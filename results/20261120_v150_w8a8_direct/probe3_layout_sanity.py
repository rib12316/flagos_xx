"""Layout-flip sanity: every routing tier bit-exact vs the CPU int32 oracle.

Covers, under VLLM_FL_INTMM_TN=1 ([N,K] storage):
  - prefill M=2048  -> ixformer TN route (cuinfer kernel)
  - prefill M=257   -> ixformer TN odd-tail route
  - prefill M=2048+bias -> ported Triton fused kernel (strided [N,K] B loads)
  - decode M=64     -> ported decode-tier Triton kernel
  - M=100           -> cuBLAS chain via .t() view
and under VLLM_FL_INTMM_TN=0 ([K,N] legacy storage): the same tiers as a
regression pass. One kernel-trace assertion pins the M=2048 route to cuinfer.
"""
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vllm-plugin-FL'))
from vllm_fl.quantization.w8a8 import int_mm as impl

SHAPES = [('gate_up', 2048, 12288), ('qkv', 2048, 2560),
          ('o', 2048, 2048), ('down', 6144, 2048)]


def oracle_rows(x_q, w_mat, xs, ws, bias, rows):
    acc = x_q[rows].cpu().to(torch.int32) @ w_mat.cpu().to(torch.int32)
    y = acc.float() * xs[rows].cpu().float()[:, None]
    y = y * ws.cpu().float()[None, :]
    if bias is not None:
        y = y + bias.cpu().float()[None, :]
    return y.bfloat16().cuda()


def run_case(tn, tag, m, k, n, with_bias=False):
    os.environ['VLLM_FL_INTMM_TN'] = '1' if tn else '0'
    torch.manual_seed(7)
    wt = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    w_mat = wt.t().contiguous()  # logical [K, N]
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
    xs = torch.rand(m, device='cuda') / 8 + 1e-4
    ws = torch.rand(n, device='cuda') / 8 + 1e-4
    bias = torch.randn(n, device='cuda') * 0.1 if with_bias else None
    weight = wt if tn else w_mat
    out = impl.int8_scaled_mm_int_mm(a, weight, xs, ws, bias)
    rows = torch.linspace(0, m - 1, 7).long().cuda()
    ref = oracle_rows(a, w_mat, xs, ws, bias, rows)
    ok = torch.equal(out[rows], ref)
    print(f"{'TN' if tn else 'KN'} {tag:8s} M={m:5d} bias={int(with_bias)}: "
          f"{'OK' if ok else 'MISMATCH'}", flush=True)
    return ok


def trace_route():
    """Assert the M=2048 TN route lands on the cuinfer vendor kernel."""
    os.environ['VLLM_FL_INTMM_TN'] = '1'
    torch.manual_seed(7)
    k, n = 2048, 2560
    wt = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
    a = torch.randint(-127, 128, (2048, k), dtype=torch.int8, device='cuda')
    xs = torch.rand(2048, device='cuda') / 8 + 1e-4
    ws = torch.rand(n, device='cuda') / 8 + 1e-4
    impl.int8_scaled_mm_int_mm(a, wt, xs, ws, None)  # warm
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA]) as prof:
        impl.int8_scaled_mm_int_mm(a, wt, xs, ws, None)
        torch.cuda.synchronize()
    kernels = [e.name for e in prof.events()
               if e.device_type == torch.autograd.DeviceType.CUDA]
    hit = any('cuinfer' in name for name in kernels)
    print('trace M=2048 TN kernels:', kernels[:3], '-> cuinfer' if hit else '')
    return hit


def main():
    results = []
    for tag, k, n in SHAPES:
        results.append(run_case(True, tag, 2048, k, n))          # ix TN
    results.append(run_case(True, 'gate_up', 257, 2048, 12288))  # ix odd tail
    results.append(run_case(True, 'qkv', 2048, 2048, 2560, True))    # fused+bias
    results.append(run_case(True, 'gate_up', 2048, 2048, 12288, True))
    results.append(run_case(True, 'qkv', 64, 2048, 2560))        # decode tier
    results.append(run_case(True, 'down', 64, 6144, 2048))
    results.append(run_case(True, 'qkv', 100, 2048, 2560))       # chain view
    results.append(run_case(True, 'qkv', 12, 2048, 2560))        # pad chain view
    # legacy-layout regression
    results.append(run_case(False, 'qkv', 2048, 2048, 2560))     # fused KN
    results.append(run_case(False, 'gate_up', 2048, 2048, 12288, True))
    results.append(run_case(False, 'qkv', 64, 2048, 2560))       # decode KN
    results.append(run_case(False, 'o', 100, 2048, 2048))        # chain KN
    results.append(trace_route())
    print('ALL OK' if all(results) else 'FAILURES PRESENT')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
