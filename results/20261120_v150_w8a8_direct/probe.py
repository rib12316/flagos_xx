"""Direct W8A8 operator comparison on BI-V150. Run from a free competition GPU.

Historical note (post-TN-flip): this probe predates the [N,K] TN storage
default. It pins VLLM_FL_INTMM_TN=0 so the archived numbers stay
reproducible; the TN-layout evidence lives in probe2.py/probe3_layout_sanity.py.
"""
import os

os.environ.setdefault("VLLM_FL_INTMM_TN", "0")  # legacy-layout evidence
import argparse
import itertools
import json
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

import torch
import triton
from ixformer.inference.functions.w8a8 import w8a8_gemm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'vllm-plugin-FL'))
from vllm_fl.quantization.w8a8 import int_mm as impl

PROJECTIONS = [('gate_up', 2048, 12288), ('qkv', 2048, 2560),
               ('o', 2048, 2048), ('down', 6144, 2048)]


def compare(out, ref):
    # Count BF16 differences and measure their distance in representable values.
    def ordered(x):
        bits = x.view(torch.int16).int()
        return torch.where(bits < 0, -32768 - bits, bits)
    diff = (ordered(out) - ordered(ref)).abs()
    return {'different': int((diff != 0).sum()), 'max_ulp': int(diff.max()),
            'nonfinite': int((~torch.isfinite(out)).sum()), 'elements': out.numel()}


def benchmark(methods, mode):
    calls = {}
    graphs = {}
    outputs = {}
    replay_batch = 4
    num_calls = 24
    for name, fn in methods.items():
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        if mode == 'graph':
            graphs[name] = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graphs[name]):
                for _ in range(replay_batch):
                    outputs[name] = fn()
            graphs[name].replay()
            calls[name] = graphs[name].replay
        else:
            calls[name] = fn
    torch.cuda.synchronize()
    times = {name: [] for name in calls}
    for order in itertools.permutations(calls):
        for name in order:
            # Warm immediately before the timed interval on both paths.
            calls[name]()
            torch.cuda.synchronize()
            start = time.perf_counter_ns()
            for _ in range(num_calls // replay_batch if mode == 'graph' else num_calls):
                calls[name]()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter_ns() - start) / (1000 * num_calls))
    return {name: {'us': statistics.median(v), 'rounds_us': v} for name, v in times.items()}, outputs


def trace_once(fn):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = fn()
        torch.cuda.synchronize()
    return [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]


def run_shape(tag, m, k, n, seed):
    torch.manual_seed(seed)
    a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
    w = torch.randint(-127, 128, (k, n), dtype=torch.int8, device='cuda')
    wt = w.T.contiguous()
    xs = torch.rand(m, device='cuda') / 8 + 1e-4
    ws = torch.rand(n, device='cuda') / 8 + 1e-4
    # Test zeros, signed values and rounding with independent row and column scales.
    a[0].zero_()
    w[:, 0].zero_()
    wt[0].zero_()
    acc = torch._int_mm(a, w)
    ref = ((acc.float() * xs[:, None]) * ws[None, :]).bfloat16()
    rows = torch.linspace(0, m - 1, 13).long()
    cols = torch.linspace(0, n - 1, 31).long()
    rows_gpu, cols_gpu = rows.cuda(), cols.cuda()
    cpu_acc = a[rows_gpu].cpu().int() @ w[:, cols_gpu].cpu().int()
    assert torch.equal(cpu_acc, acc[rows_gpu[:, None], cols_gpu].cpu())

    def current():
        out = impl._run_fused_gemm(a, w, xs, ws, None, m, n, k)
        if out is not None:
            return out
        return impl._run_fused_epilogue(torch._int_mm(a, w), xs, ws, None, m, n)

    def vendor_nn():
        return w8a8_gemm(a, w, xs, ws, format='NN', out_dtype=torch.bfloat16)

    def vendor_tn():
        return w8a8_gemm(a, wt, xs, ws, format='TN', out_dtype=torch.bfloat16)

    methods = {'current': current, 'ix_NN': vendor_nn, 'ix_TN': vendor_tn}
    checks = {name: compare(fn(), ref) for name, fn in methods.items()}
    assert checks['current']['different'] == 0, checks
    result = dict(projection=tag, M=m, K=k, N=n, seed=seed, parity=checks)
    for mode in (['graph'] if m == 64 else ['graph', 'eager']):
        timings, captured = benchmark(methods, mode)
        result[mode] = timings
        if captured:
            result['replay_parity'] = {name: compare(out, ref) for name, out in captured.items()}
    if tag == 'gate_up':
        result['kernel_trace'] = {name: trace_once(fn) for name, fn in methods.items()}
    print(tag, m, json.dumps({mode: {key: round(t['us'], 2) for key, t in result[mode].items()}
                              for mode in ('graph', 'eager') if mode in result}),
          'parity', [(key, t['different'], t['max_ulp']) for key, t in checks.items()], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1120)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--gate-only', action='store_true')
    parser.add_argument('--rows', nargs='+', type=int, default=[64, 2048])
    args = parser.parse_args()
    data = dict(torch=torch.__version__, triton=triton.__version__, ixformer=version('ixformer'),
                gpu=torch.cuda.get_device_name(), results=[])
    for tag, k, n in PROJECTIONS[:1] if args.gate_only else PROJECTIONS:
        for m in args.rows:
            data['results'].append(run_shape(tag, m, k, n, args.seed))
            args.out.write_text(json.dumps(data, indent=2) + '\n')


if __name__ == '__main__':
    with torch.inference_mode():
        main()
