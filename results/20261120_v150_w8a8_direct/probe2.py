"""Probe 2 for the ixformer TN integration: fallback viability under [N,K] storage.

A) torch._int_mm with a .t() view of [N,K] weights (cuBLAS transposed-B operand):
   correctness vs contiguous [K,N], graph timing (hidden copies show up), and
   CUDA-graph capture+replay safety. Decides whether the cuBLAS chain can serve
   decode capture sizes / unknown shapes from TN storage with zero copies.

B) ix TN at the decode capture sizes (M in 16/24/32/40/48/72) vs the production
   chain: parity + graph timing. Decides whether non-{64,128} decode batches can
   route to ixformer instead of the chain.
"""
import json
import statistics
import time
from importlib.metadata import version
from pathlib import Path

import torch
import triton
from ixformer.inference.functions.w8a8 import w8a8_gemm

ROOT = Path(__file__).resolve().parents[2]
from vllm_fl.quantization.w8a8 import int_mm as impl

SHAPES = [('gate_up', 2048, 12288), ('qkv', 2048, 2560),
          ('o', 2048, 2048), ('down', 6144, 2048)]


def med_us(fn, num_calls=24, replay_batch=4):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(replay_batch):
            fn()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(6):
        g.replay()
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        for _ in range(4):
            g.replay()
        torch.cuda.synchronize()
        times.append((time.perf_counter_ns() - start) / (1000 * num_calls))
    return statistics.median(times)


def probe_a():
    """_int_mm on transposed views of [N,K] storage."""
    results = {}
    for tag, k, n in SHAPES:
        torch.manual_seed(42)
        wt = torch.randint(-127, 128, (n, k), dtype=torch.int8, device='cuda')
        w_kn = wt.t().contiguous()
        view = wt.t()  # [K, N] strides (1, K)
        a = torch.randint(-127, 128, (64, k), dtype=torch.int8, device='cuda')
        row = {}
        try:
            r_view = torch._int_mm(a, view)
            r_ref = torch._int_mm(a, w_kn)
            row['equal'] = bool(torch.equal(r_view, r_ref))
            row['view_strides'] = tuple(view.stride())
            # graph timing: view (no copy) vs contiguous
            row['us_view'] = med_us(lambda: torch._int_mm(a, view))
            row['us_contig'] = med_us(lambda: torch._int_mm(a, w_kn))
            # capture with the view created inside the captured region
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                o = torch._int_mm(a, wt.t())
            g.replay()
            torch.cuda.synchronize()
            row['capture_inline_view_ok'] = bool(torch.equal(o, r_ref))
        except Exception as e:
            row['error'] = f'{type(e).__name__}: {e}'[:300]
        results[tag] = row
        print('A', tag, json.dumps(row), flush=True)
    return results


def probe_b():
    """ix TN at decode capture sizes vs the production chain ([K,N] today)."""
    results = {}
    for tag, k, n in SHAPES:
        for m in (16, 24, 32, 40, 48, 72, 64):
            torch.manual_seed(43)
            a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device='cuda')
            w = torch.randint(-127, 128, (k, n), dtype=torch.int8, device='cuda')
            wt = w.t().contiguous()
            xs = torch.rand(m, device='cuda') / 8 + 1e-4
            ws = torch.rand(n, device='cuda') / 8 + 1e-4
            if m <= 16:  # _int_mm needs M > 16; zero-pad like the chain does
                acc = torch._int_mm(
                    torch.cat([a, a.new_zeros(1, k)]), w)[:m]
            else:
                acc = torch._int_mm(a, w)
            ref = ((acc.float() * xs[:, None]) * ws[None, :]).bfloat16()

            def chain():
                return impl._int8_scaled_mm_int_mm_impl(
                    a, w, xs, ws, None, torch.bfloat16)

            def ix_tn():
                return w8a8_gemm(a, wt, xs, ws, format='TN',
                                 out_dtype=torch.bfloat16)

            row = {}
            try:
                out_c, out_i = chain(), ix_tn()
                eq = torch.equal(out_i, ref)
                # chain is the production oracle-holder; assert it too
                eq_c = torch.equal(out_c, ref)
                row['parity_ix'] = eq
                row['parity_chain'] = eq_c
                if eq and eq_c:
                    row['us_chain'] = med_us(chain)
                    row['us_ix_tn'] = med_us(ix_tn)
            except Exception as e:
                row['error'] = f'{type(e).__name__}: {e}'[:300]
            results[f'{tag}_M{m}'] = row
            print('B', tag, m, json.dumps(row), flush=True)
    return results


def main():
    data = dict(torch=torch.__version__, triton=triton.__version__,
                ixformer=version('ixformer'), gpu=torch.cuda.get_device_name(),
                probe_a={}, probe_b={})
    out = Path(__file__).with_name('probe2.json')
    import sys
    if '--probe-b-only' in sys.argv:
        data['probe_a'] = 'skipped (already collected 19:3x)'
    else:
        data['probe_a'] = probe_a()
    data['probe_b'] = probe_b()
    out.write_text(json.dumps(data, indent=2) + '\n')
    print('written', out)


if __name__ == '__main__':
    with torch.inference_mode():
        main()
