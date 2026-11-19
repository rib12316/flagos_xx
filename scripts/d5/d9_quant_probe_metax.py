#!/usr/bin/env python3
# D9 T0：per-token int8 quant kernel 变体对决（沐曦 C500，L2-warm 口径）
#
# 背景：serve trace 实测 `_dynamic_per_token_quant_int8_bare_kernel` 168 次/步×9.2µs
# =1.55ms/步（decode 现役：每行一 program + BLOCK 垫 next_pow2(K)，K=12288 垫 16384
# =25% lane 浪费；M=64 时 64 programs < 104 SM）。B 机 P1-B3 的「quant 37%」为 L2-cold
# 链口径，本探针以 serve 真实形状 + L2-warm（quant 不破坏输入，重复跑同张量=热态）定真价。
# 变体：
#   A 现役 bare（对照，位精确基准）
#   B kloop：1 行/program + BLOCK_K 内循环（免 2 的幂垫档，降寄存器压力）
#   C blocked：ROWS 行/program + BLOCK_K 循环（phase0d 思路，prefill 形状主攻）
# 判据：位精确（int8 输出与 scale 逐位相等）+ ≥1.3× 才立项。
import os
import statistics
import subprocess
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "vllm-plugin-FL"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vllm_fl.dispatch.backends.flaggems.impl.quantization import (  # noqa: E402
    _dynamic_per_token_quant_int8_bare_kernel as _bare,
    _round_half_to_even,
    dynamic_per_token_quant_int8_bare as ref_impl,
)

DEV = "cuda"


@triton.jit
def _kloop_kernel(input_ptr, output_ptr, scale_ptr, M, K: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    base = input_ptr + row * K
    absmax = 0.0
    for k in range(0, K, BLOCK_K):
        v = tl.load(base + k + offs).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(v), axis=0))
    inv = tl.where(absmax != 0.0, tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0, 0.0)
    for k in range(0, K, BLOCK_K):
        v = tl.load(base + k + offs).to(tl.float32)
        q = _round_half_to_even(v * inv)
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + row * K + k + offs, q.to(tl.int8))
    tl.store(scale_ptr + row, absmax * (1.0 / 127.0))


@triton.jit
def _blocked_kernel(input_ptr, output_ptr, scale_ptr, M, K: tl.constexpr,
                    ROWS: tl.constexpr, BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    offs = tl.arange(0, BLOCK_K)
    absmax = tl.zeros((ROWS,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        v = tl.load(input_ptr + rows[:, None] * K + k + offs[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(v), axis=1))
    inv = tl.where(absmax != 0.0,
                   tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0, 0.0)
    for k in range(0, K, BLOCK_K):
        v = tl.load(input_ptr + rows[:, None] * K + k + offs[None, :],
                    mask=rmask[:, None], other=0.0).to(tl.float32)
        q = _round_half_to_even(v * inv[:, None])
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(output_ptr + rows[:, None] * K + k + offs[None, :],
                 q.to(tl.int8), mask=rmask[:, None])
    tl.store(scale_ptr + rows, absmax * (1.0 / 127.0), mask=rmask)


def bench(fn, n_warm=10, n_iter=50, reps=10):
    """CUDA-graph 口径：小 kernel 的 host launch 开销(~50µs)会淹没 GPU 时间，
    捕获 reps 次进 graph 后 replay 计时（phase0d 先例；serve 内 kernel 为
    编译/流式 launch，graph replay 才是同口径）。"""
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        st = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(e) * 1e3 / reps)
    return statistics.median(ts)


def main():
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"# PROVENANCE d9_quant_probe: git={sha} torch={torch.__version__}", flush=True)
    rows = []
    shapes = [(64, 2048), (64, 2560), (64, 6144), (64, 12288),   # decode 4/层
              (2048, 2048), (2048, 6144), (2048, 12288)]         # prefill chunk
    for M, K in shapes:
        torch.manual_seed(1234 + M + K)
        x = (torch.randn(M, K, device=DEV) * 3).to(torch.bfloat16)
        q_ref, s_ref = ref_impl(x)
        o = torch.empty(M, K, device=DEV, dtype=torch.int8)
        sc = torch.empty(M, device=DEV, dtype=torch.float32)

        def run_ref():
            ref_impl(x)

        us_a = bench(run_ref)

        best_b, cfg_b = None, None
        for bk in (1024, 2048, 4096):
            if bk > triton.next_power_of_2(K):
                continue

            def run_b(bk=bk):
                _kloop_kernel[(M,)](x, o, sc, M, K=K, BLOCK_K=bk, num_warps=8 if bk >= 4096 else 4)

            run_b()
            torch.cuda.synchronize()
            ok = torch.equal(o, q_ref) and torch.equal(sc.view(M, 1), s_ref)
            us = bench(run_b)
            if ok and (best_b is None or us < best_b):
                best_b, cfg_b = us, f"BLOCK_K={bk}"
        best_c, cfg_c = None, None
        for rows_n, bk in ((2, 2048), (4, 2048), (4, 1024), (8, 1024)):
            grid = (M + rows_n - 1) // rows_n

            def run_c(rows_n=rows_n, bk=bk, grid=grid):
                _blocked_kernel[(grid,)](x, o, sc, M, K=K, ROWS=rows_n,
                                         BLOCK_K=bk, num_warps=8 if bk >= 4096 else 4)

            try:
                run_c()
                torch.cuda.synchronize()
            except Exception:
                continue
            ok = torch.equal(o, q_ref) and torch.equal(sc.view(M, 1), s_ref)
            if not ok:
                continue
            us = bench(run_c)
            if best_c is None or us < best_c:
                best_c, cfg_c = us, f"ROWS={rows_n},BLOCK_K={bk}"

        line = (f"[M={M} K={K}] bare={us_a:.1f}µs"
                f" | kloop={best_b if best_b is None else f'{best_b:.1f}µs'}({cfg_b}) {'' if best_b else 'FAIL/慢'}"
                f" | blocked={best_c if best_c is None else f'{best_c:.1f}µs'}({cfg_c})")
        print(line, flush=True)
        rows.append((M, K, us_a, best_b, best_c))

    out = os.path.join(REPO, "results", "20261120_c500_reprofiling", "d9_quant_metax.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("M,K,bare_us,kloop_us,blocked_us\n")
        for r in rows:
            f.write(",".join(str(v) if v is not None else "" for v in r) + "\n")
    print(f"# CSV -> {out}", flush=True)
    print("# 判据：decode 形状（M=64）最优变体 ≥1.3× 且位精确 → D9 立项", flush=True)


if __name__ == "__main__":
    sys.exit(main())
