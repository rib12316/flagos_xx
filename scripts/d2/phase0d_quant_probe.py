# quant kernel 大 M 优化探针：blocked 多行 × K 分块两遍法 vs 现 1 行/program。
# 现 kernel K=6144 垫 BLOCK 8192（25% lane 浪费）且单行 ILP 差，~430GB/s（流上限 829）。
# 位精确：max/abs 可结合 → 分块归约不变；逐元素量化数学照抄。
import os
import statistics
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch
import triton
import triton.language as tl

DEV = "cuda"

import sys
sys.path.insert(0, "/flagos_比赛/flagos_xx/vllm-plugin-FL")
from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
    _dynamic_per_token_quant_int8_bare_kernel as _row_kernel,
    _round_half_to_even,
)
from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8 as ref_quant


@triton.jit
def _blocked_quant_kernel(
    input_ptr, output_ptr, scale_ptr,
    M, K: tl.constexpr,
    ROWS: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * ROWS + tl.arange(0, ROWS)
    rmask = rows < M
    offs_k = tl.arange(0, BLOCK_K)
    absmax = tl.zeros((ROWS,), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        vals = tl.load(
            input_ptr + rows[:, None] * K + k + offs_k[None, :],
            mask=rmask[:, None], other=0.0,
        ).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(vals), axis=1))
    inv_scale = tl.where(
        absmax != 0.0,
        tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0,
        0.0,
    )
    for k in range(0, K, BLOCK_K):
        ptrs = input_ptr + rows[:, None] * K + k + offs_k[None, :]
        vals = tl.load(ptrs, mask=rmask[:, None], other=0.0).to(tl.float32)
        q = _round_half_to_even(vals * inv_scale[:, None])
        q = tl.minimum(tl.maximum(q, -128.0), 127.0)
        tl.store(
            output_ptr + rows[:, None] * K + k + offs_k[None, :],
            q.to(tl.int8), mask=rmask[:, None],
        )
    tl.store(scale_ptr + rows, absmax * (1.0 / 127.0), mask=rmask)


def graph_time_us(fn, iters=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        rounds.append((time.perf_counter() - t0) / iters * 1e6)
    return statistics.median(rounds), out


def main():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    torch.manual_seed(0)
    for K, M in [(2048, 2048), (6144, 2048), (2048, 8192), (6144, 8192)]:
        x = (torch.randn(M, K, device=DEV) * 3).to(torch.bfloat16)
        out = torch.empty(M, K, device=DEV, dtype=torch.int8)
        scale = torch.empty(M, 1, device=DEV, dtype=torch.float32)
        ref_q, ref_s = ref_quant(x)
        nbytes = M * K * 3  # bf16 读 + int8 写（第二遍读走 L2 记一半）

        def cur():
            _row_kernel[(M,)](
                x, out, scale, hidden_size=K,
                BLOCK_SIZE=triton.next_power_of_2(K),
                num_warps=8 if triton.next_power_of_2(K) >= 8192 else 4)
            return out
        cur(); torch.cuda.synchronize()
        ok_cur = torch.equal(out, ref_q) and torch.equal(scale, ref_s)
        t_cur, _ = graph_time_us(cur)
        print(f"\n== K={K} M={M} | 现 kernel {t_cur:.1f}µs = "
              f"{M*K*3/(t_cur*1e-6)/1e9:.0f} GB/s(3B 口径) bitexact="
              f"{'OK' if ok_cur else 'FAIL'} ==")
        best = None
        for ROWS, BK, w in [
            (1, 1024, 4), (2, 1024, 4), (4, 1024, 4), (8, 1024, 8),
            (2, 2048, 8), (4, 2048, 8), (1, 2048, 8), (8, 512, 4),
            (4, 512, 4), (16, 1024, 8),
        ]:
            try:
                def call():
                    _blocked_quant_kernel[(triton.cdiv(M, ROWS),)](
                        x, out, scale, M, K=K, ROWS=ROWS, BLOCK_K=BK,
                        num_warps=w)
                    return out
                call(); torch.cuda.synchronize()
                ok = torch.equal(out, ref_q) and torch.equal(scale, ref_s)
                t_us, _ = graph_time_us(call)
                mark = ""
                if ok and (best is None or t_us < best[0]):
                    best = (t_us, (ROWS, BK, w))
                    mark = " *"
                print(f"  R{ROWS} BK{BK} w{w}: {t_us:7.1f} µs = "
                      f"{M*K*3/(t_us*1e-6)/1e9:4.0f} GB/s  "
                      f"{'OK' if ok else 'FAIL!!'}{mark}", flush=True)
            except Exception as e:
                print(f"  R{ROWS} BK{BK} w{w}: EXC {str(e)[:60]}")
        if best:
            print(f"  -> 最优 {best[1]} 快 {t_cur/best[0]:.2f}× vs 现 kernel")


if __name__ == "__main__":
    main()
