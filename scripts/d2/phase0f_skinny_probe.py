# decode skinny 探针（两段归约 split-K，替代 pad17+_int_mm+epi 的 65µs）：
#   stage1: x_q[16,K] int8 × w[K,N] → int32 partials [SK, M, N]（B 每元素读一次，
#           grid = (N/BN, SK) 喂饱 16 SM；整段 dot 位精确）
#   stage2: partials 整数求和（可结合=位精确）+ epilogue（乘法次序照抄）→ bf16
# 对照现路径：pad17 + torch._int_mm + _run_fused_epilogue。判据 graph 口径 ≥1.3×。
# quant kernel 保留不动（x_q 相同，位精确零重推导）。
import os
import statistics
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch
import triton
import triton.language as tl

import sys
sys.path.insert(0, "/flagos_比赛/flagos_xx/vllm-plugin-FL")
from vllm_fl.quantization.w8a8.int_mm import _run_fused_epilogue

DEV = "cuda"
SHAPES = [(2048, 12288, "gate_up"), (6144, 2048, "down"),
          (2048, 2560, "qkv"), (2048, 2048, "o")]


def graph_time_us(fn, iters=50):
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


@triton.jit
def _skinny_stage1_kernel(
    x_ptr,  # [M, K] int8（M<=16，行不足掩码补 0）
    w_ptr,  # [K, N] int8
    p_ptr,  # [SK, M, N] int32 partials（N 为输出列维）
    M, N, K,
    K_PER: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k0 = pid_k * K_PER
    offs_k = k0 + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K_PER, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    # partials 布局 [SK, M, N]
    p_ptrs = p_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :]
    tl.store(p_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _skinny_stage2_kernel(
    p_ptr,  # [SK, M, N] int32
    y_ptr,  # [M, N] bf16
    xs_ptr, ws_ptr,
    M, N,
    SK: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for s in tl.static_range(SK):
        p = tl.load(p_ptr + s * M * N + offs_m[:, None] * N + offs_n[None, :],
                    mask=m_mask[:, None], other=0)
        acc += p
    xs = tl.load(xs_ptr + offs_m, mask=m_mask, other=0.0)
    ws = tl.load(ws_ptr + offs_n)
    y = acc.to(tl.float32)
    y = y * xs[:, None]
    y = y * ws[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             y.to(tl.bfloat16), mask=m_mask[:, None])


def run(M, K, N, tag):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    xs = (torch.rand(M, 1, device=DEV) / 8 + 1e-4)
    ws = (torch.rand(N, device=DEV) / 8 + 1e-4)
    # 现路径 oracle：pad17 + _int_mm + 融合 epilogue
    ap = torch.zeros(17, K, device=DEV, dtype=torch.int8)
    ap[:M] = a
    acc_ref = torch._int_mm(ap, b)[:M]
    y_ref = _run_fused_epilogue(acc_ref, xs, ws, None, M, N)

    def cur():
        ap2 = torch.zeros(17, K, device=DEV, dtype=torch.int8)
        ap2[:M] = a
        acc = torch._int_mm(ap2, b)
        return _run_fused_epilogue(acc, xs, ws, None, M, N)
    t_cur, out_cur = graph_time_us(cur)
    assert torch.equal(out_cur, y_ref)
    wbytes = K * N

    print(f"\n== {tag} M={M} K={K} N={N} | 现路径 {t_cur:.1f}µs "
          f"({wbytes/(t_cur*1e-6)/1e9:.0f} GB/s 权重) ==")
    best = None
    for SK, BN, BK, w1, w2 in [
        (4, 128, 128, 4, 4), (4, 128, 256, 4, 4), (4, 64, 256, 4, 4),
        (8, 128, 128, 4, 4), (8, 64, 128, 4, 4), (2, 128, 256, 4, 4),
        (4, 128, 512, 8, 4), (8, 128, 64, 4, 4), (4, 256, 128, 8, 4),
        (16, 64, 64, 4, 4),
    ]:
        if (K // SK) % BK != 0:
            continue
        try:
            p = torch.empty(SK, M, N, device=DEV, dtype=torch.int32)
            y = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)

            def call():
                _skinny_stage1_kernel[(triton.cdiv(N, BN), SK)](
                    a, b, p, M, N, K, K_PER=K // SK,
                    BLOCK_M=16, BLOCK_N=BN, BLOCK_K=BK,
                    num_warps=w1, num_stages=2)
                _skinny_stage2_kernel[(triton.cdiv(N, 256),)](
                    p, y, xs, ws, M, N, SK=SK, BLOCK_M=16, BLOCK_N=256,
                    num_warps=w2)
                return y
            call(); torch.cuda.synchronize()
            ok = torch.equal(y, y_ref)
            t_us, _ = graph_time_us(call)
            mark = ""
            if ok and (best is None or t_us < best[0]):
                best = (t_us, (SK, BN, BK, w1))
                mark = " *"
            print(f"  SK{SK} BN{BN} BK{BK} w{w1}: {t_us:7.1f} µs = "
                  f"{wbytes/(t_us*1e-6)/1e9:4.0f} GB/s  "
                  f"{'OK' if ok else 'FAIL!!'}{mark}", flush=True)
        except Exception as e:
            print(f"  SK{SK} BN{BN} BK{BK} w{w1}: EXC {str(e)[:60]}")
    if best:
        print(f"  -> 最优 {best[1]} = {t_cur/best[0]:.2f}× vs 现路径"
              f"（{'过' if t_cur/best[0] >= 1.3 else '未过'} 1.3× 判据）")


def main():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    h = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(150):
        h @ h
    torch.cuda.synchronize()
    for M in (16, 1):
        for K, N, tag in SHAPES:
            run(M, K, N, f"skinny{M} {tag}")


if __name__ == "__main__":
    main()
