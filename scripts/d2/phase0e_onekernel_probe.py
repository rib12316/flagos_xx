# 1-kernel 融合判决探针：bf16-A 融合 GEMM（量化前移进 GEMM）实测 vs 现 2-kernel 分工。
#
# 候选形态（absmax 预 kernel + bf16-A GEMM，仍 2 kernel）：
#   A1 (256,256,64,s2) 96KB → 1CTA；A2 (128,256,64,s2) 64KB → 2CTA（但 BM128 慢 tile）
#   A3 (256,128,64,s2) 48KB → 2CTA
# 真单 kernel（absmax 在 GEMM program 内预扫，1 kernel）也测一把给上限参考。
# 对照：现生产 (quant kernel + int8-A 融合 GEMM)。位精确：absmax/round/clamp/dot 数学
# 与生产链逐字同款。判据：总 GPU 时间（graph）beat 现分工 >5% 才值得产线化。
import os
import statistics
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch
import triton
import triton.language as tl

import sys
sys.path.insert(0, "/flagos_比赛/flagos_xx/vllm-plugin-FL")
from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
    _dynamic_per_token_quant_int8_bare_kernel as _row_quant_kernel,
    _round_half_to_even,
)
from vllm_fl.quantization.w8a8 import int_mm as im

DEV = "cuda"


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


@triton.jit
def _row_absmax_kernel(x_ptr, am_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < K
    v = tl.load(x_ptr + row * K + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(am_ptr + row, tl.max(tl.abs(v), axis=0))


@triton.jit(do_not_specialize=["M"])
def _bf16a_fused_gemm_kernel(
    x_ptr,  # [M, K] bf16
    w_ptr,  # [K, N] int8
    inv_ptr,  # [M] fp32 = fdiv(1,absmax)*127
    y_ptr,  # [M, N] bf16
    ws_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    inv = tl.load(inv_ptr + offs_m)  # [BM]
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs).to(tl.float32)          # bf16 tile → f32
        a = tl.minimum(tl.maximum(_round_half_to_even(a * inv[:, None]),
                                  -128.0), 127.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a.to(tl.int8), b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    ws = tl.load(ws_ptr + offs_n)
    y = acc.to(tl.float32)
    # 与 emu 的乘法次序一致需要 xs=absmax/127：inv=127/absmax → xs=1/inv 不精确！
    # 所以 epilogue 用真实 xs 由外部传入（xs_ptr 复用 inv_ptr 所在的 absmax 张量换算
    # 不可位精确）——此处探针只测吞吐，位精确在产线化时用第二个指针解决。
    y = y * tl.load(inv_ptr + 0)  # 占位（探针吞吐口径，不参与位精确判定）
    y = y * ws[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             y.to(tl.bfloat16))


@triton.jit(do_not_specialize=["M"])
def _true_one_kernel(
    x_ptr, w_ptr, y_ptr, ws_ptr, xs_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # 真单 kernel：program 内先全 K 预扫 absmax（bf16 重读），再量化+dot。
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # pass1: absmax
    absmax = tl.zeros((BLOCK_M,), dtype=tl.float32)
    p = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    for _ in range(0, K, BLOCK_K):
        v = tl.load(p).to(tl.float32)
        absmax = tl.maximum(absmax, tl.max(tl.abs(v), axis=1))
        p += BLOCK_K
    inv = tl.where(absmax != 0.0,
                   tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0, 0.0)
    # pass2: 量化 + dot
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs).to(tl.float32)
        a = tl.minimum(tl.maximum(_round_half_to_even(a * inv[:, None]),
                                  -128.0), 127.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a.to(tl.int8), b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    xs = absmax * (1.0 / 127.0)
    ws = tl.load(ws_ptr + offs_n)
    y = acc.to(tl.float32)
    y = y * xs[:, None]
    y = y * ws[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             y.to(tl.bfloat16))
    tl.store(xs_ptr + offs_m, xs)


def main():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    torch.manual_seed(0)
    M, K, N = 2048, 2048, 12288  # gate_up（L2 压力最重的形状，1-kernel 最不利场）
    x = (torch.randn(M, K, device=DEV) * 3).to(torch.bfloat16)
    w = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    ws = torch.rand(N, device=DEV) / 8 + 1e-4
    # 热身
    h = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        h @ h
    torch.cuda.synchronize()

    # ===== 现生产 2-kernel 分工（基线）=====
    xq = torch.empty(M, K, device=DEV, dtype=torch.int8)
    xs = torch.empty(M, 1, device=DEV, dtype=torch.float32)
    y_prod = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    BM, BN, BK, GM, W, S = im._FUSED_GEMM_CONFIGS[(K, N)]

    def prod():
        _row_quant_kernel[(M,)](x, xq, xs, hidden_size=K,
                                BLOCK_SIZE=triton.next_power_of_2(K),
                                num_warps=4)
        grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
        im._w8a8_fused_gemm_kernel[grid](
            xq, w, y_prod, xs, ws, xq, M, N, K,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            HAS_BIAS=False, EVEN_M=(M % BM == 0), num_warps=W, num_stages=S)
        return y_prod
    prod(); torch.cuda.synchronize()
    t_prod, _ = graph_time_us(prod)
    print(f"现生产分工（quant {triton.next_power_of_2(K)}w4 + 融合GEMM "
          f"{(BM,BN,BK,GM,W,S)}）: {t_prod:.1f} µs")

    # 位精确 oracle（emu 公式）
    acc = torch._int_mm(xq, w)
    y_emu = acc.to(torch.float32).mul_(xs).mul_(ws).to(torch.bfloat16)
    assert torch.equal(y_prod, y_emu), "production output mismatch?!"

    # ===== A: absmax kernel + bf16-A 融合 GEMM（候选 tile）=====
    am = torch.empty(M, device=DEV, dtype=torch.float32)
    y_a = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    for BM2, BN2, BK2, W2, S2 in [
        (256, 256, 64, 8, 2), (128, 256, 64, 8, 2), (256, 128, 64, 8, 2),
        (128, 128, 64, 8, 2),
    ]:
        smem = (BM2 * BK2 * 2 + BK2 * BN2) * S2

        def variantA():
            _row_absmax_kernel[(M,)](x, am, K=K,
                                     BLOCK=triton.next_power_of_2(K),
                                     num_warps=4)
            grid = (triton.cdiv(M, BM2) * triton.cdiv(N, BN2),)
            _bf16a_fused_gemm_kernel[grid](
                x, w, am, y_a, ws, M, N, K,
                BLOCK_M=BM2, BLOCK_N=BN2, BLOCK_K=BK2, GROUP_M=4,
                num_warps=W2, num_stages=S2)
            return y_a
        try:
            variantA(); torch.cuda.synchronize()
            t_a, _ = graph_time_us(variantA)
            print(f"A absmax+bf16GEMM (BM{BM2} BN{BN2} BK{BK2} w{W2} s{S2}) "
                  f"smem~{smem//1024}KB: {t_a:7.1f} µs  "
                  f"({'beat' if t_a < t_prod*0.95 else 'LOSE'} vs 生产 "
                  f"{t_prod/t_a:.2f}×)")
        except Exception as e:
            print(f"A (BM{BM2} BN{BN2} BK{BK2} w{W2} s{S2}): EXC {str(e)[:70]}")

    # ===== B: 真单 kernel（absmax program 内预扫）——位精确版 =====
    y_b = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
    xs_b = torch.empty(M, device=DEV, dtype=torch.float32)
    for BM2, BN2, BK2, W2, S2 in [(256, 256, 64, 8, 2), (128, 256, 64, 8, 2)]:
        def variantB():
            grid = (triton.cdiv(M, BM2) * triton.cdiv(N, BN2),)
            _true_one_kernel[grid](
                x, w, y_b, ws, xs_b, M, N, K,
                BLOCK_M=BM2, BLOCK_N=BN2, BLOCK_K=BK2, GROUP_M=4,
                num_warps=W2, num_stages=S2)
            return y_b
        try:
            variantB(); torch.cuda.synchronize()
            ok = torch.equal(y_b, y_emu) and torch.equal(
                xs_b, xs.reshape(-1))
            t_b, _ = graph_time_us(variantB)
            print(f"B 真单kernel (BM{BM2} BN{BN2} BK{BK2} w{W2} s{S2}): "
                  f"{t_b:7.1f} µs  bitexact={'OK' if ok else 'FAIL!!'}  "
                  f"({'beat' if t_b < t_prod*0.95 else 'LOSE'} vs 生产 "
                  f"{t_prod/t_b:.2f}×)")
        except Exception as e:
            print(f"B (BM{BM2} BN{BN2} BK{BK2} w{W2} s{S2}): EXC {str(e)[:70]}")


if __name__ == "__main__":
    main()
