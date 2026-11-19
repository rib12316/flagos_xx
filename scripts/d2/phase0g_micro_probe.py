# 微调探针 a+b：融合 kernel 冠军 (256,256,64,G4,w8,s2) 的边缘点 + B 流 cache_modifier。
# a) cache_modifier=".cg"（B 权重单次流读绕 L1）与 eviction_policy="evict_first"
# b) 边缘配置：s3 / w16 / GROUP_M∈{2,16}
# 全部驱动生产 kernel（.cg 用带参副本 kernel），位精确对拍 emu，graph 口径。
import os
import statistics
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch
import triton
import triton.language as tl

import sys
sys.path.insert(0, "/flagos_比赛/flagos_xx/vllm-plugin-FL")
from vllm_fl.quantization.w8a8 import int_mm as im

DEV = "cuda"
SHAPES = [(2048, 12288, "gate_up"), (6144, 2048, "down"),
          (2048, 2560, "qkv"), (2048, 2048, "o")]
BASE = (256, 256, 64, 4, 8, 2)  # 生产冠军


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


@triton.jit(do_not_specialize=["M"])
def _fused_gemm_cg_kernel(
    x_ptr, w_ptr, y_ptr, xs_ptr, ws_ptr, bias_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr, HAS_BIAS: tl.constexpr, EVEN_M: tl.constexpr,
    B_CG: tl.constexpr,  # 1 = B 流 .cg；0 = 默认
    B_EF: tl.constexpr,  # 1 = B 流 evict_first
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
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        if B_CG:
            b = tl.load(b_ptrs, cache_modifier=".cg")
        elif B_EF:
            b = tl.load(b_ptrs, eviction_policy="evict_first")
        else:
            b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    xs = tl.load(xs_ptr + offs_m)
    ws = tl.load(ws_ptr + offs_n)
    y = acc.to(tl.float32)
    y = y * xs[:, None]
    y = y * ws[None, :]
    tl.store(y_ptr + offs_m[:, None] * N + offs_n[None, :],
             y.to(tl.bfloat16))


def run_prod(a, b, xs, ws, y, M, N, K, cfg, kernel=None):
    BM, BN, BK, GM, W, S = cfg
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)
    if kernel is None:
        def call():
            im._w8a8_fused_gemm_kernel[grid](
                a, b, y, xs, ws, a, M, N, K,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                HAS_BIAS=False, EVEN_M=(M % BM == 0),
                num_warps=W, num_stages=S)
            return y
    else:
        def call():
            kernel[grid](
                a, b, y, xs, ws, a, M, N, K,
                BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
                HAS_BIAS=False, EVEN_M=True, B_CG=True, B_EF=False,
                num_warps=W, num_stages=S)
            return y
    return call


def main():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    h = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        h @ h
    torch.cuda.synchronize()
    M = 2048
    for K, N, tag in SHAPES:
        torch.manual_seed(0)
        a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
        b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
        xs = torch.rand(M, device=DEV) / 8 + 1e-4
        ws = torch.rand(N, device=DEV) / 8 + 1e-4
        y = torch.empty(M, N, device=DEV, dtype=torch.bfloat16)
        acc = torch._int_mm(a, b)
        emu = acc.to(torch.float32).mul_(xs[:, None]).mul_(ws[None, :]) \
            .to(torch.bfloat16)
        # cuBLAS 参照
        t_cublas, _ = graph_time_us(lambda: torch._int_mm(a, b))

        call = run_prod(a, b, xs, ws, y, M, N, K, BASE)
        call(); torch.cuda.synchronize()
        assert torch.equal(y, emu)
        t_base, _ = graph_time_us(call)
        print(f"\n== {tag} K={K} N={N} M={M} | 生产 {t_base:.1f}µs "
              f"| cuBLAS {t_cublas:.1f}µs（{t_cublas/t_base*100:.0f}%）==")

        # b) 边缘配置（生产 kernel）
        for cfg in [
            (256, 256, 64, 4, 8, 3),   # s3
            (256, 256, 64, 4, 16, 2),  # w16
            (256, 256, 64, 2, 8, 2),   # GM2
            (256, 256, 64, 16, 8, 2),  # GM16
        ]:
            try:
                c2 = run_prod(a, b, xs, ws, y, M, N, K, cfg)
                c2(); torch.cuda.synchronize()
                ok = torch.equal(y, emu)
                t2, _ = graph_time_us(c2)
                mark = " ✓" if ok and t2 < t_base * 0.97 else ""
                print(f"  {cfg}: {t2:7.1f} µs ({t_base/t2:.3f}× vs 生产)"
                      f"{' OK' if ok else ' FAIL!!'}{mark}", flush=True)
            except Exception as e:
                print(f"  {cfg}: EXC {str(e)[:60]}")

        # a) cache modifier（同冠军配置，cg 副本 kernel）
        for name, cg, ef in [(".cg", True, False), ("evict_first", False, True)]:
            try:
                def c3():
                    grid = (triton.cdiv(M, 256) * triton.cdiv(N, 256),)
                    _fused_gemm_cg_kernel[grid](
                        a, b, y, xs, ws, a, M, N, K,
                        BLOCK_M=256, BLOCK_N=256, BLOCK_K=64, GROUP_M=4,
                        HAS_BIAS=False, EVEN_M=True, B_CG=cg, B_EF=ef,
                        num_warps=8, num_stages=2)
                    return y
                c3(); torch.cuda.synchronize()
                ok = torch.equal(y, emu)
                t3, _ = graph_time_us(c3)
                mark = " ✓" if ok and t3 < t_base * 0.97 else ""
                print(f"  B流 {name}: {t3:7.1f} µs ({t_base/t3:.3f}× vs 生产)"
                      f"{' OK' if ok else ' FAIL!!'}{mark}", flush=True)
            except Exception as e:
                print(f"  B流 {name}: EXC {str(e)[:70]}")


if __name__ == "__main__":
    main()
