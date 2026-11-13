# Phase 0 第二轮：gate_up 热态复测 + 扩展 tile 配置 + skinny split-K。
# 第一轮疑点：gate_up 21.4 T-MAC/s（19% cuBLAS）vs down 74.4（59%）——首形状冷时钟
# 伪影嫌疑；skinny 全输现路径（grid 96 program 串行 K 延迟主导）。
import statistics
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


def graph_time_us(fn, iters: int = 30) -> float:
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
def _tiled_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                  BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr, GROUP_M: tl.constexpr):
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
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc)


@triton.jit
def _skinny_splitk_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                          BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    # grid = (cdiv(N, BN), SPLIT_K)；各 split 段 int32 atomic_add（整数可结合，位精确）
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    K_per = tl.cdiv(K, SPLIT_K)
    k0 = pid_k * K_per
    offs_k = k0 + tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K_per, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.atomic_add(c_ptrs, acc, mask=m_mask[:, None])


def run_tiled(M, K, N, tag, cfgs):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    ref = torch._int_mm(a, b)
    def mm():
        return torch._int_mm(a, b)
    t_ref, _ = graph_time_us(mm)
    print(f"\n== {tag}: M={M} K={K} N={N} | cuBLAS {t_ref:.1f}µs = "
          f"{M*K*N/(t_ref*1e-6)/1e12:.1f} T-MAC/s ==")
    for BM, BN, BK, w, st in cfgs:
        try:
            c = torch.empty((M, N), device=DEV, dtype=torch.int32)
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

            def call():
                _tiled_kernel[grid](a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN,
                                    BLOCK_K=BK, GROUP_M=8, num_warps=w,
                                    num_stages=st)
                return c
            call(); torch.cuda.synchronize()
            ok = torch.equal(c, ref)
            t_us, _ = graph_time_us(call)
            tmacs = M * K * N / (t_us * 1e-6) / 1e12
            print(f"  BM{BM} BN{BN} BK{BK} w{w} s{st}: {t_us:8.1f} µs = {tmacs:6.1f} "
                  f"T-MAC/s ({tmacs/(M*K*N/(t_ref*1e-6)/1e12)*100:.0f}% cublas) "
                  f"bitexact={'OK' if ok else 'FAIL!!'}", flush=True)
        except Exception as e:
            print(f"  BM{BM} BN{BN} BK{BK} w{w} s{st}: EXC {type(e).__name__} {str(e)[:70]}")


def run_skinny_splitk(M, K, N, tag):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    ap = torch.zeros(17, K, device=DEV, dtype=torch.int8); ap[:M] = a
    ref = torch._int_mm(ap, b)[:M]
    def cur():
        ap2 = torch.zeros(17, K, device=DEV, dtype=torch.int8)
        ap2[:M] = a
        return torch._int_mm(ap2, b)[:M]
    t_cur, _ = graph_time_us(cur)
    bytes_w = K * N
    print(f"\n== {tag}: M={M} K={K} N={N} | 现路径 {t_cur:.1f}µs = "
          f"{bytes_w/(t_cur*1e-6)/1e9:.0f} GB/s ==")
    for BM, BN, BK, SK, w, st in [
        (16, 64, 128, 4, 4, 3), (16, 64, 256, 2, 4, 2), (16, 32, 256, 4, 4, 2),
        (16, 64, 128, 8, 4, 2), (16, 128, 128, 4, 4, 3), (16, 32, 128, 8, 2, 2),
        (16, 64, 64, 8, 4, 2), (16, 128, 64, 4, 4, 3),
    ]:
        try:
            c = torch.zeros(M, N, device=DEV, dtype=torch.int32)
            grid = (triton.cdiv(N, BN), SK)

            def call():
                _skinny_splitk_kernel[grid](a, b, c, M, N, K, BLOCK_M=BM,
                                            BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK,
                                            num_warps=w, num_stages=st)
                return c
            call(); torch.cuda.synchronize()
            ok = torch.equal(c, ref)
            c.zero_()  # graph 里也要能重复：每次重放前清零在 fn 外做不到——
            # graph 捕获的是 call()，其中不含清零 → 重放会累加。改在 call 内清零成本高；
            # 计时口径：先捕获含 zero_ 的完整版。
            def call2():
                c.zero_()
                _skinny_splitk_kernel[grid](a, b, c, M, N, K, BLOCK_M=BM,
                                            BLOCK_N=BN, BLOCK_K=BK, SPLIT_K=SK,
                                            num_warps=w, num_stages=st)
                return c
            t_us, _ = graph_time_us(call2)
            print(f"  BM{BM} BN{BN} BK{BK} SK{SK} w{w} s{st}: {t_us:8.1f} µs = "
                  f"{bytes_w/(t_us*1e-6)/1e9:5.0f} GB/s(+zero) bitexact="
                  f"{'OK' if ok else 'FAIL!!'}", flush=True)
        except Exception as e:
            print(f"  BM{BM} BN{BN} BK{BK} SK{SK} w{w} s{st}: EXC {type(e).__name__} {str(e)[:70]}")


def main():
    # 热身：先把 GPU 时钟打热（跑一遍大 GEMM）
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()

    run_tiled(2048, 2048, 12288, "gate_up 热态复测", [
        (128, 256, 64, 8, 2), (128, 128, 64, 8, 3), (64, 256, 64, 8, 2),
        (128, 256, 32, 8, 2), (128, 128, 128, 8, 2), (256, 128, 64, 8, 2),
        (128, 256, 64, 8, 3), (64, 128, 128, 4, 3),
    ])
    run_tiled(2048, 6144, 2048, "down 复测（扩展 stage/BN）", [
        (128, 256, 64, 8, 2), (128, 256, 64, 8, 3), (128, 256, 128, 8, 2),
        (64, 256, 128, 8, 2), (128, 128, 64, 8, 4),
    ])
    run_skinny_splitk(16, 2048, 12288, "skinny16 gate_up split-K")
    run_skinny_splitk(16, 6144, 2048, "skinny16 down split-K")


if __name__ == "__main__":
    main()
