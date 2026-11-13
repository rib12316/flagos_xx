# Phase 0 探针（不改插件）：Triton int8 tl.dot 可行性闸门。
#
# 问题：prefill 归因显示 int8 链的 epilogue+quant = GPU 税 27% + 多 43% launch。
# 解药候选 = 融合单 kernel（quant prologue + int8 GEMM + epilogue）。但 M=2048 档
# 是算力 bound（gate_up 51.5 G-MAC vs 访存 83MB）——融合 kernel 必须接近 cuBLAS
# 的 MAC 吞吐（_int_mm gate_up@2048 = 464µs 实测）。本探针回答：
#   P1. tl.dot(int8,int8,int32) 在本栈是否位精确（vs torch._int_mm 逐位）——R1 风险
#   P2. tiled 形态（BM 64/128）能达到多少 MAC 吞吐 vs cuBLAS（判据 ≥70% 才值得融合）
#   P3. skinny 形态（BM 1/16，M=1/16 全 K）带宽形态（decode 融合用）
#   P4. 25MB int8 纯读流带宽天花板（性能预期分母）
# 计时一律 CUDA graph 重放口径（eager 三伪影）。

import statistics
import sys
import time

import torch
import triton
import triton.language as tl

DEV = "cuda"


def gpu_free_mib() -> float:
    import subprocess
    out = subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True)
    return float(out.strip().splitlines()[0])


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


@triton.jit  # 裸 jit
def _tiled_int8_dot_kernel(
    a_ptr, b_ptr, c_ptr,  # a [M,K] int8, b [K,N] int8, c [M,N] int32
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
def _skinny_int8_dot_kernel(
    a_ptr, b_ptr, c_ptr,  # a [M,K] (M<=BLOCK_M), b [K,N], c [M,N] int32
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_n = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m_mask = offs_m < M
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_k[:, None] * N + offs_n[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _read_reduce_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0)
    tl.atomic_add(o_ptr, tl.sum(x.to(tl.int32), axis=0) * 0)


def run_tiled(M, K, N, tag):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    ref = torch._int_mm(a, b)  # M>16 OK；位精确 oracle（int32 精确已证）

    print(f"\n== {tag}: M={M} K={K} N={N}（{M*K*N/1e9:.1f} G-MAC）==")
    best = None
    for BM, BN, BK, w, st in [
        (128, 128, 64, 4, 3), (128, 128, 64, 8, 3), (128, 256, 64, 8, 2),
        (64, 128, 64, 4, 3), (64, 256, 64, 8, 2), (128, 128, 32, 4, 4),
        (64, 64, 64, 4, 4), (128, 64, 64, 4, 3),
    ]:
        try:
            c = torch.empty((M, N), device=DEV, dtype=torch.int32)
            grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

            def call():
                _tiled_int8_dot_kernel[grid](
                    a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                    GROUP_M=8, num_warps=w, num_stages=st)
                return c
            call()
            torch.cuda.synchronize()
            ok = torch.equal(c, ref)
            t_us, _ = graph_time_us(call)
            tmacs = M * K * N / (t_us * 1e-6) / 1e12
            line = f"  BM{BM} BN{BN} BK{BK} w{w} s{st}: {t_us:8.1f} µs = {tmacs:6.1f} T-MAC/s  bitexact={'OK' if ok else 'FAIL!!'}"
            print(line, flush=True)
            if ok and (best is None or t_us < best[0]):
                best = (t_us, (BM, BN, BK, w, st), tmacs)
        except Exception as e:
            print(f"  BM{BM} BN{BN} BK{BK} w{w} s{st}: EXC {type(e).__name__}: {str(e)[:90]}")
    # cuBLAS 对照
    def mm():
        return torch._int_mm(a, b)
    t_ref, out_ref = graph_time_us(mm)
    print(f"  cuBLAS _int_mm:  {t_ref:8.1f} µs = {M*K*N/(t_ref*1e-6)/1e12:6.1f} T-MAC/s")
    if best:
        print(f"  -> best Triton {best[1]} = {best[2]:.1f} T-MAC/s "
              f"({best[2]/(M*K*N/(t_ref*1e-6)/1e12)*100:.0f}% of cuBLAS, "
              f"{t_ref/best[0]:.2f}× slower)")
    return best, t_ref


def run_skinny(M, K, N, tag):
    torch.manual_seed(0)
    a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    # oracle：CPU int32（CUDA 无 int GEMM 小 M 垫零版也行：pad 到 17）
    ap = torch.zeros(17, K, device=DEV, dtype=torch.int8)
    ap[:M] = a
    ref = torch._int_mm(ap, b)[:M]
    print(f"\n== {tag}: M={M} K={K} N={N} ==")
    for BM, BN, BK, w, st in [
        (16, 128, 256, 4, 2), (16, 128, 512, 4, 2), (16, 64, 512, 4, 3),
        (16, 256, 256, 8, 2), (1, 128, 512, 4, 2) if M == 1 else (16, 128, 128, 4, 3),
    ]:
        try:
            c = torch.empty((M, N), device=DEV, dtype=torch.int32)
            grid = (triton.cdiv(N, BN),)

            def call():
                _skinny_int8_dot_kernel[grid](
                    a, b, c, M, N, K, BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                    num_warps=w, num_stages=st)
                return c
            call()
            torch.cuda.synchronize()
            ok = torch.equal(c, ref)
            t_us, _ = graph_time_us(call)
            bw = (K * N + M * K + M * N * 4) / (t_us * 1e-6) / 1e9
            print(f"  BM{BM} BN{BN} BK{BK} w{w} s{st}: {t_us:8.1f} µs = {bw:6.0f} GB/s  bitexact={'OK' if ok else 'FAIL!!'}", flush=True)
        except Exception as e:
            print(f"  BM{BM} BN{BN} BK{BK} w{w} s{st}: EXC {type(e).__name__}: {str(e)[:90]}")
    # 现路径对照：pad 17 + _int_mm
    def cur():
        ap2 = torch.zeros(17, K, device=DEV, dtype=torch.int8)
        ap2[:M] = a
        return torch._int_mm(ap2, b)[:M]
    t_cur, _ = graph_time_us(cur)
    print(f"  现路径(pad17+_int_mm): {t_cur:8.1f} µs")


def run_bw_probe():
    n = 25 * 1024 * 1024  # 25MB int8
    x = torch.randint(-127, 128, (n,), device=DEV, dtype=torch.int8)
    o = torch.zeros(1, device=DEV, dtype=torch.int32)
    for BLOCK, w in [(4096, 4), (8192, 8), (16384, 8)]:
        grid = (triton.cdiv(n, BLOCK),)

        def call():
            _read_reduce_kernel[grid](x, o, n, BLOCK=BLOCK, num_warps=w)
        call(); torch.cuda.synchronize()
        t_us, _ = graph_time_us(call)
        print(f"  read {n/1e6:.0f}MB BLOCK{BLOCK} w{w}: {t_us:8.1f} µs = {n/(t_us*1e-6)/1e9:6.0f} GB/s", flush=True)


def main():
    used = gpu_free_mib()
    assert used < 3000, f"GPU 独占门禁失败: {used} MiB"
    print("== P1/P2: tiled int8 tl.dot 位精确 + 吞吐（M=2048 档）==")
    run_tiled(2048, 2048, 12288, "gate_up")
    run_tiled(2048, 6144, 2048, "down")
    run_tiled(2048, 2048, 2560, "qkv")
    print("\n== P3: skinny（M=1/16，decode 融合用）==")
    run_skinny(16, 2048, 12288, "skinny16 gate_up")
    run_skinny(1, 2048, 12288, "skinny1 gate_up")
    run_skinny(16, 6144, 2048, "skinny16 down")
    print("\n== P4: 25MB 读流带宽天花板 ==")
    run_bw_probe()


if __name__ == "__main__":
    main()
