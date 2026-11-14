# Phase 3：W4A8 探针（时间盒 0.5 天，判决性——go/no-go，不做完整 W4 管线）。
#
# 假设：decode 档（M=64/128）线性链是权重读带宽主导（W int8 读 ~267MB-1GB/步），
# W4 将权重字节减半 → 若解包 ALU 税不吃掉收益，decode 线性再 ~1.3-1.8×。
# 本 backend 无 int4×int8 tl.dot——唯一开放形态 = packed-int4 装载 + kernel 内
# 位解包 → int8 tl.dot。仅 decode 侧有意义：M/BM=1 时 B tile 恰好解包一次；
# prefill BM=256 时同一权重块被 32 个 M-tile 重复解包，必亏（不测）。
#
# 判据：graph 口径 M=64 @(2048,12288)/(2048,130560) vs P0 decode 优胜配置
# （56.2µs / 567.7µs）≥1.15× 才 go。
#
# 打包格式：沿 N 方向 2 个 int4/int8（offset-binary [-8,7]：w = nibble - 8），
# packed[k, n//2] = (hi<<4)|lo；解包 = tl.interleave(p&0xF, p>>4) - 8。
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "vllm-plugin-FL"))
from vllm_fl.quantization.w8a8 import int_mm as im  # noqa: E402

DEV = "cuda"
# (K, N, tag, P0 优胜时间 µs（phase0d sweep 实测），P0 优胜 cfg)
BASE = [
    (2048, 12288, "gate_up", 56.2, (64, 256, 64, 1, 4, 2)),
    (2048, 130560, "lm_head", 567.7, (64, 128, 64, 1, 4, 2)),
]


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


@triton.jit(do_not_specialize=["M"])  # 裸 jit（libentry 病理前科，红线）
def _w4a8_fused_gemm_kernel(
    x_ptr,   # [M, K] int8（激活侧不变）
    wp_ptr,  # [K, N/2] int8 packed（沿 N 两 nibble/byte）
    y_ptr,   # [M, N] bf16
    xs_ptr,  # fp32 [M]
    ws_ptr,  # fp32 [N]
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n  # GROUP_M=1 恒等映射（decode 档 M 块 ≤2）
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_nh = pid_n * (BLOCK_N // 2) + tl.arange(0, BLOCK_N // 2)  # packed 列
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = wp_ptr + offs_k[:, None] * (N // 2) + offs_nh[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for _ in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        p = tl.load(b_ptrs)  # (BLOCK_K, BLOCK_N//2) int8
        lo = (p & 0x0F).to(tl.int8)
        hi = (p >> 4).to(tl.int8)
        b = tl.interleave(lo, hi) - 8  # (BLOCK_K, BLOCK_N) int8, [-8,7]
        acc = tl.dot(a, b, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K
        b_ptrs += (BLOCK_K // 1) * (N // 2)
    xs = tl.load(xs_ptr + offs_m)
    y = acc.to(tl.float32)
    y = y * xs[:, None]
    y = y * tl.load(ws_ptr + offs_n)[None, :]
    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        y.to(tl.bfloat16),
    )


def run():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()

    M = 64
    for K, N, tag, t_p0, p0cfg in BASE:
        torch.manual_seed(0)
        a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
        w = torch.randint(-8, 8, (K, N), device=DEV, dtype=torch.int8)
        # packed[k, n//2] = (hi<<4)|lo，lo=偶列 nibble、hi=奇列 nibble
        lo_n = w[:, ::2].to(torch.int16) & 0xF
        hi_n = w[:, 1::2].to(torch.int16) & 0xF
        wp = ((hi_n << 4) | lo_n).to(torch.int8).contiguous()
        xs = torch.rand(M, device=DEV) / 8 + 1e-4
        ws = torch.rand(N, device=DEV) / 8 + 1e-4
        ref = torch._int_mm(a, w)
        emu = ref.to(torch.float32).mul_(xs[:, None]).mul_(ws[None, :]) \
            .to(torch.bfloat16)

        results = []
        for BM, BN, BK, w_, st in [
            (64, 128, 64, 4, 2), (64, 256, 64, 4, 2),
            (64, 128, 128, 4, 2), (64, 256, 128, 8, 2),
            (64, 128, 64, 8, 2), (64, 256, 64, 8, 2),
        ]:
            try:
                c = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
                grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

                def call():
                    _w4a8_fused_gemm_kernel[grid](
                        a, wp, c, xs, ws, M, N, K,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
                        num_warps=w_, num_stages=st)
                    return c
                call()
                torch.cuda.synchronize()
                if not torch.equal(c, emu):
                    print(f"  {tag} BM{BM} BN{BN} BK{BK} w{w_} s{st}: "
                          f"BITEXACT FAIL", flush=True)
                    continue
                t_us, _ = graph_time_us(call)
                results.append((t_us, (BM, BN, BK, w_, st)))
            except Exception as e:
                print(f"  {tag} BM{BM} BN{BN} BK{BK} w{w_} s{st}: EXC "
                      f"{type(e).__name__} {str(e)[:80]}", flush=True)
        results.sort()
        print(f"\n== {tag} K={K} N={N} M={M} | P0 int8 优胜 {t_p0}µs =="
              flush=True)
        if not results:
            print("  -> 全配置 FAIL（含 tl.interleave 支持性）→ no-go",
                  flush=True)
            continue
        for t_us, cfg in results[:4]:
            r = t_p0 / t_us
            print(f"  {t_us:8.1f} µs  {cfg}  ({r:.3f}× P0-int8)"
                  f"{' ✓GO' if r >= 1.15 else ''}", flush=True)
        verdict = "GO" if results[0][0] and t_p0 / results[0][0] >= 1.15 \
            else "NO-GO"
        print(f"  -> 判决: {verdict}（判据 ≥1.15× vs P0 int8）", flush=True)


if __name__ == "__main__":
    run()
