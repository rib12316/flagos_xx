# TileLang GEMM 校准对决赛（阶段1，docs/交付-B机-TileLang试点单.md）
#
# 方法论复刻 scripts/d7/gemm_calib.py（commit a18e64a）：单 kernel 单 grid
# (104,) = C500 CTA 数；每 CTA 把 A[BM,BK]/B[BK,BN] 拷进 shared 后循环
# iters=1024 次 gemm 累加——数据驻留寄存器/共享内存（L2-warm 等效口径），
# 计时 bp.graph_time_us（CUDA graph 3×5 replay 中位，µs；graph 不可捕时退
# 事件计时并在结果标注）。TFLOPS = 104 × iters × BM·BN·BK / t(µs) × 1e-6。
# 对决赛锚：沐曦 triton 3.0 tl.dot 峰值 56T（256,64,64,w8,s1）。
# 判决：X≥100T 立项 / X<70T 判死 / 70-100T 灰区上报。
import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")  # 无关本测，但保持环境一致

import torch  # noqa: E402
import tilelang  # noqa: E402
import tilelang.language as T  # noqa: E402

sys.path.insert(0, "scripts/d2")
import bottleneck_probe as bp  # noqa: E402

ITERS = 1024
GRID = 104  # C500 CTA 数（与 d7 对齐）


def make_kernel(BM, BN, BK, threads, dtype=T.bfloat16, target="auto"):
    @tilelang.jit(target=target)
    def gemm_loop(
        A: T.Tensor((GRID * BM, BK), dtype),
        B: T.Tensor((BK, BN), dtype),
        C: T.Tensor((GRID * BM, BN), "float32"),
    ):
        with T.Kernel(GRID, threads=threads) as bx:
            A_shared = T.alloc_shared((BM, BK), dtype)
            B_shared = T.alloc_shared((BK, BN), dtype)
            C_local = T.alloc_fragment((BM, BN), "float32")
            T.copy(A[bx * BM, 0], A_shared)
            T.copy(B[0, 0], B_shared)
            T.clear(C_local)
            for _ in range(ITERS):
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[bx * BM, 0])

    return gemm_loop


def bench_one(BM, BN, BK, warps, target):
    threads = warps * 32
    a = torch.randn(GRID * BM, BK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(BK, BN, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(GRID * BM, BN, device="cuda", dtype=torch.float32)
    kern = make_kernel(BM, BN, BK, threads, target=target)
    kern(a, b, c)
    torch.cuda.synchronize()
    caliber = "graph"
    try:
        t = bp.graph_time_us(lambda: kern(a, b, c), iters=5)
    except Exception:
        caliber = "event"
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            for _ in range(3):
                kern(a, b, c)
        torch.cuda.synchronize()
        import statistics
        import time
        rounds = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                kern(a, b, c)
            torch.cuda.synchronize()
            rounds.append((time.perf_counter() - t0) / 10 * 1e6)
        t = statistics.median(rounds)
    macs = GRID * ITERS * BM * BN * BK
    return t, macs / t * 1e-6, caliber


def main():
    target = os.environ.get("TILELANG_TARGET", "auto")
    cfgs = [(128, 32, 128, 8), (128, 64, 128, 8), (64, 64, 128, 4),
            (128, 128, 64, 8), (256, 64, 64, 8), (128, 256, 64, 8),
            (256, 128, 64, 8), (64, 128, 128, 8), (128, 64, 64, 4)]
    print(f"tilelang {tilelang.__version__} target={target} "
          f"grid={GRID} iters={ITERS}")
    best = 0.0
    for BM, BN, BK, w in cfgs:
        try:
            t, tf, cal = bench_one(BM, BN, BK, w, target)
            best = max(best, tf)
            print(f"BM{BM} BN{BN} BK{BK} w{w}: {t:8.1f}µs({cal}) "
                  f"→ {tf:5.0f} TFLOPS")
        except Exception as e:
            print(f"BM{BM} BN{BN} BK{BK} w{w}: {type(e).__name__} "
                  f"{str(e)[:80]}")
    print(f"\nPEAK={best:.0f}T  (锚: triton 56T; 判决 ≥100T 立项 / <70T 判死)")


if __name__ == "__main__":
    main()
