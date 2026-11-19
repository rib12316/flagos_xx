# Phase 0c：融合 kernel 生产形态配置深扫（驱动真实 _w8a8_fused_gemm_kernel，
# 含 epilogue 尾，非探针副本）。
#
# ixkncli 归因：理论占用 12.5%，瓶颈 smem 49KB/block → 2 CTA/CU。本扫攻击：
#   ① ≤32KB smem 的 4-CTA 配置族（(128,128,64,s2) 等——probe1 恰好漏扫的点）
#   ② GROUP_M ∈ {4,8,16}（从未扫过）
#   ③ num_stages/warps 网格加密
# （cache_modifier=".cg" 需 kernel 加参数变体，若本扫仍显 L2 bound 再补）
# 判据：graph 口径 vs 现配置表基线；>5% 才回填。
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch
import triton

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "vllm-plugin-FL"))
from vllm_fl.quantization.w8a8 import int_mm as im  # noqa: E402

DEV = "cuda"
SHAPES = [  # (K, N, tag, 生产配置)
    (2048, 12288, "gate_up", (256, 128, 64, 8, 2)),
    (2048, 2560, "qkv", (128, 256, 64, 8, 2)),
    (2048, 2048, "o", (128, 256, 64, 8, 2)),
    (6144, 2048, "down", None),  # 现走链；若深扫 >=80% cuBLAS 可重新融合
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


CFGS = [  # (BM, BN, BK, warps, stages) —— smem ≈ (BM*BK + BK*BN)*stages 字节
    # 4-CTA 族（≤32KB）：probe1 漏扫点
    (128, 128, 64, 8, 2), (128, 128, 64, 4, 2), (128, 64, 64, 4, 2),
    (128, 64, 64, 4, 3), (64, 128, 64, 4, 2), (64, 128, 64, 4, 3),
    (128, 128, 32, 4, 4), (64, 64, 64, 4, 4),
    # 生产基线族 + 加密
    (256, 128, 64, 8, 2), (128, 256, 64, 8, 2), (128, 256, 64, 8, 3),
    (256, 128, 64, 4, 2), (256, 64, 64, 4, 2), (256, 64, 64, 8, 2),
    (256, 128, 32, 8, 3), (128, 128, 64, 8, 3), (128, 128, 128, 8, 2),
    (64, 256, 64, 8, 2), (256, 256, 64, 8, 2), (128, 256, 128, 8, 2),
    # skill 推荐补点：小 tile × 深流水（4-CTA 族的 stages 加密）
    (128, 128, 64, 8, 4), (64, 128, 64, 4, 4), (64, 128, 64, 4, 5),
    (128, 64, 64, 4, 4), (64, 64, 64, 4, 5),
]
GROUP_MS = [4, 8, 16]


def run():
    # GPU 门禁 + 热身
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()

    M = 2048
    for K, N, tag, prod_cfg in SHAPES:
        torch.manual_seed(0)
        a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
        b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
        xs = torch.rand(M, device=DEV) / 8 + 1e-4
        ws = torch.rand(N, device=DEV) / 8 + 1e-4
        ref = im._int_mm_equiv(a, b) if hasattr(im, "_int_mm_equiv") \
            else torch._int_mm(a, b)
        emu = ref.to(torch.float32).mul_(xs[:, None]).mul_(ws[None, :]) \
            .to(torch.bfloat16)

        def cu():
            return torch._int_mm(a, b)
        t_cublas, _ = graph_time_us(cu)

        results = []
        for BM, BN, BK, w, st in CFGS:
            smem = (BM * BK + BK * BN) * st
            if smem > 128 * 1024:
                continue
            try:
                c = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
                grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

                def call():
                    im._w8a8_fused_gemm_kernel[grid](
                        a, b, c, xs, ws, a, M, N, K,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=8,
                        HAS_BIAS=False, EVEN_M=(M % BM == 0),
                        num_warps=w, num_stages=st)
                    return c
                call()
                torch.cuda.synchronize()
                if not torch.equal(c, emu):
                    print(f"  {tag} BM{BM} BN{BN} BK{BK} w{w} s{st}: "
                          f"BITEXACT FAIL")
                    continue
                t_us, _ = graph_time_us(call)
                results.append((t_us, (BM, BN, BK, w, st), smem))
            except Exception as e:
                print(f"  {tag} BM{BM} BN{BN} BK{BK} w{w} s{st}: EXC "
                      f"{type(e).__name__} {str(e)[:60]}")
        results.sort()
        print(f"\n== {tag} K={K} N={N} M={M} | cuBLAS(_int_mm) {t_cublas:.1f}µs "
              f"==")
        for t_us, cfg, smem in results[:6]:
            cta = 128 * 1024 // max(smem, 1)
            print(f"  {t_us:8.1f} µs  {cfg}  smem~{smem//1024}KB"
                  f"(~{cta}CTA)")
        if prod_cfg:
            base = [r for r in results if r[1] == prod_cfg]
            if base and results[0][1] != prod_cfg:
                print(f"  -> 生产配置 {prod_cfg} {base[0][0]:.1f}µs，最优 "
                      f"{results[0][1]} 快 {base[0][0]/results[0][0]:.3f}×")
        else:
            best = results[0]
            tm = M * K * N / (best[0] * 1e-6) / 1e12
            tm_ref = M * K * N / (t_cublas * 1e-6) / 1e12
            print(f"  -> down 最优 {best[1]} = {tm:.1f} T-MAC/s "
                  f"({tm/tm_ref*100:.0f}% cuBLAS，含 epilogue 尾)")
        # GROUP_M 细扫 top-2
        for t_us, cfg, smem in results[:2]:
            BM, BN, BK, w, st = cfg
            for gm in GROUP_MS:
                if gm == 8:
                    continue
                c = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
                grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

                def call():
                    im._w8a8_fused_gemm_kernel[grid](
                        a, b, c, xs, ws, a, M, N, K,
                        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=gm,
                        HAS_BIAS=False, EVEN_M=(M % BM == 0),
                        num_warps=w, num_stages=st)
                    return c
                try:
                    call()
                    torch.cuda.synchronize()
                    assert torch.equal(c, emu)
                    t2, _ = graph_time_us(call)
                    if t2 < t_us * 0.97:
                        print(f"  GROUP_M{gm} on {cfg}: {t2:.1f}µs "
                              f"(vs 8: {t_us:.1f}) ✓")
                except Exception as e:
                    print(f"  GROUP_M{gm} on {cfg}: EXC {str(e)[:50]}")


if __name__ == "__main__":
    run()
