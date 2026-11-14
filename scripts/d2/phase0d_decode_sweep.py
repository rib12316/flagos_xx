# Phase 0d：decode M=64/128 融合 GEMM 小 tile 配置扫（官方 decode 并发 64/128 →
# 每步 M=64/128，现全走 cuBLAS 链；该 M 档从未扫过小 BM tile——_FUSED_GEMM_MIN_M=256
# 是按 (256,256) 大 tile 的败绩设的，偏保守）。
#
# 与 phase0c 的区别：
#   ① M ∈ {64,128}（decode 档），BM ≤ M（even_m 规则自动排除 BM>M）
#   ② 基线臂 = serve 今日真实支付的整链：_int_mm + _run_fused_epilogue（graph replay；
#      quant kernel 两臂相同相消；M=64/128 无 pad-17）
#   ③ GROUP_M 固定 1（num_pid_m ≤ 2 时 group 调度恒等，无需扫）
#   ④ 新增收 lm_head 形状 (2048,130560)（P1 预备；合成 int8 操作数，先于 ckpt 存在）
# 判据：融合 ≥ 1.15× 整链（graph 口径，3×30 中位）才回填 _FUSED_GEMM_DECODE_CONFIGS。
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
SHAPES = [  # (K, N, tag)
    (2048, 12288, "gate_up"),
    (2048, 2560, "qkv"),
    (2048, 2048, "o"),
    (6144, 2048, "down"),
    (2048, 130560, "lm_head"),
]
MS = [64, 128]


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


def cfgs_for(M):
    # (BM, BN, BK, warps, stages) —— BM>M 被 even_m 规则挡掉，不在此重复过滤
    out = []
    for BM in (32, 64, 128):
        if BM > M:
            continue
        warps = (4,) if BM == 32 else (4, 8)
        for BN in (128, 256):
            for BK in (64, 128):
                for w in warps:
                    for st in (2, 3):
                        if BM == 128:  # 大 BM 子族收窄（BK64 为主）
                            if BK != 64:
                                continue
                        out.append((BM, BN, BK, w, st))
    return out


def run():
    # GPU 门禁 + 热身
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()

    for K, N, tag in SHAPES:
        for M in MS:
            torch.manual_seed(0)
            a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
            b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
            xs = torch.rand(M, device=DEV) / 8 + 1e-4
            ws = torch.rand(N, device=DEV) / 8 + 1e-4
            ref = torch._int_mm(a, b)
            emu = ref.to(torch.float32).mul_(xs[:, None]).mul_(ws[None, :]) \
                .to(torch.bfloat16)

            # 基线臂：serve 真实整链（_int_mm + 融合 epilogue）
            def chain():
                acc = torch._int_mm(a, b)
                return im._run_fused_epilogue(acc, xs, ws, None, M, N)
            t_chain, out_chain = graph_time_us(chain)
            assert torch.equal(out_chain, emu), "chain baseline != emu?"

            results = []
            for BM, BN, BK, w, st in cfgs_for(M):
                smem = (BM * BK + BK * BN) * st
                if smem > 128 * 1024:
                    continue
                try:
                    c = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
                    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

                    def call():
                        im._w8a8_fused_gemm_kernel[grid](
                            a, b, c, xs, ws, a, M, N, K,
                            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=1,
                            HAS_BIAS=False, EVEN_M=(M % BM == 0),
                            num_warps=w, num_stages=st)
                        return c
                    call()
                    torch.cuda.synchronize()
                    if not torch.equal(c, emu):
                        print(f"  {tag} M{M} BM{BM} BN{BN} BK{BK} w{w} s{st}"
                              f": BITEXACT FAIL", flush=True)
                        continue
                    t_us, _ = graph_time_us(call)
                    results.append((t_us, (BM, BN, BK, w, st), smem))
                except Exception as e:
                    print(f"  {tag} M{M} BM{BM} BN{BN} BK{BK} w{w} s{st}: "
                          f"EXC {type(e).__name__} {str(e)[:60]}", flush=True)
            results.sort()
            bw = (M * K + K * N + M * N * 2) / (t_chain * 1e-6) / 1e9
            print(f"\n== {tag} K={K} N={N} M={M} | 整链 t_chain="
                  f"{t_chain:.1f}µs (~{bw:.0f}GB/s) ==", flush=True)
            for t_us, cfg, smem in results[:6]:
                r = t_chain / t_us
                mark = " ✓GATE" if r >= 1.15 else ""
                print(f"  {t_us:8.1f} µs  {cfg}  "
                      f"({r:.3f}× chain){mark}", flush=True)
            if results:
                best = results[0]
                print(f"  -> 最优 {best[1]} = {t_chain/best[0]:.3f}× chain"
                      f"（判据 ≥1.15× "
                      f"{'过' if t_chain/best[0] >= 1.15 else '不过'}）",
                      flush=True)
            else:
                print("  -> 无可用配置（全 BITEXACT FAIL/EXC）", flush=True)


if __name__ == "__main__":
    run()
