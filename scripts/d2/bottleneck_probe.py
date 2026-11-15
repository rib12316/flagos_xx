# 瓶颈快诊 L1：屋顶线分类探针（~5min，无需 serve/官方 bench）。
#
# 原则：时间占比（谁慢）≠ 瓶颈类型（为什么慢）。量化/优化选型前先回答后者：
#   bandwidth-bound   → 字节减半类（量化权重/KV）有效
#   compute-bound     → 整数算力类（int8 dot 原生路径）有效
#   latency/occupancy → 都无效，需 kernel 形态工程（split/并行度/流水）
# 每组件实测「达成速率 vs 本机天花板」→ 分类。天花板：读流 829GB/s（BLOCK16384
# 实测）、_int_mm 110-125 T-MAC/s、16 SM。
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch  # noqa: E402
import triton  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "vllm-plugin-FL"))
import vllm.v1.attention.ops.triton_unified_attention as tua  # noqa: E402
from vllm_fl.quantization.w8a8 import int_mm as im  # noqa: E402

DEV = "cuda"
NH, NKV, HD, BLOCK = 16, 2, 128, 16
BW_CEIL = 829.0   # GB/s 读流天花板（ledger 实测）
MAC_CEIL = 118.0  # T-MAC/s _int_mm 中位（110-125）


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
    return statistics.median(rounds)


def attn_case(S, q_len, kv_len):
    n_tok = S * kv_len
    q = torch.randn(S * q_len, NH, HD, device=DEV, dtype=torch.bfloat16)
    nb = triton.cdiv(kv_len, BLOCK) * S + 4
    kv = torch.randn(2, nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16)
    bt = torch.arange(S * triton.cdiv(kv_len, BLOCK), dtype=torch.int32,
                      device=DEV).reshape(S, -1)
    cu = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
    su = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)

    def call():
        tua.unified_attention(
            q, kv[0], kv[1], out, cu, q_len, su, kv_len,
            softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
            block_table=bt, softcap=0.0, q_descale=None, k_descale=None,
            v_descale=None)
        return out
    return call, S * kv_len * 2 * NKV * HD * 2  # KV 读字节


def gemm_case(M, K, N, cfg=None):
    a = torch.randint(-127, 128, (M, K), device=DEV, dtype=torch.int8)
    b = torch.randint(-127, 128, (K, N), device=DEV, dtype=torch.int8)
    xs = torch.rand(M, device=DEV) / 8 + 1e-4
    ws = torch.rand(N, device=DEV) / 8 + 1e-4

    if cfg is None:  # cuBLAS 链
        def call():
            acc = torch._int_mm(a, b)
            return im._run_fused_epilogue(acc, xs, ws, None, M, N)
        return call, M * K + K * N + M * N * 2
    BM, BN, BK, GM, w, st = cfg
    c = torch.empty((M, N), device=DEV, dtype=torch.bfloat16)
    grid = (triton.cdiv(M, BM) * triton.cdiv(N, BN),)

    def call():
        im._w8a8_fused_gemm_kernel[grid](
            a, b, c, xs, ws, a, M, N, K,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, GROUP_M=GM,
            HAS_BIAS=False, EVEN_M=(M % BM == 0),
            num_warps=w, num_stages=st)
        return c
    return call, M * K + K * N + M * N * 2


def elem_bw_case():
    x = torch.randn(64 * 1024 * 1024 // 2, device=DEV, dtype=torch.bfloat16)
    y = torch.empty_like(x)

    def call():
        torch.mul(x, 0.5, out=y)
        return y
    return call, x.numel() * 2 * 2  # 读+写


def cls(pct_bw, pct_mac=None):
    if pct_bw >= 60:
        return "bandwidth-bound"
    if pct_mac is not None and pct_mac >= 60:
        return "compute-bound"
    if pct_bw < 60 and (pct_mac is None or pct_mac < 60):
        return "latency/occupancy-bound（量化无效，需 kernel 形态工程）"


def run():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    # 关键：装载 serve 同款 launch 调优（D0 表：decode w1s1 / prefill w2s1）。
    # 不装则测的是未调优 kernel——attention 会慢 ~3×，墙图失真（首版教训）。
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_launch_config as alc
    alc.install()  # type: ignore[attr-defined]
    print(f"launch 表已装载：decode w1s1 / prefill w2s1（serve 等效口径）",
          flush=True)
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()

    rows = []
    # 0) 元素带宽参照（本机可达带宽下界）
    call, by = elem_bw_case()
    t = graph_time_us(call)
    rows.append(("元素参照 mul(64MB)", t, by / t * 1e-3, None))

    # 1) attention（decode/prefill 各一）
    for tag, S, ql, kl in [("attn decode S=64 kv=2k", 64, 1, 2048),
                           ("attn decode S=64 kv=16k", 64, 1, 16384),
                           ("attn prefill S=2 q=2k kv=2k", 2, 2048, 2048)]:
        call, by = attn_case(S, ql, kl)
        t = graph_time_us(call)
        rows.append((tag, t, by / t * 1e-3, None))

    # 2) 线性 decode 档（P0 融合 vs 链）+ prefill 融合
    for tag, M, K, N, cfg in [
        ("lin decode M=64 gate_up 链", 64, 2048, 12288, None),
        ("lin decode M=64 gate_up 融合", 64, 2048, 12288, (64, 256, 64, 1, 4, 2)),
        ("lin decode M=64 down 链", 64, 6144, 2048, None),
        ("lin decode M=64 down 融合", 64, 6144, 2048, (64, 128, 128, 1, 8, 2)),
        ("lin decode M=64 lm_head 链", 64, 2048, 130560, None),
        ("lin prefill M=2048 gate_up 融合", 2048, 2048, 12288, (256, 256, 64, 4, 16, 2)),
    ]:
        call, by = gemm_case(M, K, N, cfg)
        t = graph_time_us(call)
        mac = M * K * N / (t * 1e-6) / 1e12
        rows.append((tag, t, by / t * 1e-3, mac))

    print(f"{'组件':<32} {'µs':>9} {'GB/s':>7} {'%BW':>6} {'T-MAC/s':>8} "
          f"{'%MAC':>6}  分类")
    for tag, t, bw, mac in rows:
        pct_bw = bw / BW_CEIL * 100
        pct_mac = mac / MAC_CEIL * 100 if mac else None
        c = cls(pct_bw, pct_mac)
        print(f"{tag:<32} {t:>9.1f} {bw:>7.0f} {pct_bw:>5.0f}% "
              f"{mac if mac else 0:>8.1f} {pct_mac if mac else 0:>5.0f}%  {c}")
    print("判读：attention 若 %BW<25% 且 decode 形态 → latency/occupancy（KV 量化/带宽类免疫）；"
          "线性 %MAC≥85% → 已贴 vendor 算力顶。", flush=True)


if __name__ == "__main__":
    run()
