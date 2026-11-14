# Phase 2：KV cache int8_per_token_head 原生路径探针（编译/数值/时间三闸，90min 时间盒）。
#
# 背景：本栈 VLLM_PLUGINS=fl 的 attention 后端 = TritonAttentionBackend（unified_attention
# 可读 Triton），vLLM 0.24 自带 int8_per_token_head KV 量化（写入侧
# triton_reshape_and_cache_flash_per_token_head_quant per-(token,head) absmax 动态 scale
# 存 inline head padding（head_size 128→132，容量比 1024/528≈1.94）；读取侧
# unified_attention KV_QUANT_MODE=INT8_PER_TOKEN_HEAD kernel 内反量化）。旧路线图 F9
# 「独立路径已证伪」针对外挂包装路径，本原生 in-kernel 路线在本栈未测过。
#
# 三闸：
#   ① 编译闸：flagtree 3.6 能编 _reshape_cache_per_token_head + unified_attention 的
#      KV_QUANT_MODE=2 分支（EXC/abort 即失败）
#   ② 数值闸：int8 vs bf16 attention 输出 max_rel ≤ ~3%（sanity 级，非精度 gate）
#   ③ 时间闸（信息性）：graph 口径 attention kernel 时间 bf16 vs int8——decode 长 KV
#      形态（KV 读带宽减半的直接兑现点）与 prefill 形态（dequant ALU 税观察点，
#      F9 旧恐惧 -25% 的对照）
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch  # noqa: E402
import triton  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "vllm-plugin-FL"))

from vllm.v1.attention.backends.triton_attn import (  # noqa: E402
    TritonAttentionBackend, TritonAttentionImpl,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.attention.ops.triton_unified_attention import (  # noqa: E402
    unified_attention,
)
from vllm.v1.kv_cache_interface import KVQuantMode  # noqa: E402

DEV = "cuda"
NKV, NQ, HD = 2, 16, 128          # MiniCPM5-2B GQA 8:1，head_dim 128
BLOCK = 16                        # iluvatar 平台默认 block_size
SCALE_PAD = 4                     # fp32/int8 = 4 元素 inline scale

# (tag, S, q_len, kv_len)：decode 短/长 KV + prefill 两形态
CASES = [
    ("decode_kv2k", 64, 1, 2048),
    ("decode_kv16k", 64, 1, 16384),
    ("prefill_2k", 2, 2048, 2048),
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


def build_case(S, q_len, kv_len, seed):
    gen = torch.Generator().manual_seed(seed)
    n_tok = S * kv_len
    n_blocks = triton.cdiv(kv_len, BLOCK) * S + 4
    q = torch.randn(S * q_len, NQ, HD, generator=gen).to(torch.bfloat16).to(DEV)
    key = torch.randn(n_tok, NKV, HD, generator=gen).to(torch.bfloat16).to(DEV)
    val = torch.randn(n_tok, NKV, HD, generator=gen).to(torch.bfloat16).to(DEV)
    cu_q = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
    seqused_k = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)
    # block_table：每序列一段连续块；slot = block_table[seq, t//B]*B + t%B
    blocks_per_seq = triton.cdiv(kv_len, BLOCK)
    bt = torch.arange(S * blocks_per_seq, dtype=torch.int32, device=DEV) \
        .reshape(S, blocks_per_seq)
    seq_of = torch.arange(n_tok, device=DEV) // kv_len
    t_in = torch.arange(n_tok, device=DEV) % kv_len
    blk = bt[seq_of, t_in // BLOCK]
    sm = (blk * BLOCK + t_in % BLOCK).to(torch.int32)
    return q, key, val, cu_q, seqused_k, bt, sm, n_blocks


def run():
    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    from vllm.v1.attention.backends.utils import get_kv_cache_layout
    layout = get_kv_cache_layout()
    print(f"kv_cache_layout={layout}（NHD → get_kv_cache_shape 顺序即物理连续，"
          f"探针布局与 serve 一致）", flush=True)
    assert layout == "NHD", "HND 布局下探针需按 stride_order 重排（本栈默认 NHD）"

    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()

    for tag, S, q_len, kv_len in CASES:
        q, key, val, cu_q, seqused_k, bt, sm, n_blocks = build_case(
            S, q_len, kv_len, seed=hash(tag) % 10000)

        # ---- bf16 参照池（物理连续，advanced-index 写入）----
        cache_bf = torch.zeros(n_blocks, 2, BLOCK, NKV, HD,
                               dtype=torch.bfloat16, device=DEV)
        cache_bf[sm // BLOCK, 0, sm % BLOCK] = key
        cache_bf[sm // BLOCK, 1, sm % BLOCK] = val
        kbf, vbf = cache_bf.unbind(1)

        # ---- int8 池（head pad 132）+ strided fp32 scale views ----
        shape8 = TritonAttentionBackend.get_kv_cache_shape(
            n_blocks, BLOCK, NKV, HD, "int8_per_token_head")
        cache8 = torch.zeros(*shape8, dtype=torch.int8, device=DEV)
        impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
        impl._k_scale_cache = None
        impl._ensure_scale_caches(cache8)  # 建 k/v scale strided 视图
        k8, v8 = cache8.unbind(1)
        ks, vs = impl._k_scale_cache, impl._v_scale_cache

        # 写入闸①：int8 量化写入 kernel 在 flagtree 编译且运行
        triton_reshape_and_cache_flash_per_token_head_quant(
            key, val, k8, v8, ks, vs, sm)
        torch.cuda.synchronize()

        # 写入数值：dequant(key_cache)·scale ≈ key（int8 动态 scale 舍入 ~0.4%）
        k_dq = k8[..., :HD].to(torch.float32) \
            * ks.unsqueeze(-1).to(torch.float32)
        rel = (k_dq - key.to(torch.float32)).abs().max() \
            / key.to(torch.float32).abs().max()
        print(f"[{tag}] write dequant max_rel={rel:.4f} "
              f"{'OK' if rel < 0.02 else 'FAIL'}", flush=True)

        # ---- attention 双臂 ----
        out_bf = torch.empty_like(q)
        out_i8 = torch.empty_like(q)

        def attn_bf():
            unified_attention(
                q, kbf, vbf, out_bf, cu_q, q_len, seqused_k, kv_len,
                softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
                block_table=bt, softcap=None, q_descale=None, k_descale=None,
                v_descale=None, kv_quant_mode=KVQuantMode.NONE,
                k_scale_cache=None, v_scale_cache=None)
            return out_bf

        def attn_i8():
            unified_attention(
                q, k8, v8, out_i8, cu_q, q_len, seqused_k, kv_len,
                softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
                block_table=bt, softcap=None, q_descale=None, k_descale=None,
                v_descale=None, kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                k_scale_cache=ks, v_scale_cache=vs)
            return out_i8

        attn_bf()
        attn_i8()
        torch.cuda.synchronize()

        denom = out_bf.to(torch.float32).abs().max().clamp_min(1e-6)
        arel = (out_i8.to(torch.float32) - out_bf.to(torch.float32)).abs() \
            .max() / denom
        t_bf, _ = graph_time_us(attn_bf)
        t_i8, _ = graph_time_us(attn_i8)
        kv_bytes_bf = S * kv_len * 2 * NKV * HD * 2
        bw_bf = kv_bytes_bf / (t_bf * 1e-6) / 1e9
        print(f"[{tag}] S={S} q={q_len} kv={kv_len} | 数值 max_rel="
              f"{arel:.4f} {'OK(≤3%)' if arel <= 0.03 else 'FAIL'} | "
              f"bf16 {t_bf:.1f}µs (~{bw_bf:.0f}GB/s KV读) vs int8 {t_i8:.1f}µs"
              f" = {t_bf/t_i8:.3f}×", flush=True)

    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    run()
