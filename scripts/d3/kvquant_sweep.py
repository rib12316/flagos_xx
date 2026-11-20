# P2 基线探针:KV int8_per_token_head(decode 为主)× TILE 轴 × warps。
#
# 假设:A 臂 187ms ITL 税的主因是每 tile 迭代的 scale cache 加载 + int8→bf16
# 转换在 TILE=16(默认)下被 kv/16≈1064 次迭代串行放大;量化路径最优 TILE
# 很可能 ≠ bf16 的 TILE16。若 TILE 重调即抹平大半税,P2 快路径=参数级修复。
#
# 复用:blockm_sweep 的锚点参数化 wrapper(UA/CFG)+ phase2 的 int8 池构建
# (NHD 布局钉死、head pad 132、strided scale views、量化写入 kernel)。
# 口径:graph_time_us 中位;数值闸 int8 vs bf16_t16 max_rel(sanity ~3%)。
import csv
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                            # blockm_sweep
sys.path.insert(0, os.path.join(_HERE, "..", "d2"))  # bottleneck_probe
sys.path.insert(0, os.path.join(_HERE, "..", "..", "vllm-plugin-FL"))

import torch  # noqa: E402
import triton  # noqa: E402
import blockm_sweep as bs  # noqa: E402  (import 安全:main 有 guard;UA/CFG 就绪)
import bottleneck_probe as bp  # noqa: E402

DEV = "cuda"
NKV, NQ, HD, BLOCK = 2, 16, 128, 16
OUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results",
                                       "20261120_kvquant_sweep"))
CSV_PATH = os.path.join(OUT_DIR, "sweep.csv")
THRESH_SERVE, DEFAULT_SEG = 64, 16

COLUMNS = ["tag", "S", "q_len", "kv_len", "quant", "tile_dec", "w", "s",
           "time_us", "vs_bf16_best", "max_rel", "verdict"]


def build_case(S, q_len, kv_len, seed):
    """固定 q + 双池(bf16/int8)+ block_table,全部同种子可复现。"""
    import vllm.v1.attention.backends.utils as _bku
    _bku._KV_CACHE_LAYOUT_OVERRIDE = "NHD"
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend, \
        TritonAttentionImpl
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import \
        triton_reshape_and_cache_flash_per_token_head_quant

    gen = torch.Generator().manual_seed(seed)
    n_tok = S * kv_len
    n_blocks = triton.cdiv(kv_len, BLOCK) * S + 4
    q = torch.randn(S * q_len, NQ, HD, generator=gen) \
        .to(torch.bfloat16).to(DEV)
    key = torch.randn(n_tok, NKV, HD, generator=gen).to(torch.bfloat16).to(DEV)
    val = torch.randn(n_tok, NKV, HD, generator=gen).to(torch.bfloat16).to(DEV)
    blocks_per_seq = triton.cdiv(kv_len, BLOCK)
    bt = torch.arange(S * blocks_per_seq, dtype=torch.int32,
                      device=DEV).reshape(S, blocks_per_seq)
    seq_of = torch.arange(n_tok, device=DEV) // kv_len
    t_in = torch.arange(n_tok, device=DEV) % kv_len
    blk = bt[seq_of, t_in // BLOCK]
    sm = (blk * BLOCK + t_in % BLOCK).to(torch.int32)

    cache_bf = torch.zeros(n_blocks, 2, BLOCK, NKV, HD,
                           dtype=torch.bfloat16, device=DEV)
    cache_bf[sm // BLOCK, 0, sm % BLOCK] = key
    cache_bf[sm // BLOCK, 1, sm % BLOCK] = val
    kbf, vbf = cache_bf.unbind(1)

    shape8 = TritonAttentionBackend.get_kv_cache_shape(
        n_blocks, BLOCK, NKV, HD, "int8_per_token_head")
    cache8 = torch.zeros(*shape8, dtype=torch.int8, device=DEV)
    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl._k_scale_cache = None
    impl._ensure_scale_caches(cache8)
    k8, v8 = cache8.unbind(1)
    ks, vs = impl._k_scale_cache, impl._v_scale_cache
    triton_reshape_and_cache_flash_per_token_head_quant(
        key, val, k8, v8, ks, vs, sm)
    torch.cuda.synchronize()
    return q, (kbf, vbf), (k8, v8, ks, vs), bt


def make_call(S, q_len, kv_len, q, pools, quant, out, bt):
    from vllm.v1.kv_cache_interface import KVQuantMode
    kbf, vbf = pools[0]
    k8, v8, ks, vs = pools[1]
    cu = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
    su = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)
    kwargs = dict(softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
                  softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
                  use_td=False)
    if q_len == 1:
        so = torch.empty(THRESH_SERVE, NQ, DEFAULT_SEG, HD,
                         dtype=torch.float32, device=DEV)
        smx = torch.empty(THRESH_SERVE, NQ, DEFAULT_SEG,
                          dtype=torch.float32, device=DEV)
        ss_ = torch.empty_like(smx)
        kwargs.update(seq_threshold_3D=THRESH_SERVE,
                      num_par_softmax_segments=DEFAULT_SEG,
                      softmax_segm_output=so, softmax_segm_max=smx,
                      softmax_segm_expsum=ss_)

    def call():
        if quant:
            bs.UA(q, k8, v8, out, cu_seqlens_q=cu, max_seqlen_q=q_len,
                  seqused_k=su, max_seqlen_k=kv_len, block_table=bt,
                  kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                  k_scale_cache=ks, v_scale_cache=vs, **kwargs)
        else:
            bs.UA(q, kbf, vbf, out, cu_seqlens_q=cu, max_seqlen_q=q_len,
                  seqused_k=su, max_seqlen_k=kv_len, block_table=bt,
                  kv_quant_mode=KVQuantMode.NONE,
                  k_scale_cache=None, v_scale_cache=None, **kwargs)
        return out
    return call


def time_us(call):
    for _ in range(2):
        call()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    call()
    torch.cuda.synchronize()
    rough = (time.perf_counter() - t0) * 1e6
    iters = int(max(5, min(30, 1.5e6 / max(rough, 1.0))))
    return bp.graph_time_us(call, iters=iters)


BF16_ARMS = [("bf16_t16_w1s1", 16, 1, 1), ("bf16_t32_w1s1", 32, 1, 1),
             ("bf16_t64_w2s1", 64, 2, 1)]
Q_ARMS = [("q8_t16_w1s1", 16, 1, 1), ("q8_t32_w1s1", 32, 1, 1),
          ("q8_t32_w2s1", 32, 2, 1), ("q8_t64_w2s1", 64, 2, 1),
          ("q8_t64_w4s1", 64, 4, 1), ("q8_t128_w4s2", 128, 4, 2)]


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[env] torch={torch.__version__} triton={triton.__version__}", flush=True)
    bs.CFG.update(bs.ZERO)
    gate = bs.ShapeCase(2, 512, 2048)
    gate.call(); torch.cuda.synchronize()
    d = (gate.out - gate.ref).abs().max().item()
    assert d == 0.0, f"gate fail {d}"
    print("[gate] patched-default bit-exact OK", flush=True)

    cases = [("D64", 64, 1, 16384), ("D32", 32, 1, 16384),
             ("P1", 1, 4096, 16384)]
    f = open(CSV_PATH, "w", newline="")
    w = csv.writer(f)
    w.writerow(COLUMNS); f.flush(); os.fsync(f.fileno())

    for name, S, ql, kl in cases:
        q, bf_pool, q_pool, bt = build_case(S, ql, kl, seed=42)
        out_bf = torch.empty_like(q)
        out_q = torch.empty_like(q)
        ref = None
        bf_best = None
        rows = []
        for tag, td, wp, st in BF16_ARMS + Q_ARMS:
            quant = tag.startswith("q8")
            bs.CFG.update(bs.ZERO)
            bs.CFG.update({"td": td if ql == 1 else 0, "w": wp, "s": st})
            call = make_call(S, ql, kl, q, (bf_pool, q_pool), quant,
                             out_q if quant else out_bf, bt)
            try:
                us = time_us(call)
            except Exception as e:
                rows.append([tag, S, ql, kl, int(quant), td, wp, st, "", "",
                             "", f"ERR:{str(e)[:60]}"])
                print(f"[{name}] {tag}: ERR {str(e)[:60]}", flush=True)
                continue
            call(); torch.cuda.synchronize()
            if quant:
                rel = ((out_q.float() - ref.float()).abs().max()
                       / ref.float().abs().max().clamp(min=1e-6)).item()
            else:
                if ref is None:
                    ref = out_bf.clone()
                rel = 0.0
            if not quant and (bf_best is None or us < bf_best):
                bf_best = us
            rows.append([tag, S, ql, kl, int(quant), td, wp, st, round(us, 1),
                         "", round(rel, 4), "OK"])
            print(f"[{name}] {tag}: {us:.1f}us rel={rel:.4f}", flush=True)
        for r in rows:
            if r[11] == "OK" and bf_best and r[8]:
                r[9] = round(bf_best / float(r[8]), 3)
            w.writerow(r)
        f.flush(); os.fsync(f.fileno())
    f.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
