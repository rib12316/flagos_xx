# P2 合体探针:P1(BM256/T128/W16/S2)× KV int8 的 prefill 税 + TD×quant 共存。
# 背景:kvquant_sweep 判决 kernel 级量化税 decode +12%(t16)/prefill +25%(默认
# tile,BM 关);自研 dequant 不值。剩余关键未知:
#   ① BM 大 tile 形态下量化税是否仍可控(TILE=128 每 tile scale 摊薄?)
#   ② USE_TD × KV_QUANT_MODE=2 是否编译/运行/数值共存(①×④互毁雷的正面验证)
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "d2"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "vllm-plugin-FL"))

import torch  # noqa: E402
import triton  # noqa: E402
import blockm_sweep as bs  # noqa: E402
import bottleneck_probe as bp  # noqa: E402
import kvquant_sweep as kq  # noqa: E402

DEV = "cuda"
OUT = os.path.abspath(os.path.join(_HERE, "..", "..", "results",
                                   "20261120_kvquant_sweep"))


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


def main():
    print("=== ① P1 形态 × quant prefill 税 (S1 q4096 kv16384) ===", flush=True)
    S, ql, kl = 1, 4096, 16384
    q, bf_pool, q_pool, bt = kq.build_case(S, ql, kl, seed=42)
    out = torch.empty_like(q)
    ref = None
    for tag, cfg, quant in [
        ("bf16_bm256_t128_w16s2", dict(bm=256, t=128, w=16, s=2), False),
        ("q8_bm256_t128_w16s2",   dict(bm=256, t=128, w=16, s=2), True),
        ("q8_bm256_t64_w16s2",    dict(bm=256, t=64, w=16, s=2), True),
        ("q8_bm256_t128_w8s2",    dict(bm=256, t=128, w=8, s=2), True),
    ]:
        bs.CFG.update(bs.ZERO); bs.CFG.update(cfg)
        call = kq.make_call(S, ql, kl, q, (bf_pool, q_pool), quant, out, bt)
        try:
            us = time_us(call)
            call(); torch.cuda.synchronize()
            if not quant:
                ref = out.clone()
                rel = 0.0
            else:
                rel = ((out.float() - ref.float()).abs().max()
                       / ref.float().abs().max().clamp(min=1e-6)).item()
            print(f"  {tag}: {us:.1f}us rel={rel:.4f}", flush=True)
        except Exception as e:
            print(f"  {tag}: ERR {str(e)[:90]}", flush=True)

    print("=== ② TD × quant decode 共存 (bs64/kv16384) ===", flush=True)
    S, ql, kl = 64, 1, 16384
    q, bf_pool, q_pool, bt = kq.build_case(S, ql, kl, seed=43)
    out = torch.empty_like(q)
    from vllm.v1.kv_cache_interface import KVQuantMode
    kbf, vbf = bf_pool
    k8, v8, ks, vs = q_pool
    cu = torch.arange(0, S + 1, 1, dtype=torch.int32, device=DEV)
    su = torch.full((S,), kl, dtype=torch.int32, device=DEV)
    segm = dict(seq_threshold_3D=64, num_par_softmax_segments=16,
                softmax_segm_output=torch.empty(64, 16, 16, 128,
                                                dtype=torch.float32, device=DEV),
                softmax_segm_max=torch.empty(64, 16, 16,
                                             dtype=torch.float32, device=DEV),
                softmax_segm_expsum=torch.empty(64, 16, 16,
                                                dtype=torch.float32, device=DEV))
    ref = None
    for tag, quant, use_td in [
        ("bf16_td", False, True), ("q8_td", True, True), ("q8_notd", True, False),
    ]:
        bs.CFG.update(bs.ZERO); bs.CFG.update(dict(td=16, w=1, s=1))
        base = dict(softmax_scale=128 ** -0.5, causal=True,
                    window_size=(-1, -1), softcap=0.0, q_descale=None,
                    k_descale=None, v_descale=None, use_td=use_td, **segm)

        def call(tag=tag, quant=quant, base=base):
            if quant:
                bs.UA(q, k8, v8, out, cu_seqlens_q=cu, max_seqlen_q=1,
                      seqused_k=su, max_seqlen_k=kl, block_table=bt,
                      kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
                      k_scale_cache=ks, v_scale_cache=vs, **base)
            else:
                bs.UA(q, kbf, vbf, out, cu_seqlens_q=cu, max_seqlen_q=1,
                      seqused_k=su, max_seqlen_k=kl, block_table=bt,
                      kv_quant_mode=KVQuantMode.NONE,
                      k_scale_cache=None, v_scale_cache=None, **base)
            return out
        try:
            us = time_us(call)
            call(); torch.cuda.synchronize()
            if ref is None:
                ref = out.clone()
                rel = 0.0
            else:
                rel = ((out.float() - ref.float()).abs().max()
                       / ref.float().abs().max().clamp(min=1e-6)).item()
            print(f"  {tag}: {us:.1f}us rel={rel:.4f}", flush=True)
        except Exception as e:
            print(f"  {tag}: ERR {str(e)[:90]}", flush=True)
    print("COMBO DONE", flush=True)


if __name__ == "__main__":
    main()
