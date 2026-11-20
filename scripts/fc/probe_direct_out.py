#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F-c 探测矩阵：一次性 GPU run 反推 fwd_kvcache 的 out 合同（一次性取证脚本）。

背景：T0 v2 仍炸 `out must have shape (batch_size, seqlen_q, num_heads,
head_size_og)`，但 out=(32,1,16,128)=empty_like(q) 与静态反汇编推出的合同
（4 维、逐维==q、同 dtype、cuda、stride(-1)==1）完全一致 → 存在静态看不到
的运行期因素。本脚本一次跑完形状×位置矩阵，逐 variant 打印
通过/错误串，通过的做 wrapper 基准 bitwise 对拍。

矩阵（每项独立 try/except）：
  V1  out=4D q形（已知失败，作对照——若这次过了说明前两跑另有隐情）
  V2  out=3D (32,16,128)
  V3  q 改 3D (32,16,128) + out 3D 同构
  V4  q 改 3D + out 4D q形
  V5  out 放到 python 位 20（s_aux 槽）、位 12 传 None（位置假设）
  V6  out=(1,32,16,128)
  V7  out=4D q形但 stride(-1)≠1 —— 期待报 contiguous 错（证明 out 到达检查链）
  V8  out=4D q形但 dtype=fp16 —— 期待报 dtype 错（同上）
  V9  out=(32,1,2,128)（num_heads 用 KV 头数的合同假设）
"""
import hashlib
import sys

import torch

BS, HQ, HKV, D = 32, 16, 2, 128
PAGE, SEQ = 16, 1024
NBLOCKS = BS * (SEQ // PAGE) + 8
SCALE = D ** -0.5


def build():
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(BS, 1, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    kc = torch.randn(NBLOCKS, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    vc = torch.randn(NBLOCKS, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    bt = torch.arange(BS * (SEQ // PAGE), device="cuda", dtype=torch.int32).reshape(BS, -1)
    sl = torch.full((BS,), SEQ, device="cuda", dtype=torch.int32)
    return q, kc, vc, bt, sl


def base_call(fac, q, kc, vc, bt, sl, out12, out20):
    return fac.fwd_kvcache(
        q, kc, vc, None, None, sl, None, None, None, None, bt, None,
        out12, SCALE, True, -1, -1, 0.0, True, 0, out20,
    )


def report(tag, fn, ref_flat):
    try:
        r = fn()
        torch.cuda.synchronize()
        out = r[0] if isinstance(r, (list, tuple)) else r
        eq = "n/a"
        if out is not None and ref_flat is not None:
            eq = str(torch.equal(ref_flat, out.reshape(-1).float()))
        print(f"[{tag}] OK  ret0.shape={tuple(out.shape)} dtype={out.dtype} "
              f"bitwise_vs_ref={eq}")
    except Exception as e:
        print(f"[{tag}] ERR {type(e).__name__}: {e}")


def main():
    import flash_attn_2_cuda as fac
    print("torch", torch.__version__, "| flash_attn_2_cuda:", fac.__file__)
    with open(fac.__file__, "rb") as f:
        b = f.read()
    print("so size", len(b), "md5", hashlib.md5(b).hexdigest()[:16])
    print("doc:", " ".join(fac.fwd_kvcache.__doc__.split())[:200])

    q, kc, vc, bt, sl = build()
    from flash_attn import flash_attn_with_kvcache
    ref = flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, block_table=bt,
                                  cache_seqlens=sl, softmax_scale=SCALE, causal=True)
    ref_flat = ref.reshape(-1).float()
    print("wrapper ref:", tuple(ref.shape), ref.stride(), ref.dtype)

    q3 = q.reshape(BS, HQ, D)

    def v1():
        return base_call(fac, q, kc, vc, bt, sl, torch.empty_like(q), None)
    def v2():
        return base_call(fac, q, kc, vc, bt, sl, torch.empty(BS, HQ, D, device="cuda", dtype=torch.bfloat16), None)
    def v3():
        return base_call(fac, q3, kc, vc, bt, sl, torch.empty(BS, HQ, D, device="cuda", dtype=torch.bfloat16), None)
    def v4():
        return base_call(fac, q3, kc, vc, bt, sl, torch.empty_like(q), None)
    def v5():
        return base_call(fac, q, kc, vc, bt, sl, None, torch.empty_like(q))
    def v6():
        return base_call(fac, q, kc, vc, bt, sl, torch.empty(1, BS, HQ, D, device="cuda", dtype=torch.bfloat16), None)
    def v7():
        o = torch.empty(HQ, D, BS, 1, device="cuda", dtype=torch.bfloat16).permute(2, 3, 0, 1)
        return base_call(fac, q, kc, vc, bt, sl, o, None)
    def v8():
        return base_call(fac, q, kc, vc, bt, sl, torch.empty_like(q, dtype=torch.float16), None)
    def v9():
        return base_call(fac, q, kc, vc, bt, sl, torch.empty(BS, 1, HKV, D, device="cuda", dtype=torch.bfloat16), None)

    for tag, fn in [("V1 4Dq形", v1), ("V2 3D", v2), ("V3 q3D+out3D", v3),
                    ("V4 q3D+out4D", v4), ("V5 out@pos20", v5), ("V6 (1,32,16,128)", v6),
                    ("V7 stride(-1)!=1", v7), ("V8 dtype=fp16", v8), ("V9 (32,1,2,128)", v9)]:
        report(tag, fn, ref_flat)


if __name__ == "__main__":
    main()
