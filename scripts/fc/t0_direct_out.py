#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F-c T0：flash decode 直写 out 的位精确对拍 + cudagraph 相容（需 GPU，主会话跑）。

对照三臂：
  A（基准）= flash_attn.flash_attn_with_kvcache 现役 wrapper 路径；
  B（补丁）= vllm_fl ... flash_attn.py::_flash_kvcache_direct_out 直写 out；
  C（图内）= B 包进 torch.cuda.CUDAGraph 捕获后 replay 两次。
判据：B/C 与 A 逐位相等（torch.equal），且 C 两次 replay 结果一致、且与
静态 buffer 地址一致（输出写进预分配 out，不新分配）。

用法（GPU 机）：
  cd <repo>/vllm-plugin-FL && pip install --no-deps --no-build-isolation . \
    && python3 ../scripts/fc/t0_direct_out.py
"""
import sys

import torch

BS, HQ, HKV, D = 32, 16, 2, 128   # MiniCPM5-2B decode 形状
PAGE, SEQ = 16, 1024              # 与 serve 同页块；seq 取 1K 覆盖 decode 常态
NBLOCKS = BS * (SEQ // PAGE) + 8
SCALE = D ** -0.5

sys.path.insert(0, "/data/flagos_feat_fc/vllm-plugin-FL")


def build(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(BS, 1, HQ, D, device="cuda", dtype=torch.bfloat16, generator=g)
    kc = torch.randn(NBLOCKS, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    vc = torch.randn(NBLOCKS, PAGE, HKV, D, device="cuda", dtype=torch.bfloat16, generator=g)
    # 每序列用不相交页，保证确定性
    bt = torch.arange(BS * (SEQ // PAGE), device="cuda", dtype=torch.int32).reshape(BS, -1)
    sl = torch.full((BS,), SEQ, device="cuda", dtype=torch.int32)
    return q, kc, vc, bt, sl


def arm_a(q, kc, vc, bt, sl):
    from flash_attn import flash_attn_with_kvcache
    return flash_attn_with_kvcache(
        q=q, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=sl,
        softmax_scale=SCALE, causal=True,
    )


def arm_b(out, q, kc, vc, bt, sl):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import (
        _flash_kvcache_direct_out,
    )
    _flash_kvcache_direct_out(
        q, kc, vc, bt, sl, SCALE, (-1, -1), 0.0, None, out,
    )


def main():
    assert torch.cuda.is_available()
    q, kc, vc, bt, sl = build()
    ref = arm_a(q, kc, vc, bt, sl).clone()

    out = torch.empty_like(ref)
    arm_b(out, q, kc, vc, bt, sl)
    torch.cuda.synchronize()
    ok_b = torch.equal(ref, out)
    print(f"[B] direct-out vs wrapper bitwise: {'PASS' if ok_b else 'FAIL'}")
    if not ok_b:
        diff = (ref.float() - out.float()).abs()
        print(f"    max_abs={diff.max().item():.3e} mean_abs={diff.mean().item():.3e}")
        sys.exit(1)

    # C: cudagraph 捕获 + 两次 replay（换输入数值验证真重算、地址不变）
    static_q = q.clone()
    out_c = torch.empty_like(ref)
    g = torch.cuda.CUDAGraph()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        arm_b(out_c, static_q, kc, vc, bt, sl)
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(g):
        arm_b(out_c, static_q, kc, vc, bt, sl)
    ptr_before = out_c.data_ptr()
    static_q.copy_(q); g.replay(); torch.cuda.synchronize()
    ok1 = torch.equal(ref, out_c)
    static_q.copy_(q); g.replay(); torch.cuda.synchronize()
    ok2 = torch.equal(ref, out_c) and out_c.data_ptr() == ptr_before
    print(f"[C] cudagraph replay bitwise ×2 + 静态地址: "
          f"{'PASS' if (ok1 and ok2) else 'FAIL'} (r1={'ok' if ok1 else 'bad'}, "
          f"r2={'ok' if ok2 else 'bad'})")

    print("T0", "PASS" if (ok_b and ok1 and ok2) else "FAIL")


if __name__ == "__main__":
    main()
