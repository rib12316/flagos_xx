#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F-c T0：flash decode 直写 out 的位精确对拍 + cudagraph 相容（需 GPU，主会话跑）。

v2（11-20 13:2x 修正）：首跑失败根因=arm_b 的 out 用了 empty_like(wrapper返回值)——
MACA wrapper 对 seqlen_q==1 返回 **3D [bs,H,D]**（vllm 侧
reshape_attn_output_for_spec_decode 的 dim()==3 分支即为此存在），而 C++ 侧
TORCH_CHECK 要求 out 与 q 同为 4 元素形状 (bs, seqlen_q, H, D)
（.so 反汇编：out.dtype==q.dtype / is_cuda / stride(-1)==1 /
ArrayRef::equals(out.sizes(), {q0,q1,q2,q3}, len=4)）。
v2 一律传 `torch.empty_like(q)`（4D q 形，与 serve 补丁
`output[:n].view(*decode_query.shape)` 同构），对拍按 flatten 逐位比。

对照三臂：
  A（基准）= flash_attn.flash_attn_with_kvcache 现役 wrapper 路径；
  B（补丁）= vllm_fl ... flash_attn.py::_flash_kvcache_direct_out 直写 out；
  C（图内）= B 包进 torch.cuda.CUDAGraph 捕获后 replay 两次。
判据：B/C 与 A 逐位相等（flatten 后 torch.equal），C 两次 replay 结果一致、
输出地址不变（写进预分配 out，不新分配）。

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


def arm_b(out4d, q, kc, vc, bt, sl):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import (
        _flash_kvcache_direct_out,
    )
    _flash_kvcache_direct_out(
        q, kc, vc, bt, sl, SCALE, (-1, -1), 0.0, None, out4d,
    )


def main():
    assert torch.cuda.is_available()
    q, kc, vc, bt, sl = build()
    ref = arm_a(q, kc, vc, bt, sl)
    print(f"[A] wrapper 返回 shape={tuple(ref.shape)} stride={ref.stride()} "
          f"dtype={ref.dtype}")

    out = torch.empty_like(q)          # 4D (bs, seqlen_q, H, D) —— 与 serve 补丁同构
    arm_b(out, q, kc, vc, bt, sl)
    torch.cuda.synchronize()
    ok_b = torch.equal(ref.reshape(-1), out.reshape(-1))
    print(f"[B] direct-out(4D q形) vs wrapper flatten bitwise: "
          f"{'PASS' if ok_b else 'FAIL'} (out shape={tuple(out.shape)})")
    if not ok_b:
        d = (ref.float().reshape(-1) - out.float().reshape(-1)).abs()
        print(f"    max_abs={d.max().item():.3e} mean_abs={d.mean().item():.3e}")
        return 1

    # C: cudagraph 捕获 + 两次 replay（换输入数值验证真重算、地址不变）
    static_q = q.clone()
    out_c = torch.empty_like(q)
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
    ok1 = torch.equal(ref.reshape(-1), out_c.reshape(-1))
    static_q.copy_(q); g.replay(); torch.cuda.synchronize()
    ok2 = torch.equal(ref.reshape(-1), out_c.reshape(-1)) and out_c.data_ptr() == ptr_before
    print(f"[C] cudagraph replay bitwise ×2 + 静态地址: "
          f"{'PASS' if (ok1 and ok2) else 'FAIL'} (r1={'ok' if ok1 else 'bad'}, "
          f"r2={'ok' if ok2 else 'bad'})")

    passed = ok_b and ok1 and ok2
    print("T0", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
