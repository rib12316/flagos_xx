# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""⑥ split-KV metax hop：路由判定表 + flash vs splitkv 数值 parity。

真机命令（沐曦实例，GPU 空闲时）：
    VLLM_FL_ATTN_SPLITKV=1 python3 -m pytest \
        vllm-plugin-FL/tests/unit_tests/attention/test_splitkv_metax_hop.py -x -q
"""

import os

import pytest
import torch

os.environ.setdefault("VLLM_FL_ATTN_SPLITKV", "1")

NH, NKV, HD, BLOCK = 16, 2, 128, 16


def _mk(S, kv_len, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(S, NH, HD, device="cuda", dtype=torch.bfloat16, generator=g)
    nb = (kv_len + BLOCK - 1) // BLOCK * S + 4
    kc = torch.randn(nb, BLOCK, NKV, HD, device="cuda", dtype=torch.bfloat16,
                     generator=g)
    vc = torch.randn(nb, BLOCK, NKV, HD, device="cuda", dtype=torch.bfloat16,
                     generator=g)
    cols = (kv_len + BLOCK - 1) // BLOCK
    bt = torch.randperm(nb, device="cuda")[: S * cols].to(torch.int32).reshape(S, -1)
    su = torch.full((S,), kv_len, dtype=torch.int32, device="cuda")
    return q, kc, vc, bt, su


def _call(q, kc, vc, bt, su):
    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import (
        flash_attn as _fa,
    )
    return _fa.flash_attn_with_kvcache(
        q=q, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=su,
        softmax_scale=HD ** -0.5, causal=True, window_size=None,
        alibi_slopes=None, softcap=None, s_aux=None)


@pytest.fixture(scope="module")
def hop():
    from vllm_fl.dispatch.backends.vendor.metax import attn_splitkv_hop
    assert attn_splitkv_hop.install(), "hop 未装载"
    yield attn_splitkv_hop


@pytest.mark.parametrize("S,kv", [(8, 2048), (8, 8192), (32, 16384),
                                  (32, 17408), (64, 17408), (64, 4096)])
@pytest.mark.parametrize("variant", [0, 1, 2, 3])
def test_parity_vs_flash(hop, S, kv, variant):
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
    q, kc, vc, bt, su = _mk(S, kv, seed=S * 7 + kv)
    try:
        skv.set_active(False)
        base = _call(q, kc, vc, bt, su).float()
        skv.set_active(True)
        skv.set_variant(variant)
        out = _call(q, kc, vc, bt, su).float()
    finally:
        skv.set_active(None)
        skv.set_variant(None)
    rel = (out - base).abs().max().item() / max(base.abs().max().item(), 1e-6)
    assert rel < 2e-2, f"S={S} kv={kv} v={variant} maxrel={rel:.3e}"
    assert not torch.isnan(out).any()


def test_route_table(hop):
    """应拦 / 应透传各形态（CPU 侧静态量判定）。"""
    q, kc, vc, bt, su = _mk(32, 16384)
    base_hit = dict(q=q, k_cache=kc, v_cache=vc, cache_seqlens=su,
                    causal=True, window_size=None, alibi_slopes=None,
                    softcap=None, s_aux=None)
    assert hop._route_decision_flash(base_hit) is None

    q2 = torch.randn(64, NH, HD, device="cuda", dtype=torch.bfloat16)
    miss = dict(base_hit, q=q2, cache_seqlens=su)  # q_len 64 tokens / 32 seqs
    assert hop._route_decision_flash(miss) is not None
    assert hop._route_decision_flash(dict(base_hit, causal=False)) is not None
    assert hop._route_decision_flash(dict(base_hit, softcap=0.5)) is not None
    ws = (-1, 0)
    assert hop._route_decision_flash(dict(base_hit, window_size=ws)) is not None
    kc32 = torch.randn(kc.shape[0], 32, NKV, HD, device="cuda",
                       dtype=torch.bfloat16)
    vc32 = torch.randn_like(kc32)
    assert hop._route_decision_flash(
        dict(base_hit, k_cache=kc32, v_cache=vc32)) is not None  # block_size


def test_cudagraph_capture_replay(hop):
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
    S, kv = 32, 16384
    q, kc, vc, bt, su = _mk(S, kv, seed=3)
    out = torch.empty((S, NH, HD), device="cuda", dtype=torch.bfloat16)

    def call():
        r = _call(q, kc, vc, bt, su)
        out.copy_(r)

    try:
        skv.set_active(True)
        skv.set_variant(1)
        call()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        g.replay()
        torch.cuda.synchronize()
        skv.set_active(False)
        ref = _call(q, kc, vc, bt, su).float()
    finally:
        skv.set_active(None)
        skv.set_variant(None)
    rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert rel < 2e-2, f"replay maxrel={rel:.3e}"
