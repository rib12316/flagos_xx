# Copyright (c) 2026 BAAI. All rights reserved.
"""⑥ 自研 split-KV decode kernel 单测（无卡写好**不跑**，真机 T0 跑）。

════════════════════════════════════════════════════════════════════════
真机 T0 命令序列（15min 时间盒，照抄即用；详见交接书 §8 与 debug 预案）
════════════════════════════════════════════════════════════════════════
  cd /flagos_比赛/flagos_xx
  pip install --no-deps --no-build-isolation ./vllm-plugin-FL
  # 1) smoke 层（~2-3min，编译错误在此暴露）：
  python -m pytest vllm-plugin-FL/tests/unit_tests/attention/test_splitkv_decode.py \
      -k smoke -x -q
  # 2) smoke 全过 → 边界+对拍全量（~8min）：
  python -m pytest vllm-plugin-FL/tests/unit_tests/attention/test_splitkv_decode.py -q
  # 3) 失败时二分路径（按序，勿跳步）：
  #    a. 只跑 writethrough 臂:  -k writethrough     → 隔离 stage1（无归约）
  #    b. 只跑乱序表臂:          -k scatter          → 隔离页表 gather
  #    c. 只跑边界 kv:           -k boundary_kv      → masking off-by-one
  #    d. 失败输出已含 per-(token,head) partial dump → 按归并公式手验
  #    e. 三方仲裁（本文件自带 torch 参考实现）判断 kernel 错还是对照错

判据（§7/§9）：对照 unified_attention（同 segm/TD 态）与 torch 参考实现，
max_rel ≤1%、无 NaN。均匀 ~1e-3 级差异 = floating 归约顺序（放过）；
个别 (token,head) 大误差 = 真 bug（dump 定位）。
════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import math
import os
import sys

import pytest

torch = pytest.importorskip("torch")

# 无卡环境（无 CUDA / 无 triton）自动跳过收集，不炸（无卡纪律）
if not torch.cuda.is_available():
    pytest.skip("无 GPU：单测真机跑（交接书 §8 T0）", allow_module_level=True)

NH, NKV, HD, BLOCK = 16, 2, 128, 16          # MiniCPM5-2B（§3）
DEV = "cuda"
VARIANTS = (0, 1, 2, 3)


# ---------------------------------------------------------------------------
# 用例构造（与 scripts/d2/attn_surface_probe.py attn_case5 同构）
# ---------------------------------------------------------------------------
def _make_case(bs, kv, seed=0, scatter=False, q_pad=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(bs, NH, HD, generator=g, device=DEV, dtype=torch.bfloat16)
    if q_pad:  # 非连续 head 布局（stride 传参正确性）
        buf = torch.empty(bs, NH, HD + q_pad, device=DEV, dtype=torch.bfloat16)
        buf[:, :, :HD] = q
        q = buf[:, :, :HD]
    nb = math.ceil(kv / BLOCK) * bs + 4
    kv_cache = torch.randn(
        2, nb, BLOCK, NKV, HD, generator=g, device=DEV, dtype=torch.bfloat16)
    if scatter:  # 乱序页表（serve 分配器散射口径）
        bt = torch.randperm(nb - 4, generator=g).to(torch.int32).to(DEV)
        bt = bt[: bs * math.ceil(kv / BLOCK)].reshape(bs, -1)
    else:
        bt = torch.arange(bs * math.ceil(kv / BLOCK), dtype=torch.int32,
                          device=DEV).reshape(bs, -1)
    seq_lens = torch.full((bs,), kv, dtype=torch.int32, device=DEV)
    cu = torch.arange(0, bs + 1, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    return dict(q=q, k=kv_cache[0], v=kv_cache[1], out=out, bt=bt,
                seq_lens=seq_lens, cu=cu, sm_scale=HD ** -0.5, nb=nb)


def _torch_reference(case):
    """纯 torch decode attention 参考实现（三方仲裁用，不依赖 triton）。

    逐序列 gather 全部 K/V 到稠密矩阵做标准 softmax——慢但无歧义。"""
    q, k, v, bt, sl = (case["q"], case["k"], case["v"], case["bt"],
                       case["seq_lens"])
    outs = []
    for i in range(q.shape[0]):
        n = int(sl[i])
        pages = bt[i, : (n + BLOCK - 1) // BLOCK].to(torch.long)
        kk = k[pages].reshape(-1, NKV, HD)[:, :, :]        # [n, NKV, HD]
        vv = v[pages].reshape(-1, NKV, HD)
        qi = q[i].float()                                   # [NH, HD]
        # GQA: q head h → kv head h // (NH // NKV)
        kexp = kk.repeat_interleave(NH // NKV, dim=1)       # [n, NH, HD]
        vexp = vv.repeat_interleave(NH // NKV, dim=1)
        s = torch.einsum("hd,nhd->hn", qi, kexp.float()) * case["sm_scale"]
        p = torch.softmax(s, dim=-1)
        o = torch.einsum("hn,nhd->hd", p, vexp.float())
        outs.append(o.to(q.dtype))
    return torch.stack(outs)


def _run_splitkv(case, variant, use_td=True, splits=None):
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
    old = os.environ.get("VLLM_FL_ATTN_SPLITKV_SPLITS")
    if splits is not None:
        os.environ["VLLM_FL_ATTN_SPLITKV_SPLITS"] = str(splits)
    try:
        skv.splitkv_decode_attention(
            case["q"], case["k"], case["v"], case["out"], case["cu"],
            case["seq_lens"], case["bt"], case["sm_scale"],
            variant=variant, use_td=use_td)
        nseq = case["seq_lens"].shape[0]
        mkv = splits * BLOCK * 64 if splits else max(
            int(case["seq_lens"].max()), 0) or (
                case["bt"].shape[1] * BLOCK)
        return skv.resolve_split(nseq, mkv)
    finally:
        if old is None:
            os.environ.pop("VLLM_FL_ATTN_SPLITKV_SPLITS", None)
        else:
            os.environ["VLLM_FL_ATTN_SPLITKV_SPLITS"] = old


def _run_upstream(case, segments=16, use_td=True):
    """上游 unified_attention（同 segm/TD 态对照，§7 口径）。"""
    import vllm.v1.attention.ops.triton_unified_attention as tua
    bs = case["q"].shape[0]
    segm_out = torch.empty(bs, NH, segments, HD, dtype=torch.float32, device=DEV)
    segm_max = torch.empty(bs, NH, segments, dtype=torch.float32, device=DEV)
    segm_sum = torch.empty_like(segm_max)
    tua.unified_attention(
        case["q"], case["k"], case["v"], case["out"], cu_seqlens_q=case["cu"],
        max_seqlen_q=1, seqused_k=case["seq_lens"],
        max_seqlen_k=int(case["seq_lens"].max()), block_table=case["bt"],
        softmax_scale=case["sm_scale"], causal=True, window_size=(-1, -1),
        softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
        use_td=use_td, seq_threshold_3D=64, num_par_softmax_segments=segments,
        softmax_segm_output=segm_out, softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_sum)
    return case["out"]


def _rel_stats(got, ref):
    g, r = got.float(), ref.float()
    denom = r.abs().clamp_min(1e-3)
    rel = (g - r).abs() / denom
    return rel


def _dump_case(case, variant, split, got, ref, name):
    """失败即诊断：一次跑出定位信息，避免真机上加打印重跑。"""
    rel = _rel_stats(got, ref)
    flat = rel.flatten()
    print(f"[FAIL:{name}] variant={variant} split={split} "
          f"shape={tuple(got.shape)} nan={int(torch.isnan(got).sum())} "
          f"inf={int(torch.isinf(got).sum())}", flush=True)
    print(f"[FAIL:{name}] max_rel 分布: mean={flat.mean():.2e} "
          f"p50={flat.median():.2e} p99={flat.quantile(0.99):.2e} "
          f"max={flat.max():.2e}（均匀 ~1e-3=floating 顺序差异可放过；"
          f"个别大误差=真 bug）", flush=True)
    # top-5 分歧 (token, head) + 该点的 split partials
    tok_head = rel.amax(dim=-1)                       # [bs, NH]
    vals, idxs = tok_head.flatten().topk(5)
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
        buf = skv._BUFFER_CACHE.get(0)
    except Exception:
        buf = None
    for val, idx in zip(vals.tolist(), idxs.tolist()):
        t, h = divmod(idx, NH)
        line = (f"[FAIL:{name}]  token={t} head={h} rel={val:.3e} "
                f"got={got[t, h, :4].tolist()} ref={ref[t, h, :4].tolist()}")
        if buf is not None and not math.isinf(val):
            mm = buf["max"][t, h].tolist()
            ll = buf["expsum"][t, h].tolist()
            live = [i for i in range(split) if ll[i] > 0.0]
            line += f" | live_splits={live[:8]}{'...' if len(live) > 8 else ''}"
            for sidx in live[:4]:
                line += (f" s{sidx}:M={mm[sidx]:.3e},L={ll[sidx]:.3e},"
                         f"|acc|={buf['out'][t, h, sidx].norm():.3e}")
        print(line, flush=True)


def _assert_parity(got, ref, variant, split, name, tol=1e-2):
    rel = _rel_stats(got, ref)
    assert not torch.isnan(got).any(), f"{name}: NaN（见 dump）"
    assert not torch.isinf(got).any(), f"{name}: Inf"
    if rel.max().item() > tol or torch.isnan(rel).any():
        _dump_case(None, variant, split, got, ref, name)
        pytest.fail(f"{name}: max_rel={rel.max():.3e} > {tol:g}")


# ---------------------------------------------------------------------------
# 1) smoke 层（T0 第一步，~2-3min，编译错误在此暴露）
# ---------------------------------------------------------------------------
@pytest.mark.smoke
@pytest.mark.parametrize("variant", VARIANTS)
def test_smoke_vs_upstream_16k(variant):
    case = _make_case(32, 16384, seed=1)
    split = _run_splitkv(case, variant)
    got = case["out"].clone()
    ref = _run_upstream(_make_case(32, 16384, seed=1))
    _assert_parity(got, ref, variant, split, f"smoke16k-v{variant}")


@pytest.mark.smoke
@pytest.mark.parametrize("variant", VARIANTS)
def test_smoke_vs_upstream_4k(variant):
    case = _make_case(8, 4096, seed=2)
    split = _run_splitkv(case, variant)
    got = case["out"].clone()
    ref = _run_upstream(_make_case(8, 4096, seed=2))
    _assert_parity(got, ref, variant, split, f"smoke4k-v{variant}")


@pytest.mark.smoke
def test_smoke_vs_torch_reference():
    """三方仲裁臂：kernel vs 纯 torch 参考（判定「对照错了」的盲调试）。"""
    case = _make_case(8, 2048, seed=3)
    split = _run_splitkv(case, variant=1)
    got = case["out"].clone()
    ref = _torch_reference(case)
    _assert_parity(got, ref, 1, split, "smoke-vs-torch", tol=1.5e-2)


# ---------------------------------------------------------------------------
# 2) writethrough 臂（split==1：隔离 stage1，无归约参与）
# ---------------------------------------------------------------------------
@pytest.mark.smoke
def test_smoke_writethrough():
    case = _make_case(4, 1024, seed=4)      # kv≤2048 → split=1 直写
    split = _run_splitkv(case, variant=1)
    assert split == 1, f"writethrough 臂应 split=1，得 {split}"
    got = case["out"].clone()
    _assert_parity(got, _torch_reference(case), 1, split, "writethrough",
                   tol=1.5e-2)


# ---------------------------------------------------------------------------
# 3) 边界层
# ---------------------------------------------------------------------------
@pytest.mark.boundary_kv
@pytest.mark.parametrize("kv", [1, 15, 16, 17, 4096, 16384, 17408])
@pytest.mark.parametrize("variant", [0, 1])
def test_boundary_kv(variant, kv):
    case = _make_case(8, kv, seed=10 + kv)
    split = _run_splitkv(case, variant)
    _assert_parity(case["out"].clone(), _torch_reference(case), variant, split,
                   f"kv{kv}-v{variant}", tol=1.5e-2)


@pytest.mark.boundary_bs
@pytest.mark.parametrize("bs", [1, 8, 32, 33, 64])
def test_boundary_bs(bs):
    case = _make_case(bs, 4096, seed=20 + bs)
    split = _run_splitkv(case, variant=1)
    _assert_parity(case["out"].clone(), _torch_reference(case), 1, split,
                   f"bs{bs}", tol=1.5e-2)


@pytest.mark.scatter
def test_boundary_scatter_block_table():
    """乱序页表（serve 分配器散射口径）+ 非连续 q 布局（stride 正确性）。"""
    for kv in (16, 16384):
        case = _make_case(32, kv, seed=30 + kv, scatter=True, q_pad=8)
        split = _run_splitkv(case, variant=1)
        _assert_parity(case["out"].clone(), _torch_reference(case), 1, split,
                       f"scatter-kv{kv}", tol=1.5e-2)


@pytest.mark.forced_splits
def test_forced_split32_short_kv():
    """强制 split=32 + 短 kv：act_num_segments < 32（空段掩码逻辑）。"""
    case = _make_case(4, 512, seed=40)      # 512/16/32=1 tile/段 → act=1
    split = _run_splitkv(case, variant=0, splits=32)
    assert split == 32
    _assert_parity(case["out"].clone(), _torch_reference(case), 0, split,
                   "forced32-short", tol=1.5e-2)


@pytest.mark.forced_splits
def test_forced_split32_odd_kv():
    """非 2 幂 kv（16500）：段边界不对齐 + 尾段部分 tile。"""
    case = _make_case(8, 16500, seed=41)
    split = _run_splitkv(case, variant=1, splits=32)
    _assert_parity(case["out"].clone(), _torch_reference(case), 1, split,
                   "forced32-16500", tol=1.5e-2)


# ---------------------------------------------------------------------------
# 4) 集成层（hop1 wrapper 路由）
# ---------------------------------------------------------------------------
@pytest.mark.wrapper
def test_wrapper_routes_and_parity():
    os.environ["VLLM_FL_ATTN_SPLITKV"] = "1"
    os.environ["VLLM_FL_ATTN_SPLITKV_VARIANT"] = "1"
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
    assert skv.install(), "hop1 未装载（env 已设 1？）"
    skv.set_active(True)
    skv.set_variant(1)
    case = _make_case(32, 16384, seed=50)
    import vllm.v1.attention.ops.triton_unified_attention as tua
    r0 = skv._route_stats["routed"]
    tua.unified_attention(
        case["q"], case["k"], case["v"], case["out"], cu_seqlens_q=case["cu"],
        max_seqlen_q=1, seqused_k=case["seq_lens"],
        max_seqlen_k=int(case["seq_lens"].max()), block_table=case["bt"],
        softmax_scale=case["sm_scale"], causal=True, window_size=(-1, -1),
        softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
        use_td=True)
    routed = skv._route_stats["routed"] > r0
    skv.set_active(None)
    skv.set_variant(None)
    assert routed, "应路由到自研 kernel（route_stats 未增长）"
    ref_case = _make_case(32, 16384, seed=50)
    _assert_parity(case["out"].clone(), _run_upstream(ref_case), 1, 32,
                   "wrapper-parity")


@pytest.mark.wrapper
def test_wrapper_passthrough_prefill():
    """q_len>1 必须透传（prefill 不拦）。"""
    os.environ["VLLM_FL_ATTN_SPLITKV"] = "1"
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
    skv.install()
    skv.set_active(True)
    import vllm.v1.attention.ops.triton_unified_attention as tua
    case = _make_case(2, 512, seed=51)
    p0 = skv._route_stats["passed"]
    q2 = torch.randn(4, NH, HD, device=DEV, dtype=torch.bfloat16)  # q_len=2
    cu2 = torch.tensor([0, 2, 4], dtype=torch.int32, device=DEV)
    out2 = torch.empty_like(q2)
    segm_out = torch.empty(4, NH, 16, HD, dtype=torch.float32, device=DEV)
    segm_max = torch.empty(4, NH, 16, dtype=torch.float32, device=DEV)
    tua.unified_attention(
        q2, case["k"], case["v"], out2, cu_seqlens_q=cu2, max_seqlen_q=2,
        seqused_k=torch.tensor([512, 512], dtype=torch.int32, device=DEV),
        max_seqlen_k=512, block_table=case["bt"][:2],
        softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
        softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
        use_td=False, seq_threshold_3D=64, num_par_softmax_segments=16,
        softmax_segm_output=segm_out, softmax_segm_max=segm_max,
        softmax_segm_expsum=torch.empty_like(segm_max))
    skv.set_active(None)
    assert skv._route_stats["passed"] > p0, "prefill 应透传"
    assert not torch.isnan(out2).any()
