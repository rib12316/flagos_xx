#!/usr/bin/env python3
"""D0 T0 复证 micro-bench：unified_attention launch 调优三臂对照（进程内 monkeypatch）。

复证对象（废弃仓库 flagos2_infragogogo 口径，D3a/D3c 实测）：
  - decode  3D launch (len(grid)==3): num_warps=1 / num_stages=1
  - prefill 2D launch (len(grid)==2): num_warps=2 / num_stages=1
  - 当年微基准：decode 4K 形状 1970.7→1164.7µs (1.69×, bit-exact)；
    prefill 8×1024q×kv4096 69.9→46.5ms (1.50×)

本脚本三臂 = default（triton auto，不注入）vs w1s1 vs w2s1，两条路径各跑一遍：
  - decode : bs=64, q_len=1, kv_len∈{4096,16384}, 16Q/2KV 头, head_dim=128,
             bf16, block=16, seg=16（MiniCPM5-2B decode 形状口径）
  - prefill: 8seq×1024q, kv_len∈{4096,16384}（chunked prefill 形状口径）

输出：每臂 per-call 耗时 µs + 名义有效带宽 GB/s（KV 读字节/时间）、D2D copy
自校准、bit-exact 校验。RESULT| 单行可 grep。
判据：调优臂 vs default 在其路径上的带宽差 ≥10% → 复证通过。
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import torch

import vllm.v1.attention.ops.triton_unified_attention as tua
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

DEV = "cuda"
DTYPE = torch.bfloat16
NH, NKV, HD, BLOCK, NSEG = 16, 2, 128, 16, 16  # MiniCPM5-2B decode 口径
SEQ_THRESHOLD_3D = 64

torch.manual_seed(0)


# ---------------------------------------------------------------- monkeypatch
def unpatch_existing():
    """若环境里已装了旧 patch（_fl_iluvatar_orig 标记），先还原为上游 JITFunction。"""
    k = tua.kernel_unified_attention
    orig = getattr(k, "_fl_iluvatar_orig", None)
    if orig is not None:
        tua.kernel_unified_attention = orig
        print("[bench] 检测到已安装的旧 launch patch，已还原为上游 kernel（default 臂不受污染）")
    return orig is not None


class LaunchProxy:
    """进程内 monkeypatch：按 grid 维度注入 num_warps/num_stages（同旧仓库机制）。"""

    def __init__(self, jit, cfg3d, cfg2d):
        self._jit, self._cfg3d, self._cfg2d = jit, cfg3d, cfg2d

    def __getitem__(self, grid):
        cfg = self._cfg3d if len(grid) == 3 else self._cfg2d
        sub = self._jit[grid]
        if cfg is None:
            return sub

        def _launch(*args, **kwargs):
            w, s = cfg
            if w > 0:
                kwargs.setdefault("num_warps", w)
            if s > 0:
                kwargs.setdefault("num_stages", s)
            return self._jit.run(grid=grid, warmup=False, *args, **kwargs)

        return _launch

    def __getattr__(self, name):
        return getattr(self._jit, name)


def install(cfg3d, cfg2d):
    jit = getattr(tua.kernel_unified_attention, "_fl_iluvatar_orig", tua.kernel_unified_attention)
    proxy = LaunchProxy(jit, cfg3d, cfg2d)
    proxy._fl_iluvatar_orig = jit
    tua.kernel_unified_attention = proxy


# ---------------------------------------------------------------- 用例构造
def make_decode_case(bs, kv_len):
    q = torch.randn(bs, NH, HD, dtype=DTYPE, device=DEV)
    npages = (kv_len + BLOCK - 1) // BLOCK
    kv = torch.randn(2, bs * npages + 64, BLOCK, NKV, HD, dtype=DTYPE, device=DEV)
    bt = torch.arange(bs * npages, dtype=torch.int32, device=DEV).reshape(bs, npages)
    cu = torch.arange(0, bs + 1, dtype=torch.int32, device=DEV)
    seqused = torch.full((bs,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty(bs, NH, HD, dtype=DTYPE, device=DEV)
    segm_out = torch.empty(SEQ_THRESHOLD_3D, NH, NSEG, HD, dtype=torch.float32, device=DEV)
    segm_max = torch.empty(SEQ_THRESHOLD_3D, NH, NSEG, dtype=torch.float32, device=DEV)
    segm_sum = torch.empty(SEQ_THRESHOLD_3D, NH, NSEG, dtype=torch.float32, device=DEV)
    kwargs = dict(
        q=q, k=kv[0], v=kv[1], out=out, cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=HD**-0.5,
        causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=NSEG,
        softmax_segm_output=segm_out, softmax_segm_max=segm_max,
        softmax_segm_expsum=segm_sum,
    )
    return kwargs, out


def make_prefill_case(nseq, qlen, kv_len):
    ntok = nseq * qlen
    q = torch.randn(ntok, NH, HD, dtype=DTYPE, device=DEV)
    npages = (kv_len + BLOCK - 1) // BLOCK
    kv = torch.randn(2, nseq * npages + 64, BLOCK, NKV, HD, dtype=DTYPE, device=DEV)
    bt = torch.arange(nseq * npages, dtype=torch.int32, device=DEV).reshape(nseq, npages)
    cu = torch.arange(0, ntok + 1, qlen, dtype=torch.int32, device=DEV)
    seqused = torch.full((nseq,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty(ntok, NH, HD, dtype=DTYPE, device=DEV)
    kwargs = dict(
        q=q, k=kv[0], v=kv[1], out=out, cu_seqlens_q=cu, max_seqlen_q=qlen,
        seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=HD**-0.5,
        causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=None, num_par_softmax_segments=None,
        softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None,
    )
    return kwargs, out


def bench_call(kwargs, reps, warmup=10):
    for _ in range(warmup):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(reps):
        unified_attention(**kwargs)
    ev1.record()
    torch.cuda.synchronize()
    return ev0.elapsed_time(ev1) * 1e3 / reps  # µs per call


def d2d_calibration(nbytes):
    src = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=DEV)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(20):
        dst.copy_(src)
    ev1.record()
    torch.cuda.synchronize()
    ms = ev0.elapsed_time(ev1) / 20
    return 2 * nbytes / (ms * 1e-3) / 1e9  # GB/s（读+写合计）


ARMS = [("default", None), ("w1s1", (1, 1)), ("w2s1", (2, 1))]


def run_path(name, maker, shape_args, kv_read_bytes, reps):
    results = {}
    outs = {}
    # 同一输入跨臂复用（bit-exact 才有意义）：随机数据只构造一次
    torch.manual_seed(0)
    kwargs, out = maker(*shape_args)
    for arm, cfg in ARMS:
        unpatch_existing()
        if cfg is not None:
            install(cfg3d=cfg if name == "decode" else None,
                    cfg2d=cfg if name == "prefill" else None)
        us = bench_call(kwargs, reps)
        results[arm] = us
        outs[arm] = out.clone()
        print(f"  {name} kv={shape_args[-1]:>6} arm={arm:<7} {us:10.1f} µs "
              f"{kv_read_bytes / us * 1e6 / 1e9:7.1f} GB/s")
    unpatch_existing()
    for arm in ("w1s1", "w2s1"):
        diff = (outs[arm].float() - outs["default"].float()).abs().max().item()
        print(f"  bit-exact[{arm} vs default]: max_abs_diff={diff}")
    for arm in ("w1s1", "w2s1"):
        base, tune = results["default"], results[arm]
        gain = (base - tune) / base * 100
        bw_gain = (base / tune - 1) * 100
        print(f"RESULT|bench={name}|kv={shape_args[-1]}|arm={arm}"
              f"|default_us={base:.1f}|tuned_us={tune:.1f}"
              f"|speedup={base / tune:.3f}x|bw_gain_pct={bw_gain:.1f}"
              f"|GBps_default={kv_read_bytes / base * 1e6 / 1e9:.1f}"
              f"|GBps_tuned={kv_read_bytes / tune * 1e6 / 1e9:.1f}")
    return results


def main():
    print(f"环境: torch={torch.__version__} cuda_dev={torch.cuda.get_device_name(0)}")
    import triton
    print(f"triton={triton.__version__}")
    print(f"形状: {NH}Q/{NKV}KV 头 × head_dim {HD}, bf16, block={BLOCK}, seg={NSEG}")

    # D2D 自校准（用 4K decode KV 读量做参考容量）
    ref_bytes = 64 * 4096 * 2 * NKV * HD * 2
    bw = d2d_calibration(ref_bytes)
    print(f"自校准: D2D copy 带宽（读+写）= {bw:.1f} GB/s")

    all_res = {}
    for kv in (4096, 16384):
        kwargs, _ = make_decode_case(64, kv)
        kv_bytes = 64 * kv * 2 * NKV * HD * 2  # K+V 读量/step
        print(f"\n== decode 3D 路径 (bs=64, kv={kv}) KV读={kv_bytes / 1e9:.3f} GB ==")
        all_res[("decode", kv)] = run_path("decode", make_decode_case, (64, kv), kv_bytes, 50)

    for kv in (4096, 16384):
        kwargs, _ = make_prefill_case(8, 1024, kv)
        # 名义带宽口径：全量 KV 读（causal 实际约为 ×(1-qlen/2kv)，臂间可比）
        kv_bytes = 8 * 1024 * kv * 2 * NKV * HD * 2
        print(f"\n== prefill 2D 路径 (8seq×1024q, kv={kv}) 名义KV读={kv_bytes / 1e9:.3f} GB ==")
        all_res[("prefill", kv)] = run_path("prefill", make_prefill_case, (8, 1024, kv), kv_bytes, 15)

    print("\n== 判据核对（调优臂在其路径带宽增益 ≥10%）==")
    ok = True
    for path, arm in (("decode", "w1s1"), ("prefill", "w2s1")):
        for kv in (4096, 16384):
            r = all_res[(path, kv)]
            g = (r["default"] / r[arm] - 1) * 100
            verdict = "PASS" if g >= 10 else "FAIL"
            ok &= g >= 10
            print(f"VERDICT|path={path}|kv={kv}|arm={arm}|bw_gain_pct={g:.1f}|{verdict}")
    print(f"OVERALL|{'REPRODUCED' if ok else 'NOT_REPRODUCED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
