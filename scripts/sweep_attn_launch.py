#!/usr/bin/env python3
"""sweep unified_attention 的 per-shape launch 参数，再生 attn_launch_config 表。

用法：
    python scripts/sweep_attn_launch.py                 # quick：计分形状（~3min）
    python scripts/sweep_attn_launch.py --full          # 全矩阵（~20-40min）
    python scripts/sweep_attn_launch.py --out results/xxx

产出：
    1) 每形状全配置 CSV（sweep_raw.csv）
    2) 每形状最优配置 + 相对 default 加速（stdout 表）
    3) 可直接贴回 vllm_fl/.../iluvatar/attn_launch_config.py 的 LAUNCH_TABLE
       字面量（sweep_table.py）+ 环境变量覆盖用 JSON（sweep_table.json，
       VLLM_FL_ATTN_LAUNCH_TABLE=<path> 直接 A/B，无需改码）

重调检查点（P2′ 调度落地后 / P3′ 量化落地后 / 冻结前终扫）用法：
    重跑本脚本 -> A/B 验证 JSON -> 把 sweep_table.py 内容贴回 LAUNCH_TABLE。
判据沿 micro_bench_d0.py：bit-exact 必须成立，否则该配置作废。
"""
import argparse
import csv
import json
import os
import sys
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

import torch

import vllm.v1.attention.ops.triton_unified_attention as tua

DEV, DTYPE = "cuda", torch.bfloat16
NH, NKV, HD, BLOCK, NSEG = 16, 2, 128, 16, 16
SEQ_THRESHOLD_3D = 64

torch.manual_seed(0)


# ------------------------------------------------------------------ 用例构造
def make_decode_case(bs, kv_len):
    q = torch.randn(bs, NH, HD, dtype=DTYPE, device=DEV)
    npages = (kv_len + BLOCK - 1) // BLOCK
    kv = torch.randn(2, bs * npages + 64, BLOCK, NKV, HD, dtype=DTYPE, device=DEV)
    bt = torch.arange(bs * npages, dtype=torch.int32, device=DEV).reshape(bs, npages)
    cu = torch.arange(0, bs + 1, dtype=torch.int32, device=DEV)
    seqused = torch.full((bs,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty(bs, NH, HD, dtype=DTYPE, device=DEV)
    so = torch.empty(SEQ_THRESHOLD_3D, NH, NSEG, HD, dtype=torch.float32, device=DEV)
    sm = torch.empty(SEQ_THRESHOLD_3D, NH, NSEG, dtype=torch.float32, device=DEV)
    se = torch.empty(SEQ_THRESHOLD_3D, NH, NSEG, dtype=torch.float32, device=DEV)
    return dict(
        q=q, k=kv[0], v=kv[1], out=out, cu_seqlens_q=cu, max_seqlen_q=1,
        seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=HD**-0.5,
        causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=SEQ_THRESHOLD_3D, num_par_softmax_segments=NSEG,
        softmax_segm_output=so, softmax_segm_max=sm, softmax_segm_expsum=se,
    ), out


def make_prefill_case(nseq, qlen, kv_len):
    ntok = nseq * qlen
    q = torch.randn(ntok, NH, HD, dtype=DTYPE, device=DEV)
    npages = (kv_len + BLOCK - 1) // BLOCK
    kv = torch.randn(2, nseq * npages + 64, BLOCK, NKV, HD, dtype=DTYPE, device=DEV)
    bt = torch.arange(nseq * npages, dtype=torch.int32, device=DEV).reshape(nseq, npages)
    cu = torch.arange(0, ntok + 1, qlen, dtype=torch.int32, device=DEV)
    seqused = torch.full((nseq,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty(ntok, NH, HD, dtype=DTYPE, device=DEV)
    return dict(
        q=q, k=kv[0], v=kv[1], out=out, cu_seqlens_q=cu, max_seqlen_q=qlen,
        seqused_k=seqused, max_seqlen_k=kv_len, softmax_scale=HD**-0.5,
        causal=True, window_size=(-1, -1), block_table=bt, softcap=0.0,
        q_descale=None, k_descale=None, v_descale=None,
        seq_threshold_3D=None, num_par_softmax_segments=None,
        softmax_segm_output=None, softmax_segm_max=None,
        softmax_segm_expsum=None,
    ), out


# ------------------------------------------------------- 注入机制（bench 用）
class LaunchProxy:
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


def neutralize_plugin_hook():
    """若 dispatch 层表已装载，还原上游默认（保证 default 臂干净）。"""
    k = tua.kernel_unified_attention
    if getattr(k, "_fl_launch_config_hook", False):
        k.run = k._fl_launch_orig_run
        k._fl_launch_config_hook = False
        w = tua.unified_attention
        orig_w = getattr(w, "_fl_orig_wrapper", None)
        if orig_w is not None:
            tua.unified_attention = orig_w
            import vllm.v1.attention.backends.triton_attn as _ta
            if getattr(_ta.unified_attention, "_fl_orig_wrapper", None) is orig_w:
                _ta.unified_attention = orig_w
        print("[sweep] 已还原 dispatch 层 launch 装载（default 臂为上游默认）")


def install_arms(cfg3d=None, cfg2d=None):
    jit = tua.kernel_unified_attention
    proxy = LaunchProxy(jit, cfg3d, cfg2d)
    tua.kernel_unified_attention = proxy


def uninstall_arms():
    k = tua.kernel_unified_attention
    jit = getattr(k, "_jit", k)
    tua.kernel_unified_attention = jit


def bench(kwargs, reps, warmup=5):
    from vllm.v1.attention.ops.triton_unified_attention import unified_attention
    for _ in range(warmup):
        unified_attention(**kwargs)
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(reps):
        unified_attention(**kwargs)
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) * 1e3 / reps


# ------------------------------------------------------------------ main
QUICK_DECODE = [(16, 4096), (32, 4096), (32, 16384), (64, 4096), (64, 16384)]
QUICK_PREFILL = [(8, 1024, 4096), (8, 1024, 16384)]
FULL_DECODE = [(8, 1024), (16, 1024), (16, 16384), (32, 4096), (32, 16384),
               (64, 4096), (64, 16384), (128, 4096), (128, 16384), (128, 32768)]
FULL_PREFILL = [(1, 4096, 4096), (2, 2048, 4096), (4, 2048, 4096),
                (8, 1024, 4096), (8, 1024, 16384), (2, 4096, 16384)]
ARMS = [("auto", None)] + [(f"w{w}s{s}", (w, s))
                           for w in (1, 2, 4, 8) for s in (1, 2, 3)]


def sweep(cases, maker, reps, rows, best):
    is_decode = maker is make_decode_case
    path = "decode" if is_decode else "prefill"
    for shape in cases:
        torch.manual_seed(0)
        kwargs, out = maker(*shape)
        bs, kv = shape[0], shape[-1]
        qlabel = f" q={shape[1]}" if not is_decode else ""
        label = f"{path} bs={bs} kv={kv}{qlabel}"
        res, ref = {}, None
        for arm, cfg in ARMS:
            uninstall_arms()
            if cfg is not None:
                install_arms(cfg3d=cfg) if is_decode else install_arms(cfg2d=cfg)
            us = bench(kwargs, reps)
            res[arm] = us
            if arm == "auto":
                ref = out.clone()
            else:
                diff = (out.float() - ref.float()).abs().max().item()
                if diff != 0.0:
                    print(f"  [警告] {label} {arm} 非 bit-exact (max_abs={diff})，作废")
                    res[arm] = float("inf")
            rows.append({"shape": label, "arm": arm, "us": round(us, 1)})
        uninstall_arms()
        ok = {a: t for a, t in res.items() if t != float("inf")}
        b = min(ok, key=ok.get)
        gain = ok["auto"] / ok[b]
        best.append((path, bs, kv, label, b, ok[b], ok["auto"], gain))
        top3 = sorted(ok.items(), key=lambda kv_: kv_[1])[:3]
        print(f"{label:<34} default={ok['auto']:9.1f}µs | " +
              " ".join(f"{a}={t:.1f}" for a, t in top3) +
              f" | best={b} ({gain:.3f}x)")


def bucket_bs(n):
    # 档位定义与插件内 attn_launch_config.BS_TIERS 保持一致（单一事实源：直接复用）
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar.attn_launch_config import (
            bucket_bs as _bbs)
        return _bbs(n)
    except Exception:
        return next((f"le{b}" for b in (16, 32, 64, 128) if n <= b), "gt128")


def bucket_kv(n):
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar.attn_launch_config import (
            bucket_kv as _bkv)
        return _bkv(n)
    except Exception:
        return next((f"le{b // 1024}k" for b in (8192, 32768) if n <= b), "gt32k")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="全矩阵（默认 quick 计分形状）")
    ap.add_argument("--out", default=None, help="输出目录（默认 results/<ts>_launch_sweep）")
    args = ap.parse_args()

    neutralize_plugin_hook()
    ts = __import__("datetime").datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = args.out or f"results/{ts}_launch_sweep"
    os.makedirs(outdir, exist_ok=True)

    print(f"环境: {torch.cuda.get_device_name(0)}; ARMS={[a for a, _ in ARMS]}")
    rows, best = [], []
    dcases = FULL_DECODE if args.full else QUICK_DECODE
    pcases = FULL_PREFILL if args.full else QUICK_PREFILL
    print("\n== decode 3D ==")
    sweep(dcases, make_decode_case, 30, rows, best)
    print("\n== prefill 2D ==")
    sweep(pcases, make_prefill_case, 8, rows, best)

    with open(f"{outdir}/sweep_raw.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=["shape", "arm", "us"])
        wcsv.writeheader()
        wcsv.writerows(rows)

    # 汇总到档位：同档一致 -> 具体行；不一致 -> 列出告警人工裁决
    agg = defaultdict(dict)
    for path, bs, kv, label, arm, us, auto, gain in best:
        bsk, kvk = bucket_bs(bs), bucket_kv(kv)
        agg[(path, bsk, kvk)][label] = (arm, gain)
    table = {}
    for key, shapes in sorted(agg.items()):
        # 档内多个形状时：所有形状最优 arm 一致才写具体行，否则列出告警人工裁决
        winners = {arm for arm, _ in shapes.values()}
        if len(winners) > 1 or "auto" in winners:
            print(f"[告警] {key} 档内最优不一致 {winners}（{dict(shapes)}），不写行，需人工复核")
            continue
        pick = winners.pop()
        table[key] = tuple(int(x) for x in pick.removeprefix("w").split("s"))

    lines = [
        "# sweep_attn_launch.py 再生于 " + ts +
        "（重调检查点：P2′/P3′/冻结前终扫后重跑）",
        "LAUNCH_TABLE = {"
    ]
    for (p, b, k), (w, s) in sorted(table.items()):
        lines.append(f"    (\"{p}\", \"{b}\", \"{k}\"): ({w}, {s}),")
    lines.append("}")
    py = "\n".join(lines)
    js = json.dumps([[p, b, k, w, s] for (p, b, k), (w, s) in
                     sorted(table.items())], ensure_ascii=False, indent=1)
    with open(f"{outdir}/sweep_table.py", "w") as f:
        f.write(py + "\n")
    with open(f"{outdir}/sweep_table.json", "w") as f:
        f.write(js + "\n")

    print("\n== 再生 LAUNCH_TABLE（贴回 attn_launch_config.py）==")
    print(py)
    print(f"\n产出: {outdir}/sweep_raw.csv, sweep_table.py, sweep_table.json")
    print("A/B: VLLM_FL_ATTN_LAUNCH_TABLE=" + os.path.abspath(f"{outdir}/sweep_table.json"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
