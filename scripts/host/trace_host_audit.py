#!/usr/bin/env python3
"""任务③ host 开销审计 —— trace 侧定量脚本（纯 CPU，不触 GPU）。

数据源：kineto (MACA mc* runtime) trace，serve 实测 rank0。
口径与判决规则（开发单 §5）：
  - decode 步锚点 = compute_global_hist_kernel（采样直方图，每 decode 步恰 1 次；
    1027 次 == gpu_user_annotation 1027 == 官方口径 decode 步数）。
  - decode 稳态 = dominant execute_new_*_cached_<bs> 重放 span（bs=modal，887 个）。
  - 诚实门：稳态每步同步类（Stream/Event/DeviceSynchronize + 同步形态 memcpy）
    < 2 次 → 负判决（host≈0，任务③关账）。
  - 同步"代价"以 host 侧 API 调用时长为准（µs 级 no-op 与 ms 级阻塞分开统计）。

用法：python3 trace_host_audit.py <trace.json.gz> [outdir]
输出：stdout 摘要 + <outdir>/host_audit_summary.txt
"""
import bisect
import collections
import gzip
import json
import os
import sys

SYNC_APIS = (
    "mcStreamSynchronize",
    "mcEventSynchronize",
    "mcDeviceSynchronize",
)
PAGEABLE_D2H = "Memcpy DtoH (Device -> Pageable)"
PINNED_D2H = "Memcpy DtoH (Device -> Pinned)"
ANCHOR_KERNEL = "compute_global_hist_kernel"
GAP_US = 50  # intra-step gap threshold


def q(xs, p):
    if not xs:
        return 0.0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def fmt_us(v):
    return f"{v:.1f}us" if v < 1000 else f"{v / 1000:.2f}ms"


def main():
    trace = sys.argv[1] if len(sys.argv) > 1 else (
        "/data/flagos_wt/results/20261120_c500_reprofiling/trace_all/"
        "rank0.1795110863960817196.pt.trace.json.gz")
    outdir = sys.argv[2] if len(sys.argv) > 2 else "."
    os.makedirs(outdir, exist_ok=True)
    lines = []

    def P(s=""):
        print(s)
        lines.append(s)

    P(f"# trace: {trace}")
    P("loading json (pure CPU) ...")
    with gzip.open(trace, "rt") as f:
        ev = json.load(f)["traceEvents"]
    t0 = min(e["ts"] for e in ev if e.get("ph") == "X" and e.get("ts"))
    P(f"# events: {len(ev)}, trace span {(max(e['ts'] + e.get('dur', 0) for e in ev if e.get('ph') == 'X') - t0) / 1e6:.1f}s")

    kern, runtime, memcpy, cpuop, ann = [], [], [], [], []
    for e in ev:
        ph, cat = e.get("ph"), e.get("cat")
        if ph != "X":
            continue
        ts, du = e["ts"], e.get("dur", 0)
        if cat == "kernel":
            kern.append((ts, ts + du, e["name"]))
        elif cat == "cuda_runtime":
            runtime.append((ts, du, e["name"]))
        elif cat == "gpu_memcpy":
            memcpy.append((ts, du, e["name"], e.get("args", {}).get("bytes")))
        elif cat == "cpu_op":
            cpuop.append((ts, ts + du, e["name"]))
        elif cat == "user_annotation":
            ann.append((ts, ts + du, e["name"]))

    # ---- 1. 全局 runtime API / memcpy 表 ----
    P("\n== [1] cuda_runtime API 全量表 (count | host 总时长) ==")
    agg = collections.defaultdict(lambda: [0, 0.0])
    for ts, du, nm in runtime:
        agg[nm][0] += 1
        agg[nm][1] += du
    for nm, (n, d) in sorted(agg.items(), key=lambda x: -x[1][1])[:14]:
        P(f"  {n:8d}  {d / 1e3:10.1f}ms  {nm}")
    mc = collections.Counter(nm for _, _, nm, _ in memcpy)
    P("== [1b] gpu_memcpy 方向表 ==")
    for nm, n in mc.most_common():
        P(f"  {n:8d}  {nm}")

    n_steps = sum(1 for _, _, nm in kern if nm == ANCHOR_KERNEL)
    P(f"\n# decode 步锚点 ({ANCHOR_KERNEL}): {n_steps}")

    # ---- 2. decode 活动段（锚点间隔 <2s 连成段）----
    anchors = sorted(ts for ts, _, nm in kern if nm == ANCHOR_KERNEL)
    segs = []
    s = p = anchors[0]
    for a in anchors[1:]:
        if a - p > 2e6:
            segs.append((s, p))
            s = a
        p = a
    segs.append((s, p))
    P(f"# decode 活动段: {len(segs)} 段 " +
      str([(round((a - t0) / 1e6, 1), round((b - a) / 1e6, 1)) for a, b in segs]))

    # ---- 3. 稳态 decode span（dominant 重放形状）----
    namec = collections.Counter(nm for _, _, nm in ann if "execute_new" in nm)
    dom_name, dom_n = namec.most_common(1)[0]
    steady = sorted((a, b) for a, b, nm in ann if nm == dom_name)
    P(f"\n== [2] 稳态 decode 重放 span: '{dom_name}' n={len(steady)} "
      f"host dur p50={fmt_us(q([b - a for a, b in steady], .5))} ==")

    ss = [a for a, _ in steady]

    def cnt_in(events_ts):
        c = 0
        for ts in events_ts:
            i = bisect.bisect_right(ss, ts) - 1
            if i >= 0 and ts < steady[i][1]:
                c += 1
        return c

    # ---- 4. 稳态每步同步/D2H/item 计数（诚实门）----
    P("\n== [3] 稳态每步计数（诚实门输入）==")
    gate = 0.0
    for nm in SYNC_APIS:
        ts_list = [ts for ts, du, n2 in runtime if n2 == nm]
        c = cnt_in(ts_list)
        dus = [du for ts, du, n2 in runtime if n2 == nm and
               (lambda i: i >= 0 and ts < steady[i][1])(bisect.bisect_right(ss, ts) - 1)]
        P(f"  {nm:22s} in-span {c:5d} = {c / len(steady):5.2f}/step  "
          f"in-span host 时长合计 {sum(dus) / 1e3:.3f}ms (p50 {fmt_us(q(dus, .5))})")
        gate += c / len(steady)
    for label, nm in (("D2H pageable", PAGEABLE_D2H), ("D2H pinned", PINNED_D2H)):
        c = cnt_in([ts for ts, du, n2, _ in memcpy if n2 == nm])
        P(f"  {label:22s} in-span {c:5d} = {c / len(steady):5.2f}/step")
    items = [ts for ts, te, nm in cpuop if nm == "aten::item"]
    c_item = cnt_in(items)
    all_item_du = [te - ts for ts, te, nm in cpuop if nm == "aten::item"]
    P(f"  aten::item             in-span {c_item:5d} = {c_item / len(steady):5.2f}/step  "
      f"全 trace item host 时长 p50={fmt_us(q(all_item_du, .5))} p99={fmt_us(q(all_item_du, .99))} "
      f"(CPU 张量读，非 GPU 同步)" if all_item_du else "")
    P(f"\n** 诚实门: 稳态每步同步类 = {gate:.2f} "
      f"{'<' if gate < 2 else '>='} 2 → "
      f"{'负判决（host≈0）' if gate < 2 else '需逐项消除'} **")

    # ---- 5. eventSync 长等待的落点（by-design join 论证）----
    P("\n== [4] mcEventSynchronize 时长分解 ==")
    evs = [(ts, du) for ts, du, nm in runtime if nm == "mcEventSynchronize"]
    in_du = [du for ts, du in evs if (lambda i: i >= 0 and ts < steady[i][1])(bisect.bisect_right(ss, ts) - 1)]
    out_du = [du for ts, du in evs if not (lambda i: i >= 0 and ts < steady[i][1])(bisect.bisect_right(ss, ts) - 1)]
    P(f"  span 内: n={len(in_du)} 总 {sum(in_du) / 1e3:.1f}ms p50 {fmt_us(q(in_du, .5))}（no-op）")
    P(f"  span 外: n={len(out_du)} 总 {sum(out_du) / 1e3:.1f}ms p90 {fmt_us(q(out_du, .9))}"
      f"（async 输出/输入 prep join，等待 GPU 完成，GPU-bound by design）")
    # 按 GPU busy 秒分桶
    kern_sec = collections.defaultdict(float)
    for a, b, _ in kern:
        for k in range(int(a // 1e6), int(b // 1e6) + 1):
            kern_sec[k] += (min(b, (k + 1) * 1e6) - max(a, k * 1e6)) / 1e6
    busy_d = idle_d = busy_n = idle_n = 0.0
    for ts, du in evs:
        if kern_sec.get(int(ts // 1e6), 0) > 0.5:
            busy_d += du
            busy_n += 1
        else:
            idle_d += du
            idle_n += 1
    P(f"  落点: GPU busy 秒(>50%)内 n={int(busy_n)} 阻塞 {busy_d / 1e6:.2f}s | "
      f"低 busy 秒 n={int(idle_n)} 阻塞 {idle_d / 1e6:.2f}s → 长等待与 kernel 忙共现（已重叠）")

    # ---- 6. 稳态窗口 busy% + 步间气泡 + 步内 gap 归因 ----
    # 选稳态 span 最多的活动段内部窗口
    inseg = collections.Counter(
        i for i, (a, b) in enumerate(segs)
        for a2, b2 in steady if a <= a2 < b)
    gi = inseg.most_common(1)[0][0]
    w_spans = [(a, b) for a, b in steady if segs[gi][0] <= a < segs[gi][1]]
    # 去掉段头 prefill 段：取窗口内 span 第 5 个起到倒数第 5 个
    w0, w1 = w_spans[4][0], w_spans[-5][1]
    wk = sorted((a, b) for a, b, _ in kern if w0 <= a < w1)
    merged = []
    for a, b in wk:
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    busy = sum(b - a for a, b in merged)
    P(f"\n== [5] 稳态窗口 [{(w0 - t0) / 1e6:.1f}s, {(w1 - t0) / 1e6:.1f}s] "
      f"n={len(w_spans) - 9} 步: kernel busy {busy / (w1 - w0) * 100:.1f}% ==")
    wsp = [a for a, _ in w_spans]
    run_by = collections.defaultdict(list)
    for a, b in merged:
        run_by[bisect.bisect_right(wsp, a) - 1].append((a, b))
    inter = []
    for i in range(len(w_spans) - 1):
        r0, r1 = run_by.get(i), run_by.get(i + 1)
        if r0 and r1:
            inter.append(r1[0][0] - r0[-1][1])
    P(f"  步间气泡（步 N 末 kernel → 步 N+1 首 kernel）: p50={fmt_us(q(inter, .5))} "
      f"p90={fmt_us(q(inter, .9))} max={fmt_us(q(inter, 1.0))} → async 流水无 host 气泡")
    pairs = collections.Counter()
    pdur = collections.Counter()
    wkn = sorted((a, b, nm) for a, b, nm in kern if w0 <= a < w1)
    intra_n = intra_t = 0
    for i in range(len(wkn) - 1):
        g = wkn[i + 1][0] - wkn[i][1]
        if g >= GAP_US:
            key = (wkn[i][2].split("<")[0][:40], wkn[i + 1][2].split("<")[0][:40])
            pairs[key] += 1
            pdur[key] += g
            intra_n += 1
            intra_t += g
    n_ws = max(1, len(w_spans) - 9)
    P(f"  步内 gap>={GAP_US}us: n={intra_n} 总 {intra_t / 1e3:.0f}ms = {intra_t / 1e3 / n_ws:.2f}ms/步"
      f"（eager 采样尾 host-launch 缝隙）")
    for (a, b), n in pairs.most_common(5):
        P(f"    {n:4d} 次 | {pdur[(a, b)] / 1e3:6.1f}ms | {a} -> {b}")

    # ---- 7. H2D（静态缓存现状）----
    h2d = [(ts, by) for ts, du, nm, by in memcpy if "HtoD" in nm]
    c = cnt_in([ts for ts, _ in h2d])
    byt = 0
    for ts, by in h2d:
        i = bisect.bisect_right(ss, ts) - 1
        if i >= 0 and steady[i][0] <= ts < steady[i][1] + 2000:
            byt += by or 0
    P(f"\n== [6] H2D(稳态) {c / len(steady):.1f}/步 {byt / len(steady) / 1024:.0f}KiB/步 "
      f"（pinned 异步，静态 pinned buffer 已在位）==")

    with open(os.path.join(outdir, "host_audit_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    P(f"\n# summary saved -> {os.path.join(outdir, 'host_audit_summary.txt')}")


if __name__ == "__main__":
    main()
