#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""F-c: _copy_kernel 归因脚本（纯 CPU，不触 GPU）。

把 kineto trace 里的 `_copy_kernel*` GPU 核归因到发起它的 cpu_op 调用链。

关联通道（trace 无 python 栈、无 Input Dims，三条互补通道）：
  1. kernel.args["External id"] == cpu_op.args["External id"]  —— kineto 官方父子链；
  2. kernel.args["correlation"] == cuda_runtime.args["correlation"] —— 找到 launch API
     （mcModuleLaunchKernel=急发 / mcGraphLaunch=cudagraph 回放）及其 tid/ts；
  3. 同 tid 上按时间区间包含关系重建 cpu_op 祖先链 + 包裹的 user_annotation
     （execute_new_N_cached_M = 每个 decode/prefill 步一条）。

输出：
  - 按 (user_annotation, cpu_op 链签名) 聚合的次数/核时长/grid 分布表；
  - 每组每步次数（46074 核 / 1027 步 = 44.9/步 的拆解）；
  - JSON 明细供复核。

用法：python3 scripts/fc/trace_copy_attribution.py <trace.json.gz> [out_prefix]
"""
import collections
import gzip
import json
import sys

TGT = "_copy_kernel"  # 匹配 _copy_kernel / _copy_kernel_kernel_rank_{1,2,3}


def load_events(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        return json.load(f)["traceEvents"]


def sweep_chains(events, queries_by_tid):
    """栈式扫描求包含关系祖先链（内→外），O((n+q)log(n+q))。

    events: X 事件；queries_by_tid: {(pid,tid): [查询 ts,...]} → 返回同 key
    的 {query_ts: [祖先事件,...]}（内→外）。同 ts 排序先长事件（父先入栈）。
    """
    by_tid = collections.defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and "ts" in e:
            by_tid[(e["pid"], e["tid"])].append(e)
    out = {}
    for key, qs in queries_by_tid.items():
        evs = sorted(by_tid.get(key, []), key=lambda e: (e["ts"], -e.get("dur", 0)))
        items = [(e["ts"], 1, e) for e in evs] + [(q, 0, None) for q in qs]
        items.sort(key=lambda it: (it[0], it[1]))  # 查询点排在同 ts 事件后
        res = {}
        stack = []
        for ts, kind, e in items:
            if kind == 1:
                while stack and stack[-1][1] <= ts:
                    stack.pop()
                stack.append((ts, ts + e.get("dur", 0), e))
            else:
                while stack and stack[-1][1] < ts:
                    stack.pop()
                res[ts] = [s[2] for s in reversed(stack)]
        out[key] = res
    return out


def grid_key(grid, block):
    g = grid if isinstance(grid, list) else [grid]
    b = block if isinstance(block, list) else [block]
    gx = int(g[0]) if g else 1
    bx = int(b[0]) if b else 1
    return gx, bx, gx * bx


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/data/flagos_wt/results/20261120_c500_reprofiling/trace_all/"
        "rank0.1795110863960817196.pt.trace.json.gz")
    out_prefix = sys.argv[2] if len(sys.argv) > 2 else "/tmp/fc_copy_attr"

    ev = load_events(path)
    print(f"# events: {len(ev)}")

    # ---- 索引 ----
    extid2cp = {}
    corr2rt = {}
    cpu_ops, anns, kernels = [], [], []
    for e in ev:
        if e.get("ph") != "X":
            continue
        cat = e.get("cat")
        if cat == "cpu_op":
            cpu_ops.append(e)
            x = e.get("args", {}).get("External id")
            if x is not None:
                extid2cp[x] = e  # 同 id 多事件时后写覆盖——实测一对一
        elif cat == "user_annotation":
            anns.append(e)
        elif cat == "cuda_runtime":
            c = e.get("args", {}).get("correlation")
            if c is not None:
                corr2rt[c] = e
        elif cat == "kernel" and TGT in e.get("name", ""):
            kernels.append(e)
    print(f"# copy kernels={len(kernels)} cpu_ops={len(cpu_ops)} "
          f"anns={len(anns)} corr={len(corr2rt)}")

    # 查询点：每个 copy kernel 的 launch ts（按 tid 分组）
    launch_info = {}
    queries_by_tid = collections.defaultdict(list)
    for k in kernels:
        args = k.get("args", {})
        rt = corr2rt.get(args.get("correlation"))
        lts = rt["ts"] if rt else k["ts"]
        ltid = (rt["pid"], rt["tid"]) if rt else (k["pid"], k["tid"])
        launch_info[k["ts"]] = (rt["name"] if rt else "?", lts, ltid)
        queries_by_tid[ltid].append(lts)
    chains_by_tid = sweep_chains(cpu_ops + anns, queries_by_tid)

    # ---- 逐核归因 ----
    rows = []
    for k in kernels:
        args = k.get("args", {})
        extid = args.get("External id")
        launch_api, lts, ltid = launch_info[k["ts"]]
        # 首选 External id 直连 cpu_op；再叠加祖先链（链里通常已含直连点）
        direct = extid2cp.get(extid)
        chain = chains_by_tid.get(ltid, {}).get(lts, [])
        chain_names = [c["name"] for c in chain] or (
            [direct["name"]] if direct else ["<no-cpu-op>"])
        # 包裹 annotation（链里最外层通常就是）
        ann = next((c["name"] for c in reversed(chain)
                    if c.get("cat") == "user_annotation"), "<none>")
        gx, bx, lanes = grid_key(args.get("grid"), args.get("block"))
        rows.append({
            "kname": k["name"], "dur_us": k.get("dur", 0), "ts": k["ts"],
            "launch_api": launch_api, "ann": ann, "chain": chain_names,
            "direct": direct["name"] if direct else None,
            "grid0": gx, "block0": bx, "lanes": lanes,
            "extid": extid, "corr": args.get("correlation"),
        })

    # ---- 聚合 ----
    def sig(r):
        return (r["ann"], " <- ".join(r["chain"][:6]), r["launch_api"])

    groups = collections.defaultdict(list)
    for r in rows:
        groups[sig(r)].append(r)

    # 步数：按 annotation 实例数（execute_new_* 每步一条）
    n_steps = collections.Counter(
        a["name"] for a in anns if a["name"].startswith("execute_new"))

    report = {
        "trace": path,
        "total_copy_kernels": len(rows),
        "kernel_name_hist": collections.Counter(r["kname"] for r in rows),
        "launch_api_hist": collections.Counter(r["launch_api"] for r in rows),
        "ann_step_hist": dict(n_steps),
        "groups": [],
        "rows": rows,
    }
    print("\n== kernel 名字直方图 ==")
    for n, c in report["kernel_name_hist"].most_common():
        print(f"  {c:7d}  {n}")
    print("\n== launch API 直方图 ==")
    for n, c in report["launch_api_hist"].most_common():
        print(f"  {c:7d}  {n}")
    print("\n== user_annotation 步直方图（前 10）==")
    for n, c in sorted(n_steps.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {c:7d}  {n}")

    print("\n== 归因分组（按总核时排序）==")
    glist = []
    for s, rs in groups.items():
        dur = sum(r["dur_us"] for r in rs)
        lanes = collections.Counter(r["lanes"] for r in rs)
        grids = collections.Counter(
            (r["grid0"], r["block0"]) for r in rs)
        glist.append({
            "ann": s[0], "chain": s[1], "launch_api": s[2],
            "count": len(rs), "total_dur_us": round(dur, 1),
            "avg_dur_us": round(dur / len(rs), 2),
            "lanes_hist": dict(lanes),
            "grid_hist": {f"g{g}xb{b}": c for (g, b), c in grids.most_common(6)},
            "kernel_names": dict(collections.Counter(r["kname"] for r in rs)),
            "per_step_main": round(len(rs) / max(
                n_steps.get(s[0], 1), 1), 2),
        })
    glist.sort(key=lambda g: -g["total_dur_us"])
    report["groups"] = glist
    for g in glist[:25]:
        print(f"\n[{g['count']}x | {g['total_dur_us']/1000:.2f}ms | "
              f"avg {g['avg_dur_us']}us] ann={g['ann']} api={g['launch_api']}")
        print(f"  chain: {g['chain']}")
        print(f"  grids: {g['grid_hist']}")
        print(f"  lanes: {g['lanes_hist']}  kernels: {g['kernel_names']} "
              f"per_step={g['per_step_main']}")

    with open(out_prefix + ".json", "w") as f:
        json.dump({k: v for k, v in report.items() if k != "rows"}, f,
                  indent=1, ensure_ascii=False)
    with open(out_prefix + "_rows.json", "w") as f:
        json.dump(rows, f, ensure_ascii=False)
    print(f"\n写出: {out_prefix}.json / {out_prefix}_rows.json")


if __name__ == "__main__":
    main()
