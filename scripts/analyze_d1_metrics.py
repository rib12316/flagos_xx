#!/usr/bin/env python3
"""D1 裁决轮 metrics.log 分析：E1（抢占/排队证据）+ E4（跨轮 APC 命中率）。

用法: python3 scripts/analyze_d1_metrics.py <metrics.log> [--fl-sched <serve.log>]

metrics.log 行格式（scrape_metrics.sh 产出）：
  <epoch_sec> vllm:metric_name{labels} value
按 generation_tokens 计数器是否增长切分"活动段"（= 一轮 bench），逐段算差分。
"""

import argparse
import re
import sys

LINE_RE = re.compile(
    r"^(?P<ts>\d+\.\d+)\s+vllm:(?P<name>[\w:]+)(?P<labels>\{[^}]*\})?\s+(?P<val>[\d.eE+-]+)$"
)

WANT = (
    "num_preemptions",
    "kv_cache_usage_perc",
    "num_requests_running",
    "num_requests_waiting",
    "prompt_tokens",
    "prompt_tokens_cached",
    "generation_tokens",
)


def parse(path):
    series = {k: [] for k in WANT}  # name -> [(ts, val)]
    bad = 0
    with open(path) as f:
        for line in f:
            m = LINE_RE.match(line.strip())
            if not m:
                continue
            name = m.group("name").rstrip("_total") if m.group("name").endswith("_total") else m.group("name")
            if name not in series:
                continue
            try:
                series[name].append((float(m.group("ts")), float(m.group("val"))))
            except ValueError:
                bad += 1
    return series, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_log")
    ap.add_argument("--fl-sched", default=None, help="serve.log 路径，附加 [fl.sched] 指纹统计")
    args = ap.parse_args()

    series, bad = parse(args.metrics_log)
    for k in WANT:
        if not series[k]:
            print(f"WARN: 指标 {k} 无样本", file=sys.stderr)
    print(f"# 解析样本: { {k: len(v) for k, v in series.items()} }（坏行 {bad}）")

    gen = series["generation_tokens"]
    pre = series["prompt_tokens"]
    cache = series["prompt_tokens_cached"]
    preempt = series["num_preemptions"]
    usage = series["kv_cache_usage_perc"]
    run = series["num_requests_running"]
    wait = series["num_requests_waiting"]

    # 按活动段切分：generation_tokens 严格递增视为活跃
    segs = []
    cur = None
    for i in range(1, len(gen)):
        active = gen[i][1] > gen[i - 1][1]
        if active and cur is None:
            cur = [i - 1, i]
        elif active:
            cur[1] = i
        elif cur is not None:
            segs.append(tuple(cur))
            cur = None
    if cur is not None:
        segs.append(tuple(cur))
    # 合并间隔 <30s 的相邻段（同一轮 bench 的间隙抖动）
    merged = []
    for s in segs:
        if merged and gen[s[0]][0] - gen[merged[-1][1]][0] < 30:
            merged[-1] = (merged[-1][0], s[1])
        else:
            merged.append(list(s))
    segs = [tuple(s) for s in merged]

    def seg_delta(arr, s):
        vals = [v for t, v in arr if gen[s[0]][0] - 1 <= t <= gen[s[1]][0] + 1]
        # 用段起止最近的样本
        lo = min([i for i, (t, v) in enumerate(arr) if t >= gen[s[0]][0] - 2], default=None)
        hi = max([i for i, (t, v) in enumerate(arr) if t <= gen[s[1]][0] + 2], default=None)
        if lo is None or hi is None or hi <= lo:
            # 段内样本兜底
            inseg = [v for t, v in arr if gen[s[0]][0] <= t <= gen[s[1]][0]]
            return inseg[-1] - inseg[0] if len(inseg) >= 2 else 0
        return arr[hi][1] - arr[lo][1]

    def seg_stat(arr, s, fn):
        vals = [v for t, v in arr if gen[s[0]][0] <= t <= gen[s[1]][0]]
        return fn(vals) if vals else float("nan")

    print(f"\n# 活动段（=bench 轮次）: {len(segs)}")
    print(f"{'轮':>2} {'起(时分秒)':>8} {'时长s':>6} {'Δprompt':>10} {'Δcached':>10} {'命中率%':>7} "
          f"{'Δgen':>8} {'Δpreempt':>8} {'max usage':>9} {'max run':>7} {'max wait':>8}")
    for i, s in enumerate(segs):
        dp = seg_delta(pre, s)
        dc = seg_delta(cache, s)
        dg = seg_delta(gen, s)
        dpr = seg_delta(preempt, s)
        import time as _t
        t0 = _t.localtime(gen[s[0]][0])
        dur = gen[s[1]][0] - gen[s[0]][0]
        hit = 100.0 * dc / dp if dp else float("nan")
        print(f"{i:>2} {t0.tm_hour:02d}{t0.tm_min:02d}{t0.tm_sec:02d} {dur:>6.0f} {dp:>10.0f} {dc:>10.0f} "
              f"{hit:>7.2f} {dg:>8.0f} {dpr:>8.0f} {seg_stat(usage, s, max):>9.3f} "
              f"{seg_stat(run, s, max):>7.0f} {seg_stat(wait, s, max):>8.0f}")

    if args.fl_sched:
        import subprocess
        out = subprocess.run(
            ["grep", "-c", r"\[fl.sched\]", args.fl_sched], capture_output=True, text=True
        ).stdout.strip()
        print(f"\n# [fl.sched] 行数: {out}")
        out2 = subprocess.run(
            ["grep", "-c", r"preempt=[1-9]", args.fl_sched], capture_output=True, text=True
        ).stdout.strip()
        print(f"# preempt>0 的步数: {out2}")
        # free_blk 谷值与 pre_tok/dec_tok 构成
        steps = []
        with open(args.fl_sched) as f:
            for line in f:
                if "[fl.sched] step=" in line:
                    d = dict(re.findall(r"(\w+)=([\d.]+)", line))
                    if d:
                        steps.append(d)
        if steps:
            import statistics as st
            free = [int(float(d["free_blk"])) for d in steps if "free_blk" in d]
            pre_t = [int(d["pre_tok"]) for d in steps if "pre_tok" in d]
            dec_t = [int(d["dec_tok"]) for d in steps if "dec_tok" in d]
            print(f"# free_blk 谷值={min(free)} 中位={st.median(free)}  均值pre_tok={st.mean(pre_t):.0f} "
                  f"均值dec_tok={st.mean(dec_t):.0f}  pre_tok=0 步占比="
                  f"{sum(1 for x in pre_t if x==0)/len(pre_t)*100:.1f}%")


if __name__ == "__main__":
    main()
