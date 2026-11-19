#!/usr/bin/env python3
"""D2 fake-vs-real 对拍：serve 真实路径 greedy 逐题比对 fake_quant_eval 结果。

用途：serve 侧 W8A8 kernel 数值缺陷与 libentry 类流竞争的检出门（handover §5.4 第 2 步）。
fake（results/20261107_d2_fakequant/r_L2a05.json）= 离线 emu=int32 位精确仿真 + 纯 torch
量化链；real = serve 经 CachedOp/dispatch 的裸 triton 量化 + _int_mm + 融合 epilogue。
同口径：greedy（temperature=0）、max_tokens=3072、官方 user 消息模板、冻结 30 题
（hash 校验）。显著偏离 = kernel 数值缺陷或流竞争，立即止损排查。

用法：python scripts/d2/compare_fake_real.py [--port 9031] [--fake results/.../r_L2a05.json]
输出：RESULT 行 + JSON 落盘（--out，默认 results/20261112_d2_fakevsreal/compare.json）
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.d2.common import (
    build_messages,
    extract_pred,
    is_correct,
    load_problems,
    same_answer,
)


def chat(port: int, problem: str, max_tokens: int, timeout: int = 600) -> str:
    body = json.dumps({
        "model": "minicpm",
        "messages": build_messages(problem),
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())["choices"][0]["message"]["content"]
    return out, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9031)
    ap.add_argument("--fake", default="results/20261107_d2_fakequant/r_L2a05.json")
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="results/20261112_d2_fakevsreal/compare.json")
    args = ap.parse_args()

    fake = json.load(open(args.fake))
    problems, frozen = load_problems(args.n)
    assert frozen == fake["frozen"], \
        f"冻结集不一致: {frozen} vs {fake['frozen']}"
    fake_by_id = {r["unique_id"]: r for r in fake["records"]}

    def run_one(p):
        gen, dt = chat(args.port, p["problem"], args.max_tokens)
        pred = extract_pred(gen)
        return {
            "unique_id": p["unique_id"],
            "pred": pred,
            "ok": bool(is_correct(pred, p["answer"])),
            "len": len(gen),
            "truncated": len(gen) >= args.max_tokens,
            "head": gen[:120],
            "tail": gen[-120:],
            "sec": round(dt, 1),
        }

    t0 = time.time()
    with cf.ThreadPoolExecutor(args.workers) as ex:
        real = list(ex.map(run_one, problems))
    wall = time.time() - t0

    same_ans = same_pred = 0
    diffs = []
    for r in real:
        f = fake_by_id[r["unique_id"]]
        r["fake_pred"] = f["pred"]
        r["fake_ok"] = f["ok"]
        r["same_pred"] = (r["pred"] == f["pred"])
        r["same_ans"] = bool(same_answer(r["pred"], f["pred"]))
        same_pred += r["same_pred"]
        same_ans += r["same_ans"]
        if not r["same_ans"]:
            diffs.append(r["unique_id"])
    acc = sum(r["ok"] for r in real) / len(real)
    fake_acc = sum(r["fake_ok"] for r in real) / len(real)
    line = ("RESULT|stage=fake_vs_real|n=%d|acc_real=%.4f|acc_fake=%.4f|"
            "same_ans=%.4f|same_pred=%d|trunc=%d|wall_sec=%.0f|diff_ids=%s"
            % (len(real), acc, fake_acc, same_ans / len(real), same_pred,
               sum(r["truncated"] for r in real), wall, ",".join(diffs) or "-"))
    print(line)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"params": {"temperature": 0.0, "max_tokens": args.max_tokens,
                              "fake": args.fake, "frozen": frozen},
                   "summary": {"acc_real": acc, "acc_fake": fake_acc,
                               "same_ans": same_ans / len(real),
                               "same_pred": same_pred},
                   "records": real}, fh, indent=1, ensure_ascii=False)

    # 判读：fake 口径 Δacc=0 且 emu=int32 与 _int_mm 位精确 ⇒ real acc 应与 fake 一致。
    # same_ans 低但 acc 相同 = 生成轨迹漂移（数值微差或非确定算子）；acc 崩 = 真缺陷。
    ok = acc >= fake_acc - 1 / len(real) and same_ans / len(real) >= 0.8
    print(f"RESULT|stage=fake_vs_real_gate|pass_={'true' if ok else 'false'}")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
