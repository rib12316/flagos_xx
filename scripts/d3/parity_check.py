# 对拍 sanity（非官方 105 题）：16 条固定 prompt，greedy（temperature=0），
# max_tokens=64，逐条比对生成文本一致率。capture 模式存基线，compare 模式
# 输出一致率与首处分歧（供 TD/段数/KV int8 等非 bit-exact 改动的风险量化）。
import argparse
import json
import os
import sys
import urllib.request

PORT = 9031
URL = f"http://127.0.0.1:{PORT}/v1/completions"

# 固定 16 条：中英混合、事实/推理/代码/数学各几条，长度错开
PROMPTS = [
    "The capital of France is",
    "简述牛顿第二定律：",
    "Write a Python function to reverse a string:",
    "计算 17 * 23 - 45 = ",
    "Explain the difference between TCP and UDP in one paragraph:",
    "把下面的话翻译成英文：今天天气很好，我们去公园散步。",
    "List three renewable energy sources:",
    "证明：一个三角形的内角和等于180度。简述思路。",
    "def fibonacci(n):",
    "What causes the seasons on Earth?",
    "用一句话解释什么是机器学习：",
    "Solve: if x + 7 = 12, then x =",
    "Summarize the plot of Romeo and Juliet in two sentences:",
    "化学反应式中，催化剂的作用是",
    "Convert 100 kilometers to miles:",
    "Write a haiku about autumn leaves:",
]


def run_one(prompt):
    body = json.dumps({
        "model": "minicpm",
        "prompt": prompt,
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        out = json.loads(r.read())
    return out["choices"][0]["text"]


def run_all():
    return {p: run_one(p) for p in PROMPTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["capture", "compare"])
    ap.add_argument("--out",
                    default="/flagos_比赛/flagos_xx/results/d3_dev/parity_baseline.json")
    ap.add_argument("--baseline", default=None)
    args = ap.parse_args()
    path = args.baseline or args.out

    if args.mode == "capture":
        os.makedirs(os.path.dirname(path), exist_ok=True)
        res = run_all()
        with open(path, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=1)
        print(f"captured {len(res)} prompts -> {path}")
        return

    with open(path) as f:
        base = json.load(f)
    cur = run_all()
    match = 0
    for p in PROMPTS:
        a, b = base[p], cur[p]
        if a == b:
            match += 1
        else:
            i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                     min(len(a), len(b)))
            print(f"[DIFF] {p[:36]!r} @char{i}: base={a[max(0,i-8):i+16]!r} "
                  f"cur={b[max(0,i-8):i+16]!r}")
    print(f"MATCH {match}/{len(PROMPTS)} ({match / len(PROMPTS) * 100:.0f}%)")
    sys.exit(0)


if __name__ == "__main__":
    main()
