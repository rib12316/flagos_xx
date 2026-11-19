#!/usr/bin/env python3
"""D2 Phase 2.0b 激活统计采集：SmoothQuant-lite（L2）的前置。

语料合规（红线）：机上无 wikitext/c4 —— 用模型自生成通用文本（48 条非数学主题
prompt 内嵌常量：新闻/文案/历史/代码讲解/翻译/菜谱/旅行/科普/邮件，零数学内容，
不涉 math_500/gsm8k）。语料选择写入台账与报告。

机制：offline LLM 生成 512 tok/条（temperature=0.8），前向 hook 挂在 168 个融合
Linear 上做逐输入通道 absmax / mean-abs 归约（不存激活）。o_proj 额外维护按
GQA 8 头组聚合的 256 维统计（折叠面向量，GQA 陷阱见 kernel-matrix §4）。
生成过程本身（prefill+decode 全部步）都被 hook 覆盖 → 统计含 prompt+续写 ≈3 万 token。

用法：
  python scripts/d2/collect_act_stats.py --out results/20261106_d2_actstats/
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_PLUGINS", "fl")

import argparse
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import (  # noqa: E402
    GQA_REP, HEAD_DIM, KV_HEADS, MODEL_DIR, NUM_LAYERS,
    result_line, write_provenance,
)

# 48 条通用（非数学）prompt —— 内嵌常量保证可复现（seed 固定生成）
GENERIC_PROMPTS = [
    "Write a short news article about a new city park opening downtown.",
    "Draft a product description for a stainless steel water bottle.",
    "Summarize the causes of the Industrial Revolution in a few paragraphs.",
    "Explain how a refrigerator keeps food cold, for a curious teenager.",
    "Translate the following idea into plain English: persistence beats talent.",
    "Write a recipe for vegetarian chili for four people.",
    "Describe a five-day itinerary for visiting Kyoto in autumn.",
    "Write an email to a landlord asking to fix a leaking faucet.",
    "Explain why the sky is blue during the day and red at sunset.",
    "Write a short story about a librarian who discovers a talking cat.",
    "Summarize the plot of Romeo and Juliet in one paragraph.",
    "Explain the difference between weather and climate.",
    "Write a cover letter for a junior software engineering position.",
    "Describe how coffee is grown, harvested, and roasted.",
    "Write a persuasive paragraph about the benefits of daily walking.",
    "Explain how vaccines train the immune system.",
    "Draft a friendly announcement for a neighborhood garage sale.",
    "Describe the water cycle in simple terms for children.",
    "Write a review of a fictional restaurant called The Golden Spoon.",
    "Explain what blockchain technology is to someone non-technical.",
    "Write a short biography of a fictional explorer.",
    "Describe the rules of football (soccer) for a complete beginner.",
    "Write instructions for planting a small vegetable garden.",
    "Explain why leaves change color in autumn.",
    "Draft a social media post announcing a local book club meeting.",
    "Describe a memorable train journey across the countryside.",
    "Explain how batteries store and release energy.",
    "Write a dialogue between a tourist and a hotel receptionist in Paris.",
    "Summarize the importance of the printing press in history.",
    "Write a paragraph encouraging people to recycle at home.",
    "Explain the role of bees in growing food crops.",
    "Describe your ideal quiet weekend at a lakeside cabin.",
    "Write a short explanation of how the internet delivers a web page.",
    "Draft a thank-you note after a job interview.",
    "Explain the difference between a virus and a bacterium.",
    "Write a scene where two old friends meet after ten years apart.",
    "Describe traditional breakfast foods from three different countries.",
    "Explain how earthquakes happen and how they are measured.",
    "Write a product review for a pair of running shoes.",
    "Summarize how photosynthesis works in a plant leaf.",
    "Draft an itinerary for a one-day museum visit with kids.",
    "Explain why sleep is important for memory.",
    "Write a short essay on the history of the Olympic Games.",
    "Describe the process of making bread from scratch.",
    "Explain what renewable energy sources exist and how they differ.",
    "Write a postcard message from a trip to the mountains.",
    "Explain how gravity keeps the Moon orbiting the Earth.",
    "Describe the work of a lighthouse keeper in the nineteenth century.",
]

SUFFIX = " Please write a few paragraphs."


def site_of(name: str) -> str | None:
    if name.endswith("self_attn.qkv_proj"):
        return "qkv"
    if name.endswith("self_attn.o_proj"):
        return "o"
    if name.endswith("mlp.gate_up_proj"):
        return "gate_up"
    if name.endswith("mlp.down_proj"):
        return "down"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/20261106_d2_actstats/")
    ap.add_argument("--prompts", type=int, default=48)
    ap.add_argument("--gen-tokens", type=int, default=512)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=MODEL_DIR, enforce_eager=True, max_model_len=8192,
              gpu_memory_utilization=0.80, seed=args.seed)

    state = {}   # name -> {"absmax": [K] cpu fp32, "sum": [K], "n": int}
    o_group = {} # o_proj name -> {"absmax256": [256]}

    def install(model):
        for name, mod in model.named_modules():
            site = site_of(name)
            if site is None:
                continue
            st = {"absmax": None, "sum": None, "n": 0}
            state[name] = st

            def make_hook(st, name, site):
                def hook(mod, inp, out):
                    x = inp[0]
                    if not torch.is_tensor(x) or x.ndim < 2:
                        return
                    x2 = x.reshape(-1, x.shape[-1]).abs().to(torch.float32)
                    if st["absmax"] is None:
                        st["absmax"] = torch.zeros(x2.shape[1], device=x2.device)
                        st["sum"] = torch.zeros(x2.shape[1], device=x2.device)
                    torch.maximum(st["absmax"], x2.amax(0), out=st["absmax"])
                    st["sum"] += x2.sum(0)
                    st["n"] += x2.shape[0]
                    if site == "o":
                        # GQA 聚合：[T,2048] → [T,2,8,128].amax(1) → [T,256]
                        T = x2.shape[0]
                        xg = x2.view(T, KV_HEADS, GQA_REP, HEAD_DIM).amax(2)
                        og = o_group[name]
                        if og["absmax256"] is None:
                            og["absmax256"] = torch.zeros(KV_HEADS * HEAD_DIM, device=x2.device)
                        torch.maximum(og["absmax256"], xg.amax(0).reshape(-1), out=og["absmax256"])
                return hook
            if site == "o":
                o_group[name] = {"absmax256": None}
            mod.register_forward_hook(make_hook(st, name, site))
    llm.apply_model(install)

    msgs = [[{"role": "user", "content": p + SUFFIX}] for p in GENERIC_PROMPTS[: args.prompts]]
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=args.gen_tokens, seed=args.seed)
    outs = llm.chat(msgs, sp, use_tqdm=False)
    gen_tokens = sum(len(o.outputs[0].token_ids) for o in outs)

    # 序列化（cpu numpy，6 位有效数字）
    import math
    def ser(t):
        return [None if v is None else round(v, 6) for v in t.tolist()]
    layers = {}
    for name, st in state.items():
        mean = (st["sum"] / max(st["n"], 1)) if st["sum"] is not None else None
        layers[name] = {"absmax": ser(st["absmax"]), "mean_abs": ser(mean), "n_tokens": st["n"]}
    og = {name: {"absmax256": ser(v["absmax256"])} for name, v in o_group.items()}

    os.makedirs(args.out, exist_ok=True)
    meta = {"meta": {
        "prompts": args.prompts, "gen_tokens_total": gen_tokens,
        "seed": args.seed, "corpus": "model-generated generic non-math text (48 embedded prompts, "
        "news/copywriting/history/code-explainer/translation/recipe/travel/science/email; "
        "no math_500/gsm8k content)",
    }, "layers": layers, "o_group": og}
    with open(f"{args.out}/act_stats.json", "w") as f:
        json.dump(meta, f)
    torch.save({"layers": {n: {"absmax": s["absmax"]} for n, s in state.items()},
                "o_group": {n: v["absmax256"] for n, v in o_group.items()}},
               f"{args.out}/act_stats.pt")

    write_provenance(args.out, extra={"prompts": args.prompts, "gen_tokens": gen_tokens,
                                      "seed": args.seed})
    n_sites = sum(1 for n in state if state[n]["absmax"] is not None)
    print(result_line(mode="act_stats", sites=f"{n_sites}/168",
                      o_groups=sum(1 for v in o_group.values() if v["absmax256"] is not None),
                      gen_tokens=gen_tokens, out=f"{args.out}/act_stats.json"))


if __name__ == "__main__":
    main()
