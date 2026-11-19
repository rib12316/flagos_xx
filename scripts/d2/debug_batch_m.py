#!/usr/bin/env python3
"""D2 隔离实验：定位 M 依赖的假量化 bug（5 题健康 vs 30 题瞬死 EOS）。

设计：同一道题，batch=1 vs 复制 30 份（greedy 64 tok），四组：
  A ctrl  b=1 / B ctrl b=30 / C L0 b=1 / D L0 b=30
判定：
  C≈A 且 D 早期偏离 C → patch 的 M 依赖 bug → 逐模块数值定位（dump 首步 decode 每
  站点输出 vs batch=1 行切片的最大相对误差）
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_PLUGINS", "fl")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from common import MODEL_DIR, build_messages, load_problems

from vllm import LLM, SamplingParams
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
import fake_quant_eval as fq

PROBE = 24  # 第 24 题（wave1 里 len=2 瞬死的那类）


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    problems, _ = load_problems(30)
    mode = os.environ.get("D2_DEBUG_MODE", "identical")
    if mode == "real30":  # 真实 30 题（错落完成、M 随生成跌破 17 → 垫零路径）
        msgs1 = [build_messages(problems[PROBE]["problem"])]
        msgs30 = [build_messages(p["problem"]) for p in problems]
        max_tok = int(os.environ.get("D2_DEBUG_TOK", "256"))
    else:
        msgs1 = [build_messages(problems[PROBE]["problem"])]
        msgs30 = [build_messages(problems[PROBE]["problem"])] * 30
        max_tok = 64
    sp = SamplingParams(temperature=0.0, max_tokens=max_tok)

    llm = LLM(model=MODEL_DIR, enforce_eager=True, max_model_len=8192,
              gpu_memory_utilization=0.80, seed=1234)
    UnquantizedLinearMethod.apply = fq.patched_apply
    surg = fq.Surgeon(llm)

    surg.control()
    a = llm.chat(msgs1, sp, use_tqdm=False)[0].outputs[0].text
    b_outs = llm.chat(msgs30, sp, use_tqdm=False)
    b = b_outs[0].outputs[0].text

    surg.arm_surgery(fq.parse_arm_arg("r_L0", None))
    c = llm.chat(msgs1, sp, use_tqdm=False)[0].outputs[0].text
    d_outs = llm.chat(msgs30, sp, use_tqdm=False)
    d = d_outs[0].outputs[0].text
    surg.control()

    def head(t, n=90):
        return repr(t[:n])
    print("A ctrl b1 :", head(a))
    print("B ctrl b30:", head(b))
    print("C L0   b1 :", head(c))
    print("D L0   b30:", head(d))
    print("A==B:", a == b, "| C==D:", c == d, "| A~C prefix eq:", a[:20] == c[:20])

    if mode == "real30":
        # 每题首 80 字符 + 与 ctrl 逐题前缀一致长度（找分歧点）
        import json
        rows = []
        for i, (bo, do) in enumerate(zip(b_outs, d_outs)):
            bt = bo.outputs[0].text
            dt = do.outputs[0].text
            k = 0
            while k < min(len(bt), len(dt)) and bt[k] == dt[k]:
                k += 1
            rows.append({"i": i, "id": problems[i]["unique_id"][-12:],
                         "ctrl_len": len(bt), "q_len": len(dt),
                         "prefix_eq_chars": k,
                         "q_head": dt[:70]})
        with open("/tmp/d2debug_real30.json", "w") as f:
            json.dump(rows, f, ensure_ascii=False, indent=1)
        for r in rows[:10]:
            print(f"[{r['i']:2d}] {r['id']:14s} ctrl={r['ctrl_len']:4d} q={r['q_len']:4d} "
                  f"diverge@{r['prefix_eq_chars']:4d}  q_head={r['q_head'][:48]!r}")
        return

    # ---- 逐模块数值定位：batch1 vs batch30 行 0 ----
    x1 = None
    print("\n--- 逐模块 M 依赖定位（同一输入复制 30 行 vs 1 行，量化路径直接调用） ---")
    dev = "cuda"

    def probe_model(model):
        import torch.nn.functional as F
        torch.manual_seed(7)
        # 找 4 类站点各取 L21 一层，真实激活幅度随机近似
        rows = []
        for name, mod in model.named_modules():
            if not fq.site_of(name):
                continue
            li = fq.layer_idx(name)
            if li != 21:
                continue
            site = fq.site_of(name)
            W = mod.weight.data
            K = W.shape[1]
            x = (torch.randn(1, K, device=dev) * 2).to(torch.bfloat16)
            x30 = x.expand(30, K).contiguous()
            recipe = fq.Recipe("probe")
            STATE = fq.STATE
            STATE["recipe"] = recipe
            STATE["keep_pred"] = lambda n: False
            STATE["folds"] = None
            STATE["handlers"] = {}
            fq.STATE["quant_w"] = {}
            fq._PAD_BUF.clear()
            cached = fq._build_quant_weight(mod, name, site, recipe)
            STATE["quant_w"][id(mod)] = cached
            wq, ws, wq_t = cached
            # M=1（垫零路径）
            xq1, xs1 = fq.act_quant(x)
            M = xq1.shape[0]
            buf = torch.zeros(17, K, dtype=torch.int8, device=dev)
            buf[:M].copy_(xq1)
            y1 = torch._int_mm(buf, wq_t)[:M].to(torch.float32).mul_(xs1).mul_(ws)
            # M=30（直连路径）取第 0 行
            xq30, xs30 = fq.act_quant(x30)
            y30 = torch._int_mm(xq30.contiguous(), wq_t)[:30].to(torch.float32).mul_(xs30).mul_(ws)
            rel = ((y30[0] - y1[0]).abs() / (y1[0].abs() + 1e-2)).max().item()
            rows.append((site, rel))
        for site, rel in rows:
            flag = " <-- M 依赖!" if rel > 0.05 else ""
            print(f"L21 {site:10s} max_rel(b30 vs b1) = {rel:.4f}{flag}")
    llm.apply_model(probe_model)


if __name__ == "__main__":
    main()
