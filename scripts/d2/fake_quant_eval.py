#!/usr/bin/env python3
"""D2 Phase 2 假量化筛查 harness：offline LLM + monkeypatch UnquantizedLinearMethod.apply。

机制（计划 Phase 2.0，细节与核验见 results/d2-kernel-matrix.md）：
  - VLLM_ENABLE_V1_MULTIPROCESSING=0 + llm.apply_model → 进程内活体模型；patch 在
    LLM() 返回后安装（enforce_eager 必须开，否则编译路径烙死原 apply）
  - lm_head/embed 走 UnquantizedEmbeddingMethod 天然不进 patch；非四站点 Linear 直通
  - emu=int32（默认）：逐 token 假量化后 fp32 整值矩阵乘（int8 积 ≤16129 在 fp32 精确，
    与 Triton kernel 的 int32 累加仅差 >2^24 病态和的 1 ulp）——kernel 忠实仿真
  - emu=bf16：反量化 bf16 F.linear，仅粗排（大网格才用）
  - SmoothQuant 折叠 = 进程内活手术：norm 权重与 keep-list 层的 v/up 行折叠就地改，
    量化层的列折叠进权重缓存；arm 间从 pristine 快照完整恢复（头号交叉污染防线）

用法：
  python scripts/d2/fake_quant_eval.py --arms r_ctrl,r_L0 --problems 30 \
      --stats results/20261106_d2_actstats/act_stats.json \
      --control-cache results/20261106_d2_fakequant/control.json \
      --out results/20261106_d2_fakequant/
  python scripts/d2/fake_quant_eval.py --mse-scan        # L1a 权重 MSE 预扫（无 GPU 模型加载）

判据（筛查阶段）：same_ans 为主指标 + Δacc；promote 规则见 results/d2-precision-search.md。
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_PLUGINS", "fl")
# bisect 调试开关（D2_DEBUG="quant=ref,pad=new,alloc=off"）——必须在 torch/cuda 初始化前解析
_D2_DEBUG = dict(
    kv.split("=", 1) for kv in os.environ.get("D2_DEBUG", "").split(",") if "=" in kv)
# 假量化臂每步产生多形状临时张量，expandable_segments 抑制分配器碎片引发的同步 cudaMalloc 风暴
if _D2_DEBUG.get("alloc") != "off":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

from common import (  # noqa: E402
    GQA_REP, HEAD_DIM, KV_HEADS, MODEL_DIR, NUM_LAYERS,
    Recipe, build_messages, extract_pred, is_correct, keep_predicate,
    load_fold_scales, load_problems, rtn_channel_quant, result_line,
    same_answer, write_provenance,
)

SITES = ("qkv", "o", "gate_up", "down")


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


def layer_idx(name: str) -> int | None:
    import re
    m = re.search(r"layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# 全局 patch 状态（单进程串行 arm，无并发）
# ---------------------------------------------------------------------------
STATE: dict = {
    "recipe": None,        # None → 直通（对照臂）
    "folds": None,         # {li: FoldScales(GPU)}
    "keep_pred": lambda n: False,
    "quant_w": {},         # id(layer) -> (wq int8 [N,K], ws fp32 [N], wq_t int8 [K,N])
    "handlers": {},        # id(layer) -> site（本 arm 需量化的模块；keep/非站点不在表 = 直通）
    "name_by_id": {},
    "emu": "int32",
    "cache_builds": 0,
}

_PAD_BUF: dict = {}   # K -> zeros[17,K] int8（_int_mm 要求 M>16 的小 M 垫零缓冲）

from vllm.model_executor.layers.linear import UnquantizedLinearMethod  # noqa: E402

try:  # 默认纯 torch 逐 token 量化（标准 CUDA 流，无竞争；实测 libentry Triton 启动器
      # 在本进程内存在流竞争——b30 读到未就绪输入，见台账 11-07 记录与 debug_batch_m.py）
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8 as _ref_quant,
    )

    def act_quant(x2):
        return _ref_quant(x2.contiguous())
except Exception:  # pragma: no cover
    act_quant = None

def _act_quant_triton(x2):  # D2_DEBUG=quant=triton 才启用（诊断用）
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_flaggems_triton as tq,
    )
    return tq(x2)

if _D2_DEBUG.get("quant") == "triton":
    act_quant = _act_quant_triton

ORIG_APPLY = UnquantizedLinearMethod.apply


def _site_scales(li: int, site: str):
    f = STATE["folds"][li]
    return {"qkv": f.s_attn, "o": None, "gate_up": f.s_mlp_in, "down": f.s_mlp_mid}[site]


def _build_quant_weight(layer, name: str, site: str, recipe: Recipe):
    """从（可能已含行折叠的）layer.weight 构建 int8/scale 缓存；列折叠在此应用。

    返回 (wq int8 [N,K], ws fp32 [N], wq_t int8 [K,N] contiguous)——wq_t 供
    torch._int_mm（实测与本 box cuBLAS int8 GEMM 位精确一致，5-10x 快于 fp32 仿真）。
    """
    W = layer.weight.data.to(torch.float32)
    li = layer_idx(name)
    if STATE["folds"] is not None and li is not None:
        f = STATE["folds"][li]
        if site == "qkv":
            W = W * f.s_attn[None, :]
            if not STATE["keep_pred"](name.replace("qkv_proj", "o_proj")):
                W[2304:2560, :] = W[2304:2560, :] / f.s_o[:, None]
        elif site == "o":
            from common import s_o_expand
            W = W * s_o_expand(f.s_o)[None, :]
        elif site == "gate_up":
            W = W * f.s_mlp_in[None, :]
            if not STATE["keep_pred"](name.replace("gate_up_proj", "down_proj")):
                W[6144:12288, :] = W[6144:12288, :] / f.s_mlp_mid[:, None]
        elif site == "down":
            W = W * f.s_mlp_mid[None, :]
    wq, ws = rtn_channel_quant(W, recipe.clip)
    STATE["cache_builds"] += 1
    return wq, ws, wq.t().contiguous()


HAS_INT_MM = hasattr(torch, "_int_mm")

from vllm_fl.quantization.w8a8.reference import dynamic_per_token_quant_int8 as _ref_quant_debug  # noqa: E402


def patched_apply(self, layer, x, bias=None):
    if _D2_DEBUG.get("sync") == "1":
        torch.cuda.synchronize()  # 诊断：全同步排除流竞争
    if id(layer) not in STATE["handlers"]:
        return ORIG_APPLY(self, layer, x, bias)
    lid = id(layer)
    cached = STATE["quant_w"].get(lid)
    if cached is None:
        name = getattr(layer, "prefix", "") or STATE["name_by_id"].get(lid, "")
        cached = _build_quant_weight(layer, name, STATE["handlers"][lid], STATE["recipe"])
        STATE["quant_w"][lid] = cached
    wq, ws, wq_t = cached

    x2 = x.reshape(-1, x.shape[-1])
    if _D2_DEBUG.get("quant") == "ref":  # bisect 开关
        xq, xs = _ref_quant_debug(x2)
    else:
        xq, xs = act_quant(x2)
    if STATE["emu"] == "int32":
        M, K = xq.shape
        if HAS_INT_MM:
            if M < 17:  # cuBLAS int8 GEMM 要求 M>16：垫零（零行不影响结果）
                if _D2_DEBUG.get("pad") == "new":
                    buf = xq.new_zeros(17, K)
                else:
                    buf = _PAD_BUF.get(K)
                    if buf is None:
                        buf = torch.zeros(17, K, dtype=torch.int8, device=xq.device)
                        _PAD_BUF[K] = buf
                buf[:M].copy_(xq)
                xq = buf
            y = torch._int_mm(xq, wq_t)[:M].to(torch.float32).mul_(xs).mul_(ws)
        else:
            y = xq.to(torch.float32) @ wq.to(torch.float32).t()
            y = y.mul_(xs).mul_(ws)
    else:  # bf16 粗排
        w_dq = (wq.to(torch.float32) * ws[:, None]).to(x.dtype)
        x_dq = (xq.to(torch.float32) * xs).to(x.dtype)
        y = F.linear(x_dq, w_dq)
    if bias is not None:
        y = y + bias
    return y.reshape(*x.shape[:-1], -1).to(x.dtype)


# ---------------------------------------------------------------------------
# 模型手术：快照 / 恢复 / 折叠
# ---------------------------------------------------------------------------
class Surgeon:
    def __init__(self, llm):
        self.llm = llm
        self.pristine = {}   # name -> CPU tensor
        self.modules = {}    # fused/norm name -> module

        def scan(model):
            for name, mod in model.named_modules():
                if site_of(name) or name.endswith(("input_layernorm", "post_attention_layernorm")):
                    self.modules[name] = mod
                    w = mod.weight
                    self.pristine[name] = w.detach().to("cpu", copy=True)
            STATE["name_by_id"] = {id(m): n for n, m in model.named_modules()}
        llm.apply_model(scan)
        assert len(self.pristine) == NUM_LAYERS * 6, f"expect {NUM_LAYERS*6} tensors, got {len(self.pristine)}"

    def restore(self):
        def _rest(model):
            for name, mod in self.modules.items():
                mod.weight.data.copy_(self.pristine[name].to(mod.weight.device, non_blocking=True))
        self.llm.apply_model(_rest)
        STATE["quant_w"].clear()

    def arm_surgery(self, recipe: Recipe):
        """设置 patch 状态 + 施加 keep-list 依赖的就地折叠（norm / keep 层的 v、up 行）。"""
        STATE["recipe"] = recipe
        STATE["keep_pred"] = keep_predicate(recipe.keep)

        def build_handlers(model):
            keep = STATE["keep_pred"]
            handlers = {}
            for name, mod in model.named_modules():
                site = site_of(name)
                if site is not None and not keep(name):
                    handlers[id(mod)] = site
            STATE["handlers"] = handlers
        self.llm.apply_model(build_handlers)
        if recipe.smooth > 0:
            assert recipe.stats, "smooth>0 需要 --stats 激活统计文件"
            folds = load_fold_scales(recipe.stats, recipe.smooth)
            def _to_gpu(model):
                dev = next(model.parameters()).device
                for li, f in folds.items():
                    f.s_attn = f.s_attn.to(dev); f.s_mlp_in = f.s_mlp_in.to(dev)
                    f.s_o = f.s_o.to(dev); f.s_mlp_mid = f.s_mlp_mid.to(dev)
            self.llm.apply_model(_to_gpu)
            STATE["folds"] = folds
        else:
            STATE["folds"] = None

        keep = STATE["keep_pred"]
        if STATE["folds"] is None:
            return

        def _surgery(model):
            for li in range(NUM_LAYERS):
                f = STATE["folds"][li]
                qkv_n = f"model.layers.{li}.self_attn.qkv_proj"
                o_n = qkv_n.replace("qkv_proj", "o_proj")
                gu_n = f"model.layers.{li}.mlp.gate_up_proj"
                dn_n = gu_n.replace("gate_up_proj", "down_proj")
                # norm 折叠：仅当消费者量化（fp32 除后单次舍入回 bf16，避免原地 dtype 报错）
                if not keep(qkv_n):
                    w = self.modules[f"model.layers.{li}.input_layernorm"].weight
                    w.data.copy_((w.data.to(torch.float32) / f.s_attn).to(w.dtype))
                if not keep(gu_n):
                    w = self.modules[f"model.layers.{li}.post_attention_layernorm"].weight
                    w.data.copy_((w.data.to(torch.float32) / f.s_mlp_in).to(w.dtype))
                # keep-list 层的行折叠就地做（量化层在缓存构建里做，避免双折）
                if keep(qkv_n) and not keep(o_n):
                    w = self.modules[qkv_n].weight
                    rows = w.data[2304:2560, :]
                    rows.copy_((rows.to(torch.float32) / f.s_o[:, None]).to(w.dtype))
                if keep(gu_n) and not keep(dn_n):
                    w = self.modules[gu_n].weight
                    rows = w.data[6144:12288, :]
                    rows.copy_((rows.to(torch.float32) / f.s_mlp_mid[:, None]).to(w.dtype))
        self.llm.apply_model(_surgery)

    def control(self):
        STATE["recipe"] = None
        STATE["folds"] = None
        STATE["keep_pred"] = lambda n: False
        STATE["handlers"] = {}
        STATE["quant_w"].clear()


# ---------------------------------------------------------------------------
# 生成与判分
# ---------------------------------------------------------------------------
def run_arm(llm, problems, max_tokens: int, seed: int):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=seed)
    msgs = [build_messages(p["problem"]) for p in problems]
    t0 = time.time()
    outs = llm.chat(msgs, sp, use_tqdm=False)
    dt = time.time() - t0
    recs = []
    for p, o in zip(problems, outs):
        text = o.outputs[0].text
        pred = extract_pred(text)
        recs.append({
            "unique_id": p["unique_id"],
            "ref": p["answer"],
            "pred": pred,
            "ok": is_correct(pred, p["answer"]),
            "len": len(o.outputs[0].token_ids),
            "truncated": o.outputs[0].finish_reason == "length",
        })
    return recs, dt


def summarize(quant_recs, ctrl_recs):
    n = len(quant_recs)
    acc_q = sum(r["ok"] for r in quant_recs) / n
    acc_c = sum(r["ok"] for r in ctrl_recs) / n
    same = sum(same_answer(q["pred"], c["pred"]) for q, c in zip(quant_recs, ctrl_recs)) / n
    ml_q = sum(r["len"] for r in quant_recs) / n
    ml_c = sum(r["len"] for r in ctrl_recs) / n
    return {
        "n": n, "acc_quant": round(acc_q, 4), "acc_ctrl": round(acc_c, 4),
        "same_ans": round(same, 4), "dacc": round(acc_q - acc_c, 4),
        "mean_len_quant": round(ml_q, 1), "mean_len_ctrl": round(ml_c, 1),
        "dlen_pct": round(100 * (ml_q - ml_c) / ml_c, 2),
        "trunc_quant": sum(r["truncated"] for r in quant_recs),
        "trunc_ctrl": sum(r["truncated"] for r in ctrl_recs),
    }


# ---------------------------------------------------------------------------
# L1a MSE 预扫（无 GPU 模型加载）
# ---------------------------------------------------------------------------
def mse_scan(outdir: str):
    from safetensors.torch import load_file
    idx = json.load(open(f"{MODEL_DIR}/model.safetensors.index.json"))
    need = [k for k in idx["weight_map"] if any(
        k.endswith(s) for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight",
                                "o_proj.weight", "gate_proj.weight", "up_proj.weight",
                                "down_proj.weight"))]
    shard_file = sorted(set(idx["weight_map"].values()))[0]  # 单 shard 模型
    all_t = load_file(f"{MODEL_DIR}/{shard_file}")
    alphas = [1.0, 0.999, 0.99, 0.98, 0.95]
    table = {a: 0.0 for a in alphas}
    samples = []
    for k in need:
        W = all_t[k].to(torch.float32)
        for a in alphas:
            wq, ws = rtn_channel_quant(W, a)
            rec = (wq.to(torch.float32) * ws[:, None] - W)
            mse = (rec * rec).mean().item()
            table[a] += mse
            if "layers.20." in k and k.endswith(("q_proj.weight", "gate_proj.weight")):
                samples.append({"tensor": k, "alpha": a, "mse": mse})
    total = {a: table[a] / len(need) for a in alphas}
    best = min(total, key=total.get)
    out = {"per_alpha_mean_mse": total, "best_alpha": best, "n_tensors": len(need),
           "samples": samples}
    os.makedirs(outdir, exist_ok=True)
    with open(f"{outdir}/mse_scan.json", "w") as f:
        json.dump(out, f, indent=2)
    print(result_line(mode="mse_scan", best_alpha=best,
                      **{f"mse_a{str(a).replace('.', '')}": f"{v:.3e}" for a, v in total.items()}))
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="r_ctrl,r_L0")
    ap.add_argument("--problems", type=int, default=30)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--emu", choices=["int32", "bf16"], default="int32")
    ap.add_argument("--stats", default="results/20261106_d2_actstats/act_stats.json")
    ap.add_argument("--control-cache", default="results/20261106_d2_fakequant/control_greedy30.json")
    ap.add_argument("--out", default="results/20261106_d2_fakequant/")
    ap.add_argument("--pilot", action="store_true", help="只跑前 5 题 r_ctrl+r_L0 验截断率")
    ap.add_argument("--mse-scan", action="store_true")
    args = ap.parse_args()

    if args.mse_scan:
        mse_scan(args.out)
        return

    from vllm import LLM
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    n_probs = 5 if args.pilot else args.problems
    problems, frozen = load_problems(n_probs)
    print(f"frozen set: {n_probs} problems, sha={frozen}")

    llm = LLM(model=MODEL_DIR, enforce_eager=True, max_model_len=8192,
              gpu_memory_utilization=0.80, seed=args.seed)
    STATE["emu"] = args.emu
    UnquantizedLinearMethod.apply = patched_apply
    surg = Surgeon(llm)

    # 对照臂（缓存跨 arm 复用；非 30 题规模自动换名，避免污染正式冻结集缓存）
    ctrl_path = args.control_cache if n_probs == 30 else \
        os.path.join(args.out, f"control_greedy{n_probs}.json")
    if os.path.exists(ctrl_path):
        cached = json.load(open(ctrl_path))
        assert len(cached["records"]) == n_probs, "control cache size mismatch"
        assert cached.get("max_tokens") == args.max_tokens, "control cache max_tokens mismatch"
        ctrl_recs = cached["records"]
        print(f"control loaded from {ctrl_path}")
    else:
        ctrl_recs, dt = run_arm(llm, problems, args.max_tokens, args.seed)
        os.makedirs(os.path.dirname(ctrl_path), exist_ok=True)
        json.dump({"records": ctrl_recs, "frozen": frozen, "max_tokens": args.max_tokens},
                  open(ctrl_path, "w"), ensure_ascii=False, indent=1)
        print(result_line(rid="r_ctrl", acc=round(sum(r['ok'] for r in ctrl_recs) / n_probs, 4),
                          trunc=sum(r["truncated"] for r in ctrl_recs), n=n_probs, sec=round(dt, 1)))

    write_provenance(args.out, extra={"arms": args.arms, "emu": args.emu, "frozen": frozen,
                                      "problems": n_probs, "max_tokens": args.max_tokens})

    arms = [a.strip() for a in args.arms.split(",") if a.strip() and a.strip() != "r_ctrl"]
    summary_all = {}
    for spec in arms:
        recipe = parse_arm_arg(spec, args.stats)
        surg.restore()
        surg.arm_surgery(recipe)
        t0 = time.time()
        recs, _ = run_arm(llm, problems, args.max_tokens, args.seed)
        s = summarize(recs, ctrl_recs)
        s["arm_sec"] = round(time.time() - t0, 1)
        s["cache_builds"] = STATE["cache_builds"]; STATE["cache_builds"] = 0
        summary_all[recipe.rid] = s
        payload = {"recipe": {"rid": recipe.rid, "clip": recipe.clip, "smooth": recipe.smooth,
                              "keep": recipe.keep}, "summary": s, "frozen": frozen,
                   "records": recs, "emu": args.emu}
        with open(f"{args.out}/{recipe.rid}.json", "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(result_line(rid=recipe.rid, emu=args.emu, clip=recipe.clip, smooth=recipe.smooth,
                          keep=recipe.keep, acc_q=s["acc_quant"], acc_ctrl=s["acc_ctrl"],
                          same=s["same_ans"], dacc=s["dacc"], dlen=f"{s['dlen_pct']}%",
                          trunc=s["trunc_quant"], n=s["n"], sec=s["arm_sec"]))
        surg.restore()
        surg.control()

    if summary_all:
        best = max(summary_all, key=lambda r: (summary_all[r]["same_ans"], summary_all[r]["acc_quant"]))
        print(result_line(best=best, **{"best_same": summary_all[best]["same_ans"]}))


def parse_arm_arg(spec: str, stats_default: str | None) -> Recipe:
    from common import parse_arm
    r = parse_arm(spec)
    if r.smooth > 0 and not r.stats:
        return Recipe(r.rid, r.clip, r.smooth, r.keep, stats_default)
    return r


if __name__ == "__main__":
    main()
