#!/usr/bin/env python3
"""D2 共用件：配方注册表、冻结筛查集加载、evalscope 同口径判分、SmoothQuant-lite 折叠数学。

折叠数学唯一定义处（计划要求 single source of truth），两处共享：
  - scripts/d2/fake_quant_eval.py            进程内活手术（GPU，融合运行时模块）
  - vllm-plugin-FL/tools/quantize_int8_rtn.py 转换脚本（CPU，checkpoint 分离张量）

折叠点数学（逐行核验记录见 results/d2-kernel-matrix.md §4）：
  RMSNorm w 乘在 sqrt 外 → w/=s ⟺ 输出×1/s（residual 流不受影响）
  RoPE 只作用 q/k、attention 对 V 线性 → o_proj 的 s 折进 v_proj 行（GQA：16Q/2KV，
  o 输入通道 (h,d) 对应 v 行 (h//8,d)，s 必须按 8 头一组聚合）
  SiluAndMul=silu(gate)×up 对 up 分支线性 → down_proj 的 s 折进 up_proj 行
  model.norm 永不折（lm_head 不量化）

本文件不得 import vllm（转换脚本环境也要能用）。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field

MODEL_DIR = "/workspace/MiniCPM5-2B"
MATH500 = "/workspace/evalscope-datasets/math_500/test.jsonl"

# ---- 模型结构常量（MiniCPM5-2B，运行时融合口径；断言防呆用） ----
NUM_LAYERS = 42
HIDDEN = 2048
QKV_FUSED_N = 2560          # q 0:2048 | k 2048:2304 | v 2304:2560
GATE_UP_FUSED_N = 12288     # gate 0:6144 | up 6144:12288
KV_HEADS, Q_HEADS, HEAD_DIM = 2, 16, 128
GQA_REP = Q_HEADS // KV_HEADS  # 8


# ---------------------------------------------------------------------------
# 配方
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Recipe:
    rid: str
    clip: float = 1.0        # 权重 per-channel absmax clip 比例（L1a）
    smooth: float = 0.0      # SmoothQuant-lite α（L2），0=关
    keep: str = "none"       # keep-list 预设名（被保留=保 bf16 不量化的模块）
    stats: str | None = None # 激活统计 json（smooth>0 时必填）


def keep_predicate(keep: str):
    """keep 预设 → 运行时融合模块名（如 model.layers.3.self_attn.qkv_proj）的判定函数。

    预设名（按「量化什么」的语言命名，keep=保留不量化）：
      none          全量化（L0）
      attn          保留 attention（qkv/o）→ MLP-only 量化（L4①）
      mlp           保留 MLP（gate_up/down）→ 仅 attn 量化（2.3 探针）
      first_last:K  保留首末各 K 层（L4②）
      band:a-b      只量化层号闭区间 [a,b]，其余保留（2.3 三段分层探针）
      layers:i,j,k  保留指定层（L4③ 敏感层定向，2.3 扫描结果驱动）
    """
    if keep == "none":
        return lambda name: False
    if keep == "attn":
        return lambda name: bool(re.search(r"\.(qkv_proj|o_proj)$", name))
    if keep == "mlp":
        return lambda name: bool(re.search(r"\.(gate_up_proj|down_proj)$", name))
    if keep.startswith("first_last:"):
        k = int(keep.split(":")[1])
        idx = set(range(k)) | set(range(NUM_LAYERS - k, NUM_LAYERS))
        return _layer_idx_pred(idx)
    if keep.startswith("band:"):
        a, b = (int(v) for v in keep[len("band:"):].split("-"))
        idx = set(range(a, b + 1))  # 这是「量化带」，keep = 其余层
        return _layer_idx_pred(set(range(NUM_LAYERS)) - idx)
    if keep.startswith("layers:"):
        idx = {int(v) for v in keep[len("layers:"):].split(",")}
        return _layer_idx_pred(idx)
    raise ValueError(f"unknown keep preset: {keep}")


def _layer_idx_pred(keep_idx: set[int]):
    def pred(name: str) -> bool:
        m = re.search(r"layers\.(\d+)\.", name)
        return bool(m) and int(m.group(1)) in keep_idx
    return pred


# 预注册 arm 集（阶梯实验一次性定义；组合档按结果补跑用 --arms inline 语法）
RECIPES: dict[str, Recipe] = {r.rid: r for r in [
    Recipe("r_L0"),
    Recipe("r_L1a0999", clip=0.999),
    Recipe("r_L1a099", clip=0.99),
    Recipe("r_L1a098", clip=0.98),
    Recipe("r_L1a095", clip=0.95),
    Recipe("r_L2a05", smooth=0.5),
    Recipe("r_L2a06", smooth=0.6),
    Recipe("r_L2a07", smooth=0.7),
    Recipe("r_L2a08", smooth=0.8),
    # 2.3 敏感度探针（不叠加 clip/smooth，基于 L0）
    Recipe("r_probe_mlp_only", keep="mlp"),        # 只量化 attention
    Recipe("r_probe_attn_only", keep="attn"),      # 只量化 MLP（=L4① MLP-only）
    Recipe("r_probe_band0_13", keep="band:0-13"),
    Recipe("r_probe_band14_27", keep="band:14-27"),
    Recipe("r_probe_band28_41", keep="band:28-41"),
    # L4 首末层保留
    Recipe("r_keep_fl2", keep="first_last:2"),
    # L4 敏感层定向（占位，待 2.3 结果填层号）
    Recipe("r_keep_sens", keep="layers:0,1,40,41"),
]}


def parse_arm(spec: str, default_stats: str | None = None) -> Recipe:
    """解析 arm：'r_L0'（注册表）或 'rid:k=v,k=v'（inline）。"""
    if spec in RECIPES:
        return RECIPES[spec]
    rid, _, kv = spec.partition(":")
    kwargs = {}
    for item in filter(None, kv.split(",")):
        k, _, v = item.partition("=")
        if k in ("clip", "smooth"):
            kwargs[k] = float(v)
        elif k in ("keep", "stats"):
            kwargs[k] = v
        else:
            raise ValueError(f"unknown arm kwarg {k} in {spec}")
    return Recipe(rid, stats=default_stats, **kwargs)


# ---------------------------------------------------------------------------
# 冻结筛查集
# ---------------------------------------------------------------------------
def load_problems(n: int, level: int | str = 3) -> tuple[list[dict], str]:
    """math_500 Level 3，按 unique_id 排序取前 n。返回 (题目列表, 冻结 hash)。

    level 字段官方为整数 3（任务书误写为字符串，兼容两种）。
    """
    out = []
    with open(MATH500) as f:
        for line in f:
            rec = json.loads(line)
            if int(rec["level"]) == int(level):
                out.append(rec)
    out.sort(key=lambda r: r["unique_id"])
    frozen = out[:n]
    h = hashlib.sha256("|".join(r["unique_id"] for r in frozen).encode()).hexdigest()[:12]
    return frozen, h


def build_messages(problem: str) -> list[dict]:
    """与官方 evalscope 完全同口径的 user 消息（已对 20261105 官方 review 记录逐字核对）。"""
    return [{"role": "user",
             "content": f"{problem}\nPlease reason step by step, and put your final answer within \\boxed{{}}."}]


# ---------------------------------------------------------------------------
# 判分（与 evalscope 官方链路同函数）
# ---------------------------------------------------------------------------
def _scoring_fns():
    from evalscope.metrics.math.parser import (
        extract_answer,
        math_equal,
        strip_answer_string,
    )
    return extract_answer, math_equal, strip_answer_string


def extract_pred(text: str) -> str | None:
    fn = _scoring_fns()[0]
    return fn(text, use_last_number=True)


def is_correct(pred: str | None, ref_answer: str) -> bool:
    _, math_equal, strip_answer_string = _scoring_fns()
    if pred is None:
        return False
    try:
        return bool(math_equal(pred, strip_answer_string(ref_answer)))
    except Exception:
        return pred == ref_answer


def same_answer(pred_q: str | None, pred_c: str | None) -> bool:
    """同题答案一致率分子：量化臂 vs bf16 对照臂的答案判等。"""
    _, math_equal, strip_answer_string = _scoring_fns()
    if pred_q is None and pred_c is None:
        return True
    if pred_q is None or pred_c is None:
        return False
    try:
        return bool(math_equal(pred_q, strip_answer_string(pred_c)))
    except Exception:
        return pred_q == pred_c


# ---------------------------------------------------------------------------
# SmoothQuant-lite 折叠数学（唯一定义处）
# ---------------------------------------------------------------------------
@dataclass
class FoldScales:
    """单层的 4 个站点折叠向量（全部 ≤1，max=1 → 权重只缩不放）。"""
    s_attn: "object"    # [2048] qkv 站点：input_layernorm 输出
    s_mlp_in: "object"  # [2048] gate_up 站点：post_attention_layernorm 输出
    s_o: "object"       # [256]  o_proj 站点：v 输出（KV 头聚合后）
    s_mlp_mid: "object" # [6144] down_proj 站点：SiluAndMul 输出


def smooth_scale(absmax_per_channel, alpha: float, eps: float = 1e-6):
    """s = (A/A.max())^α，clamp 防 0。s∈(0,1]，x/=s 压平通道剖面，W 列 *=s 补偿。"""
    import torch
    a = absmax_per_channel.to(torch.float32)
    s = (a / a.max().clamp_min(eps)).pow(alpha)
    return s.clamp_min(1e-6)


def load_fold_scales(stats_path: str, alpha: float) -> dict[int, FoldScales]:
    """从激活统计 json 计算逐层 FoldScales。键 = 层号。"""
    with open(stats_path) as f:
        stats = json.load(f)
    layers_in = stats["layers"]
    scales = {}
    for li in range(NUM_LAYERS):
        def A(suffix, sub=None):
            node = layers_in[f"model.layers.{li}.{suffix}"]
            import torch
            return torch.tensor(node[sub or "absmax"], dtype=torch.float32)
        scales[li] = FoldScales(
            s_attn=smooth_scale(A("self_attn.qkv_proj"), alpha),
            s_mlp_in=smooth_scale(A("mlp.gate_up_proj"), alpha),
            s_o=smooth_scale(torch.tensor(
                stats["o_group"][f"model.layers.{li}.self_attn.o_proj"]["absmax256"],
                dtype=torch.float32), alpha),
            s_mlp_mid=smooth_scale(A("mlp.down_proj"), alpha),
        )
    return scales


def s_o_expand(s_o):
    """[256] → [2048]：o_proj 输入通道 (h,d) ← v 通道 (h//GQA_REP, d)。"""
    return s_o.view(KV_HEADS, 1, HEAD_DIM).expand(KV_HEADS, GQA_REP, HEAD_DIM).reshape(-1)


def apply_folds_separate(tensors: dict, li: int, folds: FoldScales,
                         quant: dict[str, bool]) -> dict:
    """转换脚本口径：对一层 checkpoint 分离张量做折叠（就地返回新 dict）。

    quant: 站点 → 是否量化（keep-list 站点跳过其列折叠；但 v/up 行折叠仍要
    应用——它服务 o/down 的量化，bf16 层做纯数学缩放同样精确）。
    """
    t = dict(tensors)
    if folds is None:
        return t
    s_a, s_mi, s_o, s_mm = folds.s_attn, folds.s_mlp_in, folds.s_o, folds.s_mlp_mid
    # norm 折叠：仅当消费者量化
    if quant["qkv"]:
        t["input_layernorm.weight"] = t["input_layernorm.weight"] / s_a
    if quant["gate_up"]:
        t["post_attention_layernorm.weight"] = t["post_attention_layernorm.weight"] / s_mi
    # q/k/v：列 *= s_attn（若量化）；v 行 *= 1/s_o（若 o 量化，无论自身是否量化）
    if quant["qkv"]:
        t["q_proj.weight"] = t["q_proj.weight"] * s_a[None, :]
        t["k_proj.weight"] = t["k_proj.weight"] * s_a[None, :]
    t["v_proj.weight"] = t["v_proj.weight"] * (s_a[None, :] if quant["qkv"] else 1.0)
    if quant["o"]:
        t["v_proj.weight"] = t["v_proj.weight"] / s_o[:, None]
    # o_proj：列 *= s_o 扩张（若量化）
    if quant["o"]:
        t["o_proj.weight"] = t["o_proj.weight"] * s_o_expand(s_o)[None, :]
    # gate/up：列 *= s_mlp_in（若量化）；up 行 *= 1/s_mlp_mid（若 down 量化）
    if quant["gate_up"]:
        t["gate_proj.weight"] = t["gate_proj.weight"] * s_mi[None, :]
        t["up_proj.weight"] = t["up_proj.weight"] * s_mi[None, :]
    if quant["down"]:
        t["up_proj.weight"] = t["up_proj.weight"] / s_mm[:, None]
    # down：列 *= s_mlp_mid（若量化）
    if quant["down"]:
        t["down_proj.weight"] = t["down_proj.weight"] * s_mm[None, :]
    return t


# ---------------------------------------------------------------------------
# 权重 RTN（L1a clip 网格）
# ---------------------------------------------------------------------------
def rtn_channel_quant(w_fp32, clip: float):
    """per-channel 对称 RTN：scale=α·absmax_row/127，round-half-even，clamp ±127。

    与运行时 round 语义一致（torch.round fp32）。返回 (int8, scale_fp32[N])。
    """
    import torch
    a = w_fp32.abs().amax(dim=1) * float(clip)
    scale = (a / 127.0).clamp_min(1e-12)
    q = (w_fp32 / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return q, scale


def result_line(**kv) -> str:
    """scripts/ 惯例：stdout 单行 RESULT 可 grep。"""
    return "RESULT|" + "|".join(f"{k}={v}" for k, v in kv.items())


def write_provenance(outdir: str, extra: dict | None = None):
    import subprocess
    import datetime
    os.makedirs(outdir, exist_ok=True)
    meta = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "env": {"VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
                "VLLM_PLUGINS": os.environ.get("VLLM_PLUGINS")},
    }
    try:
        repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        meta["git"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            cwd=repo).stdout.strip()
    except Exception:
        pass
    if extra:
        meta.update(extra)
    with open(os.path.join(outdir, "PROVENANCE"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
