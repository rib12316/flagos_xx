# SPDX-License-Identifier: Apache-2.0
# 2026 - Modified by team infragogogo (flagos_xx). All rights reserved.
"""MetaX C500 采样 top-p 过滤 torch 快路径（D5，docs/二期计划-1120.md Phase 2 #1）。

机理（20261120 P1-B4 实测，scripts/d5/sampling_probe_metax.py）：
- vendor patch 把 apply_top_k_top_p 永久指向 PyTorch fallback：bs=64 [64,130560] fp32
  实测 **9.66ms/步**（其中 torch.sort 仅 2.6ms，softmax/cumsum/masked_fill/scatter 链
  另占 7.0ms）；16K 稳态步时 ~67ms → **占 ITL ~14%**。
- vLLM Triton kernel（Qrita pivot 式）在沐曦 triton3.0 **编译通过但运行期 ATU 非法
  地址**（BLOCK 8192/4096/2048/1024 五档全崩，results/20261120_c500_reprofiling/
  sampling_metax.csv）→ kernel 路线在本栈判死，只能 torch 原生重写。

两级设计（分布等价 + 任意分布兜底）：
- 快路径：top-k(K) 降序 + 全词表 logsumexp 分母 → 阈值 τ（第 m 大，降序累计质量
  首达 p）→ ``logits < τ`` 整体置 -inf。免全词表 sort、免 scatter、免逐位回写。
  K 内累计质量未达 p（病态平坦分布）时返回 None，调用方整批回退慢路径。
- 慢路径（位精确）：原版 ``apply_top_k_top_p_pytorch``（vendor patch 现行为）。

与 fallback 的语义差（T0 对拍 + 105 题双 gate 的依据）：
  1) 浮点累加序：降序部分和 vs 升序全量和——边界 token 偶有差异（分布等价级）；
  2) 并列 tie：``logits < τ`` 保留所有 =τ 的 token（fallback 按序数截断）——
     fp32 全词表精确并列罕见，保留侧偏差可忽略。
env：``VLLM_FL_SAMPLE_FAST=0`` 关闭（默认开）；``VLLM_FL_SAMPLE_FAST_K`` 调 K（默认 1024）。
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("VLLM_FL_SAMPLE_FAST", "1") == "1"
_K = max(16, int(os.environ.get("VLLM_FL_SAMPLE_FAST_K", "1024")))
_MIN_BS = int(os.environ.get("VLLM_FL_SAMPLE_FAST_MIN_BS", "8"))
_DEBUG = os.environ.get("VLLM_FL_SAMPLE_FAST_DEBUG", "0") == "1"
_neg_inf = float("-inf")


def top_p_filter_fast(logits: torch.Tensor, p: torch.Tensor, k: int = _K):
    """TOPP-only 快路径 v2（sort 基座）。返回过滤后 logits（新张量）；异常由调用方回退。

    v1 教训（20261120 T1）：torch.topk 在 serve 上下文抛
    "Triton Error [MACA]: memory size or pointer value too large to fit in 32 bit"
    （离线任何形状/48GB 压力下均不复现 → 弃 topk），v2 改用与 fallback 同源的
    torch.sort（serve 内每日实测可用）：

        s, _ = logits.sort(-1, descending=True)   # 2.6ms（vendor sort，与 fallback 同引擎）
        c   = softmax(s).cumsum(-1)               # 降序累计（fallback 数学，方向翻转）
        τ   = s[每行首个 c≥p 的位置]              # 阈值 = 第 m 大
        out = logits.masked_fill(logits < τ, -inf) # 免 scatter 全宽回写/免双重 masked_fill

    vs fallback（sort 升序 + softmax + cumsum + masked_fill×2 + scatter 回写 = 9.66ms）：
    省 scatter（0.75ms）+ 全宽 masked_fill 链（~4ms）+ bool 中间张量，实测口径见 T1。
    语义差：① 累加序（降序 vs 升序）边界 token 偶差一位；② 并列 τ 全保留 vs 按序截断
    ——分布等价级（T0 对拍 + 105 题双 gate）。
    """
    with torch.no_grad():
        s, _ = torch.sort(logits, dim=-1, descending=True)      # [bs,V] 降序
        probs = torch.softmax(s, dim=-1)                        # 与 fallback 同式
        c = probs.cumsum(dim=-1)                                # 降序累计
        # 每行保留前缀 = 首个 c≥p 的位置（含）——与 fallback 升序口径逐位对偶
        need = (c < p.unsqueeze(1)).sum(dim=-1)                 # [bs] 0..V-1
        need = need.clamp(max=logits.shape[1] - 1)              # 保底：最后一列永不 mask
        tau = s.gather(1, need.unsqueeze(1))                    # [bs,1] 阈值
        return logits.masked_fill(logits < tau, _neg_inf)


def install() -> bool:
    """把 vendor 的 no-triton 分发替换为 快路径→fallback 两级。幂等。"""
    if not _ENABLED:
        logger.info("sample_fast_path: VLLM_FL_SAMPLE_FAST=0，保持 vendor fallback")
        return False
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as _tts
    except Exception as e:
        logger.warning("sample_fast_path: topk_topp_sampler 不可用: %s", e)
        return False
    if getattr(_tts, "_fl_sample_fast_patched", False):
        return True

    orig = _tts.apply_top_k_top_p  # vendor patch 后的 no-triton 分发（或上游）

    stats = {"n": 0, "fallback": 0, "dbg": 0}

    def _dbg(logits, p):
        """T1 诊断：top-K 质量/耗时/步长（仅 VLLM_FL_SAMPLE_FAST_DEBUG=1 时前 60 次调用）"""
        import time
        torch.cuda.synchronize()
        t0 = time.time()
        kk = min(_K, logits.shape[1])
        tv, _ = torch.topk(logits, kk, dim=-1)
        rm = tv[:, :1]
        ex = torch.exp(logits - rm)
        z = ex.sum(dim=-1, keepdim=True)
        pt = torch.exp(tv - rm) / z
        ct = pt.cumsum(dim=-1)
        cmin = ct[:, -1].min().item()
        cmed = ct[:, -1].median().item()
        torch.cuda.synchronize()
        logger.info(
            "sample_fast DEBUG[%d]: bs=%d stride=%s topK_mass min=%.4f med=%.4f "
            "attempt=%.2fms", stats["dbg"], logits.shape[0],
            str(logits.stride()), cmin, cmed, (time.time() - t0) * 1e3)

    def _fl_apply(logits, k, p, *args, **kwargs):
        if _DEBUG and stats["dbg"] < 60 and p is not None and k is None \
                and logits.is_cuda and logits.ndim == 2:
            stats["dbg"] += 1
            try:
                _dbg(logits, p)
            except Exception as e:
                logger.warning("sample_fast DEBUG 失败: %s", e)
        if (p is not None and k is None
                and logits.is_cuda and logits.ndim == 2
                and logits.dtype == torch.float32
                and logits.shape[0] >= _MIN_BS):
            try:
                out = top_p_filter_fast(logits, p)
                if out is not None:
                    stats["n"] += 1
                    if stats["n"] % 1000 == 0:
                        logger.info(
                            "sample_fast_path: 快路径命中 %d 次 / 回退 %d 次"
                            "（回退=该批分布 top-%d 质量<p，走位精确慢路，正确性无损）",
                            stats["n"], stats["fallback"], _K)
                    return out
                stats["fallback"] += 1
            except Exception as e:  # 快路径异常不致命
                logger.warning("sample_fast_path: 快路径异常回落: %s", e)
        return orig(logits, k, p, *args, **kwargs)

    _tts.apply_top_k_top_p = _fl_apply
    _tts._fl_sample_fast_patched = True
    _tts._fl_sample_fast_orig = orig
    logger.info(
        "sample_fast_path: top-p 过滤快路径启用（top-k%d+阈值法，bs=64 实测 "
        "9.66→目标 ~1.5ms/步；VLLM_FL_SAMPLE_FAST=0 可关闭）", _K)
    return True
