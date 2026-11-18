# Copyright (c) 2026 BAAI. All rights reserved.
"""Faster launch grid for vLLM's top-k/top-p sampling filter on Iluvatar.

机理（BI-V150 实测，2026-11-15 scripts/d3/sample_filter_bench.py）
----------------------------------------------------------------------------
serve 端默认采样参数来自 ckpt generation_config.json（top_p=0.95, T=1.0），
每步 decode 对 [bs, 130560] fp32 logits 跑 ``_topk_topp_kernel`` 全词表
top-p 过滤。上游 wrapper 的 grid 取 ``NUM_PROGRAMS = min(num_compute_units,
batch_size)`` = **16**（BI-V150 仅 16 SM）→ 每程序串行处理 4 行 × 130K 词表的
多趟扫描，单 CTA/SM 零延迟隐藏，bs=64 实测 3.86ms/步（trace 口径 4.4ms，
占 ITL ~18%）。

kernel 行映射为 ``tl.range(pid, BATCH_SIZE, num_programs)`` 按程序分行、行间
独立——**加大 grid 不改任何单行数学**（实测 16 vs 128 programs 输出
bitwise_equal=True）。扫参结果（fp32, p=0.95, TOPP-only）：

    grid(bs=64):  16→3.72ms  32→1.99ms  64→1.09ms  128→1.10ms（甜点 64）
    grid(bs=32):  16→1.90ms  32→1.05ms  64→1.02ms

本模块把 grid 提到 ``min(next_pow2(batch), 64)`` 并自管 per-program 工作缓冲
（上游 wrapper 把 buffer 钳到 ≤num_sm 行，是加大 grid 的唯一障碍）。kernel、
BLOCK（8192/4096）、数学全不动——**结果与 stock 逐位一致**（等价性单测覆盖）。
``VLLM_FL_SAMPLE_FILTER=0`` 回退上游；``VLLM_FL_SAMPLE_FILTER_PROGRAMS``
改 grid 上限（默认 64）。
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_installed = False
_BUFFER_CACHE: dict = {}   # (device, vocab) -> (num_progs, vocab) fp32


def _programs_for(batch: int, cap: int) -> int:
    p = 1
    while p < batch and p < cap:
        p *= 2
    return max(16, min(p, cap))


def _apply_topp_fast(logits, p, cap: int):
    """TOPP-only 快路径：stock kernel + 加大 grid + 自管 buffer。逐位等价。"""
    import vllm.v1.sample.ops.topk_topp_triton as _ttt

    if logits.stride(1) != 1:
        logits = logits.contiguous()
    batch, vocab = logits.shape
    n_prog = _programs_for(batch, cap)

    key = (logits.device, vocab)
    buf = _BUFFER_CACHE.get(key)
    if buf is None or buf.shape[0] < n_prog:
        buf = logits.new_empty((n_prog, vocab))
        _BUFFER_CACHE[key] = buf
    if buf.shape[0] > n_prog:
        buf = buf[:n_prog]

    tables = _ttt._TRITON_TABLE_CACHE.get(logits.device)
    if tables is None:
        tables = (logits.new_tensor(_ttt._NORMAL_CDF_TO_SIGMA_TABLE),
                  logits.new_tensor(_ttt._PERCENTILE_TO_STD_TABLE))
        _ttt._TRITON_TABLE_CACHE[logits.device] = tables
    n_cdf, p_std = tables

    _ttt._topk_topp_kernel[(n_prog,)](
        logits, logits.stride(0), buf, p_std, n_cdf,
        logits,  # k 未启用的哑指针（与 stock 同法）
        p.to(torch.float32),
        BATCH_SIZE=batch, MASK_VALUE=float("-inf"), VOCAB_SIZE=vocab,
        BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
        TOPK_ENABLED=False, TOPP_ENABLED=True,
    )
    return logits


def install() -> bool:
    """替换 apply_top_k_top_p 的 TOPP-only 分支（幂等；默认启用）。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_SAMPLE_FILTER", "1") != "1":
        logger.info("sample_filter_mode: VLLM_FL_SAMPLE_FILTER=0，保持上游")
        return False
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as _tts
    except Exception as e:
        logger.warning("sample_filter_mode: topk_topp_sampler 不可用: %s", e)
        return False
    if getattr(_tts, "_fl_sample_filter_patched", False):
        _installed = True
        return True

    cap = max(16, int(os.environ.get("VLLM_FL_SAMPLE_FILTER_PROGRAMS", "64")))
    orig = _tts.apply_top_k_top_p

    def _fl_apply_top_k_top_p(logits, k, p, *args, **kwargs):
        # 仅拦截 GPU fp32 TOPP-only（本栈 serve 常态）；其余形态原路透传。
        if (p is not None and k is None
                and logits.is_cuda and logits.ndim == 2
                and logits.dtype == torch.float32
                and logits.shape[0] >= 8):
            try:
                return _apply_topp_fast(logits, p, cap)
            except Exception as e:  # 快路径异常不致命：回落上游
                logger.warning("sample_filter_mode: 快路径失败回落: %s", e)
        return orig(logits, k, p)

    _tts.apply_top_k_top_p = _fl_apply_top_k_top_p
    _tts._fl_sample_filter_patched = True
    _tts._fl_sample_filter_orig = orig
    _installed = True
    logger.info(
        "sample_filter_mode: top-p 过滤 grid 已提至 ≤%d programs（逐位等价，"
        "bs=64 实测 3.7→1.1ms；VLLM_FL_SAMPLE_FILTER=0 可关闭）", cap)
    return True
