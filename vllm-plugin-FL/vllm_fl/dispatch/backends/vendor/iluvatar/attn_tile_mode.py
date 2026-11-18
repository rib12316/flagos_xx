# Copyright (c) 2026 BAAI. All rights reserved.
"""Prefill TILE_SIZE=16 for vLLM's unified attention on Iluvatar (BI-V150).

机理（2026-11-16 results/d3_dev/item5_tile/tile_sweep.csv，graph 口径）
----------------------------------------------------------------------------
上游 ``_get_tile_size`` 对 prefill 恒返回 32（is_prefill=True 分支）。16 SM 的
BI-V150 上扫 TILE×warps×stages（3 形态 × 11 臂）：**TILE=16（配既有表 w2s1）
一致最优**，TILE=32（默认）慢 ~10%，TILE=64/128 大幅劣化（0.55-0.64×）——
方向与"大 TILE 抬吞吐"的直觉相反：小卡上更小的 kv 块 → 更多 CTA 波次 →
更低的单 CTA 串行延迟（prefill %MAC 仅 2-5%，延迟主导）。

    形态                 t32(默认)   t16 w2s1    加速
    S1 q4096 kv4k        13641.7µs   12330.8µs   1.107x
    S2 q2048 kv16k       99850.5µs   90767.9µs   1.100x
    S4 q4096 kv4k        53729.4µs   48415.1µs   1.110x

注意：TILE 是 constexpr，改 tile = 改 kv 循环内累加分组 → **非 bit-exact**
（与段数同类，softmax 数学等价、舍入序变化）。TD 若开启，wrapper 的
``min(tile, block_size)`` 钳制与 16 一致，无冲突。
``VLLM_FL_ATTN_TILE=0``（默认）保持上游 32；=1 启用 prefill TILE=16。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False


def install() -> bool:
    """prefill TILE_SIZE=16（幂等；默认关闭——非 bit-exact，过对拍门后再定）。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_ATTN_TILE", "0") != "1":
        logger.info("attn_tile_mode: VLLM_FL_ATTN_TILE 未设为 1，保持上游 TILE=32")
        return False
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _tua
    except Exception as e:
        logger.warning("attn_tile_mode: triton_unified_attention 不可用: %s", e)
        return False
    if getattr(_tua, "_fl_tile_patched", False):
        _installed = True
        return True

    orig_tile = _tua._get_tile_size

    def _tile16(head_size, sliding_window, element_size, is_prefill):
        if is_prefill:
            return 16
        return orig_tile(head_size, sliding_window, element_size, is_prefill)

    _tua._get_tile_size = _tile16
    _tua._fl_tile_patched = True
    _tua._fl_tile_orig = orig_tile
    _installed = True
    logger.info(
        "attn_tile_mode: prefill TILE_SIZE 32→16（探针 1.10-1.11×，非 bit-exact；"
        "VLLM_FL_ATTN_TILE=0 可关闭）")
    return True
