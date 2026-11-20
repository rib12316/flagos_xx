# Copyright (c) 2026 BAAI. All rights reserved.
"""Prefill BLOCK_M 耦合重标定 for vLLM's unified attention on Iluvatar (BI-V150).

机理（2026-11-20 results/20261120_blockm_sweep/sweep.csv，graph 口径）
----------------------------------------------------------------------------
上游 wrapper 对 GQA（num_queries_per_kv<=16）固定 ``BLOCK_M=16`` →
``BLOCK_Q=2``：每个 program 只算 2 个 query 位置，prefill 算力利用率仅
1.4-2.8% 峰值 MAC（16Q/2KV、hd128、bf16 实测）。本模块做耦合重标定：

    BLOCK_M=256 / num_warps=16 / num_stages=2 / TILE_SIZE_PREFILL=128

    形态                          upstream    本补丁      加速    vs 旧生产
    S1 q4096 kv16384              139239µs    14087µs    9.88x    6.03x
    S2 q2048 kv16384（×2seq）     149963µs    15335µs    9.78x    5.94x
    S1 q8192 kv16384（整chunk）   ~237ms      24082µs    9.86x     —
    S4 q4096 kv4096（×4seq）      80283µs      8519µs    9.42x    5.68x

36 配网格（BM{64,128,256}×T{32,64,128}×W{8,16}×S{1,2}）无更优；消融：W16
是命门（换 W4 剩 1.95×），T128 在 BM256 下再 +1.31×（BM16 时反向劣化——
与 attn_tile_mode 的 TILE16 结论是不同 BLOCK_M 前提下的耦合关系，非矛盾）。
数值：T128 改 KV 归约分块，非 bit-exact，max|diff| ≤ 2 bf16 ULP（量级相关，
与 ⑤TILE16/①TD 同漂移类，B 臂 105 题 0.9714 已背书该类无害）。decode 侧
不动（attn_launch_config 表 w1s1 + TILE16 已最优，同 CSV dect 轴结案）。

与其它 hook 的关系
----------------------------------------------------------------------------
- **attn_td_mode（①TD）**：TD 的 wrapper 钳位 ``min(TILE, block_size=16)`` 会把
  prefill TILE=128 静默钳回 16 → 本模块在 prefill 分支内**强制 use_td=False**
  （仅 prefill）；decode 侧 TD 不受影响——实测 decode TD 124ms vs 非 TD 269ms
  （bs16/kv17K，20261120_bm_smoke），**建议 VLLM_FL_ATTN_TD=1 保留 decode TD**。
- attn_tile_mode（⑤TILE16）：本模块注入点在 wrapper 内 TILE 赋值链之后，
  天然优先；建议显式 ``VLLM_FL_ATTN_TILE=0`` 避免双写困惑。
- attn_launch_config（D0 表）：本模块显式设 prefill warps/stages，
  kernel.run ``setdefault`` 对已显式键不生效；decode 仍走 D0 表。
- 双绑定：``triton_attn`` 后端 ``from ... import unified_attention`` 持有
  自己的引用，必须与 ops 模块一并替换（与上游 PR#671 同处理）。

实现：取 ``unified_attention`` 源码，三处锚点定点替换后原模块命名空间重
exec；锚点任一不唯一即整体放弃（无半补丁状态，仅失去加速不影响正确性）；
幂等。``VLLM_FL_ATTN_BM=1`` 启用（**默认关**，待官方 4 轮 + 105 题门后转
默认开）；``VLLM_FL_ATTN_BM_VAL`` / ``VLLM_FL_ATTN_BM_W`` / ``VLLM_FL_ATTN_BM_S``
/ ``VLLM_FL_ATTN_BM_TILE`` 可调（供后续分桶表化）。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False

_ANCHOR_BLOCK = """    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv"""

_ANCHOR_WARP = """    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = None
    launch_num_stages: int | None = None"""

_ANCHOR_TILE = """    if tuned_large_head:
        TILE_SIZE_PREFILL = 128"""


def install() -> bool:
    """prefill BLOCK_M=256/W16/S2/T128（幂等；默认关；VLLM_FL_ATTN_BM=1 启用）。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_ATTN_BM", "0") != "1":
        return False
    bm = int(os.environ.get("VLLM_FL_ATTN_BM_VAL", "256"))
    wp = int(os.environ.get("VLLM_FL_ATTN_BM_W", "16"))
    st = int(os.environ.get("VLLM_FL_ATTN_BM_S", "2"))
    tile = int(os.environ.get("VLLM_FL_ATTN_BM_TILE", "128"))
    if os.environ.get("VLLM_FL_ATTN_TD", "1") != "1":
        logger.warning(
            "attn_bm_mode: VLLM_FL_ATTN_TD 未开——decode 侧 TD 快路径将不生效"
            "（实测 bs16/kv17K ITL 124ms(TD) vs 269ms(非TD)），建议 TD=1")
    try:
        import inspect

        import vllm.v1.attention.ops.triton_unified_attention as _tua
    except Exception as e:  # pragma: no cover
        logger.warning("attn_bm_mode: triton_unified_attention 不可用: %s", e)
        return False
    if getattr(_tua, "_fl_bm_patched", False):
        _installed = True
        return True

    orig = _tua.unified_attention
    src = inspect.getsource(orig)
    inject_block = _ANCHOR_BLOCK + f"""
    if _fl_bm and max_seqlen_q > 1 and head_size <= 128 \\
            and {bm} % num_queries_per_kv == 0:
        BLOCK_M = {bm}
        BLOCK_Q = BLOCK_M // num_queries_per_kv
        use_td = False  # prefill 侧强制非 TD：否则 TD 钳位 TILE<=block_size
        # 会把 TILE={tile} 静默钳回 16；decode（q<=1）不受本行影响，TD 照常"""
    inject_warp = _ANCHOR_WARP + f"""
    if _fl_bm and max_seqlen_q > 1 and head_size <= 128 \\
            and {bm} % num_queries_per_kv == 0:
        launch_num_warps = {wp}
        launch_num_stages = {st}"""
    inject_tile = _ANCHOR_TILE + f"""
    if _fl_bm and max_seqlen_q > 1 and head_size <= 128:
        TILE_SIZE_PREFILL = {tile}"""
    patches = ((_ANCHOR_BLOCK, inject_block), (_ANCHOR_WARP, inject_warp),
               (_ANCHOR_TILE, inject_tile))
    for old, _ in patches:
        n = src.count(old)
        if n != 1:
            logger.warning("attn_bm_mode: 锚点 x%d（需 1），整体放弃回退原生: %r",
                           n, old[:60])
            return False
    for old, new in patches:
        src = src.replace(old, new)
    _tua._fl_bm = True
    exec(compile(src, "<fl_attn_bm>", "exec"), _tua.__dict__)
    patched = _tua.__dict__["unified_attention"]

    # 双绑定替换：triton_attn 后端持有自己的 from-import 引用
    rebound = ["ops.triton_unified_attention"]
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
        if getattr(_ta, "unified_attention", None) is orig:
            _ta.unified_attention = patched
            rebound.append("backends.triton_attn")
    except Exception:
        pass
    _tua._fl_bm_patched = True
    _tua._fl_bm_orig = orig
    _installed = True
    logger.info(
        "attn_bm_mode: prefill 重标定 BLOCK_M=%d/W%d/S%d/TILE=%d 生效（探针 "
        "9.4-9.9x vs upstream，5.7-6.0x vs 旧生产；TILE 非 bit-exact；替换绑定 "
        "%s）", bm, wp, st, tile, "+".join(rebound))
    return True
