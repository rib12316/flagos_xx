# Copyright (c) 2026 BAAI. All rights reserved.
"""Data-driven launch-config table for vLLM's unified attention kernel on Iluvatar.

机理（BI-V150, 16 SM, triton 3.6.0 flagtree, 2026-11-03 纯净基线复证）
----------------------------------------------------------------------------
vllm 0.24 ``kernel_unified_attention``（上游 @triton.jit kernel）的默认 launch
（triton auto ≈ num_warps=4 / num_stages=3）在本卡的小 CTA 形状上被 launch 级
开销限制：decode 3D split（BLOCK_M=16/TILE=16）仅跑到 ~160-172 GB/s（D2D copy
自校准 587 GB/s、顺序读 ~352 GB/s）。注入 num_warps/num_stages 后：

    路径/形状（16Q/2KV × head_dim128, bf16, block=16, seg=16）      default  调优    加速
    decode 3D  bs=64  kv=4096   -> w1/s1                            1703.8µs 959.0µs  1.78x
    decode 3D  bs=64  kv=16384  -> w1/s1                            6240.4µs 3374.3µs 1.85x
    prefill 2D 8x1024q kv=4096  -> w2/s1                            69995.8µs 46754.7µs 1.50x
    prefill 2D 8x1024q kv=16384 -> w2/s1                            307562.5µs 204825.1µs 1.50x

四组对照输出 bit-exact（max_abs_diff=0.0）——launch 参数不改 kernel 数学。
端到端官方口径（废弃仓库 flagos2_infragogogo 同机制实测，D3a+D3c）：4K
1879.68→2664 (+41.7%)，16K 862.06→1263 (+46.6%)，精度 97.1%。

配置表与再生成
----------------------------------------------------------------------------
``LAUNCH_TABLE`` 按（路径 × bs 档 × kv 长度档）索引，非单值硬编码；
``scripts/sweep_attn_launch.py`` 扫描各档最优 launch 参数并可再生成本表。
``'*'`` 为通配兜底行（当前取值来自上表 + 废弃仓库 bs=30/kv~17k、bs=64/kv4600
及全量 E2E 验证的一致最优），重调检查点（P2′ 调度落地后 / P3′ 量化落地后 /
冻结前终扫）应重跑 sweep 换表，不要手改单值。

运行时集成方式（为何是 dispatch 层运行时装载而非 vllm 配置项）
----------------------------------------------------------------------------
vllm 0.24 的 ``unified_attention`` wrapper 内部虽已有 launch_num_warps/stages
变量，但仅服务 B200 large-head 特例，没有对外配置入口；kernel 又是上游
site-packages 内的 @triton.jit。因此本模块在插件内做两跳装载（默认启用，
``VLLM_FL_ATTN_LAUNCH=0`` 关闭恢复上游默认）：

  hop1  wrapper 级：以 CPU 侧入参 max_seqlen_k 为本次 launch 提供 kv 档提示
        （``seqused_k`` 是 device 张量，运行时取 max 会同步，不可用）
  hop2  kernel.run 级：从 launch 实参精确取路径（len(grid)）与 bs
        （seq_lens_ptr.shape[0]），合并 hop1 提示查表，setdefault 注入
        num_warps/num_stages

两跳均为插件源码内的一次性装载（幂等、try 保护、不替换 vllm 模块对象身份），
kernel 数学与 constexpr 完全不动。
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 档位（bucket）定义。边界改动 = 表结构变化，必须重跑 sweep 再生表。
# ---------------------------------------------------------------------------
BS_TIERS = (16, 32, 64, 128)   # <=16 / <=32 / <=64 / <=128 / >128
KV_TIERS = (8192, 32768)      # <=8k / <=32k / >32k


def bucket_bs(num_seqs: int) -> str:
    for bound in BS_TIERS:
        if num_seqs <= bound:
            return f"le{bound}"
    return "gt128"


def bucket_kv(kv_len: int) -> str:
    for bound in KV_TIERS:
        if kv_len <= bound:
            return f"le{bound // 1024}k"
    return "gt32k"


# (path, bs_tier, kv_tier) -> (num_warps, num_stages)；'*' 通配。
# 具体行 = 实测档位；通配行 = 该路径的一致最优兜底（见模块 docstring 出处）。
LAUNCH_TABLE: dict[Tuple[str, str, str], Tuple[int, int]] = {
    # ---- 具体行：sweep_attn_launch.py --out results/20261103_d0_launch_sweep
    #      于 20261103_232448 再生（quick 矩阵，bit-exact 校验通过）----
    ("decode", "le16", "le8k"): (1, 1),    # 测 bs=16 kv=4096 (1.647x)
    ("decode", "le32", "le8k"): (2, 1),    # 测 bs=32 kv=4096 (1.694x, w2s1 反超 w1s1 16%)
    ("decode", "le32", "le32k"): (1, 1),   # 测 bs=32 kv=16384 (1.690x)
    ("decode", "le64", "le8k"): (1, 1),    # 测 bs=64 kv=4096 (1.776x)
    ("decode", "le64", "le32k"): (1, 1),   # 测 bs=64 kv=16384 (1.846x)
    ("prefill", "le16", "le8k"): (2, 1),   # 测 8seq×1024q kv=4096 (1.507x)
    ("prefill", "le16", "le32k"): (2, 1),  # 测 8seq×1024q kv=16384 (1.505x)
    # ---- 通配兜底：该路径一致最优（T0 复证 + 废弃仓库 E2E 全量验证），
    #      未扫到的形状继承；重调检查点后由 sweep 细化 ----
    ("decode", "*", "*"): (1, 1),
    ("prefill", "*", "*"): (2, 1),
}


def resolve(
    path: str, bs: int, kv: Optional[int]
) -> Optional[Tuple[int, int]]:
    """按（路径 × bs 档 × kv 档）查表；kv 为 None 时仅匹配通配行。

    优先级：精确档 > (path, bs档, *) > (path, *, kv档) > (path, *, *)。
    未命中返回 None = 不注入，保持上游默认 launch。
    """
    bs_t = bucket_bs(bs) if bs is not None else "*"
    kv_t = bucket_kv(kv) if kv is not None else None
    table = _active_table()
    if kv_t is not None and (path, bs_t, kv_t) in table:
        return table[(path, bs_t, kv_t)]
    if (path, bs_t, "*") in table:
        return table[(path, bs_t, "*")]
    if kv_t is not None and (path, "*", kv_t) in table:
        return table[(path, "*", kv_t)]
    return table.get((path, "*", "*"))


# ---------------------------------------------------------------------------
# 环境变量覆盖表（sweep A/B 与重调检查点用；JSON: [[path,bs,kv,w,s], ...]）
# ---------------------------------------------------------------------------
_OVERRIDE_TABLE: Optional[dict] = None
_OVERRIDE_LOADED = False


def _active_table() -> dict:
    global _OVERRIDE_TABLE, _OVERRIDE_LOADED
    if not _OVERRIDE_LOADED:
        _OVERRIDE_LOADED = True
        path = os.environ.get("VLLM_FL_ATTN_LAUNCH_TABLE", "")
        if path:
            try:
                with open(path) as f:
                    rows = json.load(f)
                _OVERRIDE_TABLE = {
                    (str(p), str(b), str(k)): (int(w), int(s))
                    for p, b, k, w, s in rows
                }
                logger.info(
                    "attn_launch_config: 覆盖表已从 %s 装载（%d 行）", path, len(rows)
                )
            except Exception as e:  # 覆盖表损坏不致命：回落内置表
                logger.warning("attn_launch_config: 覆盖表 %s 加载失败(%s)，用内置表", path, e)
                _OVERRIDE_TABLE = None
    return _OVERRIDE_TABLE if _OVERRIDE_TABLE is not None else LAUNCH_TABLE


# ---------------------------------------------------------------------------
# 两跳装载
# ---------------------------------------------------------------------------
# hop1 -> hop2 的单次 kv 提示（unified_attention 调用链同步执行，随取随清）
_KV_HINT: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "fl_attn_launch_kv_hint", default=None
)

_installed = False
# 首次命中的 (path, bs_tier, kv_tier) -> cfg 记录用（serve 日志可 grep 验证生效）
_hit_log: set = set()


def _note_hit(key, cfg, kv):
    if key in _hit_log:
        return
    _hit_log.add(key)
    logger.info(
        "fl.attn-launch: %s (bs档=%s kv=%s) -> num_warps=%d num_stages=%d",
        key[0], key[1], kv if kv is not None else "?", cfg[0], cfg[1],
    )


def install() -> bool:
    """装载 launch 配置（幂等；默认启用）。返回是否装载。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_ATTN_LAUNCH", "1") != "1":
        logger.info("attn_launch_config: VLLM_FL_ATTN_LAUNCH=0，保持上游默认 launch")
        return False
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _tua
    except Exception as e:
        logger.warning("attn_launch_config: triton_unified_attention 不可用: %s", e)
        return False

    kernel = _tua.kernel_unified_attention
    if getattr(kernel, "_fl_launch_config_hook", False):
        _installed = True
        return True  # 已由别的进程路径装载（幂等）

    # ---- hop2: kernel.run 级注入（路径/bs 取自 launch 实参，精确且零同步）----
    orig_run = kernel.run

    def _tuned_run(*args, grid, warmup=False, **kwargs):
        cfg = None
        if not warmup and "seq_lens_ptr" in kwargs and len(grid) in (2, 3):
            path = "decode" if len(grid) == 3 else "prefill"
            try:
                bs = int(kwargs["seq_lens_ptr"].shape[0])
            except Exception:
                bs = None
            kv = _KV_HINT.get()
            cfg = resolve(path, bs, kv)
            if cfg is not None:
                key = (
                    path,
                    "*" if bs is None else bucket_bs(bs),
                    "*" if kv is None else bucket_kv(kv),
                )
                _note_hit(key, cfg, kv)
                kwargs.setdefault("num_warps", cfg[0])
                kwargs.setdefault("num_stages", cfg[1])
        return orig_run(*args, grid=grid, warmup=warmup, **kwargs)

    kernel.run = _tuned_run
    kernel._fl_launch_config_hook = True
    kernel._fl_launch_orig_run = orig_run  # 供 sweep/bench 还原默认臂

    # ---- hop1: wrapper 级 kv 提示（max_seqlen_k 是 CPU 侧 int，免费）----
    orig_wrapper = _tua.unified_attention

    def _hinted_unified_attention(*args, **kwargs):
        mk = kwargs.get("max_seqlen_k")
        tok = None
        if isinstance(mk, int) and mk > 0:
            tok = _KV_HINT.set(mk)
        try:
            return orig_wrapper(*args, **kwargs)
        finally:
            if tok is not None:
                _KV_HINT.reset(tok)

    _hinted_unified_attention._fl_orig_wrapper = orig_wrapper
    _tua.unified_attention = _hinted_unified_attention
    # vllm 的 triton_attn 后端按名导入了该符号，两处绑定都要接上
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
        if _ta.unified_attention is orig_wrapper:
            _ta.unified_attention = _hinted_unified_attention
    except Exception as e:
        logger.warning("attn_launch_config: triton_attn 符号绑定未接上: %s", e)

    _installed = True
    n = len(_active_table())
    logger.info(
        "attn_launch_config: 已装载 unified_attention launch 配置表（%d 行，"
        "路径×bs档×kv档 索引；VLLM_FL_ATTN_LAUNCH=0 可关闭）",
        n,
    )
    return True
