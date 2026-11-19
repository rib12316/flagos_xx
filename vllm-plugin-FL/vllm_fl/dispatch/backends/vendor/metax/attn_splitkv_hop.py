# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""⑥ split-KV decode kernel 的 MetaX hop（无卡交付 attn_splitkv.py 的沐曦接线）。

形态（与 iluvatar 侧 attn_splitkv.py 同一 kernel 本体，零改动复用）：
----------------------------------------------------------------------------
沐曦 attention backend = MACA flash 闭源库：decode 统一走
``MacaFlashAttentionBackend.forward`` → ``flash_attn_with_kvcache``（flash_attn.py
L830），``kv_cache.unbind(0)`` 后恰为 kernel 假设的 ``[nb, 16, NKV, HD]`` 布局。
本模块把该符号双绑定替换为路由 wrapper：命中 decode 形态 → 自研 split-KV
kernel（V0-V3，见 iluvatar/attn_splitkv.py 交接书）；否则透传 MACA flash。

TD stub：沐曦栈 triton 3.0 < 3.3，DependencyFinder 连死代码分支一起解析
``tl.make_tensor_descriptor`` → 必须打 stub 且 USE_TD 恒 False（本栈无 TD 路径）。

env 门控（全部默认关）：``VLLM_FL_ATTN_SPLITKV=1`` 启用；变体/调试/强制段数
沿用 ``VLLM_FL_ATTN_SPLITKV_VARIANT/_DEBUG/_SPLITS/_GROUP``（见 kernel 模块头）。
``VLLM_FL_ATTN_SPLITKV_TD`` 在本栈仅接受 0/-1（无 TD 载体，非 0 即拒装）。
"""
from __future__ import annotations

import logging
import os

import torch

logger = logging.getLogger(__name__)

_installed = False
_route_stats = {"routed": 0, "passed": 0}  # probe 口径与 iluvatar hop 对齐


def _ensure_td_stub() -> None:
    """triton < 3.3 的 make_tensor_descriptor stub（iluvatar.py 同款，勿跨
    vendor import——那会连带装载 iluvatar installer 链）。"""
    try:
        import triton.language as tl

        if hasattr(tl, "make_tensor_descriptor"):
            return

        class _TensorDescriptorStub:
            def __call__(self, *args, **kwargs):
                raise RuntimeError(
                    "tl.make_tensor_descriptor is not available on triton < 3.3; "
                    "USE_TD must stay False on the MetaX stack."
                )

            def __hash__(self):
                return hash("_tl_make_tensor_descriptor_stub")

            def __repr__(self):
                return "<tl.make_tensor_descriptor stub for triton<3.3"

        tl.make_tensor_descriptor = _TensorDescriptorStub()
        logger.info(
            "fl.attn-splitkv(metax): 已为 triton<3.3 打 make_tensor_descriptor "
            "stub（USE_TD 恒 False）")
    except Exception as e:
        logger.warning("fl.attn-splitkv(metax): TD stub 失败: %s", e)


def _route_decision_flash(kw) -> str | None:
    """flash_with_kvcache 口径的路由判定。返回 None=路由自研 kernel。只看
    CPU 侧静态量（cache_seqlens 是 device 张量，绝不取 max——同 hop1 纪律）。"""
    q = kw.get("q")
    k_cache = kw.get("k_cache")
    v_cache = kw.get("v_cache")
    cache_seqlens = kw.get("cache_seqlens")
    if not (isinstance(q, torch.Tensor) and isinstance(k_cache, torch.Tensor)
            and isinstance(v_cache, torch.Tensor)
            and isinstance(cache_seqlens, torch.Tensor)):
        return "qkv/lens 非张量"
    if q.ndim != 3:
        return f"q.ndim={q.ndim}"
    num_tokens, num_heads, head_size = q.shape
    num_seqs = cache_seqlens.shape[0]
    if num_tokens != num_seqs:
        return f"q_len={num_tokens // max(num_seqs, 1)}>1"
    if num_seqs > 64:
        return f"bs={num_seqs}>64"
    if kw.get("causal") is not True:
        return f"causal={kw.get('causal')}"
    ws = kw.get("window_size")
    if not (ws is None or (isinstance(ws, tuple) and len(ws) == 2
                           and ws[0] < 0 and ws[1] < 0)):
        return f"window={ws}"
    for key in ("alibi_slopes", "softcap", "s_aux"):
        val = kw.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val != 0):
            return key
    if q.element_size() < 2:
        return "fp8 q"
    if head_size & (head_size - 1) or head_size > 256:
        return f"head_size={head_size}"
    num_kv_heads = k_cache.shape[2]
    if num_heads % num_kv_heads or num_heads // num_kv_heads > 16:
        return f"gqa {num_heads}:{num_kv_heads}"
    block_size = k_cache.shape[1]
    if block_size != 16:
        return f"block_size={block_size}"
    if k_cache.shape != v_cache.shape:
        return "k/v 形状不一致"
    return None


_CU_CACHE: dict = {}     # device_index -> int32 arange 缓冲（内容静态，只增长）
_OUT_CACHE: dict = {}    # device_index -> bf16 输出缓冲（只增长不搬迁，graph 安全）


def _get_cu(num_seqs: int, device) -> torch.Tensor:
    key = device.index if device.index is not None else 0
    buf = _CU_CACHE.get(key)
    if buf is None or buf.shape[0] < num_seqs + 1:
        cap = max(num_seqs + 1, 65)
        _CU_CACHE[key] = torch.arange(cap, dtype=torch.int32, device=device)
        buf = _CU_CACHE[key]
        logger.info("fl.attn-splitkv(metax): cu 缓冲 → %d", cap)
    return buf[: num_seqs + 1]


def _get_out(num_tokens: int, num_heads: int, head_size: int, device):
    key = device.index if device.index is not None else 0
    buf = _OUT_CACHE.get(key)
    if (buf is None or buf.shape[0] < num_tokens or buf.shape[1] != num_heads
            or buf.shape[2] != head_size):
        cap = max(num_tokens, 64)
        _OUT_CACHE[key] = torch.empty(
            (cap, num_heads, head_size), dtype=torch.bfloat16, device=device)
        buf = _OUT_CACHE[key]
        logger.info(
            "fl.attn-splitkv(metax): out 缓冲 → %dx%dx%d（graph capture 前完成"
            "即可）", cap, num_heads, head_size)
    return buf[:num_tokens]


def install() -> bool:
    """双绑定 flash_attn_with_kvcache（幂等；默认**关闭**）。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_ATTN_SPLITKV", "0") != "1":
        logger.info("fl.attn-splitkv(metax): VLLM_FL_ATTN_SPLITKV=0，保持 MACA flash")
        return False
    td_mode = int(os.environ.get("VLLM_FL_ATTN_SPLITKV_TD", "-1") or -1)
    if td_mode == 1:
        logger.warning("fl.attn-splitkv(metax): 本栈无 TD 载体，TD=1 拒装")
        return False
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar import (
            attn_splitkv as _ask,
        )
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention import (
            flash_attn as _fa,
        )
        from vllm_fl.dispatch.backends.vendor.metax.impl.attention.utils import (
            fa_utils as _fau,
        )
    except Exception as e:
        logger.warning("fl.attn-splitkv(metax): 依赖不可用: %s", e)
        return False

    orig_flash = getattr(_fa, "flash_attn_with_kvcache", None)
    orig_flash_src = getattr(_fau, "flash_attn_with_kvcache", None)
    if orig_flash is None:
        logger.warning("fl.attn-splitkv(metax): flash_attn_with_kvcache 符号缺失")
        return False

    _ensure_td_stub()
    variant = max(0, min(_env_int("VLLM_FL_ATTN_SPLITKV_VARIANT", 1), 3))
    debug = _env_int("VLLM_FL_ATTN_SPLITKV_DEBUG", 0) == 1
    _fallback_logged = False

    def _splitkv_flash_with_kvcache(q=None, k_cache=None, v_cache=None, *args,
                                    **kwargs):
        nonlocal _fallback_logged
        # probe A/B：_active/_variant_override 是 kernel 模块的模块级全局，
        # 经模块引用读取才跟随 set_active/set_variant 的重赋值
        active = _ask._active
        active = True if active is None else active
        var = variant if _ask._variant_override is None else _ask._variant_override
        kw = dict(kwargs)
        kw.update({"q": q, "k_cache": k_cache, "v_cache": v_cache})
        reason = _route_decision_flash(kw) if active else "disabled"
        if reason is None:
            try:
                block_table = kw["block_table"]
                # CPU 侧上界（block_table 列数 × block_size），避免 device 同步
                _ask._CURRENT_MAX_KV[0] = block_table.shape[1] * k_cache.shape[1]
                out = _get_out(q.shape[0], q.shape[1], q.shape[2], q.device)
                cu = _get_cu(kw["cache_seqlens"].shape[0], q.device)
                _ask.splitkv_decode_attention(
                    q, k_cache, v_cache, out, cu, kw["cache_seqlens"],
                    block_table, kw.get("softmax_scale"),
                    variant=var, use_td=False)
                if debug:
                    logger.info(
                        "fl.attn-splitkv(metax): routed bs=%d kv<=%d split=%d "
                        "variant=%s", kw["cache_seqlens"].shape[0],
                        _ask._CURRENT_MAX_KV[0],
                        _ask.resolve_split(kw["cache_seqlens"].shape[0],
                                           _ask._CURRENT_MAX_KV[0]),
                        _ask.VARIANT_NAMES[var])
                _route_stats["routed"] += 1
                return out
            except Exception as e:
                if not _fallback_logged:
                    _fallback_logged = True
                    logger.warning(
                        "fl.attn-splitkv(metax): 自研 kernel 异常，本进程永久"
                        "回落 MACA flash（原因: %s）", e, exc_info=debug)
        elif debug:
            logger.info("fl.attn-splitkv(metax): passthrough (%s)", reason)
        _route_stats["passed"] += 1
        full = dict(kwargs)
        full.update({"q": q, "k_cache": k_cache, "v_cache": v_cache})
        return orig_flash(*args, **full)

    _splitkv_flash_with_kvcache._fl_splitkv_hook = True
    _splitkv_flash_with_kvcache._fl_splitkv_orig = orig_flash
    _fa.flash_attn_with_kvcache = _splitkv_flash_with_kvcache
    # 定义源同步绑定（防其他路径从 fa_utils 按名取符号）
    if orig_flash_src is not None and orig_flash_src is orig_flash:
        _fau.flash_attn_with_kvcache = _splitkv_flash_with_kvcache

    _installed = True
    logger.info(
        "fl.attn-splitkv(metax): 已装载（variant=%s group=%d TD=False；"
        "VLLM_FL_ATTN_SPLITKV=0 可关闭）",
        _ask.VARIANT_NAMES[variant],
        _ask._env_int("VLLM_FL_ATTN_SPLITKV_GROUP", 2))
    return True


# probe/serve A/B 直用 kernel 模块的 set_active/set_variant（同一全局对象）
__all__ = ["install"]
