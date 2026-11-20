# Copyright (c) 2026 BAAI. All rights reserved.
"""F-a：rope(Q)+rope(K) 融合 compile pass（插件首个自定义 inductor pass）。

配方逐行照抄上游 vllm/compilation/passes/fusion/rope_kvcache_fusion.py 与
mla_rope_kvcache_cat_fusion.py（direct_register_custom_op + VllmPatternReplacement
+ VllmFusionPatternMatcherPass）。

归因结论（为什么匹配 native 分解）
----------------------------------------------------------------------------
compile 模式下 vllm CustomOp 默认解绑（forward_native），serve 图中 rope 即
RotaryEmbedding.forward_static 的 aten 分解（index_select → chunk → mul/sub/add
→ cat → cat）。本 pass 的 pattern 侧用 MatcherRotaryEmbedding(enabled=False)
追踪**同一段 python**（forward_native 路径），fx 结构天然与真实图一致——上游
配方成立的同一理由。图内无独立 cat 算子待融合（cat 是 rope 分解的内件，已被
inductor 并入 cat_3/cat_4 两核）；本 pass 做的是把两核换成一个 opaque 融合 op。

替换语义
----------------------------------------------------------------------------
  pattern:     q,k,v = qkv.split([qs,ks,vs],-1); q,k = rope(positions,q,k,cs)
               → q.view(T,NQ,D), k.view(T,NKV,D), v.view(T,NKV,D)
               （v 必须进输出：matcher 要求节点 users 数与真实图一致，缺第三
               getitem 整模式失配——CPU 干跑实证 MATCH_COUNT 0→1 的根因）
  replacement: vllm.fl_rope_qk_cat(qkv, positions, cs, NQ, NKV, D) → (Q',K')
               （纯函数，输出即 [T,NQ,D]/[T,NKV,D]，布局与 view 后原张量
               逐位同构）+ v 原样 view
其它 qkv 消费者不在匹配域内、原样保留。op 无 mutation → 不需要
auto_functionalized（上游用它只因 fused op 原地写）。

接线（platform.py 注入点，装载期静态）
----------------------------------------------------------------------------
compilation_config.inductor_compile_config["post_grad_custom_post_pass"] =
<本 pass 实例>（backends.py configure_post_pass 会把它 add 进 PostGradPassManager
并在 pass 尾部执行——与 vllm 官方 ActivationFusionPass 同机制、免改 vllm 源码）。
env 门：VLLM_FL_FA_ROPECAT=0 完全不注入。无运行时分支 → cudagraph 捕获安全。

风险登记
----------------------------------------------------------------------------
- dynamo/pattern 不匹配（真实图与 pattern 追踪出现结构差，如 inductor 预处理
  改写 view/reshape 形态）：pass 静默 0 命中（log debug "Replaced 0"），退化为
  现状两核，无正确性风险；上机用 VLLM_PATTERN_MATCH_DEBUG=1 取证。
- pass 在 config 期构造（模型未建），头数从 hf_config 读；TP>1 时头数需整除，
  当前单卡 C500 无 TP。
"""

from __future__ import annotations

from typing import Any

import torch

from vllm.compilation.passes.vllm_inductor_pass import (
    VllmFusionPatternMatcherPass,
    VllmPatternReplacement,
)
from vllm.compilation.passes.fusion.matcher_utils import MatcherRotaryEmbedding
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _fl_rope_qk_cat_impl(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.dispatch.backends.flaggems.impl.rope_cat import rope_qk_cat_alloc

    return rope_qk_cat_alloc(
        qkv,
        positions,
        cos_sin_cache,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def _fl_rope_qk_cat_fake(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    return (
        torch.empty(
            (num_tokens, num_q_heads, head_size), dtype=qkv.dtype, device=qkv.device
        ),
        torch.empty(
            (num_tokens, num_kv_heads, head_size), dtype=qkv.dtype, device=qkv.device
        ),
    )


direct_register_custom_op(
    op_name="fl_rope_qk_cat",
    op_func=_fl_rope_qk_cat_impl,
    mutates_args=[],  # 纯函数：输出为全新张量，无原地写
    fake_impl=_fl_rope_qk_cat_fake,
)

FUSED_OP = torch.ops.vllm.fl_rope_qk_cat.default


class FLRopeQKCatPattern(VllmPatternReplacement):
    """match {split → native rope(q,k) → view} → 替换为 vllm.fl_rope_qk_cat。"""

    def __init__(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_size: int,
        is_neox: bool,
    ) -> None:
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.q_size = num_q_heads * head_size
        self.k_size = num_kv_heads * head_size
        self.v_size = num_kv_heads * head_size
        self.is_neox = is_neox
        # enabled=False → 追踪 forward_native（aten 分解），与 compile 图同构；
        # 不跟随 RotaryEmbedding.enabled()（本机 _C.rotary_embedding 存在但
        # compile 模式仍走 native，默认态会错配 custom-op 形态的 pattern）。
        self.rope_matcher = MatcherRotaryEmbedding(
            is_neox=is_neox,
            head_size=head_size,
            num_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            enabled=False,
        )

    def get_inputs(self) -> list:
        T = 5
        L = 4096
        qkv = torch.empty(T, self.q_size + self.k_size + self.v_size,
                          dtype=torch.bfloat16, device="cuda")
        positions = torch.empty(T, dtype=torch.int64, device="cuda")
        cos_sin_cache = torch.empty(L, self.head_size,
                                    dtype=torch.bfloat16, device="cuda")
        return [qkv, positions, cos_sin_cache]

    @property
    def pattern(self):
        rope_matcher = self.rope_matcher
        q_size, k_size, v_size = self.q_size, self.k_size, self.v_size
        num_q_heads, num_kv_heads = self.num_q_heads, self.num_kv_heads
        head_size = self.head_size

        def pattern(qkv, positions, cos_sin_cache):
            # v 必须绑定并作为输出（pattern matcher 要求匹配节点的 users 数
            # 一致：真实图 split 的第三个 getitem 喂 v-view，缺它整模式失配）
            q, k, v = qkv.split([q_size, k_size, v_size], dim=-1)
            q, k = rope_matcher(positions, q, k, cos_sin_cache)
            return (
                q.view(-1, num_q_heads, head_size),
                k.view(-1, num_kv_heads, head_size),
                v.view(-1, num_kv_heads, head_size),
            )

        return pattern

    @property
    def replacement(self):
        num_q_heads, num_kv_heads = self.num_q_heads, self.num_kv_heads
        head_size = self.head_size
        q_size, k_size, v_size = self.q_size, self.k_size, self.v_size

        def replacement(qkv, positions, cos_sin_cache):
            _, _, v = qkv.split([q_size, k_size, v_size], dim=-1)
            q, k = FUSED_OP(
                qkv,
                positions,
                cos_sin_cache,
                num_q_heads,
                num_kv_heads,
                head_size,
            )
            return q, k, v.view(-1, num_kv_heads, head_size)

        return replacement


class FLRopeCatFusionPass(VllmFusionPatternMatcherPass):
    """F-a pass 实体。构造期注册 pattern（fake mode 追踪，不触设备）。"""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config, "fl_rope_cat_fusion_pass")
        hf_config: Any = (
            config.model_config.hf_config if config.model_config else None
        )
        num_q_heads = getattr(hf_config, "num_attention_heads", None)
        num_kv_heads = getattr(hf_config, "num_key_value_heads", None) or num_q_heads
        head_size = getattr(hf_config, "head_size", None)
        if head_size is None and num_q_heads:
            hidden = getattr(hf_config, "hidden_size", 0)
            if hidden and hidden % num_q_heads == 0:
                head_size = hidden // num_q_heads
        if not (num_q_heads and num_kv_heads and head_size):
            logger.warning(
                "F-a rope_cat: 无法从 hf_config 解出头形（%s/%s/%s），pass 空转",
                num_q_heads, num_kv_heads, head_size,
            )
            return
        # is_neox 两态都注册（Llama 系 True；成本低，防 config 变体）
        for is_neox in (True, False):
            self.register(
                FLRopeQKCatPattern(num_q_heads, num_kv_heads, head_size, is_neox)
            )
        self.dump_patterns(config, self.pm_pass)


def make_rope_cat_pass(config: VllmConfig) -> FLRopeCatFusionPass:
    """platform.py 注入用工厂（set_current_vllm_config 上下文由调用方管）。"""
    return FLRopeCatFusionPass(config)
