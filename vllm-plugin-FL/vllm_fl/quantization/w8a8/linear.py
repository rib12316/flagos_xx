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
"""Dispatched scaled-mm adapter for dynamic-token/per-channel W8A8."""

from __future__ import annotations

import torch

from vllm.model_executor.kernels.linear import (
    Int8ScaledMMLinearKernel,
    Int8ScaledMMLinearLayerConfig,
)
from vllm.model_executor.layers.quantization.utils import replace_parameter

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import (
    is_oot_enabled,
    use_flaggems_op,
)

FLAGGEMS_W8A8_LINEAR_OP = "w8a8_dynamic_per_token_linear"

_dynamic_per_token_quant_int8 = CachedOp("dynamic_per_token_quant_int8")
_w8a8_scaled_mm = CachedOp(FLAGGEMS_W8A8_LINEAR_OP)

# D13 §3.5：配对标志的类级默认（False）。apply_weights 里必须直读属性——
# getattr(x, name, default) 在 dynamo fullgraph 下遇到 nn.Module 缺失属性会
# 内联 __getattr__ 并抛 AttributeError（EngineCore 启动失败实证）；类级默认
# 使属性在任何 linear（含 lm_head 的 ParallelLMHead 分支）上恒可解析，env
# 门关闭/离线直调同样安全。配对实例在装载期以实例属性覆盖为 True
# （d13_norm_quant._install_pair）。
try:
    from vllm.model_executor.layers.linear import LinearBase as _LinearBase

    _LinearBase._fl_expects_prequant = False
except Exception:  # pragma: no cover
    pass
try:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        ParallelLMHead as _ParallelLMHead,
    )

    _ParallelLMHead._fl_expects_prequant = False
except Exception:  # pragma: no cover
    pass


class FLW8A8DynamicLinearKernel(Int8ScaledMMLinearKernel):
    """Use the dispatched W8A8 scaled-mm with the vLLM dynamic-W8A8 contract.

    The GEMM runs through the ``w8a8_dynamic_per_token_linear`` dispatch op
    (torch._int_mm impl preferred, FlagGems scaled_mm as ordered fallback).
    Runtime operands are ``A=[M,K] int8``, ``B=[K,N] int8``,
    ``scale_a=[M,1] fp32`` and ``scale_b=[N] fp32``.
    """

    @classmethod
    def is_supported(
        cls,
        compute_capability: int | None = None,
    ) -> tuple[bool, str | None]:
        del compute_capability
        if not is_oot_enabled():
            return False, "FL W8A8 kernels are disabled"
        if not use_flaggems_op(FLAGGEMS_W8A8_LINEAR_OP):
            return False, "FlagGems W8A8 linear is disabled by policy"
        return True, None

    @classmethod
    def can_implement(
        cls,
        config: Int8ScaledMMLinearLayerConfig,
    ) -> tuple[bool, str | None]:
        if not config.is_channelwise:
            return False, "requires per-channel weights"
        if config.is_static_input_scheme:
            return False, "requires dynamic input quantization"
        if not config.input_symmetric:
            return False, "requires symmetric INT8 activations"
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        w_q_name, w_s_name, i_s_name, i_zp_name, azp_adj_name = self.layer_param_names
        weight = getattr(layer, w_q_name)
        weight_scale = getattr(layer, w_s_name)
        if weight.ndim != 2 or weight.dtype != torch.int8:
            raise ValueError("FL W8A8 expects checkpoint weight as 2D int8")
        if weight_scale.numel() != weight.shape[0]:
            raise ValueError("FL W8A8 requires one scale per output channel")

        # FlagGems scaled_mm consumes B in [K, N] layout. Keep scales as [N].
        # MetaX C500: mctlassEx w8a8_scaled_mm_azp consumes B=[N,K] row-major
        # (the unpacker's natural layout) — skip the transpose, zero copies
        # (see w8a8/mctlass_mm.py; VLLM_FL_MCTLASS_W8A8=0 restores [K,N]).
        from .mctlass_mm import metax_mctlass_w8a8

        runtime_weight = (
            weight.contiguous()
            if metax_mctlass_w8a8()
            else weight.t().contiguous()
        )
        replace_parameter(
            layer,
            w_q_name,
            torch.nn.Parameter(runtime_weight, requires_grad=False),
        )
        replace_parameter(
            layer,
            w_s_name,
            torch.nn.Parameter(
                weight_scale.reshape(-1).to(torch.float32).contiguous(),
                requires_grad=False,
            ),
        )
        setattr(layer, i_s_name, None)
        setattr(layer, i_zp_name, None)
        setattr(layer, azp_adj_name, None)

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight, weight_scale, _, _, _ = self._get_layer_params(layer)
        # D13 §3.5 装载期静态配对（d13_norm_quant.py）：配对 norm 已在融合
        # kernel 内完成 per-token int8 量化，激活以 int8 流入——跳过量化，scale
        # 从配对 norm 的侧信道 buffer 取前 M 行。判别符用 x.dtype==int8（非
        # 仅 layer 标志）：同一 linear 在 M>阈值 的 shape 下仍收 bf16（路由分支
        # 在 norm 侧），dtype 是 dynamo 合法 guard、每 shape 固定 → graph 安全。
        # 属性须直读（类级默认见模块头注释）——getattr-default 在 dynamo
        # fullgraph 下不可用。
        if layer._fl_expects_prequant and x.dtype == torch.int8:
            x_q = x
            x_scale = layer._fl_prequant_norm._fl_d13_scale[: x.shape[0]]
            output = _w8a8_scaled_mm(
                x_q,
                weight,
                x_scale,
                weight_scale,
                bias=bias,
                out_dtype=torch.bfloat16,
            )
            return output.reshape(x.shape[0], weight_scale.numel())
        original_shape = x.shape
        x_2d = x.reshape(-1, original_shape[-1]).contiguous()
        x_q, x_scale = _dynamic_per_token_quant_int8(x_2d)
        output = _w8a8_scaled_mm(
            x_q,
            weight,
            x_scale,
            weight_scale,
            bias=bias,
            out_dtype=x.dtype,
        )
        # 输出维度 = 权重 scale 元素数（N）：对 [K,N] 与沐曦 [N,K] 两种
        # 运行时布局都成立（weight.shape 在两者间含义互换）。
        return output.reshape(*original_shape[:-1], weight_scale.numel())


def register_fl_w8a8_linear_kernel(registry: dict) -> bool:
    """Prepend FlagGems while retaining native INT8 kernels as fallback."""
    from vllm.platforms import PlatformEnum

    # PlatformFL always selects through the OOT registry. On NVIDIA that list
    # has already inherited CUDA's native Cutlass/Triton candidates, so
    # prepending here keeps them available as ordered fallbacks.
    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if FLW8A8DynamicLinearKernel not in candidates:
        candidates.insert(0, FLW8A8DynamicLinearKernel)

    # Warm the W8A8 dispatch ops before vLLM's AOT fullgraph capture: a
    # first-time resolve inside torch.compile would enter locks dynamo
    # cannot trace (EngineCore startup failure). Runs at quant registration,
    # well before the first forward.
    try:
        _dynamic_per_token_quant_int8.pre_resolve()
        _w8a8_scaled_mm.pre_resolve()
    except Exception:  # pragma: no cover - never block kernel registration
        pass
    return True


__all__ = [
    "FLAGGEMS_W8A8_LINEAR_OP",
    "FLW8A8DynamicLinearKernel",
    "register_fl_w8a8_linear_kernel",
]
