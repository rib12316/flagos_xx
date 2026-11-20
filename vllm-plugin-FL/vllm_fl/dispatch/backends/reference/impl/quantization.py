# Copyright (c) 2026 BAAI. All rights reserved.

"""PyTorch reference quantization operator implementations."""

from __future__ import annotations

import torch


def dynamic_per_token_quant_int8_torch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8,
    )

    return dynamic_per_token_quant_int8(x)


def w8a8_dynamic_per_token_linear_torch(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Portable torch fallback for the W8A8 dynamic-token linear op.

    The accumulator uses an fp32 matmul of the integer operands: this stack
    has no CUDA integer GEMM, and fp32 holds int8 products exactly up to
    2^24, beyond which the sub-ulp rounding (relative ~6e-8) is negligible
    against the int8 quantization noise. Last-resort reference only.
    """
    M, _ = x_q.shape
    # 布局自适应：常规栈 weight=[K,N]；沐曦 mctlass 栈 weight=[N,K]（见
    # w8a8/mctlass_mm.py——unpack 天然布局，process 层跳过转置）。方形层
    # 用运行时门控消歧。
    from vllm_fl.quantization.w8a8.mctlass_mm import metax_mctlass_w8a8

    N = weight_scale.numel()
    is_nk = (
        weight.shape[0] == N and weight.shape[1] != N
    ) or metax_mctlass_w8a8()
    operand = weight.to(torch.float32).t() if is_nk else weight.to(torch.float32)
    accumulator = x_q.to(torch.float32) @ operand
    y = accumulator.mul_(x_scale.reshape(M, 1).to(torch.float32))
    y = y.mul_(weight_scale.reshape(1, -1).to(torch.float32))
    if bias is not None:
        y = y + bias
    return y.to(out_dtype if out_dtype is not None else torch.bfloat16)


def fused_add_rms_norm_quant_int8_torch(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference two-step torch chain (norm → quant) for the fused D13 op.

    Matches the Triton bare kernel's intermediate-precision semantics:
    h=(x+residual) lands in bf16, the norm output lands in bf16 before the
    f32 quant math. Uses the w8a8 reference quantizer for the second step.
    """
    h = x + residual                                    # bf16 intermediate
    hf = h.float()
    var = hf.pow(2).sum(-1, keepdim=True) / h.shape[1]
    y = (hf * torch.rsqrt(var + epsilon)).to(torch.bfloat16)
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8,
    )
    q, scale = dynamic_per_token_quant_int8(y)
    return q, scale, h
