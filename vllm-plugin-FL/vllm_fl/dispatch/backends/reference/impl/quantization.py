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
    scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference two-step torch chain (norm → quant) for the fused D13 op.

    Matches the Triton bare kernel's intermediate-precision semantics:
    h=(x+residual) lands in bf16, the norm output lands in bf16 before the
    f32 quant math. Uses the w8a8 reference quantizer for the second step.
    ``scale_out``（§3.5 接线侧信道）：给定时写入其前 M 行并返回该切片（与
    flaggems impl 签名 parity）。
    """
    h = x + residual                                    # bf16 intermediate
    hf = h.float()
    var = hf.pow(2).sum(-1, keepdim=True) / h.shape[1]
    y = (hf * torch.rsqrt(var + epsilon)
         * weight.to(torch.float32)).to(torch.bfloat16)
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8,
    )
    q, scale = dynamic_per_token_quant_int8(y)
    if scale_out is not None:
        if scale_out.ndim != 2 or scale_out.shape[1] != 1 \
                or scale_out.shape[0] < x.shape[0] \
                or scale_out.dtype != torch.float32:
            raise ValueError(
                f"scale_out must be [≥{x.shape[0]},1] fp32, "
                f"got {tuple(scale_out.shape)}")
        scale_out[: x.shape[0]].copy_(scale)
        scale = scale_out[: x.shape[0]]
    return q, scale, h
