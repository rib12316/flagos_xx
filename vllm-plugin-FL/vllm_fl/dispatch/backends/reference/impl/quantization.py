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
    accumulator = x_q.to(torch.float32) @ weight.to(torch.float32)
    y = accumulator.mul_(x_scale.reshape(M, 1).to(torch.float32))
    y = y.mul_(weight_scale.reshape(1, -1).to(torch.float32))
    if bias is not None:
        y = y + bias
    return y.to(out_dtype if out_dtype is not None else torch.bfloat16)
