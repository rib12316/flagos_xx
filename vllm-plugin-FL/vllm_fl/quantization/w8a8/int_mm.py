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
"""W8A8 scaled GEMM via ``torch._int_mm`` (cuBLAS int8, int32 accumulation).

On BI-V150 the FlagGems Triton int8 ``scaled_mm`` achieves less than half
the bandwidth efficiency of the bf16 cuBLAS path, making W8A8 a net loss.
``torch._int_mm`` is bit-exact against an int32 matrix multiply and 4-8x
faster than the Triton kernel on the fused decode shapes, so this module
is registered as the high-priority impl of the
``w8a8_dynamic_per_token_linear`` dispatch op (flag_gems ``scaled_mm``
stays as the ordered fallback).

The epilogue mirrors the fake-quant emulator bit-for-bit
(int32 -> fp32 -> mul x_scale -> mul weight_scale -> optional fp32 bias
add -> cast), so the frozen 30-question precision verdicts carry over to
this path unchanged.
"""

from __future__ import annotations

import torch

HAS_INT_MM = hasattr(torch, "_int_mm")

# cuBLAS int8 GEMM requires M > 16. Pad with leading zero rows (zero rows
# contribute nothing to the accumulation; bit-exactness verified against
# the unpadded int32 reference) and slice back after the GEMM.
_MIN_ROWS = 17

# Persistent per-(device, K) zero [17, K] int8 buffers. The tail rows are
# never written after creation, so one zero-fill lasts for the process.
# Buffers are allocated on first call (vLLM's profile/dummy forward runs
# before CUDA graph capture), have a stable address, and every reader and
# writer shares the single compute stream, so reuse across layers and
# graphs is race-free.
_PAD_BUFS: dict = {}


def int_mm_available() -> bool:
    """Whether the ``torch._int_mm`` path can run (dispatch probing hook)."""
    return HAS_INT_MM and torch.cuda.is_available()


def _pad_rows(x_q: torch.Tensor) -> torch.Tensor:
    """Copy ``x_q [M, K]`` into the head of a persistent zero ``[17, K]``."""
    M, K = x_q.shape
    key = (x_q.device.index, K)
    buf = _PAD_BUFS.get(key)
    if buf is None:
        buf = torch.zeros(_MIN_ROWS, K, dtype=torch.int8, device=x_q.device)
        _PAD_BUFS[key] = buf
    buf[:M].copy_(x_q)
    return buf


def int8_scaled_mm_int_mm(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Dynamic-token/per-channel W8A8 linear GEMM on ``torch._int_mm``.

    Contract (operand semantics match ``flag_gems.scaled_mm``):

    - ``x_q``: ``[M, K]`` int8 activations (quantized per token)
    - ``weight``: ``[K, N]`` int8 runtime layout (transposed checkpoint)
    - ``x_scale``: fp32 with ``M`` values (per-token, usually ``[M, 1]``)
    - ``weight_scale``: fp32 with ``N`` values (per-channel, usually ``[N]``)
    - ``bias``: optional ``[N]`` bias, added in fp32 before the output cast

    Returns ``[M, N]`` in ``out_dtype`` (default bfloat16).
    """
    if not HAS_INT_MM:
        raise RuntimeError("torch._int_mm is unavailable on this torch build")
    if x_q.ndim != 2 or x_q.dtype != torch.int8:
        raise ValueError(
            f"x_q must be 2D int8 [M, K], got {x_q.ndim}D {x_q.dtype}"
        )
    if weight.ndim != 2 or weight.dtype != torch.int8:
        raise ValueError(
            f"weight must be 2D int8 [K, N], got {weight.ndim}D {weight.dtype}"
        )
    M, K = x_q.shape
    if weight.shape[0] != K:
        raise ValueError(
            f"K mismatch: x_q has K={K}, weight has K={weight.shape[0]}"
        )
    if K % 8:
        raise ValueError(
            f"torch._int_mm requires K to be a multiple of 8, got {K}"
        )
    N = weight.shape[1]
    if x_scale.numel() != M:
        raise ValueError(f"x_scale must hold M={M} values, got {x_scale.numel()}")
    if weight_scale.numel() != N:
        raise ValueError(
            f"weight_scale must hold N={N} values, got {weight_scale.numel()}"
        )

    if not x_q.is_contiguous():
        x_q = x_q.contiguous()
    if not weight.is_contiguous():
        weight = weight.contiguous()
    if M < _MIN_ROWS:
        x_q = _pad_rows(x_q)

    # Epilogue order is the emu contract: slice -> fp32 -> x_scale ->
    # weight_scale -> fp32 bias -> output cast. Keep it verbatim.
    y = torch._int_mm(x_q, weight)[:M].to(torch.float32)
    y = y.mul_(x_scale.reshape(M, 1).to(torch.float32))
    y = y.mul_(weight_scale.reshape(1, N).to(torch.float32))
    if bias is not None:
        y = y + bias
    return y.to(out_dtype if out_dtype is not None else torch.bfloat16)


__all__ = [
    "HAS_INT_MM",
    "int8_scaled_mm_int_mm",
    "int_mm_available",
]
