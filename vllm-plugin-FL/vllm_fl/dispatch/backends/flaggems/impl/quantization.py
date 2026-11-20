# Copyright (c) 2026 BAAI. All rights reserved.

"""FlagGems-backed quantization operator implementations."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry


@triton.jit
def _round_half_to_even(x):
    """Match torch.round without relying on a vendor-specific libdevice."""
    floor = tl.floor(x)
    fraction = x - floor
    is_odd = tl.abs(floor - 2.0 * tl.floor(floor / 2.0)) > 0.5
    round_up = (fraction > 0.5) | ((tl.abs(fraction - 0.5) < 1e-10) & is_odd)
    return tl.where(round_up, floor + 1.0, floor)


@libentry()
@triton.jit
def _dynamic_per_token_quant_int8_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(
        input_ptr + token_idx * hidden_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(values), axis=0)
    inv_scale = tl.where(absmax != 0.0, 127.0 / absmax, 0.0)
    quantized = _round_half_to_even(values * inv_scale)
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)

    tl.store(
        output_ptr + token_idx * hidden_size + offsets,
        quantized.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + token_idx, absmax / 127.0)


def dynamic_per_token_quant_int8_flaggems_vllm(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run FlagGems-vLLM dynamic symmetric per-token INT8 quantization.

    ``scaled_int8_quant`` selects its dynamic path when ``scale`` is ``None``.
    Import and runtime failures propagate to dispatch, which then tries the
    local FlagGems/Triton implementation before the PyTorch reference.
    """
    from flaggems_vllm.ops.scaled_int8_quant import scaled_int8_quant

    output, scale, _ = scaled_int8_quant(
        x,
        scale=None,
        azp=None,
        symmetric=True,
    )
    return output, scale


@triton.jit
def _dynamic_per_token_quant_int8_bare_kernel(
    input_ptr,
    output_ptr,
    scale_ptr,
    hidden_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Same math as _dynamic_per_token_quant_int8_kernel above, launched as a
    # bare @triton.jit (no @libentry): libentry's event-timing fallback adds
    # ~0.1-0.9ms per call on this backend and has a known stream race when
    # called in-process, so the W8A8 hot path must not go through it.
    token_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < hidden_size
    values = tl.load(
        input_ptr + token_idx * hidden_size + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)

    absmax = tl.max(tl.abs(values), axis=0)
    # Bit-compatibility with the torch reference (the fake-quant emu):
    # torch lowers scalar-divisor division to a reciprocal multiply
    # (t/127 == t*(1/127)) and scalar/tensor to reciprocal().mul, so this
    # kernel replicates exactly that arithmetic instead of a correctly
    # rounded division (which differs by ~1ulp and flips round-half-even
    # ties on ~0.02% of elements).
    inv_scale = tl.where(
        absmax != 0.0,
        tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0,
        0.0,
    )
    quantized = _round_half_to_even(values * inv_scale)
    quantized = tl.minimum(tl.maximum(quantized, -128.0), 127.0)

    tl.store(
        output_ptr + token_idx * hidden_size + offsets,
        quantized.to(tl.int8),
        mask=mask,
    )
    tl.store(scale_ptr + token_idx, absmax * (1.0 / 127.0))


def dynamic_per_token_quant_int8_bare(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bare-launch Triton per-token INT8 quantization (no libentry wrapper)."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D [tokens, hidden_size] tensor")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if x.shape[1] == 0:
        raise ValueError("hidden_size must be positive")

    x = x.contiguous()
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(x.shape[1])
    num_warps = 8 if block_size >= 8192 else 4
    _dynamic_per_token_quant_int8_bare_kernel[(x.shape[0],)](
        x,
        output,
        scale,
        hidden_size=x.shape[1],
        BLOCK_SIZE=block_size,
        num_warps=num_warps,
    )
    return output, scale


def dynamic_per_token_quant_int8_flaggems_triton(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the in-tree backend-neutral FlagGems/Triton fallback kernel."""
    if x.ndim != 2:
        raise ValueError("x must be a 2D [tokens, hidden_size] tensor")
    if not x.is_floating_point():
        raise TypeError(f"x must be floating point, got {x.dtype}")
    if x.shape[1] == 0:
        raise ValueError("hidden_size must be positive")

    x = x.contiguous()
    output = torch.empty_like(x, dtype=torch.int8)
    scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    block_size = triton.next_power_of_2(x.shape[1])
    with torch_device_fn.device(x.device):
        _dynamic_per_token_quant_int8_kernel[(x.shape[0],)](
            x,
            output,
            scale,
            hidden_size=x.shape[1],
            BLOCK_SIZE=block_size,
        )
    return output, scale


def w8a8_dynamic_per_token_linear_scaledmm(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Run the FlagGems scaled_mm for the W8A8 dynamic-token linear op.

    Signature matches ``vllm_fl.quantization.w8a8.int_mm`` 1:1 so this impl
    is a pure pass-through of the FlagGems kernel with identical operand
    semantics (``x_q [M,K] int8``, ``weight [K,N] int8``, scales fp32).
    Under the V150 TN layout weights arrive [N,K] row-major — materialize
    the transpose for FlagGems (cold fallback path only; the hot path never
    routes here).
    """
    from flag_gems import scaled_mm

    from vllm_fl.quantization.w8a8.int_mm import _intmm_tn_layout

    if _intmm_tn_layout():
        weight = weight.t().contiguous()
    return scaled_mm(
        x_q,
        weight,
        x_scale,
        weight_scale,
        bias=bias,
        out_dtype=out_dtype if out_dtype is not None else torch.bfloat16,
    )


# D13 fused_add_rms_norm + per-token int8 quant —— 交付头注（开发单 §0 四要素，
# 模板 vendor/iluvatar/attn_splitkv.py）
# ----------------------------------------------------------------------------
# 数学口径来源：两步链「同数学 norm-only kernel + 本文件
#   _dynamic_per_token_quant_int8_bare_kernel（:76，位兼容 quant 口径）」，
#   T0 对拍位精确（scripts/d13/norm_quant_fuse_t0.py，2026-11-20，
#   results/20261120_c500_reprofiling/d13_norm_quant_fuse/README.md）。
#   两处中间 bf16 round 必须保留：h=(x+resid) 落 bf16、norm 输出落 bf16 再读
#   f32（对齐 vllm inductor triton 链的中间精度），删任一处即破位一致。
# 位精确声明：q/scale/residual_out vs 上述两步链 bitwise 全一致（K6144 档
#   1/25M 差异为编译器 fma 折叠级，详见 T0 README）。
# graph 相容设计：裸 @triton.jit（禁 @libentry——事件计时 ~0.9ms/次 + 流竞态
#   前科，见 :83 注释）、无 autotune、grid/warps 纯 shape 函数（
#   next_power_of_2(N)，w=8 若 BLOCK≥8192 否则 4）——capture/replay 行为确定。
# 接线（§3.5）：vllm_fl/quantization/w8a8/d13_norm_quant.py 装载期静态配对，
#   scale 经 scale_out 侧信道 buffer 直写（本 wrapper 的可选参数），配对
#   linear 见 w8a8/linear.py apply_weights 的 _fl_expects_prequant 分支。
# 纪律声明：性能结论只认 CUDA graph 口径；prefill 大 M 档（M4096 K6144）
#   0.894× 寄存器压力——调用侧 per-shape 路由（decode 档 M≤m_fuse_max 才走
#   融合），勿整链启用。
@triton.jit
def _fused_add_rms_norm_quant_int8_bare_kernel(
    x_ptr,
    resid_ptr,
    w_ptr,
    resid_out_ptr,
    q_ptr,
    scale_ptr,
    eps,
    N: tl.constexpr,
    BLOCK: tl.constexpr,
):
    # D13 fused_add_rms_norm + per-token int8 quant（单 kernel 直出）。
    # 位口径：复刻两步链的两处中间 bf16 round（h=(x+resid) 落 bf16、norm 输出
    # 落 bf16 再读 f32），quant 段与 _dynamic_per_token_quant_int8_bare_kernel
    # 逐位同口径（ieee fdiv recip*127 + _round_half_to_even）。
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row.to(tl.int64) * N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(resid_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    h16 = (x + r).to(tl.bfloat16)
    tl.store(resid_out_ptr + base + offs, h16, mask=mask)
    hf = h16.to(tl.float32)
    var = tl.sum(hf * hf, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y16 = (hf * rrms * w).to(tl.bfloat16)
    yf = y16.to(tl.float32)
    absmax = tl.max(tl.abs(yf), axis=0)
    inv_scale = tl.where(
        absmax != 0.0,
        tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0,
        0.0,
    )
    q = _round_half_to_even(yf * inv_scale)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(q_ptr + base + offs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, absmax * (1.0 / 127.0))


def fused_add_rms_norm_quant_int8_bare(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    scale_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused add-RMSNorm + per-token INT8 quantization (bare launch).

    Returns (q_int8, scale, residual_out) where residual_out = (x + residual)
    in bf16 (the fused_add side effect) and (q, scale) are the per-token
    dynamic int8 quantization of the RMSNorm output — bit-compatible with the
    two-step chain [norm-only kernel of identical math] + [bare quant kernel].

    ``scale_out``（§3.5 接线侧信道）：给定时 [≥M,1] fp32 常驻 buffer，kernel
    直写其前 M 行并原样返回该切片——scale 张量身份跨 forward 恒定，dynamo
    guard 稳定、cudagraph 静态内存传递；None 时维持现分配逻辑（T0 回归不破）。
    """
    if x.ndim != 2 or residual.shape != x.shape:
        raise ValueError("x/residual must be matching 2D [tokens, hidden]")
    if x.shape[1] == 0:
        raise ValueError("hidden_size must be positive")
    x = x.contiguous()
    residual = residual.contiguous()
    M, N = x.shape
    resid_out = torch.empty_like(residual)
    q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    if scale_out is None:
        scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    else:
        if scale_out.ndim != 2 or scale_out.shape[1] != 1 \
                or scale_out.shape[0] < M or scale_out.dtype != torch.float32:
            raise ValueError(
                f"scale_out must be [≥{M},1] fp32, got {tuple(scale_out.shape)}")
        scale = scale_out[:M]
    block = triton.next_power_of_2(N)
    nw = 8 if block >= 8192 else 4
    _fused_add_rms_norm_quant_int8_bare_kernel[(M,)](
        x, residual, weight, resid_out, q, scale, epsilon,
        N=N, BLOCK=block, num_warps=nw,
    )
    return q, scale, resid_out
