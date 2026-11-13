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

import os

import torch

try:  # Triton is optional for the pure-torch fallback path.
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

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


def _fused_epilogue_enabled() -> bool:
    """Single-kernel epilogue toggle (on by default; bit-identical either way)."""
    return os.environ.get("VLLM_FL_INTMM_FUSED_EPILOGUE", "1").lower() in (
        "1",
        "true",
        "yes",
    )


if _HAS_TRITON:

    @triton.jit  # bare jit: no libentry (stream-race + event-timing history)
    def _w8a8_int32_epilogue_kernel(
        acc_ptr,
        out_ptr,
        xs_ptr,
        ws_ptr,
        bias_ptr,
        M,
        N,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        HAS_BIAS: tl.constexpr,
    ):
        # int32 -> fp32 -> *x_scale -> *weight_scale -> (+bias fp32) -> bf16,
        # same op order and roundings as the torch-chain fallback below, so
        # the two paths are bit-identical (asserted by unit tests).
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        acc = tl.load(
            acc_ptr + offs_m[:, None] * N + offs_n[None, :],
            mask=mask,
            other=0,
        ).to(tl.float32)
        xs = tl.load(xs_ptr + offs_m, mask=offs_m < M, other=0.0)
        ws = tl.load(ws_ptr + offs_n, mask=offs_n < N, other=0.0)
        y = acc * xs[:, None]
        y = y * ws[None, :]
        if HAS_BIAS:
            bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(
                tl.float32
            )
            y = y + bias[None, :]
        tl.store(
            out_ptr + offs_m[:, None] * N + offs_n[None, :],
            y.to(tl.bfloat16),
            mask=mask,
        )

    def _run_fused_epilogue(
        acc: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        M: int,
        N: int,
    ) -> torch.Tensor:
        out = torch.empty((M, N), dtype=torch.bfloat16, device=acc.device)
        xs = x_scale if x_scale.is_contiguous() else x_scale.contiguous()
        ws = (
            weight_scale
            if weight_scale.is_contiguous()
            else weight_scale.contiguous()
        )
        bias_arg = bias if bias is not None else acc  # unread when HAS_BIAS=0
        if bias is not None and not bias.is_contiguous():
            bias_arg = bias.contiguous()
        BLOCK_M = 32
        BLOCK_N = 256
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        _w8a8_int32_epilogue_kernel[grid](
            acc,
            out,
            xs,
            ws,
            bias_arg,
            M,
            N,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HAS_BIAS=bias is not None,
            num_warps=4,
        )
        return out


# ---- Fused GEMM+epilogue (single Triton kernel, chunked-prefill regime) ----
#
# Serve profiling (results/20261113_d2_prefill_attribution) shows the separate
# epilogue costs ~24% of all GPU kernel time — the int32 [M,N] round trip
# cancels most of the int8 GEMM savings — plus one extra kernel launch per
# linear (Corex launches are ~50-105us host-side in eager prefill). This
# kernel consumes the already-quantized int8 activations (produced by the
# bit-exact bare-quant kernel, unchanged), accumulates via int8 tl.dot into
# int32 (bit-exact vs ``torch._int_mm`` — probe-verified on this stack,
# scripts/d2/phase0_int8_dot_probe.py), and applies the epilogue in
# registers with the exact op order of ``_w8a8_int32_epilogue_kernel``,
# writing bf16 exactly once. For M >= _MIN_ROWS it replaces the
# ``_int_mm`` + epilogue pair; every other shape keeps the cuBLAS chain.
_FUSED_GEMM_CONFIGS = {
    # (K, N) -> (BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages)
    # phase0c production-kernel sweep (2026-11-13): (256,256,64) dominates
    # every model shape — gate_up 96% / o 96% / qkv 93% / down 86% of
    # cuBLAS _int_mm (M=2048, graph caliber). down (6144, 2048) re-enters
    # the fused table: at 86% the epilogue+launch saving pays back (its
    # earlier 59% tile lost to the cuBLAS chain). GROUP_M=4 beat 8 on the
    # top configs (L2 swizzle; down -8%).
    (2048, 12288): (256, 256, 64, 4, 8, 2),
    (6144, 2048): (256, 256, 64, 4, 8, 2),
    (2048, 2560): (256, 256, 64, 4, 8, 2),
    (2048, 2048): (256, 256, 64, 4, 8, 2),
}
_FUSED_WARMED: set = set()  # (K, N, even_m) specializations launched once
_FUSED_GEMM_BROKEN = False  # memoized off-switch (never risk mgr.call crawl)
_FUSED_GEMM_MIN_M = 256  # below this the cuBLAS chain wins (see _run_fused_gemm)


def _fused_gemm_enabled() -> bool:
    """Fused single-kernel GEMM+epilogue toggle (on by default)."""
    return os.environ.get("VLLM_FL_INTMM_FUSED_GEMM", "1").lower() in (
        "1",
        "true",
        "yes",
    )


if _HAS_TRITON:

    @triton.jit(do_not_specialize=["M"])  # bare jit; M tails must not recompile
    def _w8a8_fused_gemm_kernel(
        x_ptr,  # [M, K] int8 (bare-quant output)
        w_ptr,  # [K, N] int8 contiguous
        y_ptr,  # [M, N] bf16 out
        xs_ptr,  # fp32, M values (per-token scale)
        ws_ptr,  # fp32, N values (per-channel scale)
        bias_ptr,  # fp32 [N] or unread dummy
        M,
        N,
        K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
        GROUP_M: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        EVEN_M: tl.constexpr,
    ):
        # K % BLOCK_K == 0 and N % BLOCK_N == 0 are guaranteed by the config
        # table. EVEN_M (M % BLOCK_M == 0, the chunked-prefill common case)
        # compiles the M masking out entirely — predicated A-loads/store cost
        # ~3-6x on this backend. Odd-M chunk tails take the masked path;
        # zero-padded rows contribute nothing to tl.dot, staying bit-exact.
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BLOCK_M)
        num_pid_n = tl.cdiv(N, BLOCK_N)
        num_pid_in_group = GROUP_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        b_ptrs = w_ptr + offs_k[:, None] * N + offs_n[None, :]
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
        if EVEN_M:
            for _ in range(0, K, BLOCK_K):
                a = tl.load(a_ptrs)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * N
        else:
            m_mask = offs_m < M
            for _ in range(0, K, BLOCK_K):
                a = tl.load(a_ptrs, mask=m_mask[:, None], other=0)
                b = tl.load(b_ptrs)
                acc = tl.dot(a, b, acc, out_dtype=tl.int32)
                a_ptrs += BLOCK_K
                b_ptrs += BLOCK_K * N
        # Epilogue tail: verbatim op order of _w8a8_int32_epilogue_kernel —
        # int32 -> fp32 -> *x_scale -> *weight_scale -> (+bias) -> bf16. The
        # two muls stay separate statements (no FMA reassociation).
        if EVEN_M:
            xs = tl.load(xs_ptr + offs_m)
            y = acc.to(tl.float32)
            y = y * xs[:, None]
            y = y * tl.load(ws_ptr + offs_n)[None, :]
            if HAS_BIAS:
                y = y + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
            tl.store(
                y_ptr + offs_m[:, None] * N + offs_n[None, :],
                y.to(tl.bfloat16),
            )
        else:
            m_mask = offs_m < M
            xs = tl.load(xs_ptr + offs_m, mask=m_mask, other=0.0)
            y = acc.to(tl.float32)
            y = y * xs[:, None]
            y = y * tl.load(ws_ptr + offs_n)[None, :]
            if HAS_BIAS:
                y = y + tl.load(bias_ptr + offs_n).to(tl.float32)[None, :]
            tl.store(
                y_ptr + offs_m[:, None] * N + offs_n[None, :],
                y.to(tl.bfloat16),
                mask=m_mask[:, None],
            )

    def _run_fused_gemm(
        x_q: torch.Tensor,
        weight: torch.Tensor,
        x_scale: torch.Tensor,
        weight_scale: torch.Tensor,
        bias: torch.Tensor | None,
        M: int,
        N: int,
        K: int,
    ) -> torch.Tensor | None:
        """Launch the fused kernel, or None to fall back to the cuBLAS chain."""
        global _FUSED_GEMM_BROKEN
        if _FUSED_GEMM_BROKEN:
            return None
        cfg = _FUSED_GEMM_CONFIGS.get((K, N))
        if cfg is None:
            return None
        # Decode batches (official bench conc <= 128 -> M <= 128) run inside
        # cudagraphs where launches are free and the cuBLAS chain already
        # wins; at small M the fused grid also starves the 16 SMs. Fused is
        # for the chunked-prefill regime.
        if M < _FUSED_GEMM_MIN_M:
            return None
        even_m = M % cfg[0] == 0
        if not even_m:
            # Odd-M chunk tails keep the cuBLAS chain: the masked fused path
            # pays 3-6x predication on this backend. Official bench inputs
            # (4096/16384) divide the 2048 token budget exactly, so tails are
            # rare; the masked kernel branch stays for generality.
            return None
        # Triton JIT compiles with device-side syncs — illegal inside a CUDA
        # graph capture. vLLM's dummy forward (M=2048) warms the even-M
        # specialization of each (K, N); anything else inside a capture
        # falls back to the cuBLAS chain.
        key = (K, N, even_m)
        if torch.cuda.is_current_stream_capturing() and key not in _FUSED_WARMED:
            return None
        try:
            BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M, num_warps, num_stages = cfg
            out = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)
            xs = x_scale if x_scale.is_contiguous() else x_scale.contiguous()
            ws = (
                weight_scale
                if weight_scale.is_contiguous()
                else weight_scale.contiguous()
            )
            bias_arg = bias if bias is not None else x_q  # unread when HAS_BIAS=0
            if bias is not None and not bias.is_contiguous():
                bias_arg = bias.contiguous()
            grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
            _w8a8_fused_gemm_kernel[grid](
                x_q,
                weight,
                out,
                xs,
                ws,
                bias_arg,
                M,
                N,
                K,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
                BLOCK_K=BLOCK_K,
                GROUP_M=GROUP_M,
                HAS_BIAS=bias is not None,
                EVEN_M=even_m,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            _FUSED_WARMED.add(key)
            return out
        except Exception:
            # Any failure permanently disables the fused path (an impl
            # exception would otherwise drop the CachedOp into the ~0.7ms
            # mgr.call slow path for the whole process).
            _FUSED_GEMM_BROKEN = True
            return None


def _int8_scaled_mm_int_mm_impl(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
    out_dtype: torch.dtype | None,
) -> torch.Tensor:
    """Raw implementation. Call through ``int8_scaled_mm_int_mm`` instead."""
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

    # Chunked-prefill regime (M >= 17): single fused GEMM+epilogue kernel —
    # removes the int32 [M,N] round trip (~24% of serve GPU kernel time) and
    # one launch per linear. Falls back to the cuBLAS chain for unknown
    # shapes, env-off, or first-call-inside-capture.
    if (
        M >= _MIN_ROWS
        and _HAS_TRITON
        and _fused_gemm_enabled()
        and x_scale.dtype == torch.float32
        and weight_scale.dtype == torch.float32
        and (out_dtype is None or out_dtype == torch.bfloat16)
    ):
        out = _run_fused_gemm(
            x_q, weight, x_scale, weight_scale, bias, M, N, K
        )
        if out is not None:
            return out

    if M < _MIN_ROWS:
        x_q = _pad_rows(x_q)

    # Epilogue order is the emu contract: slice -> fp32 -> x_scale ->
    # weight_scale -> fp32 bias -> output cast. Keep it verbatim. The fused
    # single-kernel path implements the identical math (bit-equal, tested);
    # a 4-kernel torch chain costs ~90us at M=64 on this backend, i.e. more
    # than the GEMM itself.
    acc = torch._int_mm(x_q, weight)
    if (
        _HAS_TRITON
        and _fused_epilogue_enabled()
        and x_scale.dtype == torch.float32
        and weight_scale.dtype == torch.float32
        and (out_dtype is None or out_dtype == torch.bfloat16)
    ):
        return _run_fused_epilogue(acc, x_scale, weight_scale, bias, M, N)
    y = acc[:M].to(torch.float32)
    y = y.mul_(x_scale.reshape(M, 1).to(torch.float32))
    y = y.mul_(weight_scale.reshape(1, N).to(torch.float32))
    if bias is not None:
        y = y + bias
    return y.to(out_dtype if out_dtype is not None else torch.bfloat16)


def _register_custom_op() -> bool:
    """Wrap the bf16 hot path as a torch custom op (dynamo opaque boundary).

    vLLM AOT-compiles the model forward once with a large M; a traced
    ``if M < _MIN_ROWS`` gets baked to the unpadded branch, so a decode
    cudagraph capture at M=16 would hit ``torch._int_mm`` with M=16 and
    abort EngineCore. Inside a custom op the branch executes for real at
    capture/run time. Returns False when the op is already registered.
    """
    try:
        if hasattr(torch.ops.vllm_fl, "w8a8_int_mm_scaled_gemm"):
            return False

        @torch.library.custom_op(
            "vllm_fl::w8a8_int_mm_scaled_gemm", mutates_args=()
        )
        def w8a8_int_mm_scaled_gemm(
            x_q: torch.Tensor,
            weight: torch.Tensor,
            x_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            return _int8_scaled_mm_int_mm_impl(
                x_q, weight, x_scale, weight_scale, bias, torch.bfloat16
            )

        @w8a8_int_mm_scaled_gemm.register_fake
        def _(
            x_q: torch.Tensor,
            weight: torch.Tensor,
            x_scale: torch.Tensor,
            weight_scale: torch.Tensor,
            bias: torch.Tensor | None,
        ) -> torch.Tensor:
            return torch.empty(
                (x_q.shape[0], weight.shape[1]),
                dtype=torch.bfloat16,
                device=x_q.device,
            )

        return True
    except Exception:  # pragma: no cover - registration is best-effort
        return False


_CUSTOM_OP_OK = _register_custom_op()


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

    Returns ``[M, N]`` in ``out_dtype`` (default bfloat16). The bf16 hot
    path routes through a torch custom op so dynamo treats it as an opaque
    call (shape-dependent M<17 padding must run at capture time, not trace
    time); other dtypes fall back to the raw implementation.
    """
    if _CUSTOM_OP_OK and (out_dtype is None or out_dtype == torch.bfloat16):
        return torch.ops.vllm_fl.w8a8_int_mm_scaled_gemm(
            x_q, weight, x_scale, weight_scale, bias
        )
    return _int8_scaled_mm_int_mm_impl(
        x_q, weight, x_scale, weight_scale, bias, out_dtype
    )


__all__ = [
    "HAS_INT_MM",
    "int8_scaled_mm_int_mm",
    "int_mm_available",
]
