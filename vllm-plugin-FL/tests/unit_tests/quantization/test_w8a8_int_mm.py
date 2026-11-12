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

import pytest
import torch

from vllm_fl.quantization.w8a8.int_mm import (
    _MIN_ROWS,
    _PAD_BUFS,
    HAS_INT_MM,
    int8_scaled_mm_int_mm,
)
from vllm_fl.quantization.w8a8.reference import (
    dynamic_per_token_quant_int8,
    w8a8_linear_reference,
)

needs_int_mm = pytest.mark.skipif(
    not (torch.cuda.is_available() and HAS_INT_MM),
    reason="requires CUDA and torch._int_mm",
)

# CUDA integer matmul is unavailable on this stack ("addmm_cuda" not
# implemented for Int), so the bit-exact oracle runs the int32 matmul on
# CPU. Output row i depends only on x_q[i] / x_scale[i] / weight, so
# fused-shape cases compare a sampled row subset against per-row CPU
# matmuls to keep the oracle cheap.
_FUSED_SHAPES = [(2048, 2560), (6144, 2048)]


def _make_operands(M, K, N, *, with_bias, seed=0):
    """Deterministic per-token int8 x and per-channel int8 weight [K, N]."""
    gen = torch.Generator().manual_seed(seed)
    x = (torch.randn(M, K, generator=gen) * 2).to(torch.bfloat16).cuda()
    weight_fp = torch.randn(N, K, generator=gen) * 0.02
    weight_scale = (weight_fp.abs().amax(dim=1) / 127).clamp_min(1e-12).cuda()
    weight = (
        (weight_fp.cuda() / weight_scale[:, None])
        .round()
        .clamp(-127, 127)
        .to(torch.int8)
    )
    x_q, x_scale = dynamic_per_token_quant_int8(x)
    bias = (
        (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16).cuda()
        if with_bias
        else None
    )
    return x_q, x_scale, weight.t().contiguous(), weight_scale, bias


def _cpu_oracle_rows(x_q, weight, x_scale, weight_scale, bias, out_dtype, rows):
    """Exact int32 CPU matmul for selected output rows + emu epilogue order."""
    selected = x_q[rows].cpu().to(torch.int32)
    accumulator = selected @ weight.cpu().to(torch.int32)
    y = accumulator.to(torch.float32)
    y = y * x_scale[rows].cpu().reshape(-1, 1)
    y = y * weight_scale.cpu().reshape(1, -1)
    if bias is not None:
        y = y + bias.cpu()
    return y.to(out_dtype)


def _row_indices(M):
    return sorted({0, M // 2, M - 1, min(M - 1, 3)})


@pytest.mark.parametrize("M", [1, 2, 5, 16, 17, 64])
@pytest.mark.parametrize(
    "K_N",
    [(64, 32), (128, 256), (72, 48)],
    ids=["k64n32", "k128n256", "k72n48"],
)
@pytest.mark.parametrize("with_bias", [False, True])
@needs_int_mm
def test_bitexact_small_shapes_vs_cpu_int32(M, K_N, with_bias):
    K, N = K_N
    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=with_bias
    )

    actual = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )
    rows = list(range(M))
    expected = _cpu_oracle_rows(
        x_q, weight, x_scale, weight_scale, bias, torch.bfloat16, rows
    ).cuda()

    assert actual.shape == (M, N)
    assert actual.dtype == torch.bfloat16
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("M", [1, 16, 17, 64, 1024])
@pytest.mark.parametrize("K_N", _FUSED_SHAPES, ids=["qkv", "down"])
@pytest.mark.parametrize("with_bias", [False, True])
@needs_int_mm
def test_bitexact_fused_shapes_row_sampled(M, K_N, with_bias):
    K, N = K_N
    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=with_bias
    )

    actual = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )
    rows = _row_indices(M)
    expected = _cpu_oracle_rows(
        x_q, weight, x_scale, weight_scale, bias, torch.bfloat16, rows
    ).cuda()

    assert actual.shape == (M, N)
    assert torch.equal(actual[rows], expected)


@needs_int_mm
@pytest.mark.parametrize("M", [1, 5, 64])
def test_matches_w8a8_linear_reference_end_to_end(M):
    """Full-chain match: same quant outputs fed to the portable reference."""
    K, N = 2048, 2048
    gen = torch.Generator().manual_seed(7)
    x = (torch.randn(M, K, generator=gen) * 2).to(torch.bfloat16)
    weight_fp = torch.randn(N, K, generator=gen) * 0.02
    weight_scale = (weight_fp.abs().amax(dim=1) / 127).clamp_min(1e-12)
    checkpoint_weight = (
        (weight_fp / weight_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    )
    bias = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)

    expected = w8a8_linear_reference(
        x, checkpoint_weight, weight_scale.reshape(-1, 1), bias
    )

    x_q, x_scale = dynamic_per_token_quant_int8(x)
    actual = int8_scaled_mm_int_mm(
        x_q.cuda(),
        checkpoint_weight.t().contiguous().cuda(),
        x_scale.cuda(),
        weight_scale.cuda(),
        bias=bias.cuda(),
        out_dtype=torch.bfloat16,
    )

    assert actual.shape == expected.shape
    assert torch.equal(actual.cpu(), expected)


@needs_int_mm
def test_output_dtype_and_default():
    x_q, x_scale, weight, weight_scale, _ = _make_operands(
        5, 2048, 2560, with_bias=False
    )

    out_f32 = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.float32
    )
    out_default = int8_scaled_mm_int_mm(x_q, weight, x_scale, weight_scale)

    assert out_f32.dtype == torch.float32
    assert out_default.dtype == torch.bfloat16
    assert torch.equal(out_f32.to(torch.bfloat16), out_default)


@needs_int_mm
def test_pad_buffer_reuse_is_stateless_across_calls():
    K, N = 2048, 2048
    _PAD_BUFS.clear()
    operands = {
        M: _make_operands(M, K, N, with_bias=False, seed=M) for M in (3, 5)
    }

    def _run(M):
        x_q, x_scale, weight, weight_scale, _ = operands[M]
        return int8_scaled_mm_int_mm(x_q, weight, x_scale, weight_scale)

    first = _run(3)
    _run(5)
    third = _run(3)

    assert torch.equal(first, third)
    buf = _PAD_BUFS[(torch.cuda.current_device(), K)]
    assert buf.shape == (_MIN_ROWS, K)
    assert buf.dtype == torch.int8
    # Tail rows must stay zero: they are never written and feed the pad GEMM.
    assert torch.equal(
        buf[5:].to(torch.int32), torch.zeros_like(buf[5:], dtype=torch.int32)
    )


def test_validates_operand_contract():
    weight = torch.ones(8, 3, dtype=torch.int8)
    weight_scale = torch.ones(3, dtype=torch.float32)
    x_scale = torch.ones(2, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="x_q must be 2D int8"):
        int8_scaled_mm_int_mm(
            torch.ones(2, dtype=torch.int8), weight, x_scale, weight_scale
        )
    with pytest.raises(ValueError, match="weight must be 2D int8"):
        int8_scaled_mm_int_mm(
            torch.ones(2, 8, dtype=torch.int8),
            torch.ones(8, 3, dtype=torch.bfloat16),
            x_scale,
            weight_scale,
        )
    with pytest.raises(ValueError, match="K mismatch"):
        int8_scaled_mm_int_mm(
            torch.ones(2, 16, dtype=torch.int8), weight, x_scale, weight_scale
        )
    with pytest.raises(ValueError, match="multiple of 8"):
        int8_scaled_mm_int_mm(
            torch.ones(2, 12, dtype=torch.int8),
            torch.ones(12, 3, dtype=torch.int8),
            x_scale,
            weight_scale,
        )
    with pytest.raises(ValueError, match="x_scale must hold"):
        int8_scaled_mm_int_mm(
            torch.ones(2, 8, dtype=torch.int8),
            weight,
            torch.ones(3, 1, dtype=torch.float32),
            weight_scale,
        )
    with pytest.raises(ValueError, match="weight_scale must hold"):
        int8_scaled_mm_int_mm(
            torch.ones(2, 8, dtype=torch.int8),
            weight,
            x_scale,
            torch.ones(4, dtype=torch.float32),
        )
