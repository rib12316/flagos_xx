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

from vllm_fl.dispatch import (
    CachedOp,
    get_default_manager,
    reset_default_manager,
    reset_global_policy,
)
from vllm_fl.quantization.w8a8.int_mm import (
    _MIN_ROWS,
    _PAD_BUFS,
    _intmm_tn_layout,
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

# 融合档表在 MetaX 栈被清空（int_mm.py：smem 67584B>65536B 超限 + mctlass
# 承载主路径）——warm-key 语义在沐曦无载体。
IS_METAX = "metax" in torch.__version__.lower()
DEFAULT_W8A8_IMPL = "default.flagos_mctlass" if IS_METAX else "default.flagos_intmm"
needs_fused_tables = pytest.mark.skipif(
    IS_METAX,
    reason="fused GEMM tables are empty on the MetaX stack (mctlass path)",
)

# CUDA integer matmul is unavailable on this stack ("addmm_cuda" not
# implemented for Int), so the bit-exact oracle runs the int32 matmul on
# CPU. Output row i depends only on x_q[i] / x_scale[i] / weight, so
# fused-shape cases compare a sampled row subset against per-row CPU
# matmuls to keep the oracle cheap.
_FUSED_SHAPES = [(2048, 2560), (6144, 2048)]


def _make_operands(M, K, N, *, with_bias, seed=0, tn=None):
    """Deterministic per-token int8 x and per-channel int8 weight.

    ``tn`` picks the storage layout handed to the impl ([N, K] row-major vs
    the legacy [K, N] transpose); None follows the active env layout, so
    unparametrized tests track the production default.
    """
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
    if tn is None:
        tn = _intmm_tn_layout()
    stored = weight.contiguous() if tn else weight.t().contiguous()
    return x_q, x_scale, stored, weight_scale, bias


def _cpu_oracle_rows(x_q, weight, x_scale, weight_scale, bias, out_dtype, rows):
    """Exact int32 CPU matmul for selected output rows + emu epilogue order."""
    w = weight.cpu().to(torch.int32)
    if _intmm_tn_layout():
        w = w.t().contiguous()  # storage [N, K] -> logical [K, N]
    selected = x_q[rows].cpu().to(torch.int32)
    accumulator = selected @ w
    y = accumulator.to(torch.float32)
    y = y * x_scale[rows].cpu().reshape(-1, 1)
    y = y * weight_scale.cpu().reshape(1, -1)
    if bias is not None:
        y = y + bias.cpu()
    return y.to(out_dtype)


def _row_indices(M):
    return sorted({0, M // 2, M - 1, min(M - 1, 3)})


def _assert_bf16_matches_reference(actual, expected, bias=None):
    """torch.equal, except the bias path of the fused epilogue.

    The fused kernel's mul+bias-add can be FMA-contracted by the compiler
    (single rounding vs the torch chain's two roundings), showing up as a
    rare 1-ulp bf16 difference at cancellation points (~1e-5 of elements).
    The model's quantized projections are bias-free, where the fused path
    stays bit-identical; tolerance covers <=2 ulp of the operand magnitude.
    """
    if torch.equal(actual, expected):
        return
    assert bias is not None, "no-bias path must be bit-identical"
    mag = expected.float().abs() + bias.float().abs().reshape(1, -1)
    diff = (actual.float() - expected.float()).abs()
    assert (diff <= mag * 2**-7 + 1e-30).all(), (
        f"bias-path mismatch beyond FMA tolerance: "
        f"max diff {(diff / mag.clamp_min(1e-30)).max().item():.3e} rel"
    )


@pytest.mark.parametrize("layout", ["tn", "kn"], ids=["TN", "KN"])
@pytest.mark.parametrize("M", [1, 2, 5, 16, 17, 64])
@pytest.mark.parametrize(
    "K_N",
    [(64, 32), (128, 256), (72, 48)],
    ids=["k64n32", "k128n256", "k72n48"],
)
@pytest.mark.parametrize("with_bias", [False, True])
@needs_int_mm
def test_bitexact_small_shapes_vs_cpu_int32(monkeypatch, layout, M, K_N, with_bias):
    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1" if layout == "tn" else "0")
    K, N = K_N
    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=with_bias, tn=layout == "tn"
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
    _assert_bf16_matches_reference(actual, expected, bias)


@pytest.mark.parametrize("layout", ["tn", "kn"], ids=["TN", "KN"])
@pytest.mark.parametrize("M", [1, 16, 17, 64, 1024])
@pytest.mark.parametrize("K_N", _FUSED_SHAPES, ids=["qkv", "down"])
@pytest.mark.parametrize("with_bias", [False, True])
@needs_int_mm
def test_bitexact_fused_shapes_row_sampled(monkeypatch, layout, M, K_N, with_bias):
    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1" if layout == "tn" else "0")
    K, N = K_N
    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=with_bias, tn=layout == "tn"
    )

    actual = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )
    rows = _row_indices(M)
    expected = _cpu_oracle_rows(
        x_q, weight, x_scale, weight_scale, bias, torch.bfloat16, rows
    ).cuda()

    assert actual.shape == (M, N)
    _assert_bf16_matches_reference(actual[rows], expected, bias)


@needs_int_mm
@pytest.mark.parametrize("layout", ["tn", "kn"], ids=["TN", "KN"])
@pytest.mark.parametrize("M", [1, 5, 64])
def test_matches_w8a8_linear_reference_end_to_end(monkeypatch, layout, M):
    """Full-chain match: same quant outputs fed to the portable reference."""
    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1" if layout == "tn" else "0")
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
    stored = (
        checkpoint_weight.contiguous()
        if layout == "tn"
        else checkpoint_weight.t().contiguous()
    )
    actual = int8_scaled_mm_int_mm(
        x_q.cuda(),
        stored.cuda(),
        x_scale.cuda(),
        weight_scale.cuda(),
        bias=bias.cuda(),
        out_dtype=torch.bfloat16,
    )

    assert actual.shape == expected.shape
    _assert_bf16_matches_reference(actual.cpu(), expected, bias)


@needs_int_mm
@pytest.mark.parametrize("M", [1, 5, 64, 1024])
@pytest.mark.parametrize("K_N", _FUSED_SHAPES, ids=["qkv", "down"])
@pytest.mark.parametrize("with_bias", [False, True])
def test_fused_epilogue_bitidentical_to_torch_chain(
    monkeypatch, M, K_N, with_bias
):
    """VLLM_FL_INTMM_FUSED_EPILOGUE toggles two paths that must be bit-equal."""
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    # Chain-internal toggle: keep the ixformer route out so both arms share
    # one GEMM and only the epilogue differs (ix has its own bit-exact test).
    monkeypatch.setenv("VLLM_FL_INTMM_IX", "0")
    K, N = K_N
    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=with_bias
    )

    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_EPILOGUE", "1")
    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_GEMM", "0")
    fused = int_mm_mod.int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )
    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_EPILOGUE", "0")
    chain = int_mm_mod.int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias, out_dtype=torch.bfloat16
    )

    _assert_bf16_matches_reference(fused, chain, bias)


@pytest.mark.parametrize("layout", ["tn", "kn"], ids=["TN", "KN"])
@pytest.mark.parametrize("M", [17, 64, 1024, 2050])
@pytest.mark.parametrize(
    "K_N",
    [(2048, 2560), (6144, 2048), (2048, 12288), (2048, 2048)],
    ids=["qkv", "down", "gate_up", "o"],
)
@pytest.mark.parametrize("with_bias", [False, True])
@needs_int_mm
def test_fused_gemm_bitidentical_to_cublas_chain(
    monkeypatch, layout, M, K_N, with_bias
):
    """VLLM_FL_INTMM_FUSED_GEMM toggles fused-kernel vs cuBLAS-chain paths.

    The fused kernel must reproduce the cuBLAS chain bit-for-bit on the
    no-bias path (model quantized projections are bias-free); M=2050
    exercises the chunk-tail M masking (2050 % 256 != 0, 2050 % 128 != 0).
    The ixformer route is disabled so the Triton tier is what gets toggled
    (it has its own bit-exact test below).
    """
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1" if layout == "tn" else "0")
    monkeypatch.setenv("VLLM_FL_INTMM_IX", "0")
    K, N = K_N
    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=with_bias, tn=layout == "tn"
    )

    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_GEMM", "1")
    fused = int_mm_mod.int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias,
        out_dtype=torch.bfloat16,
    )
    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_GEMM", "0")
    chain = int_mm_mod.int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, bias=bias,
        out_dtype=torch.bfloat16,
    )

    assert fused.shape == (M, N)
    _assert_bf16_matches_reference(fused, chain, bias)


_IX_SHAPES = [(2048, 2560, "qkv"), (6144, 2048, "down"),
              (2048, 12288, "gate_up"), (2048, 2048, "o")]


@pytest.mark.parametrize("M", [256, 1024, 2048, 257, 2047])
@pytest.mark.parametrize("K_N", [s[:2] for s in _IX_SHAPES],
                         ids=[s[2] for s in _IX_SHAPES])
@needs_int_mm
def test_ixformer_tn_route_bitexact_and_gated(monkeypatch, M, K_N):
    """The ixformer TN vendor route: bit-exact vs the CPU oracle on table
    shapes (including odd chunk tails 257/2047), actually dispatched to
    w8a8_gemm, and VLLM_FL_INTMM_IX=0 removes it without changing results."""
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    if int_mm_mod._v150_prefill_gemm(torch.device("cuda")) is None:
        pytest.skip("ixformer w8a8_gemm unavailable on this stack")
    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1")
    K, N = K_N

    calls = []
    real_resolver = int_mm_mod._v150_prefill_gemm

    def spy_resolver(device):
        gemm = real_resolver(device)

        def wrapped(*args, **kwargs):
            calls.append(1)
            return gemm(*args, **kwargs)

        return wrapped if gemm is not None else None

    monkeypatch.setattr(int_mm_mod, "_v150_prefill_gemm", spy_resolver)

    x_q, x_scale, weight, weight_scale, bias = _make_operands(
        M, K, N, with_bias=False, tn=True
    )
    actual = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.bfloat16
    )
    assert calls, f"M={M} did not route to ixformer w8a8_gemm"
    rows = _row_indices(M)
    expected = _cpu_oracle_rows(
        x_q, weight, x_scale, weight_scale, None, torch.bfloat16, rows
    ).cuda()
    assert torch.equal(actual[rows], expected)

    calls.clear()
    monkeypatch.setenv("VLLM_FL_INTMM_IX", "0")
    actual2 = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.bfloat16
    )
    assert not calls, "VLLM_FL_INTMM_IX=0 must disable the vendor route"
    _assert_bf16_matches_reference(actual2, actual, None)


_DECODE_SHAPES = [
    (2048, 2560, "qkv"),
    (6144, 2048, "down"),
    (2048, 12288, "gate_up"),
    (2048, 2048, "o"),
    (2048, 130560, "lm_head"),
]


def _raw_operands(M, K, N, seed=0, tn=None):
    """Arbitrary int8 operands — enough for fused-vs-chain bit-identity
    (the GEMM contract), without the fp32 [N, K] CPU oracle tensors that
    make _make_operands heavy at lm_head's N=130560."""
    gen = torch.Generator().manual_seed(seed)
    x_q = torch.randint(
        -127, 128, (M, K), generator=gen, dtype=torch.int8
    ).cuda()
    weight = torch.randint(
        -127, 128, (N, K), generator=gen, dtype=torch.int8
    ).cuda()
    if tn is None:
        tn = _intmm_tn_layout()
    if not tn:
        weight = weight.t().contiguous()
    xs = (torch.rand(M, generator=gen) / 8 + 1e-4).cuda()
    ws = (torch.rand(N, generator=gen) / 8 + 1e-4).cuda()
    return x_q, xs, weight, ws


@pytest.mark.parametrize("layout", ["tn", "kn"], ids=["TN", "KN"])
@pytest.mark.parametrize("M", [64, 128])
@pytest.mark.parametrize("K_N", [s[:2] for s in _DECODE_SHAPES],
                         ids=[s[2] for s in _DECODE_SHAPES])
@pytest.mark.parametrize("with_bias", [False, True])
@needs_int_mm
def test_decode_fused_bitidentical_to_cublas_chain(
    monkeypatch, layout, M, K_N, with_bias
):
    """VLLM_FL_INTMM_FUSED_DECODE toggles decode-tier fused vs cuBLAS chain
    at the exact-M table points; must stay bit-equal on the bias-free path.
    lm_head@M=64 is deliberately absent from the table (1.106x < gate) —
    both arms take the chain there, a trivial pass that documents intent."""
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1" if layout == "tn" else "0")
    K, N = K_N
    x_q, xs, weight, ws = _raw_operands(M, K, N, seed=M, tn=layout == "tn")
    bias = None
    if with_bias:
        gen = torch.Generator().manual_seed(M + 1)
        bias = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16).cuda()

    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_DECODE", "1")
    fused = int_mm_mod.int8_scaled_mm_int_mm(
        x_q, weight, xs, ws, bias=bias, out_dtype=torch.bfloat16
    )
    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_DECODE", "0")
    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_GEMM", "0")
    chain = int_mm_mod.int8_scaled_mm_int_mm(
        x_q, weight, xs, ws, bias=bias, out_dtype=torch.bfloat16
    )

    assert fused.shape == (M, N)
    _assert_bf16_matches_reference(fused, chain, bias)


@needs_int_mm
@needs_fused_tables
def test_decode_tier_routes_exact_m_only(monkeypatch):
    """The decode fused tier must fire only at exact M=64/128 on table
    shapes; every other decode size (17/32/72/96/255 — the cudagraph
    capture grid around them) keeps the cuBLAS chain. Spied via the shared
    launch body."""
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    K, N = 2048, 2560
    real_launch = int_mm_mod._launch_fused_gemm
    calls = []

    def spy(x_q, weight, x_scale, weight_scale, bias, M, N_, K_, cfg):
        calls.append((M, cfg))
        return real_launch(
            x_q, weight, x_scale, weight_scale, bias, M, N_, K_, cfg
        )

    monkeypatch.setattr(int_mm_mod, "_launch_fused_gemm", spy)
    for M in (17, 32, 72, 96, 255):
        x_q, xs, weight, ws = _raw_operands(M, K, N, seed=M)
        out = int_mm_mod.int8_scaled_mm_int_mm(
            x_q, weight, xs, ws, out_dtype=torch.bfloat16
        )
        assert out.shape == (M, N)
        assert all(c[0] != M for c in calls), (
            f"M={M} reached the fused launch; decode tier must be exact-M"
        )
    for M in (64, 128):
        calls.clear()
        x_q, xs, weight, ws = _raw_operands(M, K, N, seed=M)
        int_mm_mod.int8_scaled_mm_int_mm(
            x_q, weight, xs, ws, out_dtype=torch.bfloat16
        )
        want_cfg = int_mm_mod._FUSED_GEMM_DECODE_CONFIGS[(M, K, N)]
        assert (M, want_cfg) in calls, (
            f"M={M} did not route to the decode-tier config"
        )


@needs_int_mm
@needs_fused_tables
def test_warm_key_distinguishes_decode_and_prefill_cfg(monkeypatch):
    """A warmed prefill specialization must not green-light an unwarmed
    decode tile inside a cudagraph capture: Triton JIT compiles carry
    device-side syncs, so the warm key carries the config tuple."""
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    # This is about the Triton tier's warm bookkeeping — keep the ixformer
    # vendor route from swallowing the M=1024 prefill call.
    monkeypatch.setenv("VLLM_FL_INTMM_IX", "0")
    K, N = 2048, 2560
    int_mm_mod._FUSED_WARMED.clear()
    try:
        x_q, xs, weight, ws = _raw_operands(1024, K, N, seed=1)
        int_mm_mod.int8_scaled_mm_int_mm(
            x_q, weight, xs, ws, out_dtype=torch.bfloat16
        )
        prefill_cfg = int_mm_mod._FUSED_GEMM_CONFIGS[(K, N)]
        assert (K, N, True, prefill_cfg) in int_mm_mod._FUSED_WARMED
        decode_cfg = int_mm_mod._FUSED_GEMM_DECODE_CONFIGS[(64, K, N)]
        assert decode_cfg != prefill_cfg
        assert (K, N, True, decode_cfg) not in int_mm_mod._FUSED_WARMED

        # Behavior: an M=64 call inside a fresh capture (decode tile never
        # warmed) must fall back to the cuBLAS chain instead of compiling
        # mid-capture, and the replayed graph must match the eager chain.
        M = 64
        x_q, xs, weight, ws = _raw_operands(M, K, N, seed=2)
        eager = int_mm_mod.int8_scaled_mm_int_mm(
            x_q, weight, xs, ws, out_dtype=torch.bfloat16
        )
        int_mm_mod._FUSED_WARMED.discard((K, N, True, decode_cfg))

        def fn():
            return int_mm_mod.int8_scaled_mm_int_mm(
                x_q, weight, xs, ws, out_dtype=torch.bfloat16
            )

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            fn()  # stream warm-up of the chain kernels
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = fn()
        torch.cuda.synchronize()
        g.replay()  # capture only records; replay executes into `captured`
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)
    finally:
        int_mm_mod._FUSED_WARMED.clear()


@needs_int_mm
def test_fused_gemm_survives_fullgraph_compile():
    """The fused branch hides inside the custom op boundary — dynamo must
    stay able to fullgraph-compile across M (AOT bakes a large-M trace, then
    capture/run-time M must still route correctly), mirroring the serve
    invariant that first kept the M<17 padding inside the custom op."""
    from vllm_fl.quantization.w8a8 import int_mm as int_mm_mod

    K, N = 2048, 2560  # fused config-table shape
    gen = torch.Generator().manual_seed(11)
    w8_shape = (N, K) if _intmm_tn_layout() else (K, N)
    w8 = torch.randint(
        -127, 128, w8_shape, generator=gen, dtype=torch.int8
    ).cuda()
    ws = (torch.rand(N, generator=gen) / 8 + 1e-4).cuda()

    def fn(x_q, xs):
        return int_mm_mod.int8_scaled_mm_int_mm(
            x_q, w8, xs, ws, out_dtype=torch.bfloat16
        )

    for M in (64, 128, 2050):
        gen_m = torch.Generator().manual_seed(M)
        x_q = torch.randint(
            -127, 128, (M, K), generator=gen_m, dtype=torch.int8
        ).cuda()
        xs = (torch.rand(M, 1, generator=gen_m) / 8 + 1e-4).cuda()
        compiled = torch.compile(fn, fullgraph=True, dynamic=False)
        out = compiled(x_q, xs)
        assert out.shape == (M, N)
        assert torch.equal(out, fn(x_q, xs))


@needs_int_mm
def test_bare_quant_kernel_bitidentical_to_reference():
    """The bare (non-libentry) quant impl must match the portable reference."""
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_bare,
    )

    for M, K in [(1, 2048), (5, 6144), (64, 2048), (128, 6144)]:
        gen = torch.Generator().manual_seed(M * 1000 + K)
        x = (torch.randn(M, K, generator=gen) * 3).to(torch.bfloat16).cuda()

        q, s = dynamic_per_token_quant_int8_bare(x.contiguous())
        ref_q, ref_s = dynamic_per_token_quant_int8(x)

        assert q.shape == (M, K) and q.dtype == torch.int8
        assert s.shape == (M, 1) and s.dtype == torch.float32
        assert torch.equal(q, ref_q)
        assert torch.equal(s, ref_s)


@needs_int_mm
def test_cachedop_quant_resolves_to_fast_impl():
    """Quant must not land on the libentry impl in this environment."""
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)
    quant = CachedOp("dynamic_per_token_quant_int8")

    q, s = quant(x)

    impl_id = get_default_manager().get_selected_impl_id(
        "dynamic_per_token_quant_int8"
    )
    assert impl_id in ("default.flagos", "default.flagos_bare")
    # The impl cache must stay populated across calls: a call-time failure of
    # the top impl would stick this CachedOp on the manager slow path
    # (~0.6ms overhead per call) for the life of the process.
    quant(x)
    assert quant._impl is not None
    assert quant._impl.impl_id == impl_id
    ref_q, ref_s = dynamic_per_token_quant_int8(x)
    assert torch.equal(q, ref_q)
    assert torch.equal(s, ref_s)


@needs_int_mm
def test_pre_resolved_ops_survive_fullgraph_compile():
    """vLLM AOT-compiles the model forward; an un-resolved CachedOp inside
    dynamo enters locks it cannot trace (EngineCore startup failure on the
    W8A8 path). The serve invariant: pre_resolve() at registration time
    makes the whole chain fullgraph-traceable."""
    quant = CachedOp("dynamic_per_token_quant_int8")
    mm = CachedOp("w8a8_dynamic_per_token_linear")
    assert quant.pre_resolve() == "default.flagos_bare"
    assert mm.pre_resolve() == DEFAULT_W8A8_IMPL

    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    # K=64,N=32；沐曦布局契约为 [N,K]（mctlass 承载，见 w8a8/mctlass_mm.py），
    # V150 TN 布局同为此约定（_intmm_tn_layout）。
    w8_shape = (32, 64) if (IS_METAX or _intmm_tn_layout()) else (64, 32)
    w8 = torch.randint(-127, 128, w8_shape, device="cuda", dtype=torch.int8)
    ws = torch.rand(32, device="cuda")

    def fn(x):
        q, s = quant(x)
        return mm(q, w8, s, ws, out_dtype=torch.bfloat16)

    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    out = compiled(x)
    assert out.shape == (4, 32)
    assert torch.equal(out, fn(x))


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


@pytest.mark.parametrize("layout", ["tn", "kn"], ids=["TN", "KN"])
def test_validates_operand_contract(monkeypatch, layout):
    monkeypatch.setenv("VLLM_FL_INTMM_TN", "1" if layout == "tn" else "0")

    def stored(k, n, dtype=torch.int8):
        ones = torch.ones(n, k, dtype=dtype) if layout == "tn" else torch.ones(
            k, n, dtype=dtype
        )
        return ones

    weight = stored(8, 3)
    weight_scale = torch.ones(3, dtype=torch.float32)
    x_scale = torch.ones(2, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="x_q must be 2D int8"):
        int8_scaled_mm_int_mm(
            torch.ones(2, dtype=torch.int8), weight, x_scale, weight_scale
        )
    with pytest.raises(ValueError, match="weight must be 2D int8"):
        int8_scaled_mm_int_mm(
            torch.ones(2, 8, dtype=torch.int8),
            stored(8, 3, dtype=torch.bfloat16),
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
            stored(12, 3),
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


# ==================== dispatch integration ====================


@pytest.fixture(autouse=True)
def _reset_dispatch_state():
    reset_default_manager()
    reset_global_policy()
    yield
    reset_default_manager()
    reset_global_policy()


@needs_int_mm
def test_dispatch_resolves_intmm_impl_by_default():
    manager = get_default_manager()

    impl_id = manager.get_selected_impl_id("w8a8_dynamic_per_token_linear")

    assert impl_id == DEFAULT_W8A8_IMPL


@needs_int_mm
def test_dispatch_per_op_switch_to_flaggems_scaledmm(monkeypatch):
    pytest.importorskip("flag_gems")
    x_q, x_scale, weight, weight_scale, _ = _make_operands(
        64, 2048, 2560, with_bias=False
    )
    via_intmm = int8_scaled_mm_int_mm(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.bfloat16
    )

    monkeypatch.setenv(
        "VLLM_FL_PER_OP",
        "w8a8_dynamic_per_token_linear=impl:default.flagos_scaledmm",
    )
    reset_global_policy()
    reset_default_manager()
    manager = get_default_manager()

    assert (
        manager.get_selected_impl_id("w8a8_dynamic_per_token_linear")
        == "default.flagos_scaledmm"
    )
    via_scaledmm = CachedOp("w8a8_dynamic_per_token_linear")(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.bfloat16
    )
    # Different GEMM backends only need to agree within the int8 noise band.
    rel = (
        (via_scaledmm.float() - via_intmm.float()).abs()
        / via_intmm.float().abs().clamp_min(1e-3)
    )
    assert rel.median().item() <= 1e-2
    assert rel.quantile(0.99).item() <= 5e-2


@needs_int_mm
def test_dispatch_falls_back_to_scaledmm_when_intmm_fails(monkeypatch):
    pytest.importorskip("flag_gems")

    def broken_int_mm(*_args, **_kwargs):
        raise RuntimeError("simulated cuBLAS int8 failure")

    x_q, x_scale, weight, weight_scale, _ = _make_operands(
        64, 2048, 2560, with_bias=False
    )
    # Exercise the cuBLAS chain specifically: the fused single-kernel path
    # (on for config-table shapes like this one) does not call torch._int_mm
    # and would legitimately succeed past the simulated failure.
    monkeypatch.setenv("VLLM_FL_INTMM_FUSED_GEMM", "0")
    monkeypatch.setattr(torch, "_int_mm", broken_int_mm)

    out = CachedOp("w8a8_dynamic_per_token_linear")(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.bfloat16
    )

    assert out.shape == (64, 2560)
    assert torch.isfinite(out.float()).all()
    # The failing impl is marked failed for the process and later calls
    # keep going through the fallback (VLLM_FL_STRICT=0 default).
    manager = get_default_manager()
    assert "default.flagos_intmm" in manager._failed_impls.get(
        "w8a8_dynamic_per_token_linear", set()
    )
    again = CachedOp("w8a8_dynamic_per_token_linear")(
        x_q, weight, x_scale, weight_scale, out_dtype=torch.bfloat16
    )
    assert torch.isfinite(again.float()).all()


@needs_int_mm
def test_apply_weights_end_to_end_through_dispatch(monkeypatch):
    from vllm_fl.quantization.w8a8.linear import FLW8A8DynamicLinearKernel

    K, N, M = 2048, 2048, 5
    gen = torch.Generator().manual_seed(11)
    x = (torch.randn(1, M, K, generator=gen) * 2).to(torch.bfloat16)
    weight_fp = torch.randn(N, K, generator=gen) * 0.02
    weight_scale = (weight_fp.abs().amax(dim=1) / 127).clamp_min(1e-12)
    checkpoint_weight = (
        (weight_fp / weight_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    )
    bias = (torch.randn(N, generator=gen) * 0.1).to(torch.bfloat16)

    expected = w8a8_linear_reference(
        x.reshape(M, K), checkpoint_weight, weight_scale.reshape(-1, 1), bias
    )

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                checkpoint_weight.cuda(), requires_grad=False
            )
            self.weight_scale = torch.nn.Parameter(
                weight_scale.to(torch.float32).cuda(), requires_grad=False
            )
            self.input_scale = None
            self.input_zero_point = None
            self.azp_adj = None

    layer = _Layer()
    kernel = object.__new__(FLW8A8DynamicLinearKernel)
    kernel.layer_param_names = [
        "weight",
        "weight_scale",
        "input_scale",
        "input_zero_point",
        "azp_adj",
    ]
    # 走真实 process_weights_after_loading：本测试同时覆盖 TN/MetaX/legacy
    # 三分支的存储布局决策（linear.py 单一事实源 _intmm_tn_layout）。
    kernel.process_weights_after_loading(layer)
    assert layer.weight.shape == (
        (N, K) if (IS_METAX or _intmm_tn_layout()) else (K, N)
    )
    monkeypatch.setattr(
        "vllm_fl.quantization.w8a8.linear._dynamic_per_token_quant_int8",
        lambda value: dynamic_per_token_quant_int8(value),
    )

    actual = kernel.apply_weights(layer, x.cuda(), bias.cuda())

    assert actual.shape == (1, M, N)
    assert actual.dtype == torch.bfloat16
    _assert_bf16_matches_reference(actual.reshape(M, N).cpu(), expected, bias)
