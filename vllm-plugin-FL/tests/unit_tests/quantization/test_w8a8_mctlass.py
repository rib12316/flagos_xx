# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
"""MetaX C500 mctlassEx W8A8 GEMM：数值 parity / 布局契约 / 捕获安全 / 路由。

真机命令（沐曦实例）：
    python3 -m pytest vllm-plugin-FL/tests/unit_tests/quantization/ \
        test_w8a8_mctlass.py -x -q
"""

import pytest
import torch

from vllm_fl.quantization.w8a8.mctlass_mm import (
    int8_scaled_mm_mctlass,
    mctlass_available,
    metax_mctlass_w8a8,
)

needs_mctlass = pytest.mark.skipif(
    not (torch.cuda.is_available() and mctlass_available()),
    reason="requires CUDA and the image-shipped mctlassEx",
)


def _operands(M, K, N, seed=0, bias=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x_q = torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8, generator=g)
    # 沐曦布局契约：weight [N,K] row-major（process 层跳过转置）
    w = torch.randint(-127, 127, (N, K), device="cuda", dtype=torch.int8, generator=g)
    xs = torch.rand((M, 1), device="cuda", dtype=torch.float32, generator=g) * 0.01
    ws = torch.rand((N,), device="cuda", dtype=torch.float32, generator=g) * 0.01
    b = None
    if bias:
        b = torch.randn(N, device="cuda", dtype=torch.bfloat16, generator=g) * 0.1
    return x_q, w, xs, ws, b


@pytest.mark.parametrize(
    "M,K,N",
    [
        (1, 2048, 2560),
        (8, 2048, 2560),
        (30, 2048, 2560),
        (64, 2048, 2560),
        (2048, 2048, 2560),
        (64, 2048, 130560),  # lm_head
        (17, 6144, 2048),   # 奇数 M + 第二类形状
    ],
)
@needs_mctlass
def test_mctlass_parity_vs_fp32(M, K, N):
    x_q, w, xs, ws, _ = _operands(M, K, N, seed=42)
    out = int8_scaled_mm_mctlass(x_q, w, xs, ws, out_dtype=torch.bfloat16)
    assert out.shape == (M, N) and out.dtype == torch.bfloat16
    ref = (x_q.float() @ w.float().t()) * (xs @ ws.reshape(1, -1))
    denom = max(ref.abs().max().item(), 1e-6)
    rel = (out.float() - ref).abs().max().item() / denom
    assert rel < 5e-2, f"maxrel={rel}"  # int8 量化噪声 + bf16 舍入的宽容界


@needs_mctlass
def test_mctlass_bias_and_layout_guard():
    M, K, N = 32, 2048, 2560
    x_q, w, xs, ws, b = _operands(M, K, N, seed=7, bias=True)
    out = int8_scaled_mm_mctlass(x_q, w, xs, ws, bias=b)
    ref = (x_q.float() @ w.float().t()) * (xs @ ws.reshape(1, -1)) + b.float()
    denom = max(ref.abs().max().item(), 1e-6)
    rel = (out.float() - ref).abs().max().item() / denom
    assert rel < 5e-2
    # [K,N] 误传必须报错（布局契约防线）
    with pytest.raises(ValueError):
        int8_scaled_mm_mctlass(x_q, w.t().contiguous(), xs, ws)


@needs_mctlass
def test_mctlass_cudagraph_capture_replay():
    M, K, N = 64, 2048, 2560
    x_q, w, xs, ws, _ = _operands(M, K, N, seed=9)
    out = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)

    def call():
        result = int8_scaled_mm_mctlass(x_q, w, xs, ws)
        out.copy_(result)

    call()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        call()
    x_q.copy_(torch.randint(-127, 127, (M, K), device="cuda", dtype=torch.int8))
    g.replay()
    torch.cuda.synchronize()
    ref = (x_q.float() @ w.float().t()) * (xs @ ws.reshape(1, -1))
    rel = (out.float() - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert rel < 5e-2


@needs_mctlass
def test_mctlass_wins_dispatch_resolution():
    """metax 栈上 w8a8 op 必须解析到 flagos_mctlass（prio DEFAULT+20）。"""
    from vllm_fl.dispatch import get_default_manager

    if not metax_mctlass_w8a8():
        pytest.skip("mctlass runtime gate off")
    impl = get_default_manager().resolve("w8a8_dynamic_per_token_linear")
    assert "mctlass" in getattr(impl, "__qualname__", ""), impl
