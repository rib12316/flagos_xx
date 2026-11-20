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
"""D13 §3.5 装载期静态配对接线单元测试（五件套第 4 件，随库走）。

完整 GPU 对拍/计时见 scripts/d13/t0_pair_wiring.py（CUDA graph 口径）。
"""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

import vllm_fl.quantization.w8a8.d13_norm_quant as d13
from vllm_fl.quantization.w8a8.d13_norm_quant import (
    _pairable,
    install_d13_pairings,
)


class _Proj(torch.nn.Module):
    """W8A8 per-token linear 的 duck-typing 外观（_pairable/apply 契约）。"""

    _fl_expects_prequant = False   # 真栈由 linear.py 类级默认提供

    def __init__(self, k, n, device="cuda"):
        super().__init__()
        w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * 0.02
        ws = (w.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
        self.weight = torch.nn.Parameter(
            (w.float() / ws[:, None]).round().clamp(-127, 127)
            .to(torch.int8).contiguous(), requires_grad=False)
        self.weight_scale = torch.nn.Parameter(
            ws.reshape(-1).float().contiguous(), requires_grad=False)
        self.quant_method = object()
        self.input_size = k


class _LlamaLikeDecoder(torch.nn.Module):
    def __init__(self, k, device):
        super().__init__()
        self.input_layernorm = RMSNorm(k, eps=1e-6).to(device)
        self.post_attention_layernorm = RMSNorm(k, eps=1e-6).to(device)
        self.self_attn = torch.nn.Module()
        self.self_attn.qkv_proj = _Proj(k, 2 * k, device)
        self.mlp = torch.nn.Module()
        self.mlp.gate_up_proj = _Proj(k, 4 * k, device)


def _model(k=64, layers=3, device="cuda"):
    m = torch.nn.Module()
    m.model = torch.nn.Module()
    m.model.layers = torch.nn.ModuleList(
        [_LlamaLikeDecoder(k, device) for _ in range(layers)])
    return m


@pytest.fixture()
def vc():
    with set_current_vllm_config(VllmConfig()):
        yield VllmConfig()


@pytest.fixture(autouse=True)
def _fake_layer_allowlist(monkeypatch):
    monkeypatch.setenv("VLLM_FL_FUSION_BUNDLE", "1")
    d13._SUPPORTED_DECODER_LAYERS.add("_LlamaLikeDecoder")
    yield
    d13._SUPPORTED_DECODER_LAYERS.discard("_LlamaLikeDecoder")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 MACA 设备")
class TestPairing:
    def test_pair_count_and_layer0_skip(self, vc, monkeypatch):
        monkeypatch.setenv("VLLM_FL_D13_M_FUSE_MAX", "64")
        m = _model()
        n = install_d13_pairings(m, vc)
        assert n == 5  # 3 层 × 2 − layer0 input_layernorm
        assert not getattr(
            m.model.layers[0].input_layernorm, "_fl_d13_paired", False)
        assert m.model.layers[1].input_layernorm._fl_d13_paired
        assert m.model.layers[2].mlp.gate_up_proj._fl_expects_prequant
        assert (m.model.layers[2].mlp.gate_up_proj._fl_prequant_norm
                is m.model.layers[2].post_attention_layernorm)
        # 侧信道 buffer 已注册为非持久 buffer
        assert m.model.layers[1].input_layernorm._fl_d13_scale.dtype \
            == torch.float32

    def test_idempotent(self, vc, monkeypatch):
        monkeypatch.setenv("VLLM_FL_D13_M_FUSE_MAX", "64")
        m = _model()
        assert install_d13_pairings(m, vc) == 5
        assert install_d13_pairings(m, vc) == 0

    def test_env_master_gate_off(self, vc, monkeypatch):
        monkeypatch.setenv("VLLM_FL_D13_NORM_QUANT", "0")
        m = _model()
        assert install_d13_pairings(m, vc) == 0
        assert not getattr(
            m.model.layers[1].input_layernorm, "_fl_d13_paired", False)

    def test_pairable_rejects_non_int8_weight(self, vc):
        norm = RMSNorm(64, eps=1e-6).cuda()
        lin = _Proj(64, 128)
        lin.weight = torch.nn.Parameter(
            torch.randn(128, 64, device="cuda", dtype=torch.bfloat16),
            requires_grad=False)
        assert not _pairable(norm, lin)

    def test_m_routing_boundary(self, vc, monkeypatch):
        monkeypatch.setenv("VLLM_FL_D13_M_FUSE_MAX", "64")
        m = _model()
        install_d13_pairings(m, vc)
        norm = m.model.layers[1].input_layernorm
        x64 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        r64 = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        y, _ = norm(x64, r64)
        assert y.dtype == torch.int8          # M = 阈值 → 融合
        x65 = torch.randn(65, 64, device="cuda", dtype=torch.bfloat16)
        r65 = torch.randn(65, 64, device="cuda", dtype=torch.bfloat16)
        y2, _ = norm(x65, r65)
        assert y2.dtype == torch.bfloat16     # M > 阈值 → 两步链
        y3 = norm(x64)                        # residual=None → 原版
        assert torch.is_tensor(y3) and y3.dtype == torch.bfloat16

    def test_fused_matches_two_step_bitwise(self, vc, monkeypatch):
        """融合路径 (q, scale, resid) 与 stock 链 bitwise（GEMM 输入级）。"""
        from vllm_fl.dispatch import CachedOp

        monkeypatch.setenv("VLLM_FL_D13_M_FUSE_MAX", "64")
        m = _model()
        install_d13_pairings(m, vc)
        norm = m.model.layers[1].post_attention_layernorm
        x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        r = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
        q, resid = norm(x.clone(), r.clone())
        q_ref, s_ref, resid_ref = CachedOp(
            "fused_add_rms_norm_quant_int8")(
            x, r, norm.weight.data, 1e-6)
        assert torch.equal(q, q_ref)
        assert torch.equal(resid, resid_ref)
        assert torch.equal(norm._fl_d13_scale[: 64], s_ref)
