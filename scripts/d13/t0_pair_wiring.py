# D13 接线 T0（§3.5 装载期静态配对）验证：模拟 Llama 结构的 norm→proj 数据流。
#
# 验证六层（对应开发单 §0 五件套第 3/4 件 + §3.5 契约）：
#   1. dispatch 路由 smoke：op 解析到 default.flagos_bare
#   2. install_d13_pairings 配对数与 layer0 跳过（3 层 × 2 − layer0 前norm = 5）
#   3. 融合全链 vs stock 两步全链：GEMM 输出 bitwise 一致 + 侧信道 scale
#      buffer 内容 bitwise + residual bitwise
#   4. M 路由：M≤m_fuse_max 返回 int8 q（融合）；M>m_fuse_max 返回 bf16（两步）
#      residual=None 走原版
#   5. 幂等：二次 install 配对数 0；env 总门 =0 时零配对零行为变化
#   6. 计时（CUDA graph 口径，铁律 2）：融合 vs 两步全链
# 用法：python3 scripts/d13/t0_pair_wiring.py
import functools
import os
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
# 路由测试用小阈值：M=64 融合 / M>64 两步（默认公式另在报告里打印）
os.environ["VLLM_FL_D13_M_FUSE_MAX"] = "64"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

sys.path.insert(0, "scripts/d2")
import bottleneck_probe as bp  # noqa: E402

from vllm.config import VllmConfig, set_current_vllm_config  # noqa: E402
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: E402

from vllm_fl.dispatch import get_default_manager  # noqa: E402
from vllm_fl.quantization.w8a8.d13_norm_quant import (  # noqa: E402
    install_d13_pairings,
)
from vllm_fl.quantization.w8a8.linear import (  # noqa: E402
    FLW8A8DynamicLinearKernel,
)

DEV = "cuda"
K = 2048          # MiniCPM5-2B hidden
N_QKV = 3072
N_GATE_UP = 8192
NLAYER = 3


class FakeProj(nn.Module):
    """模拟 vllm LinearBase 的 W8A8 外观（apply_weights/_pairable 契约）。"""

    _fl_expects_prequant = False   # 真栈由 linear.py 类级默认提供

    def __init__(self, k, n):
        super().__init__()
        w = torch.randn(n, k, device=DEV, dtype=torch.bfloat16) * 0.02
        ws = (w.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
        wq = (w.float() / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
        self.weight = nn.Parameter(wq.contiguous(), requires_grad=False)
        self.weight_scale = nn.Parameter(
            ws.reshape(-1).to(torch.float32).contiguous(), requires_grad=False)
        self.quant_method = object()   # _pairable duck-typing
        self.input_size = k
        from vllm.model_executor.kernels.linear import (
            Int8ScaledMMLinearLayerConfig,
        )
        self.kernel = FLW8A8DynamicLinearKernel(
            Int8ScaledMMLinearLayerConfig(
                is_static_input_scheme=False,  # 动态 per-token 激活
                is_channelwise=True,           # per-channel 权重 scale
                input_symmetric=True,
            ),
            layer_param_names=[
                "weight", "weight_scale", "input_scale",
                "input_zero_point", "azp_adj"],
        )
        self.kernel.process_weights_after_loading(self)

    def forward(self, x):
        return self.kernel.apply_weights(self, x)


class FakeDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layernorm = RMSNorm(K, eps=1e-6).to(DEV)
        self.post_attention_layernorm = RMSNorm(K, eps=1e-6).to(DEV)
        self.self_attn = nn.Module()
        self.self_attn.qkv_proj = FakeProj(K, N_QKV)
        self.mlp = nn.Module()
        self.mlp.gate_up_proj = FakeProj(K, N_GATE_UP)
        # 命名复刻 Llama（install 按 named_modules 路径末段判 layer idx）
        # 外层用 ModuleList 命名 layers.<i> 即可。


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            [FakeDecoderLayer() for _ in range(NLAYER)])

    def forward(self, x, residual):
        # 复刻 llama.py decoder forward 数据流（直连，中间零算子；qkv 输出
        # 折回 hidden 用切片模拟 attention 投影——不影响 bitwise 对拍）
        for layer in self.model.layers:
            x, residual = layer.input_layernorm(x, residual)
            x = layer.self_attn.qkv_proj(x)[:, :K]
            x, residual = layer.post_attention_layernorm(x, residual)
            x = layer.mlp.gate_up_proj(x)[:, :K]
        return x, residual


def _two_step_triton_forward(norm, orig, x, residual=None):
    """stock 臂 = serve inductor 执行形态：同数学 norm-only triton kernel +
    linear 内部现役 bare quant（native C++ fused_add 与 triton 本征 6.6% 后端差，
    见 results/.../d13_norm_quant_fuse/README——不作位一致参照）。"""
    if residual is None:
        return orig(x, residual)
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from norm_quant_fuse_t0 import same_math_two_step

    _, _, resid_out, y = same_math_two_step(
        x.contiguous(), residual.contiguous(),
        norm.weight.data, norm.variance_epsilon)
    return y, resid_out


def _copy(dst, src):
    dst.input_layernorm.weight.data.copy_(src.input_layernorm.weight.data)
    dst.post_attention_layernorm.weight.data.copy_(
        src.post_attention_layernorm.weight.data)
    dst.self_attn.qkv_proj.weight.data.copy_(src.self_attn.qkv_proj.weight.data)
    dst.self_attn.qkv_proj.weight_scale.data.copy_(
        src.self_attn.qkv_proj.weight_scale.data)
    dst.mlp.gate_up_proj.weight.data.copy_(src.mlp.gate_up_proj.weight.data)
    dst.mlp.gate_up_proj.weight_scale.data.copy_(
        src.mlp.gate_up_proj.weight_scale.data)


def _run():
    torch.manual_seed(0)
    vc = VllmConfig()
    # Fake 结构入 allowlist（生产只认 LlamaDecoderLayer；测试结构等价）
    import vllm_fl.quantization.w8a8.d13_norm_quant as d13
    d13._SUPPORTED_DECODER_LAYERS.add("FakeDecoderLayer")

    # ---- 1. dispatch 路由 smoke ----
    impl = get_default_manager()._resolve_impl("fused_add_rms_norm_quant_int8")
    print(f"[route] impl_id={impl.impl_id} (expect default.flagos_bare)")
    assert impl.impl_id == "default.flagos_bare", impl.impl_id

    # ---- 2. 配对 ----
    fused = FakeModel()
    with set_current_vllm_config(vc):
        n = install_d13_pairings(fused, vc)
    print(f"[pair] paired={n} (expect {NLAYER * 2 - 1} = {NLAYER}层×2 "
          f"− layer0 input_layernorm)")
    assert n == NLAYER * 2 - 1, n
    assert not getattr(
        fused.model.layers[0].input_layernorm, "_fl_d13_paired", False
    ), "layer0 input_layernorm 不应配对"
    assert fused.model.layers[0].post_attention_layernorm._fl_d13_paired
    assert fused.model.layers[1].input_layernorm._fl_d13_paired

    # stock 对照（同权重、未配对；norm 换同数学 triton 两步口径，见上。
    # layer0 input_layernorm 两臂都保持原版——融合臂因 layer0 跳过配对走原生）
    stock = FakeModel()
    for li, (lf, ls) in enumerate(zip(fused.model.layers, stock.model.layers)):
        _copy(ls, lf)
        for attr in ("input_layernorm", "post_attention_layernorm"):
            if li == 0 and attr == "input_layernorm":
                continue
            norm = getattr(ls, attr)
            norm._forward_method = functools.partial(
                _two_step_triton_forward, norm, norm._forward_method)

    # ---- 3. bitwise 对拍（M=64 ≤ m_fuse_max=64，全走融合） ----
    M = 64
    x0 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 2
    r0 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    out_f, res_f = fused(x0.clone(), r0.clone())
    out_s, res_s = stock(x0.clone(), r0.clone())
    bit_out = torch.equal(out_f, out_s)
    bit_res = torch.equal(res_f, res_s)
    print(f"[bitwise] 全链输出 {'PASS' if bit_out else 'FAIL'} | "
          f"residual {'PASS' if bit_res else 'FAIL'}")
    assert bit_out and bit_res

    # 侧信道 scale buffer：直接驱动该站融合路径，buffer 内容须与现算一致
    from vllm_fl.dispatch import CachedOp
    norm1 = fused.model.layers[0].post_attention_layernorm
    _, res_direct = norm1(x0.clone(), r0.clone())     # 融合路径，写 buffer
    _, s_ref, res_ref = CachedOp("fused_add_rms_norm_quant_int8")(
        x0, r0, norm1.weight.data, 1e-6)
    buf = norm1._fl_d13_scale[:M]
    bit_scale = torch.equal(buf, s_ref) and torch.equal(res_direct, res_ref)
    print(f"[bitwise] 侧信道 scale buffer + resid {'PASS' if bit_scale else 'FAIL'}")
    assert bit_scale

    # ---- 4. M 路由 / residual=None ----
    x_big = torch.randn(4096, K, device=DEV, dtype=torch.bfloat16)
    r_big = torch.randn(4096, K, device=DEV, dtype=torch.bfloat16)
    y_big, _ = fused.model.layers[1].input_layernorm(x_big, r_big)
    ok_big = y_big.dtype == torch.bfloat16
    print(f"[route] M=4096 两步链回退: {'PASS' if ok_big else 'FAIL'} "
          f"(dtype={y_big.dtype})")
    assert ok_big
    y_mid, _ = fused.model.layers[1].input_layernorm(
        torch.randn(65, K, device=DEV, dtype=torch.bfloat16),
        torch.randn(65, K, device=DEV, dtype=torch.bfloat16))
    ok_mid = y_mid.dtype == torch.bfloat16
    print(f"[route] M=65(>64 边界) 两步链: {'PASS' if ok_mid else 'FAIL'}")
    assert ok_mid
    y_s, _ = fused.model.layers[1].input_layernorm(x0, r0)
    ok_s = y_s.dtype == torch.int8
    print(f"[route] M=64(=阈值) 融合: {'PASS' if ok_s else 'FAIL'}")
    assert ok_s
    y0 = fused.model.layers[1].input_layernorm(x0)   # residual=None → 原版
    ok_none = torch.is_tensor(y0) and y0.dtype == torch.bfloat16
    print(f"[route] residual=None 原版: {'PASS' if ok_none else 'FAIL'}")
    assert ok_none

    # ---- 5. 幂等 + env 总门 ----
    n2 = install_d13_pairings(fused, vc)
    print(f"[idempotent] 二次 install paired={n2} (expect 0)")
    assert n2 == 0
    os.environ["VLLM_FL_D13_NORM_QUANT"] = "0"
    try:
        gate = FakeModel()
        n3 = install_d13_pairings(gate, vc)
        print(f"[env-gate] 总门=0 paired={n3} (expect 0)")
        assert n3 == 0
        # 总门=0 → 全站原生路径：与「未配对未替换」的原生模型对拍（同权重）
        native = FakeModel()
        for lf, ln in zip(fused.model.layers, native.model.layers):
            _copy(ln, lf)
        for lf, lg in zip(fused.model.layers, gate.model.layers):
            _copy(lg, lf)
        xg = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        rg = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        assert torch.equal(gate(xg.clone(), rg.clone())[0],
                           native(xg.clone(), rg.clone())[0])
    finally:
        del os.environ["VLLM_FL_D13_NORM_QUANT"]

    # ---- 6. 计时（graph 口径） ----
    t_s = bp.graph_time_us(lambda: stock(x0, r0), iters=30)
    t_f = bp.graph_time_us(lambda: fused(x0, r0), iters=30)
    print(f"[perf] 两步 {t_s:.1f}µs vs 融合 {t_f:.1f}µs = {t_s / t_f:.3f}× "
          f"({NLAYER} 层全链 M=64 K={K})")

    print("\nT0 pair-wiring: ALL PASS")


if __name__ == "__main__":
    from vllm.config import VllmConfig, set_current_vllm_config

    with set_current_vllm_config(VllmConfig()):  # CustomOp 构造期需要
        _run()
