#!/usr/bin/env python3
# A-H4 T0：fused_add_rms_norm 两臂对决（沐曦 C500，L2-warm + CUDA graph 口径）
#
# 背景（serve 真实路径，已文件级核实）：
#   norm 在 compiled 态走 vllm IR 系统：RMSNorm.forward_cuda → forward_native →
#   ir.ops.fused_add_rms_norm.maybe_inplace（vllm/model_executor/layers/layernorm.py:88,157）。
#   该 ir op 有两个 impl：
#     · native（fp32 分解，vllm/ir/ops/layernorm.py:39-58）——VllmIRLoweringPass 将其
#       内联进 inductor 图 → 生成 triton 核（serve trace 所见 triton_red_fused_
#       fused_add_rms_norm_0/2，~0.60ms/步，43134 次/1027 步）。默认 priority=["native"]
#       （vllm/platforms/interface.py:1143），插件从未 override → 排除根因。
#     · vllm_c（vllm/kernels/vllm_c.py:51 → torch.ops._C.fused_add_rms_norm，mcoplib
#       C++ 核，双 in-place：input/residual 均原地改写）。
#
# ArmA（serve 现役）最小复现口径声明：
#   serve 的编译链 = vllm-compile PostGradPassManager.VllmIRLoweringPass 对图中
#   vllm_ir::fused_add_rms_norm 节点 dispatch(native) 后用 replace_by_example 追踪
#   impls["native"].impl_fn 内联。离线单复现整套 pass manager 需构造完整 VllmConfig，
#   过于繁琐；本脚本退而求其次：直接 torch.compile **同一个 impl_fn 函数对象**
#   （ir.ops.fused_add_rms_norm.impls["native"].impl_fn，非手写近似）——数学与内联
#   代码逐字一致，inductor 生成同族 triton_red_fused 核。口径差异：serve 内该核可与
#   邻算子融合（trace 有 _0/_2 两变体），本脚本为孤立核——若 ArmB 在孤立口径下胜出，
#   serve 内优势只会更难（ArmA 融合更强），故本口径对判活阈值是保守方向。
#
# ArmB 正确性陷阱自查结论（脚本内再跑一遍 guard 实证）：
#   · weight+1.0 约定属于 GemmaRMSNorm.forward_native（fp32 weight），vllm_c guard
#     `rms_add_no_var_size`（vllm/kernels/vllm_c.py:44：variance_size is None 且
#     weight.dtype == x.dtype）会因 fp32≠bf16 自动排除 Gemma → 无 +1 语义风险；
#     本模型 LlamaForCausalLM 走 RMSNorm，weight 为 raw bf16，guard 成立。
#   · variance_size：Llama 不传 var_hidden_size → None → guard 成立。
#   · 双 in-place（Tensor! input, Tensor! residual）：计时反复跑会原地改写输入值，
#     但 kernel 耗时与数值无关（数据无关 FLOPs），计时有效；正确性对拍一律用
#     新克隆输入、在对拍窗口内只跑一次。
#   正确性判据 = vs eager fp32 native 参考（即 serve native 数学本身），容差抄 vllm
#   ir 口径：bf16 DEFAULT_TOLERANCES（vllm/ir/tolerances.py）atol=1e-3/rtol=1.6e-2；
#   layernorm.py 对 fp16 另有 override atol=1e-2/rtol=2e-3（bf16 不适用）。报 max rel
#   diff + 超容差元素数。
#
# 判活门（主会话掌握）：ArmB/ArmA ≥1.10×（M∈{32,64} decode 档）才判活；M=2048 为
# prefill 档记录用。serve e2e 裁决以 T1 quick 双形状三轮中位为准。
import json
import os
import statistics
import subprocess
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")  # d9 先例：装载 fl 插件平台（metax→CUDA_ALIKE）
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "vllm-plugin-FL"))

import torch  # noqa: E402

# 触发注册链：vllm.ir.ops（ir op 本体 + native impl）；vllm.kernels（vllm_c impl：
# vllm/kernels/__init__.py → vllm_c.py → current_platform.import_kernels() → mcoplib._C）
import vllm.kernels  # noqa: F401,E402
from vllm import ir  # noqa: E402
from vllm.platforms import current_platform  # noqa: E402

DEV = "cuda"
CFG_PATH = "/workspace/MiniCPM5-2B-w8a8-v2/config.json"
# 兜底常量（config.json 缺失时用；与 2026-11-20 实测 config 一致）
HIDDEN_FALLBACK = 2048
EPS_FALLBACK = 1e-6
DT = torch.bfloat16
# vllm/ir/tolerances.py DEFAULT_TOLERANCES[torch.bfloat16]
ATOL, RTOL = 1e-3, 1.6e-2
SHAPES_M = [32, 64, 2048]  # 前两档 decode（bs=32/64 现役），2048 prefill 记录用


def bench(fn, n_warm=10, n_iter=50, reps=10):
    """CUDA-graph 口径：小 kernel 的 host launch 开销(~50µs)会淹没 GPU 时间，
    捕获 reps 次进 graph 后 replay 计时（phase0d 先例；serve 内 kernel 为
    编译/流式 launch，graph replay 才是同口径）。【逐字抄 d9_quant_probe_metax.py】"""
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        st = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        st.record()
        g.replay()
        e.record()
        torch.cuda.synchronize()
        ts.append(st.elapsed_time(e) * 1e3 / reps)
    return statistics.median(ts)


def load_model_caliber():
    """从 serve 同一份 config.json 取 hidden/eps（自己核实，不引用二手数字）。"""
    hidden, eps = HIDDEN_FALLBACK, EPS_FALLBACK
    try:
        with open(CFG_PATH) as f:
            cfg = json.load(f)
        hidden = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]
        print(f"# model caliber: {CFG_PATH} hidden={hidden} eps={eps} "
              f"layers={cfg.get('num_hidden_layers')} dtype={cfg.get('torch_dtype')}")
    except Exception as e:  # noqa: BLE001
        print(f"# WARN: 读 {CFG_PATH} 失败({e})，用兜底 hidden={hidden} eps={eps}")
    return hidden, eps


def diffs(a: torch.Tensor, b: torch.Tensor):
    """返回 (max_abs, max_rel, 超容差元素数)。rel 分母 clamp 1e-2 防 bf16 近零爆表。"""
    af, bf = a.float(), b.float()
    d = (af - bf).abs()
    over = (d > ATOL + RTOL * bf.abs()).sum().item()
    maxrel = (d / bf.abs().clamp_min(1e-2)).max().item()
    return d.max().item(), maxrel, over


def main():
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"# PROVENANCE t0_norm_ab: git={sha} torch={torch.__version__} "
          f"vendor={current_platform.vendor_name} "
          f"cuda_alike={current_platform.is_cuda_alike()}", flush=True)

    hidden, eps = load_model_caliber()
    op = ir.ops.fused_add_rms_norm
    assert "vllm_c" in op.impls, (
        "vllm_c impl 未注册：vllm.kernels→vllm_c.py→import_kernels(mcoplib._C) 链断裂")
    vllm_c_impl = op.impls["vllm_c"]
    assert vllm_c_impl.supported, "vllm_c impl supported=False（is_cuda_alike?）"
    print(f"# ir impls={sorted(op.impls)} vllm_c.inplace={vllm_c_impl.inplace}")

    # ArmA = serve 现役 native impl 的同一函数对象（VllmIRLoweringPass 内联的就是它）
    native_impl = op.impls["native"].impl_fn

    print(f"# shape(MxK) | ArmA_us(中位) ArmB_us(中位) speedup | "
          f"A_vs_ref: maxabs/maxrel/over | B_vs_ref同 | B_vs_A同 | guard")
    print(f"# 容差口径: bf16 atol={ATOL} rtol={RTOL} (vllm ir DEFAULT_TOLERANCES)")
    for M in SHAPES_M:
        torch.manual_seed(1234 + M + hidden)
        x0 = torch.randn(M, hidden, device=DEV).to(DT)
        r0 = torch.randn(M, hidden, device=DEV).to(DT)
        w = (torch.randn(hidden, device=DEV) * 0.5 + 1.0).to(DT)  # 权重量级~1，避全零

        # vllm_c guard 实证（对拍前的先决条件）
        guard = vllm_c_impl.supports_args(x0, r0, w, eps, None)

        # ---- 正确性（克隆输入，一次性）----
        ref_o, ref_r = native_impl(x0.clone(), r0.clone(), w, eps)  # eager fp32 参考
        armA = torch.compile(native_impl, dynamic=False)
        a_o, a_r = armA(x0.clone(), r0.clone(), w, eps)
        b_x, b_r = x0.clone(), r0.clone()  # ArmB 双 in-place → 必须克隆后单跑
        torch.ops._C.fused_add_rms_norm(b_x, b_r, w, eps)
        da = (diffs(a_o, ref_o), diffs(a_r, ref_r))
        db = (diffs(b_x, ref_o), diffs(b_r, ref_r))
        dab = (diffs(b_x, a_o), diffs(b_r, a_r))

        # ---- 计时（L2-warm：同一输入张量反复跑；ArmB 原地改值不碍计时）----
        tA = bench(lambda: armA(x0, r0, w, eps))
        tB = bench(lambda: torch.ops._C.fused_add_rms_norm(x0, r0, w, eps))

        print(f"M={M:<5d} K={hidden} | ArmA={tA:8.2f}us ArmB={tB:8.2f}us "
              f"speedup={tA / tB:5.2f}x | "
              f"A_out {da[0][0]:.2e}/{da[0][1]:.2e}/{da[0][2]} "
              f"A_res {da[1][0]:.2e}/{da[1][1]:.2e}/{da[1][2]} | "
              f"B_out {db[0][0]:.2e}/{db[0][1]:.2e}/{db[0][2]} "
              f"B_res {db[1][0]:.2e}/{db[1][1]:.2e}/{db[1][2]} | "
              f"B_vs_A {dab[0][0]:.2e}/{dab[0][1]:.2e}/{dab[0][2]} "
              f"res {dab[1][0]:.2e}/{dab[1][1]:.2e}/{dab[1][2]} | guard={guard}",
              flush=True)

    print("# 判读：speedup 是 ArmA/ArmB（>1 = ArmB 快）。判活门：M∈{32,64} 均 ≥1.10×。")
    print("# serve 投影（粗，孤立核口径）：decode 档 speedup × trace 0.60ms/步 norm 合计"
          " = 每步省时上限；e2e 以 T1 为准。")


if __name__ == "__main__":
    main()
