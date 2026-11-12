#!/usr/bin/env python3
"""D2 Phase 3 硬件闸门：sm71 int8 GEMM 微基准（15 形状 × 四臂）。

形状（运行时融合口径）：(K,N) ∈ {qkv(2048,2560), o(2048,2048), gate_up(2048,12288),
down(6144,2048), lm_head 参照(2048,130560)} × M ∈ {64,128,1024}（--m-extra 8192 加
chunked-prefill 档）。

四臂：
  A bf16   —— F.linear 基线（量化模型必须打败的对象）
  B int8   —— FlagGems 调度路径：CachedOp("dynamic_per_token_quant_int8") + flag_gems
              scaled_mm，权重预转 [K,N]（镜像 process_weights_after_loading）；
              主计时 = quant+mm 成对（一层实际付出的代价），副计时 = mm-only
  C 证伪   —— int8 值 cast bf16 后普通 GEMM × scales（_iluvatar/matmul_int8.py 占位
              模式）。C 赢 B = Triton int8 失调，立即报警
  D int_mm —— 真实调度路径（接线后）：CachedOp("w8a8_dynamic_per_token_linear")
              = torch._int_mm（cuBLAS int8, int32 累加）+ emu 位对齐 epilogue。
              启动断言 dispatch 选中 default.flagos_intmm；主计时 = quant+mm 成对
              （与 B 同一 CachedOp 量化实例），副计时 = mm-only

冒烟四查先行（失败非零退出）：编译 / 输出有限 / vs fp32 参考误差 ≤1e-2 /
D 臂 vs emu 公式逐位相等。
屋顶线：M=64 → 权重字节/时间 vs 1200 GB/s（decode 带宽界）；M=1024 → 2MNK/时间
vs 384 TOPS INT8（bf16 臂对 192 TFLOPS）。
判据（写入脚本头供台账引用）：任一 M=1024 ≥1.5x 或 M=64 ≥1.2x（D vs A，**CUDA graph
重放口径**——两臂同法捕获，消除三算子链 ~200µs/iter 的 host 开销与轮间 drain 后的 GPU
降频伪影；graph 是 serve decode（FULL_DECODE_ONLY）的真实执行形态，捕获输出与 eager
逐位对拍保正确性；eager 数字并列保留供对照）→ 硬件闸门过（B 保留为对照与报警，
gate_pass_flaggems_b 另存）。

GPU 独占门禁：ixsmi 已用显存 ≥3000MiB 即拒绝启动。
用法：python scripts/d2/bench_int8_gemm.py [--m-extra 8192] [--real-weight-smoke]
      [--out-dir results/20261112_d2_intmm_gemmbench/]
"""
from __future__ import annotations

import os
import re
import statistics
import subprocess
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch

SHAPES = [  # (K, N, tag)
    (2048, 2560, "qkv"),
    (2048, 2048, "o"),
    (2048, 12288, "gate_up"),
    (6144, 2048, "down"),
    (2048, 130560, "lm_head_ref"),
]
HBM_GBPS = 1200.0
INT8_TOPS = 384.0
BF16_TFLOPS = 192.0


def result_line(**kv):
    return "RESULT|" + "|".join(f"{k}={v}" for k, v in kv.items())


def gpu_free_mib() -> int:
    out = subprocess.run(["/usr/local/corex/bin/ixsmi"], capture_output=True, text=True).stdout
    m = re.search(r"(\d+)MiB / 32768MiB", out)
    return int(m.group(1)) if m else 32768


def bench(fn, iters: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        rounds.append((time.perf_counter() - t0) / iters * 1e3)  # ms
    return statistics.median(rounds)


def graph_bench(fn, iters: int):
    """CUDA-graph 捕获后按重放计时（两臂同法，消除 host 开销与降频伪影）。

    eager 循环在本机上受两种伪影干扰：①三算子链每 iter ~200µs host 开销
    （分配/cuBLAS host/launch）；②轮间全量 drain 后 GPU 降频使后续轮劣化
    2-3×（bf16 单 kernel 不受影响，成对比较失真）。graph 重放计时反映
    serve decode（FULL_DECODE_ONLY cudagraph）的真实可部署性能。
    返回 (重放 ms, 捕获输出)。
    """
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    rounds = []
    for _ in range(3):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            g.replay()
        torch.cuda.synchronize()
        rounds.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(rounds), out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--m-extra", type=int, default=0)
    ap.add_argument("--iters-small", type=int, default=100)
    ap.add_argument("--iters-large", type=int, default=30)
    ap.add_argument("--out-dir", default="results/20261112_d2_intmm_gemmbench/",
                    help="结果目录（20261107 为历史判定书证据，勿复用）")
    ap.add_argument("--real-weight-smoke", action="store_true",
                    help="真实权重 qkv L0：RTN→scaled_mm vs fp32 参考（median/p99/cos）")
    args = ap.parse_args()

    used = gpu_free_mib()
    assert used < 3000, f"GPU 独占门禁失败：ixsmi 已用 {used}MiB（≥3000）"

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda"

    from vllm_fl.dispatch import CachedOp, get_default_manager
    _quant = CachedOp("dynamic_per_token_quant_int8")
    _mm_op = CachedOp("w8a8_dynamic_per_token_linear")
    from flag_gems import scaled_mm
    from vllm_fl.quantization.w8a8.reference import (
        dynamic_per_token_quant_int8 as ref_quant,
    )

    # dispatch 接线自检：D 臂必须命中 int_mm impl（否则 bench 测的不是目标路径）
    linear_impl = get_default_manager().get_selected_impl_id(
        "w8a8_dynamic_per_token_linear")
    print(result_line(stage="dispatch", linear_impl=linear_impl))
    assert linear_impl == "default.flagos_intmm", (
        f"D 臂 dispatch 未选中 int_mm：got {linear_impl}")

    # ---------------- 冒烟四查 ----------------
    K, N, M = 2048, 2048, 64
    x = (torch.randn(M, K, device=dev) * 2).to(torch.bfloat16)
    W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    ws = (W.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
    wq = (W.float() / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
    w8 = wq.t().contiguous()  # [K,N] 运行时布局
    xq, xs = _quant(x.contiguous())
    y = scaled_mm(xq, w8, xs, ws, out_dtype=torch.bfloat16)
    assert torch.isfinite(y.float()).all(), "冒烟失败：输出非有限"
    y_ref = (xq.float() @ wq.float().t()) * xs * ws
    # 相对误差判据（bf16 输出舍入 ulp≈0.4%，绝对阈值会被值域误伤）
    rel = ((y.float() - y_ref).abs() / y_ref.abs().clamp_min(1e-3))
    med, p99 = rel.median().item(), rel.quantile(0.99).item()
    assert med <= 0.01 and p99 <= 0.05, \
        f"冒烟失败：scaled_mm vs 参考 median {med:.3e} / p99 {p99:.3e}"
    # 第四查：D 臂（dispatch int_mm）与 fake-quant emu 公式逐位相等
    y_d = _mm_op(xq, w8, xs, ws, out_dtype=torch.bfloat16)
    y_emu = torch._int_mm(xq, w8).to(torch.float32).mul_(xs).mul_(ws) \
        .to(torch.bfloat16)
    assert torch.equal(y_d, y_emu), "冒烟失败：int_mm 路径 vs emu 公式非逐位相等"
    print(result_line(stage="smoke", compiled=True, finite=True,
                      rel_median=f"{med:.2e}", rel_p99=f"{p99:.2e}",
                      intmm_emu_equal=True))

    if args.real_weight_smoke:
        real_weight_smoke(dev, _quant, scaled_mm)

    # ---------------- 15 形状四臂 ----------------
    ms = [64, 128, 1024] + ([args.m_extra] if args.m_extra else [])
    print("shape|M|arm|med_ms|ratio_vs_bf16|bw_GBs|tops")
    rows = []
    gate_ok = False
    gate_ok_flaggems_b = False
    for K, N, tag in SHAPES:
        x = (torch.randn(max(ms), K, device=dev) * 2).to(torch.bfloat16)
        W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ws = (W.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
        wq = (W.float() / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
        w8 = wq.t().contiguous()
        wq_bf = wq.to(torch.bfloat16)  # 证伪臂用
        for M in ms:
            xm = x[:M].contiguous()
            iters = args.iters_small if M <= 128 else args.iters_large
            # graph 模式（gate 口径）：A/D 两臂同法捕获重放，消 host 开销与
            # 轮间降频伪影；捕获输出与 eager 输出对拍保正确性
            t_a_g, y_a_g = graph_bench(
                lambda: torch.nn.functional.linear(xm, W), iters)
            def d_pair():
                q, s = _quant(xm)
                return _mm_op(q, w8, s, ws, out_dtype=torch.bfloat16)
            t_d_g, y_d_g = graph_bench(d_pair, iters)
            assert torch.equal(y_a_g, torch.nn.functional.linear(xm, W)), \
                "graph 捕获 bf16 输出与 eager 不一致"
            y_d_eager = d_pair()
            assert torch.equal(y_d_g, y_d_eager), \
                "graph 捕获 int_mm 输出与 eager 不一致"
            ratio_d = t_a_g / t_d_g
            # eager 数字并列保留（口径对照）
            t_a = bench(lambda: torch.nn.functional.linear(xm, W), iters)
            xq, xs = _quant(xm)  # 预热 + 供 mm-only 臂复用同批量化输入
            # B 主计时 = quant+mm 成对（真实一层代价）
            def b_pair():
                q, s = _quant(xm)
                return scaled_mm(q, w8, s, ws, out_dtype=torch.bfloat16)
            t_b = bench(b_pair, iters)
            t_b_mm = bench(lambda: scaled_mm(xq, w8, xs, ws, out_dtype=torch.bfloat16), iters)
            def c_falsify():
                q, s = _quant(xm)
                return (q.to(torch.bfloat16) @ w8.to(torch.bfloat16)) * s * ws
            t_c = bench(c_falsify, iters)
            t_d = bench(d_pair, iters)
            t_d_mm = bench(
                lambda: _mm_op(xq, w8, xs, ws, out_dtype=torch.bfloat16), iters)
            ratio = t_a / t_b
            flops = 2 * M * N * K
            row = dict(shape=f"{K}x{N}", tag=tag, M=M,
                       t_bf16_graph_ms=t_a_g, t_intmm_graph_ms=t_d_g,
                       ratio_graph=ratio_d,
                       t_bf16_ms=t_a, t_int8_pair_ms=t_b,
                       t_int8_mmonly_ms=t_b_mm, t_falsify_ms=t_c, ratio_vs_bf16=ratio,
                       t_intmm_pair_ms=t_d, t_intmm_mmonly_ms=t_d_mm,
                       speedup_intmm_vs_flaggems=t_b / t_d,
                       intmm_bw_GBs=K * N / (t_d_mm / 1e3) / 1e9,
                       intmm_tops=flops / (t_d_mm / 1e3) / 1e12,
                       int8_bw_GBs=K * N / (t_b_mm / 1e3) / 1e9,
                       int8_tops=flops / (t_b_mm / 1e3) / 1e12,
                       bf16_tops=flops / (t_a / 1e3) / 1e12)
            rows.append(row)
            print(result_line(arm="BD", **{k: (f"{v:.4f}" if isinstance(v, float) else v)
                                           for k, v in row.items()}))
            if (M == 1024 and ratio_d >= 1.5) or (M == 64 and ratio_d >= 1.2):
                gate_ok = True
            if (M == 1024 and t_a / t_b >= 1.5) or (M == 64 and t_a / t_b >= 1.2):
                gate_ok_flaggems_b = True
            if t_c < t_b:
                print(f"!! 证伪臂 C ({t_c:.3f}ms) 赢了 int8 臂 B ({t_b:.3f}ms) @ {tag} M={M}")
            if t_c < t_d:
                print(f"!! 证伪臂 C ({t_c:.3f}ms) 赢了 int_mm 臂 D ({t_d:.3f}ms) @ {tag} M={M}")

    import json
    os.makedirs(args.out_dir, exist_ok=True)
    with open(f"{args.out_dir}/bench.json", "w") as f:
        json.dump({"rows": rows, "gate_pass": gate_ok,
                   "gate_pass_flaggems_b": gate_ok_flaggems_b,
                   "gate_arm": "D_intmm_graph_vs_bf16_graph",
                   "dispatch_impl": linear_impl,
                   "roofline": {"HBM_GBPS": HBM_GBPS, "INT8_TOPS": INT8_TOPS}}, f,
                  indent=1)
    best_1024 = max((r["ratio_graph"] for r in rows if r["M"] == 1024), default=0)
    best_64 = max((r["ratio_graph"] for r in rows if r["M"] == 64), default=0)
    best_b_1024 = max((r["ratio_vs_bf16"] for r in rows if r["M"] == 1024), default=0)
    print(result_line(stage="gate", pass_=str(gate_ok).lower(),
                      best_intmm_M1024=f"{best_1024:.2f}",
                      best_intmm_M64=f"{best_64:.2f}",
                      best_flaggems_M1024=f"{best_b_1024:.2f}",
                      gate_flaggems_b=str(gate_ok_flaggems_b).lower()))
    return 0 if gate_ok else 2


def real_weight_smoke(dev, _quant, scaled_mm):
    """真实权重假层对拍：q_proj L0 RTN → scaled_mm vs fp32（median≤1% / p99≤5% / cos≥0.999）。"""
    import json as _json
    from safetensors.torch import load_file
    idx = _json.load(open("/workspace/MiniCPM5-2B/model.safetensors.index.json"))
    shard = sorted(set(idx["weight_map"].values()))[0]
    t = load_file(f"/workspace/MiniCPM5-2B/{shard}")
    W = t["model.layers.0.self_attn.q_proj.weight"].to(torch.float32)
    ws = (W.abs().amax(dim=1) / 127).clamp_min(1e-12)
    wq = (W / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
    w8 = wq.t().contiguous().to(dev)
    ws_d = ws.to(dev)
    x = (torch.randn(512, 2048, device=dev) * 3).to(torch.bfloat16)
    xq, xs = _quant(x.contiguous())
    y = scaled_mm(xq, w8, xs, ws_d, out_dtype=torch.bfloat16).float()
    y_ref = (xq.float() @ wq.float().to(dev).t()) * xs * ws_d
    rel = ((y - y_ref).abs() / y_ref.abs().clamp_min(1e-3))
    cos = torch.nn.functional.cosine_similarity(y.flatten(), y_ref.flatten(), dim=0).item()
    med, p99 = rel.median().item(), rel.quantile(0.99).item()
    print(result_line(stage="real_weight_smoke", median_rel=f"{med:.2e}",
                      p99_rel=f"{p99:.2e}", cos=f"{cos:.6f}",
                      pass_=str(med <= 0.01 and p99 <= 0.05 and cos >= 0.999).lower()))
    assert med <= 0.01 and p99 <= 0.05 and cos >= 0.999, "真实权重对拍未过"


if __name__ == "__main__":
    sys.exit(main())
