# D2 prefill 回退归因：int8 三算子链在 prefill M 档的分解测量。
#
# 背景：快速烟测 case1（prefill 重）w8a8 0.764× vs bf16，而 graph 微基准 M=1024
# 档 int8 链 GPU 侧 1.22-1.60× 赢——矛盾指向 serve eager prefill 的 host 开销/
# launch 间隙/int32 往返。本脚本把每个 (shape, M) 分解为：
#   eager wall（serve prefill 实际每层付出的）vs 三 kernel GPU 时间之和（事件计时）
#   差值 = host/launch/分配间隙；graph 口径两臂 = GPU 侧真值（对照）。
# 口径注意：eager 轮间降频伪影对 D 臂（多 kernel）更不利，但 serve prefill 本就是
# eager——eager wall 就是部署真值，graph 仅作 GPU 侧归因对照。

import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch

SHAPES = [  # (K, N, tag) — 每层 4 个量化线性
    (2048, 2560, "qkv"),
    (2048, 2048, "o"),
    (2048, 12288, "gate_up"),
    (6144, 2048, "down"),
]
MS = [1024, 2048, 8192]


def gpu_free_mib() -> float:
    import subprocess
    out = subprocess.check_output(
        ["/usr/local/corex/bin/ixsmi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True)
    return float(out.strip().splitlines()[0])


def bench(fn, iters: int, rounds: int = 3):
    """eager wall 计时（连续轮，无轮间 drain——serve prefill 也是连续流）。"""
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    outs = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        outs.append((time.perf_counter() - t0) / iters * 1e3)
    return statistics.median(outs)


def event_bench(fn, iters: int = 20):
    """单 kernel GPU 时间（CUDA 事件，免疫 host 抖动）。"""
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        times.append(e0.elapsed_time(e1))
    return statistics.median(times)


def main():
    used = gpu_free_mib()
    assert used < 3000, f"GPU 独占门禁失败：ixsmi 已用 {used}MiB"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda"

    from vllm_fl.dispatch import CachedOp, get_default_manager
    from vllm_fl.quantization.w8a8.int_mm import (
        _run_fused_epilogue, _fused_epilogue_enabled)
    _quant = CachedOp("dynamic_per_token_quant_int8")
    _mm_op = CachedOp("w8a8_dynamic_per_token_linear")

    linear_impl = get_default_manager().get_selected_impl_id(
        "w8a8_dynamic_per_token_linear")
    assert linear_impl == "default.flagos_intmm", linear_impl
    quant_impl = get_default_manager().get_selected_impl_id(
        "dynamic_per_token_quant_int8")
    print(f"# dispatch: linear={linear_impl} quant={quant_impl} "
          f"fused_epilogue={_fused_epilogue_enabled()}")
    assert _fused_epilogue_enabled(), "epilogue 融合关闭时分解口径不同，先开"

    print("shape|M|eagerA_ms|eagerD_ms|eagerD/A|gpuQ_ms|gpuMM_ms|gpuEPI_ms|"
          "gpuSUM_ms|launch_gap_ms|gap_ratio")
    layer_host_ms = {}
    for K, N, tag in SHAPES:
        x = (torch.randn(max(MS), K, device=dev) * 2).to(torch.bfloat16)
        W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ws = (W.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
        wq = (W.float() / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
        w8 = wq.t().contiguous()  # [K,N] 运行时布局
        for M in MS:
            xm = x[:M].contiguous()
            iters = 20 if M <= 2048 else 10

            def d_pair():
                q, s = _quant(xm)
                return _mm_op(q, w8, s, ws, out_dtype=torch.bfloat16)

            t_a = bench(lambda: torch.nn.functional.linear(xm, W), iters)
            t_d = bench(d_pair, iters)

            xq, xs = _quant(xm)
            acc = torch._int_mm(xq, w8)
            t_q = event_bench(lambda: _quant(xm))
            t_mm = event_bench(lambda: torch._int_mm(xq, w8))
            t_epi = event_bench(
                lambda: _run_fused_epilogue(acc, xs, ws, None, M, N))
            gpu_sum = t_q + t_mm + t_epi
            gap = t_d - gpu_sum
            layer_host_ms.setdefault(M, 0.0)
            layer_host_ms[M] += max(gap, 0.0)
            print(f"{tag}|{M}|{t_a:.4f}|{t_d:.4f}|{t_a / t_d:.3f}|"
                  f"{t_q:.4f}|{t_mm:.4f}|{t_epi:.4f}|{gpu_sum:.4f}|"
                  f"{gap:.4f}|{gap / t_d:.3f}", flush=True)

    # 每层 4 线性 → host 间隙/层；MiniCPM5-2B 28 层（13 blocks? 以模型实际为准，见输出注）
    print("\n# 每层 4 量化线性的 launch/host 间隙合计（ms/layer，eager prefill）:")
    for M, v in sorted(layer_host_ms.items()):
        print(f"# M={M}: {v:.3f} ms/layer")


if __name__ == "__main__":
    main()
