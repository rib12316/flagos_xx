# D2 prefill 链分解（沐曦 C500 / mctlass 版）。
#
# 原版 profile_prefill_chain.py 面向天数（ixsmi 门禁 / intmm 断言 / [K,N] 布
# 局 / cuBLAS int8 + 融合 epilogue 三算子链）。沐曦侧 W8A8 主路径 =
# mctlassEx w8a8_scaled_mm_azp（量化 + GEMM+scale+bias 单算子，epilogue 融合
# 在 GEMM 内），本脚本按两算子分解：
#   D 臂 eager wall = _quant(x) + mctlass op 连发；GPU 侧 = 两算子事件计时之和
#   差值 = host/launch 间隙；A 臂 = bf16 F.linear 同口径。

import os
import statistics

os.environ.setdefault("VLLM_PLUGINS", "fl")

import torch

SHAPES = [  # (K, N, tag) — 每层 4 个量化线性
    (2048, 2560, "qkv"),
    (2048, 2048, "o"),
    (2048, 12288, "gate_up"),
    (6144, 2048, "down"),
]
MS = [16, 64, 128, 1024, 2048, 8192]  # 16/64/128 = decode 档（P1-B3，稳态 Running 63-64）


def gpu_used_mib() -> float:
    # mx-smi 表格解析脆弱，改用运行时 mem_get_info（设备级空闲）
    free_b, total_b = torch.cuda.mem_get_info()
    return (total_b - free_b) / 1024 / 1024


def bench(fn, iters: int, rounds: int = 3):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    outs = []
    for _ in range(rounds):
        t0 = __import__("time").perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        outs.append((__import__("time").perf_counter() - t0) / iters * 1e3)
    return statistics.median(outs)


def event_bench(fn, iters: int = 20):
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
    used = gpu_used_mib()
    assert used < 3000, f"GPU 独占门禁失败：mx-smi 已用 {used}MiB"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(0)
    dev = "cuda"

    from vllm_fl.dispatch import CachedOp, get_default_manager
    _quant = CachedOp("dynamic_per_token_quant_int8")
    _mm_op = CachedOp("w8a8_dynamic_per_token_linear")

    linear_impl = get_default_manager().get_selected_impl_id(
        "w8a8_dynamic_per_token_linear")
    assert linear_impl == "default.flagos_mctlass", linear_impl
    print(f"# dispatch: linear={linear_impl} (mctlassEx 承载)")

    print("shape|M|eagerA_ms|eagerD_ms|eagerD/A|gpuQ_ms|gpuMM_ms|gpuSUM_ms|"
          "launch_gap_ms|gap_ratio")
    for K, N, tag in SHAPES:
        x = (torch.randn(max(MS), K, device=dev) * 2).to(torch.bfloat16)
        W = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
        ws = (W.float().abs().amax(dim=1) / 127).clamp_min(1e-12)
        wq = (W.float() / ws[:, None]).round().clamp(-127, 127).to(torch.int8)
        w8 = wq.contiguous()  # 沐曦布局契约 [N,K]（process 层不转置）
        for M in MS:
            xm = x[:M].contiguous()
            iters = 20 if M <= 2048 else 10

            def d_pair():
                q, s = _quant(xm)
                return _mm_op(q, w8, s, ws, out_dtype=torch.bfloat16)

            t_a = bench(lambda: torch.nn.functional.linear(xm, W), iters)
            t_d = bench(d_pair, iters)

            xq, xs = _quant(xm)
            t_q = event_bench(lambda: _quant(xm))
            t_mm = event_bench(lambda: _mm_op(xq, w8, xs, ws,
                                              out_dtype=torch.bfloat16))
            gpu_sum = t_q + t_mm
            gap = t_d - gpu_sum
            print(f"{tag}|{M}|{t_a:.4f}|{t_d:.4f}|{t_a / t_d:.3f}|"
                  f"{t_q:.4f}|{t_mm:.4f}|{gpu_sum:.4f}|"
                  f"{gap:.4f}|{gap / t_d:.3f}", flush=True)


if __name__ == "__main__":
    main()
