# ② 采样过滤微基准：top_p 过滤两条实现的同口径对比 + 等价性检查。
# 事件计时（每 iter 新输入，避免 pytorch 路径原地改写的非幂等），20 iter 中位；
# A 臂绝对值与 serve trace 的 4.4ms 交叉校验。掩码 logits 等价性逐位对比。
import statistics
import sys
import time

import torch

sys.path.insert(0, "/flagos_比赛/flagos_xx/vllm-plugin-FL")
from vllm.v1.sample.ops.topk_topp_sampler import (  # noqa: E402
    apply_top_k_top_p_pytorch, apply_top_k_top_p_triton)

DEV = "cuda"
VOCAB = 130560


def bench(fn_make, iters=20):
    times = []
    for _ in range(3):
        fn_make()[0]()
    torch.cuda.synchronize()
    for _ in range(iters):
        x = fn_make()[0]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        x()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def arm_triton(bs, dtype, p_val):
    def make():
        logits = (torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32) * 3.0
                  ).to(dtype)
        p = torch.full([bs], p_val, device=DEV, dtype=torch.float32)
        return (lambda: apply_top_k_top_p_triton(logits.clone(), None, p)), logits, p
    return make


def arm_pytorch(bs, dtype, p_val):
    def make():
        logits = (torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32) * 3.0
                  ).to(dtype)
        p = torch.full([bs], p_val, device=DEV, dtype=torch.float32)
        # pytorch 路径原地改写 → 包 clone 计入成本（serve 里 logits 本就独占，
        # clone 是本基准的保守开销，两臂同加做对照时单独报告）
        return (lambda: apply_top_k_top_p_pytorch(logits.clone(), None, p,
                                                  allow_cpu_sync=True)), logits, p
    return make


def equivalence(bs, dtype, p_val):
    """掩码结果逐位对比：数值相同的 -inf 位置集合 + 有限值逐位相等。"""
    logits = (torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32) * 3.0).to(dtype)
    p = torch.full([bs], p_val, device=DEV, dtype=torch.float32)
    a = apply_top_k_top_p_triton(logits.clone(), None, p)
    b = apply_top_k_top_p_pytorch(logits.clone(), None, p, allow_cpu_sync=True)
    inf_a, inf_b = torch.isinf(a), torch.isinf(b)
    mask_same = bool((inf_a == inf_b).all())
    finite_diff = ((a[~inf_a] - b[~inf_b]).abs().max().item()
                   if (~inf_a).any() else 0.0)
    n_flip = int((inf_a != inf_b).sum())
    return mask_same, n_flip, finite_diff


def arm_grid(bs, dtype, p_val, num_programs):
    """直接 launch _topk_topp_kernel，可控 NUM_PROGRAMS（行数学不变=逐位安全）。
    wrapper 的 buffer 钳制（≤num_sm 行）在此自管。"""
    from vllm.v1.sample.ops.topk_topp_triton import (
        _topk_topp_kernel, _PERCENTILE_TO_STD_TABLE, _NORMAL_CDF_TO_SIGMA_TABLE)

    tables = {}
    buf = {}

    def make():
        logits = (torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32) * 3.0
                  ).to(dtype).to(torch.float32).contiguous()
        p = torch.full([bs], p_val, device=DEV, dtype=torch.float32)
        dev = logits.device
        if dev not in tables:
            tables[dev] = (logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE),
                           logits.new_tensor(_PERCENTILE_TO_STD_TABLE))
        n_cdf, p_std = tables[dev]
        if dev not in buf or buf[dev].shape[0] < num_programs:
            buf[dev] = logits.new_empty((num_programs, VOCAB))

        def call():
            _topk_topp_kernel[(num_programs,)](
                logits, logits.stride(0), buf[dev], p_std, n_cdf,
                logits,  # k 未启用的哑指针
                p,
                BATCH_SIZE=bs, MASK_VALUE=float("-inf"), VOCAB_SIZE=VOCAB,
                BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
                TOPK_ENABLED=False, TOPP_ENABLED=True)
        return call, logits
    return make


def grid_equiv_check(bs, num_programs):
    """不同 NUM_PROGRAMS 的掩码结果应与 stock（16 programs）逐位一致。"""
    from vllm.v1.sample.ops.topk_topp_triton import (
        _topk_topp_kernel, _PERCENTILE_TO_STD_TABLE, _NORMAL_CDF_TO_SIGMA_TABLE)
    logits = torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32) * 3.0
    logits = logits.contiguous()
    p = torch.full([bs], 0.95, device=DEV, dtype=torch.float32)
    outs = []
    for np_ in (16, num_programs):
        buf = logits.new_empty((np_, VOCAB))
        n_cdf = logits.new_tensor(_NORMAL_CDF_TO_SIGMA_TABLE)
        p_std = logits.new_tensor(_PERCENTILE_TO_STD_TABLE)
        x = logits.clone()
        _topk_topp_kernel[(np_,)](
            x, x.stride(0), buf, p_std, n_cdf, x, p,
            BATCH_SIZE=bs, MASK_VALUE=float("-inf"), VOCAB_SIZE=VOCAB,
            BLOCK_SIZE=8192, BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=False, TOPP_ENABLED=True)
        outs.append(x)
    torch.cuda.synchronize()
    return bool(torch.equal(outs[0], outs[1]))


def main():
    print(f"{'arm':<28}{'bs':>4}{'dtype':>8}{'p':>6}{'ms':>10}")
    for dtype in (torch.float32,):  # stock kernel 断言仅支持 fp32（链路即 fp32）
        for bs in (32, 64):
            for p_val in (0.95,):
                for name, arm in (("triton", arm_triton),
                                  ("pytorch(+clone)", arm_pytorch)):
                    make = arm(bs, dtype, p_val)
                    ms = bench(make)
                    print(f"{name:<28}{bs:>4}{str(dtype).split('.')[-1]:>8}"
                          f"{p_val:>6}{ms:>10.3f}", flush=True)
    print("\n== grid 扫（fp32, p=0.95, TOPP-only, 逐位安全维度）==")
    for bs in (32, 64):
        for np_ in (16, 32, 64, 128, 256):
            ms = bench(arm_grid(bs, torch.float32, 0.95, np_))
            print(f"grid={np_:>4} programs bs={bs}: {ms:.3f} ms", flush=True)
    ok = grid_equiv_check(64, 128)
    print(f"grid 等价性 bs64 (16 vs 128 programs): bitwise_equal={ok}")
    print("\n== 等价性（fp32/fp16 随机 logits, p=0.95）==")
    for dtype in (torch.float32, torch.bfloat16):
        for bs in (8, 64):
            same, n_flip, d = equivalence(bs, dtype, 0.95)
            print(f"bs{bs} {dtype}: mask_identical={same} flip_positions={n_flip} "
                  f"finite_maxdiff={d:.2e}")


if __name__ == "__main__":
    main()
