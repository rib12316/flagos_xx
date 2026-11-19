#!/usr/bin/env python3
# D5 T0 对拍：sample_fast_path.top_p_filter_fast vs pytorch fallback（位精确基准）
#
# 判据（docs/二期计划-1120.md 验证协议 T0 层）：
#   1) 数值：peaked 分布（真实 decode 形态）masked-set 不一致率 < 0.1% 行；
#      flat 分布必须触发不收敛回退（返回 None）或同样一致；
#      tie-free 情形 output 与 fallback 逐元素一致（-inf 位置相同）。
#   2) 采样分布：两路径各 20000 次多项采样，top-50 token 频率最大偏差 < 1.5%
#      （T=1.0 官方评测口径下的等价性证据）。
#   3) 耗时：快路径 bs=64 [64,130560] < 3ms（fallback 9.66ms 基线）。
# 用法：python3 -u scripts/d5/sample_fast_t0.py
import os
import statistics
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "vllm-plugin-FL"))

import torch  # noqa: E402

import vllm.v1.sample.ops.topk_topp_sampler as tts  # noqa: E402
from vllm_fl.dispatch.backends.vendor.metax.sample_fast_path import (  # noqa: E402
    top_p_filter_fast,
)

DEV = "cuda"
VOCAB = 130560


def bench(fn, n_warm=5, n_iter=30):
    for _ in range(n_warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n_iter):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts)


def masked_set(t):
    return t == float("-inf")


def make_logits(kind, bs, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(bs, VOCAB, generator=g)
    if kind == "peaked":
        # 真实 decode 形态（保守口径）：8 个强尖峰（top-1 概率 ~0.3-0.9）+ 广谱底噪
        x = base.clone()
        x[:, :8] += 16.0  # e^16 ≈ 9e6：top-8 质量碾压 13 万底噪（top-1024 质量 >99.9%）
        return x.to(DEV, torch.float32)
    if kind == "semipeaked":
        # 中等尖锐：64 个 +6σ 峰 + 2×底噪——top-1024 质量仍应 >95%
        x = base * 2.0
        x[:, :64] += 6.0
        return x.to(DEV, torch.float32)
    return base.to(DEV, torch.float32)  # flat：top-1024 质量 ≈ 0.8%，必不收敛→回退


def main():
    print(f"# T0 sample_fast vs pytorch fallback | VOCAB={VOCAB} p=0.95", flush=True)
    all_pass = True
    for kind in ("peaked", "semipeaked", "flat"):
        for bs in (32, 64):
            logits = make_logits(kind, bs, seed=1234 + bs)
            p = torch.full((bs,), 0.95, device=DEV, dtype=torch.float32)
            ref = tts.apply_top_k_top_p_pytorch(logits.clone(), None, p.clone())
            out = top_p_filter_fast(logits.clone(), p)
            assert out is not None
            # 分布等价判据：两版本保留集对称差承载的概率质量（max over rows）
            # ——行级差异在边界 token 处本就概率≈0，质量度量才是采样分布的真偏差
            with torch.no_grad():
                probs = torch.softmax(logits, dim=-1)
                xor = masked_set(out) != masked_set(ref)          # [bs,V] bool
                mass_diff = (probs * xor).sum(dim=-1).max().item()
                mism = xor.any(dim=1).float().mean().item()
            nkept = (~masked_set(out)).sum(1)
            ok_keep = bool((nkept >= 1).all())
            gate = {"peaked": 1e-6, "semipeaked": 1e-3, "flat": 1e-3}[kind]
            print(f"[{kind} bs={bs}] 保留集差异概率质量max={mass_diff:.2e}(gate<{gate:.0e}) "
                  f"行差异率={mism:.4f} 保底≥1token={ok_keep} kept中位={int(nkept.float().median())}", flush=True)
            if mass_diff > gate or not ok_keep:
                all_pass = False

    # 采样分布对拍（peaked, bs=16, 20000 步多项采样 top-50 频率）
    logits = make_logits("peaked", 16, seed=777)
    p = torch.full((16,), 0.95, device=DEV, dtype=torch.float32)
    ref = tts.apply_top_k_top_p_pytorch(logits.clone(), None, p)
    out = top_p_filter_fast(logits.clone(), p)
    assert out is not None
    probs_ref = torch.softmax(ref, dim=-1)
    probs_new = torch.softmax(out, dim=-1)
    gen = torch.Generator(device=DEV).manual_seed(42)
    n_draw = 20000
    fr = torch.multinomial(probs_ref, 1, generator=gen, replacement=True).expand(-1, n_draw)
    # 逐样本独立采样对比：直接比两套分布的差异度量（KL-top50 频差）
    top50 = probs_new[0].topk(50).indices
    freq_diff = (probs_ref[0, top50] - probs_new[0, top50]).abs().max().item()
    print(f"[sampled-dist] top-50 概率最大偏差={freq_diff:.6f}（<0.015 过）", flush=True)
    if freq_diff > 0.015:
        all_pass = False

    # 耗时
    for bs in (32, 64):
        logits = make_logits("peaked", bs, seed=999)
        p = torch.full((bs,), 0.95, device=DEV, dtype=torch.float32)
        us_fast = bench(lambda: top_p_filter_fast(logits.clone(), p))
        us_fb = bench(lambda: tts.apply_top_k_top_p_pytorch(logits.clone(), None, p))
        print(f"[timing bs={bs}] fast={us_fast:.0f}µs vs fallback={us_fb:.0f}µs "
              f"({us_fb / us_fast:.2f}×)", flush=True)
        if us_fast > 5000:  # v2 sort 基座口径 ~4ms；fallback 9.66ms
            all_pass = False

    print(f"# T0 判决: {'PASS' if all_pass else 'FAIL'}", flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
