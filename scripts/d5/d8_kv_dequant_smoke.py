#!/usr/bin/env python3
# D8 复活探针：metax flash_attn.flash_attn_with_kvcache_dequant 可用性验证
# 三问：(1) 能跑吗（编译/精度）(2) 数值对吗（vs bf16 反量化参考）(3) 带宽赚吗
# （int8 读取字节减半——若该 op 有效带宽 ≥ bf16 版本的 ~43% 即净赢）
# 形态：官方 decode 形状 bs=64 / kv=16384 / GQA 16Q-2KV / headdim 128 / block 256
import os
import statistics
import subprocess
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch  # noqa: E402
import flash_attn  # noqa: E402
from flash_attn import flash_attn_with_kvcache, flash_attn_with_kvcache_dequant  # noqa: E402

DEV = "cuda"
BS, KV, HQ, HKV, D, BLK = 64, 16384, 16, 2, 128, 256
GROUP = 8


def make_kv():
    torch.manual_seed(1234)
    nblocks = BS * (KV // BLK + 2)
    k = torch.randn(nblocks, BLK, HKV, D, device=DEV, dtype=torch.bfloat16) * 0.7
    v = torch.randn(nblocks, BLK, HKV, D, device=DEV, dtype=torch.bfloat16) * 0.6
    # 组内 absmax 量化（dim 每 8 元素一组）
    def q8(x):
        xb = x.view(nblocks, BLK, HKV, D // GROUP, GROUP)
        amax = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = (amax / 127.0).to(torch.bfloat16)
        q = (xb / amax * 127.0).round().clamp(-127, 127).to(torch.int8)
        return q.view_as(x), scale.view(nblocks, BLK, HKV, D // GROUP)

    kq, ks = q8(k)
    vq, vs = q8(v)
    return k, v, kq, ks, vq, vs


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


def main():
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"# PROVENANCE d8_kv_dequant_smoke: git={sha} flash_attn={flash_attn.__version__}", flush=True)
    k, v, kq, ks, vq, vs = make_kv()
    q = torch.randn(BS, 1, HQ, D, device=DEV, dtype=torch.bfloat16) * 0.5
    seqlens = torch.full((BS,), KV, device=DEV, dtype=torch.int32)
    bt = torch.arange(BS * (KV // BLK), device=DEV, dtype=torch.int32).view(BS, KV // BLK)

    # bf16 参考（同 block 布局）
    out_ref = flash_attn_with_kvcache(
        q=q, k_cache=k, v_cache=v, cache_seqlens=seqlens, block_table=bt,
        causal=True, num_splits=0)
    us_bf16 = bench(lambda: flash_attn_with_kvcache(
        q=q, k_cache=k, v_cache=v, cache_seqlens=seqlens, block_table=bt,
        causal=True, num_splits=0))

    # int8 + dequant
    try:
        out_i8 = flash_attn_with_kvcache_dequant(
            q=q, k_cache=kq, v_cache=vq, k_scale=ks, v_scale=vs,
            cache_seqlens=seqlens, block_table=bt, causal=True,
            num_splits=0, dequant_group=GROUP)
        torch.cuda.synchronize()
    except Exception as e:
        print(f"DEQUANT_OP_FAIL: {type(e).__name__}: {str(e)[:300]}", flush=True)
        return 1
    # 数值：参考 = 反量化后的 bf16 flash
    kd = (kq.view_as(k).to(torch.bfloat16) * ks.repeat_interleave(GROUP, dim=-1))
    vd = (vq.view_as(v).to(torch.bfloat16) * vs.repeat_interleave(GROUP, dim=-1))
    out_deref = flash_attn_with_kvcache(
        q=q, k_cache=kd, v_cache=vd, cache_seqlens=seqlens, block_table=bt,
        causal=True, num_splits=0)
    rel = ((out_i8.float() - out_deref.float()).abs().max() /
           out_deref.float().abs().max()).item()
    rel_vs_bf16 = ((out_i8.float() - out_ref.float()).abs().max() /
                   out_ref.float().abs().max()).item()
    us_i8 = bench(lambda: flash_attn_with_kvcache_dequant(
        q=q, k_cache=kq, v_cache=vq, k_scale=ks, v_scale=vs,
        cache_seqlens=seqlens, block_table=bt, causal=True,
        num_splits=0, dequant_group=GROUP))
    # 带宽账：KV 字节 int8 = 0.5×bf16（scale 读取 ~1/8×2B≈0.25B/elem 折算）
    byt_bf16 = BS * KV * HKV * D * 2 * 2
    eff_bf16 = byt_bf16 / (us_bf16 * 1e-6) / 1e9
    byt_i8 = BS * KV * HKV * D * (1 + 1 / GROUP) * 2  # k+v int8 + scale fp16/8
    eff_i8 = byt_i8 / (us_i8 * 1e-6) / 1e9
    print(f"[bs={BS} kv={KV}] bf16: {us_bf16:.0f}µs ({eff_bf16:.0f}GB/s eff) | "
          f"int8+dequant: {us_i8:.0f}µs ({eff_i8:.0f}GB/s eff)", flush=True)
    print(f"[数值] vs 反量化参考 maxrel={rel:.2e} | vs 原 bf16 maxrel={rel_vs_bf16:.2e}", flush=True)
    print(f"[判决素材] 加速比={us_bf16/us_i8:.2f}×（>1.0 即 KV 读带宽净赢；>1.2 立项 D8 serve 级）", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
