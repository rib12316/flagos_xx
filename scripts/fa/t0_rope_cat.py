#!/usr/bin/env python3
# F-a T0：rope(Q)+rope(K) 单核融合 vs 现役 inductor 双核（沐曦 C500）
#
# 归因（2026-11-20，/tmp/torchinductor_root + results/20261120_c500_reprofiling/
# trace_all/topk_kernels.txt）：
#   triton_poi_fused_cat_3 = rope(Q)：42107 次 244.74ms（5.81µs/层·步）
#   triton_poi_fused_cat_4 = rope(K)：42107 次 189.80ms（4.51µs/层·步）
#   （bs=32 decode；cat_4 块形按 prefill 上界选定，decode 仅 4 CTA）
# 本脚本对照臂 = 这两核的**逐字拷贝**（含 XBLOCK 与 grid 公式），融合臂 =
# vllm_fl.dispatch.backends.flaggems.impl.rope_cat.rope_qk_cat。
#
# 判据：
#   1) 正确性：融合输出与两对照核输出逐位相等（torch.equal；同输入含负位置
#      回卷用例——pos<0 → pos+131072 与现役核同语义）
#   2) 计时：M∈{32,64}（decode）speedup = (t_cat3+t_cat4)/t_fused ≥ 1.15×
#      （CUDA graph replay 中位 + L2-warm：同一输入张量反复跑，elementwise
#       不破坏输入——铁律 1/2）
#
# 只做 kernel 级 T0；pass 命中与 serve 级收益由上机 smoke serve 验证。
import os
import statistics
import subprocess
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "vllm-plugin-FL"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

from vllm_fl.dispatch.backends.flaggems.impl.rope_cat import (  # noqa: E402
    rope_qk_cat,
)

DEV = "cuda"
NQ, NKV, HEAD, MAX_POS = 16, 2, 128, 131072
Q_SIZE, K_SIZE, V_SIZE = NQ * HEAD, NKV * HEAD, NKV * HEAD
QKV_ROW = Q_SIZE + K_SIZE + V_SIZE  # 2560


# ---------------------------------------------------------------------------
# 对照臂：inductor 生成核逐字拷贝（/tmp/torchinductor_root/k7/ck7fkn….py 与
# h6/ch6hjq….py，XBLOCK=1024/2048 与 serve 一致——size_hints 按 prefill 上界选定）
# ---------------------------------------------------------------------------
@triton.jit
def _ref_cat3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 16)
    x2 = xindex // 2048
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (128 * x1 + 2560 * x2 + (x0)), tmp4 & xmask,
                   eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask,
                   eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 131072, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (128 * tmp10 + (x0)), tmp4 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.load(in_ptr0 + (64 + 128 * x1 + 2560 * x2 + (x0)), tmp4 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (64 + 128 * tmp10 + (x0)), tmp4 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 - tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (64 + 128 * x1 + 2560 * x2 + ((-64) + x0)),
                    tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (x2), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0)
    tmp24 = tl.full([XBLOCK], 131072, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tmp28 = tl.load(in_ptr2 + (128 * tmp27 + ((-64) + x0)), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp22 * tmp28
    tmp30 = tl.load(in_ptr0 + (128 * x1 + 2560 * x2 + ((-64) + x0)), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr2 + (64 + 128 * tmp27 + ((-64) + x0)), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp18, tmp35)
    tl.store(out_ptr0 + (x4), tmp36, xmask)


@triton.jit
def _ref_cat4(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 128)
    x1 = ((xindex // 128) % 2)
    x2 = xindex // 256
    x4 = xindex
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (2048 + 128 * x1 + 2560 * x2 + (x0)), tmp4 & xmask,
                   eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x2), tmp4 & xmask,
                   eviction_policy='evict_last', other=0.0)
    tmp7 = tl.full([XBLOCK], 131072, tl.int32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp6 < 0
    tmp10 = tl.where(tmp9, tmp8, tmp6)
    tmp11 = tl.load(in_ptr2 + (128 * tmp10 + (x0)), tmp4 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp12 = tmp5 * tmp11
    tmp13 = tl.load(in_ptr0 + (2112 + 128 * x1 + 2560 * x2 + (x0)), tmp4 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (64 + 128 * tmp10 + (x0)), tmp4 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp12 - tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp4, tmp16, tmp17)
    tmp19 = tmp0 >= tmp3
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp0 < tmp20
    tmp22 = tl.load(in_ptr0 + (2112 + 128 * x1 + 2560 * x2 + ((-64) + x0)),
                    tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp23 = tl.load(in_ptr1 + (x2), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0)
    tmp24 = tl.full([XBLOCK], 131072, tl.int32)
    tmp25 = tmp23 + tmp24
    tmp26 = tmp23 < 0
    tmp27 = tl.where(tmp26, tmp25, tmp23)
    tmp28 = tl.load(in_ptr2 + (128 * tmp27 + ((-64) + x0)), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp29 = tmp22 * tmp28
    tmp30 = tl.load(in_ptr0 + (2048 + 128 * x1 + 2560 * x2 + ((-64) + x0)),
                    tmp19 & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp31 = tl.load(in_ptr2 + (64 + 128 * tmp27 + ((-64) + x0)), tmp19 & xmask,
                    eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp29 + tmp32
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp19, tmp33, tmp34)
    tmp36 = tl.where(tmp4, tmp18, tmp35)
    tl.store(out_ptr0 + (x4), tmp36, xmask)


def run_ref_pair(qkv, positions, cos_sin, q_out, k_out):
    """现役路径：cat_3 + cat_4 两核（XBLOCK=1024/2048、num_warps=8/8，
    与 serve 缓存元数据一致）。"""
    n3 = Q_SIZE * qkv.shape[0]
    n4 = K_SIZE * qkv.shape[0]
    _ref_cat3[(triton.cdiv(n3, 1024),)](qkv, positions, cos_sin, q_out, n3,
                                        XBLOCK=1024, num_warps=8)
    _ref_cat4[(triton.cdiv(n4, 2048),)](qkv, positions, cos_sin, k_out, n4,
                                        XBLOCK=2048, num_warps=8)


def run_fused(qkv, positions, cos_sin, q_out, k_out):
    rope_qk_cat(qkv, positions, cos_sin, q_out, k_out,
                num_q_heads=NQ, num_kv_heads=NKV)


def bench(fn, n_warm=10, n_iter=50, reps=10):
    """CUDA-graph 口径（抄 scripts/d5/d9_quant_probe_metax.py，D9 教训：
    小 kernel host launch 开销 ~50µs 会淹没 GPU 时间，graph replay 才同口径；
    输入不克隆轮转 = L2-warm，铁律 1）。"""
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


def make_cos_sin(device):
    """与 vllm RotaryEmbedding._compute_cos_sin_cache 同式（theta=5e6）。"""
    inv_freq = 1.0 / (
        5000000.0 ** (torch.arange(0, HEAD, 2, dtype=torch.float32) / HEAD)
    )
    t = torch.arange(MAX_POS, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(torch.bfloat16).to(device)


def main():
    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"# PROVENANCE t0_rope_cat: git={sha} torch={torch.__version__}",
          flush=True)
    cos_sin = make_cos_sin(DEV)

    # ---- 正确性：含负位置回卷用例 ----
    for M in (32, 64):
        torch.manual_seed(1234 + M)
        qkv = (torch.randn(M, QKV_ROW, device=DEV) * 0.5).to(torch.bfloat16)
        pos = torch.randint(0, 4096, (M,), device=DEV, dtype=torch.int64)
        pos[0] = -5   # 回卷 → 131067
        pos[1] = -131072 + 3  # 回卷 → 3
        qr = torch.empty(M, NQ, HEAD, device=DEV, dtype=torch.bfloat16)
        kr = torch.empty(M, NKV, HEAD, device=DEV, dtype=torch.bfloat16)
        qf = torch.empty_like(qr)
        kf = torch.empty_like(kr)
        run_ref_pair(qkv, pos, cos_sin, qr, kr)
        run_fused(qkv, pos, cos_sin, qf, kf)
        torch.cuda.synchronize()
        eq_q = torch.equal(qr, qf)
        eq_k = torch.equal(kr, kf)
        d_q = (qr.float() - qf.float()).abs().max().item()
        d_k = (kr.float() - kf.float()).abs().max().item()
        print(f"[corr M={M}] q bitwise={eq_q} (maxdiff {d_q:g}) | "
              f"k bitwise={eq_k} (maxdiff {d_k:g})", flush=True)

    # vllm native fp32 交叉对拍（信息位：fma 收缩差异应 ≤1 bf16 ulp 量级）
    from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
    M = 32
    qkv = (torch.randn(M, QKV_ROW, device=DEV) * 0.5).to(torch.bfloat16)
    pos = torch.randint(0, 4096, (M,), device=DEV, dtype=torch.int64)
    qf = torch.empty(M, NQ, HEAD, device=DEV, dtype=torch.bfloat16)
    kf = torch.empty(M, NKV, HEAD, device=DEV, dtype=torch.bfloat16)
    run_fused(qkv, pos, cos_sin, qf, kf)
    q_nat, k_nat = RotaryEmbedding.forward_static(
        pos, qkv[:, :Q_SIZE].contiguous(), qkv[:, Q_SIZE:Q_SIZE + K_SIZE].contiguous(),
        HEAD, HEAD, cos_sin, True)
    q_nat = q_nat.view(M, NQ, HEAD)
    k_nat = k_nat.view(M, NKV, HEAD)
    print(f"[corr native-fp32 M={M}] q maxdiff="
          f"{(q_nat.float() - qf.float()).abs().max().item():g} | "
          f"k maxdiff={(k_nat.float() - kf.float()).abs().max().item():g}",
          flush=True)

    # ---- 计时 ----
    rows = []
    for M in (32, 64, 2048):
        torch.manual_seed(999 + M)
        qkv = (torch.randn(M, QKV_ROW, device=DEV) * 0.5).to(torch.bfloat16)
        pos = torch.randint(0, 1 << 20, (M,), device=DEV, dtype=torch.int64)
        q_out = torch.empty(M, NQ, HEAD, device=DEV, dtype=torch.bfloat16)
        k_out = torch.empty(M, NKV, HEAD, device=DEV, dtype=torch.bfloat16)

        def fn_ref():
            run_ref_pair(qkv, pos, cos_sin, q_out, k_out)

        def fn_fused():
            run_fused(qkv, pos, cos_sin, q_out, k_out)

        us_ref = bench(fn_ref)
        us_fused = bench(fn_fused)
        sp = us_ref / us_fused
        tag = "decode" if M <= 64 else "prefill(信息位)"
        print(f"[M={M} {tag}] ref(cat3+cat4)={us_ref:.2f}µs "
              f"fused={us_fused:.2f}µs speedup={sp:.2f}x", flush=True)
        rows.append((M, us_ref, us_fused, sp))

    out = os.path.join(REPO, "results", "20261120_fa_ropecat", "t0_rope_cat.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("M,ref_pair_us,fused_us,speedup\n")
        for r in rows:
            f.write(",".join(f"{v:.3f}" for v in r) + "\n")
    print(f"# CSV -> {out}", flush=True)
    print("# 判据：M∈{32,64} bitwise 全绿 且 speedup ≥1.15× → F-a T0 通过",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())
