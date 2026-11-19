# D7 prefill attention 自研 kernel T0（沐曦 C500）。
#
# 机理假设（P1-B2 判决延伸）：MACA flash varlen 在 chunked 续算（kv>q）形态的
# 96 TFLOPS 平台疑似 per-head CTA 的 KV 读放大（16 q heads 各自遍历 KV → 有效
# DRAM 流 ~94GB/s）。对策 = GQA packing：BLOCK_M 行 = (tokens × 8 q heads)
# 同 kv_head 打包，KV tile 每 CTA 只加载一次供多行共享 → KV 流 ÷8。
#
# 形态：等长 varlen paged（T0 简化——官方形状 S×q 等长成立）。
#   q [S*q, NH, HD] bf16；k/v cache [nb, PAGE, NKV, HD]；block_table [S, cols]
#   每 seq kv 总长 kv_len（含 q 自身，先写后算）；causal。
# grid=(total_q_tiles, NKV)；单 kernel 直写（并行度已足，无 split-KV）。
# smem 预算：BM=64,BN=64,nstages=1 → q32K+k16K+v16K=64KB 顶格（BM=128 变体
# 由 --bm 试）。
#
# T0 判据：S16 q2048 kv16K causal——maxrel vs flash < 5e-3 且 ≥1.3×（破 96T 平台）。
# 用法：python3 scripts/d7/prefill_gqa_t0.py [--S 16] [--q 2048] [--kv 16384]
#       [--bm 64] [--bn 64] [--warps 4]

import argparse
import sys

sys.path.insert(0, "scripts/d2")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import bottleneck_probe as bp  # noqa: E402
from flash_attn import flash_attn_varlen_func  # noqa: E402

DEV = "cuda"
NH, NKV, HD = 16, 2, 128
HPK = NH // NKV          # 8 q heads per kv head
PAGE = 16                # kv cache 页大小（沐曦 block_size）


@triton.jit
def _prefill_gqa_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr, block_table_ptr,
    seq_lens_ptr, q_tile_seq_ptr, q_tile_off_ptr,
    sm_scale,
    bt_stride: tl.int64,
    q_s0: tl.int64, q_s1: tl.int64,
    o_s0: tl.int64, o_s1: tl.int64,
    k_s0: tl.int64, k_s1: tl.int64, k_s2: tl.int64, k_s3: tl.int64,
    v_s0: tl.int64, v_s1: tl.int64, v_s2: tl.int64, v_s3: tl.int64,
    Q_LEN: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
    PAGE: tl.constexpr, HPK: tl.constexpr, HEAD_SIZE: tl.constexpr,
):
    pid_t = tl.program_id(0)
    kv_head = tl.program_id(1)

    seq_idx = tl.load(q_tile_seq_ptr + pid_t)
    q0 = tl.load(q_tile_off_ptr + pid_t)           # tile 首 token seq 内偏移
    kv_len = tl.load(seq_lens_ptr + seq_idx)
    prefix = kv_len - Q_LEN                        # 前文长度（>=0）

    offs_m = tl.arange(0, BM)
    offs_d = tl.arange(0, HEAD_SIZE)
    t_in = offs_m // HPK                           # tile 内 token 号
    q_row = seq_idx * Q_LEN + q0 + t_in            # q 全局行
    qh = kv_head * HPK + offs_m % HPK              # 行对应 q head
    row_ok = (q0 + t_in) < Q_LEN

    q = tl.load(q_ptr + q_row[:, None] * q_s0 + qh[:, None] * q_s1
                + offs_d[None, :],
                mask=row_ok[:, None], other=0.0)

    # causal 上界：本行 token 绝对位置 = prefix + q0 + t_in → 可见 kv 数 +1；
    # 无效行给超大上界（配合 q=0 行防 -inf 全行 NaN，存储时再掩掉）
    row_end = tl.where(row_ok, prefix + q0 + t_in + 1,
                       kv_len + 2 * BN)

    M = tl.full([BM], float("-inf"), dtype=tl.float32)
    L = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, HEAD_SIZE], dtype=tl.float32)

    offs_n = tl.arange(0, BN)
    for n0 in range(0, kv_len, BN):
        p = n0 + offs_n                            # 绝对 kv 位置
        col_ok = p < kv_len
        page = tl.load(block_table_ptr + seq_idx * bt_stride + p // PAGE,
                       mask=col_ok, other=0).to(tl.int64)
        inpage = p % PAGE
        krow = page * k_s0 + inpage * k_s1 + kv_head * k_s2
        vrow = page * v_s0 + inpage * v_s1 + kv_head * v_s2
        kt = tl.load(k_ptr + krow[:, None] + offs_d[None, :] * k_s3,
                     mask=col_ok[:, None], other=0.0)      # [BN, HD]
        vt = tl.load(v_ptr + vrow[:, None] + offs_d[None, :] * v_s3,
                     mask=col_ok[:, None], other=0.0)      # [BN, HD]
        s = tl.dot(q, tl.trans(kt)) * sm_scale             # [BM, BN]
        valid = (p[None, :] < row_end[:, None]) & col_ok[None, :]
        s = tl.where(valid, s, float("-inf"))
        m_new = tl.maximum(M, tl.max(s, 1))
        alpha = tl.exp(M - m_new)
        pexp = tl.exp(s - m_new[:, None])
        L = L * alpha + tl.sum(pexp, 1)
        acc = acc * alpha[:, None] + tl.dot(pexp.to(vt.dtype), vt)
        M = m_new
    out = acc / L[:, None]
    tl.store(out_ptr + q_row[:, None] * o_s0 + qh[:, None] * o_s1
             + offs_d[None, :],
             out.to(out_ptr.dtype.element_ty), mask=row_ok[:, None])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=16)
    ap.add_argument("--q", type=int, default=2048)
    ap.add_argument("--kv", type=int, default=16384)
    ap.add_argument("--bm", type=int, default=64)
    ap.add_argument("--bn", type=int, default=64)
    ap.add_argument("--warps", type=int, default=4)
    args = ap.parse_args()
    S, qlen, kvlen = args.S, args.q, args.kv
    assert kvlen >= qlen

    torch.manual_seed(0)
    q = torch.randn(S * qlen, NH, HD, device=DEV, dtype=torch.bfloat16)
    nb = (kvlen // PAGE) * S + 4
    kc = torch.randn(nb, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16)
    vc = torch.randn(nb, PAGE, NKV, HD, device=DEV, dtype=torch.bfloat16)
    cols = kvlen // PAGE
    bt = torch.randperm(nb, device=DEV)[: S * cols].to(torch.int32).reshape(S, -1)
    su = torch.full((S,), kvlen, dtype=torch.int32, device=DEV)
    cu_q = torch.arange(0, S * qlen + 1, qlen, dtype=torch.int32, device=DEV)
    cu_k = torch.arange(0, S * kvlen + 1, kvlen, dtype=torch.int32, device=DEV)

    def flash():
        return flash_attn_varlen_func(
            q=q, k=kc, v=vc, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=qlen, max_seqlen_k=kvlen, softmax_scale=HD ** -0.5,
            causal=True, window_size=(-1, -1), alibi_slopes=None, softcap=0.0,
            block_table=bt, s_aux=None)

    tok_per_tile = args.bm // HPK
    tiles_per_seq = triton.cdiv(qlen, tok_per_tile)
    tile_seq = torch.arange(S, device=DEV, dtype=torch.int32
                            ).repeat_interleave(tiles_per_seq)
    tile_off = torch.arange(tiles_per_seq, device=DEV, dtype=torch.int32
                            ).repeat(S) * tok_per_tile
    out = torch.empty_like(q)

    def mine():
        _prefill_gqa_kernel[(tiles_per_seq * S, NKV)](
            q, kc, vc, out, bt, su, tile_seq, tile_off, HD ** -0.5,
            bt.stride(0), q.stride(0), q.stride(1),
            out.stride(0), out.stride(1),
            kc.stride(0), kc.stride(1), kc.stride(2), kc.stride(3),
            vc.stride(0), vc.stride(1), vc.stride(2), vc.stride(3),
            Q_LEN=qlen, BM=args.bm, BN=args.bn, PAGE=PAGE, HPK=HPK,
            HEAD_SIZE=HD, num_warps=args.warps, num_stages=1)
        return out

    # 数值对拍
    ref = flash().float()
    got = mine().float()
    rel = ((got - ref).abs().max().item()
           / max(ref.abs().max().item(), 1e-6))
    print(f"maxrel vs flash = {rel:.3e}  ({'PASS' if rel < 5e-3 else 'FAIL'})")

    t_f = bp.graph_time_us(flash, iters=10)
    t_m = bp.graph_time_us(mine, iters=10)
    macs = S * qlen * kvlen * NH * HD * 2 * 2
    print(f"flash : {t_f:9.1f}µs  {macs / t_f * 1e-6:6.0f} TFLOPS(全量)")
    print(f"mine  : {t_m:9.1f}µs  {macs / t_m * 1e-6:6.0f} TFLOPS(全量)")
    print(f"speedup = {t_f / t_m:.3f}×")


if __name__ == "__main__":
    main()
