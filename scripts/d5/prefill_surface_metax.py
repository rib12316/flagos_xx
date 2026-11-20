# P1-B2 prefill attention 曲面（沐曦 C500 / MACA flash 现役路径）。
#
# 计时口径：抄 scripts/d2/attn_surface_probe.py 的 graph_time_us（CUDA graph
# 3×N replay wall 中位——唯一可信口径）；read 参照抄 tuned_read_ref_us
# （BLOCK=16384 大块读，ref_table 同口径）。
# 调用形态：抄 vendor/metax/impl/attention/flash_attn.py:807 prefill 段——
#   flash_attn_varlen_func(q=[total_q,NH,HD], k=k_cache(paged), v=v_cache,
#     block_table, cu_seqlens_q, cu_seqlens_k(KV 含 q 自身，已写 cache),
#     max_seqlen_q/k, softmax_scale, causal, window_size=None, alibi=None,
#     softcap=0.0, s_aux=None)
# 页表乱序（randperm）：serve 分配器散射口径（attn_surface_probe scatter 教训——
# arange 纯顺序流会高估带宽可达性）。
# A 机指定优先点（11-20 01:46 留言）：S1q4096kv4K causal（4K 官方真实形状）、
# S16q2048kv16K（16K 官方真实形状）——--suite official 先跑。
# 全表：S∈{1,8,16,32,64}×q∈{512,1024,2048}×kv∈{4K,16K}×causal{0,1} + 官方两点。
#
# 真机：python3 scripts/d5/prefill_surface_metax.py --suite official \
#         --out-dir results/20261120_c500_reprofiling/prefill_surface

import argparse
import csv
import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "d2"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import bottleneck_probe as bp  # noqa: E402  (graph_time_us)
from flash_attn import flash_attn_varlen_func  # noqa: E402  (沐曦 2.6.3+metax)

DEV = "cuda"
NH, NKV, HD, BLOCK = 16, 2, 128, 16   # MiniCPM5-2B 真实头配置；沐曦页块 16
BW_CEIL = 829                          # 沿用 d2 口径（跨卡仅作参考分母）
_MAX_REF_BYTES = 64 * 16384 * 2 * NKV * HD * 2

COLUMNS = ["tag", "S", "q_len", "kv_len", "causal", "time_us", "kv_read_bytes",
           "read_ref_us", "GBps_attn", "GBps_ref", "pct_of_ref", "pct_of_829",
           "T_MACs", "verdict"]

_ref_x = None


@triton.jit
def _read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x, 0))


def tuned_read_ref_us(nbytes):
    global _ref_x
    n = nbytes // 2
    if _ref_x is None:
        _ref_x = torch.randn(_MAX_REF_BYTES // 2, device=DEV, dtype=torch.bfloat16)
    best = None
    for w in (4, 8):
        out = torch.empty(triton.cdiv(n, 16384), device=DEV, dtype=torch.float32)

        def call():
            _read_kernel[(triton.cdiv(n, 16384),)](
                _ref_x, out, n, BLOCK=16384, num_warps=w)
        t = bp.graph_time_us(call, iters=30)
        best = t if best is None else min(best, t)
    return best


def build_case(S, q_len, kv_len, causal=True, seed=0):
    """MACA flash varlen + paged cache（serve prefill 段同构）。
    kv_len = 每 seq KV 总长（含 q 自身——vLLM 先写 K/V 再算）。"""
    if kv_len < q_len:
        raise ValueError("kv_len >= q_len（KV 含 q 自身）")
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(S * q_len, NH, HD, device=DEV, dtype=torch.bfloat16,
                    generator=g)
    nb = triton.cdiv(kv_len, BLOCK) * S + 4
    kc = torch.randn(nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16,
                     generator=g)
    vc = torch.randn(nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16,
                     generator=g)
    cols = triton.cdiv(kv_len, BLOCK)
    bt = torch.randperm(nb, device=DEV)[: S * cols].to(torch.int32).reshape(S, -1)
    cu_q = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
    cu_k = torch.arange(0, S * kv_len + 1, kv_len, dtype=torch.int32, device=DEV)

    def call():
        return flash_attn_varlen_func(
            q=q, k=kc, v=vc, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
            max_seqlen_q=q_len, max_seqlen_k=kv_len, softmax_scale=HD ** -0.5,
            causal=causal, window_size=(-1, -1), alibi_slopes=None,
            softcap=0.0, block_table=bt, s_aux=None)
    by = S * kv_len * 2 * NKV * HD * 2  # K+V 读字节（与 d2 口径一致，causal 不折半）
    return call, by


def classify(pct_bw, pct_mac):
    if pct_mac >= 60:
        return "compute-bound"
    if pct_bw >= 60:
        return "memory-bound"
    return "latency/低占用"


def run_case(writer, out_f, done, tag, S, q_len, kv_len, causal):
    if tag in done:
        print(f"[skip] {tag}", flush=True)
        return
    t0 = time.time()
    try:
        call, by = build_case(S, q_len, kv_len, causal, seed=S * 1000 + q_len + kv_len)
        est = bp.graph_time_us(call, iters=3)
        iters = 10 if est > 3000 else 30
        t = bp.graph_time_us(call, iters=iters)
        del call
        ref_t = tuned_read_ref_us(by) if by <= _MAX_REF_BYTES else float("nan")
        gb_a, gb_r = by / t * 1e-3, by / ref_t * 1e-3
        macs = S * q_len * kv_len * NH * HD * 2 / (t * 1e-6) / 1e12
        row = dict(tag=tag, S=S, q_len=q_len, kv_len=kv_len, causal=int(causal),
                   time_us=f"{t:.1f}", kv_read_bytes=by,
                   read_ref_us=f"{ref_t:.1f}", GBps_attn=f"{gb_a:.0f}",
                   GBps_ref=f"{gb_r:.0f}", pct_of_ref=f"{gb_a / gb_r * 100:.0f}",
                   pct_of_829=f"{gb_a / BW_CEIL * 100:.0f}", T_MACs=f"{macs:.2f}",
                   verdict=classify(gb_a / BW_CEIL * 100, macs / 118 * 100))
    except Exception as e:
        row = {c: "" for c in COLUMNS}
        row.update(tag=tag, S=S, q_len=q_len, kv_len=kv_len, causal=int(causal),
                   verdict=f"ERROR: {type(e).__name__}: {e}"[:200])
    writer.writerow(row)
    out_f.flush()
    os.fsync(out_f.fileno())
    print(f"[{time.time() - t0:6.1f}s] {tag:<32} {row.get('time_us', ''):>9}µs "
          f"{row.get('GBps_attn', ''):>6}GB/s {row.get('pct_of_ref', ''):>5}%ref "
          f"{row.get('verdict', '')}", flush=True)
    torch.cuda.empty_cache()


# A 机优先点（官方场景真实形状）+ 计划全表
OFFICIAL = [(1, 4096, 4096), (16, 2048, 16384)]
GRID = [(S, q, kv) for S in (1, 8, 16, 32, 64) for q in (512, 1024, 2048)
        for kv in (4096, 16384)]


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("verdict", "").startswith("ERROR"):
                    continue
                done.add(r["tag"])
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["official", "grid", "all"], default="all")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    free_b, total_b = torch.cuda.mem_get_info()
    assert (total_b - free_b) < 3 * 1024 ** 3, \
        f"GPU busy: {(total_b - free_b) >> 20} MiB"

    prop = torch.cuda.get_device_properties(0)
    import subprocess
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    print(f"# PROVENANCE prefill_surface_metax: git={sha} dev={prop.name} "
          f"SMs={prop.multi_processor_count} torch={torch.__version__} "
          f"flash_attn_varlen(沐曦) NH={NH} NKV={NKV} HD={HD} BLOCK={BLOCK} "
          f"页表=randperm(散射) 计时=graph_time_us", flush=True)

    cases = []
    if args.suite in ("official", "all"):
        cases += [(f"pre S{S} q{q} kv{kv // 1024}k c1", S, q, kv, True)
                  for S, q, kv in OFFICIAL]
    if args.suite in ("grid", "all"):
        cases += [(f"pre S{S} q{q} kv{kv // 1024}k c{c}", S, q, kv, bool(c))
                  for S, q, kv in GRID for c in (1, 0)]
        if args.suite == "all":   # 官方点补非 causal 臂
            cases += [(f"pre S{S} q{q} kv{kv // 1024}k c0", S, q, kv, False)
                      for S, q, kv in OFFICIAL]

    p = os.path.join(args.out_dir, "prefill_surface.csv")
    done = load_done(p)
    new_file = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        print(f"===== suite={args.suite}: {len(cases)} cases ({len(done)} done) "
              f"-> {p}", flush=True)
        for tag, S, q, kv, causal in cases:
            run_case(writer, f, done, tag, S, q, kv, causal)
    print("PREFILL_SURFACE_DONE", flush=True)


if __name__ == "__main__":
    main()
