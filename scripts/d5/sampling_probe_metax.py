#!/usr/bin/env python3
# P1-B4 采样 topk_topp 沐曦实价探针（docs/二期计划-1120.md Phase 1）
#
# 背景：vendor/metax/patches/topk_topp_sampler.py 把 apply_top_k_top_p 永久指向
# PyTorch fallback（Triton kernel 在沐曦 ttgir PassManager 编译失败）——每 decode 步
# 对 [bs,130560] fp32 走全词表 sort。天数侧先验 4.4ms/步≈18% ITL。
# 本探针回答三问：
#   Q1 fallback 实价多少 µs/步（bs=32/64）？sort 占多少？
#   Q2 Triton kernel 沐曦编译失败能否通过降 BLOCK_SIZE 修复（64KB smem 假说）？
#      通过 wrapper 与直接 launch 双路测；BLOCK_SIZE∈{8192,4096,2048,1024}。
#   Q3 修好后 grid（num_programs）甜点在哪？（沐曦 104 CU，与天数 16 CU 不同）
# 判定：kernel 输出 vs fallback torch.equal（位精确）；D5 立项判据 = fallback ≥2ms/步
# 且存在可编译变体。
import os
import statistics
import subprocess
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")
import torch  # noqa: E402

DEV = "cuda"
VOCAB = 130560
NUM_SM = torch.cuda.get_device_properties(0).multi_processor_count


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
        ts.append(s.elapsed_time(e) * 1e3)  # µs
    return statistics.median(ts)


class Cycler:
    """in-place kernel 的计时适配：循环 N 份克隆，保持数据分布稳定。"""

    def __init__(self, proto, n=8):
        self.bufs = [proto.clone() for _ in range(n)]
        self.i = 0

    def next(self):
        b = self.bufs[self.i]
        self.i = (self.i + 1) % len(self.bufs)
        return b


def main():
    import vllm.v1.sample.ops.topk_topp_sampler as tts
    import vllm.v1.sample.ops.topk_topp_triton as ttt

    sha = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    print(f"# PROVENANCE sampling_probe_metax: git={sha} torch={torch.__version__} "
          f"SM={NUM_SM} dev={torch.cuda.get_device_name(0)}")
    print(f"# VOCAB={VOCAB} p=0.95 TOPP-only fp32")

    rows = []
    for bs in (32, 64):
        torch.manual_seed(1234 + bs)
        logits0 = torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32)
        p = torch.full((bs,), 0.95, device=DEV, dtype=torch.float32)

        # ---- Q1: pytorch fallback 与 sort 归因 ----
        out_ref = tts.apply_top_k_top_p_pytorch(logits0.clone(), None, p.clone())
        cy = Cycler(logits0)
        us_fb = bench(lambda: tts.apply_top_k_top_p_pytorch(cy.next(), None, p))
        us_sort = bench(lambda: torch.sort(cy.next(), dim=-1)[0])
        rows.append((bs, "pytorch_fallback", us_fb, ""))
        rows.append((bs, "torch_sort_only", us_sort, ""))
        print(f"[bs={bs}] fallback={us_fb:.0f}µs  sort={us_sort:.0f}µs")

        # ---- Q2: wrapper 直测（不经 metax patch，patch 只换 sampler 模块引用）----
        try:
            out_tr = ttt.apply_top_k_top_p_triton(logits0.clone(), None, p.clone())
            eq = torch.equal(out_tr, out_ref)
            cy2 = Cycler(logits0)
            us_tr = bench(lambda: ttt.apply_top_k_top_p_triton(cy2.next(), None, p))
            rows.append((bs, "triton_wrapper_stock", us_tr, f"equal={eq}"))
            print(f"[bs={bs}] triton_wrapper(8192/4092默认)={us_tr:.0f}µs equal={eq}")
        except Exception as ex:
            msg = f"{type(ex).__name__}: {str(ex)[:200]}"
            rows.append((bs, "triton_wrapper_stock", None, f"COMPILE/RUN FAIL {msg}"))
            print(f"[bs={bs}] triton_wrapper FAIL: {msg}")

        # ---- Q2b: 直接 launch，BLOCK_SIZE 降档扫（64KB smem 假说）----
        tables = ttt._TRITON_TABLE_CACHE.get(torch.device(DEV))
        if tables is None:
            tables = (
                torch.tensor(ttt._NORMAL_CDF_TO_SIGMA_TABLE, device=DEV, dtype=torch.float32),
                torch.tensor(ttt._PERCENTILE_TO_STD_TABLE, device=DEV, dtype=torch.float32),
            )
            ttt._TRITON_TABLE_CACHE[torch.device(DEV)] = tables
        n_cdf, p_std = tables
        buf = logits0.new_empty((max(NUM_SM, 128), VOCAB))

        for blk, blk_tr in ((8192, 4092), (4096, 4096), (2048, 2048), (1024, 1024)):
            for n_prog in {min(bs, NUM_SM), 128, 64}:
                key = (bs, f"kernel_direct_B{blk}_g{n_prog}")
                if any(r[:2] == key for r in rows):
                    continue
                try:
                    cyk = Cycler(logits0)
                    lx = cyk.next()

                    def call(lx=lx):
                        ttt._topk_topp_kernel[(n_prog,)](
                            lx, lx.stride(0), buf, p_std, n_cdf,
                            lx,  # K 哑指针（TOPK_ENABLED=False 不读）
                            p,
                            BATCH_SIZE=bs, VOCAB_SIZE=VOCAB,
                            MASK_VALUE=float("-inf"),
                            BLOCK_SIZE=blk, BLOCK_SIZE_TRUNC=min(blk_tr, blk),
                            TOPK_ENABLED=False, TOPP_ENABLED=True,
                        )

                    call()
                    torch.cuda.synchronize()
                    eq = torch.equal(lx, out_ref)
                    us = bench(call)
                    rows.append((bs, f"direct_B{blk}_g{n_prog}", us, f"equal={eq}"))
                    print(f"[bs={bs}] direct BLOCK={blk} g={n_prog}: {us:.0f}µs equal={eq}")
                except Exception as ex:
                    msg = f"{type(ex).__name__}: {str(ex)[:160]}"
                    rows.append((bs, f"direct_B{blk}_g{n_prog}", None, f"FAIL {msg}"))
                    print(f"[bs={bs}] direct BLOCK={blk} g={n_prog} FAIL: {msg}")

    outdir = os.path.join(os.path.dirname(__file__), "..", "..",
                          "results", "20261120_c500_reprofiling")
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "sampling_metax.csv"), "w") as f:
        f.write("bs,variant,us,equal_note\n")
        for bs, v, us, note in rows:
            f.write(f"{bs},{v},{us if us is not None else ''},{note}\n")
    print(f"# CSV -> results/20261120_c500_reprofiling/sampling_metax.csv")
    best = [r for r in rows if r[2] is not None and "equal=True" in r[3]]
    if best:
        b = min(best, key=lambda r: r[2])
        print(f"# 最快位精确变体: bs={b[0]} {b[1]} = {b[2]:.0f}µs "
              f"(vs fallback {min(r[2] for r in rows if r[1]=='pytorch_fallback' and r[0]==b[0]):.0f}µs)")
    else:
        print("# 无位精确 Triton 变体——D5 需走 torch 原生重写或 kernel 移植路线")


if __name__ == "__main__":
    sys.exit(main())
