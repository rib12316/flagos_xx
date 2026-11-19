#!/usr/bin/env python3
# P1-B4 采样 topk_topp 沐曦实价探针 v2（docs/二期计划-1120.md Phase 1）
#
# v1 教训（20261120 01:22 崩溃取证）：
#   - BLOCK_SIZE_TRUNC 误写 4092（非 2 幂，stock 实为 4096）——沐曦 triton3.0 不拦截、
#     错误 lowering → _topk_topp_kernel 运行时 Xnack/ATU 非法地址，进程 core dump。
#     即：kernel 在沐曦**能编译**（v1 trap 里 kernelName=_topk_topp_kernel 已真实运行），
#     vendor patch 注释"ttgir 编译失败"待本探针复核（可能是旧版本现象）。
#   - v2 改造：每 Triton 变体独立子进程（ATU fault 只杀该变体）+ -u 无缓冲 + 常量全 2 幂。
# 回答：Q1 fallback 实价？ Q2 stock/降档 BLOCK 哪些可编译可跑且位精确？ Q3 grid 甜点？
# D5 立项判据 = fallback ≥2ms/步 且存在可用的位精确变体。
import os
import statistics
import subprocess
import sys

os.environ.setdefault("VLLM_PLUGINS", "fl")

DEV = "cuda"
VOCAB = 130560
REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV = os.path.join(REPO, "results", "20261120_c500_reprofiling", "sampling_metax.csv")


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


def run_variant(bs, blk, blk_tr, n_prog):
    """子进程模式：单变体编译+对拍+计时，打印一行 RESULT 后退出。"""
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    import torch
    import vllm.v1.sample.ops.topk_topp_sampler as tts
    import vllm.v1.sample.ops.topk_topp_triton as ttt

    torch.manual_seed(1234 + bs)
    logits0 = torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32)
    p = torch.full((bs,), 0.95, device=DEV, dtype=torch.float32)
    out_ref = tts.apply_top_k_top_p_pytorch(logits0.clone(), None, p.clone())

    tables = ttt._TRITON_TABLE_CACHE.get(torch.device(DEV))
    if tables is None:
        tables = (
            torch.tensor(ttt._NORMAL_CDF_TO_SIGMA_TABLE, device=DEV, dtype=torch.float32),
            torch.tensor(ttt._PERCENTILE_TO_STD_TABLE, device=DEV, dtype=torch.float32),
        )
        ttt._TRITON_TABLE_CACHE[torch.device(DEV)] = tables
    n_cdf, p_std = tables
    buf = logits0.new_empty((max(n_prog, 8), VOCAB))
    lx = logits0.clone()

    def call():
        ttt._topk_topp_kernel[(n_prog,)](
            lx, lx.stride(0), buf, p_std, n_cdf,
            lx,  # K 哑指针（TOPK_ENABLED=False 不读）
            p,
            BATCH_SIZE=bs, VOCAB_SIZE=VOCAB, MASK_VALUE=float("-inf"),
            BLOCK_SIZE=blk, BLOCK_SIZE_TRUNC=blk_tr,
            TOPK_ENABLED=False, TOPP_ENABLED=True,
        )

    call()  # CUDA_LAUNCH_BLOCKING=1：此处同步抛错
    eq = torch.equal(lx, out_ref)
    os.environ.pop("CUDA_LAUNCH_BLOCKING", None)  # 计时段恢复异步
    # 计时用 8 份轮转，保持数据分布
    bufs = [logits0.clone() for _ in range(8)]
    i = [0]

    def cyc():
        x = bufs[i[0]]
        i[0] = (i[0] + 1) % 8

        def call2(x=x):
            ttt._topk_topp_kernel[(n_prog,)](
                x, x.stride(0), buf, p_std, n_cdf, x, p,
                BATCH_SIZE=bs, VOCAB_SIZE=VOCAB, MASK_VALUE=float("-inf"),
                BLOCK_SIZE=blk, BLOCK_SIZE_TRUNC=blk_tr,
                TOPK_ENABLED=False, TOPP_ENABLED=True,
            )
        call2()

    us = bench(cyc)
    print(f"RESULT bs={bs} blk={blk} tr={blk_tr} g={n_prog} us={us:.0f} equal={eq}", flush=True)


def main():
    import torch

    if len(sys.argv) > 1 and sys.argv[1] == "--variant":
        run_variant(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))
        return 0

    sha = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    num_sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"# PROVENANCE sampling_probe_metax v2: git={sha} torch={torch.__version__} "
          f"SM={num_sm} dev={torch.cuda.get_device_name(0)}", flush=True)

    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    rows = []

    import vllm.v1.sample.ops.topk_topp_sampler as tts

    # ---- Q1: fallback / sort（纯 torch，主进程安全）----
    for bs in (32, 64):
        torch.manual_seed(1234 + bs)
        logits0 = torch.randn(bs, VOCAB, device=DEV, dtype=torch.float32)
        p = torch.full((bs,), 0.95, device=DEV, dtype=torch.float32)
        bufs = [logits0.clone() for _ in range(8)]
        i = [0]

        def cyc_fb():
            x = bufs[i[0]]
            i[0] = (i[0] + 1) % 8
            tts.apply_top_k_top_p_pytorch(x, None, p)

        us_fb = bench(cyc_fb)
        us_sort = bench(lambda: torch.sort(bufs[i[0] % 8], dim=-1)[0])
        rows.append((bs, "pytorch_fallback", us_fb, ""))
        rows.append((bs, "torch_sort_only", us_sort, ""))
        print(f"[bs={bs}] fallback={us_fb:.0f}µs  sort={us_sort:.0f}µs", flush=True)

    # ---- Q2/Q3: Triton 变体（子进程隔离；两阶段控机时）----
    def run_one(bs, blk, tr, g):
        tag = f"direct_B{blk}_t{tr}_g{g}"
        try:
            r = subprocess.run(
                [sys.executable, "-u", os.path.abspath(__file__),
                 "--variant", str(bs), str(blk), str(tr), str(g)],
                capture_output=True, text=True, timeout=300, cwd=REPO)
            line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
            if line:
                d = dict(kv.split("=") for kv in line.split()[1:])
                rows.append((bs, tag, float(d["us"]), f"equal={d['equal']}"))
                print(f"[bs={bs}] {tag}: {d['us']}µs equal={d['equal']}", flush=True)
                return rows[-1]
            tail = (r.stderr or r.stdout or "").strip().splitlines()
            note = f"FAIL {tail[-1][:150] if tail else 'no output rc=' + str(r.returncode)}"
            rows.append((bs, tag, None, note))
            print(f"[bs={bs}] {tag} FAIL rc={r.returncode}: "
                  f"{tail[-1][:150] if tail else 'no output'}", flush=True)
        except subprocess.TimeoutExpired:
            rows.append((bs, tag, None, "FAIL timeout300s"))
            print(f"[bs={bs}] {tag} FAIL timeout", flush=True)
        return None

    # 阶段1（bs=64, g=64）：定哪些 BLOCK 档可编译可跑
    stage1 = []
    for blk, tr in ((8192, 4096), (4096, 2048), (4096, 4096), (2048, 2048), (1024, 1024)):
        r1 = run_one(64, blk, tr, 64)
        if r1 is not None and "equal=True" in r1[3]:
            stage1.append((blk, tr))
    # 阶段2：仅对可用档扫 grid × bs（编译缓存命中，子进程只付 import 税）
    for blk, tr in stage1[:2]:  # 最多两档，控机时
        for bs in (32, 64):
            for g in sorted({min(bs, num_sm), 32, 104, 128} - {64}):
                if g <= 0 or g > 128:
                    continue
                run_one(bs, blk, tr, g)

    with open(CSV, "w") as f:
        f.write("bs,variant,us,equal_note\n")
        for bs, v, us, note in rows:
            f.write(f"{bs},{v},{us if us is not None else ''},{note}\n")
    print(f"# CSV -> {CSV}", flush=True)
    fb = {r[0]: r[2] for r in rows if r[1] == "pytorch_fallback"}
    best = [r for r in rows if r[2] is not None and "equal=True" in r[3]]
    if best:
        for bs in (32, 64):
            b = min((r for r in best if r[0] == bs), key=lambda r: r[2], default=None)
            if b:
                print(f"# 判决素材 bs={bs}: 最快位精确 {b[1]}={b[2]:.0f}µs vs fallback {fb[bs]:.0f}µs "
                      f"({fb[bs]/b[2]:.2f}×)", flush=True)
    else:
        print("# 无可用 Triton 变体——D5 走 torch 原生重写或 kernel 移植", flush=True)
    return 0


if __name__ == "__main__":
    import torch  # noqa: F401  # 顶部统一 import（子进程模式也在 main 前）
    sys.exit(main())
