# 解析 vLLM torch profiler chrome trace（gzip JSON）：prefill 归因。
# 输出：kernel 时间分类聚合 / GPU busy-idle / 步进节奏 / cpu_op 形状直方图（真实 M）。
import gzip
import json
import re
import sys
from collections import Counter, defaultdict

path = sys.argv[1]

with gzip.open(path, "rt") as f:
    data = json.load(f)
ev = data["traceEvents"]
print(f"# events: {len(ev)}")

kernels = [e for e in ev if e.get("cat") == "kernel" and e.get("dur")]
ops = [e for e in ev if e.get("cat") == "cpu_op" and e.get("dur")]
api = [e for e in ev if e.get("cat") in ("cuda_runtime", "cuda_driver") and e.get("dur")]

print(f"# kernels={len(kernels)} cpu_ops={len(ops)} cuda_api={len(api)}")


def bucket(name: str) -> str:
    n = name.lower()
    if "quant" in n and ("bare" in n or "dynamic_per_token" in n):
        return "Q_quant_bare"
    if "epilogue" in n:
        return "E_epi_fused"
    if "int_mm" in n or ("gemm" in n and "int8" in n) or "igemm" in n:
        return "M_int8_gemm"
    if "attn" in n or "attention" in n or "flash" in n or "flashfwd" in n or "paged" in n:
        return "A_attention"
    if "gemm" in n or "nvjet" in n or "cutlass" in n or "matmul" in n or "s16816" in n:
        return "L_gemm_bf16(lm_head等)"
    if "norm" in n:
        return "N_norm"
    if "rotary" in n or "rope" in n:
        return "R_rope"
    if "elementwise" in n or "vectorized" in n or "copy" in n or "cat" in n or "index" in n:
        return "X_elementwise/copy"
    if "sampl" in n or "softmax" in n or "argmax" in n or "topk" in n:
        return "S_sampling/softmax"
    return "O_other"


agg = defaultdict(lambda: [0, 0])  # name -> [total_us, count]
bkt = defaultdict(float)
for e in kernels:
    agg[e["name"]][0] += e["dur"]
    agg[e["name"]][1] += 1
    bkt[bucket(e["name"])] += e["dur"]

total_k = sum(v[0] for v in agg.values())
t0 = min(e["ts"] for e in kernels)
t1 = max(e["ts"] + e["dur"] for e in kernels)
span = t1 - t0
print(f"\n# GPU kernel 总时间 {total_k/1e3:.1f} ms / trace span {span/1e3:.1f} ms "
      f"-> GPU busy {total_k/span*100:.1f}%，idle {100-total_k/span*100:.1f}%")

print("\n== 分类聚合 (ms, 占 kernel 时间比) ==")
for k, v in sorted(bkt.items(), key=lambda x: -x[1]):
    print(f"  {k:28s} {v/1e3:9.2f} ms  {v/total_k*100:5.1f}%")

print("\n== Top-20 单 kernel (ms | count | avg µs) ==")
for name, (tot, cnt) in sorted(agg.items(), key=lambda x: -x[1][0])[:20]:
    print(f"  {tot/1e3:8.2f} ms | {cnt:6d} | {tot/cnt:8.1f} µs | {name[:100]}")

# 步进节奏：ProfilerStep 标记 或 cpu_op 里的 step 区间
steps = [e for e in ev if e.get("name", "").startswith("ProfilerStep")]
if steps:
    durs = sorted(e["dur"] / 1e3 for e in steps)
    print(f"\n== ProfilerStep: n={len(durs)} median={durs[len(durs)//2]:.1f}ms "
          f"p10={durs[int(len(durs)*.1)]:.1f} p90={durs[int(len(durs)*.9)]:.1f}")

# GPU 空闲间隙 top（按 kernel 时间轴合并）
ks = sorted(kernels, key=lambda e: e["ts"])
gaps = []
cur_end = ks[0]["ts"]
for e in ks:
    if e["ts"] > cur_end:
        gaps.append((e["ts"] - cur_end, cur_end))
    cur_end = max(cur_end, e["ts"] + e["dur"])
gaps.sort(reverse=True)
print(f"\n== GPU 空闲 gap 总计 {sum(g for g,_ in gaps)/1e3:.1f} ms, top-8: ==")
for g, at in gaps[:8]:
    print(f"  {g/1e3:8.2f} ms @ t+{(at-t0)/1e3:.1f}ms")

# 真实 M：quant / linear cpu_op 的输入形状直方图
shape_hist = Counter()
for e in ops:
    n = e.get("name", "")
    if "w8a8" in n or "int_mm" in n or "scaled_mm" in n:
        dims = e.get("args", {}).get("Input Dims")
        if dims:
            shape_hist[json.dumps(dims[0] if isinstance(dims[0], list) else dims)] += 1
print("\n== w8a8/int_mm cpu_op 首输入形状直方图 (真实 M) ==")
for s, c in shape_hist.most_common(12):
    print(f"  {c:6d} × {s[:80]}")

# cuda api 耗值 top（host 侧线索）
api_agg = defaultdict(lambda: [0, 0])
for e in api:
    api_agg[e["name"]][0] += e["dur"]
    api_agg[e["name"]][1] += 1
print("\n== cuda api Top-10 (总 ms | count) ==")
for name, (tot, cnt) in sorted(api_agg.items(), key=lambda x: -x[1][0])[:10]:
    print(f"  {tot/1e3:8.2f} ms | {cnt:7d} | {name[:70]}")
