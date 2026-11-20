#!/bin/bash
# P1-A1 decode/混批 trace 墙图（docs/二期计划-1120.md Phase 1）
# 形态：官方终局逐字 + 仅 --profiler-config 观测旗标（playbook 先例允许）。
# 采集：窗1 decode 稳态（benchA [[4096,1024,64,32]] 输出期）；窗2 prefill（benchB 首 10s）。
# 产物：results/20261120_c500_reprofiling/trace_{decode,prefill}/ + parse 输出。
# 已知口径：attention 若在图内重放则 trace 不可见（天数既定口径，沐曦待本窗验证）。
# 用法：bash scripts/d5/trace_wallmap.sh （前置：GPU 空闲）
set -uo pipefail
cd /data/flagos_xx
RES=results/20261120_c500_reprofiling
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
PROF=/tmp/prof
rm -rf "$PROF"; mkdir -p "$PROF"
pkill -f 'vllm serve' || true; sleep 5

echo "===== 诊断 serve（官方形态 + profiler 旗标）====="
VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL" --port 9031 \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"'"$PROF"'","torch_profiler_with_stack":false}' \
  > "$RES/serve_diag.log" 2>&1 &
for i in $(seq 1 60); do
  sleep 10
  curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm && { echo "serve ready ($((i*10))s)"; break; }
done
grep -E 'GPU KV cache size|Maximum concurrency' "$RES/serve_diag.log" | tail -2 || true

echo "===== 窗1: decode 稳态 ====="
mkdir -p "$PROF/dec"
( timeout 600 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name minicpm --port 9031 \
    --test-cases '[[4096,1024,64,32]]' > "$RES/benchA_dec.log" 2>&1 ) &
BENCH=$!
# 等 prefill 完成进 decode（32×4096=131K tok / ~11.5K tok/s ≈ 12s，留裕量）
sleep 25
curl -s -X POST http://127.0.0.1:9031/start_profile -d '{"output_dir":"'"$PROF"/dec"'"}' || true
sleep 12
curl -s -X POST http://127.0.0.1:9031/stop_profile || true
echo "窗1 profile 完成，等 bench 收尾"
wait $BENCH 2>/dev/null || true
sleep 5

echo "===== 窗2: prefill ====="
mkdir -p "$PROF/pre"
( timeout 300 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name minicpm --port 9031 \
    --test-cases '[[4096,128,64,32]]' > "$RES/benchB_pre.log" 2>&1 ) &
BENCH=$!
sleep 3
curl -s -X POST http://127.0.0.1:9031/start_profile -d '{"output_dir":"'"$PROF"/pre"'"}' || true
sleep 10
curl -s -X POST http://127.0.0.1:9031/stop_profile || true
wait $BENCH 2>/dev/null || true

pkill -f 'vllm serve' || true; sleep 8

echo "===== 归档与解析 ====="
mkdir -p "$RES/trace_decode" "$RES/trace_prefill"
find "$PROF/dec" -name '*.gz' -o -name '*trace*' 2>/dev/null | head -3 | while read -r f; do cp "$f" "$RES/trace_decode/"; done
find "$PROF/pre" -name '*.gz' -o -name '*trace*' 2>/dev/null | head -3 | while read -r f; do cp "$f" "$RES/trace_prefill/"; done
for w in decode prefill; do
  T=$(ls "$RES/trace_$w"/*.gz 2>/dev/null | head -1)
  [ -n "$T" ] && { echo "-- parse $w: $T"; python3 scripts/d2/parse_torch_trace.py "$T" > "$RES/trace_$w/parse.txt" 2>&1; \
    python3 - "$T" > "$RES/trace_$w/topk_kernels.txt" 2>&1 <<'EOF'
import gzip, json, sys
from collections import Counter
with gzip.open(sys.argv[1], "rt") as f:
    ev = json.load(f)["traceEvents"]
c = Counter(); d = Counter()
for e in ev:
    if e.get("cat") == "kernel" and e.get("dur"):
        c[e["name"]] += e["dur"]; d[e["name"]] += 1
print(f"total_kernel_ms={sum(c.values())/1000:.1f}")
for n, t in c.most_common(25):
    print(f"{t/1000:9.2f}ms {d[n]:6d}x {n[:110]}")
EOF
  } || echo "-- $w 无 trace 文件"
done
echo "WALLMAP_DONE — $RES/trace_*/{parse.txt,topk_kernels.txt}"
