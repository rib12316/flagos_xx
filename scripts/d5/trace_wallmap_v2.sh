#!/bin/bash
# P1-A1 v2：active_iterations 大值连续记录 + 可见 curl + 受控负载
# v1 教训：start_profile 处理器忽略 body（api_router.py raw_request 未用），
# 窗口由 serve 侧 ProfilerConfig.active_iterations(默认5) 决定；且 ready 后被
# auto-start 吃掉 → 1KB 空 trace。v2：active_iterations=1000000 从 serve 起
# 连续记录，bench 完成即 stop dump。
set -uo pipefail
cd /data/flagos_xx
RES=results/20261120_c500_reprofiling
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
PROF=/tmp/prof2
rm -rf "$PROF"; mkdir -p "$PROF"
pkill -f 'vllm serve' || true; sleep 5

VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL" --port 9031 \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"'"$PROF"'","torch_profiler_with_stack":false,"active_iterations":1000000}' \
  > "$RES/serve_diag2.log" 2>&1 &
READY=0
for i in $(seq 1 60); do
  sleep 10
  if curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm; then
    echo "serve ready after $((i*10))s"; READY=1; break
  fi
  pgrep -f 'vllm serve' >/dev/null || { echo "SERVE DIED"; tail -20 "$RES/serve_diag2.log"; exit 1; }
done
[ "$READY" = 1 ] || { echo TIMEOUT; exit 1; }

# 主动确认 profiler 状态（不额外 start——auto/serve 侧已开；若日志无 auto-start 则补一发）
grep -c 'Starting profiler' "$RES/serve_diag2.log" || true
curl -s -X POST http://127.0.0.1:9031/start_profile -w 'START_HTTP=%{http_code}\n' || true

# 受控负载：32×4096 短输出（prefill 11s + decode ~60s，连续记录全程）
timeout 400 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model "$MODEL" --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,256,64,32]]' > "$RES/benchC_trace.log" 2>&1 \
  || echo "bench client timeout（数据已落 serve 侧）"

curl -s -X POST http://127.0.0.1:9031/stop_profile -w 'STOP_HTTP=%{http_code}\n' || true
sleep 10
pkill -f 'vllm serve' || true; sleep 5

echo "===== 归档 ====="
ls -la "$PROF" | head
mkdir -p "$RES/trace_all"
cp "$PROF"/*.gz "$RES/trace_all/" 2>/dev/null || echo "无 gz"
T=$(ls "$RES/trace_all"/rank0*.gz 2>/dev/null | head -1)
if [ -n "$T" ]; then
  echo "parse: $T ($(stat -c%s "$T") bytes)"
  python3 scripts/d2/parse_torch_trace.py "$T" > "$RES/trace_all/parse.txt" 2>&1 || true
  python3 - "$T" > "$RES/trace_all/topk_kernels.txt" 2>&1 <<'EOF'
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
  head -30 "$RES/trace_all/topk_kernels.txt"
else
  echo "无 rank0 trace——v2 也失败"
fi
