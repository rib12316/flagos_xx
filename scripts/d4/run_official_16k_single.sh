#!/bin/bash
# 16K 阶段接力：等 4K 官方调用结束后，新鲜 serve + 单次官方 bench 调用。
# 协议说明：官方脚本内部 RUNS=4/SKIP_FIRST=1，单次调用即"4 轮丢首取均值"，
# 不再外层套轮（旧 d4 外层 4 轮是 RUNS=1 时代做法，套用会 4×超时）。
# 用法：bash scripts/d4/run_official_16k_single.sh <4K调用的timeout wrapper pid>
set -uo pipefail
cd /data/flagos_final_submition
RES=results/20261120_final_submission
log() { echo "[$(date '+%F %T')] $*" | tee -a "$RES/master.log"; }

K4PID=$1
log "[16k-relay] 等待 4K 调用 (pid $K4PID) 结束..."
while kill -0 "$K4PID" 2>/dev/null; do sleep 10; done
log "[16k-relay] 4K 调用结束。最新 CSV："
ls -t benchmark_results/ | head -4 | tee -a "$RES/master.log"
cp "$(ls -t benchmark_results/raw_runs_20261120_2*.csv | head -1)" "$RES/raw_k4.csv" 2>/dev/null || true
cp "$(ls -t benchmark_results/summary_20261120_2*.csv | head -1)" "$RES/summary_k4.csv" 2>/dev/null || true
log "[16k-relay] 4K raw/summary 已拷贝为 raw_k4.csv/summary_k4.csv"

pkill -f 'vllm serve' || true; sleep 10
for i in $(seq 1 24); do
  mx-smi 2>/dev/null | grep -qE 'python|vllm' || break
  sleep 5
done
VLLM_PLUGINS=fl setsid nohup vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$RES/serve_bench_k16.log" 2>&1 &
READY=0
for i in $(seq 1 90); do
  sleep 10
  if curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm; then
    log "[16k-relay] serve[k16] ready ($((i*10))s)"; READY=1; break
  fi
done
[ "$READY" = 1 ] || { log "[16k-relay] FATAL: serve 起不来"; tail -30 "$RES/serve_bench_k16.log" | tee -a "$RES/master.log"; exit 1; }
grep -E 'F-c grouped|D13 fused|flagos_mctlass|F-a rope' "$RES/serve_bench_k16.log" | sort -u | head -6 | tee -a "$RES/master.log"

log "[16k-relay] 16K 单次官方调用启动（内部 4 轮丢首，超时 5400s）"
timeout 5400 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[16384,1024,64,128]]' > "$RES/round_k16_official.log" 2>&1
RC=$?
log "[16k-relay] 16K 调用 rc=$RC"
tail -c 400 "$RES/round_k16_official.log" | tr '\r' '\n' | tail -6 | tee -a "$RES/master.log"
cp "$(ls -t benchmark_results/raw_runs_20261120_2*.csv | head -1)" "$RES/raw_k16.csv" 2>/dev/null || true
cp "$(ls -t benchmark_results/summary_20261120_2*.csv | head -1)" "$RES/summary_k16.csv" 2>/dev/null || true
pkill -f 'vllm serve' || true
log "[16k-relay] K16_DONE"
