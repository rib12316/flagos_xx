#!/bin/bash
# P1-A3 4K 轮间方差归因（docs/二期计划-1120.md Phase 1）
#
# 现象：官方 4K 终局 4 个 pass（bench 内部 RUNS=4 丢首）= 6927.95/5883.03/6386.65，
# 散布 ~16%（16K 仅 0.12%）。归因三选一：
#   H-a pass 轮次效应（图形态/编译/缓存态随 pass 演化）
#   H-b serve 寿命效应（同一 serve 内位置决定）
#   H-c 物理噪声（应可复现为小方差）
# 设计：
#   Arm1 同 serve 连续 3 次 bench 调用（=12 passes）→ 方差随 pass 序演化？
#   Arm2 新鲜 serve ×2（各 1 次调用）→ 首个丢弃 pass 之后的 pass 分布 vs Arm1 同位置。
# 形态：官方终局逐字（零附加 env、无 --compilation-config）。
# 判据：Arm1 内 pass 序列自相关/趋势 显著 → H-a；Arm2 与 Arm1 同位置分布不同 → H-b；
#       两臂同分布且宽 → H-c。
# 用法：bash scripts/d5/variance_4k_probe.sh  （前置：GPU 空闲、无 vllm 残留）
set -uo pipefail
cd /data/flagos_xx
RES=results/20261120_c500_reprofiling/variance_4k
mkdir -p "$RES"
MODEL=/workspace/MiniCPM5-2B-w8a8-v2

start_serve() {  # $1 = tag
  local tag=$1
  VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL" --port 9031 \
    --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
    > "$RES/serve_${tag}.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 10
    if curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm; then
      echo "serve[$tag] ready after $((i*10))s"; return 0
    fi
    if ! pgrep -f 'vllm serve' >/dev/null; then
      echo "SERVE[$tag] DIED"; tail -30 "$RES/serve_${tag}.log"; return 1
    fi
  done
  echo "SERVE[$tag] TIMEOUT"; return 1
}

kill_serve() { pkill -f 'vllm serve' || true; sleep 8; }

run_bench() {  # $1 = arm标签 $2 = 调用序号
  timeout 1200 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name minicpm --port 9031 \
    --test-cases '[[4096,1024,64,256]]' > "$RES/bench_${1}_${2}.log" 2>&1 \
    || echo "bench $1#$2 客户端超时/挂死——从 raw CSV 取数"
  # 汇总挂死对策：直接从 raw_runs CSV 提取本调用 passes
  ls -t benchmark_results/raw_runs_*.csv 2>/dev/null | head -1 | xargs -r cat \
    | grep '4096,1024' | tail -8 >> "$RES/passes_raw_tail.csv" || true
}

echo "===== Arm1: 同 serve × 3 次调用 ====="
start_serve a1 || exit 1
for i in 1 2 3; do
  echo "-- Arm1 invocation $i --"
  run_bench a1 $i
done
kill_serve

echo "===== Arm2: 新鲜 serve × 2 ====="
for j in 1 2; do
  start_serve "a2_$j" || exit 1
  echo "-- Arm2 serve#$j invocation --"
  run_bench "a2$j" 1
  kill_serve
done

echo "VARIANCE_PROBE_DONE — 产物在 $RES/（passes_raw_tail.csv + bench_*.log + serve_*.log）"
echo "分析：对比 Arm1 12 passes 序列趋势 vs Arm2 各 4 passes 分布；"
echo "      serve log 里 grep 'Capturing\|piecewise\|compil' 看 pass 间图事件。"
