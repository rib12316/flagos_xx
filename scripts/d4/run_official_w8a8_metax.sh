#!/bin/bash
# 沐曦 w8a8-mctlass 臂官方 bench（官方口径）：每场景新鲜 serve，4 轮丢首。
# serve 形态 = d4 PROVENANCE 官方锁死：--compilation-config FULL_DECODE_ONLY
# 轮间探活+自动重启（v3 补跑器语义）+ 客户端硬超时杀（d4 已知挂死对策）。
# 用法：bash scripts/d4/run_official_w8a8_metax.sh   （前置：当前 serve 已杀、GPU 空闲）
set -uo pipefail
cd /data/flagos_xx
RES=results/20261119_d4_metax_port/official
mkdir -p "$RES"
MODEL=/workspace/MiniCPM5-2B-w8a8-v2

start_serve() {  # $1 = tag
  local tag=$1
  VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL" --port 9031 \
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
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

run_scenario() {  # $1 = tag, $2 = test-cases json, $3 = 每轮超时秒
  local tag=$1 cases=$2 tmo=$3
  echo "===== scenario $tag ====="
  if ! start_serve "$tag"; then
    echo "FATAL: serve $tag 起不来"; return 1
  fi
  grep -E 'GPU KV cache size|Maximum concurrency' "$RES/serve_${tag}.log" | tail -2 || true
  local r=1
  while [ $r -le 4 ]; do
    if ! pgrep -f 'vllm serve' >/dev/null; then
      echo "!! round $r 前 serve 死亡，重启中"
      sleep 10
      start_serve "$tag" || return 1
    fi
    echo "-- $tag round $r --"
    timeout "$tmo" python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
      --model "$MODEL" --served-model-name minicpm --port 9031 \
      --test-cases "$cases" 2>&1 | tee "$RES/round_${tag}_${r}.log" \
      | grep -E '^Prefill=' || echo "ROUND $r NO OUTPUT"
    r=$((r+1))
  done
  pkill -f 'vllm serve' || true; sleep 8
}

# 先 16K（quick 仅 2 轮，官方 4 轮增值最大）
run_scenario "k16" '[[16384,1024,64,128]]' 3600 || exit 1
run_scenario "k4"  '[[4096,1024,64,256]]'  1800 || exit 1

echo "OFFICIAL_W8A8_DONE — raw CSVs 在 benchmark_results/，轮次日志在 $RES/"
