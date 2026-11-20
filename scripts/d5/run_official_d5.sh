#!/bin/bash
# T2 官方全量复测 runner（D5 二期口径，2026-11-20）
# 前置（缺一不跑）：
#   1) 用户明示批准（红线）
#   2) 待测代码已合入本仓且 pip 本仓重装 + provenance 四查
#   3) GPU 空闲（mx-smi 无进程）
# 形态：官方终局逐字——零附加 env、无 --compilation-config、每场景新鲜 serve、
#       bench 原件 4 轮丢首均值；另跑 105 题精度（evalscope 官方配方）。
# 对策内置：serve 死亡自动重启（EngineCore 瞬态死亡前科）+ bench 客户端挂死
#           timeout -k 强杀 + raw CSV 兜底取数（DELIVERY 先例口径）。
# 用法：bash scripts/d5/run_official_d5.sh [k16|k4|both|eval]
set -uo pipefail
cd /data/flagos_xx
STAGE=results/20261120_t2_official
mkdir -p "$STAGE"
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
MODE="${1:-both}"

start_serve() {  # $1=tag
  local tag=$1
  VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL" --port 9031 \
    --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
    > "$STAGE/serve_${tag}.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 10
    if curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm; then
      echo "serve[$tag] ready ($((i*10))s)"; return 0
    fi
    pgrep -f 'vllm serve' >/dev/null || { echo "SERVE[$tag] DIED"; tail -30 "$STAGE/serve_${tag}.log"; return 1; }
  done
  echo "SERVE[$tag] TIMEOUT"; return 1
}

kill_serve() { pkill -9 -f 'vllm serve' 2>/dev/null; sleep 3; ps aux | grep '[E]ngineCore' | awk '{print $2}' | xargs -r kill -9; sleep 8; }

run_scenario() {  # $1=tag $2=cases $3=每轮超时秒
  local tag=$1 cases=$2 tmo=$3
  echo "===== scenario $tag ====="
  kill_serve
  start_serve "$tag" || exit 1
  grep -E 'GPU KV cache size|flagos_mctlass' "$STAGE/serve_${tag}.log" | tail -2 || true
  echo "-- $tag bench（bench 内部 4 pass 丢首取均值）--"
  timeout -k 10 "$tmo" python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name minicpm --port 9031 \
    --test-cases "$cases" > "$STAGE/bench_${tag}.log" 2>&1 \
    || echo "客户端挂死/超时——从 raw CSV 取数（先例口径）"
  grep -E '^Prefill=|Total tok/s' "$STAGE/bench_${tag}.log" | tail -2 || true
  kill_serve
}

run_eval() {
  echo "===== 105 题精度（evalscope 官方配方逐字，同 20261119 final）====="
  start_serve eval || exit 1
  timeout -k 10 7200 evalscope eval \
    --model minicpm \
    --api-url "http://127.0.0.1:9031/v1/chat/completions" \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets math_500 \
    --dataset-args "{\"math_500\": {\"dataset_id\": \"/workspace/evalscope-datasets/math_500\", \"subset_list\": [\"Level 3\"]}}" \
    --eval-batch-size 8 \
    --timeout 3600 \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
    --work-dir /workspace/evalscope-datasets/level3 \
    --ignore-errors \
    > "$STAGE/eval105.log" 2>&1 || echo "eval 客户端异常——查 $STAGE/eval105.log"
  grep -iE 'accuracy|score' "$STAGE/eval105.log" | tail -3 || true
  kill_serve
}

case "$MODE" in
  k16) run_scenario k16 '[[16384,1024,64,128]]' 3600 ;;
  k4)  run_scenario k4  '[[4096,1024,64,256]]'  2400 ;;
  both) run_scenario k16 '[[16384,1024,64,128]]' 3600 && run_scenario k4 '[[4096,1024,64,256]]' 2400 ;;
  eval) run_eval ;;
esac
echo "T2_STAGE_DONE[$MODE] — raw: benchmark_results/raw_runs_*.csv；产物：$STAGE/"
