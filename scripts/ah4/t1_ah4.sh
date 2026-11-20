#!/bin/bash
# A-H4 T1：fused_add_rms_norm IR 优先级 A/B 双形状筛查
# 臂0 基线：VLLM_FL_IR_NORM=0（native inductor triton，serve 现役形态）
# 臂1 治疗：默认 env（fused_add_rms_norm→vllm_c mcoplib C 核优先）
# 判据（开发单§0/§6）：quick 双形状 [[4096,1024,64,32]]/[[4096,1024,64,64]]
#   各 4 pass 丢首、后 3 pass 中位；两形状均 ≥+0.3% 才有合入资格
# v2: serve 存活校验 + case 级重试（12:0x 曾被同机他 会话 pkill 误杀致整臂作废）
# 用法：bash scripts/ah4/t1_ah4.sh（前置：GPU 空闲、插件已自本 worktree pip install）
set -uo pipefail
cd /data/flagos_feat_ah4
RES=results/20261120_ah4/t1
mkdir -p "$RES" benchmark_results
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
CASES=('[[4096,1024,64,32]]' '[[4096,1024,64,64]]')

start_serve() {  # $1=tag $2=extra_env
  local tag=$1 extra=$2
  env $extra VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL" --port 9031 \
    --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
    > "$RES/serve_${tag}.log" 2>&1 &
  for i in $(seq 1 60); do
    sleep 10
    curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm && { echo "serve[$tag] ready ($((i*10))s)"; return 0; }
    pgrep -f 'vllm serve' >/dev/null || { echo "SERVE[$tag] DIED"; tail -20 "$RES/serve_${tag}.log"; return 1; }
  done
  echo "SERVE[$tag] TIMEOUT"; return 1
}

case_ok() {  # $1=bench log
  grep -q "All requests failed" "$1" && return 1
  grep -qE "Successful requests: +0 " "$1" && return 1
  return 0
}

run_case() {  # $1=tag $2=extra_env $3=ci —— 带重试（serve 被外部杀掉则重启重跑本 case）
  local tag=$1 extra=$2 ci=$3 attempt
  for attempt in 1 2 3; do
    timeout 900 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
      --model "$MODEL" --served-model-name minicpm --port 9031 \
      --test-cases "${CASES[$((ci-1))]}" > "$RES/bench_${tag}_case${ci}.log" 2>&1
    if case_ok "$RES/bench_${tag}_case${ci}.log"; then
      echo "case$ci ok (attempt $attempt)"; return 0
    fi
    echo "case$ci INVALID (attempt $attempt)——serve 被杀/客户端挂，重启 serve 重跑"
    pkill -f 'vllm serve' || true; sleep 8
    start_serve "$tag" "$extra" || return 1
  done
  echo "case$ci 三次重试仍失败"; return 1
}

run_arm() {  # $1=tag $2=extra_env
  local tag=$1 extra=$2
  pkill -f 'vllm serve' || true; sleep 8
  start_serve "$tag" "$extra" || exit 1
  grep -E "Final IR op priority" "$RES/serve_${tag}.log" | head -1
  run_case "$tag" "$extra" 1
  run_case "$tag" "$extra" 2
  pkill -f 'vllm serve' || true; sleep 8
}

echo "===== 臂0 基线（IR_NORM=0，native inductor）====="
run_arm base "VLLM_FL_IR_NORM=0"
echo "===== 臂1 治疗（默认 env，vllm_c 优先）====="
run_arm treat ""

echo "===== pass 值提取（每臂每形状 4 pass 丢首，后3中位）====="
for tag in base treat; do
  for c in 4096,1024,64,32 4096,1024,64,64; do
    echo "-- $tag case[$c]:"
    ls -t benchmark_results/raw_runs_*.csv | head -2 | xargs -r cat 2>/dev/null | grep "$c" | tail -4
  done
done
echo "T1_DONE — 判据：两形状 treat/base 后3pass中位均 ≥+0.3%"
