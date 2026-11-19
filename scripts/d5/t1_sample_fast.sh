#!/bin/bash
# D5 T1：采样快路径 A/B 单场景筛查（4K quick case=与B机交叉锚点同协议）
# 臂0 基线：VLLM_FL_SAMPLE_FAST=0（vendor fallback 路径）——同时=A机交叉锚点
# 臂1 快路径：默认 env（top-k1024+阈值法）
# 判据：同 case Total tok/s 提升幅度 + serve 日志命中率统计（快路径命中/回退）
# 用法：bash scripts/d5/t1_sample_fast.sh（前置：GPU 空闲）
set -uo pipefail
cd /data/flagos_xx
RES=results/20261120_c500_reprofiling/t1_sample_fast
mkdir -p "$RES"
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
CASE='[[4096,1024,64,32]]'

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

run_arm() {  # $1=tag $2=extra_env
  local tag=$1 extra=$2
  pkill -f 'vllm serve' || true; sleep 8
  start_serve "$tag" "$extra" || exit 1
  grep -E 'sample_fast_path|w8a8_dynamic.*mctlass' "$RES/serve_${tag}.log" | head -3
  timeout 900 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL" --served-model-name minicpm --port 9031 \
    --test-cases "$CASE" > "$RES/bench_${tag}.log" 2>&1 \
    || echo "bench $tag 客户端挂死/超时——从 raw CSV 提取"
  grep -E 'sample_fast_path: 快路径命中' "$RES/serve_${tag}.log" | tail -2 || true
  pkill -f 'vllm serve' || true; sleep 8
}

echo "===== 臂0 基线（SAMPLE_FAST=0，兼A机交叉锚点）====="
run_arm base "VLLM_FL_SAMPLE_FAST=0"
echo "===== 臂1 快路径（默认 env）====="
run_arm fast ""

echo "===== pass 值提取（两臂各 4 pass 丢首）====="
for tag in base fast; do
  echo "-- $tag:"
  ls -t benchmark_results/raw_runs_*.csv | head -3 | xargs -r cat 2>/dev/null \
    | grep '4096,1024' | tail -4
done
echo "T1_DONE — 对比丢首均值；判据 ≥+3%（墙图口径 sort 链占步时 ~35-50%）即升级 T2"
