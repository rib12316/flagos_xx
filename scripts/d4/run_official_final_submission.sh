#!/bin/bash
# final-submition 官方全量复测 runner（用户 2026-11-20 明示批准执行）。
# 顺序：先精度（evalscope math_500 L3 105 题）→ 4K bench → 16K bench。
# serve 形态 = DELIVERY 定稿官方锁死：零附加 env、无 --compilation-config、
# --model 指向 w8a8 ckpt（其余与官方 4.3 沐曦命令逐字一致）。
# bench = 官方脚本原件（与上游 diff 为空已核），4 轮丢首取均值，每轮硬超时
# （d4 已知"客户端落盘后汇总挂死"对策：CSV 数据完整，汇总从 CSV 提取）。
# 用法：bash scripts/d4/run_official_final_submission.sh   （前置：serve 已杀、GPU 空闲）
set -uo pipefail
cd /data/flagos_final_submition
RES=results/20261120_final_submission
mkdir -p "$RES"
MODEL_SERVE=/workspace/MiniCPM5-2B-w8a8-v2
MODEL_CLIENT=/workspace/MiniCPM5-2B
MASTER="$RES/master.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

provenance() {
  { echo "== PROVENANCE $(date '+%F %T') ==";
    echo "git: $(git rev-parse HEAD) ($(git log -1 --format=%h))"
    echo "git status dirty files: $(git status --porcelain | wc -l)"
    echo "plugin: $(pip show vllm-plugin-fl 2>/dev/null | grep -E 'Version|Location' | tr '\n' ' ')"
    python3 -c 'import vllm; print("vllm:", vllm.__version__)'
    mx-smi | sed -n '5,12p'; } | tee "$RES/PROVENANCE"
}

wait_gpu_idle() {
  local i
  for i in $(seq 1 24); do
    if ! mx-smi 2>/dev/null | grep -qE 'python|vllm'; then return 0; fi
    sleep 5
  done
  log "WARN: GPU 进程未清空，继续（可能占显存）"
}

start_serve() {  # $1 = tag
  local tag=$1
  wait_gpu_idle
  VLLM_PLUGINS=fl setsid nohup vllm serve "$MODEL_SERVE" --port 9031 \
    --served-model-name minicpm --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    > "$RES/serve_${tag}.log" 2>&1 &
  local i
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm; then
      log "serve[$tag] ready after $((i*10))s"; return 0
    fi
    if ! pgrep -f 'vllm serve' >/dev/null; then
      log "FATAL: serve[$tag] DIED"; tail -40 "$RES/serve_${tag}.log" | tee -a "$MASTER"; return 1
    fi
  done
  log "FATAL: serve[$tag] TIMEOUT(900s)"; return 1
}

audit_lines() {  # $1 = tag —— 默认融合生效证据（非致命，缺失则警示）
  local tag=$1
  { echo "== serve[$tag] audit lines ==";
    grep -E 'w8a8_dynamic_per_token_linear|flagos_mctlass|F-c grouped|D13|rope|REGISTER neox|IR.*priority|Capturing CUDA graphs|fusion' \
      "$RES/serve_${tag}.log" | sort -u | head -25; } | tee -a "$MASTER"
}

smoke() {  # 官方 4.3 简单测试逐字（max_tokens 256）
  local resp
  resp=$(python3 - <<'PY'
import requests, json, sys
try:
    r = requests.post('http://127.0.0.1:9031/v1/chat/completions',
        headers={'Content-Type': 'application/json'}, timeout=300,
        json={'model': 'minicpm', 'messages': [{'role': 'user', 'content': 'introduce llm'}],
              'max_tokens': 256})
    d = r.json()
    print("SMOKE_OK" if 'choices' in d and d['choices'][0].get('message', {}).get('content') \
          else f"SMOKE_BAD:{json.dumps(d)[:300]}")
except Exception as e:
    print(f"SMOKE_FAIL:{e}")
PY
)
  log "smoke: $resp"
  [[ "$resp" == SMOKE_OK* ]]
}

kill_serve() {
  pkill -f 'vllm serve' 2>/dev/null || true
  sleep 8
}

run_accuracy() {
  log "===== STAGE 1: 精度（105 题）====="
  if ! start_serve acc; then return 1; fi
  audit_lines acc
  if ! smoke; then kill_serve; return 1; fi
  log "evalscope 启动（官方配方逐字）"
  timeout 14400 evalscope eval \
    --model minicpm \
    --api-url http://127.0.0.1:9031/v1/chat/completions \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets math_500 \
    --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
    --eval-batch-size 8 \
    --timeout 3600 \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
    --work-dir /workspace/evalscope-datasets/level3 \
    --ignore-errors > "$RES/evalscope.log" 2>&1
  local rc=$?
  log "evalscope rc=$rc（超时=124 也不代表失败，看日志与报告）"
  tail -20 "$RES/evalscope.log" | tee -a "$MASTER"
  kill_serve
  return 0
}

run_scenario() {  # $1 = tag, $2 = test-cases json, $3 = 每轮超时秒
  local tag=$1 cases=$2 tmo=$3 r
  log "===== STAGE bench $tag ====="
  if ! start_serve "bench_$tag"; then return 1; fi
  audit_lines "bench_$tag"
  if ! smoke; then kill_serve; return 1; fi
  r=1
  while [ $r -le 4 ]; do
    if ! pgrep -f 'vllm serve' >/dev/null; then
      log "!! round $r 前 serve 死亡，重启"
      sleep 10
      start_serve "bench_${tag}_r${r}" || return 1
    fi
    log "-- $tag round $r/$tmo s --"
    timeout "$tmo" python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
      --model "$MODEL_CLIENT" --served-model-name minicpm --port 9031 \
      --test-cases "$cases" > "$RES/round_${tag}_${r}.log" 2>&1
    log "round $r rc=$?；tail: $(tail -1 "$RES/round_${tag}_${r}.log" | head -c 200)"
    r=$((r+1))
  done
  kill_serve
}

provenance
run_accuracy || { log "ABORT: 精度阶段失败"; exit 1; }
run_scenario "k4"  '[[4096,1024,64,256]]'  1800 || { log "ABORT: 4K 失败"; exit 1; }
run_scenario "k16" '[[16384,1024,64,128]]' 3600 || { log "ABORT: 16K 失败"; exit 1; }
log "ALL_DONE — raw CSVs 在 benchmark_results/，轮次日志在 $RES/"
