#!/bin/bash
# V150 合并树(afa5fda: 四项含 W8A8-TN + P1 BM 默认开)官方全量验证。
# 形态 = 官方 4.3 逐字：零附加 env、含 compilation-config、仅 --model 指向 w8a8-v2。
# 顺序：精度 105 题 → 4K → 16K。bench 单次调用（脚本内部 RUNS=4/SKIP_FIRST=1）。
set -uo pipefail
cd /flagos_比赛/final_build
RES=results/20261120_v150_final_tn
MODEL_SERVE=/workspace/MiniCPM5-2B-w8a8-v2
MODEL_CLIENT=/workspace/MiniCPM5-2B
PORT=9031
MASTER=$RES/master.log
mkdir -p "$RES"

# 清空环境：剥离一切 VLLM_FL_*/P3_DIAG_*，只留官方要求的 VLLM_PLUGINS=fl
for k in $(env | grep -oE '^(VLLM_FL_[A-Z0-9_]+|P3_DIAG_[A-Z0-9_]+)' || true); do unset "$k"; done
export VLLM_PLUGINS=fl

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MASTER"; }

kill_serve() {
  local pids
  pids=$(pgrep -f 'vllm serve' 2>/dev/null || true)
  [ -n "$pids" ] && kill $pids 2>/dev/null || true
  sleep 8
  pids=$(pgrep -f 'vllm serve' 2>/dev/null || true)
  [ -n "$pids" ] && kill -9 $pids 2>/dev/null || true
  sleep 4
}

start_serve() {  # $1 = tag
  kill_serve
  local used
  used=$(ixsmi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  [ "${used:-0}" -gt 3000 ] && log "WARN: GPU mem ${used}MiB 非空"
  nohup vllm serve "$MODEL_SERVE" \
    --port $PORT \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --served-model-name minicpm \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    > "$RES/serve_$1.log" 2>&1 &
  local i
  for i in $(seq 1 90); do
    sleep 10
    if curl -s -m 3 http://127.0.0.1:$PORT/v1/models | grep -q minicpm; then
      log "serve[$1] ready after $((i*10))s"; return 0
    fi
    if ! pgrep -f 'vllm serve' >/dev/null; then
      log "FATAL: serve[$1] DIED"; tail -40 "$RES/serve_$1.log" | tee -a "$MASTER"; return 1
    fi
  done
  log "FATAL: serve[$1] TIMEOUT(900s)"; return 1
}

audit() {  # $1 = tag
  { echo "== serve[$1] audit =="; grep -E 'ixformer|TN|attn_bm|fl.attn-launch|Capturing CUDA graphs|GPU KV cache size|flagos_intmm' "$RES/serve_$1.log" | sort -u | head -15; } | tee -a "$MASTER"
}

smoke() {
  local resp
  resp=$(python3 - <<'PY'
import requests, json
try:
    r = requests.post('http://127.0.0.1:9031/v1/chat/completions',
        headers={'Content-Type': 'application/json'}, timeout=300,
        json={'model': 'minicpm', 'messages': [{'role': 'user', 'content': 'introduce llm'}],
              'max_tokens': 256})
    d = r.json()
    print("SMOKE_OK" if 'choices' in d and d['choices'][0].get('message', {}).get('content')
          else f"SMOKE_BAD:{json.dumps(d)[:300]}")
except Exception as e:
    print(f"SMOKE_FAIL:{e}")
PY
)
  log "smoke: ${resp:0:80}"
  [[ "$resp" == SMOKE_OK* ]]
}

run_accuracy() {
  log "===== STAGE 1: 精度 105 题 ====="
  start_serve acc || return 1
  audit acc
  smoke || { kill_serve; return 1; }
  timeout 14400 evalscope eval \
    --model minicpm \
    --api-url http://127.0.0.1:$PORT/v1/chat/completions \
    --api-key EMPTY \
    --eval-type openai_api \
    --datasets math_500 \
    --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
    --eval-batch-size 8 \
    --timeout 3600 \
    --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
    --work-dir /workspace/evalscope-datasets/level3 \
    --ignore-errors > "$RES/evalscope.log" 2>&1
  log "evalscope rc=$?"
  grep -E 'Accuracy|Score' "$RES/evalscope.log" | tail -3 | tee -a "$MASTER"
  kill_serve
}

run_scenario() {  # $1 = tag, $2 = cases
  log "===== STAGE bench $1 ====="
  start_serve "bench_$1" || return 1
  audit "bench_$1"
  smoke || { kill_serve; return 1; }
  (cd /workspace && timeout 5400 python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model "$MODEL_CLIENT" --served-model-name minicpm --port $PORT \
    --test-cases "$2") > "$RES/bench_$1.log" 2>&1
  log "bench $1 rc=$?；summary:"
  ls -t /workspace/benchmark_results/summary_*.csv 2>/dev/null | head -1 | xargs -r tail -2 | tee -a "$MASTER"
  kill_serve
}

{ echo "== PROVENANCE $(date '+%F %T') ==";
  echo "tree: afa5fdada6ca721bcfb34a31ef124af8809ab2af (final-submition merge)";
  echo "env VLLM*: $(env | grep -c '^VLLM' || true) vars (expect 1: VLLM_PLUGINS)";
  pip show vllm-plugin-fl 2>/dev/null | grep -E 'Version|Location'; } | tee "$RES/PROVENANCE"

run_accuracy || { log "ABORT: 精度失败"; exit 1; }
run_scenario k4 '[[4096,1024,64,256]]' || { log "ABORT: 4K"; exit 1; }
run_scenario k16 '[[16384,1024,64,128]]' || { log "ABORT: 16K"; exit 1; }
log "ALL_DONE"
