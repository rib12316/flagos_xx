#!/usr/bin/env bash
# D1 步骤 2 · 前置裁决轮：抢占是否存在 + 跨轮 APC 命中率（E1-E4 证据一次拿齐）。
# serve 参数与官方一字不差（同 run_t1_d0.sh），仅 env 增加 VLLM_FL_SCHED_FL=1（诊断注入，
# 行为不变）；bench 用 vllm bench serve 原生命令行 + --save-result。
# 用法: bash scripts/run_d1_adjudication.sh
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=9031
MODEL_PATH=/workspace/MiniCPM5-2B
SERVED_NAME=minicpm
OUT="$REPO_ROOT/results/20261104_d1_adjudication"
mkdir -p "$OUT"

# ---- PROVENANCE ----
{
  echo "date: $(date -Is)"
  echo "git_rev: $(git -C "$REPO_ROOT" rev-parse HEAD)"
  echo "git_branch: $(git -C "$REPO_ROOT" branch --show-current)"
  echo "git_dirty: $(git -C "$REPO_ROOT" status --porcelain | wc -l) files"
  echo "env: VLLM_PLUGINS=fl VLLM_FL_SCHED_FL=1 SCRAPE_INTERVAL=2"
  echo "serve_cmd: vllm serve $MODEL_PATH --port $PORT --compilation-config {cudagraph_mode: FULL_DECODE_ONLY} --served-model-name $SERVED_NAME --gpu-memory-utilization 0.85 --max-model-len 131072"
  echo "purpose: D1 adjudication (preemption + cross-round APC hit rate)"
} > "$OUT/PROVENANCE"

# ---- 起服（官方参数逐字）+ metrics 抓取 ----
pkill -f "vllm serve" 2>/dev/null || true; sleep 3
pkill -f scrape_metrics.sh 2>/dev/null || true
VLLM_PLUGINS=fl VLLM_FL_SCHED_FL=1 nohup vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
echo "serve PID=$SERVE_PID"
nohup bash "$REPO_ROOT/scripts/scrape_metrics.sh" "$PORT" "$OUT" >/dev/null 2>&1 &
echo "scraper PID=$!"
for i in $(seq 1 360); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { echo "服务就绪(${i}x5s)"; break; }
    sleep 5
    if ! kill -0 $SERVE_PID 2>/dev/null; then echo "❌ serve 进程退出"; tail -40 "$OUT/serve.log"; exit 1; fi
done

# ---- 骨架生效检查（T0）+ 冒烟（防呆：起服 6min 白等）----
echo "== T0: SchedulerFL 生效检查 =="
grep -m1 "SchedulerFL(diag) banner" "$OUT/serve.log" || { echo "❌ banner 未出现"; tail -30 "$OUT/serve.log"; pkill -f "vllm serve"; exit 1; }
grep -m2 "GPU KV cache size\|GPU KV cache" "$OUT/serve.log" || true
SMOKE=$(curl -s http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"minicpm","prompt":"The capital of France is","max_tokens":12,"temperature":0}')
echo "smoke: ${SMOKE:0:200}"
echo "$SMOKE" | grep -q '"text"' || { echo "❌ 冒烟失败（EngineCore 可能已崩）"; pkill -f "vllm serve"; exit 1; }
grep -c "\[fl.sched\]" "$OUT/serve.log" || true

run_case() {  # $1=input_len $2=num_prompts $3=prefix
    echo "== bench $3: input=$1 prompts=$2 $(date -Is) =="
    (cd /workspace && vllm bench serve --backend vllm \
      --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
      --endpoint /v1/completions --host localhost --port "$PORT" \
      --dataset-name random --ignore-eos \
      --random-input-len "$1" --random-output-len 1024 \
      --max-concurrency 64 --num-prompts "$2" \
      --seed 0 --save-result --result-dir "$OUT" --result-filename "$3.json" \
      2>&1 | tee "$OUT/bench_$3.log" | grep -E "Successful|Benchmark duration|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT")
}

run_case 16384 128 16k_run1
run_case 16384 128 16k_run2
run_case 4096  256 4k_run1
run_case 4096  256 4k_run2

pkill -f scrape_metrics.sh 2>/dev/null || true
echo "== 完成 $(date -Is)，serve 保留运行（供后续检查），产物: $OUT =="
tail -5 "$OUT/metrics.log" || true
