#!/usr/bin/env bash
# D1 步骤 3 · T0.5 预门：带指定 env 起服（官方参数逐字）→ T0 生效检查+冒烟 → 16K 单轮。
# 用法: VARS="VLLM_FL_WATERMARK=0.03" PREFIX=wm003 [ROUNDS=1] bash scripts/run_d1_t05.sh
# 对照：冷轮=裁决 16k_run1 1190.52；热轮=裁决 16k_run2 1264.41（ROUNDS=2 时第二热轮可比）
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT=9031
MODEL_PATH=/workspace/MiniCPM5-2B
SERVED_NAME=minicpm
PREFIX="${PREFIX:-arm}"
ROUNDS="${ROUNDS:-1}"
OUT="$REPO_ROOT/results/20261105_d1_t05_${PREFIX}"
mkdir -p "$OUT"
echo "env: $VARS" | tee "$OUT/PROVENANCE"
echo "git: $(git -C "$REPO_ROOT" rev-parse HEAD) $(git -C "$REPO_ROOT" branch --show-current)" >> "$OUT/PROVENANCE"
echo "date: $(date -Is)" >> "$OUT/PROVENANCE"

pkill -f "vllm serve" 2>/dev/null || true; sleep 3
for i in $(seq 1 20); do
    MEM=$(ixsmi 2>/dev/null | grep -oE "[0-9]+MiB / 32768MiB" | head -1 | grep -oE "^[0-9]+" || echo 0)
    [[ "${MEM:-0}" -lt 1000 ]] && break
    pgrep -f "vllm|EngineCore" 2>/dev/null | while read -r pid; do kill -9 "$pid" 2>/dev/null; done
    sleep 5
done

env VLLM_PLUGINS=fl VLLM_FL_SCHED_FL=1 $VARS nohup vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
    --served-model-name "$SERVED_NAME" \
    --gpu-memory-utilization 0.85 \
    --max-model-len 131072 \
    > "$OUT/serve.log" 2>&1 &
SERVE_PID=$!
echo "serve PID=$SERVE_PID (env: $VARS)"
T0=$(date +%s)
for i in $(seq 1 400); do
    curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && break
    sleep 5
    if ! kill -0 $SERVE_PID 2>/dev/null; then echo "❌ serve 退出"; tail -30 "$OUT/serve.log"; exit 1; fi
done
echo "boot $(( $(date +%s) - T0 ))s"

BANNER=""
for i in $(seq 1 36); do
    BANNER=$(grep -m1 "SchedulerFL(diag) banner" "$OUT/serve.log" || true)
    [[ -n "$BANNER" ]] && break; sleep 5
done
echo "$BANNER" || { echo "❌ 无 banner"; exit 1; }
grep -m2 "GPU KV cache size" "$OUT/serve.log" || true
SMOKE=$(curl -s http://127.0.0.1:$PORT/v1/completions -H "Content-Type: application/json" \
  -d '{"model":"minicpm","prompt":"The capital of France is","max_tokens":12,"temperature":0}')
echo "$SMOKE" | grep -q '"text"' || { echo "❌ 冒烟失败: ${SMOKE:0:200}"; exit 1; }
echo "smoke OK"

for r in $(seq 1 "$ROUNDS"); do
    echo "== round $r start $(date -Is) =="
    (cd /workspace && vllm bench serve --backend vllm \
      --model "$SERVED_NAME" --tokenizer "$MODEL_PATH" \
      --endpoint /v1/completions --host localhost --port "$PORT" \
      --dataset-name random --ignore-eos \
      --random-input-len 16384 --random-output-len 1024 \
      --max-concurrency 64 --num-prompts 128 \
      --seed 0 --save-result --result-dir "$OUT" --result-filename "16k_r${r}.json" \
      2>&1 | tee "$OUT/bench_r${r}.log" | grep -E "Successful|Benchmark duration|Total token throughput|Mean TTFT")
done
echo "== preempt 计数: $(grep -o 'cum=[0-9]*' "$OUT/serve.log" | tail -1)"
echo "== 完成 $(date -Is)：$OUT（serve 已停）"
pkill -f "vllm serve" 2>/dev/null || true
