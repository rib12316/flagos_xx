#!/usr/bin/env bash
# D0 T1 验证流水线：起服务(官方参数,一字不动) → T0 确认新配置生效(grep dispatch 日志)
#   → 官方口径 bench(4轮丢首) → evalscope math_500 L3。用法:
#   bash scripts/run_t1_d0.sh [bench|eval|all]   # 默认 all
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE="${1:-all}"
PORT=9031
MODEL_PATH=/workspace/MiniCPM5-2B
SERVED_NAME=minicpm
OUT="$REPO_ROOT/results/20261104_d0_t1"
mkdir -p "$OUT"

start_serve() {
    pkill -f "vllm serve" 2>/dev/null || true; sleep 3
    VLLM_PLUGINS=fl nohup vllm serve "$MODEL_PATH" \
      --port "$PORT" \
      --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
      --served-model-name "$SERVED_NAME" \
      --gpu-memory-utilization 0.85 \
      --max-model-len 131072 \
      > "$OUT/serve.log" 2>&1 &
    echo "serve PID=$!"
    for i in $(seq 1 360); do
        curl -s -o /dev/null "http://127.0.0.1:$PORT/v1/models" && { echo "服务就绪(${i}x5s)"; return 0; }
        sleep 5
    done
    echo "❌ 30 分钟未就绪"; tail -30 "$OUT/serve.log"; return 1
}

t0_dispatch_check() {
    echo "== T0: dispatch 层新配置生效检查（serve.log 中 fl.attn-launch 命中行）=="
    grep -c "fl.attn-launch" "$OUT/serve.log" || true
    grep "attn_launch_config: 已装载" "$OUT/serve.log" || true
    grep "fl.attn-launch" "$OUT/serve.log" | sort -u | head -10 || true
}

run_bench() {
    echo "== T1: 官方口径 bench（两场景 × 4 轮丢首）=="
    (cd /workspace && python3 "$REPO_ROOT/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_NAME" \
      --port "$PORT" \
      --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
      2>&1 | tee "$OUT/benchmark.log" | tail -8)
    LATEST=$(ls -t /workspace/benchmark_results/summary_*.csv 2>/dev/null | head -1)
    cp "$LATEST" "$OUT/" && cp "/workspace/benchmark_results/raw_runs_$(basename "$LATEST" .csv | sed 's/summary_//')" "$OUT/ 2>/dev/null" || true
    echo "summary: $OUT/$(basename "$LATEST")"
}

run_eval() {
    echo "== 精度: evalscope math_500 Level 3（红线 ≥0.95，基线 0.962）=="
    evalscope eval \
      --model "$SERVED_NAME" \
      --api-url "http://127.0.0.1:$PORT/v1/chat/completions" \
      --api-key EMPTY \
      --eval-type openai_api \
      --datasets math_500 \
      --dataset-args "{\"math_500\": {\"dataset_id\": \"/workspace/evalscope-datasets/math_500\", \"subset_list\": [\"Level 3\"]}}" \
      --eval-batch-size 8 \
      --timeout 3600 \
      --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
      --work-dir /workspace/evalscope-datasets/level3 \
      --ignore-errors \
      2>&1 | tee "$OUT/evalscope.log" | grep -iE "accuracy|score" | tail -5
}

start_serve || exit 1
t0_dispatch_check
[[ "$STAGE" == "eval" ]] && { run_eval; pkill -f "vllm serve"; exit 0; }
[[ "$STAGE" == "bench" || "$STAGE" == "all" ]] && run_bench
[[ "$STAGE" == "all" ]] && run_eval
pkill -f "vllm serve" 2>/dev/null || true
echo "完成。结果: $OUT"
