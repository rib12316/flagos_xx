#!/usr/bin/env bash
# D2 T2 端到端验证：serve 转换 checkpoint（--model 指向 /workspace/MiniCPM5-2B-w8a8，
# 其余官方参数与 run_t1_d1.sh 一字不动——--model 换目录属部署步骤，判读见 results/
# d2-precision-search.md 与 readme 草稿）→ 激活确认三查（W8A8 scheme 装载 / 权重显存
# −1.84GiB / KV tokens ≈568K）→ 官方 bench 双场景（对照 main 基线 4K 2668.42 /
# 16K 1276.62 不回退）→ 官方配方 105 题精度（≥0.955 维持）。用法:
#   bash scripts/d2/run_d2_eval.sh [bench|eval|all]   # 默认 all
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${1:-all}"
PORT=9031
MODEL_PATH="${D2_MODEL:-/workspace/MiniCPM5-2B-w8a8}"
SERVED_NAME=minicpm
OUT="${D2_OUT:-$REPO_ROOT/results/20261111_d2_w8a8_full}"
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

t0_d2_check() {
    echo "== T0: W8A8 激活确认三查 =="
    echo "-- 1) packed W8A8 scheme 装载与量化层数:"
    grep -m3 -iE "pack|w8a8|quant" "$OUT/serve.log" | head -5 || true
    echo "-- 2) KV 池（权重 −1.84GiB 释放 → 应显著高于 bf16 参照 ~518,096 tokens，预期 ≈568K）:"
    grep -m2 "GPU KV cache size" "$OUT/serve.log" || true
    echo "-- 3) 权重显存（对比 bf16 装载行）:"
    grep -m2 -iE "loading.*took|model weights" "$OUT/serve.log" || true
    { echo "[t0_d2_check @ $(date -Is)] model=$MODEL_PATH"; } >> "$OUT/PROVENANCE"
}

run_bench() {
    echo "== T1: 官方口径 bench（两场景 × 4 轮丢首；对照 4K 2668.42 / 16K 1276.62 不回退）=="
    (cd /workspace && python3 "$REPO_ROOT/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py" \
      --model "$MODEL_PATH" \
      --served-model-name "$SERVED_NAME" \
      --port "$PORT" \
      --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
      2>&1 | tee "$OUT/benchmark.log" | tail -8)
    LATEST=$(ls -t /workspace/benchmark_results/summary_*.csv 2>/dev/null | head -1)
    LATEST_RAW="${LATEST/summary_/raw_runs_}"; cp "$LATEST" "$OUT/"; cp "$LATEST_RAW" "$OUT/" 2>/dev/null || true
    echo "summary: $OUT/$(basename "$LATEST")"
}

run_eval() {
    echo "== 精度: evalscope math_500 Level 3 官方配方（D2 门槛 ≥0.955；基线 0.962）=="
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
t0_d2_check
[[ "$STAGE" == "eval" ]] && { run_eval; pkill -f "vllm serve"; exit 0; }
[[ "$STAGE" == "bench" || "$STAGE" == "all" ]] && run_bench
[[ "$STAGE" == "all" ]] && run_eval
pkill -f "vllm serve" 2>/dev/null || true
echo "完成。结果: $OUT"
