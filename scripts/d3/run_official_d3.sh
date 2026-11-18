#!/bin/bash
# D3 官方 bench 执行器（用户确认后启动；GPU 串行，逐臂新鲜 serve）
# 用法: setsid nohup bash scripts/d3/run_official_d3.sh <arms> > chain.log 2>&1 &
#   arms: 空格分隔的臂名，可选 default B A（按序执行）
set -u
REPO=/flagos_比赛/flagos_xx
OUT_ROOT=$REPO/results/d3_official_$(date +%m%d_%H%M)
mkdir -p "$OUT_ROOT"
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
PORT=9031
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT_ROOT/chain.log"; }

run_arm() {
    local ARM=$1
    local DIR=$OUT_ROOT/$ARM
    mkdir -p "$DIR"
    local ENVS=""
    case $ARM in
        default) ENVS="" ;;
        B) ENVS="VLLM_FL_ATTN_TILE=1 VLLM_FL_ATTN_TD=1" ;;
        A) ENVS="VLLM_FL_ATTN_TILE=1 VLLM_FL_KV_INT8=1" ;;
        *) LOG "unknown arm $ARM"; return 1 ;;
    esac
    LOG "== 臂 $ARM 开始：起 serve（env: ${ENVS:-默认}）=="
    cd /workspace || return 1
    setsid nohup env VLLM_PLUGINS=fl $ENVS vllm serve $MODEL \
        --port $PORT --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --served-model-name minicpm --gpu-memory-utilization 0.85 \
        --max-model-len 131072 > "$DIR/serve.log" 2>&1 &
    local SPID=$!
    echo "$SPID" > "$DIR/serve.pid"
    local ok=0
    for i in $(seq 1 66); do
        sleep 10
        grep -q "Application startup complete" "$DIR/serve.log" 2>/dev/null && { ok=1; break; }
        # 死亡判据 = vllm serve 主进程消失（EngineCore 早期未孵化是正常现象，
        # 首个 10s 窗口后再确认一次防竞态）
        if ! pgrep -f "vllm serve $MODEL" >/dev/null; then
            sleep 20
            pgrep -f "vllm serve $MODEL" >/dev/null || { ok=0; break; }
        fi
    done
    if [ $ok -ne 1 ]; then
        LOG "臂 $ARM serve 启动失败，跳过"; tail -15 "$DIR/serve.log" >> "$OUT_ROOT/chain.log"
        kill -KILL $SPID 2>/dev/null
        return 1
    fi
    LOG "臂 $ARM serve 就绪（pool: $(grep -m1 -o 'GPU KV cache size: [0-9,]*' "$DIR/serve.log")）"
    LOG "臂 $ARM 官方 bench 开跑（两场景 × 4 轮，~2.5h）"
    (cd /workspace && python3 $REPO/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
        --model $MODEL --served-model-name minicpm --port $PORT \
        --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' \
        2>&1 | tee "$DIR/benchmark.log" | grep -E "Prefill=|Successful requests")
    LOG "臂 $ARM bench 退出码 $?"
    LATEST=$(ls -t /workspace/benchmark_results/summary_*.csv 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
        cp "$LATEST" "$DIR/"
        cp "${LATEST/summary_/raw_runs_}" "$DIR/" 2>/dev/null
        LOG "臂 $ARM summary: $(tail -2 "$DIR/$(basename "$LATEST")" | cut -c1-120)"
    fi
    LOG "臂 $ARM 杀 serve 收队"
    for p in $(ps -eo pid,cmd | grep -E "vllm serve|EngineCore" | grep -v grep | awk '{print $1}'); do
        kill -KILL "$p" 2>/dev/null
    done
    sleep 5
}

for ARM in "${@:-default B A}"; do
    run_arm "$ARM"
done
LOG "ALL ARMS DONE -> $OUT_ROOT"
