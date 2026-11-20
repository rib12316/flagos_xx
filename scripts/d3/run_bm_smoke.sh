#!/bin/bash
# BM 联调烟测:单场景 × 新鲜 serve(仿 run_official_redo.sh 协议,EngineCore
# 死亡钟下每单元独立 serve + 收队全杀)。C 臂 = ②采样grid + BM重标定,
# ①⑤ 显式关闭(TD 与 TILE=128 互斥;⑤ 被 BM 覆盖)。
# 用法: setsid nohup bash run_bm_smoke.sh C16 C4 > smoke.log 2>&1 &
set -u
REPO=/flagos_比赛/flagos_xx
BASE=/flagos_比赛/flagos_xx/results/20261120_bm_smoke
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
PORT=9032
mkdir -p "$BASE"
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$BASE/smoke_chain.log"; }

unit_cfg() {
    case $1 in
        C16) ENVS="VLLM_FL_ATTN_BM=1 VLLM_FL_ATTN_TILE=0 VLLM_FL_ATTN_TD=0"; CASE='[[16384,1024,64,128]]' ;;
        C4)  ENVS="VLLM_FL_ATTN_BM=1 VLLM_FL_ATTN_TILE=0 VLLM_FL_ATTN_TD=0"; CASE='[[4096,1024,64,256]]' ;;
        *) return 1 ;;
    esac
}

for UNIT in "$@"; do
    DIR=$BASE/${UNIT}
    mkdir -p "$DIR"
    unit_cfg "$UNIT" || { LOG "unknown unit $UNIT"; continue; }
    LOG "== 烟测 $UNIT 开始（env: $ENVS, case: $CASE）=="
    cd /workspace
    setsid nohup env VLLM_PLUGINS=fl $ENVS vllm serve $MODEL \
        --port $PORT --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --served-model-name minicpm --gpu-memory-utilization 0.85 \
        --max-model-len 131072 > "$DIR/serve.log" 2>&1 &
    SPID=$!
    ok=0
    for i in $(seq 1 90); do
        sleep 10
        grep -q "Application startup complete" "$DIR/serve.log" 2>/dev/null && { ok=1; break; }
        if ! kill -0 $SPID 2>/dev/null; then
            sleep 20
            grep -q "Application startup complete" "$DIR/serve.log" 2>/dev/null && ok=1 || { ok=0; break; }
        fi
    done
    if [ $ok -ne 1 ]; then
        LOG "$UNIT serve 启动失败（tail: $(tail -2 "$DIR/serve.log" 2>/dev/null | tr '\n' ' ')），跳过"
        for p in $(ps -eo pid,cmd | grep -E "vllm serve|EngineCore" | grep -v grep | awk '{print $1}'); do
            kill -KILL "$p" 2>/dev/null
        done
        sleep 5
        continue
    fi
    LOG "$UNIT serve 就绪（$(grep -m1 -o 'GPU KV cache size: [0-9,]*' "$DIR/serve.log")），验证 BM 装载:"
    grep -m2 "attn_bm_mode" "$DIR/serve.log" | sed 's/^/    /' | tee -a "$BASE/smoke_chain.log"
    LOG "$UNIT bench 开跑（官方 harness 原件 4 轮）"
    (cd /workspace && python3 $REPO/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
        --model $MODEL --served-model-name minicpm --port $PORT \
        --test-cases "$CASE" 2>&1 | tee "$DIR/benchmark.log" | grep -cE "Serving Benchmark" >/dev/null)
    LOG "$UNIT bench 结束（$(grep -c "Serving Benchmark Result" "$DIR/benchmark.log" 2>/dev/null || echo 0) 个结果块）"
    LATEST=$(ls -t /workspace/benchmark_results/summary_*.csv 2>/dev/null | head -1)
    [ -n "$LATEST" ] && cp "$LATEST" "$DIR/" && cp "${LATEST/summary_/raw_runs_}" "$DIR/" 2>/dev/null
    for p in $(ps -eo pid,cmd | grep -E "vllm serve|EngineCore" | grep -v grep | awk '{print $1}'); do
        kill -KILL "$p" 2>/dev/null
    done
    sleep 5
    LOG "$UNIT 收队（各轮 tok/s: $(grep 'Total token throughput' "$DIR/benchmark.log" 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')）"
done
LOG "SMOKE ALL DONE"
