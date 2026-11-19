#!/bin/bash
# D3 官方 bench 补跑器：单场景 × 新鲜 serve（EngineCore 长负载死亡时钟下的
# 结构性对策——每补跑单元一个 serve，收有效轮前缀，死了换 serve 再来）。
# 用法: setsid nohup bash run_official_redo.sh <unit...> > redo.log 2>&1 &
#   unit: A16 D16 B16 A4
set -u
REPO=/flagos_比赛/flagos_xx
BASE=/flagos_比赛/flagos_xx/results/d3_official_1116_2222
MODEL=/workspace/MiniCPM5-2B-w8a8-v2
PORT=9031
LOG() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$BASE/redo_chain.log"; }

unit_cfg() {
    case $1 in
        A16) ENVS="VLLM_FL_ATTN_TILE=1 VLLM_FL_KV_INT8=1";  CASE='[[16384,1024,64,128]]' ;;
        D16) ENVS="";                                        CASE='[[16384,1024,64,128]]' ;;
        B16) ENVS="VLLM_FL_ATTN_TILE=1 VLLM_FL_ATTN_TD=1";  CASE='[[16384,1024,64,128]]' ;;
        A4)  ENVS="VLLM_FL_ATTN_TILE=1 VLLM_FL_KV_INT8=1";  CASE='[[4096,1024,64,256]]' ;;
        B4)  ENVS="VLLM_FL_ATTN_TILE=1 VLLM_FL_ATTN_TD=1";  CASE='[[4096,1024,64,256]]' ;;
        *) return 1 ;;
    esac
}

for UNIT in "$@"; do
    DIR=$BASE/${UNIT}_redo
    mkdir -p "$DIR"
    unit_cfg "$UNIT" || { LOG "unknown unit $UNIT"; continue; }
    LOG "== 补跑 $UNIT 开始（env: ${ENVS:-默认}, case: $CASE）=="
    cd /workspace
    setsid nohup env VLLM_PLUGINS=fl $ENVS vllm serve $MODEL \
        --port $PORT --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
        --served-model-name minicpm --gpu-memory-utilization 0.85 \
        --max-model-len 131072 > "$DIR/serve.log" 2>&1 &
    SPID=$!
    ok=0
    for i in $(seq 1 66); do
        sleep 10
        grep -q "Application startup complete" "$DIR/serve.log" 2>/dev/null && { ok=1; break; }
        if ! pgrep -f "vllm serve $MODEL" >/dev/null; then
            sleep 20
            pgrep -f "vllm serve $MODEL" >/dev/null || { ok=0; break; }
        fi
    done
    if [ $ok -ne 1 ]; then
        LOG "$UNIT serve 启动失败，跳过"
        continue
    fi
    LOG "$UNIT serve 就绪（$(grep -m1 -o 'GPU KV cache size: [0-9,]*' "$DIR/serve.log")），bench 开跑"
    (cd /workspace && python3 $REPO/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
        --model $MODEL --served-model-name minicpm --port $PORT \
        --test-cases "$CASE" 2>&1 | tee "$DIR/benchmark.log" | grep -cE "Serving Benchmark" >/dev/null)
    NV=$(grep -c "Serving Benchmark Result" "$DIR/benchmark.log" 2>/dev/null || echo 0)
    LOG "$UNIT bench 结束（$NV 个结果块）"
    LATEST=$(ls -t /workspace/benchmark_results/summary_*.csv 2>/dev/null | head -1)
    [ -n "$LATEST" ] && cp "$LATEST" "$DIR/" && cp "${LATEST/summary_/raw_runs_}" "$DIR/" 2>/dev/null
    for p in $(ps -eo pid,cmd | grep -E "vllm serve|EngineCore" | grep -v grep | awk '{print $1}'); do
        kill -KILL "$p" 2>/dev/null
    done
    sleep 5
    LOG "$UNIT 收队（有效轮 tok/s: $(grep 'Total token throughput' "$DIR/benchmark.log" 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')）"
done
LOG "REDO ALL DONE"
