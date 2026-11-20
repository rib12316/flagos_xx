#!/usr/bin/env bash
set -euo pipefail
root=$(git rev-parse --show-toplevel)
export PYTHONPATH="$root/vllm-plugin-FL${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_PLUGINS=fl
model=/workspace/MiniCPM5-2B-w8a8-v2
trial=$(mktemp -d /data/fusion_quick_XXXXXX)
echo "results=$trial"
for var in VLLM_FL_FA_ROPECAT VLLM_FL_IR_NORM VLLM_FL_D13_NORM_QUANT VLLM_FL_FLASH_KVCACHE_DIRECT_OUT; do unset "$var"; done
stop_server() {
    ps -eo pid,args | awk -v id="$server_pid" '$1==id && /vllm.entrypoints.openai.api_server/ && !/awk/ {print $1}' | xargs -r kill
    wait "$server_pid" || true
}
read -r -a flags <<< "${FUSION_ARM_FLAGS:-0 1}"
for flag in "${flags[@]}"; do
    arm="$trial/arm$flag"
    mkdir "$arm"
    cd "$arm"
    export VLLM_CACHE_ROOT="$arm/cache"
    unset VLLM_FL_FUSION_BUNDLE VLLM_FL_D13_M_FUSE_MAX VLLM_FL_FA_PROBE
    case "$flag" in
        default) ;;
        0|1) export VLLM_FL_FUSION_BUNDLE="$flag" VLLM_FL_FA_PROBE="$arm/fa.log" ;;
        *) exit 2 ;;
    esac
    for i in {1..30}; do mx-smi | grep -q 'no process' && break; sleep 2; done
    mx-smi | grep 'no process'
    python3 -m vllm.entrypoints.openai.api_server --model "$model" --port 9031 \
        --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 >serve.log 2>&1 &
    server_pid=$!
    trap stop_server EXIT
    for i in {1..240}; do
        ps -p "$server_pid" >/dev/null || { tail -40 serve.log; exit 1; }
        curl -fsS -m 2 http://127.0.0.1:9031/health >/dev/null 2>&1 && break
        sleep 3
    done
    curl -fsS -m 3 http://127.0.0.1:9031/health
    curl -fsS -m 180 -H 'Content-Type: application/json' -d '{"model":"minicpm","prompt":["The capital of France is","What is 7 times 8?","写一句关于秋天的话。","def add(x, y):"],"temperature":0,"seed":0,"max_tokens":24,"return_token_ids":true}' \
        http://127.0.0.1:9031/v1/completions >smoke.json
    if [[ $flag != 0 ]]; then
        grep -E 'F-c grouped decode out active|paired.*83|83.*pair|Final IR op priority' serve.log | tail -5
        grep -q 'F-c grouped decode out active' serve.log
        grep -q 'paired 83 sites (m_fuse_max=2048' serve.log
        grep -Fq "fused_add_rms_norm=['vllm_c', 'native']" serve.log
        if [[ ! -f fa.log ]] || ! grep -Eq 'CALL_DONE matched=[1-9]' fa.log; then
            rg -q -a 'torch.ops.vllm.fl_rope_qk_cat.default' "$arm/cache/torch_compile_cache"
        fi
        rg -q -a 'torch\.ops\.vllm_fl\.d13_norm_quant\.default\(' "$arm/cache/torch_compile_cache"
    fi
    if [[ ${FUSION_SMOKE_ONLY:-0} != 1 ]]; then
        timeout 1800 python3 -u "$root/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py" --model "$model" --served-model-name minicpm --port 9031 --test-cases '[[4096,1024,64,32],[4096,1024,64,64]]' >bench.log 2>&1
    fi
    trap - EXIT
    stop_server
done
if [[ -f "$trial/arm0/smoke.json" && -f "$trial/arm1/smoke.json" ]]; then
    python3 -c 'import json; from pathlib import Path; a=json.loads(Path("'"$trial"'/arm0/smoke.json").read_text()); b=json.loads(Path("'"$trial"'/arm1/smoke.json").read_text()); print("greedy smoke text identical:", [x["text"] for x in a["choices"]] == [x["text"] for x in b["choices"]])'
fi
