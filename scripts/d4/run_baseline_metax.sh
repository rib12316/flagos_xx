#!/bin/bash
# 沐曦纯净基线（官方口径）：基线插件(13eb9be) + bf16 模型，双场景各 4 轮。
# 用法：bash scripts/d4/run_baseline_metax.sh   （前置：当前 serve 已杀、GPU 空闲）
set -euo pipefail
cd /data/flagos_xx
RES=results/20261119_d4_metax_port/baseline
mkdir -p "$RES"

echo "== [1/4] 安装基线插件（/workspace/vllm-plugin-FL @13eb9be）=="
pip install --no-deps --no-build-isolation /workspace/vllm-plugin-FL 2>&1 | grep -E 'Successfully|error' | tail -2
python3 - <<'PY'
import vllm_fl, os, json, glob
loc = os.path.dirname(vllm_fl.__file__)
print("vllm_fl at:", loc)
infos = glob.glob(os.path.join(os.path.dirname(loc), 'vllm_plugin_fl*dist-info', 'direct_url.json'))
assert infos, "direct_url missing"
url = json.load(open(infos[0]))['url']
print("direct_url:", url)
assert url.startswith('file:///workspace/vllm-plugin-FL'), f"provenance FAIL: {url}"
print("PROVENANCE OK: baseline plugin")
PY

echo "== [2/4] 基线 serve（bf16）=="
VLLM_PLUGINS=fl setsid nohup vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  > "$RES/serve_baseline.log" 2>&1 &
for i in $(seq 1 60); do
  sleep 10
  if curl -s -m 3 http://127.0.0.1:9031/v1/models | grep -q minicpm; then
    echo "serve ready after $((i*10))s"; break
  fi
  if ! pgrep -f 'vllm serve' >/dev/null; then
    echo "SERVE DIED"; tail -30 "$RES/serve_baseline.log"; exit 1
  fi
done
grep -E 'GPU KV cache size|Maximum concurrency' "$RES/serve_baseline.log" | tail -2

echo "== [3/4] 官方 bench：4K×4 + 16K×4 =="
for r in 1 2 3 4; do
  echo "-- 4K round $r --"
  timeout 1800 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
    --test-cases '[[4096,1024,64,256]]' 2>&1 | grep -E '^Prefill=' || true
done
for r in 1 2 3 4; do
  echo "-- 16K round $r --"
  timeout 3600 python3 vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
    --test-cases '[[16384,1024,64,128]]' 2>&1 | grep -E '^Prefill=' || true
done

echo "== [4/4] 收尾：杀 serve + 装回仓库分支插件 =="
pkill -f 'vllm serve' || true; sleep 8
pip install --no-deps --no-build-isolation /data/flagos_xx/vllm-plugin-FL 2>&1 | grep -c Successfully
echo "BASELINE_RUN_DONE — CSVs in /data/flagos_xx/benchmark_results/ (copy to $RES)"
