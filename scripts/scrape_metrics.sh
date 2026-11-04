#!/usr/bin/env bash
# D1 调度线：/metrics 周期抓取（抢占裁决 E1/E4 证据源）。
# 用法: bash scripts/scrape_metrics.sh <port> <out_dir>   → 后台运行，写 <out_dir>/metrics.log
# 停止: pkill -f scrape_metrics.sh
set -u
PORT="${1:-9031}"
OUT="${2:-.}"
INTERVAL="${SCRAPE_INTERVAL:-2}"
mkdir -p "$OUT"
echo "# scrape start $(date -Is) port=$PORT interval=${INTERVAL}s" >> "$OUT/metrics.log"
while true; do
    ts=$(date +%s.%N)
    curl -s --max-time 2 "http://127.0.0.1:${PORT}/metrics" \
      | grep -E '^vllm:(num_preemptions|kv_cache_usage_perc|num_requests_running|num_requests_waiting|prompt_tokens|prompt_tokens_cached|generation_tokens)' \
      | awk -v ts="$ts" '{print ts, $0}'
    sleep "$INTERVAL"
done >> "$OUT/metrics.log" 2>&1
