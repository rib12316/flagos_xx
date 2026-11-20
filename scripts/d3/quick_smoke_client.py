# 快速烟测客户端(非官方 harness):BM 臂 serve 真实路径的 prefill 信号。
#   ① 单请求 16K 输入流式 → TTFT + 首段文本(连贯性=正确性信号)
#   ② 16 并发 × 16K 输入 × 128 输出 → 墙钟吞吐 + TTFT 分布
# 用法: python3 quick_smoke_client.py <port>
import json
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "9032"
URL = f"http://127.0.0.1:{PORT}/v1/completions"
ISL = int(sys.argv[2]) if len(sys.argv) > 2 else 16384
CONC = int(sys.argv[3]) if len(sys.argv) > 3 else 16
OSL_SMALL = int(sys.argv[4]) if len(sys.argv) > 4 else 128
# prompt_token_ids 端点不支持 → tokenizer decode 随机 ids 生成定长文本
# (retokenize 有漂移,但两臂用同一字符串,A/B 可比)
import random
random.seed(7)
from transformers import AutoTokenizer
_TOK = AutoTokenizer.from_pretrained("/workspace/MiniCPM5-2B-w8a8-v2", trust_remote_code=True)
PROMPT_TEXT = _TOK.decode([random.randint(1000, 100000) for _ in range(ISL)])
_real = len(_TOK(PROMPT_TEXT)["input_ids"])
print(f"[prompt] 生成文本 retokenize 后 = {_real} tokens（目标 {ISL}）")


def _payload(max_tokens):
    return {
        "model": "minicpm", "prompt": PROMPT_TEXT,
        "max_tokens": max_tokens, "temperature": 0.0,
    }


def one_stream(max_tokens=24):
    pass
    body = json.dumps({
        "model": "minicpm", "prompt": PROMPT_TEXT,
        "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
    }).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    ttft = None
    text = []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            if line.startswith(b"data:") and b"[DONE]" not in line:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                try:
                    j = json.loads(line[5:])
                    d = j["choices"][0].get("text", "")
                    if d:
                        text.append(d)
                except Exception:
                    pass
    return ttft, time.perf_counter() - t0, "".join(text)


def batch():
    import threading
    results = []

    def worker():
        body = json.dumps(_payload(OSL_SMALL)).encode()
        req = urllib.request.Request(
            URL, data=body, headers={"Content-Type": "application/json"})
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=1200) as r:
                r.read()
            results.append(("ok", time.perf_counter() - t0))
        except Exception as e:
            results.append(("err", str(e)[:60]))

    ts = [threading.Thread(target=worker) for _ in range(CONC)]
    t0 = time.perf_counter()
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in results if r[0] == "ok"]
    toks = CONC * (_real + OSL_SMALL)
    print(f"[batch] {len(ok)}/{CONC} ok  wall={wall:.1f}s  "
          f"总tok/s={toks/wall:.0f}  per-req中位={(sorted(r[1] for r in ok)[len(ok)//2] if ok else 0):.1f}s")
    return toks / wall


SKIP_SINGLE = ISL <= 256
print(f"=== ① 单请求 ISL={ISL} max_tokens=24 流式 ===") if not SKIP_SINGLE else None
ttft, tot, text = (0,0,"") if ISL <= 256 else one_stream()
print(f"TTFT={ttft:.2f}s  total={tot:.2f}s")
print(f"输出片段: {text[:160]!r}")
print(f"=== ② {CONC} 并发 × ISL={ISL} × OSL={OSL_SMALL} ===")
batch()
print("QUICK SMOKE DONE")
