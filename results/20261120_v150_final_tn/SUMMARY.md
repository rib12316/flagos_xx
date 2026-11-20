# V150 合并树官方全量验证（2026-11-20 23:07 收官）

树：afa5fda（final-submition merge：W8A8 ixformer TN + P1 BLOCK_M 默认开 + D0-D3 全栈）。
形态：官方 4.3 逐字，零附加 env（仅官方自带 VLLM_PLUGINS=fl），serve 含 compilation-config，
--model 指 w8a8-v2 ckpt。bench 单次调用（内部 4 轮丢首均值）。

| 场景 | Total tok/s | 旧官方 | 提升 | 纯净基线 | vs 基线 |
|---|---|---|---|---|---|
| 4K [4096,1024,64,256] | 4614.33 | 3414.58 | +35.1% | 1879.68 | +145.5% |
| 16K [16384,1024,64,128] | 3259.02 | 1570.62 | +107.5% | 862.06 | +278% |
| 精度 math_500 L3 (105题) | 96.2% | 97.14% | 带 0.962-0.971 内 | 0.962 | 红线 0.95 ✓ |

TTFT：4K Median 1628ms（基线 12321，-87%）；16K Mean 151s（基线 636s，-76%）。
本轮新增成分相对旧官方口径：W8A8 ixformer TN（bit-exact，prefill 线性 0-15%+尾档2.7×）
+ P1 prefill BLOCK_M=256/W16/S2/T128（非 bit-exact，TILE128 ≤2ULP 漂移类，本表 96.2% 背书）。
