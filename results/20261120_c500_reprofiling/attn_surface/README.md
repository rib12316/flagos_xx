# P1-B1 — decode attention 曲面 + 自研 split-KV 对决（沐曦 C500）

- 日期：2026-11-20 02:40 · B 机 · GPU 独占 · `scripts/d2/splitkv_probe_metax.py`
  （q 改 4D serve 同构 + window_size/softcap 参数适配当前 flash 库）
- 原始：`splitkv_probe_metax.csv`（7 形状 × 5 臂）+ `/tmp` 顺序读 ref（1599/1576/1101 GB/s）
- **前置修复（研究档案代码环境适配，3 处，B 机 commit 同步）**：
  ① hop `_env_int` 裸调用 → `_ask._env_int`（27c8ab5 拆线误伤）；
  ② kernel 的 `tl` 局部 import → 模块顶层（沐曦 triton 3.0 编译期只在 `__globals__`
  解析自由变量 → 函数内定义 kernel + 局部 tl = NameError）；TD stub 两处（hop +
  iluvatar.py）补 `__module__` 伪造 triton 前缀（triton 3.0 visit_Call 断言）；
  ③ kernel 删 `continue`（沐曦 triton 3.0 不支持该 AST 节点）——空段改写
  M=-inf 占位，stage2 合并 exp(-inf)=0 数学等价。
  另：**hop 路由判据补 4D q 支持**（serve 经 reshape_query_for_spec_decod 传
  [bs,1,NH,HD]，原判据 q.ndim!=3 全 passthrough——⑥ 在当前环境曾静默失效）。

## 判决一：D6 对决——自研 split-KV 大败，MACA flash 胜出

官方形态平均（bs32/40 × kv16.4K/17.4K，flash=491.3µs）：

| 臂 | µs | 加速 | routed | rel |
|---|---|---|---|---|
| V0-plain | 908.8 | 0.541× | 1 | ~3.8e-3 |
| V1-dbuf | 1322.7 | 0.371× | 1 | ~4.4e-3 |
| V2-persistent | 1974.6 | 0.249× | 1 | ~3.8e-3 |
| V3-wide | — | （全程 routed=0 静默回落，无有效数据） | 0 | — |

**自研 kernel 数值正确（rel≈4e-3 = bf16 噪声量级）但性能全形态大败 2-4×。**
天数侧的赢面（vs triton unified attention）不可迁移——沐曦基线是闭源 MACA flash。

## 判决二：MACA flash decode 曲面（首次沐曦实测）——已 83-88% 带宽利用

| 形状 | flash µs | GB/s | 顺序读上限 | 利用率 |
|---|---|---|---|---|
| bs32 kv16K | 409-410 | 1312 | 1576 | **83%** |
| bs32 kv17.4K | 431-433 | 1320 | — | 83% |
| bs40 kv16K | 505-507 | 1323 | 1576 | **84%** |
| bs40 kv17.4K | 535 | 1333 | 1599 | **83%** |
| bs64 kv16K | 776-778 | 1384 | 1576 | **88%** |
| bs64 kv4K | 225 | 1195 | — | — |
| bs8 kv4K | 67 | 499 | 1101 | 45%（小 bs 低占用） |

**二期计划 A-H1 预设「decode 带宽利用率 55%→80% ⇒ 16K +25-60%」不成立：现实
83-88%，天花板只剩 12-17%，且现有自研路线大败。**

## 对 Phase 2 排名的影响

- **D6（decode attention 对决）基本判死**：自研 V0-V2 无竞争力（0.25-0.54×）、
  闭源 kernel 已 83-88% 利用。除非 Kernel-Smith 生成意外好 kernel，优先级应大降。
- **D7 成为 kernel 线唯一主候选**（P1-B2：prefill chunked 续算 96 TFLOPS 平台，
  上限 2.2×）——资源应集中于此。
- 16K 场景优化排序（B 机证据链）：D5 采样（A 机，fallback 9.66ms/步）>
  D7 prefill kernel（45.6ms/层·chunk 批）> D9 融合 GEMM（quant 37%）> D6（死）。

PROVENANCE：B 机 C500 · git（本 probe commit）· flash_attn 2.6.3+metax ·
hop install 断言通过 · routed 计数 CSV 可查 · rel=maxrel vs flash 同输入。
