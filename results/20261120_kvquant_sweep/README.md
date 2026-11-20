# P2 KV int8 判决探针(2026-11-20,graph 口径,全 kernel 级)

背景:开发单 P2 投影 +40-90%(容量臂);A 臂官方 −9.9%(ITL 187ms"≈2×")。
本目录三个探针把 P2 拆到 kernel 级重审。

## 实测事实

1. **A 臂"2× 税"重新归因**(sweep.csv):kernel 级量化税 decode 仅 **+12%**
   (q8_t16 3799 vs bf16_t16 3386 @bs64/kv16K);187 vs 93ms 的主差额是
   **bs64 vs bs33 每步 2× KV 读**——容量翻倍的必然代价,非 dequant 慢。
2. **量化路径最优 TILE 仍是 16**(t32/t64/t128 单调劣化,t128 灾难 0.13×)。
   上游 dequant 数学已最优(K scale 后置 S、V scale 前置 P),自研 dequant
   上限 = 省 12% 的几个点 ≈ e2e 2-3%,不值 kernel 手术。
3. **①×④互毁精确刻画**(combo):TD×quant 编译共存、数值对(rel 0.9%),
   但 q8_td 5150μs vs q8_notd 3799 vs **bf16_td 2705**——开 quant 弃 TD,
   **真实 decode 税 = +40%**。
4. **prefill 合体税**(combo):BM256/T128/S2×quant **OOR**(flagtree smem
   上限实测 131072B);最优可行 = **q8_bm256_t128_w16_S1 = 19.9ms**,税 +41%
   (vs bf16 S2 14.1ms)/+28%(vs bf16 S1 15.6ms)。BM128 全线劣化。

## 16K 墙钟模型判决

| 形态 | 波结构 | 墙钟 | tok/s |
|---|---|---|---|
| P1-only | 4波×(90s+82s@bs33-TD) | ~688s | ~3240 |
| +P2 env-flip | 2波×(127s+~180s@bs64-q8) | ~615s | ~3650(**+8~13%**) |
| +kernel 手术 | decode 夺回 TD 级 | ~570s | ~3900(+15~20%) |

**env-flip 收益在官方轮间方差(±10%)内,且押 2.1pp 精度余量——现在不做。**
kernel 手术(快 int8 描述符加载/转换)是 P2 唯一正杠杆路径,排 P1 官方门后议。

## 数值

int8 rel:decode 0.9-1.1%(t16)/2.8%(t32+);prefill 1.1%。sanity 级(105 题门未开)。

## 文件

- `sweep.py/.csv/.log`:bf16 vs q8 × TILE×warms 三形态(D64/D32/P1)
- `combo_probe.py`(脚本在 scripts/d3/):BM 形态×quant + TD×quant
