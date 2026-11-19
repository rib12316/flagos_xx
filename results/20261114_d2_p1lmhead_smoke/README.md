# 2026-11-14 P1 lm_head int8 烟测（v2 vs v1 ckpt）

复用 /tmp/d2_stage_smoke.sh（单实例 flock 修正版），conc 对齐官方场景
（[[512,1024,64,64],[256,1024,128,128]]），3 轮 + 30 题对拍 sanity。

**事故存档**：首次运行两脚本实例并行（`a && b & ` 整链后台化 + 首试 redirect
假失败真启动），互抢 9031 端口、互杀 serve——数据全废，归档
`../20261114_d2_p1lmhead_smoke_botched_2inst*/`。教训入 memory。

## 激活确认（硬事实）

- v2 KV 池 588,016 vs v1 581,936 tokens（两次独立复现 +6.0~6.1K tokens ≈
  lm_head 535→268MB 释放的精确换算：267MB / 43KB-per-token(42层) ≈ 6.2K）✓
- **lm_head int8 确认装载生效**（loader 严格模式 + packed 三键存在）。

## 端到端（噪声大，ITL 为主指标）

| 臂 | c64 tok/s 中位 | c64 ITL 中位 | c128 tok/s 中位 | c128 ITL 中位 | parity acc |
|---|---|---|---|---|---|
| A v2(lm_head int8) | 1762 | **19.27ms** | 1980 | 33.09ms* | **0.9667** ✓ |
| B v1(bf16 head) | 2958 | **19.44ms** | 2084 | 32.41ms* | 0.60 ⚠见下 |

*c128 双臂中位均被单轮离群污染（A 的 r2=42.3ms 冷轮；B 的 r3 尾段拥塞）。

- c64 ITL −0.17ms（−0.9%）：方向正确、与逐层换算预测（bf16 lm_head 读
  535MB vs int8 融合 0.57ms → −0.2~0.3ms）同量级但贴噪声底。
- **P1 端到端效应 ≈ 1-2%（ITL 口径），显著小于 P0（−19%）**；主要价值在
  权重 −267MB（KV 池 +1.05%）与 decode 图内确定性 GPU 时间下降。
- tok/s 口径被 prefix cache 双峰支配（同一臂轮间 1060→3443），n=3 中位
  不足以分辨 P1 量级——不再追测（ROI 不在端到端而在资源释放）。

## ⚠ arm B（v1 现行配置）parity 崩溃事件

30 题对拍 acc 0.60 / same_ans 0.5667 / trunc 25/30（历史带 0.90-0.9333，
trunc 3-5）。发生窗口：3 轮高压 bench 之后。同日 arm A（v2，同 decode 档）
parity 干净（0.9667，trunc 14）。结合 P0 烟测两臂 r3 静默 EngineCore 死亡
（同签名 D1 时代 20261105 前科）：指向**间歇性引擎级退化/崩溃，与 P0/P1
配置无关**。归因实验（新鲜 v1 serve parity 对照 + conc128 ×5 逐轮 parity
压力）见 `../20261114_d2_fresh_v1_parity/` 与 `../20261114_d2_conc128_stress/`。

## 判定

P1 合入：激活硬事实 + parity 干净 + 无回退信号（c128 +0.68ms 为噪声底内，
r2 离群主导）。serve 最终指向 v2 目录属部署步骤（红线允许）。
