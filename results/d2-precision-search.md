# D2 精度搜索台账 — W8A8 动态 int8（MiniCPM5-2B @ BI-V150）

> 冻结筛查集：math_500 L3 按 unique_id 排序前 30 条（sha=9ea59c28f528，5 题 pilot sha=429b0b44ca57）；
> 105 全量复核 sha=cef9383ec394。greedy 筛查（temperature=0, max_tokens=3072, seed=1234）。
> 对照臂 bf16 greedy30 正确率见下（sanity ≈0.96 闸门）。
> 指标：same_ans = 量化臂 vs bf16 臂同题答案一致率（主）；Δacc = 正确率差。
> emu=int32 = `torch._int_mm` 位精确 int32 累加（与 Triton kernel 整数加法精确一致）+ 与运行时同一
> Triton per-token 量化 kernel —— fake 与 real 的唯一差异只剩 kernel 内部调度顺序。
>
> 晋升规则：same_ans ≥ 0.93 且 acc_q ≥ acc_ctrl − 1 题且无退化病理（截断率 <10%、dlen <25%）
> → 105 条全量 fake 复核 → 真实路径对拍 → 官方配方（temperature=1.0, top_p=0.95,
> max_tokens=32768）105 题 ≥ 0.955 终判。

## L1a 权重 MSE 预扫（CPU，无生成）

per-tensor 平均 MSE（294 张量，越小越好）：α=1.0 → 5.302e-08；0.999 → 5.290e-08；
**0.99 → 5.242e-08（最优）**；0.98 → 5.278e-08；0.95 → 6.061e-08。
→ 权重侧 clip 收益浅（最优仅 −1.1%），权重不是主要误差源；筛查仍验证 0.99 一档。

## 激活离群剖面（L2 依据，results/20261107_d2_actstats/，模型自生成通用文本 48 条×~324 tok）

| 站点 | 典型 max/med | 典型 max/p1 | 结论 |
|---|---|---|---|
| down_proj 输入（SiluAndMul 出） | 12.9–120.9（L41 最毒） | 34.9–463.3 | **主战场**（up 行折叠） |
| qkv_proj 输入（Layernorm 出） | 4.5–82.4（L0 离群） | 5.9–148.3 | 次战场（norm 折叠） |
| gate_up 输入 | 1.7–21.7 | 2.6–28.4 | 中等 |
| o_proj 输入（GQA 聚合 256 维） | 1.4–2.6 | 1.9–5.1 | 温和（折叠近恒等） |

## 筛查主表

| rid | 日期 | git | emu | clip_α | smooth_α | keep | acc_q | acc_ctrl | same_ans | Δacc | dlen | trunc | 判定 | 产物 |
|-----|------|-----|-----|--------|----------|------|-------|----------|----------|------|------|-------|------|------|
| r_ctrl(5题pilot) | 11-07 | feat/d2 | — | — | — | — | 1.000 | — | — | — | — | 0/5 | harness sanity ✓ | control_greedy5.json |
| r_L0(5题pilot) | 11-07 | feat/d2 | int32 | 1.0 | 0 | none | 1.000 | 1.000 | 1.000 | 0 | +8.8% | 0/5 | 起点正常 ✓ | （bisect3，/tmp） |

（wave1 = r_ctrl/r_L0/r_L1a099/r_L2a05–08 30 题进行中，跑完补行）

## 备注

- 昨日（11-06 深夜）一次 r_L0 pilot 出现 len=1/len=5 瞬死 + 满截断病理（acc 0）——同配置今日三次复跑
  （alloc=off@768、alloc=on@768、alloc=on@3072）全部健康，判定为会话拆除期瞬时异常（分配器/GPU 状态），
  非代码缺陷。防线：后续每臂检查退化病理（len<10 计数），异常即重跑该臂。
- emu int32 最初用 fp32 矩阵乘仿真 → 5 题臂 25 分钟不可用；换 `torch._int_mm`（实测与 fp32 参考位精确
  一致、5-10x 快）后 5 题臂 469s。
