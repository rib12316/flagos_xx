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

| r_ctrl | 11-07 | feat/d2 | — | — | — | — | — | **0.9333** | — | — | — | 3/30 | sanity ✓ | control_greedy30.json |
| r_L0 | 11-11 | feat/d2 | int32 | 1.0 | 0 | none | 0.9333 | 0.9333 | **0.9000** | 0 | +15.8% | 3/30 | 起点（未达 0.93 晋升线，需 L1a/L2/L4 找回） | r_L0.json |

（进行中：r_L1a099 / r_L2a05–08，跑完补行；r_probe_* 与组合档视结果加跑）

## 备注

- **11-07 流竞争 bug 定位记录（报告素材：框架层发现）**：wave1 首轮 r_L0 30 题 acc 崩至 0.20（大量
  len=2~9 瞬死 EOS + 满长度游荡），5 题 pilot 却两次全对。隔离实验（scripts/d2/debug_batch_m.py，
  同题 b=1 vs b=30 复制对照）四臂定位：ctrl 两臂逐字一致；量化 b=1 连贯、b=30 首 token 即垃圾 →
  M 依赖。逐模块数值对拍排除数学问题（act_quant、`torch._int_mm`、scale 全行位精确，含 M=4500）。
  双盲收口：patched_apply 加 `torch.cuda.synchronize()` → b=30 恢复与 b=1 逐字一致；换纯 torch 量化链
  （不加 sync）→ 同样恢复。**元凶 = `@libentry()` 包装的 Triton 逐 token 量化 kernel 的启动器（event
  -timing 旁路流），b>1 时下游读到未就绪输入**。修复：harness 默认纯 torch 量化链。真实 serve 路径经
  CachedOp/dispatch 包装，是否同病待 Phase 4 真实路径对拍检验（若复现 = 插件级 bug，须在 dispatch 层
  修复后量化才能上线）。
- emu int32 最初用 fp32 矩阵乘仿真 → 5 题臂 25 分钟不可用；换 `torch._int_mm`（实测与逐行 fp32 整值
  矩阵乘全行位精确、5-10x 快；M>16 限制用垫零解决）后 5 题臂 469s。
- 11-06 深夜一次 pilot 同症状（当时误判瞬时异常）——实为同一流竞争（该跑 max_tokens=3072 复现条件
  恰好凑齐），今日三次"健康"复跑均在未触发竞争的口径下（b=1 / 768 tok）。
