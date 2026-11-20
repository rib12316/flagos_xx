# 交付文档 · C500 算子工程三任务开发单（新 agent 接手）

> 写给：新开发 agent（沐曦 C500 A 机同型号环境）。你接手的是三项**互不依赖**的算子工程任务：
> ①小算子融合（多项菜单制）、②逐算子性能替换扫描、③host 开销消除。
> 全部为自主可控项（零 vendor 依赖）。本文自包含——按序读完即可开工，无需今日会话上下文。

## 0. 必读与红线（违者回滚）

1. 通读顺序：`CLAUDE.md` → `README.md`（纪律红线）→ `docs/二期计划-1120.md`（背景）→
   `docs/00-统一假设台账.md` §9（前人判决，**不要重判已判死项**）→ 本文档。
2. 红线：serve 启动参数冻结（优化一律走**插件代码默认值**+env 逃生门，①⑤ 翻转先例）；
   `benchmark_throughput_serve.py` 原件零改动；官方 bench 与 105 题精度评测**必须用户明示批准**；
   GPU 测量互斥（bench/剖析/微基准不同时）；产物落 `results/<日期>_<id>/` + 回写台账 §9；当日 push。
3. 分支：从 `delivery/s2-final` 切（**注意：main 已停用**），单假设一条分支：
   `feat/f-*`（融合）/`feat/op-*`（替换）/`feat/host-*`；`git diff delivery/s2-final..feat/<id>` 只含本假设。
4. 环境：`bash scripts/d5/env_audit_metax.sh` 必须 AUDIT_GREEN；改插件后
   `pip install --no-deps --no-build-isolation ./vllm-plugin-FL` + provenance 四查（README）。

## 1. 三条方法论铁律（今日用一天换来的，违反必白干）

1. **微基准必须 L2-warm**：被测 kernel 的输入若是克隆轮转（冷缓存），测出的耗时高估 4-8×
   （D5/D9/D7 三案同构伪影，台账 §9）。融合类 kernel 测试用**同一输入张量反复跑**（elementwise 不破坏输入）。
2. **小 kernel 计时必须 CUDA graph 口径**：python launch 开销 ~50-90µs 会淹没 µs 级 kernel
   （D9 教训；参考 `scripts/d5/d9_quant_probe_metax.py` 的 `bench()` 实现）。
3. **一切结论以 serve trace 为准**：kernel 级收益 × serve 内占比 = 真实 e2e 预期；离线数字
   在上报时必须标注口径。墙图采集法：`scripts/d5/trace_wallmap_v2.sh`（active_iterations 连续记录法）。

## 2. 今日实测基础（你的起点数据）

- decode 步（bs=32, 26ms/步）kernel 清单（48MB trace，`results/20261120_c500_reprofiling/trace_all/topk_kernels.txt`）：
  attention 4.75ms / GEMM 链 4.4ms / quant 1.55ms / norm+rope+cat+cache ~1.7ms / 采样 ~1-2ms / 杂项 0.2ms；**GPU busy 96%**
- 这些数字就是你的"矿藏地图"：三项任务全部针对其中 <10% 的小件池与 4% 空隙，**别去动 attention/GEMM 大件**
  （大件已判最优：flash 83-88% 带宽、mctlass 近算力上限、triton dot 本征 56T——台账 §9 全记录）。
- 前人已完成：D13 norm+quant 融合（B 机交付，kernel 级 1.10-1.16×，e2e +0.4-0.6%，**待 A 机集成裁定——不要重做**）。

## 3. 任务①：小算子融合（菜单制，逐项立项逐项关账）

按预期收益排序的候选菜单（每项独立 feat 分支，T0 位精确 + graph 口径 + T1 三轮中位）：

| # | 融合项 | 现状证据（trace 每步） | 预期 | 难度 |
|---|---|---|---|---|
| F-a | **rope + cat**（qkv 拼接与 RoPE 相邻链） | triton_poi_fused_cat_3/4 合计 0.42ms + fused_1(rope) 0.32ms | +0.5-1% | 中 |
| F-b | **reshape_and_cache + rope 融合**（写 KV 时顺便 rope，FlashMLA 模式） | cache 写回 0.23ms + rope 部分 | +0.5-1% | 中高 |
| F-c | **copy 消除**（`_copy_kernel_kernel_rank_3` 46074 次=0.19ms/步——先查是谁在做布局搬运，能删则删，不能删则融合进邻算子） | 0.19ms | +0.3-0.5% | 低-中 |
| F-d | D13 norm+quant 已做（勿重做）；其姊妹项 **silu-mul + quant**（gate_up 后激活量化） | — | +0.3-0.5% | 中 |
| F-e | 机会项：遍历 trace 中其余 <100µs/步的 triton_poi_* 碎核，两两相邻可合则合 | 合计 ~0.3ms 池 | 各 +0.1-0.3% | 按项 |

实现通道（按优先）：
- 首选 **torch.compile 自定义 pass 注入**（vLLM 官方 ActivationFusionPass 同机制：PostGradPassManager，
  插件进程内改默认值，免改 vllm 源码）——适合模式化融合（silu+quant、cat 类）；
- 直写 **triton kernel 替换**（plugin 的 FlagGems impl 层；数学逐位照抄——舍入口径见
  `_round_half_to_even`，norm/quant 先例在 `dispatch/backends/flaggems/impl/quantization.py`）；
- 纪律模板抄 `vendor/iluvatar/attn_splitkv.py` 头注（裸 jit/禁 autotune/graph 相容静态查表）。
门：T0 位精确（norm 类）或分布等价（激活类，概率质量差 <1e-3，参考 `scripts/d5/sample_fast_t0.py` 判据实现）→
T1 quick case [[4096,1024,64,32]] 与 [[4096,1024,64,64]] 双形状 ≥3 轮中位 ≥+0.3% 才立项合入。

## 4. 任务②：逐算子性能替换扫描（审计表制）

方法：
1. 从 `trace_all/topk_kernels.txt` 枚举全部在跑 kernel → 映射到插件路由（dispatch/backends/*/metax.yaml 的
   op_backends 表）→ 每算子找**候选替代实现**：aten 原生 / mcoplib vendor op（`torch.ops._C.*` 枚举）/ FlagGems 别档；
2. **优先查一个旧线索**（台账 §1 A-H4，2h 工作量）：mcoplib 的 `torch.ops._C.fused_add_rms_norm`
   已注册但被 metax.yaml 路由顺序排除——改路由顺序 A/B，若更快即白捡（本子项独立成 feat）；
3. 逐算子 A/B：graph 口径 + L2-warm，输出审计表（op / 现实现 / 候选 / 实测 / 判决），
   胜出者改默认 + env 回退门（#384 拉黑模式先例，README 与 survey 有引）；
4. 注意：大件（attention/GEMM/quant）已判最优勿重扫；目标池=norm/rope/cat/copy/采样杂项 <10% 小件。
预期合计 ~1-2%。

## 5. 任务③：host 开销消除（先审计后动手）

- 天花板：decode busy 96% → 全部空隙 ≤4%；async scheduling 与 FULL graph 已确认开启（B 机实锤，
  台账 §9）——**不要重查这两项**。
- 动作：
  1. grep 插件热路径的 `.item()/.cpu()/synchronize()`（vllm_fl/ 全目录），逐个评估是否在 decode 步内；
  2. 静态张量缓存（#778 模式可取子集：slot_mapping 等跨步不变张量）；
  3. 产出：host 审计表 + 每项消除的 A/B（quick case 双形状三轮中位）。
- 预期 ~1-2%；超过 2% 的声称先自查测量口径（铁律 3）。

## 6. 验证与交付协议（每项改动逐级）

T0（秒-分钟级，位精确/分布等价 + graph 计时）→ T1（quick 双形状 ≥3 轮中位，新鲜 serve，GPU 独占）→
（多项累积后）T2 官方全量：**用户批准后**用 `scripts/d5/run_official_d5.sh`（官方形态逐字）。
交付物格式：`results/<日期>_<id>/`（raw+log+README 判决）+ 台账 §9 追加行 + 分支 push；重大结论同步任务板留言。
每完成一项在 `docs/双机任务板.md` 留言区追加一行（新 agent 区块自建：`## 新agent区`）。

## 7. 预期总账与纪律

三项合计自主可控预期 **+2-4%**（今日口径）；单 <0.3% 项不合入。诚实负判决同样宝贵——按台账格式记录。
