# D2 瓶颈快诊体系（15 分钟定位真瓶颈，无需官方 bench）+ 终态墙图

> 回答两个问题：①项目里真实诊断过吗？②怎么在跑任何优化/bench 之前快速知道
> 「时间花在哪、瓶颈是什么类型」？本文 = 审计结论 + 可复用的三层快诊方案 +
> 终态配置（v2 ckpt + P0/P1）实测墙图（2026-11-15）。

## 一、审计：项目诊断史与两个结构性缺口

| 诊断 | 时点 | 方法 | 回答了什么 |
|---|---|---|---|
| D0 launch 复证 | 11-03 | micro_bench + 旧仓 patch 数据 + T1 | attention 默认 launch 慢 1.5-1.9×（调优表来源） |
| **D2 prefill 归因** | 11-13 | 双臂 torch profiler 12s 窗（`--profiler-config` 诊断 serve + /start_profile）+ `parse_torch_trace.py` + 链路微基准 | prefill 每 kernel 时间占比（attention 65%、epilogue 24% 税）、真实 M=2048 chunk、lm_head 931µs@577GB/s |
| 融合 kernel 深挖 | 11-13 | ixkncli 指令统计 + phase0c sweep | kernel 内瓶颈（smem 占用 12.5%、矩阵指令路径） |
| KV int8 探针 | 11-14 | phase2_kv_probe 双臂 graph 计时 | **首个 bound 类型判定**：attention 非带宽 bound |
| 本轮墙图 | 11-15 | bottleneck_probe + profiler 窗口 | 终态配置完整墙图（§三） |

**两个缺口（这就是"做无用功"的来源）**：
1. **时间占比 ≠ 瓶颈类型**。11-13 的归因回答了"谁占 65%"，没回答"它为什么慢
   （带宽/算力/延迟-占用）"。P2/P4 的立项依据是"attention 占 65-80% + KV 读
   占其流量 71-91%"——听起来像带宽问题，直到实测带宽利用率才发现是延迟
   bound。**先测利用率再立项，这个顺序缺位导致 P2 空转。**
2. **一次性 forensic ≠ 例行体检**。每份归因都是为某个具体疑问临时搭的，
   没有"改配置前先跑一遍体检"的固定动作。本文的三层方案就是把它标准化。

**第三个方法论坑（本轮踩的，已修）**：探针必须复刻 serve 的运行时调优
（launch 表/图模式），否则测的是"未调优 kernel"——首版 L1 探针没装 D0 表，
attention 慢 3×（2680 vs 696µs），带宽利用率 6% vs 真实 23%，墙图失真。
`bottleneck_probe.py` 已内置 `attn_launch_config.install()`。

## 二、三层快诊方案（总计 ~15min，全离线/诊断 serve，不碰官方 bench）

### L0 步进分解（3min，只要一个 serve）
小参 case（[512,1024,64,64]）跑一轮 bench → 取 ITL/TTFT。物理 sanity：
`ITL × 并发 ≈ 每 token 时延上限`（ITL 20ms × 64 = 每 token 1.28s·GPU，
反推 step 内 GPU 预算）。ITL 变化 = 端到端体检脉搏。

### L1 屋顶线分类（5min，`scripts/d2/bottleneck_probe.py`，无需 serve）
每组件在 serve 真实形状 + serve launch 表下，graph 口径测「达成速率 vs
天花板」，输出分类：

| 分类 | 判据 | 有效杠杆 | 无效杠杆 |
|---|---|---|---|
| bandwidth-bound | ≥60% 实用带宽（本机 544GB/s 元素参照） | 字节减半类（量化权重/KV） | — |
| compute-bound | ≥60% vendor 算力（118 T-MAC/s） | int8 原生 dot / 更低位宽算力 | 带宽类 |
| **latency/occupancy** | 两者都低 | **kernel 形态工程**（split-K/并行度/流水） | **一切量化** |

### L2 serve 墙图（7min + serve 启动，红线允许的诊断观测）
`vllm serve … --profiler-config '{"profiler":"torch","torch_profiler_dir":…,
"torch_profiler_with_stack":false}'` → burst 流量 → `/start_profile`/
`/stop_profile` → `scripts/d2/parse_torch_trace.py <rank0 trace>`。
注意：decode 在 cudagraph 内重放**对本 trace 不可见**——L2 只给 prefill/
eager 侧占比；decode 侧用 L1 分类 + `42层×单层数值` 与 ITL 对账（自洽
校验：算出的 step GPU 预算 ≈ ITL ±10%）。

### 判定链（选武器前必过）
`L2 谁占比大 → L1 它什么 bound → 查杠杆表`。占比大 + latency-bound =
**先做形态工程，量化免谈**（P2/P4 的教训）；占比小 + bandwidth-bound =
量化值得做（P1 的正确路径）。

## 三、终态配置墙图（2026-11-15 实测）

### prefill（profiler 窗口，4×4096 输入）
| 成分 | 占 GPU kernel 时间 | L1 分类 |
|---|---|---|
| **attention（bf16, w2s1）** | **87.7%**（11.2ms/层/chunk） | latency/compute 混合，非带宽（实用带宽的 0%） |
| 线性 int8（融合+链+quant+epi） | 8.5% | prefill 融合 89% vendor 算力 = **贴顶** |
| norm/other | 3.8% | 平凡 |

### decode（L1 + 对账推导，M=64，kv~1.5K 实域）
| 成分 | 每 step 估算 | L1 分类 |
|---|---|---|
| **attention** | ~15ms（42 层 × ~360µs）≈ **70%** | 193-197GB/s = **实用带宽 36%** → latency/occupancy |
| 线性 int8（P0 融合档） | ~5.1ms ≈ 23% | gate_up 88% / down 78% 实用带宽 = **已饱和** |
| norm/rope/采样 | ~2ms ≈ 10% | 平凡 |

对账：15+5.1+2 ≈ 22ms ≈ 实测 ITL 20-22ms ✓。

### P2 复测（serve 等效 launch，双臂同表）
int8/bf16 = decode **0.929×** / kv16k **0.925×** / prefill **0.820×**
——判决在调优口径下成立（原 0.90×/0.726× 为未调优口径），仍净负。

## 四、墙图的处方（下一步该做什么、不该做什么）

1. **该做：attention kernel 形态工程**（decode 36% → 目标 70%+ 实用带宽，
   理论 2×，对应 decode step −35%；prefill attention 同查 split/流水）。
   入口：fork `triton_unified_attention.py` 做 flash-decoding 式 KV split +
   两段归约；工程 2-3 天，无精度风险。**这是 65-88% 占比上唯一的未开采矿。**
2. **不该做：更多量化**。线性两档已贴物理顶（88% 实用带宽 / 89% vendor
   算力）；KV/int8-attention 被实测封死；W4 无原生路径。
3. **待裁定**：KV int8 的容量收益（池 ×1.94）在官方 16K 容量 binding 场景
   是否净正——需一次官方单场景实测（~15min，等用户令）。
