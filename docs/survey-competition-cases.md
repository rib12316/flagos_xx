# 同类比赛获奖方案与实战案例调研（FlagOS S2 赛道二借鉴）

> 调研日期：2026-11-20。方法：WebSearch + GitHub API/WebReader 抓取（vllm-plugin-FL 全部 611 个 PR 标题 + 关键 PR 正文/补丁，FlagGems 1000 个 PR 标题，FlagOS 官网/知乎/CSDN，厂商与国际赛事公开资料）。
> 我方背景：MiniCPM5-2B（Llama 稠密 2B，bf16，GQA 16Q/2KV）× 天数 BI-V150（Triton/flagtree）+ 沐曦 C500（MACA/Triton），vllm-plugin-FL（vllm 0.24）+ FlagGems，4K/16K 两场景 total tokens/s，math_500 ≥0.95 红线，serve 参数锁死、禁超参开量化/投机采样。已合入 D0 attention launch 调优（+42%/+48%）、D1 watermark 调度（+0.4%/+1.1%）；进行中 D2 W8A8；备选 D3 KV-int8×attention 重写。

---

## 0. TL;DR 案例总表

| # | 来源 | 硬件/模型 | 核心招式 | 实测收益 | 对我方映射 |
|---|---|---|---|---|---|
| 1 | **S2 赛道二竞队 PR #671**（RobertFengOPC） | BI-V150 / MiniCPM5-2B（同赛题！） | unified_attention launch 参数耦合重标定：prefill BLOCK_M 16→256、warps 4→16、TILE_SIZE 32→128、门控阈值 8192→0；decode warps 8→2/stages 3→1 | **4K +81.5%、16K +199.7%**，逐位一致，math_500 97.1–98.1% | D0 上调空间巨大；C500 需独立网格（其 64KB 共存上限使大 tile 物理不可行） |
| 2 | **S2 赛道二竞队 PR #778**（fanqh18，中科院应数所） | V150+C500 双卡 / 同赛题 | 纯 Python 层：张量元数据缓存、消除 `.cpu()/.item()` 同步、>64 并发禁 graph、`flag_gems.enable()` 换 FP32 累加融合算子 | V150 +23.9%/+21.3%，C500 +24.9%/+22.8% | 新增方向：**host 侧开销削减**是零风险大西瓜 |
| 3 | **S1 赛道二唯一公开提交 PR #178**（E4_8 队） | 2×Ascend 910C / Qwen3-4B | 只留 Ascend+FlagGems 分发、逐场景 benchmark 调参、HCCL AIV | 1K +86.2% / 4K +95.3% / 6K +102.4%（未合入） | 逐场景（4K/16K）分别调参数的思路；大删文件不受接受 |
| 4 | vllm-plugin-FL **#384**（合入） | H100 / 16K 服务 | Nsight 定位：Triton unified attention 比 FA3 慢 11×、占 GPU 差距 76.2% → 按架构路由 vendor attention + 拉黑 FlagGems `mm` | 535→692 tok/s（+29%） | **逐算子 profile→拉黑/路由**：FlagGems 在 V150/C500 上未必处处最优 |
| 5 | vllm-plugin-FL **#381/#417**（合入/未合） | 多后端 | graph 显存计入 KV sizing；`AttentionCGSupport.UNIFORM_BATCH` 让 decode 走 FULL CUDA Graph（昆仑xon 后端曾被降级到 PIECEWISE） | 显式 A/B 未给，属正确性修复型增益 | 检查 V150/C500 后端是否真走 FULL graph decode |
| 6 | vllm-plugin-FL **#299**（未合） | Qwen3.6-35B / 32K–64K | prefill 序列分片 context parallelism + FlagCX PD | 32K +33.0%、64K +38.1% | 仅多卡可用；单卡不适用 |
| 7 | **MLPerf Inference Closed Division**（v5.0/v5.1，2025） | H200/B200/MI300X / Llama 系 | 固定模型+99% 精度门槛下：FP8/W8A8 权重+激活+**KV cache** 量化、in-flight batching、分页 KV | Blackwell 3.1×（含硬件）；软件侧量化+调度为大头 | D2/D3 的方法论背书：量化收益排序 = 权重激活 → KV → 调度 |
| 8 | **AMD Developer Challenge / GPU MODE**（2025–2026） | MI300X/MI355X | 冠军打法：megakernel（整个 decode 循环驻留单 kernel）、消除 launch 开销、LDS scratchpad、TileLang 手写汇编级 kernel 比 Triton AMD 后端快 ~1.53× | decode 1.9–3.3×（ROCm megakernel 博客） | 2B 小模型 decode 段 launch/访存占比高 → 融合与 graph 路径收益大；TileLang 是报告创新点素材 |
| 9 | **NeurIPS LLM Efficiency Challenge**（2023，复盘 2025） | A100/4090 | 夺冠配方 = 简单且调透的配置（QLoRA+FlashAttn+paged AdamW），非花活 | — | 「简单方案彻底调参」胜过复杂方案；对应我方把 D0 参数网格搜到极致 |
| 10 | **沐曦「揭榜挂帅」**（2026） | C500 / FusedMoE-GEMM、MLA、NSA | 官方主推 **TileLang+MXMACA**，赛题即前沿算子；AI Agent 自动调优范式 | 擂主 10 万元 | 沐曦自家生态重心在 TileLang；Triton 之外的报告创新叙事 |
| 11 | **MiniCPM-o 4.5 × 昇腾挑战赛**（2026.7） | 昇腾 NPU | 评分维度：TTFT/chunk latency/E2E/并发/资源利用率/精度损失/复现质量 | — | 评分维度与我们的「性能+报告」双轨同构，报告写法可参考 |
| 12 | **KernelGen 24h 五芯算子赛**（2026 WAIC） | 昇腾/天数/海光/摩尔/沐曦 | Triton/Triton-TLE 一次开发多芯运行；launch 参数+访存模式是得分关键 | — | 证明「同一 Triton kernel 分芯片重调参数」是 FlagOS 生态公认打法（D0 合法性与方向背书） |

---

## 1. FlagOS 生态自身（最高优先级）

### 1.1 赛季一（2026.1–6）概况
- 主办：智源 BAAI × 众智 FlagOS 社区，奖金池 200 万元，三赛道：算子开发（FlagGems，20 题）、**推理吞吐优化（Qwen3-4B，即赛道二）**、自动数据标注。
- 2026-06-13 智源大会颁发赛道一/三；**赛道二结果 7 月中旬公布，但至今未检索到公开的获奖名单与技术复盘**（知乎/DOIT/官网/CSDN 均无）。赛道一获奖者 Yunyi Liu（悉尼，海外参赛者），代码按指定开源证并入官方算子库。
- 结论：赛季一赛道二**没有可抄的公开报告**，其真实答卷沉淀在两个地方——(a) vllm-plugin-FL 的 PR 痕迹（见 1.2）；(b) S2 赛题与评分规则相对 S1 的变化（见 1.3）。

### 1.2 案例一：S1 赛道二唯一可见提交 —— PR #178（E4_8 队，Ascend）
①来源：https://github.com/flagos-ai/vllm-plugin-FL/pull/178（closed，未合入）
②赛题与硬件：Qwen3-4B 推理吞吐，2×Ascend 910C
③做了什么：
- 删除 138 个非 Ascend 后端文件（-34,847 行），只保留 FlagGems 分发框架 + Ascend 后端；
- 增加逐场景 benchmark 调参脚本（HCCL AIV + FlagGems）；
- 声称 1K +86.2% / 4K +95.3% / 6K +102.4%。
④可借鉴：**按场景（输入长度×并发）分别保存最优配置并热切换**——4K 与 16K 的最优 launch 参数/graph 策略不同，应做成场景自适应而非全局一组参数。
⑤不能抄：大删上游文件不被接受（该 PR 未合入）；正确姿势是仓库已确立的「运行时 patch、wheel 不动」模式（`vllm_fl/patches/*.py`，如 moe_sum.py）。

### 1.3 案例二/三：S2 赛道二竞队直接情报（本赛季！）
S2 比赛规则是**获奖后才正式发 PR**（模板 PR #463/#413 均为占位），但已有两支队伍提前泄露：

**PR #671 RobertFengOPC（unified_attention launch 参数重标定）**
①https://github.com/flagos-ai/vllm-plugin-FL/pull/671（open，2026-11-02）
②BI-V150（非 sm100 平台）× MiniCPM5-2B（GQA 8:1、head_dim=128、42 层——注意其描述与我方 16Q/2KV 认知有出入，以官方 checkpoint 为准）
③做了什么（对我方最有价值的一手资料）：
- 瓶颈定位：vLLM v1 Triton `unified_attention` prefill 沿用 decode 导向默认 `BLOCK_M=16`，每 program 仅 2 个 query 位置，算力利用率 4.5/96.4 TFLOPS；decode 16 行 tile 配 8 warps/3 stages 属空转；
- **耦合重标定**：prefill `BLOCK_M 16→256`、`num_warps 4→16`、`TILE_SIZE 32→128`、TILE 门控阈值 `8192→0`；decode `num_warps 8→2`、`num_stages 3→1`；
- 实现：取 `unified_attention()` 源码做三处定点锚点替换后在原模块命名空间 `exec`，同时替换 `triton_unified_attention` 与 `triton_attn` 两处引用；锚点不全命中即整体回退；`FLAGOS_P1=0` A/B 开关；
- 实测（官方 benchmark_throughput_serve.py，RUNS=4 弃首轮）：4K 1818→3299.55 tok/s（**+81.5%**），16K 862→2583.27 tok/s（**+199.7%**）；TTFT −59%/−69%，TPOT −43%/−66%；跨场次复现偏差 ≤3.8%；
- 精度：prefill 形状逐位一致（max|diff|=0）；TILE 变更 <1 ULP；math_500 两次 97.1%/98.1%（基线 96.2%）；
- **C500 负结果（重要）**：P1 配置需 262,144 B 共享内存，C500 SM 上限 65,536 B → `OutOfResources`→EngineDead；降级参数后收益无意义。结论：**共享内存 ≥256KB 的卡才能吃大 tile；64KB 级设备需重做参数网格**。
④映射：与我方 D0 同族但**其参数远比我们激进**（prefill BM=256 vs 通常 BM=64/128；TILE 门控归零）。行动项：a) 复核我方 D0 是否覆盖了「prefill 大 BLOCK_M + warps=16 + TILE_SIZE=128 + 门控=0」这一格；b) 16K 的 +199.7% 说明 prefill 段仍是最大矿脉，尤其 16K=[16384,1024] prefill token 占绝对多数；c) C500 上我方必须做**独立的小 tile 网格**（其内存上限约束），不能复用 V150 参数。
⑤规则差异：其声称「launch 参数重标定属于内核优化而非调参」的论证（对照上游 `tuned_large_head` 分支先例）已被写进 PR 供评委裁量——我方 D0 报告应采用同样论证框架自保。

**PR #778 fanqh18（Python 层开销削减）**
①https://github.com/flagos-ai/vllm-plugin-FL/pull/778（open，2026-11-18，仅 2 文件 +96 行）
②V150+C500 双卡 × MiniCPM5-2B（同赛题、双卡都测）
③做了什么：
- `MetadataCachePool`：跨 decode 步缓存静态张量（slot_mapping 等）减少动态分配；
- 消除冗余 host-device 同步（`.cpu()`/`.item()`）；
- 16K@64 并发下 graph capture 显存池保护（超过 `VLLM_GRAPH_CAPTURE_MAX_BATCH_SIZE` 即置 `use_cuda_graph=False`）；
- `flag_gems.enable()` 触发 FlagGems 高精度融合算子（RMSNorm/SiLU/RoPE，FP32 register 累加）替换默认 aten 路径；
- 声称 V150 +23.9%(4K)/+21.3%(16K)，C500 +24.9%/+22.8%，math_500 98.7%。
④映射：**新增方向 D4「host 侧/框架开销削减」**——零精度风险、双卡通用、实现量小（~100 行 patch）。特别注意其中 `flag_gems.enable()` 一行就能换上 FP32 累加的融合 norm/激活算子，这同时是 D2 量化期间的精度保险。
⑤注意：其补丁极薄而声称收益 +20%+，含金量需存疑（metadata cache 反而多一次 `copy_`）；但「graph capture 与 64 并发/16K 显存的冲突保护」与我方 D1 watermark 思路互证，报告可互引。

### 1.4 案例四：vllm-plugin-FL 已合入 perf PR 金矿（同框架被接受的优化，全量扫描 611 PR 后精选）

| PR | 类别 | 内容 | 收益 | 对我方启示 |
|---|---|---|---|---|
| [#384](https://github.com/flagos-ai/vllm-plugin-FL/pull/384) 合入 | 图/路由 | Nsight 定位 Hopper 16K：Triton unified-attn 比 FA3 慢 11×（占差距 76.2%）；FlagGems `mm` 慢于原生 GEMM → 架构级 dispatch profile：attention 路由 vendor 后端 + 拉黑 FlagGems mm，环境变量 A/B | 535→692 tok/s（+29%） | **方法论**：逐算子 profile（V150/C500 上 FlagGems 全表未必最优）；「拉黑单个算子回退原生」是零风险高收益动作 |
| [#381](https://github.com/flagos-ai/vllm-plugin-FL/pull/381) 合入 | 调度/显存 | CUDA graph 池显存计入 KV cache sizing，避免高并发下 KV 过分侵占 graph replay 显存 | 正确性型修复 | 与我方 D1 watermark 预留同族；可检查我方 D1 是否也覆盖 graph 池 |
| [#417](https://github.com/flagos-ai/vllm-plugin-FL/pull/417) 未合 | 图 | 昆仑xon 后端误报 `AttentionCGSupport.NEVER` → decode 被降级 PIECEWISE；加 `UNIFORM_BATCH` 后 decode 走 FULL CUDA Graph | decode 大项 | **立即检查 V150/C500 attention backend 的 CGSupport 上报**——若同样被降级，恢复 FULL graph 是白捡收益 |
| [#414](https://github.com/flagos-ai/vllm-plugin-FL/pull/414) 合入 | 调度 | AsyncSchedulerFL + xGrammar 掩码批量填充 + 调优 MoE Triton config | 归一化吞吐 +89%（含任务特性） | 我们无结构化输出需求，仅借鉴「调度器异步化」叙事 |
| [#299](https://github.com/flagos-ai/vllm-plugin-FL/pull/299) 未合 | 并行 | prefill 序列分片 CP + FlagCX PD，P 侧 CP/L-1 对齐 | 32K +33%、64K +38% | 多卡才可用；若我方单卡则排除，报告可作 related work |
| [#264](https://github.com/flagos-ai/vllm-plugin-FL/pull/264)/[#372](https://github.com/flagos-ai/vllm-plugin-FL/pull/372)/[#378](https://github.com/flagos-ai/vllm-plugin-FL/pull/378) | 量化 | Hygon W8A8 INT8（Qwen3.5）、MetaX DeepSeek-V4 W8A8、MetaX GLM-5.2 W8A8 均为**后端内置支持** | — | **D2 的框架先例**：C500 上 W8A8 有现成适配路径，照抄其接入方式；V150 需自证 |
| [#409](https://github.com/flagos-ai/vllm-plugin-FL/pull/409) 合入 | 图 | Ascend 全 decode graph 模式 | — | 图 decode 是各后端共同方向 |
| [#244](https://github.com/flagos-ai/vllm-plugin-FL/pull/244) | 算子 | split_qkv_rmsnorm_mrope 融合（Ascend, Qwen3.5/3.6） | — | MiniCPM5 无 mrope，但 qkv+rmsnorm 融合思路可查 FlagGems 是否有现成核 |
| [#406](https://github.com/flagos-ai/vllm-plugin-FL/pull/406) 合入 | 基建 | iluvatar（天数）Triton patch 统一收敛到模块级 | — | V150 的 triton patch 集中地，读代码找 V150 已知短板 |

FlagGems 仓库（flagos-ai/FlagGems，~1000 PR）无带比赛标记的 PR——S1 赛道一获奖算子以普通标题合入；对赛道二有直接参考价值的是其 attention/gemm/rmsnorm 的 tuning config 目录结构（按形状分档的 autotune 键），与我方 D0 的 launch 调优同构。

### 1.5 KernelGen / 算子赏金赛（FlagOS 自家算子赛，简述）
- KernelGen（2026 WAIC）：24h、3 题、$3000，Triton/Triton-TLE 为昇腾/天数/海光/摩尔/沐曦五芯写算子——官方叙事承认「一次开发、分芯片调参」的合法性；
- 72h 赏金赛（CSDN 复盘均为直播预告页，无文字干货）。
- 对我方：报告里可引「KernelGen 五芯同源 kernel 分芯调参」为 D0 正名。

---

## 2. 国产芯片厂商比赛

### 2.1 案例五：沐曦「揭榜挂帅」擂台赛（2026.5–11）
①https://github.com/MetaX-MACA/op_optimization（README 全文已抓）
②曦云 C500（64GB）+ MXMACA + **TileLang**（官方主推语言，非 Triton）
③赛题：初赛 Fused MoE GEMM、决赛 MLA / NSA（原生稀疏注意，号称显存省 50%+）；赛题二为 AI Agent 自动调优范式（算子迁移/性能分析/自动 Kernel 优化/Benchmark 迭代全流程）。奖励：擂主 10 万 + 2 张高端 GPU。
④映射：a) 沐曦官方生态重心在 TileLang——若 FlagOS 的 Triton-TLE 在 C500 可用，「attention 热核用 TLE/TileLang 重写」可作 D3 之外的**报告创新点**；b) 其 NSA 方向说明长序列注意力稀疏化是厂商认可路线，但改注意力语义有精度风险，我方红线内不宜动。
⑤差异：该赛是纯算子赛（单算子 vs baseline），无端到端精度门槛；我方必须端到端 math_500 ≥0.95。

### 2.2 案例六：MiniCPM-o 4.5 × 昇腾推理优化挑战赛（2026.7，进行中）
①https://www.mindspore.cn/activities/zh/2026-7-13（官网 ascend.openbmb.cn）
②MiniCPM-o 4.5 全模态 × 昇腾 NPU（统一环境复现，每天限提交 3 次）
③高性能赛道评分维度：适配、TTFT、chunk latency、E2E latency、吞吐/并发、资源利用率、精度损失控制、复现质量。
④映射：我方 30% 创新分由技术报告定——按其评分维度组织报告章节（定位方法学→分项收益→复现脚本）是同类比赛公认加分结构；「复现质量」对应我们 PR 里的可执行验证脚本（#671 的 verify_plugin_*.py 值得模仿）。
⑤差异：无公开获奖复盘（赛程未完），只取评分框架。

### 2.3 共性打法小结（厂商赛）
Triton/TileLang 算子调优（launch 参数+访存）、融合（qkv+norm、MoE 路由）、量化（W8A8/INT4 为各厂商标配能力）、显存管理（KV 分页/量化）、端侧小模型重参数——与我方 D0–D3 完全同谱。厂商赛几乎不做投机采样（无法保证精度可控），与我们的禁令一致。

---

## 3. 国际同类

### 3.1 案例七：MLPerf Inference Closed Division（方法论同构：固定模型+精度门槛+极限优化）
①https://mlcommons.org/2025/04/llm-inference-v5 等（v5.0/v5.1）
②Llama 2 70B / Llama 3.1 8B & 405B，Closed Division 精度门槛 = FP16 参考的 99%（与我方 math_500≥0.95 同构）
③厂商配方（NVIDIA H200/TRT-LLM、AMD MI300X、HPE）：
- **FP8/INT8 权重+激活量化**（ModelOpt PTQ）；
- **KV cache 量化（INT8/FP8）**——显存大头，直接换取更大 batch/更长序列；
- in-flight/continuous batching + 分页 KV；
- 算子层按架构选 kernel（TRT builder tactic 自动选优；AMD v5.1 新增 FP8 技巧、MoE kernel 调优）；
- v5.1 NVIDIA 相对 v5.0 纯软件再 +15.4%（LLM 项）——说明同一硬件上软件仍有两位数空间。
④映射：收益排序对我方 D2→D3 排期给出背书：W8A8（权重+激活）先做 → KV-int8（D3）第二 → 调度/graph 收尾。注意 2B 模型 weight 常驻小（~4GB bf16），16K×64 并发的 KV 与激活才是显存/带宽压力中心，**KV int8 的杠杆在 16K 场景可能超过 W8A8**。
⑤差异：MLPerf 允许改 batch size 等服务超参（closed 也允许 batching 策略自由）、允许 FP8 KV——我方 serve 参数锁死，量化必须做在 kernel/插件内部且过 math_500 红线；投机采样 MLPerf 允许（部分轮次）、我们被禁。

### 3.2 案例八：AMD Developer Challenge 2025 + GPU MODE 内核赛（非 CUDA 后端经验）
①https://www.datamonsters.com/amd-developer-challenge-2025；GPU MODE $100K LLM kernels on MI300X；后续 $1.1M MI355X DeepSeek-R1 speedrun
②MI300X/MI355X；任务=MLA+RoPE attention、Fused MoE routing、FP8 GEMM、All-to-All（8 卡）；KernelBot 自动评测
③获奖手法（复盘共识）：
- **megakernel/monokernel**：整个 decode 循环做成单驻留 kernel，消除 launch 开销与 HBM 往返（ROCm 官方博客：decode 1.9–3.3×）；
- 用 **CDNA LDS（compute-unit 本地 scratchpad）** 做 tile 缓冲；
- **TileLang/汇编级手写 kernel 比 Triton AMD 后端快 ~1.53×**（MI300X attention）——非 CUDA 后端上 Triton 代码生成质量是公认短板；
- 通信-计算重叠（all-to-all writeup：自定义 HIP dispatch/combine）。
④映射：a) 2B 模型单层计算薄，decode step 的 launch/框架开销占比高——V150/C500 上「Triton 后端代码生成差」同样成立，**FlagGems kernel 的手写级 micro-opt（向量化加载、swizzle、pipeline stage 数）还有一层矿**；b) megakernel 太重不宜照搬，但「graph 全覆盖 decode + 减少 kernel 数（融合）」是其温和版；c) 报告可引用「Triton 在非 CUDA 后端的生成质量差异」论证我方逐芯调参必要性。
⑤差异：内核赛无端到端精度门槛与 serving 限制；megakernel 与 vllm 0.24 调度器不兼容，不可抄。

### 3.3 案例九：NeurIPS LLM Efficiency Challenge（2023，arXiv 2503.13507 复盘）
①https://arxiv.org/abs/2503.13507
②1 LLM + 1 GPU + 24h 微调赛（A100/4090）——注意这是**训练侧**效率赛，与推理不同
③复盘结论：头部方案全是「简单但调透」的配方（QLoRA 4-bit + FlashAttention + paged AdamW 32bit），复杂方法无一胜出；A100 冠军同时赢开放赛道。
④映射：唯一可迁移的元教训——**把一个正确方向（如 D0 launch 调优）的参数网格搜到极致，胜过铺开五个半吊子方向**。竞队 #671 正是这么做的（同方向，参数更彻底，收益 2–4 倍于我方）。
⑤差异：其允许量化微调、无 serving 参数限制。

### 3.4 AI 编程类比赛中的推理优化案例
未检索到以推理性能为评分主体的 AI 编程赛（各「AI Agent 编程赛」均为功能/质量评分）；唯一沾边的是沐曦揭榜挂帅赛题二（AI Agent 做 kernel 自动调优），归入 2.1。

---

## 4. 映射总表：案例招式 → 我方行动

| 我方方向 | 状态 | 调研结论 | 行动项 |
|---|---|---|---|
| **D0 attention launch** | 已合入 +42/48% | 竞队 #671 同方向 +81.5/+199.7%：prefill BM=256/warps=16/TILE=128/门控=0、decode warps=2/stages=1 | 补网格：prefill 大 BM 格未搜过则立即补；16K prefill 段重估；C500 因 64KB 共存上限需独立小 tile 网格（防 OOR 崩溃） |
| **D1 调度/watermark** | 已合入 +0.4/1.1% | #381（graph 池计入 sizing）同族；#778 的 graph-batch 门控同族 | 复核 graph 池显存项；检查两后端 `AttentionCGSupport` 是否被误降级 PIECEWISE（#417 教训） |
| **D2 W8A8** | 进行中 | MLPerf 配方第一项；框架已有 Hygon/MetaX W8A8 先例（#264/#372/#378）；#778 用 FP32 累加融合算子保精度 | C500 走 #372 接入模式；量化后配合 FlagGems FP32 累加 norm/激活保 math_500 |
| **D3 KV-int8** | 备选 | MLPerf 配方第二项（KV 量化换 batch/长度空间）；16K×64 并发下 KV 是显存/带宽大头 | 升级优先级：16K 场景收益潜力可能大于 W8A8；需 attention kernel 配合（与 D0 补丁同一落点） |
| **新增 D4 host 开销削减** | 未启动 | #778 声称 +20%+：元数据缓存、消 `.item()/.cpu()`、`flag_gems.enable()` 一行换融合算子 | 一天工作量内做 profile（CPU 侧火焰图）验证，性价比极高 |
| **新增 D5 逐算子路由/拉黑** | 未启动 | #384：FlagGems 不必然处处最优；11× 差距案例 | 在 V150/C500 上对 FlagGems 全算子 A/B（ aten vs gems），拉黑慢算子；环境变量 A/B 已是仓库惯例 |
| **报告创新分 30%** | — | MLPerf/昇腾赛的报告结构；KernelGen 分芯调参叙事；AMD TileLang>Triton 论据 | 报告按「定位方法学→分项消融→复现脚本→普适性论证」组织；引 #671 式「内核优化 vs 调参」论证模板 |

## 5. 规则红线对照（不能抄清单）
1. **MLPerf 的 FP8 KV + in-flight batching 自由度**：我方 serve 参数锁死、禁超参开量化——量化/调度改动必须落在插件与 kernel 内部，不得出现在 serve 命令行。
2. **投机采样**：MLPerf/部分厂商赛允许，我方明令禁止（MTP PR #775、#258 只可作 related work）。
3. **megakernel/monokernel**：与 vllm 0.24 调度器不兼容，工程量与风险越界。
4. **PR #178 式删库瘦身**：上游不接受（未合入）；一切走运行时 patch 模式。
5. **PR #671 的 C500 参数**：直接复用会 EngineDead（共享内存 4× 超限）——C500 必须独立网格。
6. **NeurIPS 赛的量化微调**：我方无训练环节，不可比。

## 6. 主要来源
- vllm-plugin-FL PR（curl GitHub API 实测抓取）：#671、#778、#178、#463、#384、#381、#417、#414、#299、#264、#372、#378、#409、#244、#406 等，https://github.com/flagos-ai/vllm-plugin-FL
- FlagGems：https://github.com/flagos-ai/FlagGems（1000 PR 扫描）
- FlagOS 赛季一颁奖（赛道一/三）：https://zhuanlan.zhihu.com/p/2050920091970188492 ；https://www.doit.com.cn/computing/816315045126213.html ；https://flagos.io/events?tab=competition
- S2 赛道二 AMA（评分细则）：https://zhuanlan.zhihu.com/p/2102131327101297410
- 赛季一赛道二赛题：https://zhuanlan.zhihu.com/p/2003203433658851341
- 沐曦×FlagOS S2 赛道二赛题发布：https://www.metax-tech.com/ndetail/12669.html
- 沐曦揭榜挂帅：https://github.com/MetaX-MACA/op_optimization
- MiniCPM-o 4.5 × 昇腾：https://www.mindspore.cn/activities/zh/2026-7-13 （ascend.openbmb.cn）
- KernelGen：https://zhuanlan.zhihu.com/p/2040381415238415770
- MLPerf：https://mlcommons.org/2025/04/llm-inference-v5 ；https://developer.nvidia.com/blog/nvidia-h200-tensor-core-gpus-and-nvidia-tensorrt-llm-set-mlperf-llm-inference-records
- AMD/Data Monsters：https://www.datamonsters.com/amd-developer-challenge-2025 ；ROCm megakernel：https://rocm.blogs.amd.com/software-tools-optimization/megakernel-intro/README.html ；MI300X a2a writeup：https://gau-nernst.github.io/amd-a2a
- NeurIPS 效率赛复盘：https://arxiv.org/abs/2503.13507 ；https://github.com/llm-efficiency-challenge/neurips_llm_efficiency_challenge
- 天数开发者活动：https://developer.iluvatar.com/events
