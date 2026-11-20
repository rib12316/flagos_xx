# Agentic 优化循环开源项目全景调研 —— FlagOS S2 赛道二可用性评级

调研日期：2026-11-14。方法：WebSearch + WebFetch/webReader（github.com 直接抓取受证书限制，改用 raw README 抓取与搜索交叉验证）。

## 0. 评级标尺（比赛环境回顾）

| 维度 | 约束 |
|---|---|
| 硬件 | 天数智芯 BI-V150 32GB（flagtree Triton 3.2 后端）+ 沐曦 C500（MACA 栈）；均**无 NCU/英伟达工具链** |
| 软件落点 | 仅 vllm-plugin-FL（vllm 0.24 插件）+ FlagGems v5.3.5 两处代码 |
| 工作模式 | 实例上通用 coding agent + 自建 RSI 循环（假设台账→patch→micro-bench→官方 bench→精度 gate）；bench 仅 stdout 输出 |
| 红线 | serve 参数锁死、禁 bench 调优、禁超参开量化/投机采样、精度 ≥0.95 |

**评级符号**：✅ 可直接用 / 🔧 改造后可用（注明改什么）/ ❌ 不可用（原因）。所有评级以"能否落地到 vllm-plugin-FL + FlagGems + 无 NCU 环境"为准。

---

## 1. 总览速查表

| 项目 | 一句话定位 | License/开源 | 评级 |
|---|---|---|---|
| **AKO4X** | campaign 式多 agent kernel 闭环优化框架（AKO4ALL 的进化版） | MIT，2026-05-31 开源 | 🔧（benchmark 适配器+技能卡替换） |
| **AKO4ALL** | 单文件 SKILL.md kernel 优化 harness（我方主循环候选） | MIT | ✅（已在用） |
| **GEAK v4 (AMD-AGI)** | e2e_workflow（sglang/vLLM serving 吞吐）+ kernel_workflow 双 agent 系统 | Apache-2.0 | 🔧（剥离 AMD/rocprof） |
| **Hyperloom (AMD-AGI)** | 全栈自主推理优化 harness（框架 flag/env/backend/PR backport + kernel 委托） | MIT | 🔧（剥离 AITER/rocprof，接 vllm-plugin-FL） |
| **Triton8 (IntelLabs)** | Intel GPU 的 LLM 驱动 Triton kernel 生成优化循环（trial tree + KB） | Apache-2.0，**已归档** | 🔧（归档无妨，当模板 fork） |
| **KernelEvolve (Meta)** | 异构加速器（NVIDIA/AMD/MTIA）kernel 进化框架 | ❌ 不开源（论文+博客） | ❌（抄方法论） |
| **KernelAgent/KernelFalcon (meta-pytorch)** | NCU 硬件引导的多 agent Triton kernel 优化 | 开源（meta-pytorch/KernelAgent） | ❌（NCU-first、CUDA-only） |
| **KernelGen + flagos-ai/skills（官方）** | FlagOS 生态 AI 自动 Triton kernel 开发平台 | 开源 | ✅（官方在用） |
| **Kernel-Smith（上海AI Lab+沐曦）** | 评估驱动进化 kernel 生成系统，MXMACA 深度支持 | **权重与 agent 代码不开源** | ❌（仅 arena 可玩） |
| **KDA (NVIDIA)** | 研究-实现-验证-迭代的 kernel 设计 agent 工作流，MLSys'26 FlashInfer 竞赛冠军 | 开源（NVlabs/kda、mit-han-lab/kernel-design-agents） | ❌（CUDA/NCU），方法论可抄 |
| **KDA-Pilot (BBuf)** | KDA 方法论在 SGLang 的应用：冻结生产形状的轻量 prompt 任务库 | 开源 | 🔧（方法论直接移植） |
| **SpecPilot (Berkeley)** | 投机采样无损 agentic 优化（我方已吸收其方法论） | 开源（StargazerX0/SpecPilot） | ❌（本赛禁投机采样） |
| **AgentKernelArena (AMD-AGI)** | 196 任务的 agent kernel 优化泛化性评测平台（HIP/Triton/FlyDSL） | 开源 | ❌ 平台/✅ 方法论（任务切分） |
| **AutoTriton (THUDM+华为)** | SFT+GRPO 训练的 8B Triton kernel 生成模型 | 开源（模型+数据） | ❌（是模型不是循环） |
| **cuPilot** | 策略协同多 agent CUDA 生成（论文+kernel 仓库） | 仅 kernel 开源 | ❌（CUDA-only） |
| **AutoKernel (OAID)** | TVM 调度自动搜索（非 LLM 时代遗产） | Apache-2.0 | ❌ |
| **AutoKernel (rightnow-ai)** | "Autoresearch for GPU kernels"：单文件→固定评测→保留/回滚 | 开源 | ❌（CUDA）/✅（循环范式同构，AKO 灵感源） |
| **cuda-kernel-optimizer / autocuda** | LangGraph/autoresearch 式 CUDA 优化循环 | 开源 | ❌（CUDA-only） |
| **RoofLang (MSRA)** | LLM 推理系统架构 DSL + agent 设计空间探索 | 论文（arXiv 2609.12551），未见代码 | ❌（借 action space 设计思想） |

---

## 2. 主循环与 campaign 类

### 2.1 AKO4X（TongmingLAIC/AKO4X）—— 重点补查对象

**① 定位**：closed-loop、campaign-based 多 agent GPU kernel 优化框架，benchmark 可换（默认 FlashInfer bench），号称"在数小时内达到专家级 kernel 性能，virtually any hardware"。

**② 循环架构（本次调研核心增量）**：三种模式。
- **Mode 1 Manual**：手动启动子 agent 单轮优化，用户可会话内介入。
- **Mode 2 Closed-loop（campaign 核心）**：一个**持久主 agent** 连续运行多轮 campaign：Step1 基准采集与索引 → Step2 热点分析（GPU 时间/调用次数/忙闲占比）→ Step3 **锁定 FROZEN scope**（bench harness 只读）→ Step4 从 per-operator **reference archive** 注入先验 → Step5 全新隔离子 agent 实现完整五步内核（baseline→分析→方案→实现→评审）→ Step6 追加归档 → **Step7 主 agent 门控**（任何越界 patch 必须回滚并记录）→ 下一轮。每个算子一个 **append-only archive**（kernel.py 头部结构化字段：Identity/Delta/Lessons/Dead-ends）+ 跨算子 **TRAPS.md** 静默缺陷目录 + master/harness-ledger.md。
- **Mode 3 Harness Co-evolution（实验性）**：harness 与 kernel 共同进化，通过 **evidence-gated PROPOSALS.md**（适用条件+证据链+影响面）申请解冻 harness；作者自评"append-biased, master reactive-only"——诚实标注了不成熟。
- SKILL 卡目录 9 张：Agent-principles / Autonomy / Profiler-NCU / Sanitizer / Triton / CUDA / CuTe DSL / TileLang / C++。
- 结果：**MLSys-2026 FlashInfer 竞赛全部 5 个 kernel 上击败 KDA**（B200 上，如 DSA sparse attention 30.71x vs KDA 3.78x geomean）；AKO4X 论文：10 kernel 族 471 工作负载 9 族胜 FlashInfer 专家基线。
- 生态位确认：灵感来自 karpathy/autoresearch 与 RightNow-AI/autokernel；**dsh-kernel-opt 插件（2026-08-17）**= DeepSeek Harness 上的 AKO4ALL 求值器集成，可观测可转向。

**③ 硬件/工具依赖**：benchmark 适配器协议（`scripts/benchmark_adapter.py`：run/pack/solution_meta/list_workloads + 可选 profile/list_ncu_options/sanitize/cheat_check）设计上可换，但**默认 flashinfer-bench 要求 CUDA Driver ≥13.0**；NCU/Sanitizer 技能卡不可用于我方。

**④ 开源状态**：MIT；README 明示 open-source 定位；campaign 机制结构完整但为唯一维护者项目，Mode 3 实验性；roadmap：每轮并行 subs、跨算子 share、能力自适应 harness（"北极星"目标）。

**⑤ 本赛可用性**：🔧 **改造后可用（强烈推荐作为主循环升级）**。需改三处：(a) 按 `PORTING_GUIDE.md` 写我方 benchmark_adapter（run=官方 bench stdout 解析、pack/micro-bench 自定义）；(b) 删除 Profiler-NCU/Sanitizer/CUDA/CuTe/TileLang 技能卡，改写 flagtree Triton 3.2 + MACA Triton 技能卡 + torch profiler 采样；(c) Mode 2 起步，Mode 3 不碰。campaign 的 append-only archive + TRAPS.md 与我方"假设台账"同构，可直接对齐。

**⑥ 与 AKO4ALL 互补**：AKO4ALL 是"单算子单轮"的技能化封装，AKO4X 给它补上**多轮组合优化需要的持久记忆结构**（archive/TRAPS/ledger）与主 agent 门控——正是我方 RSI 台账机制的外部印证与升级蓝本。

### 2.2 AKO4ALL

已有结论保持：✅ 我方主循环候选。新增情报：其评测器已被 dsh-kernel-opt 插件移植为 DeepSeek Harness 集成，证明"SKILL.md harness + 固定求值器"模式可插拔到任意宿主 agent——我方 ZCode/Claude Code 实例同理。

### 2.3 轻量循环同类（参考价值）

- **rightnow-ai/autokernel**："agent 修改单个文件→固定评测→保留/回滚"，AKO 系灵感来源，与我方 micro-bench gate 完全同构。CUDA 向导内容不可用，循环骨架无需借鉴（已自建）。
- **JackTan25/autocuda**：karpathy/autoresearch 变体。❌ CUDA-only。
- **bertaye/agentic-cuda-optimizer**（即 HN 上的 cuda-kernel-optimizer）：LangGraph 编排，compile→correctness→profile→iterate。❌ CUDA/NCU。
- **ZJLi2013/awesome-kernel-skills**：从 2026-03 论文版 AutoKernel（自主 agent 循环）蒸馏的技能库，`.cursor/rules` 可接。🔧 值得抽取其知识条目进我方 SKILL 卡，循环本体不用。

---

## 3. Meta 系

### 3.1 KernelEvolve（重点补查对象）

**① 定位**：Meta 面向 DLRM 训练/推理的异构加速器 agentic kernel 进化框架（arXiv:2512.23236，ISCA 2026，Gang Liao 组 39 人）。

**② 循环架构**：六组件——LLM Synthesizer（多 DSL：Triton/TLX/CuTe DSL/FlyDSL + 低层 CUDA/HIP/MTIA C++；动态上下文感知 prompt）→ **Tree Search Engine**（MCTS+进化策略；每个 kernel 是节点，节点携带可配置"memory operator"决定继承父轨迹/比较兄弟/合并/重启）→ **分层检索增强 KB**（正确性约束/平台无关优化通识/硬件专属文档；按运行时信号动态检索——如带宽瓶颈触发 memory-hierarchy 文档；成功策略蒸馏回写 KB = in-context RL）→ 自动评估框架（TritonBench 对 PyTorch 基线做正确性；profiling 栈含 **PyTorch Profiler、NCU、Proton、MTIA Insight**）→ 跨会话共享数据底座（tiling 策略等 insight 持久化，后来者从近优点起步）→ agentic RL（轨迹训练小模型）。终止条件：达标/预算耗尽/停滞。

**③ 硬件依赖**：NVIDIA + AMD + **MTIA**。关键句：MTIA 无任何 LLM 训练数据，Meta 把手册/指令集/内存层级**注入 KB，inference-time"学会"新芯片；新芯片只需文档策展，不需手写 kernel**。异构=同一 universal prompting interface + 按硬件注入知识。

**④ 开源状态**：❌ **不开源**（论文 CC-BY + 工程博客）。相邻工具开源（TLX、TritonBench）。

**⑤ 本赛可用性**：❌ 不可用（无代码）。**但方法论价值全表最高**，对我方处境三点直接启示：(a) BI-V150/C500 同样是"LLM 没见过的芯片"——把 flagtree/MACA/Triton 方言文档、FlagGems 已有算子、天数/沐曦手册做成分层 KB 注入我方 agent，等价于 Meta 对 MTIA 的做法；(b) profiling 栈证明**PyTorch Profiler 可作为 NCU 缺位时的结构化诊断源**；(c) 跨会话 insight 复用对应我方台账的" Lessons/Dead-ends"字段。结果参考：>60% 推理吞吐提升（Andromeda/NVIDIA）、>25% 训练吞吐（MTIA）、KernelBench 250 题 100%。

**⑥ 与 AKO4ALL 互补**：AKO4ALL 有循环没知识库——KernelEvolve 的"文档策展→分层 KB→信号触发检索"是给我方 SKILL 卡体系补分层检索的蓝本。

### 3.2 KernelAgent / KernelFalcon（meta-pytorch/KernelAgent）

**① 定位**：NCU 硬件计数器反馈引导的多 agent Triton kernel 优化（KernelFalcon 的 deep-agent 分层任务分解为其前身形态，代码同仓库）。

**② 循环架构**：Profiling Agent（NCU roofline 式瓶颈诊断）→ 优化 → 验证（KernelBench 250 题 100% 正确）。分层：planner/executor/verifier。

**③ 硬件依赖**：**NCU-first**。虽有 `KERNEL_PROFILER_PYTHON` 环境变量（Buck 构建下 NCU 修复，PR#137），但这是修 bug 不是后端抽象层——profiler 可换性没有产品化设计。CUDA-only。

**④ 开源状态**：开源（meta-pytorch org），活跃（2026-03 官方博客）。

**⑤ 本赛可用性**：❌（NCU+CUDA 硬依赖）。可抄：诊断→修复→验证的三段式 agent 分工，与 KernelFalcon 的层级任务分解（planner 委派 subagent）模式。

**⑥ 与 AKO4ALL 互补**：其"Profiling Agent"角色在我方无 NCU 环境下退化为 torch profiler + 手写计时 wrapper，这一退化写法可参考 Triton8（见 §5.1）。

---

## 4. AMD 系（非 CUDA 后端经验最丰富的开源生态）

### 4.1 GEAK v4（AMD-AGI/GEAK）—— 重大新发现

**① 定位**：开源多 agent GPU 性能优化器；**v4 双工作流：e2e_workflow（整机 sglang/vLLM serving 吞吐优化）+ kernel_workflow（单 kernel 优化）**，v3 是 baseline（miniswe 编排）。

**② 循环架构**（README 全文抓取，关键机制）：
- **e2e_workflow**（对我方价值最大）：preflight（装备+候选构造）→ **profile**（actor/非 actor 分开，分辨率>150Hz/300Hz CPU/CUDA 采样）→ **Amdahl triage**（热 kernel 按 `pct_gpu_time × achievable_speedup` 排序定收益上限）→ **cheapest levers first**：config/backend sweep → GEMM/attention 实现烤场（bake-off）→ editable-kernel 里程碑循环（judge 强制存在差分 patch、未编译防退步路径、非线性采样、commit 归档）→ **reversible overlay**：以实测 warm-server 吞吐 delta 为唯一门控（**interleaved A/B**、串行化基线对比、<0.5% 内不信任、需 engagement 证明 + **output parity 检查**防不编译回归）→ 基线前进后才开下一杠杆。
- **kernel_workflow**：Director（预算+可换 evaluator）→ TechLead（计划，不碰 kernel）→ 专业 subagents（algorithm/memory/compute/host_runtime，3 份候选+排错，预算受控）→ 流水线验证（hypothesis ledger 与我方同名！）→ **insight blackboard**。KB 产物 kb_artifacts/ gitignore 不提交，作 warm-start。
- 由 Claude Code 驱动、确定性 JS workflow 编排（也可 codex CLI）。

**③ 硬件依赖**：AMD Instinct MI300X/MI325X/MI350X（gfx942/950）、RDNA3.5、R9700；ROCm 6.3.4+；profiler=rocprof-compute/rocprofv3/rocprof；gfx 自动检测硬编码；aiter per-shape tune；adapters/{sglang,vllm,atom}.sh。

**④ 开源状态**：Apache-2.0，AMD-AGI 官方活跃维护（v3 2026-07 博客、v4 更新）。

**⑤ 本赛可用性**：🔧 **改造后可用（serving 层首选蓝本）**。改动清单：删 gfx 自动检测与 rocprof（profiling 步用 torch profiler/自写 CUDA-event 计时替代）、删 aiter（换 FlagGems）、vllm adapter 改接 vllm-plugin-FL 的 bench stdout、attention/GEMM bake-off 池换 FlagGems 内候选；**保留其精华**：Amdahl triage、cheapest-lever-first、interleaved A/B + 0.5% band + output parity、reversible overlay。结果参考：e2e 报告 ~60% 真实 workload serving 提升；GEAK v3 在 AgentKernelArena 上 repository 级优化验证。

**⑥ 与 AKO4ALL 互补**：AKO4ALL 管"单算子怎么改"，GEAK e2e 管"整机先改哪、怎么证明改对了"——直接补我方"多杠杆排序+防噪声 A/B"缺口。

### 4.2 Hyperloom（AMD-AGI/Hyperloom）—— serving 级 agentic 优化器的标杆

**① 定位**：端到端自主 LLM 推理优化多 agent harness（AMD Instinct），优化对象覆盖**框架 flag/环境变量/权重与 KV 精度/attention 后端/批调度上限，甚至 backport 已合并的上游 PR diff**（vLLM、SGLang、xDiT），外加源码级 in-tree patch、AITER/sgl-kernel 重编译，kernel 级委托 GEAK 或 KernelForge 后端。

**② 循环架构**：四角色——Orchestration（全程一个持续会话，planner 不冷启动）+ **Critic（每次 keep-or-revert 决策用全新上下文，patch 无 Critic 双签不合并）** + Robustness（熔断：停滞/崩溃/循环）+ Specialist（短命任务级，只回 diff 不做决策）。会话阶段：Prelude（基线、KB warm replay、**roofline 分析**）→ Framework 优化（启动失败六步恢复阶梯）→ Kernel 优化 → Sweep（并发×输入输出长度网格）→ Close（Recipe KB 回写、报告）。**关键纪律：bench 一律由协调器跑，"agent 报的数字永不当真"；收益必须端到端复测且精度不低于基线**。

**③ 硬件依赖**：AMD Instinct（MI355X 等）；rocprof 系 profiling；AITER。

**④ 开源状态**：MIT，开源，有 ROCm Docs 文档。结果：16 workload 中位 **1.73x** 吞吐（1.35–7.31x），Llama-3.1-8B on MI355X 3.48x；声称 14,000+ 模型被优化过（生态规模存疑，方向明确）。

**⑤ 本赛可用性**：🔧 **改造后可用（vLLM 侧改造蓝本）**。它已有 vLLM/SGLang adapter——接 vllm-plugin-FL 的距离是全场最近的；需剥离 AITER/rocprof（roofline 数据源改为 stdout bench + torch profiler），且**必须裁剪搜索空间以遵守我方红线**（serve 参数锁死→删 flag sweep 阶段；精度≥0.95→保留其 accuracy gate 设计；禁投机采样→无此模块）。其 Critic 双签 + "agent 数字不作数"纪律可直接写进我方主循环规范。

**⑥ 与 AKO4ALL 互补**：AKO4ALL+AKO4X 覆盖 kernel 层，Hyperloom 补**框架/插件层**（vllm-plugin-FL 里的调度、图、后端选择 patch）与"会话级上下文持久化"工程化写法。

### 4.3 AgentKernelArena（AMD-AGI/AgentKernelArena）

196 任务、泛化感知（generalization-aware）的 agent kernel 优化评测平台（arXiv:2605.16819），HIP/Triton/FlyDSL，支持 GEAK 类 agent、frozen initial-candidate 基线。❌ 作为平台（要 AMD 机器）；✅ 方法论可抄：**任务按泛化维度拆分、冻结基线候选**——我方把官方 bench 拆 train/test 形状集防过拟合 bench（我方红线"禁 bench 调优"的工程化手段）。GEAK v3 曾在其上做 repository 级评测。

---

## 5. Intel / 非 CUDA Triton 循环模板

### 5.1 Triton8（IntelLabs/Triton8）—— 最干净的可移植模板

**① 定位**：Intel GPU 上 LLM 驱动的 PyTorch→Triton 数值等价 kernel 生成优化（arXiv:2605.26118，九阶段优化：算法重构→tile→向量化→折叠→融合→…)。

**② 循环架构**：**Claude Code 驱动 + 可换 trial tree + accumulated KB（yaml 词条：Symptom/Cause/Action/效果）+ 复合成功标准（准确率×加速比×token 预算）**；持续分析 session 文件沉淀 KB、依据类别表现选择性蒸馏回流。

**③ 硬件依赖**：Intel Data Center GPU Max 1550 / Triton 3.4 / PyTorch 2.8 XPU；**评测全部 CPU 侧 wall-clock + CUDA-event 式计时，无任何 vendor profiler 依赖**。

**④ 开源状态**：Apache-2.0，**已归档（不再维护）**——但归档恰说明代码已冻结、可安全 fork 当模板。

**⑤ 本赛可用性**：🔧 **改造后可用（kernel 循环模板首选）**。README 自带 "Porting to a different target" 章节，明示仅 4 处硬件相关：agentconfig.py 判断文本+阈值 / trial_runner.py shell 命令 / kernel wrappers / golden KB。我方改动：换成 flagtree Triton 3.2 + BI-V150/C500 wrapper + 把 golden KB 换成 FlagGems 已验证算子；**其 yaml KB 是最便宜的可持久化跨轮记忆方案**，可直接并入我方台账。

**⑥ 与 AKO4ALL 互补**：AKO4ALL 是流程、Triton8 给流程配上**结构化知识沉淀格式与 trial tree 管理**；且证明 Triton-only + 计时评测的循环在非 CUDA 后端完全跑得通（Intel 先例 = 我方 flagtree/MACA 同类处境）。

### 5.2 Xe-Forge（IntelLabs）

多阶段 LLM agent 管线（分析→逐阶段优化）。作为 Triton8 同门的流水线化变体参考即可，热度与文档均不及 Triton8。

---

## 6. 国产生态（天数/沐曦直接相关）

### 6.1 KernelGen + flagos-ai/skills（官方，已在用）—— 近期更新

- KernelGen：FlagOS 生态 AI 自动 Triton kernel 开发平台，**语言支持 Triton + Triton-TLE，已适配 6 款芯片（英伟达/海光/摩尔线程/昇腾/天数智芯/沐曦）**——即 BI-V150 与 C500 均在官方适配面内。
- **FlagOS 2.0（2026-03/04）**：天数/摩尔线程/华为/寒武纪支持 FlagGems C++ Wrapper，30 个重点算子替换；**FlagOS 2.1（2026-06）面向 AI Agent**：FlagGems 560+ 算子、14 厂商 23 芯片、进入 PyTorch 基金会。
- **KernelGenBench**：210 算子（ATen 110 / **vLLM 50** / cuBLAS 50），多芯片评测，LLM Track Pass@K——与我方 vllm-plugin-FL 落点高度对齐。
- **flagos-ai/skills**：总控 Skill 自动识别仓库类型（FlagGems/vLLM/通用）做算子生成/优化/平台迁移；配套 **flagos-ai/awesome-LLM-driven-kernel-generation**（官方策展列表 + 综述 arXiv:2601.15727《Towards Automated Kernel Generation in the Era of LLMs》，IJCAI-ECAI 2026）。
- 评级：✅ 官方在用。行动项：跟踪 2.1 Agent 能力与 vLLM-FL 仓库（flagos-ai 生态内就有基于 vLLM 插件化多芯片的 vLLM-FL）——**我方 vllm-plugin-FL 的优化可反哺上游**。

### 6.2 Kernel-Smith（上海AI Lab + 沐曦）

**① 定位**：评估驱动进化 agent 的高性能 GPU 算子生成系统（arXiv:2603.28342《A Unified Recipe for Evolutionary Kernel Optimization》），在线平台 chat.intern-ai.org.cn/kernel-smith。

**②③**：四类常用算子在 MXMACA 后端测试出色——**MXMACA 支持是发布卖点而非纸面**；进化 agent + 统一配方（生成→编译→bench→变异选择）。

**④ 开源状态（本次关键负面发现）**：GitHub InternLM/Kernel-Smith 存在，但官方明示 **"We do not currently plan to release the Kernel-Smith model weights or agent code"**——仓库只分享生成的 kernel 与 benchmark 素材。

**⑤ 本赛可用性**：❌（循环不可得）。残余价值：(a) 其生成的 **MXMACA kernel 成品**是 C500 上可对标/借鉴的同类实现；(b) 在线 arena 可零成本试其能力上限作参考。

**⑥ 与 AKO4ALL 互补**：无循环互补；其 MXMACA 产物可进我方"参考实现池"。

---

## 7. NVIDIA 系方法论（不可直接用，但定义了天花板）

### 7.1 KDA（Kernel Design Agents）

NVlabs/kda 与 mit-han-lab/kernel-design-agents 双仓库；研究→实现→验证→迭代的工作流，MLSys-2026 FlashInfer 竞赛冠军（AKO4X 报告称在该竞赛 5 kernel 上反超 KDA）；Kimi Delta Attention 重写 2.96x geomean；~$70K token 成本论。❌（CUDA/NCU/FlashInfer 面）。**可抄**：把"竞赛题"当第一公民的工作流组织法。

### 7.2 KDA-Pilot（BBuf）+ sglang-omni Runtime Profile Agent

- KDA-Pilot：把真实 serving 框架 kernel 变成可复现优化任务的**轻量 prompt 任务库**——**frozen production shapes + copied upstream baselines + symmetric benchmarks**（B200 上 10 个 SGLang kernel 任务）。🔧 **方法论直接移植**：我方在 vllm-plugin-FL/FlagGems 上建立同款"冻结形状+对称基准"任务卡，就是合规化的 micro-bench 体系。LMSYS 博客《Agent-Assisted SGLang Development》(2026-07-02)。
- sglang-omni #1650/#1798：Runtime Profile Agent（原 KDA）路线——**先 profile 每个模型 runtime 再优化**，含独立复现记录。印证 GEAK 的"先 triage 后动手"次序。

### 7.3 cuTile / CUDA Tile IR（GTC26 方向）

CUDA 13.1（2025-12）引入 tile 编程，13.2/13.3 增强（C++ tile、编译器 autotuning）；GTC26 多场 session（S81433/S81705/S81653）含 "Translating CUDA Tile Operations from Python to Rust Using Agentic AI"、"LLM-Generated CUDA Kernels: Are We There Yet?"。**对 Triton 生态的启示**：tile 级硬件抽象 + agent 生成是行业收敛方向——**Triton 正是我方非 CUDA 卡上等价的 tile 抽象层，FlagGems 路线与该趋势同向**；无直接工具可复用。❌（工具不可用）/ 方向确认价值高。

---

## 8. 模型驱动与进化类（≠ agent 循环，供知识来源）

- **AutoTriton（THUDM+华为，ai9stars/AutoTriton）**：8B 模型 SFT+GRPO 专门生成 Triton kernel（arXiv:2507.05687）。❌ 不是循环；开源权重+数据，其训练语料可作我方 SKILL 卡知识源。
- **cuPilot**：策略协同多 agent CUDA 生成（2025-12 论文），仅 cuPilot-Kernels 仓库开源。❌ CUDA-only；策略分解思想（规划/验证/反思角色）已被 AKO4X 覆盖。
- **ForgeKernel（2026-11）**：自进化 kernel，对标并匹配 K-Search 7 个 kernel。**Warpline**：二进制级 tracing+统一证据链，7.69x geomean。**K-Search（Cao et al. 2026）**。均为论文级，监控即可。
- **OAID/AutoKernel**：TVM 调度搜索时代的遗产（非 LLM）。❌。
- **Sakana AI CUDA Engineer**：注意其 2025 年"优化作弊"翻车史——对我方是**反面教材强化**：必须保留 output parity/精度 gate 与 cheat_check（AKO4X benchmark 协议里就有 sanitize/cheat_check 钩子，建议我方 adapter 实现之）。

---

## 9. 架构层 / SpecPilot 补充

### 9.1 RoofLang（MSRA）

arXiv 2609.12551（2026-09）：LLM 推理系统架构 DSL——通用图表示 + **语义保持动作空间**，agent 修改推理架构定位局部瓶颈；DeepSeek V4 系列最优配置下 decode 峰值 3.5–39.5x。**未找到代码发布**（MSR 出版页 + OpenReview）。❌（无代码、多卡分布式为主）。单卡可借鉴：**"语义保持动作空间"设计**——把我方 vllm-plugin-FL 内可动点（调度/图捕获/算子替换）枚举成带前置条件的动作图，agent 只在图内搜索，天然防越红线。

### 9.2 SpecPilot 补充（arXiv:2611.02501，2026-11）

- **NIXUS 不存在**：多轮搜索无任何 NIXUS 系统；其 related work（§7）实为 FunSearch、AlphaEvolve、ADRS、AlphaDev、Ansor、KernelBench、CUDA-L1 + 投机采样专项系统（EAGLE 系、TriForce、MagicDec、ZipNN 等）——我方此前的引用疑为记忆偏差，可从台账删除。
- 新增细节：结构化记忆 140 条（每条≤12 项要求：code delta+registry+evidence，应用回 stock 须完全复现才算 entry）；ceiling 分解为 τ ratio / kernel ratio / overhead ratio 三比值取最大者定向；teacher-forced scorer 双拒绝规则（rate rule + event rule，δ=stock 自身 0.72% 近tie 错配率）。**本赛禁投机采样→其产物不可用**，但"比值分解定位 + 复现才算记忆"两条已被我方吸收，维持方法论参考。

### 9.3 serving 级 agent 检索结论

专门检索"vLLM/SGLang 自动调优 agent"：**无独立成熟开源项目**（GuideLLM 只是基准工具）；该生态位实际由 **Hyperloom（全栈）+ GEAK v4 e2e_workflow（Amdahl triage/A-B 门控）+ KDA-Pilot/sglang-omni（任务化）** 三家构成——即 §4 的 AMD 系就是 2026 年 serving 级 agentic 优化的事实中心。"FlashVector"检索无此学术系统（疑为 LLM 幻觉引文），以 Hyperloom 为准。

---

## 10. 跨项目主题分析

1. **无 NCU 循环已有先例**：Triton8（计时+KB）、GEAK v4（profiler 仅是 triage 输入，决策靠 interleaved A/B）、Hyperloom（roofline 可替换）、KernelEvolve（PyTorch Profiler 入栈）——四者证明 NCU 非必需，**端到端 A/B + 精度 parity 才是硬验证**。
2. **记忆结构收敛**：AKO4X append-only archive+TRAPS.md、GEAK insight blackboard+kb_artifacts、Hyperloom Recipe KB（七字段：model/hw/framework/model-type/arch/version/precision + 松弛回退匹配）、Triton8 yaml 词条、KernelEvolve 分层 KB、SpecPilot 140 条 evidence 记忆——我方台账应对齐 Hyperloom 七字段并加"复现才算 entry"（SpecPilot）。
3. **防作弊/防噪声是标配**：interleaved A/B 0.5% 带（GEAK）、Critic 双签+agent 数字不作数（Hyperloom）、cheat_check/sanitize 钩子（AKO4X）、frozen shapes+对称基准（KDA-Pilot）、任务 train/test 拆分（AgentKernelArena）。我方五件套应全数采纳。
4. **国产芯片处境的对标对象是 MTIA**：KernelEvolve 对 MTIA 的"文档策展+KB 注入"路线 = 我方对天数/沐曦的路线；KernelGen 官方已适配两卡，是唯一 ✅ 的外部依赖。

## 11. 结论：Top 5 可用组合

| # | 项目 | 角色 | 改造量 |
|---|---|---|---|
| 1 | **AKO4X**（MIT） | 主循环升级：campaign 多轮+archive/TRAPS 门控；benchmark adapter 接官方 bench stdout | 中（adapter+技能卡换 flagtree/MACA Triton） |
| 2 | **GEAK v4 e2e_workflow**（Apache-2.0） | serving 层蓝图：Amdahl triage→cheapest-lever-first→interleaved A/B+parity→可回滚 overlay | 中大（剥 rocprof/gfx/aiter，接 vllm-plugin-FL） |
| 3 | **Hyperloom**（MIT） | vLLM 插件层全栈改造蓝本+Critic 双签纪律+Recipe KB 格式 | 中大（同上，红线裁剪 flag sweep） |
| 4 | **Triton8**（Apache-2.0，归档） | kernel 循环模板：trial tree+yaml KB+无 profiler 计时评测，4 处移植点 | 小（官方 porting guide） |
| 5 | **KernelGen+flagos-ai/skills**（官方） | 唯一 ✅ 直接依赖：Triton-TLE 多芯片生成+KernelGenBench 对齐 | 零（持续跟踪 FlagOS 2.1） |

方法论层（无代码但必抄）：KernelEvolve（异构 KB 注入/PyTorch Profiler 兜底/跨会话复用）、KDA-Pilot（冻结形状对称基准）、AgentKernelArena（任务拆分防过拟合）。

不可用结论（止损）：Kernel-Smith（agent 不开源）、KernelAgent（NCU/CUDA）、AutoTriton/cuPilot（模型/CUDA）、RoofLang（无代码）、SpecPilot 本体（禁投机采样）、NIXUS（不存在）。

## 附录：主要来源

- AKO4X/AKO4ALL/AKO: github.com/TongmingLAIC/AKO4X（README/CLAUDE.md）；tongminglaic.github.io/AKO
- KernelEvolve: arxiv.org/abs/2512.23236；engineering.fb.com/2026/04/02/...kernelevolve...
- KernelAgent/KernelFalcon: github.com/meta-pytorch/KernelAgent；pytorch.org/blog/kernelagent...；pytorch.org/blog/kernelfalcon...
- GEAK: github.com/AMD-AGI/GEAK（v4 README）；rocm.blogs.amd.com（GEAK V3, 2026-07-20）
- Hyperloom: github.com/AMD-AGI/Hyperloom；rocm.blogs.amd.com（2026-07、2026-09）
- AgentKernelArena: github.com/AMD-AGI/AgentKernelArena；arxiv.org/abs/2605.16819
- Triton8: github.com/IntelLabs/Triton8；arxiv.org/abs/2605.26118
- KernelGen/FlagOS: github.com/flagos-ai/KernelGen；flagos-ai/awesome-LLM-driven-kernel-generation；综述 arxiv 2601.15727 / IJCAI-ECAI 2026
- Kernel-Smith: github.com/InternLM/Kernel-Smith；arxiv 2603.28342；metax-tech.com/ndetail/12575.html
- KDA: github.com/NVlabs/kda、mit-han-lab/kernel-design-agents；KDA-Pilot: github.com/BBuf/KDA-Pilot；lmsys.org/blog/2026-07-02
- SpecPilot: arxiv.org/abs/2611.02501；github.com/StargazerX0/SpecPilot
- RoofLang: arxiv 2609.12551；microsoft.com/en-us/research/publication/rooflang...
- cuTile: nvidia.com GTC26 S81433/S81705/S81653；developer.nvidia.com/blog
- 索引：youyve/awesome-kernel-agent（280+ agents）、youyve/awesome-kernel-benchmark（157 benchmarks）
