# 工业界「特定模型 × 特定硬件」推理优化实战案例调研

> FlagOS S2 赛道二 · MiniCPM5-2B × 天数 BI-V150 / 沐曦 C500 · vLLM 0.24 系
> 调研日期:2026-11-20。共 9 次有效抓取(GitHub README×4、官方博客×3、SGLang 技术文×1、第三方镜像×1)+ 定向搜索佐证。
> 可信度标注:[一手]=官方仓库/官方博客原文;[摘要]=搜索结果摘要,未取全文;[宣传]=厂商营销口径,未独立验证。

---

## 案例 A:DeepSeek FlashMLA —— 自研 attention kernel 全家桶(价值最高)

**① 来源** [一手] https://github.com/deepseek-ai/FlashMLA (README,2026-01 stable 版)

**② 模型×硬件** DeepSeek V3/R1 系(MLA 注意力)× NVIDIA Hopper/H800/Blackwell + **华为昇腾 NPU**(官方声明 FlashMLA 支撑 DeepSeek 在 NVIDIA GPU 与昇腾 NPU 上的部署)

**③ 瓶颈定位法** README 直接用双口径给结论:memory-bound 场景报有效带宽(GB/s)、compute-bound 场景报 TFLOPS——即「先判断 kernel 属于哪一墙,再对着墙的极限算利用率」。这正是无 NCU 环境可用的定位法:理论字节量 ÷ 实测 kernel 时长 = 有效带宽,与硬件峰值带宽之比即利用率。

**④ 手段清单 + 量化收益**
| 手段 | 细节 | 收益 |
|---|---|---|
| **fused norm-RoPE-attn-RoPE-cast 单 kernel**(decode) | 把 MLA 前后一串小算子(norm、双向 RoPE、类型转换)全部融进 attention 主体,消掉小 kernel launch 与中间张量 HBM 往返;权重排列(permute)预计算 | **decode +10~15%**;B200 上 prefill 1460 TFLOPS / decode 950 TFLOPS |
| **FP8/FP4 KV cache + 尺度紧邻数据布局** | 每 32 元素(FP8)/16 元素(FP4)一个 scale;**scale 直接排在对应数据后面**(data followed by scales),kernel 一次 load 连续取出,无独立 dequant 旁路访存 | KV 体积 2x/4x ↓,等效带宽同比例 ↑ |
| joint prefill+decode kernel | 一个 kernel 同时处理混合 batch | 减少切换与 launch |
| 2025-04 kernel 迭代 | 更优 tile/流水 | compute-bound +5~15%,H800 660 TFLOPS |
| 昇腾 950 sparse attention | 与华为联合,附技术报告 | prefill 410 TFlops(**95% 峰值**)、decode 360 TFlops(**83% 峰值**) |
| **Community 移植矩阵**(对我方最有价值) | 沐曦 MetaX-MACA、摩尔线程 MT-flashMLA/FlashMLEX、海光 OpenDAS/MLAttention、**天数智芯 Deep-Spark**、AMD AITER 均有官方/社区移植 | 证明「把业界最优 kernel 移植到自家软件栈」是国产厂商标准打法 |

**⑤ 映射我方**
- 我们的备选方向「KV-int8 × attention 重写」应直接抄它的**布局设计**:int8 KV + per-32 通道 scale,scale 紧跟数据块存放,Triton kernel 内联反量化——比「int8 存储 + 独立 dequant kernel 再 attention」少一整轮 KV 读。16K 场景我们 91% 流量在 KV 读,KV 减半 ≈ decode 理论上限提速 ~1.8x。
- **小算子融合链**:GQA 下我们同样有 norm/RoPE/cast/分 logits 等碎算子,launch 调优(D0)之后下一步就是融合。FlagGems 是 Triton 库,融合无障碍。
- 天数(Deep-Spark/FlashMLA)与沐曦(MetaX-MACA/FlashMLA)的移植**可以作为我方 attention Triton 实现的参考实现对照**(它们用 mctlass 写,我们用 Triton,但 tiling/分页参数可对照)。

---

## 案例 B:DeepSeek DeepGEMM —— 小 M GEMM 的专用 kernel 设计

**① 来源** [一手] https://github.com/deepseek-ai/DeepGEMM (README)

**② 模型×硬件** DeepSeek V3/R1 训练+推理 × H800/Hopper(社区移植到昇腾/AMD 等)

**③ 瓶颈定位法** 观察 MoE/decode GEMM 的形状特征:**M 维极小(通常 64/128),N、K 大且固定(如 7168)**——通用 BLAS 为大 M 设计,tile/流水全不匹配小 M;另用「同精度输出对比」(<0.5% loss)守住质量线。

**④ 手段清单 + 量化收益**
- **四类 kernel 全部 JIT**:normal dense / grouped contiguous / grouped masked(decode 专用,支持 CUDA graph 捕获)/ dense-normdequant(bf16)。JIT 不存 cubin,运行时按 shape 生成代码——每种 batch/seq 组合拿到专属 kernel。
- **常驻低精度 + 边界高精累加**:计算用 FP8,关键边界升精度,精度损失 <0.5%(随机数据)。
- **SF(scaling factor)布局按 TMA 对齐**重排,消非合并访存。
- **`set_num_sms()` + cuda-graph 模式**:限制 GEMM 占用 SM 数、给通信/其它流留核,削 tail wave。
- **PDL(programmatic dependent launch)overlap Tail wave**:下一个 kernel 提前起跑,吃掉波尾空窗。
- 性能:H800 1550 TFLOPS(FP8);dense-normdequant bf16 2690 TFLOPS;小 kernel 约 20 µs 级。[宣传] 官方 X:1350+ TFLOPS。

**⑤ 映射我方**
- 我们 decode bs=64 时 M=64,正是 DeepGEMM 的目标形状。**Triton 侧等价物**:按「4K/16K 定长 × bs=64」几个固定 shape 离线 autotune 出专属配置(BLOCK_M/N/K、num_warps、num_stages),固化成查表,不走运行时通用 autotune。
- 「限制 kernel 占用 SM 数、给旁路留核」在双流场景(见案例 G 多流并行)可用 Triton grid 大小近似实现。
- 精度策略:W8A8 用 FP32 累加 + 少量边界高精,对应案例 C 的实测教训。

---

## 案例 C:SGLang × DeepSeek V3 —— 工程调优方法论最完整的一篇

**① 来源** [一手镜像] https://verda.com/blog/deepseek-v3-sglang-inference-optimization (原发 blog.sglang.ai《Optimizing DeepSeek V3: Inference Speed Improvements》);配套 [一手] https://github.com/sgl-project/sglang/issues/2591

**② 模型×硬件** DeepSeek-V3(671B MoE, MLA)× H800/H200

**③ 瓶颈定位法(无 NCU,可直接抄)**
- `SGLANG_TORCH_PROFILER_DIR` + `bench_one_batch --profile` 导出 **PyTorch profiler trace**;
- **prefill(512in/1out)与 decode(1in/512out)分开压测**,分别归因调度开销与访存开销;
- 大批量(10,000 prompts)复现生产负载再对比小批量,识别「只在大 batch 生效」的优化。

**④ 手段清单 + 量化收益**
| 手段 | 量化收益 |
|---|---|
| CUDA graph(`--cuda_graph_max_bs`)+ torch.compile 叠加 | 端到端延迟 7.322s→1.256s→**1.011s**;吞吐 39→229→**285 tok/s(≈7x)**;代价:prefill 略降(0.212→0.261s,graph 不覆盖 prefill) |
| Overlap scheduler(CPU 调度提前一批算元数据) | 每输出 token 时间 348.10→196.79(**1.77x**);TTFT 变差 |
| FlashInfer MLA kernel 替换 | 吞吐 +6%(1810→1920 tok/s),GSM8K 0.951→0.948 |
| DP attention(MLA head=1,TP 切不动 → 按请求切) | 仅大 batch 有效,小 batch 反效果 |
| **FP8 blockwise**(权重 128×128 块 / 激活 1×128 向量各自 scale)+ FP32 累加(H800 TensorCore 累加仅 ~14 bit) | 同精度(0.955 vs 0.957)吞吐更高 |
| ⚠️ 反面教训:bf16→FP8 的 bmm **不融合反更慢**(cast 计算开销吃掉收益) | — |
| **Triton 调优脚本方法论**:`tuning_block_wise_fp8.py` 遍历 BLOCK_M/N/K、GROUP_SIZE_M、num_warps、num_stages;FusedMoE 同法(per-N 配置表,调优 5267s) | SGLang vs vLLM FusedMoE 延迟 ~**2x** |
| NextN(EAGLE)投机解码 ~6x | 我方禁用,仅记录 |

**⑤ 映射我方**
- D0(attention launch 调优)之上的最大候选:**确认 CUDA graph 在 vllm0.24 天数/沐曦后端对 bs=64 decode 全步覆盖**(含 lm-head/sampling);不覆盖的算子是 Triton 可改写的就补。
- W8A8(D1)必须做「cast/dequant 融合进 GEMM epilogue」验证,否则会复刻他们「FP8 反而更慢」的坑;FP32 累加保精度。
- 把 SGLang 的 **shape 定点 Triton 调优脚本**流程搬过来:只对 (bs=64, 4K) 与 (bs=64, 16K) 两个点扫配置,产出 FlagGems 配置表。
- 瓶颈定位流程(torch profiler + 两阶段分离压测)作为我方标准动作,替代 NCU。

---

## 案例 D:沐曦 MetaX-MACA/FlashMLA —— 沐曦 C500 一手工程事实

**① 来源** [一手] https://github.com/MetaX-MACA/FlashMLA (README)

**② 模型×硬件** DeepSeek MLA × **沐曦 C500/C600, MXMACA 软件栈**

**③ 瓶颈定位法** 移植基准:对齐原版 FlashMLA 的 kernel 级性能(README 明示以 FA2 v2.6.3 为基线移植)。

**④ 手段清单(工程事实)**
- 工具链:**MXMACA 2.27.426+**,PyTorch 2.1 + **Triton 2.1(maca toolkit wheel)**;矩阵算子用 **Mctlass**(MACA 版 CUTLASS,从 sw-download.metax-tech.com 下载);CUDA 兼容走 `CUDA_PATH=$MACA_PATH/tools/cu-bridge`。
- 支持 fp16/bf16、**MTP≥1**(多头预测)、paged KV block = 2^n(要求对齐)。
- 生态侧佐证:沐曦官方 vLLM 镜像 `vllm-metax:0.15.0-maca.ai3.5.3.203-torch2.8`、SGLang `0.5.8-maca.ai3.5.3.104-torch2.8` 已流通([摘要] GPUStack 实践文)。
- 规模:27 stars / 41 commits(2025-11),活跃但早期。

**⑤ 映射我方**
- 直接证据:**MACA 上的 Triton 路线可行且是沐曦自家选择**(其 Triton 2.1 来自 maca toolkit)。我们对齐其 PyTorch/Triton 版本组合可少踩兼容坑;cu-bridge 说明大量 CUDA 语义代码可零改跑。
- 它的 paged KV「block 必须 2^n」约束提醒我方 KV-int8 重写时**保持 block 对齐**,否则在两家卡上都会掉到非合并访存路径。
- 沐曦用 mctlass 而非纯 Triton 写 MLA——说明最内层 attention 想吃满 C500,可能需要超出现成 Triton 模板的深度调优;我们受约束只能 Triton,收益预期应按「拿到大部分」设定,并以 kernel 带宽利用率(案例 A 双口径法)验收。

---

## 案例 E:昇腾 vllm-ascend + MindIE —— 非 CUDA 全栈官方优化矩阵

**① 来源** [一手] https://github.com/vllm-project/vllm-ascend (README);[摘要] 昇腾社区博客(见下)

**② 模型×硬件** DeepSeek 全系/Qwen3-MoE 等 × Atlas 800I A2 / 300I Duo / 910B / 950

**③ 瓶颈定位法** 版本化性能回归(README 公开 v0.9.x 起逐版本性能曲线与已知精度/性能 gap 表)。

**④ 手段清单 + 量化收益**
- **架构:把厂商支持做成 vLLM plugin**(v0.7.3+ 平台化,vLLM 上游可直接罗列其模型)——软硬解耦、跟住 vLLM 主干。昇腾侧计算经 **CANN**。
- **torchair graph 模式**:NPU 图执行,替换 Attention/RoPE/AllGather 等为昇腾优化算子;支持 **ACL graph 捕获**。
- 量化:W8A8 / W8A16(ANT)+ 社区 FP8;官方发布 DeepSeek-R1-0528-**W8A8** checkpoint。
- [摘要] 官方博客案例:DeepSeek-V4-w8a8 在 A2 上 **PD 分离部署吞吐 2.8x**;DeepSeek-V4 Flash 在 910B 上 **融合内核+多流并行,吞吐 6600 tok/s,TTFT P99 4.2s→0.9s**。
- [一手,FlashMLA README] 昇腾 950 sparse attention:prefill 410 TFlops(95% 峰值)/ decode 360 TFlops(83% 峰值),附算法与优化技术报告。

**⑤ 映射我方**
- 「**融合内核 + 多流并行**」是国产栈官方最常拿结果的两个抓手,与我方 Triton 融合方向一致;多流(attention 与 GEMM/杂算子异流水)值得在天数/沐曦双卡或单卡内试。
- vllm-ascend 的「plugin 化 + 公开性能回归表」是团队工程管理可抄的范式:每轮优化出一条可复现曲线。
- PD 分离对我们 4K/16K 定长、输出恒 1024 的负载未必适用(prefill/decode 比例固定且并发高),优先级放低。

---

## 案例 F:摩尔线程 MTT —— 教训型案例(降级检测 → 自建算子库)

**① 来源** [摘要] mthreads.com 官方新闻/腾讯云报道;[一手] FlashMLA README community 节(MT-flashMLA、FlashMLEX)

**② 模型×硬件** DeepSeek-V4 Flash / Qwen × MTT S4000/S5000,MUSA 栈

**③ 瓶颈定位法** **逐 kernel 审计指令支持**:适配 Qwen 时实测原生 PyTorch FA2 在 S4000 上有 **63% 的 kernel launch 因硬件不支持 `__shfl_sync` 被静默降级为通用矩阵乘**——先找到「看不见的性能黑洞」再动手。

**④ 手段清单 + 量化收益**
- 2024-11 开源 **MUTLASS**(MUSA 版 CUTLASS)重写 FlashAttention 类算子。
- [宣传] DeepSeek-V4 Flash 在 S5000 多轮优化:单 GPU 吞吐较月初基线 +18.8x、高并发场景 ~80x(口径夸张,佐证「未适配 → 适配」的头部空间)。
- [一手] FlashMLEX kernel 有效带宽 ~**0.5 PB/s**(S5000 集群口径),并移植进 FlashMLA 生态。

**⑤ 映射我方**
- 立刻可做的审计动作:**核对 FlagGems/我方 Triton kernel 在天数/沐曦上没有被降级**(如某 intrinsic 不支持时编译器回退慢路径)。这类问题用「kernel 实测时长 vs 理论字节/峰值带宽」的比值一抓一个准。
- 厂商自建模板库(MUTLASS/Mctlass)说明:Triton 之外总有 20~30% 尾部收益在 ISA 级——受约束我们不做,但把它记为「已知收益上限缺口」写进报告预期。

---

## 案例 G:天数智芯 Deep-Spark 生态 —— 我方同门卡的最优参照

**① 来源** [一手] FlashMLA README community 节(Deep-Spark/FlashMLA 移植);[摘要] DeepSpark 26.03/26.09 release、developer.iluvatar.com(IxFormer/ixRT/IGIE)

**② 模型×硬件** DeepSeek MLA 等 × 天数智芯通用 GPU(天垓/智铠系列,BI 系同栈)

**③ 手段清单**
- **Deep-Spark/FlashMLA**:天数官方组织维护的 FlashMLA 移植——即「MLA kernel 已在天数栈被打通」。
- **IxFormer(vLLM)**:天数官方 vLLM 适配(兼容第三方推理框架);**ixRT** 推理引擎(动态形状、多精度、图优化);**IGIE** 推理框架;DeepSparkInference:多算子库 + **算子自动调优** + 量化 + 图优化。
- DeepSpark 26.03 一次新增 18 个 LLM 推理示例(10 个 vLLM + 6 个 xDiT),26.09 再加 2 个 vLLM 示例——vLLM 是其主推路径。

**⑤ 映射我方**
- **BI-V150 上最直接的对标物**:天数自家的 FlashMLA 移植仓库可作为我方 GQA attention Triton 重写的正确性/性能对照组(同卡同栈);IxFormer 的版本兼容表可反推我方 vllm0.24 栈该对齐的依赖组合。
- 生态重心在 vLLM,证明我方选型方向与厂商投入一致,遇到 kernel 问题可向 Deep-Spark 社区/仓库找现成解。

---

## 案例 H:Nebius —— Triton attention kernel 工程细节(Kvax)+ vLLM 服务指南

**① 来源** [一手] https://nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax (Kvax, 2025-02);[一手] https://nebius.com/blog/posts/serving-llms-with-vllm-practical-guide (2025-12);[摘要] nebius.com/customer-stories/sglang(SGLang 案例页:注意力算法优化 + FP8 → 单节点吞吐 2x)

说明:任务书中提到的「99% GPU 利用率 / attention kernel 对比 / 量化收益」系列在本次检索中未能定位到独立文章(Nebius 站内相关内容散见 MLPerf v6.0 结果帖与上述两篇),以下为已验证内容。

**② 模型×硬件** Llama 3.1 8B 级 × Hopper;Kvax 为 Triton 实现的 FA2 变体(JAX 生态)

**③ 瓶颈定位法** Kvax 的出发点:框架层(XLA 静态形状)导致 padding 浪费算力 + 完整 attention mask 的数据搬运开销;对比实验按「单长序列 / 多序列 packed」分桶测。

**④ 手段清单 + 量化收益**
- **Triton 写 FA 的实用边界**:选 FA2 而非 FA3,因 FA3 的关键特性(Hopper 专属异步流水)**Triton 语言表达不了**——对「只能 Triton」的我们是直接预警:C500/BI-V150 上 Triton attention 的上限由语言决定,别按 CUDA FA3 的数字定 KPI。
- **块级 mask 四分类**(Full/Left-partial/Right-partial/Empty),Empty 块整块跳过;mask 压缩成每行 4 个 int32 边界,显存 ≈ 3×bs×seq/block×16B——**变长/交叠 batch 下省下可观带宽**。
- Dense packing + document mask 消 padding;CP 负载均衡按 Llama3 式 2×CP 配对分块。
- 对比:多序列场景显著快于 CuDNN(后者受 mask 数据搬运拖累)、JAX FA2 与 FlexAttention。
- vLLM 指南要点:attention backend 选择矩阵(Triton attention 适合 ALiBi 且需快于 SDPA 的场景);**优先厂商预量化 checkpoint**(在线量化拖启动+占 CPU/GPU);TP 取节点内卡数。

**⑤ 映射我方** 4K/16K 定长输入无 padding 问题,但「Empty/部分块跳过」思想可用于 watermark 驱动的**KV 部分回收**与 batch 内不等长残余;Triton 上限预警应写进我方收益预期管理。

---

## 专题综合:小模型高并发 decode 与 16K 长上下文

**(1) <3B × bs>32 的 decode(我方主战场)**
- 理论框架:batch≥64 时权重读取被摊薄,GEMM 进入 M=64 小 M 区(DeepGEMM 立项依据);此时**kernel launch 与碎算子开销占比急剧上升**——SGLang 的 7x 中大头来自 CUDA graph+compile,FlashMLA 的 +10~15% 来自小算子融合,两者都以 launch/碎片为敌。
- 行动:graph 覆盖率审计(bs=64 decode 全步)→ 融合 norm/RoPE/cast 链 → 定点 shape 的 Triton 配置表 → SM 配额/多流。

**(2) 16K 长上下文(KV 带宽 91%)**
- KV 压缩是唯一杠杆:FlashMLA 的 **int8/FP8 KV + scale 紧邻数据布局**是已验证模板;昇腾/沐曦/天数全都移植了 MLA kernel 而非改模型结构,说明「kernel 层解决」是行业共识。
- paged KV block 对齐 2^n(沐曦移植硬约束);融合 RoPE 进 attention 减少 KV 多次往返(FlashMLA fused 链收益佐证)。
- 容量侧:天数 32GB 卡 29 并发上限 → W8A8 权重(4GB→2GB)直接换 KV 空间,与 KV-int8 叠加是「容量+带宽」双收方向(需按 2KV-head×16K×layers 精确核算增量并发数)。

---

## 汇总:对我方路线的映射(约束:禁投机 / 仅 Triton / 无 NCU)

| 我方路线 | 借鉴来源 | 具体动作 |
|---|---|---|
| D0 attention launch 调优(已做 +42/48%) | SGLang、FlashMLA | 下一步做「小算子融合链」(norm-RoPE-attn-cast),预期再 +10~15%(decode) |
| D1 W8A8(进行中) | SGLang 反面教训、DeepGEMM | cast/dequant 必须融进 GEMM;FP32 累加;离线 shape 定点调优脚本出配置表 |
| D2 KV-int8×attention 重写(备选) | **FlashMLA** | 抄布局:per-32 scale 紧跟数据、Triton 内联反量化、block 2^n 对齐;16K 收益上限 ≈ KV 流量占比×压缩率 |
| D3/新增:graph 与调度 | SGLang、昇腾 | bs=64 decode 全步 graph 覆盖审计;overlap/双微批流水(小 batch 失效风险,先试点) |
| 新增:无 NCU 定位法 | SGLang、FlashMLA | torch profiler trace + prefill/decode 分离压测 + 有效带宽反推(字节/时长 vs 峰值),建立每轮优化回归表(vllm-ascend 范式) |
| 新增:降级审计 | 摩尔线程 63% 教训 | 全 kernel 过一遍「实测 vs 理论带宽比」,揪静默降级 |
| 参照对照 | 沐曦 MetaX-MACA/FlashMLA、天数 Deep-Spark | 同栈正确性对照 + 依赖版本对齐(Triton 2.1/macа toolkit、cu-bridge) |

**明确不做**(约束/性价比):投机采样(SGLang 6x 不可用)、mctlass/MUTLASS 级 ISA 重写(违反 Triton-only;记为已知收益缺口 ~20-30%)、PD 分离(负载不匹配)。

---

## 来源清单
1. FlashMLA README — github.com/deepseek-ai/FlashMLA [一手]
2. DeepGEMM README — github.com/deepseek-ai/DeepGEMM [一手]
3. SGLang《Optimizing DeepSeek V3》镜像 — verda.com/blog/deepseek-v3-sglang-inference-optimization [一手镜像];github.com/sgl-project/sglang/issues/2591
4. MetaX-MACA/FlashMLA README — github.com/MetaX-MACA/FlashMLA [一手]
5. vllm-ascend README — github.com/vllm-project/vllm-ascend [一手]
6. Nebius Kvax — nebius.com/blog/posts/kvax-open-source-flash-attention-for-jax [一手]
7. Nebius vLLM 指南 — nebius.com/blog/posts/serving-llms-with-vllm-practical-guide [一手]
8. 昇腾社区博客(搜索摘要):hiascend.com/developer/blog/details/02202220239553783193(PD 分离 2.8x)、…/02201220510120818196(融合内核+多流 6600 tok/s)
9. 摩尔线程官方新闻/腾讯云报道(搜索摘要);Deep-Spark releases、GPUStack 沐曦 C500 实践(cnblogs,部署向)
