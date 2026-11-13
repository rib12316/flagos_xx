# 2026-11-13 D2 算子深挖阶段台账（prefill 优先线）

## 0. 本日终章：融合 GEMM+epilogue 落地（commit 6dfe8fd + 961cac7）

**融合 kernel（`_w8a8_fused_gemm_kernel`，int_mm.py）端到端收益矩阵**：

| 口径 | 数字 |
|---|---|
| bench eager（prefill 部署真值）M=2048 每层 3 融合线性 | **1.78×**（6.22→3.49ms：gate_up 2.50× / o 2.16× / qkv 1.89×；down 走链不回退） |
| bench graph M=2048 | gate_up 3.01× / o 1.21× / qkv 1.18× / down 走链 |
| serve 烟测 case1 [2048,512,16,32] 冷启 | **1415.16 → 2939.72 tok/s（2.08×）**；混合态 pass 971-1280 → 2485-2576 |
| serve 烟测 case2 [512,1024,16,32]（decode 重） | 1365.52 vs 原 1372.36 持平（M<256 恒走链，decode 零回退） |
| 精度 30 题对拍 | **acc_real 0.9333 = fake 基线**（原链 0.90），same_ans 0.90，gate 过 |
| serve fallback | 0 次 |

路由定案：仅 (2048,12288)/(2048,2560)/(2048,2048) 且 M≥256 且 M%BLOCK_M==0 走融合；
其余（M<256 decode graph 档 / down 全档 / 非整除尾 chunk）走 cuBLAS 链。
迭代教训两则：① `do_not_specialize=["M"]` + 恒真掩码 → 谓词化 load/store 实测劣化
3-6×，EVEN_M constexpr 显式双分支解决；② M=64/128 融合 grid 饥饿（10-20 program/
16 SM），M<256 回落链后 decode graph 档零扰动。

**ixkncli 归因（六步循环第 2-3 步）**：Multi-Lane Matrix Instructions 6.29M——int8
tl.dot 确实下探矩阵指令路径；grid 768×512thr，109 vec regs，49KB smem/block，**理论
占用 12.5% 瓶颈 = 共享内存**（2 CTA/CU 上限）——与 cuBLAS（110-125 T-MAC/s）的差距
主因，后续调参方向 = 减小 BK/BN 降 smem 提 CTA 数（当前 68-74 T-MAC/s 已够赢 eager）。
ixobjdump 步省略（指令统计已证实矩阵路径）。

遗留优化空间（明日按 ROI，深扫脚本 scripts/d2/phase0c_sweep_prod.py 已就绪待 GPU）：

## 0.4 开源算子调研结论（用户问「能不能直接拿开源的用」，2026-11-13 夜）

| 候选 | 判定 | 依据 |
|---|---|---|
| DeepSparkInference（天数自家，330+ 模型） | ❌ 无 kernel 可用 | 模型部署脚本库（ixrtexec/igie 驱动闭源引擎）；`models/benchmark/quantize_w8a8.py` 仅 llmcompressor SmoothQuant 转换脚本（与我们同配方家族）——int8 快路径全在 ixRT/IGIE 闭源 .so 内 |
| iluvatar-corex-ixrt 引擎源码 | ❌ 取不到 | gitee deep-spark/iluvatar-corex-ixrt clone 无权限/不存在（ixrt skill 文档链接已失效） |
| IST-DASLab/gemm-int8 | ❌ 编译不了 | CUTLASS + **SM80+**（Ada/Hopper 优化）+ nvcc/CUDA 11.8 工具链；Corex 栈是 clang→PTX→ptxas + ivcore11 方言，CUTLASS 不可编（tile 策略可参考但 ixrt skill 已有同源参考） |
| vLLM cutlass/cutednn int8 | ❌ 已知死路 | sm75+/sm90 门（handover §P4 前科） |
| FlagGems 上游（> in-tree 5.3.5） | ⚠️ 不跟进 | 2026-09 release 加的是 mm/gemv 等 bf16 算子（[releases](https://github.com/FlagOpen/FlagGems/releases)）；FlagOS weekly 的 BI-V150 算子队列重测是 ixdnn/ixblas vendor slot 工程非 scaled_mm int8 提速；且红线 pip 只从本仓库 + 冻结线内升级 in-tree FlagGems 的 dispatch 兼容风险 > 收益 |
| cublasLt（Corex SDK `/usr/local/corex/include/cublasLt.h` = ixblasLt CUDA 兼容层） | ❌ 数学卡死 | int8 epilogue 表达力只有 per-tensor alpha + per-col bias；W8A8 需要 per-row(xs)×per-col(ws) 双向 scale——表达不了（NVIDIA 侧同理，vLLM 因此自写 cutlass_scaled_mm） |
| torch._int_mm（ixblas） | ✅ 已在用 | vendor 最强 int8 GEMM（125 T-MAC/s @down）就是现路径的链内核 |

**总结论**：本硬件栈（Corex clang/Triton flagtree 生态）与主流开源 int8 kernel 生态
（CUTLASS/sm80+/nvcc）方言隔离，天数自家的快 kernel 全在闭源引擎（ixRT/ixblas/ixdnn）
且 ixblas 已通过 `_int_mm` 在用——**「直接拿来用」无可编译现成品，自研 Triton 融合
是本栈唯一开放路径**；开源价值集中在 tile 策略参考（IST-DASLab / ixrt skill INT8 参考）。

## 0.5 融合 kernel 深挖计划（Step 3b，用户问责「未到极致」后立档）

已知差距与主攻方向（按预期 ROI）：
1. **4-CTA smem 族配置**（ixkncli：占用 12.5% 瓶颈 = 49KB smem → 2 CTA/CU）：
   (128,128,64,s2)=32KB 等 ≤32KB 族 probe1 恰好漏扫——理论上 4 CTA/CU 翻倍占用，
   latency hiding 翻倍，gate_up 62%→? cuBLAS 的最大候选。
2. **GROUP_M ∈ {4,8,16}**（从未扫过，L2 swizzle 敏感参数）。
3. **ixobjdump SASS**：确认 int8 dot 下探的 mma 形态/操作数宽度（指令统计已证实
   矩阵路径 6.29M，但形态未知；cache cubin 用 --fun '.*fused_gemm.*' --sass）。
4. **down 深扫**：≥80% cuBLAS 则重新入融合表（现 59% 走链）。
5. **quant prologue 融进 GEMM（2→1 kernel）**：kernel 内两遍法——pass1 对本 program
   的 BM 行全 K bf16 absmax（x ≤8MB 全驻 L2 16MB，重读放大 ~96×由 L2 吃），pass2
   K 循环边量化边 dot。位精确数学逐字照抄 `_dynamic_per_token_quant_int8_bare_kernel`
   （quantization.py L75-143：`fdiv(1.0,absmax,ieee_rounding=True)*127.0` /
   `absmax*(1.0/127.0)` / round-half-even / clamp）。收益：省 quant kernel
   ~28µs GPU + 一次 launch（eager ~40-80µs）×3 融合线性/层 ≈ 每层 6-8%。
   风险：寄存器压力（bf16 tile + absmax 累加）；位精确单测锁死（oracle 同款）。
6. cache_modifier=".cg" B 流变体：若 sweep 后仍 L2 bound 再加 kernel 参数变体。

补充候选（ROI 靠后）：qkv/o 试 BM256（gate_up 靠它 18%→62%；qkv/o 权重 4-5MB 可全驻
L2，预期增益小）；官方双场景 bench ≥3 中位 + 105 题精度（Step 5a 过夜跑中）。



## 1. 快速端到端烟测对比（两臂收齐，本阶段方向定盘数据）

`/tmp/d2_quick_e2e.sh` 口径：同参 serve（红线参数）+ 官方 bench 原件换小参 `[[2048,512,16,32],[512,1024,16,32]]`。
原始日志：`results/20261112_d2_e2e_quick/`（w8a8 arm 23:49-00:08，bf16 arm 00:08-00:27，两臂 serve 均 0 次 fallback）。

| 场景 | w8a8 (tok/s) | bf16 (tok/s) | w8a8/bf16 |
|---|---|---|---|
| case1 [2048,512,16,32] prefill 重 | **1415.16** | 1853.03 | **0.764× 回退** |
| case2 [512,1024,16,32] decode 重 | **1372.36** | 882.96 | **1.554× 显著加速** |

TTFT：case1 两臂同为 ~19-20s（prefill 主导段）；case2 w8a8 343ms vs bf16 9925ms（bf16 decode 慢导致排队）。

**结论与决策（2026-11-13 归因后修订，详见 `01_attribution.md`）**：
- **0.764× 被推翻**：quick bench 汇总口径（4 pass 跳首取 mean）受 prefix cache 双峰性支配
  （同 seed 数据 pass 间前缀复用，两臂命中率同构）。3 轮中位：**w8a8 2492 vs bf16 1853
  （领先 1.35×）**；w8a8 最佳 pass 3799 > bf16 3524；decode ITL 17.9 vs 19.9ms；GPU
  每 forward 147.6 vs 178.5ms——w8a8 全面≥bf16。
- 真实存在的税（下一步攻击对象）：**int8 链 epilogue=24.2% GPU kernel 时间 + quant 3.2%
  ≈ 抵消 int8 GEMM 节省**（serve 内 attention 冲刷 L2，epilogue 有效带宽 ~420GB/s）；
  每线性 3 kernel × Corex 慢 launch 在 eager 冷 prefill 段多付 launch 税（冷 pass ITL
  mean 卡到 138ms）。→ **融合单 kernel（quant+GEMM+epilogue）为第一优先**，动机从
  「救回退」改为「官方 4K/16K 冷缓存场景的确定性大幅收益 + 全 regime 通用」。
- 单轮 quick bench 汇总不可作结论（本轮教训入档）：一切对比 ≥3 轮中位 + 记录缓存态。

## 2. skill 调研结论（开放任务收口）

### 2.1 新装：tensormux/kernel-skills（8 件软链就位）

`git@github.com:tensormux/kernel-skills.git` SSH clone 至 `/root/kernel-skills`（npx/https 禁用红线遵守），以下软链进 `~/.claude/skills/`（Claude Code 已识别）：

| skill | 用途对位 |
|---|---|
| `write-int8-quantized-kernel` | dp4a/INT32 累加/per-channel epilogue（P0/Step 3 内核形态） |
| `write-triton-gemm-kernel` | tile/block tiling/tl.dot 累加（大 M 融合 GEMM） |
| `optimize-triton-block-parameters` | BLOCK/num_warps/num_stages 选型（P2 调参） |
| `optimize-prefill-vs-decode-kernels` | prefill compute-bound vs decode memory-bound 分治（本阶段主问题） |
| `write-vllm-custom-op-integration-plan` | vLLM custom op/cudagraph 捕获/TP 集成（AOT 坑对位） |
| `fuse-elementwise-ops` | epilogue 融合模式（P1） |
| `write-kernel-test-plan` | 位精确测试计划设计 |
| `debug-quantized-kernel-accuracy` | 量化精度回归归因（备用） |

评估记录：anthropics/skills 官方库无 GPU kernel 类目（出局）；NVIDIA/skills、huggingface/kernels cuda-kernels（NVIDIA-only，不对位天数栈，出局）；intel/gpu-ai-skills（CUDA→XPU 迁移向，不对位）；Kernel-Smith（前会话已否决）；Kernelbench Eval 封装（NVIDIA 基准评测向，非开发 skill）。**结论：tensormux/kernel-skills 是当前最对位可装的算子开发 skill 库**，硬件标注 nvidia/amd 但 Triton/patterns/inference/quantization 类目内容硬件中立，已按「适配 sm71/corex 再用」原则使用。

### 2.2 Iluvatar 官方三 MCP（已确认存在，执行待批）

《让 AI Agent 有纪律地做 GPU 性能优化——三 MCP 联动指南》（developer.iluvatar.com/blogs，2026-10-22，李超，id=`trimcp-tutorial`）确认：
- `@ixsys/mcp-app` 0.2.1（trace→热点榜/SOL%/roofline）、`@ixkn/mcp-app` 0.3.1（kernel profile→SOL%/warp stall/指令统计）、`@ixloop/mcp-app` 0.3.3（实验纪律：基线冻结/逐轮 PASS 判定/scorecard/plateau 检测）。
- tarball：`http://files.deepspark.org.cn:880/deepspark/MCP/*.tgz`。三 MCP 只认落盘产物（.ptrace/.ixkn-rep），数据级耦合。
- **状态**：三 tarball 已下载解包至 `/root/ix-mcp/`（46M/82M/192K）；执行被权限分类器拦截（plain-HTTP 无签名源）——**待用户明确批准后**再配 `.mcp.json`（node 直起绕开 npx）。本阶段归因先走 CLI 正路（ixsys/ixkn-cli/ixobjdump = CoreX Toolkit 预装，即 4 件 iluvatar skill 文档化的官方用法），无阻塞。

### 2.3 已装 4 件 iluvatar skill 读毕

ixkncli-guide（kernel profile：`--section`/`-k`/`--launch-skip/-c`，先 ixsys 找热点再 ixkn 深挖）、ixsys-guide（系统级 trace：`-o x.ptrace -t cuda,osrt`，debugfs 需 root）、ixobjdump-guide（SASS/IR：`--sass --fun '.*name.*'`）、ixrt（INT8 量化参考 + 「ixrt 上 int8 反而比 fp16 慢」逐层 profiling 案例——与本次 prefill 回退同类问题形态）。注意本机 Toolkit **V4.4.0** vs skill 基线 v5.0.0，option 以 `-h` 实测为准。

## 3. 硬件事实修正（Plan agent 本机实测，后续所有算术以此为准）

| 项 | 旧假设 | 实测 |
|---|---|---|
| SM 数 | ~120 | **16**（multi_processor_count） |
| smem/SM | 96KB | **128KB（max 128KB/block）** |
| L2 | — | **16MB** |
| Triton int8 tl.dot | 未知 | iluvatar backend `min_dot_size` 8-bit = **(1,1,32)**，M=1 可直接 dot |
| cuBLAS _int_mm 天花板 | — | 3400 int8-MAC/SM/clk（M=1024 实测 163 TOPS） |

**推论**：M=16 档手工 mul+sum 逐元素内积必 ALU-bound（4-16× 缺口）——内积必须 tl.dot/dp4a 形态。**R1 风险**：vendor backend 可能将 int8 dot 偷换 fp16-mma（FlagGems scaled_mm 从未位精确验证过）→ 一切 dot kernel 先过 CPU oracle `torch.equal` 闸门。

## 3.5 Phase 0 探针判决（scripts/d2/phase0_int8_dot_probe.py + phase0b_probe2.py，2026-11-13）

| 探针 | 判决 |
|---|---|
| tl.dot(int8,int8,int32) 位精确 | **全配置 OK**（vs torch._int_mm 逐位相等）——R1（vendor 偷换 fp16-mma）解除 |
| tiled M=2048 吞吐 | down **75.1 T-MAC/s（60% cuBLAS）**、qkv 66.1（67%）、gate_up **BM256/BN128 = 68.4（62%）**（BM128 档仅 18-20%：大 BM 降低 25MB 权重块的 L2 重读压力）；cuBLAS 109-125 T-MAC/s |
| skinny（M=16）split-K+atomic | down 39.5µs vs 现路径 50.1µs（1.27×）但 **fast 配置位精确全 FAIL**（仅 BN32 慢配置 OK）——vendor atomic 加速路径疑似数值坑，未深挖；gate_up 79µs 仍输 55µs → **P0 skinny 推迟**（decode 链在 graph 内 launch 免费，现状已赢 bf16） |
| 读流带宽天花板 | 829 GB/s（BLOCK16384；4096 档仅 241——per-program 串行长归约延迟主导） |

**融合定案**：实现「quant（复用现有验证 kernel）+ 融合 GEMM+epilogue 单 kernel」两步链（3→2 kernel），
prefill M>16 生效，decode M≤16 维持 cuBLAS 垫零链。预期：epilogue 24% GPU 税消除 + 每 linear
省 1 launch；eager prefill 每 linear wall 收益估 −15~−37%（launch 税为主）。tile 起点：
gate_up (256,128,64,w8,s2) / down,qkv,o (128,256,64,w8,s2)，按形状查表，未知形状回落 cuBLAS 链。

## 4. P0 decode skinny GEMM 设计存档（推迟；若重启须换两段归约替代 atomic——split-K atomic 位精确存疑）

要点：kernel 落 `int_mm.py`（L206 垫零分支前插 M≤16 分支），裸 triton.jit，grid=(cdiv(N,BN),)，BLOCK_M∈{1,16} 两实例，int8 tl.dot int32 acc + epilogue 融尾（`(acc·xs)·ws` 两条独立乘）；参数初值 M=1:(BM1,BN128,BK512,w4,s2) / M=16:(BM16,BN128,BK256,w4,s2-3)；env `VLLM_FL_INTMM_SKINNY` 默认 1 + capture 守卫（`is_current_stream_capturing()` 未 warm 回退旧路径）+ try/except 静默回退；变体 A（M=1 复用 BM16 实例）/B（mul+sum 仅诊断）/C（dp4a inline_asm，需 [N,K] 重排 +2GB 显存，须用户批准）。判据 graph 口径 M=1/16 ≥1.3×；单测 oracle/模板见 test_w8a8_int_mm.py L69-82/L256-278/L297-320。

## 0.6 Step 3b 深挖战报（skill 方法论驱动，2026-11-13 深夜）

方法论：optimize-triton-block-parameters skill（smem 模型/配置空间/≥6-8 配置/边界形状/
>70% 效率判据）。生产 kernel 全配置深扫（phase0c_sweep_prod.py，驱动真实 kernel 含
epilogue 尾）：**(256,256,64,GROUP_M=4,w8,s2) 全面碾压**——

| shape | 旧配置 | 新最优 | vs cuBLAS(_int_mm) | graph vs 链 M=2048 | eager vs 链 |
|---|---|---|---|---|---|
| gate_up | (256,128)799.7µs | **486.8µs** | **96%** | **4.78×** | 3.79× |
| o | (128,256)90.9µs | **81.0µs** | **96%** | 1.42× | 2.63× |
| qkv | (128,256)126.1µs | **118.0µs** | **93%** | 1.35× | 1.84× |
| down | 曾 59% 走链 | **221.3µs(G_M4)** | **86%（重入融合表）** | 1.10× | 1.18× |

- **每层 4 线性全部融合，eager 合计 4.16→1.61ms = 2.59×**（深挖前 1.78×）；M=1024 档
  亦全赢（1.10-2.06×）
- ixkncli 复核：生产 kernel 488µs 实测落地（=sweep 值）；grid 384、smem 64KB、矩阵
  指令路径维持
- 指令级验证闭环：Triton cache `.asm` 实为 **iluvatar 原生 ISA**（非 NVIDIA SASS，
  ixobjdump --sass 不适用方言）：1273 条 ml + 441 条 sl 矩阵装载族指令主导 K 循环，
  has_wgmma=0——vendor 矩阵路径与 ixkncli 6.29M 矩阵指令互证
- 精度 30 题对拍：acc 0.9333=fake、same_ans 0.90→**0.9333**、gate 过
- case1 烟测单轮 2288（缓存双峰噪声带内，pass4 未全热；kernel 层 2.59× 为硬数据）
- 未竟：quant prologue 单 kernel 化（每层 4×quant≈174µs 占融合链 10.8%，下一候选）
