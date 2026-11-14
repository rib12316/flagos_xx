# D2 W8A8 量化线交付文档 + 极限榨取新 agent prompt（2026-11-14）

> 交付人：D2 算子深挖会话 agent。分支 `feat/d2-w8a8-quant`（32 commits 已 push 至 `1dd43f5`）。
> 用户裁定：①W8A8 端到端未达应有提升，**继续榨干加速潜力，精度让路**；②**官方完整
> 性能 bench 与精度测试只在用户明确要求时才允许开测**；③run3 与 105 题已按令中止
> （run1/2 数据有效）。

---

## 一、结果总表（截至交付）

### 官方双场景（对照基线 = D0+D1 合入后的 main 基线 2668.42 / 1276.62）

| 形态 | 4K [4096,1024,64,256] | 16K [16384,1024,64,128] |
|---|---|---|
| 纯净起点（无 D0/D1/D2） | 1879.68 | 862.06 |
| + D0 attention launch 调优 | 2658.72（+41.5%） | 1262.66（+46.5%） |
| + D1 watermark=0.06 调度 | 2668.42 | 1276.62 |
| **+ D2 W8A8（本线，run1/run2）** | **2838.46 / 2835.73（+6.4%）** | **1345.90 / 1346.04（+5.4%）** |
| **全项目 vs 起点** | **+51.0%** | **+56.1%** |

- 105 题精度：**未跑**（按令中止）。30 题对拍 gate 过（acc 0.90-0.9333 波动带，fake 基线 0.9333）。
- 附带收益：权重 **−1.84GB**，KV 池 582K vs bf16 518K tokens（+12%）。

### D2 各层成果（同一线性链提速在不同场景的显形）

| 层 | 结果 |
|---|---|
| 精度配方 | SmoothQuant α=0.5，30 题 Δacc=0；`tools/quantize_int8_rtn.py` 幂等可再生 |
| 算子一代 | FlagGems Triton int8（0.10-0.90× 全面慢）→ `torch._int_mm` 链过硬件闸门（down M=1024 1.604×） |
| 算子二代（融合） | `_w8a8_fused_gemm_kernel` (256,256,64,G4,w16,s2)：**qkv/o 103% 反超 cuBLAS、down 96%、gate_up 95%**；每层 4 线性 eager **4.16→1.58ms = 2.6×**；167 单测位精确 |
| quick 小场景 vs bf16 双臂 | case1 中位 +35%、case2 +55%（线性占比高时收益显形） |
| 官方场景 | +5-6%（见下节稀释分析） |

## 二、为什么端到端「只有」+5-6%：量化收益的天花板结构

官方场景（4K/16K 输入、并发 64/128）的 GPU 时间构成（双臂 torch profiler 实测）：

| 成分 | 占比 | 是否已量化 |
|---|---|---|
| **attention（bf16）** | **65-80%+**（16K 长上下文更高） | ❌ —— **这是量化路线的真正阻塞点** |
| 线性层（int8，本线战场） | 15-25% | ✅ 已打到 95-103% cuBLAS |
| lm_head（bf16 540MB） | decode 侧 ~5% | ❌（P3 原被精度红线否决，现已解锁） |
| norm/rope/采样/调度 | ~10% | ❌ |

线性层 2.6× 作用于 ~20% 占比 → 端到端 ≈ +7%（与实测吻合）。**结论：线性层已无大幅
空间（95-103% cuBLAS），量化的下一步杠杆只能来自把「未量化的 80%」搬进低精度——
KV cache 量化（attention 带宽减半 + KV 池翻倍）与 lm_head int8 是两个明确候选。**

## 三、终态架构（新 agent 起点，全部代码级默认生效）

| 组件 | 位置 | 状态 |
|---|---|---|
| 融合 GEMM+epilogue kernel | `vllm-plugin-FL/vllm_fl/quantization/w8a8/int_mm.py`（`_w8a8_fused_gemm_kernel` + `_FUSED_GEMM_CONFIGS`） | M≥256 且 M%256==0 且 (K,N)∈表 → 融合；`VLLM_FL_INTMM_FUSED_GEMM=0` 关回链 |
| 裸 quant kernel | `vllm_fl/dispatch/backends/flaggems/impl/quantization.py`（每行/program 单遍，已证最优形态） | 165 prio 最高 |
| cuBLAS 链（M<256/尾 chunk/未知形状） | 同 int_mm.py（pad17 + `_int_mm` + 融合 epilogue） | decode 在 cudagraph 内 launch 免费 |
| dispatch/custom op/AOT | `vllm_fl::w8a8_int_mm_scaled_gemm` custom op + `CachedOp.pre_resolve()` | vLLM AOT 三坑已解 |
| 单测 | `tests/unit_tests/quantization/` 167 全绿（位精确 oracle=CPU int32 matmul 行采样 + fullgraph 模板） | 改码必重跑 |
| checkpoint | `/workspace/MiniCPM5-2B-w8a8`（SmoothQuant α=0.5，lm_head/norm 未量化） | 转换脚本可再生成 |

## 四、已证死的路（勿重试，数据在 `results/20261113_d2_prefill_attribution/00_ledger.md` §0.7-0.8）

1. **1-kernel 融合**（quant 前移进 GEMM）：0.10-0.15×，K 循环内量化转换链摧毁 dot 流水。
2. **quant kernel blocked 多行/两遍法**：0.89-0.94× 全负。
3. **decode skinny（M≤16 自研 GEMV）**：三振（无 split 输 2× / atomic 位精确坑 / 两段归约 M=16 全输 0.74-1.08×——cuBLAS pad-17 已 427GB/s）。
4. **B 流 cache_modifier（.cg/evict_first）**：本 backend 无效。
5. **开源算子直接拿来用**：CUTLASS/sm80+ 方言隔离编译不了；cublasLt epilogue 表达不了双向 scale；天数闭源引擎不可取（§0.4 全表）。
6. **num_stages=3**（96KB smem → 1 CTA/CU）：0.54-0.63× 大负。
7. quick bench 单轮汇总作结论（prefix cache 双峰噪声）——一切对比 ≥3 轮中位。

## 五、剩余潜力清单（新 agent 弹药，按预期 ROI 排序）

| # | 方向 | 预期 | 依据/入口 |
|---|---|---|---|
| P0 | **decode M=64/128 融合 sweep**：官方 decode 并发 64/128 → 每步 M=64/128 全走 cuBLAS 链，**该 M 档从未用小 BM tile 扫过**（M<256 门是按 (256,256) tile 的败绩设的） | decode 线性 +10-30% | `phase0c_sweep_prod.py` 改 M∈{64,128} + BM∈{32,64,128}；过判据就放开 `_FUSED_GEMM_MIN_M` |
| P1 | **lm_head int8**（原 P3 解锁）：540MB→270MB，decode 每步 −0.9ms（ITL −5%）；M=decode batch 走 P0 的路由 | decode +5% | `tools/quantize_int8_rtn.py` 的 targets/ignore 配置；重转 ckpt |
| P2 | **KV cache int8/fp8**（用户点名的阻塞点候选）：attention 读带宽减半 + **KV 池 582K→~1.1M tokens**（同驻翻倍 → D1 鸽笼空间翻倍）。先探 vLLM `kv_cache_dtype` 在本栈可用性，不可用则自实现 KV int8 写/读 kernel（reshape_and_cache + unified_attention 的量化钩子） | attention 侧最大杠杆，端到端或 +10-20% | 上游 `kv_cache_dtype=fp8_e4m3/int8` 路径 + `attn_launch_config.py` 同款注入模式 |
| P3 | **W4A8 权重**（decode 权重读减半）：int4×int8 dot 在本 backend 支持性未知，先探针 | decode 线性再 2× | Triton int4 支持 / 手动 unpack |
| P4 | **int8 attention 自研 kernel**：最大 lift（80% 占比）但工程量最大，最后动 | 端到端或 +20-30% | 上游 unified_attention Triton 源码起步 |
| P5 | **D0 重调耦合**：P0-P2 落地后批分布/KV 池变化 → `sweep_attn_launch.py --full` 换表（~20-40min） | 数个点 | D0 线遗留检查点① |

## 六、环境与工具箱

- **skills**：`~/.claude/skills/` 12 件（iluvatar×6 + tensormux kernel-skills×8 软链，`/root/kernel-skills`）。方法论主轴：`optimize-triton-block-parameters`（本轮深挖 86%→103% 即其产物）。
- **Iluvatar 三 MCP**：tarball 已下载 `/root/ix-mcp/`（@ixsys/@ixkn/@ixloop），执行被权限分类器拦截——**用户批准后**配 `.mcp.json`（node 直起）。CLI 正路（ixkn-cli/ixobjdump/ixsys V4.4.0）可用。
- **探针脚本**（`scripts/d2/`，全部可复跑）：phase0c（生产 kernel 配置扫）、phase0e（1-kernel 判决）、phase0f（skinny 两段）、phase0g（边缘+cache mod）、profile_prefill_chain（链路分解）、parse_torch_trace（serve profile 解析，配合 `--profiler-config` 诊断 serve + `/start_profile`）。
- **bench**：`bench_int8_gemm.py`（graph/eager 双口径四臂）、`/tmp/d2_quick_e2e.sh`（快速烟测口径）。
- **硬件事实**：16 SM、smem 128KB、L2 16MB、Triton iluvatar backend（8-bit dot 下限 (1,1,32)，`.asm` 为 iluvatar 原生 ISA 非 SASS）、Corex launch ~40-105µs（graph 内免费）。

## 七、官方数据现状与测试纪律

- run1/run2 官方双场景已完成并超基线（见 §一）；run3 中止于 16K 中段（用户令）；**105 题从未跑**。
- **测试纪律（用户令，2026-11-14）**：官方完整性能 bench 与精度测试**只在用户明确提出要求时执行**，其余一切验证走烟测循环（quick 双场景 + 30 题对拍 sanity）。

---

## 八、新 agent 完整 prompt（复制即用）

```
角色：FlagOS S2 赛道二开发 agent，接手 D2 量化线「极限榨取」阶段。环境：天数 BI-V150
（torch 2.10 corex / vllm 0.24.0 / triton 3.6 flagtree / flag_gems 5.3.5 / 单卡 32GB /
sm(7,1)，实测 16 SM、smem 128KB/SM、L2 16MB、Corex kernel launch ~40-105µs、CUDA graph
内 launch 免费）。仓库 /flagos_比赛/flagos_xx，分支 feat/d2-w8a8-quant（32 commits 已
push 至 1dd43f5）。

核心目标（唯一）：把 W8A8 量化推理的加速潜力完全榨干——用户已明确「不考虑精度下降」
（精度让路，但保持配方可追溯与可回退：每步量化加深都留 env/开关回退与转换脚本可再
生）。现状：线性层已 95-103% cuBLAS（融合 kernel 上线），官方场景端到端仅 +5-6%——
结构性原因 = attention(bf16) 占 GPU 时间 65-80% 未量化、lm_head(bf16 540MB) 未量化。
若优化到极致仍遇瓶颈，最后交付一份「量化路线阻塞点分析」（回答：下一步是否必须对
KV cache 量化 / int8 attention / 更低位宽，各自的预期收益与工程代价）。

第一步（必做，按序）：①通读 docs/d2-w8a8-delivery.md（本交付文档：结果总表/终态架
构/七条已证死路/剩余潜力清单 P0-P5/工具箱）；②通读 results/20261113_d2_prefill_
attribution/00_ledger.md（全部探针数据与四死案判决）；③跑通环境自检（~/.claude/
skills/ 12 件已装；python3 scripts/d2/phase0c_sweep_prod.py 的 GPU 门禁会防误占）。

开发优先级（按 ROI，每项独立 env 开关 + 六步循环：写 kernel→ixkncli 判 bound→
指令验证→graph 微基准过判据→ixsys 占比→烟测）：
P0 decode M=64/128 融合 sweep——官方 decode 并发 64/128 → 每步 M=64/128 走 cuBLAS
   链，该档从未扫过小 BM tile（M<256 门按 (256,256) 败绩设的，偏保守）：改
   phase0c 探针 M∈{64,128} × BM∈{32,64,128} × BN∈{128,256}，graph 判据 ≥1.15× 即
   放开 _FUSED_GEMM_MIN_M 并回填该档配置（decode 在 cudagraph 内，纯 GPU 收益直接
   兑现 ITL）。
P1 lm_head int8（已解锁）：tools/quantize_int8_rtn.py 的 targets 加 lm_head 重转
   ckpt（M 档路由复用 P0）；decode 每步 −0.9ms。30 题对拍仅作 sanity 记录（不 gate）。
P2 KV cache int8/fp8（阻塞点头号候选）：先探 vLLM kv_cache_dtype(fp8_e4m3/int8)
   在本栈（corex）可用性与数值形态；不可用则自实现——reshape_and_cache 写入侧量化 +
   unified_attention 读取侧反量化（参考 attn_launch_config.py 的注入模式），收益双杀：
   attention 读带宽减半 + KV 池 582K→~1.1M tokens（D1 同驻鸽笼空间翻倍）。
P3 W4A8：int4×int8 dot 本 backend 支持性探针先行，可行则 decode 权重读再减半。
P4 int8 attention 自研 kernel：最大 lift（80% 占比）最大工程量，P0-P3 收完后动。
P5 全部落地后 D0 重调：scripts/sweep_attn_launch.py --full（批分布/KV 池已变）。

验证策略（用户令，严格执行）：烟测循环为常态——性能只跑快速端到端烟测（官方 bench
原件换小参 [[2048,512,16,32],[512,1024,16,32]]，/tmp/d2_quick_e2e.sh 口径，单轮汇总
受 prefix cache 双峰噪声支配、结论必须 ≥3 轮中位）+ graph 口径微基准；精度只跑冻结
30 题对拍作 sanity（不 gate，用户已放弃精度约束）。官方完整双场景 bench 与 105 题精
度【只在用户明确提出要求时才允许开测，否则一律禁止】。

红线：serve 参数一字不动（--port 9031 --compilation-config
'{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm
--gpu-memory-utilization 0.85 --max-model-len 131072 + VLLM_PLUGINS=fl；--model 换目
录/诊断性 --profiler-config 属部署/观测步骤允许）；官方 bench 脚本原件不改
（--test-cases 可换）；pip 只从本仓库（pip install --no-deps --no-build-isolation
./vllm-plugin-FL）；GPU 独占串行；git 只走 SSH；11/17 硬止；自写 kernel 一律裸
triton.jit 严禁 @libentry（流竞争+0.9ms/调用病理前科）；形状依赖分支必须包
torch.library.custom_op + CachedOp.pre_resolve（vLLM AOT 三坑，单测模板在
test_w8a8_int_mm.py）；改码必 pip install 后再 serve/bench（site-packages 隔离）。

陷阱速查：pkill -f 自杀（按精确 pid；EngineCore 孤儿按显存循环清）；Bash 工具 cwd
会重置、命令一律绝对路径（单测从 vllm-plugin-FL 目录内跑否则 no tests ran）；Bash
单次 ≤10min 长等待分次轮询；【新增】会话/agent 进程退出可能连带收割 setsid nohup
子进程（11-14 晨官方 run3 即如此暴死）——长跑任务要 setsid nohup + 中途状态落盘
（chain.log 逐段 append）+ 完成监听放后台 watcher；eager 微基准三伪影（host 开销/
轮间降频/56ms 尖峰）——性能判据必须 graph 口径；CUDA 无 int32 matmul 与 fp64 GEMM
（oracle 落 CPU）；torch._int_mm 约束 M>16（垫零位精确已证）+K%8；Triton mul+add 会被
FMA 收缩（bias 路径 1ulp）；Triton cache 的 .asm 是 iluvatar 原生 ISA（非 SASS，
ixobjdump --sass 不适用，指令形态用 ixkncli 指令统计看）；num_stages=3 在 (256,256)
tile 下 96KB smem→1CTA 大负；量化数学复刻纪律：absmax=max（可结合位精确）、inv_scale
= fdiv(1.0,absmax,ieee_rounding=True)*127.0、scale=absmax*(1.0/127.0)、round-half-
even、clamp[-128,127]、epilogue 乘法次序 (acc·xs)·ws 两条独立乘防 FMA 重结合。

已证死路（勿重试，详单在交付文档 §四）：1-kernel 融合（quant 前移进 GEMM，0.10-0.15×）
/ quant kernel blocked 多行（全负）/ decode skinny M≤16（三振，cuBLAS pad-17 已
427GB/s）/ B 流 cache_modifier（本 backend 无效）/ 开源算子直接移植（CUTLASS 方言隔
离，cublasLt epilogue 表达力不够）/ num_stages=3 大 tile。

最终交付（11/17 前）：①P0-P5 按序落地，每项 graph 判据+烟测过闸才合入；②若仍有瓶颈，
交付「量化路线阻塞点分析」：量化还能搬多少未量化占比进低精度、KV/int8-attention/
W4 各自收益-代价测算、以及本栈（16 SM/闭源引擎/CUTLASS 隔离）的结构性上限；③台账
回写（results/d2-kernel-matrix.md、handover）+ push（SSH）；④官方 bench 与 105 题
等用户指令。
```
