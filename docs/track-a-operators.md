# Track A：算子融合 + 图捕获 假设台账（FlagOS S2 赛道二 / MiniCPM5-2B）

版本：2026-11-03 · 基于 vllm-plugin-FL `flagos-2026-s2`（commit 13eb9be）与 FlagGems `v5.3.5`（tag v5.3.5）本地代码盘点 + Web 调研。
定位：纯研究产出，直接指导开发排序；所有路径均为本地真实路径（`VL = vllm-plugin-FL`，`FG = FlagGems`）。

---

## 0. 先说结论：预算重定向（本台账最重要的判断）

对 2B 稠密模型按场景做 decode 一步的访存预算（bf16，bs=64 并发）：

| 流量项 | 计算 | 4K 场景 | 16K 场景 |
|---|---|---|---|
| 层权重（42 层 GEMM 权重） | 42×47.2M 参数×2B ≈ 3.96GB | 25% | 8% |
| lm_head（130560×2048×2B） | 0.54GB | 3% | 1% |
| KV cache 读取 | 42 层×2×ctx×2头×128×2B×64 req | 11.3GB（71%）| 45GB（91%）|
| elementwise（norm/act/rope） | 64×2048 级别 | <1%（纯延迟） | <1% |

（每 token 每层 KV = 2头×128×2(K+V)×2B = 1KB；每请求 4K ctx ≈ 176MB，16K ≈ 704MB。）

**推论**：
1. decode 的主导项是 **KV cache 读取带宽利用率**（attention kernel）与 **小 M GEMM 的权重读取带宽**，不是逐元素算子融合。16K 场景 attention 占 ~91% 访存，是唯一大杠杆；4K 场景 attention ~71% + GEMM ~29%。
2. 传统"融合叙事"（fused_add_rmsnorm / silu_and_mul）在 bs=64 下每个 kernel 只有 ~4µs 量级，融合收益是"每层省几个 launch"，在 FULL_DECODE_ONLY 图捕获下进一步稀释 → 单项 ≤1-2% 步时。它们的价值在于**防止 reference 慢实现被走到**（每个 reference rms_norm 是 6 个 kernel 的 PyTorch 链），以及 prefill 大 M 下的真实流量减半。
3. prefill（compute-bound）由 GEMM 与 varlen attention 的计算效率决定；逐元素融合在 prefill 是真流量项（bs≈万级 token），但占比仍小（~1-2%）。

因此台账排序：**decode attention > 小 M GEMM > 路由正确性/慢实现清除 > 图覆盖 > 逐元素融合 > KV fp8（高风险）**。

---

## 1. 代码盘点

### 1a. MiniCPM5（LlamaForCausalLM）前向热点现状

机制回顾：`VL/vllm_fl/worker/worker.py` 在模型构建前调用 `register_oot_ops()`（`VL/vllm_fl/ops/custom_ops.py`），把 `RMSNormFL / SiluAndMulFL / GeluAndMulFL / RotaryEmbeddingFL` 通过 vLLM `CustomOp.register_oot` 注入，替换上游同名层；层内 `forward_oot` → `CachedOp("rms_norm")` 等 → dispatch 按**平台 yaml 的 op_backends 顺序**解析（`VL/vllm_fl/dispatch/manager.py::_resolve_impl`：按 token 顺序取第一个可用实现，与 BackendPriority 无关）。attention 后端由 `PlatformFL.get_attn_backend_cls` → `call_op("attention_backend")` 在初始化期选择。

| 热点 | 沐曦 metax（C500） | 天数 iluvatar（BI-V150） |
|---|---|---|
| RMSNorm(+残差) | `flagos`（FlagGems Triton `fused_add_rms_norm`，`FG/src/flag_gems/fused/fused_add_rms_norm.py`，经 `gems_rms_forward`） | `flagos` 同左；**vendor:iluvatar 兜底是纯 PyTorch 链**（`VL/.../vendor/iluvatar/impl/normalization.py` 明确是 reference 代码） |
| RoPE（GQA 2KV） | `flagos`（`FG/fused/rotary_embedding.py` 的 `apply_rotary_pos_emb_kernel`，NUM_Q/NUM_K_HEADS 均为 constexpr，结构上支持 GQA，无 2-KV 特化） | 同左；vendor 兜底 `impl/rotary.py` 是 PyTorch rotate_half 链（多 kernel、非原地） |
| SiLU-mul | `flagos`（`FG/fused/silu_and_mul.py`，注意 `VL/.../flaggems/impl/activation.py` 先做两次切片视图再调用） | 同左；vendor 兜底是 `F.silu(x1)*x2` |
| QKV/o/gate_up/down GEMM | 不走 dispatch——FlagGems 以 ATen 后端身份全局接管 `mm/addmm/linear`（`FG/runtime/backend/_metax/ops/mm.py`、`matmul_bf16.py`、`addmm.py` + `mm_metax_expand.yaml`/`tune_configs.yaml` 调优配置）。平台 `flagos_blacklist` 只有 masked_fill → Triton mm 生效 | 同机制（`_iluvatar/ops/mm.py`、`linear.py` 等） |
| decode attention | `attention_backend: [vendor:metax]` → `MacaFlashAttentionBackend`（`VL/.../vendor/metax/impl/attention/flash_attn.py`），decode 用 maca 版 flash-attn2 `flash_attn_with_kvcache`，FA 工具在 `impl/attention/utils/fa_utils.py`（`from flash_attn import ...`，mcoplib 的 `_C` ops + `reshape_and_cache_flash`） | `attention_backend: [flagos, vendor:iluvatar, reference]` → flagos 分支返回 vLLM `TRITON_ATTN`（`VL/.../flaggems/flaggems.py::attention_backend`，默认 `VLLM_FL_USE_FLAGGEMS_ATTN=0`）；即 vLLM Triton attention，且 `VLLM_TRITON_ATTN_USE_TD=0`（triton 3.2 无 `make_tensor_descriptor`，见 `vendor/iluvatar/iluvatar.py` 三处 patch） |
| prefill attention | 同 backend 的 `flash_attn_varlen_func`；含"prefill decode split"魔改：混合 batch 里 prefill/decode 拆两次 kernel 调用（flash_attn.py L797-847） | vLLM TRITON_ATTN 的 extend 路径 |
| 采样 | vLLM 原生；iluvatar 上 `patch_sampler_compile_for_iluvatar` **主动解除了 sampler 的 torch.compile**（topk_topp/logprobs，跑 eager） | 同左 |

**已存在但未被路由用上的实现**（"融合好了没插线"清单）：
1. `vendor.metax` 的原生 `torch.ops._C.fused_add_rms_norm` / `_C.rms_norm`（`VL/.../vendor/metax/impl/layernorm.py`，mcoplib 原生 kernel）——已注册进 registry，但 `VL/vllm_fl/dispatch/config/metax.yaml` 的 `op_backends.rms_norm = [flagos, reference]` **不含 vendor**，永远不会被选中。silu_and_mul、rotary_embedding 同样被 yaml 排除 vendor。
2. `VL/.../vendor/metax/impl/attention/ops/triton_decode_attention.py`（722 行，SGLang/LightLLM 血统的 split-KV flash-decoding，已带 `is_maca_=True` 修改、支持 page size≥1）与 `triton_unified_attention.py`（1048 行）——**全仓库无任何 import**（仅 `impl/mla.py` 一行注释提及），纯死代码。
3. FlagGems 的 `AttentionFLBackend/AttentionFLImpl`（`VL/.../flaggems/impl/attention.py`，包装 `flag_gems.flash_attn_varlen_func` = `FG/src/flag_gems/ops/attention.py` 的 Triton FA2）——只在 `VLLM_FL_USE_FLAGGEMS_ATTN=1` 时启用，默认死路。
4. `FG/src/flag_gems/fused/` 里的现成融合算子大量未接入 vllm-plugin-FL：`swiglu.py`（GEMM 形 swiglu，metax 黑名单标"不可用"）、`matmul_bias_activation.py`（KernelGen 生成的 GEMM+bias+ReLU epilogue，iluvatar 目录有 override 版）、`add_rms_norm.py`、`top_k_per_row_decode.py / top_k_per_row_prefill.py`（采样）、`apply_repetition_penalties.py`、`reshape_and_cache(_flash).py`、`mrope.py`、`fused_deepseek_v4_qnorm_rope_kv_rope_insert.py`（**q-norm+rope+KV写入三融合的 MLA 版**，是 MHA 版可直接抄的模板）。
5. `VL/vllm_fl/ops/_C_ops_registry.py`：当平台无 vllm 原生 `_C` 扩展时，预加载 `mcoplib._C` 并用 FRAGMENT 注册 `_C` 系 op schema（含 `apply_repetition_penalties_` 纯 torch fallback）——说明 metax 侧 mcoplib 原生 kernel 池可通过 `torch.ops._C.*` 直接取用。

**reference 慢实现被走到的风险点**：
- iluvatar：`op_backends` 三 op 的中间兜底 `vendor:iluvatar` 全是 PyTorch reference 代码（normalization/activation/rotary 三个 impl 文件头部自认 "template/stub"）。若 FlagGems Triton kernel 在 flagtree triton 上编译失败，dispatch 静默落到慢路径（strict=false），日志只有一行 WARNING。**必须在真实机器上用 `VLLM_FL_DISPATCH_DEBUG=1` 验证实际 impl_id**。
- `RotaryEmbeddingFL.forward_oot`（`VL/vllm_fl/ops/rotary_embedding.py`）每次 forward 重复 `cos_sin_cache.to(device)` 与 `chunk` ——图捕获后是录进图里的冗余节点，微小但免费可清。

### 1b. FlagGems v5.3.5 可直接/差一步接入的融合算子

注册机制：`FG/src/flag_gems/__init__.py` 的 `_FULL_CONFIG` 把 op 注册到 ATen（`enable()/only_enable()`），平台差异由 `FG/src/flag_gems/runtime/backend/_metax|_iluvatar/`（`ops/` per-backend 覆盖、`tune_configs.yaml` 调优参数、`op_black_list.yaml` 兼容性黑名单、`only_enable_configs.yaml`）处理；`flagtune/` 子包是自动调优工具。`use_c_extension` 开关决定走 `torch.ops.flag_gems.*` C++ 包装还是 Python/Triton 直调。

与本模型直接相关：
- `fused/fused_add_rms_norm.py`（单块 + loop 两版 kernel，fp16/bf16 累加 fp32）——已接入
- `fused/silu_and_mul.py`、`fused/silu_and_mul_with_clamp.py`——前者已接入
- `fused/rotary_embedding.py`（neox/interleaved、GQA constexpr）——已接入
- `ops/attention.py::flash_attn_varlen_func`（Triton FA2 varlen + paged KV）+ `flash_attn_varlen_opt_func`——差一步接入（AttentionFLBackend 已有壳）
- `fused/reshape_and_cache_flash.py`——已由 metax 路径用 mcoplib 版替代
- `fused/top_k_per_row_decode.py / top_k_per_row_prefill.py`、`fused/apply_repetition_penalties.py`——未接入（采样路径）
- `fused/swiglu.py`（metax 黑名单：全 dtype 失败）、`fused/matmul_bias_activation.py`（iluvatar 有 backend override）——GEMM epilogue 方向的现成素材
- metax 黑名单中值得注意：`flash_attention_forward`、`flash_attn_varlen_func`、`cutlass_scaled_mm`、`geglu/reglu/swiglu`、`topk_softmax` 在 metax 全 dtype 失败（2026-01-27 实测）→ metax 上 FlagGems Triton attention 不可用，attention 必须走 maca flash-attn 或修 kernel。iluvatar 黑名单仅 `grouped_topk`、`topk_softmax` → Triton attention 在 iluvatar 是可行候选。

### 1c. metax / iluvatar vendor 后端覆盖

- metax（`VL/.../vendor/metax/`）：rms_norm（mcoplib 原生 fused_add_rms_norm）、silu/gelu_and_mul（转 vLLM CUDA kernel `forward_cuda`）、rotary（`vllm._custom_ops.rotary_embedding`）、attention（MacaFlashAttentionBackend：FA2 varlen/kvcache、cascade、prefill-decode-split、merge_attn_states）、fused_moe 系、topk_softmax；另有 `patches/`（cuda_wrapper、fix_standalone_compile——torch2.8+metax standalone 编译热修、topk_topp_sampler、pynccl_wrapper 等）。
- iluvatar（`VL/.../vendor/iluvatar/`）：rms_norm/silu/rotary 三个 impl 均为 PyTorch 占位；attention 返回 vLLM TRITON_ATTN；模块级做了 4 个关键 patch：triton<3.3 的 `make_tensor_descriptor` stub、vllm triton attention 源码里链式 `or` 重写（3.2.x 编译兼容）、matmul perf model 时钟固定 1500MHz、**inductor GPUTarget 'cuda'→'corex/iluvatar' 重映射**（torch.compile 前提）；外加 sampler 解 compile。→ iluvatar 的"vendor 加速"目前是**合规性修补而非性能实现**，性能全部押在 FlagGems Triton + vLLM Triton attention 上。

### 1d. vllm_fl/compilation 图捕获机制

- `VL/vllm_fl/compilation/break_graph.py`：薄 re-export vLLM 0.24 的 `breakable_cudagraph`——单次前向流捕获在 attention/KV 边界切成 graph 段与 eager 段，**无需 torch.compile FX 切分**；`unified_attention_with_output` 已带 `@eager_break_during_capture`，OOT attention impl 自动参与。`attention_backend` 是初始化期选择器，不进捕获。
- `VL/vllm_fl/compilation/graph.py`：FL 自绘 `GraphWrapper`（平台无关 capture/replay，按 `BatchDescriptor` 缓存 graph，weak_ref 输出，输入地址 debug 校验），`PlatformFL.get_static_graph_wrapper_cls()` 指向它；`Graph.graph` 按 device_type 选 CUDAGraph/NPUGraph/MUSAGraph/PTPUGraph（txda=None）。`support_static_graph_mode()` 对 nvidia/ascend/metax/hygon/mthreads/**iluvatar** 返回 True。
- FULL_DECODE_ONLY 语义（vLLM 0.24）：纯 decode（每请求 1 token）batch 整步一条 graph，attention 在图内；混合/首 prefill batch 回退 eager。天数基线 serve 已锁 `FULL_DECODE_ONLY`。
- 会断图/回退 eager 的点：非 decode-only batch（chunked prefill 混步）、超出捕获 batch 集合的形状（pad 或 eager）、attention metadata 里的 host 侧动态构造（metax FA 的 prefill 分支里有 `torch.tensor([0]+...tolist()).cumsum()` 的 H2D+host sync，好在只在含 prefill 的 batch 触发，不进 decode 图）、iluvatar sampler 解 compile 后 eager 化的采样段。
- 扩大覆盖的挂点：①`VL/vllm_fl/platform.py::PlatformFL` 的 `maca specific config updates`（L253 一带）已有代码级改写 vllm_config 的先例（禁 cascade/cudnn prefill 等）——可在此对 metax 补 cudagraph 模式/捕获尺寸类默认；②metax FA backend 的 decode 路径（flash_attn.py L824-845）确认全部张量来自 metadata 缓冲区即可全图捕获；③`GraphWrapper.concrete_graph_entries` 的 batch descriptor 集合（由 vllm cudagraph_capture_sizes 决定）。

---

## 2. 融合技术调研（业界证据 → 本项目适配结论）

1. **fused_add_rmsnorm**：FlashInfer/vLLM/SGLang 标配，收益机理是残差+归一化读一次 HBM、省一个中间张量写（[vLLM fusions 设计文档](https://docs.vllm.ai/en/latest/design/fusions)、[fused AllReduce+RMSNorm 论文](https://arxiv.org/html/2505.11329v5)）。对本项目：decode 端已被 FlagGems 覆盖且是延迟项（≤1% 步时）；**真正的风险是 iluvatar 兜底链**；prefill 端（万级 token）融合省 ~50% 该 op 流量，但该 op 只占 prefill 步 ~1%。结论：当"正确性保险"处理，不当性能主杠杆。
2. **silu_and_mul / SwiGLU**：带宽数 <1 的纯带宽算子（[FlashInfer kernel 研究 arXiv 2607.14541]），融合意义=单次访存。同上：已融合、已路由，收益封顶。GEMM epilogue 融合（activation 进 GEMM 尾部，vLLM #43224/#38733 RFC、FlagGems `matmul_bias_activation`）在本模型 decode 端省 ~4.7MB 流量/层 ≈ 3µs+1 launch——**不值得做**（除非顺带解决 Triton GEMM M=64 的 split-K 问题）。
3. **RoPE 融合**：业界趋势是 rope+kv-cache-write 乃至 rope+quant 三融合（FlashInfer roadmap、FlagGems 已有 MLA 版 `fused_..._qnorm_rope_kv_rope_insert.py`）。本项目 bs=64 下 K/V 写入仅 64KB/步——融合收益是 launch 数不是流量，估 <0.5% 步时；作为低优先"顺手项"。
4. **decode 小 batch GEMV/GEMM 特化**：M=64 时全部 4 类 GEMM 是权重读取带宽受限，业界做法是 split-K / persistent kernel / per-shape 调优（FlagGems `flagtune` + per-backend `tune_configs.yaml` 正是为此）。GEMM+lm_head 占 4K decode 流量 ~29%，若 Triton mm 当前只跑到峰值的 50-60%，调到 75-80% 可拿 **-4~-7% 步时**（4K 场景为主，16K 也沾边）。这是"确定性中等收益"项。
5. **flash-decoding / split-KV**：小 batch×长上下文时 attention 并行度 = bs×Q头 = 64×16=1024（尚可），但每 KV 分块扫描的带宽利用率取决于 2-KV-head 的访存合并；split-KV 把并行轴扩到 KV 序列维（[Flash-Decoding 原文：长上下文最高 8×](https://pytorch.org/blog/flash-decoding)、[FlashInfer GQA decode](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)）。16K 场景 attention 占 91% 流量，kernel 带宽利用率每 +10% ⇒ 步时约 -9%。metax 有现成未接线的 `triton_decode_attention.py`（SGLang 血统），maca FA2 的 `flash_attn_with_kvcache` 在 2-KV/16K 的表现未验证——**这是全台账最大单点**。
6. **非 CUDA（Triton 系）适配**：FlagGems 的价值就是 per-backend tune_configs + 黑名单兜底；iluvatar 侧 flagtree triton 3.2 的 TD 缺失已由 stub+USE_TD=0 规避（付出 load 路径性能）；`matmul_perf_model` 时钟 patch 说明调优启发式在非 NV 平台也需要手喂参数。结论：所有 Triton 假设在两平台都必须"先 micro-bench 后接线"。
7. **KV cache fp8**：16K 场景理论上限最大（KV 流量减半 → 步时 -40% 级），但精度红线余量仅 1.2 点（math_500 0.962→0.95），fp8 KV 典型掉 0.3-1.0 点，且 serve 参数锁定意味着要在代码里强改 kv dtype——**列为高风险观察项，不做首选**。

---

## 3. Agent 工作流落地（单人队 + 远程租 GPU + 本地无 GPU）

总原则：**生成侧多样化（KernelGen / Kernel-Smith），执行侧单一 harness（AKO4ALL 纪律），度量侧固定口径（stdout micro-bench + evalscope 门禁）**。

### 3.1 AKO4ALL（github.com/TongmingLAIC/AKO4ALL）——执行主循环
- 装什么：单文件 SKILL.md 的 Claude Code/Codex skill。`git clone` 到 agent 技能目录（ZCode 侧可放 `~/.agents/skills/ako4all/SKILL.md` 或项目 `.zcode` 技能位；Claude Code 为 `~/.claude/skills/ako4all/`）。无服务、无 token。
- 放哪：在 vllm-plugin-FL 仓库里建 AKO 工作区：`source/`（目标 kernel 原文 + 调用点）、`solution/`（agent 产出）、`bench/`（bench 脚本，**必须命令行运行且结果打到 stdout**）、`knowledge/`（C500/BI-V150 规格卡：HBM 峰值带宽、SM 数、L2、triton 版本、mcoplib op 清单）、`HINTS.md`、`ITERATIONS.md`（台账：每轮 baseline/新方案 µs、加速比、正确性 max_err、结论、commit）。
- 循环怎么转：每个 kernel 一条 `opt/<kernel>` 分支 → agent 把 kernel 拷进 `solution/` → 生成 `bench/bench.sh`（stdout！）→ baseline 校验（正确性+计时，反作弊三层：必须 import solution/、不许动 bench、ncu/profiler 复核真实加速）→ profile → 改 → bench → 记 ITERATIONS.md → commit；连续 3 轮无改善强制重新 profile/换方向。官方数据：B200 级赛题单 kernel 约 55 分钟 agent 时间，1.12-1.50× 超 FlashInfer 人类专家。
- 远程 GPU 适配：本地无 GPU 完全成立——`ssh c500 'cd ~/vllm-plugin-FL && bash bench/bench.sh'`，stdout 流回本地贴进对话；反作弊的 profile 层用 metax `mcu_profiler`/iluvatar 工具替代 ncu（KernelAgent 的 ncu 依赖在两平台都不存在，必须换）。

### 3.2 kernelgen-flagos（flagos-ai/skills + kernelgen-mcp）——FlagGems 感知的生成器
- 装什么：①`github.com/flagos-ai/skills` 克隆进技能目录；②KernelGen MCP：`kernelgen.flagos.io/login` 取 Bearer token，在 ZCode MCP 配置里注册 kernelgen-mcp 服务。skill 启动先强制 MCP 连通检查。
- 怎么用：在 **FlagGems 仓库**里触发（skill 自动识别 repo 类型：FlagGems / vLLM / 通用 Triton，各走子流程）。对本赛：让它以 `fused/` 目录现有算子为模板生成/优化目标 kernel（decode attention stage1/2、mm M=64 特化、rope+reshape_and_cache 融合），目标后端 metax/iluvatar；产出的候选直接丢进 AKO 工作区 `solution/` 走 bench 循环。vLLM 侧子流程遵循 flagos-ai/FlagGems-vllm 的 workflow.md（代码落点、测试与 benchmark 要求、deep optimization 触发条件）——与我们"只改两个仓库"的边界天然对齐。
- 定位：它是**生成器**不是裁判；所有产出必须过 AKO bench 才进台账。

### 3.3 KernelAgent（Profile→Diagnose→Prescribe→Orchestrate→Explore→Measure）——方法论骨架
- 开源（PyTorch 官方博客配套，github），依赖 NCU。**两台国产卡都没有 NCU → 不装工具、装纪律**：把六段式当 checklist 嵌进 AKO 循环——Profile（serve 上 torch.profiler 抓 3 个 decode-only step + 1 个 prefill chunk，每场景每卡各一份，存档）、Diagnose（对每 kernel 算 achieved GB/s / 峰值，标 memory/compute/latency-bound，对应第 0 节预算表逐项核对）、Prescribe（从本台账选题）、Orchestrate（AKO 单 kernel 分支）、Explore（KernelGen/Kernel-Smith 候选）、Measure（bench.sh + evalscope math_500 门禁 ≥0.95）。
- Profile 产物是全局一次性投入：约 2 小时/卡，换来所有后续假设的定量排序依据。

### 3.4 Kernel-Smith（chat.intern-ai.org.cn/kernel-smith）——MXMACA 特化生成器
- 不装任何东西：在线 chat，选 Kernel-Smith-MACA-30B（沐曦+上海AI Lab，MXMACA 算子生成平均加速比超 DeepSeek-v3.2/Qwen3-235B，已落地 SGLang/DLBlas）。原生支持 MXMACA → **metax 侧优先用它出候选**（Triton/CUDA for MACA），iluvatar 侧用 KernelGen。
- 用法：贴 kernel 源码 + 形状（bs=64、ctx 4096/16384、2 KV 头、128 head_dim、bf16）+ 目标（带宽利用率、split 数），要它给候选与理由；候选同样进 AKO bench。人工浏览器操作，单人是可承受的（每 kernel 10-20 分钟）。

### 3.5 micro-bench 设计规范（stdout、秒级、可进 ITERATIONS.md）
- 每个 op 一个 python 文件（放 `VL/benchmarks/ops/` 或 AKO `bench/`），固定形状 = 真实形状（decode: M∈{64,128}、ctx∈{4096,16384}、hidden 2048、kv_heads 2、head_dim 128、bf16；prefill: M∈{4096,8192}）；warmup 25 + iter 100，CUDA event 计时；输出单行可 grep：
  `RESULT op=decode_attn impl=triton_splitkv bs=64 ctx=16384 us=812.4 gbps=1450 speedup=1.71 status=PASS maxerr=3.1e-3`
- 正确性同脚本内做：对 reference（vLLM/PyTorch 原路径）`allclose`，bf16 容差 atol=2e-2/rtol=2e-2，打印 max_abs_err；FAIL 即退出码非 0。
- 单脚本运行 5-30 秒；总入口 `bench/run_all.sh` 顺序跑全部并汇总 stdout 表格。端到端门禁另跑 evalscope math_500（Level 3 ≥0.95）。

---

## 4. 假设台账（按优先级）

> 通用约定：预期收益给的是"对 total tokens/s 的贡献"区间，依据 = 第 0 节流量预算 × 业界证据 × kernel 现状；工作量含远程机验证；所有假设先 micro-bench 后接线。

### H1（P0）metax decode attention：maca FA2 vs 现成 triton split-KV 对决并路由胜者 【16K 主、4K 次】
- 目标文件：`VL/vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py`（`forward` 的 `num_decodes>0` 分支 L824-845）；候选 kernel `.../attention/ops/triton_decode_attention.py`（现成、已带 maca 修改、未被引用）。
- 机制：micro-bench 两者（bs=64、ctx 4096/16384、GQA16/2、head128、bf16、paged KV）；若 triton split-KV 胜，在 FlashAttentionImpl decode 分支接 `triton_decode_attention`（或注册新 attention backend 供 `op_backends` 选择），KV 布局/stride 按其 `Req_to_tokens` 约定适配。
- 预期收益：若 maca FA2 在 2-KV/16K 只达 50-60% 带宽而 split-KV 达 80%（Flash-Decoding 证据：长上下文低并行度下数倍级），16K decode 步时 -20~-40% ⇒ 16K 场景 tokens/s +25~60%；4K +5~15%。这是全台账上限最大的单项。
- 精度风险：低（exact attention，两者数值等价，bf16 累加差 ~1e-3）。
- micro-bench：§3.5 脚本，双实现同形状对比 gbps；5 分钟出结果。
- 工作量：bench+路由 4-8h；若需修 kernel 的 maca 特有问题 +8-16h（交 Kernel-Smith/AKO 循环）。

### H2（P0）小 M GEMM（M=64/128）调优与实现选择【4K 主、16K 次、prefill 兼得】
- 目标文件：`FG/src/flag_gems/runtime/backend/_metax/{mm_metax_expand.yaml,tune_configs.yaml}`、`_metax/ops/{mm.py,matmul_bf16.py}`、`_iluvatar/ops/{mm.py,linear.py}`；路由开关 `VL/vllm_fl/utils.py::use_flaggems_op`（blacklist env / 平台 yaml）。
- 机制：对四类 GEMM 真实形状（qkv 64×2048:2304、o 64×2048:2048、gate_up 64×2048:12288、down 64×6144:2048、lm_head 64×2048:130560；以及 prefill M=4096/8192 同列）跑 flagtune 生成 per-shape 配置写入 tune_configs；对比"FlagGems Triton mm vs 平台原生（metax: torch cublas 路径，即 blacklist 掉 mm）vs vendor"选优。
- 预期收益：GEMM+lm_head 占 4K decode 流量 ~29%；带宽利用率 55%→80% ⇒ 4K 步时 -6~-9%，prefill GEMM 段 -5~15%（M 大时更接近 compute roofline，收益取决于当前配置质量）。
- 精度风险：无（同算法调块尺寸）。
- micro-bench：torch.matmul 三方对比脚本（triton-with-config / cublas / 原配置），秒级。
- 工作量：6-12h（flagtune 跑形状网格 + yaml 落盘 + A/B serve）。

### H3（P0）iluvatar 慢实现清除：确认三 op 实际 impl 并把 vendor:iluvatar 兜底换成真 kernel 【两场景，保险+小赚】
- 目标文件：`VL/vllm_fl/dispatch/backends/vendor/iluvatar/impl/{normalization,activation,rotary}.py`（当前为 PyTorch 占位）；对照 `vendor/cuda/impl/*`（调 `vllm._custom_ops`，iluvatar CUDA 兼容大概率可用）。
- 机制：①远程机 `VLLM_FL_DISPATCH_DEBUG=1 VLLM_FL_LOG_LEVEL=DEBUG` 起 serve，grep `Op 'rms_norm' ->` 实际 impl_id 与 fallback WARNING；②若 flagos 失败被兜到 vendor:iluvatar/reference：把 vendor impl 改为 `vllm._custom_ops.fused_add_rms_norm/silu_and_mul/rotary_embedding`（CUDA 兼容路径），或修 FlagGems kernel 在 flagtree triton 的编译问题。
- 预期收益：若当前真在走 PyTorch 链：decode 每 op 链 6 kernel→1，~0.3-0.8ms/步 ⇒ 4K +3~8%，16K +1~2%；prefill 该项流量减半 ⇒ +1~2%。若本来就走 flagos：收益 0，但 1 小时确诊是必要保险。
- 精度风险：无。
- micro-bench：dispatch 三后端同 op 对比脚本（resolve_op 各 impl 直调计时）。
- 工作量：诊断 1h；替换 3-6h。

### H4（P1）metax 三 op vendor 原生 kernel 启用与否（mcoplib fused_add_rms_norm 等）【两场景，小】
- 目标文件：`VL/vllm_fl/dispatch/config/metax.yaml`（op_backends 加 `vendor:metax`）或 `.../vendor/metax/register_ops.py` 调 priority；kernel 已在 `impl/layernorm.py`。
- 机制：mcoplib 原生 `torch.ops._C.fused_add_rms_norm` vs FlagGems Triton 同形状对比，胜者进 yaml 顺序首位。
- 预期收益：每 op 差 1-3µs ×（42×2+42+42）次/步 ⇒ 0.2-0.6ms ⇒ ≤2-5% 上限，大概率 <2%；但零风险零成本，与 H3 同一脚本顺带做。
- 工作量：2-4h（含 bench）。

### H5（P1）metax 图模式核验与代码级提升（FULL_DECODE_ONLY 等效覆盖）【4K 主】
- 目标文件：`VL/vllm_fl/platform.py`（L253 `maca specific config updates` 处，仿 iluvatar 基线在代码里确保 decode 全图）；`VL/vllm_fl/compilation/graph.py`（捕获集合）；验证面 `.../metax/impl/attention/flash_attn.py` decode 分支无 host 动态张量。
- 机制：先在 metax 远程机 torch.profiler + `compilation_counter`/日志确认基线 decode 是否整图重放、捕获 batch 集是否含 64；若为 PIECEWISE 或 eager：通过 platform.py 代码级置 `cudagraph_mode`（不改 serve 参数——yaml/代码内默认属于插件代码，合规性与 HOPPER_LONG_CONTEXT_OPT 同类）；同时清 `RotaryEmbeddingFL.forward_oot` 每步 `to()/chunk()` 冗余。
- 预期收益：PIECEWISE→FULL_DECODE_ONLY：42 层×2 段 graph 断点 + eager attention CPU 开关 ≈ 1-3ms/步 ⇒ 4K +8~20%（若已是 FULL 则 0，同样先确诊）；bs=64 若被 pad 到 128 则省一半冗余 token 计算。
- 精度风险：无（执行图不变）。
- micro-bench：serve 日志/counter 检查 + step 时间对比（1 次 serve 拉起，10 分钟）。
- 工作量：诊断 2h；改默认+回归 4-6h。

### H6（P1）iluvatron Triton attention 路线选择：vLLM TRITON_ATTN(USE_TD=0) vs FlagGems AttentionFLBackend（Triton FA2）【两场景】
- 目标文件：`VL/.../flaggems/flaggems.py::attention_backend`（`VLLM_FL_USE_FLAGGEMS_ATTN` 默认值逻辑，可按平台在代码内改默认）；`FG/src/flag_gems/ops/attention.py`（flash_attn_varlen_func，iluvatar 黑名单未禁）。
- 机制：iluvatar 黑名单允许 Triton attention；micro-bench vLLM triton attention（无 TD）与 FlagGems FA2 varlen（decode+prefill 两形状）；胜者接线。若 flagtree 3.6（triton≥3.3）可用，进一步开 TD/TMA 路径（`patch_triton_language_for_iluvatar` 自动跳过）。
- 预期收益：两 Triton kernel 在 GQA2/head128 的访存效率差常见 10-30% ⇒ 16K +5~15%、4K +2~6%（attention 占比加权）。
- 精度风险：低。
- 工作量：bench 4h；接线 4-8h。

### H7（P1）prefill attention（compute-bound）核验：varlen 因果遮罩与 GQA 打包效率【两场景 prefill 段】
- 目标文件：metax `.../metax/impl/attention/flash_attn.py`（prefill 分支 L800-823，含 host 侧 `cu_prefix_kv_lens` 构造——顺带挪到 metadata builder 消除 H2D+sync）；iluvatar vLLM triton extend 路径。
- 机制：prefill 段 kernel 时间占比 profile 定位；对 4096-token 单序列 varlen 做 bf16 roofline（C500/BI-V150 bf16 峰值），缺口>30% 才立项改（如 split 头、block 表布局 NHD/HND 选择——`get_kv_cache_stride_order` 已支持双布局，A/B 即可）。
- 预期收益：不确定，缺口存在时 prefill 段 -10~25%；两场景 prefill token 数均为 decode 的 4 倍，总分约 +2~6%。
- 工作量：profile 2h；改 8-16h（视缺口）。

### H8（P2）rope + reshape_and_cache 融合（MHA 版三融合）【decode 微收益，低成本展示项】
- 目标文件：新增 `FG/src/flag_gems/fused/rotary_and_cache.py`（模板抄 `fused/fused_deepseek_v4_qnorm_rope_kv_rope_insert.py`）；接入 `VL/vllm_fl/dispatch/backends/flaggems/impl/rotary.py` 或 metax FA backend 的 KV 写入点（现用 `ops.reshape_and_cache_flash`）。
- 机制：Q/K rope 后 K 立即写 cache——融合后 K 只过一次寄存器→HBM；省 1-2 launch/层。
- 预期收益：<0.5-1% 步时（bs64 下 K/V 仅 64KB）；价值在低风险练手与 prefill 段的 KV 写流量。
- 精度风险：低。
- 工作量：6-10h（KernelGen 生成 + AKO 验证）。

### H9（P2）iluvatar 采样段 Triton 化（FlagGems top_k_per_row_decode / apply_repetition_penalties）【4K 次】
- 目标文件：`VL/.../vendor/iluvatar/iluvatar.py::patch_sampler_compile_for_iluvatar`（现在解 compile 跑 eager）；候选 `FG/fused/top_k_per_row_{decode,prefill}.py`、`fused/apply_repetition_penalties.py`；`VL/vllm_fl/ops/_C_ops_registry.py` 的 fallback 也可换。
- 机制：lm_head 后 130560 vocab 上的 top-k/top-p 与惩罚算子 eager 化是每步 0.1-0.5ms 纯开销；用 FlagGems Triton 版替换。
- 预期收益：4K +1~4%；16K <1%。精度风险：低（同分布采样需验证随机数路径，greedy 则无风险——eval 走 greedy）。
- 工作量：4-8h。

### H10（P2，高风险观察）KV cache fp8（代码内强置）【仅 16K，上限大红线近】
- 目标文件：`VL/vllm_fl/platform.py`（config 改写挂点，同 H5）；attention backend 的 kv dtype 视图逻辑（metax `flash_attn.py` 已有 fp8 视图分支但 `get_fp8_dtype_for_flashattn` 抛 NotImplementedError——需 maca flash-attn fp8 支持确认，大概率不支持 → 仅 iluvatar Triton 路线可试）。
- 机制：16K KV 45GB→22.5GB，步时上限 -30~40%。但：math_500 余量 1.2 点，fp8 KV 典型 -0.3~-1.0 点；且属"代码内改 serve 行为"，需先与赛规确认。
- 结论：排最后，仅在 H1/H2 收益不及预期且精度实测掉 <0.8 点时启动。
- 工作量：8h + 高验证成本。

### 反向结论（明确不做）
- **GEMM epilogue 融合（silu 进 gate_up 尾部）**：decode 省流量 ~3µs/层、prefill <0.1%——不做（除非 H2 顺带）。
- **swiglu/geglu/reglu GEMM 融合版**：metax 黑名单全 dtype 失败，iluvatar 无急需——不立项。
- **进一步融合 norm/act（已各自单 kernel）**：收益 <0.5%，不做。

---

## 5. 执行顺序建议（第一周）

1. Day1：H3+H4+H5 的诊断三件套（dispatch debug 日志、graph counter、torch.profiler 各一份/卡）——全部 stdout 化归档进 AKO `knowledge/`。
2. Day1-2：搭 §3.5 micro-bench 骨架 + AKO 工作区 + kernelgen-mcp 配置；Kernel-Smith 账号就绪。
3. Day2-3：H1 对决 bench（metax）与 H6 对决 bench（iluvatar）——同一天出结果，胜者立项接线。
4. Day3-5：H2 flagtune 形状网格（两卡并行跑）。
5. 每个 >=2% 的假设落地后跑一次 evalscope math_500 门禁；台账 ITERATIONS.md 记录。

## 附：关键文件速查
- 路由配置：`VL/vllm_fl/dispatch/config/{metax,iluvatar}.yaml`
- 解析逻辑：`VL/vllm_fl/dispatch/manager.py::_resolve_impl`（token 顺序优先于 priority）
- OOT 注入：`VL/vllm_fl/ops/custom_ops.py::OOT_OPS` + `VL/vllm_fl/ops/{layernorm,activation,rotary_embedding}.py`
- metax 原生 kernel：`VL/.../vendor/metax/impl/{layernorm,activation,rotary_embedding}.py`、mcoplib via `VL/vllm_fl/ops/_C_ops_registry.py`
- metax attention：`VL/.../vendor/metax/impl/attention/flash_attn.py`；死代码宝藏：同目录 `ops/{triton_decode_attention,triton_unified_attention}.py`
- iluvatar 补丁：`VL/.../vendor/iluvatar/iluvatar.py`（4 个 patch + sampler 解 compile）
- 图捕获：`VL/vllm_fl/compilation/{graph.py,break_graph.py}`、`VL/vllm_fl/platform.py`（maca config 挂点 L253、static graph L349）
- FlagGems 融合池：`FG/src/flag_gems/fused/`（silu_and_mul、fused_add_rms_norm、rotary_embedding、matmul_bias_activation、swiglu、top_k_per_row_*、apply_repetition_penalties、reshape_and_cache*、DSAv4 三融合模板）
- FlagGems 平台层：`FG/src/flag_gems/runtime/backend/{_metax,_iluvatar}/`（ops/、tune_configs.yaml、op_black_list.yaml、mm_metax_expand.yaml）+ `FG/src/flag_gems/flagtune/`
- 业界证据：vLLM fusions 设计文档与 #36066/#43224 融合 tracker、Flash-Decoding（pytorch.org/blog/flash-decoding）、FlashInfer GQA decode、ServingStudio kernel 表。

## 附：调研来源
- [vLLM torch.compile fusions 设计文档](https://docs.vllm.ai/en/latest/design/fusions)
- [FlashInfer #2771 rmsnorm kernel 对比](https://github.com/flashinfer-ai/flashinfer/issues/2771)
- [fused AllReduce+RMSNorm (arXiv 2505.11329)](https://arxiv.org/html/2505.11329v5)
- [ServingStudio kernel 基准表](https://servingstudio.cs.washington.edu/kernels.html?kind=residual_rms_norm&view=grid)
- [Flash-Decoding 官方博客](https://pytorch.org/blog/flash-decoding) / [Stanford CRFM](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)
- [FlashInfer 介绍（GQA/split-KV decode）](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
- [vLLM 融合工作 tracker #36066](https://github.com/vllm-project/vllm/issues/36066) / [#43224 手工融合 RFC](https://github.com/vllm-project/vllm/issues/43224) / [#38733 activation IR 化](https://github.com/vllm-project/vllm/issues/38733)
- [vLLM 101: compile 与 CUDA graph 捕获](https://aiengineeringfromscratch.com/manual-vllm-101/how-compile-and-cuda-graphs-are-captured)
- [TongmingLAIC/AKO4ALL](https://github.com/TongmingLAIC/AKO4ALL)
- [flagos-ai/skills](https://github.com/flagos-ai/skills) / [KernelGen releases](https://github.com/flagos-ai/KernelGen/releases) / [FlagOS skills 知乎详解](https://zhuanlan.zhihu.com/p/2024207186176288648) / [FlagGems-vllm workflow.md](https://github.com/flagos-ai/FlagGems-vllm/blob/main/workflow.md)
- [KernelAgent PyTorch 博客](https://pytorch.org/blog/kernelagent-hardware-guided-gpu-kernel-optimization-via-multi-agent-orchestration)
- [Kernel-Smith 论文 arXiv 2603.28342](https://arxiv.org/html/2603.28342v1) / [InternLM/Kernel-Smith](https://github.com/InternLM/Kernel-Smith) / [沐曦官方发布](https://www.metax-tech.com/ndetail/12575.html)
