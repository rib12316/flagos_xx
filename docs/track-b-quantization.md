# 赛道二量化方向决策文档：MiniCPM5-2B 框架内自研量化

> 研究日期：2026-11-03。纯研究结论，不含量化实现代码。
> 核心问题：**走 weight-only 还是权重+激活都量化？硬件和现成算子支不支持？精度余量够不够？**

## 0. 决策速览

| 问题 | 结论 |
|---|---|
| 主方案 | **W8A8 dynamic int8（per-channel 权重 + per-token 动态激活，无校准）**——插件内整条管线已存在，只缺 bf16→int8 权重转换脚本 |
| 保底方案 | **W8A16 weight-only int8**（激活保 bf16），FlagGems 已有现成 kernel，需自写一个 scheme 类 |
| FP8 | **排除**。C500 无 FP8 硬件（C600 才有）；BI-V150 仅 ixTE 私有路径；插件 `PlatformFL.supports_fp8()` 对非 NVIDIA 一律返回 False |
| int4（GPTQ/AWQ） | **排除**。2B 模型精度风险大 + 无校准工装 + 两卡 int4 GEMM 路径未建立 |
| KV cache int8 | **暂缓为二期**。收益上限高（16K 场景 KV 流量/容量减半）但需改 attention kernel（vendor 后端仅支持 auto/fp8），与算子方向冲突、长上下文误差累积，一期不做 |
| 精度余量 | 基线 0.962、红线 0.95，**绝对余量仅 1.2pt**。W8A8 动态 int8 业界平均损失 1–3% 且小模型偏不利 → **必须以 math_500 实测一票否决**；不过则切 W8A16（近无损） |
| 预期收益 | decode 权重流量 4.5GB→约 2.3–2.8GB（约减半）→ 权重受限段 decode 提升 ~1.6–1.9x；另省 ~1.7–2.2GB 显存使 16K/64 并发 KV 预算从临界变宽松；prefill 若 int8 tensor core 打满另有最多 ~2x 计算收益（需实测） |
| 主方案工作量 | **约 12–24 人时**（转换脚本 4–8h + 双卡接线验证 4–8h + 精度回归 2–4h + 性能验证 2–4h）；保底 W8A16 另加 8–16h |

---

## 1. 硬件低精度支持调研

### 1.1 沐曦曦云 C500（MACA 3.7.0.38）

| 项目 | 数值 / 结论 | 来源 |
|---|---|---|
| INT8 算力 | **480 TOPS**（OAM 版部分资料称 560） | 公开规格汇总 |
| BF16/FP16 算力 | 约 240–280 TFLOPS | 同上 |
| INT8:BF16 比 | **2:1**，存在专用 INT8 张量单元 | 推算自上两项 |
| FP8 | **无**。FP8 在下一代 C600（HBM3e, 144GB, 3.35TB/s）才引入 | 公开报道 |
| 显存 | 64GB HBM2e，**1.8 TB/s** | 公开规格 |
| 定位 | 对标 A100/A800 级，XCORE 统一架构 | 招股/IPO 资料 |
| 官方量化支持 | MXMACA 发布说明明确 **"支持W8A8_INT8和W4A16量化"**（mxc500-sglang-2.33.1.7 / SDK 2.33.1.15，已核实原文） | developer.metax-tech.com 文档 613 |
| int8 GEMM 库 | mcBLAS 高性能库存在（对标 cuBLAS），INT8 GEMM 具体接口/性能未公开；**我们不走 mcBLAS，走 Triton（FlagGems）路线** | developer.metax-tech.com |
| Triton int8 | FlagGems 仓库内自带 `_metax` 后端的 `matmul_int8` Triton kernel（`tl.dot` int8→int32 累积），证明 **MACA Triton 编译链支持 int8 dot** | 本地 FlagGems 代码 |

关键澄清：本插件体系里 **MetaX 的 `device_type` 是 `"cuda"`**（走 MACA 的 CUDA 兼容层，通信用 mccl），不是 `"musa"`——`vllm_fl/utils.py` 的 `VENDOR_DEVICE_MAP`：`"metax": {"device_type": "cuda", ...}`；`"musa"` 是摩尔线程（mthreads）。因此 `vllm_fl/__init__.py:register_quant_linear()` 里"musa 跳过注册"的保护分支**不影响 C500**（它只挡摩尔线程），C500 和 BI-V150 都会正常注册 W8A8 OOT kernel。

### 1.2 天数智芯 BI-V150（Iluvatar CoreX）

| 项目 | 数值 / 结论 | 来源 |
|---|---|---|
| INT8 算力 | **384 TOPS** | mirrorfrog 规格页 / 南开大学 2025 采购公告 |
| FP16 算力 | 192 TFLOPS（FP32 48） | 同上 |
| INT8:FP16 比 | **2:1** | 推算 |
| FP8 | 规格页称"支持但需配合 ixTE 库"——**只有私有库路径，无 vLLM/FlagGems/Triton 路径**，对我们不可用 | mirrorfrog；插件代码 |
| 显存 | 64GB HBM2e，**约 1.2 TB/s** | 同上 |
| INT4 | 模力方舟文档称支持 INT8/INT4 量化但"需配合 IGIE 推理框架或 ixRT 推理引擎"——**绑定天数自家推理栈，非我们的 vLLM 管线** | moark.com 文档 |
| Triton int8 | FlagGems `_iluvatar` 后端同样自带 `matmul_int8` Triton kernel；黑名单仅 2 个算子（grouped_topk、topk_softmax，2026-03-11 测试），`scaled_mm`/量化算子不在列 | 本地 FlagGems 代码 |

### 1.3 两卡 bf16→int8 实际加速比证据

- 两卡 INT8:FP16 稀疏/稠密算力比均为 **2:1**（480 vs 240；384 vs 192），这是 int8 tensor core 存在的直接证据；沐曦官方 SDK 已将 W8A8_INT8 列为发布特性（SGLang 管线），说明 int8 GEMM 在其推理栈内已产品化。
- **未找到**两卡上公开的 bf16 vs int8 GEMM 实测加速比数据。且我们走 Triton 路线，`tl.dot(int8,int8)→int32` 能否映射到满血 480/384 TOPS（还是经 fp16 管线模拟）**必须上机实测**。因此本决策对收益的口径是：**decode 权重流量减半是保底收益（访存受限场景与计算精度无关）；prefill 计算加速是待验证的增量**。
- FlagGems 两后端的算子兼容性黑名单（仓库内 YAML，实测日期标注）：
  - `_metax/op_black_list.yaml`（2026-01-27）：`cutlass_scaled_mm`、`per_token_group_quant_fp8`、`copy_` 等 19 个失败；**`scaled_mm`、`weight_int8pack_mm`、`dynamic_per_token_quant_int8` 不在黑名单**。
  - `_iluvatar/op_black_list.yaml`（2026-03-11）：仅 `grouped_topk`、`topk_softmax` 失败。

### 1.4 结论

硬件层面 int8 双卡都支持（2:1 算力比 + 官方量化背书 + Triton int8 dot 编译可行）；FP8 双卡实际不可用；int4 无我们管线的可用路径。**方案空间收敛到 int8 的 W8A8 / W8A16 两选。**

---

## 2. 现成资产盘点（本地代码）

### 2.1 vllm-plugin-FL：一条完整的动态 W8A8 INT8 管线（对稠密 Linear 直接可用）

加载侧（checkpoint 驱动，**不需要 `--quantization` 超参**）：

- `vllm_fl/quantization/w8a8/int8_mode.py` — `should_use_packed_w8a8()`：识别 compressed-tensors `pack-quantized` 格式 + per-channel 对称 int8 静态权重 + per-token 对称 int8 **动态**激活的 checkpoint 元数据。
- `vllm_fl/quantization/w8a8/packed.py` — `CompressedTensorsPackedW8A8Int8`（继承 vLLM 0.24 的 `CompressedTensorsW8A8Int8`）：`create_weights` 接收 4×int8 打包的 int32 权重（`PackedvLLMParameter`，要求 in_features % 4 == 0，2048/5k 均满足），`process_weights_after_loading` 用 `unpack_uint8b128_int32` 解包后走原生流程。`install_packed_w8a8_scheme()` monkey-patch `CompressedTensorsConfig._get_scheme_from_parts`——这是插件接入 vLLM 量化机制的证明：**不动 vLLM 主库、经 checkpoint config 自动选 scheme**。
- `vllm_fl/quantization/w8a8/reference.py` — `unpack_uint8b128_int32`（offset-binary-128 打包）、`dynamic_per_token_quant_int8`、`w8a8_linear_reference`（纯 torch 参考实现，做数值对拍用）。

执行侧：

- `vllm_fl/quantization/w8a8/linear.py` — `FLW8A8DynamicLinearKernel`（继承 vLLM `Int8ScaledMMLinearKernel`）：`apply_weights` = `dynamic_per_token_quant_int8(x)` → FlagGems `scaled_mm(x_q, w[K,N], x_scale[M], w_scale[N], out bf16)`。`register_fl_w8a8_linear_kernel()` 把它 **prepend 进 `_POSSIBLE_INT8_KERNELS[PlatformEnum.OOT]`**，原生 CUDA kernel 保序为 fallback。
- `vllm_fl/quantization/quant_linear.py` — `add_oot_quant_kernel()`：从源平台继承 int8/fp8 kernel 候选并安装上述三件套（linear kernel + packed scheme + MoE selector），由 `vllm_fl/__init__.py:register_model() → register_quant_linear()` 触发；仅 `device_type == "musa"`（摩尔线程）跳过。
- `vllm_fl/dispatch/backends/flaggems/impl/quantization.py` — 后端中立的 per-token int8 量化 Triton kernel（自带 banker's rounding），优先调 `flaggems_vllm.ops.scaled_int8_quant`，失败回落本地 Triton，再回落 PyTorch 参考——三级 fallback 已注册在 `register_ops.py`。
- `vllm_fl/quantization/w8a8/moe.py` + `moe_experts.py` — W8A8 MoE 路由（FlagGemsW8A8Experts）。MiniCPM5 是稠密模型用不上，但说明该量化体系按整套设计。
- 测试：`tests/unit_tests/dispatch/test_flaggems_quantization.py`、`tests/unit_tests/quantization/test_packed_int8.py`、`test_w8a8_reference.py`——单测范式现成。

**覆盖范围**：以上作用于 vLLM `RowParallelLinear/ColumnParallelLinear`（q/k/v/o/gate/up/down 全部稠密 Linear），即对 MiniCPM5-2B 的 LlamaForCausalLM 路径**结构上直接适用**（MoE 部分才需要额外适配，我们没有 MoE）。

### 2.2 关键缺口：bf16 → int8 checkpoint 的转换脚本不存在

插件只消费"已量化的 compressed-tensors pack-quantized checkpoint"（注释称来自 FlagOS-Compressor）。MiniCPM5-2B 官方给的是 bf16，仓库内（含 `tools/` 目录）**没有任何量化转换脚本**。这就是主方案唯一要写的代码（约 100–200 行，见 §5）。

### 2.3 FlagGems：量化算子库存充足

- `src/flag_gems/ops/scaled_mm.py` — 后端中立 Triton scaled_mm：**int8×int8→int32 累积** + per-token scale_a[M] × per-channel scale_b[N] + bias，out bf16；cutlass 快路径仅 CUDA sm90（双卡不适用，走 Triton 主路径）。同时在 `flag_gems/__init__.py` 以 `_scaled_mm`/`_scaled_mm.out` aten override 注册（torch≥2.5）。
- `src/flag_gems/ops/weight_int8pack_mm.py` — **W8A16 weight-only kernel 现成**：A bf16[M,K] × B int8[N,K] + per-channel scale[N]，kernel 内 int8→bf16 精确表示、fp32 累积后乘 scale。注意 int8 值（≤7bit 尾数）在 bf16 中精确可表示，数值上等价于"bf16 GEMM + int8 权重舍入误差"。尚未被插件引用（W8A16 需要自己接线）。
- `src/flag_gems/fused/act_quant.py` — vLLM/sGLang 风格 per-token/per-tensor 激活量化 kernel。
- `src/flag_gems/runtime/backend/_metax/ops/matmul_int8.py`、`_iluvatar/ops/matmul_int8.py` — 两卡的 int8 Triton matmul 各有厂商定制版（int32 累积）。
- int4 相关仅有打包工具（`_dyn_quant_pack_4bit_weight.py`、`_convert_weight_to_int4pack.py`、`_weight_int4pack_mm_with_scales_and_zeros.py`），无两卡可用的 int4 GEMM 落地路径。
- fp8：`per_token_group_quant_fp8.py` 在 metax 黑名单（"All dtypes failed"），与"C500 无 FP8 硬件"互相印证。

### 2.4 vLLM 0.24 量化注册机制与插件接入方式

vLLM 0.24 的量化选择有两条路：① `--quantization` 超参 → `QuantizationRegistry`（比赛禁用）；② **checkpoint `config.json` 的 `quantization_config` 字段** → `CompressedTensorsConfig` 等 → `_get_scheme_from_parts` 选 scheme。插件走的正是 ② + monkey-patch，配合线性 kernel 的平台注册表（`_POSSIBLE_INT8_KERNELS[PlatformEnum.OOT]`）插入自研 kernel——**完全可以不动 vLLM 主库接入自定义量化**，且不触碰禁用的超参。这条路径天然满足"框架内有开发量"：我们的开发量 = 自研转换脚本 + 插件内 scheme/kernel 补丁的适配验证。

---

## 3. 方案空间对比（回答核心问题）

### 3.1 W8A8 dynamic int8（推荐主方案）

- **形式**：权重 per-channel 对称 int8（离线 RTN，无需校准数据）；激活 per-token 动态对称 int8（运行时逐 token 计算 absmax，无需校准）。
- **精度**：业界平均损失 1–3%/任务（arXiv:2411.02355 等），Red Hat 四套基准实测"无可测退化"；但社区共识 **<4B 小模型偏差更大**。我们余量 1.2pt，W8A8 的损失区间骑在红线上 → **胜负手在实测**，有明确的退路（W8A16）。
- **decode 收益**：每步权重流量约 4.5GB→2.3GB（lm_head/embed 不量化时 ≈2.8GB）。C500 1.8TB/s：权重段 2.5ms→1.3–1.6ms；BI-V150 1.2TB/s：3.75ms→1.9–2.3ms。64 并发共享一次权重读，权重受限段提升 ~1.6–1.9x。额外开销仅每 token 一次激活量化 kernel（微秒级，相对 1–2ms 的 step 可忽略）。
- **prefill 收益**：若 Triton int8 dot 打到 tensor core，计算受限段最多 ~2x；未打到则持平——上机实测决定宣传口径。
- **开发量**：只写转换脚本（§5），管线其余全部现成。**全部方案里最小**。
- **显存红利**：省 ~1.7–2.2GB → 16K/64 并发 KV 峰值需求 ≈46GB vs 可用 45–48GB 从临界变宽松，与调度方向直接耦合。

### 3.2 W8A16 weight-only int8（保底方案，精度最安全）

- **形式**：权重 per-channel int8，激活保持 bf16，kernel 内反量化（int8→bf16 精确）。
- **精度**：weight-only int8 业界普遍报告**近无损（<1%，多数 <0.5%）**（OmniQuant/LLM.int8() 等引用的证据链）。math_500 这类数学推理对权重 int8 舍入的敏感度低——是最安全的一档。
- **decode 收益**：与 W8A8 相同（瓶颈是权重字节而非计算，减半流量照拿）；无激活量化开销。
- **prefill 收益**：无（反量化后仍是 bf16 dot）。
- **开发量**：需要 ① 自写 scheme 类（仿 `packed.py`，`create_weights` int8+scale、`apply` 调 FlagGems `weight_int8pack_mm`）② patch `_get_scheme_from_parts` 接 weight-only int8 元数据 ③ 转换脚本（dense int8，无需打包）。估计 8–16h，kernel 不用写。
- **定位**：W8A8 在 math_500 掉线时的直接替补；也作为"精度问题定位工具"（W8A16 过线而 W8A8 不过 ⇒ 损失来自激活量化，可进一步考虑 SmoothQuant 或只量化 MLP 的混合方案）。

### 3.3 FP8 —— 排除

C500 无 FP8 硬件单元（C600 才有）；BI-V150 的"FP8 需 ixTE 库"是私有栈；插件 `vllm_fl/platform.py:supports_fp8()` 对非 NVIDIA 返回 False；FlagGems fp8 量化算子在 metax 黑名单。三重否定，无讨论空间。

### 3.4 int4（GPTQ/AWQ 类）—— 排除

- 精度：2B 模型 int4 普遍需要 GPTQ/AWQ 误差补偿与校准集；RTN int4 在小模型上掉点显著，红线 1.2pt 余量承受不起。
- 工程：需要校准数据管线（比赛未提供、时间不允许）；两卡 int4 GEMM kernel 路径未建立（FlagGems 只有 pack 工具）；"W4A16"是沐曦自家 SGLang 的能力，不在我们的 vLLM 管线里。
- 结论：直接排除，不投入。

### 3.5 KV cache int8 —— 暂缓（二期候选）

- 收益上限其实很高：16K 场景 KV 读流量与容量同时减半——每 token 42KB→21KB，16K/64 并发 KV 峰值 46.8GB→23.4GB，理论上并发容纳力翻倍；且长上下文 decode 的 KV 读取（16K 尾段每步理论 ~44GB/64seq）可能超过权重流量成为带宽大头，KV int8 对 16K 场景的总吞吐杠杆可能大于权重量化。
- 但代价与我们约束冲突：vendor attention 后端不支持 int8 KV（`vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py:supports_kv_cache_dtype` 仅接受 `auto`/`fp8`），需要动 attention kernel/后端 = 与算子优化方向抢同一批人力且有回归风险；KV 量化误差随上下文长度累积，恰好在 16K 场景风险最大；输出异常即成绩无效。
- **决策**：一期只做权重 W8。等 W8 落地后用 profiling（decode 每 step 权重读 vs KV 读占比）决定是否二期投入 KV int8；若做，建议 per-channel(K)+per-head(V) scale 的极简方案并保留 bf16 回退开关。

### 3.6 补充：分层混合量化（风险阶梯中间档）

若 W8A8 全量不过线、W8A16 又想多挤性能，可只量化占权重 ~70% 的 MLP（gate/up/down），注意力四矩阵保 bf16——牺牲 30% 流量收益换精度。转换脚本按 tensor 名单过滤即可实现，零额外框架代码。列为风险阶梯第 3 档。

### 3.7 决策矩阵

| 方案 | 精度风险 | decode 收益 | prefill 收益 | 开发量 | 判定 |
|---|---|---|---|---|---|
| W8A8 dynamic | 中（1–3% 损失 vs 1.2pt 余量） | 权重段 ~1.6–1.9x | 待实测（≤2x） | 12–24h | **主方案** |
| W8A16 weight-only | 低（近无损） | 同上 | 无 | +8–16h | **保底** |
| 混合（仅 MLP W8A8/W8） | 低–中 | ~70% 权重段 | 部分 | +1h（脚本过滤） | 风险阶梯 |
| FP8 | — | — | — | — | 排除（硬件不支持） |
| int4/GPTQ/AWQ | 高 | — | — | 大 | 排除 |
| KV int8 | 中–高（长上下文累积） | 高（16K 场景） | 中 | 大（改 attention） | 二期评估 |

---

## 4. 精度风险评估与回归策略

### 4.1 2B 模型对量化的敏感度证据

- 通用结论：INT8 量化对 ≥7B 模型近无损（LLM.int8()、OmniQuant 引用链）；**<4B 模型每比特损失放大**，社区反复报告 1–3B 模型 W8A8 相对损失偏高（HF 论坛/llm-compressor issue #1394 讨论专门针对 W8A8 精度恢复）。
- W8A8 动态 int8：arXiv:2411.02355 实测 **1–3% per-task 平均损失**（动态激活优于静态）；Red Hat 2026-09 文章四套基准（含 MMLU）未见可测退化——但均是大中模型。
- weight-only int8：多方报告 **<1%**，是唯一"余量内大概率安全"的档位。
- MiniCPM5-2B 特有风险：①数学推理（math_500）对 logits 漂移比 MMLU 类选择题敏感——舍入误差改变 argmax/采样路径会直接掉分；②42 层较深，误差逐层复合；③词表 130,560 大，lm_head 量化对 logits 分布影响面大（建议 lm_head/embed 保 bf16，业界 llm-compressor 默认也忽略 lm_head）。

### 4.2 结论性判断

- **W8A16：大概率过线（预期损失 <0.5pt，余量 1.2pt）**——作为"必然有收益"的兜底。
- **W8A8：五五开到略偏乐观（损失 0.3–1.5pt 区间）**——因为它是现成管线、开发量最小，先实测，掉了再退，期望收益最大化。若一次实测掉到 0.95 附近（0.95–0.955），可尝试"仅 MLP W8A8"再救一档。

### 4.3 回归测试策略（红线纪律）

1. **任何量化改动后必须全量 evalscope math_500 Level 3**（≥0.95 硬门槛），且检查输出健全性（无乱码/复读/截断——输出异常即成绩无效）。换卡必须分别回归（C500 与 BI-V150 的 Triton 编译链不同，数值路径不同）。
2. 开发期快速门（不替代全量）：固定 50 题子集 + 与 bf16 基线对比生成文本 diff；对首层/末层线性用 `w8a8_linear_reference`（`vllm_fl/quantization/w8a8/reference.py`）做逐层数值对拍，误差应 ≤1e-2（相对）。
3. 排查顺序固定：W8A8 全量 → 不过 → W8A16（定位是否激活量化锅）→ 混合（仅 MLP）→ 仍不过则放弃量化方向保 bf16 优化（量化收益不是唯一得分点，调度/算子方向仍有空间）。
4. 性能改动与精度改动分离提交：每次只动一个变量，math_500 与 benchmark_throughput_flagos 成对记录。

---

## 5. 实现路径与工作量（主方案 W8A8）

### 5.1 权重转换：离线脚本（推荐），而非加载时在线量化

- **离线转换**：新脚本 `vllm-plugin-FL/tools/quantize_int8_rtn.py`（仓库 `tools/` 目录已存在，放这里）。逻辑：
  1. 读 MiniCPM5-2B bf16 safetensors + config.json；
  2. 对所有 decoder Linear 权重（`model.layers.*.self_attn.{q,k,v,o}_proj`、`model.layers.*.mlp.{gate,up,down}_proj`）做 per-channel 对称 RTN：`scale = absmax(out_dim)/127`，`q = round(W/scale)`；
  3. 按 compressed-tensors `pack-quantized` 格式打包（与 `unpack_uint8b128_int32` 严格互逆：`(q+128).view(uint8)` 四个一组塞 int32，注意与 `packed_factor=4, packed_dim=1, [N, K/4] int32` 的布局约定对齐）；
  4. 写出 `weight_scale`（fp32, [N]）与改造后的 `config.json`：注入 `quantization_config`（format=pack-quantized, weights: int8/channel/symmetric/static, input_quant: int8/token/symmetric/dynamic）——元数据必须精确匹配 `should_use_packed_w8a8` 的判定（`int8_mode.py`），否则 scheme 不会被选中；
  5. **跳过** `embed_tokens`、`lm_head`、norm（保 bf16，精度优先）；
  6. 自检模式：随机抽 3 层与 `w8a8_linear_reference` 对拍重建误差。
- **为什么不在线量化**：vLLM 的 scheme 在建图时就需确定 create_weights 的参数布局，在线改权重要侵入 vLLM 加载器，开发量和风险都更大；离线脚本产物是"一个新 checkpoint 目录"，组委会复现 = 跑一遍脚本 + 正常 serve，无任何量化超参。
- **合规性说明**：脚本自研（不依赖 llm-compressor 包，目标机无需装新依赖），运行时走的是插件自研 scheme（`CompressedTensorsPackedW8A8Int8`）+ 自研 kernel 选择（`FLW8A8DynamicLinearKernel`）+ FlagGems Triton kernel——"框架内有开发量"三处证据齐备，且全程不出现 `--quantization`。

### 5.2 需要改动/验证的文件清单（本地真实路径）

| 动作 | 文件 |
|---|---|
| 新增 | `vllm-plugin-FL/tools/quantize_int8_rtn.py`（转换脚本，4–8h） |
| 只读复用 | `vllm_fl/quantization/w8a8/{packed.py, linear.py, int8_mode.py, reference.py}`、`vllm_fl/quantization/quant_linear.py` |
| 验证（预期不改） | `vllm_fl/__init__.py:register_quant_linear()`（确认 metax/iluvatar 下 device_type 均为 cuda、`is_oot_enabled()` 为真、`VLLM_FL_FLAGOS_BLACKLIST` 未禁 `w8a8_dynamic_per_token_linear`） |
| 可能微调 | `vllm_fl/dispatch/backends/flaggems/impl/quantization.py` 的 fallback 顺序（若 flaggems_vllm 包缺失，确认本地 Triton 路径被选中） |
| 测试参考 | `tests/unit_tests/quantization/test_packed_int8.py`（新增 MiniCPM5 形状的用例：[2048,2048]、[5120,2048]、GQA 的 [256,2048]） |

### 5.3 工作量与里程碑

| 阶段 | 内容 | 工时 | 出口判据 |
|---|---|---|---|
| M1 | 转换脚本 + 参考实现对拍 | 4–8h | 逐层重建误差 ≤ int8 理论舍入 |
| M2 | C500 + BI-V150 各自跑通 W8A8 smoke（offline_inference） | 4–8h | 生成正常、日志确认 `Using FlagGems W8A8` 类路径生效 |
| M3 | math_500 全量精度回归 | 2–4h（另计评测时长） | ≥0.95 → 继续；<0.95 → 触发 W8A16 分支（+8–16h） |
| M4 | benchmark_throughput_flagos 4K/16K 对比 bf16 | 2–4h | decode 段 tokens/s 提升量化归因（对比开/关量化、prefill/decode 分解） |
| 合计 | 主方案 | **12–24h**；含保底分支最坏 **20–40h** | |

---

## 6. 附：关键证据索引

**本地代码**（相对仓库根）：
- `vllm-plugin-FL/vllm_fl/quantization/w8a8/packed.py`、`linear.py`、`int8_mode.py`、`reference.py`、`moe.py`、`moe_experts.py`
- `vllm-plugin-FL/vllm_fl/quantization/quant_linear.py`、`vllm_fl/__init__.py`（register_quant_linear 的 musa 判定）、`vllm_fl/platform.py`（supports_fp8）、`vllm_fl/utils.py`（VENDOR_DEVICE_MAP：metax→cuda）
- `vllm-plugin-FL/vllm_fl/dispatch/backends/flaggems/impl/quantization.py`、`register_ops.py`
- `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py`（KV dtype 仅 auto/fp8）
- `FlagGems/src/flag_gems/ops/scaled_mm.py`、`ops/weight_int8pack_mm.py`、`fused/act_quant.py`
- `FlagGems/src/flag_gems/runtime/backend/_metax/{ops/matmul_int8.py, op_black_list.yaml}`、`_iluvatar/{ops/matmul_int8.py, op_black_list.yaml}`

**Web 来源**：
- 沐曦官方发布说明（W8A8_INT8/W4A16 支持）：developer.metax-tech.com 文档 #613
- MXMACA-3.3.0.X 技术报告（mcBLAS 等算子库）：metax-tech.com/ndetail/12532.html
- 曦云 C500 规格（480 TOPS INT8 / 1.8TB/s / 无 FP8，C600 引入 FP8）：公开规格汇总（腾讯云/网易/IPO 资料）
- BI-V150 规格（384 TOPS INT8 / 192 TFLOPS FP16 / 1.2TB/s / FP8 需 ixTE）：mirrorfrog.com/en/docs/cards/iluvatar/bi-v150；chips.techbuzzchina.com
- BI-V150 量化需 IGIE/ixRT 工具链：moark.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu
- W8A8 动态 int8 精度（1–3%）：arXiv:2411.02355；Red Hat "Understanding W8A8 INT8 LLM Quantization"（developers.redhat.com, 2026-09）
- weight-only int8 近无损：OmniQuant（arXiv:2211.10438）及其引用；vllm-project/llm-compressor issue #1394（小模型 W8A8 精度恢复）
- 沐曦×FlagOS/FlagGems 合作（FlagGems 覆盖量化线性算子）：metax-tech.com/ndetail/12639.html
