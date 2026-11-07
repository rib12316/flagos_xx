# D2 Kernel 支持矩阵 — W8A8 动态 int8（vllm_fl × FlagGems，BI-V150 sm71）

> Phase 1 产出（2026-11-06，feat/d2-w8a8-quant 规划期逐行核验）。
> 后续 per-vendor 复用的基础文档；所有结论带 file:line，代码版本 = main@9012af6（与 site-packages 逐字节一致，已 diff 验证）。

## 0. 结论速览（定义 Phase 2 搜索空间）

| 配方 | kernel-ready | 说明 |
|---|---|---|
| **L0** per-channel 对称 RTN 权重 + per-token 对称动态激活 | **✅ 就绪** | 全链路端到端可用 |
| **L1a** 权重 scale clip 网格（α∈{1.0,0.999,0.99,0.98,0.95}） | **✅ 就绪** | 纯离线（只改转换时的 absmax 截断），运行时结构不变 |
| **L2** SmoothQuant-lite（激活离群迁移进权重/norm，离线折叠） | **✅ 就绪** | 三处折叠点数学精确（见 §4），运行时结构不变 |
| **L4** keep-list 混合精度 | **✅ 就绪** | config ignore 列表 + 该层权重保 bf16 |
| L1b 激活 percentile 截断 | ❌ 需 dispatch 层小改 | 动态量化 absmax/127 **硬编码**（§2），需把 clip 参数穿透 apply_weights→quant op（估 2-4h + 重装验证）；被 L2 实质覆盖，仅兜底 |
| L3 per-group 权重（g=128/64） | ❌ 不可行（本时间盒） | `int8_mode.py:68-71` 非 channel 直接 ValueError；FlagGems block-wise smm kernel sm≥90 门（`fused/cutlass_scaled_mm.py:407`）。需新 scheme + 新 Triton group-scale kernel，估 >4h → 记入报告「下一层假设」 |

**搜索空间 = {L0, L1a, L2, L4} 自由组合**——全部离线侧改动，运行时 kernel 路径一字不变。

## 1. 权重量化参数支持

| 能力 | 支持 | 证据 |
|---|---|---|
| per-channel scale `[N]` | ✅ 唯一就绪路径 | `linear.py:75-76`（scale numel 必须等于 out 通道数）；`linear.py:84-91` 展平 fp32 |
| per-group `[N, K/g]` | ❌ dispatch 即 raise | `int8_mode.py:68-71`；is_packed_int8_weight 元数据虽匹配 GROUP（`int8_mode.py:38-39`）但选择策略硬否 |
| 非对称（zero point） | ❌ | `int8_mode.py:36` 要求 symmetric；无 weight_zero_point 参数（`linear.py:92-94` 显式 None） |
| bias | ✅ | `scaled_mm.py:252-257` |

**scale 应用公式**（输出侧、fp32 后累加）：Triton kernel 内 `acc * scale_a[:,None] * scale_b[None,:]`（FlagGems `ops/scaled_mm.py:128`）；参考实现同构（`reference.py:100-102`）。

**打包格式**（compressed-tensors pack-quantized）：
- int8 值 offset-binary：`(q.int16+128).to(uint8)`，4 个沿 dim1（K 维）塞一个 int32，小端低位在前 → `weight_packed int32 [N, K/4]`（`packed.py:78-89`，packed_factor=4）
- 解包互逆已 bit-exact 验证：`unpack_uint8b128_int32`（`reference.py:20-33`：view uint8 → 取前 K 列 → int16 − 128）
- checkpoint 每SmartyHeaderCode层三键：`weight_packed` int32 [N,K/4]、`weight_scale` **fp32 [N,1]**（加载断言严格，`parameter.py:153,175,229`；scheme 建 scale 为 `(sum(out_shards),1)` `compressed_tensors_w8a8_int8.py:71-75`）、`weight_shape` int64 [2] = **[K,N]**

## 2. 激活量化

- **per-token 动态、对称、scale [M,1] fp32**。公式（`reference.py:49-68`；运行时 Triton 同构 `vllm_fl/dispatch/backends/flaggems/impl/quantization.py:33-53`）：
  `absmax=行abs最大; scale=absmax/127; xq=round_half_even(x*127/absmax).clamp(-128,127)`；absmax==0 行 scale=0 守卫
- **截断硬编码**：无任何 clip 参数可从 config 穿透（`can_implement` 要求 input_symmetric，`linear.py:65-66`）→ L1b 不就绪的直接原因
- 三级 dispatch：CachedOp("dynamic_per_token_quant_int8")（`linear.py:34`）→ (1) flaggems_vllm.ops.scaled_int8_quant (2) in-tree Triton（round-half-even，:16-23）(3) PyTorch 参考

## 3. INT8 GEMM dispatch（sm71 实际路径）

`linear.py:96-116 apply_weights` → `flag_gems scaled_mm`（`ops/scaled_mm.py:425`）→ 依次尝试：
1. **cutlass 快路径**：`_can_use_cutlass_scaled_mm` 门在 `scaled_mm.py:292-294`（`90 ≤ sm < 100`）→ **sm71 必不触发**（另有 stride 双重门 :306-307）
2. **Triton 通用 int8 kernel**：`scaled_mm_kernel`（:51-136），int8 输入、`ACC_DTYPE=tl.int32`（:366）、libtuner 自动调优 → **BI-V150 实际执行路径**
3. Ascend 变体仅 vendor==ascend 触发（:315-317）

**_iluvatar/matmul_int8.py 证伪结论**：确证 bf16 降级占位——`get_output_dtype` 硬编码 bf16（:33-34），`tl.dot` 前 cast bf16（:118）；但它**不在 w8a8 调用链上**（w8a8 走专用 int8 scaled_mm_kernel，int32 累加）。若任何路径把 int8 matmul 路由到它，int8 精度会被静默丢给 bf16——Phase 3 微基准设证伪臂专门盯这一点。

**能力门（本会话新发现，D2 第一阻断）**：vLLM 侧 `compressed_tensors.py:860` `_check_scheme_supported(scheme.get_min_capability())`，`CompressedTensorsW8A8Int8.get_min_capability()=75`（"turing and up"），本卡 `get_device_capability()==(7,1)`→71 → **packed checkpoint 加载即 RuntimeError**。FL kernel 本身无能力门（`linear.py:44-54`）。修复：`packed.py` 的 `CompressedTensorsPackedW8A8Int8` override `get_min_capability()=71`（本分支落地 + 单测）。

## 4. SmoothQuant-lite 折叠点数学（逐行核验，全部精确）

| 站点 | 折叠方式 | 精确性证据 |
|---|---|---|
| q/k/v ← input_layernorm | `norm.weight *= 1/s`；q/k/v 权重列 `*= s` | RMSNorm 的 w 乘在 sqrt 外（`layernorm.py:37-94`），w/=s ⟺ 输出×1/s；residual 流不受影响（fused_add_rms_norm 返回的 residual 未过 w） |
| gate/up ← post_attention_layernorm | 同上 | 同上 |
| o_proj ← v_proj | o 权重列 `*= s`；**v_proj 行 `*= 1/s`** | RoPE 只作用 q/k（`llama.py:231`），attention 对 V 线性 → 输出通道 j 精确缩放。**GQA 陷阱：16 Q 头共享 2 KV 头，s 必须按 8 头一组聚合（2048→256 再 repeat 扩张）** |
| down_proj ← up_proj | down 权重列 `*= s`；up_proj 行 `*= 1/s` | SiluAndMul = silu(gate 半)×up 半（`activation.py:118-126`），对 up 分支线性 |
| model.norm | **永不折** | lm_head 不量化，折了白损精度 |

被折叠方自身的量化不受行缩放伤害（per-channel [N] scale 按输出通道，行缩放被 scale 吸收）。keep-list 站点跳过折叠（数学上无意义且省一次 bf16 舍入）。

## 5. config 驱动激活（should_use_packed_w8a8）

判定链：config.json `quantization_config` → vLLM CompressedTensorsConfig → monkeypatch 的 `_get_scheme_from_parts`（`packed.py:110-147`，marker `_vllm_fl_packed_w8a8_v024`）→ `should_use_packed_w8a8`（`int8_mode.py:56-72`）。精确 schema：

```json
{
  "quant_method": "compressed-tensors",
  "format": "pack-quantized",
  "config_groups": {"group_0": {
    "weights":           {"num_bits": 8, "type": "int", "symmetric": true, "dynamic": false, "strategy": "channel"},
    "input_activations": {"num_bits": 8, "type": "int", "symmetric": true, "dynamic": true,  "strategy": "token"},
    "targets": ["Linear"]}},
  "ignore": ["<keep-list 运行时融合模块名>"]
}
```

- input_activations **必填**（`int8_mode.py:64-65`，无 weight-only 模式）
- `targets:["Linear"]` 按模块**类名子串**匹配（`compressed_tensors/utils.py:154-162`）→ QKVParallelLinear/MergedColumnParallelLinear/RowParallelLinear 全中；ParallelLMHead/VocabParallelEmbedding 不中 → **embed/lm_head 自动保 bf16，无需 ignore**
- ignore 粒度必须整融合模块（gate ignored 而 up not → ValueError，`utils.py:50-97`）
- 运行时融合 vs checkpoint 分离：运行时 qkv_proj [2560,2048]（q/k/v = 0:2048/2048:2304/2304:2560）、gate_up_proj [12288,2048]（gate/up = 0:6144/6144:12288），片宽读 `layer.logical_widths`；**转换脚本按分离命名输出，loader 自动融合**（`linear.py:708-742`）

## 6. 环境事实

- `torch.cuda.get_device_capability(0) == (7,1)`（实测）；iluvatar 上报 device_type=cuda（`vllm_fl/utils.py:41`）→ 真实 capability 生效
- 屋顶线常数：384 TOPS INT8 / ~1.2TB/s HBM（docs/track-b-quantization.md §1.2）
- FP8 两卡均无；cutlass int8 sm90 门 → 均排除，勿再评估
- B 线遗留数据（T1/T2/T4）不存在于本机（/workspace/verify-b、research/results 均无）→ Phase 3 全新跑
