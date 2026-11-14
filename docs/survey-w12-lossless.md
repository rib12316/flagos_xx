# W12 无损 12-bit 权重格式调研（SpecPilot / arXiv 2611.02501）

> 调研日期 2026-11-14。范围：W12 编码与内核细节（论文全文 + 仓库源码级阅读）、Triton 移植可行性、无损/近无损权重压缩全景、按本队负载画像的收益模型与决策建议。
> 信息源：arXiv 2611.02501v1（HTML 全文）、github.com/StargazerX0/SpecPilot@main（codeload tarball，逐行读了 h100 setting 的 `specpilot_d5_w12.py` 1367 行全文、sm_120 的 d2/d7 变体、unquant.py/qwen2_moe.py 集成补丁）、NF12/DFloat11/ZipNN 等检索。

---

## 0. TL;DR

1. **W12 是真无损**：不是营销词。原理 = 每个输出通道（权重行）内指数动态范围 ≤16 binade，把 bf16 的 8 位指数换成"行内 4 位指数偏移 + 行级公共指数"，12 bit 可逐位还原原始 bf16（仓库在加载时逐 tensor `torch.equal` 强校验，不精确直接抛异常）。测得 8 个稠密模型全部落在 **12.013–12.020 bit/weight**，escape 率仅 0.7–1.2×10⁻⁴。
2. **单 W12 加速 1.19–1.28×（batch=1）/ 1.13–1.20×（batch=32）**，但那是 **8–30B 模型、权重流主导 decode** 的场景。我们 2B 模型 4K 场景权重只占步时 29%，理论上限 **+7.8%**，现实 **+4~7%**；16K 场景（权重占 9%）**+1.5~2.5%**。W12 的真实价值更多在**显存**：Linear 3.7GB→2.78GB，**腾出 ~0.9GB 给 KV**。
3. **Triton 移植可行、工程量中等（~40–60 人时）**：解码只需 shift/and/or + bitcast（无 LOP3 依赖），vLLM awq_triton、纯 Triton W4A16 等先例证明该模式在非 CUDA 后端可用；瓶颈风险在厂商 Triton 分支的 bitcast/整数路径质量与 FlagGems GEMM 效率，需 1 天硬件验证 spike。
4. **同类里没有"更好且可直接用"的**：NF12（Together AI，NeurIPS 2026）是并行同款工作（12.02 bpw，解码更便宜），但内核只有 CUDA + AMD ROCm，无 Triton 实现，BI-V150/C500 用不上其内核；DFloat11 无损但 GEMM 慢（比 cuBLAS 慢 4.8–6.7×）；ZipNN 只省磁盘不省显存/带宽。仓库里还有 **W11 变体（11 bpw 仍逐位无损）**，是免费的一档备选。
5. **决策位置**：作为 Plan B，W12 在"速度收益"上被 **W8A16（weight-only int8 g=64/128）** 支配（同样 −50% 字节、近无损历史记录、Triton 解包更简单）；W12 的独特卖点是**逐位精确 = 零精度风险 + 免评测周期**。建议：W8A8 红线不过 → 先试 W8A16；W8A16 也不过或要求绝对零风险 → W12。

---

## 1. W12 技术细节

### 1.1 编码：为什么 12 bit 能逐位精确

（来源：论文 §4.1/Fig.5a/Appendix E Table A2；源码 `specpilot_d5_w12.py` 模块 docstring 与 `_encode_rows()`）

- 设某输出通道（W 的一行 n）内权重 w = s·2^(e−127)·1.m（bf16：1 符号 + 8 指数 + 7 尾数）。**训练好的稠密模型每行权重的指数跨度实际 ≤16 个 binade**（动态范围 < 2¹⁶，通常只有 2⁸~2¹²）。
- 对每行选 16-binade 窗口 `[E_n−15, E_n]`（直方图贪心，覆盖最多权重，平局取最高窗口）。窗内权重的指数编码为 4 位偏移 `c = e − (E_n−15) ∈ [0,15]`。
- 解码值 w̃ = s·2^(c−15)·1.m 的 bf16 位型固定为 **`[s 0111 c3c2c1 | c0 m6..m0]`**——高字节的指数高 4 位恒为 `0111`（指数域 112–127），完全可预测，**这 4 位就是被省掉的 4 bit**。
- 每权重存 12 bit，**两个平面**（不是任务假设的"三个 4-bit 平面"，实际是 1 字节 + 1 半字节）：
  - **Q 平面（1 B/权重）**：`[c0 m6 m5 m4 | s c3 c2 c1]`
  - **B 平面（0.5 B/权重，nibble）**：`[m3 m2 m1 m0]`
  - Q 单独读出就是嵌套的 8-bit"e4m3-like"近似视图（给投机采样两级验证用，本队用不到）；Q|B 合并才是精确值。
- 还原：`lo = (Q & 0xF0) | B_nibble`，`hi = ((Q<<4) & 0x80) | (Q & 0x07) | 0x38`，`(hi<<8)|lo` bitcast 回 bf16 得 w̃；**真实 w = w̃ · 2^(E_n−127)**，即 GEMM 对整行共用一个 2 的幂行缩放，epilogue 对 fp32 累加器乘 `scale[n] = 2^(E_n−127)`。乘 2 的幂是纯指数平移，**逐位无损**。
- **escape 机制**：窗口外权重（离群指数）、零、denormal（escape 率 0.69–1.21×10⁻⁴，8 模型实测）存为占位 w̃=2⁻¹⁵，由每行 CSR 稀疏表精确校正：`acc += x·w̃_true − x·2⁻¹⁵`（两个乘积都在 fp32 精确）。连带 CSR/尺度等元数据，总密度 **12.013–12.020 bpw**（Table A2：Llama-3.1-8B 12.015，Qwen3.8-27B 12.014…），比 1.5 B/w 只多 0.01–0.02 bit。
- bit-exact 是被强制的：`d5_encode_target()` 编码后 `_check_roundtrip` 用 `torch.equal(dec, w)` 逐层校验，不通过抛 `RuntimeError`。论文亦声明服务器加载时逐 tensor 与 checkpoint 逐位比对。

**为何"无损"成立（信息论角度）**：任意 bf16 串 16 bit 不能塞进 12 bit；但按行利用指数相关性（行内动态范围 ≤ 2¹⁶ ⇒ 4 位指数偏移足够）+ 2 的幂缩放的精确性，就把"每权重信息量"压到了 ~12 bit。这是**针对稠密 LLM 权重分布的结构性无损**，不是通用压缩。

### 1.2 Kernel 设计（H100/sm_90a 版，源码级）

`w12s_kernel`：persistent **stream-K** 内核，一 CTA/SM，处理展平的 (tile × k-stage) 区间，segment 顺序旋转。分工：
- **1 producer warp**：权重记录用 1-D `cp.async.bulk`（bulk TMA，带 L2 cache hint）；激活 x 用 2-D TMA tensor map（128B swizzle），5 级 mbarrier 流水。
- **2 consumer warpgroup（8 warp）**：从 smem `lds` 平面 → 寄存器解码 → **wgmma m64nNTk16，A 操作数直接来自寄存器**（B=x 走 smem descriptor）。解码路径（`decode_n`）：每对权重约 `2 AND + 2 OR/移位 + 2 __byte_perm`，代码注释 **"~3.3 ops per register (was ~7)"**；旧版两平面路径 docstring 写明 **"2 byte_perm + 2 LOP3 per weight pair"**（论文正文未提 LOP3，LOP3 在代码里以 `(n*17)&0x87|0x38` 这类可被编译器融合成 LOP3 的位逻辑出现，**不是硬依赖**——另有 mask/shift 形式的 `decode_v1` 留作低寄存器压力备选，证明纯移位+与+或即可）。
- **2 escape warp**：按 CSR 算稀疏校正 tile 进双缓冲 smem，消费者在 epilogue 前加回。
- **确定性**：split-K 贡献者以 `red.release` 发布 fp32 部分和，k=0 段 owner 按固定段序求和 → 固定 P 下逐位确定（这是与参考 bf16 GEMM 逐位一致的必要条件）。
- **融合 epilogue**：gate_up 走 `[64 gate | 64 up]` tile + 融合 SiLU(gate)·up（操作数先 bf16 舍入以对齐参考路径）。
- **PDL**（programmatic dependent launch）：权重平面首批 stage 在 `griddepcontrol.wait` 之前预取，与上游内核尾部重叠。
- **M 分档**：M≤64 走 W12 内核（NT∈{16,32,48,64}）；64<M≤128 可单发射 NT=128（`w12q_kernel`，权重只流一次）；M>128（真 prefill）**解码进共享 bf16 scratch 再走 cuBLAS**（`w12_decode_kernel`：位组装 + `__vadd2` 按半字加回指数偏移）。
- sm_120 版（d2）：同格式，mma.sync m16n8k16 + swap-AB（权重做 M=16 侧，`ldmatrix.x4`），且提供子开关扩展覆盖 o_proj/GDN out_proj/lm_head/prefill 切片。

**内存布局**：每 (128 行 tile × 64 k chunk) 一条 12 KB record = Q 区 [8 warp][1 KB] + B 区 [8 warp][512 B]，**按 wgmma A-fragment lane 顺序预排**（编码时用纯 torch permute 完成，内核免换序）。平面内 K 连续，TMA bulk 拉取天然合并。

### 1.3 实测数据（论文 §6.3/Fig.9/Appendix C,E）

| 指标 | 数值 |
|---|---|
| 单 W12、AR decode、SGLang、batch=1 | **1.19–1.28×**（Muse-Glimmer-30B / Qwen3.8-27B / Gemma-4-12B / Llama-3.1-8B；TPOT 例：Qwen3.8-27B 19.1→15.6 ms） |
| batch=32 | **1.13–1.20×**（23.6→20.5 ms）→ batch 增大衰减温和（这些模型下权重流始终主导） |
| vs cuBLAS bf16（Qwen3.8-27B 各投影，H100，M=1–32 行） | **1.17–1.47×**；warm-L2 1.08–1.45×；lm_head 收益最大（1.35–1.47×） |
| vs cuBLAS（RTX PRO 6000/sm_120） | 全部 13 个形状 **1.18–1.68×**（≤32 行）；48/64 行仍有 1.02–1.68× |
| 显存 | 权重内存降 **14–22%**（整模型口径，含未编码的 embedding 等）；KV pool 增 2.0×/2.5×（27B/30B 单卡，权重占比高所致） |
| 密度 | 12.013–12.020 bpw，escape 率 ≤1.3×10⁻⁴（最小模型 Llama-3.1-8B，**未测 2B 级**） |
| 在完整 SpecPilot 栈中 | 两轮晋升分别 +7.6%/+6.3%（含内核重调） |

注意：README 口径"1.15–1.26×（batch 1–32）"与论文正文"1.13–1.28×"略有出入，以论文为准。

### 1.4 与投机采样的关系

W12 压的是**目标模型**权重（另有 EAGLE-3 部署中压 MLP）；draft 模型走独立的 FP8/INT4 路径。W12 的嵌套 Q 平面（8-bit 近似视图，丢弃尾数的中点替代）服务于其两级验证（TLV）的粗筛 pass——**对本队无投机采样的场景，Q 平面只是顺带产物，不需要实现**。

### 1.5 工程量与耦合度

- 核心 = **一个 1367 行自包含 Python 文件**（`kernels/ops/gemm/specpilot_d5_w12.py`）：内嵌 CUDA 源（inline JIT 编译，~800 行：2 个 GEMM 内核 + 1 个 decode 内核 + host launcher）+ **纯 PyTorch 编码器**（`_encode_rows`，直方图选窗、位打包、CSR，~40 行，与硬件无关）+ dispatch。
- SGLang 集成面极薄：`layers/quantization/unquant.py` 约 4 行分支（`layer._d5_w12` 存在则走 `d5_linear`）；模型文件（qwen2_moe.py 等）每处 forward 约 3 行分支。环境变量 `SPECPILOT_D5_W12=1` 开关，unset 即原生路径。
- **可以剥离**：W12Weight 类不依赖 SGLang（仅 logger/env），把 CUDA 内核换成 Triton、去掉 wgmma lane 预排即可独立使用。约束：K%64==0（我们 K=2048 ✓）、bf16 无 bias、TP=1 ✓。

---

## 2. Triton 移植可行性（重点）

### 2.1 Triton 位操作能力：够用

解码所需原语：`>> << & |`（uint8/int32）、常数折叠、`bitcast` 到 bf16。全部是 Triton 核心语义：
- `tl.load` uint8/int32 → 整数位运算 → `(uint16).to(tl.bfloat16, bitcast=True)`，官方 `tl.dot` 支持 bf16 输入 + fp32 累加。
- 已知坑（要在 spike 里验证）：pytorch#143821 记录的 upcast-to-fp32 会悄悄重解释 bitcast 整数——对策是整数运算全程显式 int32/uint16，最后一步才 bitcast；openai/triton#7161 展示的 uint8 打包/移位模式是标准做法。
- W12 解码**不依赖 LOP3/PRMT**：SpecPilot 自己保留了 mask/shift 版 `decode_v1`；NF12 的解码同样是"register shifts, logic and byte permutations"（13 条寄存器指令/权重 + 3 条 escape 检查），纯基本位运算。

### 2.2 先例确认（任务问的 AWQ/Marlin Triton 移植线）

- **vLLM `awq_triton`**：`awq_dequantize_kernel` 内核内 `(packed >> shifts) & 0xF` 解包 + 组缩放，是 CUDA 不可用（AMD ROCm、CPU）时的可移植回退路径——**该模式在非 CUDA 后端跑通的直接证据**。
- **"Beating FP16 with 4-Bit Weights: A Portable W4A16 GEMM"**（subhadipmitra.com，2026-07）：纯 Triton W4A16，解包在 GEMM 内部，明确 NVIDIA+AMD 双端、无 CUDA 依赖，性能超 FP16。
- **triton_w4a16**（ROCm MI300，GPTQ uint4b8）；**OpenAI Triton 官方博客 "Accelerating Triton Dequantization Kernels for GPTQ"**（2024-01）= 官方背书的"dequant 融进 matmul"教程。
- 社区经验：NVIDIA 上无 Marlin 的 Triton GEMM 路径可能慢于 FP16（同字节下），但在无 Marlin 的硬件上它是唯一路径且普遍可用——**结论：内核内位解包在 Triton/非 CUDA 后端是成熟模式，风险不在"能不能"而在"效率折扣"**。
- FlagGems 本身即 Triton 算子库且官方列明 MetaX（沐曦）/Iluvatar CoreX（天数）后端，bf16 GEMM 路径已被这些卡踩过。

### 2.3 无 LOP3 的指令开销估算

- CUDA 版解码 ~1 条等价逻辑指令/权重（2 LOP3 + 2 PRMT per 权重对）；纯 shift/and/or 版（我们 Triton 移植）每权重约 **6–8 条整数指令**（2 AND + 2–3 OR/移位 + 组装 uint16 + bitcast）。
- decode（M≤64）是访存受限：每权重流 1.5 B。以 1.5 TB/s 带宽计 ≈ 1×10¹² 权重/s，需要 ~6–8×10¹² int op/s——H100 级 INT32 吞吐 ~2.8×10¹³，**留有余量；且这些指令填充的是访存等待空隙，不占 tensor core**。厂商卡 ALU:BW 比例同量级，预计开销 ≤5%，即 25% 字节节省净保留 ~20%+（对应 4K 步时 +4~7% 净收益）。**只有在厂商 Triton 编译器把整数链编得很差（如不向量化 nibble 处理）时才会吃掉收益**——这正是 spike 要测的。
- 结论：**不会吃掉节省**，但要按"权重段时间 ×0.75 ÷ 内核效率 δ"建模，δ 初版取 0.90–0.95。

### 2.4 移植方案草案

入手点：**FlagGems 的 bf16 GEMM（或 Triton 官方 matmul tutorial）做骨架**，把 B 操作数（权重）加载换成双平面加载+解码。

编码侧（直接复用 SpecPilot `_encode_rows`，纯 torch，改一处）：
- 保留：行直方图选窗 E_n、Q/B 打包、CSR escape、`scale[n]`。
- 修改：**去掉 wgmma A-fragment lane 预排**，Q 存 `[N,K]` uint8、B 存 `[N,K/2]` uint8（nibble），K 连续 → Triton `tl.load` 天然合并访存（平面各自连续，无双平面交错问题）。

内核伪代码（W12A16 GEMM，M≤64 decode 路径）：
```python
@triton.jit
def w12_gemm(x_ptr, q_ptr, b_ptr, scale_ptr, out_ptr, M, N, K, ...):
    # 常规 GEMM 循环: acc[BLOCK_M, BLOCK_N] fp32 = sum_k dot(x_tile, w_tile)
    offs_k = k0 + tl.arange(0, BLOCK_K)                    # K 连续
    q  = tl.load(q_ptr  + offs_n[:,None]*K + offs_k[None,:])          # uint8 [BN, BK]
    b8 = tl.load(b_ptr  + offs_n[:,None]*(K//2) + offs_k[None,:]//2)  # uint8
    b  = (b8 >> ((offs_k & 1) * 4)) & 0xF
    q32 = q.to(tl.int32)
    lo = (q32 & 0xF0) | b                                   # bf16 低字节
    hi = ((q32 << 4) & 0x80) | (q32 & 0x07) | 0x38          # [s|0111|c3c2c1]
    w16 = ((hi << 8) | lo).to(tl.uint16).to(tl.bfloat16, bitcast=True)
    acc = tl.dot(x_tile.to(tl.bfloat16), w16, acc)          # bf16 MMA, fp32 acc
    # epilogue: acc *= tl.load(scale_ptr + offs_n)[:,None]  # 行缩放, 2 的幂, 精确
    # (escape 校正单独小内核或 torch 端补: out += x[:,cols]·(vals−2⁻¹⁵) 按行 gather)
```
- **escape 校正**：率 ~10⁻⁴（K=2048 行均值 0.2 个/行），一个独立 Triton/torch 小内核把 `Σ x·(w̃_true−2⁻¹⁵)` 加到输出即可，纳秒级成本；不追求与 cuBLAS 逐位一致，只要自洽确定（Triton 编译产物归约序固定 → 逐次运行可复现）。
- **prefill（M>128）**：学 SpecPilot，解码进 bf16 scratch（一个 ~34MB 的层缓冲）→ FlagGems bf16 GEMM，与基线打平；M≤128 也可直接走 W12 内核（论文测过比"解码+cuBLAS"快 ~17%）。
- **输出 dtype**：bf16 输出、fp32 累加（与主线一致；SiLU·up 融合可选）。
- autotune BLOCK_M/N/K + num_stages 按 BI-V150/C500 实测。

### 2.5 工作量与风险

| 项 | 估时 |
|---|---|
| 编码器移植（torch，近乎照抄）+ MiniCPM5 上量 escape 率/bpw | 4–8 h |
| Triton W12 GEMM（M≤64 + M≤128）+ autotune | 16–24 h |
| escape 校正 + prefill scratch 路径 | 6–8 h |
| 集成（替换目标 Linear forward、CUDA graph 兼容、位精确单测 vs bf16） | 8–16 h |
| **合计（一名熟手）** | **≈ 40–60 h** |

Day-0 验证 spike（1 天）：① 编码器跑 MiniCPM5-2B，确认 bpw ≤12.05、escape 率 <2×10⁻⁴（论文未测 2B 级，唯一数据空白）；② 厂商 Triton 3.2 分支上 bitcast int16→bf16 + 整数链是否正确/向量化；③ 2K×2K 小 GEMM 对照 bf16 逐位验证 w̃·scale 路径。

---

## 3. 同类技术全景

### 3.1 无损/位精确类

| 技术 | 机制 | 对本队可用性结论 |
|---|---|---|
| **W12**（SpecPilot，arXiv 2611.02501，2026-11） | 行内 16-binade 窗口 + 4 位指数偏移 + 7 位尾数 + CSR escape；寄存器解码进 wgmma | 本报告主体。CUDA/sm_90/sm_120 实现，**无 Triton 版**，需自移植 |
| **NF12 / NyoomFloat12**（Together AI，NeurIPS 2026，arXiv 2610.*） | per-matrix 2 的幂缩放把指数移进 [112,127]（>99.7% 权重高指数位=0111），保低字节 + 权重对共享 meta byte；8 权重=12 字节；escape 组（meta=0）转 per-matrix 原值缓冲。12.02 bpw | **并行同款工作，非 W12 前作**。解码固定长、13 寄存器指令，GEMV 1.26× vs 定制 bf16、1.35× vs cuBLAS（70B/H100）；内核覆盖 Ampere→Blackwell + **AMD ROCm RDNA4**，并入 GPTQModel。**无 Triton/国产后端实现，内核不可直接用**；其 per-matrix 缩放编码思想比 W12 的 per-row 更简单（单标量缩放），可作我们编码方案的简化参考 |
| **DFloat11**（arXiv 2504.11651，NeurIPS 2025） | LZ77+Huffman 动态长浮点，~11 bpw 无损，LLaMA.cpp 系 | **先例**（证明 <16 bit 无损可行），但变长解码慢：SpecPilot 测其 GEMM 比_cuBLAS 慢 4.8–6.7×，NF12 测其解码仅 27% 峰值带宽。**对本队无用**（省显存不省带宽，且慢） |
| **ZipNN**（arXiv 2411.05239，IBM） | bf16 字节级无损压缩（~33% 文件缩减），load 时解压 | **确认：只省存储/下载/加载，不省运行时显存与 GEMM 带宽**。排除 |
| **ZipServ** | GPU 上 GEMM 前即时解压，开销 ~2–4% | 变长/流式解压路线，比 W12/NF12 的定长寄存器解码弱；无国产实现。排除 |
| **W11**（SpecPilot 仓库内，pro6000 setting `specpilot_d7_w11.py`） | 行内 7-binade 窗口（3 位指数码）+ 8 位平面 `[s m6..m0]` + 顺序 escape 侧流（每 escape 1 字节）→ **11 bpw 仍逐位无损** | 仓库自带的进一步压缩点（−31% 字节）；escape 率更高、解码需 popc+shfl scan，工程量大于 W12。作可选加档 |

**首创性结论**：定长无损 ≤12-bit + **GEMM 内寄存器解码**这个组合，W12 与 NF12 是 **2026 年下半年的一对并行首发**（NF12 alphaxiv 2610 略早，SpecPilot 投稿 11-02）；"无损 <16-bit 权重"本身可追溯到 DFloat11（2025-04，但 GEMM 慢）。**没有任何项目已发布 Triton/国产 GPU 版"无损位平面 GEMM"，没有现成可直接用的**——要用就得自己移植（好在 W12 的编码器 + 论文 + 仓库公开，Apache 2.0）。

### 3.2 近无损类（W12 与 W8A8 之间的参考档）

- **W8A16（weight-only int8，per-group g=64/128，非对称）**：社区/文献共识 **近无损**——Llama 家族 ppl 降幅典型 0.01–0.1（g128/group-wise），per-channel 也在 0.5 以内；AWQ/GPTQ 的主要战场在 4-bit，int8 组内通常被当作无损上界基线。2B 模型敏感度略高于 7B+，g=64 更稳。**Triton 解包比 W12 还简单**（单平面 `(q−z)·s`，无位平面组装）。−50% 字节。
- **W8A8（本队主案）**：SmoothQuant 系处理激活离群后有损但可控；红线余量 1.2pt 已实测。int8 tensor core 两卡均支持（FlagOS 已发 W8A8-iluvatar 模型版可佐证）。
- **FP8（E4M3/E5M2）权重**：**确认不可用**——BI-V150/C500 的 FP8 硬件支持不保证（BI 系"部分新卡支持 FP8"，C500 未确认），且 Triton 3.2 系厂商后端 FP8 路径不成熟。排除。
- **任意位宽（10-bit 等）定制**：MX/微缩放标准（OCP MX，arXiv 2310.10537）有 FP6/FP4 定长先例，但都是**有损**；无损的任意位宽只有 W12（12）/W11（11）/DFloat11（~11 变长）这几个点。10-bit 无损没有现成先例（窗口 4-binade 太窄，escape 率会爆炸）。

---

## 4. 收益模型（按本队画像）

### 4.1 前提数字

- 可优化 Linear ≈ 3.7 GB（bf16，≈1.8 B 参数：42 层 × (qkv+o+gate_up+down ≈ 42.8M)）→ W12 后 **2.78 GB**，**省 0.90 GB**（+scale/CSR ~5 MB）；总权重 4.5→3.6 GB。
- 若加码 lm_head（d2_HEAD 同款，vocab 10–15 万 × 2048 ≈ 0.4–0.6 GB）与 o_proj，可再省 ~0.1–0.5 GB 且 decode 头部 GEMM 同样 −25% 流量（论文中 lm_head 恰是收益最大的 GEMM，1.35–1.47×）。
- 解码步时构成（实测）：4K 权重 GEMM 流量占 ~29%，KV 读 ~71%；16K 权重 ~9%。prefill 计算受限。

### 4.2 decode 步时（访存受限，权重段按字节线性）

步时变化 = 1 − w·(1 − 0.75/δ)，w=权重占比，δ=Triton 移植内核相对原生 bf16 GEMM 的效率。

| 场景 | w | δ=1.0 上限 | δ=0.95 | δ=0.90（保守） |
|---|---|---|---|---|
| 4K | 29% | **+7.8%**（1/(1−0.0725)） | +6.6% | +5.4% |
| 16K | 9% | +2.3% | +2.0% | +1.6% |

叠加解码整数开销后给出区间：**4K +4~7%，16K +1.5~2.5%**。batch 增大时 KV/激活占比升高，权重份额下降，收益向 16K 一行收敛（SpecPilot 8–30B 下 batch 1→32 衰减仅 ~0.06–0.08 绝对值，因为其权重份额高；我们 2B 衰减会更快）。**prefill ≈ 0**（走 scratch+普通 GEMM，解码开销摊在整批 token 上 <2%）。

作为对照：NF12 的受控实验（KV 配额锁死到 bf16 水平后吞吐跌到 0.41×）说明**这类格式的吞吐收益大头来自腾出的显存换 KV 并发，而非单纯带宽**——这与我们 16K 排队区场景的机制一致。

### 4.3 显存/KV 侧

- 腾出 **~0.90–1.0 GB**（含 lm_head 可到 ~1.3 GB，即任务假设的 ~1.1GB 量级）。
- KV/token = 42 层 × 2 × n_kv·d_h × dtype 字节。若 GQA 8×128 bf16：172 KB/token → +0.9GB ≈ **+5.2K token**；4×128：86 KB/token → +10.5K token；KV 已 FP8/INT8 则翻倍。
- 按满 16K 请求计：bf16 KV 下 +0.3~0.65 条；KV 量化或请求平均驻留 <16K（爬坡中）时 → **+1~2 条**（与任务预估一致，但依赖 KV dtype/GQA 配置，建议用真实 config 复算一遍）。另注意：论文"KV pool 增 2–2.5×"是 27–30B 单卡权重占大头的特例，我们 2B 上是线性加法不是翻倍。

### 4.4 与 W8A8/W8A16 的决策树

```
W8A8 评测 ≥0.95 红线且余量 ≥~1pt？
├─ 是 → 主案不变（decode 4K 理论 +17%，W12 不做，可选作为 KV 拥挤时的叠加项？否——格式互斥，不做）
└─ 否 →
   W8A16 (int8 weight-only, g=64 非对称, bf16 act) 一轮评测
   ├─ 过线（历史概率高）→ 用 W8A16：−50% 字节，decode 4K 理论 +17%，Triton 工作量 ≤ W12
   └─ 不过线 / 无评测预算 / 要求零风险 → W12 Triton 移植（40–60h）：
        bit-exact 零精度风险、免评测周期；4K +4~7%、16K +1.5~2.5%、KV +0.9GB
        （时间富余可上 W11：−31% 字节，仍无损，+8~12h 内核改造）
```

**W12 相对 W8A16 的唯一优势是"零风险 + 零评测成本"与工程确定性（不依赖量化误差分布），不是性能**。若 W8A8 在中期评测已显精度压力，W12 spike（1 天）可以提前启动作为保险，编码器与位精确测试与最终是否采用解耦。

---

## 5. 行动建议（若立项）

1. **Day-0 spike（~1 天，无需内核）**：用 SpecPilot 的 `_encode_rows`（可逐行照抄，Apache 2.0，注明出处）在 MiniCPM5-2B checkpoint 上测 bpw / escape 率 / 逐位 roundtrip；同时在 BI-V150 与 C500 的 Triton 3.2 上跑 bitcast+整数链最小 kernel。
2. **内核**：从 FlagGems bf16 GEMM 克隆改 B 操作数为 Q/B 双平面解码（伪代码见 §2.4），先做 M≤64 decode 路径 + scratch prefill 路径；escape 校正独立小内核。
3. **验证**：单测逐位 `w̃·scale == w`（全量 1.8B 权重 torch.equal，分钟级）；端到端与 bf16 基线比对 logits（应逐位一致或仅在累加序上有 1ulp 级差异——我们不追求与 cuBLAS 逐位一致，只要求自洽确定）。
4. **量测**：4K/16K 两负载的 TPOT/吞吐、以及腾出显存后的并发增益（16K 排队区 +1~2 条的实测确认）。

## 6. 来源

- 论文：arXiv 2611.02501v1（abs / html 全文提取 §4.1、§5.3、§6.3、Fig.9、Appendix C/E）
- 代码：github.com/StargazerX0/SpecPilot@main——`settings/h100_c1_qwen38-27b_dflash2/files/python/sglang/kernels/ops/gemm/specpilot_d5_w12.py`（全文精读）、`settings/pro6000_c1_muse-30b_dflash2/files/.../specpilot_d2_w12.py`（sm_120）、`.../specpilot_d7_w11.py`（W11）、`layers/quantization/unquant.py`、`srt/models/qwen2_moe.py`（集成面）
- NF12：alphaxiv.org/abs/2610.nyoomfloat12（OpenReview yYFj1E1mSZ 有人机验证，走 alphaxiv）；GPTQModel NF12 wiki；NeurIPS 2026 poster 153072
- DFloat11：arXiv 2504.11651，github.com/LeanModels/DFloat11
- ZipNN：arXiv 2411.05239
- Triton 先例：vLLM `awq_triton`（`awq_dequantize_kernel`）；subhadipmitra.com/blog/2026/w4a16-weight-only-gemm-triton；OpenAI Triton blog "Accelerating Triton Dequantization Kernels for GPTQ"（2024-01）；openai/triton#7161；pytorch/pytorch#143821（bitcast 坑）；triton_w4a16（ROCm）
- 生态：FlagGems/FlagOS 多后端（MetaX、Iluvatar CoreX 等）；沐曦 MXMACA（C500 64GB）；MX 标准 arXiv 2310.10537
