# D7 T0 判决 — prefill 自研 kernel 手写路线的平台证据（feat/d7-prefill-attn）

- 日期：2026-11-20 04:10 · B 机 · `scripts/d7/prefill_gqa_t0.py` + `scripts/d7/gemm_calib.py`
- 设计：GQA packing（BLOCK_M 行 = tokens×8 heads 共享 KV，流 ÷8）+ varlen paged
  + online softmax 单 kernel；数值 maxrel vs flash = 5.3e-3（bf16 噪声量级）✓

## 判决：手写 triton 路线判死（平台铁证）

| 证据 | 数字 |
|---|---|
| 我的 kernel（BM128/BN32/w8/s2，S16q2048kv16K） | **57 TFLOPS**（0.59× vs flash 96T） |
| 配置扫描（16 组：BM 32-256 × BN 16-128 × w × s） | 全部 ≤57T；BN≥64 全部 smem OutOfResources（64KB 硬限） |
| 效率 vs kv 长度（2K→16K） | 恒 55-57T（与迭代深度无关 → 非延迟问题） |
| 顺序 vs 乱序页表 | 77210 vs 77211 µs（无差 → 非散射问题） |
| **纯 GEMM 循环校准（无 attention 开销，数据驻留）** | **29-56 TFLOPS**（最好 BM256/BN64/BK64=56T） |

**结论：沐曦 triton 3.0 的 dot 后端本征峰值 ≈56T——我的 attention kernel 已打平
编译器上限。MACA flash（mctlass 原生）的 96T 与满速 224T 在 triton 路线不可达。**

## D7 剩余路线（全部外部依赖，需用户/A 机决策）

1. **Kernel-Smith 在线**（chat.intern-ai.org.cn/kernel-smith，已验证可达）：
   MXMACA 原生生成，需**用户人工在 Web 端发起**（agent 无 API）——战绩 SGLang
   normal_decode 4.78×；产出过本仓 T0 gate。
2. **kernelgen MCP**（kernelgen.flagos.io）：需用户取 token 配置。
3. mctlass 原生 C++：超今日工程范围。

## 附：GEMM 校准明细（scripts/d7/gemm_calib.py，104 CTA×1024 iter）

BM128/BN32/BK128/w8/s2=29T；BM128/BN64/BK128/w8/s2=29T；BM64/BN64/BK128/w4/s2=23T；
BM128/BN128/BK64/w8/s1=49T；**BM256/BN64/BK64/w8/s1=56T**；BN256 s1=6T（崩）。
（对照：同卡 mctlass GEMM w8a8 链实测 224T 量级、MACA flash prefill 96-224T。）

PROVENANCE：B 机 C500 · flash_attn 2.6.3+metax · triton 3.0.0+metax3.7.0.7 ·
graph_time_us 口径 · GPU 独占。
