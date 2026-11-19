# flagos_xx 双平台优化报告（s2 交付版 · 终局数字版）

> 归档自 `submit_pkg/report.md`（2026-11-20）。沐羲侧官方 4 轮丢首数字已落盘，
> 替换原 quick 口径占位。原始 raw CSV：`results/20261119_d4_metax_port/final/`。

## 总览

MiniCPM5-2B · vLLM 0.24 + vllm-plugin-FL 插件 · 官方双场景（4K/16K）·
天数智芯 BI-V150 + 沐曦 C500 双平台 · 官方 bench 丢首均值口径 · 105 题精度门 ≥0.95

## 终局数字

| 平台 | 场景 | 纯净基线 tok/s | 终局 tok/s | 提升 | 精度 math_500 L3 |
|---|---|---|---|---|---|
| 天数 BI-V150 | 4K | 1879.68 | **3414.58** | **+81.7%** | **0.9714**（基线 0.962） |
| 天数 BI-V150 | 16K | 862.06 | **1570.62** | **+82.2%** | 同上 |
| 沐曦 C500 | 4K | 3786.30 | **6399.21**（4 轮丢首均值；轮值 6927.95/5883.03/6386.65） | **+69.0%** | **0.981**（105/105 全样本） |
| 沐曦 C500 | 16K | 5583.49 | **7509.87**（4 轮丢首均值；三热轮散布 0.12%） | **+34.5%** | 同上 |

沐曦侧 Mean TTFT 见 `results/20261119_d4_metax_port/final/raw_runs_*.csv`。

## 天数侧（BI-V150）工作构成

- D0 attention launch 调优：16-SM 饥饿机理，launch 表注入（w1s1 decode / w2s1 prefill）——4K +41.5%、16K +46.5%
- D1 调度 watermark=0.06
- D2 W8A8：`torch._int_mm`（cuBLAS int8）链 + 自研融合 GEMM（int8 dot→int32，epilogue 融合，位精确对齐）+ 逐 token quant kernel；decode 小 BM tile 档 1.16–1.62×；lm_head int8
- D3 注意力三件套：① TD（tensor descriptor）/ ⑤ prefill TILE 32→16 / ② 采样 grid——终局 B 臂 +20.3%/+16.7%（vs D2 基线）；105 题 0.9714 过线
- 权重显存 −1.84GB；KV 池 518K→582K tokens

## 沐曦侧（C500）工作构成

- D2 等价：mctlassEx W8A8（`w8a8_scaled_mm_azp`，int8×int8→bf16，per-token×per-channel scale）承载全链 int8 线性 + lm_head int8；[N,K] 零拷贝布局契约（unpack 天然产出，省 2.1GB→KV 池 1,205,696→1,257,392 tokens）
- 微基准（bf16 为 1×）：prefill M1024/M2048 1.52×/1.68×；lm_head M30/M64 3.42×/2.62×；decode M30/M64 0.79×/0.68×（由 prefill/lm_head 大赢找平，e2e decode 侧仍领先：Output tok/s ≈440 vs 328）
- ⑥ split-KV hop：flash_attn_with_kvcache 双绑定路由，kernel 零改动复用
- 沐曦 serve 官方形态（零附加 env、无 --compilation-config）成立，全部优化默认激活

## 数值与复现

- 量化配方：SmoothQuant α=0.5，clip=1.0，RTN，295 tensors 位精确 roundtrip；
  ckpt 由 `tools/quantize_int8_rtn.py` 幂等再生（--lm-head-int8）
- 沐曦 parity：M∈[1,2049] maxrel ~3e-3（int8 噪声量级）；cudagraph 捕获+replay 安全
- 天数 parity：bit-exact（int32 位精确 + epilogue 顺序冻结）

## 消融

- 沐曦官方形态 vs FULL_DECODE_ONLY（19 时 quick）：4K +9.9%、16K 持平——无 compilation-config 形态全面占优或持平
- 沐曦 4K 轮间散布 ~16%（prefill 图形态方差，均值仍 +69%）；16K 极稳（0.12%）
