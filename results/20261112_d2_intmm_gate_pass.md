# D2 硬件闸门判定书（二）— _int_mm 路线 **✅ 过线**

> 2026-11-12，feat/d2-w8a8-quant。数据：`results/20261112_d2_intmm_gemmbench/bench.json`
> （脚本 `scripts/d2/bench_int8_gemm.py --m-extra 8192`，GPU 独占）。前判：`20261112_d2_hardware_gate_verdict.md`
> （FlagGems Triton 路径 15 形状全败 0.10-0.90×，本篇为其后续）。

## 判定：✅ 过线（graph 口径）

**down 6144×2048 M=1024 = 1.604×（≥1.5 判据）**；M=64 档部署形状 gate_up = 1.753× / down = 1.410× /
qkv = 1.308×（均 ≥1.2 判据）。判据：任一 M=1024 ≥1.5× 或 M=64 ≥1.2×（int8 对 bf16）。

### 加速比矩阵（ratio_graph = A/D 两臂 CUDA graph 捕获重放同法计时）

| (K,N) | M=64 | M=128 | M=1024 | M=8192 |
|---|---|---|---|---|
| qkv 2048×2560 | 1.31 | 0.99 | 1.35 | 1.10 |
| o 2048×2048 | 0.99 | 0.78 | 1.28 | 3.68 |
| gate_up 2048×12288 | **1.75** | 1.21 | 1.22 | 1.46 |
| down 6144×2048 | **1.41** | 1.28 | **1.60** | 1.12 |
| lm_head 参照 2048×130560 | 1.82 | 1.11 | 0.96 | 1.50 |

（lm_head 实际部署为 bf16，仅参照。vs FlagGems eager pair 最快 4.97×。）

## 路径（三层修复，全部已提交）

1. `torch._int_mm`（cuBLAS int8, int32 累加）+ M<17 垫零 + 融合 epilogue 单 kernel
   （裸 triton.jit，int32×xs×ws→bf16 一遍，与 emu 位对齐）——dispatch op
   `w8a8_dynamic_per_token_linear`，intmm(prio160) > flag_gems scaled_mm(150) > reference。
2. 裸 triton.jit 逐 token 量化 kernel（`default.flagos_bare` prio165，无 libentry），
   **位精确复刻 torch reference**（含标量除法乘倒数语义：`t/127 ≡ t·fp32(1/127)`、
   `127/t ≡ reciprocal(t)·127`；用 `tl.fdiv(..., ieee_rounding=True)` 显式构造）。
3. 外部 `flaggems_vllm` 可用性预探测——根治「包缺失只在调用时暴露 → CachedOp 异常路径
   → 永久坠落 mgr.call 慢路径（~0.6ms/调用）」。

## 测量方法论（eager vs graph，重要）

- **eager 成对数字不可用于本机判据**：三算子链 host 开销 ~200µs/iter（aten::_int_mm host
  105µs + 3×aten::empty ~80µs + 2×triton launch ~25µs），且轮间全量 drain 后 GPU 降频使
  后续轮劣化 2-3×（首轮 0.30ms → 后续 0.77ms，逐迭代直方图证实中位 0.22ms + 周期 56ms
  尖峰）。bf16 单 kernel 臂不受降频影响 → eager 成对比较系统性失真。
- **graph 口径**：两臂同法捕获重放，消除 host 开销与降频伪影；即 serve decode
  （FULL_DECODE_ONLY cudagraph）的真实执行形态；捕获输出与 eager 逐位对拍（脚本内置断言）。
- GPU 侧纯 kernel 时间（profiler 实测 gate_up M=64）：GEMM 59.3µs + epilogue 10.9µs +
  quant 7.7µs = 77.9µs vs bf16 ~126µs。
- 已知异常：bf16 lm_head M=64/128 graph 2.08/2.00ms vs eager 0.97/1.06（捕获期 cuBLAS
  选了不同 algo）——lm_head 不部署 int8，不影响判据；台账记录在案。

## 后续

serve 全链路验证（run_d2_eval.sh 冒烟 → fake-vs-real 30 题对拍 → 官方双场景 ≥3 中位
vs 4K 2668.42 / 16K 1276.62 → 105 题精度）。serve prefill 为 eager（FULL_DECODE_ONLY），
int8 链 host 开销在整模型 prefill 中占比小于微基准，以官方 bench 为准。
