# D2 硬件闸门判定书 — sm71 Triton int8 scaled_mm（BI-V150）

> Phase 3 产出，2026-11-12，feat/d2-w8a8-quant。bench 原始数据：
> `results/20261107_d2_gemmbench/bench.json`（脚本 `scripts/d2/bench_int8_gemm.py`，
> GPU 独占，20 预热 + 3 轮×100/30 迭代中位，TF32 关）。
> **本结论 per-vendor：仅对天数 BI-V150 + FlagGems 5.3.5 Triton 路径成立；沐曦侧必须独立重测。**

## 判定：❌ 未过线（int8 全面慢于 bf16）

判据（任务书）：任一 M=1024 形状 ≥1.5× 或任一 M=64 形状 ≥1.2×（int8 对 bf16 加速比）。
实测：**15 形状全部 <1.0×**，M=1024 最佳 0.54×（即慢 1.85×），M=64 最佳 0.90×。

### 加速比矩阵（int8=quant+mm 成对 / bf16=F.linear；mm-only 见 bench.json）

| (K,N) | M=64 | M=128 | M=1024 |
|---|---|---|---|
| qkv 2048×2560 | 0.21 | 0.14 | 0.54 |
| o 2048×2048 | 0.17 | 0.10 | 0.34 |
| gate_up 2048×12288 | 0.53 | 0.47 | 0.45 |
| down 6144×2048 | 0.34 | 0.38 | 0.33 |
| lm_head 参照 2048×130560 | 0.90 | 0.63 | 0.50 |

### 屋顶线归因

- **M=64（decode 带宽界，1200 GB/s HBM）**：int8 mm-only 实测 42.6–253.7 GB/s（qkv 42.6、
  gate_up 160、lm_head 254）；对照 bf16 GEMM 在 lm_head 形状做到 ≈553 GB/s —— **Triton int8
  kernel 的带宽效率只有 bf16 cuBLAS 路径的一半以下**，权重字节减半的优势被效率完全吞掉。
- **M=1024（TOPS 界，384 TOPS INT8 标称）**：int8 实测 27.6–45.5 TOPS（峰值 12%）；bf16 臂
  55–90 TFLOPS（对 192 TFLOPS 标称 29–47%）。标称 384 TOPS 需要 ixTE 私有栈路径（本任务
  栈不可用），torch/Triton 可达路径远低于标称。
- **证伪臂（int8→bf16 cast 后普通 GEMM，_iluvatar/matmul_int8.py 占位模式）**：同样全面慢于
  bf16 基线（M=64 0.26–0.44×）→ 排除「B 臂失调」解释，是 int8 路径本身在该栈上无收益。

### 数值侧（管线冒烟）全过 ✅

- 合成冒烟：编译 ✓ / 输出有限 ✓ / vs fp32 参考 rel_median 1.35e-03、p99 3.46e-03
- 真实权重假层对拍（q_proj L0 RTN→scaled_mm vs fp32）：median 1.35e-03 / p99 3.46e-03 /
  cos 0.999999 —— **kernel 数值正确，问题纯在吞吐**。

### 路径结论（kernel-matrix §3 印证）

sm71 上 cutlass int8 快路径被 sm90 门挡死（`scaled_mm.py:292-294`），实际执行 FlagGems Triton
通用 int8 kernel（int32 累加）——该 kernel 在本后端带宽效率不足，W8A8 在天数侧是**性能负资产**：
权重显存 −1.84GiB 的收益换来每层 GEMM 1.1–3× 减速，官方 bench 双场景必然回退。

## 后续动作（按任务书降级链第 2 条 + 负责人决策）

1. 工程线仍走完 Phase 4 端到端（负责人指令：工程先行，精度算法另行改进）——serve 级实测
   回退幅度与官方配方精度照测照报，作为基线数据与沐曦侧复用素材。
2. W8A16 weight-only（FlagGems weight_int8pack_mm 现成）留作下一层假设：不涉激活量化与
   scaled_mm，若其 kernel 带宽正常则显存收益可保性能不回退——不在本时间盒展开。
3. 沐曦 C500：转换脚本/config/方法论直接复用，硬件闸门须重跑（480 TOPS INT8 + MXMACA
   官方 W8A8 支持，结论不可迁移）。
