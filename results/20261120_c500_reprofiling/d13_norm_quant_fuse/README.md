# D13 fused_add_rms_norm + per-token-int8-quant 融合 T0（feat/d13-norm-quant-fuse）

- 日期：2026-11-20 10:5x · B 机 · GPU 独占 · `scripts/d13/norm_quant_fuse_t0.py`
- 数学：h=(x+resid).bf16 → var=sum(h²)/N → rrms=1/sqrt(var+eps) → y=(h·rrms·w).bf16
  → yf=f32 → absmax → inv_scale=fdiv(1,absmax,ieee)·127 → round_half_to_even
  （照抄现役 bare kernel 位兼容口径）→ clamp → int8 + scale=absmax/127。
  两处中间 bf16 round 均复刻（两步链的中间张量落盘语义）。

## T0 结果

| 形状 | 位精确（vs 同数学两步链） | 两步链 | 融合 | 加速 |
|---|---|---|---|---|
| M64 K2048（decode 主档） | **PASS（0 mism）** | 28.8µs | 25.4µs | **1.133×** |
| M64 K6144（decode down） | PASS（1/393216，fma 级） | 35.6µs | 32.2µs | 1.105× |
| M2048 K2048（prefill） | **PASS（0 mism）** | 72.2µs | 62.2µs | 1.161× |
| M4096 K6144（大 prefill） | PASS（1/25M） | 432.9µs | 484.2µs | **0.894×（反慢）** |

- **位判据 PASS**：融合 vs「同数学 norm-only kernel + serve 现役 bare quant」
  全部位一致（K=6144 档 1 个元素/25M，编译器 fma 折叠级）——**融合零数值损失**。
- vs C++ `_C.fused_add_rms_norm` 链：q mism 6.6%、scale maxrel 7.75e-3——
  **后端间差量级**（实测 C++ vs torch 精确链本就 21% mism/1e-3 相对差；serve
  实际走 inductor triton 版与 C++ 亦不同）——非融合引入。
- **性能**：decode/prefill 档 1.10-1.16×；**大 prefill 档（K6144 大 BLOCK 单
  CTA 寄存器压力）0.894× 反慢**——若集成需 per-shape 路由（M≤2048 融合 /
  大 M 走原两步），或做分块循环版（two-pass sum）。

## e2e 收益重估（供 A 机裁定）

微基准节省 ~3µs/次（decode 档）。serve 口径（墙图 norm 7.7µs + quant 9.2µs）
保守估省 ~2µs/次 × 56 次/步（28 层×2）≈ 112µs/步 / ITL 26ms ≈ **+0.4-0.6%**
——低于调研报告 F7 的 2-3% 预期（该预期假设省 0.7-1ms/步）。是否推进 T1/
集成（dispatch op 注册 + per-shape 路由）由 A 机按此证据裁定。

PROVENANCE：B 机 C500 · graph_time_us 3×30 中位 · torch 2.8.0+metax ·
参照系：vllm `_C.fused_add_rms_norm` + vllm_fl flaggems bare quant（serve 现役）。
