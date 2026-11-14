# 2026-11-14 P3 W4A8 探针判决：NO-GO

`scripts/d2/phase3_w4a8_probe.py`：裸 jit packed-int4（offset-binary，沿 N 两
nibble/byte）+ `tl.interleave` + uint8 逻辑右移解包 → int8 tl.dot，M=64，
vs P0 decode 优胜配置（graph 3×30 中位）。**位精确 0 失败**（vs `torch._int_mm`
emu，解包数学正确），判决纯粹来自性能：

| 形状 | W4 最优 | vs P0 int8 优胜 |
|---|---|---|
| gate_up (2048,12288) | (64,256,64,w4,s2) 214.1µs | **0.262×**（P0 56.2µs） |
| lm_head (2048,130560) | (64,128,128,w4,s2) 2162.6µs | **0.263×**（P0 567.7µs） |

## 结论

- **判决 NO-GO（判据 ≥1.15×）**：解包 ALU 税（每 int8 元素 ~3 ops：and/shift/
  sub + interleave shuffle）在 16 SM 上摧毁收益——权重读减半的带宽收益（理论
  ~2×）被 4× 计算劣化吞没。本 backend 无 int4×int8 tl.dot，唯一开放形态即此，
  **W4A8 在本栈判死**（除非未来 backend 原生支持 int4 dot）。
- 探针首版两 bug 存档：①打包数学用 `w & 0xF` ≠ offset-binary 的 `w+8`；②
  int8 算术右移在高半字节≥8 时错位，须 uint8 逻辑右移。修正后位精确全过——
  判决可信。
- prefill 侧不测（BM=1 时 B tile 恰解包一次的前提只在 decode 档成立，prefill
  32× 冗余解包必亏——设计上就排除）。
