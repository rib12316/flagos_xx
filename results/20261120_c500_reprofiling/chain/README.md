# P1-B3 — w8a8 链税 + decode M 档（沐曦 C500 终局配置）

- 日期：2026-11-20 01:38 · B 机 · GPU 独占（serve 已清，起始显存 ~1.5GB）
- 脚本：`scripts/d2/profile_prefill_chain_metax.py`（MS 扩至 [16,64,128,1024,2048,8192]）
- 原始：`chain_probe.log`（本目录）
- 口径：D 臂 = `_quant(x)` + mctlass `w8a8_scaled_mm_azp` 连发 eager wall；A 臂 = bf16
  `F.linear` 同口径；gpuQ/gpuMM = CUDA event 单核计时中位（20 iters×3 rounds eager / 20 events）。
  dispatch 断言 `default.flagos_mctlass` 通过。

## 判决（数字见 log，此处摘 decode 档 M=64）

| 形状 | D/A（越低 w8a8 越快） | gpuQ | gpuMM | 链 floor |
|---|---|---|---|---|
| qkv | 0.189 | 0.039 | 0.068 | 0.106 |
| o | 0.186 | 0.039 | 0.068 | 0.102 |
| gate_up | 0.452 | 0.039 | 0.082 | 0.105 |
| down | 0.440 | 0.038 | 0.076 | 0.107 |

1. **decode 档 w8a8 链级大赢**：D/A 0.18~0.50（M∈{16,64,128}），即量化链比 bf16 快
   2~5.5×。与 1119 微基准"decode GEMM 单算子 0.79×/0.68×"不冲突——单算子 vs 链级
   （int8 权重读减半在链级兑现）。
2. **小 M floor 与 M 无关**：M≤128 时 D 臂恒 ~0.102-0.106ms、gpuQ 恒 0.039ms、
   gpuMM 0.067-0.089ms → 纯权重读带宽主导（compute 未上量）。
   M=64 全链推算：每层 4 线性 0.419ms × 28 层 = **11.7ms/步**；对 bench 实测 ITL 26ms
   ≈ **45%**；其中 quant 独立核 4.3ms（ITL 16.5%）。
3. **无 launch/host 正税**：launch_gap 全负（-1%~-70%）= eager 连发流水覆盖事件计时，
   D0 类 launch 调优在沐曦**无收益**（负结论，回台账）。
4. **D9（融合 GEMM metax 重tile）判据成立**：quant 独立核占小 M 链 37%（≥8% 立项线）。
   融合上限 = 省 quant 核 0.039ms/线性（+一次 launch 合并）→ decode 链保守 -30%
   → ITL -13% → decode 速度 +15% 量级。前置挡板：64KB smem（bedf8ca），需 tile 缩进。
   注意 event 计时含 launch 开销，此上限偏保守（真实 GPU kernel 时间可能更小）。
5. **down（K=6144）大 M quant 突出**：gpuQ 0.10-0.58ms（M1024→8192），占 D 臂 46-52%
   ——D11 flagtune per-shape 的候选点。

## 对 Phase 2 排名的影响

- D9 升格为「有实测依据的 decode 侧主候选」（原排 #5 仅当 P1-B3 见缺口——已见）。
- D0 类 launch 工作在沐曦除名。
- decode 线性链仅占 ITL 45%，余量在 attention/采样/其他 —— 与 P1-B4（采样）、P1-B1
  （attention 曲面）的预期一致。

PROVENANCE：git 2dc099a · torch 2.8.0+metax3.7.0.7 · mctlassEx 承载（dispatch 断言）·
B 机 C500 64GB 单卡 · 温度 35C/P0 · 与 A 机交叉锚点 6020.04 tok/s（等 A 机对带）。
