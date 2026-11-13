# D2 prefill「0.764× 回退」归因报告（2026-11-13）

## 结论速览

1. **原始 0.764× 不是稳定的算子回退**——quick bench 的汇总口径（RUNS=4、SKIP_FIRST=1、
   后 3 次 mean）被 **prefix cache 双峰性**支配：同 seed 随机数据使 pass 2-4 复用 pass 1 的
   KV 前缀（两臂命中率同构：冷 pass 0%、热 pass 66-79%），热 pass ~2600-3800 tok/s、
   冷 pass ~1000-1800 tok/s，3 次 mean 由命中比例决定，臂内汇总方差 1415→3789。
2. **GPU 侧 w8a8 每 forward 反而快 17%**（两臂 12s torch profile 窗实测：147.6 vs
   178.5 ms/forward，attention 同价 3.4ms/call，线性链 62.5 vs 70ms）。
3. **真实但更小的回退只存在于冷 prefill 段**：int8 链 GPU 税（epilogue 24% kernel 时间
   + quant 3%）≈ 抵消 int8 GEMM 的省（链 GPU ≈ bf16 单 GEMM），外加每线性 3 kernel
   × Corex 慢 launch（~50-105µs/次）在 eager prefill 里多付 ~2200 次额外 launch/窗，
   冷 pass 的 decode ITL 被间歇卡到 mean 138ms（bf16 21.7ms）。
4. decode 侧（cudagraph 内，launch 免费）w8a8 已稳定赢：ITL 17.9 vs 19.9ms（warm 态
   3789 tok/s case1 汇总）。

## 证据链

### A. quick bench 汇总口径与 pass 分布（bench 脚本 RUNS=4/SKIP_FIRST=1/mean(2-4)）

| 臂 | case1 汇总（tok/s）→ **3 轮中位** | pass 明细 |
|---|---|---|
| w8a8 | 1415.16 → 2492.18 → 3788.75（全热）→ **中位 2492.18** | [411,1995,1280,971] / [685,2659,1072,**3745**] / [3770,3799,3768,3800] |
| bf16 | 1853.03 → 2522.05 → 1300.55（冷启）→ **中位 1853.03** | [584,1224,**3541**,794] / [533,1795,2247,**3524**] |

**3 轮中位：w8a8 2492 vs bf16 1853 = w8a8 领先 1.35×——「prefill 回退」正式推翻**
（原始 0.764× 为单轮噪声抽签：两臂汇总方差 1300→3789，由缓存双峰性支配）。
最佳单 pass：w8a8 3799 > bf16 3524；decode ITL：17.9 vs 19.9ms；GPU/forward：147.6
vs 178.5ms——w8a8 全面≥。bf16 全热轮因前轮被杀缓存态不干净、pass 拖长未出汇总（非
结论性数据，弃用；其稳态窗口 gen ~400 tok/s@16 并发与 ITL 19.9ms 自洽）。

- w8a8 打出 3745/3799（>bf16 最佳 3541/3524）→ 两臂天花板同量级，warm 态 w8a8 更快。
- TTFT 指认双峰成因：3745-pass TTFT 544ms（前缀全命中，prefill 几乎免费）；1072-pass
  TTFT mean 27.4s（缓存被逐，真实冷 prefill + 排队）。
- 物理 sanity：16 并发 × ITL 20ms ≈ 800 tok/s decode 上限 → 3500+ 只有 prefill 免费才可能。

### B. 两臂 torch profile（诊断 serve + /start_profile 12s 窗，case1 稳态，M=2048 chunk）

| 指标 | w8a8 | bf16 |
|---|---|---|
| kernels / 窗 | 8590（quant 1848 + epi 1848 + gemm 1848 + attn 462 + …） | 5842（mm 2184 + attn 546 + …） |
| GPU kernel 总时间 | 2436ms（16.5 forward → **147.6ms/fwd**） | 3481ms（19.5 fwd → **178.5ms/fwd**） |
| attention | 1586ms = 65.1%（96.6ms/fwd） | 1895ms = 54.4%（97.2ms/fwd）同价 ✓ |
| 线性部分 GPU | 链 1033ms（GEMM 366 + **epi 589** + quant 78）= 62.5ms/fwd | mm_kernel 1365ms = 70ms/fwd |
| cuLaunchKernel host | 1044ms/6742 次 + cudaLaunchKernel 318ms/1848 次（含 profiler 开销上限） | 1927ms/5842 次（GPU-bound 队列满阻塞型） |

- **epilogue = 24.2% 全部 GPU kernel 时间**（589ms，avg 120.6µs×1848），比 int8 GEMM
  本身（15%）还显眼；int32 中间往返税实证。
- 微基准（`profile_prefill_chain.py`，热 L2）链每形状赢 1.2-1.3×，但 serve 里 attention
  流 KV 冲刷 L2 → epilogue 有效带宽 ~420GB/s，链 GPU 总账 ≈ bf16 单 GEMM（省的全交税）。
- 每 forward kernel 数：w8a8 ~520 vs bf16 ~300（+73% launch）；Corex launch 实价
  ~50-105µs（profiler 前科口径）→ eager prefill 冷段 host 压力 w8a8 显著更高。
- `linear_kernel`(lm_head bf16) 931µs×11：decode 每步 lm_head 读 537MB@~577GB/s——P3
  候选实证（仅 1.7% kernel 时间，低优先）。

### C. 真实形状

cpu_op record_shapes：prefill chunk **M=2048**（[2048,2048]×2520 + [2048,6144]×840，
3:1 = qkv/o/gate_up:down ✓），无 [16,*]（decode 在 cudagraph 内重放，不产生 cpu_op）。

## 对开发优先级的修正（数据定案）

- 「prefill 大回退」降级为「冷 prefill 段的小额真实税 + bench 方差」；**不改变 prefill
  融合为第一优先**——理由从『救回退』改为『官方 4K/16K 场景 KV 足迹大（16K×128 提示
  ≈2M token ≫ 582K 池）、缓存更冷、真实 prefill 计算占比更高，链 GPU 税（epi+quant
  27%）+ launch 税是官方数字上唯一可控的大头』。
- 顺序：**Phase 0 探针（tl.dot 位精确/吞吐）→ 融合单 kernel（quant+GEMM+epilogue）**
  一步到位（同时消 GPU 税与 2/3 launch），中间不再做 epilogue 单独调参（会被融合取代）。
- P0 skinny（decode M≤16）恢复价值：warm 态 3789 由 decode 主导（ITL 17.9ms 中链
  ~4.2ms），skinny + epilogue 融合可再压 → 融合 kernel 家族（BM∈{16,64,128} tile）
  一次开发两档覆盖 decode+prefill。
- 官方 bench 的噪声特性记入台账：quick 单轮汇总不可作结论，一切对比 ≥3 轮中位 + 记录
  缓存态。

## 工具与方法备注

- ixsys V4.4.0 print-only 模式不可用（参数组合被拒，skill 文档基线 v5.0.0）；
  system-wide 只采「采集启动后 launch 的进程」无法附加已运行 serve。
- 替代路径：诊断 serve 加 `--profiler-config '{"profiler":"torch",...}'`（冻结参数不动，
  仅观测标志，不影响执行；性能结论均出自无 profiler 的正式 serve）+ `/start_profile`
  API + chrome trace 解析（`scripts/d2/parse_torch_trace.py`）。
- 本轮诊断 serve 启动过 3 次（w8a8/bf16 各 1 + w8a8 正式复测），GPU 独占串行未破坏。
