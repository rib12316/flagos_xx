# D1 抢占判定书（2026-11-04~05 裁决轮）

## 实验设定
- serve：官方锁死参数逐字（port 9031 / FULL_DECODE_ONLY / gpu-mem 0.85 / max-model-len 131072），env 仅增 `VLLM_PLUGINS=fl VLLM_FL_SCHED_FL=1`（诊断注入，行为不变——见 §骨架行为不变证明）
- bench：`vllm bench serve` 官方子命令 flags + 显式 `--seed 0` + `--save-result`；同一 serve 进程连跑 16K×2 + 4K×2
- 生效调度参数（banner 实测，commit b454717 前的 main 形态）：`mnbt=2048 mnst_eff=2048 seqs=256 wm=0.0 lpt=0 mnp=1 rsv_full_isl=True apc=True bs=16 blocks=32621(=521,936 tokens) async_sched=True`
- PROVENANCE：本目录 PROVENANCE；分析：`python3 scripts/analyze_d1_metrics.py results/20261104_d1_adjudication/metrics.log --fl-sched .../serve.log`

## 骨架行为不变证明 ✅
- 16k_run2（热身轮）Total tok/s = **1264.41**，落在官方轮 2-4（1262.47/1262.68/1262.84）带内正中（+0.12%）
- 16k_run1（冷轮）= 1190.52，完整复现官方丢首轮逻辑（官方 round1=1238→round2=1263，+4.9%；本文 run1→run2 +6.2%，均为冷启动预热差）
- 结论：SchedulerFL(AsyncScheduler 基类) + [fl.sched] 诊断日志 = 零可测行为差异

## E1 · Prometheus metrics（直接证据）
| 轮 | Δnum_preemptions | max kv_cache_usage | max running | max waiting | 时长s | Total tok/s |
|---|---|---|---|---|---|---|
| 16k_run1(冷) | **4** | **1.000** | 31 | 63 | 1849 | 1190.52 |
| 16k_run2(热) | **4** | **1.000** | 31 | 62 | 1753 | **1264.41** |
| 4k_run1(冷) | **0** | 0.621 | 64 | 60 | 675 | 1941.8* |
| 4k_run2(热) | **0** | 0.620 | 64 | 60 | 525 | 2496.6* |
*4K tok/s = (1,048,576+262,144)/时长

## E2 · [fl.sched] 诊断指纹（1676 行）
- preempt>0 步数 = 8（与 metrics Δ 一致）；全部为 `pre_tok=0 dec_tok=30 run=30` 的纯 decode 步、free_blk≈1033（抢占后值，受害者刚释放 ~1024 块）
- **锯齿机制**：抢占事件步号 591/1610/2667/3691/…，间隔 ≈1024 步——准入在 free≈1033 时擦边过 ISL 检查（需 1024 块），之后净耗 ~1 块/步，~1024 步后余量耗尽 → decode 分配失败 → victim=running.pop()（最新者，半程 prefill 作废，num_computed_tokens=0）
- **鸽笼论证**：第 30 条准入后已承诺 30×1024=30,720 块，余 1,901 块 < 30 条 decode 终身需求 30×64=1,920 块 → 周期性抢占是 ISL-only 准入（reserve_full_isl、wm=0）的必然结果，非偶发
- free_blk 谷值=0（容量绝对钉死）；中位 13,772
- 步构成：均值 pre_tok=1495（< 预算 2048：准入节奏约束）、dec_tok=40；pre_tok=0（decode-only）步占 24.2%（大头为波尾：每波最后 ~1024 步全 decode；余为钉死期等释放的间隙）
- 准入节奏：ramp 期 new=1 每 ~8 步 = 恰好 16384/2048 chunk 串行化（mnbt=2048 同时只容一个 prefill 在飞）

## E3 · TTFT 分布
| 轮 | Mean TTFT | Median | P99 | 备注 |
|---|---|---|---|---|
| 16k_run1 | 453,438ms | 481,259ms | 871,889ms | 单峰长尾（排队型特征，非抢占双峰）|
| 16k_run2 | 434,877ms | — | — | |

## E4 · 跨轮 APC 命中率（run2−run1 差分，metrics 实测）
- 16K：Δcached/Δprompt = **0.00%**（预测吻合：LRU 驱逐动态）
- 4K：Δcached/Δprompt = **0.39%**（4,080/1,048,576）——**推翻"4K 逐轮爬升=跨轮命中"旁证假设**；官方 4K 轮间 +10% 爬升与 APC 无关（编译/图预热等其他因素）
- **H6 裁定：保持 APC 开（上游默认）**——两场景命中均≈0，关掉仅省哈希 CPU（async 调度下与 GPU 重叠，不可测收益）而丢 0.39% 命中，净收益≈0，不值得偏离默认。VLLM_FL_APC 逃生门保留作报告消融用

## 判定树结论
- **分支 B-plus**：抢占存在但量小（4 次/轮，8 次事件累计），usage 触顶 1.0 + running<64 与 waiting>0 长期并存 = 容量排队型为主、叠加周期性 razor-thin 抢占
- 证据等级：E1 直接 metrics + E2 逐步指纹（双路一致）
- 主打杠杆（按机制确定性 × 成本排序）：
  1. **H7 watermark 准入预留**（VLLM_FL_WATERMARK=0.03 → 978 块 ≈ 1 条请求 decode 余量）：机制上直接消灭 8 次/轮抢占废工（≈2-3% 输入 token 纯浪费），纯 config 零子类，4K 侧无影响（usage 0.62 永不触闸）
  2. **mnbt 2048→8192**：chunk 串行化解除（8 chunk→2 chunk）+ prefill GEMM M 增大利用率；风险=T0 必查 KV 池缩水（profile 峰值随预算放大）
  3. H4 block64：块表管理 ÷4（CPU 侧，async 重叠下收益存疑），时间允许才做
- H1（子类准入预留）降级：watermark 即其 config 等价物；H2/H10 不立项（抢占量小）；H3'（波次插入）由 mnbt 覆盖主收益

## 指纹 → 杠杆映射（实测回填）
| 指纹 | 实测 | 对应杠杆 |
|---|---|---|
| preempt≥1 集中在 new>0 之后 | ❌ 集中在 decode-only 钉死步（每 ~1024 步锯齿）| watermark（非 H1 子类）|
| preempt=0 + free_blk≈0 + wait>0 并存 | ✅（16K 钉死，谷值 0）| watermark + mnbt |
| dec_tok≤64 且 pre_tok=0 空窗成段 | 部分：24.2%（大头波尾）| mnbt（缩尾）/ 波次插入 |
| pre_tok 长期 ≪2048 且 wait=0 | ❌（wait 常驻 34，准入节奏受限）| mnbt（2chunk 加速准入节奏）|
