# 20261115 瓶颈深挖（阶段 A 全量数据）

终态配置（ckpt `/workspace/MiniCPM5-2B-w8a8-v2` + 默认 env + serve 红线参数）上的
L1+L2 重跑与四维度细化。工具：`scripts/d2/attn_surface_probe.py`（新增，复用
playbook 探针的 graph 口径计时）；本 README 为墙图刷新版，详细方案见
`docs/bottleneck-research-report.md`（阶段 B）。

## 0. 目录

| 文件 | 内容 |
|---|---|
| `decode_surface.csv` | decode 曲面：kv{1K..32K}×bs{8..128}×{3d,2d,3dx}，55 case |
| `segm_sweep.csv` | 3D 段数扫 {2,4,8,32}×4 代表点，16 case |
| `td_ab.csv` / `tdx_ab.csv` | use_td A/B（decode 4 点 / prefill 5 点 + TD×段数叠加 4 点） |
| `prefill_surface.csv` | prefill 曲面 S×q×kv 24+5 case |
| `ref_table.csv` | 逐字节量带宽参照：sum() 顺序读 vs 调优读核（BLOCK=16384） |
| `scatter_ab.csv` | 乱序 block_table（serve 分配器散射口径）vs 顺序，4 点 |
| `l1_refresh.log` | playbook 原版 L1 重跑（线性层基线刷新） |
| `trace_decode/` `trace_prefill/` | L2 双窗 trace + parse 输出 |
| `capacity/` | 16K 容量路径：metrics.log / serve2.log / bench_c32.log / bench_c48.log |
| `dec_warm.log` `dec_prof.log` `pre_prof.log` | L0 ITL 采样（4 轮）与 profiled 臂 |

## 1. 墙图刷新（L2，graph 外可见侧）

**decode 窗**（[[256,2048,64,64]]，纯 decode 步、bs≈64、kv≤2.3K）：
- **`_topk_topp_kernel` 4407µs/步，占可见 GPU kernel 时间 76.2%**（8193 步 × 4.4ms）
  ——全新发现的第 4 大件：vllm `vllm/v1/sample/ops/topk_topp_sampler` 的全词表
  topk/topp 过滤（[64,130560]），采样链合计 ≈5.3ms/步。
- 其余可见：bf16 lm_head GEMM 488µs、epilogue/量化链 ~1ms。attention 在
  FULL_DECODE_ONLY graph 内重放，对 trace 不可见（既定口径）。
- **L0-L1 对账**：42×(attn ~330µs@kv1.3K 3D + 融合线性 87µs) + lm_head 628µs
  + **采样 5.3ms** + norm/rope ~1.5ms ≈ **23.3ms vs L0 中位 ITL 25.1ms**（丢首
  4 轮中位 21.3/25.2/24.5/25.1）→ 缺口 ~8% **< 10%：host pacing 不立案**。
  老预算缺采样项，缺口其实被采样 kernel 解释。

**prefill 窗**（[[4096,128,64,64]]）：**A_attention 82.1%**（kernel_unified_attention
7427µs×5460 次，2D 路径）；M_int8_gemm 9.6%；quant 2.0%；norm/rope 2.9%；
采样 1.7%。刷新历史 87.7%（同量级确认）。

## 2. decode attention 双路径曲面（核心修正）

**修正一个历史口径偏差**：playbook 探针 `attn_case` 不传 segm buffer →
wrapper 判据强制 2D——历史「decode attention 193-197GB/s = 36% 实用带宽」是
**2D 路径口径**。serve 中 bs≤64 走 3D（固定 16 段 split-KV + reduce），实际更快：

| 形态（bs/kv） | 2D | 3D(n16 现状) | 3D 相对 2D |
|---|---|---|---|
| bs64 / 4K | 178 GB/s (50%ref) | 283 GB/s (80%ref) | 1.60× |
| bs64 / 16K | 198 (61%) | 317 (99%) | 1.61× |
| bs128 / 16K | 242 (69%) | 321* (100%*) | 1.32×（*p3dx=强制 3D） |
| bs8 / 8K | 53 (16%) | 243 (73%) | 4.6× |

关键读数（%ref=同字节量 sum() 顺序读参照；tuned=调优读核）：
- **3D 路径 72-99% of sum-ref，但 sum-ref 本身只有 ~350GB/s**；调优读核
  （BLOCK=16384）达 **650-664GB/s**（`ref_table.csv`）→ 3D 现状 ≈ tuned 天花板
  的 40-49%，**理想 kernel 仍有 ~1.7-2× 空间**。
- **bs=128 悬崖实测**（kv≥8K）：退化 2D 比 3D 慢 1.26-1.35×；本赛两场景 conc=64
  恒在 3D 区，但混批（prefill chunk+decode 同步）强制 2D——serve 混批步的
  decode 行吃 2D 速度。
- **散射保真度**：乱序 block_table vs 顺序 ≈ 无差（0-5%），探针顺序口径有效。

### 2.1 段数（split 切块）最优档

| 形态 | n16(默认) 最优段 | 提速 |
|---|---|---|
| bs64/kv8K | **n8** 303GB/s | **1.24×** |
| bs64/kv4K | **n8** | ~1.24×（tdx 臂 731µs 复核） |
| bs16/kv8K | n32 284GB/s | 1.07× |
| bs64/kv32K | n8/n32 | 0.96-1.00× |

4K 场景 decode 常驻形态（bs64/kv4-8K）正好落在 n8 收益区。

### 2.2 use_td（tensor descriptor）——单点最大杠杆

iluvatar/flagtree 后端可编译可跑（本机实测）：

| 臂 | decode 3D | decode 2D | prefill 2D |
|---|---|---|---|
| td0→td1 | **1.20-1.31×**（bs64/kv2k 479→400µs；bs16/kv8k 504→386µs=348GB/s） | **1.55×**（bs128/kv16k 9996→6460µs） | **1.10-1.14×**（S1q4096kv4k 13654→12420µs） |

**TD + 最优段数叠加**：bs64/kv8k 2198→**1393µs（1.58×，386GB/s=58% tuned）**；
bs64/kv4k 1.30×；bs16/kv8k 1.34×；bs64/kv32k 1.28×（414GB/s）。
**数值注意：TD 非 bit-exact**（max_abs_diff 0.12-0.31，累加序不同）——上线必须过
parity 门（105 题/烟测对拍，新鲜 serve）。

## 3. prefill 曲面：结构性低效确认

q>1 恒 2D。全部 29 形态 **%MAC 仅 2-4%、%BW≈0**（vs 线性层 89% vendor MAC）：
纯 kernel 结构问题（TILE=32 串行 kv 循环、BLOCK_M=16 小 MMA、w2s1 无流水）。
规模参考：S1q4096kv4k=13.6ms/层（causal 减半≈6.8ms，与墙图 7.4ms/次吻合）。
同 total-q 对照：S8q512 vs S1q4096（kv16K：105.2ms vs 94.1ms）——S 越大越亏
（每 seq 独立扫 kv，无共享）。TD 只给 1.10-1.14×，**深挖需 kernel 形态改造**
（kv 维 split / 更大 TILE / FA2 式分块）。

## 4. 16K 容量路径：机理改写

- 池实测 **588,112 tokens**；**同驻上限实测 33**（kv 峰值 0.98 时 running=33，
  理论 588112/17408=33.8 精确吻合），max_wait=39。
- **全程 Δpreempt=0**（含 48 并发×16K 超发组）——`reserve_full_isl` 全长准入
  不超发：**16K 瓶颈是并发钳制（33/64）+ 排队，不是 preemption-recompute 税**。
  402s TTFT = 排队。与官方 16K/4K 吞吐比 0.474≈33/64 自洽。
- 推论：KV int8 池翻倍（588K→1.13M）→ 同驻 33→64 → 16K 场景 decode 并发
  解放，这是 16K 场景第一杠杆（前提：per-step 速度不掉，量化线已有免疫证据）。
- seg0（decode 窗）APC=71%、seg1（16K 组）APC=43%——跨轮前缀复用显著，
  bench 口径注意残留 cache 虚增。

## 5. 收益上界折算（kernel 层 → 端到端）

锚点：官方 4K Total 2838.46 tok/s（Median ITL 63.16ms 含 prefill 交错）、
16K 1345.90（Median ITL 100.14ms）；本机纯 decode 步 25.1ms（bs64/kv≤2.3K）。
4K 场景时间构成推算：prefill 侧 ≈300s / decode-only ≈100s / 其余排队间隙。

| 杠杆 | kernel 层 | 端到端折算（假设见报告） |
|---|---|---|
| TD+段数（decode 3D） | 1.28-1.58× | 4K +3~5%；16K +8~15%（attn 占 ITL 80%） |
| TD（prefill 2D） | 1.10-1.14× | 4K +4~5%（82% 墙图×~10%） |
| 采样 topk_topp 快路径 | 5.3→~0.5ms/步 | 4K +4%；16K +4%（decode 步 −17%） |
| KV int8 扩池 | 池 ×1.94 | 16K 并发 33→64：**+40~90%**（上界，需实测） |
| 自写 split-KV kernel（追 tuned 660） | 再 1.3-1.6× | TD+段数之上的增量，16K 主战场 |

## 6. 判定链汇总

1. prefill attention 82.1% 墙图 + %MAC 2-4% → **kernel 形态工程**（TD 先收
   1.1×，深挖需 kv-split/tiling 改造）
2. decode attention（3D）latency/occupancy → **TD+段数即收 1.3-1.58×**；
   上限 tuned 660 尚有 1.7× 空间（自写 kernel 范畴）
3. 混批步 decode 行走 2D（1.3-1.6× 慢）→ 调度/门槛问题，非 kernel 问题
4. host pacing：**不立案**（对账缺口 8%）
5. 16K 容量：**准入排队机理**，KV int8 扩池为第一杠杆
6. 采样 topk_topp 4.4ms/步：**新立案**，占 ITL ~18%，疑为全词表排序路径
