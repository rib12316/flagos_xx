# P1-B2 — prefill attention 曲面（MACA flash varlen，沐曦 C500）

> **⚠ 02:15 判决修正版**（替代 01:55 初版的「16× 加载流/90GB/s」模型——该模型系读放大
> 假设的错误拟合，本版以 `rootcause_isolation.csv` 的隔离实验为准；台账 §9 同步修正）。

## 修正后核心事实

1. **chunked 续算形态（q 对超过自身长度的 prefix KV 做 attention）效率平台 ~96
   TFLOPS（全量口径，峰值 224 的 43%）**；纯自注意（q=kv）223-240 TFLOPS 满速。
   两者皆 compute 域（加载流从未饱和，非带宽问题）。
2. **全部官方可调旋钮排除**（`rootcause_isolation.csv`）：
   页块 BLOCK 16→128 无效（1.00×）；dense 比 paged 仅 +7%（q512 档 +51% 但仍 138
   TFLOPS）；两段式拆分（ctx 非 causal + self causal）无效（ctx 段本身即慢）；
   `flash_attn_with_kvcache` 多 token 查询 = 同一 kernel 底座同慢；`num_splits`
   0→16 反而单调变慢（0.88×）。
3. **结论：闭源 kernel 内在限制，无调用形态绕道** → D7 唯一路径 = 自研/生成
   prefill kernel（KernelGen/Kernel-Smith/天数 ⑥ 骨架移植），**上限 2.2×**
   （16K 官方点 45.6→~20.4ms/层·批，28 层省 ~700ms/chunk-batch）。
4. C-H4（block_size 16→64）**在加载效率维度判死**（BLOCK A/B 1.00×）；调度层
   块表/清零开销论点不受本实验影响，但预期维持原 +1-3% 低位。
5. 16K 官方点（S16 q2048 kv16K causal）单层调用 45.6ms 事实不变；q=kv 纯新
   prompt（4K 场景）免疫事实不变。

## 以下为初版曲面数据（形状 scaling 仍有效，仅机理归因修正）



- 日期：2026-11-20 01:50 · B 机 · GPU 独占（起始 ~1.5GB）
- 脚本：`scripts/d5/prefill_surface_metax.py`（新写；计时=graph_time_us 3×N replay 中位，
  调用形态抄 vendor/metax flash_attn.py:807 prefill 段，页表 randperm 散射口径）
- 原始：`prefill_surface.csv`（64 cases；头两行 ERROR 为 window_size=None 修复前残留，
  成功行在后）+ 本 README 附录的隔离小实验
- 形状语义：q = 本 chunk 新 token，kv = 每 seq KV 总长（含 q 自身，vLLM 先写后算）；
  NH=16/NKV=2/HD=128/BLOCK=16（MiniCPM5-2B 真实配置）

## 判决（按 q 分组，causal=1，S∈{8,16,32,64} 趋势一致）

| 形态（kv=16K） | 表观 GB/s | 读放大 ×(q/128) | 有效加载 GB/s |
|---|---|---|---|
| q=512 | 22 | 4 | **88** |
| q=1024 | 11 | 8 | **88** |
| q=2048 | 6 | 16 | **96** |
| 顺序大块读参照（268MB 档） | — | — | **1477**（tuned ref，181.8µs） |

**初版机理归因（⚠已弃用——「读放大×q/128→有效加载 88-96GB/s」模型是错误拟合，
正确归因见顶部修正版：compute 域 96 TFLOPS 效率平台。以下数字仅形状 scaling 有效）：**
1. ~~MACA flash varlen paged 在 chunked 续算（kv>q）形态下，DRAM 加载流恒定
   ~90-120GB/s（跨 S/q/kv 全形状），= 顺序读上限的 6-8%~~（归因弃用）
2. **纯新 prompt（q=kv）不受此限**：S1 q4096 kv4K = 224 TFLOPS（compute-bound）——
   4K 场景 prefill 基本免疫；**16K 场景（q2048 续算 14K 前文）是重灾区**。（有效）
3. **与 causal / 页表散射无关**：causal=0 同慢（48.7ms vs 45.6ms）；顺序页表
   （arange）45.3ms vs 乱序 45.6ms——散射税 <1%。（有效）
4. **16K 官方点（A 机指定 S16 q2048 kv16K causal）= 45.6ms/层调用**：28 层
   → **1.28s/chunk-batch**。16K prefill 段候选主墙。（有效）
5. S1 全形态绝对值小（S1 q2048 kv16K 2.56ms），serve 里 S=1 chunked 不是主形态。（有效）

## 对 Phase 2 排名的影响（修正版）

- D7 仍成立但上限修正为 **2.2×**（自研 kernel 收割 96→224 TFLOPS 平台差）；
  16K 场景专属（4K 纯新 prompt 免疫）。工作量 2-4 天对 2.2× 上限，排期需与
  D5（采样，A 机在办）/D6（decode attention 对决）按 A 机 A1 墙图的 prefill 段
  占比定优先级。
- 原README 附录隔离实验已固化进 `rootcause_isolation.csv` + `blocksize_ab.csv`。

## 附录：隔离实验（已固化 CSV；数字见对话记录）

- S16 q2048 kv16K：顺序页表 45.3ms / 乱序 45.6ms / 非 causal 48.7ms
- S1 q2048 kv16K 2.57ms；S16 q2048 kv2048（kv=q）3.29ms
- S8 q2048 kv16K 22.7ms（=S16 ½，线性于 S）；S16 q1024 23.6ms（线性于 q）；
  S16 kv8K 21.1ms（线性于 kv）

PROVENANCE：git 1758748+（脚本随本 probe 提交）· MetaX C500 104 SMs ·
torch 2.8.0+metax3.7.0.7 · flash_attn 2.6.3+metax · B 机 · GPU 独占。
