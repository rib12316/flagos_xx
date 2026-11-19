# P1-B2 — prefill attention 曲面（MACA flash varlen，沐曦 C500）

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

**核心事实：**
1. **MACA flash varlen paged 在 chunked 续算（kv>q）形态下，DRAM 加载流恒定
   ~90-120GB/s（跨 S/q/kv 全形状），= 顺序读上限的 6-8%**。表观 GB/s 随 q 下降
   （22→6）只是 q-tile 读放大（无 KV 跨 tile 复用）的镜像；时间严格 ∝ S×q×kv。
2. **纯新 prompt（q=kv）不受此限**：S1 q4096 kv4K = 224 TFLOPS（112% 峰值口径，
   compute-bound）——KV 4.2MB 全在 L2，放大读免费。4K 场景 prefill（大 q、短 kv）
   基本免疫；**16K 场景（q2048 续算 14K 前文）是重灾区**。
3. **与 causal / 页表散射无关**：causal=0 同慢（48.7ms vs 45.6ms）；顺序页表
   （arange）45.3ms vs 乱序 45.6ms——散射税 <1%，慢是 kernel 内在加载模式。
4. **16K 官方点（A 机指定 S16 q2048 kv16K causal）= 45.6ms/层调用**：28 层
   → **1.28s/chunk-batch**。这是 16K prefill 段的候选主墙。
5. S1 全形态只有 6-13GB/s 表观（单 seq 并行度不足，latency 域）——但绝对值小
   （S1 q2048 kv16K 2.56ms），serve 里 S=1 chunked 不是主形态。

## 对 Phase 2 排名的影响（D7 升格）

- **D7（prefill attention 形态工程）判据强力成立**：加载效率空间 16×（90→1477GB/s）。
  即使只收割 1/4（360GB/s），16K 官方点 45.6ms → ~12ms → 每层省 33ms、28 层省
  ~0.93s/chunk-batch。候选路径（按可行性序）：
  a. **block_size 16→64**（C-H4 旧假设复活：页粒度 8KB→32KB，gather 效率 4× 空间；
     platform.py 16 处，代码默认非 serve 参数）——建议第一子假设，近零成本
  b. 自研 split-KV prefill kernel（feat/d2-w8a8-quant 的 V0-V3 + hop 骨架可复用，
     decode 版已有沐曦 hop 先例）
  c. KernelGen/Kernel-Smith 候选（在线端可达已验证）
- 4K 场景 prefill 免疫 → D7 收益集中在 16K（官方 16K 7509.87 的 prefill 段占比
  待 A 机 A1 墙图交叉确认）。

## 附录：隔离实验（临时脚本，非 CSV，数字见对话记录）

- S16 q2048 kv16K：顺序页表 45.3ms / 乱序 45.6ms / 非 causal 48.7ms
- S1 q2048 kv16K 2.57ms；S16 q2048 kv2048（kv=q）3.29ms
- S8 q2048 kv16K 22.7ms（=S16 ½，线性于 S）；S16 q1024 23.6ms（线性于 q）；
  S16 kv8K 21.1ms（线性于 kv）

PROVENANCE：git 1758748+（脚本随本 probe 提交）· MetaX C500 104 SMs ·
torch 2.8.0+metax3.7.0.7 · flash_attn 2.6.3+metax · B 机 · GPU 独占。
