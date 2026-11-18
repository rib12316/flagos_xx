# 瓶颈开源调研报告（阶段 B，2026-11-15）

数据基线：`results/20261115_bottleneck_deepdive/`（阶段 A 实测，下称「深挖数据」）；
历史结论：`docs/d2-bottleneck-diagnosis.md`、`docs/d2-quant-blocking-analysis.md`。
本报告对每个确认瓶颈过「本栈约束过滤器」，逐候选输出六件套，最后给推荐排序。
**只调研不定向，未写任何生产代码。**

## 0. 瓶颈清单（阶段 A 实测刷新）

| # | 瓶颈 | 量化 | 性质 |
|---|---|---|---|
| B1 | prefill attention | 82.1% 墙图，%MAC 2-4%，7.4ms/层/次 | kernel 结构（tile 太小/无 kv 并行） |
| B2 | decode attention（3D 路径） | 244-324GB/s = 40-49% of tuned 660 | latency/occupancy（TD+段数可收 1.3-1.58×） |
| B3 | 混批步 decode 行强制 2D | 2D 比 3D 慢 1.3-1.6× | 调度/门槛 |
| B4 | 采样 `_topk_topp_kernel` | 4.4ms/步（采样链 5.3ms）= ITL ~21% | 全词表过滤路径 |
| B5 | 16K 并发钳制 | 同驻 33/64，0 抢占，max_wait 39 | KV 池容量（准入排队，非重算税） |

过滤器（全候选适用）：① Triton 3.6 flagtree/iluvatar 可编译（无 CUTLASS/nvcc，
闭源 ixRT/ixdnn 不可用，`torch._int_mm` 唯一 vendor 快 GEMM）；② 16 SM / smem
128KB / 无 fp8 / 无 int4 的占用形态可承载；③ 不动 serve 参数红线；monkeypatch
注入落 dispatch 层（`attn_launch_config.py` 两跳模式：保留原 run 引用、幂等标记、
env 开关、`VLLM_FL_*` 表热换）。

---

## 候选 1：use_td 开启（tensor descriptor 加载路径）【实测已验证】

- **机理**：`tl.make_tensor_descriptor` 做 Q/K/V 2D 块读与输出写（上游 PR
  [#40327](https://github.com/vllm-project/vllm/pull/40327)，Intel Xe2/Xe3 HW 块读；
  `USE_TD` constexpr，关=编译期死码消除）。改善加载向量化/边界处理宽度。
- **适配性**：①✅ 本机实测可编译可跑（flagtree 落降已实现）；②✅ 不改占用形态；
  ③✅ vllm 侧 gating 是 `current_platform.is_xpu()` → 需插件 dispatch 层对
  iluvatar 强制 `use_td=True`（triton_attn.py L520-528 的 tri-state env
  `VLLM_TRITON_ATTN_USE_TD` 也可，但 env 属 serve 口径，应走插件默认代码路径）。
- **预期结果（实测数字折算）**：decode 3D 1.20-1.31×；decode 2D 1.55×；
  prefill 2D 1.10-1.14×。端到端：4K 场景 decode-only 步 ~100s×(attn 占 60%)与
  prefill 步 ~300s×(attn 82%)同时受益 → **+7~10%**；16K 场景 attn 占 ITL ~80%
  → **+12~18%**（与段数叠加前）。
- **工程量**：0.5-1 人·天（dispatch 层一个 install 函数 + parity 门）。
- **风险**：**非 bit-exact**（实测 max_abs_diff 0.12-0.31，累加序变化）→ 必须过
  105 题对拍 + 烟测 parity（新鲜 serve）；Intel 小卡（B580）有 TD 回归前科
  （[epic #6704](https://github.com/intel/intel-xpu-backend-for-triton/issues/6704)），
  本机实测优势明确但需全形态复测。回退：env/插件开关一键关。
- **判定：可行（第一优先）**。

## 候选 2：3D 段数自适应 + 门槛设备化（metadata 层，免写 kernel）

- **机理**：`NUM_PAR_SOFTMAX_SEGMENTS=16` 固定段在短 kv 切太碎（kv=1K 时每段
  64 token、`tiles_per_segment=cdiv(seq_len, 16×TILE)`，源码可验算）；`seq_threshold_3D
  =128//NKV=64` 对 SM 数盲（上游 [issue #48076](https://github.com/vllm-project/vllm/issues/48076)：
  B200 补丁后 SLA 内 +68%；但 GB10 48-SM 小卡反例 nseq≥8 时 split-KV 0.83-0.97×）。
  FlashInfer 的正经做法（[arXiv:2501.01005](https://arxiv.org/html/2501.01005)
  Algorithm 1）：`target_CTA=k×#SM`、`L_kv=总KV工作量/#CTA`、短请求不切
  （writethrough 免归约）。
- **适配性**：①✅ 段数是 wrapper 参数+grid 维，kernel 现成；②✅；③✅ patch
  `TritonAttentionMetadataBuilder`（segm buffer 尺寸随段数）——dispatch 层
  monkeypatch builder 方法，两跳模式同款。
- **预期结果**：深挖段数扫：bs64/kv4-8K 用 n8 = **1.24×**（正中 4K 场景 decode
  形态）；bs16 用 n32 = 1.07×；长 kv 无收益。门槛 64→128 仅对 bs>64 混批/高并发
  有意义（本赛 conc=64 边缘）。端到端：**4K +3~5%，16K +3~5%**（与 TD 叠加
  后 bs64/kv8k 合计 1.58×）。
- **工程量**：1-1.5 人·天（含 per-shape 表与 cudagraph 捕获相容性：段数须
  init-time 固定，#48076 教训）。
- **风险**：段数改变 → grid 变 → cudagraph 捕获形状变化（须按 bs/kv 桶静态化）；
  数值不变（段内归约序同构，需 bit-exact 复核）。回退：表置默认 16。
- **判定：可行（第二优先，与候选 1 叠加）**。

## 候选 3：自写 flash-decoding split-KV decode kernel（追 tuned 660 上限）

- **机理**：FlashDecoding（[CRFM 博客](https://crfm.stanford.edu/2023/10/12/flashdecoding.html)）/
  FlashInfer plan→main→merge 三段；仓库内现成模板：metax vendor
  `triton_decode_attention.py`（`NUM_KV_SPLITS` constexpr + stage2 合并）。
- **适配性**：①✅ bare `@triton.jit`（严禁 @libentry——流竞争前科）；②✅ smem
  预算：TILE=64×128×2(K+V)×2B×3 级流水 ≈ 96KB < 128KB；③✅ hop2 `kernel.run`
  patch 处路由（`attn_launch_config` 同款）。
- **预期结果**：现状 TD+段数最优 386-414GB/s = tuned 660 的 58-63%。自写 kernel
  攻更大 TILE+流水+段数自适应，目标 450-520GB/s → **再 +15~30%**（attn 层）。
  端到端：16K +10~20%，4K +3~6%。上限：除非突破 660（读流形态极限），否则
  不超过 1.7×总空间（`ref_table.csv`）。
- **工程量**：2-3 人·天（IBM 论文 [arXiv:2511.11581](https://arxiv.org/abs/2511.11581)
  的决策树式静态配置、离线微基准蒸馏，不做运行时 autotune）。
- **风险**：数值门（bit-exact 不可求，须 parity）；与 TD 路径收益重叠（若候选 1
  落地，自写边际收益缩水）。回退：不接路由即无影响。
- **判定：可行（第三优先，候选 1/2 收完后再评估边际）**。

## 候选 4：prefill attention kernel 形态改造（B1 主攻）

- **机理**：2D 路径 `grid=(q_blocks, 2)` y 维仅 2、TILE=32 串行 kv 循环、
  %MAC 2-4%。FA2 §3.2/3.3（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）：
  序列维并行 + warp 切 KV 行；kv 维 split（对 prefill 做 chunked softmax 两段
  归约）= 把 3D 机制扩到 q>1（上游 [#56148](https://github.com/vllm-project/vllm/pull/56148)
  multi-row split-K 已合并，是现成参照；阻断点：`softmax_segm_*` 按序列分配按
  token 索引，需扩 per-token 语义）。TD 先收 1.10-1.14×（候选 1 含）。
- **适配性**：①✅；②✅（大 TILE 版 smem 预算同候选 3）；③✅ 自写 kernel 走
  dispatch 路由。**chunked prefill 预算（mnbt=8192）是 serve 参数红线**——大
  chunk 抬 occupancy 的路（Sarathi-Serve [arXiv:2403.02310](https://arxiv.org/html/2403.02310v3)
  甜点 2048-8192）只能标注不可做。
- **预期结果**：B1 占 4K 场景时间 ~65%（300s/464s×82% 墙图内 attn）。kv-split
  prefill 若复刻 decode 侧收益量级（1.3-1.6×）→ **4K +15~25%**；仅 tiling/warp
  级（不动结构）→ +5~10%。16K 场景 prefill 占比低（decode 主导），+3~8%。
- **工程量**：3-5 人·天（改 kernel + wrapper + per-shape 验证；无上游现成
  prefill-split 可整体照搬）。
- **风险**：工程量最大项；数值门同上。回退：路由开关。
- **判定：可行但重（按 4K 场景价值排序）**。

## 候选 5：采样 topk_topp 快路径（B4，新立案）

- **机理**：`_topk_topp_kernel` 4407µs/步×8193 步占可见 kernel 76.2%——对
  [64,130560] 全词表做 top-k/top-p 过滤（排序型）。bench 客户端不设采样参数 →
  serve 端默认温度采样路径。快路径：k/p 为恒等（top_p=1, top_k=-1）时跳过；
  或换 FlashInfer 式采样核（Gumbel/rejection 采样，免全词表排序）。
- **适配性**：①✅（bare triton 或 torch 链）；②✅；③✅ vllm sampler 层 patch
  （`vllm/v1/sample/ops/topk_topp_sampler.py`，同 `patch_sampler_compile` 模式）。
- **预期结果**：采样链 5.3→~0.5ms/步 → 纯 decode 步 −19%。端到端：**4K +4~6%，
  16K +4~6%**（decode 步占比分别 ~22%/90%×步内 19%）。叠加低风险。
- **工程量**：1-2 人·天（确认默认参数语义 + 快路径 + 分布等价性单测）。
- **风险**：**采样语义红线**——必须保证输出分布不变（k 恒等跳过是零风险的；
  换采样算法需对拍）。数值路径变化影响 105 题对拍的风险低于 TD（分布等价可证）。
- **判定：可行（性价比高，先做恒等快路径）**。

## 候选 6：KV int8 扩池（B5 主攻，P2 复活——仅容量角度）

- **机理**：16K 场景供给比 0.53 → 同驻 33/64、排队 39、TTFT 402s（实测 0 抢占，
  纯准入排队）。池 588K→1.13M → 同驻 33→64。老 P2 的「decode 提速」前提证伪
  关案，**容量角度复活**：INT8 KV 精度证据（[#33480](https://github.com/vllm-project/vllm/issues/33480)
  内 100 题对照：int8_per_token_head 73% = 基线 73%；FP8 未校准仅 51%）；
  FP8 KV 官方同构场景 +14.9%（[vLLM 博客](https://vllm.ai/blog/2026-04-22-fp8-kvcache)）；
  本栈无 fp8 硬件 → int8 per-token-head 是唯一路线（插件已有 KVQuantMode
  管道与 scale cache 骨架）。
- **适配性**：①✅ unified_attention 原生 `KVQuantMode.INT8_PER_TOKEN_HEAD`
  分支（反量化在 kernel 内）；②✅；③✅ 模型侧量化 ckpt + env 开关（VLLM_FL_KV_INT8
  已有）。**attention kernel 速度免疫已证**（带宽 bound 不成立）——收益全部
  来自扩池。
- **预期结果**：16K 同驻 33→64 → 若 per-step 速度保持，**16K 输出吞吐
  +50~90%**（1346→2000-2560）；4K 场景无感（0.46 峰值占用）。这是全场最大
  单项端到端收益。注意：更大 bs 推高单步 attn 时长（bs64/kv16k 3D=317GB/s vs
  bs32=294GB/s——带宽利用率反而升），净收益为正的概率高。
- **工程量**：2-3 人·天（老 P2 代码大部分可复用：池分配、ckpt 量化、env）。
- **风险**：105 题对拍硬门（历史 KV int8 parity 0.9667 过线记录在 lm_head int8；
  KV 的 73%=73% 证据面好）；16K 场景 bench 需用户明示才开。回退：env 一键关。
- **判定：可行（16K 场景第一杠杆）**。

## 候选 7：调度器 patch（LCF/准入/块保护）

- **机理**：V1 抢占受害者=最晚准入者，LCF（least-computed-first）策略在线浪费
  −24-43%（[RFC #54644](https://github.com/vllm-project/vllm/issues/54644)、
  [PR #53723](https://github.com/vllm-project/vllm/pull/53723)）。
- **适配性/预期**：**阶段 A 证伪了前提**——本栈实测 Δpreempt=0（全长准入不
  超发），抢占税不存在，LCF 无税可省。块保护（防被抢块 LRU 淘汰）同样无对象。
  剩余价值：混批步 B3（decode 行被拉进 2D）可用「decode 与 prefill 分步调度」
  缓解，但这改变调度节奏、影响 TTFT/吞吐平衡，且收益已被候选 1 的 2D 提速
  （1.55×）大幅稀释。**判定：存疑（不做）**。

## 候选 8：投机解码（DSpark）

- **机理**：`openbmb/MiniCPM5-2B-DSpark` 官方 draft（5 层 324M、7 token/步），
  vLLM 通用 dspark 路径可跑（[#57013](https://github.com/vllm-project/vllm/issues/57013)：
  接受长度 5.78、单请求 12.8 tok/s）。
- **预期**：**负收益**。四重不利：①吞吐赛 64 并发压满（官方博客数据：高 QPS
  下 draft model 1.4-1.8× 减速）；②2B 小模型 drafting 开销占比最高档（Berkeley
  研究：目标越小越亏）；③MagicDec 的「长上下文 KV 读主导」前提不成立（2 KV 头，
  KV 读占比低）；④greedy 对拍已知分歧 issue 链（#7627/#41758/#54928）。
  **判定：不可行（比赛口径）**。

## 候选 9：block_size / max_num_seqs / APC 微调

- block_size 16→32：长输入下池占用影响 <1%（SOSP'23 论文口径），收益符号依
  kernel 代际翻转，且本栈 attention 8KB 粒度 gather 与 16 对齐——**不动**。
- max_num_seqs：32GB 卡默认 256 档，64 并发从未触顶——**调了没用**。
- APC：保持开启（0 命中开销 <1%；跨轮 71%/43% 命中实测在账）；bench 纪律注意
  残留 cache 虚增（官方文档明示）。**判定：不动作**。

## 候选 10：变相扩池（graph 捕获内存压缩）

- **机理**：KV 池 = 0.85×显存 − 权重 − 激活 − cudagraph 捕获 − 编译缓冲（后
  两项不入池账目可代码级压缩）。与候选 6 正交叠加。
- **预期**：不确定（需查当前捕获内存占比；捕获列表已含 64 档）。**判定：存疑
  （先量化再定，0.5 人·天盘点）**。

---

## 推荐排序（供用户裁定）

| 序 | 候选 | 场景 | 预期端到端 | 工程量 | 累计口径 |
|---|---|---|---|---|---|
| 1 | 候选 1 TD 开启 | 双场景 | 4K +7~10%，16K +12~18% | 0.5-1d | 含 parity 门 |
| 2 | 候选 5 采样快路径 | 双场景 | 各 +4~6% | 1-2d | 恒等跳过零数值风险 |
| 3 | 候选 2 段数自适应 | 双场景 | 各 +3~5%（与 1 叠加 1.58×） | 1-1.5d | cudagraph 静态化 |
| 4 | 候选 6 KV int8 扩池 | 16K | **+50~90%** | 2-3d | 16K binding 场景需用户令 |
| 5 | 候选 4 prefill 形态 | 4K | +15~25% | 3-5d | 重工程 |
| 6 | 候选 3 自写 split-KV | 16K | +10~20%（边际） | 2-3d | 1/2 落地后重估 |

叠加粗算（不含相互稀释）：4K 场景 +18~26%（TD+采样+段数+prefill 部分）；
16K 场景 +70~120%（TD+采样+段数+扩池）。所有数字的推导链在
`results/20261115_bottleneck_deepdive/README.md` §5 可回查。

## 引用索引

FlashDecoding（[CRFM](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) /
[PyTorch 博客](https://pytorch.org/blog/flash-decoding/)，注：无正式论文，
arXiv 2310.10713 与其无关）· FlashDecoding++（[arXiv:2311.01282](https://arxiv.org/abs/2311.01282)）·
FlashInfer（[arXiv:2501.01005](https://arxiv.org/html/2501.01005)，[plan.cuh](https://github.com/flashinfer-ai/flashinfer)）·
unified_attention 解剖（[arXiv:2511.11581](https://arxiv.org/abs/2511.11581) /
[deep dive](https://vllm.ai/blog/2026-03-04-vllm-triton-backend-deep-dive)）·
2D/3D 门槛（[#48076](https://github.com/vllm-project/vllm/issues/48076) /
[PR #47520](https://github.com/vllm-project/vllm/pull/47520)）· multi-row split-K
（[#56148](https://github.com/vllm-project/vllm/pull/56148)）· use_td（[PR #40327](https://github.com/vllm-project/vllm/pull/40327)，
[Intel TD 回归 epic](https://github.com/intel/intel-xpu-backend-for-triton/issues/6704)）·
FA2（[arXiv:2307.08691](https://arxiv.org/abs/2307.08691)）· Sarathi-Serve
（[OSDI'24](https://www.usenix.org/system/files/osdi24-agrawal.pdf)）· 投机解码
（[官方博客](https://blog.vllm.ai/2024/10/17/spec-decode.html) /
[文档](https://docs.vllm.ai/en/stable/features/speculative_decoding) /
[#16258](https://github.com/vllm-project/vllm/issues/16258) /
[#57013](https://github.com/vllm-project/vllm/issues/57013)）· KV 量化
（[#33480](https://github.com/vllm-project/vllm/issues/33480) /
[FP8 KV 博客](https://vllm.ai/blog/2026-04-22-fp8-kvcache) /
[LMDeploy](https://lmdeploy.readthedocs.io/en/latest/quantization/kv_quant.html)）·
调度（[RFC #54644](https://github.com/vllm-project/vllm/issues/54644) /
[APC](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)）·
vLLM SOSP'23（[arXiv:2309.06180](https://arxiv.org/abs/2309.06180)）·
比赛口径（[FlagOS 赛季二](https://flagos.io/events?tab=competition) /
[MiniCPM5](https://github.com/OpenBMB/MiniCPM)）
