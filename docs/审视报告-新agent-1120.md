# C500 双机技术审视 · 2026-11-20

**交付按整包评估。** 将已有融合及配套的去拷贝作为一次优化，同机比较全部关闭与按适用形状全部开启的吞吐。双形状 +0.3% 门槛用于整包。历史单项 T1 用于解释结果，整包净收益待测。11 月 19 日交付锚点为 4K **6399.21**、16K **7509.87** total tok/s、105 题 **98.1%**。

截至 16:02，F-a 已修复并命中真实图。F-c 在同一 `.so` 上通过逐位、别名及两次 graph replay，接入真实 decode 输出仍是整包实测的前置工作。误判风险前三为 **F-c、D7 的 16K 外推、A-H4 的单形状收益**。新增算子研发集中在 **TileLang**，首选验证长上下文分块 prefill attention。

### 整包边界

| 整包成员 | 全开臂应验证的行为 |
|---|---|
| F-a rope+cat | 支持的 NeoX 形状在真实编译图调用 `fl_rope_qk_cat`，其余形状走已有实现。 |
| D13 norm+quant 与 A-H4 norm 优先级 | D13 在已配对站点、有 residual 且 M≤512 时融合。A-H4 是配套的实现选择，仅在实际调用对应 norm 时有机会生效。记录两条路径各自的命中，收益按整包实测一次。 |
| F-c decode 去拷贝 | 将已验分组视图接入真实 output slice，在 graph replay 核对输出、别名与 staging copy 计数。现有分支直接打开开关仍会回退。 |

在同一集成提交显式固定两臂各开关，用短 smoke 核对输出与路由后，关闭诊断探针做 T1。F-b、F-d 的 SiLU 姊妹项及 F-e 在本轮只有筛选记录。逐项表中的留档判决描述当时的独立实验。仅当整包出现精度问题或性能回退时，才通过逐项关开关定位原因。

## 逐项判决

「可信」指列出的配置、数据和决策可支持。「存疑」指归因或外推还缺证据。「需重测」指有明确的新证据和有界验证办法。下表中的 `32p/64p` 指请求数，两个 quick case 的 `max_concurrency` **都是 64**。下列 `R/` 代表 `results/20261120_c500_reprofiling/`，分支前缀标明证据所在位置。

| 项目 | 判决可信度与建议 | 依据和边界 |
|---|---|---|
| Host 同步审计、D12 | **可信**，关闭本配置稳态 decode 的逐站同步优化。**存疑**的是把剩余 host 成本也写成零。 | `feat/s2-host-audit` 的 `results/20261120_host_audit/{host_audit_table.md,host_audit_summary.txt}`。887 个 span 内 0.19+0.85=1.04 次/步，总阻塞约 2 µs/步。19 站包括必要 join 和 CPU 读。pageable D2H 全 span 为 0.09/步，纯稳态为 0。244 步窗口 busy 95.6%、步间 p50 2.8 µs。D12 的 95.7–98.4% 是最忙秒，不能代表 ramp、mixed 或全场均值。 |
| 采样尾 0.45 ms/步 | **可信**地观察到空档，**存疑**的是可全部回收。交给独立的采样提交开销假设是合理分工。 | 同一摘要 `[5]`，≥50 µs 间隔共 111 ms/244 步，最大一组 64.2 ms。0.45/26≈1.7% 是该窗口的理想上界。按 stream 求时间并集、关联 CPU launch/依赖、扣除异步重叠，才能给每个空档定价。D5 算法替换失败与这项归因相容。 |
| A-H4 IR norm | **可信**，按双门留档。**存疑**的是泛称无收益或全形状位一致。应记「64p 单形状观测过线，效果待复现」。 | `feat/ah4-irnorm-vllmc` 的 `scripts/ah4/`、A 机同工作树 `results/20261120_ah4/{t0_norm_ab.log,t1/bench_*}`。T0 M32/64/2048 为 1.10/1.14/1.24×。前两档两臂输出相同；M2048 `B_vs_A maxabs=7.81e-03`，超过容差的元素为零。serve 日志确认 IR priority 翻转。热轮详见下表。 |
| F-c attention staging copy | **可信**，前后邻核支持 42×3.30 µs/步。**旧合同结论已有实测反例**，serve 收益需重测。 | `feat/fc-copy-attribution` 的 `scripts/fc/{ATTRIBUTION.md,probe_direct_out.py}` 和 `results/20261120_fc/probe_matrix.txt`。V1 已对连续 q 使用 `empty_like(q)`。V7/V8 的错误优先级及 V5 的 `s_aux` 错误支持 Python 零基槽 12=out、20=s_aux。新实测见 `results/20261120_review/fa_review_fc_group.log`。 |
| F-a rope+cat | **可信**，已修复构造回归并在真实 95-node 图命中。T0 位精确的参照为现役生成核。 | `feat/fa-rope-cat-fusion` 的 `results/20261120_fa_ropecat/{t0_r2.log,t0_rope_cat.csv}`，有效范围 M32/64、含合法负位置。两核合一 1.61/1.54×，M2048 0.997×。以 41 层×3.46–3.52 µs 估算，每 decode 步约省 0.14 ms，26/44 ms 下约 0.55%/0.33%，mixed 和请求总墙钟会进一步改变比例。本次 serve 结果见末节。 |
| D5 采样快路径、低 BLOCK top-k/top-p | **可信**，归档已测实现。**存疑**的是 L2 可单独解释所有差值。 | `R/README.md`、A 机 `t1_sample_fast/`、`R/sampling_metax.csv`。T0 peaked 位一致、平坦分布有微差，serve 命中 4000+、零回退，ITL 32p 为 26.46 vs 26.05 ms，64p 两组也未改善。Cycler、eager 调度、profile 干扰和不同分母同时存在。5 档 Triton 变体在本栈有运行期 ATU 记录，保留隔离。 |
| D6 split-KV | **可信**，V0/V1/V2 各为 0.541/0.371/0.249×，V3 routed=0，缺有效比较。**存疑**的是把经验带宽当作硬上限。 | `R/attn_surface/{README.md,splitkv_probe_metax.csv}`、`scripts/d2/splitkv_probe_metax.py`。分子为逻辑 `S×kv_len×2(K,V)×2(KV heads)×128×2 B`，不含索引、重复读和 cache 命中；分母为同卡另一顺序读实测。1312–1384/1576–1599 GB/s 在长 KV 大批中约 83–88%。假定字节数与参照固定，剩余时间占比 12–17%，相应吞吐提升约 14–20%，再乘 attention 墙钟占比。bs8/4K 为 499/1101，约 45%。 |
| D7 手写 Triton、prefill 定价 | **可信**，当前实现慢于 MACA，归档该原型。**需重测**将 16K prefill 空间判零的依据。 | `origin/feat/d7-prefill-attn` 的 `scripts/d7/{prefill_gqa_t0.py,gemm_calib.py}`、`R/prefill_surface/{d7_t0_verdict.md,d7_serve_caliber_verdict.md}`。16 配置最高 57，MACA 同形状约 96；maxrel 5.3e-3 还略高于脚本头的 `<5e-3`。56 是已扫 GEMM 的最佳值。grid、算量和层数复核见下节。 |
| D8 INT8 KV | **可信**，16K 的扩池理由和本版去量化路径均不足以立项。 | 任务板 10:0x、`scripts/d5/d8_kv_dequant_smoke.py`。池 1,257,344 tokens、零 preempt。vendor 已有 group=8、page=256 的 INT8 接口，maxrel 5.4e-3（vs 反量化参考），bs64/16K 为 0.46×、扫 splits 最好 0.48×。问 vendor 应带上 332 vs 1274 GB/s 和本模型 BF16 页 16 的约束。模型精度仍需另验。 |
| D9 per-token quant | **可信**，M64/K2048、12288 独立 graph 复核支持原方向判决。37% 的链内比例和 serve 的 1.55 ms/步采用不同分母。 | `0231c70` 提交的 `R/d9_quant_metax.csv` 在 K12288 给出 113.41/50.94 µs，表观 2.23×。限定整除形状复核后，两组 bare 为 17.856/17.766 µs，最优 kloop 为 1.065/1.066×。K2048 bare 均为 7.680 µs，变体均较慢。各候选逐位通过，均低于 1.3× 立项门。旧 CSV 的计时出处待核。`results/20261120_review/fa_review_d9_shapes.log`。 |
| D10 4K 方差 | **可信**，存在大幅状态依赖，保留冻结参数。**存疑**的是「单调」「均由 APC 造成」及「普适 16% 噪声底」。 | 台账 §9、`results/20261119_d4_metax_port/final/SUMMARY.md`。32p/64p/官方 256p 曾见 1.1%/41%/16% 散布，数列本身不单调。74.7% 是特定 quick 会话的缓存记录，16K 的 APC=0 也不能外推所有 4K pass。需按 pass 比较命中 token 数、池压力、排队及 phase。 |
| D13 norm+quant | **可信**，五件套已交，本轮双门未过。**存疑**的是「效应必然小于噪声」和「多数步 mixed」。 | `origin/feat/d13-norm-quant-fuse:results/20261120_c500_reprofiling/d13_norm_quant_fuse/WIRING.md`、任务板 11:00/14:1x。83 站、6/6 单测、接线位精确、贪心对拍、graph 1.15×。T1 共四会话（臂×形状），每格一个 serve。M≤512 按 token 数路由，配对数不能证明实际融合步数。早期 K6144 单点有 1/25M 差异，M4096 曾 0.894×。记录每个 M 档占用的**墙钟**后，才有理由申请官方尺度抽样。 |
| TileLang 校准、阶段 2 | **可信**，驻留 GEMM 校准过 100T 试点门。**存疑**的是把 2.07× 用作实际算子收益。 | `origin/feat/tilelang-pilot:results/20261120_tilelang/README.md` 和 `scripts/tilelang/gemm_calib_tilelang.py`。116/56 是 TMAC/s，传统 FLOP/s 是 232/112。Triton 校准的 104 CTA 写同一 C，TileLang 按 CTA 分块写且 C 为 fp32，循环计算指标可作筛查，访存与写回不同。D7 attention 脚本另乘 2 计 FLOP，且按完整 Q×KV 计数。先统一有效 causal MAC、softmax 与页访问口径，再与真实 MACA 算子比。 |
| F5/F6、替换扫描及其余机会 | **可信**，调度和图模式已开启，保持现行设置。未测候选记为估算。 | 任务板 10:15 显示 async enabled、decode FULL 37、mixed PIECEWISE 49。`feat/s2-opscan-audit:results/20261120_s2_opscan/audit_table.md` 有 SiLU 邻融合、F-b/cache 和 F-e 筛选依据。P1-B3 w8a8/mctlass 的现役收益成立。D11 为机会清单，开源报告中的推测、其它设备加速比和 HTTP 200 不能替代本机 T0。 |

## 三项优先复核

1. **F-c 的“合同不可满足”最易用反例检验，也是今晚最可能回收现成开销的归档项。** 公开的 FlashAttention v2.6.3 `csrc/flash_attn/flash_api.cpp` 中，`mha_fwd_kvcache` 在 `seqlen_q=1, Hq>Hkv` 等条件下先把 Q 改成 `[B,Hq/Hkv,Hkv,D]`，随后按更新过的尺寸检查 out，末尾再转回原布局。对本例，应把下游连续缓冲区 `dst` 以 `dst.view(B,2,8,128).transpose(1,2)` 传入，即 `[B,8,2,128]`、stride `[2048,128,1024,1]`。上游只是提出该变体的依据，vendor fork 需要实测。Python wrapper 的完整实参表和 V5/V7/V8 支持两个 optional 的映射；比较 C++ 局部 q 与 Python 入参 q 时要记录检查发生的时刻。验收看 `dst` 的逐位结果、返回指针是否别名、graph replay 是否覆写、serve copy 次数是否真的减少。构造探针 5–15 GPU 分钟，trace 10–20 分钟，T1 另计。零额外开销时最多省 0.139 ms/步，约 0.53% 的 26 ms decode。
2. **D7 的“16K 无空间”是方法论判死链里最弱的一环。** `scripts/d5/trace_wallmap_v2.sh` 实际发 **32×4096 输入、256 输出**。`scripts/d5/prefill_surface_metax.py` 的 S16×q2048 创建 **32,768 个新 Q tokens**；该 serve 调度预算为每步 2048，4K 流量也没有 16K KV。`parse_prefill_shapes.py` 只读 kernel 名、grid、duration，无法还原每条请求的 `cu_seqlens_q` 和 KV 长度。437.6 µs、1428 次是该 bucket 的实测，45.6 ms 是另一个工作量。模型有 **42 层**；即便同一形态覆盖每层，437.6 µs×42≈18.38 ms，原 28 层导出的 12.3 ms/1.28 s 两个投影均失去依据。先取 `sum(q_i)`、KV 长度分布、因果有效 pair 数、每步该 kernel 的实例数与 phase 墙钟；抽真实 16K 续算诊断窗口需约 15–30 GPU 分钟。现有手写 kernel 维持归档。
3. **A-H4 的 64p 正向信号仅有一对独立启动。** 全部三个热轮的区间与基线分开，且 64p 的同会话热态散布可以低至 0.14%。优先在整包 smoke 记录 D13 接管后仍调用该 norm 的站点。若须为整包回归做定向消融，两对 AB/BA 双形状预算约 1–1.3 GPU h。

第一项的分组视图验证已于本次完成。`dst` 与返回对象首地址相同，BF16 输出逐位相等，两次 replay 均覆写初始化的 NaN，第二次翻转 Q 后仍与新参考相等。F-c 功能分支旧补丁仍传外部 Q 形状，开 env 即回退。接入后在真正的 output slice 上验证别名，以 trace 核对消除的 copy 数，吞吐随整包测量。

两个诊断的可复现入口为 `results/20261120_review/fa_review_fc_group.py <F-c工作树>` 及 `results/20261120_review/fa_review_d9_shapes.py <0231c70工作树>`。F-c 沿用原九臂探针的输入、seed 及 ABI，D9 沿用原分支的 bare/变体与 graph timer，仅测整除的 K 和两个独立 seed。

## T1 的分辨率

**D10 的 41% 包含丢弃轮。** 对应的 64p 原始总吞吐为 4956.27 / 7019.43 / 7014.54 / 7009.52。若用最快比首轮，差 41.6%，按后三轮极差除以中位，只有 **0.141%**。官方 4K 的 16% 则来自同一 serve 的三个计分轮。启动间配对方差需另外采样。归档见 `results/20261120_review/raw_runs_20261120_031010.csv`。下表 A-H4 的四份原始 CSV 及精度日志见 `results/20261120_review/ah4/`，文件名 123048/123646 为 base，124838/125438 为 treat。

| 项目 | 32p 热三轮（base → treat） | 64p 热三轮（base → treat） |
|---|---|---|
| A-H4 total tok/s | 5932.30 / 5965.28 / 5931.62 → 5916.60 / 5937.37 / 5949.38；中位 **+0.0855%** | 6991.17 / 6989.17 / 7001.86 → 7013.36 / 7021.72 / 7010.64；中位 **+0.3174%** |
| D13 output tok/s | 1206.38 / 1208.50 / 1208.39 → 1195.60 / 1206.07 / 1205.16；**−0.267%** | 1422.68 / 1422.46 / 1423.37 → 1422.70 / 1421.72 / 1410.06；**−0.067%** |

A-H4 的 64p 三个治疗值全高于三个基线值，值得保留该方向。同一 PID 的三次测量共享 APC、编译态和启动顺序，独立样本数仍是每格一台次。32p 两臂的热轮极差约 0.55–0.57%，64p 约 0.16–0.18%。D13 治疗臂极差约 0.87–0.89%，基线臂约 0.06–0.18%。这些都是观察到的极差，不能当配对标准差或显著性界。4096:1024 的固定请求若全数、足长成功，total/output=5，两指标的臂内倍率相同；跨机绝对数先统一口径。A 机 5932 与 B 机历史 6020 只是约 1.5% 的方向锚，其中 6020 为历史均值。A-H4 的 norm 池仅约 0.60 ms，以 M32/64 的倍率推算能省约 0.055/0.074 ms/步。D13 板上 +0.4–0.6% 也是条件估计。效应小与测量分辨率不足可以同时成立。

**建议协议与机时**

1. 保留原件、冻结启动参数及 case，主指标 `Total tok/s`。同一机器、插件 SHA、同种 seed/负载、每臂相同编译缓存初态与预热。预先固定形状顺序或按区组平衡。每个会话 4 pass 丢首，保留所有原始值、成功/失败数、输入/输出 token 总数，区分脚本官方均值与 T1 热三中位。失败或服务被杀记录整组原因并按事先规则成对重做。
2. A 为现役路径（本轮整包开关全关），B 为整包开启。每对用同一台机器上相邻的两次独立 serve 启动，多对时平衡 AB/BA 顺序。A/A 可额外估计启动间噪声。保存按 pass 的 prefix hit/query **增量**、KV 水位、preempt、M 分布、TTFT/ITL。固定种子产生的暖 pass 也属于合法数据，按缓存状态分层报告，避免事后挑轮。
3. 每形状/每对计算 `d=log(median_treat/median_base)`，以 serve 对为重采样单位。报告全部配对值、几何均比与配对 95% 区间。计划 4 对作 pilot，再按配对标准差 `s` 定总量。用双侧 5%、80% 把握度区分 **0 与 +0.3%** 的正态近似 `n≈(2.8s/0.003)²`，若 `s=0.2/0.3/0.5%`，分别约 4/8/22 对；小样本的 t 临界值会增加需求。证明超过 +0.3% 需按真实效应与门槛之差另算。双形状均报：点估计过双门才有资格，CI 跨门标「分辨率不足」，上界低于门可排除有意义收益。固定样本或预先定停止规则。
4. A-H4 实际 32p 每 pass 约 28–40 s，64p 47–55 s。另有逐次拉起 bench 客户端、serve 编译/捕获的固定费用。从 12:30/12:36、12:48/12:54 两组 raw 文件及 serve 完成时刻，单臂双形状预算 **15–20 min**，一对 **30–40 min**。4 对约 2–2.7 GPU h，8 对 4–5.3 h，22 对 11–15 h。加长同一进程 pass 只能减少臂内抖动，不能替代独立重启。
5. 官方尺度仅在用户逐轮明示批准后由 A 机安排，候选为通过 T1 的整包。4K `256p`、16K `128p` 每形状/臂独立新 serve。依 11-19 原始 total/s，负载约 205 s 和 297 s/pass。复刻既有外层协议的实际墙钟为 21:51:16 至 23:20:13，约 **89 min/臂**（另加首个启动），双臂预算 **3–3.3 GPU h**，精度另计。`final/raw_runs_20261119_*.csv` 中四个 4K 文件各一条成功行，四个 16K 文件各两条，`final/SUMMARY.md` 的 16K 轮值采用各文件第二行。复核时须保留外层轮号及内层行号。单次官方对照可作量级抽查，4K 的历史 16% 极差使其难以单独裁定 0.3%。

## 后续优先级

| 时限 | 顺序、预期收益与预算 |
|---|---|
| 今晚 23:59 前 | A 机将已有实现接到同一提交，补 F-c 的真实输出接线，然后核对整包路由与正确性。接线后的 trace 预算 10–20 GPU 分钟，首对整包双形状 T1 预算 30–40 GPU 分钟，集成开发另计。F-c 对 26 ms decode 步的理想上界约 0.53%，整包吞吐由实测给出。B 机可先用 15–30 CPU 分钟整理已有 prefill 记录，标出 TileLang 选题缺失的 q/KV 信息。 |
| 明日（赛后） | 整包结果若接近门槛，完成 4 对整包 AB/BA，总计约 2–2.7 GPU h。新增研发只投入 TileLang。先取 15–30 GPU 分钟的长上下文分块 prefill 诊断窗口，记录每请求新增 q、已有 KV、页布局及 MACA Flash 的关键路径时间占比 `f`，覆盖 `Σq_i≤2048` 的真实调度步。 |
| 长期 | TileLang 首个原型针对时间占比最大的真实 prefill bucket，与现役 MACA Flash 在相同输入、页表、因果 mask、精度下比较。每个代表性 bucket 估半天开发、1–3 GPU h。若实测加速 `s`，该 bucket 对总墙钟的预计节省比例为 `f×(1−1/s)`。多 bucket 按互不重叠的占比加权，数值正确且预计收益足够时，以融合包固定开启、TileLang 关/开测增量 T1。2.07× 仅是 TileLang 与 Triton 的驻留 GEMM 校准比，真实 attention 加速待测。 |

## 本次验证与交付

F-a v3 的构造异常在 `3a2fb50`、真实 `VllmConfig` 上复现。第一次 `REGISTER total=2` 已证明 `ApplyRotaryEmb.forward_static(x, cos, sin, True)` 经 fake 追踪成功。`flat_outputs` 两值均返回同一图，第二次注册抛 `RuntimeError: Duplicate pattern`。旧逻辑还会把交错式 GPT-J 模式接到仅支持 NeoX 的 kernel。修复在功能分支限定已验形状/布局、只注册 NeoX、对齐 Q/K 的 `[T,-1,128]`；沿用同子进程注入与 backend 同步。新 EngineCore 的 56/95/46-node 图依次 `matched=1/1/0`，末图没有 cat。`/v1/completions` 的贪心 smoke 成功。

| F-a T1 total tok/s | 基线热三轮，中位 | 治疗热三轮，中位 | 单次配对结果 |
|---|---|---|---|
| 32p | 5911.20 / 5904.11 / 5946.49，**5911.20** | 5915.19 / 5918.98 / 5967.23，**5918.98** | **+0.132%** |
| 64p | 6999.93 / 6989.56 / 7009.09，**6999.93** | 7035.81 / 7048.01 / 7034.04，**7035.81** | **+0.513%** |

F-a 五件套为已有 impl、已注册 custom op、真实编译图替换、随库 T0 与本轮 T1、开发单 §8 与台账 §9 登记。测量版 `14db6f0`，先治疗后基线，各一个新 serve，两臂贪心输出 12 tokens 文本相同，16 个 pass 均足额且输入/输出 token 总数一致。功能分支 `results/20261120_fa_ropecat/review_t1/` 有探针和原始 CSV。两个生成的 Inductor 模块都调用 `torch.ops.vllm.fl_rope_qk_cat.default`，基线 EngineCore 明文记录 gate off。归档时 F-a 默认开关由 1 调为 0，复现实验显式设 1/0。A-H4 功能分支 `platform.py` 的实际默认仍为 1，单独装该分支时基线臂要显式设 0。

证据优先序为原始日志/代码、A/B 同机对照、台账与有时序的任务板、最后才是开源预期。11:49 A-H4 的零成功会话作废，后续有效日志每轮成功数足额。B 机早班 D13 并发编辑已经交接，采信最终 `WIRING.md` 的安装、配对与 smoke。F-a 本次用 `api_server` 形态、各臂独立 compile-cache 根、每次 `mx-smi` 空闲后起服，未改公开 benchmark。B 机分支中仅存汇总而缺本地 raw 的项目，均按二手证据标明。

## 四项融合整包实测（A 机晚班）

全关对四项全开，代码 `feat/c500-fusion-bundle@6c5eaa2`。F-a 合并 Q/K 旋转，D13 让归一化直接产生下游 INT8 和 scale，A-H4 调整余下 fused_add_rms_norm 的实现优先级，F-c 让厂商 decode 直接写目标视图。两臂各一次独立启动，并发 64，4096 输入、1024 输出，各取 4 轮中的热三轮中位。

| 请求数 | 全关总吞吐（tok/s） | 全开总吞吐（tok/s） | 变化 |
|---:|---:|---:|---:|
| 32 | 5892.74 | 6061.27 | +2.86% |
| 64 | 7016.42 | 7106.88 | +1.29% |

全开显式将 D13 上限设为 2048，编译产物含 D13 和 F-a 两个 custom op，服务记录有 83 处配对、A-H4 优先级及 F-c 直写启用。逐形状成功请求数及输入输出 token 数全部足额。4 条相同输入的贪心生成（各 24 token），英文、算术、中文在第 21、2、11 个生成 token 分叉，代码样例全同。这是一次顺序对照，效果尚待跨启动复现，精度影响待定。准备默认关闭的合入候选。功能分支 `results/20261120_fusion_bundle/` 有双臂原始 CSV、完整贪心响应与实验条件。
