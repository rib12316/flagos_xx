# Track C：「KV Cache 管理 + 调度优化」假设台账（主攻 16K 场景）

- 赛道：FlagOS S2 赛道二（MiniCPM5-2B / 沐曦 C500 64GB MACA / 天数 BI-V150，vllm-plugin-FL `flagos-2026-s2` 分支，基于 **vllm==0.24.0**）
- 约束：只改 vllm-plugin-FL 与 FlagGems 代码；serve 启动参数锁死；评分 = `benchmark_throughput_serve.py` 的 **Total token throughput (tok/s)**（输入+输出合计 / 时长），4 轮取后 3 轮均值，任何一轮 `successful_requests != num_prompts` 则该 case 作废。
- 本文所有本地代码路径均相对 `C:\Users\Rib\.zcode\workspace\draft-1e9bed95-0ab6-48d4-b566-acac730de60b\vllm-plugin-FL\`。

---

## 0. 结论速览

| 项 | 结论 |
|---|---|
| 负载真相 | **随机 token（无共享前缀）+ ignore-eos 定长输出 + 客户端 64 并发闸门**。prefix caching 对本负载零命中；total tok/s 中 prefill 占 80%（4K）/94%（16K），**抢占重算的 token 不计分、纯浪费** |
| 16K 临界区 | 数学成立：64 并发 × 峰值 0.75GB/条 ≈ 47.9GB vs KV 池 ≈46–48GB，且 vLLM V1 准入只查 prompt 块数、不预留 decode 增长 → 波次切换期**必然进入 recompute 抢占区**（待开机测量证实） |
| 插件可改动面 | 插件**已整建制持有** Worker（gpu_worker 全拷贝）、ModelRunner（gpu_model_runner 全拷贝）、Platform（check_and_update_config 配置改写钩子）、以及 **AsyncScheduler 子类先例 + `scheduler_cls` 字符串注入先例**（worker.py:206-211）。自定义 V1 Scheduler 并在 platform.py 注入 = **框架内合法开发**，非改主库 |
| Top 假设 | H1 抢占免除式准入（decode 增长预留）> H8 cudagraph 覆盖核查 > H4 block_size 调优 > H2 抢占受害者选择 > H10 KV 卸载式抢占（创新分） |

---

## 1. 负载真相（bench 脚本逐行证据）

脚本：`benchmarks/benchmark_throughput_serve.py`。

### 1.1 关键机制（代码证据）

| 机制 | 证据 | 含义 |
|---|---|---|
| 数据集 = 随机 token | `build_common_args()` L157-177：`--dataset-name random` | vLLM bench random 数据集对每个 prompt 从词表均匀采样 `random_input_len` 个 token（range_ratio 默认 1.0，长度精确等于设定值）。**所有 prompt 之间无任何共享前缀，单个 prompt 内部也无重复结构** |
| 定长输出 | `--ignore-eos`（L176）+ `--random-output-len` | 每条请求恰好生成 output_len 个 token 后停止，无视 EOS。请求完成时间可精确预测 |
| 并发控制 | `--max-concurrency 64`（L326） | 客户端 asyncio 信号量：**在飞请求 ≤64**。一条完成立刻放下一条。4K: 256 条 = 4 个波次；16K: 128 条 = 2 个波次 |
| 接口 | `--endpoint /v1/completions` | 裸 completions；max_tokens=output_len。请求体只有随机 token_ids |
| 评分口径 | PATTERNS L180-198：`Total token throughput (tok/s)` = (Total input + Total generated)/Benchmark duration | **重算的 prefill token 不计入分子**：抢占重算 16K = 纯时间浪费；反之 TTFT/TPOT/ITL 全部不影响得分（可以自由牺牲延迟换吞吐） |
| 轮次 | `RUNS=4, SKIP_FIRST=1`（L47-50） | 每个场景跑 4 轮丢首轮取均值；**serve 进程常驻**，跨轮状态（KV 碎片、prefix 池）延续；调度器必须在 4 轮内无死锁/无失败请求（失败一条 → 该 case 整体作废） |
| 失败判定 | L377-394：`successful_requests == num_prompts` | 请求级失败是二值灾难：任何 OOM/断言/死锁直接出局。调度改动必须带守卫 |

### 1.2 对调度技术的直接推论清单

**无效/近零收益（不要投入）：**
- **prefix caching（APC）**：随机 token → 块哈希零命中。唯一残余影响是准入时对 1024 块/条做哈希树遍历的 CPU 开销与 cached 池管理（见 H6，属"关掉无用机制"而非"用好机制"）。bench 脚本用法注释 L19 甚至示例了 `--no-enable-prefix-caching`，但官方锁死的 serve 参数未知——开机必须确认。
- **cascade attention / 共享前缀类优化**：无公共前缀，metax 已 `disable_cascade_attn=True`（platform.py:256）。
- **PD 分离、多卡并行**：单卡赛制，无关。
- **输出长度感知的早停/投机**：ignore-eos 定长，无提前完成空间（MTP 若启用属 Track B 内核侧）。

**有效（按权重）：**
1. **杜绝/减少抢占重算**（16K 唯一大杠杆）：重算 token 不计分。
2. **波次切换期的准入节奏**：16K 只有 2 个波次，第二波何时插入 prefill 决定了「老请求 decode 增长 vs 新请求 prefill 需求」的冲突窗口。
3. **chunked prefill 的 chunk 尺寸与 decode 混批**：prefill 占计分 token 的 80%（4K）/94%（16K），大 chunk → MACA GEMM 效率高、kernel launch 少；decode 混入 prefill 批（Sarathi stall-free）避免 decode-only 小步。
4. **cudagraph 覆盖 decode 批尺寸**：64 并发下 decode 批 ≈64（波尾递减），若 56-64 不在 capture sizes 内，decode 全程走低效路径。
5. **块元数据开销**：block table 尺寸、块清零（`_zero_block_ids`）随块数线性放大，block_size 调优直接作用于此。

---

## 2. 插件可改动面盘点（本地代码证据）

### 2.1 插件已持有的调度/KV 相关资产

| 文件 | 内容 | 作用 |
|---|---|---|
| `vllm_fl/platform.py` | `PlatformFL.check_and_update_config()`（L176-260） | **官方配置改写钩子**。已在此改 `parallel_config.worker_cls`、`cache_config.block_size`（npu=128/musa=64/其余=16）、metax 禁 cascade attention 等。我们注入 `scheduler_cls`、watermark、block_size、cudagraph sizes 的天然落点 |
| `vllm_fl/worker/worker.py` | `WorkerFL`（gpu_worker 全量拷贝，头注 "Adapted from v0.19.0"，注释引用 "vllm 0.24.0's kernel_warmup"） | L205-211：**`vllm_config.scheduler_config.scheduler_cls = "vllm_fl.worker.scheduler_fl.AsyncSchedulerFL"`** —— 插件替换 vLLM 调度器类的已验证先例（MTP=1 + async_scheduling 时触发）。`determine_available_memory()` 完全插件自有：KV 池大小 = 0.85×total − (权重+激活峰值+graph) 由这里的 profile 决定 |
| `vllm_fl/worker/scheduler_fl.py` | `AsyncSchedulerFL(_AsyncScheduler)` | 子类化 vLLM `vllm.v1.core.sched.async_scheduler.AsyncScheduler` 的样板（当前仅包 xGrammar 批量填 mask） |
| `vllm_fl/worker/model_runner.py` | `ModelRunnerFL`（gpu_model_runner 全量拷贝，~7500 行） | 插件整建制持有：`_update_states()`（L1187+，处理 preempt/resumed 请求的 block_ids 替换路径 L1431-1441）、`_zero_block_ids()`（L1165）、`initialize_kv_cache*`（L7306+）、`capture_model()`（L6674）、`profile_run()`（L6323） |
| `vllm_fl/platform.py` L353-375 | `insert_blocks_to_device()` / `swap_out_blocks_to_host()` | 平台级 KV 块搬运原语（设备内/主机），H10 卸载式抢占可直接复用 |
| `vllm_fl/dispatch/config/metax.yaml` / `iluvatar.yaml` | attention 后端选择 | metax：`attention_backend: vendor:metax`（MACA flash attn）；iluvatar：`flagos → vendor:iluvatar → reference` |
| `vllm_fl/patches/` | moe_sum / qwen3_5_text / gdn_packed_decode 等 | **monkey-patch vLLM 内部函数的既定实践**（从 `register()`/`register_model()` 入口安装） |
| `pyproject.toml` L36-40 | entry-points `vllm.platform_plugins`→`register`、`vllm.general_plugins`→`register_model` | 官方插件注册通道 |

### 2.2 判断：插件内自定义/替换 V1 Scheduler 是否合法

**结论：合法，且是插件已内建的能力，不构成「改主库」。**

依据：
1. **规则字面**：铁律 = 只改 vllm-plugin-FL 与 FlagGems 的**代码**。在 `vllm_fl/` 包内写 `Scheduler(Scheduler)` 子类、并由插件入口把它设为 `scheduler_config.scheduler_cls`，一行主库文件都不动。
2. **vLLM 官方机制**：`scheduler_cls` 是 SchedulerConfig 的字符串字段，主库解析该字符串动态实例化调度器；主库源码对该字段的警告只是「非公开 API、跨版本可能变」——版本已锁死 0.24.0，风险可控。
3. **仓库先例**：`worker.py:206-211` 已经在用同一机制替换调度器（AsyncSchedulerFL）；`platform.py:180` 已经在改 worker_cls。若这算违规，插件主干自己先违规了。
4. **注入点选择**：优先在 `PlatformFL.check_and_update_config()` 设 `scheduler_cls`（API-server 进程构造 VllmConfig 阶段执行，随 config 序列化传入 EngineCore，单卡/多卡皆生效）；worker.py 内的改写在 TP=1 uniproc 下也生效（worker 与 EngineCore 同进程），但平台钩子更稳。
5. **边界（不要越）**：不修改 `site-packages/vllm/` 的任何文件；不改 benchmark 脚本；serve CLI 参数一律不动。所有行为变化经 `vllm_fl/` 代码 + 可选环境变量开关。

### 2.3 插件当前的空白

常规（非 MTP）推理路径完全使用 vLLM 0.24 原版 Scheduler/kv_cache_manager/block_pool——插件没有任何抢占、准入、批组成逻辑。这正是本赛道的作业空间。

---

## 3. 技术调研（对照 vLLM V1 机制 + 本负载适配性）

> 机制来源：vLLM main（0.24 之后继版本）`vllm/v1/core/sched/scheduler.py`、`vllm/config/scheduler.py` 源码 + V1 架构公开资料。0.24.0 与 main 的字段位置可能有漂移（如 `watermark` 从 CacheConfig 移到 SchedulerConfig、`scheduler_reserve_full_isl` 是否已存在），开机第一件事核对（§4）。

### 3.1 vLLM V1 抢占机制（已逐行确认 main 源码）

- 调度循环 `schedule()`：先 `schedule_running()`，再 `schedule_waiting()`。token 预算 = `max_num_batched_tokens`（decodes 优先排入，剩余给 prefill chunk = chunked prefill，V1 恒开）。
- **decode 增长不加保护**：`schedule_running` 对每条 running 请求 `allocate_slots(req, num_new_tokens=1...)`；返回 None（自由块+可逐出块不足）即触发抢占。
- **受害者 = `self.running[-1]`**（FCFS 尾部 = 最新请求）。`_preempt_request()`：释放其全部块，**`num_computed_tokens = 0`（全量重算，V1 无 swap）**，`num_preemptions += 1`，请求 `prepend` 回 waiting 队首。
- **准入只查 prompt**：`schedule_waiting` 调 `allocate_slots(..., full_sequence_must_fit=scheduler_reserve_full_isl)`，即使 reserve_full_isl=True 也只校验 **input sequence（prompt 已生成部分）** 能放下，不预留 max_tokens 的 decode 增长。
- watermark：`BlockPool.can_allocate` 要求 `free - required > watermark_blocks`（`watermark_blocks = watermark × num_gpu_blocks`；main 默认 0.0，≤0.13 时代 CacheConfig 默认 0.01）。

**16K 抢占风暴成因（结合负载推演）**：64 条全速 prefill（16384/8192 ≈ 2 chunk/条，token 预算串行）+ 交错 decode → 波次 1 尾段 64 条同时逼近 17408 token 峰值 ≈ 47.9GB > 池子；此后每有一条波次 1 请求完成（释放 1088 块），FCFS 立刻准入一条波次 2 的 16K prefill（需 1024+ 块）→ 自由池回到零 → 波次 2 新请求 mid-prefill 被 `running[-1]` 选中抢占 → **已算的 8192-token chunk 全部作废重来**。每次这类事件浪费 ≈0.5–1 个计分 token 等效的 8K chunk（不计分但耗时）。若发生 10–20 次 ≈ 浪费 5–10% 总时长。

### 3.2 抢占/重算策略 → 适配性

| 策略 | 适用性 | 落点 |
|---|---|---|
| **准入时预留 prompt+max_tokens**（本负载 max_tokens 恒 1024，Request 对象自带 `max_tokens`） | **高（H1）**：把「乐观准入+事后抢占」变「悲观准入+零抢占」。46GB 池 / 0.75GB 全程需求 ≈ 只能同驻 61 条（vs 64），损失 61/64 的 decode 批并行 ≪ 抢占重算的浪费 | 新 `SchedulerFL.schedule_waiting` |
| 受害者选择最小化浪费（argmin num_computed_tokens 而非 running[-1]） | 中（H2）：H1 生效后仍可能在波尾瞬时压力下触发，选「已算 token 最少」者抢占可把单次浪费从 ~16K 降到 ~1K | `SchedulerFL.schedule_running` 抢占循环 |
| 水位线调高（防抢占廉价实验） | 低-中（H7）：一行改动，作为 H1 的标定对照 | platform.py 设 `watermark` |
| KV 卸载式抢占（swap 复活） | 研究项（H10）：平台已有 `swap_out_blocks_to_host` 原语；vLLM 恢复路径支持 `resumed_req_ids` 整表替换 block_ids（model_runner L1431-1441） | 创新分主打 |

### 3.3 chunked prefill（Sarathi-Serve 思想）

Sarathi-Serve（OSDI'24）：prompt 切成 512–2048 的均匀 chunk，与 decode 混批组成 stall-free 调度；chunk C 直接控制 TTFT–TPOT–吞吐三角，过小 → 碎开销，过大 → decode 饥饿。**本负载无延迟 SLO（total tok/s 唯一评分）→ 应取偏大 chunk**：V1 已是「decodes 先排、prefill 吃满剩余 token 预算」的 Sarathi 型调度，默认形状已接近最优；可做的只剩 (a) 确认 `long_prefill_token_threshold`（main 新字段，限制单条长 prefill 每步 chunk）未把 16K chunk 切小；(b) 波次尾段主动让 prefill 提前插入（归入 H1 准入控制器的动态余量设计）。4K 与 16K 都受益但幅度有限（<2%）。

### 3.4 block size

默认 16（platform.py:193 对 metax/iluvatar 明确设 16）。trade-off：
- 调大（32/64/128）：块表条目数 /req 从 1088 → 136（16K 场景），`_update_states` 的 block table 追加、CachedRequestData 序列化、逐块清零调用数全部同比下降；vendor paged-attention kernel 的页内访存连续性更好。
- 代价：尾部碎片 = 每条平均 block_size/2 个 token（64 条 × 64 token × 42KB ≈ 0.17GB，可忽略）；需验证 metax flash_attn 与 iluvatar FlagGems attention 的页粒度约束（NPU 系常见 128 对齐、CUDA 系 16 即可）。
- GQA 2 KV 头特殊性：每 block 每 KV 头仅 block_size×128×2B 的 K/V，2 头意味着单 block 很「窄」（16 token block = 8KB/头/层×2），过小 block 会放大 gather 次数；16→64 是低风险第一步。
- **精度零风险**（纯布局），预期能量级 1–3%，两场景同受益（16K 块表更大、收益略高）。

### 3.5 批组成/准入收缩

「64 并发要不要主动降到 KV 不溢出」：**要，但只降准入不降客户端并发**——客户端信号量恒 64，服务器端把同驻上限收到 ~61（由预算/全程序需求动态算出），等待队列自然积压，波次衔接更平滑。这就是 H1 的动态形式。静态 `max_num_seqs` 属 serve 参数（锁死），代码内做软上限即可绕开。

### 3.6 其他

- **KV 释放时机**：V1 在请求 finished 的下一个调度步整批释放块（APC 开启时转入 evictable 池）。无迟滞问题；不做。
- **free-block 分配局部性**：BlockPool free list 是 LIFO/分配顺序驱动，跨请求块地址高度交错，paged attention 是 gather 访存（本来就随机），收益存疑，暂不立项（可作为 H4 的附加观测）。
- **`needs_kv_cache_zeroing`**（H5）：新分配块逐块 memset（防脏数据 NaN）。16K 全程要分配 ~46GB 的块；若两平台 attention kernel 的 partial-block masking 正确，可在插件侧关掉（KVCacheConfig/ModelRunnerFL 路径），省 memset 带宽 + 每步宿主循环。精度风险非零（依赖 kernel 正确性）→ 必须门控 + logprobs A/B 验证。
- **APC 机制旁路**（H6）：若开机发现锁死参数开了 prefix caching，准入时跳过 1024 块/条的哈希树查找（反正零命中）省调度 CPU。确认默认关闭则此条作废。

---

## 4. 可观测性：开机后第一个 30 分钟

### 4.1 前提核清单（serve 启动日志，~2 分钟）

启动日志（`WorkerFL.determine_available_memory` / engine 初始化）直接回答所有版本漂移问题：

1. `Available KV cache memory: X GiB`（worker.py L614-617 `logger.info_once`）→ **KV 预算实测值**。
2. `GPU KV cache size: N, ... tokens` / `Maximum concurrency for ... tokens per request` → 块数与块大小（确认 block=16 与 N×16×42KB ≈ X GiB 自洽）。
3. `Chunked prefill enabled with max_num_batched_tokens=K` → chunk 预算 K（预期 8192 或平台默认）。
4. `Prefix caching is enabled/disabled` → 决定 H6 是否存在。
5. cudagraph capture sizes 列表 → 64（及 40–64 区间）是否被覆盖（H8）。
6. `max_num_seqs` 实际值（V1 默认 1024 或平台覆写）。
7. （代码侧一次性确认）`python -c "import vllm; from vllm.config import SchedulerConfig as S; print([f for f in ('scheduler_reserve_full_isl','watermark','long_prefill_token_threshold') if hasattr(S,f)])"` → 0.24.0 有哪些新字段。

### 4.2 「16K 确实在抢占」证实方案（~25 分钟）

跑一次 16K case（`--test-cases '[[16384,1024,64,8]]'` 小规模或直接 128 条），四路证据交叉：

1. **Prometheus**：`curl localhost:8000/metrics` 每 5s 抓一次，盯
   - `vllm:num_preemptions`（V1 若导出）或等价 preemption 计数 > 0 即实锤；
   - `vllm:gpu_cache_usage_perc`（= `kv_cache_manager.usage`）持续 ≥0.95 并触顶；
   - `vllm:num_requests_running` vs `vllm:num_requests_waiting`：waiting>0 且 running<64 = 准入被内存卡死。
2. **服务端周期日志**：INFO 级每 10s 打 `Scheduler stats: ... Running: x, Waiting: y, GPU KV cache usage: z%`；同时 `grep -ci preempt serve.log`。
3. **bench 侧旁证**：`vllm bench serve` 输出的 TTFT 分布——抢占请求 TTFT 爆炸，**p99/median TTFT 比值 > 2 且呈双峰** 即抢占指纹（重算使 TTFT ≈ 2×prefill 时间）。Duration 异比拉长同理。
4. **无侵入对照实验（量化上限）**：另起一个**仅用于测量**的 serve 实例，加 `--max-num-seqs 56`（开发测量不受锁死约束），若 16K total tok/s 上升 ≥3% → 抢占损失坐实且 ≈ 该差值 + 56<64 的并行损失，H1 的收益上界拿到。

> 通过标准：证据 1 或 2 命中 + 证据 3 双峰 → H1/H2 立项；全部未命中且 4K/16K 都跑满 → 瓶颈在内核侧（Track B），调度线转 H4/H8/H9 低风险项。

---

## 5. 假设台账（按 16K 预期收益 × 风险排序）

> 通栏：所有条目均为调度/布局类改动，**对生成 token 的数值路径零触碰，精度风险 ≈ 0**（H5 除外，已标注）。预期收益以「total tok/s 相对提升」计，需基线测量后校准。

### H1 抢占免除式准入：prompt + max_tokens 全程预留 【16K 主攻】
- **落点**：新文件 `vllm_fl/worker/scheduler_fl.py` 增加 `SchedulerFL(Scheduler)`，覆写 `schedule_waiting` 准入条件：`free_blocks ≥ ceil((prompt_len + max_tokens)/block_size) + margin`（max_tokens 取 `req.max_tokens`，本负载恒 1024）；池子全空仍放不下时回退原版逻辑防死锁。注入：`platform.py::check_and_update_config` 设 `scheduler_config.scheduler_cls = "vllm_fl.worker.scheduler_fl.SchedulerFL"`。
- **机制**：乐观准入→悲观准入，消灭 recompute 抢占（§3.1 推演的波次切换 thrash 归零）。
- **预期**：16K +5–12%（若基线确有抢占；收益 = 抢占浪费占比）；4K 0%（无压力）。
- **风险**：预留过度 → 同驻降到 61、decode 批略小（净收益仍正）；实现风险低（单函数覆写）。
- **工作量**：1–2 天（含开关环境变量 `VLLM_FL_ADMISSION_RESERVE`）。
- **前置测量**：§4.2 证实 preemption > 0。

### H2 最小浪费抢占受害者选择 【16K 备手】
- **落点**：`SchedulerFL.schedule_running` 抢占循环：受害者从 `running[-1]` 改为 `argmin(num_computed_tokens)`（并列取最新者）。
- **机制**：H1 之后残余的瞬时压力下，把单次抢占浪费从 ~16K token 降到 ~1K。
- **预期**：16K +0–3%（H1 的补丁）；4K 0%。
- **风险**：几乎无；注意不要抢占正在等 async KV 事件的请求（复用原版守卫）。
- **工作量**：0.5 天。**前置**：同 H1。

### H3 波次尾段 prefill 提前插入（动态余量准入） 【16K】
- **落点**：`SchedulerFL` 准入控制器的余量项：`margin = f(当前 running 的剩余 decode 需求)`——波次 1 收尾（多条接近 17408）时主动放宽，让波次 2 prefill 与剩余 decode 重叠，填满 token 预算。
- **机制**：16K 只有 2 波，波间空转窗口（decode-only 小批步）对 total tok/s 是纯损失；Sarathi 混批思想的「何时插入」侧。
- **预期**：16K +1–4%；4K +0–2%（波次更多）。与 H1 同文件实现。
- **工作量**：H1 之上 +0.5 天。

### H4 block_size 16 → 64（metax/iluvatar）【双场景】
- **落点**：`platform.py::check_and_update_config`（L192-193 现设 16 处），改由环境变量 `VLLM_FL_BLOCK_SIZE` 驱动 {16,32,64,128} 扫描。
- **机制**：块表/清零/元数据开销随块数线性降；页内连续性提升（§3.4）。
- **预期**：+1–3%（16K 略高）；需 vendor attention 页粒度验证。
- **风险**：布局类，精度 0；若 vendor kernel 有对齐要求则回退。
- **工作量**：0.5 天 + 扫描。

### H5 关闭 KV 块清零（门控）【双场景，唯一精度风险项】
- **落点**：`vllm_fl/worker/model_runner.py` `_zero_block_ids`（L1165）与 KVCacheConfig `needs_kv_cache_zeroing` 的插件侧覆写：平台环境变量门控跳过。
- **机制**：16K 全程 ~46GB 新块分配的 memset + 宿主循环开销（部分后端正确 mask 时属纯浪费）。
- **预期**：+0.5–2%；**精度风险非零**（依赖 metax flash_attn / iluvatar FlagGems attention 对脏块的处理）→ 必须 A/B logprobs 全序列比对后才可默认开。
- **工作量**：1 天（含验证）。

### H6 APC 机制旁路【条件立项：开机确认 prefix caching 开启】
- **落点**：`SchedulerFL.schedule_waiting` 在 APC 开启时跳过 `get_computed_blocks`（随机 token 零命中）。
- **预期**：省准入 CPU（1024 块哈希/条 × 128–256 条）；量级 <1%，4K（256 次准入）略高。若默认已关 → 作废。
- **工作量**：0.5 天。

### H7 watermark 扫描【廉价标定，兼独立小收益】
- **落点**：`platform.py` 设 watermark（0.24 位置待核：CacheConfig 或 SchedulerConfig）∈ {0, 0.01, 0.03, 0.05}。
- **预期**：单独看 ±1%；主要价值是给 H1 的 margin 取值提供实测曲线。0.5 天内与 H4 一并扫描。

### H8 cudagraph 覆盖 decode 批尺寸【双场景，先核查再动手】
- **落点**：`platform.py::check_and_update_config` 对 `compilation_config.cudagraph_capture_sizes` 的补充（插件对 compilation_config 已有既定改写先例）；确认 40–64 区间被覆盖。
- **机制**：64 并发 decode 批（波尾 64→1 递减）若落在 capture 缺口，decode 步退化为 eager/pad，损失可达两位数百分比。
- **预期**：若缺口存在 → +5%+；若已覆盖 → 0。**开机 §4.1 第 5 项 30 秒核查**，性价比最高的一条。
- **工作量**：核查 0；补 sizes 数小时（需权衡多 capture 的显存占用，worker.py 已有 cudagraph 内存预估路径）。

### H9 异步调度（async_scheduling）插件内启用【4K 侧】
- **落点**：`platform.py` 设 `scheduler_config.async_scheduling = True`（serve 参数锁死→代码内启用）；调度器换用插件已有的 AsyncSchedulerFL 一系。
- **机制**：调度 CPU 与 GPU 步重叠。16K 步长大（8192-token chunk）收益小；4K decode 段步长 ~几 ms，重叠价值中等。
- **预期**：4K +0–5%；16K ~0。风险中等（MACA/corex 流与事件语义、与 H1 调度器的组合需回归）。1–2 天。

### H10 卸载式抢占（recompute → host swap）【创新分主打，16K】
- **落点**：`SchedulerFL._preempt_request` 改为：用 `PlatformFL.swap_out_blocks_to_host()`（platform.py L365-375，现成原语）把受害者 1088 块搬到主机 64GB RAM，标记 paused；恢复时 `insert_blocks_to_device` 回填 + 走既有 `resumed_req_ids` 整表替换路径（model_runner L1431-1441 已支持）。
- **机制**：抢占成本从「16K token 全量重算（占 GPU）」变「0.7GB PCIe 搬运（~30ms，不占 GPU）」。临界区内抢占不可避免时的最优解，也是「系统级调度」叙事的最佳素材（vLLM V1 明确砍掉了 swap，我们以插件形式在特定负载下复活它）。
- **预期**：若基线抢占频繁（§4.2 证据 4 显示损失 >5%）→ 与 H1 二选一或叠加 +3–8%；若 H1 已把抢占清零，则 H10 仅剩保险价值 → **先测后做**。
- **风险**：工程复杂度最高（块表重建、OOM-on-host、与 4 轮连跑的状态残留）；精度 0。3–5 天。

### 排序总表（16K 收益预期 × 风险）

| # | 假设 | 16K | 4K | 风险 | 工作量 | 前置 |
|---|---|---|---|---|---|---|
| 1 | H1 准入预留 | +5–12% | 0 | 低 | 1–2d | 抢占证实 |
| 2 | H8 cudagraph 核查 | 0~+5%+ | 0~+5%+ | 低 | ~0–1d | 开机日志 |
| 3 | H4 block_size | +1–3% | +1–3% | 低 | 0.5d | 无 |
| 4 | H3 波次 prefill 插入 | +1–4% | +0–2% | 低 | +0.5d | H1 |
| 5 | H2 受害者选择 | +0–3% | 0 | 低 | 0.5d | 抢占证实 |
| 6 | H10 卸载式抢占 | +3–8% | 0 | 高 | 3–5d | 抢占证实且 H1 不够 |
| 7 | H5 关清零 | +0.5–2% | +0.5–2% | 中(精度) | 1d | A/B 验证 |
| 8 | H9 async sched | ~0 | +0–5% | 中 | 1–2d | 回归测试 |
| 9 | H7 watermark | ±1% | ±1% | 低 | 0.5d | 无 |
| 10 | H6 APC 旁路 | <1% | <1% | 低 | 0.5d | APC 开启确认 |

---

## 6. 参考来源

- 本地：`benchmarks/benchmark_throughput_serve.py`；`vllm_fl/platform.py`；`vllm_fl/worker/{worker,model_runner,scheduler_fl}.py`；`vllm_fl/dispatch/config/{metax,iluvatar}.yaml`；`pyproject.toml`（vllm==0.24.0、entry-points）；git 分支 `flagos-2026-s2`（commit 13eb9be "based on empty vllm==0.24.0"）。
- vLLM main 源码（2026-11 抓取）：[v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)、[config/scheduler.py](https://github.com/vllm-project/vllm/blob/main/vllm/config/scheduler.py)、[single_type_kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/single_type_kv_cache_manager.py)。
- 抢占观测：[vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics)、[Optimization & Tuning](https://docs.vllm.ai/en/stable/configuration/optimization)（preemption 计数与 recompute 说明）、[metrics 设计文档](https://docs.vllm.ai/en/stable/design/metrics)。
- Sarathi-Serve：[OSDI'24 论文](https://arxiv.org/abs/2403.02310)、[USENIX PDF](https://www.usenix.org/system/files/osdi24-agrawal.pdf)、[chunk size 分析](https://ashwanijha.dev/blog/sarathi-chunked-prefill-what-the-paper-actually-says)。
- V1 调度解析：[K4i: Request Queues to Scheduler Output](https://k4i.top/posts/scheduler-request-queue-to-scheduler-output)、[V1 preemption 深读](https://labhub.hopto.org/blog/ai-platform/2026-08-12-vllm-scheduler-and-preemption?lang=en)、[vLLM 论坛: preemption option](https://discuss.vllm.ai/t/request-preemption-option/1672)。
