# 任务③ host 开销审计表 —— 负判决（host≈0，任务③关账）

> 日期 2026-11-20｜分支 `feat/s2-host-audit`｜审计 agent：agent-host
> 数据源（同源）：`/data/flagos_wt/results/20261120_c500_reprofiling/trace_all/rank0.1795110863960817196.pt.trace.json.gz`
> （kineto/MACA mc* 口径，1027 decode 步，bs=32 稳态 887 步；纯 CPU 解析，未触 GPU）
> 复现：`python3 scripts/host/trace_host_audit.py <trace.gz> results/20261120_host_audit`
> 前置：async scheduling 与 FULL graph 已确认开启（B 机实锤，台账 §9 F5/F6——本审计未重查）；
> D12（launch/host 缺口不立案，稳态最忙秒 busy 95.7-98.4%）为背景判决，本审计做**逐点消除核实**。

## A. trace 侧定量（摘要，全量见 host_audit_summary.txt）

### A.1 稳态 decode 每步同步类 / D2H（诚实门输入）

| 类别 | 每步次数 | host 侧代价 | 说明 |
|---|---|---|---|
| mcStreamSynchronize | **0.19** | p50 3.2µs（no-op） | 6048 次总时长仅 19.3ms，99% 集中在首段捕获期 |
| mcEventSynchronize（span 内） | **0.85** | p50 1.3µs（no-op） | 见 A.2，span 外的长等待是 by-design join |
| mcDeviceSynchronize | **0.00** | — | 全 trace 仅 1 次（26ms，收尾） |
| **同步类合计** | **1.04 < 2** | **span 内合计 1.6ms / 887 步 ≈ 2µs/步** | **诚实门触发 → 负判决** |
| D2H pageable | 0.09（纯稳态段 **0.00**） | 4-8B 标量 | 83 次全部落在 t+11.9s 单秒（捕获尾巴）；3024 次总量 89% 集中在首段（捕获+预热+prefill 头） |
| D2H pinned | 0.71 in-span（含 span 外合计 1.0） | 128B/次，GPU 2µs | 采样 token 异步回读（async 输出处理），by design |
| aten::item | 3.0 in-span / 6.3 全窗口 | p50 **1.3µs**、p99 5.3µs | **全部 CPU 张量读**（无一次伴随 D2H） |
| H2D pinned | 5.2/步，722KiB/步 | 异步 copy engine | 静态 pinned buffer 已在位（#778 模式已是现状） |

### A.2 长等待的落点（为何 16s 的 eventSync 不是 host 开销）

- mcEventSynchronize 全 trace 阻塞 16.0s，其中 span 外 1301 次 = 15.995s，p90 **18.3ms**。
- 落点分桶：**GPU busy 秒（>50%）内 1864 次 / 15.57s**，低 busy 秒仅 0.42s → 长等待与 kernel 忙**共现**。
- 语义：`get_output()`（model_runner.py:348）等采样 token 的 pinned D2H 完成 + async 输出 join——
  等的是 **GPU 正在执行的活**（GPU-bound），不是 host 自造的串行化。删除它=破坏 async scheduling 正确性。

### A.3 流水线完整性（空隙归因）

- 稳态窗口（115.5-122.0s，244 步）：kernel busy **95.6%**；**步间气泡 p50=2.8µs、max=3.1µs → async 流水零 host 气泡**。
- 步内 gap≥50µs：**0.45ms/步（≈1.7%）**，全部为 eager 采样尾的 host-launch 缝隙，固定三处 seam：
  - `_to_copy_func → fill_scalar`：244 次 / 64.2ms（**264µs/步**，采样 python 守卫链 eq/or/all/is_nonzero + 小 kernel 间隔）
  - `fill_scalar → _index_jit`：228 次 / 12.4ms（54µs/步）
  - 末层 `fused_add_rms_norm_2 → _index_jit`：212 次 / 24.2ms（113µs/步）
- 其余 idle 为 <50µs 的 kernel 切换微缝（GPU 侧，图内节点切换，非 host）。
- MACA 驱动层 launch 内在开销：mcPointerGetAttribute 248/步 + mcGetDeviceProperties 53/步（合计 354ms host / 39.6s decode 活动），
  已被 busy 96% 吸收，只在上述缝隙显形——无插件侧修复点。

## B. 代码侧逐点判决（vllm_fl/ 全目录 grep `.item()|.cpu()|synchronize()|.tolist()|.numpy()`）

装载链核实：serve 用 `VLLM_PLUGINS=fl` 装载本仓库插件（`vllm_fl/platform.py` 注册 metax 平台），
`vllm_fl/worker/model_runner.py` 即 metax 实际执行的 worker model runner（7673 行，v1 API）。

| # | site | 在本 serve 每 decode 步路径? | 每步次数 | 代价 | 判决 | 修复状态 |
|---|---|---|---|---|---|---|
| 1 | worker/model_runner.py:348 `async_copy_ready_event.synchronize()`（get_output） | **在**（span 外，每步 1 次） | 1 | 阻塞≈GPU 剩余步时（p90 18.3ms），与 kernel 忙共现 | **by design**（async 输出 join 本体；:2902 注释即此设计——GPU 标量滞留避免同步） | 不修 |
| 2 | :455 同名（pooler 输出变体） | 否（无 pooler） | 0 | — | 判死 | — |
| 3 | :3832 `prepare_inputs_event.synchronize()`（synchronize_input_prep） | **在**（span 间，输入 prep） | ~1 | µs 级 no-op | **by design**（保护复用 CPU 缓冲，async H2D 正确性所需） | 不修 |
| 4 | :1821 `cu_num_tokens[cur_index].item()-1` | 在（输入 prep 循环） | 每请求 1 | ~µs | **判死（非同步项）**：cu_num_tokens 是 `np.cumsum`（:1745）numpy 数组，纯 CPU 读 | — |
| 5 | :2315 `optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()` | 在（attn metadata init） | 1 | trace 佐证 aten::max 5.7µs + item 0.7µs | **判死**：`_cpu` 张量 numpy max，纯 CPU | — |
| 6 | :3978 `num_tokens_across_dp[dp_rank].item()` | **否** | 0 | — | **判死**：DP=1 时 `coordinate_batch_across_dp` 早退返回 None（vllm dp_utils.py:197-199），分支不进 | — |
| 7 | :2108 `num_accepted_tokens_event.synchronize()` | 否（spec/mamba 路径，且 async 模式跳过） | 0 | — | 判死 | — |
| 8 | :2882/:4867/:4905 draft/sample_count 事件与 tolist | 否（spec decode） | 0 | — | 判死 | — |
| 9 | :7596 `_to_list` 的 `transfer_event.synchronize()+tolist()` | **否** | 0 | — | **判死**：仅 `not use_async_scheduling` 分支（:3721）调用，本 serve async 开启走 :348 事件路径 | — |
| 10 | :3067 LoRA connector tolist | 否（无 LoRA） | 0 | — | 判死 | — |
| 11 | :4717-18 `routed_experts_cpu.numpy()` | 否（MoE；Llama dense） | 0 | — | 判死 | — |
| 12 | :5621 `query_start_loc.np[req_idx].item()` | 仅 prefill prompt-logprobs | 0（decode） | — | 判死（`.np` CPU；decode 无 prompt logprob） | — |
| 13 | :1107/:2095/:2315 `optimistic_seq_lens_cpu` tolist/numpy | 在 | 1-2 | µs | 判死（CPU） | — |
| 14 | :354 `sampled_token_ids_cpu.tolist()` | 在（输出处理） | 1 | CPU pinned 读 | by design（join 之后） | — |
| 15 | platform.py:477 `swap_out_blocks_to_host` `.cpu()` | 否（KV 换出；本场景零 preempt，P1-A2） | 0 | — | 判死 | — |
| 16 | vendor/metax/…/mla/common.py:779/780/834 `.item()` | **否**（MLA 路径；本模型 Llama GQA 走 flash_attn） | 0 | — | **判死**（且 779/780 是 `*_cpu` 张量） | — |
| 17 | vendor/metax/…/flash_attn.py:802/900/1151/1177 `.tolist()` | 否（802 是 prefill-split 混批分支；1151/1177 是 cascade/DCP 公共前缀路径；均非 decode 步） | 0 | — | 判死（decode 口径） | — |
| 18 | compilation/graph.py:251 / :118-120 synchronize | 否（捕获/初始化期一次性） | 0 | — | 判死 | — |
| 19 | aten::item 全体（trace 实测 6.3/步） | 在 | 6.3 | p50 1.3µs | **判死**：全为 CPU 张量读，稳态 span 内零 D2H 伴随 | — |

## C. 判决与建议

1. **负判决：本配置 decode 稳态 host≈0，任务③关账。** 诚实门数字 1.04 < 2（A.1）；
   全部同步类要么是 µs 级 no-op、要么是 DP/spec/MM 路径死点、要么是 async scheduling 本体的 by-design join
   （删除即破坏正确性，且其等待与 GPU 忙共现=GPU-bound）。
2. **无修复补丁、无上机验证申请**（C 部分按开发单前置条件跳过：不存在"已证实在跑"的可消除同步项）。
   #778 静态张量缓存：现状已等价实现（pinned 常驻 buffer + non_blocking H2D，A.1 末行），无增量空间。
3. **唯一残留 host 成本（记录在案，非任务③口径）**：eager 采样尾 host-launch 缝隙 0.45ms/步（1.7%），
   三处固定 seam（A.3）。属 sampler python 精简/小算子融合领域（D5 后继，需 with_stack trace 定位到行）；
   采样语义正确性风险高、天花板 <1.7%，**不建议在本任务立项**；若 F-e（碎核融合）启动可顺带覆盖其 GPU 侧。
4. 对齐 D12：本审计以逐点证据补强了 D12 的方向性判决——96% busy 的另 4% 中，
   ~1.7% 是采样尾 host 缝隙（D5 后继地盘），其余为 GPU 侧微缝；**同步/D2H 维度的 host 开销实测为零**。

## 附：复现

```bash
python3 scripts/host/trace_host_audit.py \
  /data/flagos_wt/results/20261120_c500_reprofiling/trace_all/rank0.1795110863960817196.pt.trace.json.gz \
  results/20261120_host_audit
# 纯 CPU（json 解析 ~2min），输出 host_audit_summary.txt
```
