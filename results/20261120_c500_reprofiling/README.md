# 20261120_c500_reprofiling — 沐曦 C500 终局配置真实痕迹（Phase 1 电池）

> 计划：docs/二期计划-1120.md。背景：既有深度剖析（20261115_bottleneck_deepdive 等）
> 全部为天数侧/旧配置；本目录为沐曦终局形态（零 env + 无 compilation-config +
> w8a8-v2 ckpt + mctlass）下的全新取证。PROVENANCE：本机 git 见各 probe 头行，
> torch 2.8.0+metax3.7.0.7 / triton 3.0.0+metax / SM=104 / mx-smi 2.3.1。

## P1-B4 采样 fallback 实价（01:22-01:35，sampling_probe_metax.py v2）— 判决：D5 立项

| 测项 | bs=32 | bs=64 | 说明 |
|---|---|---|---|
| pytorch fallback（现 serve 每步走这条） | 5054µs | **9663µs** | 即 vendor patch 强制的路径 |
| torch.sort 单项 | 1331µs | 2619µs | fallback 的大头之一 |
| Triton kernel 五档 BLOCK（8192/4096×2/2048/1024） | — | **全部运行期 ATU 非法地址崩溃** | 编译通过、运行 Xnack trap（mx_device.cpp 10106 实名 _topk_topp_kernel）→ **kernel 路线本栈判死**（vendor patch 注释"ttgir 编译失败"实为运行期错；patch 决策正确） |

- 16K 稳态步时 ~67ms（final log：~945 gen tok/s @ Running=64）→ **采样 fallback 占 ITL ~14%**；
  4K 侧步时更短、占比更高（待 P1-A1 trace 定量）。
- D5 走向：torch 原生快路径（top-k 降序 + logsumexp 分母 + 阈值过滤，免全词表 sort/scatter）
  → feat/d5-sample-fast-metax，T0 对拍脚本 scripts/d5/sample_fast_t0.py。

## P1-A2 容量政权（01:25，final serve_bench.log 复核）— 判决：16K 无容量钳制（确认）

- KV 池 **1,257,344 tokens**；全程 **0 条 preempt 日志**；APC 全程 **0.0%**（随机 token 零命中，天数侧缓存双峰机理不迁移）；
- prefill 吞吐（16K ramp 段实测）**11.5K-13.1K tok/s**；
- cudagraph 捕获 sizes 到 512、FULL_AND_PIECEWISE 模式（serve_diag 可复核）；
- 结论：天数侧"KV int8 扩池 +40-90%"论点在 C500 归零 → D8 仅剩 KV 读流量减半一途，维持三重条件门。

## P1-A1 墙图 / P1-A3 方差（进行中）

- trace_wallmap.sh：诊断 serve 双窗 trace（decode 稳态 + prefill）→ trace_*/。
- variance_4k_probe.sh：同 serve 3×调用 vs 新鲜 serve 2×，pass 级方差归因 → variance_4k/。

## D5 采样快路径 · 终局判决（03:2x）：❌ serve 级零收益，负判决归档

- T0（严格判据）全绿：peaked 保留集逐位一致；平坦形态保留集差异概率质量 2.1e-6（<<int8 噪声 3e-3）；bs=64 离线 2.23×（3.94 vs 8.77ms）。
- T1 两轮 A/B（quick case [[4096,1024,64,32]] 与官方形状 [[4096,1024,64,64]]，各 4 pass 丢首）：
  - bs=32：fast 26.46 vs base 26.05ms（≈零差）
  - bs=64：fast 44.50/44.63 vs base 44.10/44.32ms（≈零差）；serve 日志快路径命中 4000+/回退 0
- **根因：B4 探针的 9.66ms/步是 L2-cold 微基准伪影**（Cycler 8 份克隆轮转使每次调用冷缓存）；serve 内 logits 由 lm_head 刚写出（L2 热态），整条采样链真实成本 ~1-2ms，优化空间在 serve 口径下不存在。
- 附带修正：墙图 sort 链 46% 占比含 profiler 膨胀；eager 侧归因以 A/B 臂差为准。
- 处置：feat/d5-sample-fast-metax 归档（含 T0/T1 全套与 DEBUG 探针），不合入 delivery；site-packages 已还原 vendor 插件。
- 方法论沉淀：**凡 kernel 微基准外推 serve，必须 L2-warm 口径**（台账 §9 方法论行）。

## P1-A1 补充判决（04:0x）：decode 稳态 GPU busy ≈ 96%（非 launch-bound）

- trace 最忙秒的 kernel 时间 957-984ms/s → **GPU 饱和**；早前「可见 kernel 仅解释 57% 步时」为混合阶段重建错误。
- 推论：e2e 增益只能来自 kernel 自身效率（D7 prefill / D9 quant），无 launch/调度免费午餐（D12 类假设不立案）。
