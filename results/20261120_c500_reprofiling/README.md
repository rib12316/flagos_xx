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
