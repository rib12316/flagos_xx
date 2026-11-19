# 推理瓶颈快诊手册（15 分钟三层 · 不碰官方 bench）

> 目的：任何优化/bench 之前，先回答「时间花在哪（L0/L2）、瓶颈是什么类型（L1）」。
> 本机：天数 BI-V150，16 SM / smem 128KB/SM / L2 16MB / HBM 16.6GB，读流天花板
> 829GB/s，**实用带宽参照 544GB/s**（元素 kernel 实测），`_int_mm` vendor 算力
> 110-125 T-MAC/s。仓库 `/flagos_比赛/flagos_xx`，分支 `feat/d2-w8a8-quant`。
> 历史墙图快照见 §五；机理详证 `docs/d2-bottleneck-diagnosis.md`。

## 核心纪律（先记住再动手）

1. **时间占比 ≠ 瓶颈类型**。"X 占 65%" 不等于 "X 是带宽问题"。占比找大头上
   L2，类型用 L1 判，**两步都做完才准选武器**（P2 空转的教训：按"占比+
   直觉"立项 KV 量化，实测才发现 attention 是延迟 bound，量化免疫）。
2. 性能判据一律 **graph 口径**（eager 有 host 开销/降频/尖峰三伪影）。
3. 探针必须**复刻 serve 运行时调优**（launch 表等），否则测的是假 kernel
   （未装 D0 表的 attention 慢 3×）。
4. GPU 独占串行；任何 serve/bench 期间不并发其他 GPU 任务。

---

## L0 · 步进分解（3 min，只要一个 serve）

```bash
# 红线 serve 参数（一字不动；--model 换目录/诊断 --profiler-config 允许）
VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B-w8a8-v2 \
  --port 9031 --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
# 小参快速口径（官方 bench 原件换 --test-cases；单轮受 prefix cache 双峰支配，≥3 轮中位）
cd /workspace && python3 /flagos_比赛/flagos_xx/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B-w8a8-v2 --served-model-name minicpm --port 9031 \
  --test-cases '[[512,1024,64,64],[256,1024,128,128]]'
```

读数：`Median ITL`（decode 每 token 时延）与 TTFT。sanity 对账：
`ITL ≈ 42 层 ×(单层 attention + 4 线性) + 采样/norm`（数值从 L1 来）——
对不上 ±10% 说明有 host/调度间隙，查 L2 的 CPU 侧。

## L1 · 屋顶线分类（5 min，无需 serve）★核心层

```bash
python3 /flagos_比赛/flagos_xx/scripts/d2/bottleneck_probe.py
```

自动：装载 serve launch 表（D0：decode w1s1 / prefill w2s1）→ 在 serve 真实
形状上 graph 口径测每组件「达成 GB/s（vs 829 天花板 / 544 实用）与 T-MAC/s
（vs 118 vendor）」→ 输出分类。**判读表**：

| 分类 | 判据 | 有效杠杆 | 无效杠杆 |
|---|---|---|---|
| bandwidth-bound | ≥60% 实用带宽 544GB/s | 字节减半类：量化权重/KV | — |
| compute-bound | ≥60% vendor 算力 118 T-MAC/s | int8 原生 dot/低位宽算力 | 带宽类 |
| **latency/occupancy** | 两者皆低 | **kernel 形态工程**（split-K/并行度/流水/launch） | **一切量化** |

改动 kernel/配置后重跑本层即可复核（改配置 → 体检 → 再优化，循环标准化）。

## L2 · serve 墙图（7 min + serve 启动 ~10 min）

```bash
# 1) 带 profiler 的诊断 serve（profile 路由只有带此 flag 才挂载）
VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072 \
  --profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/prof","torch_profiler_with_stack":false}'
# 2) burst 流量窗口（示例：32 并发 decode 流 / 4×4096 prefill 流）
# 3) curl -X POST :9031/start_profile -d '{"output_dir":"/tmp/prof/dec"}' … /stop_profile
# 4) 解析（分桶聚合 + Top-20 kernel + GPU busy）
python3 /flagos_比赛/flagos_xx/scripts/d2/parse_torch_trace.py /tmp/prof/rank0.*.trace.json.gz
```

**坑**：decode 在 FULL_DECODE_ONLY cudagraph 内重放，**对本 trace 不可见**——
L2 只给 prefill/eager 侧占比；decode 侧占比 = L1 单层数值 × 42 层 与 ITL 对账。

## 判定链与杠杆表（选武器前必过）

```
L2 谁占比大 → L1 它什么 bound → 查下表选杠杆 → 预期收益用占比×杠杆上限估算
```

| 瓶颈组合 | 可用杠杆（本栈已验证或候选） | 已证死路 |
|---|---|---|
| 占比大 + latency/occupancy | kernel 形态工程：split-KV+两段归约（flash-decoding）、tile/launch 重扫、并行度 | KV/int8 量化（带宽免疫）、W4（无原生 int4 路径） |
| 占比大 + bandwidth | 量化字节（W8 已做尽；KV 需容量 binding 场景才可能净正） | — |
| 占比大 + compute | int8 原生 dot（W8A8 已 89-103% vendor） | 更低位宽算力（无原语） |
| 占比小（<10%） | 不值得动；除非零成本（launch 表） | — |

## 陷阱速查（本机实证过）

- 探针不装 launch 表 → attention 慢 3×，墙图失真（L1 脚本已内置）。
- eager 微基准三伪影（host 开销/轮间降频/56ms 尖峰）→ 只认 graph。
- 单轮 bench 汇总受 prefix cache 双峰支配 → ≥3 轮中位。
- parity/精度 sanity 必须在**新鲜 serve** 上跑（负载后引擎退化，见
  `results/20261114_d2_conc128_stress/README.md` 的 ~60min 退化-死亡时钟）。
- 归约严禁 atomic（数值不确定，skinny 三振前科）——写 partials + 独立归约。
- `pkill -f` 会自杀（exit 144）；按精确 pid 杀；Bash 命令绝对路径。
- results/ 目录元数据滞后：判文件存在用 `find` 或 `/proc/<pid>/fd/1`，勿信单次 ls。
- 长跑 setsid nohup + 中途状态落盘；GPU 门禁 `ixsmi --query-gpu=memory.used` < 3000MiB。

## 五、墙图快照（2026-11-15，终态 v2 ckpt + P0/P1，供对照）

| 场景 | 构成 | L1 分类 |
|---|---|---|
| prefill（4×4096 输入） | **attention 87.7%** / 线性 int8 8.5% / norm+其他 3.8% | attention 非 bandwidth；线性 89% vendor 算力=贴顶 |
| decode step（M=64, kv~1.5K） | **attention ~70%** / 线性 23% / 其他 10%（对账 ITL ✓） | attention 193-197GB/s=实用 36% → latency/occupancy；线性 78-88% 实用带宽=饱和 |

结论：两个场景的墙都是 attention；decode 侧形态工程理论余量 ~2×，
prefill 侧次之；线性侧量化已无矿（详 `docs/d2-quant-blocking-analysis.md`）。
