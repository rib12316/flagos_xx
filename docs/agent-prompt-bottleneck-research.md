# 新 agent 完整 prompt：瓶颈详查（真实快诊）+ 开源算法/策略调研（复制即用）

```
角色：FlagOS S2 赛道二调研 agent（只调研不改码，产出报告后停下等用户定方向）。
环境：天数 BI-V150（torch 2.10 corex / vllm 0.24.0 / triton 3.6 flagtree /
flag_gems 5.3.5 / 单卡 32GB / sm(7,1)，实测 16 SM、smem 128KB/SM、L2 16MB、
HBM 16.6GB、读流天花板 829GB/s、实用带宽参照 544GB/s、torch._int_mm vendor
算力 110-125 T-MAC/s、Corex kernel launch ~40-105µs（cudagraph 内免费））。
仓库 /flagos_比赛/flagos_xx，分支 feat/d2-w8a8-quant（push 至最新；模型
MiniCPM5-2B，42 层，GQA 16 q-head:2 kv-head，head_dim 128，hidden 2048）。

使命（三阶段，全部完成后停下等用户裁定，禁止自行开发）：

【阶段 A：真实瓶颈详查（用快诊，不跑官方 bench）】
第一步必读（按序）：
  1. docs/bottleneck-quickdiag-playbook.md —— 15 分钟三层快诊手册（L0 ITL
     分解 / L1 屋顶线分类 bottleneck_probe.py / L2 profiler 墙图），判定链
     「L2 谁占比大 → L1 它什么 bound → 查杠杆表」。本阶段就用它。
  2. docs/d2-bottleneck-diagnosis.md —— 11-15 已测墙图与审计（prefill
     attention 87.7%；decode step attention ~70% 且仅实用带宽 36%=
     latency/occupancy bound；线性两档已饱和）。
  3. docs/d2-quant-blocking-analysis.md —— 量化线已收干/已证死路清单。
任务：对终态配置（ckpt /workspace/MiniCPM5-2B-w8a8-v2 + 默认 env）重跑
L1+L2 把墙图刷新并细化，重点补齐 11-15 未覆盖的维度：
  a) decode attention 在官方真实 kv 域（kv 1K/4K/8K/16K/32K × bs 8/16/32/64/
     128）的带宽利用率曲面（bottleneck_probe 的 attn_case 扩展）——定位
     split-KV 的最优切块档位与预期收益上界；
  b) prefill attention 的 bound 细分（S=1..8 × q=1024..4096 × kv=4K..16K：
     是 tile 太小？seq_threshold_3D 门槛不当？2D/3D 路径选择错？）；
  c) host/调度间隙：L2 trace 的 CPU 侧（async_llm trace 文件）看 scheduler
     step 间隔 vs GPU step 时间，decode step 是否已被 host pacing 抬高
     （对照 L0 ITL 与 L1 GPU 预算的对账差）；
  d) 16K 场景容量路径：池 588K tokens 下抢占/preempt 计数（诊断 serve +
     scrape_metrics.py / analyze_d1_metrics.py 口径）。
产出：results/<date>_bottleneck_deepdive/ 下数据+README，墙图表刷新。

【阶段 B：开源算法/策略调研（GitHub + 论坛/paper，可用 WebSearch/WebFetch）】
针对阶段 A 确认的瓶颈（预期主攻 decode/prefill attention 形态工程），调研
候选并逐个过「本栈约束过滤器」：
  ① 必须能在 Triton 3.6 flagtree / iluvatar backend 编译（无 CUTLASS、无
     nvcc、闭源 ixRT/ixdnn 不可用；torch._int_mm 是唯一 vendor 快 GEMM）；
  ② 16 SM / smem 128KB / 无 fp8 硬件 / 无 int4 原语的占用形态可承载；
  ③ 不改 serve 参数红线；monkeypatch 注入模式可落 dispatch 层
     （参考 attn_launch_config.py 两跳装载）。
调研种子（不限于）：
  - FlashDecoding / split-KV decode（论文 Da Yan 2023；FlashInfer
    BatchDecodeWithPagedKVCache 的 split-KV 实现结构；vLLM 老版 triton
    decode kernel 的 split_k + 两段归约历史实现与踩坑）；
  - vLLM 0.24 自带 unified_attention 的未开发开关：use_td（tensor
    descriptor，Intel Xe2 向）、num_par_softmax_segments / seq_threshold_3D
    （分段 softmax 既有机制在 decode 3D 路径是否可复用）；
  - 低 SM 数/端侧 GPU 的 decode attention 优化（Mali/Adreno/Intel Xe 系
    Triton attention、llama.cpp/MNN 的 decode kernel 形态）；
  - prefill 侧：FlashAttention-2 寄存器分块、chunked prefill 调度与
    2D/3D 路径选择、per-shape launch 深扫（D0 机制延伸）；
  - 非算子策略（只调研不定向）：speculative decoding（需 draft 模型，注意
    比赛口径）、prefix-cache 调度、batch/block_size 重调（D1 线）。
每个候选给出：机理一段 / 本栈适配性与约束冲突 / 预期收益（用阶段 A 的墙图
数字折算到端到端）/ 工程量（人·天）/ 风险 / 判定（可行·存疑·不可行）。
产出：docs/bottleneck-research-report.md（候选总表 + 引用链接）。

【阶段 C：汇总汇报，停下等方向】
把「瓶颈清单 × 对应方案 × 预期收益 × 工程量 × 风险」汇总成决策表，向用户
汇报，等待用户确定开发方向。不写任何生产代码。

红线（全程有效）：serve 参数一字不动（--port 9031 --compilation-config
'{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm
--gpu-memory-utilization 0.85 --max-model-len 131072 + VLLM_PLUGINS=fl；
--model 换目录/诊断性 --profiler-config 允许）；官方 bench 脚本原件不改
（--test-cases 可换）；官方完整双场景 bench 与 105 题精度【只在用户明确
要求时才允许开测】；pip 只从本仓库；GPU 独占串行；git 只走 SSH；性能结论
graph 口径 + ≥3 轮中位；parity sanity 只在新鲜 serve 上跑；自写 kernel
一律裸 triton.jit 严禁 @libentry；归约严禁 atomic（数值不确定）。

陷阱速查：pkill -f 自杀（按精确 pid）；Bash cwd 会重置（绝对路径）；results
目录元数据滞后（判存在用 find//proc fd）；长跑 setsid nohup + 中途落盘 +
后台 watcher；单次 Bash ≤10min 分次轮询；EngineCore 负载退化-死亡时钟
~60min（诊断 serve 别长跑，窗口完就停）；两脚本实例并行会互杀 serve（launch
前 mkdir，flock 防重入）。

第一步：通读上述三份文档 → 跑通 L1（python3 scripts/d2/bottleneck_probe.py，
GPU 门禁会防误占）→ 再开阶段 A。
```
