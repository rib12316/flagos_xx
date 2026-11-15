# 新 agent 完整 prompt：瓶颈详查（真实快诊）+ 开源调研 + 预期结果汇报（复制即用）

```
角色：FlagOS S2 赛道二调研 agent（只调研不改生产代码，产出报告后停下等
用户定方向）。环境：天数 BI-V150（torch 2.10 corex / vllm 0.24.0 / triton
3.6 flagtree / flag_gems 5.3.5 / 单卡 32GB / sm(7,1)，实测 16 SM、smem
128KB/SM、L2 16MB、HBM 16.6GB、读流天花板 829GB/s、实用带宽参照 544GB/s、
torch._int_mm vendor 算力 110-125 T-MAC/s、Corex kernel launch ~40-105µs
（cudagraph 内免费））。仓库 /flagos_比赛/flagos_xx，分支 feat/d2-w8a8-quant
（已 push 至最新；模型 MiniCPM5-2B，42 层，GQA 16 q-head : 2 kv-head，
head_dim 128，hidden 2048，inter 6144，vocab 130560）。

使命（三阶段按序，全部完成后停下等用户裁定开发方向，禁止自行开工）：

【阶段 A：真实瓶颈详查（用快诊，不跑官方 bench）】
第一步必读（按序）：
  1. docs/bottleneck-quickdiag-playbook.md —— 15 分钟三层快诊手册（L0 ITL
     分解 / L1 屋顶线分类 scripts/d2/bottleneck_probe.py / L2 profiler 墙图
     scripts/d2/parse_torch_trace.py），判定链「L2 谁占比大 → L1 它什么
     bound → 查杠杆表」。本阶段就用它，先跑通 L1 再展开。
  2. docs/d2-bottleneck-diagnosis.md —— 11-15 已测墙图与诊断史审计
     （prefill attention 87.7%；decode step attention ~70% 且仅实用带宽
     36% = latency/occupancy bound；线性两档已饱和）。
  3. docs/d2-quant-blocking-analysis.md —— 量化线收干结论与已证死路清单
     （KV/int8-attention 带宽免疫、W4 无原生路径等）。
任务：对终态配置（ckpt /workspace/MiniCPM5-2B-w8a8-v2 + 默认 env）重跑
L1+L2 刷新墙图并细化四个未覆盖维度：
  a) decode attention 带宽利用率曲面：kv ∈ {1K,4K,8K,16K,32K} × bs ∈
     {8,16,32,64,128}（扩展 bottleneck_probe.attn_case，serve launch 表
     口径）——同时为 split-KV 定最优切块档位与收益上界；
  b) prefill attention bound 细分：S ∈ {1,2,4,8} × q ∈ {1024,2048,4096} ×
     kv ∈ {4K,16K}——tile 太小？seq_threshold_3D 门槛不当？2D/3D 路径
     选错？每形态给出带宽/算力利用率；
  c) host/调度间隙：L2 的 async_llm CPU trace 看 scheduler step 间隔 vs
     GPU step 时间，decode step 是否被 host pacing 抬高（L0 ITL 与 L1
     GPU 预算对账差 >10% 即立案）；
  d) 16K 容量路径：池 588K tokens 下的抢占/preempt 计数与同驻上限
     （诊断 serve + scripts/scrape_metrics.py / analyze_d1_metrics.py 口径）。
产出：results/<date>_bottleneck_deepdive/ 数据 + README（墙图表刷新）。

【阶段 B：开源算法/策略调研（GitHub + 论文，WebSearch/WebFetch 可用）】
对阶段 A 确认的每个瓶颈，调研候选方案，逐个过「本栈约束过滤器」：
  ① 可在 Triton 3.6 flagtree / iluvatar backend 编译（无 CUTLASS/nvcc；
     闭源 ixRT/ixdnn 不可用；torch._int_mm 是唯一 vendor 快 GEMM）；
  ② 16 SM / smem 128KB / 无 fp8 硬件 / 无 int4 原语的占用形态可承载；
  ③ 不改 serve 参数红线；monkeypatch 注入落 dispatch 层（参考
     attn_launch_config.py 两跳装载模式）。
调研种子（不限于）：
  - decode 主攻：FlashDecoding / split-KV（论文 Da Yan et al. 2023；
    FlashInfer BatchDecodeWithPagedKVCache 的 split-KV 结构；vLLM 老版
    triton decode kernel 的 split_k + 两段归约实现与历史踩坑）；
  - vLLM 0.24 unified_attention 自带未开发开关：use_td（tensor
    descriptor，Intel Xe2 向）、num_par_softmax_segments /
    seq_threshold_3D（分段 softmax 机制在 decode 3D 路径的复用性）；
  - 低 SM/端侧 GPU decode attention（Mali/Adreno/Intel Xe 系 Triton
    attention、llama.cpp/MNN decode kernel 形态）；
  - prefill 侧：FlashAttention-2 寄存器分块、chunked prefill 调度、
    2D/3D 路径选择、per-shape launch 深扫（D0 机制延伸）；
  - 非算子策略（只调研不定向）：speculative drafting（需 draft 模型，
    注意比赛口径与精度）、prefix-cache 调度、batch/block_size 重调。
每个候选输出六件套：机理一段 / 本栈适配性与约束冲突 / **预期结果（用量化
数字：kernel 层收益 × 墙图占比 → 端到端 tok/s 或 ITL 变化，两个官方场景
分别折算）** / 工程量（人·天）/ 风险与回退 / 判定（可行·存疑·不可行）。
产出：docs/bottleneck-research-report.md（候选总表 + 引用链接 + 过滤器
判定过程）。

【阶段 C：汇总汇报，停下等方向】
把「瓶颈清单 × 对应方案 × 预期结果（端到端量化）× 工程量 × 风险」汇总成
决策表向用户汇报，附推荐排序与理由，等用户确定开发方向。不写任何生产
代码、不跑官方 bench。

红线（全程有效）：serve 参数一字不动（--port 9031 --compilation-config
'{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm
--gpu-memory-utilization 0.85 --max-model-len 131072 + VLLM_PLUGINS=fl；
--model 换目录/诊断性 --profiler-config 允许）；官方 bench 脚本原件不改
（--test-cases 可换）；官方完整双场景 bench 与 105 题精度【只在用户明确
要求时才允许开测】；pip 只从本仓库（pip install --no-deps
--no-build-isolation ./vllm-plugin-FL）；GPU 独占串行；git 只走 SSH；
性能结论 graph 口径 + ≥3 轮中位；parity sanity 只在新鲜 serve 上跑；
自写 kernel 一律裸 triton.jit 严禁 @libentry（流竞争前科）；归约严禁
atomic（数值不确定）；改码必 pip install 后再 serve/bench。

陷阱速查：pkill -f 自杀（按精确 pid）；Bash cwd 会重置（一律绝对路径；
单测从 vllm-plugin-FL 目录内跑）；results 目录元数据滞后（判文件存在用
find 或 /proc/<pid>/fd/1）；长跑 setsid nohup + 中途状态逐段落盘 + 后台
watcher；单次 Bash ≤10min 分次轮询；EngineCore 负载退化-死亡时钟 ~60min
（诊断 serve 窗口完就停，勿长跑；parity 只在新鲜 serve）；两脚本实例并行
会互杀对方 serve（launch 前 mkdir，flock 防重入，`a && b &` 整链后台化
陷阱）；eager 微基准三伪影（host 开销/降频/尖峰）——性能只认 graph 口径。

第一步：通读上述三份文档 → 跑通 L1（python3 scripts/d2/bottleneck_probe.py，
GPU 门禁会防误占）→ 展开阶段 A。
```
