# V150 P1/P2/P3 交接文档(2026-11-20 下午,交接人:BM/P2/P3 会话 agent)

> 分支:`feat/v150-attn-bm`(基于 delivery/s2-final,已 push,含 2 个 commit)。
> 起点 = 《天数V150回归开发单-1120》(docs/) 的执行现场。本文档自包含,新 agent
> 从零读完即可接手。**所有结论都给出了数据出处;没有出处的判断都标注"未验证"**。

---

## 0. 三十分钟状态快照

| 项 | 状态 | 一句话 |
|---|---|---|
| **P1 prefill BLOCK_M 重标定** | **代码完成,默认关,等官方门** | kernel 9.4-9.9×,serve TTFT 15.5→2.3s;官方 4 轮+105 题未开(等用户令) |
| **P2 KV int8** | **判决:env-flip 不做** | 增量 +8~13% 在官方轮方差内;kernel 手术路径(+15~20%)冻结待议 |
| **P3 segm 自适应** | **接线完成,默认关,serve 级回退未归因** | kernel 级 2.96× 优,serve 级反转为 ~2× 劣——下一个 agent 的第一优先案 |
| P4 混批步 2D→3D | 未开工 | post-P1 价值缩水到 +1-2% |
| P5 host busy 摸底 | 未开工 | post-P1 价值上升(GPU 时间缩短后 host 占比相对变大) |
| P6 逐算子拉黑 | 未开工 | +1-2% 收尾件 |

当前 16K/4K 官方口径(尚未复测):B 臂 1570.62 / 3414.58(D3 官方判决,基线 1345.90/2838.46)。
P1 合体臂烟测投影:16K ~2600-3600(波结构模型,见 §2.4)。

---

## 1. 环境与纪律(必读,全是本机踩过的坑)

- **官方 bench(4K/16K 4 轮)与 105 题精度门:只在用户明示时开测**。常态只跑烟测/探针。
- **kill 必须按端口限定**:`ps -eo pid,cmd | grep "port 90XX" | grep -v grep` 再杀。
  11-20 11:49 曾发生全量 kill 误杀队友 ah4-T1 基线 serve(远程 f88db09 通报)。
  占 GPU 前先 `ixsmi` + 查队友窗口。
- GPU 清理:EngineCore 孤儿按显存循环清(见 memory `pkill-selfmatch-and-vllm-orphans`);
  **交互命令勿用 `pkill -f`(自匹配自杀,exit 144)**。
- 后台任务:`mkdir -p` 必须**前置**(shell 重定向先于脚本内 mkdir);git/GitHub 只走
  SSH(https 出网不通;大 repo 全量 fetch 会挂,blobless 浅克隆可解);后台命令用绝对路径。
- serve 形态(V150 官方口径,逐字):
  `vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port <P> --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`,env 前缀 `VLLM_PLUGINS=fl`。
- 插件是**双拷贝**:`/flagos_比赛/flagos_xx/vllm-plugin-FL/`(repo,真源)+ `/usr/local/lib/python3.12/site-packages/vllm_fl/`(serve 实际加载)。**改 repo 后必须 cp 同步**,否则 serve 看不到。
- 探针口径:唯一可信 = CUDA graph replay 中位(`scripts/d2/bottleneck_probe.py::graph_time_us`);
  eager 微基准有三重伪影(memory `biv150-perf-forensics-toolbox`)。
- 每场景官方轮前 fresh serve(EngineCore 9 serve 6 死/挂前科;16K 一轮 ~12-25min)。

---

## 2. P1:prefill BLOCK_M 耦合重标定(已完成,等门)

### 2.1 机理与数据

上游 wrapper(`vllm/v1/attention/ops/triton_unified_attention.py:860`)对 GQA(≤16 q/kv)
固定 `BLOCK_M=16` → BLOCK_Q=2,每 program 只算 2 个 query 位置,prefill 仅 1.4-2.8% 峰值
MAC。重标定 **BM256/W16/S2/TILE128**(=竞队 PR#671 配方):

| 形态 | upstream | 旧生产(B臂prefill) | 新配置 | 加速 |
|---|---|---|---|---|
| S1 q4096 kv16K | 139.2ms | 85.1ms | **14.1ms** | 9.88× / 6.03× |
| S2 q2048 kv16K×2 | 150.0ms | 91.0ms | **15.3ms** | 9.78× / 5.94× |
| S1 q8192 kv16K(整chunk) | ~237ms | — | **24.1ms** | 9.86× |
| S4 q4096 kv4K×4 | 80.3ms | 48.4ms | **8.5ms** | 9.42× / 5.68× |

36 配网格(BM{64,128,256}×T{32,64,128}×W{8,16}×S{1,2})无更优;消融:W16 是命门
(换W4剩1.95×),T128 在 BM256 下 +1.31×。**decode 侧不用动**(T16+w1s1 已最优,
D0 表保留)。数据:`results/20261120_blockm_sweep/`。

### 2.2 实现(commit 1)

`vllm_fl/dispatch/backends/vendor/iluvatar/attn_bm_mode.py`:锚点重 exec 参数化
wrapper(源码三锚点,任一不唯一整体放弃),双绑定替换(ops 模块 + triton_attn 后端),
`VLLM_FL_ATTN_BM=1` 启用,**默认关**(过官方门后翻默认)。

三个必须知道的坑:
1. **必须装在 alc(attn_launch_config)之前**——alc hop1 闭包捕获当时的绑定;iluvatar.py
   已调整 install 顺序(BM 在最前)。链 = hop1 hint → BM-patched → hop2 setdefault。
2. **TD 与 TILE128 互斥仅在 prefill 侧**(TD 钳位 min(TILE, block_size=16));①TD 的
   真正值在 decode(TD 124ms vs 非TD 269ms @bs16/kv17K)。BM 模块在 prefill 分支内
   强制 `use_td=False`,decode 保留 TD。
3. serve 真实调用链走 `vllm::unified_attention_with_output` 分裂算子,但它模块内调
   `unified_attention` 全局名 → 重绑有效(实测证明)。

### 2.3 烟测结果(results/20261120_bm_smoke/)

| 测项 | B臂(现役) | 合体臂(BM+TD) |
|---|---|---|
| 单请求 16.8K 冷提示 TTFT | 15.53s | **≈2.3s(6.7×)** |
| decode-only ITL@bs16 | 124ms | **123ms(零回退)** |
| 16并发×16K×128 | 8525 tok/s | **11243(+32%)** |

测量陷阱:同文本重复请求→prefix cache 全命中(TTFT 0.28s 假象);进程首个真 prefill
付 BM kernel ~6s Triton 编译税(官方 4 轮丢首轮天然吸收,105 题首题需 warmup)。
终选臂 env:`VLLM_FL_ATTN_BM=1 VLLM_FL_ATTN_TILE=0 VLLM_FL_ATTN_TD=1`。

### 2.4 剩余工作(P1 收尾)

1. **官方 4 轮双场景 + 105 题(等用户令!)**;新鲜 serve 协议照 D3
   (`scripts/d3/run_official_redo.sh` 模板)。
2. 过门后:attn_bm_mode 默认开(改 env 缺省值)+ DELIVERY.md 更新。
3. 16K 投影(波结构模型,自下而上):attention 占 65-75% ÷6 → 理论 2.2-2.7×;
   受 33/64 准入四波串行 + decode 地板(中位 ITL 77ms)压制 → **2600-3600**。

---

## 3. P2:KV int8(判决:env-flip 不做)

全部数据 `results/20261120_kvquant_sweep/`(README 有完整判读)。要点:

1. **A 臂"187ms≈2×"重新归因**:kernel 级量化税 decode 仅 +12%(q8_t16 3799 vs
   bf16_t16 3386 @bs64/kv16K);主差额是 bs64 vs bs33 每步 2× KV 读(容量翻倍的代价)。
2. **①×④互毁精确刻画**:TD×quant 编译共存、数值对(rel 0.9%)但 q8_td 5150μs vs
   q8_notd 3799 vs bf16_td 2705 → 开 quant 弃 TD,**真实 decode 税 = +40%**。
3. **prefill 合体税**:BM256/T128/S2×quant **OOR**(flagtree smem 上限实测 **131072B**);
   最优 q8_bm256_t128_w16_S1=19.9ms,税 +41%。BM128 全线劣化。
4. **16K 墙钟模型**:P1-only ~3240 → +P2 env-flip ~3650(**+8~13%,在轮间方差内**)→ 不做。
   kernel 手术(快 int8 加载/转换,夺回 TD 级 decode)→ ~3900(+15~20%),
   **排 P1 官方门之后由用户裁定是否开工**。
5. 量化路径最优 TILE 仍是 16(t32/64/128 单调劣化);上游 dequant 数学已最优
   (K scale 后置 S、V scale 前置 P)——纯 dequant 优化上限 ≈ e2e 2-3%,不值。

---

## 4. P3:segm 自适应(接线完成,serve 级回退未归因)——**下一个 agent 第一优先**

### 4.1 已确证的(内核级,graph 口径,可复现)

TD 口径 segm 探针(生产 decode 配置 T16/w1s1/TD on):

| 形态 | n8 | n16(现役) | n32 |
|---|---|---|---|
| bs64/kv4K | **730.6μs** | 2163.3μs | 772.9μs |
| bs64/kv8K | **1396.4μs** | 1445.4μs | 1472.9μs |
| bs64/kv16K | 2716.5μs | **2699.4μs** | 2746.1μs |
| bs32/kv8K | 741.4μs | **730.0μs** | 3208.9μs(灾难) |

bs64/kv4K n16→n8 = **2.96×**(TD off 时只有 1.24×——TD 把 n16 病理化)。
serve 全链 sanity(装 hop1 降档):774μs,兑现 2.79×。

### 4.2 实现(commit 2,默认关)

`attn_launch_config.py` hop1(`_hinted_unified_attention`)内,`VLLM_FL_ATTN_SEGM=1`
启用:`num_par_softmax_segments==16 且 max_seqlen_q<=1 且 bs>=48 且 0<kv<=8192`
时降为 8。**必须在 wrapper 网格计算前改**(hop2 kernel.run 级来不及);segm 缓冲按
16 分配、n8 只用前 8 槽,越界安全。

### 4.3 反常(未解):kernel 级 2.79× 优 → serve 级 ~2× 劣

A/B 烟测(4K 域:ISL=4096/conc=64/OSL=128,quick_smoke_client,两臂仅差 SEGM env):
- **segm_on: wall 82.3s / 3367 tok/s**
- **segm_off: wall 43.0s / 6444 tok/s**

数据:`results/20261120_segm_smoke/`(两臂 serve.log + quick.txt)。
**归因假设清单(按嫌疑排序,交下一个 agent):**
1. **cudagraph 捕获时桶判定烘焙错值**:FULL_DECODE_ONLY 捕获时 dummy 输入的
   max_seqlen_k/seq_lens 是什么值?捕获时 hop1 的 bucket 条件用的 kv/bs 可能与
   runtime 不同 → 烘焙了错误 n。验证:hop1 加一次性 log(capture 期间 vs runtime),
   或 serve log 里 grep 捕获期的降档记录。
2. **探针图口径伪影**:探针的 n16-TD@bs64/kv4k 病态格(2163μs)可能是探针自身
   graph capture 的伪影(探针自捕自放 vs serve 走 vllm cudagraph 机制)。
   验证:serve 挂 torch profiler / ixkn trace,直接看 decode 步 kernel 时间 n8 vs n16。
3. 混批步(预填充+decode eager 路径)hop1 python 逻辑每调用执行的额外开销(小,难成 2×)。
4. segm 缓冲跨 capture size 的别名问题。
**对照实验(便宜且决定性)**:同 serve 设 `VLLM_FL_ATTN_SEGM=1` 但把阈值改成
`kv<=1`(桶永不命中)→ 若仍 82s 慢,则慢因不在 n8 本身(嫌疑 3/4 或 env 存在性);
若恢复 43s,则锁定 n8 烘焙/执行问题(嫌疑 1/2)。
**判决前 P3 保持默认关;探针结论(n8 kernel 级优势)与 serve 反常并案存档,勿单方面采信。**

---

## 5. 未开工项(开发单原文见 docs/天数V150回归开发单-1120.md)

- **P4 混批步 decode 行 2D→3D**(deepdive 判定链 3):post-P1 混批步更少,价值 +1-2%。
  若做:调 batch 形态判定门槛;注意混批步不走 decode cudagraph。
- **P5 host 热路径审计**:先 torch profiler 摸 busy(墙图法同 C500),.item()/.cpu()
  消除、静态张量缓存(#778 子集)、async scheduling 确认。post-P1 价值上升。
- **P6 逐算子 A/B 拉黑**(#384 模式):W8A8 已换大 GEMM,只剩元素类,+1-2%。
- 机会项:flagtune 元素类 per-shape;W4A8 仍禁(无 int4 dot)。

---

## 6. 文件地图

| 路径 | 内容 |
|---|---|
| `vllm_fl/.../iluvatar/attn_bm_mode.py` | P1 模块(锚点重 exec,env 门控默认关) |
| `vllm_fl/.../iluvatar/attn_launch_config.py` | D0 表 + P3 segm hop1(§4.2) |
| `vllm_fl/.../iluvatar/iluvatar.py` | install 顺序:BM 必须在 alc 前 |
| `scripts/d3/blockm_sweep.py` | P1 探针(锚点参数化 wrapper 模板,可复用) |
| `scripts/d3/kvquant_sweep.py` / `kvquant_combo_probe.py` | P2 探针(int8 池构建模板) |
| `scripts/d3/quick_smoke_client.py` | 快速烟测客户端(用法:port ISL conc OSL) |
| `scripts/d3/run_bm_smoke.sh` / `run_official_redo.sh` | serve 烟测/官方补跑模板 |
| `results/20261120_blockm_sweep/` | P1 全数据 |
| `results/20261120_bm_smoke/` | P1 烟测(README 判读) |
| `results/20261120_kvquant_sweep/` | P2 全数据(README 判读) |
| `results/20261120_segm_smoke/` | P3 A/B 反常现场 |
| `results/d3_official_1116_2222/README.md` | D3 官方判决(基线/协议/EngineCore 事故史) |
| PR#671 竞品补丁 | `/tmp/pr671/`(参考;勿直接抄,我们栈有 AOT/W8A8 差异) |

## 7. 已证死/勿重开

KV int8 env-flip(§3)、TD×quant 组合(性能互毁)、量化 TILE>16、n32 segm、
BM128、W4A8、1-kernel 融合 quant(D2)、mnbt=8192(D1 负结果存档)、
launch 网格超越 PR671 配方(36 配已扫尽)。
