# ⑥ split-KV decode kernel 无卡开发交接书（2026-11-18 夜）

> 使命：自写 decode attention split-KV kernel，开采 16K 场景 decode attention
> 的 ~1.9× 带宽 headroom（现状 307-354GB/s vs 本卡可达 660）。本文档面向
> **无 GPU 的开发 agent**：只写代码/测试/集成，编译与实测留给次日 6h 真机窗口
> （验证计划见 §8，判据与回退见 §9）。开发语言环境：python3.12 + triton 3.6
> （仅做 py_compile/AST 级自检，**不要尝试 import triton 编译**——目标后端
> iluvatar 只有真机上有）。

## 1. 硬件与栈约束（不可协商）

- 天数 BI-V150：**16 SM**，smem **128KB/SM**，L2 16MB，读流天花板 829GB/s，
  **可达参照 660GB/s**（调优读核 BLOCK=16384 实测，`results/20261115_
  bottleneck_deepdive/ref_table.csv`）。无 fp8/int4 硬件。
- triton 3.6 flagtree（iluvatar backend）；**无 CUTLASS/nvcc**；闭源 ixRT 不可用。
- 纪律（前科在案）：自写 kernel **一律裸 `@triton.jit`，严禁 `@libentry`**
  （流竞争前科：进程内直调 b>1 读未就绪输入）；**归约严禁 atomic**（要确定性）；
  **严禁运行时 autotune**（与 cudagraph 冲突，launch 开销 40-105µs）；
  性能结论只认 **CUDA graph 口径**（eager 三伪影：host 开销/降频/尖峰）。

## 2. 本卡已确立的设计规律（违反必败，全有数据）

**「多而小的 CTA」**：一切加大单 CTA 工作的臂全败——
- decode TILE：t16 最优，t32 慢 10-27%、t64 慢 45-58%（`results/d3_dev/
  item6_splitkv/dtile_sweep.csv`）；
- num_stages=2 在 TD 态大劣化（bs32/kv16K：w1s1 1748µs vs w1s2 2603µs，
  `results/d3_dev/tdlaunch_sweep.csv`）；
- 段数（现有 kernel 上）：n16 已是最优——bs32/kv16K n8 倒挂 0.92×（`results/
  d3_dev/item3_segments/segm_ext.csv`）。**新 kernel 的收益必须来自结构
  （分段策略/归约融合/加载路径），不是简单调段数**；
- 正面对照：采样过滤 grid 16→64 programs = 3.4×（行间独立+占用翻倍）。

## 3. 目标形态与基线（官方 16K 场景实态，全 graph 口径实测）

模型 MiniCPM5-2B：42 层，GQA 16Q:2KV，head_dim 128，bf16 KV，paged
block_size=16，causal，window(-1,-1)，softcap 0，descale 全 None。
KV 读字节/层 = bs × kv × 2(K+V) × 2(kv_heads) × 128 × 2B = bs×kv×1024 B。

| 形态（decode 3D, TD=on, w1s1） | 基线 | GB/s | 要击败的数 |
|---|---|---|---|
| **bs32 / kv16384（官方 16K 常驻）** | **1748.0µs** | 307 | 验收线 <1520µs（1.15×），冲刺 1345µs（1.3×） |
| bs64 / kv16384 | 3035.5µs | 354 | 同比例 |
| bs64 / kv4096（4K 场景参考） | 731.0µs（TD+n8 叠加臂，tdx_ab.csv） | 367 | 不得回退 |
| TD=off 对照 | bs32/kv16K 1827µs；bs64/kv16K 3382µs（decode_surface.csv） | 262-317 | — |

官方场景信息：16K 场景 conc=64 但 KV 池 588,112 tokens → 全长准入下同驻
≈33 → decode bs≈33（capture 32/40 档），kv∈[16K,17.4K] 单调增长；4K 场景
bs≤64、kv∈[4K,5.1K]。**首攻 bs32-40/kv16-18K**。

## 4. 现有 kernel 逆向事实（site-packages vllm 0.24，路径行号实测）

文件 `/usr/local/lib/python3.12/site-packages/vllm/v1/attention/ops/
triton_unified_attention.py`（wrapper `unified_attention` L780 起；kernel
`kernel_unified_attention` @triton.jit；`reduce_segments` L663）：

- 3D 路径即 split-KV：grid = `(total_num_q_blocks, num_kv_heads, segments)`，
  段数 = `NUM_PAR_SOFTMAX_SEGMENTS=16`（metadata builder 常量，triton_attn.py
  L55/L150-171）；每段串行扫 kv 的 `tiles_per_segment = cdiv(seq_len,
  segments×TILE)` 个 tile，写 segm 缓冲 (result,max,expsum)，再由
  `reduce_segments`（grid=(q_tokens, num_q_heads)）二段合并。
- decode 常量：BLOCK_M=16（16Q:2KV→num_queries_per_kv=8，pow2）、BLOCK_Q=2、
  **TILE_SIZE_DECODE=16**（bf16）、w1s1（插件 launch 表注入）。
- segm 缓冲三件套 fp32：`(seq_threshold_3D=64, 16 q-heads, segments, 128)`
  ——**按序列分配、按 token 索引**（上游 #48076：q_len>1 越界的根因，你的
  kernel 若支持 q>1 必须改成 per-token 语义）。
- 2D/3D 判据 L969-978（segm 非 None 且 q≤1 且 bs≤64）；USE_TD 分支：加载走
  `tl.make_tensor_descriptor`（上游 PR #40327），TD 时 TILE 被 min(tile,
  block_size=16) 钳制。**TD 在本栈实测可用且是当前最优态**——新 kernel 应
  原生用 descriptor 加载（或至少不慢于 TD 态基线）。
- 参考模板（仓库内，两段式骨架可直接抄结构）：
  `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/
  triton_decode_attention.py`（`_fwd_kernel_stage1` NUM_KV_SPLITS constexpr +
  stage2 合并）。

## 5. 设计要求（按 FlashInfer Algorithm 1 + 本卡规律裁剪）

1. **自适应分段**：target_CTA = k×16（k=1-2），`split = clamp(ceil(target_CTA
   /(bs×2)), 1, cdiv(kv, MIN_SEG_TOKENS))`，MIN_SEG≈8-16 tile；**split==1 的
   请求 writethrough 直写输出、跳过归约 launch**（FlashInfer D.2/D.3）。
   短 kv（≤2K）不切，长 kv（≥8K）才切——与 ③ 的「官方 16K 形态 n16 无益」
   不矛盾：你要赢的是**结构开销**（空段、固定 16 段的负载不均、二次 launch），
   不是段数本身。
2. **确定性两段归约**：per (token, q-head) 一个 program，读 (result, max,
   expsum) 按 max 归一合并；无 atomic。数值防护照抄上游：整行 mask 时
   `m>-inf` 防护、L 初值 1.0/M=-inf、expsum==0→输出 0。
3. **形态纪律**：TILE=16 基线（如试 32 必须带 A/B 臂）；w1s1 起步；smem 预算
   ≤128KB（TILE=16×128×2B×2(K+V)×流水级——留 1-2 级余量）。
4. **GQA 头组打包**：一个 CTA 吃满 16 q-head（BLOCK_M=16、CTA=kv-head），
   `tl.dot` 保留（elementwise+sum 不会被识别为 MMA）。
5. **不做** prefill/decode 融合单 kernel（IBM 实测 ≥2× 掉速）；不做
   FlashDecoding++「统一 max」捷径（数值语义变化）。

## 6. 集成方式（dispatch 层，零 vllm 源码改动）

仿 `vllm_fl/dispatch/backends/vendor/iluvatar/attn_launch_config.py` 两跳模式，
新文件 `vllm_fl/dispatch/backends/vendor/iluvatar/attn_splitkv.py`：

- hop1 式包装 `vllm.v1.attention.ops.triton_unified_attention.unified_attention`
  （注意 `vllm.v1.attention.backends.triton_attn` 按名导入需同步换绑，参照
  attn_launch_config L232-237）：仅当 **max_seqlen_q≤1 且 bs≤64 且未启用 KV
  量化（kv_quant_mode==NONE）** 时路由到自研 kernel，否则原样透传；
- env `VLLM_FL_ATTN_SPLITKV`（默认 0）；幂等哨兵、try 保护、不替换对象身份；
- **cudagraph 相容**：路由决策只依赖 (bs, q≤1) 等 capture 期静态量——grid/
  split 按 capture bs 查表确定，graph 内行为确定；segm 工作缓冲按最大 split
  预分配（参照 sample_filter_mode 的自管 buffer 缓存）；
- 新 kernel 与段缓冲的分配放 install 时或首调用惰性（capture 前完成）。

## 7. 数值与验收（无卡阶段可写的测试）

- 单测 `tests/unit_tests/attention/test_splitkv_decode.py`（真机跑）：
  随机输入对照 `unified_attention`（同 segm/TD 态），max_rel ≤1%、无 NaN；
  边界 kv∈{1,15,16,17,4096,16384,17408}、bs∈{1,8,32,33,64}、block_table
  顺序+乱序（randperm）双形态、causal 边界、GQA 布局 stride。
- graph 微基准：扩 `scripts/d2/attn_surface_probe.py` 加 suite "splitkv"
  （照抄 run_dtile_suite 的 override 骨架，把 launch override 换成
  VLLM_FL_ATTN_SPLITKV 路由开关），形态固定 §3 表格四行，CSV 列沿用。
- 验收判据见 §9。

## 8. 次日 6h 真机验证计划（按序，含时间盒）

1. **T0 编译/装载自检（15min）**：`pip install --no-deps --no-build-isolation
   ./vllm-plugin-FL` → 单测跑通（§7）。失败即修，不动下一步。
2. **T1 kernel 微基准（20min）**：`python3 scripts/d2/attn_surface_probe.py
   --suite splitkv --out-dir results/d3_dev/item6_splitkv`，对照 §3 基线。
   **门槛：bs32/kv16K ≥1.15×（<1520µs）才继续；未达 → 数据结案收工**
   （不要现场调参超 30min，那是无卡阶段该做完的事）。
3. **T2 serve 集成烟测（45min）**：B 臂 env + `VLLM_FL_ATTN_SPLITKV=1` 起
   serve（红线命令见下）→ `python3 scripts/d3/parity_check.py compare`
   （预期 <16/16，记录分歧率）→ decode 代理 bench `--test-cases
   '[[256,1024,64,64]]'`（4 轮，ITL 对照 B 态基线 ~16.2ms）。
4. **T3 官方 16K 单轮试金石（35min）**：新鲜 serve + `--test-cases
   '[[16384,1024,64,128]]'` 一轮即看（timeout 300 截断亦可）——**对照 B16
   轮值 1358/1573/1569/1570**。≥+5% 才值得 T4。
5. **T4 官方 16K 完整 4 轮（~2h）**：`scripts/d3/run_official_redo.sh B16`
   改 env 或手动等价序列；每场景独立新鲜 serve（EngineCore 死亡时钟 27-80min
   随机，死亡=换 serve 重跑该场景，勿长跑串联）。
6. **T5 105 题精度门（45min）**：同 readme §5 命令，红线 ≥0.95（B 态基线
   0.9714）。过线 → 更新 docs/improvements-master.md §0 与交付包数字。
7. 剩余时间：缓冲/失败重试/文档回写。

serve 红线命令（一字不动，只加 env）：
`VLLM_PLUGINS=fl VLLM_FL_ATTN_TILE=1 VLLM_FL_ATTN_TD=1 VLLM_FL_ATTN_SPLITKV=1
vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 --compilation-config
'{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm
--gpu-memory-utilization 0.85 --max-model-len 131072`

## 9. 判据 / 回退 / 交付

- **性能门槛**：T1 ≥1.15×（bs32/kv16K vs 1748µs）；T3 单轮 ≥1620（vs B16
  热态均值 1570 +5%）；T4 丢首均值即新官方数。
- **精度门槛**：T5 ≥0.95（目标 ≥0.96 与 B 态一致量级）。
- **回退**：`VLLM_FL_ATTN_SPLITKV=0` 一键回 B 臂；任何阶段失败按 §8 时间盒
  结案留档，不硬撑。
- **交付**：kernel + 集成 + 单测 + probe suite + results 数据 + DEVLOG 增节
  + docs/improvements-master.md §0/§3 行更新（⑥ 从「数据结案」改判）。

## 10. 数据档案索引（全在本仓库）

- 全部基线：`results/20261115_bottleneck_deepdive/`（decode_surface/
  td_ab/tdx_ab/segm_sweep/ref_table/scatter）+ `results/d3_dev/`（item3
  段数扩展、item6 dtile、tdlaunch_sweep、tdx 叠加）。
- 官方战役：`results/d3_official_1116_2222/`（三臂+补跑+105 题，README 有
  全表）。
- 调研报告（FlashInfer/上游考古/死案）：`docs/bottleneck-research-report.md`。
- 台账总账：`docs/improvements-master.md`；开发日志 `results/d3_dev/DEVLOG.md`。

## 11. 运行纪律速查（真机阶段）

GPU 独占串行（ixsmi <3000MiB 门禁）；Bash cwd 会重置一律绝对路径；单次
Bash ≤10min 分次轮询；pkill 禁 -f 自匹配（按精确 pid）；serve 用完即杀
（EngineCore 死亡时钟）；results 元数据滞后判存在用 find；官方 bench 脚本
原件不改（--test-cases 可换）；改插件必 pip install 后再 serve。

## 12. 2026-11-18 深夜补强实验（TD 态段数扫 + 官方精确形态基线）——计划重排

数据：`results/d3_dev/item6_splitkv/td_segm_sweep.csv`（graph 口径，TD=on，w1s1）。

### 12.1 核心发现：TD 态最优段数反转（n16 → n32）

| 形态 | n4 | n8 | n16（现状） | n32 | n32 收益 |
|---|---|---|---|---|---|
| bs32/kv16384 | 1775.5 | 1820.3 | 1748.0 | **1374.3** | **1.27×**（307→391GB/s） |
| bs64/kv16384 | 3801.5 | 3597.5 | 3035.5 | **2793.4** | 1.09×（354→384GB/s） |

TD-off 时（③ 台账）n8/n16 最优、n32 倒挂；TD=on 后方向反转（n4/n8 变差、
n32 变优）——向量化加载让每 CTA 更快消化 tile，更多段（更少 tile/段）成为
正收益。这是 §5.1 自适应分段论点的直接实证，也推翻了 ③ 的「官方形态段数
无收益」结论（那是 TD-off 口径）。

### 12.2 官方精确形态基线（TD=on, n16, w1s1）

| 形态 | µs | GB/s |
|---|---|---|
| bs32/kv17408 | 1482.7 | **385** |
| bs40/kv16384 | 1721.6 | **390** |
| bs40/kv17408 | 1967.9 | 362 |

真实形态带宽 362-390GB/s（高于 §3 的 307 口径）→ 自写 kernel 的真实
headroom 修正为 **~1.7×**（390 vs 660）。另：kv17408 比 kv16384 快 ~15%
（1088 vs 1024 块的波形效应，未深究，复测确认即可）。

### 12.3 明日 6h 计划重排：段数生产化（P0'）优先于自写 kernel（降级 P2）

**P0'（快赢，1-1.5h）：段数 patch 生产化——③ 的半途设计 + 本轮数据**
- 实现：包装 `TritonAttentionMetadataBuilder.__init__`（segm 三件套按
  MAX_SEG=32 分配）+ 包装 `build_for_cudagraph_capture` 按 capture bs 查表
  设 `self.num_par_softmax_segments`（表：bs≤48→32，其余 16；capture 后还原）。
  参照 attn_tile_mode.py 的 _get_tile_size patch + sample_filter_mode 的自管
  buffer 模式。env `VLLM_FL_ATTN_SEGMENTS`（默认 0）。
- 判据：T1' probe（bs32/kv16K n32 ≤1450µs 复现）→ T2' serve 烟测 decode 代理
  ITL −3~5%（attn 占 ITL~80%×层内 1.27×，端到端 16K 估 +8~15%）→ T3' 官方
  16K 单轮 ≥1650（vs B16 热态 1570+5%）→ T4' 完整 4 轮 + T5' 105 题（段数
  改归约分组=非 bit-exact，门必须过）。
- **注意与 ③ 结论的关系**：③ 关的是「TD-off 态无收益」；本节是 TD 态数据。
  若 P0' 落地，总账 ③ 从「数据结案」改判「TD 态复活」。

**P2（可选，剩余时间）：自写 split-KV kernel**——P0' 落地后基线变为
n32 态（bs32/kv16K 1374µs / 452GB/s 口径@kv17408 外推），headroom 缩至
~1.45×，按 §5-§7 原计划执行但门槛重估：T1 ≥1.15× vs n32 态新基线。

**P1（保险，10min）：** 复测 bs32/kv16K n32 与 kv17408 n16/n32（确认波形
效应非伪影）后再投入 P0' 编码。

### 12.4 无卡开发 agent prompt（复制即用；P0' 与 P2 二选一或按序）

```
角色：FlagOS S2 无卡开发 agent（只写代码，不碰 GPU；禁止 import triton 编译/
运行 GPU 代码/pip 安装/改 serve 参数）。仓库 /flagos_比赛/flagos_xx。

第一步必读（唯一信息源）：docs/splitkv-kernel-handoff.md——重点 §12（最新
数据与计划重排：P0' 段数生产化优先）、§2（设计规律）、§4（kernel 逆向）、
§6（集成模式）、§8-§9（次日真机计划与判据）。与本文档冲突以 §12 为准。

任务 A（P0'，先做）：段数 patch 生产化
  新文件 vllm_fl/dispatch/backends/vendor/iluvatar/attn_segments_mode.py：
  包装 TritonAttentionMetadataBuilder.__init__（segm output/max/expsum 三件套
  按 MAX_SEG=32 分配，形状 (threshold,16,32,128)fp32）+ 包装
  build_for_cudagraph_capture（按 common_attn_metadata.num_reqs 查表：bs≤48
  →num_par_softmax_segments=32，否则 16；临时设置调 orig 后还原）+ env
  VLLM_FL_ATTN_SEGMENTS（默认 0）+ 幂等哨兵/try 保护/首启用打日志
  fl.attn-segments。iluvatar.py 接线仿 install_attn_tile_mode_for_iluvatar。
  单测（无卡写好不跑）：构造伪 builder 对象验证表路由与 buffer 尺寸；
  cudagraph 相容论证写进 docstring（段数=capture bs 的纯函数）。

任务 B（P2，A 完成后）：attn_splitkv.py 自写 kernel——按交接书 §5/§6/§7
  全量要求（TILE=16/w1s1/自适应分段/writethrough/确定性两段归约/无 atomic/
  bare jit/静态化），基线用 §12.2 精确形态 + P0' 落地后的 n32 态数字。

交付：代码 + 单测 + scripts/d2/attn_surface_probe.py 增 suite（"segm32"：
VLLM_FL_ATTN_SEGMENTS 路由下的四形态复测；"splitkv" 照 §7）+ DEVLOG 增节
（设计决策对照 §2/§5 逐条）+ 在 §8 T0 前插「无卡阶段已知风险清单 5 条与
预案」。明日真机会话按 §8+§12.3 执行（P1 复测 → P0' 判据链 → 可选 P2）。
```

## 13. Profile 数据包（2026-11-18 23:20-23:25 补测，kernel 作者必读）——含 P0' 降级修正

数据：`results/d3_dev/item6_splitkv/cta_scaling.csv`（12+4 臂，graph 口径，TD=on）。

### 13.1 CTA 标度曲线（bs32，w1s1）——现有结构的饱和点

| 配置 | CTAs | µs | GB/s |
|---|---|---|---|
| kv16K n16 | 1536 | 1648.1 | 326 |
| kv16K n32 | 3072 | **1373.0** | **391** ← kv16K 峰值 |
| kv16K n64 | 6144 | 1824.7 | 294（过切回落） |
| 任意 n×w2 | — | 全部劣化 15-35% | 双 warp 同步税 > 收益 |
| **kv17.4K n16** | 1536 | **1482-1856*** | **382-385** ← 无需 n32 已达峰值区 |
| kv17.4K n32 | 3072 | 1454.7 | 392（与 n16 持平） |
| **bs33/kv17.4K n16** | 1568 | **1539.4** | **382**（官方真形态） |
| bs33/kv17.4K n32 | 3136 | 1748.7 | 336（**n32 反而劣化**） |
| bs40/kv16K n32 | 3840 | 1710.5 | 392 |
| bs40/kv17.4K n32 | 3840 | 1813.9 | 393 |

*kv17408 的 n16 两次测 1482.7/1855.7，会话间方差 ~±10%（降频/波形敏感）。

### 13.2 三个判决性结论

1. **P0'（段数 n32）降级为「形态依赖，默认不做」**：1.27× 增益只存在于
   kv=16384 精确点（波形伪影：1024 块的 2 的幂对齐劣势）；官方 kv 漂移区
   （16.4K→17.4K）里 n16 与 n32 在 ~390GB/s 汇合，bs33 处 n32 还劣化 12%。
   **若明日仍想试 P0'：必须先跑 kv∈{16500,17000,17408} 三点平均再判**，
   否则跳过直接 P2。
2. **现有 kernel 在官方真实形态的基线 = 382-393GB/s（660 的 58-60%）**，
   且 CTA 数已饱和（3072 CTAs 是峰值区，n64/w2 全回落）——**剩余 1.7× 缺口
   的本质是 per-warp 访存并行度（MLP）**：w1s1 单 warp 串行发射 descriptor
   加载，每 tile 的 load→dot→load→softmax 依赖链无重叠。
3. **已量化的非瓶颈**（排除项，kernel 作者不用再看）：segm 缓冲往返流量
   ~2-6%（537MB KV 读的占比）；reduce_segments 第二 launch ~10µs 量级
   （graph 内）；散射块表 0-5%（scatter_ab.csv）；smem 占用远低于 128KB
   （TILE16×128×2×2B=8KB/CTA）。**唯一战场：让每个 warp 同时有更多在飞的
   加载**。

### 13.3 自写 kernel 的假设阶梯（按优先级，每条都是可单测的机制）

1. **手动双缓冲预取**（首攻）：w1 CTA 内显式预取下一 tile 的 K/V（在当前
   tile 的 dot 之前发起加载），模仿 s=2 流水但手工控制——注意编译器 s=2 在
   现 kernel 上大劣化（2603µs），所以必须手动排指令而不是开 num_stages。
2. **多段常驻 CTA**（persistent）：一个 CTA 顺序处理本 (seq,kvh) 的多段，
   段间加载与归约写重叠；CTA 数回落到 1536 但在飞加载翻倍。
3. **descriptor 宽形状**：若 backend 支持非 16 倍内维，单次加载 2 个 tile
   的 K（等效 TILE=32 的读宽度但保持两个独立 16 宽的 dot，避开 t32 的劣化
   机制——t32 慢的原因推测是 dot 形状而非读宽度，值得 A/B 验证）。
4. 验收线不变：**官方形态（bs32-40/kv16.4-17.4K 平均）≥1.15× vs 390GB/s
   （≤1350µs@bs32 口径）**；kv=16384 单点的任何数字不作为判据（波形伪影区）。

### 13.4 明日 6h 修订版顺序

P1 复测（kv 三点平均，10min，决定 P0' 生死）→ **P2 自写 kernel（主攻，
4h：T0 编译+单测 30min → T1 probe 对照 390GB/s 基线 20min → T2 serve 烟测
45min → T3 官方 16K 单轮试金石 35min → T4 完整 4 轮 2h → T5 105 题 45min）**
→ 若 T1 未达 1.15×：数据结案，成绩锁定交付包（zip 已在
/flagos_比赛/submission/参赛队伍.zip，4K 3414.58/16K 1570.62/精度 0.9714）。

§12 的 prompt 中「任务 A（P0'）优先」据此改为：**任务 A 仅需完成「kv 三点
复测脚本」（probe suite "segm32"），生产化 patch 降为可选**；任务 B（自写
kernel）为主攻，按 §13.3 假设阶梯实现。
