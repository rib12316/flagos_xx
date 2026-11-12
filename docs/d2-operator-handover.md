# D2 交接文档 — W8A8 算子加速线（供新 agent 接手）

> 写于 2026-11-12，交接人：D2 会话 agent（feat/d2-w8a8-quant 分支，9 个提交，未合入 main）。
> **新核心目标：通过算子优化打通真量化（W8A8）推理加速**——精度线已由前一阶段解决（配方定案），
> 性能线卡在 int8 GEMM 算子（详见 §4），本文件给出全部上下文、数据与首要线索。

---

## 0. 一页速览

| 维度 | 状态 |
|---|---|
| 精度配方 | **定案**：per-channel RTN（clip=1.0）+ SmoothQuant-lite α=0.5 离线折叠 + per-token 动态激活。30 题冻结集 greedy：acc 0.9333=对照、same_ans 0.9333、Δacc 0 |
| 转换产物 | `/workspace/MiniCPM5-2B-w8a8`（294 张量，round-trip 逐位全过，数值 ≤0.76% 输出幅值，config 注入完成） |
| 阻断修复 | G2 能力门（sm71<75 错杀加载）已修：`packed.py` override `get_min_capability()=71` + 单测 |
| **性能** | **❌ 未过**：FlagGems Triton int8 scaled_mm 在 15 形状上全面慢于 bf16（0.10–0.90×）。数值正确（rel 1e-3/cos 0.9999999），纯吞吐问题 |
| **头号线索** | `torch._int_mm`（cuBLAS int8 GEMM，int32 累加）实测**位精确**且比 Triton scaled_mm **快 4-8×**，对 bf16 有望 ≥1.5×——见 §5.1，这是最短路径 |
| 未验证 | serve 真实路径（冒烟/官方 bench/官方 105 精度）——等新算子接通后一并走 |

---

## 1. 环境与不变约束（全部继续有效）

- 实例：天数 BI-V150 单卡 32GB（`/usr/local/corex/bin/ixsmi`），torch 2.10 corex，vllm 0.24.0+empty，
  triton 3.2 flagtree，flag_gems 5.3.5，compute capability **(7,1)**。
- 仓库：`/flagos_比赛/flagos_xx`（唯一落地处）。**git 只能走 SSH**（`git@github.com:rib12316/flagos_xx.git`，
  https 智能协议在本机挂死——npx/npm 装 GitHub 上的东西都会失败，用 SSH clone 替代）。
- 红线：serve 参数锁死（`--port 9031 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072` + `VLLM_PLUGINS=fl`）；
  bench 脚本原件不可改（`vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py`）；pip 只从本仓库装
  （provenance 四重验证：direct_url 指向本仓库 / site-packages diff 一致 / 无旧仓库符号 / flag_gems==5.3.5）；
  校准语料禁 math_500/gsm8k（已用模型自生成通用文本，合规）；GPU 独占串行；采纳数字 ≥3 次中位。
- 对照基线（main 分支 D0+D1）：**4K 2668.42 / 16K 1276.62 total tok/s / math_500 L3 96.2%**。
- 环境陷阱（血泪教训，全部踩过）：
  1. `pkill -f "vllm"` 在交互命令里会**自杀**（匹配到自己的 bash 命令行）——按精确 pid kill。
  2. 长任务用 `setsid nohup ... < /dev/null > log 2>&1 &` detach，且**会话拆除不保证不杀它**——重要
     运行要尽早落盘中间产物，恢复后先查 `/tmp/*.log` 与 results 目录。
  3. Bash 工具单次超时上限 10 分钟——长等待用轮询。
  4. **libentry 流竞争**（框架级 bug，详见 §3.3）：进程内直调 `@libentry()` 包装的 Triton kernel，
     batch>1 时下游可能读到未就绪输入。自写 kernel **不要用 libentry 包装**（裸 triton.jit 即可），
     或走 dispatch 层。
- 已安装 skills（本机 `~/.claude/skills/`，仓库 `/root/iluvatar-agent-skills`）：
  `iluvatar-ixkncli-guide`（kernel profile，规范屋顶线归因）/ `iluvatar-ixobjdump-guide`（SASS/IR 查看，
  验证指令是否下探 int8 快路径）/ `iluvatar-ixsys-guide`（系统级 trace）/ `iluvatar-ixrt`（IxRT INT8
  官方引擎参考——384 TOPS 标称靠 ixTE 私有栈，torch/Triton 只摸到 12%）。

## 2. 已有改动全清单（9 commits on feat/d2-w8a8-quant）

### 2.1 插件侧（vllm-plugin-FL/）

| 文件 | 改动 | 说明 |
|---|---|---|
| `vllm_fl/quantization/w8a8/packed.py` | **+get_min_capability()=71** | G2 修复：上游 `CompressedTensorsW8A8Int8` 门槛 75，sm71 加载 packed checkpoint 即 RuntimeError。已 pip install 装机（`pip install --no-deps --no-build-isolation ./vllm-plugin-FL`，**不加 --no-deps 会因外网解析挂死**） |
| `tests/unit_tests/quantization/test_packed_int8.py` | +能力门回归测试 | 6/6 过 |
| `tools/quantize_int8_rtn.py` | **新增**（红线合规核心件） | bf16→packed int8 转换：RTN per-channel（clip α 或 auto MSE）+ SmoothQuant-lite 折叠（调 `scripts/d2/common.py` 同一折叠数学）+ offset-binary-128 打包 int32 [N,K/4] + config 注入（见 §3.1 schema）+ 三重自检。幂等（seed 1234）。注意：本地有一处未提交小改（自检数值判据改全局尺度相对 ≤2%，勿丢） |

### 2.2 评测/工具侧（scripts/d2/）

| 文件 | 作用 | 新 agent 复用点 |
|---|---|---|
| `common.py` | Recipe 注册表、冻结筛查集（L3 前 30/105 按 unique_id 排序，sha 落盘）、evalscope 同口径判分、**SmoothQuant-lite 折叠数学（唯一定义处）**、RTN、RESULT 行惯例 | 折叠/打包/判分全部复用，勿另写一份 |
| `fake_quant_eval.py` | 假量化筛查 harness：offline LLM（`VLLM_ENABLE_V1_MULTIPROCESSING=0`、enforce_eager、gmu 0.80）+ monkeypatch `UnquantizedLinearMethod.apply` + `_int_mm` 位精确 int32 仿真 + 进程内活体折叠手术 + MSE 预扫（`--mse-scan`） | **换新 kernel 后重验精度直接用它**（`--arms r_L2a05` 等，30 题 ~50min/臂）；emu=int32 的算术 == 任何 int32 累加的 int8 GEMM == `_int_mm`，所以精度结论对 `_int_mm` 路线的真实路径直接成立 |
| `collect_act_stats.py` | 激活统计（48 条内嵌非数学 prompt 自生成，168 站点 + o_proj GQA 256 维聚合） | 已有产物 `results/20261107_d2_actstats/act_stats.json`，无需重跑 |
| `bench_int8_gemm.py` | **15 形状 × 3 臂微基准**（A=bf16 F.linear / B=quant+flag_gems scaled_mm 成对+mm-only / C=证伪臂 int8→bf16 cast GEMM）+ 冒烟三查 + 真实权重对拍 + 屋顶线 | **加新臂 D（_int_mm+epilogue）改这里**，判据/口径现成 |
| `run_d2_eval.sh` | T2 端到端 wrapper（镜像 run_t1_d1.sh：serve 转换 ckpt + 激活三查 + 官方双场景 + 官方 105 精度） | kernel 打通后直接跑 |
| `debug_batch_m.py` | 流竞争隔离实验（同题 b=1 vs b=30 四臂 + 逐模块数值定位） | 排查 kernel 正确性问题时好用 |

### 2.3 台账/证据（results/）

- `d2-kernel-matrix.md` — **kernel 支持矩阵**（权重/激活 scale 支持形状、dispatch 三路径、sm71 结论、
  配方就绪表）——新 agent 必读
- `d2-precision-search.md` — 精度搜索台账（L0/L1a/L2 各配方全部数字 + 两条 bug 定位记录）
- `20261112_d2_hardware_gate_verdict.md` — **硬件闸门判定书**（§4 的原始出处）
- `20261107_d2_gemmbench/bench.json` — 15 形状三臂原始数据
- `20261107_d2_fakequant/*.json` — 各配方逐题记录（含文本头尾，供病理检查）
- `20261107_d2_actstats/act_stats.json` — 激活统计（SmoothQuant 用）

## 3. 已确证地面真相（新 agent 直接采用，勿重查）

### 3.1 W8A8 管线结构（vllm_fl/quantization/w8a8/）

- **加载链**：config.json `quantization_config` → `should_use_packed_w8a8`（int8_mode.py:56-72）→
  monkeypatch 的 `_get_scheme_from_parts`（packed.py:110-147）→ `CompressedTensorsPackedW8A8Int8`。
  精确 schema：`quant_method="compressed-tensors"`、`format="pack-quantized"`、
  `config_groups.*.weights={8,int,symmetric,static,strategy:"channel"}`、
  `input_activations={8,int,symmetric,dynamic,strategy:"token"}`、`targets:["Linear"]`
  （类名子串匹配 → embed/lm_head 天然 bf16，无需 ignore）。
- **checkpoint 张量**：`weight_packed` int32 [N,K/4]（offset-binary-128，与 `unpack_uint8b128_int32`
  严格互逆）；`weight_scale` **fp32 [N,1]**（加载断言严格）；`weight_shape` int64 [2]=[K,N]。
- **运行链**（`linear.py:96-116 apply_weights`）：`CachedOp("dynamic_per_token_quant_int8")(x_2d)`
  → `flag_gems.scaled_mm(x_q, weight[K,N], x_scale[M,1], weight_scale[N], out_dtype=bf16)`。
  权重在 `process_weights_after_loading` 转成 **[K,N] int8 contiguous**。
- **运行时融合模块**：qkv_proj [2560,2048]（q 0:2048/k 2048:2304/v 2304:2560）、gate_up_proj
  [12288,2048]（gate 0:6144/up 6144:12288），片宽可读 `layer.logical_widths`。

### 3.2 dispatch 路径（性能问题的根源所在）

`flag_gems/ops/scaled_mm.py:425 scaled_mm` → `_scaled_mm_impl` 依次尝试：
1. **cutlass 快路径**：`_can_use_cutlass_scaled_mm` 门 `90 ≤ sm < 100`（:292-294）→ **sm71 必不触发**；
2. **Triton 通用 int8 kernel**（`scaled_mm_kernel` :51-136，int8 输入 `ACC_DTYPE=tl.int32`，libtuner
   自动调优）→ **当前实际路径，性能不行**（§4）；
3. Ascend 变体仅 vendor==ascend。
- `_iluvatar/matmul_int8.py` 是 bf16 降级占位（输出 dtype 硬编码 bf16，:33-34/:118），**不在 w8a8 链上**。

### 3.3 两个已修/已绕的框架 bug（报告素材 + 新 agent 注意）

1. **G2 能力门**（已修，见 §2.1）。
2. **libentry 流竞争**（已绕过未修）：`@libentry()` 包装的 Triton kernel（如
   `dynamic_per_token_quant_int8_flaggems_triton`）进程内直调时，b>1 下游读到未就绪输入（症状：
   生成瞬死 EOS / 满长度游荡 / acc 崩）。定位双盲：patched_apply 加全 sync 恢复健康；换纯 torch 量化链
   （不加 sync）同样恢复。libentry 的 event-timing fallback 疑走旁路流。**真实 serve 路径经
   CachedOp/dispatch 包装——是否同病未验证**（新 kernel 上 serve 后第一件事就是 fake-vs-real 对拍）。

### 3.4 精度结论（30 题冻结集，greedy，对照 0.9333）

| 配方 | acc_q | same_ans | 备注 |
|---|---|---|---|
| L0 全层 RTN | 0.9333 | 0.90 | 起点，Δacc=0 |
| L1a clip 0.99 | 0.8667 | 0.8667 | 证伪（权重 MSE≠任务一致） |
| **L2 smooth α=0.5** | **0.9333** | **0.9333** | **定案配方**（checkpoint 已按此转换） |
| L2 α=0.6/0.7 | 0.9333 | 0.90 | 未达线 |
| L2 α=0.8 | 0.8667 | 0.8333 | 过平滑，证伪 |
| L2 α=0.3 | 0.9333 | 0.9333 | 打平 0.5 但 dlen +34%/trunc 9/30 病理 |

- SmoothQuant-lite 折叠点（数学精确，逐行核验）：RMSNorm w 乘在 sqrt 外→norm 权重/=s；RoPE 只作用
  q/k→o_proj 的 s 折进 v_proj 行（**GQA 16Q/2KV 必须按 8 头组聚合 2048→256**）；SiluAndMul 对 up
  分支线性→down_proj 的 s 折进 up_proj 行。激活离群主战场：down_proj 输入（L41 max/med=121）。
- 105 题全量 fake 复核未完成（进程被会话拆除杀掉两次）——**精度数字的最终口径以官方 evalscope 真实
  路径为准**（run_d2_eval.sh eval 阶段），官方配方 temp=1.0 与 greedy 筛查是两套口径。

## 4. 性能问题陈述（Phase 3 实测，判定书为证）

判据：任一 M=1024 ≥1.5× 或 M=64 ≥1.2×（int8 对 bf16）。实测 **15/15 形状 <1.0×**：

| (K,N) | M=64 | M=128 | M=1024 |
|---|---|---|---|
| qkv 2048×2560 | 0.21 | 0.14 | 0.54 |
| o 2048×2048 | 0.17 | 0.10 | 0.34 |
| gate_up 2048×12288 | 0.53 | 0.47 | 0.45 |
| down 6144×2048 | 0.34 | 0.38 | 0.33 |
| lm_head 参照 2048×130560 | 0.90 | 0.63 | 0.50 |

（比值为 quant+mm 成对 / bf16 F.linear；mm-only 数据在 bench.json）

归因：int8 mm-only 实测带宽 42.6–253.7 GB/s（qkv 42.6 / gate_up 160 / lm_head 254），对照 bf16
cuBLAS 同形状 ≈553 GB/s——**Triton int8 kernel 带宽效率不足 bf16 的一半**，权重字节减半的优势被吞。
M=1024 TOPS 最高 45.5（标称 384 的 12%；384 需 ixTE 私有栈）。证伪臂（int8→bf16 cast GEMM）同样慢
→ 非失调，是路径本身。数值侧全过（rel median 1.35e-03 / cos 0.9999999）。

## 5. 给新 agent 的开发路线（建议）

### 5.1 ⭐ 头号线索：`torch._int_mm` 路线（最短路径，先验证它）

本会话实测（scripts/d2 内多次验证）：
- `torch._int_mm(A[M,K] int8, B[K,N] int8) → int32 [M,N]`，与逐行 fp32 整值矩阵乘**全行位精确**；
- 速度（单次调用）：M=30, K=2048, N=12288 → **0.058ms**（fp32 matmul 0.453ms）；M=30, K=6144,
  N=2048 → 0.051ms；M=4500, K=2048, N=12288 → 1.047ms（比 fp32 快 9.7×）；
- 对照 bf16 F.linear：gate_up M=64 = 0.121ms、down M=64 = 0.077ms、qkv M=64 = 0.044ms。
  → **_int_mm + 小型 epilogue（×x_scale×w_scale、cast bf16）对 bf16 有望 1.2–2×**，可能直接过闸门。
- 约束：M>16（小 M 垫零已验证位精确，见 fake_quant_eval.py 的 _PAD_BUF 逻辑）；A、B 须 int8
  contiguous（权重运行时本就是 [K,N] contiguous，正好）。

**第一步建议**：`bench_int8_gemm.py` 加 D 臂（`_int_mm` + epilogue）跑 15 形状——若过判据，
第二步在 dispatch 层接线：新增 FL dispatch op（如 `w8a8_int8_linear`）或直接改
`vllm_fl/quantization/w8a8/linear.py apply_weights` 走 `_int_mm + epilogue`（保留 flag_gems
scaled_mm 作 fallback 顺序位），这属于**框架内开发量**（合规核心论据）。epilogue 可以先纯 torch
三算子（int32→fp32 cast、×s、cast bf16），再用 ixkncli profile 决定是否值得合成单 kernel。
激活量化侧：现 Triton kernel 有 libentry 竞争前科——自写**裸 triton.jit**（非 libentry）逐 token
量化 kernel，或直接 torch 链起步再 profile。

### 5.2 二号线索：修 Triton scaled_mm kernel 本身

用 `iluvatar-ixkncli-guide` 对 `scaled_mm_kernel`（flag_gems/ops/scaled_mm.py:51-136）做 profile，
判 memory/compute bound；`iluvatar-ixobjdump-guide` 看生成的 SASS 有没有用上 int8 双发/DP4A 类
指令；对照 `iluvatar-ixrt` skill 里 IxRT INT8 路径的实现（官方吃到算力的引擎）抄 tile/向量化策略。
libtuner 的自动调优可能没为 sm71 出好配置（num_warps/num_stages/BLOCK 尺度）。

### 5.3 三号线索：cutlass 门

`_can_use_cutlass_scaled_mm` 的 sm90 门是 NVIDIA 语义的硬移植；BI-V150 自报 (7,1) 但其 int8
tensor core 能力是否存在以 ixobjdump/ixrt 证据为准。若存在等价快路径，改门+适配是正道但工作量
大，放最后。

### 5.4 kernel 打通后（顺序照旧）

1. serve 冒烟：`bash scripts/d2/run_d2_eval.sh`（D2_MODEL=/workspace/MiniCPM5-2B-w8a8）——激活三查
   （scheme 装载 / 权重显存 −1.84GiB / KV tokens ≈568K vs bf16 参照 ~518K）
2. fake-vs-real 30 题对拍（fake_quant_eval 结果 vs serve greedy 逐题比对；显著偏离=kernel 数值缺陷，
   注意 libentry 竞争在真实路径的复现）
3. 官方双场景 bench ≥3 次中位 vs 4K 2668.42 / 16K 1276.62
4. 官方 evalscope 105 题精度（≥0.955 目标；当前配方 30 题口径 Δacc=0）
5. 耦合验证（[fl.sched] 同驻数）+ D0 重调检查点②（scripts/sweep_attn_launch.py --full）+ 台账回写

### 5.5 止损

- 时间盒到 11/17 硬止；`_int_mm` 路线若 M=64 档仍 <1.2×（decode 是带宽界，epilogue 开销占比大），
  考虑 quant+GEMM+epilogue 融合单 kernel（Triton 写，int32 累加语义必须与 fake-quant 位精确对齐）。
- 全败 → W8A16 weight-only（FlagGems weight_int8pack_mm 现成）或按原任务书记录止损。

## 6. 快速上下文恢复命令

```bash
cd /flagos_比赛/flagos_xx && git log main..feat/d2-w8a8-quant --oneline   # 9 commits
cat results/d2-kernel-matrix.md        # kernel 支持矩阵
cat results/20261112_d2_hardware_gate_verdict.md   # 性能判定书 + 数据
cat results/d2-precision-search.md     # 精度台账
python scripts/d2/bench_int8_gemm.py   # 复跑微基准（GPU 独占门禁内置）
ls /workspace/MiniCPM5-2B-w8a8/        # 转换产物（config.json 含 quantization_config）
```
