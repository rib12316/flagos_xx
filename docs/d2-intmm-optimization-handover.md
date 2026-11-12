# D2 交接文档 — _int_mm 算子线成果与下一步算子优化（供新 agent 接手）

> 写于 2026-11-13 凌晨，交接人：D2 算子优化会话 agent（feat/d2-w8a8-quant，本会话新增 8 commits，
> 累计 17 commits 未合入 main）。前序交接：`docs/d2-operator-handover.md`（精度线+硬件闸门判定，
> 其地面真相继续有效，本文不重复）。
> **新核心目标：算子层面继续深挖（自实现融合算子等），把 W8A8 真量化推理加速从「过闸门」
> 推向「显著加速」**。当前 int8 链 GPU 侧只有 42% 标称效率，且存在结构性带宽浪费（见 §5）。

---

## 0. 一页速览

| 维度 | 状态 |
|---|---|
| 硬件闸门 | **✅ 过**：down 6144×2048 M=1024 = **1.604×**（≥1.5 判据）；M=64 档 gate_up **1.753×** / down **1.410×** / qkv **1.308×**（≥1.2）。graph 口径（两臂 CUDA graph 同法捕获重放），判定书 `results/20261112_d2_intmm_gate_pass.md` |
| 路径 | `torch._int_mm`（cuBLAS int8, int32 累加）+ 裸 triton.jit 逐 token 量化 + 融合 epilogue 单 kernel，全部经 dispatch op `w8a8_dynamic_per_token_linear`（intmm prio160 > flag_gems scaled_mm 150 > reference） |
| 位精确 | 整链与 fake-quant emu **位对齐**（62+22 例单测）：量化 kernel 复刻 torch 标量除法乘倒数语义；epilogue 与 torch 链逐位相等（bias 路径 FMA 1ulp 容差，模型无 bias） |
| serve | **✅ 能起能跑**：KV 582K tokens（bf16 参照 ~518K），cudagraph FULL_DECODE_ONLY 含 M=16 档捕获通过，日志确认 `flagos_bare` 量化 + `flagos_intmm` GEMM 选中、无 fallback |
| fake-vs-real | ✅ gate 过：acc_real 0.90 vs fake 0.9333（差 1 题），same_ans 0.90；3 个 diff 全为截断游荡形态（2 题 fake 亦错），非 kernel 缺陷。`results/20261112_d2_fakevsreal/compare.json` |
| 快速端到端烟测 | **arm1(w8a8) 已完 / arm2(bf16) 进行中**：`/tmp/d2_quick_e2e.sh`（同参 serve + 小场景 [2048,512,16,32]/[512,1024,16,32] 官方 bench 原件换参），结果 `results/20261112_d2_e2e_quick/`。w8a8 case1(prefill 重) 稳态 Total ~1380 tok/s。**接手第一件事：收 bf16 arm 结果出对比**（若未完照脚本 arm2 段补跑） |
| 验证策略（用户定案） | **烟测循环为常态**：精度只跑冻结 30 题（`scripts/d2/compare_fake_real.py`，acc/same_ans 大致正确即可）；性能只跑快速端到端烟测（同上脚本口径）→ 查瓶颈 → 继续算子优化 → 循环。**官方完整双场景 bench 与官方 105 题精度只在「算子层基本无瓶颈」后最后各走一次**，不做常规验证 |

---

## 1. 本会话改动全清单（8 commits，全部已提交未 push）

### 1.1 插件侧（vllm-plugin-FL/，装机方式不变：`pip install --no-deps --no-build-isolation ./vllm-plugin-FL`）

| 文件 | 改动 | 要点 |
|---|---|---|
| `vllm_fl/quantization/w8a8/int_mm.py` | **新建（核心）** | `_int8_scaled_mm_int_mm_impl`（校验 + M<17 垫零 `_pad_rows` 持久 buffer + `_int_mm` + epilogue）；`_register_custom_op()` 包成 **`vllm_fl::w8a8_int_mm_scaled_gemm`** torch custom op（fake kernel 给 shape，dynamo 不 trace 内部分支——vLLM AOT 动态编译会把 `if M<17` 按大 M 烘焙，capture M=16 复用 artifact 直击 `_int_mm` M>16 约束崩 EngineCore）；融合 epilogue kernel `_w8a8_int32_epilogue_kernel`（裸 triton.jit，int32×xs×ws(+bias)→bf16 单遍，BLOCK 32×256/warps4，env `VLLM_FL_INTMM_FUSED_EPILOGUE=0` 可关回 torch 链） |
| `vllm_fl/dispatch/backends/flaggems/impl/quantization.py` | +裸量化 kernel | `_dynamic_per_token_quant_int8_bare_kernel`（裸 triton.jit，同 in-tree libentry 版数学）；**位精确关键**：`tl.fdiv(1.0,absmax,ieee_rounding=True)*127.0` 复刻 torch `127/t ≡ reciprocal(t)·127`、`absmax*(1.0/127.0)` 复刻 `t/127 ≡ t·fp32(1/127)`——torch 标量除法是乘倒数（差 1ulp，翻转 ~0.02% 的 round-half-even tie），用 div.rn 反而不位精确！ |
| `vllm_fl/dispatch/backends/flaggems/flaggems.py` | +4 方法 | intmm/scaledmm linear 两方法 + `dynamic_per_token_quant_int8_bare` 委托 |
| `vllm_fl/dispatch/backends/flaggems/register_ops.py` | +3 OpImpl | quant：`default.flagos_bare` **prio 165**（最高，位精确+快+无 libentry）；linear：`default.flagos_intmm` prio 160 / `default.flagos_scaledmm` prio 150；`_flaggems_vllm_available()` 探测（外部包缺失若只在调用时暴露 → CachedOp 异常路径 → **永久坠落 mgr.call 慢路径 ~0.6ms/调用**，必须解析期探测跳过） |
| `vllm_fl/dispatch/backends/reference/`（3 文件） | +reference 实现纯 torch fp32 matmul 累加器（CUDA 无整数 GEMM 的最后兜底，2^24 内精确） |
| `vllm_fl/dispatch/ops.py` | +schema 方法 `w8a8_dynamic_per_token_linear`（非 abstract，不破坏外部 backend） |
| `vllm_fl/dispatch/__init__.py` | +`CachedOp.pre_resolve()` | 注册期预解析实例缓存——vLLM AOT fullgraph 编译 forward，**未预解析的 CachedOp 首次 resolve 在 capture 内进 RLock 即 EngineCore 崩**（"Dynamo does not know how to enter a RLock"）；预解析后 capture 走全程无锁快路径 |
| `vllm_fl/dispatch/manager.py` | +`warm()`；`_record_first_use` 编译期跳锁（`torch.compiler.is_compiling()`） |
| `vllm_fl/quantization/w8a8/linear.py` | **接线** | `_w8a8_scaled_mm = CachedOp(FLAGGEMS_W8A8_LINEAR_OP)`；apply_weights 走 dispatch（quant 保持独立 CachedOp）；`register_fl_w8a8_linear_kernel` 末尾对两 op `pre_resolve()`；G2 修复（get_min_capability=71）与 packed 链不变 |

### 1.2 评测/工具侧（scripts/d2/）

| 文件 | 改动 |
|---|---|
| `bench_int8_gemm.py` | 四臂（A bf16 / B flag_gems / C 证伪 / **D int_mm 真实调度路径**）+ **graph 模式**（`graph_bench()`：两臂 CUDA graph 捕获重放计时，捕获输出与 eager 逐位对拍断言；gate 判 graph 比，eager 并列保留）+ `--out-dir`（默认 `results/20261112_d2_intmm_gemmbench/`，20261107 是历史判定书勿覆盖）+ dispatch 自检断言 |
| `compare_fake_real.py` | **新建**：serve greedy（temp 0/max_tokens 3072）逐题比对 fake_quant_eval 结果，冻结 hash 校验，same_ans/acc 双判读 + gate |

### 1.3 台账（results/）

- `20261112_d2_intmm_gate_pass.md` — 闸门过线判定书（矩阵+方法论+三层修复路径）
- `20261112_d2_intmm_gemmbench/bench.json` — 20 形状四臂原始数据（graph+eager 双口径）
- `20261112_d2_fakevsreal/compare.json` — 对拍逐题记录（含 head/tail 供病理检查）

### 1.4 提交序列（git log main..feat/d2-w8a8-quant 尾部 8 条）

`f3d6a21` _int_mm 模块+62 单测 → `c3c400b` dispatch op+接线(391 全绿) → `8c7983e` bench D 臂 → `5c45df6` 裸 quant+融合 epilogue+graph bench → `f33582f` 闸门过线判定书 → `0c0ed16` dynamo 兼容(pre_resolve)+对拍脚本 → custom op 修复 → 对拍结果。单测终态 **134 全绿**（`tests/unit_tests/quantization/`，从 vllm-plugin-FL 目录内跑=仓库源码；**绝对路径跑**否则 no tests ran）。

---

## 2. 已确证地面真相（勿重查）

### 2.1 性能数字（graph 口径 = serve decode 真实形态）

| (K,N) | M=64 | M=128 | M=1024 | M=8192 | eager int8 pair（对照，失真）|
|---|---|---|---|---|---|
| qkv 2048×2560 | 1.31 | 0.99 | 1.35 | 1.10 | ~0.8ms 地板 |
| o 2048×2048 | 0.99 | 0.78 | 1.28 | 3.68 | |
| gate_up 2048×12288 | **1.75** | 1.21 | 1.22 | 1.46 | |
| down 6144×2048 | **1.41** | 1.28 | **1.60** | 1.12 | |
| lm_head 参照 | 1.82 | 1.11 | 0.96 | 1.50 | lm_head 部署为 bf16 |

- GPU 侧纯 kernel 时间（profiler，gate_up M=64）：**GEMM 59.3µs + epilogue 10.9µs + quant 7.7µs = 77.9µs** vs bf16 F.linear ~126µs（bf16 是单 kernel 126µs → int8 三个 kernel 78µs，launch 间隙是主要损耗）。
- `_int_mm` 裸调：M=30/K=2048/N=12288 = 0.058ms；M=1024 = 0.316ms（163 TOPS = 标称 42%，与 bf16 效率持平 → cuBLAS 本身没失调，是口径天花板）。
- vs FlagGems Triton（旧路径）：eager pair 最快 **4.97×**（gate_up M=1024）。

### 2.2 硬约束与位精确链

- `torch._int_mm`：**M>16**（垫零位精确已证）、**K%8==0**（真实形状 2048/6144 ✓）；int32 累加与 CPU int32 matmul 位精确。
- **CUDA 无 int32 matmul、无 fp64 GEMM**（"addmm_cuda not implemented for Int" / "gemm of double is not supported on CoreX"）→ 位精确 oracle = CPU int32 matmul + 大行状行采样。
- 整链位精确路径：bare quant（复刻 torch 除法语义）→ `_int_mm`（int32 精确）→ 融合 epilogue（乘法次序同 emu：`(acc·xs)·ws`，两条独立乘防重结合；mul+add 会被 **FMA 收缩**成单次舍入 → bias 路径 1ulp 差，模型量化层无 bias 不受影响）。
- 激活量化经 dispatch：`default.flagos`（外部 flaggems_vllm，**本机未装**，探测期跳过）→ `default.flagos_bare`（165）→ libentry triton（150，弃用）。libentry 病理：interleaved 调用 ~0.9ms/call + b>1 流竞争前科——**自写 kernel 一律裸 triton.jit**。

### 2.3 测量方法论（eager 三伪影，判据级测量必须 graph 口径）

eager 成对循环在本机：①三算子链每 iter ~200µs host 开销（`aten::_int_mm` host 105µs + 3×`aten::empty` ~80µs + triton launch 10-25µs）；②轮间全量 drain 后 **GPU 降频**使后续轮劣化 2-3×（首轮 0.30ms→后续 0.77ms；bf16 单 kernel 臂不受影响 → 成对比较系统性失真）；③周期性 56ms 尖峰污染轮均值。修法 = `graph_bench()`（两臂捕获重放，输出与 eager 对拍保正确性）。**任何新算子的性能结论都过 graph 口径**。

### 2.4 vLLM AOT 编译三坑（新算子入 forward 必读）

1. **RLock**：dynamo fullgraph 不能 enter RLock。CachedOp 必须 `pre_resolve()`（算子注册期）；`_record_first_use` 有 is_compiling 兜底。
2. **分支烘焙**：动态编译把 `if M<17` 按 trace 期大 M 烘死。**形状依赖分支的实现必须包 torch.library.custom_op**（fake kernel 只给 shape）。
3. **验证模板**（单测即可，不必起 serve）：`torch.compile(fn, fullgraph=True, dynamic=True)` 后跨 M=64/16/1/128 逐一 `torch.equal(fn(x))`。

### 2.5 残留已知问题（低优先级）

- **mgr.call 慢路径 0.7ms/调用**：任何 impl 运行期异常 → CachedOp 永久坠落此路径（`_use_manager_call=True` 不复位、`_failed_impls` 不清除）。当前链解析期探测已避开；若未来加新 impl 失败面，serve 会静默劣化——例行 grep serve.log `fallback to` / `switched from`。
- bf16 lm_head M=64/128 graph 捕获选了不同 cuBLAS algo（graph 2.08ms vs eager 0.97ms）——lm_head 不部署 int8，无影响，记录在案。
- serve prefill 是 eager（FULL_DECODE_ONLY 只 graph decode）——int8 链 prefill 付 host 开销，但整模型 prefill 中占比小于微基准；以官方 bench 为准。

---

## 3. 环境与红线（全部继续有效）

- 红线：serve 参数一字不动（`--port 9031 --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072` + `VLLM_PLUGINS=fl`）；官方 bench 脚本原件不改（`vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py`，**换 --test-cases 参数允许**，`run_d2_eval.sh` 的调用样例可抄）；pip 只 `--no-deps --no-build-isolation ./vllm-plugin-FL`；精度配方勿动（SmoothQuant α=0.5，checkpoint `/workspace/MiniCPM5-2B-w8a8`）；GPU 独占串行；≥3 次中位；git 只走 SSH；**11/17 硬止**。
- 陷阱速查：`pkill -f` 自杀（按精确 pid）；长任务 `setsid nohup` + 早落盘；Bash 单次 ≤10min，长等待轮询（sleep 上限 ~4min 一次）；Bash 工具 **cwd 会重置，命令一律绝对路径**；GitHub https 挂死。
- **skills 调研结论（用户已验证，接手直接采用）**：适用的是 Deep-Spark/iluvatar-agent-skills 仓库 4 件——**本机 `~/.claude/skills/` 已全部装好，无需再装**（若需重装：git 走 SSH clone + `bash install.sh` 软链；**npx skills add 走 GitHub https 在本机挂死，禁用**）：
  - `iluvatar-ixkncli-guide`：kernel profile 规范归因（SOL%/带宽/占用），**每个算子变体必跑**，替代手搓屋顶线；
  - `iluvatar-ixobjdump-guide`：SASS/IR 查指令——验证 Triton 生成是否真下探 int8 快路径（对上「cutlass 快路径被 sm90 门挡死」类归因）；
  - `iluvatar-ixsys-guide`：系统级 trace/时间线——serve 端到端跑起来后定位 int8 kernel 真实占比；
  - `iluvatar-ixrt`：IxRT INT8 路径（吃到算力的官方引擎）——手写融合 GEMM 时抄 tile/向量化策略的第一参考（384 TOPS 需 ixTE 私有栈，torch/Triton 天花板 ~42% 效率已被 bf16 侧印证）。
  - `iluvatar-cuda-base`（CUDA 兼容差异）：暂不发布但仓内可读（skills 仓库 `/root/iluvatar-agent-skills`），CUDA→Corex 兼容排查时翻。
  - **Kernel-Smith 不采用**（用户已核实）：独立自动算子生成系统（非 skill 包形态）+ 评测基准在沐曦/MACA 侧；本会话纯天数侧。若日后开沐曦会话可作参考基线。
  - **待办一次检索**：官方《三 MCP 联动指南》（developer.iluvatar.com，2026-10-22）——若天数栈有官方 MCP（性能优化联用）可能比 skills 更强，开工时检索确认一次获取方式。

---

## 4. 接手头三件事

1. **深度调研 4 个已装 skill**（§3 列表；读 `~/.claude/skills/` 下各 skill 的 SKILL.md，
   跑一遍帮助命令摸熟输出格式）+ 检索一次官方《三 MCP 联动指南》（developer.iluvatar.com，
   2026-10-22）确认是否有比 skills 更强的官方 MCP 可用——**这是算子开发的方法论地基，
   最先做**。
2. 收快速端到端烟测对比（bf16 arm 若未完照 `/tmp/d2_quick_e2e.sh` arm2 段补跑）：
```bash
cd /flagos_比赛/flagos_xx && git log main..feat/d2-w8a8-quant --oneline   # 应见 17+ commits
tail /tmp/d2_quick_e2e_run.log
grep "Total token throughput" results/20261112_d2_e2e_quick/bench_{w8a8,bf16}.log
```
3. 用 `ixsys`/`ixkncli` 对当前 serve（起法照 `/tmp/d2_quick_e2e.sh`）做一次时间线/profile
   基线，定位 int8 kernel 真实占比——P0/P1 的优先级用数据复核后再开工。
   （改码必 `pip install --no-deps --no-build-isolation ./vllm-plugin-FL` 才进 bench/serve。）

---

## 5. 开发方法论：skill 驱动的算子开发循环（每个变体都走）

> skills 是知识/命令生成型，不写算子——算子生成本身是 agent 的活；skills 的价值是把
> profiling/归因/验证从手搓（屋顶线手算、ixsmi 肉眼盯）换成天数官方工具链规范用法。
> **开工第一步就是深度调研这 4 个 skill 的用法**（读 skill 内容 + 跑一遍帮助命令摸熟），
> 并检索一次官方《三 MCP 联动指南》（见 §3）。

每个算子变体的标准循环：
1. **写 kernel**（裸 triton.jit，位精确语义先在单测对拍 oracle 锁死——口径见 §2.2/§1.1）
2. **ixkncli profile**：`--section` 看 SOL%/带宽/占用，判 memory/compute bound（对照 §2.1 的
   数字基线）
3. **ixobjdump 验指令**：SASS 有没有真下探 int8 双发/DP4A 类路径，tile/向量化形态对不对
4. **graph 口径微基准**（`bench_int8_gemm.py`，D 臂或新臂）过判据
5. **ixsys 时间线**（端到端阶段）：serve 跑起来后定位 int8 kernel 真实占比，确认优化
   目标没找错
6. **烟测**（见 §6 验证策略）→ 瓶颈仍在 → 回 1

---

## 6. 优化空间分析（新 agent 主战场，按 ROI 排序）

> 现状：int8 链 77.9µs（M=64 gate_up）vs 理论下界。结构性浪费按大小排：

**P0 — decode skinny GEMM 专用 kernel（最大空间，M≤16 档 17× 计算浪费）**
decode M=1 时 `_int_mm` 垫零到 17 行——**17× 计算浪费**，且 decode 每步每层都是 GEMV 形态
（weight 读带宽主导：gate_up 25MB int8 → 21µs@1200GB/s 理论，实测链路远高于此）。
自写裸 triton.jit **int8 GEMV/skinny GEMM**（M≤16：每 program 处理若干 N 列，K 维 int8
点积用 `tl.dot(int8, int8, acc int32)` 或手工 DP4A 式展开），M>16 回落 `_int_mm`。
位精确语义不变（int32 累加 + 同款 epilogue 融合进 kernel 尾部，一次写出 bf16——
连 epilogue kernel 都省了）。判据：graph 口径 M=1/M=16 档 vs 现路径 ≥1.3×，且官方
bench decode 重场景（16K）可感。**风险低**（GEMV 是最容易写对写满带宽的 kernel 形态）。

**P1 — GEMM+epilogue 融合（消 int32 中间往返，prefill 档最大浪费）**
现路径：`_int_mm` 写 int32 [M,N]（M=1024 gate_up = 50MB）→ epilogue kernel 读回 50MB +
写 25MB bf16。**~100MB/层往返 ≈ 83µs@1200GB/s 纯浪费**。方向 A：自写融合 GEMM
（int8 MMA tile + epilogue 寄存器内完成直接写 bf16）——要打 cuBLAS 的 59.3µs，先
ixkncli profile `gemm_tcu_mr_kernel` + ixobjdump 看 SASS tile 形状，**抄 `iluvatar-ixrt`
skill 的 INT8 tile/向量化策略**（吃到算力的官方参考）。方向 B（更稳）：custom op 内
两步变一步的 CUDA Graph 段已在 decode 生效；prefill eager 侧先用 ixkncli 确认 epilogue
占比再决定。量化也可融进 GEMM prologue（读 bf16 x 一遍完成 absmax→量化→MMA，省 int8
中间写读）——与 P0 同一 kernel 体系，P0 验证手写 int8 点积效率后再上。

**P2 — 现有三个 kernel 的参数调优（半天工作量，确定性收益）**
- 融合 epilogue：BLOCK 32×256/warps4 固定——M=1024 大形状按 ixkncli 实测调 BLOCK/warps；
- 裸量化 kernel：BLOCK=next_pow2(K) 单行/program——M 大时可 2-4 行/program 或调 warps（现在 8192→8 warps，2048→4）；
- `_int_mm` 的 cuBLASLt algo：`CUBLASLT_WORKSPACE_SIZE` env / torch 的 workspace 配置是否有更优 heuristic 可试（低风险实验）。

**P3 — lm_head int8 化（540MB bf16 权重减半，decode 每步必读）**
lm_head 现为 bf16（targets 类名匹配不中）。量化它需重转 checkpoint + 30 题复验（精度
配方红线边缘，需用户同意）。收益：decode 每步 lm_head 读 540MB→270MB ≈ 省 225µs/步
（1200GB/s）——decode 重场景或显著。**先问用户。**

**P4 — 二线工具路线（打不动 cuBLAS 时的归因武器）**
`iluvatar-ixkncli-guide` profile `gemm_tcu_mr_kernel`（SOL%/带宽/占用）；`iluvatar-ixobjdump`
看 int8 是否下探 DP4A/IMMA 双发；对照 `iluvatar-ixrt` 的 INT8 tile 配置。注意 384 TOPS
标称要 ixTE 私有栈（不可用），torch/Triton 天花板 ~42% 效率已被 bf16 侧印证。

**止损线**：任何新 kernel 位精确不过（对拍 oracle 见 §2.2）或 graph 口径无收益即回退；
11/17 硬止前 2 天（11/15）应冻结算子改动转入收尾。

---

## 7. 验证策略与收尾清单（用户定案）

**常态 = 烟测循环**（算子优化期间反复走，不跑官方全量）：
1. 精度烟测：冻结 30 题对拍 `python scripts/d2/compare_fake_real.py`（serve 在 9031 时直跑；
   acc/same_ans 与 fake（0.9333/基线）大致一致即可，不追位级一致——已知残差来自 serve 栈
   执行形态，见 §0 对拍记录）
2. 性能烟测：`/tmp/d2_quick_e2e.sh` 口径（同参 serve 两臂 + 官方 bench 原件换小参
   `[[2048,512,16,32],[512,1024,16,32]]`），或只跑 w8a8 臂与历史自比 + 关键节点跑双臂
3. 烟测 → ixsys 时间线/ixkncli 查瓶颈 → 回算子优化循环，直到「基本无瓶颈」

**算子层基本无瓶颈后，最后一次性走官方全量**：
1. 官方双场景 bench ≥3 次中位：`bash scripts/d2/run_d2_eval.sh`（对照 4K 2668.42 /
   16K 1276.62 不回退，目标显著超）
2. 官方 105 题精度 ≥0.955（`STAGE=eval`；30 题口径 Δacc=0 + 对拍在噪声带，105 题
   max_tokens=32768 无截断游荡干扰，预期稳）
3. 耦合验证（[fl.sched] 同驻数）+ D0 重调（`scripts/sweep_attn_launch.py --full`）
4. 台账回写：`results/d2-kernel-matrix.md` 加 int_mm/自研 kernel 行、handover 状态更新、
   push（SSH）
