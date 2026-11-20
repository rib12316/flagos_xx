# 算子工程三任务 · 交接 handoff（2026-11-20 15:05 · 新agent → 接手 agent）

> 任务源：`docs/交付-新agent-C500算子工程开发单.md`（必读 §0 DoD 五件套 / §1 三铁律）。
> 本文件是交接事实源：已做判决、未完成任务、可复用资产、踩坑清单。比赛截止 23:59。

## 0. 一句话状态

四项判决 **全部负/未完成**，但每项都有完整证据链与可复用机制资产。唯一仍有救的是 **F-a rope+cat 融合**（kernel T0 绿 1.61×、接线已打通、pattern 差最后一改），其代码停在 pattern v3（构造期回归，见 §3）。

## 1. 判决总表（详见 `docs/00-统一假设台账.md` §9 + 开发单 §8 登记表 + 审计表）

| 项 | 判决 | 关键数字 | 分支（均已 push） |
|---|---|---|---|
| ③ host 开销 | ❌ 负判决（诚实门） | 稳态每步同步 1.04<2；步间气泡 p50 2.8µs；19 站点全判死 | `feat/s2-host-audit` |
| ② A-H4 norm 替换 | ❌ T1 双形状门未全过 | kernel 级 1.10/1.14/1.24×（位一致）；serve +0.086%/+0.32% | `feat/ah4-irnorm-vllmc` |
| ① F-c copy 消除 | ❌ 结构性不可动（vendor .so） | 42/步×3.3µs=0.14ms/步；探测矩阵九变体定案（md5 0953e2b9） | `feat/fc-copy-attribution` |
| ② 审计表 | ✅ 交付（任务②本体） | 三判决行+机制修正+排除论证 | `feat/s2-opscan-audit` |
| ① F-a rope+cat | ⏳ **接线通、pattern 未命中** | T0 1.61×/1.54× 位精确；探针实证 pass 已进 manager 且被调 3 次、matched=0 | `feat/fa-rope-cat-fusion` |

## 2. 复用资产（机制发现，比判决本身更值钱）

1. **A-H4 真机制**：compiled 态 norm 走 vllm IR 系统（`RMSNorm.forward_cuda → ir.ops.fused_add_rms_norm`），排除根因是 `get_default_ir_op_priority` 默认 `["native"]` —— 修法=platform override per-op 翻转（`vllm_fl/platform.py`，env `VLLM_FL_IR_NORM`），**yaml 路由说法的修正**。该接线范式可复用于任何 norm 类替换。
2. **post-grad 自定义 pass 的四个坑**（F-a 全程踩出，vllm 0.24 + metax）：
   - `inductor_passes` 字符串通道**死路**（只在 `CompilationConfig.__post_init__` 解析，早于插件变异；EngineCore pickle 恢复不重跑）。
   - pass 实例**不可跨进程 pickle**（pattern matcher 含 PyCapsule）→ 必须 WorkerFL.__init__（EngineCore 子进程内、编译前）注入。
   - `VllmBackend.__init__` 对 `inductor_compile_config` 做 **deepcopy 构造期快照**，`configure_post_pass` 判快照 → 注入早于/晚于构造决定成败；F-a 用类级 patch + 自愈（`rope_cat_fusion.py:_install_backend_sync`）。
   - `vllm_fl.*` 域名的 logger **不在 vllm logger 树下 → INFO 静默丢弃**；调试必须走文件探针（模板：`rope_cat_fusion.py` 内 `/tmp/fa_pass.log` 写法）。
3. **F-c 归因方法论**：kineto 无 python 栈时，graph 回放核的 cpu_op 链是异步交错伪影；归因须 GPU 时间线邻接法（前驱/后继核定位）。
4. **vendor .so out 合同缺陷**：`flash_attn_2_cuda.fwd_kvcache` 的 out 参存在但当前构建运行期不可满足（九变体矩阵），vendor 升级后翻 `VLLM_FL_FLASH_KVCACHE_DIRECT_OUT=1` 可 0 成本回收 0.14ms/步。

## 3. 未完成：F-a 最后一改（接手首选动作）

**现状（分支 `feat/fa-rope-cat-fusion` HEAD=3a2fb50）**：
- kernel `vllm_fl/dispatch/backends/flaggems/impl/rope_cat.py`：T0 位精确、M32/64 = 1.61×/1.54×、prefill 1.00×（`scripts/fa/t0_rope_cat.py`，结果 `results/20261120_fa_ropecat/t0_r2.log`）。
- pass `vllm_fl/compilation/passes/rope_cat_fusion.py`：注入链已通（探针 `UUID`+`APPLICABLE`+`CALL` 实证），pattern matched=0。
- **真实 post-grad 图形态（探针实证，最关键的交接情报）**：
  ```
  [CALL nodes=95]  aten.cat.default([sub_44, add_151], -1) / ([sub_61, add_213], -1)
                   aten.reshape.default(getitem_12, [arg1_1, -1, 128])   ← reshape，不是 view
                   aten.index.Tensor(arg14_1, [arg13_1])
  ```
- **pattern v3（commit 480b6c9）构造期回归**：探针 `REGISTER` 只到 `total=2` 即断——第 2 个变体注册时异常（疑 `ApplyRotaryEmb.forward_static` 签名/参数不符或 fake-mode 追踪失败），被 WorkerFL try/except 静默吞掉（日志同样被吞）。
- **下一步**：①复现 v3 构造异常（`cd /data/flagos_feat_fa && python3 -c "from vllm_fl.compilation.passes.rope_cat_fusion import make_rope_cat_pass; ..."` 走真实 VllmConfig，或在 `_register_variants` 里加文件日志打 traceback）；②修好后一轮 smoke 看探针 `CALL_DONE matched>0`；③命中即跑 T1 双形状（`scripts/ah4/t1_ah4.sh` 的 A/B 骨架可改用 env `VLLM_FL_FA_ROPECAT=0` 做基线臂）。

## 4. 环境与纪律要点（接手必读）

- 工作树：`/data/flagos_wt`=delivery（docs/sync）、`/data/flagos_feat_{ah4,fc,fa,host,opscan}`=feat 分支。**当前 pip 装的是 fa v3**（site-packages 版本），跑别的项前先重装对应 worktree。
- **GPU 独占**：serve 启动前 `mx-smi | grep 'no process'` 必须成立（显存未释放时 vLLM 会报 `Free memory ... less than desired` 直接死）。**杀进程必须用 `ps|awk '!/awk/'` 排除自身**——`pgrep -f` 会匹配执行它的 bash 本身，`kill -9` 自杀过一次。
- serve 形态：`VLLM_PLUGINS=fl setsid nohup python3 -m vllm.entrypoints.openai.api_server --model /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`（api_server 形态可躲过 `pkill -f 'vllm serve'`；bench 客户端用**绝对路径**）。
- bench：`python3 <repo>/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py --model ... --port 9031 --test-cases '[[4096,1024,64,32]]'`（4 pass 丢首取后 3 中位）。
- 诊断 trace：`--profiler-config '{"profiler":"torch","torch_profiler_dir":"/tmp/xxx","torch_profiler_with_stack":false,"active_iterations":1000000}'` + 手动 `POST /start_profile`（api_server 形态不自启）+ 受控负载 + `POST /stop_profile`。
- 已知基线锚：`[[4096,1024,64,32]]` 后3中位 ≈5932（本机今天实测，与 B 机 6020 带内）；`[[4096,1024,64,64]]` ≈6991。
- 同机协调：A 机原会话可能也在活动（会 pkill serve）；上板声明窗口后再压测。B 机线（D13/TileLang）不碰。
- 变更流程：T0 → T1（双形状 ≥3 轮中位 ≥+0.3%）→ 才可合入 delivery；负判决同样按台账格式入账。

## 5. 已交付文档位置

- 台账 §9：host/A-H4/F-c 三行（F-a 行在 §3 更新后补）
- 开发单 §8 算子库登记表：四行（含 env 门/复活路径）
- 审计表：`/data/flagos_feat_opscan/results/20261120_s2_opscan/audit_table.md`
- 各分支 `results/2026*_<id>/` 含 raw/log/判决
