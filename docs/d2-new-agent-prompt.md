# 新 agent 指挥 prompt（D2 算子加速线）

角色：FlagOS S2 赛道二（infragogogo）开发 agent，接手 D2 量化线的**算子优化阶段**。
环境：天数 BI-V150 实例（torch 2.10 corex / vllm 0.24.0+empty / triton 3.2 flagtree / flag_gems
5.3.5 / 单卡 32GB / compute capability (7,1)）。仓库 /flagos_比赛/flagos_xx，工作分支
feat/d2-w8a8-quant（前一阶段 10 个提交已 push）。

## 核心目标（唯一）

**通过算子优化打通 W8A8 真量化推理加速**：让 int8 路径对 bf16 达到「任一 M=1024 形状 ≥1.5× 或
M=64 ≥1.2×」，并最终在官方 serve bench 双场景不回退（对照 4K 2668.42 / 16K 1276.62 tok/s）。
精度已由前一阶段定案（SmoothQuant α=0.5，30 题 Δacc=0），checkpoint 已就绪
（/workspace/MiniCPM5-2B-w8a8）——除非你改了量化算术语义，否则不要动精度配方。

## 第一步（必做）

通读 `docs/d2-operator-handover.md`（交接文档：全部改动清单、已确证地面真相、性能判定数据、
环境陷阱、三线开发路线）。所有带 file:line 的结论都已核验过，直接采用，不要重查。

## 开发路线（按优先级，交接文档 §5 有展开）

1. **头号线索**：`torch._int_mm`（cuBLAS int8 GEMM，int32 累加）——已实测**位精确**且比现行
   FlagGems Triton scaled_mm 快 4-8×，对 bf16 有望过线。第一步在 scripts/d2/bench_int8_gemm.py
   加 D 臂（_int_mm + epilogue）跑 15 形状验证；过线后在 vllm_fl dispatch 层接线（框架内开发量）。
   注意 M>16 约束（垫零方案已验证）、libentry 流竞争前科（自写 kernel 用裸 triton.jit）。
2. 二线：用 iluvatar-ixkncli-guide profile 现 Triton kernel 判 bound，ixobjdump 看指令是否下探
   int8 快路径，对照 iluvatar-ixrt 的 INT8 实现抄 tile/向量化。
3. kernel 打通后按序：serve 冒烟（scripts/d2/run_d2_eval.sh）→ fake-vs-real 30 题对拍 →
   官方双场景 bench（≥3 次中位）→ 官方 105 题精度 → 耦合验证 + D0 重调②。

## 红线（继承，违者判负）

serve 参数锁死一字不动（--model 指向转换 checkpoint 属部署步骤，允许）；bench 脚本原件不改；
pip 只从本仓库装（--no-deps --no-build-isolation，外网解析会挂死）；校准语料禁 math_500/gsm8k；
GPU 独占串行；采纳数字 ≥3 次中位；git 只走 SSH。时间盒 11/17 硬止（到期未达标回退量化改动保 P4′）。

## 环境陷阱速查

交互命令 pkill -f 自杀；长任务 setsid nohup detach 且早落盘中间产物；Bash 单次超时上限 10 分钟；
libentry Triton kernel 进程内直调 b>1 有流竞争；GitHub https 挂死只用 SSH。
skills 已装在 ~/.claude/skills/（ixkncli/ixobjdump/ixsys/ixrt，仓库 /root/iluvatar-agent-skills）。
