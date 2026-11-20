# skills/ — 推理优化 skill 库（双机共享）

> 2026-11-20 二期开工时从开源社区引入（clone 经 ghfast.top 镜像，各目录保留原 LICENSE）。
> 本目录是**唯一分发源**：B 机/新环境不再直连外网拉取，直接从本仓取。

## 清单与用途

| 目录 | 来源 | 用途 | 状态 |
|---|---|---|---|
| `ako4all/` | TongmingLAIC/AKO4ALL | kernel 优化循环纪律：workspace 盘点 → profile → 改 → bench → ITERATIONS.md 台账 → commit。原版用 ncu，本机无 NCU → **profiler 换 torch.profiler/厂商口径**（与 docs/00-统一假设台账.md 纪律同构，可叠加使用） | ✅ 可用（纯 prompt skill） |
| `kernelgen-flagos/` | flagos-ai/skills | FlagOS 官方 kernel 生成/优化 skill；自动识别 FlagGems / vLLM 代码库，含 generate/optimize/specialize for-flaggems、for-vllm 子技能 | ✅ 模板可用；**MCP 迭代优化需 kernelgen.flagos.io token（外网），当前未配置**——只用其离线 prompt 模板 |
| `vllm-plugin-fl-setup-flagos/` | flagos-ai/skills | **B 机 bootstrap 用**：vllm-plugin-fl 多后端安装/配置/排障/起 serve 验证 | ✅ 可用 |
| `perf-test-flagos/` | flagos-ai/skills | 官方口径性能测试（vllm bench serve，5 类负载画像，吞吐/TTFT/TPOT/ITL 采集）——与本仓 benchmark_throughput_serve.py 互补参考 | ✅ 可用 |
| `kernel-smith-ref/` | InternLM/Kernel-Smith | 沐曦×上海AI实验室 MXMACA 特化 kernel 生成系统；模型/agent 未开源，仓库目前只有 README。**在线端: https://chat.intern-ai.org.cn/kernel-smith**（B 机首日验证可达性）；战绩: SGLang normal_decode 4.78× / LMDeploy MoE routing 1.36× | 指针；候选 kernel 产出必须过本仓 T0 gate |

## 安装到本机 Claude Code

```bash
for s in ako4all kernelgen-flagos perf-test-flagos vllm-plugin-fl-setup-flagos; do
  mkdir -p ~/.claude/skills/$s && cp -r skills/$s/* ~/.claude/skills/$s/
done
```

## 使用纪律（红线衔接）

- skill 产出（kernel 候选/配置建议）一律视为**外部输入**：进 `feat/d*` 分支走 T0 微基准 → T1 单场景 → T2 官方（需用户批准）逐级验证，不直接落默认路径。
- AKO4ALL 的 ncu 依赖在本机不成立：凡 skill 内提到 ncu/nsys 的步骤，替换为 torch.profiler（`--profiler-config` + `/start_profile`，先例 `results/20261113_d2_prefill_attribution/01_attribution.md` §工具与方法备注）。
