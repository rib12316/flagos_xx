# final-submition 官方全量复测（2026-11-20 19:45–20:50，用户批准）

- 分支：`final-submition` = 83ddda7（四项融合默认 metax 开 + delivery 证据链）
- serve：官方沐曦命令逐字（零附加 env、无 `--compilation-config`），`--model` →
  `/workspace/MiniCPM5-2B-w8a8-v2`；插件 = 83ddda7 pip 安装（PROVENANCE 在档）
- bench：官方脚本原件（与上游 diff 为空）；**发现官方脚本内部 RUNS=4/SKIP_FIRST=1，
  单次调用即"4 轮丢首取均值"**（summary CSV 行即官方值）——与 11-19 外层 4 轮×RUNS=1
  旧协议统计内容等价
- 精度：evalscope math_500 L3 官方配方逐字

## 结果

| 项 | 值 | 对照 |
|---|---:|---|
| 105 题精度 | **0.9714**（105 题，accuracy mean） | 红线 0.95 ✓；上版 98.1%（−1.0pp，融合贪心分叉的代价） |
| 4K total tok/s | **5986.27**（轮值 5960.52/6025.25/5973.03，丢 5432.91） | 基线 3786.30 = **+58.1%**；上版 6399.21 = −6.4% ⚠（见下） |
| 16K total tok/s | **7662.39**（轮值 7661.70/7663.28/7662.19，丢 7532.72） | 基线 5583.49 = **+37.3%**；上版 7509.87 = **+2.0%** ✓ |
| 4K Mean/Median TTFT | 2627.42 / 1119.98 ms | summary CSV 在档 |
| 16K Mean TTFT | 24506.69 ms | 同上 |

轮值散布：4K 计分轮 ±0.55%（带内紧）；16K 计分轮 ±0.02%（历史最紧）。

## 4K −6.4% 归因口径（诚实记录）

- 今日三轮（5960–6025）内部一致，但整体落在 11-19 历史轮值带（5883–6928，极差 16%，
  台账 D10 判决"大幅状态依赖"）下沿；16K 同会话 +2.0% 且极紧，不支持"融合伤害 4K"
  的简单解释（同栈同融合）。
- quick 样本（32p/64p）四项全开为 +2.86%/+1.29%，与 16K 方向一致。
- 不排除会话级状态差异（APC/时钟/温度）；如需裁定可另开会话补一轮 4K（~25 min）。
- 有效性无虞：两场景均远高于 baseline−1% 下限；排名以组委会统一复现为准。

## 融合生效证据（serve 日志，两阶段一致）

`w8a8_dynamic_per_token_linear → default.flagos_mctlass` ｜ A-H4
`fused_add_rms_norm=['vllm_c','native']` ｜ D13 `paired 83 sites (m_fuse_max=2048)` ｜
F-c `grouped decode out active` ｜ F-a `pass injected, ROPECAT=True`

## 文件

- 原始轮值：`raw_runs_20261120_200414.csv`（4K）/ `raw_runs_20261120_202545.csv`（16K）
- 官方汇总行：`summary_20261120_200414.csv` / `summary_20261120_202545.csv`
- 精度：`evalscope_reports/`、predictions `math_500_Level 3.jsonl`、`evalscope.log`
- serve 全量日志：`serve_acc.log` / `serve_bench_k4.log` / `serve_bench_k16.log`
- 过程：`master.log`；runner：`scripts/d4/run_official_final_submission.sh` +
  `scripts/d4/run_official_16k_single.sh`（协议修正记录见脚本头注）
