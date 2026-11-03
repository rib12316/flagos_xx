# flagos_xx · FlagOS S2 赛道二唯一开发仓库

> 队伍 infragogogo（xxinfra）｜MiniCPM5-2B 推理吞吐优化（total tokens/s，4K/16K 双场景）｜沐曦 C500 64GB + 天数 BI-V150 32GB｜截止 2026-11-20 23:59

## 仓库定位（重要）

- 本仓库是**唯一的开发仓库**，从上游官方基线从零搭建，与旧仓库 `flagos2_infragogogo` **完全隔离**：旧仓库的代码、分支、改动一律不混入；其历史数据（D3a/D3c launch 调优证据、KV 量化负结果、profile 数据）仅作为**证据引用**（口径需注明"调优版/废弃仓库"）。
- 天数实例上的 `/flagos_比赛/flagos2_infragogogo` 视为**隔离区**：**禁止**从该路径 pip install（F5 事故来源）、禁止作为任何开发基线。

## 基线锚点

| 组件 | 版本 | commit |
|---|---|---|
| vllm-plugin-FL | branch `flagos-2026-s2` | `13eb9be69ecc5b5ca4f79c44e9ee40081eaa1bf0`（== 官方分支头 == 实例 `/workspace/vllm-plugin-FL` 干净树） |
| FlagGems | tag `v5.3.5` | `a7620cc191a0b42e040194622c5758b22a7a25dc` |
| 天数基线（10-30 纯净版实测） | 4K=1879.68 tok/s｜16K=862.06｜math_500 L3=0.962 | 待补沐曦基线数字入 `docs/00-统一假设台账.md` |

纯基线 commit 打 tag **`baseline-2026-11-03`**；任何时刻的总优化量 = `git diff baseline-2026-11-03 -- vllm-plugin-FL FlagGems`。

## 分支约定

- `main`：集成分支，只接受验证通过（T0→T1→精度 gate）的改动，保持随时可提交状态。
- `feat/<假设编号>`：每个假设一条分支，从 `baseline-2026-11-03` 切出；验证不过线 → revert 并在 `开发记录.md` 记录失败原因后删除分支。
- 一次只改一个变量；每次改动前在 `开发记录.md` 登记假设（目标文件/函数 + 预测机制 + 预期指标）。

## 安装与验证（实例侧）

```bash
git clone https://github.com/rib12316/flagos_xx.git
cd flagos_xx/vllm-plugin-FL && pip install --no-build-isolation .   # 或 --no-deps --force-reinstall
```

**pip provenance 纪律**（F5 教训，每次测量前执行）：`pip show vllm-fl | grep -i location` + `direct_url.json` 指向本仓库路径；diff `site-packages` 与本仓库源码一致；确认旧仓库 patch 符号不存在；`flag_gems` 版本 == v5.3.5。

## 目录结构

```
flagos_xx/
├── vllm-plugin-FL/   # 比赛框架源码（全部优化提交落这里）
├── FlagGems/         # 算子库（算子改动落这里）
├── scripts/          # 一键脚本（安装/基线复跑/结果对比）
├── docs/             # 研究文档、实测方案、基线档案（00-统一假设台账.md 是唯一事实源）
├── results/          # 各轮 benchmark CSV 存档（按日期命名）
├── 开发记录.md        # 假设→验证→分支号 三列台账
└── README.md
```

## 验证分层协议（详见 docs/00-统一假设台账.md 与路线图）

| 层 | 内容 | 耗时 |
|---|---|---|
| T0 | 算子级 micro-bench（数值+耗时） | 秒级 |
| T1 | 官方脚本单场景（4K 或 16K） | 10-20min |
| T2 | 双场景 + evalscope L3 精度 gate（≥0.95） | ≈1h |
| T3 | 冻结复现（提交前） | ≈2h |
