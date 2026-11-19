# delivery/s2-final —— 交付分支（唯一事实源）

> 天数 BI-V150 侧开发**封板**（2026-11-19）。后续：沐曦 C500 clone 本分支、
> 在本分支上完成 metax 适配与官方 bench/精度，最后双平台合并出提交 zip。

## 天数侧终局状态（本分支基线 = 9ddaa0f + ①⑤ 默认翻转）

| 项 | 值 | 口径 |
|---|---|---|
| 4K 场景 | **3414.58 tok/s**（vs 纯净起点 1879.68 = +81.7%） | 官方 bench 丢首均值 |
| 16K 场景 | **1570.62 tok/s**（vs 862.06 = +82.2%） | 同上 |
| 105 题精度 | **0.9714**（红线 ≥0.95，基线 0.962） | evalscope math_500 L3 官方配方 |
| Mean TTFT | 4K 12321→**7521ms**（−39%）；16K 636→**342s**（−46%） | 官方 CSV（罚则为上限，双场景大优） |

## 激活方式（评测审计后定稿）

- **官方 serve 命令零附加 env**：① TD 与 ⑤ TILE 的默认值已翻转为**开**
  （`VLLM_FL_ATTN_TD=0` / `VLLM_FL_ATTN_TILE=0` 为显式回退逃生门）。
  其余优化（D0 launch 表 / D1 调度默认 / D2 w8a8 管线 / ② 采样过滤）本就默认开。
- **w8a8 量化线 = 纯文档指引**（ckpt 路径激活）：serve 命令除 `--model`
  指向量化 ckpt 外与官方一字不动。ckpt 由转换脚本从原始模型**幂等再生**：

```
python3 vllm-plugin-FL/tools/quantize_int8_rtn.py \
    --src /workspace/MiniCPM5-2B \
    --dst /workspace/MiniCPM5-2B-w8a8-v2 \
    --smooth-alpha 0.5 \
    --act-stats results/20261107_d2_actstats/act_stats.json \
    --lm-head-int8 --seed 1234
```
（act_stats 来源 PROVENANCE：48 prompts / seed 7 / git 9012af6，见同目录）

- 天数 serve 形态（官方 4.3 逐字，仅 --model）：
```
export VLLM_PLUGINS=fl
vllm serve /workspace/MiniCPM5-2B-w8a8-v2 \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

## 沐曦侧待办（clone 本分支后）

1. `git merge feat/d4-metax-port`（bedf8ca mctlassEx w8a8 + c948594 接线；与本分支仅
   attn_td/tile_mode 两文件可能小冲突，保留本分支默认开语义）
2. 官方沐曦 serve 命令**无 `--compilation-config`**——数字须在该形态成立
3. 全部优化默认激活审计（评测只设 `VLLM_PLUGINS=fl`；默认关 = 复现不存在）
4. 官方 bench 4K/16K 各 4 轮（丢首均值，记 Mean TTFT）+ 105 题 ≥0.95
5. 双平台源码合并 → 参赛队伍.zip（模板：FlagGems(v5.3.5 原样)/readme/report/
   vllm-plugin-FL）→ 提交平台 + 评审后 PR

## 纪律红线（全程有效）

serve 参数除文档注明的 `--model` 外一字不动；benchmark 脚本原件；官方 bench 与
精度仅用户明示后开测；每场景新鲜 serve；产物落 results/ + DEVLOG；pip 只从本仓库
`--no-deps --no-build-isolation` 安装 + provenance 四查。

## 审计与验证档案（2026-11-19 晚）

- ①⑤ 翻转烟测：零附加 env 起 serve，日志确认 `use_td=True 已强制` +
  `TILE 32→16` + `w8a8→flagos_intmm` 路由，主进程/EngineCore 双行
  （results/20261119_submit_audit/serve_defaults.log）
- parity 0/16 = ①⑤ 非位精确的**预期分歧**（基线采自无 patch 态，DEVLOG 步骤 0
  在案；真精度门 = 105 题 0.9714）
- w8a8 再生命令已实跑（295 tensors packed，RESULT 行可查）；逐位 diff 对照
  w8a8-v2 的最终校验待办（沐曦侧合并后一并做）
- ttft 双场景大优（上表）
