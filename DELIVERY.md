# final-submition 双平台交付汇总

2026-11-20，C500 在 `delivery-final-1119`（`ffd0daf`）之上集成四项融合。当前 C500 版本待测官方 4K、16K 和 105 题精度。下表为上一封板版本的官方结果，吞吐单位 tok/s。

| 平台 | 已验收的默认优化 | 4K | 16K | 105 题精度 |
|---|---|---:|---:|---:|
| 沐曦 C500 | W8A8 线性层和 lm_head、mctlassEx INT8 矩阵乘及零拷贝权重布局 | 6399.21 | 7509.87 | 98.1% |
| 天数 BI-V150 | attention 启动参数、TD 和 TILE、调度 watermark、W8A8 及采样过滤 | 3414.58 | 1570.62 | 97.14% |

官方测量记录见 [C500 汇总](results/20261119_d4_metax_port/final/SUMMARY.md) 和 [V150 逐轮结果与精度](results/d3_official_1116_2222/)。

## C500 运行清单

| 调用路径 | `final-submition` 的默认行为 |
|---|---|
| D2 模型与激活量化 | SmoothQuant + RTN 再生 W8A8 权重，线性层和 lm_head 使用逐通道权重 scale 与逐 token 激活 scale。 |
| D4 矩阵乘 | `mctlassEx` 承载 INT8 GEMM，直接使用 `[N,K]` 权重布局。 |
| F-a 位置编码 | 匹配 NeoX Q/K RoPE，将两路旋转与拼接合成一次 kernel 调用。 |
| D13 残差归一化与量化 | 83 处装载期配对，直接为下游 W8A8 GEMM 生成 INT8 和 scale，默认 M 上限为 2048。 |
| A-H4 其余归一化 | `fused_add_rms_norm` 优先调用沐曦 `vllm_c`，保留 `native` 后备。 |
| F-c Decode 输出 | 在实测的 BF16 GQA 布局上通过分组视图让 MACA Flash 直接写入目标输出。 |
| Attention 与调度 | MACA Flash prefill/decode，D1 准入 watermark 0.06，异步调度、decode 全图及 mixed 分段图。 |

四项融合默认在 `metax` 启用，V150 走原路由。在 C500 上显式设置 `VLLM_FL_FUSION_BUNDLE=0` 并清除单项覆盖可回退四项融合。已显式开启四项的 4K 小样本（32/64 条请求）总吞吐分别提高 2.86%/1.29%，4 条贪心样例中有 3 条输出分叉。[逐轮数据与输出](results/20261120_fusion_bundle/)

官方复测使用文末 C500 启动命令，4K 为 `[4096,1024,64,256]`，16K 为 `[16384,1024,64,128]`，每形状独立起服、四轮丢首取均值；精度为 math_500 L3 全部 105 题，门槛 0.95。D5 采样、D6 split-KV、D7 手写 prefill、D8 INT8 KV、D9 quant 变体均为归档实验。D10 方差、D11 机会项、D12/host、F-b、SiLU+quant 与 F-e 为诊断和筛选记录。TileLang 已结项。

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

## 11 月 19 日封板记录

1. ✅ `git merge feat/d4-metax-port`（`4eb72df` 入本分支；bedf8ca mctlassEx w8a8 +
   c948594 接线，保留默认开语义）
2. ✅ 官方沐曦 serve 命令**无 `--compilation-config`**——终局数字在该形态成立
   （4K 6399.21 / 16K 7509.87）
3. ✅ 全部优化默认激活审计（零附加 env 起 serve，审计行
   `w8a8_dynamic_per_token_linear → default.flagos_mctlass` ✓）
4. ✅ 官方 bench 4K/16K 各 4 轮（丢首均值）+ 105 题 98.1% ≥0.95
   （Mean TTFT 未单独汇总，raw_runs CSV 在档可查）
5. ⬜ 双平台源码合并 → 参赛队伍.zip（模板：FlagGems(v5.3.5 原样)/readme/report/
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
- w8a8 再生命令已实跑（295 tensors packed，RESULT 行可查）；沐曦侧已再生并
  roundtrip 位校验（295 tensors bit_exact，`results/20261119_d4_metax_port/PROVENANCE`
  在案）——原「逐位 diff 对照」待办就此关闭
- ttft 双场景大优（上表）

## 沐曦上版官方成绩（2026-11-19 23:20）

- **形态**：`VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`（零附加 env、**无 --compilation-config**，每场景新鲜 serve、4 轮丢首）
- **4K = 6399.21 tok/s**（vs 沐曦纯净基线 3786.30 = **+69.0%**；轮值 6927.95/5883.03/6386.65）
- **16K = 7509.87 tok/s**（vs 5583.49 = **+34.5%**；三热轮散布 0.12%）
- **105 题 = 98.1%**（105/105，红线 ≥0.95 大幅过；超天数侧 0.9714）
- 沐曦纯净基线（bf16+基线插件 13eb9be，同口径 4 轮丢首）：4K 3786.30 / 16K 5583.49
- 激活链：w8a8-v2 ckpt（tools/quantize_int8_rtn.py 幂等再生，沐曦已再生并 roundtrip 位校验）+ **mctlassEx w8a8_scaled_mm_azp 承载**（镜像预装算子，vendor='metax' 路由，[N,K] 零拷贝布局）
- 产物：results/20261119_d4_metax_port/{final,baseline,PROVENANCE,splitkv_iluvatar_prompt.md}
- 注意：本机 bench 客户端有"数据落盘后汇总阶段挂死"现象——raw_runs CSV 数据完整有效，
  汇总从 CSV 提取（已按此口径出全部数字）
