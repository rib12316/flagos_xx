# D3 六项开发交付（2026-11-16）

数据目录 `results/d3_dev/`（DEVLOG.md 为逐项过程与判决）。官方 bench 与 105 题
精度**未跑**（按纪律等用户明示；六项已全部完成，条件满足）。

## 终态总表

| # | 项 | 状态 | 默认 | 代理口径收益 | 对拍 | 开关 |
|---|---|---|---|---|---|---|
| ① | TD 加载路径 | 已落地 | **关** | decode ITL −6.4%（探针 attn 1.20-1.55×，官方 4K kv 区应更高） | **0/16** | `VLLM_FL_ATTN_TD=1` |
| ② | 采样过滤 grid 16→64 | 已落地 | **开** | decode ITL **−9~12%**（3.7→1.1ms/步，逐位等价） | **16/16** | `VLLM_FL_SAMPLE_FILTER=0` 关 |
| ③ | 段数自适应 | 数据结案 | n/a | 官方形态无收益（kv8K 收益区到不了；bs32/kv16K 倒挂 0.92×） | — | 未落码 |
| ④ | KV int8 扩池 | 零代码烟测 | **关** | 池 ×1.94（588K→1.14M）、16K 0 抢占 kv 峰 0.58、decode 税 +4%、prefill 税未观察到 | **1/16** | `VLLM_FL_KV_INT8=1` |
| ⑤ | prefill TILE=16 | 已落地 | **关** | prefill **+12.8%**（6941 vs 6153 tok/s；探针 attn 1.10-1.11×） | **0/16** | `VLLM_FL_ATTN_TILE=1` |
| ⑥ | 自写 split-KV kernel | 数据结案 | n/a | 机理假设被零成本探针证伪（decode TILE=16 已最优、重分段无益） | — | 未动工 |

## 交付配置（默认态 = 仅 ② 开）

终态烟测（4 轮全）：**对拍 16/16（逐位干净）**；decode ITL 中位 **16.2ms**
（基线 17.6-18.2，**−9~12%**）；prefill 最优轮 **6303 tok/s**（基线 6153，+2.4%）。
- 代码：`attn_td_mode.py` / `sample_filter_mode.py` / `attn_tile_mode.py` +
  `iluvatar.py` 三处接线（pip 装入 site-packages，serve 默认即生效）。
- 4K 场景推算 +4~6%、16K 场景 +8~10%（decode 占比加权，待官方 bench 验证）。

## 用户门禁决策空间（官方 bench + 105 题后裁）

- **+⑤ TILE**：4K 场景再 +~10%（prefill 主导），对拍 0/16 风险与 ① 同类。
- **+④ KV int8**：16K 场景主杠杆（同驻 33→64，上界 +50~90%），decode 税 +4%，
  对拍 1/16——**精度是硬闸门**（公开证据 int8_per_token_head 73=73 基线）。
- **+① TD**：decode 再 −6%+（官方 kv 区或更高），对拍 0/16。
- ⚠️ **①×④ 禁止同开**（实测组合 decode ITL 18.7ms = 比默认臂差 16%，
  TD×descale 路径互毁）；all-on 臂 prefill 也净负（−5.4%）——推荐组合为
  「②+⑤+④」或「②+⑤+①」二选一，由官方两场景得分与 105 题结果定夺。

## 稳定性备注

- ④ 首 serve 曾一次 EngineCore 无栈静默死亡（prefill round2）；复现 serve
  重 prefill 13+min 未复现——归档为已知 EngineCore 间歇竞态嫌疑（11-07 同签名
  前科），非 int8 确定性缺陷；官方 bench 前建议新鲜 serve 起测（既定纪律）。

## 回退清单

全部行为变化走插件 env：`VLLM_FL_SAMPLE_FILTER=0` / `VLLM_FL_ATTN_TD=0` /
`VLLM_FL_ATTN_TILE=0` / `VLLM_FL_KV_INT8=0`，默认态即基线+②；② 本身逐位等价。
