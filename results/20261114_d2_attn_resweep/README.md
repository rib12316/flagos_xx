# 2026-11-14 P5 D0 重调检查点：LAUNCH_TABLE 复核（P0-P3 落地后）

`sweep_attn_launch.py --full`（16 形态 × 13 臂，bit-exact 校验通过，
22:45-22:52）。目的：P0 decode 融合档改变 decode 线性占比与批节奏后，
复核 D0 表是否漂移。P4（int8 attention 自研）已因 P2 探针证伪收益前提
（attention 非带宽 bound）而关案并入本检查点。

## 结论：**表不动**（无实质漂移）

| 档 | 现表 (11-03) | 再生 (11-14) | 判 |
|---|---|---|---|
| decode le32/le8k | (2,1) | (1,1) | **唯一分歧**：11-03 实测 w2s1 反超 16%，今夜 w1s1 反超 5%——跨会话 ±5% 在方差带内且与历史 16% 矛盾，保守不动 |
| decode le16/le32k, le32/le32k, le64/*, prefill le16/* | (1,1)/(2,1) | 同值 | 一致 ✓ |
| decode le16/le8k, le128/* | 有具体/通配 | 告警（档内不一致）→ 落通配 | 通配值与现表一致，无损失 |

- decode 大档最优稳定 w1s1（bs≥32 kv≥4k 增益 1.76-1.85× vs default），
  prefill 稳定 w2s1（1.49-1.50×）——与 11-03 首扫一致，D0 调优机制无漂移。
- `--kv int8` 臂未跑：P2 判受阻塞候选（kernel 层净负），量化 kernel 的
  launch 重扫无意义。

## 附注

- 本轮 sweep 数字与 11-03 的单档翻转提示跨会话方差 ~5% 量级存在——
  未来换表判据应要求 ≥10% 且方向两次一致。
- 原始数据：`sweep_raw.csv` / `sweep_table.py` / `sweep_table.json`
  （VLLM_FL_ATTN_LAUNCH_TABLE JSON A/B 入口保留）。
