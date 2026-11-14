# 2026-11-14 EngineCore 负载退化-死亡全周期刻画（conc128 压力 ×5 + 逐轮 parity）

背景：P0/P1 烟测两臂均在第 3 轮现 EngineCore 静默硬死（无 Python 栈），
D1 时代（2026-11-05，无任何 D2 kernel）有同签名前科。本压力脚本
（`/tmp/d2_conc128_stress.sh`）对单实例 v1 serve 连续 5 轮 conc128 bench
（[256,1024,128,128]），每轮后跑冻结 30 题对拍。

## 全周期曲线（同一 serve 进程）

| 时点 | 30 题 acc | trunc | 备注 |
|---|---|---|---|
| 新鲜 serve | 0.9333 | 15 | 历史带内（`../20261114_d2_fresh_v1_parity/`） |
| 压 1 轮后 | 0.8667 | 14 | |
| 压 2 轮后 | 0.7333 | 18 | |
| 压 3 轮后 | 0.4667 | 22 | |
| 压 4 轮后 | 0.5000 | 24 | |
| 压 5 轮后 | — | — | **EngineCore 静默死亡复现**（uptime ~60min） |

期间 tok/s 正常（1901-2989）、引擎零报错、显存恒定 28656MiB 无爬漏。

## b=1 隔离诊断（决定性）

在已退化至 acc 0.4667 的 serve 上（第 4 轮进行中）单发 b=1 逐题生成：
**与新鲜 serve 输出逐字一致**（pred 相同、正常收尾、长度同量级）。
→ 与 2026-11-07 libentry 流竞争事件同签名：**b=1 干净 / b>1 退化 =
批路径竞态（读未就绪输入），非 KV/状态腐坏**（后者会连 b=1 一起坏）。

serve 运行全程持续出现 flag_gems libentry `resolve_benchmarker`（event-timing
旁路流）RuntimeWarning——嫌疑机制与 11-07 案同族（当时元凶是 libentry 包装
的 quant kernel，已改 bare；现役候选 = 仍在 libentry 包装下的其余 flaggems
算子）。精确到 op 的定位留待后续（时间盒）。

## 对交付的影响判定

1. **非 P0/P1 引入**：arm B（v1，decode 走链、P0 关闭等价路径）同样崩；
   D1 时代前科；退化随负载累积与 kernel 选择无关。
2. **官方 bench 有效性**：官方每场景 ~10-15min、单次 run 总 uptime <30min，
   在死亡时钟（~60min）与深度退化前完成——与 run1/run2 吞吐 0.1% 级复现
   自洽。但第 2 场景后半的输出质量可能已轻度受染（官方计分看吞吐，用户已
   放弃精度约束）。
3. **烟测纪律修正**：parity（sanity）必须在**新鲜 serve**上跑（本日 P1
   arm A 0.9667 / arm B 0.60 的差异即此）——已入操作规程。

## 复现/定位入口

- 压力脚本：`/tmp/d2_conc128_stress.sh`（逐轮 parity 版）
- b=1 探针：`/tmp/d2_b1_probe.py <port>`（对照 `../20261114_d2_fresh_v1_parity/parity.json`）
- 下一步定位思路：逐 op 关闭 dispatch（VLLM_FL_PER_OP）二分找出参与竞态
  的 libentry op；或 libentry warning 的调用栈捕获。
