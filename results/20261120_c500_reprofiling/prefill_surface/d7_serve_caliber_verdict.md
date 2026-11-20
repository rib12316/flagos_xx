# D7 锚点 serve 口径复核 — 45.6ms 判为 L2-cold 伪影，D7 上限坍缩（B 机）

- 日期：2026-11-20 04:35 · B 机 · 数据源：A 机 `results/20261120_c500_reprofiling/trace_all/rank0.*.trace.json.gz`（48MB，本仓）
- 方法：解析 chrome trace 的 flash_fwd_splitkv_kernel 实例 grid/时长直方图
  （`/tmp/parse_prefill_shapes.py`，3158255 events）

## 核心证据：serve 内 prefill kernel 实测

| kernel（traits） | grid=(q块, batch, splits) | 次数 | 均值 |
|---|---|---|---|
| `<128,128,64,4>` | **(16, 16, 1) = S16 q2048（A 机指定官方形状）** | 1428 | **437.6µs** |
| `<128,64,64,4>` | (16, 18~32, 2) 混批 | ~1000 | 255-449µs |

**vs 我微基准同形状 S16 q2048：kv2048=3.29ms、kv16K=45.6ms —— serve 实测 437.6µs，
差 7.5~104×。P1-B2 的「96 TFLOPS 平台/45.6ms/层墙」是 L2-cold 微基准伪影。**
（serve 内 KV 为上一 chunk 刚写入+逐层访问的 L2-warm 数据；微基准 268MB 全 cold。）

## 判决

1. **D7 的 2.2× 上限坍缩**：serve 真实 prefill attention = 437.6µs×28 层 ≈
   **12.3ms/chunk-batch**（且此形态 flash 效率并不低）。自研 kernel 即使打平
   也无空间（triton 路线本就 ≤57T 判死）——**D7 建议判死**（A 机裁定）。
2. **P1-B2 判决全面重定性**（B 机自我修正，镜像 A 机 D5/D9 判死链）：
   - 「chunked 续算 96T 平台」→ L2-cold 口径伪影，serve 内不存在此墙
   - 「16K prefill 主墙 1.28s/chunk-batch」→ 撤回（serve 实际 ~12ms）
   - 数值曲面（形状 scaling/散射无差等）仍有效，仅绝对时间作废
3. **方法论第三课**（与 A 机"小 kernel 必须 graph 口径"并列）：**大工作集 kernel
   （268MB ≫ L2）的微基准绝对时间不可外推 serve**——L2 冷热主导，graph replay
   也救不了。serve 侧锚点必须用 trace 墙图或 /start_profile 实测。
4. **今日 Phase 2 判死链完整**：D5（采样 9.66ms→serve 1-2ms）、D9（quant 37%→
   3.5% 步时）、D7（prefill 45.6ms→437µs）、D6/D8/D10/D12（前判）——全部微基准
   伪影或证据不足。Phase 2 无剩余 kernel 级活口；建议 A 机以 T2 官方验证收尾。

PROVENANCE：B 机解析 A 机 trace（未动 GPU）· grid 语义 = FA2 varlen
(cdiv(max_q,BM), batch, splits) · 437.6µs 为 1428 实例全 chunk 形态均值。
