# 2026-11-14 P2 KV int8_per_token_head 探针判决：kernel 层净负，转「受阻塞候选」

`scripts/d2/phase2_kv_probe.py`：vLLM 0.24 原生 int8_per_token_head 路径
（写 kernel + unified_attention KV_QUANT_MODE=2 in-kernel 反量化）在本栈
（flagtree 3.6 / TritonAttentionBackend）首次实测。

## 三闸结果

| 闸 | 结果 |
|---|---|
| ① 编译 | **PASS**——写 kernel 与 attention int8 分支均编译运行 |
| ② 数值 | **PASS**——attention 输出 max_rel 0.94-1.16%（≤3%）；写侧 dequant 0.39% |
| ③ 时间（graph 3×30 中位） | **FAIL（净负 10-27%）** |

| 形态 | bf16 | int8 | 比 |
|---|---|---|---|
| decode S=64 q=1 kv=2048 | 1124.4µs（KV 读 ~119GB/s） | 1256.2µs | 0.895× |
| decode S=64 q=1 kv=16384 | 8813.1µs（~122GB/s） | 9770.5µs | 0.902× |
| prefill S=2 q=2048 kv=2048 | 10335.9µs | 14232.5µs | 0.726× |

## 结构性结论（改写阻塞点分析的前提）

1. **bf16 attention 的 KV 有效带宽仅 ~120GB/s（读流天花板 829GB/s）**——
   本栈 decode attention（GQA 2 kv-heads × 16 SM）是延迟/占用 bound，不是
   KV 带宽 bound。**「KV 字节减半」在此硬件形态兑不成时间**；dequant ALU +
   per-token-head scale 装载 + head 132 非幂步长的税反而显性化。
2. prefill -27% 与旧路线图 F9「prefill -25%」精确互证——F9 的负结果在本
   原生路径上同样成立（外挂/原生都逃不掉 dequant 税 + 非带宽瓶颈）。
3. **推论：int8 attention 自研（原 P4）的收益前提同样不成立**——不是
   kernel 写不写得好的问题，是瓶颈不在带宽。P4 折叠进 P5 的决定获数据支持。

## 容量收益路径（未关闭，但不可本地验证）

池 tokens ×1.94（582K→~1.13M）对官方 16K 场景（16384×64=1.05M 需求 vs
582K 池，容量排队型）可能净正：同驻 29→58 缓解排队 vs attention 每 step
-10%。但：烟测场景（98K/164K 需求）容量不 binding → 烟测必净负；官方
bench 禁自跑 → **本线内无法过闸**。

## 处置

- `VLLM_FL_KV_INT8` 保留（默认 0，opt-in），P2 不进默认路径。
- 判「受阻塞候选」：若用户愿开官方 16K 单场景验证（一次性，~15min），可
  实测容量收益是否盖过 kernel 层 -10%；未开测前不采纳。
- 探针三 bug 修复存档：get_kv_cache_layout 需 config 上下文（用
  _KV_CACHE_LAYOUT_OVERRIDE 钉 NHD）；写入校验须按 slot_mapping 收回
  cache 块；softcap=None 在 `USE_SOFTCAP=(softcap>0)` 崩（传 0.0）。
