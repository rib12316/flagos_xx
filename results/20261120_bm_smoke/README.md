# BM 重标定快速烟测(2026-11-20,非官方口径)

集成:`vllm_fl/dispatch/backends/vendor/iluvatar/attn_bm_mode.py`(锚点重 exec,
BM256/W16/S2/TILE=128,`VLLM_FL_ATTN_BM=1` 启用,默认关;**必须装在 alc 之前**,
iluvatar.py 已调整顺序)。内核级探针见 `../20261120_blockm_sweep/`(9.4-9.9×
vs upstream,5.7-6.0× vs 旧生产,36 配网格无更优)。

## 快速烟测结果(quick_smoke_client.py,16.8K token 文本)

| 测项 | B 臂(TILE1+TD1) | BM臂(TD全关,弃) | **合成臂(BM+TD1)** |
|---|---|---|---|
| 单请求冷提示 TTFT | 15.53s | 2.34s(热kernel) | 首发8.16s(含~6s编译税),热态冷提示≈2.3s |
| decode-only ITL@bs16 | 124ms | **269ms(TD丢失)** | **123ms ✓** |
| 16并发×16K×128 墙钟 | 31.8s | 36.4s | **24.1s** |
| 同形总 tok/s(含缓存共享) | 8525 | 7434 | **11243 (+32%)** |

## 判决

1. **BM prefill 重标定在 serve 真实路径兑现 ~6×**(TTFT 15.5→2.3s,与探针一致)。
2. **①TD 的价值在 decode 侧**(TD 124ms vs 非TD 269ms @bs16/kv17K)——TD/TILE
   互斥仅存在于 prefill;attn_bm_mode 现在在 prefill 分支内强制 use_td=False,
   decode 保留 TD。**终选臂 env:`VLLM_FL_ATTN_BM=1 VLLM_FL_ATTN_TILE=0
   VLLM_FL_ATTN_TD=1`**(= 合成臂 = ② + BM + decode TD + D0表)。
3. 测量陷阱(本目录数据已被这两条污染的地方已标注):同文本重复发 → prefix
   cache 全命中(TTFT 0.28s 假象);进程首个真 prefill 付 BM kernel 一次 ~6s
   Triton 编译税(官方 4 轮丢首轮天然吸收)。
4. 未验:官方 4K/16K 4 轮口径、105 题精度门(等用户令)。

## 目录

`C16/`(官方 bench 中止,仅 serve 日志可用)、`bm_serve2/`、`baseline_serve/`、
`composite_serve/`(三臂 serve 日志);`quick_*_arm.txt`(客户端输出);
`smoke_chain.log`。
