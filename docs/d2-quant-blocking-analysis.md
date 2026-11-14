# D2 量化路线阻塞点分析（2026-11-14 极限榨取阶段收官）

> 回答交接令的三问：①量化还能搬多少未量化占比进低精度；②KV / int8-attention /
> W4 各自收益-代价；③本栈（16 SM / 闭源引擎 / CUTLASS 隔离）的结构性上限。
> 全部结论以本日实测数据立档（results/20261114_d2_*），可复核。

## 〇、本轮（11-14）落地与判决总表

| 项 | 判决 | 数据 | 回退开关 |
|---|---|---|---|
| P0 decode M=64/128 融合档 | ✅ **上线** | 9/10 点过 ≥1.15× 闸（1.16-1.62×）；serve ITL M=64 档 **−19.4%**、M=128 −6.2%（与逐层换算精确吻合） | `VLLM_FL_INTMM_FUSED_DECODE=0` |
| P1 lm_head int8 | ✅ **上线**（ckpt v2） | 权重 −267MB（池 +6080 tokens 精确换算复现）；parity acc **0.9667**>fake 基线；ITL 效应 ~1-2%（贴烟测噪声底） | serve 指回 v1 目录 |
| P2 KV int8_per_token_head | ⛔ **受阻塞候选** | 编译✓ 数值✓（≤1.2%）**时间✗**：decode 0.90×、prefill 0.726× | `VLLM_FL_KV_INT8`（默认 0） |
| P3 W4A8 | ❌ **死案** | 位精确下 0.24-0.26×（解包 ALU 税 4× > 带宽收益 2×） | — |
| P4 int8 attention 自研 | ❌ **前提证伪关案** | 见 §二；收益前提（KV 带宽减半）在本栈不成立 | — |
| P5 D0 重调 | ✅ 复核无漂移 | 16 形态复扫 6/7 具体行一致，表不动 | `VLLM_FL_ATTN_LAUNCH=0` |

## 一、量化还能搬多少？——线性侧已榨干，剩余大头搬不动

官方场景 GPU 时间构成与量化状态（P0/P1 后）：

| 成分 | 占比 | 状态 | 剩余可搬空间 |
|---|---|---|---|
| attention（bf16） | **65-80%** | ❌ P2/P4 双双受挫（§二） | **带宽类量化搬不动；仅剩非量化的 kernel 形态工程** |
| 线性层（W8A8） | 15-25% | ✅ prefill 融合 95-103% cuBLAS + decode 小 tile 档 1.16-1.62× | ≈0（decode 融合已贴 tile 上限，链已 95-103% vendor） |
| lm_head | decode ~5% | ✅ P1 int8 | ≈0 |
| norm/rope/采样/调度 | ~10% | bf16 elementwise，带宽平凡 | 量化无意义 |

**结论：量化路线在本栈的可用杠杆已收干。** W8A8 把可量化的 ~20-25% 占比打到
vendor 等效（95-103%）并延伸到 decode 档与 lm_head；剩余 65-80% 的 attention
对「降精度减带宽」免疫（§二），对「降精度减计算」也无需求（decode attention
非计算 bound）。线性层侧任何进一步低位宽（W4）被无原生 int4 路径封死（§三）。

## 二、KV 量化与 int8 attention：为什么双受挫——瓶颈不在带宽

P2 探针（graph 3×30 中位，原生 int8_per_token_head 全可读 Triton 路径）：

| 形态 | bf16 | int8 | int8/bf16 | bf16 有效 KV 读带宽 |
|---|---|---|---|---|
| decode S=64 kv=2K | 1124µs | 1256µs | 0.895× | **119GB/s** |
| decode S=64 kv=16K | 8813µs | 9771µs | 0.902× | **122GB/s** |
| prefill S=2 q=2K | 10336µs | 14233µs | **0.726×** | — |

- **要害**：bf16 attention 的 KV 有效带宽仅 ~120GB/s（本机读流天花板 829GB/s，
  16.6GB HBM 的 ~14%）——GQA 2 kv-heads × 16 SM 形态下 decode attention 是
  **延迟/占用 bound**（每 program KV 循环串行 + softmax 累积链），不是带宽 bound。
  「KV 字节减半」兑不成时间；dequant ALU + per-token-head scale 装载 + head
  132 非幂步长的税反而显性化（prefill -27% 与旧 F9「-25%」跨路径互证）。
- **推论封死 P4**：int8 attention 自研（Q·K/V 进 int8）的收益叙事同样是
  「读带宽减半」——同一前提，同一免疫。
- **KV int8 唯一残存收益 = 容量**（池 tokens ×1.94，582K→1.13M）：官方 16K
  场景（16384×64=1.05M 需求 vs 582K 池，容量排队型，同驻 29→58）可能净正
  ——排队缓解 vs attention 每 step −10% 的赛跑。烟测场景容量不 binding
  （必净负）、官方 bench 禁自跑 → **本地无法过闸**，留 env 开关等用户裁定
  是否开一次官方 16K 单场景实测（~15min）。

## 三、W4A8：死案机理（不是调参问题）

- 本栈两层 int8 高性能算子护体：Triton `tl.dot(int8,int8)→int32` 下探原生
  矩阵指令（ixkncli 6.29M 矩阵指令族实锤）+ vendor `torch._int_mm`（ixblas
  110-125 T-MAC/s）。int4 **两层全无**（Triton 无 int4 dot / 无 ldmatrix-int4 /
  vendor 无 int4 GEMM）。
- 唯一开放形态 = kernel 内软件解包→int8 dot：位精确复刻下 0.24-0.26×——
  解包链（and/shift/sub + interleave shuffle）插进 K 循环摧毁流水，税 ~4×
  > 带宽收益 ~2×。代数绕法封死（`dot(a, p)=16·dot(a,hi)+dot(a,lo)` 但 hi/lo
  逐元素混合不可分离）。**除非 backend/vendor 未来提供 int4 原语，此路不通。**

## 四、本栈结构性上限（D2 视角）

1. **算子层**：线性链已到 vendor 等效（95-103%）；16 SM × smem 128KB 的占用
   形态决定 Triton tile 上限（(256,256,64) 86-103%、小 tile decode 档 1.16-1.62×
   已双档扫尽）；CUTLASS/sm80+ 方言隔离 + 天数快 kernel 全在闭源引擎（ixRT/
   ixblas，后者已经由 _int_mm 在用）——没有更快的第三方 int8 可搬。
2. **attention 层（真正的墙）**：65-80% 占比、延迟/占用 bound、D0 launch
   调优已收 1.76-1.85×、量化免疫。**下一杠杆是非量化的 kernel 形态工程**：
   fork 上游 unified_attention 做 flash-decoding 式 KV split + 两段归约
   （提高每 seq 的 SM 并行度，攻串行 KV 循环）——工程量 ~2-3 天量级、无量化
   精度风险，属 D2 之外的 D0-延伸线，冻结前（11/17）时间盒外，建议下期。
3. **基础栈风险（非本线引入）**：EngineCore 负载退化-死亡全周期已刻画
   （0.9333→0.87→0.73→0.47→0.50→静默死亡，~60min 时钟；b=1 干净 = 批路径
   竞态，11-07 libentry 同签名）。官方 bench 单 run uptime <30min 在时钟内
   完成（run1/run2 0.1% 复现自洽）；长跑场景（>45min 单 serve）需重启策略
   或定位 libentry 残留 op（时间盒内未竟，入口在 conc128_stress/README）。

## 五、终态与预期

- **默认配置**：ckpt `/workspace/MiniCPM5-2B-w8a8-v2`（+lm_head int8）；
  env 全默认（decode 融合开、KV int8 关）。
- **对官方双场景的预期**（未经官方验证，等用户令）：decode 密集段 ITL −19%
  （M=64 档）→ 端到端估 +2-5%；P1 另有 ~1%。run1/run2 基线 2838/1346 之上
  的增量预期 ~+3-6%（4K/16K 场景 decode 占比不同）。
- 官方双场景 bench 与 105 题：**只在用户明示时开测**。
