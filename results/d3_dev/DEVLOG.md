# D3 开发日志（六项按序）

纪律：烟测协议见计划；对拍 = 16 条固定 prompt greedy（scripts/d3/parity_check.py）；
decode 代理 [[256,1024,64,64]] / prefill 代理 [[4096,128,64,32]]，bench 内部 4 轮
（超时截断时取已完成轮），跨轮 TTFT 有排队伪影——比较以 Median ITL（decode）与
最优轮 Total tok/s（prefill）为准。官方 bench/105 题：用户明示且六项完成后。

## 步骤 0：基线（2026-11-15 21:22-23:00 前）

- 对拍基线：`parity_baseline.json`（16 prompts，greedy）。
- decode 代理：Median ITL **17.57 / 18.24 ms**（2 轮完整），Total 776/612 tok/s。
- prefill 代理：最优稳定轮 Total **6153 tok/s**（TTFT 9458ms；round1 527 为暖机）。
- 池 588,016；boot ~450-460s。

## ① TD 开启（2026-11-15 22:25-23:10）

代码：`vllm_fl/.../iluvatar/attn_td_mode.py`（包装 TritonAttentionImpl.__init__
强制 use_td=True）+ `iluvatar.py` 接线。serve 日志确认生效（"已强制 ... use_td=True"）。

| 臂 | 对拍 | decode ITL（轮） |
|---|---|---|
| 基线（无 patch） | 16/16 | 17.57 / 18.24 |
| TD on（默认=1 时） | **0/16**（全部前 2 字符内分歧） | 17.04 / 17.36 / 18.24 |
| TD off 对照（env=0） | **16/16（100%）** | 18.71 / 17.24 / 18.55 |
| prefill（TD on） | — | 最优轮 6258 tok/s（vs 基线 6153，+1.7%，噪声级） |

**判决**：
1. 收益真实但代理口径偏小：ITL 中位 −6.4%（on 17.36 vs off 18.55）。代理场景
   kv≤1.3K 偏短——探针口径 TD 收益区在 kv≥2K（1.20-1.31× attn 层），官方 4K
   场景（kv 4-5K）预期更高，待官方 bench 才能定量。
2. **精度风险显著**：off 臂 16/16 复现证明 greedy 跨 serve 稳定，0/16 分歧完全
   归因 TD 数值漂移（42 层复合，早期 argmax 全翻）。非 bit-exact 预期内但幅度
   超预期（探针单层 max_abs 0.12-0.31）。
3. **处置：默认关**（env 默认翻 "0"）。代码保留、一键开（VLLM_FL_ATTN_TD=1），
   终局 105 题门后由用户裁默认值。后续 ②-⑥ 烟测都在 TD off 态跑（隔离更干净），
   收尾终态烟测加做 on/off 双臂。

## ② 采样过滤加速（2026-11-15 23:15-23:35）

根因：`apply_top_k_top_p_triton` wrapper 的 grid=**min(num_compute_units=16, bs)**
→ 16 个 program 串行扛 bs64×130K 词表多趟扫描 = 3.86ms/步（trace 4.4ms 交叉验证）。
kernel 行映射按程序分行、行间独立 → 加大 grid 逐位安全（实测 bitwise_equal=True）。

- 扫参：bs64 grid 16→**64 programs：3.72→1.09ms（3.4×）**；bs32 1.90→1.02。
  pytorch sort 回退路径反而慢 3×（12.5ms，本卡 sort 弱）——不可用。
- 代码：`sample_filter_mode.py`（grid 提至 min(next_pow2(bs),64)，自管 buffer，
  stock kernel/BLOCK/数学全不动）+ iluvatar.py 接线。env VLLM_FL_SAMPLE_FILTER
  （默认 1）+ VLLM_FL_SAMPLE_FILTER_PROGRAMS（默认 64）。
- 单测：bs{8,32,64,100}×3 随机输入 patched vs stock **逐位相等**；k+p 透传分支相等。
- 烟测：**对拍 16/16**；decode ITL **16.25/16.32/16.18**（基线 17.57/18.24，
  off 对照 18.71/17.24/18.55 → 中位 **−12.4%**）。
- **判决：默认开**。逐位等价零精度风险，收益确定性高。

## ③ 段数自适应（2026-11-15 23:37-23:50，数据结案——不落 patch）

补测（results/d3_dev/item3_segments/segm_ext.csv，graph 口径）推翻原前提：
- bs64：n8 仅在 **kv8K** 收 1.24×；kv4K 仅 1.03×，kv16K 0.98×。
- bs32：n8 全线倒挂（0.92-0.96×）。
- bs16：n32 边际（kv4K 1.04×），短 kv 0.91×。
- 官方场景实态：4K 场景 decode bs≤64、kv∈[4K,5.1K]（1.03× 区）；16K 场景
  bs~33、kv∈[16K,17.4K]（0.92× 倒挂区）。**1.24× 收益区（kv8K）官方到不了。**
- 且 cudagraph 捕获把段数按 capture bs 烘焙（kv 运行时增长），无法按 kv 分档；
  改段数还非 bit-exact。
**判决：维持 n16 默认，不写生产 patch。** 探针数据留档供 ⑥ 参考。

## ④ KV int8 扩池（2026-11-15 深夜-11-16，env 烟测——零代码）

- **池 ×1.94 达成**：`VLLM_FL_KV_INT8=1` → cache_dtype=int8_per_token_head，池
  588,112 → **1,140,576 tokens**（写/读/池全链 vllm 0.24 上游原生，P2 管道复活）。
- decode 税：ITL 16.25→~16.9（**+4%**，远好于 kernel 历史 0.929× 悲观预期）。
- prefill 税：端到端**未观察到**（最优轮 6184 vs 基线 6153 tok/s，单轮噪声级）——
  与 kernel 级历史 0.820× 不符，待官方 bench 精判（存疑记录）。
- 容量：16K×40 负载下 0 抢占、kv 峰 0.58（bf16 池同负载 >1.0 必然挤压）；
  同驻解放直观（21 路时 kv 仅 0.30）。
- **对拍 1/16**：int8 KV 数值漂移大（可见分歧多为同义正确表述），105 题门是
  硬闸门——精度风险真实存在，与公开证据（int8_per_token_head 73=73 基线）需
  实测对齐。
- **稳定性**：首 serve 在 prefill round2 无栈静默死亡一次（EngineDeadError）；
  复现 serve 上 16K 重 prefill 9min + prefill 代理 4 轮完整跑完均存活——
  **未复现**，归档为已知 EngineCore 间歇竞态嫌疑（11-07 同签名前科）。
- **判决：默认关（platform.py 不动），决策数据已齐**：16K 收益大/4K 损失小/
  精度与稳定性双闸门，终局由用户裁。

## ⑤ prefill TILE 深扫（2026-11-16 19:40-20:05）

扫参（results/d3_dev/item5_tile/tile_sweep.csv，3 形态×11 臂，graph 口径）：
**TILE=16 w2s1 一致最优**——默认 t32 慢 ~10%，t64/t128 劣化至 0.55-0.64×
（16 SM 上小块=更多波次=更低单 CTA 串行延迟，方向与直觉相反）。

- 最优臂：S1q4096kv4k 1.107× / S2q2048kv16k 1.100× / S4q4096kv4k 1.110×（attention 层）。
- 代码：`attn_tile_mode.py`（patch `_get_tile_size` prefill 分支 32→16）+ 接线。
  env `VLLM_FL_ATTN_TILE`（**默认 0**——非 bit-exact 类，同 ① 处置）。
- ON 臂烟测：**prefill 代理最优轮 6941 tok/s vs 基线 6153 = +12.8%**；
  对拍 0/16（kv 循环重分组翻转早期 argmax，与 TD 同形态）。
- **判决：默认关、env 开、数据入册。**（≥1.15× 阈值未到但近乎零成本，落码
  不默认启用；kernel 形态改造的深水区见 ⑥ 与终局报告。）

## ⑥ split-KV decode kernel（2026-11-16 20:10-20:25，前置重估——数据结案）

dtile_sweep.csv（decode 3D 路径 TILE 深扫，TD=off，graph 口径）：
**TILE=16（现状默认）全线最优**——t32 慢 10-27%（bs64/kv4k 948.7→1198.2µs），
t64 慢 45-58%。本卡偏好「多而小 CTA」（与 ② grid64、⑤ prefill t16 同一规律）。

- 自写 kernel 的机理假设逐项被零成本探针证伪：更大 TILE ✗（本轮）、重分段 ✗
  （③：官方形态无收益）、唯一残存杠杆 = TD 式向量化加载（已由 ① 覆盖，待门）。
- 残留 headroom：tuned 660 vs 现状最优 318-386GB/s ≈ 1.7-2×，但需从零写
  kernel（2-3 天）+ 非 bit-exact 风险，且边际与 TD 重叠。
- **判决：本轮不动工**。续作设计备忘（若终局 TD 未过门再启）：metax
  `triton_decode_attention.py` 两段骨架 + FlashInfer Algorithm 1 自适应分段
  （target_CTA=k×16、短请求 writethrough）+ 「多小 CTA」约束（TILE=16、
  w1s1 基线）+ bare jit/无 atomic 纪律。

## 收尾：终态烟测（2026-11-16 20:30-21:15）

- **默认臂（仅 ② 开，交付配置）**：对拍 **16/16**；decode ITL 15.96/16.35/
  16.12/16.20（中位 16.2，vs 基线 **−9~12%**）；prefill 最优轮 6303（+2.4%）。
- **all-on 臂（①+④+⑤+②）**：对拍 0/16；decode ITL 18.68/18.71/18.80（比默认
  臂**差 16%**）——**①×④ 组合互毁**（TD 加载 × int8 descale 路径）；prefill
  最优轮 5963（−5.4%）。**结论：不可同开，推荐 ②+⑤+④ 或 ②+⑤+① 二选一。**
- 交付文档：docs/d3-dev-delivery.md。六项完成，**停，等用户明示官方 bench/105 题**。
