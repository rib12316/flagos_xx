# Survey 两份报告 × 11-20 双机实测 的交叉裁定（A 机）

> 两份 survey（competition/industry）内容定格在 D2 期（~11-08），来源质量高、方法论严谨；
> 但其中多处核心假设已被 11-20 全天实测改写。本文件为逐条裁定，防止按过时假设行动。
> 判定依据：docs/00-统一假设台账.md §9 + results/20261120_c500_reprofiling/。

## 一、被今日实测改写/收窄的 survey 结论

| Survey 主张 | 今日实测裁定 |
|---|---|
| 「16K prefill 段仍是最大矿脉」（依 #671 +199.7%） | **C500 上不成立**：#671 修的是 Triton unified_attention 的 prefill 默认参数病（BLOCK_M=16）；我方 C500 路由的是 **MACA flash（vendor 闭源）**，serve 实测 prefill attention 437.6µs/层 = **e2e <1%**（B 机 trace 复核，45.6ms/层锚点系 L2-cold 伪影）。且 #671 参数在 C500 直接 OOR 崩溃（262KB>64KB smem，其自述） |
| #778 Python 层 +23.9/24.9%（host 开销大西瓜） | **收窄至 ≤4%**：我方 decode GPU busy 96%（trace 实测）→ host/launch 空隙只有 4% 可回收；其数字是 vs 比赛基线（含 .item() 同步等），我方 D1 调度默认已消大半 |
| 「KV-int8 收益潜力可能大于 W8A8」（MLPerf 排序） | **门控实测**：C500 无容量钳制（池 1.26M/零抢占→扩池收益=0）；带宽路径唯一载体 `flash_attn_with_kvcache_dequant` 实测 **0.46×**（332 vs 1274GB/s，num_splits 全扫无效）——钥匙在 vendor |
| FlashMLA「int8 KV 理论上限 decode ~1.8×」 | 上限算术正确但需 kernel 载体；另 FlashMLA 的 **scale 紧邻数据布局**恰是 vendor op 所缺（其 scale 独立张量=第二访存流）——**已加入 vendor 问题清单** |
| 「小算子融合链 +10-15%」 | 部分成立：norm/rope/cat 合计 1.7ms/步(6.5%)+quant 1.55ms(6%)——**D13 norm+quant 融合已派 B 机**（+2-3% 预期）；FlashMLA 式深融合进 attention 被 vendor 闭源阻断 |
| DeepGEMM 定点 shape 调优 | 已做尽（D0 launch 表） |
| SGLang「FP8 cast 不融合反更慢」反面教训 | 我方已规避（mctlass w8a8_scaled_mm_azp epilogue 已融） |
| 摩尔线程「63% kernel 静默降级」审计 | 等价动作已做：flash 83-88% 带宽、quant/采样/norm 各在下限、triton dot 本征 56T（B 机 GEMM 校准）——无静默降级黑洞 |

## 二、survey 中仍然成立且可做的（按性价比排序）

1. ~~#417 模式：AttentionCGSupport 误报→decode 被降级~~ **已关账（B 机 32d3262，11-20）**：decode FULL 捕获 37 条实锤（A 机此前看到的 PIECEWISE 进度条只是 piecewise 阶段）+ async scheduling 日志明文已启用——**双项均最优态，无静默降级**；昆仑xon 的 #417 病在 MACA 后端不存在。教训：进度条≠全集，结论需逐条实证
2. **D13 norm+quant 融合**（已派 B）：survey 的融合链主张的可执行子集
3. **vendor 问题清单增强**：拿 FlashMLA 布局对照问沐曦（scale 紧邻数据 vs 独立张量；332GB/s 是否可修）
4. **报告写作素材**（30% 创新分）：#671 的「内核优化 vs 调参」论证模板、KernelGen 五芯调参叙事、我方台账 §9 八方向负判决+三条方法论铁律（L2-warm/graph 口径/serve trace 准）——"证明前沿"的审计链本身是强报告内容
5. **#384 逐算子拉黑审计**（低优）：FlagGems vs aten 全算子 A/B——我方小算子占比已低（<10%），上限 ~1%

## 三、竞品情报要点（存档）

- **#671（RobertFengOPC）在 V150 上 16K=2583.27**，我方 V150 交付 1570.62——竞品 16K 快 1.64×（其修 Triton attention 参数病的收益）。V150 机器已不可用，此差距仅作复盘记录；若有复议窗口，该参数网格是 V150 侧已知最大遗漏
- #778 声称双卡 +20%+，survey 自身已标注含金量存疑（metadata cache 反多一次 copy_）
- C500 侧我方交付 6399/7510 vs 两竞队公开数字（V150 口径）无直接可比项（不同卡）

## 四、明确不可做（survey 已列 + 今日新增证据）

#671 参数直接复用（C500 OOR 崩溃）、megakernel（调度器不兼容）、PD 分离（负载不匹配）、投机采样（禁令+busy 96% 前提不成立）、mctlass/ISA 级重写（超范围，记为已知缺口 20-30%）、自研 int8-KV triton kernel（盈亏平衡 0.50× vs V0-V2 实测 0.54×，上限 1.07× 薄）。
