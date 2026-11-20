# 任务② 逐算子替换扫描 · 审计表（feat/s2-opscan-audit，新 agent 11-20）

方法（开发单 §4）：trace `topk_kernels.txt` 枚举在跑 kernel → 映射现实现 → 找 drop-in 候选（aten 原生 / mcoplib `torch.ops._C.*` / FlagGems 别档）→ graph 口径 + L2-warm A/B → 判决入表。大件（attention/GEMM/quant）已判最优不扫；目标池 = norm/rope/cat/copy/采样杂项 <10% 小件。

**口径教训（A-H4 实测新增，适用于全表）**：kernel 级孤立 A/B 的赢幅在 serve 内会被 inductor 邻融合削弱——凡现役实现是 inductor 生成核（`triton_*_fused_*` 族）的候选，serve 级预期按 kernel 级赢幅 × 0.3-0.5 折扣预估，低于 +0.3% 门的不上 T1。

## 审计表

| op | 现实现（serve 实测口径） | 候选 | 实测 | 判决 |
|---|---|---|---|---|
| fused_add_rms_norm | native inductor triton（`triton_red_fused_fused_add_rms_norm_0/2`，0.60ms/步） | mcoplib C++ `torch.ops._C.fused_add_rms_norm`，经 `get_default_ir_op_priority` per-op 翻转接线（接线成功：日志明文+37 FULL 捕获） | kernel M32/64/2048 = **1.10×/1.14×/1.24×，输出位一致（0 mism）**；T1 双形状后3中位 **+0.086%（5937.37 vs 5932.30）/ +0.32%（7013.36 vs 6991.17）** | **❌ 不合入**（双形状 ≥+0.3% 门未全过；e2e 上限 ~0.25% 贴线）。分支 feat/ah4-irnorm-vllmc 留档。**机制修正入档：开发单"metax.yaml 路由排除"说不成立，真机制=IR priority 默认 native** |
| rotary_embedding（Q/K rope） | native 分解两核 `triton_poi_fused_cat_3/4`（0.42ms/步，"cat"名源自 rotate_neox 内部 cat；decode T=32 时 cat_4 仅 4 CTA=3.8% SM） | ①mcoplib `torch.ops._C.rotary_embedding`（单核 QK 同调）②**F-a 融合核**（rope_qk_cat 两核合一，feat/fa-rope-cat-fusion） | F-a **T0 已绿**：M32/64 = **1.61×/1.54×**（门槛1.15×）、q/k bitwise maxdiff 0（含负位置回卷）、prefill 1.00× 无回归；e2e 投影 +0.58%；**接线 debug 中**（pass 注入 WorkerFL 后 serve 内零执行痕迹，探针定位轮进行） | **进行中（F-a 通道）**。慢因=块形按 prefill 上界选定，decode 低 occupancy——融合核实测 ~1.6× |
| silu_and_mul | inductor 融合核 `triton_poi_fused_1`（0.32ms/步，**含邻算子融合**——sigmoid+mul+邻接 pointwise） | mcoplib `torch.ops._C.silu_and_mul` 单核 | 未测 | **预估负，不立项**：现役已与邻融合，单独替换必丢融合（A-H4 同型教训更糟——C 核无邻融合能力）；F-d（silu-mul+quant 融合）才是正路，阻塞于 B 机 D13 接线 |
| reshape_and_cache_flash | `vllm::reshape_and_cache_flash_kernel`（0.23ms/步 ≈0.9%） | mcoplib `torch.ops._C.reshape_and_cache_flash` | 未测（F-b 推迟项的 scan 级探针，机会窗口给 F-a 后） | 待定：单项上限 +0.4-0.5%，仅当 F-a 失败后视余量补测 |
| copy（attention 输出 staging） | `_copy_kernel_kernel_rank_3` 42/步（0.14ms/步，主站=MacaFlash decode 输出 staging 拷贝，三通道证据链） | 直写 out（F-c 补丁，位精确 by construction，env 门+永久回退） | **探测矩阵九变体**：V1（out 逐维==q 的 4D）/V2 3D/V6/V9 全部同串拒收 "out must have shape (bs, seqlen_q, H, head_size_og)"；V7 stride≠1 / V8 fp16 换错误串（contig/dtype 检查活着）；V5 排除位置差；wrapper 同输入 out=None 正常 | **❌ 结构性不可动关账（当前 .so 构建的 out 合同运行期不可满足，md5 0953e2b9）**——vendor 升级后翻 env 门=1 重跑 t0 即可复活（0.14ms/步 免费回收点）。分支 feat/fc-copy-attribution 留档（归因报告+矩阵证据） |
| 采样尾杂项（true_div/scatter/hist 等） | `true_div_func_kernel_rank_2` 0.07ms/步 + `_scatter_jit` + sort 链 | —（kernel 路线在采样已判死：topk_topp 五档 BLOCK 全运行期 ATU，台账 §9） | — | **不扫**（D5 已全域覆盖判决：serve 内采样真实成本 1-2ms，快路径 serve 级零收益判死） |

## 小结（13:15 时点）

- 扫描池在 A-H4/F-a/F-c 三通道外已无 ≥0.3% 的独立 drop-in 项：现役小件多已是 inductor 融合核（替换单核必丢融合）或 vendor 二进制（无替代实现）。
- 任务②的净产出 = 本表 + A-H4 机制修正（IR priority 真机制，后续任何 norm 类替换的接线模板）+ F-a/F-c 两通道的判决行（随其 T0/T1 落地回填）。
