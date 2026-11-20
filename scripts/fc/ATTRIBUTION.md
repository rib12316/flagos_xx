# F-c · `_copy_kernel` 归因报告（branch feat/fc-copy-attribution，11-20）

数据源：`/data/flagos_wt/results/20261120_c500_reprofiling/trace_all/rank0.1795110863960817196.pt.trace.json.gz`
（1027 decode 步，bs=32，LlamaForCausalLM 42 层，hidden 2048，16Q/2KV head_dim 128，FULL_AND_PIECEWISE）。
归因脚本：`scripts/fc/trace_copy_attribution.py`（纯 CPU，可重复；关联通道=External id / correlation / GPU 时间线邻接）。

## 1. 结论（主站是谁）

**`_copy_kernel_kernel_rank_3` 稳态 decode 恰为 42.0 次/步 = 每层 1 次，100% 是
attention 输出 → o_proj 输入的 staging 拷贝**，即插件
`vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py`
`MacaFlashAttentionBackend.forward` 的 "Metax Modification" decode 分支：

```python
output_unreshape = flash_attn_with_kvcache(q=decode_query, ...)   # flash 内部 new out
output[:num_decode_tokens] = reshape_attn_output_for_spec_decode(output_unreshape)  # ← 拷贝在这
```

拷贝形状 **[32(=bs), 2048(=16头×128)] bf16**（grid(32)×block(512)，4 元素/线程，65536 元素），
avg **3.30µs**，合计 **138.5µs/步 ≈ 0.14ms/步**（42 层 × 1 次）。开发单的 0.19ms/步 中其余
~0.05ms 为采样器杂项拷贝（§4）。

### 证据链（三条独立通道互证）

1. **GPU 时间线邻接**（决定性）：以 `reshape_and_cache_flash`（43134 次 = 42/步）为层标记切步，
   稳态 10 步窗口内 rank_3 共 420 次 = **42.0/步**；每一次的前驱核都是
   `flash_fwd_splitkv_combine_kernel`（attention 输出汇合）、后继核都是
   `_dynamic_per_token_quant_int8_bare_kernel`（o_proj 的 w8a8 激活量化）——精确锁定
   「attention 出 → o_proj 进」这条边。
2. **机制推理**：`reshape_attn_output_for_spec_decode` 只做免费 view；MACA flash python
   wrapper（`flash_attn/flash_attn_interface.py:1663`）对 `flash_attn_cuda.fwd_kvcache` 的
   第 13 参恒传 `None`，flash 因此内部新分配 out，`output[:n] = view(...)` 必然发射一次逐元素
   拷贝。上游契约（vllm `attention.py:493` `output = torch.empty(...)` + mutating custom op
   `torch.ops.vllm.unified_attention_with_output(..., output, ...)`）强制 impl 把结果写进
   caller 的 output 缓冲 → 该拷贝是两边契约夹出的「桥」。
3. **反证排除**：开发单怀疑的 `linear.py:117` `x.reshape(-1,K).contiguous()`（4 linear/层）
   **不是主站**——若 o_proj 输入不连续，会出现额外拷贝且每层不止 1 次；实测 42.0/步与
   「仅 attention 输出一次搬运」严格一致。o_proj 的输入（output 缓冲）本身连续，
   `.contiguous()` 为 no-op。

### 方法论注记（给后续 trace 归因复用）

- kineto 无 python 栈时，**cudagraph 回放内核的 cpu_op 链不可信**（`correlation` 指向
  mcGraphLaunch，包含关系命中的是异步交错的采样器 aten 事件，如 `aten::fill_`/`softmax`/`sort`）；
  本案 46074 次中约 3/4 的 CPU 链都是这种伪影。**可靠的归因 = GPU 时间线上以每层标记核
  （reshape_and_cache_flash）切步 + 前后邻核定位**。
- 46074（全程累计）≠ 稳态 44.9/步：累计数含 cudagraph 捕获期一次性发射（各 bs 档 42 次/图，
  ts 聚集在 8ms 内可证）、prefill/混批 eager 发射与采样器 eager 拷贝。

## 2. 修复状态：已实现（可删型，插件侧最小改动）

文件与机制：

| 文件 | 改动 |
|---|---|
| `vllm-plugin-FL/vllm_fl/envs.py` | 新增 env 逃生门 `VLLM_FL_FLASH_KVCACHE_DIRECT_OUT`（默认 1 开，=0 回退旧 staging 路径） |
| `vllm-plugin-FL/vllm_fl/dispatch/backends/vendor/metax/impl/attention/flash_attn.py` | 模块级 `_flash_direct_out_active()` + `_flash_kvcache_direct_out()`；decode 分支优先直写 |

直写依据（静态二进制取证，`flash_attn_2_cuda.so` 的 `mha_fwd_kvcache` 反汇编引用的
TORCH_CHECK 字符串）：`"out must be on CUDA"`、`"out must have shape (batch_size,
seqlen_q, num_heads, head_size_og)"`、`"Output must have the same dtype as inputs"`、
`"Output tensor must have contiguous last dimension"`——**C++ 层支持预分配 out**，python
wrapper 只是没透出。`output[:n].view(bs,1,16,128)` 与 flash 内部 out 内存布局逐元素一致
（均连续、同 dtype），combine 核直接写 output，数值**位精确 by construction**（同一组核、
仅目的地址不同）。

graph 相容：分支条件 = env + 装载期 `import flash_attn_2_cuda` 符号探测 + 进程内一次性
失败记录（异常后永久回退并 warning），均在 cudagraph 捕获前定型，无运行时数据依赖；
直写反而降低图内私有池分配次数。

预期收益：稳态 decode 每 step 少 42 个核 × 3.30µs = **0.14ms/步**（26ms 步长 →
**~+0.5% 上限**，与开发单 F-c 预估 +0.3-0.5% 一致）；混批/prefill 步的 eager 发射同样受益。

## 3. 上机验证申请（主会话执行；GPU，本 agent 未跑）

**首跑失败与修正（11-20 13:2x，T0 v2）**：首版 T0 以 `empty_like(wrapper返回值)` 构造
out，被 C++ 拒以 `out must have shape (batch_size, seqlen_q, num_heads, head_size_og)`。
反汇编复核（`mha_fwd_kvcache` @0x1b5ec90；返回 `std::vector<Tensor>` 走 sret，故
rsi=q、rdx=kcache、rcx=vcache、r8/r9=k/v、栈 0x38=block_table、**0x48=out optional**）
确认 out 检查链：`dtype==q.dtype → is_cuda → stride(-1)==1 →
ArrayRef::equals(out.sizes(), {q.size(0..3)}, len=4)`——out 必须**恰好 4 维且逐维
等于 q 形状**。而 MACA wrapper 对 seqlen_q==1 的返回是 **3D [bs,H,D]**（vllm
`reshape_attn_output_for_spec_decode` 的 `dim()==3: Already in the correct shape`
分支即为此存在），`empty_like(ref)` 得 3D → 维数 3≠4 必炸——**T0 脚本 bug，非补丁
bug**：serve 补丁传的是 `output[:n].view(*decode_query.shape)` = 4D [32,1,16,128]
（= q 形状），正是检查所要。T0 v2 已改 `torch.empty_like(q)` + flatten 逐位对拍。

```bash
# 0) 环境自检 + 插件重装（provenance 四查见 README）
cd /data/flagos_feat_fc/vllm-plugin-FL && pip install --no-deps --no-build-isolation . && pip show vllm-fl | grep -i location

# 1) T0 位精确 + cudagraph 相容（~1min；v2）
python3 /data/flagos_feat_fc/scripts/fc/t0_direct_out.py            # 期望: [A] shape=(32,16,128) 3D；[B][C] PASS → T0 PASS

# 2) 逃生门回归（=0 时门关闭；serve 侧 decode 分支即回退旧 staging 路径）
VLLM_FL_FLASH_KVCACHE_DIRECT_OUT=0 python3 -c "import sys; sys.path.insert(0,'/data/flagos_feat_fc/vllm-plugin-FL'); import torch; from vllm_fl.dispatch.backends.vendor.metax.impl.attention import flash_attn as fa; print('gate:', fa._flash_direct_out_active())"
# 期望输出: gate: False（T0 的 arm_b 直调 helper 不经门，故门回归用本行验证）

# 3) T1 serve quick case（双形状 ≥3 轮中位，≥+0.3% 才立项合入；现役冻结参数，零附加 env）
#    按仓里现役 T1 快跑协议执行 4096/1024/64/32 与 4096/1024/64/64 两形状；
#    对照臂 = delivery/s2-final 同协议轮次。
```

观察点：T0 `[A]` 行打出 wrapper 返回形状（预期 3D (32,16,128)，顺带验证 3D-return
推断）；serve 日志应出现 `fl.flash-direct-out: on`；若出现 `off:runtime(...)` 说明
C++ 运行期另有未覆盖限制，补丁自动回退零风险，届时 F-c 转判「结构性不可动（等
vendor 透出 out）」并仅留报告。

## 4. 其余拷贝判决（非主站，不动）

| 类别 | 量（稳态） | 归因 | 判决 |
|---|---|---|---|
| `_copy_kernel` rank_1/rank_2/plain + rank_3 g2048xb512 | ~5/步，合计 ~0.02ms | 采样器 eager 链（logits `.to(fp32)`/clone/sort/contiguous，[32,130560] fp32 ~2 次 + 标量小拷贝） | 结构性（vllm 采样器），量小不立项 |
| 捕获期发射 | ~1.5k 次（一次性） | cudagraph 各 bs 档捕获时的同代码 eager 发射 | 无稳态成本，忽略 |
| `gpu_memcpy`（mcMemcpyAsync 14533 次） | — | D2H 元数据回读（host 侧任务③范畴） | 不属 F-c |

## 5. 交付物清单

- `scripts/fc/trace_copy_attribution.py` —— 可重复归因脚本（纯 CPU）
- `scripts/fc/t0_direct_out.py` —— T0 对拍脚本（GPU，待主会话跑）
- `scripts/fc/ATTRIBUTION.md` —— 本报告
- 补丁 2 文件（envs.py / metax flash_attn.py），git commit 于 feat/fc-copy-attribution（未 push）
