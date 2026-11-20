# F-a：rope + cat 融合（feat/fa-rope-cat-fusion）——T0 就绪包

> agent-fa 交付（2026-11-20 11:30-13:00，无卡开发，T0 脚本只写不跑）。
> 分支 `feat/fa-rope-cat-fusion`（自 `delivery/s2-final` 切出），未 push。

## 1. 归因结论（证据：/tmp/torchinductor_root 生成码 + trace_all/topk_kernels.txt）

**菜单项「rope + cat」的真实形态与预设不同，但融合前提成立（判决：相邻，可融）。**

decode 步（bs=32，1027 步窗口）逐层数据流（LlamaForCausalLM 前向，
vllm 0.24 + 插件，MiniCPM5-2B-w8a8-v2：16 Q heads / 2 KV heads / head_dim 128 /
rope_theta 5e6 / max_position_embeddings 131072 / bf16 / w8a8 int8）：

```
qkv = qkv_proj(hidden)                 # buf29 [T,2560] 连续（w8a8_mctlass_scaled_mm 输出）
q, k, v = qkv.split([2048,256,256],-1) # 三段 strided view，无拷贝
q, k = rotary_emb(positions, q, k)     # compile 模式 CustomOp 解绑 → forward_native
  ├─ triton_poi_fused_cat_3 = rope(Q)  # 42107 次 244.74ms = 5.81µs/层·步
  │    读 qkv[:,0:2048] + positions + cos_sin[131072,128] → 写 Q' [T,16,128] 连续
  └─ triton_poi_fused_cat_4 = rope(K)  # 42107 次 189.80ms = 4.51µs/层·步
       读 qkv[:,2048:2304] → 写 K' [T,2,128] 连续
v 不做 rope：直接 strided view（qkv offset 2304, stride 2560）
→ Q'/K'/v_view → unified_attention + vllm::reshape_and_cache_flash(0.23ms/步)
```

关键订正（对任务书的三个假设）：
1. **「cat」不是独立算子**：cat_3/cat_4 的命名来自 native rope 分解
   （`ApplyRotaryEmb.forward_static`，common.py:145-185）中 `rotate_neox` 的
   `torch.cat((o1,o2),-1)`（生成码 graph fragment：`%cat = aten.cat([%sub_44,
   %add_151], -1)` 实证）。图内没有 qkv-cat 算子与 rope 相邻——**菜单的「rope+cat
   融合」实际可做项 = rope(Q)+rope(K) 两核合一**（Q/K rope 同输入、同位置表、
   无插算子，数据流完全相邻 → 不触发止损条款）。
2. **fused_1（0.32ms/步）不是 rope，是 silu_and_mul**：trace 43134 次 = 42×1027
   （42 层每层 MLP 一个）；ttir 实证 sigmoid+mul 读 [T,12288] gate_up 输出。
   rope 池 = cat_3+cat_4 = **0.42ms/步**（26ms 步长下 ~1.6%）。
3. **41-vs-42 异层**：cat_3/cat_4 各 42107 = 41×1027；其余逐层核（norm/silu/
   cache 写）都是 42×1027。缺的 1 层 rope 被并入了其它融合组、kernel 名不在
   top 榜（该层每步仅 ~2×10µs=0.02ms，占 0.08%）。fx 层面该层 rope 代码相同，
   pattern pass 若匹配则连它一起替换（41/42 收益 + 1/42 零头）。

**慢因**：两核块形按 prefill 上界选定（size_hints 4194304/524288，
XBLOCK=1024/2048、num_warps=8——缓存 json 实证），decode T=32 时 cat_3 仅 64
CTA、cat_4 仅 **4 CTA**（104 SM 的 3.8%）→ 纯延迟 bound：~300KB 流量跑了
10.3µs（等效 ~30GB/s）。

**负位置回卷**：现役核内含 `pos<0 → pos+131072`（ttir 实证；fx 图层无此算子，
CPU 符号追踪 `forward_static` 证实——系 inductor 间接索引 codegen 引入）。
融合核保留同语义，任意输入下与现役核同值。

## 2. 设计

### 融合 kernel（`vllm_fl/dispatch/backends/flaggems/impl/rope_cat.py`）
- 一个核同时算 Q' 与 K'：grid = `(T × (16+2),)` 纯形状函数，每 program 一个
  (token, head)，HEAD=128 恒 2 的幂 → **零尾部掩码**；num_warps=1/num_stages=1
  静态（纪律模板 attn_splitkv.py）。
- 数学逐位照抄现役核：bf16 load → f32；`o1 = x1*c - x2*s`、`o2 = x2*c + x1*s`
  （与 inductor 源码同形 sub(mul,mul)/add(mul,mul) → triton 同机制收缩 fma）；
  RTNE 截回 bf16。cos_sin 行 = [cos64|sin64]，gather 按回卷后 pos。
- I/O 契约（与现役逐位同构——graph 不破坏的关键）：入 qkv[T,2560] 连续 +
  positions[T] i64 + cos_sin[131072,128]；出 Q' [T,16,128] 连续（stride
  2048,128,1）+ K' [T,2,128] 连续（stride 256,128,1）。
- 预期：T=32 → 576 CTA（vs 64+4），单 launch（vs 2）→ 目标 10.3µs → ~4-5µs。
  池 0.42ms/步 × 41 层收益比 ≈ 若 kernel 级 2×，e2e ≈ +0.6%（26ms 步长）。

### compile pass（`vllm_fl/compilation/passes/rope_cat_fusion.py`，插件首个自定义 pass）
- 照上游 rope_kvcache_fusion.py 配方：`direct_register_custom_op` 注册纯函数 op
  `vllm.fl_rope_qk_cat(qkv, positions, cos_sin, NQ, NKV, D) → (Q', K')`
  （无 mutation → 不需要 auto_functionalized）→ `VllmPatternReplacement` 匹配
  `{qkv.split → native rope(q,k) → view(T,H,D)}` 整段替换。
- MatcherRotaryEmbedding 显式 `enabled=False`（追 native 分解与 compile 图同构；
  本机 `_C.rotary_embedding` 存在但 compile 模式仍走 native，默认态会错配）。
- **pattern 必须绑定 v 并作第三输出**：pattern matcher 要求节点 users 数一致，
  真实图 split 的第三个 getitem 喂 v-view——CPU 干跑实测缺它整模式失配
  （MATCH_COUNT 0 → 补上后 1）。
- 注入：`platform.py` metax 段（compilation_config 变异先例区）把 pass 实例塞
  `inductor_compile_config["post_grad_custom_post_pass"]`（backends.py
  configure_post_pass 会 add 进 PostGradPassManager——vllm 官方
  ActivationFusionPass 同机制；`inductor_passes` dict 路由只在
  CompilationConfig.__post_init__ 解析，config 期后注入必须走实例直塞）。
  env 门 `VLLM_FL_FA_ROPECAT`（默认开，=0 完全不注入）。头数从 hf_config 读
  （16/2/128），装载期一次定型，无运行时分支 → cudagraph 安全。

## 3. 口径声明

- **位精确**：融合核 vs 现役 cat_3/cat_4（同输入同运算序同舍入）。T0 对照臂 =
  这两核的逐字拷贝（XBLOCK 1024/2048、num_warps 8/8 与缓存元数据一致）。
  风险：triton fma 收缩形态若与本核不同 → 末位差；T0 位不齐则按分布等价降级
  （atol 1 bf16 ulp 量级）并论证后再判。
- 计时铁律：L2-warm（同输入反复跑，elementwise 不破坏输入）+ CUDA graph replay
  中位（抄 d9_quant_probe_metax.py 的 bench()）。
- 上机顺序：T0 过门槛（bitwise + ≥1.15×）→ smoke serve（log 确认
  `F-a rope+cat fusion pass injected` + pattern 命中数 + decode 步 trace 里
  cat_3/cat_4 消失、fl_rope_qk_cat 出现 ~42 次/步）→ T1 quick 双形状三轮中位
  ≥+0.3% 才立项合入。

## 4. CPU 干跑实证（无卡，FakeTensorMode，不触设备）

| 检查 | 结果 |
|---|---|
| 5 文件 py_compile | 全过 |
| pass 构造（真实 MiniCPM5 ModelConfig，hf 16/2/128） | OK，2 patterns（neox 两态），uuid 稳定 |
| backends.py 式 deepcopy(pass) | OK（inductor_config 深拷贝路径） |
| pattern 匹配模拟生产图（split→forward_static→view，view_to_reshape 规范化） | **MATCH_COUNT=1**，替换为 vllm.fl_rope_qk_cat.default |
| 替换后图 fake 执行 | 输出 (5,16,128)/(5,2,128) bf16，Q'/K' 连续、v 保持 strided view（stride 2560）——与生产布局一致 |

未验证（需上机）：triton 核编译/运行、真实 dynamo postgrad 图匹配（模拟图用
pm.fwd_only 追踪，与 aot 分解表理论同构——上游同款风险）、位精确性、耗时。

## 5. 上机申请（精确命令）

```bash
# 0) 确认 GPU 空闲、无他人 serve
pgrep -f "api_server --model" || echo GPU_FREE

# 1) 装插件（provenance 四查按 README）
cd /data/flagos_feat_fa/vllm-plugin-FL && \
  pip install --no-deps --no-build-isolation . && \
  pip show vllm-fl | grep -i location

# 2) T0（秒-分钟级）
python3 scripts/fa/t0_rope_cat.py 2>&1 | tee results/20261120_fa_ropecat/t0.log
#    判据：corr 行 bitwise 全 True（或 maxdiff=0）；M∈{32,64} speedup ≥1.15×

# 3) smoke serve（确认注入 + 命中；VLLM_PATTERN_MATCH_DEBUG=1 可看匹配明细）
VLLM_FL_FA_ROPECAT=1 python3 -m vllm.entrypoints.openai.api_server \
  --model /workspace/MiniCPM5-2B-w8a8-v2 --... (serve 参数照抄 d5 冻结集) \
  2>&1 | tee /tmp/smoke_fa.log
#    看点：`F-a rope+cat fusion pass injected`、`Replaced N patterns`（期望 ~42/图）、
#    decode trace 中 triton_poi_fused_cat_3/4 消失、fl_rope_qk_cat 出现
# 3b) 回退臂（A/B）
VLLM_FL_FA_ROPECAT=0 python3 -m vllm.entrypoints.openai.api_server ...（同上）
```

T1（quick [[4096,1024,64,32]]/[[4096,1024,64,64]] 双形状 ≥3 轮中位，GPU 独占）
按 §6 验证协议另行排队。

## 6. 风险登记

1. **pattern 失配**（最大风险）：生产 postgrad 图与模拟图结构差（aot 分解表
   差异、empty-cat 残留、reshape 链折叠）→ pass 静默 0 命中，退回现状（无正确
   性风险，纯无收益）。取证：`VLLM_PATTERN_MATCH_DEBUG=1` + `--compilation-config
   '{"compile_debug_dump_path": ...}'` dump 图与 pattern 对 diff。
2. fma 收缩末位差 → 位精确降级为分布等价（判据放宽至 1 ulp，需论证）。
3. cat_4 的 4-CTA 现状下 4.51µs 里 launch 尾占大头，融合后单核若只拿到 ~2×
   而非 2.5×，e2e 预期降为 +0.4%——仍在立项线上。
4. 第 42 层 rope 若真在别的融合组里且结构不同 → 该层不替换（收益 41/42）。
5. prefill（T=2048）同核同 grid 公式（36864 CTA），理论无回归；T0 的 M=2048
   信息位会实测确认。
