# D13 落地三工序 —— 接线层交付记录（§3.5 装载期静态配对）

2026-11-20 B 机 · feat/d13-norm-quant-fuse · 本文件为 §0 五件套第 3/4 件的证据链。

## 接线形态

- **配对**：`vllm_fl/quantization/w8a8/d13_norm_quant.py::install_d13_pairings`
  （挂点 `vllm_fl/worker/model_runner.py` ModelRunnerFL.load_model，weights+
  process_weights 之后、GraphWrapper/compile 包裹之前）。Llama 族 allowlist
  （`LlamaDecoderLayer`），layer0 input_layernorm 跳过（residual=None 无 add
  可融）→ **83 站**（42 层 × 2 − 1）。
- **`model.norm→lm_head` 不配对（v1 降级，用户裁定）**：runner 在 norm 与
  lm_head 之间做 `hidden_states[logits_indices]` 切片（M'<M），scale 无法逐行
  联动 gather，需 patch 5 处切片 site；该站收益 ≈ 一次 µs 级 launch。列后续
  独立假设。
- **数据流**：q_int8 走 norm 返回值（模型张量流，中间零算子）；scale 走配对
  norm 的非持久 buffer `_fl_d13_scale`（融合 kernel `scale_out` 直写），配对
  linear 经 `_fl_prequant_norm` 回指读 `[:M]`。buffer 身份恒定 → dynamo guard
  稳定 + cudagraph 静态指针。
- **per-shape 路由**：`M > m_fuse_max` 走原两步链。默认 m_fuse_max =
  `compilation_config.max_cudagraph_capture_size`（本 serve = 512；capture
  sizes [1..512]，prefill/mixed M≥1024 全排除）= **decode-only**。env
  `VLLM_FL_D13_M_FUSE_MAX` 可调。
- **env 门**：`VLLM_FL_D13_NORM_QUANT`（默认 1；=0 模块图零改动一键回退）；
  `VLLM_FL_FLAGOS_BLACKLIST=fused_add_rms_norm_quant_int8`（现有机制，kernel
  回退 reference 两步，用于隔离数值疑虑）。

## 踩坑记录（后续 agent 会撞到）

1. **dynamo fullgraph 禁 `getattr(mod, name, default)`**：缺属性时内联
   `nn.Module.__getattr__` 抛 AttributeError（EngineCore 启动失败实证）。
   解法：`LinearBase/ParallelLMHead` 类级默认 `_fl_expects_prequant=False`，
   apply_weights 直读；配对实例装载期覆盖 True。
2. **EngineCore 里裸 `vllm_fl.*` logger 的 INFO 被吞**：挂 `vllm.` 命名空间
   （`logging.getLogger("vllm.d13_norm_quant")`）才上控制台。
3. **本机部署是 site-packages 拷贝而非 editable**：源码改动必须 cp 同步
   `/opt/conda/lib/python3.12/site-packages/vllm_fl/`（env_audit 的一致性检查
   即为此）。不同步 = 单测绿（源码树导入）serve 无感（拷贝导入）的假阳性。
4. reference torch 保底初版漏乘 norm weight（收参未用）——已修，逃生门路径
   数值才正确。

## 验证证据

| 项 | 结果 |
|---|---|
| 单测（tests/unit_tests/quantization/test_d13_pair.py） | 6/6 PASS（配对/layer0 跳过/幂等/env 门/M 边界/bitwise） |
| T0 接线对拍（scripts/d13/t0_pair_wiring.py） | 全链 GEMM 输出/侧信道 scale/residual **bitwise PASS**；graph 口径融合 1.148-1.152×（M=64 K2048 三层全链，vs 同数学 norm-only triton + 现役 bare quant = serve inductor 执行形态；native C++ fused_add 与 triton 本征 6.6% 后端差不作位一致参照） |
| serve 烟测（frozen 参数原文） | 启动日志 `D13 fused_add_rms_norm+quant: paired 83 sites (m_fuse_max=512, buf_rows=2048)`；cudagraph capture 正常（2 段）、零 dynamo 报错；贪心 96 token 与 `VLLM_FL_D13_NORM_QUANT=0` 臂**逐字符一致** |
| dispatch 路由 | OpManager 13 ops；`fused_add_rms_norm_quant_int8 → default.flagos_bare` |

## PROVENANCE

- 机器：沐曦 C500 64GB（本机 B 机），MACA 3.7.0.38，torch 2.8.0+metax3.7.0.7
- 计时：scripts/d2/bottleneck_probe.py::graph_time_us（CUDA graph 3×5 replay 中位）
- GPU 独占声明：单测/对拍/烟测期间 mx-smi 空闲、无并发测量
- serve 命令：`VLLM_PLUGINS=fl vllm serve /workspace/MiniCPM5-2B-w8a8-v2 --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`（frozen，A/B 仅 env 翻转）
- 待补：T1 quick 双形状 [[4096,1024,64,32]]/[[4096,1024,64,64]] ≥3 轮中位 ≥+0.3%（B3）

## T1 quick A/B 结果（2026-11-20 13:3x-14:0x，双形状 4 会话新鲜 serve）

| case | fused 热三轮 | base 热三轮 | fused 中位 | base 中位 | Δ | 门 |
|---|---|---|---|---|---|---|
| [[4096,1024,64,32]] | 1195.60/1206.07/1205.16 | 1206.38/1208.50/1208.39 | 1205.16 | 1208.39 | **-0.267%** | FAIL |
| [[4096,1024,64,64]] | 1422.70/1421.72/1410.06 | 1422.68/1422.46/1423.37 | 1421.72 | 1422.68 | **-0.067%** | FAIL |

判据 = Output token throughput ≥3 热轮中位 ≥+0.3%。**两形状均未过门（噪声带内
微负）→ 按 §7「单 <0.3% 项不合入」，不合入 delivery**。fused 臂 serve 日志均
含 `paired 83 sites`、base 臂均无（env 门生效，A/B 有效）。

### 归因（诚实记录）

- kernel 级收益真实：bitwise + graph 口径 1.15×（T0 接线版）。但 e2e 预估
  本就只有 +0.4-0.6%（83 站 × ~2-6µs / ITL 26ms），而本 quick 协议轮间散布
  ±0.5-0.9%（fused 轮内 1195.6→1206.1 = 0.9%）——效应量 ≤ 噪声分辨率。
- 稀释因素：chunked prefill 使大部分步为 mixed 批（M 达 2048 > m_fuse_max
  → 走两步链），仅纯 decode 步吃到融合；且 case 墙钟 prefill 占比高。
- 结论：**kernel 与接线工程均成立（bitwise/graph 安全/83 站/两级回退门），
  e2e 收益在当前 bench 协议下不可分辨**。归档留档，A 机裁定是否保留基础设施
  （默认关合入 or 纯留档）；lm_head 联动 gather、m_fuse_max 扩档等后续假设
  价值相应调低。
