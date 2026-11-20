# A-H4 · fused_add_rms_norm IR 优先级接线（vllm_c / mcoplib C 核 vs native triton）

分支 `feat/ah4-irnorm-vllmc`（自 `delivery/s2-final` 切出）。改动两处：
`vllm-plugin-FL/vllm_fl/platform.py`（override `get_default_ir_op_priority`）+
本目录 T0 脚本。`git diff delivery/s2-final..feat/ah4-irnorm-vllmc` 只含本假设。

## 1. 设计

serve 编译态 norm 路径（文件级核实）：

```
RMSNorm.forward_cuda → forward_native
  → ir.ops.fused_add_rms_norm.maybe_inplace(x, residual, weight.data, eps, None)
     ├─ impl "native"  = fp32 分解（vllm/ir/ops/layernorm.py:39-58）
     │   → VllmIRLoweringPass 内联进 inductor 图 → triton_red_fused_*_0/2
     │     （serve ~0.60ms/步，43134 次/1027 步）
     └─ impl "vllm_c"  = torch.ops._C.fused_add_rms_norm（vllm/kernels/vllm_c.py:51，
         mcoplib C++，双 in-place）
默认 priority=["native"]（vllm/platforms/interface.py:1143，基类实现），
插件 PlatformFL 继承基类从未 override → vllm_c 被排除（排除根因）。

修复：PlatformFL.get_default_ir_op_priority 返回
  IrOpPriorityConfig.with_default(["native"], fused_add_rms_norm=["vllm_c","native"])
——只翻 fused_add_rms_norm 一项；rms_norm（final-norm 用）不动。
```

上游先例（签名照抄）：`platforms/cuda.py:684`、`platforms/rocm.py:1000`（ROCm 在
compile+cudagraph 下用 aiter vendor impl，即"inplace vendor impl 走 lowering pass"
的上游实证）、`platforms/xpu.py:341`。

环境门：`VLLM_FL_IR_NORM`，默认开；`=0/false/off` 或 vendor≠metax 时完全不
override（=上游默认 native）。env 门默认值/注释风格抄 `VLLM_FL_MCTLASS_W8A8` 先例。
装载期静态决定（`VllmConfig.__post_init__ → KernelConfig.set_platform_defaults →
本方法`），无运行时分支，cudagraph 安全。

## 2. 关键核实结论（离线已完成，勿重查）

| 项 | 结论 |
|---|---|
| weight+1.0 约定 | 属 GemmaRMSNorm（fp32 weight）；vllm_c guard `weight.dtype==x.dtype` 自动排除 Gemma。本模型 Llama → RMSNorm raw bf16 weight，无 +1 风险 |
| guard `rms_add_no_var_size` | 本模型成立：variance_size=None（Llama 不传 var_hidden_size），weight bf16 == x bf16 |
| IrOpPriorityConfig per-op | 支持（字段=rms_norm/fused_add_rms_norm 各自独立 list，vllm/config/kernel.py:31-35） |
| collateral ops | 无：vllm_c.py 全文只注册 rms_norm/fused_add_rms_norm 两个 ir op 的 impl，且本 override 只翻 fused_add 一项 |
| mcoplib Meta/fake | `_C::fused_add_rms_norm` 无 Meta 注册，但双 in-place 无返回值 schema 在 FakeTensorMode 下 fallthrough 可跑（已 CPU 实测 OK）→ VllmIRLoweringPass.replace_by_example 追踪不炸，**插件无需补 fake 注册** |
| inplace impl 的 clone 税 | lowering 用 func_impl_fn（功能性 clone）→ 紧随其后的 UnsafeCloneEliminationPass 消除（pass_manager.py:197-199，上游为此设计） |
| 编译缓存 | priority 进 VllmIRLoweringPass.uuid → 切换后缓存自动失效，无需手动清 |

## 3. T0 脚本口径声明（t0_norm_ab.py）

- ArmA = serve 现役：`torch.compile(ir.ops.fused_add_rms_norm.impls["native"].impl_fn)`
  ——**serve lowering pass 内联的同一函数对象**（非手写近似）。离线未复现整套
  vllm-compile PostGradPassManager（需构造完整 VllmConfig，繁琐）；口径差异：serve 内
  norm 核可与邻算子融合（trace 有 _0/_2 两变体），离线为孤立核——对"ArmB 须快 ≥10%"
  判活门是**保守方向**（serve ArmA 融合更强、更难赢）。
- ArmB = `torch.ops._C.fused_add_rms_norm(x, residual, w, eps)`，双 in-place。
- 计时：`bench()` 逐字抄 `feat/d9-quant-metax:scripts/d5/d9_quant_probe_metax.py`
  （L2-warm 同一张量反复跑 + CUDA graph 捕获 reps=10 后 replay + 50 iter 取中位）。
  ArmB 原地改写输入值不碍计时（kernel 数据无关），正确性对拍一律克隆输入单跑。
- 正确性：参考 = eager native impl（即 serve native 数学，fp32 内部）；容差 = vllm ir
  DEFAULT_TOLERANCES bf16（atol 1e-3 / rtol 1.6e-2；layernorm 的 fp16 override 不适用）。
  报 max_abs/max_rel/超容差元素数，输出+residual 双口都查。
- 形状：M∈{32,64}（decode bs=32/64 现役）×K=2048 bf16，+M=2048 prefill 记录档。
  hidden/eps 运行时读 `/workspace/MiniCPM5-2B-w8a8-v2/config.json`（2048/1e-6）。

## 4. 上机申请（主会话 GPU 排队执行；cwd 均为绝对路径）

```bash
# 0) 环境审计（必须 AUDIT_GREEN）
cd /data/flagos_feat_ah4 && bash scripts/d5/env_audit_metax.sh

# 1) 重装插件（platform.py 改动进 site-packages）+ provenance 查
cd /data/flagos_feat_ah4/vllm-plugin-FL && pip install --no-deps --no-build-isolation .
python3 -c "import vllm_fl.platform as p, inspect; print(inspect.getsourcefile(p))"
#   ↑ 打印路径应指向 site-packages；且
grep -n "VLLM_FL_IR_NORM" $(python3 -c "import vllm_fl.platform as p, inspect; print(inspect.getsourcefile(p))")
#   ↑ 应命中 get_default_ir_op_priority 内的门

# 2) T0 A/B（GPU ~1-2min；输出直接可读）
mkdir -p /data/flagos_feat_ah4/results/20261120_ah4
cd /data/flagos_feat_ah4 && python3 scripts/ah4/t0_norm_ab.py 2>&1 | tee results/20261120_ah4/t0_norm_ab.log

# 3)（T0 过线后）T1 quick A/B：默认臂（override 开）vs VLLM_FL_IR_NORM=0 臂（对照）
#    serve 命令参数冻结，仅 env 差一行；quick 双形状 [[4096,1024,64,32]]/[[4096,1024,64,64]]
#    ≥3 轮中位，≥+0.3% 才立项合入。serve 启动日志应出现：
#    "Final IR op priority after setting platform defaults: ... fused_add_rms_norm=['vllm_c','native']"
#    （vllm/config/kernel.py:276 INFO 行）= 接线生效的日志证据。
```

## 5. 输出判读

- **赢（判活候选）**：M∈{32,64} ArmB/ArmA 均 ≥1.10×，且正确性超容差元素数=0
  （B_out/B_res over=0；B_vs_A 允许少量超差但 vs ref 必须全绿）→ 进 T1。
- **输（负判决）**：任一 decode 档 <1.10×，或正确性超容差 → revert 分支，台账 §9
  记负结果（诚实负判决）。
- M=2048 档仅记录（D13 经验：prefill 大 M 有反例可能，如 0.894×——若 ArmB 在
  M=2048 明显劣化，T1 观察点加 4K 场景）。
- 粗投影：decode 档 speedup × 0.60ms/步 = 每步省时上限（trace 口径 norm 合计），
  除以 26ms/步 ≈ e2e 上限百分比；实际以 T1 为准。

## 6. 风险

1. **ArmA 离线口径偏乐观**（孤立核 vs serve 融合变体 _0/_2）→ T0 赢幅需留裕度，
   T1 才是裁决。
2. **C 核数值口径**：native 在 `x.to(weight.dtype)*weight` 处先降 bf16 再乘；C 核
   可能全程 fp32 再终降——bf16 尾数下应落在容差内（对拍验证），若超差先查此处。
3. **clone 消除依赖 donated_input_ids**（UnsafeCloneEliminationPass）——serve 真图
   若消除不完全，ArmB 图内多 clone 会吃掉收益（T1 观察点：trace 中 clone/copy 核
   数量 vs 对照）。
4. **rms_norm 未翻**：final-norm 仍走 native（1 次/步，预期影响 <0.01ms/步，忽略；
   若 T0 顺带发现非融合档大差异可另立分支）。
5. mcoplib C 核对非连续/高维输入的行为：vllm_c impl 的 ROCm 分支做了 reshape 兜底，
   metax（IS_ROCM=False）直连——serve 内 RMSNorm 输入为 2D 连续 [M,2048]，无此风险；
   但若有其它模型 3D 调用需另行核实。
