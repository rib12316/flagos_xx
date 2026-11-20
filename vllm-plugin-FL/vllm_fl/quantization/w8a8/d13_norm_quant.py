# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""D13 fused_add_rms_norm + per-token-int8-quant 的调用点接线（§3.5 装载期
静态配对）——开发单《交付-新agent-C500算子工程开发单》§0 五件套第 3 件。

接线设计（§3.5，用户/A 机裁定方案的代码化）
----------------------------------------------------------------------------
- 模型加载完成后（ModelRunnerFL.load_model，weights+process_weights 之后、
  compile/GraphWrapper 包裹之前）遍历模块图一次，静态配对
  ``*.input_layernorm → 同层 self_attn.qkv_proj``、
  ``*.post_attention_layernorm → 同层 mlp.gate_up_proj``。
  本栈 MiniCPM5-2B = LlamaForCausalLM（42 层，hidden 2048）：norm 输出经核对
  仅喂对应 proj（llama.py attention/MLP 直连，中间零算子）→ 融合安全。
  layer0 的 input_layernorm residual=None（无 add 可融）→ 跳过，共 83 站。
  ``model.norm → lm_head`` 不配对（v1 降级，用户裁定）：runner 在两者之间做
  ``hidden_states[logits_indices]`` 切片（M'<M，scale 无法逐行联动 gather，
  需 patch runner 5 处切片 site），而该站收益 ≈ 一次 µs 级 launch——收益/工程
  量严重不成比例，列后续独立假设。
- 数据流契约：配对 norm 的 ``_forward_method``（CustomOp 的实例级调用入口）
  换为融合版，**q_int8 作为返回值走模型张量流**（模型代码
  ``x, residual = norm(...)`` 解包后直喂 proj，dtype 变 int8、形状不变，中间
  零算子不受影响）；**scale 走侧信道**——配对 norm 上的非持久 buffer
  ``_fl_d13_scale``（融合 kernel 经 scale_out 直写），配对 linear 经装载期
  回指 ``_fl_prequant_norm`` 读取 ``[:M]``。buffer 身份跨 forward 恒定 →
  dynamo guard 稳定、cudagraph 静态指针传递（replay 期不执行 python）。
  （tuple 返回值/tensor 附属性案已否决：O2 默认 VLLM_COMPILE fullgraph 下
  脆弱。）
- per-shape 路由（decode-only）：forward 内唯一数据流分支
  ``M > _fl_d13_m_max → 原两步链``。分支只依赖 x.shape[0]——每个 cudagraph
  按固定 padded M 捕获，capture/replay 同支；dynamo 对其生成 shape guard，
  每个编译产物对应固定一支。「无运行时分支」的准确含义 = 无 per-call 策略
  查找/env 读取；shape-static 分支正是 graph 安全形态。大 M 档（T0 实测
  M4096 K6144 = 0.894×，单 BLOCK 寄存器压力）回落两步链。
- env 门（两级）：
  * ``VLLM_FL_D13_NORM_QUANT``（默认 "1"）=0 时本模块零改动模块图，完整
    回退 stock 两步链——一键回退；
  * ``VLLM_FL_FLAGOS_BLACKLIST=fused_add_rms_norm_quant_int8``（现有机制，
    utils.use_flaggems_op）：配对仍装、kernel 回退 reference torch 两步
    （位相同）——用于隔离「kernel 数值疑虑」vs「接线问题」。
- ``VLLM_FL_D13_M_FUSE_MAX`` 覆盖融合 M 上限；默认取
  ``compilation_config.max_cudagraph_capture_size``（= min(max_num_seqs*2,
  512)，compilation.py 默认公式）——恰覆盖 decode 全部 padded M 档，排除
  prefill/mixed（M 达 max_num_batched_tokens）。

位精确声明：融合路径 GEMM 输入 (q, scale) 与 stock 两步链 bitwise 一致
（T0：scripts/d13/norm_quant_fuse_t0.py；接线版对拍 scripts/d13/
t0_pair_wiring.py）→ GEMM 输出 bitwise 一致。
"""

from __future__ import annotations

import functools
import logging

import torch

from vllm_fl.dispatch import CachedOp
from vllm_fl.utils import use_flaggems_op

# logger 挂 vllm 命名空间：EngineCore 只路由 vllm.* 的 INFO 到控制台
# （裸 vllm_fl.* 模块 logger 会被吞——烟测实证；dispatch 自带 handler 除外）。
logger = logging.getLogger("vllm.d13_norm_quant")

_OP_NAME = "fused_add_rms_norm_quant_int8"
_fused_op = CachedOp(_OP_NAME)


@torch.library.custom_op("vllm_fl::d13_norm_quant", mutates_args=("scale_out",))
def _compiled_d13(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor,
                  epsilon: float, scale_out: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q, _, resid = _fused_op(x, residual, weight, epsilon, scale_out=scale_out)
    return q, resid


@_compiled_d13.register_fake
def _fake_compiled_d13(x, residual, weight, epsilon, scale_out):
    return torch.empty_like(x, dtype=torch.int8), torch.empty_like(residual)

# 配对规则：(decoder 层上的 norm 属性, 消费者属性链, 是否跳过 layer0)。
# layer0 的 input_layernorm 前无 residual（llama.py 首 layer 残差链未起），
# 无 add 可融。逐模型核对勿默认——本表仅对 _SUPPORTED 里的 decoder 类成立。
_PAIR_RULES = (
    ("input_layernorm", ("self_attn", "qkv_proj"), True),
    ("post_attention_layernorm", ("mlp", "gate_up_proj"), False),
)

# 允许配对的 decoder 层类名 allowlist（逐模型核对后固化；新模型须先核对
# norm 输出单消费者再入表）。MiniCPM5-2B ckpt architectures=LlamaForCausalLM。
_SUPPORTED_DECODER_LAYERS = {"LlamaDecoderLayer"}


def _env_enabled() -> bool:
    from vllm_fl.envs import fusion_enabled

    return fusion_enabled("VLLM_FL_D13_NORM_QUANT")


def _default_m_fuse_max(vllm_config) -> int:
    import os

    raw = os.environ.get("VLLM_FL_D13_M_FUSE_MAX", "")
    if raw:
        try:
            v = int(raw)
            if v > 0:
                return v
        except ValueError:
            logger.warning(
                "VLLM_FL_D13_M_FUSE_MAX=%r 非法，忽略（用默认公式）", raw)
    try:
        cs = getattr(
            vllm_config.compilation_config, "max_cudagraph_capture_size", None)
        if isinstance(cs, int) and cs > 0:
            return cs
    except Exception:  # pragma: no cover - config shape varies in tests
        pass
    try:
        return min(vllm_config.scheduler_config.max_num_seqs * 2, 512)
    except Exception:  # pragma: no cover
        return 256


def _pairable(norm, consumer) -> bool:
    """结构校验：RMSNorm 可融 + 消费者是本栈 W8A8 per-token linear。"""
    from vllm.model_executor.layers.layernorm import RMSNorm

    if not isinstance(norm, RMSNorm):
        return False
    # ir.ops.fused_add_rms_norm 的 weight 可选语义：has_weight=False 时上游
    # 传 None（无 per-channel 乘法），本融合 kernel 恒乘 w——只配 pass 的。
    if not getattr(norm, "pass_weight_add", False):
        return False
    # variance_size_override（分块方差）本 kernel 不支持。
    if getattr(norm, "variance_size_override", None) is not None:
        return False
    if getattr(norm, "_fl_d13_paired", False):
        return False  # 幂等
    if consumer is None or getattr(consumer, "quant_method", None) is None:
        return False
    # 只配 per-token 动态 int8 packed scheme 的 linear（与 w8a8/linear.py
    # apply_weights 的 quant 点同源）。duck-typing：process_weights_after_
    # loading 已为 packed W8A8 置 int8 weight + fp32 weight_scale。
    w = getattr(consumer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dtype != torch.int8:
        return False
    if not isinstance(getattr(consumer, "weight_scale", None), torch.Tensor):
        return False
    in_size = getattr(consumer, "input_size", None)
    if isinstance(in_size, int) and in_size != norm.weight.shape[0]:
        return False
    return True


def _install_pair(norm, consumer, m_fuse_max: int, buf_rows: int) -> None:
    norm.register_buffer(
        "_fl_d13_scale",
        torch.zeros(buf_rows, 1, dtype=torch.float32,
                    device=norm.weight.device),
        persistent=False,
    )
    norm._fl_d13_orig = norm._forward_method
    norm._fl_d13_m_max = m_fuse_max
    norm._forward_method = functools.partial(_d13_forward, norm)
    norm._fl_d13_paired = True
    consumer._fl_expects_prequant = True
    consumer._fl_prequant_norm = norm


def _d13_forward(norm, x, residual=None):
    """配对 norm 的融合入口。residual=None 与 M>m_fuse_max 走原两步链。"""
    if residual is None or x.shape[0] > norm._fl_d13_m_max:
        return norm._fl_d13_orig(x, residual)
    q, resid = _compiled_d13(
        x, residual, norm.weight.data, norm.variance_epsilon, norm._fl_d13_scale)
    # 契约同 fused_add_rms_norm：(out, residual)；out 为 int8 预量化激活，
    # 由配对 linear 的 _fl_expects_prequant 分支消费（w8a8/linear.py）。
    return q, resid


def install_d13_pairings(model: torch.nn.Module, vllm_config) -> int:
    """装载期静态配对入口（ModelRunnerFL.load_model 调用）。

    env 门关闭或 dispatch 层拉黑时零改动返回；返回配对站数（日志
    ``D13 fused_add_rms_norm+quant: paired N sites`` 供烟测 grep）。
    """
    if not _env_enabled():
        logger.info("D13 fused_add_rms_norm+quant: disabled by env, skip")
        return 0
    if not use_flaggems_op(_OP_NAME):
        logger.info(
            "D13 fused_add_rms_norm+quant: op blacklisted, skip pairing")
        return 0
    # AOT fullgraph capture 前锁外预解析（先例 w8a8/linear.py:147）：
    # 首次 resolve 在 capture 内会进 dynamo 无法 trace 的锁。
    try:
        _fused_op.pre_resolve()
    except Exception:  # pragma: no cover - never block model load
        logger.exception("D13 pre_resolve failed, pairing anyway")

    m_fuse_max = _default_m_fuse_max(vllm_config)
    try:
        buf_rows = max(
            vllm_config.scheduler_config.max_num_batched_tokens, m_fuse_max)
    except Exception:  # pragma: no cover - config shape varies in tests
        buf_rows = m_fuse_max

    paired = 0
    for name, layer in model.named_modules():
        if type(layer).__name__ not in _SUPPORTED_DECODER_LAYERS:
            continue
        try:
            layer_idx = int(name.rsplit(".", 1)[-1])
        except ValueError:
            continue
        for norm_attr, consumer_path, skip_layer0 in _PAIR_RULES:
            if skip_layer0 and layer_idx == 0:
                continue
            norm = getattr(layer, norm_attr, None)
            head = getattr(layer, consumer_path[0], None)
            consumer = getattr(head, consumer_path[1], None) if head else None
            if not _pairable(norm, consumer):
                continue
            _install_pair(norm, consumer, m_fuse_max, buf_rows)
            paired += 1
    logger.info(
        "D13 fused_add_rms_norm+quant: paired %d sites (m_fuse_max=%d, "
        "buf_rows=%d)", paired, m_fuse_max, buf_rows)
    return paired
