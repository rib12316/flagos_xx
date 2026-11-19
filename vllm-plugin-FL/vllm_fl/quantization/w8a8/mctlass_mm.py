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
"""MetaX C500 W8A8 GEMM via the image-shipped ``mctlassEx.ScaledGEMM`` op.

为何存在（C500 实测，2026-11-19 /data/flagos_xx 沐曦实例）：
----------------------------------------------------------------------------
``torch._int_mm``（cuBLAS int8）与 FlagGems metax scaled_mm（triton）在本卡
全热点形态输给 bf16 matmul（0.14-0.91×），天数侧的 W8A8 加速链在沐曦失效。
镜像预装的 ``mctlassEx``（MetaX CUTLASS 移植）``w8a8_scaled_mm_azp`` 为
int8×int8→bf16、per-token scale_a × per-channel scale_b、对称量化（azp=None），
与 packed W8A8 动态方案的算子契约同构。实测（bf16 matmul 为 1×）：

    prefill M1024/M2048 (K2048→N2560): 1.52× / 1.68×
    lm_head M30/M64     (K2048→N130560): 3.42× / 2.62×
    decode  M30/M64:    0.79× / 0.68×（小 M 亏损由 prefill/lm_head 大赢找平）

M∈[1,2049] 数值全对（maxrel ~3e-3，int8 噪声量级）；cudagraph 捕获+replay 安全。

布局契约（沐曦专用）：``FLW8A8DynamicLinearKernel.process_weights_after_loading``
在本 impl 生效时**跳过 [N,K]→[K,N] 转置**——mctlass 消费 B=[N,K] row-major，
unpack 天然产出该布局，零拷贝零冗余（省 2.4GB 显存 → KV 池放大）。
dispatch op 的 ``weight`` 参数在沐曦侧因此是 [N,K]；reference impl 按权重
scale 元素数自适应两种布局。``VLLM_FL_MCTLASS_W8A8=0`` 一键回退（布局回
[K,N]，走 int_mm/flaggems 旧链）。

dynamo 契约：实调用包成 ``torch.library.custom_op``（黑盒）——mctlassEx 的
Python 包装层（stream 句柄属性等）不可被 fullgraph 跟踪，裸函数会在 AOT
编译期抛 Unsupported 并把 CachedOp 打入 slow path。
"""
from __future__ import annotations

import logging
import os

import torch
from torch.library import custom_op, register_fake

logger = logging.getLogger(__name__)

_GEMM = None
_IMPORT_PROBED = False
_IMPORT_OK = False


def _is_metax_stack() -> bool:
    try:
        return "metax" in torch.__version__.lower()
    except Exception:
        return False


def metax_mctlass_w8a8() -> bool:
    """Runtime gate: mctlass impl selected AND running on the MetaX stack."""
    if os.environ.get("VLLM_FL_MCTLASS_W8A8", "1").lower() in ("0", "false", "off"):
        return False
    return _is_metax_stack()


def mctlass_available() -> bool:
    """注册期探针（纯 import，不碰 CUDA——引擎前端进程也会走到注册）。

    真正的 warm launch 在第一次实调用时进行（vllm 捕获前必有 eager 预热）。
    """
    global _IMPORT_PROBED, _IMPORT_OK, _GEMM
    if _IMPORT_PROBED:
        return _IMPORT_OK
    _IMPORT_PROBED = True
    if not _is_metax_stack() or not torch.cuda.is_available():
        return False
    try:
        import mctlassEx

        _GEMM = mctlassEx.ScaledGEMM()  # 仅构造句柄，无 kernel launch
        _IMPORT_OK = True
        logger.info("mctlass_mm: mctlassEx ScaledGEMM 句柄就绪（C500 int8 GEMM 承载）")
    except Exception as e:  # pragma: no cover - probe failure = impl 不可选
        _IMPORT_OK = False
        logger.warning("mctlass_mm: mctlassEx 不可用，回退 int_mm/flaggems 链: %s", e)
    return _IMPORT_OK


@custom_op("vllm_fl::w8a8_mctlass_scaled_mm", mutates_args=())
def _w8a8_mctlass_scaled_mm(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """int8×int8→bf16 scaled GEMM（布局契约见模块 docstring）。"""
    global _GEMM
    if _GEMM is None:
        if not mctlass_available() or _GEMM is None:
            raise RuntimeError("mctlassEx ScaledGEMM unavailable")
    M, K = x_q.shape
    N = weight_scale.numel()
    out = torch.empty((M, N), dtype=torch.bfloat16, device=x_q.device)
    sa = x_scale.reshape(M, 1)
    if sa.dtype != torch.float32:
        sa = sa.to(torch.float32)
    if not sa.is_contiguous():
        sa = sa.contiguous()
    sb = weight_scale.reshape(N, 1)
    if sb.dtype != torch.float32:
        sb = sb.to(torch.float32).contiguous()
    if not sb.is_contiguous():
        sb = sb.contiguous()
    bias_arg = None
    if bias is not None:
        bias_arg = bias.reshape(-1, 1)
        if bias_arg.dtype != torch.bfloat16:
            bias_arg = bias_arg.to(torch.bfloat16).contiguous()
        if not bias_arg.is_contiguous():
            bias_arg = bias_arg.contiguous()
    _GEMM([M, N, K], x_q, weight, out, sa, sb, bias_arg)
    return out


@_w8a8_mctlass_scaled_mm.register_fake
def _(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    return torch.empty(
        (x_q.shape[0], weight_scale.numel()),
        dtype=torch.bfloat16,
        device=x_q.device,
    )


def int8_scaled_mm_mctlass(
    x_q: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """W8A8 dynamic-token GEMM on mctlassEx（[N,K] 布局契约见模块 docstring）。

    签名与 ``vllm_fl.quantization.w8a8.int_mm.int8_scaled_mm_int_mm`` 一致；
    差异仅 weight 为 [N,K] row-major int8（process_weights_after_loading
    沐曦分支保证）。
    """
    M, K = x_q.shape
    N = weight_scale.numel()
    if weight.shape != (N, K):
        raise ValueError(
            f"mctlass_mm expects weight [N,K]=[{N},{K}] (MetaX layout), "
            f"got {tuple(weight.shape)}"
        )
    if not x_q.is_contiguous():
        x_q = x_q.contiguous()
    out = _w8a8_mctlass_scaled_mm(x_q, weight, x_scale, weight_scale, bias)
    if out_dtype is None or out_dtype == torch.bfloat16:
        return out
    return out.to(out_dtype)


__all__ = [
    "int8_scaled_mm_mctlass",
    "mctlass_available",
    "metax_mctlass_w8a8",
]
