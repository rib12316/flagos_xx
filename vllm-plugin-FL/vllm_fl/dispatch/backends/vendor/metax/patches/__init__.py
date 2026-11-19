from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False

# --------------------------------------------------
# FL ⑥ split-KV decode hop（env 门控默认关：VLLM_FL_ATTN_SPLITKV=1 启用）。
# 装载在全部内核 patch 之后：wrapper 链以 flash_attn_with_kvcache 为最内层。
try:
    from .. import attn_splitkv_hop
    attn_splitkv_hop.install()
except Exception as _e:  # pragma: no cover - 永不阻塞 metax 后端装载
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "attn_splitkv_hop install skipped: %s", _e)

# --------------------------------------------------
# FL D5 采样 top-p torch 快路径（默认开，VLLM_FL_SAMPLE_FAST=0 关）：
# top-k(1024)+logsumexp 分母+阈值法，替换 vendor 的全词表 sort fallback
# （bs=64 实测 9.66ms/步 → 目标 ~1.5ms；Triton 路线 ATU 判死，见
#  results/20261120_c500_reprofiling/sampling_metax.csv）。装载于
# topk_topp_sampler patch 之后，链：快路径 → vendor no-triton → pytorch。
try:
    from .. import sample_fast_path
    sample_fast_path.install()
except Exception as _e:  # pragma: no cover - 永不阻塞 metax 后端装载
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "sample_fast_path install skipped: %s", _e)
