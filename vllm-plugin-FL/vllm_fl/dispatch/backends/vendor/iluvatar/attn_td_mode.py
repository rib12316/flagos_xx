# Copyright (c) 2026 BAAI. All rights reserved.
"""Force tensor-descriptor (USE_TD) loads in vLLM's unified attention on Iluvatar.

机理（BI-V150 实测，2026-11-15 results/20261115_bottleneck_deepdive）
----------------------------------------------------------------------------
vllm 0.24 ``unified_attention`` 的 ``use_td`` 开关让 Q/K/V 加载与输出写走
``tl.make_tensor_descriptor``（上游 PR #40327，为 Intel Xe2/Xe3 的 HW 2D 块读
而写），默认仅 XPU 平台自动开启（``triton_attn.py`` __init__ 三态决策：
env VLLM_TRITON_ATTN_USE_TD 未设时 ``is_xpu()`` 为准）。Iluvatar flagtree
后端实测可编译可跑且更快（graph 口径，attn_surface_probe td_ab/tdx_ab）：

    臂                                td0        td1       加速
    decode 3D  bs64/kv2k            479.5µs    400.1µs    1.20x
    decode 3D  bs16/kv8k            504.4µs    386.2µs    1.31x
    decode 2D  bs128/kv16k         9995.8µs   6460.1µs    1.55x
    prefill 2D S1q4096kv4k        13654.5µs  12419.5µs    1.10x
    TD+段数n8  bs64/kv8k           2197.8µs   1392.6µs    1.58x

约束：wrapper 对 ``use_td_qo`` 有 pow2/head_size/stride 硬 assert（MiniCPM5
16Q:2KV、head 128、连续布局全满足）；USE_TD 要求 BLOCK_SIZE%TILE==0，wrapper
会把 TILE 钳到 block_size（16）。**非 bit-exact**（加载/累加序变化，探针实测
max_abs_diff 0.12-0.31，softmax 数学等价）——上线须过对拍/精度门。

运行时集成
----------------------------------------------------------------------------
与 attn_launch_config 同为 dispatch 层一次性装载（幂等、try 保护、不替换类
身份）：包装 ``TritonAttentionImpl.__init__`` 事后强制 ``self.use_td = True``
（唯一决策点在 __init__ 末尾；唯一读取点在 forward 传参）。diffkv 子类
``TritonAttentionDiffKVImpl`` 未重写 __init__，经 MRO 自动继承。
``VLLM_FL_ATTN_TD=0`` 关闭恢复上游行为（A/B 臂）。
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_installed = False


def install() -> bool:
    """强制 unified_attention use_td=True（幂等）。

    默认**关闭**（2026-11-15 烟测判决）：decode 代理 ITL 中位 −6.4%（收益真实
    但小幅），但 16 条 greedy 对拍 0/16 一致——42 层数值漂移复合后早期 argmax
    全翻，精度风险未过对拍 sanity。VLLM_FL_ATTN_TD=1 开启（终局 105 题门后再
    定默认值）。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_ATTN_TD", "0") != "1":
        logger.info("attn_td_mode: VLLM_FL_ATTN_TD=0，保持上游 use_td 决策")
        return False
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
    except Exception as e:
        logger.warning("attn_td_mode: triton_attn 不可用: %s", e)
        return False

    impl = getattr(_ta, "TritonAttentionImpl", None)
    if impl is None or getattr(impl, "_fl_td_patched", False):
        _installed = True
        return True  # 已装载（幂等）

    orig_init = impl.__init__

    def _td_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        # 覆盖 __init__ 末尾的三态决策（is_xpu()/env），其余属性不动。
        self.use_td = True

    impl.__init__ = _td_init
    impl._fl_td_patched = True
    impl._fl_td_orig_init = orig_init  # 供还原臂
    _installed = True
    logger.info(
        "attn_td_mode: 已强制 unified_attention use_td=True（TD 加载路径，"
        "VLLM_FL_ATTN_TD=0 可关闭）")
    return True
