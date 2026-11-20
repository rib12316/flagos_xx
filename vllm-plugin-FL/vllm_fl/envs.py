# Copyright (c) 2025 BAAI. All rights reserved.

import os
from typing import Any, Callable

from vllm_fl.utils import use_flaggems

fl_vllm_environment_variables: dict[str, Callable[[], Any]] = {
    "VLLM_FL_PREFER_ENABLED": lambda: (
        os.environ.get("VLLM_FL_PREFER_ENABLED", "True").lower() in ("true", "1")
    ),
    "FLAGGEMS_ENABLE_OPLIST_PATH": lambda: os.environ.get(
        "FLAGGEMS_ENABLE_OPLIST_PATH", "/tmp/flaggems_enable_oplist.txt"
    ),
    "USE_FLAGGEMS": use_flaggems,
    # ---- D1 调度线（env 门控，默认全部关闭 = 零行为变化）----
    # 诊断版调度器注入（行为不变，仅 [fl.sched] 日志）
    "VLLM_FL_SCHED_FL": lambda: os.environ.get("VLLM_FL_SCHED_FL", "0") == "1",
    # 诊断日志节流秒数
    "VLLM_FL_SCHED_LOG_SEC": lambda: os.environ.get("VLLM_FL_SCHED_LOG_SEC", "5"),
    # KV cache block size 覆盖（H4 扫描 {16,32,64}；未设 = 跟随上游按平台默认）
    "VLLM_FL_BLOCK_SIZE": lambda: os.environ.get("VLLM_FL_BLOCK_SIZE", ""),
    # prefix caching 强制开关（H6：未设 = 跟随上游默认开；"0"/"1" 强制）
    "VLLM_FL_APC": lambda: os.environ.get("VLLM_FL_APC", ""),
    # max_num_batched_tokens 覆盖（混批节奏第一级杠杆）
    "VLLM_FL_MAX_BATCHED_TOKENS": lambda: os.environ.get(
        "VLLM_FL_MAX_BATCHED_TOKENS", ""
    ),
    # 准入 watermark 覆盖（H7/H1：消灭容量钉死期的周期性抢占）
    "VLLM_FL_WATERMARK": lambda: os.environ.get("VLLM_FL_WATERMARK", ""),
    # F-c: decode flash attention 直写 out（消除 attention 输出 staging copy；
    # trace 实测 _copy_kernel 42 次/步 ≈ 0.14ms/步）。
    # 【11-20 探测矩阵九变体判决：该构建（flash_attn_2_cuda md5 0953e2b9…）的
    # out 形状合同运行期不可满足（V1/V2/V6/V9 同串拒收 vs V7/V8 检查链活着，
    # 证据 results/20261120_fc/probe_matrix.txt）→ 默认 0 封存为休眠逃生门；
    # vendor 未来更新 .so 后翻 =1 复活重测（scripts/fc/t0_direct_out.py）】
    "VLLM_FL_FLASH_KVCACHE_DIRECT_OUT": lambda: os.environ.get(
        "VLLM_FL_FLASH_KVCACHE_DIRECT_OUT", "0"
    ).lower()
    in ("1", "true", "on"),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in fl_vllm_environment_variables:
        return fl_vllm_environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(fl_vllm_environment_variables.keys())


def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in fl_vllm_environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
