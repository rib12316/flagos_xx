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
