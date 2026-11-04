# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

"""SchedulerFL：D1 调度线诊断骨架（行为不变版）。

注入方式：platform.py::check_and_update_config 在 VllmConfig.__post_init__ 阶段
（早于 EngineCore get_scheduler_cls()）设 scheduler_config.scheduler_cls 指向本类，
env 门控 VLLM_FL_SCHED_FL=1，默认关闭 = 零行为变化。

设计约束：
- 不覆写 schedule()（vllm 0.24 该方法是 ~700 行单体，fork 即埋雷）；
- 仅覆写 _update_after_schedule()（scheduler.py L1130，schedule() 尾部 L1099 调用）
  做只读采集 + 日志，随后原样调用 super()——调度决策路径一字不动；
- 采集必须在 super() 之前：super() 会推进 num_computed_tokens，之后
  prefill/decode 归类会错一步。

日志行格式（E2 证据 + H3' 指纹分析数据源）：
  [fl.sched] step= run= wait= free_blk= new= fin= pre_tok= dec_tok= nreq= preempt= cum= usage=
打印门控：事件触发（preempt>0 / new>0 / fin>0）或 VLLM_FL_SCHED_LOG_SEC 秒节流（默认 5）。
"""

import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class SchedulerFL(Scheduler):
    """诊断版调度器：行为与上游 Scheduler 完全一致，仅增加调度态日志。"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._fl_step = 0
        self._fl_cum_preempt = 0
        self._fl_last_log = 0.0
        try:
            self._fl_log_sec = float(os.environ.get("VLLM_FL_SCHED_LOG_SEC", "5"))
        except ValueError:
            self._fl_log_sec = 5.0
        sc = self.scheduler_config
        cc = self.cache_config
        # banner：一次性把本进程生效的调度参数定案（后续判定/归因的分母）
        logger.info(
            "[fl.sched] SchedulerFL(diag) banner: mnbt=%s mnst_cfg=%s mnst_eff=%s "
            "seqs=%s wm=%s lpt=%s mnp=%s rsv_full_isl=%s apc=%s bs=%s blocks=%s",
            sc.max_num_batched_tokens,
            sc.max_num_scheduled_tokens,
            self.max_num_scheduled_tokens,
            sc.max_num_seqs,
            sc.watermark,
            sc.long_prefill_token_threshold,
            sc.max_num_partial_prefills,
            sc.scheduler_reserve_full_isl,
            cc.enable_prefix_caching,
            self.block_size,
            self.kv_cache_config.num_blocks,
        )

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        so = scheduler_output
        # 只读采集（在 super() 推进 num_computed_tokens 之前）
        preempt = len(so.preempted_req_ids) if so.preempted_req_ids else 0
        new = len(so.scheduled_new_reqs)
        fin = len(so.finished_req_ids)
        pre_tok = 0
        dec_tok = 0
        for rid, n in so.num_scheduled_tokens.items():
            req = self.requests.get(rid)
            if req is None:
                continue
            if req.num_computed_tokens < req.num_tokens:
                pre_tok += n
            else:
                dec_tok += n
        now = time.monotonic()
        if preempt or new or fin or (now - self._fl_last_log) >= self._fl_log_sec:
            logger.info(
                "[fl.sched] step=%d run=%d wait=%d free_blk=%d new=%d fin=%d "
                "pre_tok=%d dec_tok=%d nreq=%d preempt=%d cum=%d usage=%.3f",
                self._fl_step,
                len(self.running),
                len(self.waiting),
                self.kv_cache_manager.block_pool.get_num_free_blocks(),
                new,
                fin,
                pre_tok,
                dec_tok,
                len(so.num_scheduled_tokens),
                preempt,
                self._fl_cum_preempt + preempt,
                self.kv_cache_manager.usage,
            )
            self._fl_last_log = now
        self._fl_cum_preempt += preempt
        self._fl_step += 1
        return super()._update_after_schedule(so)
