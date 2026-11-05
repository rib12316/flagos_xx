# Copyright (c) 2025 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project

"""SchedulerFL：D1 调度线诊断骨架（行为不变版）。

注入方式：platform.py::check_and_update_config 在 VllmConfig.__post_init__ 阶段
（早于 EngineCore get_scheduler_cls()）设 scheduler_config.scheduler_cls 指向本类，
env 门控 VLLM_FL_SCHED_FL=1，默认关闭 = 零行为变化。

基类选择（关键，2026-11-04 裁决轮首跑踩坑后修正）：
- 本 vllm 0.24 构建 async_scheduling 默认 None → VllmConfig.__post_init__
  （config/vllm.py L986-1002）对生成模型**自动置 True**，默认调度器是
  AsyncScheduler 而非 sync Scheduler；
- AsyncScheduler._update_after_schedule 在 super() 之外还做 async 簿记
  （num_output_placeholders / spec_token_ids / next_decode_eligible_step /
  pending_structured_output_tokens），这是 model_runner（含 vllm_fl fork）
  _update_states 异步恢复路径（all_token_ids 契约）的依赖——
  若继承 sync Scheduler 会丢掉整条簿记链，首请求即 KeyError 崩溃
  （实证：results/20261104_d1_adjudication/serve.log L185 起堆栈）；
- 故本类继承 AsyncScheduler，覆写 _update_after_schedule 时把采集放在
  最前、然后原样调用 super()（AsyncScheduler 链，内含 Scheduler 链），
  全部簿记保持不变。

设计约束：
- 不覆写 schedule()（上游该方法 ~700 行单体，fork 即埋雷）；
- 采集必须在 super() 之前：super() 会推进 num_computed_tokens，之后
  prefill/decode 归类会错一步。

日志行格式（E2 证据 + H3' 指纹分析数据源）：
  [fl.sched] step= run= wait= free_blk= new= fin= pre_tok= dec_tok= nreq= preempt= cum= usage=
打印门控：事件触发（preempt>0 / new>0 / fin>0）或 VLLM_FL_SCHED_LOG_SEC 秒节流（默认 5）。
"""

import os
import time

import logging

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput

logger = logging.getLogger(__name__)

# 自防御日志挂载：vllm 的 dictConfig 只配置 "vllm" 层级（propagate=False），
# EngineCore 子进程内 root logger 未配置（有效级别 WARNING）——vllm_fl.* 的
# INFO 会在级别检查即被丢弃（实证：stderr 探针可打而 logger.info 静默）。
# 诊断模块自挂 StreamHandler 直通 stderr（EngineCore stderr 会被父进程
# 中继进 serve.log，实证：[fl.sched.probe] 行带 (EngineCore pid=) 前缀）。
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class SchedulerFL(AsyncScheduler):
    """诊断版调度器：行为与上游 AsyncScheduler 完全一致，仅增加调度态日志。"""

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
            "[fl.sched] SchedulerFL(diag) banner: base=AsyncScheduler mnbt=%s "
            "mnst_cfg=%s mnst_eff=%s seqs=%s wm=%s lpt=%s mnp=%s "
            "rsv_full_isl=%s apc=%s bs=%s blocks=%s async_sched=%s",
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
            sc.async_scheduling,
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
        # 原样保留 AsyncScheduler 簿记链（内含 Scheduler 的推进逻辑）
        return super()._update_after_schedule(so)
