# Copyright (c) 2026 BAAI. All rights reserved.
"""Self-written split-KV decode attention kernel for Iluvatar BI-V150.

交接书 docs/splitkv-kernel-handoff.md §5/§6（设计要求/集成方式）+ §13.3（假设阶梯）
的无卡实现。上游参照 docs/splitkv-ref/vllm0.24_triton_unified_attention.py。

设计纪律（§1/§2，违反必败——前科在案）
----------------------------------------------------------------------------
- 裸 ``@triton.jit``，严禁 ``@libentry``（流竞争前科：进程内直调 b>1 读未就绪输入）
- 归约严禁 atomic：stage2 按 split 向量序合并（与上游 reduce_segments 同数学，
  确定性）；性能结论只认 CUDA graph 口径
- 严禁运行时 autotune：launch/grid/split 全部由 (bs, kv) 纯函数静态决定
  （cudagraph 相容：capture 期查表一次，graph 内行为确定）
- TILE=16 基线：TILE == BLOCK_SIZE == 16 → 每 tile 恰一整页，K/V 是物理连续
  的 [16,128] slab（上游 TD 路径同构；USE_TD 要求 BLOCK_SIZE%TILE==0 天然满足）
- w1s1 起步（num_warps=1/num_stages=1，§2：w2 双 warp 同步税 15-35%、
  s=2 在 TD 态大劣化 1748→2603µs）——V1 的流水必须**手动排指令**，绝不开
  num_stages
- smem 预算：TILE16×128×2B×2(K+V)=8KB/级；G=2 分组 16KB、G=3 24KB，远低于
  128KB/SM，留足余量（§5.3）

结构（对照上游 3D 路径的异同）
----------------------------------------------------------------------------
- 上游 grid=(total_num_q_blocks, kvh, seg)，其中 total_num_q_blocks =
  q.shape[0]//BLOCK_Q + num_seqs 是**上界**（bs32 时 48，其中 16 个死块
  early-return——§13.1 的 1536/1568 CTA 数即含死块口径）。本 kernel 用
  grid=(num_seqs, kvh, split)：decode 时 token==seq，**零死块**。
- 打包与上游一致：BLOCK_M=16 行 = q_per_kv(8) 有效 + 填充（上游 BLOCK_Q=2
  × 8 heads，decode 时每块仅 1 token × 8 head 有效；MMA 形状 [16,128]×
  [128,16] 保留 tl.dot，带宽瓶颈下多余行 flops 免费）
- 分段方案沿用上游公式（页对齐）：tiles_per_seg = cdiv(total_tiles, NSPLIT)；
  空段 early-return 不写；stage2 按 act_num_segments 掩码跳过（与上游
  reduce_segments 同法，容忍工作缓冲陈旧值，确定性）
- segm 缓冲**按 token 索引**（上游 #48076 的 per-seq 分配 + token 索引越界
  根因，本 kernel 分配 [num_tokens, NH, MAX_SPLIT, HD] 直接按 token 索引）
- 自适应分段（§5.1）：split = f(bs, max_kv) 纯函数查表；split==1 走
  WRITETHROUGH 特化直写输出、**跳过 stage2 launch**（FlashInfer D.2/D.3）
- 数值防护照抄上游：M 初值 -inf / L 初值 1.0（softmax_step 语义）、
  整段无有效 tile 时 L=1.0×exp(-inf-·)=0、stage2 expsum==0→输出 0

§13.3 假设阶梯（VARIANT constexpr，env 选择，供真机一条命令 A/B）
----------------------------------------------------------------------------
  V0 结构基线（对照臂，最接近上游但零死块+自适应分段）
  V1 手动双缓冲预取（**首攻**，默认）：tile 循环按 G 分组（G=VLLM_FL_ATTN_
     SPLITKV_GROUP，默认 2，可 3）：组内先发起全部 G 个 K/V 加载再逐 tile 消化
     ——每个 warp 在飞加载 ×G。不改 num_stages（§2 前科）。
  V2 persistent 多段常驻：每 CTA 顺序处理 SEGS_PER_CTA(默认 4) 段，
     段内用 G=2 分组加载（与 V1 同机制）；「段尾 partial 写与下一段首
     tile 加载重叠」的跨段 peel 为二期（若 V2 臂接近阈值再上）
  V3 宽 descriptor：物理相邻页(page[j+1]==page[j]+1)时单次加载 [32,128]
     （等效 TILE=32 读宽度、保持两个独立 16 宽 dot，避开 t32 的 dot 形状劣化
     机制——t32 慢因推测是 dot 形状而非读宽度，A/B 验证）；不相邻时回落
     两次 16 宽加载。注意 BI-V150 的 K cache 布局下 16*stride1 == stride0
     （NKV=2 时成立），相邻页在行 stride 视角下连续

运行时集成（§6，零 vllm 源码改动——复刻 attn_launch_config 两跳模式）
----------------------------------------------------------------------------
hop1：包装 ``unified_attention``（同时换绑 ``triton_attn`` 按名导入，参照
attn_launch_config L232-237）。仅当 **decode（max_seqlen_q≤1）且 bs≤64 且
kv 量化关闭且无 alibi/sinks/qq_bias/mm_prefix/softcap/window/swa 且 causal
且 head_size 为 2 的幂** 时路由到自研 kernel，其余原样透传。env::

    VLLM_FL_ATTN_SPLITKV         总开关（默认 0；=1 启用）
    VLLM_FL_ATTN_SPLITKV_VARIANT 0-3（默认 1=双缓冲）
    VLLM_FL_ATTN_SPLITKV_TD      TD 加载（默认 -1=跟随调用的 use_td；0/1 强制）
    VLLM_FL_ATTN_SPLITKV_SPLITS  段数覆盖（默认 0=自适应表）
    VLLM_FL_ATTN_SPLITKV_GROUP   V1 分组 G（默认 2；3 为深预取降档备选）
    VLLM_FL_ATTN_SPLITKV_DEBUG   =1 时每次路由决策打日志（probe/debug 用）

幂等哨兵 + try 保护 + 不替换对象身份；首启用打日志 ``fl.attn-splitkv``（serve
日志 grep 判据）。工作缓冲自管缓存（sample_filter_mode 模式），按需增长、跨
调用复用地址（graph 重放安全：capture 期首次分配后不再变更）。
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 常量与段数表（§3/§12.1/§13.1 实测锚点）
# ---------------------------------------------------------------------------
MAX_SPLIT = 32            # segm 工作缓冲最大段数（TD 态 kv16K 峰值在 n32）
MIN_SEG_TILES = 8         # §5.1：每段至少 8 tile（128 token）才有切分价值
VARIANT_NAMES = ("v0-plain", "v1-dbuf", "v2-persistent", "v3-wide")

_installed = False
_active = None            # 三态：None=未定(看 env)，True/False=运行时覆盖(probe)
_variant_override = None   # probe 扫变体用：None=用 install 时的 env 值
_fallback_logged = False  # 异常回落只打一次
_route_stats = {"routed": 0, "passed": 0}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        return default


def resolve_split(bs: int, max_kv: int) -> int:
    """自适应分段：split = f(bs, max_kv) 纯函数（graph 重放安全）。

    依据（graph 口径 TD=on w1s1）：
      - §12.1: bs32/kv16384 n32=1374µs vs n16=1748µs（1.27×）；bs64/kv16K n32 优
      - §13.1: kv17.4K 区 n16/n32 在 ~390GB/s 汇合；bs33 n32 反劣化 12%
      - §13.2: 短 kv 不切（FlashInfer D.3）；kv=16384 单点是波形伪影区不作为判据
      - §5.1: split==1 → writethrough 直写、跳过归约 launch
    """
    forced = _env_int("VLLM_FL_ATTN_SPLITKV_SPLITS", 0)
    if forced > 0:
        return max(1, min(forced, MAX_SPLIT))
    if max_kv <= 2048:
        split = 1
    elif max_kv <= 4096:
        split = 8
    elif max_kv <= 8192:
        split = 16
    else:
        split = 32 if bs <= 32 else 16
    cap = max(1, (max_kv + 15) // 16 // MIN_SEG_TILES)
    return max(1, min(split, cap, MAX_SPLIT))


# ---------------------------------------------------------------------------
# 工作缓冲自管缓存（sample_filter_mode 模式：按需增长，地址跨调用稳定）
# ---------------------------------------------------------------------------
_BUFFER_CACHE: dict = {}   # device_index -> {"out","max","expsum","ntok","nheads"}


def _get_buffers(num_tokens: int, num_heads: int, head_size: int, device):
    """segm 三件套（fp32，按 token 索引）。graph 重放安全：只增长不搬迁。"""
    key = device.index if device.index is not None else 0
    buf = _BUFFER_CACHE.get(key)
    ntok = num_tokens if buf is None else max(num_tokens, buf["ntok"])
    nh = num_heads if buf is None else max(num_heads, buf["nheads"])
    hd = head_size if buf is None else max(head_size, buf["nheads_size"])
    if buf is None or ntok > buf["ntok"] or nh > buf["nheads"] or hd > buf["nheads_size"]:
        new = {
            "out": torch.empty((ntok, nh, MAX_SPLIT, hd),
                               dtype=torch.float32, device=device),
            "max": torch.empty((ntok, nh, MAX_SPLIT),
                               dtype=torch.float32, device=device),
            "expsum": torch.empty((ntok, nh, MAX_SPLIT),
                                  dtype=torch.float32, device=device),
            "ntok": ntok, "nheads": nh, "nheads_size": hd,
        }
        if buf is not None:
            logger.info("fl.attn-splitkv: 工作缓冲增长 %dx%dx%d → %dx%dx%d（graph "
                        "capture 前完成即可）", buf["ntok"], buf["nheads"],
                        buf["nheads_size"], ntok, nh, hd)
        _BUFFER_CACHE[key] = new
        buf = new
    return buf


# ---------------------------------------------------------------------------
# kernels（裸 @triton.jit）。triton 仅真机导入（无卡纪律：只 py_compile）。
# ---------------------------------------------------------------------------
_K1 = None
_K2 = None


def _build_kernels():
    global _K1, _K2
    if _K1 is not None:
        return _K1, _K2

    from vllm.triton_utils import tl, triton

    @triton.jit
    def _splitkv_decode_stage1(
        out_ptr,               # [num_tokens, NH, HD]（WRITETHROUGH 直写）
        q_ptr, k_ptr, v_ptr,
        block_table_ptr,       # [num_seqs, bt_stride] int32
        seq_lens_ptr,          # [num_seqs] int32
        cu_seqlens_ptr,        # [num_seqs+1] int32（decode: token 起点）
        sm_scale,
        segm_out_ptr,          # [ntok, NH, MAX_SPLIT, 128] fp32（3D partial）
        segm_max_ptr,          # [ntok, NH, MAX_SPLIT] fp32
        segm_expsum_ptr,       # [ntok, NH, MAX_SPLIT] fp32
        num_q_heads: tl.constexpr,
        q_per_kv: tl.constexpr,
        BLOCK_M: tl.constexpr,          # 16（8 有效行 + 填充，MMA 形状）
        block_table_stride: tl.int64,
        q_stride0: tl.int64, q_stride1: tl.int64,
        out_stride0: tl.int64, out_stride1: tl.int64,
        stride_k0: tl.int64, stride_k1: tl.int64,
        stride_k2: tl.int64, stride_k3: tl.int64,
        stride_v0: tl.int64, stride_v1: tl.int64,
        stride_v2: tl.int64, stride_v3: tl.int64,
        BLOCK_SIZE: tl.constexpr,       # 16（kv cache 页大小）
        TILE: tl.constexpr,             # 16（== BLOCK_SIZE：一 tile 一页）
        HEAD_SIZE: tl.constexpr,        # 128（2 的幂，dim 无填充）
        NUM_SPLITS: tl.constexpr,
        MAX_SPLIT: tl.constexpr,        # 工作缓冲第 3 维（索引步长用它，跨
                                        # 不同 split 的调用共享同一缓冲布局）
        WRITETHROUGH: tl.constexpr,
        VARIANT: tl.constexpr,
        GROUP: tl.constexpr,            # V1 预取分组 G（2 或 3）
        SEGS_PER_CTA: tl.constexpr,     # V2 每常驻 CTA 的段数
        USE_TD: tl.constexpr,
    ):
        seq_idx = tl.program_id(0)
        kv_head = tl.program_id(1)
        pid_job = tl.program_id(2)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        q_token = tl.load(cu_seqlens_ptr + seq_idx)

        total_tiles = tl.cdiv(seq_len, TILE)
        tiles_per_seg = tl.cdiv(total_tiles, NUM_SPLITS)

        # ---------- 确定本 CTA 负责的段列表（V2 常驻多段，其余单段） ----------
        if VARIANT == 2:
            seg_lo = pid_job * SEGS_PER_CTA
            seg_hi = tl.minimum(seg_lo + SEGS_PER_CTA, NUM_SPLITS)
        else:
            seg_lo = pid_job
            seg_hi = seg_lo + 1

        offs_m = tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_SIZE)
        offs_t = tl.arange(0, TILE)
        row_valid = offs_m < q_per_kv
        # decode q_len=1：第 m 行 ↔ q head = kv_head*q_per_kv + m（m<q_per_kv 有效）
        qh = kv_head * q_per_kv + offs_m
        q_mask = row_valid[:, None] & (offs_d[None, :] < HEAD_SIZE)
        q = tl.load(
            q_ptr + q_token * q_stride0 + qh[:, None] * q_stride1 + offs_d[None, :],
            mask=q_mask, other=0.0,
        )

        bt_row = block_table_ptr + seq_idx * block_table_stride

        for seg in range(seg_lo, seg_hi):
            tile_lo = seg * tiles_per_seg
            if tile_lo >= total_tiles:
                continue                      # 空段：不写（stage2 同规则跳过）
            tile_hi = tl.minimum(tile_lo + tiles_per_seg, total_tiles)

            M = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
            L = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
            acc = tl.zeros([BLOCK_M, HEAD_SIZE], dtype=tl.float32)

            # ---------- tile 循环（VARIANT 选择加载策略） ----------
            if VARIANT == 1 or VARIANT == 2:
                # V1 手动双缓冲（及 V2 的段内循环，G 固定 2）：G 分组，
                # 先发起组内全部加载再逐 tile 消化。不开 num_stages
                # （§2 前科：TD 态 s=2 大劣化）。
                G: tl.constexpr = GROUP if VARIANT == 1 else 2
                for jg in range(tile_lo, tile_hi, G):
                    j0 = jg
                    p0 = tl.load(bt_row + j0).to(tl.int64)
                    base_k0 = k_ptr + p0 * stride_k0 + kv_head * stride_k2
                    base_v0 = v_ptr + p0 * stride_v0 + kv_head * stride_v2
                    tmask0 = (j0 * TILE + offs_t) < seq_len
                    if USE_TD:
                        kd0 = tl.make_tensor_descriptor(
                            base=base_k0, shape=(BLOCK_SIZE, HEAD_SIZE),
                            strides=(stride_k1, stride_k3),
                            block_shape=(TILE, HEAD_SIZE))
                        vd0 = tl.make_tensor_descriptor(
                            base=base_v0, shape=(BLOCK_SIZE, HEAD_SIZE),
                            strides=(stride_v1, stride_v3),
                            block_shape=(TILE, HEAD_SIZE))
                        k0 = kd0.load([0, 0]).T
                        v0 = vd0.load([0, 0])
                    else:
                        k0 = tl.load(
                            base_k0 + offs_t[None, :] * stride_k1
                            + offs_d[:, None] * stride_k3,
                            mask=tmask0[None, :], other=0.0)
                        v0 = tl.load(
                            base_v0 + offs_t[:, None] * stride_v1
                            + offs_d[None, :] * stride_v3,
                            mask=tmask0[:, None], other=0.0)
                    if G >= 2:
                        j1 = jg + 1
                        tmask1 = (j1 * TILE + offs_t) < seq_len
                        lm1 = j1 < tile_hi
                        p1 = tl.load(bt_row + j1, mask=lm1, other=0).to(tl.int64)
                        base_k1 = k_ptr + p1 * stride_k0 + kv_head * stride_k2
                        base_v1 = v_ptr + p1 * stride_v0 + kv_head * stride_v2
                        if USE_TD:
                            kd1 = tl.make_tensor_descriptor(
                                base=base_k1, shape=(BLOCK_SIZE, HEAD_SIZE),
                                strides=(stride_k1, stride_k3),
                                block_shape=(TILE, HEAD_SIZE))
                            vd1 = tl.make_tensor_descriptor(
                                base=base_v1, shape=(BLOCK_SIZE, HEAD_SIZE),
                                strides=(stride_v1, stride_v3),
                                block_shape=(TILE, HEAD_SIZE))
                            k1 = kd1.load([0, 0]).T
                            v1 = vd1.load([0, 0])
                        else:
                            k1 = tl.load(
                                base_k1 + offs_t[None, :] * stride_k1
                                + offs_d[:, None] * stride_k3,
                                mask=tmask1[None, :], other=0.0)
                            v1 = tl.load(
                                base_v1 + offs_t[:, None] * stride_v1
                                + offs_d[None, :] * stride_v3,
                                mask=tmask1[:, None], other=0.0)
                    if G >= 3:
                        j2 = jg + 2
                        tmask2 = (j2 * TILE + offs_t) < seq_len
                        lm2 = j2 < tile_hi
                        p2 = tl.load(bt_row + j2, mask=lm2, other=0).to(tl.int64)
                        base_k2 = k_ptr + p2 * stride_k0 + kv_head * stride_k2
                        base_v2 = v_ptr + p2 * stride_v0 + kv_head * stride_v2
                        if USE_TD:
                            kd2 = tl.make_tensor_descriptor(
                                base=base_k2, shape=(BLOCK_SIZE, HEAD_SIZE),
                                strides=(stride_k1, stride_k3),
                                block_shape=(TILE, HEAD_SIZE))
                            vd2 = tl.make_tensor_descriptor(
                                base=base_v2, shape=(BLOCK_SIZE, HEAD_SIZE),
                                strides=(stride_v1, stride_v3),
                                block_shape=(TILE, HEAD_SIZE))
                            k2 = kd2.load([0, 0]).T
                            v2 = vd2.load([0, 0])
                        else:
                            k2 = tl.load(
                                base_k2 + offs_t[None, :] * stride_k1
                                + offs_d[:, None] * stride_k3,
                                mask=tmask2[None, :], other=0.0)
                            v2 = tl.load(
                                base_v2 + offs_t[:, None] * stride_v1
                                + offs_d[None, :] * stride_v3,
                                mask=tmask2[:, None], other=0.0)

                    # ---- 消化（组内逐 tile；加载已在飞） ----
                    s0 = tl.dot(q, k0.to(q.dtype)) * sm_scale
                    s0 = tl.where(
                        row_valid[:, None] & tmask0[None, :], s0, float("-inf"))
                    m0 = tl.maximum(M, tl.max(s0, 1))
                    p0e = tl.exp(s0 - m0[:, None])
                    alpha0 = tl.exp(M - m0)
                    L = L * alpha0 + tl.sum(p0e, 1)
                    acc = acc * alpha0[:, None] + tl.dot(p0e.to(v0.dtype), v0)
                    M = m0
                    if G >= 2:
                        if jg + 1 < tile_hi:
                            s1 = tl.dot(q, k1.to(q.dtype)) * sm_scale
                            s1 = tl.where(
                                row_valid[:, None] & tmask1[None, :],
                                s1, float("-inf"))
                            m1 = tl.maximum(M, tl.max(s1, 1))
                            p1e = tl.exp(s1 - m1[:, None])
                            alpha1 = tl.exp(M - m1)
                            L = L * alpha1 + tl.sum(p1e, 1)
                            acc = acc * alpha1[:, None] + tl.dot(
                                p1e.to(v1.dtype), v1)
                            M = m1
                    if G >= 3:
                        if jg + 2 < tile_hi:
                            s2 = tl.dot(q, k2.to(q.dtype)) * sm_scale
                            s2 = tl.where(
                                row_valid[:, None] & tmask2[None, :],
                                s2, float("-inf"))
                            m2 = tl.maximum(M, tl.max(s2, 1))
                            p2e = tl.exp(s2 - m2[:, None])
                            alpha2 = tl.exp(M - m2)
                            L = L * alpha2 + tl.sum(p2e, 1)
                            acc = acc * alpha2[:, None] + tl.dot(
                                p2e.to(v2.dtype), v2)
                            M = m2

            elif VARIANT == 3:
                # V3 宽加载：相邻页配对一次 [2*TILE,128] 加载，保持两个独立
                # 16 宽 dot（规避 t32 的 dot 形状劣化——§13.3.3 A/B 臂）。
                # BI-V150 布局 NKV=2 时 16*stride1==stride0，行 stride 视角下
                # 相邻页连续；不相邻/尾块回落单 tile。
                offs_t2 = tl.arange(0, 2 * TILE)
                j = tile_lo
                while j < tile_hi:
                    p0 = tl.load(bt_row + j).to(tl.int64)
                    pair_ok = j + 1 < tile_hi
                    if pair_ok:
                        p1w = tl.load(bt_row + j + 1).to(tl.int64)
                        pair_ok = pair_ok & (p1w == p0 + 1)
                    if pair_ok:
                        base2k = k_ptr + p0 * stride_k0 + kv_head * stride_k2
                        base2v = v_ptr + p0 * stride_v0 + kv_head * stride_v2
                        tmask2w = (j * TILE + offs_t2) < seq_len
                        k2w = tl.load(
                            base2k + offs_t2[None, :] * stride_k1
                            + offs_d[:, None] * stride_k3,
                            mask=tmask2w[None, :], other=0.0)
                        v2w = tl.load(
                            base2v + offs_t2[:, None] * stride_v1
                            + offs_d[None, :] * stride_v3,
                            mask=tmask2w[:, None], other=0.0)
                        ka = k2w[:, 0:TILE]
                        kb = k2w[:, TILE:2 * TILE]
                        va = v2w[0:TILE, :]
                        vb = v2w[TILE:2 * TILE, :]
                        ma = tmask2w[0:TILE]
                        mb = tmask2w[TILE:2 * TILE]
                        # —— half A（显式展开，避免运行时分支赋值张量） ——
                        sa = tl.dot(q, ka.to(q.dtype)) * sm_scale
                        sa = tl.where(
                            row_valid[:, None] & ma[None, :], sa, float("-inf"))
                        mna = tl.maximum(M, tl.max(sa, 1))
                        pa = tl.exp(sa - mna[:, None])
                        ala = tl.exp(M - mna)
                        L = L * ala + tl.sum(pa, 1)
                        acc = acc * ala[:, None] + tl.dot(pa.to(va.dtype), va)
                        M = mna
                        # —— half B ——
                        sb = tl.dot(q, kb.to(q.dtype)) * sm_scale
                        sb = tl.where(
                            row_valid[:, None] & mb[None, :], sb, float("-inf"))
                        mnb = tl.maximum(M, tl.max(sb, 1))
                        pb = tl.exp(sb - mnb[:, None])
                        alb = tl.exp(M - mnb)
                        L = L * alb + tl.sum(pb, 1)
                        acc = acc * alb[:, None] + tl.dot(pb.to(vb.dtype), vb)
                        M = mnb
                        j += 2
                    else:
                        base_k = k_ptr + p0 * stride_k0 + kv_head * stride_k2
                        base_v = v_ptr + p0 * stride_v0 + kv_head * stride_v2
                        tmask = (j * TILE + offs_t) < seq_len
                        k0 = tl.load(
                            base_k + offs_t[None, :] * stride_k1
                            + offs_d[:, None] * stride_k3,
                            mask=tmask[None, :], other=0.0)
                        v0 = tl.load(
                            base_v + offs_t[:, None] * stride_v1
                            + offs_d[None, :] * stride_v3,
                            mask=tmask[:, None], other=0.0)
                        s = tl.dot(q, k0.to(q.dtype)) * sm_scale
                        s = tl.where(
                            row_valid[:, None] & tmask[None, :], s,
                            float("-inf"))
                        mn = tl.maximum(M, tl.max(s, 1))
                        pe = tl.exp(s - mn[:, None])
                        al = tl.exp(M - mn)
                        L = L * al + tl.sum(pe, 1)
                        acc = acc * al[:, None] + tl.dot(pe.to(v0.dtype), v0)
                        M = mn
                        j += 1

            else:
                # V0 基线：逐 tile。
                for j in range(tile_lo, tile_hi):
                    p0 = tl.load(bt_row + j).to(tl.int64)
                    base_k = k_ptr + p0 * stride_k0 + kv_head * stride_k2
                    base_v = v_ptr + p0 * stride_v0 + kv_head * stride_v2
                    tmask = (j * TILE + offs_t) < seq_len
                    if USE_TD:
                        kd = tl.make_tensor_descriptor(
                            base=base_k, shape=(BLOCK_SIZE, HEAD_SIZE),
                            strides=(stride_k1, stride_k3),
                            block_shape=(TILE, HEAD_SIZE))
                        vd = tl.make_tensor_descriptor(
                            base=base_v, shape=(BLOCK_SIZE, HEAD_SIZE),
                            strides=(stride_v1, stride_v3),
                            block_shape=(TILE, HEAD_SIZE))
                        k0 = kd.load([0, 0]).T
                        v0 = vd.load([0, 0])
                    else:
                        k0 = tl.load(
                            base_k + offs_t[None, :] * stride_k1
                            + offs_d[:, None] * stride_k3,
                            mask=tmask[None, :], other=0.0)
                        v0 = tl.load(
                            base_v + offs_t[:, None] * stride_v1
                            + offs_d[None, :] * stride_v3,
                            mask=tmask[:, None], other=0.0)
                    s = tl.dot(q, k0.to(q.dtype)) * sm_scale
                    s = tl.where(
                        row_valid[:, None] & tmask[None, :], s, float("-inf"))
                    mn = tl.maximum(M, tl.max(s, 1))
                    pe = tl.exp(s - mn[:, None])
                    al = tl.exp(M - mn)
                    L = L * al + tl.sum(pe, 1)
                    acc = acc * al[:, None] + tl.dot(pe.to(v0.dtype), v0)
                    M = mn

            # ---------- 段尾：写 partial（或 writethrough 直写） ----------
            if WRITETHROUGH:
                o = acc / L[:, None]
                tl.store(
                    out_ptr + q_token * out_stride0 + qh[:, None] * out_stride1
                    + offs_d[None, :],
                    o.to(out_ptr.dtype.element_ty),
                    mask=q_mask,
                )
            else:
                so = (q_token.to(tl.int64) * num_q_heads * MAX_SPLIT * HEAD_SIZE
                      + qh.to(tl.int64) * MAX_SPLIT * HEAD_SIZE
                      + seg * HEAD_SIZE)
                tl.store(
                    segm_out_ptr + so[:, None] + offs_d[None, :], acc,
                    mask=q_mask)
                sm = (q_token.to(tl.int64) * num_q_heads * MAX_SPLIT
                      + qh.to(tl.int64) * MAX_SPLIT + seg)
                tl.store(segm_max_ptr + sm, M, mask=row_valid)
                tl.store(segm_expsum_ptr + sm, L, mask=row_valid)

    @triton.jit
    def _splitkv_decode_stage2(
        out_ptr,               # [num_tokens, NH, HD]
        segm_out_ptr, segm_max_ptr, segm_expsum_ptr,
        seq_lens_ptr,
        num_q_heads: tl.constexpr,
        out_stride0: tl.int64, out_stride1: tl.int64,
        TILE: tl.constexpr,
        HEAD_SIZE: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
        MAX_SPLIT: tl.constexpr,
    ):
        # per (seq==token, q-head) 一个 program；确定性向量合并（无 atomic）。
        seq_idx = tl.program_id(0)
        qh = tl.program_id(1)

        seq_len = tl.load(seq_lens_ptr + seq_idx)
        total_tiles = tl.cdiv(seq_len, TILE)
        tiles_per_seg = tl.cdiv(total_tiles, NUM_SPLITS)
        act_num_segments = tl.cdiv(seq_len, tiles_per_seg * TILE)

        offs_s = tl.arange(0, NUM_SPLITS)
        segm_mask = offs_s < act_num_segments
        offs_d = tl.arange(0, HEAD_SIZE)

        base = (seq_idx.to(tl.int64) * num_q_heads + qh) * MAX_SPLIT
        segm_max = tl.load(segm_max_ptr + base + offs_s,
                           mask=segm_mask, other=float("-inf"))
        overall_max = tl.max(segm_max)

        segm_expsum = tl.load(segm_expsum_ptr + base + offs_s,
                              mask=segm_mask, other=0.0)
        w = tl.where(segm_mask, tl.exp(segm_max - overall_max), 0.0)
        overall_expsum = tl.sum(segm_expsum * w)

        offs2 = (base * HEAD_SIZE + offs_s[:, None] * HEAD_SIZE
                 + offs_d[None, :])
        segm_out = tl.load(segm_out_ptr + offs2,
                           mask=segm_mask[:, None], other=0.0)
        acc_sum = tl.sum(segm_out * w[:, None], axis=0)
        acc = tl.where(overall_expsum == 0.0, 0.0, acc_sum / overall_expsum)

        tl.store(
            out_ptr + seq_idx * out_stride0 + qh * out_stride1 + offs_d,
            acc.to(out_ptr.dtype.element_ty),
        )

    _K1, _K2 = _splitkv_decode_stage1, _splitkv_decode_stage2
    return _K1, _K2


# ---------------------------------------------------------------------------
# host 侧：launch 入口
# ---------------------------------------------------------------------------
def splitkv_decode_attention(
    q, k, v, out,
    cu_seqlens_q, seqused_k, block_table,
    sm_scale,
    variant: int,
    use_td: bool,
) -> None:
    """decode 专用入口（q_len==1 由路由保证）。布局假设与上游 wrapper 一致：
    q/out [num_tokens, NH, HD]；k/v [num_blocks, BLOCK_SIZE, NKV, HD]。"""
    k1, k2 = _build_kernels()

    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    q_per_kv = num_heads // num_kv_heads
    block_size = k.shape[1]
    num_seqs = seqused_k.shape[0]
    # max_kv 只用 CPU 侧已知量（seqused_k 是 device 张量，运行时取 max 会同步——
    # attn_launch_config hop1 同一结论）；serve 侧由 probe/metadata 传入的
    # max_seqlen_k 在路由层算好了 split，这里直接用缓存值
    max_kv = _CURRENT_MAX_KV[0] if _CURRENT_MAX_KV[0] else block_table.shape[1] * block_size
    num_splits = resolve_split(num_seqs, max_kv)
    variant = int(variant)
    group = _env_int("VLLM_FL_ATTN_SPLITKV_GROUP", 2)
    group = max(1, min(group, 3))
    segs_per_cta = 4

    BLOCK_M = 16 if q_per_kv <= 16 else (q_per_kv + 7) & ~7
    writethrough = num_splits == 1

    if not writethrough:
        buf = _get_buffers(num_tokens, num_heads, q.shape[2], q.device)
        segm_out, segm_max, segm_expsum = buf["out"], buf["max"], buf["expsum"]
    else:
        # stage1 需要 non-null 指针（writethrough 分支不读，传 out 占位）
        segm_out = segm_max = segm_expsum = out

    if variant == 2:
        grid_jobs = (num_splits + segs_per_cta - 1) // segs_per_cta
    else:
        grid_jobs = num_splits

    grid = (num_seqs, num_kv_heads, grid_jobs)

    common = dict(
        out_ptr=out,
        q_ptr=q, k_ptr=k, v_ptr=v,
        block_table_ptr=block_table,
        seq_lens_ptr=seqused_k,
        cu_seqlens_ptr=cu_seqlens_q,
        sm_scale=sm_scale,
        segm_out_ptr=segm_out, segm_max_ptr=segm_max,
        segm_expsum_ptr=segm_expsum,
        num_q_heads=num_heads,
        q_per_kv=q_per_kv,
        BLOCK_M=BLOCK_M,
        block_table_stride=block_table.stride(0),
        q_stride0=q.stride(0), q_stride1=q.stride(1),
        out_stride0=out.stride(0), out_stride1=out.stride(1),
        stride_k0=k.stride(0), stride_k1=k.stride(1),
        stride_k2=k.stride(2), stride_k3=k.stride(3),
        stride_v0=v.stride(0), stride_v1=v.stride(1),
        stride_v2=v.stride(2), stride_v3=v.stride(3),
        BLOCK_SIZE=block_size,
        TILE=block_size,          # TILE=16 基线（== BLOCK_SIZE）
        HEAD_SIZE=q.shape[2],
        NUM_SPLITS=num_splits,
        MAX_SPLIT=MAX_SPLIT,
        WRITETHROUGH=writethrough,
        VARIANT=variant,
        GROUP=group,
        SEGS_PER_CTA=segs_per_cta,
        USE_TD=use_td,
        num_warps=1,
        num_stages=1,
    )
    k1[grid](**common)

    if not writethrough:
        k2[(num_tokens, num_heads)](
            out_ptr=out,
            segm_out_ptr=segm_out, segm_max_ptr=segm_max,
            segm_expsum_ptr=segm_expsum,
            seq_lens_ptr=seqused_k,
            num_q_heads=num_heads,
            out_stride0=out.stride(0), out_stride1=out.stride(1),
            TILE=block_size,
            HEAD_SIZE=q.shape[2],
            NUM_SPLITS=num_splits,
            MAX_SPLIT=MAX_SPLIT,
        )


# 路由层 → host 层的单次 kv 提示（CPU 侧 max_seqlen_k，避免 device 同步）
_CURRENT_MAX_KV = [0]


# ---------------------------------------------------------------------------
# hop1 装载（attn_launch_config 两跳模式的 hop1 部分）
# ---------------------------------------------------------------------------
def _route_decision(kwargs) -> Optional[str]:
    """返回 None（路由到自研 kernel）或透传原因字符串。只依赖 CPU 侧静态量。"""
    max_seqlen_q = kwargs.get("max_seqlen_q")
    if not isinstance(max_seqlen_q, int) or max_seqlen_q > 1:
        return f"q_len={max_seqlen_q}"
    seqused_k = kwargs.get("seqused_k")
    if not isinstance(seqused_k, torch.Tensor) or seqused_k.ndim != 1:
        return "seqused_k 非张量"
    num_seqs = seqused_k.shape[0]
    if num_seqs > 64:
        return f"bs={num_seqs}>64"
    if kwargs.get("softcap") not in (None, 0, 0.0):
        return "softcap"
    for key in ("q_descale", "k_descale", "v_descale", "alibi_slopes",
                "sinks", "mm_prefix_range", "qq_bias", "output_scale",
                "k_scale_cache", "v_scale_cache"):
        if kwargs.get(key) is not None:
            return key
    kv_quant_mode = kwargs.get("kv_quant_mode")
    if kv_quant_mode is not None and int(kv_quant_mode) != 0:
        return "kv_quant"
    causal = kwargs.get("causal")
    if causal is not True:
        return f"causal={causal}"
    window_size = kwargs.get("window_size")
    if not (isinstance(window_size, tuple) and len(window_size) == 2
            and window_size[0] < 0 and window_size[1] < 0):
        return f"window={window_size}"
    if kwargs.get("chunk_lookback", -1) != -1:
        return "chunk"
    q = kwargs.get("q")
    k = kwargs.get("k")
    v = kwargs.get("v")
    if not (isinstance(q, torch.Tensor) and isinstance(k, torch.Tensor)
            and isinstance(v, torch.Tensor)):
        return "qkv 非张量"
    if q.element_size() < 2:
        return "fp8 q"
    head_size = q.shape[2]
    if head_size & (head_size - 1) or head_size > 256:
        return f"head_size={head_size}"
    num_heads = q.shape[1]
    num_kv_heads = k.shape[2]
    if num_heads % num_kv_heads or num_heads // num_kv_heads > 16:
        return f"gqa {num_heads}:{num_kv_heads}"
    if k.shape[1] != v.shape[1] or k.shape[1] != 16:
        return f"block_size={k.shape[1]}"
    return None


def install() -> bool:
    """装载 hop1 路由（幂等；默认**关闭**——VLLM_FL_ATTN_SPLITKV=1 启用）。"""
    global _installed
    if _installed:
        return True
    if os.environ.get("VLLM_FL_ATTN_SPLITKV", "0") != "1":
        logger.info("attn_splitkv: VLLM_FL_ATTN_SPLITKV=0，保持上游 unified_attention")
        return False
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _tua
    except Exception as e:
        logger.warning("attn_splitkv: triton_unified_attention 不可用: %s", e)
        return False

    orig_wrapper = _tua.unified_attention
    if getattr(orig_wrapper, "_fl_splitkv_hook", False):
        _installed = True
        return True

    variant = max(0, min(_env_int("VLLM_FL_ATTN_SPLITKV_VARIANT", 1), 3))
    td_mode = _env_int("VLLM_FL_ATTN_SPLITKV_TD", -1)
    debug = _env_int("VLLM_FL_ATTN_SPLITKV_DEBUG", 0) == 1

    def _splitkv_unified_attention(
        q, k, v, out, cu_seqlens_q=None, max_seqlen_q=None, seqused_k=None,
        max_seqlen_k=None, softmax_scale=None, causal=None, window_size=None,
        block_table=None, softcap=None, *args, **kwargs,
    ):
        # 运行时开关（probe A/B 用；serve 恒为 env 初值）
        active = _active if _active is not None else True
        kw = dict(kwargs)
        kw.update({
            "q": q, "k": k, "v": v, "out": out,
            "cu_seqlens_q": cu_seqlens_q, "max_seqlen_q": max_seqlen_q,
            "seqused_k": seqused_k, "max_seqlen_k": max_seqlen_k,
            "softmax_scale": softmax_scale, "causal": causal,
            "window_size": window_size, "block_table": block_table,
            "softcap": softcap,
        })
        if active:
            reason = _route_decision(kw)
        else:
            reason = "disabled"
        if reason is None:
            global _fallback_logged
            try:
                _CURRENT_MAX_KV[0] = (max_seqlen_k if isinstance(
                    max_seqlen_k, int) else 0)
                use_td = kwargs.get("use_td", False)
                if td_mode >= 0:
                    use_td = bool(td_mode)
                var = variant if _variant_override is None else _variant_override
                splitkv_decode_attention(
                    q, k, v, out, cu_seqlens_q, seqused_k, block_table,
                    softmax_scale, variant=var, use_td=use_td)
                _route_stats["routed"] += 1
                if debug:
                    logger.info(
                        "fl.attn-splitkv: routed bs=%d kv<=%s split=%d "
                        "variant=%s td=%d",
                        seqused_k.shape[0], max_seqlen_k,
                        resolve_split(seqused_k.shape[0],
                                      _CURRENT_MAX_KV[0] or 0),
                        VARIANT_NAMES[var], int(use_td))
                return None
            except Exception as e:
                if not _fallback_logged:
                    _fallback_logged = True
                    logger.warning(
                        "fl.attn-splitkv: 自研 kernel 异常，本进程永久回落上游"
                        "（原因: %s）", e, exc_info=debug)
                _route_stats["passed"] += 1
        elif debug:
            logger.info("fl.attn-splitkv: passthrough (%s)", reason)
        _route_stats["passed"] += 1
        return orig_wrapper(q, k, v, out, cu_seqlens_q, max_seqlen_q,
                            seqused_k, max_seqlen_k, softmax_scale, causal,
                            window_size, block_table, softcap, *args, **kwargs)

    _splitkv_unified_attention._fl_splitkv_hook = True
    _splitkv_unified_attention._fl_splitkv_orig = orig_wrapper
    _tua.unified_attention = _splitkv_unified_attention
    # vllm 的 triton_attn 后端按名导入了该符号，两处绑定都要接上
    # （attn_launch_config L232-237 同法）
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
        if _ta.unified_attention is orig_wrapper:
            _ta.unified_attention = _splitkv_unified_attention
    except Exception as e:
        logger.warning("attn_splitkv: triton_attn 符号绑定未接上: %s", e)

    _installed = True
    logger.info(
        "fl.attn-splitkv: 已装载自研 split-KV decode kernel（variant=%s "
        "group=%d TD=%s；VLLM_FL_ATTN_SPLITKV=0 可关闭）",
        VARIANT_NAMES[variant], _env_int("VLLM_FL_ATTN_SPLITKV_GROUP", 2),
        "跟随 use_td" if td_mode < 0 else bool(td_mode))
    return True


def set_active(flag: Optional[bool]) -> None:
    """probe A/B 用：运行时开/关路由（None=回到 env 决策）。serve 不用。"""
    global _active
    _active = flag


def set_variant(v: Optional[int]) -> None:
    """probe 扫变体用：运行时切换 VARIANT（None=回到 install 时的 env 值）。

    graph 相容：variant 必须在一次 graph capture 前设定且 capture 后不变
    （probe 每 case 先 set 再 capture，天然满足；serve 进程恒定）。"""
    global _variant_override
    _variant_override = None if v is None else max(0, min(int(v), 3))
