# Copyright (c) 2026 BAAI. All rights reserved.
"""F-a：rope(Q)+rope(K) 单核融合 kernel（feat/fa-rope-cat-fusion）。

归因钉死的现役行为（2026-11-20，/tmp/torchinductor_root + serve trace）
----------------------------------------------------------------------------
MiniCPM5-2B-w8a8-v2（LlamaForCausalLM，42 层，16 Q heads / 2 KV heads，
head_dim=128，rope_theta=5e6，bf16）decode 步每层：
  qkv = qkv_proj(hidden)                       # [T, 2560] 连续（w8a8 mm 输出 buf29）
  q, k, v = qkv.split([2048, 256, 256], -1)    # 三段 strided view，无拷贝
  q, k = rotary_emb(positions, q, k)           # vllm native 分解（compile 模式下
                                               # CustomOp 强制 forward_native）
  → inductor 产出两个 pointwise 核：
    triton_poi_fused_cat_3: rope(Q) 42107 次 244.74ms（5.81µs/层·步）
    triton_poi_fused_cat_4: rope(K) 42107 次 189.80ms（4.51µs/层·步）
  （cat 命名来自 native rope 的 rotate_neox 内部 torch.cat，无独立 qkv-cat 算子；
   fused_1(0.32ms/步) 是 silu_and_mul 非 rope——trace 43134=42×1027 与层数吻合，
   cat_3/cat_4 42107=41×1027，缺的 1 层 rope 被并进别的融合组、名字不在 top 榜）
两核缺陷：块形状按 prefill 上界（size_hints 4194304/524288）选定，decode bs=32 时
cat_4 仅 4 个 CTA（104 SM 的 3.8%）→ 纯延迟 bound。

数学口径来源（逐位照抄，来源 = 现役 inductor 核 ttir/源码）
----------------------------------------------------------------------------
- vllm RotaryEmbedding.forward_static（native，base.py:161-201）：
  cos_sin = cos_sin_cache.index_select(0, positions); cos, sin = chunk(2, -1)
  neox: o1 = x1*cos - x2*sin; o2 = x2*cos + x1*sin（x1=x[:64], x2=x[64:]，
  cos_sin 行 = [cos(64)|sin(64)]，cache [131072,128]）
- inductor 生成核（k7/ck7fkn…、h6/ch6hjq…）的精确运算序：
  load bf16 → extf f32 → o1 = fma(x1, cos, -(x2*sin))、o2 = fma(x2, cos, x1*sin)
  → truncf bf16 store。本 kernel 源码写 `x1*c - x2*s` / `x2*c + x1*s`（与
  inductor 源码 tmp16 = tmp12 - tmp15 同形），fma 收缩由 triton 编译器完成，
  与现役核同机制。
- 位置回卷（ttir 实测语义）：pos_eff = pos<0 ? pos+131072 : pos（131072 =
  cos_sin_cache.size(0)；fx 图层无此算子，系 inductor 间接索引 codegen 引入）。
  本 kernel 保留同一回卷，任何输入下与现役核同值。

位精确声明：与 triton_poi_fused_cat_3/cat_4 逐位对齐（同输入、同运算序、同舍入
路径：f32 中间值 + RTNE 截回 bf16）。T0 对照臂即这两核的逐字拷贝（scripts/fa/
t0_rope_cat.py）；若上机实测出现末位差（fma 收缩差异），按分布等价降级并论证。

graph 相容设计
----------------------------------------------------------------------------
- 输出与现役完全同构：q_out [T,16,128] 连续（stride 2048,128,1）、k_out [T,2,128]
  连续（stride 256,128,1）——下游 unified_attention / reshape_and_cache 的布局
  不变，替换不破坏图结构。
- 纯函数 op（无原地写、无内部状态、无运行时分支）：cudagraph 捕获期行为确定；
  输出在 op 内 empty 分配，走 graph 内存池（inductor extern 调用常规形态）。
- grid = (T * (NQ + NKV),) 纯形状静态函数；每 program 恰一个 (token, head)，
  head_dim=128 恒 2 的幂 → 零尾部掩码；positions/cos_sin 只读。

纪律声明（违者必败——attn_splitkv.py 头注同源）
----------------------------------------------------------------------------
- 裸 ``@triton.jit``，严禁 ``@libentry``（流竞争前科）
- 严禁运行时 autotune：num_warps=1 / num_stages=1 静态写死（V1 不排流水，
  带宽/延迟 bound 的小核开 stage 无益且 MACA 前科在案）
- 严禁 atomic、严禁设备端同步
- launch 参数全部由张量形状纯函数决定，装载期一次定型
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_qk_cat_kernel(
    qkv_ptr,  # [T, QKV_ROW] bf16：[q(16*128) | k(2*128) | v(2*128)]
    pos_ptr,  # [T] i64
    cs_ptr,  # [max_pos, HEAD] bf16，行 = [cos(HALF) | sin(HALF)]
    q_out_ptr,  # [T, NQ, HEAD] bf16 连续
    k_out_ptr,  # [T, NKV, HEAD] bf16 连续
    max_pos,  # cos_sin_cache.size(0)，负位置回卷模数
    QKV_ROW: tl.constexpr,  # 2560
    Q_ROW: tl.constexpr,  # NQ*HEAD = 2048
    K_ROW: tl.constexpr,  # NKV*HEAD = 256
    NQ: tl.constexpr,  # 16
    NKV: tl.constexpr,  # 2
    HEAD: tl.constexpr,  # 128（= rotary_dim，本模型全维 rope）
    HALF: tl.constexpr,  # 64
):
    pid = tl.program_id(0)
    tok = pid // (NQ + NKV)
    h = pid % (NQ + NKV)

    # qkv 内 Q/K 的 18 个 head 恰好连续排布：head h 的列基 = h * HEAD
    e = tl.arange(0, HALF)
    src = tok * QKV_ROW + h * HEAD + e
    x1 = tl.load(qkv_ptr + src).to(tl.float32)
    x2 = tl.load(qkv_ptr + src + HALF).to(tl.float32)

    # 位置回卷（与现役核同语义）：pos<0 → pos+max_pos
    pos = tl.load(pos_ptr + tok)
    posw = tl.where(pos < 0, pos + max_pos, pos)
    cs = posw * HEAD + e
    c = tl.load(cs_ptr + cs).to(tl.float32)
    s = tl.load(cs_ptr + cs + HALF).to(tl.float32)

    # 与 inductor 核同形：sub(mul,mul)/add(mul,mul) → triton 收缩为 fma
    o1 = x1 * c - x2 * s
    o2 = x2 * c + x1 * s

    is_q = h < NQ
    m_q = is_q & (e >= 0)
    m_k = (~is_q) & (e >= 0)
    out_dt = q_out_ptr.dtype.element_ty
    # 偏移钳 0（ATU 加固）：Q 程序的 k 偏移、K 程序的 q 偏移不再出现负值。
    # 掩码 lane 本就不写，钳 0 不改数学（首跑 t0.log 已实证原版 bitwise 全绿），
    # 只消灭负地址算术这一 MACA 风险类（topk_topp 前科：编译过、运行期 Xnack
    # trap）。钳后所有指针恒在各自张量界内。
    # int32 界核查：地址乘积上限 = 2047*2560+17*128+127 ≈ 5.24M（qkv）、
    # 2047*2048 ≈ 4.19M（q_out）、131071*128 ≈ 16.8M（cos_sin，posw 为 i64）
    # ——均 ≪ 2^31，M ≤ 2048 下无 32 位溢出。
    qh = tl.where(is_q, h, 0)
    kh = tl.where(is_q, 0, h - NQ)
    qoff = tok * Q_ROW + qh * HEAD
    koff = tok * K_ROW + kh * HEAD
    tl.store(q_out_ptr + qoff + e, o1.to(out_dt), mask=m_q)
    tl.store(q_out_ptr + qoff + HALF + e, o2.to(out_dt), mask=m_q)
    tl.store(k_out_ptr + koff + e, o1.to(out_dt), mask=m_k)
    tl.store(k_out_ptr + koff + HALF + e, o2.to(out_dt), mask=m_k)


def rope_qk_cat(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_out: torch.Tensor,
    k_out: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
) -> None:
    """ fused rope(Q)+rope(K) 启动器。契约（调用方保证，热路径零检查）：
    qkv [T, (NQ+NKV*2)*HEAD] 连续 bf16；positions [T] i64 连续；
    cos_sin_cache [max_pos, HEAD] 连续；q_out/k_out 形状 [T,NQ,HEAD]/[T,NKV,HEAD]
    连续。grid/warps 由形状纯函数决定（graph 相容）。 """
    head_size = cos_sin_cache.shape[1]
    qkv_row = qkv.shape[1]
    q_row = num_q_heads * head_size
    k_row = num_kv_heads * head_size
    grid = (qkv.shape[0] * (num_q_heads + num_kv_heads),)
    _rope_qk_cat_kernel[grid](
        qkv,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        cos_sin_cache.shape[0],
        QKV_ROW=qkv_row,
        Q_ROW=q_row,
        K_ROW=k_row,
        NQ=num_q_heads,
        NKV=num_kv_heads,
        HEAD=head_size,
        HALF=head_size // 2,
        num_warps=1,
        num_stages=1,
    )


def rope_qk_cat_alloc(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ 自带输出分配的完整入口（compile pass 的 custom op impl 调这个）。 """
    num_tokens = qkv.shape[0]
    head_size = cos_sin_cache.shape[1]
    q_out = torch.empty(
        (num_tokens, num_q_heads, head_size), dtype=qkv.dtype, device=qkv.device
    )
    k_out = torch.empty(
        (num_tokens, num_kv_heads, head_size), dtype=qkv.dtype, device=qkv.device
    )
    rope_qk_cat(
        qkv,
        positions,
        cos_sin_cache,
        q_out,
        k_out,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )
    return q_out, k_out
