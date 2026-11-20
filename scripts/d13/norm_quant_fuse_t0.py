# D13 fused_add_rms_norm + per-token-int8-quant 融合 kernel T0（沐曦 C500）。
#
# 目标（A 机指派/docs/开源方案调研报告 F7）：消 norm→quant 间的 bf16 往返与
# 双 kernel 开销。现役链 = vllm native fused_add_rms_norm（triton，A1 墙图
# 7.7µs）+ bare quant（9.2µs）；融合后单 kernel 直出 int8+scale+residual。
#
# 位精确判据（两处中间 round 必须复刻）：
#   两步链：h=(x+resid) 存 bf16 → norm 数学 f32 → 输出 y 存 bf16 →
#           quant 读 y.to(f32) absmax → recip*127 → round_half_to_even
#   融合版：h.to(bf16) → hf.to(f32) → y.to(bf16) → yf.to(f32) → 同 quant 数学
# quant 位兼容语义照抄 _dynamic_per_token_quant_int8_bare_kernel：
#   inv_scale = fdiv(1.0, absmax, ieee_rounding=True) * 127.0
#   scale = absmax * (1.0/127.0)
#
# T0：torch 语义链位对拍 + serve 真实两 kernel 计时对拍（graph 口径 3×N 中位）。
# 用法：python3 scripts/d13/norm_quant_fuse_t0.py [--m 64] [--k 2048]

import argparse
import sys

sys.path.insert(0, "scripts/d2")

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import bottleneck_probe as bp  # noqa: E402

DEV = "cuda"


@triton.jit
def _round_half_to_even(x):
    # 照抄 flaggems impl/quantization.py:15（现役 bare kernel 同款，无 libdevice 依赖）
    floor = tl.floor(x)
    fraction = x - floor
    is_odd = tl.abs(floor - 2.0 * tl.floor(floor / 2.0)) > 0.5
    round_up = (fraction > 0.5) | ((tl.abs(fraction - 0.5) < 1e-10) & is_odd)
    return tl.where(round_up, floor + 1.0, floor)


@triton.jit
def _fused_add_rms_norm_quant_kernel(
    x_ptr, resid_ptr, w_ptr,
    resid_out_ptr, q_ptr, scale_ptr,
    eps,
    N: tl.constexpr, BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row.to(tl.int64) * N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(resid_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    # vllm fused_add 语义：h = x + residual（bf16 存回 residual）
    h16 = (x + r).to(tl.bfloat16)
    tl.store(resid_out_ptr + base + offs, h16, mask=mask)
    hf = h16.to(tl.float32)
    var = tl.sum(hf * hf, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y = hf * rrms * w
    # norm 输出 bf16 中间张量（两步链在此落 bf16，quant 再读回 f32）
    y16 = y.to(tl.bfloat16)
    yf = y16.to(tl.float32)
    # —— quant 数学（位兼容 reciprocal-mul 语义）——
    absmax = tl.max(tl.abs(yf), axis=0)
    inv_scale = tl.where(
        absmax != 0.0,
        tl.fdiv(1.0, absmax, ieee_rounding=True) * 127.0,
        0.0,
    )
    q = _round_half_to_even(yf * inv_scale)
    q = tl.minimum(tl.maximum(q, -128.0), 127.0)
    tl.store(q_ptr + base + offs, q.to(tl.int8), mask=mask)
    tl.store(scale_ptr + row, absmax * (1.0 / 127.0))


def fused(x, resid, w, eps):
    M, N = x.shape
    resid_out = torch.empty_like(resid)
    q = torch.empty((M, N), device=x.device, dtype=torch.int8)
    scale = torch.empty((M, 1), device=x.device, dtype=torch.float32)
    block = triton.next_power_of_2(N)
    nw = 8 if block >= 8192 else 4
    _fused_add_rms_norm_quant_kernel[(M,)](
        x, resid, w, resid_out, q, scale, eps,
        N=N, BLOCK=block, num_warps=nw, num_stages=1)
    return q, scale, resid_out


@triton.jit
def _rms_norm_only_kernel(
    x_ptr, resid_ptr, w_ptr, resid_out_ptr, y_ptr,
    eps, N: tl.constexpr, BLOCK: tl.constexpr,
):
    """与融合 kernel 完全同数学的 norm-only 版（两步对照链的臂 1）。"""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    base = row.to(tl.int64) * N
    x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(resid_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    h16 = (x + r).to(tl.bfloat16)
    tl.store(resid_out_ptr + base + offs, h16, mask=mask)
    hf = h16.to(tl.float32)
    var = tl.sum(hf * hf, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    y16 = (hf * rrms * w).to(tl.bfloat16)
    tl.store(y_ptr + base + offs, y16, mask=mask)


def same_math_two_step(x, resid, w, eps):
    """同数学两步链：norm-only kernel（与融合版逐位同数学）+ serve 现役 bare quant。
    位一致判据的参照——证明融合本身零数值损失。"""
    M, N = x.shape
    resid_out = torch.empty_like(resid)
    y = torch.empty_like(x)
    block = triton.next_power_of_2(N)
    _rms_norm_only_kernel[(M,)](x, resid, w, resid_out, y, eps,
                                N=N, BLOCK=block,
                                num_warps=8 if block >= 8192 else 4)
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_bare as _bare_quant,
    )
    q, s = _bare_quant(y)
    return q, s, resid_out, y


def serve_chain(x, resid, w, eps):
    """serve 真实两 kernel（C++ _C fused_add + bare quant）。"""
    from vllm import _custom_ops
    xx, rr = x.clone(), resid.clone()
    _custom_ops.fused_add_rms_norm(xx, rr, w, eps)      # xx→norm 输出，rr→residual
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_bare as _bq,
    )
    q, s = _bq(xx)
    return q, s, rr, xx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=64)
    ap.add_argument("--k", type=int, default=2048)
    args = ap.parse_args()
    M, K, eps = args.m, args.k, 1e-6
    torch.manual_seed(0)
    x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16) * 2
    resid = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    w = (torch.randn(K, device=DEV, dtype=torch.bfloat16) * 0.1 + 1).abs()

    # —— 位精确对拍（判据：融合 vs 同数学两步链 = 融合零数值损失）——
    q1, s1, h1 = fused(x, resid, w, eps)
    q2, s2, h2, y2 = same_math_two_step(x, resid, w, eps)
    bit_q = (q1 == q2).all().item()
    bit_s = torch.equal(s1, s2)
    bit_h = torch.equal(h1, h2)
    print(f"[same-math 两步] q bitwise={'PASS' if bit_q else 'FAIL'} "
          f"({(q1 != q2).sum().item()}/{q1.numel()} mism) "
          f"scale bitwise={'PASS' if bit_s else 'FAIL'} "
          f"resid bitwise={'PASS' if bit_h else 'FAIL'}")

    # —— vs serve 真实链（C++ _C fused_add + bare quant；幅度口径）——
    # 注：serve 实际走 inductor triton 版；C++/inductor/torch 三者本就有
    # ~1e-3 相对级后端间差（已实测），此处量化融合版与 C++ 链的幅度同级性。
    q3, s3, h3, y3 = serve_chain(x, resid, w, eps)
    mism3 = (q1 != q3).sum().item()
    srel = ((s1 - s3).abs() / s3.abs().clamp_min(1e-9)).max().item()
    print(f"[vs C++ 链] q mism={mism3}/{q1.numel()}（后端间差量级）"
          f" scale maxrel={srel:.2e} resid={'PASS' if torch.equal(h1, h3) else 'DIFF(后端间)'}")

    # —— 计时（graph 口径；in-place 操作、clone 移出热路径）——
    from vllm import _custom_ops
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_bare as _bare_quant,
    )
    xb, rb = x.clone(), resid.clone()
    def chain():
        _custom_ops.fused_add_rms_norm(xb, rb, w, eps)
        return _bare_quant(xb)
    def fuse():
        return fused(x, resid, w, eps)
    t_a = bp.graph_time_us(chain, iters=30)
    t_b = bp.graph_time_us(fuse, iters=30)
    print(f"serve 两步链 : {t_a:8.1f}µs")
    print(f"融合单 kernel: {t_b:8.1f}µs   speedup = {t_a / t_b:.3f}×")


if __name__ == "__main__":
    main()
