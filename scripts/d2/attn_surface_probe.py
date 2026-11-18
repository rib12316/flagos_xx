# 瓶颈深挖 L1 扩展：attention 双路径(2D/3D)×形状曲面 + split 段数扫 + use_td A/B。
#
# 与 bottleneck_probe.py（playbook 5min 标准工具，保持不动）的关系：复用其
# graph_time_us（CUDA graph 3×N replay 中位，唯一可信口径）与 gemm/elem 参照，
# 新增四个维度——
#   path:    "2d"  不传 segm 参数（=原探针行为，也是 S>64 退化路径）
#            "3d"  serve 等效（seq_threshold_3D=64；S>64 时 wrapper 自动退 2D，
#                  天然复现 serve 的 3D→2D 悬崖）
#            "3dx" threshold=S（测"门槛重调让大 bs 仍走 3D"的假设，免写 patch）
#   segments: 3D split-KV 段数（上游固定 16；扫 {2,4,8,16,32} 定最优切块）
#   use_td:  tensor-descriptor A/B（Xe2 向机制，iluvatar 默认关；编译期死码消除）
#   read_ref: 同字节量连续读（graph 口径）作逐点可达带宽分母——小形态数据进
#             L2 时固定 829/544 分母会低估可收割空间
# launch 口径：alc.install() 后 hop1+hop2 生效；max_seqlen_k 必须 kwargs 传
# （hop1 只认 kwargs，位置传参 kv 档提示丢失——原探针的隐性偏差，碰巧表值相同）。
# CSV 增量落盘（每 case 立即 append+fsync）+ 默认 --resume 跳过已完成 tag。
import argparse
import csv
import os
import statistics
import sys
import time

os.environ.setdefault("VLLM_PLUGINS", "fl")

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                       # 复用同目录 bottleneck_probe
sys.path.insert(0, os.path.join(_HERE, "..", "..", "vllm-plugin-FL"))

import torch  # noqa: E402
import triton  # noqa: E402
import triton.language as tl  # noqa: E402

import bottleneck_probe as bp  # noqa: E402  (module 顶层只做 import，不跑 run())
import vllm.v1.attention.ops.triton_unified_attention as tua  # noqa: E402

DEV = "cuda"
NH, NKV, HD, BLOCK = bp.NH, bp.NKV, bp.HD, bp.BLOCK
BW_CEIL, MAC_CEIL = bp.BW_CEIL, bp.MAC_CEIL
THRESH_SERVE = 64          # serve：MIN_LAUNCH_GRID_SIZE_2D(128)//NKV(2)，吸附 capture 64
DEFAULT_SEG = 16           # 上游 NUM_PAR_SOFTMAX_SEGMENTS

COLUMNS = ["tag", "S", "q_len", "kv_len", "path_req", "path_eff", "segments",
           "use_td", "num_warps", "num_stages", "time_us", "kv_read_bytes",
           "read_ref_us", "GBps_attn", "GBps_ref", "pct_of_ref", "pct_of_829",
           "T_MACs", "pct_of_118", "verdict"]

# 同字节量连续读参照（一次分配最大档，逐 case 切前 n 个元素）
_MAX_REF_BYTES = 128 * 32768 * 2 * NKV * HD * 2
_ref_x = None


def read_ref_us(nbytes):
    global _ref_x
    n = nbytes // 2
    if _ref_x is None:
        _ref_x = torch.randn(_MAX_REF_BYTES // 2, device=DEV, dtype=torch.bfloat16)

    def call():
        return torch.sum(_ref_x[:n])
    return bp.graph_time_us(call, iters=30)


def attn_case5(S, q_len, kv_len, path="2d", segments=DEFAULT_SEG, use_td=False):
    """增强版 attn_case：path/segments/use_td 可控，其余与 bp.attn_case 同构。"""
    q = torch.randn(S * q_len, NH, HD, device=DEV, dtype=torch.bfloat16)
    nb = triton.cdiv(kv_len, BLOCK) * S + 4
    kv = torch.randn(2, nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16)
    bt = torch.arange(S * triton.cdiv(kv_len, BLOCK), dtype=torch.int32,
                      device=DEV).reshape(S, -1)
    cu = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
    su = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)

    kwargs = dict(softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
                  softcap=0.0, q_descale=None, k_descale=None, v_descale=None,
                  use_td=use_td)
    if path in ("3d", "3dx"):
        thr = THRESH_SERVE if path == "3d" else max(S, THRESH_SERVE)
        segm_out = torch.empty(thr, NH, segments, HD,
                               dtype=torch.float32, device=DEV)
        segm_max = torch.empty(thr, NH, segments,
                               dtype=torch.float32, device=DEV)
        segm_sum = torch.empty(thr, NH, segments,
                               dtype=torch.float32, device=DEV)
        kwargs.update(seq_threshold_3D=thr, num_par_softmax_segments=segments,
                      softmax_segm_output=segm_out, softmax_segm_max=segm_max,
                      softmax_segm_expsum=segm_sum)

    def call():
        tua.unified_attention(
            q, kv[0], kv[1], out, cu_seqlens_q=cu, max_seqlen_q=q_len,
            seqused_k=su, max_seqlen_k=kv_len, block_table=bt, **kwargs)
        return out
    by = S * kv_len * 2 * NKV * HD * 2  # K+V 读字节
    # wrapper 判据复算（q<=1 且 S<=thr 才真 3D）
    eff = "3d" if (path in ("3d", "3dx") and q_len <= 1 and S <= kwargs.get(
        "seq_threshold_3D", 0)) else "2d"
    return call, by, eff


def hook_cfg(S, q_len, kv_len, eff):
    """复算 alc hop1+hop2 会注入的 (warps, stages)——与钩子同一 resolve 逻辑。"""
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_launch_config as alc
    path = "decode" if eff == "3d" else "prefill"   # hop2 按 len(grid) 判
    return alc.resolve(path, S, kv_len)


def run_case(writer, done, out_f, tag, S, q_len, kv_len, path, segments, use_td):
    if tag in done:
        print(f"[skip] {tag}", flush=True)
        return
    t0 = time.time()
    try:
        call, by, eff = attn_case5(S, q_len, kv_len, path, segments, use_td)
        # 慢 case（prefill 大形态 ~1s/次）降 replay 次数，仍是 3 轮中位
        est = bp.graph_time_us(call, iters=3)
        iters = 10 if est > 3000 else 30
        t = bp.graph_time_us(call, iters=iters)
        del call
        ref_t = read_ref_us(by)
        gb_a, gb_r = by / t * 1e-3, by / ref_t * 1e-3
        macs = S * q_len * kv_len * NH * HD * 2 / (t * 1e-6) / 1e12
        w, s = hook_cfg(S, q_len, kv_len, eff) or ("", "")
        row = dict(tag=tag, S=S, q_len=q_len, kv_len=kv_len, path_req=path,
                   path_eff=eff, segments=segments if eff == "3d" else 1,
                   use_td=int(use_td), num_warps=w, num_stages=s, time_us=f"{t:.1f}",
                   kv_read_bytes=by, read_ref_us=f"{ref_t:.1f}",
                   GBps_attn=f"{gb_a:.0f}", GBps_ref=f"{gb_r:.0f}",
                   pct_of_ref=f"{gb_a / gb_r * 100:.0f}",
                   pct_of_829=f"{gb_a / BW_CEIL * 100:.0f}",
                   T_MACs=f"{macs:.2f}", pct_of_118=f"{macs / MAC_CEIL * 100:.0f}",
                   verdict=bp.cls(gb_a / BW_CEIL * 100, macs / MAC_CEIL * 100))
    except Exception as e:  # USE_TD 等可能不支持：记录后继续
        row = {c: "" for c in COLUMNS}
        row.update(tag=tag, S=S, q_len=q_len, kv_len=kv_len, path_req=path,
                   segments=segments, use_td=int(use_td),
                   verdict=f"ERROR: {type(e).__name__}: {e}"[:200])
    writer.writerow(row)
    out_f.flush()
    os.fsync(out_f.fileno())
    print(f"[{time.time() - t0:6.1f}s] {tag:<44} {row.get('time_us', ''):>9} "
          f"{row.get('GBps_attn', ''):>6}GB/s {row.get('pct_of_ref', ''):>5}%ref "
          f"{row.get('verdict', '')}", flush=True)
    torch.cuda.empty_cache()


def load_done(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("verdict", "").startswith("ERROR"):
                    continue          # 失败行重试
                done.add(r["tag"])
    return done


DECODE_GRID = [(kv, bs) for kv in (1024, 4096, 8192, 16384, 32768)
               for bs in (8, 16, 32, 64, 128)]
SEGM_POINTS = [(16, 8192), (16, 32768), (64, 8192), (64, 32768)]
TD_POINTS = [(64, 2048), (64, 16384), (16, 8192), (128, 16384)]
PREFILL_GRID = [(S, q, kv) for S in (1, 2, 4, 8) for q in (1024, 2048, 4096)
                for kv in (4096, 16384)]
PREFILL_CTRL = [(1, 4096, 4096), (8, 512, 4096), (1, 4096, 16384), (8, 512, 16384),
                (2, 2048, 4096)]   # 同 total-q 对照（S=1 q=4096 vs S=8 q=512 等）
SCATTER_POINTS = [(64, 4096), (64, 16384), (16, 16384), (32, 8192)]


@triton.jit
def _read_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(x_ptr + offs, mask=offs < n, other=0).to(tl.float32)
    tl.store(out_ptr + pid, tl.sum(x, 0))


def tuned_read_ref_us(nbytes):
    """调优读参照（BLOCK=16384 大块读，台账 829GB/s 同口径）——比 sum() 顺序读
    （~350GB/s）更接近本卡读流上限；归约写每 program 1 个 fp32，无 atomic。"""
    global _ref_x
    n = nbytes // 2
    if _ref_x is None:
        _ref_x = torch.randn(_MAX_REF_BYTES // 2, device=DEV, dtype=torch.bfloat16)
    best = None
    for w in (4, 8):
        out = torch.empty(triton.cdiv(n, 16384), device=DEV, dtype=torch.float32)

        def call():
            _read_kernel[(triton.cdiv(n, 16384),)](
                _ref_x, out, n, BLOCK=16384, num_warps=w)
        t = bp.graph_time_us(call, iters=30)
        best = t if best is None else min(best, t)
    return best


def attn_case_scatter(S, q_len, kv_len, segments=DEFAULT_SEG):
    """同 attn_case5 3D，但 block_table 行内乱序（模拟 serve 分配器散射的
    8KB 粒度 gather；probe 默认 arange=纯顺序流，会高估带宽可达性）。"""
    q = torch.randn(S * q_len, NH, HD, device=DEV, dtype=torch.bfloat16)
    nb = triton.cdiv(kv_len, BLOCK) * S + 4
    kv = torch.randn(2, nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16)
    g = torch.Generator(device="cpu").manual_seed(20261115)
    bt = torch.randperm(S * triton.cdiv(kv_len, BLOCK), generator=g
                        ).to(torch.int32).to(DEV).reshape(S, -1)
    cu = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
    su = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)
    out = torch.empty_like(q)
    segm_out = torch.empty(THRESH_SERVE if S <= THRESH_SERVE else S, NH, segments,
                           HD, dtype=torch.float32, device=DEV)
    segm_max = torch.empty(segm_out.shape[0], NH, segments,
                           dtype=torch.float32, device=DEV)
    segm_sum = torch.empty_like(segm_max)

    def call():
        tua.unified_attention(
            q, kv[0], kv[1], out, cu_seqlens_q=cu, max_seqlen_q=q_len,
            seqused_k=su, max_seqlen_k=kv_len, block_table=bt,
            seq_threshold_3D=THRESH_SERVE if S <= THRESH_SERVE else S,
            num_par_softmax_segments=segments, softmax_segm_output=segm_out,
            softmax_segm_max=segm_max, softmax_segm_expsum=segm_sum,
            softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
            softcap=0.0, q_descale=None, k_descale=None, v_descale=None)
        return out
    return call, S * kv_len * 2 * NKV * HD * 2

SUITES = {
    "decode": [(f"dec bs{bs} kv{kv//1024}k p3d", bs, 1, kv, "3d", DEFAULT_SEG, False)
               for kv, bs in DECODE_GRID]
              + [(f"dec bs{bs} kv{kv//1024}k p2d", bs, 1, kv, "2d", DEFAULT_SEG, False)
                 for kv, bs in DECODE_GRID]
              + [(f"dec bs{bs} kv{kv//1024}k p3dx", bs, 1, kv, "3dx", DEFAULT_SEG, False)
                 for kv, bs in DECODE_GRID if bs > THRESH_SERVE],
    "segm": [(f"seg bs{bs} kv{kv//1024}k n{seg}", bs, 1, kv, "3d", seg, False)
             for bs, kv in SEGM_POINTS for seg in (2, 4, 8, 32)
             if seg != DEFAULT_SEG],
    "td": [(f"td bs{bs} kv{kv//1024}k td{int(td)}", bs, 1, kv, "3d", DEFAULT_SEG, td)
           for bs, kv in TD_POINTS for td in (False, True)],
    "prefill": [(f"pre S{S} q{q} kv{kv//1024}k", S, q, kv, "2d", DEFAULT_SEG, False)
                for S, q, kv in PREFILL_GRID + PREFILL_CTRL],
    # TD 二期：prefill 2D（87.7% 墙图主力）× TD，及 decode TD×最优段数叠加
    "tdx": [(f"ptd S{S} q{q} kv{kv//1024}k td{int(td)}", S, q, kv, "2d",
             DEFAULT_SEG, td)
            for S, q, kv in [(1, 4096, 4096), (4, 4096, 4096), (1, 2048, 16384),
                             (8, 1024, 16384), (2, 2048, 4096)]
            for td in (False, True)]
           + [(f"tseg bs{bs} kv{kv//1024}k n{seg} td1", bs, 1, kv, "3d", seg, True)
              for bs, kv, seg in [(64, 8192, 8), (64, 4096, 8), (16, 8192, 32),
                                  (64, 32768, 32)]],
}


def run_ref_suite(out_dir):
    """对已有 CSV 里出现过的每种字节量测 sum 参照 + 调优读参照 → ref_table.csv。"""
    sizes = set()
    for name in ("decode_surface.csv", "prefill_surface.csv",
                 "segm_sweep.csv", "td_ab.csv"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            with open(p) as f:
                for r in csv.DictReader(f):
                    if r.get("kv_read_bytes"):
                        sizes.add(int(r["kv_read_bytes"]))
    p = os.path.join(out_dir, "ref_table.csv")
    with open(p, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["kv_read_bytes", "sum_ref_us", "tuned_ref_us",
                    "GBps_sum", "GBps_tuned"])
        for by in sorted(sizes):
            if by > _MAX_REF_BYTES:
                continue
            s_t, t_t = read_ref_us(by), tuned_read_ref_us(by)
            w.writerow([by, f"{s_t:.1f}", f"{t_t:.1f}",
                        f"{by / s_t * 1e-3:.0f}", f"{by / t_t * 1e-3:.0f}"])
            f.flush()
            os.fsync(f.fileno())
            print(f"[ref] {by/1e6:8.1f}MB  sum {by/s_t*1e-3:5.0f}GB/s  "
                  f"tuned {by/t_t*1e-3:5.0f}GB/s", flush=True)


def run_scatter_suite(out_dir):
    """乱序 block_table（serve 分配器散射口径）vs 顺序（arange）对照。"""
    p = os.path.join(out_dir, "scatter_ab.csv")
    done = load_done(p)
    new_file = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for bs, kv in SCATTER_POINTS:
            tag = f"scat bs{bs} kv{kv//1024}k p3d"
            if tag in done:
                print(f"[skip] {tag}", flush=True)
                continue
            call, by = attn_case_scatter(bs, 1, kv)
            eff = "3d"
            t = bp.graph_time_us(call, iters=30)
            gb_a = by / t * 1e-3
            macs = bs * kv * NH * HD * 2 / (t * 1e-6) / 1e12
            w_, s_ = hook_cfg(bs, 1, kv, eff) or ("", "")
            row = dict(tag=tag, S=bs, q_len=1, kv_len=kv, path_req="3d-scat",
                       path_eff=eff, segments=DEFAULT_SEG, use_td=0,
                       num_warps=w_, num_stages=s_, time_us=f"{t:.1f}",
                       kv_read_bytes=by, GBps_attn=f"{gb_a:.0f}",
                       pct_of_829=f"{gb_a / BW_CEIL * 100:.0f}",
                       T_MACs=f"{macs:.2f}",
                       pct_of_118=f"{macs / MAC_CEIL * 100:.0f}",
                       verdict=bp.cls(gb_a / BW_CEIL * 100, macs / MAC_CEIL * 100))
            writer.writerow(row)
            f.flush()
            os.fsync(f.fileno())
            print(f"[scat] {tag:<28} {t:>9.1f}µs {gb_a:>5.0f}GB/s", flush=True)
            torch.cuda.empty_cache()


TILE_SHAPES = [(1, 4096, 4096), (2, 2048, 16384), (4, 4096, 4096)]
TILE_ARMS = [(16, 2, 1), (32, 2, 1), (32, 4, 2), (64, 2, 1),
             (64, 4, 2), (64, 8, 2), (64, 8, 3), (128, 4, 1),
             (16, 4, 1), (16, 4, 2), (16, 8, 2)]
# ⑥ 前置：decode 3D 的 TILE 深扫（免写 kernel 探自写 kernel 的第一收益假设）
DTILE_SHAPES = [(64, 1, 4096), (64, 1, 16384), (32, 1, 16384), (16, 1, 8192)]
DTILE_ARMS = [(16, 1, 1), (32, 1, 1), (32, 2, 1), (64, 1, 1), (64, 2, 1)]


def run_tile_suite(out_dir):
    """⑤ prefill TILE×warps×stages 深扫（免写 kernel：patch _get_tile_size +
    覆盖 launch 表）。TD=off（默认态）。TILE 是 constexpr——每臂重编译，增量落盘。"""
    import vllm.v1.attention.ops.triton_unified_attention as _tua2
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_launch_config as alc2
    orig_tile = _tua2._get_tile_size

    p = os.path.join(out_dir, "tile_sweep.csv")
    done = load_done(p)
    new_file = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for S, q, kv in TILE_SHAPES:
            for tile, w, s in TILE_ARMS:
                tag = f"tile S{S} q{q} kv{kv//1024}k t{tile} w{w}s{s}"
                if tag in done:
                    print(f"[skip] {tag}", flush=True)
                    continue
                t0 = time.time()

                def _tile(head_size, sliding_window, element_size, is_prefill):
                    if is_prefill:
                        return tile
                    return orig_tile(head_size, sliding_window,
                                     element_size, is_prefill)
                _tua2._get_tile_size = _tile
                alc2._OVERRIDE_TABLE = {("prefill", "*", "*"): (w, s),
                                        ("decode", "*", "*"): (w, s)}
                alc2._OVERRIDE_LOADED = True
                try:
                    call, by, eff = attn_case5(S, q, kv, "2d", DEFAULT_SEG, False)
                    est = bp.graph_time_us(call, iters=3)
                    iters = 10 if est > 3000 else 30
                    t = bp.graph_time_us(call, iters=iters)
                    del call
                    gb_a = by / t * 1e-3
                    macs = S * q * kv * NH * HD * 2 / (t * 1e-6) / 1e12
                    row = dict(tag=tag, S=S, q_len=q, kv_len=kv, path_req="2d-tile",
                               path_eff=eff, segments=DEFAULT_SEG, use_td=0,
                               num_warps=w, num_stages=s, time_us=f"{t:.1f}",
                               kv_read_bytes=by, GBps_attn=f"{gb_a:.0f}",
                               pct_of_829=f"{gb_a / BW_CEIL * 100:.0f}",
                               T_MACs=f"{macs:.2f}",
                               pct_of_118=f"{macs / MAC_CEIL * 100:.1f}",
                               verdict=f"TILE={tile}")
                except Exception as e:
                    row = {c: "" for c in COLUMNS}
                    row.update(tag=tag, S=S, q_len=q, kv_len=kv, path_req="2d-tile",
                               num_warps=w, num_stages=s,
                               verdict=f"TILE={tile} ERROR: {type(e).__name__}"
                                       f" {str(e)[:80]}")
                finally:
                    _tua2._get_tile_size = orig_tile
                    alc2._OVERRIDE_TABLE = None
                    alc2._OVERRIDE_LOADED = False
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print(f"[{time.time() - t0:6.1f}s] {tag:<40} {row.get('time_us',''):>10} "
                      f"{row.get('pct_of_118',''):>6}%MAC {row.get('verdict','')[:40]}",
                      flush=True)
                torch.cuda.empty_cache()
    print("TILE_DONE", flush=True)


def run_dtile_suite(out_dir):
    """⑥ 前置重估：decode 3D 路径 TILE 深扫（patch _get_tile_size 的 decode
    分支 + 覆盖 launch 表），TD=off 默认态。免写 kernel 直接检验自写 split-KV
    kernel 的第一收益假设（更大 TILE+分段）。"""
    import vllm.v1.attention.ops.triton_unified_attention as _tua2
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_launch_config as alc2
    orig_tile = _tua2._get_tile_size

    p = os.path.join(out_dir, "dtile_sweep.csv")
    done = load_done(p)
    new_file = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for S, ql, kv in DTILE_SHAPES:
            for tile, w, s in DTILE_ARMS:
                tag = f"dtile bs{S} kv{kv//1024}k t{tile} w{w}s{s}"
                if tag in done:
                    continue

                def _tile(head_size, sliding_window, element_size, is_prefill):
                    if not is_prefill:
                        return tile
                    return orig_tile(head_size, sliding_window,
                                     element_size, is_prefill)
                _tua2._get_tile_size = _tile
                alc2._OVERRIDE_TABLE = {("decode", "*", "*"): (w, s),
                                        ("prefill", "*", "*"): (w, s)}
                alc2._OVERRIDE_LOADED = True
                try:
                    call, by, eff = attn_case5(S, ql, kv, "3d", DEFAULT_SEG, False)
                    t = bp.graph_time_us(call, iters=30)
                    del call
                    gb_a = by / t * 1e-3
                    macs = S * ql * kv * NH * HD * 2 / (t * 1e-6) / 1e12
                    row = dict(tag=tag, S=S, q_len=ql, kv_len=kv, path_req="3d-dtile",
                               path_eff=eff, segments=DEFAULT_SEG, use_td=0,
                               num_warps=w, num_stages=s, time_us=f"{t:.1f}",
                               kv_read_bytes=by, GBps_attn=f"{gb_a:.0f}",
                               pct_of_829=f"{gb_a / BW_CEIL * 100:.0f}",
                               T_MACs=f"{macs:.2f}",
                               pct_of_118=f"{macs / MAC_CEIL * 100:.0f}",
                               verdict=f"TILE={tile}")
                except Exception as e:
                    row = {c: "" for c in COLUMNS}
                    row.update(tag=tag, S=S, q_len=ql, kv_len=kv, path_req="3d-dtile",
                               num_warps=w, num_stages=s,
                               verdict=f"TILE={tile} ERROR: {type(e).__name__}")
                finally:
                    _tua2._get_tile_size = orig_tile
                    alc2._OVERRIDE_TABLE = None
                    alc2._OVERRIDE_LOADED = False
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print(f"{tag:<32} {row.get('time_us',''):>10} {row.get('GBps_attn',''):>6}GB/s "
                      f"{row.get('pct_of_ref',''):>6} {row.get('verdict','')[:30]}", flush=True)
                torch.cuda.empty_cache()
    print("DTILE_DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=list(SUITES) + ["ref", "scatter", "tile",
                                                      "dtile", "all"],
                    default="all")
    ap.add_argument("--out-dir", default=os.path.join(
        os.path.dirname(_HERE), "..", "results", "20261115_bottleneck_deepdive"))
    ap.add_argument("--fresh", action="store_true", help="忽略已有 CSV 重测")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    used = int(os.popen("/usr/local/corex/bin/ixsmi --query-gpu=memory.used "
                        "--format=csv,noheader,nounits").read())
    assert used < 3000, f"GPU busy: {used} MiB"
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_launch_config as alc
    alc.install()  # type: ignore[attr-defined]
    assert getattr(tua.kernel_unified_attention, "_fl_launch_config_hook", False), \
        "launch 钩子未生效，测的不是 serve 口径"
    x = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
    for _ in range(200):
        x @ x
    torch.cuda.synchronize()
    del x
    torch.cuda.empty_cache()

    names = list(SUITES) + ["ref", "scatter"] if args.suite == "all" else [args.suite]
    for name in names:
        if name == "ref":
            run_ref_suite(out_dir)
            continue
        if name == "scatter":
            run_scatter_suite(out_dir)
            continue
        if name == "tile":
            run_tile_suite(out_dir)
            continue
        if name == "dtile":
            run_dtile_suite(out_dir)
            continue
        fname = {"decode": "decode_surface.csv", "prefill": "prefill_surface.csv",
                 "segm": "segm_sweep.csv", "td": "td_ab.csv",
                 "tdx": "tdx_ab.csv"}[name]
        csv_path = os.path.join(out_dir, fname)
        done = set() if args.fresh else load_done(csv_path)
        new_file = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            print(f"===== suite {name}: {len(SUITES[name])} cases "
                  f"({len(done)} done) -> {csv_path}", flush=True)
            for tag, S, q_len, kv_len, path, seg, td in SUITES[name]:
                run_case(writer, done, f, tag, S, q_len, kv_len, path, seg, td)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
