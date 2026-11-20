# BLOCK_M 轴扫描（PR671 缺口验证）：prefill 2D 路径 BM×TILE×W×S 耦合网格 + decode TILE 轴。
#
# 机制：inspect 取 vllm unified_attention wrapper 源码 → 四处锚点定点替换（与 PR671
# unified_attn_launch.py 同思路）→ 在模块命名空间重 exec → 得到读模块级 _PROBE_CFG
# 的参数化 wrapper。锚点任一不唯一即整体放弃（无半补丁状态）。
#
# 口径：bp.graph_time_us（CUDA graph 3×N replay 中位，本仓唯一可信口径）。
# 污染防护：VLLM_FL_ATTN_LAUNCH/TILE/TD=0 进程级关闭插件 hook，upstream 臂纯净。
# 正确性：门=patched(cfg=0) vs 原 wrapper 逐位一致；各臂记录 vs upstream max|diff|。
# CSV 增量落盘（append+fsync）。每形状输入固定一份，参照只算一次。
import csv
import inspect
import os
import sys
import time

os.environ["VLLM_PLUGINS"] = "fl"
os.environ["VLLM_FL_ATTN_LAUNCH"] = "0"   # 关 alc setdefault，upstream 臂不被污染
os.environ["VLLM_FL_ATTN_TILE"] = "0"
os.environ["VLLM_FL_ATTN_TD"] = "0"

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "d2"))
sys.path.insert(0, os.path.join(_HERE, "..", "..", "vllm-plugin-FL"))

import torch  # noqa: E402
import triton  # noqa: E402
import bottleneck_probe as bp  # noqa: E402
import vllm.v1.attention.ops.triton_unified_attention as tua  # noqa: E402

DEV = "cuda"
NH, NKV, HD, BLOCK = bp.NH, bp.NKV, bp.HD, bp.BLOCK   # 16, 2, 128, 16
OUT_DIR = os.path.abspath(os.path.join(_HERE, "..", "..", "results",
                                       "20261120_blockm_sweep"))
CSV_PATH = os.path.join(OUT_DIR, "sweep.csv")
THRESH_SERVE = 64
DEFAULT_SEG = 16
ZERO = dict(bm=0, t=0, td=0, w=0, s=0)

COLUMNS = ["tag", "S", "q_len", "kv_len", "bm", "tile", "w", "s",
           "time_us", "GBps", "mac_pct", "vs_upstream", "vs_prod",
           "maxdiff", "verdict"]

# ---------------------------------------------------------------- 源码补丁
A_BLOCK = """    BLOCK_M = (
        16 if num_queries_per_kv <= 16 else triton.next_power_of_2(num_queries_per_kv)
    )
    BLOCK_Q = BLOCK_M // num_queries_per_kv"""
A_BLOCK_NEW = A_BLOCK + """
    if _PROBE_CFG["bm"] and max_seqlen_q > 1:
        BLOCK_M = _PROBE_CFG["bm"]
        BLOCK_Q = BLOCK_M // num_queries_per_kv"""

A_WARP = """    # Tuned launch parameters; ``None`` lets Triton pick its defaults.
    launch_num_warps: int | None = None
    launch_num_stages: int | None = None"""
A_WARP_NEW = A_WARP + """
    if _PROBE_CFG["w"]:
        launch_num_warps = _PROBE_CFG["w"]
        launch_num_stages = _PROBE_CFG["s"]"""

A_TILE = """    if tuned_large_head:
        TILE_SIZE_PREFILL = 128"""
A_TILE_NEW = A_TILE + """
    if _PROBE_CFG["t"] and max_seqlen_q > 1:
        TILE_SIZE_PREFILL = _PROBE_CFG["t"]"""

A_TILEDEC = """    TILE_SIZE_DECODE = _get_tile_size(
        head_size, sliding_window_val, q.element_size(), is_prefill=False
    )"""
A_TILEDEC_NEW = A_TILEDEC + """
    if _PROBE_CFG["td"]:
        TILE_SIZE_DECODE = _PROBE_CFG["td\"]"""


def build_patched_wrapper():
    orig = tua.unified_attention          # 必须在 exec 覆盖模块名之前抓取
    src = inspect.getsource(orig)
    patches = ((A_BLOCK, A_BLOCK_NEW), (A_WARP, A_WARP_NEW),
               (A_TILE, A_TILE_NEW), (A_TILEDEC, A_TILEDEC_NEW))
    for old, _ in patches:
        n = src.count(old)
        assert n == 1, f"anchor x{n} (need 1): {old[:60]!r}"
    for old, new in patches:
        src = src.replace(old, new)
    tua.__dict__["_PROBE_CFG"] = dict(ZERO)
    exec(compile(src, "<blockm_probe>", "exec"), tua.__dict__)
    fn = tua.__dict__["unified_attention"]
    tua.__dict__["unified_attention"] = orig   # 还原模块绑定，ORIG 干净
    return fn, tua.__dict__["_PROBE_CFG"], orig


UA, CFG, ORIG = build_patched_wrapper()


# ---------------------------------------------------------------- case 构造
class ShapeCase:
    """固定一份输入；out 由 wrapper 填充；ref = 原 wrapper(上游默认)输出。"""

    def __init__(self, S, q_len, kv_len):
        self.S, self.q_len, self.kv_len = S, q_len, kv_len
        decode = q_len == 1
        q = torch.randn(S * q_len, NH, HD, device=DEV, dtype=torch.bfloat16)
        nb = triton.cdiv(kv_len, BLOCK) * S + 4
        kv = torch.randn(2, nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16)
        bt = torch.arange(S * triton.cdiv(kv_len, BLOCK), dtype=torch.int32,
                          device=DEV).reshape(S, -1)
        cu = torch.arange(0, S * q_len + 1, q_len, dtype=torch.int32, device=DEV)
        su = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)
        self.out = torch.empty_like(q)
        self.kwargs = dict(softmax_scale=HD ** -0.5, causal=True,
                           window_size=(-1, -1), softcap=0.0, q_descale=None,
                           k_descale=None, v_descale=None, use_td=False)
        if decode:
            so = torch.empty(THRESH_SERVE, NH, DEFAULT_SEG, HD,
                             dtype=torch.float32, device=DEV)
            sm = torch.empty(THRESH_SERVE, NH, DEFAULT_SEG,
                             dtype=torch.float32, device=DEV)
            ss = torch.empty_like(sm)
            self.kwargs.update(seq_threshold_3D=THRESH_SERVE,
                               num_par_softmax_segments=DEFAULT_SEG,
                               softmax_segm_output=so, softmax_segm_max=sm,
                               softmax_segm_expsum=ss)
        self.args = (q, kv[0], kv[1])

        def wrap(fn, out):
            kk = dict(self.kwargs)

            def call():
                fn(*self.args, out, cu_seqlens_q=cu, max_seqlen_q=q_len,
                   seqused_k=su, max_seqlen_k=kv_len, block_table=bt, **kk)
                return out
            return call

        self._buf = torch.empty_like(self.out)
        self.call_orig = wrap(ORIG, self._buf)
        self.call = wrap(UA, self.out)
        # 参照：原 wrapper、上游默认参数
        self.call_orig()
        torch.cuda.synchronize()
        self.ref = self._buf.clone()
        self.decode = decode

    def time_us(self):
        for _ in range(2):
            self.call()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.call()
        torch.cuda.synchronize()
        rough = (time.perf_counter() - t0) * 1e6
        iters = int(max(5, min(30, 1.5e6 / max(rough, 1.0))))
        us = bp.graph_time_us(self.call, iters=iters)
        self.call()
        torch.cuda.synchronize()
        self.maxdiff = (self.out - self.ref).abs().max().item()
        return us


def macs_of(sc):
    return 2 * sc.S * sc.q_len * (sc.kv_len // 2) * NH * HD / 1e12  # T-MAC


def run_case(w, tag, sc, cfg, b_up, b_prod):
    CFG.update(ZERO); CFG.update(cfg)
    try:
        us = sc.time_us()
    except Exception as e:
        w.writerow([tag, sc.S, sc.q_len, sc.kv_len, cfg.get("bm", 0),
                    cfg.get("t", 0) or cfg.get("td", 0), cfg.get("w", 0),
                    cfg.get("s", 0), "", "", "", "", "", "",
                    f"ERR:{str(e)[:70]}"])
        return None
    by = sc.S * sc.kv_len * 2 * NKV * HD * 2
    row = [tag, sc.S, sc.q_len, sc.kv_len, cfg.get("bm", 0),
           cfg.get("t", 0) or cfg.get("td", 0), cfg.get("w", 0), cfg.get("s", 0),
           round(us, 1), round(by / us / 1e3, 1),
           round(macs_of(sc) / (us * 1e-6) / bp.MAC_CEIL * 100, 2),
           round(b_up / us, 3) if b_up else "",
           round(b_prod / us, 3) if b_prod else "",
           round(sc.maxdiff, 5), "OK"]
    w.writerow(row)
    return us


def main():
    print(f"[env] torch={torch.__version__} triton={triton.__version__}", flush=True)

    # 正确性门：patched(cfg=0) vs 原 wrapper 逐位一致
    CFG.update(ZERO)
    gate = ShapeCase(2, 512, 2048)
    gate.call(); torch.cuda.synchronize()
    d = (gate.out - gate.ref).abs().max().item()
    print(f"[gate] patched-default vs upstream max|diff| = {d}", flush=True)
    assert d == 0.0, "patched wrapper with cfg=0 must be bit-exact vs upstream"

    abl = {
        "upstream":  dict(ZERO),
        "prod_B":    dict(bm=0, t=16, w=2, s=1),
        "pr671":     dict(bm=256, t=128, w=16, s=2),
        "pr671_t32": dict(bm=256, t=32, w=16, s=2),
        "pr671_w4":  dict(bm=256, t=128, w=4, s=2),
        "pr671_s1":  dict(bm=256, t=128, w=16, s=1),
    }
    grid = []
    for bm in (64, 128, 256):
        for t in (32, 64, 128):
            for wp in ((4, 8) if bm == 64 else (8, 16)):
                for s in (1, 2):
                    grid.append((f"bm{bm}_t{t}_w{wp}_s{s}",
                                 dict(bm=bm, t=t, w=wp, s=s)))
    dec = [("decup", dict(ZERO))] + [
        (f"dect{td}_w{wp}_s1", dict(td=td, w=wp, s=1))
        for td in (16, 32, 64) for wp in (1, 2)]

    f = open(CSV_PATH, "w", newline="")
    w = csv.writer(f)
    w.writerow(COLUMNS); f.flush(); os.fsync(f.fileno())

    # 段1：消融臂 × 4 prefill 形状
    cases = {n: ShapeCase(S, q, k) for n, S, q, k in
             [("P1", 1, 4096, 16384), ("P2", 2, 2048, 16384),
              ("P4", 4, 4096, 4096), ("PD", 1, 8192, 16384)]}
    base = {}
    for name, sc in cases.items():
        b_up = b_prod = None
        for tag, cfg in abl.items():
            us = run_case(w, tag, sc, cfg, b_up, b_prod)
            if us is None:
                print(f"[abl] {name} {tag}: ERR", flush=True)
                continue
            if tag == "upstream":
                b_up = us
            elif tag == "prod_B":
                b_prod = us
            print(f"[abl] {name} {tag}: {us:.1f}us "
                  f"(x{b_up/us:.2f} vs up)" if b_up else
                  f"[abl] {name} {tag}: {us:.1f}us", flush=True)
            f.flush(); os.fsync(f.fileno())
        base[name] = (b_up, b_prod)

    # 段2：网格扫（16K 主形状 P2/PD）
    for name in ("P2", "PD"):
        sc = cases[name]
        b_up, b_prod = base[name]
        for tag, cfg in grid:
            us = run_case(w, tag, sc, cfg, b_up, b_prod)
            print(f"[grid] {name} {tag}: "
                  f"{us:.1f}us (x{b_up/us:.2f})" if us else
                  f"[grid] {name} {tag}: ERR", flush=True)
            f.flush(); os.fsync(f.fileno())

    # 段3：decode TILE 轴
    for name, S, q, k in [("D64", 64, 1, 16384), ("D32", 32, 1, 16384)]:
        sc = ShapeCase(S, q, k)
        b_up = None
        for tag, cfg in dec:
            us = run_case(w, tag, sc, cfg, b_up, None)
            if us is None:
                print(f"[dec] {name} {tag}: ERR", flush=True)
                continue
            if tag == "decup":
                b_up = us
            print(f"[dec] {name} {tag}: {us:.1f}us", flush=True)
            f.flush(); os.fsync(f.fileno())

    f.close()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
