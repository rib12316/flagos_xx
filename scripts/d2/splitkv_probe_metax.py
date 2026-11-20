# ⑥ split-KV decode probe（沐曦 C500 版）。
#
# 与 attn_surface_probe.run_splitkv_suite 同口径，但基线换锚为 MACA
# flash_attn_with_kvcache（沐曦现役 decode kernel），路由经 metax hop
# （vendor/metax/attn_splitkv_hop.py）。判据重锚：官方形态平均
# （bs32/40 × kv17.4K/16.4K）best-variant ≥1.10× vs MACA flash（原 1.15×
# 锚的是天数 kernel 1539µs，无跨卡继承性）。
#
# 真机命令：python3 scripts/d2/splitkv_probe_metax.py --variants 0,1,2,3 \
#             --out-dir results/20261119_d4_metax_port/item6_splitkv

import argparse
import csv
import os
import statistics
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import torch  # noqa: E402
import triton  # noqa: E402

import bottleneck_probe as bp  # noqa: E402

DEV = "cuda"
NH, NKV, HD, BLOCK = 16, 2, 128, 16
SKV_SHAPES = [(32, 16384), (32, 17408), (40, 16384), (40, 17408),
              (64, 16384), (64, 4096), (8, 4096)]  # 后三为 writethrough/小bs 参考点


def build_case(S, kv_len, seed=0):
    """MACA flash 口径数据（乱序页表）。closure 经模块属性取符号——hop 装载
    前后行为不同，装载后由 set_active 控制路由。"""
    g = torch.Generator(device=DEV).manual_seed(seed)
    # 4D [bs,1,NH,HD]：serve 经 reshape_query_for_spec_decod 的真实形态，
    # 且现版 flash_attn 2.6.3+metax 仅接受 4D（3D 直接 RuntimeError）
    q = torch.randn(S, 1, NH, HD, device=DEV, dtype=torch.bfloat16, generator=g)
    nb = triton.cdiv(kv_len, BLOCK) * S + 4
    kc = torch.randn(nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16,
                     generator=g)
    vc = torch.randn(nb, BLOCK, NKV, HD, device=DEV, dtype=torch.bfloat16,
                     generator=g)
    cols = triton.cdiv(kv_len, BLOCK)
    bt = torch.randperm(nb, device=DEV)[: S * cols].to(torch.int32).reshape(S, -1)
    su = torch.full((S,), kv_len, dtype=torch.int32, device=DEV)

    from vllm_fl.dispatch.backends.vendor.metax.impl.attention import (
        flash_attn as _fa,
    )

    def call():
        return _fa.flash_attn_with_kvcache(
            q=q, k_cache=kc, v_cache=vc, block_table=bt, cache_seqlens=su,
            softmax_scale=HD ** -0.5, causal=True, window_size=(-1, -1),
            alibi_slopes=None, softcap=0.0, s_aux=None)

    by = S * kv_len * 2 * NKV * HD * 2  # K+V 读字节
    return call, by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="0,1,2,3")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    free_b, total_b = torch.cuda.mem_get_info()
    assert (total_b - free_b) < 3 * 1024 ** 3, f"GPU busy: {(total_b-free_b)>>20} MiB"

    os.environ["VLLM_FL_ATTN_SPLITKV"] = "1"
    from vllm_fl.dispatch.backends.vendor.iluvatar import attn_splitkv as skv
    from vllm_fl.dispatch.backends.vendor.metax import attn_splitkv_hop as hop
    ok = hop.install()
    assert ok, "metax hop 未装载（VLLM_FL_ATTN_SPLITKV=1 ?）"

    prop = torch.cuda.get_device_properties(0)
    print(f"# {prop.name} SMs={prop.multi_processor_count} "
          f"hop installed, variants={args.variants}", flush=True)

    arms = [-1] + [int(v) for v in str(args.variants).split(",") if v.strip()]
    p = os.path.join(args.out_dir, "splitkv_probe_metax.csv")
    cols = ["tag", "S", "kv_len", "variant", "routed", "time_us", "GBps",
            "maxrel_vs_flash"]
    done = set()
    if os.path.exists(p):
        with open(p) as f:
            done = {r["tag"] for r in csv.DictReader(f)}
    res, parity = {}, {}
    new_file = not os.path.exists(p)
    with open(p, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        if new_file:
            writer.writeheader()
        for S, kv in SKV_SHAPES:
            call, by = build_case(S, kv, seed=S * 100 + kv)
            for arm in arms:
                tag = f"mkv bs{S} kv{kv} v{'base' if arm < 0 else arm}"
                if tag in done:
                    print(f"[skip] {tag}", flush=True)
                    continue
                if arm < 0:
                    skv.set_active(False)
                    skv.set_variant(None)
                else:
                    skv.set_active(True)
                    skv.set_variant(arm)
                r0 = hop._route_stats["routed"]
                try:
                    out = call()
                    torch.cuda.synchronize()
                    t = bp.graph_time_us(call, iters=30)
                    routed = hop._route_stats["routed"] > r0
                    row = dict(tag=tag, S=S, kv_len=kv,
                               variant=("base" if arm < 0 else arm),
                               routed=int(routed), time_us=f"{t:.1f}",
                               GBps=f"{by / t * 1e-3:.0f}", maxrel_vs_flash="")
                    if arm < 0:
                        parity[(S, kv)] = out.float()
                    else:
                        base = parity.get((S, kv))
                        if base is not None:
                            rel = ((out.float() - base).abs().max().item()
                                   / max(base.abs().max().item(), 1e-6))
                            row["maxrel_vs_flash"] = f"{rel:.2e}"
                    res[(S, kv, arm)] = t
                except Exception as e:
                    row = dict(tag=tag, S=S, kv_len=kv,
                               variant=("base" if arm < 0 else arm), routed=0,
                               time_us="", GBps="",
                               maxrel_vs_flash=f"ERROR {type(e).__name__}: "
                                               f"{str(e)[:90]}")
                finally:
                    skv.set_active(None)
                    skv.set_variant(None)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                print(f"[mkv] {tag:<26} {row['time_us']:>9}µs "
                      f"routed={row['routed']} rel={row['maxrel_vs_flash']}",
                      flush=True)
            del call
            torch.cuda.empty_cache()

    print("=" * 72, flush=True)
    official = [(32, 17408), (40, 16384), (40, 17408)]
    best = None
    for arm in arms:
        if arm < 0:
            continue
        base = [res.get((s, k, -1)) for s, k in official]
        cand = [res.get((s, k, arm)) for s, k in official]
        if all(x is not None for x in base) and all(x is not None for x in cand):
            mb, mc = statistics.mean(base), statistics.mean(cand)
            verdict = "PASS ≥1.10x" if mc <= mb / 1.10 else "FAIL"
            print(f"[T1-metax] variant={arm}: 官方形态平均 flash={mb:.1f}µs "
                  f"cand={mc:.1f}µs 加速 {mb / mc:.3f}x → {verdict}", flush=True)
            if best is None or mc < best[1]:
                best = (arm, mc)
    if best:
        print(f"BEST_VARIANT {best[0]}", flush=True)
    print("SPLITKV_METAX_DONE", flush=True)


if __name__ == "__main__":
    main()
