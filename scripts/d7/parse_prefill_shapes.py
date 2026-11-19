import gzip, json, collections

path = "results/20261120_c500_reprofiling/trace_all/rank0.1795110863960817196.pt.trace.json.gz"
with gzip.open(path, "rt") as f:
    data = json.load(f)

evs = data.get("traceEvents", [])
print(f"total events: {len(evs)}")

# flash prefill kernel 实例：名字含 flash_fwd_splitkv_kernel 且 traits 大 tile
prefill_hist = collections.Counter()
prefill_dur = collections.Counter()
decode_hist = collections.Counter()
for e in evs:
    if e.get("ph") != "X":
        continue
    name = e.get("name", "")
    if "flash_fwd_splitkv_kernel" not in name:
        continue
    args = e.get("args", {})
    grid = args.get("grid")
    dur = int(e.get("dur", 0))
    # traits 前 4 个模板参数（kHeadSize, kBlockM, kBlockN, kNWarps?）
    short = name.split("Flash_fwd_kernel_traits<")[1][:28]
    key = (short, tuple(grid) if grid else None)
    # 大 tile（128/64 M）为 prefill 候选；16 M 为 decode
    if ", 128, " in short[:22] or ", 64, " in short[:22]:
        prefill_hist[key] += 1
        prefill_dur[key] += dur
    elif ", 16, " in short[:22]:
        decode_hist[tuple(grid) if grid else None] += 1

print("\n== prefill 候选（大 tile）name片段 | grid | count | total_ms | avg_us ==")
for (short, grid), c in prefill_hist.most_common(20):
    t = prefill_dur[(short, grid)] / 1000
    print(f"{short:30} grid={str(grid):18} n={c:5} total={t:9.1f}ms avg={t/c*1000:7.1f}µs")
print("\n== decode（16 M）grid 直方图 top8 ==")
for grid, c in decode_hist.most_common(8):
    print(f"grid={grid} n={c}")
