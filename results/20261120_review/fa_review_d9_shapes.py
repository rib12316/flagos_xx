"""Retest divisible shapes only, with graph timer and variants from 0231c70."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(sys.argv[1]).resolve() / 'scripts/d5'))
import d9_quant_probe_metax as d9
import torch

print('D9 0231c70 | torch', torch.__version__, '| M=64', flush=True)
for k in (2048, 12288):
    for repeat in (1, 2):
        torch.manual_seed(1234 + 64 + k + repeat - 1)
        x = (torch.randn(64, k, device='cuda') * 3).bfloat16()
        ref, rs = d9.ref_impl(x)
        o, s = torch.empty_like(ref), torch.empty_like(rs).reshape(-1)
        bare = d9.bench(lambda: d9.ref_impl(x))
        print('K', k, 'repeat', repeat, 'bare_us', round(bare, 3), flush=True)
        for bk in (1024, 2048, 4096):
            if k % bk:
                continue
            def run():
                d9._kloop_kernel[(64,)](x, o, s, 64, K=k, BLOCK_K=bk,
                                        num_warps=8 if bk >= 4096 else 4)
            run()
            ok = torch.equal(o, ref) and torch.equal(s.reshape_as(rs), rs)
            us = d9.bench(run) if ok else float('nan')
            print('  kloop', bk, 'bitwise', ok, 'us', round(us, 3), 'speedup', round(bare/us, 3), flush=True)
        for rows, bk in ((2, 2048), (4, 2048), (4, 1024), (8, 1024)):
            def run():
                d9._blocked_kernel[(64 // rows,)](x, o, s, 64, K=k, ROWS=rows, BLOCK_K=bk, num_warps=4)
            run()
            ok = torch.equal(o, ref) and torch.equal(s.reshape_as(rs), rs)
            us = d9.bench(run) if ok else float('nan')
            print('  blocked', rows, bk, 'bitwise', ok, 'us', round(us, 3), 'speedup', round(bare/us, 3), flush=True)
