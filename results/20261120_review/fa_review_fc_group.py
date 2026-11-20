"""Check exactly the GQA layout omitted from V1-V9 of the direct-out probe."""
import hashlib
import runpy
import sys
from pathlib import Path

import torch
import flash_attn_2_cuda as fa
from flash_attn import flash_attn_with_kvcache

old = runpy.run_path(str(Path(sys.argv[1]) / 'scripts/fc/probe_direct_out.py'))
q, kc, vc, pages, lengths = old['build']()
with open(fa.__file__, 'rb') as source:
    print('so md5', hashlib.file_digest(source, 'md5').hexdigest(), flush=True)
print('public q', q.shape, q.stride(), flush=True)
ref = flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, block_table=pages,
                              cache_seqlens=lengths, softmax_scale=old['SCALE'], causal=True)
dst = torch.full_like(q, torch.nan)
bs, _, hq, dim = q.shape
hk = kc.shape[-2]
view = dst.view(bs, hk, hq // hk, dim).transpose(1, 2)
print('internal out', view.shape, view.stride(), flush=True)

def call():
    return old['base_call'](fa, q, kc, vc, pages, lengths, view, None)[0]

out = call()
torch.cuda.synchronize()
print('returned', tuple(out.shape), out.stride(), 'aliases dst', out.data_ptr() == dst.data_ptr(),
      'target bitwise vs wrapper', torch.equal(ref.reshape_as(dst), dst), flush=True)
assert out.data_ptr() == dst.data_ptr() and torch.equal(ref.reshape_as(dst), dst)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    out_capture = call()
for value in (0, 1):
    dst.fill_(torch.nan)
    if value:
        q.neg_()
    expected = flash_attn_with_kvcache(q=q, k_cache=kc, v_cache=vc, block_table=pages,
                                       cache_seqlens=lengths, softmax_scale=old['SCALE'], causal=True)
    graph.replay()
    torch.cuda.synchronize()
    print('graph replay', value, 'bitwise', torch.equal(dst, expected.reshape_as(dst)),
          'aliases dst', dst.data_ptr() == out_capture.data_ptr(), flush=True)
    assert torch.equal(dst, expected.reshape_as(dst)) and dst.data_ptr() == out_capture.data_ptr()
