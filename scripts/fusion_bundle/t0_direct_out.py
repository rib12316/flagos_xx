"""Verify the production GQA direct-out helper on a sliced caller buffer."""
import torch
from flash_attn import flash_attn_with_kvcache
from vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn import _grouped_decode_out


def check(n):
    torch.manual_seed(17)
    q = torch.randn(n, 1, 16, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(n * 8, 16, 2, 128, device="cuda", dtype=q.dtype)
    v = torch.randn_like(k)
    pages = torch.arange(n * 8, device="cuda", dtype=torch.int32).view(n, 8)
    lengths = torch.full((n,), 127, device="cuda", dtype=torch.int32)
    storage = q.new_full((n + 2, 16, 128), -19)
    dst = storage[1:-1]
    def call():
        _grouped_decode_out(q, k, v, pages, lengths, 128 ** -.5, dst)
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for _ in range(2):
        q.neg_()
        dst.fill_(torch.nan)
        graph.replay()
        ref = flash_attn_with_kvcache(q=q, k_cache=k, v_cache=v, block_table=pages,
                                      cache_seqlens=lengths, causal=True, softmax_scale=128 ** -.5)
        assert torch.equal(dst, ref.reshape_as(dst))
        assert (storage[0] == -19).all() and (storage[-1] == -19).all()
    print(f"batch={n} direct output bitwise, both replays and slice guards passed", flush=True)


if __name__ == "__main__":
    for n in (1, 32, 64):
        check(n)
