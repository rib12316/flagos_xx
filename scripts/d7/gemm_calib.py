import sys
import torch
import triton
import triton.language as tl
sys.path.insert(0, "scripts/d2")
import bottleneck_probe as bp


@triton.jit
def _gemm_loop(a_ptr, b_ptr, c_ptr, iters,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    offs_m = tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_k = tl.arange(0, BK)
    a = tl.load(a_ptr + offs_m[:, None] * BK + offs_k[None, :])
    acc = tl.zeros([BM, BN], dtype=tl.float32)
    for i in range(iters):
        b = tl.load(b_ptr + (i & 7) * 0 + offs_k[:, None] * BN + offs_n[None, :])
        acc += tl.dot(a, b)
    tl.store(c_ptr + offs_m[:, None] * BN + offs_n[None, :], acc.to(tl.bfloat16))


for BM, BN, BK, w, s in [(128, 32, 128, 8, 2), (128, 64, 128, 8, 1), (128, 64, 128, 8, 2),
                         (64, 64, 128, 4, 2), (128, 128, 64, 8, 1), (256, 64, 64, 8, 1),
                         (128, 256, 64, 8, 1)]:
    a = torch.randn(BM, BK, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(BK, BN, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(BM, BN, device="cuda", dtype=torch.bfloat16)
    iters = 1024
    try:
        def call():
            _gemm_loop[(104,)](a, b, c, iters, BM=BM, BN=BN, BK=BK,
                               num_warps=w, num_stages=s)
        call()
        torch.cuda.synchronize()
        t = bp.graph_time_us(call, iters=5)
        macs = 104 * iters * BM * BN * BK
        print(f"BM{BM} BN{BN} BK{BK} w{w} s{s}: {t:8.1f}µs → {macs/t*1e-6:5.0f} TFLOPS")
    except Exception as e:
        print(f"BM{BM} BN{BN} BK{BK} w{w} s{s}: {type(e).__name__} {str(e)[:50]}")
