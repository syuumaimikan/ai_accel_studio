from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


if TRITON_AVAILABLE:
    @triton.jit
    def _tma_matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        # On NVIDIA architectures with TMA support, tensor descriptor loads/stores
        # are lowered to the hardware TMA path by Triton.
        ad = tl.make_tensor_descriptor(a, shape=[M,K], strides=[K,1], block_shape=[BM,BK])
        bd = tl.make_tensor_descriptor(b, shape=[K,N], strides=[N,1], block_shape=[BK,BN])
        cd = tl.make_tensor_descriptor(c, shape=[M,N], strides=[N,1], block_shape=[BM,BN])
        pm, pn = tl.program_id(0), tl.program_id(1)
        acc = tl.zeros((BM,BN), tl.float32)
        for k0 in range(0, K, BK):
            av = ad.load([pm*BM, k0])
            bv = bd.load([k0, pn*BN])
            acc += tl.dot(av, bv)
        cd.store([pm*BM, pn*BN], acc.to(tl.float16))

    @triton.jit
    def _persistent_tma_matmul_kernel(a, b, c, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                                      NUM_TILES: tl.constexpr,
                                      BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
        ad = tl.make_tensor_descriptor(a, shape=[M,K], strides=[K,1], block_shape=[BM,BK])
        bd = tl.make_tensor_descriptor(b, shape=[K,N], strides=[N,1], block_shape=[BK,BN])
        cd = tl.make_tensor_descriptor(c, shape=[M,N], strides=[N,1], block_shape=[BM,BN])
        wid = tl.program_id(0)
        workers = tl.num_programs(0)
        nn = tl.cdiv(N, BN)
        tile = wid
        while tile < NUM_TILES:
            pm = tile // nn
            pn = tile - pm*nn
            acc = tl.zeros((BM,BN), tl.float32)
            for k0 in range(0, K, BK):
                acc += tl.dot(ad.load([pm*BM,k0]), bd.load([k0,pn*BN]))
            cd.store([pm*BM,pn*BN], acc.to(tl.float16))
            tile += workers


def _check(a,b):
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not installed")
    if not (a.is_cuda and b.is_cuda):
        raise RuntimeError("TMA path requires CUDA")
    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise TypeError("TMA path currently expects fp16")
    major = torch.cuda.get_device_capability(a.device)[0]
    if major < 9:
        raise RuntimeError("TMA path requires Hopper or newer (SM90+)")
    if a.shape[1] != b.shape[0]:
        raise ValueError("incompatible matmul shapes")


def tma_matmul(a: torch.Tensor,b: torch.Tensor, persistent=False,
               block_m=64,block_n=128,block_k=64,num_stages=4):
    _check(a,b)
    m,k=a.shape; _,n=b.shape
    c=torch.empty((m,n),device=a.device,dtype=a.dtype)
    if persistent:
        tiles=triton.cdiv(m,block_m)*triton.cdiv(n,block_n)
        sms=torch.cuda.get_device_properties(a.device).multi_processor_count
        _persistent_tma_matmul_kernel[(min(sms,tiles),)](
            a,b,c,m,n,k,tiles,BM=block_m,BN=block_n,BK=block_k,
            num_warps=8,num_stages=num_stages)
    else:
        grid=(triton.cdiv(m,block_m),triton.cdiv(n,block_n))
        _tma_matmul_kernel[grid](a,b,c,m,n,k,BM=block_m,BN=block_n,BK=block_k,
                                 num_warps=8,num_stages=num_stages)
    return c
