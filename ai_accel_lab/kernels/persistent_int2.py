from __future__ import annotations

import torch
from .groupwise_int2 import GroupwiseInt2Weight

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
    def _decode(c):
        return tl.where(c == 0, -1.0,
               tl.where(c == 1, -0.3333333333333333,
               tl.where(c == 2,  0.3333333333333333, 1.0)))

    @triton.jit
    def _persistent_int2_kernel(
        x, packed, scales, bias, y,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        GROUP_SIZE: tl.constexpr, NUM_TILES: tl.constexpr,
        sxm: tl.constexpr, sxk: tl.constexpr,
        spn: tl.constexpr, spb: tl.constexpr,
        sgn: tl.constexpr, sgg: tl.constexpr,
        sym: tl.constexpr, syn: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        worker = tl.program_id(0)
        workers = tl.num_programs(0)
        num_n = tl.cdiv(N, BN)
        rk = tl.arange(0, BK)

        tile = worker
        while tile < NUM_TILES:
            pm = tile // num_n
            pn = tile - pm * num_n
            rm = pm * BM + tl.arange(0, BM)
            rn = pn * BN + tl.arange(0, BN)
            acc = tl.zeros((BM, BN), tl.float32)

            for k0 in range(0, K, BK):
                kk = k0 + rk
                xv = tl.load(x + rm[:,None]*sxm + kk[None,:]*sxk,
                    mask=(rm[:,None]<M)&(kk[None,:]<K), other=0.0).to(tl.float16)
                bidx = kk[:,None] // 4
                shift = (kk[:,None] & 3) * 2
                p = tl.load(packed + rn[None,:]*spn + bidx*spb,
                    mask=(rn[None,:]<N)&(kk[:,None]<K), other=0).to(tl.int32)
                code = (p >> shift) & 3
                gid = kk[:,None] // GROUP_SIZE
                sc = tl.load(scales + rn[None,:]*sgn + gid*sgg,
                    mask=(rn[None,:]<N)&(kk[:,None]<K), other=0.0)
                w = (_decode(code) * sc).to(tl.float16)
                acc += tl.dot(xv, w)

            if HAS_BIAS:
                acc += tl.load(bias + rn, mask=rn<N, other=0.0)[None,:]
            tl.store(y + rm[:,None]*sym + rn[None,:]*syn, acc,
                     mask=(rm[:,None]<M)&(rn[None,:]<N))
            tile += workers


def persistent_groupwise_int2_linear(
    x: torch.Tensor,
    pw: GroupwiseInt2Weight,
    bias=None,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    num_stages: int = 4,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton not available")
    if not x.is_cuda or x.dtype != torch.float16 or x.ndim != 2:
        raise RuntimeError("persistent INT2 requires 2-D CUDA fp16 input")
    m,k = x.shape
    if k != pw.padded_k:
        raise ValueError("input K must match padded packed weight")
    n = pw.n
    y = torch.empty((m,n), device=x.device, dtype=x.dtype)
    props = torch.cuda.get_device_properties(x.device)
    num_tiles = triton.cdiv(m, block_m) * triton.cdiv(n, block_n)
    grid = (min(props.multi_processor_count, num_tiles),)
    if bias is None:
        b = torch.empty(1, device=x.device, dtype=x.dtype); has_bias=False
    else:
        b = bias.to(device=x.device,dtype=x.dtype).contiguous(); has_bias=True
    _persistent_int2_kernel[grid](
        x,pw.packed,pw.scales,b,y,
        m,n,k,pw.group_size,num_tiles,
        x.stride(0),x.stride(1),
        pw.packed.stride(0),pw.packed.stride(1),
        pw.scales.stride(0),pw.scales.stride(1),
        y.stride(0),y.stride(1),
        HAS_BIAS=has_bias,
        BM=block_m,BN=block_n,BK=block_k,
        num_warps=8 if block_n >= 64 else 4,
        num_stages=num_stages,
    )
    return y
