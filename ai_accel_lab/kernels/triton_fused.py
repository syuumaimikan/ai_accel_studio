from __future__ import annotations

import torch

from .packing import PackedWeight, PackedSparse24

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


ACT_NONE = 0
ACT_RELU = 1
ACT_SILU = 2


def _act_code(name: str) -> int:
    name = name.lower()
    if name in ('none', 'identity', ''): return ACT_NONE
    if name == 'relu': return ACT_RELU
    if name in ('silu', 'swish'): return ACT_SILU
    raise ValueError(f'unsupported activation: {name}')


if TRITON_AVAILABLE:
    def _configs():
        return [
            triton.Config({'BM': 16, 'BN': 32, 'BK': 64}, num_warps=4),
            triton.Config({'BM': 32, 'BN': 32, 'BK': 64}, num_warps=4),
            triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=8),
            triton.Config({'BM': 64, 'BN': 32, 'BK': 64}, num_warps=8),
            triton.Config({'BM': 32, 'BN': 64, 'BK': 128}, num_warps=8),
        ]

    @triton.jit
    def _epilogue(v, act: tl.constexpr):
        if act == ACT_RELU:
            v = tl.maximum(v, 0.0)
        elif act == ACT_SILU:
            v = v * tl.sigmoid(v)
        return v

    @triton.jit
    def _decode_int2(code):
        return tl.where(code == 0, -1.0,
               tl.where(code == 1, -0.3333333333333333,
               tl.where(code == 2,  0.3333333333333333, 1.0)))

    @triton.autotune(configs=_configs(), key=['M','N','K'])
    @triton.jit
    def _packed_int2_kernel(
        x_ptr, w_ptr, scale_ptr, bias_ptr, y_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        sxm: tl.constexpr, sxk: tl.constexpr,
        swn: tl.constexpr, swb: tl.constexpr,
        sym: tl.constexpr, syn: tl.constexpr,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k0 in range(0, K, BK):
            kk = k0 + rk
            x = tl.load(
                x_ptr + rm[:,None] * sxm + kk[None,:] * sxk,
                mask=(rm[:,None] < M) & (kk[None,:] < K), other=0.0,
            ).to(tl.float16)

            bidx = kk[:,None] // 4
            shift = (kk[:,None] & 3) * 2
            p = tl.load(
                w_ptr + rn[None,:] * swn + bidx * swb,
                mask=(rn[None,:] < N) & (kk[:,None] < K), other=0,
            )
            p = p.to(tl.int32)
            code = (p >> shift) & 3
            q = _decode_int2(code)
            sc = tl.load(scale_ptr + rn, mask=rn < N, other=0.0)
            w = (q * sc[None,:]).to(tl.float16)
            acc += tl.dot(x, w)

        if HAS_BIAS:
            b = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
            acc += b[None,:]
        acc = _epilogue(acc, ACT)
        tl.store(y_ptr + rm[:,None]*sym + rn[None,:]*syn, acc,
                 mask=(rm[:,None] < M) & (rn[None,:] < N))

    @triton.autotune(configs=_configs(), key=['M','N','K'])
    @triton.jit
    def _packed_ternary_kernel(
        x_ptr, w_ptr, scale_ptr, bias_ptr, y_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        sxm: tl.constexpr, sxk: tl.constexpr,
        swn: tl.constexpr, swb: tl.constexpr,
        sym: tl.constexpr, syn: tl.constexpr,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.float32)

        for k0 in range(0, K, BK):
            kk = k0 + rk
            x = tl.load(x_ptr + rm[:,None]*sxm + kk[None,:]*sxk,
                        mask=(rm[:,None]<M)&(kk[None,:]<K), other=0.0).to(tl.float16)
            bidx = kk[:,None] // 4
            shift = (kk[:,None] & 3) * 2
            p = tl.load(w_ptr + rn[None,:]*swn + bidx*swb,
                        mask=(rn[None,:]<N)&(kk[:,None]<K), other=0)
            p = p.to(tl.int32)
            code = (p >> shift) & 3
            q = tl.where(code == 1, 1.0, tl.where(code == 2, -1.0, 0.0))
            sc = tl.load(scale_ptr + rn, mask=rn<N, other=0.0)
            w = (q * sc[None,:]).to(tl.float16)
            acc += tl.dot(x, w)

        if HAS_BIAS:
            b = tl.load(bias_ptr + rn, mask=rn<N, other=0.0)
            acc += b[None,:]
        acc = _epilogue(acc, ACT)
        tl.store(y_ptr + rm[:,None]*sym + rn[None,:]*syn, acc,
                 mask=(rm[:,None]<M)&(rn[None,:]<N))

    # This kernel really performs only two weight multiplies for every four
    # original weights. It is optimized for small/medium M; for large dense GEMM
    # shapes, use the hardware Sparse Tensor Core backend below.
    @triton.autotune(
        configs=[
            triton.Config({'BM': 8, 'BN': 16, 'BG': 16}, num_warps=4),
            triton.Config({'BM': 16, 'BN': 16, 'BG': 16}, num_warps=4),
            triton.Config({'BM': 16, 'BN': 32, 'BG': 8}, num_warps=4),
        ],
        key=['M','N','G'],
    )
    @triton.jit
    def _sparse24_int2_kernel(
        x_ptr, p_ptr, scale_ptr, bias_ptr, y_ptr,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, G: tl.constexpr,
        sxm: tl.constexpr, sxk: tl.constexpr,
        spn: tl.constexpr, spg: tl.constexpr,
        sym: tl.constexpr, syn: tl.constexpr,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BG: tl.constexpr,
    ):
        pm = tl.program_id(0)
        pn = tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        sc = tl.load(scale_ptr + rn, mask=rn<N, other=0.0)

        # Each byte describes one 2:4 group for one output channel.
        for g0 in range(0, G, BG):
            for j in range(0, BG):
                g = g0 + j
                valid_g = g < G
                p = tl.load(p_ptr + rn * spn + g * spg,
                            mask=(rn<N) & valid_g, other=0)
                p = p.to(tl.int32)
                c0 = p & 3
                c1 = (p >> 2) & 3
                i0 = (p >> 4) & 3
                i1 = (p >> 6) & 3
                w0 = _decode_int2(c0) * sc
                w1 = _decode_int2(c1) * sc

                # idx varies by output channel, so this is an indexed sparse
                # gather. No multiply is issued for the two pruned positions.
                k0 = g * 4 + i0
                k1 = g * 4 + i1
                xv0 = tl.load(x_ptr + rm[:,None]*sxm + k0[None,:]*sxk,
                              mask=(rm[:,None]<M)&(rn[None,:]<N)&valid_g, other=0.0)
                xv1 = tl.load(x_ptr + rm[:,None]*sxm + k1[None,:]*sxk,
                              mask=(rm[:,None]<M)&(rn[None,:]<N)&valid_g, other=0.0)
                acc += xv0 * w0[None,:] + xv1 * w1[None,:]

        if HAS_BIAS:
            b = tl.load(bias_ptr + rn, mask=rn<N, other=0.0)
            acc += b[None,:]
        acc = _epilogue(acc, ACT)
        tl.store(y_ptr + rm[:,None]*sym + rn[None,:]*syn, acc,
                 mask=(rm[:,None]<M)&(rn[None,:]<N))


def _validate(x, packed, scale):
    if not TRITON_AVAILABLE:
        raise RuntimeError('Triton is not installed. Install requirements-gpu.txt')
    if not x.is_cuda:
        raise RuntimeError('Triton fused kernels require a CUDA tensor')
    if x.dtype != torch.float16:
        raise TypeError('optimized Triton path currently requires x.dtype=torch.float16')
    if not packed.is_cuda or not scale.is_cuda:
        raise RuntimeError('packed weight and scale must be on CUDA')
    if not x.is_contiguous() or not packed.is_contiguous():
        raise ValueError('x and packed weight must be contiguous')


def _bias_arg(x, bias):
    if bias is None:
        return torch.empty(1, device=x.device, dtype=x.dtype), False
    if bias.dtype != x.dtype or bias.device != x.device:
        bias = bias.to(device=x.device, dtype=x.dtype)
    return bias.contiguous(), True


def int2_linear(x: torch.Tensor, pw: PackedWeight, bias=None, activation='none'):
    _validate(x, pw.packed, pw.scale)
    if x.shape[1] != pw.padded_k:
        raise ValueError(f'x K must equal packed padded_k={pw.padded_k}; got {x.shape[1]}')
    m,k = x.shape; n = pw.packed.shape[0]
    y = torch.empty((m,n), device=x.device, dtype=x.dtype)
    b, has_bias = _bias_arg(x,bias)
    grid=lambda META:(triton.cdiv(m,META['BM']),triton.cdiv(n,META['BN']))
    _packed_int2_kernel[grid](x,pw.packed,pw.scale,b,y,m,n,k,
        x.stride(0),x.stride(1),pw.packed.stride(0),pw.packed.stride(1),
        y.stride(0),y.stride(1),HAS_BIAS=has_bias,ACT=_act_code(activation))
    return y


def ternary_linear(x: torch.Tensor, pw: PackedWeight, bias=None, activation='none'):
    _validate(x, pw.packed, pw.scale)
    if x.shape[1] != pw.padded_k:
        raise ValueError(f'x K must equal packed padded_k={pw.padded_k}; got {x.shape[1]}')
    m,k=x.shape; n=pw.packed.shape[0]
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    b,has_bias=_bias_arg(x,bias)
    grid=lambda META:(triton.cdiv(m,META['BM']),triton.cdiv(n,META['BN']))
    _packed_ternary_kernel[grid](x,pw.packed,pw.scale,b,y,m,n,k,
        x.stride(0),x.stride(1),pw.packed.stride(0),pw.packed.stride(1),
        y.stride(0),y.stride(1),HAS_BIAS=has_bias,ACT=_act_code(activation))
    return y


def int2_sparse24_linear(x: torch.Tensor, pw: PackedSparse24, bias=None, activation='none'):
    _validate(x,pw.packed,pw.scale)
    if x.shape[1] != pw.padded_k:
        raise ValueError(f'x K must equal packed padded_k={pw.padded_k}; got {x.shape[1]}')
    m,k=x.shape; n,g=pw.packed.shape
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    b,has_bias=_bias_arg(x,bias)
    grid=lambda META:(triton.cdiv(m,META['BM']),triton.cdiv(n,META['BN']))
    _sparse24_int2_kernel[grid](x,pw.packed,pw.scale,b,y,m,n,k,g,
        x.stride(0),x.stride(1),pw.packed.stride(0),pw.packed.stride(1),
        y.stride(0),y.stride(1),HAS_BIAS=has_bias,ACT=_act_code(activation))
    return y
