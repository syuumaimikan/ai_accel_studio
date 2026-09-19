from __future__ import annotations

"""Decode-specialized groupwise INT2 GEMV/GEMM-small-M kernel.

The generic INT2 kernel in v3 tiled M by 16/32 even for autoregressive M=1.
This module uses one program per (token, output tile), so M=1/2/4 does not pay
for inactive rows. Packed 2-bit decode, per-group FP16 scaling, optional exact
FP16 outlier-column correction, bias and activation happen in the same kernel.

This is a storage/dequant fusion path, not a claim of native INT2 Tensor Core
execution. Hopper/Blackwell native low-precision paths are exposed separately.
"""

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE=True
except Exception:
    triton=None; tl=None; TRITON_AVAILABLE=False

from .groupwise_int2 import GroupwiseInt2Weight, reference_linear

if TRITON_AVAILABLE:
    @triton.jit
    def _decode(c):
        return tl.where(c == 0, -1.0,
               tl.where(c == 1, -0.3333333333333333,
               tl.where(c == 2,  0.3333333333333333, 1.0)))

    @triton.autotune(
        configs=[
            triton.Config({'BN':16,'BK':128},num_warps=4,num_stages=3),
            triton.Config({'BN':32,'BK':128},num_warps=4,num_stages=3),
            triton.Config({'BN':32,'BK':256},num_warps=8,num_stages=4),
            triton.Config({'BN':64,'BK':128},num_warps=8,num_stages=4),
        ],
        key=['N','K','GROUP_SIZE','OUTLIERS'],
    )
    @triton.jit
    def _decode_gemv(
        x, packed, scales, bias, out_idx, out_w, y,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        GROUP_SIZE: tl.constexpr, OUTLIERS: tl.constexpr,
        sxm: tl.constexpr, sxk: tl.constexpr,
        spn: tl.constexpr, spb: tl.constexpr,
        ssn: tl.constexpr, ssg: tl.constexpr,
        sow_n: tl.constexpr, sow_o: tl.constexpr,
        sym: tl.constexpr, syn: tl.constexpr,
        HAS_BIAS: tl.constexpr, ACT: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr,
    ):
        m = tl.program_id(0)
        pn = tl.program_id(1)
        rn = pn * BN + tl.arange(0, BN)
        nmask = rn < N
        rk = tl.arange(0, BK)
        acc = tl.zeros((BN,), tl.float32)

        # Decode directly from packed HBM representation. No FP16 weight tensor
        # is materialized in global memory.
        for k0 in range(0, K, BK):
            kk = k0 + rk
            kmask = kk < K
            xv = tl.load(x + m*sxm + kk*sxk, mask=kmask, other=0.0).to(tl.float32)
            byte = kk[None,:] // 4
            shift = (kk[None,:] & 3) * 2
            p = tl.load(packed + rn[:,None]*spn + byte*spb,
                        mask=nmask[:,None] & kmask[None,:], other=0).to(tl.int32)
            code = (p >> shift) & 3
            gid = kk[None,:] // GROUP_SIZE
            sc = tl.load(scales + rn[:,None]*ssn + gid*ssg,
                         mask=nmask[:,None] & kmask[None,:], other=0.0).to(tl.float32)
            w = _decode(code) * sc
            acc += tl.sum(w * xv[None,:], axis=1)

        # Important activation-sensitive columns are kept exactly in FP16 and
        # corrected inside this same launch. Their base INT2 columns are zeroed
        # before packing, so this is not double-counted.
        if OUTLIERS > 0:
            ro = tl.arange(0, OUTLIERS)
            idx = tl.load(out_idx + ro).to(tl.int32)
            ox = tl.load(x + m*sxm + idx*sxk, mask=idx < K, other=0.0).to(tl.float32)
            ow = tl.load(out_w + rn[:,None]*sow_n + ro[None,:]*sow_o,
                         mask=nmask[:,None], other=0.0).to(tl.float32)
            acc += tl.sum(ow * ox[None,:], axis=1)

        if HAS_BIAS:
            acc += tl.load(bias + rn, mask=nmask, other=0.0)
        if ACT == 1:
            acc = tl.maximum(acc, 0.0)
        elif ACT == 2:
            acc = acc * tl.sigmoid(acc)
        tl.store(y + m*sym + rn*syn, acc, mask=nmask)


def decode_int2_linear(
    x: torch.Tensor,
    pw: GroupwiseInt2Weight,
    bias: torch.Tensor | None = None,
    outlier_idx: torch.Tensor | None = None,
    outlier_weight: torch.Tensor | None = None,
    activation: str = 'none',
) -> torch.Tensor:
    if not TRITON_AVAILABLE or not x.is_cuda:
        raise RuntimeError('Triton CUDA is required')
    if x.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError('decode INT2 path requires fp16/bf16 activations')
    if x.ndim != 2 or x.shape[0] > 4:
        raise ValueError('decode-specialized path is for [M,K] with M<=4')
    if x.shape[1] != pw.padded_k:
        raise ValueError('input must already be padded to packed K')
    m,k=x.shape; n=pw.n
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    if bias is None:
        bias_arg=torch.empty(1,device=x.device,dtype=x.dtype); has_bias=False
    else:
        bias_arg=bias.to(device=x.device,dtype=x.dtype).contiguous(); has_bias=True
    if outlier_idx is None or outlier_weight is None or outlier_idx.numel()==0:
        oi=torch.empty(1,device=x.device,dtype=torch.int32)
        ow=torch.empty((n,1),device=x.device,dtype=x.dtype)
        outliers=0
    else:
        oi=outlier_idx.to(device=x.device,dtype=torch.int32).contiguous()
        ow=outlier_weight.to(device=x.device,dtype=x.dtype).contiguous()
        outliers=int(oi.numel())
        if outliers not in (4,8,16,32,64):
            raise ValueError('fused outlier count must be one of 4/8/16/32/64')
    act={'none':0,'relu':1,'silu':2,'swish':2}.get(activation)
    if act is None: raise ValueError(activation)
    grid=lambda meta:(m,triton.cdiv(n,meta['BN']))
    _decode_gemv[grid](
        x,pw.packed,pw.scales,bias_arg,oi,ow,y,
        m,n,k,pw.group_size,outliers,
        x.stride(0),x.stride(1),
        pw.packed.stride(0),pw.packed.stride(1),
        pw.scales.stride(0),pw.scales.stride(1),
        ow.stride(0),ow.stride(1),
        y.stride(0),y.stride(1),
        HAS_BIAS=has_bias,ACT=act,
    )
    return y


def reference_hybrid_linear(x,pw,bias=None,outlier_idx=None,outlier_weight=None):
    y=reference_linear(x,pw,bias)
    if outlier_idx is not None and outlier_weight is not None and outlier_idx.numel():
        y=y+F.linear(x[...,outlier_idx.long()],outlier_weight.to(x.dtype))
    return y
