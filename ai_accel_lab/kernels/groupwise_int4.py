from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE=True
except Exception:
    triton=None; tl=None; TRITON_AVAILABLE=False


@dataclass
class GroupwiseInt4Weight:
    packed: torch.Tensor       # uint8 [N,Kpad/2]
    scales: torch.Tensor       # fp16 [N,Kpad/group]
    original_k: int
    padded_k: int
    group_size: int
    activation_aware: bool=False

    @property
    def n(self): return int(self.packed.shape[0])
    @property
    def compression_bits_per_weight(self): return 4.0 + 16.0/self.group_size


def _pad(weight,group_size):
    if weight.ndim!=2: raise ValueError('weight must be [N,K]')
    if group_size%2: raise ValueError('group_size must be divisible by 2')
    n,k=weight.shape; kp=math.ceil(k/group_size)*group_size
    if kp==k:return weight,k
    out=torch.zeros((n,kp),device=weight.device,dtype=weight.dtype);out[:,:k]=weight
    return out,k


def pack_groupwise_int4(weight:torch.Tensor,group_size=128,activation_rms=None,refine_steps=11):
    w,original_k=_pad(weight.detach().float(),group_size);n,kp=w.shape;ng=kp//group_size
    wg=w.view(n,ng,group_size)
    base=(wg.abs().amax(-1)/7.0).clamp_min(1e-8)
    aware=activation_rms is not None
    if activation_rms is not None:
        ar=activation_rms.detach().float().flatten()
        if ar.numel()<original_k: raise ValueError('activation_rms shorter than K')
        ar=F.pad(ar[:original_k],(0,kp-original_k)) if kp>original_k else ar[:kp]
        aw=ar.view(1,ng,group_size).square().clamp_min(1e-8)
        factors=torch.linspace(.72,1.18,refine_steps,device=w.device)
        cand=base.unsqueeze(0)*factors[:,None,None]
        q=torch.round(wg.unsqueeze(0)/cand.unsqueeze(-1)).clamp(-8,7)
        err=((q*cand.unsqueeze(-1)-wg.unsqueeze(0)).square()*aw.unsqueeze(0)).sum(-1)
        best=err.argmin(0)
        scales=torch.gather(cand.permute(1,2,0),2,best.unsqueeze(-1)).squeeze(-1)
    else:scales=base
    q=torch.round(wg/scales.unsqueeze(-1)).clamp(-8,7).to(torch.int16)+8
    q=q.to(torch.uint8).view(n,kp//2,2)
    packed=(q[...,0] | (q[...,1]<<4)).contiguous()
    return GroupwiseInt4Weight(packed,scales.to(torch.float16).contiguous(),original_k,kp,group_size,aware)


def unpack_groupwise_int4(pw:GroupwiseInt4Weight,dtype=torch.float32):
    p=pw.packed.to(torch.uint8)
    codes=torch.stack([p&15,(p>>4)&15],-1).reshape(p.shape[0],-1)
    vals=(codes.to(torch.int16)-8).to(dtype)
    kidx=torch.arange(pw.padded_k,device=p.device); gids=kidx//pw.group_size
    vals=vals*pw.scales.to(dtype)[:,gids]
    return vals[:,:pw.original_k]


def reference_linear(x,pw,bias=None):
    return F.linear(x[...,:pw.original_k],unpack_groupwise_int4(pw,x.dtype),bias)


if TRITON_AVAILABLE:
    @triton.autotune(
        configs=[
            triton.Config({'BN':16,'BK':128},num_warps=4,num_stages=3),
            triton.Config({'BN':32,'BK':128},num_warps=4,num_stages=3),
            triton.Config({'BN':32,'BK':256},num_warps=8,num_stages=4),
            triton.Config({'BN':64,'BK':128},num_warps=8,num_stages=4),
        ],key=['N','K','GROUP_SIZE','OUTLIERS'])
    @triton.jit
    def _decode_int4_gemv(x,packed,scales,bias,out_idx,out_w,y,
        M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,GROUP_SIZE:tl.constexpr,OUTLIERS:tl.constexpr,
        sxm:tl.constexpr,sxk:tl.constexpr,spn:tl.constexpr,spb:tl.constexpr,ssn:tl.constexpr,ssg:tl.constexpr,
        sow_n:tl.constexpr,sow_o:tl.constexpr,sym:tl.constexpr,syn:tl.constexpr,HAS_BIAS:tl.constexpr,
        BN:tl.constexpr,BK:tl.constexpr):
        m=tl.program_id(0);pn=tl.program_id(1);rn=pn*BN+tl.arange(0,BN);nm=rn<N
        rk=tl.arange(0,BK);acc=tl.zeros((BN,),tl.float32)
        for k0 in range(0,K,BK):
            kk=k0+rk;km=kk<K
            xv=tl.load(x+m*sxm+kk*sxk,mask=km,other=0.0).to(tl.float32)
            byte=kk[None,:]//2;shift=(kk[None,:]&1)*4
            p=tl.load(packed+rn[:,None]*spn+byte*spb,mask=nm[:,None]&km[None,:],other=0).to(tl.int32)
            code=(p>>shift)&15; q=(code-8).to(tl.float32)
            gid=kk[None,:]//GROUP_SIZE
            sc=tl.load(scales+rn[:,None]*ssn+gid*ssg,mask=nm[:,None]&km[None,:],other=0.0).to(tl.float32)
            acc+=tl.sum(q*sc*xv[None,:],axis=1)
        if OUTLIERS>0:
            ro=tl.arange(0,OUTLIERS);idx=tl.load(out_idx+ro).to(tl.int32)
            ox=tl.load(x+m*sxm+idx*sxk,mask=idx<K,other=0.0).to(tl.float32)
            ow=tl.load(out_w+rn[:,None]*sow_n+ro[None,:]*sow_o,mask=nm[:,None],other=0.0).to(tl.float32)
            acc+=tl.sum(ow*ox[None,:],axis=1)
        if HAS_BIAS:acc+=tl.load(bias+rn,mask=nm,other=0.0)
        tl.store(y+m*sym+rn*syn,acc,mask=nm)


def decode_int4_linear(x:torch.Tensor,pw:GroupwiseInt4Weight,bias=None,outlier_idx=None,outlier_weight=None):
    if not TRITON_AVAILABLE or not x.is_cuda: raise RuntimeError('Triton CUDA required')
    if x.dtype not in (torch.float16,torch.bfloat16): raise RuntimeError('fp16/bf16 activations required')
    if x.ndim!=2 or x.shape[0]>4: raise ValueError('decode INT4 is specialized for M<=4')
    if x.shape[1]!=pw.padded_k: raise ValueError('input must be padded to packed K')
    m,k=x.shape;n=pw.n;y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    if bias is None: b=torch.empty(1,device=x.device,dtype=x.dtype);hb=False
    else:b=bias.to(device=x.device,dtype=x.dtype).contiguous();hb=True
    if outlier_idx is None or outlier_weight is None or outlier_idx.numel()==0:
        oi=torch.empty(1,device=x.device,dtype=torch.int32);ow=torch.empty((n,1),device=x.device,dtype=x.dtype);oc=0
    else:
        oi=outlier_idx.to(device=x.device,dtype=torch.int32).contiguous();ow=outlier_weight.to(device=x.device,dtype=x.dtype).contiguous();oc=int(oi.numel())
        if oc not in (4,8,16,32,64):raise ValueError('outlier count unsupported')
    grid=lambda meta:(m,triton.cdiv(n,meta['BN']))
    _decode_int4_gemv[grid](x,pw.packed,pw.scales,b,oi,ow,y,m,n,k,pw.group_size,oc,
        x.stride(0),x.stride(1),pw.packed.stride(0),pw.packed.stride(1),pw.scales.stride(0),pw.scales.stride(1),
        ow.stride(0),ow.stride(1),y.stride(0),y.stride(1),HAS_BIAS=hb)
    return y
