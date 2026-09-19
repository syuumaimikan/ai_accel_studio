"""Single-token giant fusion prototype.

For MHA-style layers (num_q_heads == num_kv_heads), each Triton program handles
one batch/head and performs in one kernel:
  groupwise-INT2 Q/K/V projection -> RoPE -> KV-cache append -> causal attention.

This is specifically for decode (one new token). It is intentionally conservative:
unsupported GQA/MQA shapes use the normal model path. The goal is to remove Python
and intermediate tensor traffic for the latency-critical decode step.
"""
from __future__ import annotations

import math
import torch
from .groupwise_int2 import GroupwiseInt2Weight

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE=True
except Exception:
    triton=None; tl=None; TRITON_AVAILABLE=False


if TRITON_AVAILABLE:
    @triton.jit
    def _dec(c):
        return tl.where(c==0,-1.0,tl.where(c==1,-0.3333333333333333,tl.where(c==2,0.3333333333333333,1.0)))

    @triton.jit
    def _fused_decode(
        x, qpack,kpack,vpack, qs,ks,vs,
        cos,sin,kcache,vcache,out,
        B: tl.constexpr, H: tl.constexpr, D: tl.constexpr, K: tl.constexpr,
        MAX_SEQ: tl.constexpr, POS: tl.constexpr, GROUP: tl.constexpr,
        sx0: tl.constexpr, sx1: tl.constexpr,
        sp0: tl.constexpr, sp1: tl.constexpr,
        ss0: tl.constexpr, ss1: tl.constexpr,
        skcb: tl.constexpr, skch: tl.constexpr, skct: tl.constexpr, skcd: tl.constexpr,
        socb: tl.constexpr, soch: tl.constexpr, socd: tl.constexpr,
        BLOCK_D: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_T: tl.constexpr,
    ):
        pid=tl.program_id(0)
        b=pid//H; h=pid-b*H
        d=tl.arange(0,BLOCK_D)
        dmask=d<D
        row=h*D+d
        q=tl.zeros((BLOCK_D,),tl.float32)
        kn=tl.zeros((BLOCK_D,),tl.float32)
        vn=tl.zeros((BLOCK_D,),tl.float32)
        rk=tl.arange(0,BLOCK_K)

        for k0 in range(0,K,BLOCK_K):
            kk=k0+rk
            xv=tl.load(x+b*sx0+kk*sx1,mask=kk<K,other=0.0).to(tl.float32)
            byte=kk[None,:]//4
            sh=(kk[None,:]&3)*2
            r=row[:,None]
            gid=kk[None,:]//GROUP
            pq=tl.load(qpack+r*sp0+byte*sp1,mask=dmask[:,None]&(kk[None,:]<K),other=0).to(tl.int32)
            pk=tl.load(kpack+r*sp0+byte*sp1,mask=dmask[:,None]&(kk[None,:]<K),other=0).to(tl.int32)
            pv=tl.load(vpack+r*sp0+byte*sp1,mask=dmask[:,None]&(kk[None,:]<K),other=0).to(tl.int32)
            sq=tl.load(qs+r*ss0+gid*ss1,mask=dmask[:,None]&(kk[None,:]<K),other=0.0)
            sk=tl.load(ks+r*ss0+gid*ss1,mask=dmask[:,None]&(kk[None,:]<K),other=0.0)
            sv=tl.load(vs+r*ss0+gid*ss1,mask=dmask[:,None]&(kk[None,:]<K),other=0.0)
            qw=_dec((pq>>sh)&3)*sq
            kw=_dec((pk>>sh)&3)*sk
            vw=_dec((pv>>sh)&3)*sv
            q += tl.sum(qw*xv[None,:],axis=1)
            kn += tl.sum(kw*xv[None,:],axis=1)
            vn += tl.sum(vw*xv[None,:],axis=1)

        # Llama-style rotate_half RoPE. Reshape into [2, half], transpose so the
        # final dimension has size 2, split, rotate, then restore the vector.
        half: tl.constexpr = BLOCK_D // 2
        qpair = q.reshape(2, half).permute(1, 0)
        kpair = kn.reshape(2, half).permute(1, 0)
        q0, q1 = qpair.split()
        k0, k1 = kpair.split()
        ridx = tl.arange(0, half)
        rmask = ridx < (D // 2)
        c = tl.load(cos + POS * (D // 2) + ridx, mask=rmask, other=1.0)
        s = tl.load(sin + POS * (D // 2) + ridx, mask=rmask, other=0.0)
        qr0 = q0 * c - q1 * s
        qr1 = q1 * c + q0 * s
        kr0 = k0 * c - k1 * s
        kr1 = k1 * c + k0 * s
        q = tl.join(qr0, qr1).permute(1, 0).reshape(BLOCK_D)
        kn = tl.join(kr0, kr1).permute(1, 0).reshape(BLOCK_D)

        tl.store(kcache+b*skcb+h*skch+POS*skct+d*skcd,kn,mask=dmask)
        tl.store(vcache+b*skcb+h*skch+POS*skct+d*skcd,vn,mask=dmask)

        # Numerically stable online softmax over causal history.
        m=-float('inf'); l=0.0
        acc=tl.zeros((BLOCK_D,),tl.float32)
        scale=1.0/math.sqrt(D)
        for t0 in range(0,POS+1,BLOCK_T):
            tt=t0+tl.arange(0,BLOCK_T)
            tmask=tt<=POS
            kval=tl.load(kcache+b*skcb+h*skch+tt[:,None]*skct+d[None,:]*skcd,
                         mask=tmask[:,None]&dmask[None,:],other=0.0).to(tl.float32)
            score=tl.sum(kval*q[None,:],axis=1)*scale
            score=tl.where(tmask,score,-float('inf'))
            mt=tl.max(score,axis=0)
            mn=tl.maximum(m,mt)
            alpha=tl.exp(m-mn)
            p=tl.exp(score-mn)
            lt=tl.sum(p,axis=0)
            vval=tl.load(vcache+b*skcb+h*skch+tt[:,None]*skct+d[None,:]*skcd,
                         mask=tmask[:,None]&dmask[None,:],other=0.0).to(tl.float32)
            acc=acc*alpha+tl.sum(p[:,None]*vval,axis=0)
            l=l*alpha+lt; m=mn
        acc=acc/l
        tl.store(out+b*socb+h*soch+d*socd,acc,mask=dmask)


def supported_giant_fusion(hidden_size:int,num_heads:int,num_kv_heads:int,head_dim:int)->tuple[bool,str]:
    if not TRITON_AVAILABLE:
        return False,"Triton unavailable"
    if num_heads != num_kv_heads:
        return False,"current giant fusion supports MHA; GQA/MQA falls back"
    if head_dim not in (64,128):
        return False,"head_dim must be 64 or 128"
    if hidden_size % 64:
        return False,"hidden size must be divisible by 64"
    return True,"supported"


def fused_decode_attention_int2(x:torch.Tensor,qw:GroupwiseInt2Weight,kw:GroupwiseInt2Weight,vw:GroupwiseInt2Weight,
                                cos:torch.Tensor,sin:torch.Tensor,kcache:torch.Tensor,vcache:torch.Tensor,
                                position:int,num_heads:int):
    if not TRITON_AVAILABLE or not x.is_cuda:
        raise RuntimeError("CUDA Triton required")
    if x.ndim!=2:
        raise ValueError("x must be [B,hidden]")
    b,k=x.shape; d=qw.n//num_heads
    ok,reason=supported_giant_fusion(k,num_heads,num_heads,d)
    if not ok: raise RuntimeError(reason)
    out=torch.empty((b,num_heads,d),device=x.device,dtype=x.dtype)
    bd=triton.next_power_of_2(d)
    bk=64
    bt=32
    _fused_decode[(b*num_heads,)](
        x,qw.packed,kw.packed,vw.packed,qw.scales,kw.scales,vw.scales,
        cos,sin,kcache,vcache,out,
        b,num_heads,d,k,kcache.shape[2],position,qw.group_size,
        x.stride(0),x.stride(1),
        qw.packed.stride(0),qw.packed.stride(1),
        qw.scales.stride(0),qw.scales.stride(1),
        kcache.stride(0),kcache.stride(1),kcache.stride(2),kcache.stride(3),
        out.stride(0),out.stride(1),out.stride(2),
        BLOCK_D=bd,BLOCK_K=bk,BLOCK_T=bt,num_warps=4,num_stages=3)
    return out

@torch.no_grad()
def reference_decode_attention_int2(x,qw,kw,vw,cos,sin,kcache,vcache,position,num_heads):
    from .groupwise_int2 import unpack_groupwise_int2
    import torch.nn.functional as F
    B,K=x.shape; D=qw.n//num_heads
    q=F.linear(x[:,:qw.original_k],unpack_groupwise_int2(qw,x.dtype)).view(B,num_heads,D)
    kn=F.linear(x[:,:kw.original_k],unpack_groupwise_int2(kw,x.dtype)).view(B,num_heads,D)
    vn=F.linear(x[:,:vw.original_k],unpack_groupwise_int2(vw,x.dtype)).view(B,num_heads,D)
    half=D//2; c=cos[position,:half].view(1,1,half); s=sin[position,:half].view(1,1,half)
    def rope(z):
        a,b=z[...,:half],z[...,half:]
        return torch.cat([a*c-b*s,b*c+a*s],dim=-1)
    q=rope(q); kn=rope(kn)
    kc=kcache.clone(); vc=vcache.clone(); kc[:,:,position,:]=kn; vc[:,:,position,:]=vn
    score=torch.einsum('bhd,bhtd->bht',q,kc[:,:,:position+1,:])/(D**.5)
    p=torch.softmax(score.float(),dim=-1).to(x.dtype)
    return torch.einsum('bht,bhtd->bhd',p,vc[:,:,:position+1,:])


def giant_fusion_self_test(hidden=256,num_heads=2,seq=8,group_size=64):
    if not TRITON_AVAILABLE: return {'ok':False,'reason':'Triton unavailable'}
    if not torch.cuda.is_available(): return {'ok':False,'reason':'CUDA unavailable'}
    d=hidden//num_heads
    ok,reason=supported_giant_fusion(hidden,num_heads,num_heads,d)
    if not ok: return {'ok':False,'reason':reason}
    try:
        from .groupwise_int2 import pack_groupwise_int2
        torch.manual_seed(11); dev='cuda'
        x=torch.randn(1,hidden,device=dev,dtype=torch.float16)
        ws=[pack_groupwise_int2(torch.randn(hidden,hidden,device=dev,dtype=torch.float16)/(hidden**.5),group_size) for _ in range(3)]
        pos=seq-1; half=d//2
        ang=torch.randn(seq,half,device=dev,dtype=torch.float32)
        cos=ang.cos().half(); sin=ang.sin().half()
        kc=torch.randn(1,num_heads,seq,d,device=dev,dtype=torch.float16); vc=torch.randn_like(kc)
        ref=reference_decode_attention_int2(x,*ws,cos,sin,kc,vc,pos,num_heads)
        kc2=kc.clone(); vc2=vc.clone()
        y=fused_decode_attention_int2(x,*ws,cos,sin,kc2,vc2,pos,num_heads)
        mse=float((y.float()-ref.float()).square().mean())
        return {'ok':bool(torch.allclose(y,ref,rtol=.06,atol=.25)),'mse':mse}
    except Exception as e: return {'ok':False,'reason':repr(e)}
