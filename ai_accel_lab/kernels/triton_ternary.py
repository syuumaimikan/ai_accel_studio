from __future__ import annotations
import time
import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE=True
except Exception:
    triton=None; tl=None; TRITON_AVAILABLE=False

if TRITON_AVAILABLE:
    @triton.jit
    def _kernel(x_ptr, w_ptr, scale_ptr, y_ptr, M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                sxm:tl.constexpr,sxk:tl.constexpr,swn:tl.constexpr,swk:tl.constexpr,sym:tl.constexpr,syn:tl.constexpr,
                BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
        pidm=tl.program_id(0); pidn=tl.program_id(1)
        rm=pidm*BM+tl.arange(0,BM); rn=pidn*BN+tl.arange(0,BN); rk=tl.arange(0,BK)
        acc=tl.zeros((BM,BN),tl.float32)
        for k0 in range(0,K,BK):
            xv=tl.load(x_ptr+rm[:,None]*sxm+(k0+rk[None,:])*sxk,mask=(rm[:,None]<M)&(k0+rk[None,:]<K),other=0.0)
            # W stored [N,K]; load transposed tile KxN.
            wi=tl.load(w_ptr+rn[None,:]*swn+(k0+rk[:,None])*swk,mask=(rn[None,:]<N)&(k0+rk[:,None]<K),other=0)
            sc=tl.load(scale_ptr+rn,mask=rn<N,other=0.0)
            wf=wi.to(tl.float16)*sc[None,:]
            acc+=tl.dot(xv,wf)
        tl.store(y_ptr+rm[:,None]*sym+rn[None,:]*syn,acc,mask=(rm[:,None]<M)&(rn[None,:]<N))


def ternary_matmul(x,qweight,scale):
    if not TRITON_AVAILABLE: raise RuntimeError('Triton is not installed')
    if not x.is_cuda: raise RuntimeError('Triton kernel requires CUDA')
    M,K=x.shape; N=qweight.shape[0]; y=torch.empty((M,N),device=x.device,dtype=x.dtype)
    grid=lambda META:(triton.cdiv(M,META['BM']),triton.cdiv(N,META['BN']))
    _kernel[grid](x,qweight,scale,y,M,N,K,x.stride(0),x.stride(1),qweight.stride(0),qweight.stride(1),y.stride(0),y.stride(1),BM=32,BN=32,BK=32)
    return y


def benchmark(m=256,n=4096,k=4096,repeats=100):
    if not (TRITON_AVAILABLE and torch.cuda.is_available()): return {'available':False,'reason':'CUDA + Triton required'}
    from ..quant import ternarize_weight
    x=torch.randn(m,k,device='cuda',dtype=torch.float16); w=torch.randn(n,k,device='cuda',dtype=torch.float16); q,s,_=ternarize_weight(w); q=q.cuda(); s=s.cuda().half()
    # warmup
    for _ in range(10):ternary_matmul(x,q,s)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(repeats):ternary_matmul(x,q,s)
    torch.cuda.synchronize(); tri=(time.perf_counter()-t)*1000/repeats
    wd=q.to(torch.float16)*s[:,None]
    for _ in range(10):torch.nn.functional.linear(x,wd)
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(repeats):torch.nn.functional.linear(x,wd)
    torch.cuda.synchronize(); dense=(time.perf_counter()-t)*1000/repeats
    return {'available':True,'triton_ternary_ms':tri,'torch_dequantized_dense_ms':dense,'speedup':dense/tri}
