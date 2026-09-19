"""Experimental Blackwell warp-specialized GEMM.

This backend uses three partitions inside one CTA: a TMA producer, a tcgen05
consumer, and an epilogue/TMA-store worker. It is opt-in and guarded by self_test.
The generic runtime never depends on it for correctness.
"""
from __future__ import annotations
import torch

AVAILABLE=False; IMPORT_ERROR=None
try:
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from triton.experimental.gluon.language.nvidia.hopper import tma,mbarrier,fence_async_shared
    from triton.experimental.gluon.language.nvidia.blackwell import TensorMemoryLayout,allocate_tensor_memory,tcgen05_mma,tcgen05_commit
    AVAILABLE=True
except Exception as e: IMPORT_ERROR=repr(e)

if AVAILABLE:
    @gluon.aggregate
    class _Pipe:
        a_desc: object
        b_desc: object
        c_desc: object
        abuf: gl.shared_memory_descriptor
        bbuf: gl.shared_memory_descriptor
        empty: gl.shared_memory_descriptor
        ready: gl.shared_memory_descriptor
        acc: object
        done: gl.shared_memory_descriptor
        buffers: gl.constexpr

    @gluon.jit
    def _loader(p, om, on, K):
        bk: gl.constexpr=p.a_desc.block_type.shape[1]
        step=0
        for kk in range(0,K,bk):
            slot=step % p.buffers; phase=(step//p.buffers)&1
            mbarrier.wait(p.empty.index(slot),phase^1)
            bar=p.ready.index(slot)
            mbarrier.expect(bar,p.a_desc.block_type.nbytes+p.b_desc.block_type.nbytes)
            tma.async_load(p.a_desc,[om,kk],bar,p.abuf.index(slot))
            tma.async_load(p.b_desc,[kk,on],bar,p.bbuf.index(slot))
            step += 1

    @gluon.jit
    def _mma(p, om, on, K):
        bk: gl.constexpr=p.a_desc.block_type.shape[1]
        step=0; used=False
        for _kk in range(0,K,bk):
            slot=step%p.buffers; phase=(step//p.buffers)&1
            mbarrier.wait(p.ready.index(slot),phase)
            tcgen05_mma(p.abuf.index(slot),p.bbuf.index(slot),p.acc,use_acc=used)
            tcgen05_commit(p.empty.index(slot))
            used=True; step+=1
        tcgen05_commit(p.done)

    @gluon.jit
    def _writer(p,om,on,K):
        mbarrier.wait(p.done,0)
        tile=p.acc.load().to(p.c_desc.dtype)
        cbuf=gl.allocate_shared_memory(p.c_desc.dtype,p.c_desc.block_type.shape,p.c_desc.layout)
        cbuf.store(tile); fence_async_shared()
        tma.async_store(p.c_desc,[om,on],cbuf); tma.store_wait(0)

    @gluon.jit
    def _kernel(ad,bd,cd,buffers:gl.constexpr,num_warps:gl.constexpr):
        bm:gl.constexpr=cd.block_type.shape[0]; bn:gl.constexpr=cd.block_type.shape[1]
        pidm=gl.program_id(0); pidn=gl.program_id(1); om=pidm*bm; on=pidn*bn
        ab=gl.allocate_shared_memory(ad.dtype,[buffers]+ad.block_type.shape,ad.layout)
        bb=gl.allocate_shared_memory(bd.dtype,[buffers]+bd.block_type.shape,bd.layout)
        empty=gl.allocate_shared_memory(gl.int64,[buffers,1],mbarrier.MBarrierLayout())
        ready=gl.allocate_shared_memory(gl.int64,[buffers,1],mbarrier.MBarrierLayout())
        for i in gl.static_range(buffers):
            mbarrier.init(empty.index(i),count=1); mbarrier.init(ready.index(i),count=1)
        layout:gl.constexpr=TensorMemoryLayout([bm,bn],col_stride=1)
        acc=allocate_tensor_memory(gl.float32,[bm,bn],layout)
        done=gl.allocate_shared_memory(gl.int64,[1],mbarrier.MBarrierLayout()); mbarrier.init(done,count=1)
        p=_Pipe(ad,bd,cd,ab,bb,empty,ready,acc,done,buffers)
        gl.warp_specialize([
            (_mma,(p,om,on,ad.shape[1])),
            (_loader,(p,om,on,ad.shape[1])),
            (_writer,(p,om,on,ad.shape[1])),
        ],[1,1],[24,24])


def blackwell_warp_specialized_matmul(a,b,buffers=4):
    if not AVAILABLE: raise RuntimeError(f'Gluon unavailable: {IMPORT_ERROR}')
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10: raise RuntimeError('Blackwell SM100+ required')
    m,k=a.shape; _,n=b.shape; bm,bn,bk=128,256,64
    c=torch.empty((m,n),device=a.device,dtype=a.dtype)
    al=gl.NVMMASharedLayout.get_default_for([bm,bk],gl.float16); bl=gl.NVMMASharedLayout.get_default_for([bk,bn],gl.float16); cl=gl.NVMMASharedLayout.get_default_for([bm,bn],gl.float16)
    ad=TensorDescriptor.from_tensor(a,[bm,bk],al); bd=TensorDescriptor.from_tensor(b,[bk,bn],bl); cd=TensorDescriptor.from_tensor(c,[bm,bn],cl)
    _kernel[(triton.cdiv(m,bm),triton.cdiv(n,bn))](ad,bd,cd,buffers,4,num_warps=4,maxnreg=128)
    return c


def self_test():
    if not AVAILABLE: return {'ok':False,'reason':f'import failed: {IMPORT_ERROR}'}
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10: return {'ok':False,'reason':'Blackwell required'}
    try:
        a=torch.randn(128,128,device='cuda',dtype=torch.float16); b=torch.randn(128,256,device='cuda',dtype=torch.float16)
        y=blackwell_warp_specialized_matmul(a,b); ref=a@b
        return {'ok':bool(torch.allclose(y,ref,rtol=.03,atol=.3)),'mse':float((y.float()-ref.float()).square().mean())}
    except Exception as e: return {'ok':False,'reason':repr(e)}
