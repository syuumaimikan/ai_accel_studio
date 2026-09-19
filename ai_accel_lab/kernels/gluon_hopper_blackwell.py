"""
Optional Hopper/Blackwell backend built on Triton's experimental Gluon APIs.

The backend is intentionally isolated: the public self_test() function must pass
before the runtime enables it. This protects model execution from Triton/Gluon
API drift while still exposing TMA + asynchronous WGMMA/tcgen05 + persistent
multi-buffered execution on supported installations.
"""
from __future__ import annotations

import torch

GLUON_AVAILABLE = False
_IMPORT_ERROR = None

try:
    import triton
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
    from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
    from triton.experimental.gluon.language.nvidia.hopper import (
        tma, mbarrier, fence_async_shared,
        warpgroup_mma, warpgroup_mma_wait, warpgroup_mma_accumulator,
    )
    from triton.experimental.gluon.language.nvidia.blackwell import (
        TensorMemoryLayout, tensor_memory_descriptor, allocate_tensor_memory,
        tcgen05_mma, tcgen05_commit,
    )
    GLUON_AVAILABLE = True
except Exception as e:  # pragma: no cover - depends on GPU stack
    _IMPORT_ERROR = repr(e)


if GLUON_AVAILABLE:
    from typing import Union

    @gluon.aggregate
    class _HopperMMA:
        acc: Union[warpgroup_mma_accumulator, gl.tensor]
        used: gl.tensor

        @gluon.jit
        def create(dtype: gl.constexpr, bm: gl.constexpr, bn: gl.constexpr, warps: gl.constexpr):
            # The default blocked layout is accepted by current Gluon WGMMA lowering.
            layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 4], [4, 1], [1, 0])
            acc = gl.zeros((bm, bn), gl.float32, layout=layout)
            return _HopperMMA(acc, gl.to_tensor(False))

        @gluon.jit
        def issue(self, a, b):
            out = warpgroup_mma(a, b, self.acc, use_acc=self.used, is_async=True)
            return _HopperMMA(out, gl.to_tensor(True))

        @gluon.jit
        def drain(self):
            out = warpgroup_mma_wait(0, (self.acc,))
            return _HopperMMA(out, self.used)

        @gluon.jit
        def take(self):
            return self.acc

    @gluon.aggregate
    class _BlackwellMMA:
        acc: tensor_memory_descriptor
        bar: gl.shared_memory_descriptor
        used: gl.tensor
        phase: gl.tensor

        @gluon.jit
        def create(dtype: gl.constexpr, bm: gl.constexpr, bn: gl.constexpr, warps: gl.constexpr):
            tml: gl.constexpr = TensorMemoryLayout([bm, bn], col_stride=1)
            acc = allocate_tensor_memory(gl.float32, [bm, bn], tml)
            bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
            mbarrier.init(bar, count=1)
            return _BlackwellMMA(acc, bar, gl.to_tensor(False), gl.to_tensor(0))

        @gluon.jit
        def issue(self, a, b):
            tcgen05_mma(a, b, self.acc, use_acc=self.used)
            tcgen05_commit(self.bar)
            return _BlackwellMMA(self.acc, self.bar, gl.to_tensor(True), self.phase ^ 1)

        @gluon.jit
        def drain(self):
            mbarrier.wait(self.bar, self.phase ^ 1)
            return self

        @gluon.jit
        def take(self):
            return self.acc.load()

    @gluon.jit
    def _hb_kernel(a_desc, b_desc, c_desc, MMA: gl.constexpr,
                   buffers: gl.constexpr, warps: gl.constexpr):
        bm: gl.constexpr = c_desc.block_type.shape[0]
        bn: gl.constexpr = c_desc.block_type.shape[1]
        bk: gl.constexpr = a_desc.block_type.shape[1]
        dtype: gl.constexpr = a_desc.dtype
        K = a_desc.shape[1]

        # Each program persists across multiple output tiles.
        kid = gl.program_id(0)
        workers = gl.num_programs(0)
        nm = gl.cdiv(c_desc.shape[0], bm)
        nn = gl.cdiv(c_desc.shape[1], bn)
        total = nm * nn

        abufs = gl.allocate_shared_memory(dtype, [buffers] + a_desc.block_type.shape, a_desc.layout)
        bbufs = gl.allocate_shared_memory(dtype, [buffers] + b_desc.block_type.shape, b_desc.layout)
        ready = gl.allocate_shared_memory(gl.int64, [buffers,1], mbarrier.MBarrierLayout())
        for i in gl.static_range(buffers):
            mbarrier.init(ready.index(i), count=1)

        tile = kid
        while tile < total:
            pm = tile % nm
            pn = tile // nm
            om = pm * bm
            on = pn * bn
            mma = MMA.create(dtype,bm,bn,warps)
            producer = 0
            consumer = 0

            # Prime the ring; using >2 buffers on Blackwell lets loads overlap more deeply.
            prefetch = buffers - 1
            for kk in gl.static_range(0, bk * prefetch, bk):
                if kk < K:
                    idx = producer % buffers
                    bar = ready.index(idx)
                    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
                    tma.async_load(a_desc,[om,kk],bar,abufs.index(idx))
                    tma.async_load(b_desc,[kk,on],bar,bbufs.index(idx))
                    producer += 1

            for kk in range(bk * prefetch, K, bk):
                idxp = producer % buffers
                barp = ready.index(idxp)
                mbarrier.expect(barp, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
                tma.async_load(a_desc,[om,kk],barp,abufs.index(idxp))
                tma.async_load(b_desc,[kk,on],barp,bbufs.index(idxp))
                producer += 1

                idxc = consumer % buffers
                mbarrier.wait(ready.index(idxc), (consumer // buffers) & 1)
                mma = mma.issue(abufs.index(idxc), bbufs.index(idxc))
                consumer += 1

            for _ in gl.static_range(prefetch):
                idxc = consumer % buffers
                mbarrier.wait(ready.index(idxc), (consumer // buffers) & 1)
                mma = mma.issue(abufs.index(idxc), bbufs.index(idxc))
                consumer += 1

            mma = mma.drain()
            out = mma.take().to(dtype)
            cbuf = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
            cbuf.store(out)
            fence_async_shared()
            tma.async_store(c_desc,[om,on],cbuf)
            tma.store_wait(0)
            tile += workers


def _select_impl():
    major = torch.cuda.get_device_capability()[0]
    if major == 9:
        return _HopperMMA
    if major >= 10:
        return _BlackwellMMA
    return None


def persistent_tma_tensorcore_matmul(a: torch.Tensor,b: torch.Tensor,
                                     block_m=128,block_n=256,block_k=64,
                                     buffers: int | None=None):
    if not GLUON_AVAILABLE:
        raise RuntimeError(f"Triton Gluon unavailable: {_IMPORT_ERROR}")
    if not (a.is_cuda and b.is_cuda and a.dtype==torch.float16 and b.dtype==torch.float16):
        raise RuntimeError("Gluon HB path requires CUDA fp16")
    impl=_select_impl()
    if impl is None:
        raise RuntimeError("Hopper or newer GPU required")
    major=torch.cuda.get_device_capability()[0]
    buffers = buffers or (3 if major==9 else 4)
    warps = 8 if major==9 else 4
    m,k=a.shape; _,n=b.shape
    c=torch.empty((m,n),device=a.device,dtype=a.dtype)
    al=gl.NVMMASharedLayout.get_default_for([block_m,block_k],gl.float16)
    bl=gl.NVMMASharedLayout.get_default_for([block_k,block_n],gl.float16)
    cl=gl.NVMMASharedLayout.get_default_for([block_m,block_n],gl.float16)
    ad=TensorDescriptor.from_tensor(a,[block_m,block_k],al)
    bd=TensorDescriptor.from_tensor(b,[block_k,block_n],bl)
    cd=TensorDescriptor.from_tensor(c,[block_m,block_n],cl)
    sms=torch.cuda.get_device_properties(a.device).multi_processor_count
    tiles=triton.cdiv(m,block_m)*triton.cdiv(n,block_n)
    _hb_kernel[(min(sms,tiles),)](ad,bd,cd,impl,buffers,warps,num_warps=warps)
    return c


def self_test() -> dict:
    if not GLUON_AVAILABLE:
        return {"ok":False,"reason":f"import failed: {_IMPORT_ERROR}"}
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        return {"ok":False,"reason":"requires Hopper or newer"}
    try:
        torch.manual_seed(7)
        a=torch.randn(128,128,device='cuda',dtype=torch.float16)
        b=torch.randn(128,256,device='cuda',dtype=torch.float16)
        y=persistent_tma_tensorcore_matmul(a,b,128,256,64)
        ref=a@b
        err=(y.float()-ref.float()).pow(2).mean().item()
        ok=torch.allclose(y,ref,rtol=3e-2,atol=3e-1)
        return {"ok":bool(ok),"mse":err,"backend":"wgmma" if torch.cuda.get_device_capability()[0]==9 else "tcgen05"}
    except Exception as e:
        return {"ok":False,"reason":repr(e)}
