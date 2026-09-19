from __future__ import annotations
import time
import torch
import torch.nn.functional as F
import pandas as pd

from .metrics import output_metrics
from .power import NvmlPowerSampler
from .kernels.groupwise_int2 import pack_groupwise_int2,triton_groupwise_int2_linear
from .kernels.decode_int2 import decode_int2_linear
from .kernels.groupwise_int4 import pack_groupwise_int4,decode_int4_linear
from .kernels.persistent_int2 import persistent_groupwise_int2_linear
from .kernels.tma_tensorcore import tma_matmul
from .kernels.sparse24_native import make_sparse_weight,linear as sparse_linear
from .kernels.gluon_hopper_blackwell import persistent_tma_tensorcore_matmul,self_test as gluon_self_test
from .kernels.gluon_warp_specialized import blackwell_warp_specialized_matmul,self_test as warp_self_test


def _sync():
    if torch.cuda.is_available(): torch.cuda.synchronize()


def _bench(fn,repeats=100,warmup=10,power=False):
    for _ in range(warmup): fn()
    _sync(); s=NvmlPowerSampler(0) if power else None
    if s: s.start()
    t=time.perf_counter(); y=None
    for _ in range(repeats): y=fn()
    _sync(); sec=time.perf_counter()-t
    watts=s.stop() if s else None
    ms=sec*1000/repeats
    return y,ms,watts


def run_kernel_suite(m=32,n=4096,k=4096,group_size=128,repeats=100,power=True,methods=None):
    if not torch.cuda.is_available():
        raise RuntimeError('Kernel Lab requires NVIDIA CUDA GPU')
    methods=methods or ['torch-fp16','decode-int2-v4','decode-int4-v5','groupwise-int2','persistent-int2','native-2to4','tma','hb-gluon','blackwell-warp-specialized']
    dev='cuda'; torch.manual_seed(3)
    x=torch.randn(m,k,device=dev,dtype=torch.float16)
    w=torch.randn(n,k,device=dev,dtype=torch.float16)/(k**.5)
    b=torch.randn(n,device=dev,dtype=torch.float16)*.01
    ref=F.linear(x,w,b)
    rows=[]

    def add(name,fn,bytes_weight=None,note=''):
        try:
            y,ms,pw=_bench(fn,repeats,power=power)
            met=output_metrics(ref,y)
            rows.append(dict(method=name,latency_ms=ms,speedup=None,avg_power_w=pw,
                             energy_mj=None if pw is None else pw*ms,
                             weight_mb=None if bytes_weight is None else bytes_weight/1024**2,
                             note=note,**met))
        except Exception as e:
            rows.append(dict(method=name,latency_ms=None,speedup=None,avg_power_w=None,energy_mj=None,
                             weight_mb=None,note=f'UNAVAILABLE: {e}',mse_vs_dense=None,mae_vs_dense=None,nmse_vs_dense=None,cosine_vs_dense=None))

    if 'torch-fp16' in methods:
        add('torch-fp16',lambda:F.linear(x,w,b),w.numel()*2)

    pw=pack_groupwise_int2(w,group_size)
    packed_bytes=pw.packed.numel()+pw.scales.numel()*2
    if 'decode-int2-v4' in methods and m <= 4:
        add('decode-int2-v4',lambda:decode_int2_linear(x,pw,b),packed_bytes,'M<=4 specialized packed INT2 decode GEMV')
    if 'decode-int4-v5' in methods and m <= 4:
        p4=pack_groupwise_int4(w,group_size); p4bytes=p4.packed.numel()+p4.scales.numel()*2
        add('decode-int4-v5',lambda:decode_int4_linear(x,p4,b),p4bytes,'M<=4 packed INT4 fallback for accuracy-sensitive layers')
    if 'groupwise-int2' in methods:
        add('groupwise-int2',lambda:triton_groupwise_int2_linear(x,pw,b),packed_bytes,'packed INT2 + per-group FP16 scale')
    if 'persistent-int2' in methods:
        add('persistent-int2',lambda:persistent_groupwise_int2_linear(x,pw,b),packed_bytes,'SM-resident persistent tile scheduler')
    if 'native-2to4' in methods:
        try:
            sw=make_sparse_weight(w)
            add('native-2to4',lambda:sparse_linear(x,sw,b),None,'PyTorch semi-structured/CUTLASS/cuSPARSELt')
        except Exception as e:
            rows.append(dict(method='native-2to4',latency_ms=None,speedup=None,avg_power_w=None,energy_mj=None,weight_mb=None,note=f'UNAVAILABLE: {e}',mse_vs_dense=None,mae_vs_dense=None,nmse_vs_dense=None,cosine_vs_dense=None))
    if 'tma' in methods:
        add('tma',lambda:tma_matmul(x,w.T.contiguous(),persistent=True)+b,w.numel()*2,'TMA tensor descriptors + persistent scheduler')
    if 'hb-gluon' in methods:
        st=gluon_self_test()
        if st.get('ok'):
            add('hb-gluon',lambda:persistent_tma_tensorcore_matmul(x,w.T.contiguous())+b,w.numel()*2,
                'TMA + async WGMMA/tcgen05 + multi-buffer persistent')
        else:
            rows.append(dict(method='hb-gluon',latency_ms=None,speedup=None,avg_power_w=None,energy_mj=None,weight_mb=None,note=f"UNAVAILABLE: {st.get('reason','self-test failed')}",mse_vs_dense=None,mae_vs_dense=None,nmse_vs_dense=None,cosine_vs_dense=None))

    if 'blackwell-warp-specialized' in methods:
        st=warp_self_test()
        if st.get('ok'):
            add('blackwell-warp-specialized',lambda:blackwell_warp_specialized_matmul(x,w.T.contiguous())+b,w.numel()*2,
                'Blackwell tcgen05 with dedicated TMA load/MMA/store warp partitions')
        else:
            rows.append(dict(method='blackwell-warp-specialized',latency_ms=None,speedup=None,avg_power_w=None,energy_mj=None,weight_mb=None,note=f"UNAVAILABLE: {st.get('reason','self-test failed')}",mse_vs_dense=None,mae_vs_dense=None,nmse_vs_dense=None,cosine_vs_dense=None))

    df=pd.DataFrame(rows)
    base=df.loc[df.method=='torch-fp16','latency_ms']
    if len(base) and pd.notna(base.iloc[0]):
        df['speedup']=float(base.iloc[0])/df['latency_ms']
    return df
