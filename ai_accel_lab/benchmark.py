from __future__ import annotations
from dataclasses import dataclass, replace
import time
import pandas as pd
import torch
from .utils import resolve_device, sync, seed_all
from .layers import *
from .metrics import output_metrics
from .power import NvmlPowerSampler


@dataclass(frozen=True)
class LinearPreset:
    batch:int; in_features:int; out_features:int; warmup:int; repeats:int

LINEAR_PRESETS={
    'small':LinearPreset(16,512,512,10,50),
    'medium':LinearPreset(32,2048,2048,10,50),
    'large':LinearPreset(64,4096,4096,10,30),
}


def make_weight(preset,device,dtype=torch.float32,seed=1234):
    g=torch.Generator(device='cpu'); g.manual_seed(seed)
    w=torch.randn(preset.out_features,preset.in_features,generator=g)/preset.in_features**0.5
    b=torch.randn(preset.out_features,generator=g)*0.01
    return w.to(device=device,dtype=dtype), b.to(device=device,dtype=dtype)


def _bench(name,module,x,ref,preset,device,power=False):
    module.eval()
    with torch.no_grad():
        for _ in range(preset.warmup): _=module(x)
        sync(device)
        sampler=NvmlPowerSampler() if power and device.type=='cuda' else None
        if sampler: sampler.start()
        t=time.perf_counter(); y=None
        for _ in range(preset.repeats): y=module(x)
        sync(device); elapsed=time.perf_counter()-t
        avgp=sampler.stop() if sampler else None
    ms=elapsed*1000/preset.repeats
    items=x.numel()//x.shape[-1]
    macs=module.theoretical_macs(items)
    dense_macs=items*preset.in_features*preset.out_features
    energy=None if avgp is None else avgp*ms
    row={'method':name,'latency_ms':ms,'theoretical_macs':macs,'mac_reduction':1-macs/dense_macs,
         'estimated_weight_bytes':module.estimated_weight_bytes(),'avg_power_w':avgp,
         'energy_mj_per_inference':energy,'edp_mj_ms':None if energy is None else energy*ms}
    row.update(output_metrics(ref,y)); return row


def build_linear_methods(w,b,calibration_x=None):
    return {
        'dense':DenseLinear(w,b),
        'int8_prequant':PreQuantLinear(w,b,8),
        'int4_prequant':PreQuantLinear(w,b,4),
        'int2_prequant':PreQuantLinear(w,b,2),
        'ternary_r0':TernaryResidualLinear(w,b,0),
        'ternary_r4':TernaryResidualLinear(w,b,4),
        'ternary_r16':TernaryResidualLinear(w,b,16),
        'ternary_r8_actaware':TernaryResidualLinear(w,b,8,calibration_x=calibration_x,activation_aware=calibration_x is not None),
        'topk_50':DynamicTopKLinear(w,b,0.5),
        'topk_25':DynamicTopKLinear(w,b,0.25),
        'block_50':BlockSparseLinear(w,b,32,0.5),
        'block_25':BlockSparseLinear(w,b,32,0.25),
        'nm_2_4':NMPrunedLinear(w,b,2,4),
        'router':HybridRouterLinear(w,b,8),
    }


def run_linear(device_arg='auto',preset_name='small',repeats=None,methods='all',power=False,csv=None,compile_model=False):
    seed_all(); device=resolve_device(device_arg); p=LINEAR_PRESETS[preset_name]
    if repeats: p=replace(p,repeats=repeats)
    w,b=make_weight(p,device)
    x=torch.randn(p.batch,p.in_features,device=device)
    calibration=torch.randn(min(64,max(16,p.batch)),p.in_features,device=device)
    dense=DenseLinear(w,b).to(device)
    with torch.no_grad(): ref=dense(x)
    mods=build_linear_methods(w,b,calibration)
    if methods!='all':
        wanted=set(x.strip() for x in methods.split(',')); mods={k:v for k,v in mods.items() if k in wanted}
        if 'dense' not in mods: mods={'dense':dense,**mods}
    rows=[]
    for name,mod in mods.items():
        mod=mod.to(device); target=mod
        if compile_model and hasattr(torch,'compile'):
            try: target=torch.compile(mod)
            except Exception: target=mod
        rows.append(_bench(name,target,x,ref,p,device,power))
    df=pd.DataFrame(rows)
    base=float(df.loc[df.method=='dense','latency_ms'].iloc[0]); baseb=float(df.loc[df.method=='dense','estimated_weight_bytes'].iloc[0])
    df['speedup_vs_dense']=base/df.latency_ms
    df['weight_memory_reduction']=1-df.estimated_weight_bytes/baseb
    cols=['method','latency_ms','speedup_vs_dense','mse_vs_dense','nmse_vs_dense','cosine_vs_dense','top1_agreement','theoretical_macs','mac_reduction','estimated_weight_bytes','weight_memory_reduction','avg_power_w','energy_mj_per_inference','edp_mj_ms']
    df=df[cols].sort_values('latency_ms').reset_index(drop=True)
    if csv: pd.DataFrame(df).to_csv(csv,index=False)
    return df,device,p
