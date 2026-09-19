from __future__ import annotations
from pathlib import Path
import pandas as pd
import torch
import matplotlib.pyplot as plt
from .benchmark import LINEAR_PRESETS,make_weight,_bench
from .layers import DenseLinear,PreQuantLinear,TernaryResidualLinear,BlockSparseLinear,NMPrunedLinear
from .utils import resolve_device,seed_all


def pareto_front(df, objectives):
    # objectives: [(col,'min'|'max')]
    keep=[]
    vals=df.reset_index(drop=True)
    for i,row in vals.iterrows():
        dominated=False
        for j,other in vals.iterrows():
            if i==j:continue
            no_worse=True; strictly=False
            for col,direction in objectives:
                a=other[col]; b=row[col]
                if direction=='min': no_worse &= a<=b; strictly |= a<b
                else: no_worse &= a>=b; strictly |= a>b
            if no_worse and strictly: dominated=True; break
        keep.append(not dominated)
    out=vals.copy(); out['pareto']=keep; return out


def run_sweep(device_arg='auto',preset_name='small',out='results/pareto'):
    seed_all(); device=resolve_device(device_arg); p=LINEAR_PRESETS[preset_name]; w,b=make_weight(p,device); x=torch.randn(p.batch,p.in_features,device=device); ref=DenseLinear(w,b)(x)
    mods={'dense':DenseLinear(w,b)}
    for bits in (2,4,8):mods[f'int{bits}']=PreQuantLinear(w,b,bits)
    for r in (0,2,4,8,16,32):mods[f'ternary_r{r}']=TernaryResidualLinear(w,b,r)
    for kr in (0.125,0.25,0.5,0.75):mods[f'block_{kr:g}']=BlockSparseLinear(w,b,32,kr)
    for n,m in ((1,4),(2,4),(4,8)):mods[f'nm_{n}_{m}']=NMPrunedLinear(w,b,n,m)
    rows=[_bench(name,mod.to(device),x,ref,p,device,False) for name,mod in mods.items()]
    df=pd.DataFrame(rows); base=float(df.loc[df.method=='dense','latency_ms'].iloc[0]); baseb=float(df.loc[df.method=='dense','estimated_weight_bytes'].iloc[0])
    df['speedup_vs_dense']=base/df.latency_ms; df['weight_memory_reduction']=1-df.estimated_weight_bytes/baseb
    df=pareto_front(df,[('latency_ms','min'),('nmse_vs_dense','min'),('estimated_weight_bytes','min')])
    outp=Path(out); outp.mkdir(parents=True,exist_ok=True); df.to_csv(outp/'sweep.csv',index=False)
    fig=plt.figure(figsize=(9,6)); ax=fig.add_subplot(111); ax.scatter(df.latency_ms,df.nmse_vs_dense)
    for _,r in df[df.pareto].iterrows():ax.annotate(r.method,(r.latency_ms,r.nmse_vs_dense),fontsize=8)
    ax.set_xlabel('Latency (ms)'); ax.set_ylabel('NMSE vs dense'); ax.set_yscale('symlog',linthresh=1e-8); ax.set_title('Latency / Accuracy Pareto candidates'); fig.tight_layout(); fig.savefig(outp/'pareto.png',dpi=160); plt.close(fig)
    return df,outp
