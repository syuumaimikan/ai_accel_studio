from __future__ import annotations
import time
import pandas as pd
import torch
from .layers import DenseLinear,DeltaLinear
from .benchmark import make_weight,LinearPreset
from .metrics import output_metrics
from .utils import resolve_device,sync,seed_all

def run_delta(device_arg='auto',steps=100,batch=8,features=1024,change_ratio=0.02,threshold=0.0):
    seed_all(); device=resolve_device(device_arg); p=LinearPreset(batch,features,features,0,1); w,b=make_weight(p,device)
    dense=DenseLinear(w,b); delta=DeltaLinear(w,b,threshold); x=torch.randn(batch,features,device=device); delta(x)
    td=[]; tx=[]; mses=[]; changes=[]
    for _ in range(steps):
        xn=x.clone(); k=max(1,int(features*change_ratio)); idx=torch.randperm(features,device=device)[:k]; xn[:,idx]+=torch.randn(batch,k,device=device)*0.05
        sync(device); t=time.perf_counter(); yd=dense(xn); sync(device); td.append((time.perf_counter()-t)*1000)
        sync(device); t=time.perf_counter(); yx=delta(xn); sync(device); tx.append((time.perf_counter()-t)*1000)
        mses.append(output_metrics(yd,yx)['mse_vs_dense']); changes.append(delta.last_change_ratio); x=xn
    d=sum(td)/len(td); z=sum(tx)/len(tx)
    return pd.Series({'device':str(device),'dense_avg_ms':d,'delta_avg_ms':z,'speedup':d/z,'mean_mse':sum(mses)/len(mses),'mean_changed_ratio':sum(changes)/len(changes)})
