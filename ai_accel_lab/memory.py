from __future__ import annotations
import math,time
import pandas as pd
import torch
import torch.nn.functional as F
from .metrics import output_metrics
from .utils import resolve_device,sync,seed_all

@torch.no_grad()
def full_memory_attention(q,memory):
    s=q@memory.T/math.sqrt(memory.shape[-1]); return torch.softmax(s,-1)@memory

@torch.no_grad()
def hierarchical_memory_attention(q,memory,block_size=64,top_blocks=4):
    n,d=memory.shape; pad=(-n)%block_size; mp=F.pad(memory,(0,0,0,pad)) if pad else memory
    blocks=mp.view(-1,block_size,d); cent=blocks.mean(1); coarse=q@cent.T
    ids=torch.topk(coarse,k=min(top_blocks,blocks.shape[0]),dim=-1,sorted=False).indices
    outs=[]
    for b in range(q.shape[0]):
        sel=blocks[ids[b]].reshape(-1,d); s=q[b:b+1]@sel.T/math.sqrt(d); outs.append(torch.softmax(s,-1)@sel)
    return torch.cat(outs,0)

def run_memory(device_arg='auto',batch=8,n_memory=16384,dim=256,block_size=64,top_blocks=4,repeats=20):
    seed_all(); device=resolve_device(device_arg); mem=torch.randn(n_memory,dim,device=device); q=torch.randn(batch,dim,device=device)
    def timed(fn):
        for _ in range(3):fn()
        sync(device); t=time.perf_counter(); y=None
        for _ in range(repeats):y=fn()
        sync(device); return y,(time.perf_counter()-t)*1000/repeats
    yf,tf=timed(lambda:full_memory_attention(q,mem)); yh,th=timed(lambda:hierarchical_memory_attention(q,mem,block_size,top_blocks))
    fm=2*batch*n_memory*dim; nb=math.ceil(n_memory/block_size); hm=batch*nb*dim+2*batch*min(top_blocks,nb)*block_size*dim
    return pd.Series({'device':str(device),'full_ms':tf,'hierarchical_ms':th,'speedup':tf/th,'full_macs':fm,'hierarchical_macs':hm,'mac_reduction':1-hm/fm,**output_metrics(yf,yh)})
