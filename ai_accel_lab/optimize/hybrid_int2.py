from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F

from ..kernels.groupwise_int2 import GroupwiseInt2Weight, pack_groupwise_int2, unpack_groupwise_int2
from ..kernels.arch_dispatch import run_groupwise_int2
from ..kernels.decode_int2 import decode_int2_linear

@dataclass
class HybridInfo:
    method:str
    group_size:int
    outlier_columns:int
    residual_rank:int
    original_bytes:int
    optimized_bytes:int
    compression:float

class HybridInt2Linear(nn.Module):
    """Runtime-only INT2 layer with no retained dense source weight.

    The most activation-sensitive input columns may be kept exactly in FP16.
    Those columns are zeroed in the INT2 base before packing, so the exact
    correction can be fused into the small-M decode kernel without double count.
    Optional low-rank residual is a last-resort accuracy recovery mechanism.
    """
    def __init__(self, source:nn.Linear, group_size=128, activation_rms=None,
                 outlier_columns:int=0, residual_rank:int=0, backend='auto'):
        super().__init__()
        w=source.weight.detach()
        n,k=w.shape
        outlier_columns=max(0,min(int(outlier_columns),k))
        if outlier_columns and outlier_columns not in (4,8,16,32,64):
            # keep fused decode compile-time sizes bounded
            outlier_columns=min((4,8,16,32,64),key=lambda v:abs(v-outlier_columns))
            outlier_columns=min(outlier_columns,k)
        if outlier_columns:
            wrms=w.float().square().mean(0).sqrt()
            if activation_rms is None:
                score=wrms
            else:
                ar=activation_rms.detach().float().to(w.device)
                score=wrms*ar[:k]
            idx=torch.topk(score,k=outlier_columns,largest=True,sorted=True).indices
            exact=w[:,idx].to(torch.float16).contiguous()
            base=w.clone()
            base[:,idx]=0
            self.register_buffer('outlier_idx',idx.to(torch.int32))
            self.register_buffer('outlier_weight',exact)
        else:
            base=w
            self.register_buffer('outlier_idx',torch.empty(0,device=w.device,dtype=torch.int32))
            self.register_buffer('outlier_weight',torch.empty((n,0),device=w.device,dtype=torch.float16))
        pw=pack_groupwise_int2(base,group_size=group_size,activation_rms=activation_rms)
        self.register_buffer('packed',pw.packed)
        self.register_buffer('scales',pw.scales)
        self.original_k=pw.original_k; self.padded_k=pw.padded_k; self.group_size=pw.group_size
        self.in_features=k; self.out_features=n; self.backend=backend; self.graph_safe=False
        if source.bias is None: self.bias=None
        else: self.register_buffer('bias',source.bias.detach().to(torch.float16))

        rank=max(0,min(int(residual_rank),min(w.shape)))
        self.residual_rank=rank
        if rank:
            q=unpack_groupwise_int2(pw,dtype=torch.float32).to(w.device)
            if self.outlier_idx.numel():
                q[:,self.outlier_idx.long()]+=self.outlier_weight.float()
            err=w.float()-q
            U,S,Vh=torch.linalg.svd(err,full_matrices=False)
            self.register_buffer('res_u',(U[:,:rank]*S[:rank]).to(torch.float16).contiguous())
            self.register_buffer('res_v',Vh[:rank,:].to(torch.float16).contiguous())
        else:
            self.res_u=None; self.res_v=None

    def packed_weight(self):
        return GroupwiseInt2Weight(self.packed,self.scales,self.original_k,self.padded_k,self.group_size,True)

    def forward(self,x):
        shape=x.shape; xflat=x.reshape(-1,shape[-1])
        xp=xflat if self.padded_k==self.original_k else F.pad(xflat,(0,self.padded_k-self.original_k))
        bias=None if self.bias is None else self.bias.to(xp.dtype)
        used_fused=False
        if xp.is_cuda and xp.shape[0] <= 4:
            if self.graph_safe:
                # No exception-driven dispatch inside a compiled/static-cache graph.
                y=decode_int2_linear(xp,self.packed_weight(),bias,self.outlier_idx,self.outlier_weight)
                used_fused=True
            else:
                try:
                    y=decode_int2_linear(xp,self.packed_weight(),bias,self.outlier_idx,self.outlier_weight)
                    used_fused=True
                except Exception:
                    used_fused=False
        if not used_fused:
            try:
                y=run_groupwise_int2(xp,self.packed_weight(),bias,backend=self.backend)
            except Exception:
                w=unpack_groupwise_int2(self.packed_weight(),dtype=xp.dtype)
                y=F.linear(xflat,w,bias)
            if self.outlier_idx.numel():
                y=y+F.linear(xflat[:,self.outlier_idx.long()],self.outlier_weight.to(xflat.dtype))
        if self.residual_rank:
            corr=F.linear(F.linear(xflat.to(self.res_v.dtype),self.res_v),self.res_u).to(y.dtype)
            y=y+corr
        return y.reshape(*shape[:-1],self.out_features)

    def optimization_info(self):
        orig=self.out_features*self.in_features*2+(0 if self.bias is None else self.out_features*2)
        opt=self.packed.numel()*self.packed.element_size()+self.scales.numel()*self.scales.element_size()
        opt+=self.outlier_idx.numel()*self.outlier_idx.element_size()+self.outlier_weight.numel()*self.outlier_weight.element_size()
        if self.bias is not None: opt+=self.bias.numel()*self.bias.element_size()
        if self.residual_rank:
            opt+=self.res_u.numel()*self.res_u.element_size()+self.res_v.numel()*self.res_v.element_size()
        return HybridInfo('hybrid-int2',self.group_size,int(self.outlier_idx.numel()),self.residual_rank,orig,opt,orig/max(1,opt))
