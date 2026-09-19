from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from ..kernels.groupwise_int4 import GroupwiseInt4Weight,pack_groupwise_int4,unpack_groupwise_int4,decode_int4_linear

class HybridInt4Linear(nn.Module):
    """Packed groupwise INT4 layer with optional exact activation-sensitive columns.

    No dense source weight is retained after construction.
    """
    def __init__(self,source:nn.Linear,group_size=128,activation_rms=None,outlier_columns=0):
        super().__init__();w=source.weight.detach();n,k=w.shape
        oc=max(0,min(int(outlier_columns),k))
        if oc and oc not in (4,8,16,32,64):oc=min((4,8,16,32,64),key=lambda v:abs(v-oc));oc=min(oc,k)
        if oc:
            wrms=w.float().square().mean(0).sqrt();score=wrms if activation_rms is None else wrms*activation_rms.detach().float().to(w.device)[:k]
            idx=torch.topk(score,k=oc,largest=True,sorted=True).indices;exact=w[:,idx].to(torch.float16).contiguous();base=w.clone();base[:,idx]=0
            self.register_buffer('outlier_idx',idx.to(torch.int32));self.register_buffer('outlier_weight',exact)
        else:
            base=w;self.register_buffer('outlier_idx',torch.empty(0,device=w.device,dtype=torch.int32));self.register_buffer('outlier_weight',torch.empty((n,0),device=w.device,dtype=torch.float16))
        pw=pack_groupwise_int4(base,group_size,activation_rms)
        self.register_buffer('packed',pw.packed);self.register_buffer('scales',pw.scales)
        self.original_k=pw.original_k;self.padded_k=pw.padded_k;self.group_size=pw.group_size;self.in_features=k;self.out_features=n;self.graph_safe=False
        if source.bias is None:self.bias=None
        else:self.register_buffer('bias',source.bias.detach().to(torch.float16))

    def packed_weight(self):return GroupwiseInt4Weight(self.packed,self.scales,self.original_k,self.padded_k,self.group_size,True)
    def forward(self,x):
        shape=x.shape;xflat=x.reshape(-1,shape[-1]);xp=xflat if self.padded_k==self.original_k else F.pad(xflat,(0,self.padded_k-self.original_k));bias=None if self.bias is None else self.bias.to(xp.dtype)
        if xp.is_cuda and xp.shape[0]<=4:
            if self.graph_safe:
                y=decode_int4_linear(xp,self.packed_weight(),bias,self.outlier_idx,self.outlier_weight);return y.reshape(*shape[:-1],self.out_features)
            try:y=decode_int4_linear(xp,self.packed_weight(),bias,self.outlier_idx,self.outlier_weight);return y.reshape(*shape[:-1],self.out_features)
            except Exception:pass
        w=unpack_groupwise_int4(self.packed_weight(),xp.dtype);y=F.linear(xflat,w,bias)
        if self.outlier_idx.numel():y+=F.linear(xflat[:,self.outlier_idx.long()],self.outlier_weight.to(xflat.dtype))
        return y.reshape(*shape[:-1],self.out_features)
