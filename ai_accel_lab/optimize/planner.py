from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable
import torch
from torch import nn
import torch.nn.functional as F

from .accelerated_linear import GroupwiseInt2Linear


@dataclass
class LayerDecision:
    name: str
    method: str
    group_size: int
    residual_rank: int
    nmse: float
    cosine: float
    compression: float
    reason: str

    def to_dict(self): return asdict(self)


DEFAULT_TARGET_TOKENS = (
    "q_proj","k_proj","v_proj","o_proj",
    "gate_proj","up_proj","down_proj","fc1","fc2",
)


def collect_linear_inputs(model: nn.Module, tokenizer, texts: list[str], device,
                          max_rows=128, target_tokens=DEFAULT_TARGET_TOKENS):
    captures: dict[str,list[torch.Tensor]]={}
    hooks=[]
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear) and any(t in name for t in target_tokens):
            captures[name]=[]
            def hook_factory(key):
                def _hook(mod,args):
                    if len(captures[key]) >= 4: return
                    x=args[0].detach()
                    x=x.reshape(-1,x.shape[-1])[:max_rows].float().cpu()
                    captures[key].append(x)
                return _hook
            hooks.append(module.register_forward_pre_hook(hook_factory(name)))
    model.eval()
    with torch.no_grad():
        for text in texts:
            batch=tokenizer(text,return_tensors="pt",truncation=True,max_length=256)
            batch={k:v.to(device) for k,v in batch.items()}
            model(**batch,use_cache=False)
    for h in hooks: h.remove()
    return {k:torch.cat(v,0)[:max_rows] for k,v in captures.items() if v}


def _quality(ref: torch.Tensor, cand: torch.Tensor):
    diff=(cand.float()-ref.float())
    mse=diff.square().mean().item()
    denom=ref.float().square().mean().item()+1e-12
    nmse=mse/denom
    a=ref.float().reshape(ref.shape[0],-1)
    b=cand.float().reshape(cand.shape[0],-1)
    cosine=F.cosine_similarity(a,b,dim=-1).mean().item()
    return nmse,cosine


def decide_linear(name:str, layer:nn.Linear, sample_x:torch.Tensor,
                  group_size=128,nmse_limit=0.025,cosine_limit=0.985,
                  residual_ranks=(0,4,8,16)) -> LayerDecision:
    dev=layer.weight.device
    x=sample_x.to(dev,dtype=torch.float16 if dev.type=='cuda' else torch.float32)
    # Output-space weighting proxy for activation-aware scale search.
    rms=sample_x.float().square().mean(0).sqrt().to(dev)
    with torch.no_grad():
        ref=F.linear(x,layer.weight.to(x.dtype),None if layer.bias is None else layer.bias.to(x.dtype))
    best=None
    for rank in residual_ranks:
        cand=GroupwiseInt2Linear(layer,group_size=group_size,activation_rms=rms,
                                 residual_rank=rank,backend="auto").to(dev)
        with torch.no_grad(): out=cand(x)
        nmse,cos=_quality(ref,out)
        info=cand.optimization_info()
        decision=LayerDecision(name,"int2+residual" if rank else "int2",group_size,rank,
                               nmse,cos,info.estimated_compression,
                               "meets accuracy guard" if nmse<=nmse_limit and cos>=cosine_limit else "quality below guard")
        if decision.nmse<=nmse_limit and decision.cosine>=cosine_limit:
            return decision
        if best is None or decision.nmse<best.nmse: best=decision
    return LayerDecision(name,"fp16",group_size,0,best.nmse,best.cosine,1.0,
                         "kept dense because quantized candidates failed accuracy guard")


def build_plan(model:nn.Module,captures:dict[str,torch.Tensor],group_size=128,
               nmse_limit=0.025,cosine_limit=0.985,residual_ranks=(0,4,8,16)):
    modules=dict(model.named_modules())
    decisions=[]
    for name,x in captures.items():
        layer=modules.get(name)
        if isinstance(layer,nn.Linear):
            decisions.append(decide_linear(name,layer,x,group_size,nmse_limit,cosine_limit,residual_ranks))
    return decisions
