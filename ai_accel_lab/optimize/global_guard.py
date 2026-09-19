from __future__ import annotations
"""End-to-end calibration guard for aggressive INT2 plans.

Local layer cosine/NMSE is only a proxy. v4 additionally measures model NLL on
calibration text. It starts from an aggressive mixed plan and restores the most
sensitive layers from CPU-held temporary FP16 backups until the configured
end-to-end loss budget is met. CPU backups are destroyed before returning, so
runtime GPU memory contains only the final optimized representation.
"""
import gc
import torch
from torch import nn
import torch.nn.functional as F
from .planner import collect_linear_inputs
from .planner_v4 import build_plan_v4,LayerDecisionV4
from .hybrid_int2 import HybridInt2Linear


def _parent(root,fqn):
    ps=fqn.split('.'); o=root
    for p in ps[:-1]: o=getattr(o,p)
    return o,ps[-1]

@torch.no_grad()
def calibration_nll(model,tokenizer,texts,device,max_length=192):
    total=0.0; count=0
    for text in texts:
        b=tokenizer(text,return_tensors='pt',truncation=True,max_length=max_length)
        ids=b['input_ids'].to(device)
        if ids.shape[1]<2: continue
        o=model(input_ids=ids,use_cache=False)
        l=o.logits[:,:-1,:].float(); labels=ids[:,1:]
        loss=F.cross_entropy(l.reshape(-1,l.shape[-1]),labels.reshape(-1),reduction='sum')
        total+=float(loss.item()); count+=labels.numel()
    return total/max(1,count)

def optimize_global_guard(model,tokenizer,texts,device,group_sizes=(128,64,32),
                          local_nmse=.06,local_cosine=.96,max_loss_rel_increase=.02,
                          outliers=(0,8,16,32),residual_ranks=(0,4,8),max_rows=64):
    baseline=calibration_nll(model,tokenizer,texts,device)
    captures=collect_linear_inputs(model,tokenizer,texts,device,max_rows=max_rows)
    decisions=build_plan_v4(model,captures,group_sizes,outliers,residual_ranks,local_nmse,local_cosine)
    backups={}
    # Keep dense source only on CPU during calibration rollback.
    for d in decisions:
        if d.method=='fp16': continue
        p,leaf=_parent(model,d.name); src=getattr(p,leaf)
        if not isinstance(src,nn.Linear): continue
        backups[d.name]=(src.weight.detach().cpu().clone(), None if src.bias is None else src.bias.detach().cpu().clone())
        x=captures.get(d.name); rms=None if x is None else x.float().square().mean(0).sqrt().to(src.weight.device)
        repl=HybridInt2Linear(src,d.group_size,rms,d.outlier_columns,d.residual_rank,'auto').to(src.weight.device)
        setattr(p,leaf,repl)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    current=calibration_nll(model,tokenizer,texts,device)
    limit=baseline*(1.0+max_loss_rel_increase)
    # Restore highest local-risk layers until model-level quality passes.
    restored=[]
    candidates=sorted([d for d in decisions if d.method!='fp16'],key=lambda d:(d.nmse,-d.cosine),reverse=True)
    ci=0
    while current>limit and ci<len(candidates):
        # Restore in small batches to avoid a full model eval after every single layer.
        batch=candidates[ci:ci+4]; ci+=4
        for d in batch:
            if d.name not in backups: continue
            p,leaf=_parent(model,d.name); qw,qb=backups[d.name]
            old=getattr(p,leaf); dev=next(old.buffers()).device if any(True for _ in old.buffers()) else torch.device(device)
            dense=nn.Linear(qw.shape[1],qw.shape[0],bias=qb is not None,device=dev,dtype=qw.dtype)
            dense.weight.data.copy_(qw.to(dev));
            if qb is not None: dense.bias.data.copy_(qb.to(dev))
            setattr(p,leaf,dense)
            d.method='fp16'; d.reason='restored by global NLL guard'; d.compression=1.0
            restored.append(d.name)
        gc.collect();
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        current=calibration_nll(model,tokenizer,texts,device)
    optimized=sum(d.method!='fp16' for d in decisions)
    del backups
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    stats={'baseline_nll':baseline,'optimized_nll':current,'relative_nll_increase':current/max(baseline,1e-12)-1.0,
           'limit':max_loss_rel_increase,'optimized_layers':optimized,'restored_layers':len(restored)}
    return decisions,optimized,stats
