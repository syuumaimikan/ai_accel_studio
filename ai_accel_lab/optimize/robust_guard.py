from __future__ import annotations
"""v5 mixed-bit optimizer with a held-out, robust model-level quality guard.

Unlike v4, the prompts used to choose layer-local quantization candidates are
separated from held-out validation prompts when possible. Model-level rollback
uses mean, p95 and worst per-token NLL degradation, reducing calibration overfit.
"""
import gc,math
import torch
from torch import nn
import torch.nn.functional as F
from .planner import collect_linear_inputs
from .planner_v5 import build_plan_v5
from .hybrid_int2 import HybridInt2Linear
from .hybrid_int4 import HybridInt4Linear


def _parent(root,fqn):
    ps=fqn.split('.');o=root
    for p in ps[:-1]:o=getattr(o,p)
    return o,ps[-1]


def split_calibration_validation(texts):
    xs=[x.strip() for x in texts if x and x.strip()]
    if not xs:return ['Explain matrix multiplication.'],['Efficient inference uses computation and memory.']
    if len(xs)==1:return xs,xs
    plan=xs[::2];val=xs[1::2]
    if not val:val=plan
    return plan,val

@torch.no_grad()
def per_text_nll(model,tokenizer,texts,device,max_length=256):
    vals=[]
    for text in texts:
        b=tokenizer(text,return_tensors='pt',truncation=True,max_length=max_length);ids=b['input_ids'].to(device)
        if ids.shape[1]<2:continue
        o=model(input_ids=ids,use_cache=False);l=o.logits[:,:-1,:].float();labels=ids[:,1:]
        loss=F.cross_entropy(l.reshape(-1,l.shape[-1]),labels.reshape(-1),reduction='mean')
        vals.append(float(loss.item()))
    return vals


def robust_summary(base,current):
    if not base or not current:return {'mean_rel':float('inf'),'p95_rel':float('inf'),'worst_rel':float('inf'),'samples':0}
    n=min(len(base),len(current));rel=[current[i]/max(base[i],1e-12)-1.0 for i in range(n)];s=sorted(rel)
    p95=s[min(len(s)-1,max(0,math.ceil(.95*len(s))-1))]
    return {'mean_rel':sum(rel)/len(rel),'p95_rel':p95,'worst_rel':max(rel),'samples':len(rel)}


def _passes(s,mean_budget,p95_budget,worst_budget):
    return s['mean_rel']<=mean_budget and s['p95_rel']<=p95_budget and s['worst_rel']<=worst_budget


def optimize_robust_mixedbit(model,tokenizer,texts,device,group_sizes=(128,64,32),local_nmse=.06,local_cosine=.96,
                             mean_loss_budget=.02,p95_loss_budget=.04,worst_loss_budget=.08,
                             outliers=(0,8,16,32),residual_ranks=(0,4,8),max_rows=64,expected_decode_tokens=64):
    plan_texts,val_texts=split_calibration_validation(texts)
    baseline_val=per_text_nll(model,tokenizer,val_texts,device)
    captures=collect_linear_inputs(model,tokenizer,plan_texts,device,max_rows=max_rows)
    decisions=build_plan_v5(model,captures,group_sizes,outliers,residual_ranks,local_nmse,local_cosine,expected_decode_tokens=expected_decode_tokens)
    backups={}
    for d in decisions:
        if d.method=='fp16':continue
        p,leaf=_parent(model,d.name);src=getattr(p,leaf)
        if not isinstance(src,nn.Linear):continue
        backups[d.name]=(src.weight.detach().cpu().clone(),None if src.bias is None else src.bias.detach().cpu().clone())
        x=captures.get(d.name);rms=None if x is None else x.float().square().mean(0).sqrt().to(src.weight.device)
        if d.bits==2:repl=HybridInt2Linear(src,d.group_size,rms,d.outlier_columns,d.residual_rank,'auto').to(src.weight.device)
        elif d.bits==4:repl=HybridInt4Linear(src,d.group_size,rms,d.outlier_columns).to(src.weight.device)
        else:continue
        setattr(p,leaf,repl)
    gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    current_val=per_text_nll(model,tokenizer,val_texts,device);summary=robust_summary(baseline_val,current_val)

    # Prioritize local-risky INT2 layers first; INT4 is less aggressive and is
    # restored later. Restore in batches to keep calibration cost bounded.
    candidates=sorted([d for d in decisions if d.method!='fp16'],key=lambda d:(d.bits==2,d.nmse,-d.cosine),reverse=True)
    restored=[];ci=0
    while not _passes(summary,mean_loss_budget,p95_loss_budget,worst_loss_budget) and ci<len(candidates):
        for d in candidates[ci:ci+4]:
            if d.name not in backups:continue
            p,leaf=_parent(model,d.name);qw,qb=backups[d.name];old=getattr(p,leaf)
            try:dev=next(old.buffers()).device
            except Exception:dev=torch.device(device)
            dense=nn.Linear(qw.shape[1],qw.shape[0],bias=qb is not None,device=dev,dtype=qw.dtype)
            dense.weight.data.copy_(qw.to(dev));
            if qb is not None:dense.bias.data.copy_(qb.to(dev))
            setattr(p,leaf,dense);d.method='fp16';d.bits=16;d.compression=1.0;d.reason='restored by robust held-out NLL guard';restored.append(d.name)
        ci+=4;gc.collect()
        if torch.cuda.is_available():torch.cuda.empty_cache()
        current_val=per_text_nll(model,tokenizer,val_texts,device);summary=robust_summary(baseline_val,current_val)

    int2=sum(d.bits==2 and d.method!='fp16' for d in decisions);int4=sum(d.bits==4 and d.method!='fp16' for d in decisions);fp16=sum(d.method=='fp16' for d in decisions)
    speed_guard_fp16=sum(d.method=='fp16' and 'speed' in d.reason.lower() for d in decisions)
    projected=[d.projected_speedup for d in decisions if d.method!='fp16' and d.projected_speedup is not None]
    del backups;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    stats={'guard':'v5-heldout-robust+runtime-gate','plan_samples':len(plan_texts),'validation_samples':len(val_texts),
           'mean_rel_nll':summary['mean_rel'],'p95_rel_nll':summary['p95_rel'],'worst_rel_nll':summary['worst_rel'],
           'mean_budget':mean_loss_budget,'p95_budget':p95_loss_budget,'worst_budget':worst_loss_budget,
           'int2_layers':int2,'int4_layers':int4,'fp16_layers':fp16,'optimized_layers':int2+int4,'restored_layers':len(restored),
           'speed_guard_fp16_layers':speed_guard_fp16,'mean_projected_layer_speedup':(sum(projected)/len(projected) if projected else None)}
    return decisions,int2+int4,stats
