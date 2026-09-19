from __future__ import annotations

from dataclasses import asdict
import gc
import json
from pathlib import Path
import torch
from torch import nn

from .accelerated_linear import GroupwiseInt2Linear
from .planner import LayerDecision, collect_linear_inputs, build_plan


def _parent_and_leaf(root:nn.Module,fqn:str):
    parts=fqn.split('.')
    obj=root
    for p in parts[:-1]: obj=getattr(obj,p)
    return obj,parts[-1]


def apply_plan(model:nn.Module, decisions:list[LayerDecision], captures:dict[str,torch.Tensor] | None=None):
    captures=captures or {}
    replaced=0
    for d in decisions:
        if d.method=="fp16": continue
        parent,leaf=_parent_and_leaf(model,d.name)
        src=getattr(parent,leaf)
        if not isinstance(src,nn.Linear): continue
        x=captures.get(d.name)
        rms=None if x is None else x.float().square().mean(0).sqrt().to(src.weight.device)
        repl=GroupwiseInt2Linear(src,d.group_size,rms,d.residual_rank,"auto").to(src.weight.device)
        setattr(parent,leaf,repl); replaced+=1
    return replaced


def optimize_with_accuracy_guard(model,tokenizer,calibration_texts,device,
                                 group_size=128,nmse_limit=.025,cosine_limit=.985,
                                 residual_ranks=(0,4,8,16),max_rows=128):
    captures=collect_linear_inputs(model,tokenizer,calibration_texts,device,max_rows=max_rows)
    decisions=build_plan(model,captures,group_size,nmse_limit,cosine_limit,residual_ranks)
    replaced=apply_plan(model,decisions,captures)
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return decisions,replaced


def save_plan(decisions:list[LayerDecision],path:str|Path,meta:dict|None=None):
    payload={"meta":meta or {},"layers":[asdict(d) for d in decisions]}
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
    return p


def load_plan(path:str|Path):
    obj=json.loads(Path(path).read_text(encoding='utf-8'))
    return [LayerDecision(**x) for x in obj['layers']],obj.get('meta',{})
