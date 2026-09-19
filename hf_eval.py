"""Optional real-model accuracy evaluator. Requires requirements-optional.txt."""
from __future__ import annotations
import argparse, math
import torch
from torch import nn
from ai_accel_lab.layers import PreQuantLinear,TernaryResidualLinear
from ai_accel_lab.metrics import output_metrics

DEFAULT_TEXT = """Artificial intelligence systems can be made more efficient by reducing memory traffic, using lower precision arithmetic, exploiting structured sparsity, and allocating computation adaptively. The goal of this evaluation is to measure how much numerical approximation changes language-model predictions."""

def replace_linears(module,method,rank,min_features=128):
    count=0
    for name,child in list(module.named_children()):
        if isinstance(child,nn.Linear) and min(child.in_features,child.out_features)>=min_features:
            if method=='ternary': new=TernaryResidualLinear.from_linear(child,rank=rank)
            elif method=='int4': new=PreQuantLinear.from_linear(child,bits=4)
            elif method=='int2': new=PreQuantLinear.from_linear(child,bits=2)
            else: raise ValueError(method)
            setattr(module,name,new.to(next(child.parameters()).device)); count+=1
        else: count+=replace_linears(child,method,rank,min_features)
    return count

@torch.no_grad()
def perplexity(model,ids):
    out=model(ids,labels=ids); return float(torch.exp(out.loss.float()).item()),out.logits.detach().float().cpu()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model',required=True); p.add_argument('--method',choices=['ternary','int4','int2'],default='ternary'); p.add_argument('--rank',type=int,default=8); p.add_argument('--text-file'); p.add_argument('--max-length',type=int,default=512); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); args=p.parse_args()
    try:
        from transformers import AutoTokenizer,AutoModelForCausalLM
    except ImportError as e: raise SystemExit('Install requirements-optional.txt first') from e
    text=DEFAULT_TEXT if not args.text_file else open(args.text_file,encoding='utf-8').read()
    tok=AutoTokenizer.from_pretrained(args.model,use_fast=True); model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.float16 if args.device.startswith('cuda') else torch.float32).to(args.device).eval()
    ids=tok(text,return_tensors='pt',truncation=True,max_length=args.max_length).input_ids.to(args.device)
    base_ppl,base_logits=perplexity(model,ids); count=replace_linears(model,args.method,args.rank); approx_ppl,approx_logits=perplexity(model,ids)
    m=output_metrics(base_logits,approx_logits)
    print(f'replaced_linear_layers: {count}'); print(f'baseline_perplexity: {base_ppl:.6f}'); print(f'approx_perplexity: {approx_ppl:.6f}'); print(f'perplexity_ratio: {approx_ppl/base_ppl:.6f}')
    for k,v in m.items():print(f'{k}: {v:.8g}')
if __name__=='__main__':main()
