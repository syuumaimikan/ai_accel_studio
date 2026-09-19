from __future__ import annotations

from dataclasses import asdict,dataclass
import gc,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
import pandas as pd

from .power import NvmlPowerSampler
from .optimize.model_optimizer import optimize_with_accuracy_guard
from .optimize.accelerated_linear import GroupwiseInt2Linear
from .optimize.torchao_backend import apply_torchao_int4,apply_torchao_int8,apply_torchao_int2
from .optimize.sparse24_model import apply_sparse24
from .compare import annotate_model_matrix


@dataclass
class GenerationMetrics:
    model: str
    method: str
    prompt_tokens: int
    generated_tokens: int
    ttft_ms: float
    decode_tokens_s: float
    total_tokens_s: float
    avg_power_w: float | None
    joules_per_generated_token: float | None
    peak_vram_gb: float | None
    perplexity: float | None
    plan_layers_quantized: int | None
    output_text: str
    error: str | None = None

    def row(self): return asdict(self)


def _dtype(name:str):
    if name=='bf16': return torch.bfloat16
    if name=='fp32': return torch.float32
    return torch.float16


def load_hf(model_id:str,device='cuda',dtype='fp16',trust_remote_code=False,method:str|None=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=trust_remote_code)
    dt=_dtype(dtype)
    kwargs={"dtype":dt,"trust_remote_code":trust_remote_code}
    # Current Transformers exposes optimized kernel and attention backends at load time.
    # These are included as strong existing baselines in the comparison matrix.
    if method == 'hf-kernels-auto':
        kwargs['use_kernels'] = True
    elif method in ('sdpa','flash-attention-2','flash-attention-3','paged-flash-attention-3','hf-flash-attn2-kernel','hf-flash-attn3-kernel'):
        kwargs['attn_implementation'] = {
            'sdpa':'sdpa',
            'flash-attention-2':'flash_attention_2',
            'flash-attention-3':'flash_attention_3',
            'paged-flash-attention-3':'paged|flash_attention_3',
            'hf-flash-attn2-kernel':'kernels-community/flash-attn2',
            'hf-flash-attn3-kernel':'kernels-community/flash-attn3',
        }[method]
    if device=='cuda': kwargs["device_map"]="cuda"
    model=AutoModelForCausalLM.from_pretrained(model_id,**kwargs)
    if device!='cuda': model=model.to(device)
    model.eval()
    return model,tok


def _replace_all_int2(model,group_size=128,residual_rank=0):
    names=[]
    for n,m in model.named_modules():
        if isinstance(m,torch.nn.Linear) and any(t in n for t in ("q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj","fc1","fc2")):
            names.append(n)
    for n in names:
        ps=n.split('.'); p=model
        for q in ps[:-1]: p=getattr(p,q)
        src=getattr(p,ps[-1])
        try: setattr(p,ps[-1],GroupwiseInt2Linear(src,group_size,None,residual_rank,'auto'))
        except Exception: continue
    return len(names)


def apply_method(model,tokenizer,method,calibration_texts,device,group_size=128,nmse_limit=.025,cosine_limit=.985,residual_rank=8):
    plan_count=None
    if method in ('baseline-fp16','baseline-bf16','hf-kernels-auto','sdpa','flash-attention-2','flash-attention-3','paged-flash-attention-3','hf-flash-attn2-kernel','hf-flash-attn3-kernel'):
        return plan_count
    if method=='torch-compile':
        if hasattr(torch,'compile'): model.forward=torch.compile(model.forward,mode='reduce-overhead')
    elif method=='torchao-int4': apply_torchao_int4(model,group_size)
    elif method=='torchao-int8': apply_torchao_int8(model)
    elif method=='torchao-int2': apply_torchao_int2(model,group_size)
    elif method=='native-2to4': plan_count=apply_sparse24(model)
    elif method=='custom-int2-fast': plan_count=_replace_all_int2(model,group_size,0)
    elif method=='custom-int2-residual': plan_count=_replace_all_int2(model,group_size,residual_rank)
    elif method=='custom-int2-accuracy-guard':
        decisions,replaced=optimize_with_accuracy_guard(model,tokenizer,calibration_texts,device,group_size,nmse_limit,cosine_limit,(0,4,8,residual_rank))
        plan_count=replaced
    else: raise ValueError(f'unknown method: {method}')
    return plan_count


@torch.no_grad()
def greedy_benchmark(model,tokenizer,prompt,max_new_tokens=32,device='cuda'):
    batch=tokenizer(prompt,return_tensors='pt')
    batch={k:v.to(device) for k,v in batch.items()}
    input_ids=batch['input_ids']; attention_mask=batch.get('attention_mask',torch.ones_like(input_ids))
    if device=='cuda': torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    power=NvmlPowerSampler(0); power.start()
    t0=time.perf_counter()
    out=model(input_ids=input_ids,attention_mask=attention_mask,use_cache=True)
    if device=='cuda': torch.cuda.synchronize()
    ttft=(time.perf_counter()-t0)*1000
    past=out.past_key_values
    token=out.logits[:,-1,:].argmax(-1,keepdim=True)
    generated=[token]
    eos=getattr(tokenizer,'eos_token_id',None)
    decode_times=[]
    for _ in range(max_new_tokens-1):
        attention_mask=torch.cat([attention_mask,torch.ones((attention_mask.shape[0],1),device=device,dtype=attention_mask.dtype)],dim=1)
        if device=='cuda': torch.cuda.synchronize()
        s=time.perf_counter()
        out=model(input_ids=token,attention_mask=attention_mask,past_key_values=past,use_cache=True)
        if device=='cuda': torch.cuda.synchronize()
        decode_times.append(time.perf_counter()-s)
        past=out.past_key_values
        token=out.logits[:,-1,:].argmax(-1,keepdim=True)
        generated.append(token)
        if eos is not None and bool((token==eos).all()): break
    total=time.perf_counter()-t0
    avg_power=power.stop()
    ids=torch.cat(generated,dim=1)
    n=ids.shape[1]
    decode_s=sum(decode_times)
    decode_tps=(max(0,n-1)/decode_s) if decode_s>0 else 0.0
    total_tps=n/total if total>0 else 0.0
    peak=(torch.cuda.max_memory_allocated()/1024**3) if device=='cuda' else None
    joules=None if avg_power is None else avg_power*total/max(1,n)
    text=tokenizer.decode(ids[0],skip_special_tokens=True)
    return dict(prompt_tokens=input_ids.shape[1],generated_tokens=n,ttft_ms=ttft,
                decode_tokens_s=decode_tps,total_tokens_s=total_tps,avg_power_w=avg_power,
                joules_per_generated_token=joules,peak_vram_gb=peak,output_text=text)


@torch.no_grad()
def perplexity(model,tokenizer,text,device='cuda',max_length=512):
    if not text.strip(): return None
    b=tokenizer(text,return_tensors='pt',truncation=True,max_length=max_length)
    ids=b['input_ids'].to(device)
    if ids.shape[1]<2: return None
    out=model(input_ids=ids,use_cache=False)
    logits=out.logits[:,:-1,:].float(); labels=ids[:,1:]
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1))
    return float(torch.exp(loss).item())


def benchmark_model_method(model_id,method,prompt,calibration_texts,eval_text,max_new_tokens=32,
                           device='cuda',dtype='fp16',group_size=128,nmse_limit=.025,cosine_limit=.985,residual_rank=8):
    try:
        model,tok=load_hf(model_id,device,dtype,method=method)
        plan_count=apply_method(model,tok,method,calibration_texts,device,group_size,nmse_limit,cosine_limit,residual_rank)
        # Warmup after optimization/compile.
        b=tok(prompt,return_tensors='pt'); b={k:v.to(device) for k,v in b.items()}
        with torch.no_grad(): model(**b,use_cache=False)
        if device=='cuda': torch.cuda.synchronize()
        g=greedy_benchmark(model,tok,prompt,max_new_tokens,device)
        ppl=perplexity(model,tok,eval_text,device) if eval_text else None
        result=GenerationMetrics(model_id,method,perplexity=ppl,plan_layers_quantized=plan_count,error=None,**g)
    except Exception as e:
        result=GenerationMetrics(model_id,method,0,0,0,0,0,None,None,None,None,None,"",repr(e))
    finally:
        try: del model
        except Exception: pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def benchmark_matrix(model_ids:list[str],methods:list[str],prompt:str,calibration_texts:list[str],eval_text:str,
                     max_new_tokens=32,device='cuda',dtype='fp16',**kwargs):
    rows=[]
    for mid in model_ids:
        for method in methods:
            rows.append(benchmark_model_method(mid,method,prompt,calibration_texts,eval_text,max_new_tokens,device,dtype,**kwargs).row())
    return annotate_model_matrix(pd.DataFrame(rows))


def build_optimized_bundle(model_id:str,out_dir:str,calibration_texts:list[str],device='cuda',dtype='fp16',
                           group_size=128,nmse_limit=.025,cosine_limit=.985,residual_rank=8):
    from .optimize.bundle import save_bundle
    model,tok=load_hf(model_id,device,dtype)
    decisions,replaced=optimize_with_accuracy_guard(model,tok,calibration_texts,device,group_size,nmse_limit,cosine_limit,(0,4,8,residual_rank))
    out=save_bundle(model,tok,model_id,decisions,out_dir,{'group_size':group_size,'nmse_limit':nmse_limit,'cosine_limit':cosine_limit,'residual_rank':residual_rank,'replaced':replaced})
    df=pd.DataFrame([d.to_dict() for d in decisions])
    return df,out,replaced
