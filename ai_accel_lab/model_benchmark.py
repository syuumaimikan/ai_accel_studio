from __future__ import annotations

from dataclasses import asdict,dataclass
import gc,time,json
import torch
import torch.nn.functional as F
import pandas as pd

from .power import NvmlPowerSampler
from .optimize.model_optimizer import optimize_with_accuracy_guard
from .optimize.global_guard import optimize_global_guard
from .optimize.accelerated_linear import GroupwiseInt2Linear
from .optimize.hybrid_int2 import HybridInt2Linear
from .optimize.torchao_backend import apply_torchao_int4,apply_torchao_int8,apply_torchao_int2
from .optimize.sparse24_model import apply_sparse24
from .compare import annotate_model_matrix
from .model_arch import inspect_model_architecture
from .runtime_policy import choose_runtime_policy
from .optimize.robust_guard import optimize_robust_mixedbit
from .runtime_v5 import prepare_static_compile,generate_static_benchmark


@dataclass
class GenerationMetrics:
    model:str
    method:str
    prompt_tokens:int
    generated_tokens:int
    ttft_ms:float
    decode_tokens_s:float
    total_tokens_s:float
    avg_power_w:float|None
    joules_per_generated_token:float|None
    peak_vram_gb:float|None
    current_vram_gb:float|None
    reserved_vram_gb:float|None
    model_storage_gb:float|None
    perplexity:float|None
    plan_layers_quantized:int|None
    output_text:str
    architecture:str|None=None
    runtime_backend:str|None=None
    guard_stats:str|None=None
    int2_layers:int|None=None
    int4_layers:int|None=None
    fp16_layers:int|None=None
    error:str|None=None
    def row(self): return asdict(self)


def _dtype(name):
    if name=='bf16': return torch.bfloat16
    if name=='fp32': return torch.float32
    return torch.float16


def _load_kwargs(dtype,trust_remote_code,method,device):
    kwargs={'dtype':_dtype(dtype),'trust_remote_code':trust_remote_code}
    if device=='cuda': kwargs['device_map']='cuda'
    if method=='hf-kernels-auto': kwargs['use_kernels']=True
    elif method in ('sdpa','flash-attention-2','flash-attention-3','paged-flash-attention-3','hf-flash-attn2-kernel','hf-flash-attn3-kernel'):
        kwargs['attn_implementation']={
            'sdpa':'sdpa',
            'flash-attention-2':'flash_attention_2',
            'flash-attention-3':'flash_attention_3',
            'paged-flash-attention-3':'paged|flash_attention_3',
            'hf-flash-attn2-kernel':'kernels-community/flash-attn2',
            'hf-flash-attn3-kernel':'kernels-community/flash-attn3',
        }[method]
    return kwargs


def load_hf(model_id,device='cuda',dtype='fp16',trust_remote_code=False,method=None):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    tok=AutoTokenizer.from_pretrained(model_id,trust_remote_code=trust_remote_code)
    kwargs=_load_kwargs(dtype,trust_remote_code,method,device)
    model=AutoModelForCausalLM.from_pretrained(model_id,**kwargs)
    if device!='cuda': model=model.to(device)
    model.eval()
    return model,tok


def _target_linear_names(model):
    toks=('q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','fc1','fc2','c_attn','c_proj')
    return [n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and any(t in n for t in toks)]


def _replace_all_int2(model,group_size=128,residual_rank=0,hybrid=False,outlier_columns=0):
    names=_target_linear_names(model); replaced=0
    for n in names:
        ps=n.split('.'); p=model
        for q in ps[:-1]: p=getattr(p,q)
        src=getattr(p,ps[-1])
        try:
            if hybrid: repl=HybridInt2Linear(src,group_size,None,outlier_columns,residual_rank,'auto')
            else: repl=GroupwiseInt2Linear(src,group_size,None,residual_rank,'auto')
            setattr(p,ps[-1],repl.to(src.weight.device)); replaced+=1
        except Exception:
            continue
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return replaced


def apply_method(model,tokenizer,method,calibration_texts,device,group_size=128,
                 nmse_limit=.06,cosine_limit=.96,residual_rank=8,global_loss_budget=.02,expected_decode_tokens=64):
    plan_count=None; guard_stats=None
    passthrough=('baseline-fp16','baseline-bf16','hf-kernels-auto','sdpa','flash-attention-2','flash-attention-3',
                 'paged-flash-attention-3','hf-flash-attn2-kernel','hf-flash-attn3-kernel')
    if method in passthrough:
        if method=='baseline-static-compile':
            _,st=prepare_static_compile(model,fullgraph=False);guard_stats=json.dumps({'runtime':st},ensure_ascii=False)
        return plan_count,guard_stats
    if method=='torch-compile':
        if hasattr(torch,'compile'): model.forward=torch.compile(model.forward,mode='reduce-overhead',fullgraph=False)
    elif method=='torchao-int4': apply_torchao_int4(model,group_size)
    elif method=='torchao-int8': apply_torchao_int8(model)
    elif method=='torchao-int2': apply_torchao_int2(model,group_size)
    elif method=='native-2to4': plan_count=apply_sparse24(model)
    elif method=='custom-int2-fast': plan_count=_replace_all_int2(model,group_size,0)
    elif method=='custom-int2-residual': plan_count=_replace_all_int2(model,group_size,residual_rank)
    elif method=='custom-int2-hybrid-fast': plan_count=_replace_all_int2(model,group_size,0,True,16)
    elif method=='custom-int2-accuracy-guard':
        decisions,replaced=optimize_with_accuracy_guard(model,tokenizer,calibration_texts,device,group_size,nmse_limit,cosine_limit,(0,4,8,residual_rank))
        plan_count=replaced
    elif method in ('custom-mixedbit-v5','custom-mixedbit-v5-graph'):
        decisions,replaced,stats=optimize_robust_mixedbit(
            model,tokenizer,calibration_texts,device,
            group_sizes=tuple(dict.fromkeys([group_size,128,64,32])),
            local_nmse=nmse_limit,local_cosine=cosine_limit,
            mean_loss_budget=global_loss_budget,p95_loss_budget=max(global_loss_budget*2,.03),worst_loss_budget=max(global_loss_budget*4,.06),
            outliers=(0,8,16,32),residual_ranks=tuple(dict.fromkeys([0,4,8,residual_rank])),expected_decode_tokens=expected_decode_tokens,
        )
        plan_count=replaced
        if method=='custom-mixedbit-v5-graph':
            _,runtime_st=prepare_static_compile(model,fullgraph=False);stats['runtime']=runtime_st
        guard_stats=json.dumps(stats,ensure_ascii=False)
    elif method in ('custom-int2-v4','custom-int2-global-guard'):
        decisions,replaced,stats=optimize_global_guard(
            model,tokenizer,calibration_texts,device,
            group_sizes=tuple(dict.fromkeys([group_size,64,32])),
            local_nmse=nmse_limit,local_cosine=cosine_limit,
            max_loss_rel_increase=global_loss_budget,
            outliers=(0,8,16,32),residual_ranks=tuple(dict.fromkeys([0,4,8,residual_rank])),
        )
        plan_count=replaced; guard_stats=json.dumps(stats,ensure_ascii=False)
    else: raise ValueError(f'unknown method: {method}')
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return plan_count,guard_stats


def _tensor_storage_gb(model):
    seen=set(); total=0
    for t in list(model.parameters())+list(model.buffers()):
        if t is None: continue
        try:
            key=(t.device.type,t.device.index,t.untyped_storage().data_ptr())
            if key in seen: continue
            seen.add(key); total+=t.untyped_storage().nbytes()
        except Exception:
            total+=t.numel()*t.element_size()
    return total/1024**3


def _cuda_event_ms(fn):
    start=torch.cuda.Event(enable_timing=True); end=torch.cuda.Event(enable_timing=True)
    start.record(); result=fn(); end.record(); end.synchronize()
    return result,float(start.elapsed_time(end))


@torch.no_grad()
def greedy_benchmark(model,tokenizer,prompt,max_new_tokens=64,device='cuda',force_exact_tokens=True):
    """Fixed-length greedy benchmark.

    v3 synchronized CUDA once per generated token, which can dominate small-model
    latency. v4 records one aggregate CUDA event around the decode chain. When
    force_exact_tokens=True (default), EOS does not stop early, so models are
    compared on the same number of decode steps.
    """
    batch=tokenizer(prompt,return_tensors='pt'); batch={k:v.to(device) for k,v in batch.items()}
    input_ids=batch['input_ids']; attention_mask=batch.get('attention_mask',torch.ones_like(input_ids))
    if device=='cuda':
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    power=NvmlPowerSampler(0); power.start(); wall0=time.perf_counter()

    def prefill(): return model(input_ids=input_ids,attention_mask=attention_mask,use_cache=True)
    if device=='cuda': out,ttft=_cuda_event_ms(prefill)
    else:
        s=time.perf_counter(); out=prefill(); ttft=(time.perf_counter()-s)*1000

    past=out.past_key_values; token=out.logits[:,-1,:].argmax(-1,keepdim=True); generated=[token]
    eos=getattr(tokenizer,'eos_token_id',None)

    def decode_loop():
        nonlocal token,past,attention_mask
        for _ in range(max(0,max_new_tokens-1)):
            attention_mask=torch.cat([attention_mask,torch.ones((attention_mask.shape[0],1),device=device,dtype=attention_mask.dtype)],1)
            o=model(input_ids=token,attention_mask=attention_mask,past_key_values=past,use_cache=True)
            past=o.past_key_values; token=o.logits[:,-1,:].argmax(-1,keepdim=True); generated.append(token)
            if not force_exact_tokens and eos is not None and bool((token==eos).all()): break
        return None
    if device=='cuda': _,decode_ms=_cuda_event_ms(decode_loop)
    else:
        s=time.perf_counter(); decode_loop(); decode_ms=(time.perf_counter()-s)*1000

    if device=='cuda': torch.cuda.synchronize()
    wall_total=time.perf_counter()-wall0; avg_power=power.stop()
    ids=torch.cat(generated,1); n=ids.shape[1]; decode_s=decode_ms/1000.0
    decode_tps=(max(0,n-1)/decode_s) if decode_s>0 else 0.0
    total_gpu_s=(ttft+decode_ms)/1000.0; total_tps=n/total_gpu_s if total_gpu_s>0 else 0.0
    peak=(torch.cuda.max_memory_allocated()/1024**3) if device=='cuda' else None
    current=(torch.cuda.memory_allocated()/1024**3) if device=='cuda' else None
    reserved=(torch.cuda.memory_reserved()/1024**3) if device=='cuda' else None
    joules=None if avg_power is None else avg_power*wall_total/max(1,n)
    text=tokenizer.decode(ids[0],skip_special_tokens=True)
    return dict(prompt_tokens=input_ids.shape[1],generated_tokens=n,ttft_ms=ttft,
                decode_tokens_s=decode_tps,total_tokens_s=total_tps,avg_power_w=avg_power,
                joules_per_generated_token=joules,peak_vram_gb=peak,current_vram_gb=current,
                reserved_vram_gb=reserved,output_text=text)


@torch.no_grad()
def perplexity(model,tokenizer,text,device='cuda',max_length=512):
    if not text.strip(): return None
    texts=[x.strip() for x in text.split('---') if x.strip()]
    total_loss=0.0; total_tokens=0
    for item in texts:
        b=tokenizer(item,return_tensors='pt',truncation=True,max_length=max_length); ids=b['input_ids'].to(device)
        if ids.shape[1]<2: continue
        out=model(input_ids=ids,use_cache=False); logits=out.logits[:,:-1,:].float(); labels=ids[:,1:]
        loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),labels.reshape(-1),reduction='sum')
        total_loss+=float(loss.item()); total_tokens+=labels.numel()
    if total_tokens==0:return None
    return float(torch.exp(torch.tensor(total_loss/total_tokens)).item())


def benchmark_model_method(model_id,method,prompt,calibration_texts,eval_text,max_new_tokens=64,
                           device='cuda',dtype='fp16',group_size=128,nmse_limit=.06,cosine_limit=.96,
                           residual_rank=8,global_loss_budget=.02,force_exact_tokens=True):
    model=None
    try:
        model,tok=load_hf(model_id,device,dtype,method=method)
        arch=inspect_model_architecture(model); policy=choose_runtime_policy(m=1)
        plan_count,guard_stats=apply_method(model,tok,method,calibration_texts,device,group_size,nmse_limit,cosine_limit,residual_rank,global_loss_budget,max_new_tokens)
        # If the robust guard rejected every quantized layer, reload an exact
        # baseline model so no custom-runtime or allocator residue can distort
        # the baseline-equivalent measurement.
        if method in ('custom-mixedbit-v5','custom-mixedbit-v5-graph') and plan_count==0:
            del model; gc.collect()
            if device=='cuda': torch.cuda.empty_cache()
            model,tok=load_hf(model_id,device,dtype,method='baseline-fp16')
            if method=='custom-mixedbit-v5-graph': prepare_static_compile(model,fullgraph=False)
        b=tok(prompt,return_tensors='pt'); b={k:v.to(device) for k,v in b.items()}
        if method not in ('baseline-static-compile','custom-mixedbit-v5-graph'):
            with torch.no_grad(): model(**b,use_cache=False)
        gc.collect()
        if device=='cuda': torch.cuda.synchronize(); torch.cuda.empty_cache()
        storage=_tensor_storage_gb(model)
        if method in ('baseline-static-compile','custom-mixedbit-v5-graph'):
            g=generate_static_benchmark(model,tok,prompt,max_new_tokens,device,warmups=2)
        else:
            g=greedy_benchmark(model,tok,prompt,max_new_tokens,device,force_exact_tokens)
        ppl=perplexity(model,tok,eval_text,device) if eval_text else None
        runtime_backend=policy.decode_backend
        if method=='baseline-static-compile': runtime_backend='hf-static-cache+torch.compile'
        elif method=='custom-mixedbit-v5': runtime_backend='mixedbit-int2-int4-eager'
        elif method=='custom-mixedbit-v5-graph': runtime_backend='mixedbit-int2-int4+static-cache+torch.compile'
        i2=i4=f16=None
        if guard_stats:
            try:
                gs=json.loads(guard_stats); i2=gs.get('int2_layers');i4=gs.get('int4_layers');f16=gs.get('fp16_layers')
            except Exception: pass
        result=GenerationMetrics(model_id,method,perplexity=ppl,plan_layers_quantized=plan_count,
                                 model_storage_gb=storage,architecture=arch.family,runtime_backend=runtime_backend,
                                 guard_stats=guard_stats,int2_layers=i2,int4_layers=i4,fp16_layers=f16,error=None,**g)
    except Exception as e:
        result=GenerationMetrics(model_id,method,0,0,0,0,0,None,None,None,None,None,None,None,None,'',error=repr(e))
    finally:
        try: del model
        except Exception: pass
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def benchmark_matrix(model_ids,methods,prompt,calibration_texts,eval_text,max_new_tokens=64,device='cuda',dtype='fp16',**kwargs):
    rows=[]
    for mid in model_ids:
        for method in methods:
            rows.append(benchmark_model_method(mid,method,prompt,calibration_texts,eval_text,max_new_tokens,device,dtype,**kwargs).row())
    return annotate_model_matrix(pd.DataFrame(rows))


def build_optimized_bundle(model_id,out_dir,calibration_texts,device='cuda',dtype='fp16',
                           group_size=128,nmse_limit=.06,cosine_limit=.96,residual_rank=8,global_loss_budget=.02):
    from .optimize.bundle import save_bundle
    model,tok=load_hf(model_id,device,dtype)
    decisions,replaced,stats=optimize_global_guard(model,tok,calibration_texts,device,
        group_sizes=tuple(dict.fromkeys([group_size,64,32])),local_nmse=nmse_limit,local_cosine=cosine_limit,
        max_loss_rel_increase=global_loss_budget,outliers=(0,8,16,32),residual_ranks=tuple(dict.fromkeys([0,4,8,residual_rank])))
    out=save_bundle(model,tok,model_id,decisions,out_dir,{'group_size':group_size,'local_nmse':nmse_limit,'local_cosine':cosine_limit,
        'residual_rank':residual_rank,'replaced':replaced,'global_loss_budget':global_loss_budget,'guard_stats':stats,'format_version':4})
    df=pd.DataFrame([d.to_dict() for d in decisions]); return df,out,replaced


def build_optimized_bundle_v5(model_id,out_dir,calibration_texts,device='cuda',dtype='fp16',group_size=128,
                              nmse_limit=.06,cosine_limit=.96,residual_rank=8,mean_loss_budget=.02):
    from .optimize.bundle import save_bundle
    model,tok=load_hf(model_id,device,dtype)
    decisions,replaced,stats=optimize_robust_mixedbit(
        model,tok,calibration_texts,device,group_sizes=tuple(dict.fromkeys([group_size,128,64,32])),
        local_nmse=nmse_limit,local_cosine=cosine_limit,mean_loss_budget=mean_loss_budget,
        p95_loss_budget=max(mean_loss_budget*2,.03),worst_loss_budget=max(mean_loss_budget*4,.06),
        outliers=(0,8,16,32),residual_ranks=tuple(dict.fromkeys([0,4,8,residual_rank])))
    out=save_bundle(model,tok,model_id,decisions,out_dir,{'format_version':5,'group_size':group_size,
        'local_nmse':nmse_limit,'local_cosine':cosine_limit,'residual_rank':residual_rank,
        'replaced':replaced,'mean_loss_budget':mean_loss_budget,'guard_stats':stats})
    return pd.DataFrame([d.to_dict() for d in decisions]),out,replaced,stats
