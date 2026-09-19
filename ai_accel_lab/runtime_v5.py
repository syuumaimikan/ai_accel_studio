from __future__ import annotations
import time
import torch
from .power import NvmlPowerSampler


def prepare_static_compile(model,fullgraph=False):
    """Configure Hugging Face static KV cache + torch.compile decode path.

    Returns (model, status). Fails soft because not every architecture/custom op
    is graph-safe. The caller may benchmark the eager path if compilation fails.
    """
    status={'static_cache':False,'compiled':False,'graph_safe_custom_layers':0,'error':None}
    try:
        for mod in model.modules():
            if hasattr(mod,'graph_safe'):
                mod.graph_safe=True;status['graph_safe_custom_layers']+=1
        if hasattr(model,'generation_config'):
            model.generation_config.cache_implementation='static';status['static_cache']=True
        if hasattr(torch,'compile'):
            model.forward=torch.compile(model.forward,mode='reduce-overhead',fullgraph=fullgraph);status['compiled']=True
    except Exception as e:status['error']=repr(e)
    return model,status


def _cuda_event_ms(fn):
    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();r=fn();end.record();end.synchronize();return r,float(start.elapsed_time(end))

@torch.no_grad()
def generate_static_benchmark(model,tokenizer,prompt,max_new_tokens=64,device='cuda',warmups=2):
    """End-to-end fixed-token generation using HF StaticCache/compiled forward.

    The generated length is forced by min_new_tokens=max_new_tokens. Decode
    throughput is derived from total GPU time minus an independently measured
    prefill. This avoids synchronizing once per token and measures the real
    generate/runtime overhead that the user experiences.
    """
    old_padding=getattr(tokenizer,'padding_side','right');tokenizer.padding_side='left'
    if getattr(tokenizer,'pad_token_id',None) is None and getattr(tokenizer,'eos_token_id',None) is not None:
        tokenizer.pad_token_id=tokenizer.eos_token_id
    batch=tokenizer(prompt,return_tensors='pt',padding=True,pad_to_multiple_of=8);batch={k:v.to(device) for k,v in batch.items()}
    input_ids=batch['input_ids'];attn=batch.get('attention_mask')

    def prefill():return model(input_ids=input_ids,attention_mask=attn,use_cache=True)
    # Compile/warm prefill before timing. Compilation is a one-time deployment cost,
    # not token latency, and otherwise dwarfs small-model TTFT.
    for _ in range(max(1,int(warmups))):prefill()
    if device=='cuda':
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();_,ttft=_cuda_event_ms(prefill)
    else:
        s=time.perf_counter();prefill();ttft=(time.perf_counter()-s)*1000

    genkw=dict(do_sample=False,min_new_tokens=int(max_new_tokens),max_new_tokens=int(max_new_tokens),use_cache=True,cache_implementation='static')
    # Warm compilation/cache initialization outside the measured window.
    for _ in range(max(1,int(warmups))):
        try:model.generate(**batch,**genkw)
        except TypeError:
            genkw.pop('cache_implementation',None);model.generate(**batch,**genkw)
    if device=='cuda':torch.cuda.synchronize()
    power=NvmlPowerSampler(0);power.start();wall0=time.perf_counter()
    if device=='cuda':out,total_ms=_cuda_event_ms(lambda:model.generate(**batch,**genkw))
    else:
        s=time.perf_counter();out=model.generate(**batch,**genkw);total_ms=(time.perf_counter()-s)*1000
    if device=='cuda':torch.cuda.synchronize()
    wall=time.perf_counter()-wall0;avg=power.stop();tokenizer.padding_side=old_padding
    prompt_n=input_ids.shape[1];generated=max(0,out.shape[1]-prompt_n);decode_ms=max(.001,total_ms-ttft)
    decode_tps=generated/(decode_ms/1000.0);total_tps=generated/(total_ms/1000.0)
    peak=(torch.cuda.max_memory_allocated()/1024**3) if device=='cuda' else None;cur=(torch.cuda.memory_allocated()/1024**3) if device=='cuda' else None;res=(torch.cuda.memory_reserved()/1024**3) if device=='cuda' else None
    joules=None if avg is None else avg*wall/max(1,generated)
    text=tokenizer.decode(out[0,prompt_n:],skip_special_tokens=True)
    return dict(prompt_tokens=prompt_n,generated_tokens=generated,ttft_ms=ttft,decode_tokens_s=decode_tps,total_tokens_s=total_tps,
                avg_power_w=avg,joules_per_generated_token=joules,peak_vram_gb=peak,current_vram_gb=cur,reserved_vram_gb=res,
                output_text=text)
