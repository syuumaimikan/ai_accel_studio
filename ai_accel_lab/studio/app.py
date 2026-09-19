from __future__ import annotations

import json
from pathlib import Path
import pandas as pd

from ..hardware import detect_hardware,recommended_backends
from ..selftest import run_selftest
from ..kernel_suite import run_kernel_suite
from ..model_benchmark import benchmark_matrix,build_optimized_bundle
from ..fusion_benchmark import run_fusion_benchmark
from .results import ResultStore

STORE=ResultStore('results')
CSS='.gradio-container {max-width: 1500px !important;} .hero {font-size: 1.08rem;}'


def _fig(df,x,y,title):
    try:
        import plotly.express as px
        d=df.dropna(subset=[y])
        if d.empty: return None
        return px.bar(d,x=x,y=y,color=x,title=title,text_auto='.3s')
    except Exception:
        return None


def system_report():
    info=detect_hardware()
    d=info.to_dict(); d['recommended_backends']=recommended_backends(info)
    return '```json\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n```'


def run_kernels(m,n,k,group,repeats,methods,power):
    try:
        df=run_kernel_suite(int(m),int(n),int(k),int(group),int(repeats),bool(power),methods)
        rid,csv,js=STORE.save(df,'kernel',{'M':m,'N':n,'K':k,'group':group})
        return df,_fig(df,'method','speedup','Speedup vs torch FP16'),_fig(df,'method','energy_mj','Energy / call (mJ)'),f'Saved run: {rid}\n{csv}'
    except Exception as e:
        return pd.DataFrame([{'error':repr(e)}]),None,None,repr(e)


def run_models(model_text,methods,prompt,calibration,eval_text,max_new,dtype,group,nmse,cosine,resrank):
    models=[x.strip() for x in model_text.replace(',','\n').splitlines() if x.strip()]
    cals=[x.strip() for x in calibration.split('\n---\n') if x.strip()] or [prompt]
    try:
        df=benchmark_matrix(models,methods,prompt,cals,eval_text,int(max_new),'cuda',dtype,
                            group_size=int(group),nmse_limit=float(nmse),cosine_limit=float(cosine),residual_rank=int(resrank))
        rid,csv,js=STORE.save(df,'model',{'models':models,'methods':methods})
        return (df,
                _fig(df.assign(label=df['model']+' | '+df['method']),'label','decode_tokens_s','Decode tokens/s'),
                _fig(df.assign(label=df['model']+' | '+df['method']),'label','joules_per_generated_token','Joules / generated token'),
                _fig(df.assign(label=df['model']+' | '+df['method']),'label','perplexity','Perplexity (lower is better)'),
                f'Saved run: {rid}\n{csv}')
    except Exception as e:
        return pd.DataFrame([{'error':repr(e)}]),None,None,None,repr(e)


def list_runs():
    rows=[]
    for p in STORE.list_runs()[:100]:
        try:
            obj=json.loads(p.read_text(encoding='utf-8'))
            rows.append({'file':str(p),'id':obj.get('id'),'kind':obj.get('kind'),'records':len(obj.get('records',[]))})
        except Exception: pass
    return pd.DataFrame(rows)


def create_app():
    import gradio as gr
    with gr.Blocks(title='AI Acceleration Studio') as demo:
        gr.Markdown('''# AI Acceleration Studio v3\n<div class="hero">Hopper / Blackwell GPU kernels, groupwise INT2, 2:4 sparsity, accuracy guard, energy and multi-model comparison.</div>''')
        with gr.Tabs():
            with gr.Tab('Hardware'):
                sys=gr.Markdown(system_report())
                with gr.Row():
                    gr.Button('Refresh').click(system_report,outputs=sys)
                    selfbtn=gr.Button('Run advanced GPU self-test')
                selfout=gr.Code(label='Self-test report',language='json')
                def _selftest(): return json.dumps(run_selftest(False),ensure_ascii=False,indent=2)
                selfbtn.click(_selftest,outputs=selfout)
                gr.Markdown('Advanced backends are enabled only after architecture checks/self-tests. Unsupported paths fall back instead of silently producing wrong results.')

            with gr.Tab('Kernel Lab'):
                with gr.Row():
                    m=gr.Number(32,label='M / tokens',precision=0); n=gr.Number(4096,label='N / out',precision=0); k=gr.Number(4096,label='K / in',precision=0)
                    gs=gr.Dropdown([32,64,128,256],value=128,label='INT2 group size'); rep=gr.Number(100,label='Repeats',precision=0)
                meth=gr.CheckboxGroup(['torch-fp16','groupwise-int2','persistent-int2','native-2to4','tma','hb-gluon','blackwell-warp-specialized'],
                                      value=['torch-fp16','groupwise-int2','persistent-int2','native-2to4'],label='Backends')
                power=gr.Checkbox(True,label='Measure NVML power')
                run=gr.Button('Run kernel comparison',variant='primary')
                table=gr.Dataframe(label='Kernel results',interactive=False)
                with gr.Row(): sp=gr.Plot(label='Speed'); en=gr.Plot(label='Energy')
                status=gr.Textbox(label='Run status')
                run.click(run_kernels,[m,n,k,gs,rep,meth,power],[table,sp,en,status])

            with gr.Tab('Model Matrix'):
                models=gr.Textbox(value='Qwen/Qwen2.5-0.5B-Instruct\nHuggingFaceTB/SmolLM2-360M-Instruct',lines=3,label='Hugging Face model IDs or local paths — one per line')
                methods=gr.CheckboxGroup(
                    ['baseline-fp16','hf-kernels-auto','sdpa','hf-flash-attn2-kernel','hf-flash-attn3-kernel','flash-attention-2','flash-attention-3','paged-flash-attention-3','torch-compile','torchao-int4','torchao-int8','torchao-int2','native-2to4','custom-int2-fast','custom-int2-residual','custom-int2-accuracy-guard'],
                    value=['baseline-fp16','hf-kernels-auto','torchao-int4','custom-int2-accuracy-guard'],label='Methods')
                prompt=gr.Textbox('Explain why GPU memory bandwidth matters for LLM inference in two sentences.',lines=3,label='Benchmark prompt')
                calibration=gr.Textbox('The quick brown fox jumps over the lazy dog.\n---\nWrite a Python function that computes Fibonacci numbers.\n---\nExplain matrix multiplication.',lines=6,label='Calibration prompts (separate with ---)')
                evaltxt=gr.Textbox('Artificial intelligence systems execute matrix operations repeatedly. Efficient memory access and numerical formats can reduce inference cost while preserving useful model quality.',lines=4,label='Perplexity evaluation text')
                with gr.Row():
                    maxnew=gr.Slider(4,256,value=32,step=1,label='New tokens'); dtype=gr.Dropdown(['fp16','bf16'],value='fp16',label='Base dtype'); group=gr.Dropdown([32,64,128,256],value=128,label='INT2 group')
                    rank=gr.Dropdown([0,4,8,16,32],value=8,label='Max residual rank')
                with gr.Row(): nmse=gr.Number(.025,label='Accuracy guard NMSE max'); cos=gr.Number(.985,label='Cosine min')
                runm=gr.Button('Benchmark models sequentially',variant='primary')
                mt=gr.Dataframe(label='Model comparison',interactive=False)
                with gr.Row(): mtps=gr.Plot(); meng=gr.Plot(); mppl=gr.Plot()
                mst=gr.Textbox(label='Run status')
                runm.click(run_models,[models,methods,prompt,calibration,evaltxt,maxnew,dtype,group,nmse,cos,rank],[mt,mtps,meng,mppl,mst])


            with gr.Tab('Giant Fusion'):
                gr.Markdown('Single-token decode experiment: **groupwise INT2 Q/K/V projection + RoPE + KV-cache append + causal attention in one Triton kernel**. Current implementation targets MHA with head_dim 64/128; unsupported GQA/MQA falls back in model runtime.')
                with gr.Row():
                    fh=gr.Number(4096,label='Hidden size',precision=0); fheads=gr.Number(32,label='Heads',precision=0); fseq=gr.Number(128,label='Cache sequence',precision=0); fg=gr.Dropdown([32,64,128,256],value=128,label='INT2 group'); frep=gr.Number(50,label='Repeats',precision=0)
                frun=gr.Button('Run giant-fusion comparison',variant='primary')
                ft=gr.Dataframe(label='Fusion results',interactive=False); fp=gr.Plot(); fs=gr.Textbox(label='Status')
                def _fusion(hidden,heads,seq,group,repeats):
                    try:
                        df=run_fusion_benchmark(int(hidden),int(heads),int(seq),int(group),int(repeats))
                        rid,csv,_=STORE.save(df,'fusion',{'hidden':hidden,'heads':heads,'seq':seq})
                        return df,_fig(df,'method','speedup','Giant fusion speedup'),f'Saved run: {rid}\n{csv}'
                    except Exception as e:
                        return pd.DataFrame([{'error':repr(e)}]),None,repr(e)
                frun.click(_fusion,[fh,fheads,fseq,fg,frep],[ft,fp,fs])

            with gr.Tab('Accuracy Guard'):
                gr.Markdown('''### How it works\nThe optimizer records representative Linear inputs, estimates per-input-feature RMS, chooses groupwise INT2 scales using activation-weighted error, then tries low-rank residual ranks. A layer stays FP16 when it cannot meet the NMSE/cosine guard. This targets **accuracy preservation/recovery**, not a promise that quantization will exceed the original model quality.''')
                amodel=gr.Textbox('Qwen/Qwen2.5-0.5B-Instruct',label='Model ID / local path')
                acal=gr.Textbox('Explain matrix multiplication.\n---\nWrite a short Python function.\n---\nSummarize why memory bandwidth matters.',lines=5,label='Calibration prompts')
                with gr.Row(): aout=gr.Textbox('optimized_bundles/model-int2',label='Output bundle directory'); ag=gr.Dropdown([32,64,128,256],value=128,label='Group'); ar=gr.Dropdown([4,8,16,32],value=8,label='Max residual rank')
                with gr.Row(): an=gr.Number(.025,label='NMSE max'); ac=gr.Number(.985,label='Cosine min')
                ab=gr.Button('Build reusable optimized bundle',variant='primary')
                at=gr.Dataframe(label='Layer plan',interactive=False); ast=gr.Textbox(label='Status')
                def _bundle(mid,cal,out,g,r,n,c):
                    texts=[x.strip() for x in cal.split('---') if x.strip()]
                    try:
                        df,path,count=build_optimized_bundle(mid,out,texts,'cuda','fp16',int(g),float(n),float(c),int(r))
                        return df,f'Bundle saved: {path}\nOptimized layers: {count}'
                    except Exception as e:
                        return pd.DataFrame([{'error':repr(e)}]),repr(e)
                ab.click(_bundle,[amodel,acal,aout,ag,ar,an,ac],[at,ast])

            with gr.Tab('Architecture paths'):
                gr.Markdown('''### Execution hierarchy\n1. **Blackwell**: optional Gluon tcgen05 + Tensor Memory + TMA multi-buffer persistent path; standard TMA/INT2 fallback.\n2. **Hopper**: optional Gluon asynchronous WGMMA + TMA persistent path; standard TMA/INT2 fallback.\n3. **Ampere/Ada**: custom Triton INT2 plus native 2:4 semi-structured sparse Tensor Core path.\n4. **Other CUDA**: standard Triton/PyTorch fallback.\n\n`tools/verify_codegen.py` scans PTX/SASS for WGMMA, MMA.SP, TMA and TCGEN05 markers so benchmark claims can be checked against generated code.''')

            with gr.Tab('Runs'):
                runs=gr.Dataframe(value=list_runs(),label='Saved results',interactive=False)
                gr.Button('Refresh runs').click(list_runs,outputs=runs)
        gr.Markdown('**Important:** the fastest method depends on GPU, shape, model and context length. The studio reports measured results rather than assuming INT2/sparsity is always faster.')
    return demo


def main():
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('--host',default='127.0.0.1'); p.add_argument('--port',type=int,default=7860); p.add_argument('--share',action='store_true')
    a=p.parse_args(); create_app().launch(server_name=a.host,server_port=a.port,share=a.share,css=CSS)

if __name__=='__main__': main()
