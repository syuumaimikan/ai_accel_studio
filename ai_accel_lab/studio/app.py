from __future__ import annotations
import json,importlib.metadata as im
from pathlib import Path
import pandas as pd

from ..hardware import detect_hardware,recommended_backends
from ..selftest import run_selftest
from ..kernel_suite import run_kernel_suite
from ..model_benchmark import benchmark_matrix,build_optimized_bundle,build_optimized_bundle_v5
from ..fusion_benchmark import run_fusion_benchmark
from ..runtime_policy import choose_runtime_policy
from ..doctor import dependency_report,recommendations
from ..calibration_suite import joined_calibration,joined_eval
from .results import ResultStore

STORE=ResultStore('results')
CSS='.gradio-container {max-width: 1650px !important;} .hero {font-size: 1.08rem;}'

def _fig(df,x,y,title):
    try:
        import plotly.express as px
        d=df.dropna(subset=[y]); return None if d.empty else px.bar(d,x=x,y=y,color=x,title=title,text_auto='.3s')
    except Exception:return None

def system_report():
    info=detect_hardware(); d=info.to_dict(); d['recommended_backends']=recommended_backends(info); d['decode_policy']=choose_runtime_policy(m=1).__dict__
    return '```json\n'+json.dumps(d,ensure_ascii=False,indent=2)+'\n```'

def package_report():
    return dependency_report()

def run_kernels(m,n,k,group,repeats,methods,power):
    try:
        df=run_kernel_suite(int(m),int(n),int(k),int(group),int(repeats),bool(power),methods)
        rid,csv,_=STORE.save(df,'kernel',{'M':m,'N':n,'K':k,'group':group})
        return df,_fig(df,'method','speedup','Speedup vs torch FP16'),_fig(df,'method','energy_mj','Energy / call (mJ)'),f'Saved run: {rid}\n{csv}'
    except Exception as e:return pd.DataFrame([{'error':repr(e)}]),None,None,repr(e)

def run_models(model_text,methods,prompt,calibration,eval_text,max_new,dtype,group,nmse,cosine,resrank,global_budget):
    models=[x.strip() for x in model_text.replace(',','\n').splitlines() if x.strip()]
    cals=[x.strip() for x in calibration.split('\n---\n') if x.strip()] or [prompt]
    try:
        df=benchmark_matrix(models,methods,prompt,cals,eval_text,int(max_new),'cuda',dtype,
            group_size=int(group),nmse_limit=float(nmse),cosine_limit=float(cosine),residual_rank=int(resrank),
            global_loss_budget=float(global_budget),force_exact_tokens=True)
        rid,csv,_=STORE.save(df,'model',{'models':models,'methods':methods,'fixed_tokens':int(max_new)})
        labels=df['model']+' | '+df['method'] if not df.empty else []
        return df,_fig(df.assign(label=labels),'label','decode_tokens_s','Decode tokens/s'),_fig(df.assign(label=labels),'label','joules_per_generated_token','J/token'),_fig(df.assign(label=labels),'label','perplexity','Perplexity'),f'Saved run: {rid}\n{csv}'
    except Exception as e:return pd.DataFrame([{'error':repr(e)}]),None,None,None,repr(e)

def list_runs():
    rows=[]
    for p in STORE.list_runs()[:100]:
        try:
            obj=json.loads(p.read_text(encoding='utf-8'));rows.append({'file':str(p),'id':obj.get('id'),'kind':obj.get('kind'),'records':len(obj.get('records',[]))})
        except Exception:pass
    return pd.DataFrame(rows)

def create_app():
    import gradio as gr
    with gr.Blocks(title='AI Acceleration Studio v5') as demo:
        gr.Markdown('# AI Acceleration Studio v5\n<div class="hero">Mixed-bit INT2/INT4/FP16 planning, held-out robust quality guard, Static KV Cache + CUDA-Graph-friendly compilation, decode kernels, and measured multi-model comparison.</div>')
        with gr.Tabs():
            with gr.Tab('Hardware & diagnostics'):
                sys=gr.Markdown(system_report()); pk=gr.Dataframe(value=package_report(),interactive=False,label='Packages')
                dep=gr.Code(value=lambda:json.dumps(recommendations(),ensure_ascii=False,indent=2),label='Dependency Doctor',language='json')
                with gr.Row():
                    gr.Button('Refresh').click(lambda:(system_report(),package_report(),json.dumps(recommendations(),ensure_ascii=False,indent=2)),outputs=[sys,pk,dep])
                    selfbtn=gr.Button('Run GPU self-test')
                selfout=gr.Code(label='Self-test',language='json');selfbtn.click(lambda:json.dumps(run_selftest(False),ensure_ascii=False,indent=2),outputs=selfout)
                gr.Markdown('The Doctor reads installed package metadata to discover the exact `transformers -> kernels` and `torchao -> mslk` requirements instead of guessing a version.')
            with gr.Tab('Kernel Lab'):
                with gr.Row():
                    m=gr.Number(1,label='M',precision=0);n=gr.Number(4096,label='N',precision=0);k=gr.Number(4096,label='K',precision=0);gs=gr.Dropdown([32,64,128,256],128,label='INT2 group');rep=gr.Number(300,label='Repeats',precision=0)
                meth=gr.CheckboxGroup(['torch-fp16','decode-int2-v4','decode-int4-v5','groupwise-int2','persistent-int2','native-2to4','tma','hb-gluon','blackwell-warp-specialized'],value=['torch-fp16','decode-int2-v4','decode-int4-v5','groupwise-int2','native-2to4'],label='Backends')
                power=gr.Checkbox(True,label='NVML power');run=gr.Button('Run kernel comparison',variant='primary')
                table=gr.Dataframe(interactive=False);sp=gr.Plot();en=gr.Plot();status=gr.Textbox()
                run.click(run_kernels,[m,n,k,gs,rep,meth,power],[table,sp,en,status])
            with gr.Tab('Model Matrix'):
                models=gr.Textbox('Qwen/Qwen2.5-0.5B-Instruct\nHuggingFaceTB/SmolLM2-360M-Instruct',lines=3,label='Model IDs / local paths')
                methods=gr.CheckboxGroup([
                    'baseline-fp16','baseline-static-compile','hf-kernels-auto','sdpa','hf-flash-attn2-kernel','hf-flash-attn3-kernel','flash-attention-2','flash-attention-3','paged-flash-attention-3','torch-compile','torchao-int4','torchao-int8','torchao-int2','native-2to4','custom-int2-fast','custom-int2-hybrid-fast','custom-int2-residual','custom-int2-accuracy-guard','custom-int2-v4','custom-mixedbit-v5','custom-mixedbit-v5-graph'],
                    value=['baseline-fp16','baseline-static-compile','hf-kernels-auto','torchao-int4','custom-mixedbit-v5','custom-mixedbit-v5-graph'],label='Methods')
                prompt=gr.Textbox('Explain why GPU memory bandwidth matters for LLM inference in two sentences.',lines=3,label='Prompt')
                calibration=gr.Textbox(joined_calibration(),lines=12,label='Calibration prompts (alternating plan / held-out validation)')
                evaltxt=gr.Textbox(joined_eval(),lines=7,label='Held-out perplexity texts')
                with gr.Row():
                    maxnew=gr.Slider(16,256,value=64,step=16,label='Fixed generated tokens');dtype=gr.Dropdown(['fp16','bf16'],value='fp16',label='Base dtype');group=gr.Dropdown([32,64,128,256],128,label='Preferred INT2 group');rank=gr.Dropdown([0,4,8,16,32],8,label='Max residual rank')
                with gr.Row():
                    nmse=gr.Number(.06,label='Local NMSE max');cos=gr.Number(.96,label='Local cosine min');gb=gr.Number(.02,label='Global calibration NLL increase max')
                runm=gr.Button('Benchmark fixed-token matrix',variant='primary');mt=gr.Dataframe(interactive=False,label='Results')
                with gr.Row():mtps=gr.Plot();meng=gr.Plot();mppl=gr.Plot()
                mst=gr.Textbox(label='Status')
                runm.click(run_models,[models,methods,prompt,calibration,evaltxt,maxnew,dtype,group,nmse,cos,rank,gb],[mt,mtps,meng,mppl,mst])
                gr.Markdown('v5 keeps fixed-token comparisons and adds a StaticCache/compiled graph path. Eager methods still ignore EOS during this benchmark so every method performs the same number of decode steps. `model_storage_gb` measures live parameter/buffer storage, while peak/current/reserved VRAM are reported separately.')
            with gr.Tab('Robust Mixed-Bit Guard v5'):
                gr.Markdown('Local search first tries packed INT2, promotes failing layers to packed INT4, and keeps only the remaining sensitive layers in FP16. Plan prompts and held-out validation prompts are split when possible; rollback checks mean/P95/worst NLL rather than only one calibration average.')
                amodel=gr.Textbox('Qwen/Qwen2.5-0.5B-Instruct',label='Model');acal=gr.Textbox('Explain matrix multiplication.\n---\nWrite a short Python function.\n---\nSummarize memory bandwidth.',lines=5,label='Calibration')
                with gr.Row():aout=gr.Textbox('optimized_bundles/model-mixedbit-v5',label='Output');ag=gr.Dropdown([32,64,128,256],128,label='Preferred group');ar=gr.Dropdown([4,8,16,32],8,label='Max residual')
                with gr.Row():an=gr.Number(.06,label='Local NMSE');ac=gr.Number(.96,label='Local cosine');agb=gr.Number(.02,label='Global NLL budget')
                ab=gr.Button('Build v5 mixed-bit bundle',variant='primary');at=gr.Dataframe(interactive=False);ast=gr.Textbox()
                def _bundle(mid,cal,out,g,r,n,c,budget):
                    texts=[x.strip() for x in cal.split('---') if x.strip()]
                    try:
                        df,path,count,stats=build_optimized_bundle_v5(mid,out,texts,'cuda','fp16',int(g),float(n),float(c),int(r),float(budget));return df,f'Bundle: {path}\nOptimized layers: {count}\n'+json.dumps(stats,ensure_ascii=False,indent=2)
                    except Exception as e:return pd.DataFrame([{'error':repr(e)}]),repr(e)
                ab.click(_bundle,[amodel,acal,aout,ag,ar,an,ac,agb],[at,ast])
            with gr.Tab('Giant Fusion'):
                gr.Markdown('Microbenchmark for QKV INT2 + RoPE + KV-cache + causal attention. Model runtime still falls back when architecture/layout is unsupported; this tab never labels a fallback as fused.')
                with gr.Row():fh=gr.Number(4096,label='Hidden',precision=0);fheads=gr.Number(32,label='Heads',precision=0);fseq=gr.Number(128,label='Sequence',precision=0);fg=gr.Dropdown([32,64,128,256],128,label='Group');frep=gr.Number(50,label='Repeats',precision=0)
                frun=gr.Button('Run fusion benchmark');ft=gr.Dataframe(interactive=False);fp=gr.Plot();fs=gr.Textbox()
                def _fusion(hidden,heads,seq,group,repeats):
                    try:
                        df=run_fusion_benchmark(int(hidden),int(heads),int(seq),int(group),int(repeats));rid,csv,_=STORE.save(df,'fusion',{});return df,_fig(df,'method','speedup','Fusion speedup'),f'{rid}\n{csv}'
                    except Exception as e:return pd.DataFrame([{'error':repr(e)}]),None,repr(e)
                frun.click(_fusion,[fh,fheads,fseq,fg,frep],[ft,fp,fs])
            with gr.Tab('Architecture paths'):
                gr.Markdown('**Decode:** M=1..4 specialized packed INT2/INT4 GEMV + fused exact outlier correction. **Hopper:** WGMMA/TMA experimental paths with safe fallback. **Blackwell:** tcgen05/TMEM/TMA experimental paths with safe fallback. **Prefill:** existing Tensor Core / attention backends are retained because decode-optimized INT2 is not assumed to beat cuBLAS at large M. The v5 graph methods add Static KV Cache + `torch.compile(mode=reduce-overhead)` so decode replays can avoid most Python/C++/driver launch setup. The benchmark decides.')
            with gr.Tab('Runs'):
                runs=gr.Dataframe(value=list_runs(),interactive=False);gr.Button('Refresh').click(list_runs,outputs=runs)
        gr.Markdown('No backend is declared faster solely because it uses fewer bits. The GUI marks a method only from measured throughput/energy under the configured quality constraint.')
    return demo

def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=7860);p.add_argument('--share',action='store_true');a=p.parse_args()
    create_app().launch(server_name=a.host,server_port=a.port,share=a.share,css=CSS)
if __name__=='__main__':main()
