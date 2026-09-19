from __future__ import annotations
import argparse,json

def main():
    p=argparse.ArgumentParser(prog='python -m ai_accel_lab')
    sub=p.add_subparsers(dest='cmd',required=True)
    sub.add_parser('hardware')
    sub.add_parser('doctor')
    st=sub.add_parser('selftest'); st.add_argument('--microbench',action='store_true'); st.add_argument('--json')
    s=sub.add_parser('studio'); s.add_argument('--host',default='127.0.0.1'); s.add_argument('--port',type=int,default=7860); s.add_argument('--share',action='store_true')
    k=sub.add_parser('kernel-suite'); k.add_argument('--m',type=int,default=32); k.add_argument('--n',type=int,default=4096); k.add_argument('--k',type=int,default=4096); k.add_argument('--group-size',type=int,default=128); k.add_argument('--repeats',type=int,default=100); k.add_argument('--power',action='store_true')
    mb=sub.add_parser('model-bench'); mb.add_argument('--models',required=True,help='comma-separated HF IDs/paths'); mb.add_argument('--methods',default='baseline-fp16,baseline-static-compile,hf-kernels-auto,torchao-int4,custom-mixedbit-v5,custom-mixedbit-v5-graph'); mb.add_argument('--prompt',default='Explain GPU memory bandwidth in one sentence.'); mb.add_argument('--calibration',default='default',help='default or texts separated by ---'); mb.add_argument('--eval-text',default='default',help='default or texts separated by ---'); mb.add_argument('--new-tokens',type=int,default=64); mb.add_argument('--dtype',default='fp16',choices=['fp16','bf16']); mb.add_argument('--group-size',type=int,default=128); mb.add_argument('--nmse-limit',type=float,default=.06); mb.add_argument('--cosine-limit',type=float,default=.96); mb.add_argument('--global-loss-budget',type=float,default=.02); mb.add_argument('--residual-rank',type=int,default=8)
    # Backward-compatible v2 commands.
    a=sub.add_parser('linear'); a.add_argument('--device',default='auto'); a.add_argument('--preset',default='small',choices=['small','medium','large']); a.add_argument('--repeats',type=int); a.add_argument('--methods',default='all'); a.add_argument('--power',action='store_true'); a.add_argument('--csv'); a.add_argument('--compile',action='store_true')
    a=sub.add_parser('transformer'); a.add_argument('--device',default='auto'); a.add_argument('--preset',default='tiny',choices=['tiny','small','base']); a.add_argument('--repeats',type=int); a.add_argument('--power',action='store_true'); a.add_argument('--csv')
    a=sub.add_parser('gpu-kernels'); a.add_argument('--m',type=int,default=128); a.add_argument('--n',type=int,default=4096); a.add_argument('--k',type=int,default=4096); a.add_argument('--repeats',type=int,default=100); a.add_argument('--activation',default='none',choices=['none','relu','silu']); a.add_argument('--cuda-ext',action='store_true'); a.add_argument('--csv')
    args=p.parse_args()
    if args.cmd=='doctor':
        from .doctor import dependency_report,recommendations
        print(dependency_report().to_string(index=False)); print('\n'+json.dumps(recommendations(),ensure_ascii=False,indent=2))
    elif args.cmd=='hardware':
        from .hardware import detect_hardware,recommended_backends
        x=detect_hardware(); d=x.to_dict(); d['recommended_backends']=recommended_backends(x); print(json.dumps(d,ensure_ascii=False,indent=2))
    elif args.cmd=='selftest':
        from .selftest import run_selftest
        r=run_selftest(args.microbench); t=json.dumps(r,ensure_ascii=False,indent=2); print(t)
        if args.json:
            from pathlib import Path
            Path(args.json).parent.mkdir(parents=True,exist_ok=True); Path(args.json).write_text(t+'\n',encoding='utf-8')
    elif args.cmd=='studio':
        from .studio.app import create_app
        create_app().launch(server_name=args.host,server_port=args.port,share=args.share)
    elif args.cmd=='kernel-suite':
        from .kernel_suite import run_kernel_suite
        print(run_kernel_suite(args.m,args.n,args.k,args.group_size,args.repeats,args.power).to_string(index=False))
    elif args.cmd=='model-bench':
        from .model_benchmark import benchmark_matrix
        from .calibration_suite import DEFAULT_CALIBRATION_TEXTS,joined_eval
        models=[x.strip() for x in args.models.split(',') if x.strip()]; methods=[x.strip() for x in args.methods.split(',') if x.strip()]
        cals=DEFAULT_CALIBRATION_TEXTS if args.calibration.strip().lower()=='default' else [x.strip() for x in args.calibration.split('---') if x.strip()]
        eval_text=joined_eval() if args.eval_text.strip().lower()=='default' else args.eval_text
        df=benchmark_matrix(models,methods,args.prompt,cals,eval_text,args.new_tokens,'cuda',args.dtype,group_size=args.group_size,nmse_limit=args.nmse_limit,cosine_limit=args.cosine_limit,global_loss_budget=args.global_loss_budget,residual_rank=args.residual_rank,force_exact_tokens=True)
        print(df.to_string(index=False))
    elif args.cmd=='linear':
        from .benchmark import run_linear; df,d,_=run_linear(args.device,args.preset,args.repeats,args.methods,args.power,args.csv,args.compile); print(f'Device: {d}\n{df.to_string(index=False)}')
    elif args.cmd=='transformer':
        from .transformer import run_transformer; df,d,_=run_transformer(args.device,args.preset,args.repeats,args.power,args.csv); print(f'Device: {d}\n{df.to_string(index=False)}')
    else:
        from .kernels.benchmark_gpu import run_gpu_kernel_benchmark
        df=run_gpu_kernel_benchmark(args.m,args.n,args.k,args.repeats,args.activation,args.cuda_ext); print(df.to_string(index=False))
        if args.csv:
            from pathlib import Path
            Path(args.csv).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.csv,index=False)

if __name__=='__main__': main()
