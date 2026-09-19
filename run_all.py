from __future__ import annotations
import argparse
from pathlib import Path
from ai_accel_lab.benchmark import run_linear
from ai_accel_lab.transformer import run_transformer
from ai_accel_lab.delta import run_delta
from ai_accel_lab.memory import run_memory
from ai_accel_lab.analog import run_analog
from ai_accel_lab.speculative import run_speculative

def main():
    p=argparse.ArgumentParser(); p.add_argument('--device',default='auto'); p.add_argument('--quick',action='store_true'); p.add_argument('--out',default='results'); args=p.parse_args(); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    rep=10 if args.quick else 50
    print('\n=== LINEAR ==='); df,_,_=run_linear(args.device,'small',rep,csv=str(out/'linear.csv')); print(df.to_string(index=False))
    print('\n=== TRANSFORMER ==='); df,_,_=run_transformer(args.device,'tiny',5 if args.quick else 20,csv=str(out/'transformer.csv')); print(df.to_string(index=False))
    print('\n=== DELTA ==='); print(run_delta(args.device,20 if args.quick else 100,features=256 if args.quick else 1024).to_string())
    print('\n=== MEMORY ==='); print(run_memory(args.device,n_memory=2048 if args.quick else 16384,dim=64 if args.quick else 256,repeats=5 if args.quick else 20).to_string())
    print('\n=== ANALOG ==='); print(run_analog(args.device,features=256 if args.quick else 1024,repeats=10 if args.quick else 50).to_string())
    print('\n=== SPECULATIVE ==='); print(run_speculative(args.device,tokens=8 if args.quick else 32).to_string())
if __name__=='__main__':main()
