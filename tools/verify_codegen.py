from __future__ import annotations
import argparse, re, subprocess
from pathlib import Path

TOKENS = {
    "hopper_wgmma": [r"WGMMA", r"wgmma\.mma_async"],
    "sparse_mma": [r"MMA\.SP", r"mma\.sp", r"WGMMA.*SP"],
    "tma": [r"CPASYNC.*BULK.*TENSOR", r"cp\.async\.bulk\.tensor", r"TMA"],
    "blackwell_tcgen05": [r"TCGEN05", r"tcgen05\.mma"],
}

def scan_text(text: str):
    return {name:any(re.search(p,text,re.I) for p in pats) for name,pats in TOKENS.items()}

def main():
    ap=argparse.ArgumentParser(description="Verify PTX/SASS contains expected architecture instructions")
    ap.add_argument("binary",help="cubin/fatbin/shared library/PTX file")
    args=ap.parse_args(); p=Path(args.binary)
    if p.suffix.lower() in {'.ptx','.sass','.txt'}:
        text=p.read_text(errors='ignore')
    else:
        cmds=[["cuobjdump","--dump-sass",str(p)],["cuobjdump","--dump-ptx",str(p)]]
        parts=[]
        for cmd in cmds:
            try: parts.append(subprocess.check_output(cmd,text=True,stderr=subprocess.STDOUT))
            except Exception as e: parts.append(f"{cmd[1]} failed: {e}")
        text='\n'.join(parts)
    result=scan_text(text)
    for k,v in result.items(): print(f"{k:20s}: {'FOUND' if v else 'not found'}")

if __name__=='__main__': main()
