from __future__ import annotations

import json
import traceback
from dataclasses import asdict
from typing import Any

import torch

from .hardware import detect_hardware, recommended_backends


def _probe(name: str, fn):
    try:
        result = fn()
        if isinstance(result, tuple):
            ok = bool(result[0])
            detail = result[1] if len(result) > 1 else None
        elif isinstance(result, dict):
            ok = bool(result.get("ok", result.get("passed", False)))
            detail = result
        else:
            ok = bool(result)
            detail = result
        return {"name": name, "ok": ok, "detail": detail}
    except Exception as exc:
        return {
            "name": name,
            "ok": False,
            "detail": f"{type(exc).__name__}: {exc}",
        }


def run_selftest(run_microbench: bool = False) -> dict[str, Any]:
    hw = detect_hardware()
    report: dict[str, Any] = {
        "hardware": hw.to_dict(),
        "recommended_backends": recommended_backends(hw),
        "probes": [],
        "microbench": None,
    }

    if not torch.cuda.is_available():
        report["probes"].append({
            "name": "cuda",
            "ok": False,
            "detail": "CUDA unavailable; GPU probes were skipped.",
        })
        return report

    # Always validate basic CUDA math first.
    def basic_cuda():
        a = torch.randn(32, 64, device="cuda", dtype=torch.float16)
        b = torch.randn(64, 48, device="cuda", dtype=torch.float16)
        c = a @ b
        torch.cuda.synchronize()
        return bool(torch.isfinite(c).all().item())

    report["probes"].append(_probe("cuda-matmul", basic_cuda))

    if hw.tma:
        def tma_probe():
            from .kernels.tma_tensorcore import tma_matmul
            a = torch.randn(64, 128, device="cuda", dtype=torch.float16)
            b = torch.randn(128, 64, device="cuda", dtype=torch.float16)
            y = tma_matmul(a, b)
            ref = a @ b
            torch.cuda.synchronize()
            err = float((y.float() - ref.float()).abs().max().item())
            return {"ok": err < 0.2, "max_abs_error": err}
        report["probes"].append(_probe("triton-tma", tma_probe))

    if hw.wgmma or hw.tcgen05:
        def hb_probe():
            from .kernels.gluon_hopper_blackwell import self_test
            return self_test()
        report["probes"].append(_probe("gluon-hopper-blackwell", hb_probe))

    if hw.tcgen05:
        def ws_probe():
            from .kernels.gluon_warp_specialized import self_test
            return self_test()
        report["probes"].append(_probe("blackwell-warp-specialized", ws_probe))

    def int2_probe():
        from .kernels.groupwise_int2 import pack_groupwise_int2, unpack_groupwise_int2, triton_groupwise_int2_linear
        x = torch.randn(16, 256, device="cuda", dtype=torch.float16)
        w = torch.randn(128, 256, device="cuda", dtype=torch.float16) / 16
        packed = pack_groupwise_int2(w, group_size=64)
        y = triton_groupwise_int2_linear(x, packed)
        ref_w = unpack_groupwise_int2(packed, dtype=torch.float16)
        ref = x @ ref_w.t()
        torch.cuda.synchronize()
        max_err = float((y.float() - ref.float()).abs().max().item())
        cos = float(torch.nn.functional.cosine_similarity(y.float().reshape(1,-1), ref.float().reshape(1,-1)).item())
        return {"ok": max_err < 0.5 and cos > 0.999, "max_abs_error": max_err, "cosine": cos}
    report["probes"].append(_probe("groupwise-int2-triton", int2_probe))

    def decode_int2_probe():
        from .kernels.groupwise_int2 import pack_groupwise_int2, unpack_groupwise_int2
        from .kernels.decode_int2 import decode_int2_linear
        x=torch.randn(1,256,device='cuda',dtype=torch.float16)
        w=torch.randn(128,256,device='cuda',dtype=torch.float16)/16
        pw=pack_groupwise_int2(w,group_size=64)
        y=decode_int2_linear(x,pw)
        ref=x @ unpack_groupwise_int2(pw,dtype=torch.float16).t()
        torch.cuda.synchronize()
        max_err=float((y.float()-ref.float()).abs().max().item())
        cos=float(torch.nn.functional.cosine_similarity(y.float(),ref.float(),dim=-1).mean().item())
        return {'ok':max_err<0.5 and cos>0.999,'max_abs_error':max_err,'cosine':cos}
    report["probes"].append(_probe("decode-int2-v4",decode_int2_probe))

    def decode_int4_probe():
        from .kernels.groupwise_int4 import pack_groupwise_int4,unpack_groupwise_int4,decode_int4_linear
        x=torch.randn(1,256,device='cuda',dtype=torch.float16);w=torch.randn(128,256,device='cuda',dtype=torch.float16)/16
        pw=pack_groupwise_int4(w,group_size=64);y=decode_int4_linear(x,pw);ref=x @ unpack_groupwise_int4(pw,dtype=torch.float16).t();torch.cuda.synchronize()
        max_err=float((y.float()-ref.float()).abs().max().item());cos=float(torch.nn.functional.cosine_similarity(y.float(),ref.float(),dim=-1).mean().item())
        return {'ok':max_err<0.35 and cos>0.999,'max_abs_error':max_err,'cosine':cos}
    report["probes"].append(_probe("decode-int4-v5",decode_int4_probe))

    def cudagraph_probe():
        x=torch.randn(128,128,device='cuda',dtype=torch.float16);w=torch.randn(128,128,device='cuda',dtype=torch.float16)
        s=torch.cuda.Stream();s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): y=x@w
        torch.cuda.current_stream().wait_stream(s);g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): y=x@w
        g.replay();torch.cuda.synchronize();return bool(torch.isfinite(y).all().item())
    report["probes"].append(_probe("cuda-graph-replay",cudagraph_probe))

    def fusion_probe():
        from .kernels.fused_decode_attention import giant_fusion_self_test
        return giant_fusion_self_test()
    report["probes"].append(_probe("qkv-rope-attention-fusion", fusion_probe))

    if run_microbench:
        try:
            from .kernel_suite import run_kernel_suite
            df = run_kernel_suite(16, 512, 512, 64, 20, False)
            report["microbench"] = df.to_dict(orient="records")
        except Exception as exc:
            report["microbench"] = {"error": f"{type(exc).__name__}: {exc}"}

    return report


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--microbench", action="store_true")
    p.add_argument("--json", dest="json_path")
    args = p.parse_args()
    report = run_selftest(args.microbench)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json_path:
        from pathlib import Path
        path = Path(args.json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
