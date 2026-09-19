from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import platform
import shutil

import torch


@dataclass
class HardwareInfo:
    torch_version: str
    cuda_available: bool
    cuda_runtime: str | None
    device_name: str
    compute_capability: tuple[int, int] | None
    architecture: str
    sm_count: int | None
    total_vram_gb: float | None
    bf16: bool
    tf32: bool
    tma: bool
    wgmma: bool
    tcgen05: bool
    sparse_tensor_core: bool
    nvcc: bool
    nvidia_smi: bool
    triton: bool
    gluon: bool
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.compute_capability is not None:
            d["compute_capability"] = ".".join(map(str, self.compute_capability))
        return d


def _triton_flags() -> tuple[bool, bool]:
    try:
        import triton  # noqa: F401
        triton_ok = True
    except Exception:
        return False, False
    try:
        from triton.experimental import gluon  # noqa: F401
        gluon_ok = True
    except Exception:
        gluon_ok = False
    return triton_ok, gluon_ok


def architecture_name(cc: tuple[int, int] | None) -> str:
    if cc is None:
        return "CPU"
    major, minor = cc
    if major >= 10:
        return "Blackwell-or-newer"
    if major == 9:
        return "Hopper"
    if major == 8:
        return "Ampere/Ada"
    if major == 7:
        return "Volta/Turing"
    return f"SM{major}{minor}"


def detect_hardware(device: int = 0) -> HardwareInfo:
    cuda = torch.cuda.is_available()
    tri, gluon = _triton_flags()
    notes: list[str] = []

    if not cuda:
        return HardwareInfo(
            torch_version=torch.__version__,
            cuda_available=False,
            cuda_runtime=getattr(torch.version, "cuda", None),
            device_name=platform.processor() or platform.machine(),
            compute_capability=None,
            architecture="CPU",
            sm_count=None,
            total_vram_gb=None,
            bf16=False,
            tf32=False,
            tma=False,
            wgmma=False,
            tcgen05=False,
            sparse_tensor_core=False,
            nvcc=shutil.which("nvcc") is not None,
            nvidia_smi=shutil.which("nvidia-smi") is not None,
            triton=tri,
            gluon=gluon,
            notes=["CUDA GPU is not available; GPU-only kernels will fall back."],
        )

    props = torch.cuda.get_device_properties(device)
    cc = torch.cuda.get_device_capability(device)
    major, _minor = cc
    arch = architecture_name(cc)

    tma = major >= 9
    wgmma = major == 9
    tcgen05 = major >= 10
    sparse_tc = major >= 8

    if major < 8:
        notes.append("Hardware 2:4 sparse Tensor Core acceleration is not expected.")
    if major < 9:
        notes.append("TMA/WGMMA paths are disabled; use standard Triton/CUDA kernels.")
    if major >= 10 and not gluon:
        notes.append("Blackwell detected but Triton Gluon is missing; tcgen05 path will fall back.")

    return HardwareInfo(
        torch_version=torch.__version__,
        cuda_available=True,
        cuda_runtime=getattr(torch.version, "cuda", None),
        device_name=props.name,
        compute_capability=cc,
        architecture=arch,
        sm_count=getattr(props, "multi_processor_count", None),
        total_vram_gb=props.total_memory / (1024**3),
        bf16=torch.cuda.is_bf16_supported(),
        tf32=major >= 8,
        tma=tma,
        wgmma=wgmma,
        tcgen05=tcgen05,
        sparse_tensor_core=sparse_tc,
        nvcc=shutil.which("nvcc") is not None,
        nvidia_smi=shutil.which("nvidia-smi") is not None,
        triton=tri,
        gluon=gluon,
        notes=notes,
    )


def recommended_backends(info: HardwareInfo | None = None) -> list[str]:
    info = info or detect_hardware()
    if not info.cuda_available:
        return ["torch-fp32", "torch-compile"]
    out = ["torch-fp16", "torch-compile", "groupwise-int2"]
    if info.sparse_tensor_core:
        out.append("native-2to4")
    if info.tma:
        out += ["tma-matmul", "persistent-int2"]
    if info.wgmma:
        out.append("hopper-wgmma")
    if info.tcgen05:
        out += ["blackwell-tcgen05", "blackwell-warp-specialized"]
    return out
