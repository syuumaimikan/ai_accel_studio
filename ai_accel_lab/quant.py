from __future__ import annotations
import torch


def bitpack_estimated_bytes(numel: int, bits: int | float) -> int:
    return int((numel * bits + 7) // 8)


def quant_bounds(bits: int):
    if bits < 2: raise ValueError("bits must be >= 2")
    qmax = (1 << (bits - 1)) - 1
    qmin = -qmax
    return qmin, qmax


def quantize_per_row(weight: torch.Tensor, bits: int):
    """Signed symmetric per-output-row quantization."""
    qmin, qmax = quant_bounds(bits)
    w = weight.detach().float()
    scale = w.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(w / scale).clamp(qmin, qmax).to(torch.int8)
    return q, scale.squeeze(1)


def dequantize_per_row(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    return q.to(dtype) * scale.to(dtype).unsqueeze(1)


def fake_quant_activation(x: torch.Tensor, bits: int):
    qmin, qmax = quant_bounds(bits)
    xf = x.float()
    scale = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.round(xf / scale).clamp(qmin, qmax)
    return (q * scale).to(x.dtype)


def ternarize_weight(weight: torch.Tensor, threshold_factor: float = 0.7):
    w = weight.detach().float()
    threshold = threshold_factor * w.abs().mean(dim=1, keepdim=True)
    t = torch.where(w > threshold, 1.0, torch.where(w < -threshold, -1.0, 0.0))
    nz = t.ne(0)
    denom = nz.sum(dim=1, keepdim=True).clamp_min(1)
    scale = (w.abs() * nz).sum(dim=1, keepdim=True) / denom
    approx = t * scale
    return t.to(torch.int8), scale.squeeze(1), approx
