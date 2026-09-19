from __future__ import annotations

from dataclasses import dataclass
import torch

# Four-level weight-only INT2 codebook. It is symmetric and uses all 4 codes.
# code 0,1,2,3 => -1, -1/3, +1/3, +1
INT2_CODEBOOK = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


def _require_2d(w: torch.Tensor) -> None:
    if w.ndim != 2:
        raise ValueError(f"weight must be 2-D [N,K], got {tuple(w.shape)}")


def _pad_k(w: torch.Tensor, multiple: int = 4):
    _require_2d(w)
    k = w.shape[1]
    padded = (k + multiple - 1) // multiple * multiple
    if padded == k:
        return w, k
    out = torch.zeros((w.shape[0], padded), dtype=w.dtype, device=w.device)
    out[:, :k] = w
    return out, k


def _nearest_int2_code(norm: torch.Tensor) -> torch.Tensor:
    cb = torch.tensor(INT2_CODEBOOK, dtype=norm.dtype, device=norm.device)
    # [..., 1] - [4] -> [..., 4]
    return (norm.unsqueeze(-1) - cb).abs().argmin(dim=-1).to(torch.uint8)


def _decode_int2(code: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    c = code.to(torch.int64)
    cb = torch.tensor(INT2_CODEBOOK, device=c.device, dtype=dtype)
    return cb[c]


@dataclass
class PackedWeight:
    packed: torch.Tensor
    scale: torch.Tensor
    original_k: int
    padded_k: int
    format: str

    @property
    def n(self):
        return int(self.packed.shape[0])


def pack_int2(weight: torch.Tensor) -> PackedWeight:
    """Pack four INT2 codes per uint8. Per-output-channel scale."""
    w, original_k = _pad_k(weight.detach().float(), 4)
    scale = w.abs().amax(dim=1).clamp_min(1e-8)
    norm = (w / scale[:, None]).clamp(-1, 1)
    code = _nearest_int2_code(norm)
    n, k = code.shape
    code = code.view(n, k // 4, 4)
    packed = (
        code[..., 0]
        | (code[..., 1] << 2)
        | (code[..., 2] << 4)
        | (code[..., 3] << 6)
    ).contiguous()
    return PackedWeight(packed, scale.to(torch.float16), original_k, k, 'int2')


def unpack_int2(pw: PackedWeight, dtype=torch.float32) -> torch.Tensor:
    p = pw.packed.to(torch.uint8)
    codes = torch.stack([(p >> (2*i)) & 0x3 for i in range(4)], dim=-1)
    q = _decode_int2(codes, dtype=dtype).reshape(p.shape[0], -1)
    q = q * pw.scale.to(dtype)[:, None]
    return q[:, :pw.original_k]


def pack_ternary(weight: torch.Tensor, threshold_factor: float = 0.7) -> PackedWeight:
    """Pack four ternary values per byte: 00=0, 01=+1, 10=-1."""
    w, original_k = _pad_k(weight.detach().float(), 4)
    threshold = threshold_factor * w.abs().mean(dim=1, keepdim=True)
    pos = w > threshold
    neg = w < -threshold
    code = torch.zeros_like(w, dtype=torch.uint8)
    code[pos] = 1
    code[neg] = 2

    nz = pos | neg
    count = nz.sum(dim=1).clamp_min(1)
    scale = (w.abs() * nz).sum(dim=1) / count
    scale = scale.clamp_min(1e-8)

    n, k = code.shape
    c = code.view(n, k // 4, 4)
    packed = c[...,0] | (c[...,1] << 2) | (c[...,2] << 4) | (c[...,3] << 6)
    return PackedWeight(packed.contiguous(), scale.to(torch.float16), original_k, k, 'ternary')


def unpack_ternary(pw: PackedWeight, dtype=torch.float32) -> torch.Tensor:
    p = pw.packed.to(torch.uint8)
    codes = torch.stack([(p >> (2*i)) & 0x3 for i in range(4)], dim=-1)
    vals = torch.where(codes == 1, 1.0, torch.where(codes == 2, -1.0, 0.0)).to(dtype)
    vals = vals.reshape(p.shape[0], -1) * pw.scale.to(dtype)[:, None]
    return vals[:, :pw.original_k]


def prune_24(weight: torch.Tensor) -> torch.Tensor:
    """Magnitude prune to exactly 2 non-zero values per contiguous group of 4."""
    w, original_k = _pad_k(weight.detach(), 4)
    n, k = w.shape
    g = w.view(n, k // 4, 4)
    idx = g.abs().topk(k=2, dim=-1, largest=True, sorted=False).indices
    mask = torch.zeros_like(g, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    out = torch.where(mask, g, torch.zeros_like(g)).reshape(n, k)
    return out[:, :original_k]


@dataclass
class PackedSparse24:
    # One byte / 4 original weights:
    # bits 0..1=q0, 2..3=q1, 4..5=idx0, 6..7=idx1.
    packed: torch.Tensor
    scale: torch.Tensor
    original_k: int
    padded_k: int
    format: str = 'int2_2to4'

    @property
    def n(self):
        return int(self.packed.shape[0])


def pack_int2_sparse24(weight: torch.Tensor) -> PackedSparse24:
    """
    Fuse 2:4 pruning and INT2 quantization.

    Storage is exactly 8 bits per group of 4 original weights plus one FP16
    scale per output channel.  The two selected positions are encoded explicitly.
    """
    w, original_k = _pad_k(weight.detach().float(), 4)
    n, k = w.shape
    groups = w.view(n, k // 4, 4)
    idx = groups.abs().topk(k=2, dim=-1, largest=True, sorted=False).indices
    idx, order = idx.sort(dim=-1)
    vals = torch.gather(groups, -1, idx)

    scale = vals.abs().amax(dim=(1,2)).clamp_min(1e-8)
    norm = (vals / scale[:,None,None]).clamp(-1,1)
    q = _nearest_int2_code(norm)

    packed = (
        q[...,0]
        | (q[...,1] << 2)
        | (idx[...,0].to(torch.uint8) << 4)
        | (idx[...,1].to(torch.uint8) << 6)
    ).contiguous()
    return PackedSparse24(packed, scale.to(torch.float16), original_k, k)


def unpack_int2_sparse24(pw: PackedSparse24, dtype=torch.float32) -> torch.Tensor:
    p = pw.packed.to(torch.uint8)
    q0 = p & 0x3
    q1 = (p >> 2) & 0x3
    i0 = ((p >> 4) & 0x3).long()
    i1 = ((p >> 6) & 0x3).long()
    v0 = _decode_int2(q0, dtype=dtype) * pw.scale.to(dtype)[:,None]
    v1 = _decode_int2(q1, dtype=dtype) * pw.scale.to(dtype)[:,None]

    n, ng = p.shape
    dense = torch.zeros((n, ng, 4), device=p.device, dtype=dtype)
    dense.scatter_(-1, i0.unsqueeze(-1), v0.unsqueeze(-1))
    dense.scatter_(-1, i1.unsqueeze(-1), v1.unsqueeze(-1))
    return dense.reshape(n, ng*4)[:, :pw.original_k]


def packed_bytes(obj) -> int:
    return int(obj.packed.numel() * obj.packed.element_size() + obj.scale.numel() * obj.scale.element_size())
