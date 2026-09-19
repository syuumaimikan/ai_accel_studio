from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


INT2_LEVELS = (-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0)


@dataclass
class GroupwiseInt2Weight:
    packed: torch.Tensor       # uint8 [N, Kpad/4]
    scales: torch.Tensor       # fp16 [N, Kpad/group_size]
    original_k: int
    padded_k: int
    group_size: int
    activation_aware: bool = False

    @property
    def n(self) -> int:
        return int(self.packed.shape[0])

    @property
    def compression_bits_per_weight(self) -> float:
        scale_bits = 16.0 / self.group_size
        return 2.0 + scale_bits


def _pad(weight: torch.Tensor, group_size: int):
    if weight.ndim != 2:
        raise ValueError("weight must be [N,K]")
    if group_size % 4:
        raise ValueError("group_size must be divisible by 4")
    n, k = weight.shape
    kp = math.ceil(k / group_size) * group_size
    if kp == k:
        return weight, k
    out = torch.zeros((n, kp), device=weight.device, dtype=weight.dtype)
    out[:, :k] = weight
    return out, k


def _nearest_code(x: torch.Tensor) -> torch.Tensor:
    cb = torch.tensor(INT2_LEVELS, device=x.device, dtype=x.dtype)
    return (x.unsqueeze(-1) - cb).abs().argmin(-1).to(torch.uint8)


def _quantized_for_scale(wg: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    norm = (wg / scale.unsqueeze(-1).clamp_min(1e-8)).clamp(-1, 1)
    code = _nearest_code(norm)
    cb = torch.tensor(INT2_LEVELS, device=wg.device, dtype=wg.dtype)
    return cb[code.long()] * scale.unsqueeze(-1)


def pack_groupwise_int2(
    weight: torch.Tensor,
    group_size: int = 128,
    activation_rms: torch.Tensor | None = None,
    refine_steps: int = 13,
) -> GroupwiseInt2Weight:
    """
    Groupwise INT2 weight-only packing. If activation_rms[K] is supplied,
    the group scale is refined to minimize activation-weighted weight error.
    This is a small, dependency-free accuracy recovery step rather than GPTQ.
    """
    w, original_k = _pad(weight.detach().float(), group_size)
    n, kp = w.shape
    ng = kp // group_size
    wg = w.view(n, ng, group_size)

    base = wg.abs().amax(-1).clamp_min(1e-8)
    activation_aware = activation_rms is not None

    if activation_rms is not None:
        ar = activation_rms.detach().float().flatten()
        if ar.numel() < original_k:
            raise ValueError("activation_rms is shorter than K")
        if kp > original_k:
            ar = F.pad(ar[:original_k], (0, kp - original_k))
        else:
            ar = ar[:kp]
        weights = ar.view(1, ng, group_size).square().clamp_min(1e-8)

        factors = torch.linspace(0.55, 1.15, refine_steps, device=w.device)
        # [C,N,G,1]
        cand = base.unsqueeze(0) * factors[:, None, None]
        x = wg.unsqueeze(0) / cand.unsqueeze(-1).clamp_min(1e-8)
        cb = torch.tensor(INT2_LEVELS, device=w.device, dtype=w.dtype)
        dist = (x.unsqueeze(-1) - cb).abs()
        code = dist.argmin(-1)
        q = cb[code] * cand.unsqueeze(-1)
        err = ((q - wg.unsqueeze(0)).square() * weights.unsqueeze(0)).sum(-1)
        best = err.argmin(0)
        scales = torch.gather(cand.squeeze(-1).permute(1,2,0), 2, best.unsqueeze(-1)).squeeze(-1)
    else:
        scales = base

    q = _nearest_code((wg / scales.unsqueeze(-1)).clamp(-1,1))
    q = q.view(n, kp // 4, 4)
    packed = q[...,0] | (q[...,1] << 2) | (q[...,2] << 4) | (q[...,3] << 6)
    return GroupwiseInt2Weight(
        packed=packed.contiguous(),
        scales=scales.to(torch.float16).contiguous(),
        original_k=original_k,
        padded_k=kp,
        group_size=group_size,
        activation_aware=activation_aware,
    )


def unpack_groupwise_int2(pw: GroupwiseInt2Weight, dtype=torch.float32) -> torch.Tensor:
    p = pw.packed.to(torch.uint8)
    codes = torch.stack([(p >> (2*i)) & 3 for i in range(4)], dim=-1).reshape(p.shape[0], -1)
    cb = torch.tensor(INT2_LEVELS, device=p.device, dtype=dtype)
    vals = cb[codes.long()]
    kidx = torch.arange(pw.padded_k, device=p.device)
    gids = kidx // pw.group_size
    vals = vals * pw.scales.to(dtype)[:, gids]
    return vals[:, :pw.original_k]


def reference_linear(x: torch.Tensor, pw: GroupwiseInt2Weight, bias=None) -> torch.Tensor:
    w = unpack_groupwise_int2(pw, dtype=x.dtype)
    return F.linear(x[..., :pw.original_k], w, bias)


if TRITON_AVAILABLE:
    @triton.jit
    def _decode(code):
        return tl.where(code == 0, -1.0,
               tl.where(code == 1, -0.3333333333333333,
               tl.where(code == 2,  0.3333333333333333, 1.0)))

    @triton.autotune(
        configs=[
            triton.Config({'BM': 16, 'BN': 32, 'BK': 64}, num_warps=4, num_stages=3),
            triton.Config({'BM': 32, 'BN': 64, 'BK': 64}, num_warps=8, num_stages=3),
            triton.Config({'BM': 32, 'BN': 64, 'BK': 128}, num_warps=8, num_stages=4),
            triton.Config({'BM': 64, 'BN': 64, 'BK': 64}, num_warps=8, num_stages=4),
        ],
        key=['M','N','K','GROUP_SIZE'],
    )
    @triton.jit
    def _gw_int2_kernel(
        x, packed, scales, bias, y,
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        sxm: tl.constexpr, sxk: tl.constexpr,
        spn: tl.constexpr, spb: tl.constexpr,
        ssgn: tl.constexpr, ssg: tl.constexpr,
        sym: tl.constexpr, syn: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
    ):
        pm, pn = tl.program_id(0), tl.program_id(1)
        rm = pm * BM + tl.arange(0, BM)
        rn = pn * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), tl.float32)

        for k0 in range(0, K, BK):
            kk = k0 + rk
            xv = tl.load(x + rm[:,None]*sxm + kk[None,:]*sxk,
                         mask=(rm[:,None] < M) & (kk[None,:] < K), other=0.0).to(tl.float16)
            bidx = kk[:,None] // 4
            shift = (kk[:,None] & 3) * 2
            p = tl.load(packed + rn[None,:]*spn + bidx*spb,
                        mask=(rn[None,:] < N) & (kk[:,None] < K), other=0).to(tl.int32)
            code = (p >> shift) & 3
            q = _decode(code)
            gid = kk[:,None] // GROUP_SIZE
            sc = tl.load(scales + rn[None,:]*ssgn + gid*ssg,
                         mask=(rn[None,:] < N) & (kk[:,None] < K), other=0.0)
            w = (q * sc).to(tl.float16)
            acc += tl.dot(xv, w)

        if HAS_BIAS:
            acc += tl.load(bias + rn, mask=rn<N, other=0.0)[None,:]
        tl.store(y + rm[:,None]*sym + rn[None,:]*syn, acc,
                 mask=(rm[:,None] < M) & (rn[None,:] < N))


def triton_groupwise_int2_linear(x: torch.Tensor, pw: GroupwiseInt2Weight, bias=None) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not installed")
    if not x.is_cuda or x.dtype != torch.float16:
        raise RuntimeError("optimized groupwise INT2 requires CUDA fp16 input")
    if x.ndim != 2:
        raise ValueError("x must be 2-D; flatten leading dimensions first")
    if x.shape[1] != pw.padded_k:
        raise ValueError(f"K={x.shape[1]} does not equal packed K={pw.padded_k}")
    if not pw.packed.is_cuda:
        raise RuntimeError("packed weight must be CUDA")

    m, k = x.shape
    n = pw.n
    y = torch.empty((m,n), device=x.device, dtype=x.dtype)
    if bias is None:
        bias_arg = torch.empty(1, device=x.device, dtype=x.dtype)
        has_bias = False
    else:
        bias_arg = bias.to(device=x.device, dtype=x.dtype).contiguous()
        has_bias = True

    grid = lambda meta: (triton.cdiv(m, meta['BM']), triton.cdiv(n, meta['BN']))
    _gw_int2_kernel[grid](
        x, pw.packed, pw.scales, bias_arg, y,
        m,n,k,pw.group_size,
        x.stride(0),x.stride(1),
        pw.packed.stride(0),pw.packed.stride(1),
        pw.scales.stride(0),pw.scales.stride(1),
        y.stride(0),y.stride(1),
        HAS_BIAS=has_bias,
    )
    return y
