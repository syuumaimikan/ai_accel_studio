from __future__ import annotations
import torch


@torch.no_grad()
def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    r = reference.float()
    c = candidate.float()
    diff = c - r
    mse = (diff.square().mean()).item()
    mae = diff.abs().mean().item()
    denom = r.square().mean().item() + 1e-12
    rr = r.reshape(r.shape[0], -1)
    cc = c.reshape(c.shape[0], -1)
    cosine = torch.nn.functional.cosine_similarity(rr, cc, dim=-1).mean().item()

    # Works for [B,C] and [B,T,C].
    top1 = (r.argmax(dim=-1) == c.argmax(dim=-1)).float().mean().item()
    return {
        "mse_vs_dense": mse,
        "mae_vs_dense": mae,
        "nmse_vs_dense": mse / denom,
        "cosine_vs_dense": cosine,
        "top1_agreement": top1,
    }
