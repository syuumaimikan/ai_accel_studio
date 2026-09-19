from __future__ import annotations
import torch


def svd_residual_factors(residual: torch.Tensor, rank: int):
    """Returns U[O,r], V[r,I] such that residual ~= U @ V."""
    rank = min(rank, min(residual.shape))
    if rank <= 0:
        return None, None
    U, S, Vh = torch.linalg.svd(residual.float(), full_matrices=False)
    U = U[:, :rank] * S[:rank].unsqueeze(0)
    V = Vh[:rank, :]
    return U, V


def activation_aware_factors(
    residual: torch.Tensor,
    calibration_x: torch.Tensor,
    rank: int,
    ridge: float = 1e-4,
):
    """
    Approximate residual for the calibration distribution rather than globally.

    Goal:
        X @ residual.T ~= X @ V.T @ U.T

    We SVD the residual output Y=X@R.T, then solve a ridge regression for the
    input-side factor. This is more expensive at construction time, but can
    produce a better output approximation at the same rank.
    """
    rank = min(rank, min(residual.shape), calibration_x.shape[0])
    if rank <= 0:
        return None, None

    X = calibration_x.detach().float()
    R = residual.detach().float()
    Y = X @ R.T
    A, S, Bh = torch.linalg.svd(Y, full_matrices=False)
    A_r = A[:, :rank]
    S_r = S[:rank]
    B_r = Bh[:rank, :].T  # [O,r]
    target = A_r * S_r.unsqueeze(0)  # [samples,r]

    # Ridge solve: C=(X^T X + λI)^-1 X^T target, shape [I,r]
    xtx = X.T @ X
    eye = torch.eye(xtx.shape[0], device=xtx.device, dtype=xtx.dtype)
    C = torch.linalg.solve(xtx + ridge * eye, X.T @ target)
    U = B_r
    V = C.T
    return U, V
