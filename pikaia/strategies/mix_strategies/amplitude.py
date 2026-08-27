"""Amplitude normalisation helpers for honest multi-strategy mixing."""

from __future__ import annotations

import numpy as np


def normalize_delta_amplitudes(
    delta: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """RMS-normalise each strategy slab of a ``(N, M, K)`` delta tensor.

    For each strategy index ``k``, divide ``delta[..., k]`` by its RMS
    ``sqrt(mean(delta[..., k]**2))``. Strategies whose RMS is at most ``eps``
    are left unchanged so near-zero / ``NONE`` contributions are not amplified.

    Args:
        delta: Delta tensor of shape ``(N, M, K)``.
        eps: RMS threshold below which a strategy is treated as inactive.

    Returns:
        A copy of ``delta`` with active strategies scaled to unit RMS.
    """
    if delta.ndim != 3:
        raise ValueError(
            f"delta must have shape (N, M, K), got ndim={delta.ndim} shape={delta.shape}"
        )
    scales = np.sqrt(np.mean(delta**2, axis=(0, 1)))  # (K,)
    out = delta.copy()
    active = scales > eps
    if np.any(active):
        out[:, :, active] = delta[:, :, active] / scales[active]
    return out


def normalize_kernel_amplitude(
    D: np.ndarray | None,
    d: np.ndarray | None,
    eps: float = 1e-12,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Frobenius / L2-normalise a strategy ``(D, d)`` kernel pair.

    - ``D`` is divided by ``||D||_F`` when the Frobenius norm exceeds ``eps``.
    - ``d`` is divided by ``||d||_2`` when the L2 norm exceeds ``eps``.
    Near-zero kernels are returned unchanged.

    Args:
        D: Optional bilinear matrix of shape ``(M, M)``.
        d: Optional linear vector of shape ``(M,)``.
        eps: Norm threshold below which a term is left unchanged.

    Returns:
        ``(D_norm, d_norm)`` with independent normalisation of each present term.
    """
    D_out = D
    d_out = d
    if D is not None:
        fro = float(np.linalg.norm(D, "fro"))
        if fro > eps:
            D_out = D / fro
    if d is not None:
        nrm = float(np.linalg.norm(d))
        if nrm > eps:
            d_out = d / nrm
    return D_out, d_out
