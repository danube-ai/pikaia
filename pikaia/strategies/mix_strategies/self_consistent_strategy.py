from __future__ import annotations

import numpy as np

from pikaia.strategies.base_strategies import MixStrategy


class SelfConsistentMixStrategy(MixStrategy):
    """
    Adaptively updates mixing coefficients based on the mean absolute delta.

    This strategy computes a weighted sum of the input delta tensor using the
    current mixing coefficients, then updates the coefficients in a
    self-consistent manner based on the mean absolute value of the mixed deltas.

    Example:
        strategy = SelfConsistentMixStrategy()
        mixed_delta, updated_coeffs = strategy(delta, mix_coeffs)
    """

    def __init__(self, **kwargs):
        """Initialise the SelfConsistent mix strategy.

        Args:
            **kwargs: Keyword options forwarded to :class:`MixStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "SelfConsistent"

    def __call__(
        self, delta: np.ndarray, mix_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply self-consistent mixing to the input delta tensor and update mixing
            coefficients.

        Args:
            delta (np.ndarray):
                A 3D tensor of shape (n, m, k) representing the deltas to be mixed.
            mix_coeffs (np.ndarray):
                A 1D array of length k containing the current mixing coefficients.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - The mixed delta array of shape (n, m) after applying the coefficients.
                - The updated mixing coefficients array, adjusted based on the mean
                  absolute delta.

        Notes:
            The mixing coefficients are updated by applying a function to the mean
            absolute value of the mixed deltas, scaled by the number of columns in
            the delta array.
        """
        # Compute per-strategy mean magnitude BEFORE mixing so that strategies with
        # larger deltas genuinely grow their coefficients relative to others.
        # Averaging over both the organism axis (0) and gene axis (1) gives a
        # (k,) vector — one magnitude per strategy.
        per_strategy_mean = np.mean(np.abs(delta), axis=(0, 1))  # (k,)

        # Weighted sum over the last axis (k) of delta using mix_coeffs,
        # resulting in shape (n, m)
        delta = np.einsum("ijk,k->ij", delta, mix_coeffs)

        # Applies delta in form of the central replicator equations
        mix_coeffs = mix_coeffs * (1 + per_strategy_mean * delta.shape[1])
        mix_coeffs /= np.sum(mix_coeffs)

        return delta, mix_coeffs

    @staticmethod
    def update_coeffs_d_matrix(
        D_list: list[np.ndarray | None],
        d_list: list[np.ndarray | None],
        gamma: np.ndarray,
        mix_coeffs: np.ndarray,
    ) -> np.ndarray:
        """Update mixing coefficients for the D-matrix iteration path.

        Replaces the ``(N, M, K)`` tensor magnitude used in ``__call__`` with
        per-strategy D-matrix magnitudes:

        - Bilinear strategy ``s``: ``mean_j(|gamma_j * (D_s @ gamma)_j|)``
        - Linear (balanced org) strategy: ``mean_j(|d_j|)`` — constant

        Args:
            D_list: Per-strategy ``(M, M)`` D matrices or ``None``.
            d_list: Per-strategy ``(M,)`` d-vectors or ``None``.
            gamma: Current gene fitness vector, shape ``(M,)``.
            mix_coeffs: Current mixing coefficients, shape ``(K,)``.

        Returns:
            Updated and renormalized mixing coefficients, shape ``(K,)``.
        """
        M = len(gamma)
        magnitudes = np.array(
            [
                np.mean(np.abs(gamma * (D_s @ gamma)))
                if D_s is not None
                else (np.mean(np.abs(d_s)) if d_s is not None else 0.0)
                for D_s, d_s in zip(D_list, d_list)
            ]
        )
        mix_coeffs = mix_coeffs * (1 + M * magnitudes)
        mix_coeffs /= mix_coeffs.sum()
        return mix_coeffs
