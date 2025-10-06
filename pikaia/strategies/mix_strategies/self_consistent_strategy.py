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
        # Weighted sum over the last axis (k) of delta using mix_coeffs,
        # resulting in shape (n, m)
        delta = np.einsum("ijk,k->ij", delta, mix_coeffs)

        delta_g_mean = np.mean(np.abs(delta), axis=0)
        total_delta_g_mean = np.mean(delta_g_mean, axis=0)

        # Applies delta in form of the central replicator equations
        mix_coeffs = mix_coeffs * (1 + total_delta_g_mean * delta.shape[1])
        mix_coeffs /= np.sum(mix_coeffs)

        return delta, mix_coeffs
