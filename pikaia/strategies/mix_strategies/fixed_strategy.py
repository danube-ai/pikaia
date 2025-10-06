import numpy as np

from pikaia.strategies.base_strategies import MixStrategy


class FixedMixStrategy(MixStrategy):
    """
    Applies a fixed set of mixing coefficients to a delta tensor.

    This strategy multiplies the input delta tensor by the provided mixing coefficients
    using Einstein summation, without updating or adapting the coefficients.


    Example:
        strategy = FixedMixStrategy()
        mixed_delta, coeffs = strategy(delta, mix_coeffs)
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Fixed"

    def __call__(
        self, delta: np.ndarray, mix_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Apply fixed mixing coefficients to the input delta tensor.

        Args:
            delta (np.ndarray):
                A 3D tensor of shape (n, m, k) representing the deltas to be mixed.
            mix_coeffs (np.ndarray):
                A 1D array of length k containing the mixing coefficients to apply.

        Returns:
            tuple[np.ndarray, np.ndarray]:
                - The mixed delta array of shape (n, m) after applying the coefficients.
                - The unchanged mixing coefficients array.
        """
        # Weighted sum over the last axis (k) of delta using mix_coeffs,
        # resulting in shape (n, m)
        return np.einsum("ijk,k->ij", delta, mix_coeffs), mix_coeffs
