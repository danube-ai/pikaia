"""Implement the partial-correlation gene strategy."""

import numpy as np

from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class PartialCorrGeneStrategy(GeneStrategy):
    """A gene strategy driven by partial correlation with the target.

    !!! warning
        This strategy is experimental and its behavior may change in future
        versions.

    Rewards features whose relationship with the target survives controlling
    for all other features.  The partial correlation of feature *j* with the
    target is estimated via shrinkage precision matrices to handle
    multicollinearity::

        pc[j] = |P[j, target]| / sqrt(|P[j,j]| * |P[target,target]|)

    where ``P`` is the shrinkage precision matrix of ``[X | y]``.  Scores are
    clipped to ``[0, 1]``.

    The replicator delta is::

        delta[j] = gf[j] * (4 / N) * (pc[j] - 0.5)

    Without a target (unsupervised mode), all partial correlations are set to
    zero, making every delta negative.  For meaningful results, always pass
    ``y``.

    Pre-computed partial correlations can be supplied via ``precomputed_pc``
    to skip the expensive matrix inversion.  The scores are cached on first
    use.

    Args:
        precomputed_pc: Pre-computed partial correlations of shape
            ``(n_features,)``.  If provided, no computation is done.
        n_bins: Unused; kept for API compatibility.
        **kwargs (object): Forwarded to `GeneStrategy`.

    """

    def __init__(
        self, precomputed_pc: np.ndarray | None = None, n_bins: int = 10, **kwargs
    ):
        """Initialise score calculation and optional partial correlations.

        Args:
            precomputed_pc: Optional per-feature partial-correlation scores.
            n_bins: Retained for compatibility; not used by this strategy.
            **kwargs (object): Options forwarded to :class:`GeneStrategy`.

        """
        super().__init__(**kwargs)
        self._partial_corrs: np.ndarray | None = precomputed_pc
        self._n_bins = n_bins
        self._mode: str | None = None if precomputed_pc is None else "precomputed"

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "PartialCorr"

    @staticmethod
    def _encode_target(y: np.ndarray) -> np.ndarray:
        """Encode a target array to binary ±1 labels centred at the median.

        Args:
            y: Target array of any dtype.

        Returns:
            1D float array with values ``+1.0`` (≥ median) or ``-1.0`` (< median).

        """
        y = np.asarray(y).flatten()
        if not np.issubdtype(y.dtype, np.number):
            unique_vals = np.unique(y)
            y_numeric = np.searchsorted(unique_vals, y).astype(float)
        else:
            y_numeric = y.astype(float)
        median = np.median(y_numeric)
        return np.where(y_numeric >= median, 1.0, -1.0)

    @staticmethod
    def compute_partial_correlations(
        X: np.ndarray, y: np.ndarray | None = None, reg: float = 0.5
    ) -> np.ndarray:
        """Compute partial correlations between each feature and the target.

        Uses Ledoit-Wolf-style shrinkage towards a scaled identity to regularise
        the precision matrix.

        Args:
            X: Data matrix ``(n_samples, n_features)``, values in ``[0, 1]``.
            y: Target array ``(n_samples,)``.  If ``None``, returns zeros.
            reg: Shrinkage coefficient in ``[0, 1]``.  Higher values pull the
                precision matrix towards the diagonal.  Default ``0.5``.

        Returns:
            Array of shape ``(n_features,)`` with values in ``[0, 1]``.

        """
        n_features = X.shape[1]
        if y is None:
            return np.zeros(n_features)

        y_enc = PartialCorrGeneStrategy._encode_target(y)
        joint = np.column_stack([X, y_enc])
        cov = np.nan_to_num(np.cov(joint.T, ddof=0), nan=0.0)
        p = cov.shape[0]

        # Shrink towards scaled identity
        target_diag = (np.trace(cov) / p) * np.eye(p)
        cov_shrunk = (1.0 - reg) * cov + reg * target_diag + 1e-6 * np.eye(p)

        try:
            precision = np.linalg.inv(cov_shrunk)
        except np.linalg.LinAlgError:
            precision = np.linalg.pinv(cov_shrunk)
        precision = np.nan_to_num(precision, nan=0.0)

        last = p - 1  # index of the target column
        pc = np.zeros(n_features)
        for j in range(n_features):
            denom = np.sqrt(abs(precision[j, j]) * abs(precision[last, last]))
            pc[j] = abs(precision[j, last]) / denom if denom > 1e-12 else 0.0

        return np.clip(pc, 0.0, 1.0)

    def _get_scores(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        """Return cached partial correlation scores, recomputing if the mode changes.

        Args:
            X: Data matrix of shape ``(n_samples, n_features)``.
            y: Optional target array; triggers supervised mode when provided.

        Returns:
            Per-feature partial correlation scores of shape ``(n_features,)``.

        """
        mode = "supervised" if y is not None else "unsupervised"
        if self._partial_corrs is None or (
            self._mode != "precomputed" and self._mode != mode
        ):
            self._mode = mode
            self._partial_corrs = self.compute_partial_correlations(X, y)
        return self._partial_corrs

    def __call__(self, ctx: StrategyContext) -> float:
        """Compute delta for the PartialCorr gene strategy.

        Args:
            ctx: Strategy context.  ``ctx.y`` is used when available.

        Returns:
            float: The computed delta ``Delta_G(i,j)``.

        """
        scores = self._get_scores(ctx.population.matrix, ctx.y)
        return float(
            (4 / ctx.population.N)
            * ctx.gene_fitness[ctx.gene_id]
            * (scores[ctx.gene_id] - 0.5)
        )
