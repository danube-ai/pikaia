"""Implement the redundancy-penalty gene strategy."""

import numpy as np

from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class RedundancyPenaltyGeneStrategy(GeneStrategy):
    """A gene strategy that penalises redundant (highly correlated) features.

    !!! warning
        This strategy is experimental and its behavior may change in future
        versions.

    Computes a redundancy score for each feature as the mean absolute pairwise
    correlation with all other features::

        redundancy[j] = mean(|corr(j, k)|) for k ≠ j

    and promotes features with *low* redundancy::

        delta[j] = gf[j] * (4 / N) * (0.5 - redundancy[j])

    Features with redundancy below 0.5 receive a positive delta (promoted);
    highly correlated features receive a negative delta (suppressed).

    In supervised mode (when ``y`` is provided), the target is appended as an
    extra column before computing the correlation matrix, biasing the strategy
    towards features that are both non-redundant *among themselves* and
    non-redundant relative to the target.  The target column is excluded from
    the returned scores.

    Scores are computed once on first use and cached; a mode change triggers
    a recomputation.

    Args:
        precomputed_redundancy: Pre-computed redundancy scores of shape
            ``(n_features,)``.  If provided, skips computation entirely.
        **kwargs (object): Forwarded to `GeneStrategy`.

    """

    def __init__(self, precomputed_redundancy: np.ndarray | None = None, **kwargs):
        """Initialise score calculation and optional redundancy scores.

        Args:
            precomputed_redundancy: Optional per-feature redundancy scores.
            **kwargs (object): Options forwarded to :class:`GeneStrategy`.

        """
        super().__init__(**kwargs)
        self._redundancy: np.ndarray | None = precomputed_redundancy
        self._mode: str | None = None

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "RedundancyPenalty"

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
    def compute_redundancy(X: np.ndarray) -> np.ndarray:
        """Compute mean absolute pairwise correlation for each column of ``X``.

        Args:
            X: Matrix of shape ``(n_samples, n_cols)``.

        Returns:
            Array of shape ``(n_cols,)`` with values in ``[0, 1]``.

        """
        if X.shape[1] <= 1:
            return np.array([0.0])
        corr = np.nan_to_num(np.corrcoef(X.T), nan=0.0)
        n = corr.shape[0]
        # Mean absolute off-diagonal correlation per column
        abs_corr_sum = np.sum(np.abs(corr), axis=0) - 1.0  # subtract self-correlation
        return abs_corr_sum / (n - 1)

    def _get_scores(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        """Return cached redundancy scores, recomputing if the mode changes.

        Args:
            X: Data matrix of shape ``(n_samples, n_features)``.
            y: Optional target array; triggers supervised mode when provided.

        Returns:
            Per-feature redundancy scores of shape ``(n_features,)``.

        """
        mode = "supervised" if y is not None else "unsupervised"
        if self._redundancy is None or self._mode != mode:
            self._mode = mode
            if y is not None:
                y_col = self._encode_target(y).reshape(-1, 1)
                X_aug = np.column_stack([X, y_col])
                # Slice back to n_features — y column was appended only for correlation
                self._redundancy = self.compute_redundancy(X_aug)[: X.shape[1]]
            else:
                self._redundancy = self.compute_redundancy(X)
        return self._redundancy

    def __call__(self, ctx: StrategyContext) -> float:
        """Compute delta for the RedundancyPenalty gene strategy.

        Args:
            ctx: Strategy context.  ``ctx.y`` is used when available.

        Returns:
            float: The computed delta ``Delta_G(i,j)``.

        """
        scores = self._get_scores(ctx.population.matrix, ctx.y)
        return float(
            (4 / ctx.population.N)
            * ctx.gene_fitness[ctx.gene_id]
            * (0.5 - scores[ctx.gene_id])
        )
