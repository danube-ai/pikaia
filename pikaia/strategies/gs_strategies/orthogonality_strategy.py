"""Implement the orthogonality-promoting gene strategy."""

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class OrthoGeneStrategy(GeneStrategy):
    """A gene strategy driven by feature orthogonality (low pairwise correlation).

    !!! warning
        This strategy is experimental and its behavior may change in future
        versions.

    Rewards features that are minimally correlated with all other features.
    The orthogonality score for feature *j* is::

        orthogonality[j] = 1 - mean(|corr(j, k)|) for k ≠ j

    where correlations are computed on the MinMax-normalised data matrix.  The
    score is in ``[0, 1]``; a perfectly uncorrelated feature scores 1.

    When the target ``y`` is provided, it is appended as an extra column before
    computing correlations (supervised mode).  Because orthogonality is now
    measured against the augmented matrix, features that correlate strongly
    with ``y`` receive *lower* scores and are suppressed — the opposite of
    conventional supervised feature selection.  This makes the strategy a
    **novelty/diversity pressure**: it promotes features that add information
    beyond what the target and the other features already capture.  It is most
    useful when mixed with a target-aware strategy (e.g. ``DOMINANT`` or
    ``ENTROPY_MAX``) that handles target relevance, leaving OrthoGene to
    enforce diversity.

    The replicator delta is::

        delta[j] = gf[j] * (4 / N) * (orthogonality[j] - 0.5)

    Scores are computed once on the first call and cached; a mode change
    (supervised ↔ unsupervised) triggers a recomputation.

    Args:
        **kwargs (object): Forwarded to `GeneStrategy`.

    """

    def __init__(self, **kwargs):
        """Initialise the strategy and its cached score state.

        Args:
            **kwargs (object): Options forwarded to :class:`GeneStrategy`.

        """
        super().__init__(**kwargs)
        self._orthogonality: np.ndarray | None = None
        self._mode: str | None = None

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "OrthoGene"

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
    def _compute_orthogonality_from_matrix(X: np.ndarray) -> np.ndarray:
        """Return per-column orthogonality scores for matrix ``X``.

        Args:
            X: Matrix of shape ``(n_samples, n_cols)``.

        Returns:
            Array of shape ``(n_cols,)`` with values in ``[0, 1]``.

        """
        if X.shape[1] <= 1:
            return np.array([1.0])
        corr = np.corrcoef(X.T)
        corr = np.nan_to_num(corr, nan=0.0)
        n = corr.shape[0]
        # Sum of absolute off-diagonal correlations per column
        abs_corr_sum = np.sum(np.abs(corr), axis=0) - 1.0  # subtract self-correlation
        return 1.0 - abs_corr_sum / (n - 1)

    def _get_scores(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        """Return cached orthogonality scores, recomputing if the mode changes.

        Args:
            X: Data matrix of shape ``(n_samples, n_features)``.
            y: Optional target array; triggers supervised mode when provided.

        Returns:
            Per-feature orthogonality scores of shape ``(n_features,)``.

        """
        mode = "supervised" if y is not None else "unsupervised"
        if self._orthogonality is None or self._mode != mode:
            self._mode = mode
            if y is not None:
                y_col = self._encode_target(y).reshape(-1, 1)
                X_aug = np.column_stack([X, y_col])
                # Slice back to n_features — y column was appended only for correlation
                self._orthogonality = self._compute_orthogonality_from_matrix(X_aug)[
                    : X.shape[1]
                ]
            else:
                self._orthogonality = self._compute_orthogonality_from_matrix(X)
        return self._orthogonality

    def __call__(self, ctx: StrategyContext) -> float:
        """Compute delta for the OrthoGene strategy.

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

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Diagonal D: ``D[j,j] = 4 * (orthogonality[j] - 0.5)``.

        Args:
            population: Current population.
            gene_similarity: Gene-similarity matrix (unused).
            org_similarity: Organism-similarity matrix (unused).
            initial_org_fitness_range: Initial fitness range (unused).
            y: Optional target labels for supervised mode.

        Returns:
            ``(D, None)`` where ``D`` is a diagonal ``(M, M)`` matrix.

        """
        scores = self._get_scores(population.matrix, y)
        D = np.diag(4.0 * (scores - 0.5))
        return D, None
