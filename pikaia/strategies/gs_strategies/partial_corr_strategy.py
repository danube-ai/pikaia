import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class PartialCorrGeneStrategy(GeneStrategy):
    """
    A gene strategy driven by partial correlation with the target.

    .. warning::
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

    Without a target (unsupervised mode), partial correlations are computed
    between the last feature and every other feature, which is arbitrary and
    not recommended.  For meaningful results, always pass ``y``.

    Pre-computed partial correlations can be supplied via ``precomputed_pc``
    to skip the expensive matrix inversion.  The scores are cached on first
    use.

    Args:
        precomputed_pc: Pre-computed partial correlations of shape
            ``(n_features,)``.  If provided, no computation is done.
        n_bins: Unused; kept for API compatibility.
        **kwargs: Forwarded to :class:`GeneStrategy`.
    """

    def __init__(
        self, precomputed_pc: np.ndarray | None = None, n_bins: int = 10, **kwargs
    ):
        super().__init__(**kwargs)
        self._partial_corrs: np.ndarray | None = precomputed_pc
        self._n_bins = n_bins

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "PartialCorr"

    @staticmethod
    def _encode_target(y: np.ndarray) -> np.ndarray:
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
        if self._partial_corrs is None:
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

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Diagonal D: ``D[j,j] = 4 * (pc[j] - 0.5)``.

        Args:
            population: Current population.
            gene_similarity: Gene-similarity matrix (unused).
            org_similarity: Organism-similarity matrix (unused).
            initial_org_fitness_range: Initial fitness range (unused).
            y: Target labels.  Pass these to enable partial correlation computation.

        Returns:
            ``(D, None)`` where ``D`` is a diagonal ``(M, M)`` matrix.
        """
        scores = self._get_scores(population.matrix, y)
        D = np.diag(4.0 * (scores - 0.5))
        return D, None
