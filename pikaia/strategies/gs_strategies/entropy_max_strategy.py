import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class EntropyMaxGeneStrategy(GeneStrategy):
    """
    A supervised gene strategy driven by information-theoretic relevance.

    .. warning::
        This strategy is experimental and its behavior may change in future
        versions.

    Scores each feature as the product of its normalised mutual information
    with the target ``y`` and its normalised differential entropy (a proxy for
    feature variance).  Both components are independently normalised to
    ``[0, 1]`` before multiplication, so the combined score rewards features
    that are simultaneously informative *and* spread out.

    The replicator delta is::

        delta[j] = gf[j] * (4 / N) * (info_score[j] - 0.5)

    where ``info_score[j] ∈ [0, 1]`` is the product described above.

    When ``y`` is not supplied the strategy falls back to entropy-only scores
    (mutual information term is zero), making all ``info_score[j] = 0`` and
    the delta uniformly negative — equivalent to a mild anti-high-variance
    penalty.  For the strategy to be useful, always pass ``y``.

    The per-feature scores are computed once on the first call and cached.  To
    supply labels when using the kernel path, pass ``y`` to ``kernel()``.

    Args:
        n_bins: Number of bins used when discretising continuous features for
            mutual information estimation.  Default ``10``.
        precomputed_info: Pre-computed info scores of shape ``(n_features,)``.
            If provided, skips the MI computation entirely.
        **kwargs: Forwarded to :class:`GeneStrategy`.
    """

    def __init__(
        self, n_bins: int = 10, precomputed_info: np.ndarray | None = None, **kwargs
    ):
        super().__init__(**kwargs)
        self.n_bins = n_bins
        self._info_scores: np.ndarray | None = precomputed_info
        self._mode: str | None = None if precomputed_info is None else "precomputed"

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "EntropyMax"

    @staticmethod
    def _encode_target(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y).flatten()
        if not np.issubdtype(y.dtype, np.number):
            unique_vals = np.unique(y)
            return np.searchsorted(unique_vals, y).astype(int)
        return y

    @staticmethod
    def compute_info_scores(
        X: np.ndarray, y: np.ndarray | None = None, n_bins: int = 10
    ) -> np.ndarray:
        """Compute per-feature information scores as MI × entropy (both normalised).

        Args:
            X: Data matrix of shape ``(n_samples, n_features)``, values in ``[0, 1]``.
            y: Target array of shape ``(n_samples,)``.  Required for useful scores.
            n_bins: Number of histogram bins for MI estimation.

        Returns:
            Array of shape ``(n_features,)`` with values in ``[0, 1]``.
        """
        from sklearn.metrics import mutual_info_score

        n_features = X.shape[1]

        # Mutual information component
        if y is not None:
            y_enc = EntropyMaxGeneStrategy._encode_target(y)
            mi_scores = np.zeros(n_features)
            for j in range(n_features):
                try:
                    x_j = X[:, j]
                    bins = np.histogram(x_j, bins=n_bins)[1][:-1]
                    x_bin = np.clip(np.digitize(x_j, bins=bins), 1, n_bins)
                    mi_scores[j] = mutual_info_score(y_enc, x_bin)
                except Exception:
                    mi_scores[j] = 0.0
        else:
            mi_scores = np.zeros(n_features)

        # Differential entropy proxy: 0.5 * log(2πe * var)
        variances = np.maximum(np.var(X, axis=0), 1e-12)
        entropy_scores = 0.5 * np.log(2 * np.pi * np.e * variances)
        e_min, e_max = entropy_scores.min(), entropy_scores.max()
        entropy_norm = (
            (entropy_scores - e_min) / (e_max - e_min)
            if e_max > e_min
            else np.full(n_features, 0.5)
        )

        # Normalise MI and combine
        mi_max = mi_scores.max()
        mi_norm = mi_scores / mi_max if mi_max > 0 else np.zeros(n_features)

        return np.clip(mi_norm * entropy_norm, 0.0, 1.0)

    def _get_scores(self, X: np.ndarray, y: np.ndarray | None) -> np.ndarray:
        mode = "supervised" if y is not None else "unsupervised"
        if self._info_scores is None or (
            self._mode != "precomputed" and self._mode != mode
        ):
            self._mode = mode
            self._info_scores = self.compute_info_scores(X, y, self.n_bins)
        return self._info_scores

    def __call__(self, ctx: StrategyContext) -> float:
        """Compute delta for the EntropyMax gene strategy.

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
        """Diagonal D: ``D[j,j] = 4 * (info_score[j] - 0.5)``.

        Args:
            population: Current population.
            gene_similarity: Gene-similarity matrix (unused).
            org_similarity: Organism-similarity matrix (unused).
            initial_org_fitness_range: Initial fitness range (unused).
            y: Target labels.  Pass these to enable MI computation.

        Returns:
            ``(D, None)`` where ``D`` is a diagonal ``(M, M)`` matrix.
        """
        scores = self._get_scores(population.matrix, y)
        D = np.diag(4.0 * (scores - 0.5))
        return D, None
