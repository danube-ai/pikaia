import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BuyOrgStrategy(OrgStrategy):
    """
    Organism strategy implementing the CalSim buy-phase signal.

    In CalSim's market simulation each proband *buys* value in exercises they
    failed, using the capital they earned from selling.  Probands who solved
    many hard exercises accumulate more capital and redistribute it to the
    genes they lack — this creates a cross-gene, cross-organism interaction
    that a diagonal gene strategy cannot capture.

    For organism *i*, its capital from selling is:

    $$
    C_i = \\frac{1}{N} \\sum_k x_{ik}
          \\cdot \\frac{\\text{excl}_k}{1 - \\text{excl}_k + \\varepsilon}
    $$

    and it is normalised by the exercises it failed weighted by mean:

    $$
    Z_i = \\sum_k (1 - x_{ik}) \\cdot \\bar{x}_k
    $$

    The buy contribution from organism *i* to gene *j* is:

    $$
    \\Delta_{\\text{buy}}(i, j) =
        (1 - x_{ij}) \\cdot \\frac{C_i}{Z_i} \\cdot \\bar{x}_j
    $$

    This strategy is designed to be paired with `SellOrgStrategy` to
    fully reproduce CalSim's recalibration round within the replicator
    framework.

    !!! warning
        This strategy is experimental and its behaviour may change in future
        versions.
    """

    def __init__(self, **kwargs):
        """Initialise the Buy organism strategy.

        Args:
            **kwargs: Keyword options forwarded to `OrgStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Buy"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """Buy contributions from organism ``ctx.org_id`` to all genes.

        Args:
            ctx: Strategy context. ``org_id`` must be set; ``gene_id`` is
                ignored (organism strategies return the full gene vector).

        Returns:
            Array of shape ``(M,)`` with delta values ``Δ_buy(i, j)`` for
            every gene *j*.
        """
        X = ctx.population.matrix  # (N, M)
        N, M = X.shape
        mean_all = X.mean(axis=0)  # (M,)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)  # (M,)

        # Capital each organism earns from selling
        max_capital = (X * sell_signal[np.newaxis, :]).sum(axis=1) / N  # (N,)

        # Normaliser: weighted sum of what organism i failed
        excl_norm2 = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)  # (N,)

        i = ctx.org_id
        if excl_norm2[i] < 1e-10:
            return np.zeros(M)

        return (1.0 - X[i, :]) * max_capital[i] / excl_norm2[i] * mean_all

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector summing buy redistribution over all organisms.

        ``d[j] = mean_j * Σ_i [(1 - x_ij) * C_i / Z_i]`` where ``C_i`` is
        the per-organism sell capital and ``Z_i`` is the normaliser.  The
        signal is independent of ``gamma``, so ``D = None``.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Unused.
            initial_org_fitness_range: Unused.
            y: Unused.

        Returns:
            Tuple ``(None, d)`` where ``d`` is a ``(M,)`` vector with
            ``d[j] = mean_j * Σ_i (1 - x_ij) * C_i / Z_i``.
        """
        X = population.matrix  # (N, M)
        N = population.N
        mean_all = X.mean(axis=0)  # (M,)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)  # (M,)

        max_capital = (X * sell_signal[np.newaxis, :]).sum(axis=1) / N  # (N,)
        excl_norm2 = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)  # (N,)

        safe_norm2 = np.where(excl_norm2 < 1e-10, 1.0, excl_norm2)
        weights = np.where(excl_norm2 < 1e-10, 0.0, max_capital / safe_norm2)  # (N,)

        # d[j] = mean_j * Σ_i (1 - x_ij) * weights_i
        d = mean_all * ((1.0 - X) * weights[:, np.newaxis]).sum(axis=0)  # (M,)
        return None, d
