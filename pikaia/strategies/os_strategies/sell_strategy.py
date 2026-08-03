import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class SellOrgStrategy(OrgStrategy):
    """
    Organism strategy implementing the CalSim sell-phase signal.

    In CalSim's market simulation each proband *sells* value across all
    exercises they solved, proportional to each exercise's difficulty odds
    (``excl / (1 - excl)``).  The operation is organism-driven: each proband
    sells once per iteration across the full gene vector.

    For organism *i*, the per-gene sell contribution is:

    $$
    \\Delta_{\\text{sell}}(i,j) = -\\frac{x_{ij}}{N}
        \\cdot \\frac{\\text{excl}_j}{1 - \\text{excl}_j + \\varepsilon}
    $$

    Summed over all organisms this gives
    ``-\\bar{x}_j \\cdot \\text{excl}_j / (1 - \\text{excl}_j)``, which is
    exactly the CalSim sell loss for gene *j*.

    This strategy is designed to be paired with `BuyOrgStrategy` to
    fully reproduce CalSim's recalibration round within the replicator
    framework.

    !!! warning
        This strategy is experimental and its behaviour may change in future
        versions.
    """

    def __init__(self, **kwargs):
        """Initialise the Sell organism strategy.

        Args:
            **kwargs: Keyword options forwarded to `OrgStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Sell"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """Sell contributions from organism ``ctx.org_id`` to all genes.

        Args:
            ctx: Strategy context. ``org_id`` must be set; ``gene_id`` is
                ignored (organism strategies return the full gene vector).

        Returns:
            Array of shape ``(M,)`` with delta values ``Δ_sell(i, j)`` for
            every gene *j*.
        """
        X = ctx.population.matrix  # (N, M)
        N = ctx.population.N
        mean_all = X.mean(axis=0)  # (M,)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)  # odds ratio, (M,)
        return -(1.0 / N) * X[ctx.org_id, :] * sell_signal

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = -mean_j * excl_j / (1 - excl_j + eps)``.

        The sell loss is independent of ``gamma``, so ``D = None`` and the
        entire signal lives in the ``d``-vector.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Unused.
            initial_org_fitness_range: Unused.
            y: Unused.

        Returns:
            Tuple ``(None, d)`` where ``d`` is a ``(M,)`` vector with
            ``d[j] = -mean_j * excl_j / (1 - excl_j + eps)``.
        """
        mean_all = population.matrix.mean(axis=0)  # (M,)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)
        d = -mean_all * sell_signal  # (M,)
        return None, d
