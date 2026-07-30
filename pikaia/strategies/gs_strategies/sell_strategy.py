import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class SellGeneStrategy(GeneStrategy):
    """
    Gene strategy implementing the CalSim sell-phase signal.

    In CalSim's market simulation each proband *sells* value proportional to
    their performance on an exercise scaled by the exercise's difficulty odds
    (``excl / (1 - excl)``).  This drains value from genes that are commonly
    expressed AND carry a high difficulty weight.

    Translated to the replicator equation, the per-organism delta for gene *j*
    is:

    .. math::

        \\Delta_{\\text{sell}}(i,j) = -\\frac{x_{ij}}{N}
            \\cdot \\frac{\\text{excl}_j}{1 - \\text{excl}_j + \\varepsilon}

    Summed over all organisms this gives
    ``-\\bar{x}_j \\cdot \\text{excl}_j / (1 - \\text{excl}_j)``, exactly
    the CalSim sell loss for gene *j*.

    This strategy is designed to be paired with :class:`BuyOrgStrategy` to
    fully reproduce CalSim's recalibration round within the replicator
    framework.

    .. warning::
        This strategy is experimental and its behaviour may change in future
        versions.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "Sell"

    def __call__(self, ctx: StrategyContext) -> float:
        """Per-organism sell contribution for gene *j*.

        Args:
            ctx: Strategy context. ``org_id`` and ``gene_id`` must be set.

        Returns:
            Scalar delta ``Δ_sell(i, j)``.
        """
        mean_all = ctx.population.matrix.mean(axis=0)  # (M,)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)  # odds ratio, (M,)

        return float(
            (-1.0 / ctx.population.N)
            * sell_signal[ctx.gene_id]
            * ctx.population[ctx.org_id, ctx.gene_id]
        )

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = -mean_j * excl_j / (1 - excl_j + eps)``.

        The sell loss is independent of ``gamma``, so D=None and the entire
        signal lives in the d-vector.
        """
        mean_all = population.matrix.mean(axis=0)  # (M,)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)
        d = -mean_all * sell_signal  # (M,)
        return None, d
