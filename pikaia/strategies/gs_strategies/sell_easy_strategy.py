import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class SellEasyGeneStrategy(GeneStrategy):
    """
    Gene strategy implementing the CalSim Inverse sell signal.

    Easy genes (high mean expression, low exclusiveness) lose more value —
    the sell signal is inverted relative to `SellHardGeneStrategy`.

    For organism *i*, gene *j*:

    $$
    \\Delta_{\\text{sell\\_easy}}(i, j) =
        +\\frac{x_{ij}}{N}
        \\cdot \\frac{\\text{excl}_j}{1 - \\text{excl}_j + \\varepsilon}
    $$

    Summed over all organisms this equals
    ``+mean_j · excl_j / (1 - excl_j)``, the exact negation of the
    ``SellHardGeneStrategy`` signal.

    Pair with `BuyEasyOrgStrategy` to reproduce a full CalSim Inverse
    recalibration round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "SellEasy"

    def __call__(self, ctx: StrategyContext) -> float:
        X = ctx.population.matrix
        N = ctx.population.N
        mean_j = X[:, ctx.gene_id].mean()
        excl_j = 1.0 - mean_j
        sell_signal_j = excl_j / (1.0 - excl_j + 1e-8)
        return float((1.0 / N) * X[ctx.org_id, ctx.gene_id] * sell_signal_j)

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = +mean_j · excl_j / (1 - excl_j + eps)``."""
        mean_all = population.matrix.mean(axis=0)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)
        d = mean_all * sell_signal
        return None, d
