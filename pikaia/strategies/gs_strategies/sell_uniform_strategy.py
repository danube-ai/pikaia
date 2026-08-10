import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class SellUniformGeneStrategy(GeneStrategy):
    """
    Gene strategy implementing the CalSim Difficulty2 sell signal.

    All genes lose value at the same rate regardless of difficulty — organisms
    "sell" their solved genes uniformly.

    For organism *i*, gene *j*:

    $$
    \\Delta_{\\text{sell\\_uniform}}(i, j) = -\\frac{x_{ij}}{N}
    $$

    Summed over all organisms this equals ``-mean_j``, a uniform sell loss
    independent of gene difficulty.

    Pair with `BuyUniformOrgStrategy` to reproduce a full CalSim Difficulty2
    ("inclusive") recalibration round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "SellUniform"

    def __call__(self, ctx: StrategyContext) -> float:
        N = ctx.population.N
        return float(-(1.0 / N) * ctx.population[ctx.org_id, ctx.gene_id])

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = -mean_j``."""
        d = -population.matrix.mean(axis=0)
        return None, d
