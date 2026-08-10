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
        X = ctx.population.matrix
        N = X.shape[0]
        j = ctx.gene_id
        mean_j = X[:, j].mean()
        excl_j = 1.0 - mean_j
        # CalSim D2: vdeltaSell=0 when excl=0 (all solved) or excl=1 (none solved)
        if excl_j < 1e-6 or excl_j > 1.0 - 1e-6:
            return 0.0
        return float(-(1.0 / N) * X[ctx.org_id, j])

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = -mean_j`` for non-trivial genes only."""
        mean = population.matrix.mean(axis=0)
        excl = 1.0 - mean
        mask = (excl > 1e-6) & (excl < 1.0 - 1e-6)
        d = -mean * mask.astype(float)
        return None, d
