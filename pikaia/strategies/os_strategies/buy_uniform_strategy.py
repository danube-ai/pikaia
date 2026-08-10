import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BuyUniformOrgStrategy(OrgStrategy):
    """
    Organism strategy implementing the CalSim Difficulty2 buy-phase signal.

    Each organism spends its uniform sell capital (proportional to average
    performance) on genes it failed, weighted by how hard those genes are
    (``excl_j``).  Unlike `BuyHardOrgStrategy`, easy organisms and hard genes
    receive more attention here.

    For organism *i*, the capital from uniform selling is:

    $$
    C_i = \\frac{1}{N} \\sum_k x_{ik}
    $$

    normalised by the hard-weighted sum of failed genes:

    $$
    Z_i = \\sum_k (1 - x_{ik}) \\cdot \\text{excl}_k
    $$

    The buy contribution from organism *i* to gene *j* is:

    $$
    \\Delta_{\\text{buy\\_uniform}}(i, j) =
        (1 - x_{ij}) \\cdot \\frac{C_i}{Z_i} \\cdot \\text{excl}_j
    $$

    Pair with `SellUniformGeneStrategy` to reproduce a full CalSim Difficulty2
    ("inclusive") recalibration round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "BuyUniform"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        X = ctx.population.matrix
        N, M = X.shape
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all

        max_capital = X.sum(axis=1) / N
        excl_norm = ((1.0 - X) * excl[np.newaxis, :]).sum(axis=1)

        i = ctx.org_id
        if excl_norm[i] < 1e-10:
            return np.zeros(M)
        return (1.0 - X[i, :]) * max_capital[i] / excl_norm[i] * excl

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = excl_j · Σ_i (1-x_ij) · C_i / Z_i``."""
        X = population.matrix
        N = population.N
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all

        max_capital = X.sum(axis=1) / N
        excl_norm = ((1.0 - X) * excl[np.newaxis, :]).sum(axis=1)

        safe_norm = np.where(excl_norm < 1e-10, 1.0, excl_norm)
        weights = np.where(excl_norm < 1e-10, 0.0, max_capital / safe_norm)

        d = excl * ((1.0 - X) * weights[:, np.newaxis]).sum(axis=0)
        return None, d
