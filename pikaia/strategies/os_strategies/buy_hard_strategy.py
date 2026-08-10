import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BuyHardOrgStrategy(OrgStrategy):
    """
    Organism strategy implementing the CalSim Difficulty1 buy-phase signal.

    Each organism spends its sell capital (earned from hard genes) on genes it
    failed, weighted by how easy those genes are (``mean_j``).  Organisms that
    solved many hard genes accumulate more capital and redistribute it to the
    easy genes they missed.

    For organism *i*, the capital earned from selling is:

    $$
    C_i = \\frac{1}{N} \\sum_k x_{ik}
          \\cdot \\frac{\\text{excl}_k}{1 - \\text{excl}_k + \\varepsilon}
    $$

    normalised by the easy-weighted sum of failed genes:

    $$
    Z_i = \\sum_k (1 - x_{ik}) \\cdot \\bar{x}_k
    $$

    The buy contribution from organism *i* to gene *j* is:

    $$
    \\Delta_{\\text{buy\\_hard}}(i, j) =
        (1 - x_{ij}) \\cdot \\frac{C_i}{Z_i} \\cdot \\bar{x}_j
    $$

    Pair with `SellHardGeneStrategy` to reproduce a full CalSim Difficulty1
    ("fair") recalibration round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "BuyHard"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        X = ctx.population.matrix
        N, M = X.shape
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)

        max_capital = (X * sell_signal[np.newaxis, :]).sum(axis=1) / N
        excl_norm2 = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)

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
        """Linear d-vector: ``d[j] = mean_j · Σ_i (1-x_ij) · C_i / Z_i``."""
        X = population.matrix
        N = population.N
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)

        max_capital = (X * sell_signal[np.newaxis, :]).sum(axis=1) / N
        excl_norm2 = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)

        safe_norm = np.where(excl_norm2 < 1e-10, 1.0, excl_norm2)
        weights = np.where(excl_norm2 < 1e-10, 0.0, max_capital / safe_norm)

        d = mean_all * ((1.0 - X) * weights[:, np.newaxis]).sum(axis=0)
        return None, d
