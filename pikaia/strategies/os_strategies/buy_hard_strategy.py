import numpy as np

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
        gamma = ctx.gene_fitness
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)

        # Capital scales with current gene fitness, matching CalSim's currentValue weighting.
        max_capital = (X * (sell_signal * gamma)[np.newaxis, :]).sum(axis=1) / N
        excl_norm2 = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)

        i = ctx.org_id
        if excl_norm2[i] < 1e-10:
            return np.zeros(M)
        buy_abs = (1.0 - X[i, :]) * max_capital[i] / excl_norm2[i] * mean_all
        # Proportional delta = buy_abs / gamma_j.  At uniform start (gamma=1/M) this
        # equals the original constant formula; at equilibrium it reproduces CalSim's
        # fixed-point condition gamma_j * sell_signal_j * mean_j = buy_abs_j.
        return buy_abs / (gamma + 1e-10)
