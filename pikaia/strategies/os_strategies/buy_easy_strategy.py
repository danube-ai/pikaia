import numpy as np

from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BuyEasyOrgStrategy(OrgStrategy):
    """
    Trading buy-phase paired with `SellEasyGeneStrategy` — mirror of `BuyHardOrgStrategy`.

    Capital is earned with a negative sign (from the easy sell signal),
    so the redistribution flows in the opposite direction to `BuyHardOrgStrategy`:
    organisms that solved easy genes accumulate capital and redistribute it
    to genes they failed, weighted by how easy those genes are.

    For organism *i*, the (negative) capital from easy selling is:

    $$
    C_i = -\\frac{1}{N} \\sum_k x_{ik}
           \\cdot \\frac{\\text{excl}_k}{1 - \\text{excl}_k + \\varepsilon}
    $$

    normalised by the easy-weighted sum of failed genes (same as BuyHard):

    $$
    Z_i = \\sum_k (1 - x_{ik}) \\cdot \\bar{x}_k
    $$

    The buy contribution from organism *i* to gene *j* is:

    $$
    \\Delta_{\\text{buy\\_easy}}(i, j) =
        (1 - x_{ij}) \\cdot \\frac{C_i}{Z_i} \\cdot \\bar{x}_j
        = -\\Delta_{\\text{buy\\_hard}}(i, j)
    $$

    Pair with `SellEasyGeneStrategy` for the full easy-gene trading round.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        return "BuyEasy"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        X = ctx.population.matrix
        N, M = X.shape
        gamma = ctx.gene_fitness
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)

        # Negative capital — Inverse sell earns the opposite sign.
        max_capital = -(X * (sell_signal * gamma)[np.newaxis, :]).sum(axis=1) / N
        excl_norm2 = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)

        i = ctx.org_id
        if excl_norm2[i] < 1e-10:
            return np.zeros(M)
        buy_abs = (1.0 - X[i, :]) * max_capital[i] / excl_norm2[i] * mean_all
        return buy_abs / (gamma + 1e-10)
