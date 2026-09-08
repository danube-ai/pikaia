"""Implement the score-based strategy that buys organisms uniformly."""

import numpy as np

from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BuyUniformOrgStrategy(OrgStrategy):
    r"""Trading buy-phase paired with `SellUniformGeneStrategy`.

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

    Pair with `SellUniformGeneStrategy` for the full uniform trading round.
    """

    def __init__(self, **kwargs):
        """Initialise the strategy with options accepted by ``OrgStrategy``.

        Args:
            **kwargs (object): Options forwarded to :class:`OrgStrategy`.

        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return the stable registry name for this strategy."""
        return "BuyUniform"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """Return the uniform-trade buy contribution for one organism.

        Args:
            ctx: Evaluation context identifying the organism to evaluate.

        Returns:
            Vector of gene-fitness deltas for the selected organism.

        """
        X = ctx.population.matrix
        N, M = X.shape
        gamma = ctx.gene_fitness
        mean_all = X.mean(axis=0)
        excl = 1.0 - mean_all

        # Capital only from genes with non-trivial exclusiveness: excl ∉ {0, 1}
        sell_mask = ((excl > 1e-6) & (excl < 1.0 - 1e-6)).astype(float)
        max_capital = (X * (sell_mask * gamma)[np.newaxis, :]).sum(axis=1) / N
        excl_norm = ((1.0 - X) * excl[np.newaxis, :]).sum(axis=1)

        i = ctx.org_id
        if excl_norm[i] < 1e-10:
            return np.zeros(M)
        buy_abs = (1.0 - X[i, :]) * max_capital[i] / excl_norm[i] * excl
        return buy_abs / (gamma + 1e-10)
