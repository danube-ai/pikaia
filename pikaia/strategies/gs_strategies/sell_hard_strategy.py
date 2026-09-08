"""Implement the score-based strategy that preferentially sells hard genes."""

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class SellHardGeneStrategy(GeneStrategy):
    r"""Trading sell signal weighted by gene difficulty.

    Hard genes (low mean expression, high exclusiveness) lose more value per
    unit of performance — organisms that solved them "sell" at a premium.

    For organism *i*, gene *j*:

    $$
    \\Delta_{\\text{sell\\_hard}}(i, j) =
        -\\frac{x_{ij}}{N}
        \\cdot \\frac{\\text{excl}_j}{1 - \\text{excl}_j + \\varepsilon}
    $$

    Summed over all organisms this equals
    ``-mean_j · excl_j / (1 - excl_j)``, the proportional sell loss
    weighted by gene difficulty.

    Pair with `BuyHardOrgStrategy` for the full hard-gene trading round.
    """

    def __init__(self, **kwargs):
        """Initialise the strategy with options accepted by ``GeneStrategy``.

        Args:
            **kwargs (object): Options forwarded to :class:`GeneStrategy`.

        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """Return the stable registry name for this strategy."""
        return "SellHard"

    def __call__(self, ctx: StrategyContext) -> float:
        """Return this organism's difficulty-weighted sell contribution.

        Args:
            ctx: Evaluation context identifying the organism and gene.

        Returns:
            Signed gene-fitness delta for the selected organism and gene.

        """
        X = ctx.population.matrix
        N = ctx.population.N
        mean_j = X[:, ctx.gene_id].mean()
        excl_j = 1.0 - mean_j
        sell_signal_j = excl_j / (1.0 - excl_j + 1e-8)
        return -(1.0 / N) * X[ctx.org_id, ctx.gene_id] * sell_signal_j

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Linear d-vector: ``d[j] = -mean_j · excl_j / (1 - excl_j + eps)``."""
        mean_all = population.matrix.mean(axis=0)
        excl = 1.0 - mean_all
        sell_signal = excl / (1.0 - excl + 1e-8)
        d = -mean_all * sell_signal
        return None, d
