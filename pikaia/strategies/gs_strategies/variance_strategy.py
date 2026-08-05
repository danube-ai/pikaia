import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


def _normalized_std(matrix: np.ndarray) -> np.ndarray:
    """Return per-column population std normalised by the maximum std."""
    std = matrix.std(axis=0, ddof=0)
    return std / (std.max() + 1e-8)


class VarianceGeneStrategy(GeneStrategy):
    """
    A gene strategy that rewards features with high cross-organism dispersion.

    Scales a Dominant-style expression signal by the column's normalised
    standard deviation so that genes which separate organisms more strongly
    receive larger fitness deltas. Near-constant columns contribute ~0.
    """

    def __init__(self, **kwargs):
        """Initialise the Variance gene strategy.

        Args:
            **kwargs: Keyword options forwarded to `GeneStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Variance"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a variance-weighted gene.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            float: The computed delta value `Delta_G(i,j)` for the specified gene and organism.
        """
        s_hat = _normalized_std(ctx.population.matrix)
        return float(
            (4 / ctx.population.N)
            * ctx.gene_fitness[ctx.gene_id] ** 2
            * s_hat[ctx.gene_id]
            * (ctx.population[ctx.org_id, ctx.gene_id] - 0.5)
        )

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Diagonal D matrix scaled by normalised column std.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Unused.
            initial_org_fitness_range: Unused.
            y: Unused.

        Returns:
            Tuple ``(D, None)`` where ``D`` is a diagonal ``(M, M)`` matrix
            with ``D[j, j] = 4 * s_hat_j * (x_bar_j - 0.5)``.
        """
        s_hat = _normalized_std(population.matrix)
        D = np.diag(4.0 * s_hat * (population.matrix.mean(axis=0) - 0.5))
        return D, None
