import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BalancedOrgStrategy(OrgStrategy):
    """
    An organism strategy that promotes balanced gene contributions.

    This strategy adjusts gene fitness to favor organisms where the
    contribution of each gene to the organism's total fitness is balanced.
    It penalizes genes that contribute disproportionately (more or less) than
    the average. This implementation follows the logic from the original
    `alg.py`.
    """

    def __init__(self, **kwargs):
        """Initialise the Balanced organism strategy.

        Args:
            **kwargs: Keyword options forwarded to :class:`OrgStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Balanced"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """
        Computes deltas for a balanced organism strategy.

        The formula calculates the deviation of each gene's contribution from
        the ideal balanced state (`1/m`) and adjusts its fitness accordingly.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            np.ndarray: A vector of computed delta values `Delta_O(i,j)` of shape `(m,)`.
        """
        current_org_fitness = ctx.org_fitness[ctx.org_id]

        if current_org_fitness == 0:
            return np.zeros(ctx.population.M)

        delta_o = (
            # constant factor and normalization by population size
            (-2 / ctx.population.N)
            # deviation from ideal balanced contribution
            * (
                (ctx.population[ctx.org_id, :] * ctx.gene_fitness) / current_org_fitness
                - 1 / ctx.population.M
            )
            * current_org_fitness
        )
        return delta_o

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Rank-1 D matrix exploiting gamma normalization.

        Because ``sum_j gamma_j = 1``, a row-constant matrix
        ``D[j,k] = -2*x_bar_j`` satisfies ``(D@gamma)_j = -2*x_bar_j``
        for *any* normalized ``gamma``.  Combined with the outer
        ``gamma_j`` multiplier in the replicator step this gives:

            step_j = gamma_j * (-2*x_bar_j)

        which exactly reproduces the balanced-org contribution
        ``delta_j ≈ -2*x_bar_j*gamma_j`` (the j-independent constant
        ``2*w_bar/M`` cancels under normalisation).

        Encoding as a ``D`` matrix (rather than a ``d`` vector) keeps the
        step O(1/M) near the uniform point, ensuring numerical stability.
        """
        x_bar = population.matrix.mean(axis=0)  # (M,)
        M = population.M
        # D[j, k] = -2*x_bar_j  for all k
        D = np.outer(-2.0 * x_bar, np.ones(M))
        return D, None
