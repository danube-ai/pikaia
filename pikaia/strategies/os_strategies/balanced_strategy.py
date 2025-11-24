import numpy as np

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
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            np.ndarray:
                A vector of computed delta values `Delta_O(i,j)` of shape `(m,)`.
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
