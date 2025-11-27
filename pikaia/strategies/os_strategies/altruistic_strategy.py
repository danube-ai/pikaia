import numpy as np

from pikaia.config.logger import logger
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class AltruisticOrgStrategy(OrgStrategy):
    """
    An organism strategy that promotes altruistic behavior towards relatives.

    .. warning::
        This strategy is experimental and its behavior may change in future
        versions.

    This strategy models altruism where an organism's fitness contribution is
    adjusted based on its interaction with related organisms (kin). The delta
    is calculated based on the fitness difference between the organism and its
    relatives, weighted by their similarity. This implementation follows the
    logic from the original `alg.py`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Altruistic"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """
        Computes deltas for an altruistic organism strategy.

        Args:
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            np.ndarray:
                A vector of computed delta values `Delta_O(i,j)` of shape `(m,)`.
        """
        # Determine kin range
        kin_range = self.options.get("kin_range", ctx.population.N)
        if kin_range > 32:
            logger.warning(
                f"kin_range is very large ({kin_range}). "
                "This may severely impact performance."
            )

        # Get indices of most similar relatives, excluding self
        relatives = np.argsort(-ctx.org_similarity[ctx.org_id, :])
        relatives = relatives[:kin_range]
        relatives = relatives[relatives != ctx.org_id]

        # Early exit if no relatives or zero organism fitness
        if len(relatives) == 0 or ctx.org_fitness[ctx.org_id] == 0:
            return np.zeros(ctx.population.M)

        # Compute gene-specific term: (gene_contribution / org_fitness - 1/M)
        gene_contribution = ctx.population[ctx.org_id, :] * ctx.gene_fitness
        gene_term = (gene_contribution / ctx.org_fitness[ctx.org_id]) - (
            1 / ctx.population.M
        )

        # Compute relative weights: similarity * fitness difference
        org_similarity = ctx.org_similarity[ctx.org_id, relatives]
        fitness_diff = ctx.org_fitness[ctx.org_id] - ctx.org_fitness[relatives]
        rel_weights = org_similarity * fitness_diff

        # Vectorized computation: outer product and sum over relatives
        delta_o_matrix = np.outer(gene_term, rel_weights)
        summed_delta_o = np.sum(delta_o_matrix, axis=1)

        # Final delta calculation
        delta_o = (
            # constant factor
            (-2 / ctx.population.N)
            # normalization by kin range
            * (1 / kin_range)
            # scale by initial range
            * (summed_delta_o / ctx.initial_org_fitness_range)
        )

        return delta_o
