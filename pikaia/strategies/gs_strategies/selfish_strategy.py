import numpy as np

from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class SelfishGeneStrategy(GeneStrategy):
    """
    A gene strategy that promotes selfish behavior.

    .. warning::
        This strategy is experimental and its behavior may change in future
        versions.

    This strategy models selfish behavior where a gene's fitness is increased
    at the expense of other, dissimilar genes within the same organism. The
    effect is proportional to the similarity, meaning it acts more selfishly
    against more similar genes. This implementation follows the logic from the
    original `alg.py`.


    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Selfish"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a selfish gene.

        The formula calculates a negative delta contribution, effectively
        penalizing other genes to benefit the current one.

        Args:
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            float:
                The computed delta value `Delta_G(i,j)` for the specified gene
                and organism.
        """
        # Get all gene indices except the current gene
        indices = np.arange(ctx.population.M) != ctx.gene_id

        # Vectorized computation for all genes except self
        # -16 / N * similarity * fitness_self * (pop_self - 0.5) * fitness_others *
        # (pop_others - pop_self)
        return float(
            np.sum(
                # constant factor and normalization by population size
                (-16 / ctx.population.N)
                # similarity to other genes
                * ctx.gene_similarity[ctx.gene_id, indices]
                # fitness of current gene
                * ctx.gene_fitness[ctx.gene_id]
                # pop value of current gene minus 0.5
                * (ctx.population[ctx.org_id, ctx.gene_id] - 0.5)
                # fitness of other genes
                * ctx.gene_fitness[indices]
                # gene variant fitness of other genes
                # minus gene variant fitness of current gene
                * (
                    ctx.population[ctx.org_id, indices]
                    - ctx.population[ctx.org_id, ctx.gene_id]
                )
            )
            / ctx.population.M
        )
