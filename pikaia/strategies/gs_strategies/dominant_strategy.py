from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class DominantGeneStrategy(GeneStrategy):
    """
    A gene strategy that promotes dominant genes.

    This strategy increases the fitness of genes that are highly expressed
    (dominant), reinforcing their prevalence in the population. The delta is
    proportional to the square of the gene's fitness and its expression level.
    This implementation follows the logic from the original `alg.py`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Dominant"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a dominant gene.

        The formula reinforces the fitness of the gene based on its current
        fitness and expression.

        Args:
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            float:
                The computed delta value `Delta_G(i,j)` for the specified gene
                and organism.
        """
        return float(
            # constant factor and normalization by population size
            (4 / ctx.population.N)
            # fitness of current gene squared
            * ctx.gene_fitness[ctx.gene_id] ** 2
            # gene variant fitness minus 0.5
            * (ctx.population[ctx.org_id, ctx.gene_id] - 0.5)
        )
