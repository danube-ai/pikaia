"""Implement the dominant-gene strategy and its supported formulations."""

from typing import ClassVar

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import StrategyFormulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class DominantGeneStrategy(GeneStrategy):
    """A gene strategy that promotes dominant genes.

    This strategy increases the fitness of genes that are highly expressed
    (dominant), reinforcing their prevalence in the population. The delta is
    proportional to the square of the gene's fitness and its expression level.
    This implementation follows the logic from the original `alg.py`.
    """

    supported_formulations: ClassVar[frozenset[StrategyFormulation]] = frozenset(
        {StrategyFormulation.ORIGINAL, StrategyFormulation.MATH_PAPER}
    )

    def __init__(self, **kwargs):
        """Initialise the Dominant gene strategy.

        Args:
            **kwargs (object): Keyword options forwarded to `GeneStrategy` and
                stored in ``self.options``.

        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Dominant"

    def __call__(self, ctx: StrategyContext) -> float:
        """Compute the delta for a dominant gene.

        The formula reinforces the fitness of the gene based on its current
        fitness and expression.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            float: The computed delta value `Delta_G(i,j)` for the specified gene and organism.

        """
        if self.formulation is StrategyFormulation.MATH_PAPER:
            return float(
                (1 / ctx.population.N)
                * ctx.gene_fitness[ctx.gene_id]
                * (ctx.population[ctx.org_id, ctx.gene_id] - 0.5)
            )

        return float(
            # constant factor and normalization by population size
            (4 / ctx.population.N)
            # fitness of current gene squared
            * ctx.gene_fitness[ctx.gene_id] ** 2
            # gene variant fitness minus 0.5
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
        """Diagonal D matrix from population mean expression.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Unused.
            initial_org_fitness_range: Unused.
            y: Unused.

        Returns:
            For ``ORIGINAL``, ``(D, None)`` where ``D`` is diagonal with
            ``D[j, j] = 4 * (x_bar_j - 0.5)``.  ``MATH_PAPER`` returns
            ``(None, None)`` because its linear-in-fitness formula cannot be
            represented by the static D-matrix contract.

        """
        mean_centered_expression = population.matrix.mean(axis=0) - 0.5
        if self.formulation is StrategyFormulation.MATH_PAPER:
            # The revised delta is linear in gamma.  The D-matrix engine only
            # supports population-static d vectors and bilinear D matrices, so
            # this formulation intentionally uses the iterative path.
            return None, None

        D = np.diag(4.0 * mean_centered_expression)
        return D, None

    @property
    def supports_d_matrix(self) -> bool:
        """Support D-matrix execution only for the original quadratic equation."""
        return self.formulation is StrategyFormulation.ORIGINAL
