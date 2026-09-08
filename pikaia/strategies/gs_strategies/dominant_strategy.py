"""Implement the dominant-gene strategy and its supported formulations."""

from typing import ClassVar

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import StrategyFormulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class DominantGeneStrategy(GeneStrategy):
    """A gene strategy that promotes dominant genes.

    This strategy increases the fitness of genes that are highly expressed
    (dominant), reinforcing their prevalence in the population. ``ORIGINAL``
    makes the direct delta proportional to the square of the focal gene's
    fitness. ``MATH_PAPER`` reproduces the historical branch's revised delta,
    which is linear in the focal gene's fitness.
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
        """Return the formulation-specific exact dominant-gene D matrix.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Unused.
            initial_org_fitness_range: Unused.
            y: Unused.

        Returns:
            ``(D, None)``. For ``ORIGINAL``, ``D`` is diagonal with
            ``D[j, j] = 4 * (x_bar_j - 0.5)``. For ``MATH_PAPER``, every entry
            in row ``j`` equals ``x_bar_j - 0.5``. Because gene fitness is
            normalised to sum to one, the row-constant matrix exactly encodes
            the formulation's signal ``gamma_j * (x_bar_j - 0.5)``.

        """
        mean_centered_expression = population.matrix.mean(axis=0) - 0.5
        if self.formulation is StrategyFormulation.MATH_PAPER:
            D = np.broadcast_to(
                mean_centered_expression[:, np.newaxis],
                (population.M, population.M),
            ).copy()
            return D, None

        D = np.diag(4.0 * mean_centered_expression)
        return D, None

    @property
    def supports_d_matrix(self) -> bool:
        """Indicate that both supported formulations have exact D kernels."""
        return True
