"""Implement the balanced organism strategy."""

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import StrategyFormulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class BalancedOrgStrategy(OrgStrategy):
    """An organism strategy that promotes balanced gene contributions.

    This strategy adjusts gene fitness to favor organisms where the
    contribution of each gene to the organism's total fitness is balanced.
    It penalizes genes that contribute disproportionately (more or less) than
    the average. This implementation follows the logic from the original
    `alg.py`.
    """

    def __init__(self, **kwargs):
        """Initialise the Balanced organism strategy.

        Args:
            **kwargs (object): Keyword options forwarded to `OrgStrategy` and
                stored in ``self.options``.

        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Balanced"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """Compute deltas for a balanced organism strategy.

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
                ``(D, None)``. Returns the D-Matrix for Org-Balanced according 
                to the formula in arxiv 2605.26685v2
    
            """
            
            diagonal = -2*population.matrix.mean(axis=0)
            addvec = 2/population.M*population.matrix.mean(axis=0)
            D = np.diag(diagonal)
            
            D += np.matlib.repmat(addvec, population.M, 1)
    
            
            return D, None

    @property
    def is_bilinear(self) -> bool:
        return False
