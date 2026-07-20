import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class SelfishOrgStrategy(OrgStrategy):
    """
    An organism strategy that promotes selfish behavior.

    This strategy models selfishness where an organism aims to increase its
    own fitness, potentially at the expense of others. The delta is calculated
    based on the fitness difference between the organism and its relatives,
    weighted by their similarity. This is identical to the `AltruisticOrgStrategy`
    but is kept for semantic clarity and future independent development.
    This implementation follows the logic from the original `alg.py`.
    """

    def __init__(self, **kwargs):
        """Initialise the Selfish organism strategy.

        Keyword Args:
            kin_range (int): Maximum number of organisms to consider when
                computing the interaction term.  Defaults to ``N``
                (the full population size).
            **kwargs: Additional options forwarded to :class:`OrgStrategy`
                and stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Selfish"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """
        Computes deltas for a selfish organism strategy.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            np.ndarray: A vector of computed delta values `Delta_O(i,j)` of shape `(m,)`.
        """
        # Determine kin range
        kin_range = self.options.get("kin_range", ctx.population.N)

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
            # constant factor (negative for selfish)
            (-2 / ctx.population.N)
            # normalization by kin range
            * (1 / kin_range)
            # scale by initial range
            * (summed_delta_o / ctx.initial_org_fitness_range)
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
        """D[j,k] = (-2 / (n_rel * R)) * mean_i[sum_l s^o_il * x_ij * (x_ik - x_lk)].

        Step 1: 1/M centering dropped (j-independent constant).
        Step 2: outer w_i multiplier added to cancel 1/w_i denominator.
        """
        X = population.matrix  # (N, M)
        N = population.N
        R = initial_org_fitness_range
        kin_range = self.options.get("kin_range", N)

        D_acc = np.zeros((population.M, population.M))
        n_contributing = 0
        for i in range(N):
            sorted_idx = np.argsort(-org_similarity[i, :])
            relatives_i = sorted_idx[sorted_idx != i][:kin_range]
            if len(relatives_i) == 0:
                continue
            n_rel = len(relatives_i)
            s_il = org_similarity[i, relatives_i]  # (n_rel,)
            # x_diff_lk[l, k] = X[i, k] - X[relatives_i[l], k]
            x_diff = X[i, np.newaxis, :] - X[relatives_i, :]  # (n_rel, M)
            # sum_l s_il * (x_ik - x_lk): (M,)
            sum_l = s_il @ x_diff
            D_acc += np.outer(X[i, :], sum_l) / n_rel
            n_contributing += 1

        if n_contributing == 0:
            return np.zeros((population.M, population.M)), None

        D = D_acc / N * (-2.0 / R)
        return D, None
