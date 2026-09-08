"""Implement the selfish organism strategy and its supported formulations."""

from typing import ClassVar

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import StrategyFormulation, StrategyNormalizations
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class SelfishOrgStrategy(OrgStrategy):
    """An organism strategy that promotes selfish behavior.

    This strategy models selfishness where an organism aims to increase its
    own fitness, potentially at the expense of others. The delta is calculated
    based on the fitness difference between the organism and its relatives,
    weighted by their similarity. This is identical to the `AltruisticOrgStrategy`
    but is kept for semantic clarity and future independent development.
    This implementation follows the logic from the original `alg.py`.
    """

    supported_formulations: ClassVar[frozenset[StrategyFormulation]] = frozenset(
        {StrategyFormulation.ORIGINAL, StrategyFormulation.MATH_PAPER}
    )

    def __init__(self, **kwargs):
        """Initialise the Selfish organism strategy.

        Keyword Args:
            kin_range (int): Maximum number of organisms to consider when
                computing the interaction term.  Defaults to ``N``
                (the full population size).
            **kwargs (object): Additional options forwarded to `OrgStrategy`
                and stored in ``self.options``.

        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "Selfish"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """Compute deltas for a selfish organism strategy.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            np.ndarray: A vector of computed delta values `Delta_O(i,j)` of shape `(m,)`.

        """
        # Determine kin range
        kin_range = self.options.get("kin_range", ctx.population.N)
        if self.formulation is StrategyFormulation.MATH_PAPER:
            kin_range = min(kin_range or ctx.population.N, ctx.population.N)

        # Get indices of most similar relatives, excluding self
        relatives = np.argsort(-ctx.org_similarity[ctx.org_id, :])
        relatives = relatives[:kin_range]
        relatives = relatives[relatives != ctx.org_id]

        # Early exit if no relatives or zero organism fitness
        if len(relatives) == 0 or ctx.org_fitness[ctx.org_id] == 0:
            return np.zeros(ctx.population.M)

        # Compute gene-specific term for the selected formulation.
        gene_contribution = ctx.population[ctx.org_id, :] * ctx.gene_fitness
        if self.formulation is StrategyFormulation.MATH_PAPER:
            if ctx.normalizations is None:
                raise ValueError("MATH_PAPER requires population normalizations.")
            gene_term = gene_contribution
            normalization = (
                ctx.normalizations.require_harmonic_fitness_mean_pairwise_difference()
            )
        else:
            gene_term = (gene_contribution / ctx.org_fitness[ctx.org_id]) - (
                1 / ctx.population.M
            )
            normalization = ctx.initial_org_fitness_range

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
            # scale by the formulation-specific population range
            * (summed_delta_o / normalization)
        )

        return delta_o

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
        normalizations: StrategyNormalizations | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Return the historical ``MATH_PAPER`` D matrix for selfish organisms.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Organism similarity matrix of shape ``(N, N)``.
            initial_org_fitness_range: Unused by ``MATH_PAPER``.
            y: Unused.
            normalizations: Population-derived normalisation values required by
                the ``MATH_PAPER`` formulation; ignored by ``ORIGINAL``.

        Returns:
            For ``MATH_PAPER``, returns ``(D, None)`` where ``D`` is an ``(M, M)`` matrix with
            ``D[j, k] = (-2 / (N * R)) * sum_i[x_ij * sum_l(s^o_il * (x_ik - x_lk))]``,
            summed over kin neighbours of each organism. ``ORIGINAL`` returns
            ``(None, None)`` because it has no D-matrix implementation.

        """
        if self.formulation is not StrategyFormulation.MATH_PAPER:
            return None, None
        if normalizations is None:
            raise ValueError("MATH_PAPER requires population normalizations.")

        X = population.matrix
        N = population.N
        R = normalizations.require_harmonic_fitness_mean_pairwise_difference()
        kin_range = min(self.options.get("kin_range") or N, N)

        D_acc = np.zeros((population.M, population.M))
        n_contributing = 0
        for i in range(N):
            sorted_idx = np.argsort(-org_similarity[i, :])
            selected_relatives = sorted_idx[:kin_range]
            relatives_i = selected_relatives[selected_relatives != i]
            denominator = kin_range
            if len(relatives_i) == 0:
                continue
            s_il = org_similarity[i, relatives_i]  # (n_rel,)
            # x_diff_lk[l, k] = X[i, k] - X[relatives_i[l], k]
            x_diff = X[i, np.newaxis, :] - X[relatives_i, :]  # (n_rel, M)
            # sum_l s_il * (x_ik - x_lk): (M,)
            sum_l = s_il @ x_diff
            # The historical formulation includes self in the kin-range
            # denominator even though self does not contribute to the sum.
            D_acc += np.outer(X[i, :], sum_l) / denominator
            n_contributing += 1

        if n_contributing == 0:
            return np.zeros((population.M, population.M)), None

        D = D_acc / N * (-2.0 / R)
        return D, None

    @property
    def requires_normalizations(self) -> bool:
        """Indicate that the strategy requires population normalisations.

        The calculation uses a shared normalisation value supplied by the
        model before strategy evaluation.
        """
        return True

    @property
    def supports_d_matrix(self) -> bool:
        """Support the exact historical kernel only for ``MATH_PAPER``."""
        return self.formulation is StrategyFormulation.MATH_PAPER
