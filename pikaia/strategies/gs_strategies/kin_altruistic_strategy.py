import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class KinAltruisticGeneStrategy(GeneStrategy):
    """
    A gene strategy that promotes altruism towards kin (similar genes).

    .. warning::
        This strategy is experimental and its behavior may change in future
        versions.

    This strategy increases a gene's fitness by helping other, similar genes,
    even at a potential cost to itself. The altruistic effect is inversely
    proportional to the similarity, meaning it helps less similar genes more.
    This implementation follows the logic from the original `alg.py`.
    """

    def __init__(self, **kwargs):
        """Initialise the KinAltruistic gene strategy.

        Keyword Args:
            kin_range (int): Number of most-similar genes to consider as kin
                when computing the interaction term.  Defaults to ``M``
                (the full feature dimension).
            **kwargs: Additional options forwarded to :class:`GeneStrategy`
                and stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "KinAltruistic"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a kin-altruistic gene.

        The formula considers the interaction with other genes, weighted by a
        factor of `(0.5 - similarity)`.

        Args:
            ctx (StrategyContext):
                Context object containing all required and optional fields.

        Returns:
            float:
                The computed delta value `Delta_G(i,j)` for the specified gene
                and organism.
        """
        # Determine kin range
        kin_range = self.options.get("kin_range", ctx.population.M)

        # Get indices of most similar genes, excluding self
        indices = np.argsort(-ctx.gene_similarity[ctx.gene_id, :])
        indices = indices[:kin_range]
        indices = indices[indices != ctx.gene_id]

        # Early exit if no kin
        if len(indices) == 0:
            return 0.0

        # Vectorized computation for kin genes
        # 16 / N * (0.5 - similarity) * fitness_self * (pop_self - 0.5) *
        # fitness_others * (pop_others - pop_self)
        return float(
            np.sum(
                # constant factor and normalization by population size
                (16 / ctx.population.N)
                # kin altruism weight: 0.5 - similarity to other genes
                * (0.5 - ctx.gene_similarity[ctx.gene_id, indices])
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
            # normalization by number of genes
            / ctx.population.M
        )

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Full (M, M) D matrix with (0.5 - s^g_jk) similarity weight.

        Respects the ``kin_range`` option: per gene j, only the top
        ``kin_range`` most similar genes k contribute (self excluded).

        D[j,k] = (16/M) * (0.5 - gene_sim_masked[j,k])
                 * mean_i[(x_ij - 0.5) * (x_ik - x_ij)]
        """
        X = population.matrix  # (N, M)
        M = population.M
        kin_range = self.options.get("kin_range", M)

        # Build masked similarity: only top kin_range similar genes per row
        gene_sim_masked = np.zeros_like(gene_similarity)
        for j in range(M):
            sorted_k = np.argsort(-gene_similarity[j, :])
            top_k = sorted_k[:kin_range]
            top_k = top_k[top_k != j]  # exclude self
            gene_sim_masked[j, top_k] = gene_similarity[j, top_k]

        X_centered = X - 0.5  # (N, M)
        X_diff = X[:, np.newaxis, :] - X[:, :, np.newaxis]  # (N, M, M)
        inner = np.mean(X_centered[:, :, np.newaxis] * X_diff, axis=0)  # (M, M)
        D = (16.0 / M) * (0.5 - gene_sim_masked) * inner
        np.fill_diagonal(D, 0.0)
        return D, None
