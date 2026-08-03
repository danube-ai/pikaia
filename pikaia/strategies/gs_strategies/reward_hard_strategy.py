import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class RewardHardGeneStrategy(GeneStrategy):
    """
    A gene strategy that rewards features that are hard to achieve.

    !!! warning
        This strategy is experimental and its behavior may change in future
        versions.

    Ported from the tgeneticai CalSim ``"Difficulty1"`` sell strategy
    (``experiments/tgeneticai/calsim.py``).  In CalSim, the sell delta for an
    exercise was ``startValue * exclusiveness / (1 - exclusiveness)``, where
    ``exclusiveness`` is the fraction of probands who did NOT solve the
    exercise.  The higher the exclusiveness (i.e. the harder the feature),
    the larger the delta.

    In pikaia this is expressed as a per-gene difficulty signal
    ``difficulty_j = (1 - mean_j) / (1 - mean_j + eps)`` that scales the
    replicator-style delta, so that features with low average expression
    (hard to achieve) gain more fitness.

    """

    def __init__(self, **kwargs):
        """Initialise the Reward Hard gene strategy.

        Args:
            **kwargs: Keyword options forwarded to `GeneStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "RewardHard"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a reward-hard gene.

        Harder features (low mean expression → high exclusiveness) receive a
        positive delta boost; easier features receive less.

        Args:
            ctx (StrategyContext): Context object containing all required and
                optional fields.

        Returns:
            float: The computed delta value ``Delta_G(i,j)`` for the specified
                gene and organism.
        """
        mean_all = ctx.population.matrix.mean(axis=0)  # (M,)
        exclusiveness = 1.0 - mean_all
        odds = exclusiveness / (1.0 - exclusiveness + 1e-8)
        difficulty = odds / (odds.max() + 1e-8)

        term = (
            (16 / ctx.population.N)
            * difficulty[ctx.gene_id]
            * ctx.gene_fitness[ctx.gene_id]
            * (ctx.population[ctx.org_id, ctx.gene_id] - 0.5)
        )
        return float(term)

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Diagonal D matrix proportional to gene difficulty.

        Args:
            population: Population providing the ``(N, M)`` data matrix.
            gene_similarity: Unused.
            org_similarity: Unused.
            initial_org_fitness_range: Unused.
            y: Unused.

        Returns:
            Tuple ``(D, None)`` where ``D`` is a diagonal ``(M, M)`` matrix
            with ``D[j, j] = (16/M) * difficulty_j`` and
            ``difficulty_j = excl_j / (1 - excl_j + eps)`` normalised by
            the maximum odds across genes.
        """
        M = population.M
        mean_all = population.matrix.mean(axis=0)  # (M,)
        exclusiveness = 1.0 - mean_all
        odds = exclusiveness / (1.0 - exclusiveness + 1e-8)
        difficulty = odds / (odds.max() + 1e-8)
        D = np.diag((16.0 / M) * difficulty)
        return D, None
