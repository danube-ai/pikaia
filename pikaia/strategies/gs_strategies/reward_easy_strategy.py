import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class RewardEasyGeneStrategy(GeneStrategy):
    """
    A gene strategy that rewards features that are easy to achieve.

    .. warning::
        This strategy is experimental and its behavior may change in future
        versions.

    Ported from the tgeneticai CalSim ``"Inverse"`` sell strategy
    (``experiments/tgeneticai/calsim.py``).  The original sell delta was
    ``startValue * -exclusiveness / (1 - exclusiveness)`` — the exact same
    formula as ``Difficulty1`` but with a **negative sign**.  Easy features
    (low exclusiveness) earn positive value; hard features earn negative value.

    In pikaia this is the exact inverse of ``RewardHardGeneStrategy``: the
    difficulty signal is negated so that features with high average expression
    (easy to achieve) gain fitness.
    """

    def __init__(self, **kwargs):
        """Initialise the Reward Easy gene strategy.

        Args:
            **kwargs: Keyword options forwarded to :class:`GeneStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "RewardEasy"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the delta for a reward-easy gene.

        Easier features (high mean expression → low exclusiveness) receive a
        positive delta boost; harder features receive less.

        This is the exact inverse of :class:`RewardHardGeneStrategy`.

        Args:
            ctx (StrategyContext): Context object containing all required and
                optional fields.

        Returns:
            float: The computed delta value ``Delta_G(i,j)`` for the specified
                gene and organism.
        """
        mean_all = ctx.population.matrix.mean(axis=0)  # (M,)
        exclusiveness = 1.0 - mean_all
        difficulty = exclusiveness / (exclusiveness + 1e-8)

        term = (
            (-16 / ctx.population.N)
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
        """Diagonal D: ``D[j,j] = -(16/M) * difficulty_j``."""
        M = population.M
        mean_all = population.matrix.mean(axis=0)  # (M,)
        exclusiveness = 1.0 - mean_all
        difficulty = exclusiveness / (exclusiveness + 1e-8)
        D = np.diag(-(16.0 / M) * difficulty)
        return D, None
