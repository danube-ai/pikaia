import numpy as np

from pikaia.config.logger import logger
from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class ValuationBlendGeneStrategy(GeneStrategy):
    """
    A gene strategy that blends between reward-hard and reward-easy.

    .. warning::
        This strategy is experimental and its behavior may change in future
        versions.

    Ported from the tgeneticai CalSim ``"Mixed"`` sell strategy
    (``experiments/tgeneticai/calsim.py``).  CalSim blended three strategies
    (difficulty-based, uniform, and inverse) using ``mixFactor`` and
    ``mixFactor2`` parameters.

    Pikaia simplifies this to a single ``preference`` parameter in ``[0, 1]``:

    - ``preference = 0.0`` → pure :class:`RewardEasyGeneStrategy`
    - ``preference = 0.5`` → balanced (equal weight, zero signal)
    - ``preference = 1.0`` → pure :class:`RewardHardGeneStrategy`
    """

    def __init__(self, preference: float = 0.5, **kwargs):
        """Initialise the Valuation Blend gene strategy.

        Args:
            preference: Blend factor in ``[0, 1]``.  ``0.0`` favours easy
                features, ``1.0`` favours hard features, ``0.5`` is balanced.
                Defaults to ``0.5`` (balanced).
            **kwargs: Keyword options forwarded to :class:`GeneStrategy` and
                stored in ``self.options``.
        """
        super().__init__(**kwargs)
        self.options["preference"] = preference
        if preference < 0.0 or preference > 1.0:
            logger.warning(
                f"preference={preference} is outside the recommended range "
                "[0, 1].  Values < 0 favour easy features more strongly; "
                "values > 1 favour hard features more strongly."
            )

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "ValuationBlend"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Computes the blended delta for a valuation-blend gene.

        The delta interpolates between reward-hard and reward-easy based on the
        ``preference`` parameter.

        Args:
            ctx (StrategyContext): Context object containing all required and
                optional fields.

        Returns:
            float: The computed delta value ``Delta_G(i,j)`` for the specified
                gene and organism.
        """
        preference = self.options.get("preference", 0.5)
        mean_all = ctx.population.matrix.mean(axis=0)  # (M,)
        exclusiveness = 1.0 - mean_all
        difficulty = exclusiveness / (exclusiveness + 1e-8)

        # preference=1.0 → reward hard, preference=0.0 → reward easy
        sign = 2.0 * preference - 1.0  # -1 (easy) to +1 (hard)

        term = (
            (16 / ctx.population.N)
            * sign
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
        """Diagonal D: ``D[j,j] = (16/M) * sign * difficulty_j``.

        ``sign = 2*preference - 1`` ranges from ``-1`` (easy) to ``+1``
        (hard).
        """
        M = population.M
        preference = self.options.get("preference", 0.5)
        mean_all = population.matrix.mean(axis=0)  # (M,)
        exclusiveness = 1.0 - mean_all
        difficulty = exclusiveness / (exclusiveness + 1e-8)
        sign = 2.0 * preference - 1.0
        D = np.diag((16.0 / M) * sign * difficulty)
        return D, None
