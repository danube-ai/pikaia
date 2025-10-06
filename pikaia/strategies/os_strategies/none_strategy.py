import numpy as np

from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class NoneOrgStrategy(OrgStrategy):
    """
    An organism strategy that applies no evolutionary pressure.

    This strategy is a neutral placeholder that returns a delta vector of zeros,
    effectively making no change to the organism's fitness contribution. It can
    be used to disable organism-level selection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "None"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """
        Returns a delta vector of zeros, representing no change.

        This method ignores all input parameters and simply returns a zero vector
        of the correct shape, indicating no change in fitness contribution from
        this organism.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            np.ndarray: A zero vector of shape `(m,)`.
        """
        return np.zeros(ctx.population.M)
