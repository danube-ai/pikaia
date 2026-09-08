"""Implement the no-op organism strategy."""

from typing import ClassVar

import numpy as np

from pikaia.schemas.strategies import StrategyFormulation
from pikaia.strategies.base_strategies import OrgStrategy, StrategyContext


class NoneOrgStrategy(OrgStrategy):
    """An organism strategy that applies no evolutionary pressure.

    This strategy is a neutral placeholder that returns a delta vector of zeros,
    effectively making no change to the organism's fitness contribution. It can
    be used to disable organism-level selection. Because zero is independent of
    the selected equations, the strategy supports both ``ORIGINAL`` and
    ``MATH_PAPER``.
    """

    supported_formulations: ClassVar[frozenset[StrategyFormulation]] = frozenset(
        {StrategyFormulation.ORIGINAL, StrategyFormulation.MATH_PAPER}
    )

    def __init__(self, **kwargs):
        """Initialise the None (no-op) organism strategy.

        Args:
            **kwargs: Keyword options forwarded to `OrgStrategy` and
                stored in ``self.options``.

        """
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "None"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        """Return a delta vector of zeros, representing no change.

        This method ignores all input parameters and simply returns a zero vector
        of the correct shape, indicating no change in fitness contribution from
        this organism.

        Args:
            ctx (StrategyContext): Context object containing all required and optional fields.

        Returns:
            np.ndarray: A zero vector of shape `(m,)`.

        """
        return np.zeros(ctx.population.M)

    @property
    def supports_d_matrix(self) -> bool:
        """Allow D-matrix runs because this no-op contributes exact zero."""
        return True
