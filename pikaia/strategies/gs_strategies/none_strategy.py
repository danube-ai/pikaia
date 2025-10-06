from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class NoneGeneStrategy(GeneStrategy):
    """
    A gene strategy that applies no evolutionary pressure.

    This strategy is a neutral placeholder that returns a delta value of 0,
    effectively making no change to the gene's fitness. It serves as a baseline
    or a way to disable gene-level selection.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def name(self) -> str:
        """The name of the strategy."""
        return "None"

    def __call__(self, ctx: StrategyContext) -> float:
        """
        Returns a delta of 0, representing no change.

        This method ignores all input parameters and simply returns 0.0,
        indicating that this strategy does not contribute to any change in
        gene fitness.

        Args:
            ctx (GeneStrategyContext): Context object containing all required and
                optional fields.

        Returns:
            float: A delta value of 0.0.
        """
        return 0.0
