from __future__ import annotations

from pikaia.schemas.strategies import (
    GeneStrategyEnum,
    MixStrategyEnum,
    OrgStrategyEnum,
)
from pikaia.strategies.base_strategies import GeneStrategy, MixStrategy, OrgStrategy
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import (
    KinAltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy
from pikaia.strategies.mix_strategies.self_consistent_strategy import (
    SelfConsistentMixStrategy,
)
from pikaia.strategies.os_strategies.altruistic_strategy import AltruisticOrgStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.kin_selfish_strategy import KinSelfishOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy


class GeneStrategyFactory:
    """
    Factory class for creating gene strategy instances.

    This factory provides a centralized way to instantiate gene strategy
    objects based on the `GeneStrategyEnum`. It maps the enum members to their
    corresponding strategy classes.
    """

    _strategies = {
        GeneStrategyEnum.DOMINANT: DominantGeneStrategy,
        GeneStrategyEnum.ALTRUISTIC: AltruisticGeneStrategy,
        GeneStrategyEnum.KIN_ALTRUISTIC: KinAltruisticGeneStrategy,
        GeneStrategyEnum.SELFISH: SelfishGeneStrategy,
        GeneStrategyEnum.NONE: NoneGeneStrategy,
    }

    @classmethod
    def get_strategy(cls, name: GeneStrategyEnum, *args, **kwargs) -> GeneStrategy:
        """
        Retrieves an instance of the requested gene strategy.

        Args:
            name (GeneStrategyEnum): The enum member representing the desired
                strategy.
            ``*args``: Positional arguments to pass to the strategy's constructor.
            ``**kwargs``: Keyword arguments to pass to the strategy's constructor.

        Returns:
            GeneStrategy: An instance of the corresponding gene strategy class.

        Raises:
            ValueError: If the requested strategy name is not found in the
                factory's registry.
        """
        strategy_cls = cls._strategies.get(name)
        if strategy_cls is None:
            raise ValueError(f"Strategy '{name}' not found.")
        return strategy_cls(*args, **kwargs)


class OrgStrategyFactory:
    """
    Factory class for creating organism strategy instances.

    This factory provides a centralized way to instantiate organism strategy
    objects based on the `OrgStrategyEnum`. It maps the enum members to their
    corresponding strategy classes.
    """

    _strategies = {
        OrgStrategyEnum.BALANCED: BalancedOrgStrategy,
        OrgStrategyEnum.ALTRUISTIC: AltruisticOrgStrategy,
        OrgStrategyEnum.KIN_SELFISH: KinSelfishOrgStrategy,
        OrgStrategyEnum.SELFISH: SelfishOrgStrategy,
        OrgStrategyEnum.NONE: NoneOrgStrategy,
    }

    @classmethod
    def get_strategy(cls, name: OrgStrategyEnum, *args, **kwargs) -> OrgStrategy:
        """
        Retrieves an instance of the requested organism strategy.

        Args:
            name (OrgStrategyEnum): The enum member representing the desired
                strategy.
            ``*args``: Positional arguments to pass to the strategy's constructor.
            ``**kwargs``: Keyword arguments to pass to the strategy's constructor.

        Returns:
            OrgStrategy: An instance of the corresponding organism strategy class.

        Raises:
            ValueError: If the requested strategy name is not found in the
                factory's registry.
        """
        strategy_cls = cls._strategies.get(name)
        if strategy_cls is None:
            raise ValueError(f"Strategy '{name}' not found.")
        return strategy_cls(*args, **kwargs)


class MixStrategyFactory:
    """
    Factory class for creating mixing strategy instances.

    This factory provides a centralized way to instantiate mixing strategy
    objects based on the `MixinGeneStrategyEnum`.
    """

    _strategies = {
        MixStrategyEnum.FIXED: FixedMixStrategy,
        MixStrategyEnum.SELF_CONSISTENT: SelfConsistentMixStrategy,
    }

    @classmethod
    def get_strategy(cls, name: MixStrategyEnum, *args, **kwargs) -> MixStrategy:
        """
        Retrieves a singleton instance of the requested mixing strategy.

        Args:
            name (MixinGeneStrategyEnum): The enum member representing the desired
                strategy.
            ``*args``: Positional arguments to pass to the strategy's constructor.
            ``**kwargs``: Keyword arguments to pass to the strategy's constructor.

        Returns:
            MixStrategy: An instance of the corresponding mixing strategy class.

        Raises:
            ValueError: If the requested strategy name is not found.
        """
        strategy_cls = cls._strategies.get(name)
        if strategy_cls is None:
            raise ValueError(f"Strategy '{name}' not found.")
        return strategy_cls(*args, **kwargs)
