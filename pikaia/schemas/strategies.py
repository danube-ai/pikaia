from enum import Enum


class GeneStrategyEnum(str, Enum):
    """
    Enum representing gene-level strategies in evolutionary simulations.

    Members:
        DOMINANT: Gene expresses dominance over others.
        SELFISH: Gene acts in its own interest.
        KIN_ALTRUISTIC: Gene favors kin altruism.
        ALTRUISTIC: Gene acts altruistically toward others.
        REWARD_HARD: Rewards features that are hard to achieve (port of
            tgeneticai CalSim ``"Difficulty1"``).  Features with low average
            expression receive a positive delta boost.
        REWARD_EASY: Rewards features that are easy to achieve (port of
            tgeneticai CalSim ``"Inverse"``).  The exact inverse of
            REWARD_HARD — features with high average expression receive a
            positive delta boost.
        VALUATION_BLEND: Blends between REWARD_HARD and REWARD_EASY via a
            ``preference`` parameter in ``[0, 1]`` (port of tgeneticai CalSim
            ``"Mixed"``).  ``0.0`` → pure REWARD_EASY, ``0.5`` → balanced,
            ``1.0`` → pure REWARD_HARD.
        SELL: CalSim sell-phase signal — drains value from commonly-expressed
            genes proportional to their difficulty odds.  Pair with
            ``OrgStrategyEnum.BUY`` to reproduce a full CalSim round.
        NONE: No specific strategy.
    """

    DOMINANT = "DOMINANT"
    SELFISH = "SELFISH"
    KIN_ALTRUISTIC = "KIN_ALTRUISTIC"
    ALTRUISTIC = "ALTRUISTIC"
    REWARD_HARD = "REWARD_HARD"
    REWARD_EASY = "REWARD_EASY"
    VALUATION_BLEND = "VALUATION_BLEND"
    SELL = "SELL"
    NONE = "NONE"


class OrgStrategyEnum(str, Enum):
    """
    Enum representing organism-level strategies in evolutionary simulations.

    Members:
        BALANCED: Organism balances selfish and altruistic behaviors.
        ALTRUISTIC: Organism acts altruistically toward others.
        KIN_SELFISH: Organism is selfish toward non-kin, altruistic toward kin.
        SELFISH: Organism acts selfishly.
        BUY: CalSim buy-phase signal — redistributes sell capital from
            high-performing organisms to genes they lack.  Pair with
            ``GeneStrategyEnum.SELL`` to reproduce a full CalSim round.
        NONE: No specific strategy.
    """

    BALANCED = "BALANCED"
    ALTRUISTIC = "ALTRUISTIC"
    KIN_SELFISH = "KIN_SELFISH"
    SELFISH = "SELFISH"
    BUY = "BUY"
    NONE = "NONE"


class MixStrategyEnum(str, Enum):
    """
    Enum representing mixed strategy types in evolutionary simulations.

    Members:
        NONE: No mixed strategy applied.
        FIXED: Fixed mixed strategy.
        SELF_CONSISTENT: Self-consistent mixed strategy.
    """

    NONE = "NONE"
    FIXED = "FIXED"
    SELF_CONSISTENT = "SELF_CONSISTENT"
