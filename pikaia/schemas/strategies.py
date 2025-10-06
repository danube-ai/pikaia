from enum import Enum


class GeneStrategyEnum(str, Enum):
    """
    Enum representing gene-level strategies in evolutionary simulations.

    Members:
        DOMINANT: Gene expresses dominance over others.
        SELFISH: Gene acts in its own interest.
        KIN_ALTRUISTIC: Gene favors kin altruism.
        ALTRUISTIC: Gene acts altruistically toward others.
        NONE: No specific strategy.
    """

    DOMINANT = "DOMINANT"
    SELFISH = "SELFISH"
    KIN_ALTRUISTIC = "KIN_ALTRUISTIC"
    ALTRUISTIC = "ALTRUISTIC"
    NONE = "NONE"


class OrgStrategyEnum(str, Enum):
    """
    Enum representing organism-level strategies in evolutionary simulations.

    Members:
        BALANCED: Organism balances selfish and altruistic behaviors.
        ALTRUISTIC: Organism acts altruistically toward others.
        KIN_SELFISH: Organism is selfish toward non-kin, altruistic toward kin.
        SELFISH: Organism acts selfishly.
        NONE: No specific strategy.
    """

    BALANCED = "BALANCED"
    ALTRUISTIC = "ALTRUISTIC"
    KIN_SELFISH = "KIN_SELFISH"
    SELFISH = "SELFISH"
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
