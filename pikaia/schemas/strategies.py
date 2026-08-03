from enum import Enum


class GeneStrategyEnum(str, Enum):
    """Enum representing gene-level evolutionary strategies."""

    DOMINANT = "DOMINANT"
    """Gene expresses dominance over others."""

    SELFISH = "SELFISH"
    """Gene acts in its own interest."""

    KIN_ALTRUISTIC = "KIN_ALTRUISTIC"
    """Gene favors kin altruism."""

    ALTRUISTIC = "ALTRUISTIC"
    """Gene acts altruistically toward others."""

    REWARD_HARD = "REWARD_HARD"
    """Rewards features that are hard to achieve.

    Port of tgeneticai CalSim ``"Difficulty1"``.  Features with low average
    expression receive a positive delta boost.
    """

    REWARD_EASY = "REWARD_EASY"
    """Rewards features that are easy to achieve.

    Port of tgeneticai CalSim ``"Inverse"``.  The exact inverse of
    ``REWARD_HARD`` — features with high average expression receive a positive
    delta boost.
    """

    VALUATION_BLEND = "VALUATION_BLEND"
    """Blends between REWARD_HARD and REWARD_EASY via a ``preference`` parameter.

    Port of tgeneticai CalSim ``"Mixed"``.  ``preference=0.0`` → pure
    REWARD_EASY, ``preference=0.5`` → balanced, ``preference=1.0`` → pure
    REWARD_HARD.
    """

    ENTROPY_MAX = "ENTROPY_MAX"
    """Information-theoretic supervised strategy.

    Rewards features with high mutual information with the target weighted by
    differential entropy.  Converges within 5 iterations.
    """

    ORTHO_GENE = "ORTHO_GENE"
    """Orthogonality-based strategy.

    Promotes features that are minimally correlated with all other features.
    """

    PARTIAL_CORR = "PARTIAL_CORR"
    """Partial-correlation supervised strategy.

    Rewards features whose relationship with the target survives controlling for
    all other features.
    """

    REDUNDANCY_PENALTY = "REDUNDANCY_PENALTY"
    """Redundancy-penalty strategy.

    Suppresses features that are highly correlated with their peers.
    """

    NONE = "NONE"
    """No specific strategy — zero contribution."""


class OrgStrategyEnum(str, Enum):
    """Enum representing organism-level evolutionary strategies."""

    BALANCED = "BALANCED"
    """Organism balances gene contributions to promote uniform fitness."""

    ALTRUISTIC = "ALTRUISTIC"
    """Organism acts altruistically toward similar organisms."""

    KIN_SELFISH = "KIN_SELFISH"
    """Organism is selfish toward non-kin, altruistic toward kin."""

    SELFISH = "SELFISH"
    """Organism acts selfishly, promoting its own gene expression."""

    SELL = "SELL"
    """CalSim sell-phase signal.

    Drains value from commonly-expressed genes proportional to their difficulty
    odds.  Pair with ``OrgStrategyEnum.BUY`` to reproduce a full CalSim round.
    """

    BUY = "BUY"
    """CalSim buy-phase signal.

    Redistributes sell capital from high-performing organisms to genes they
    lack.  Pair with ``OrgStrategyEnum.SELL`` to reproduce a full CalSim round.
    """

    NONE = "NONE"
    """No specific strategy — zero contribution."""


class MixStrategyEnum(str, Enum):
    """Enum representing strategy mixing modes."""

    NONE = "NONE"
    """No mixed strategy applied."""

    FIXED = "FIXED"
    """Fixed mixing coefficients — proportions do not adapt over iterations."""

    SELF_CONSISTENT = "SELF_CONSISTENT"
    """Self-consistent mixing — coefficients adapt each iteration based on delta magnitude."""
