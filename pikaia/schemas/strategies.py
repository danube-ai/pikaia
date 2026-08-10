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

    SELL_HARD = "SELL_HARD"
    """CalSim Difficulty1 ("fair") sell signal.

    Hard genes (low mean expression) lose more value per unit of performance.
    Pair with ``OrgStrategyEnum.BUY_HARD``.
    """

    SELL_UNIFORM = "SELL_UNIFORM"
    """CalSim Difficulty2 ("inclusive") sell signal.

    All genes lose value at the same rate, independent of difficulty.
    Pair with ``OrgStrategyEnum.BUY_UNIFORM``.
    """

    SELL_EASY = "SELL_EASY"
    """CalSim Inverse sell signal.

    Easy genes (high mean expression) lose more value — the mirror of
    ``SELL_HARD``.  Pair with ``OrgStrategyEnum.BUY_EASY``.
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

    VARIANCE = "VARIANCE"
    """Rewards genes with high cross-organism dispersion (column std)."""

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

    BUY_HARD = "BUY_HARD"
    """CalSim Difficulty1 ("fair") buy-phase signal.

    Redistributes hard-gene sell capital to easy genes the organism failed.
    Pair with ``GeneStrategyEnum.SELL_HARD``.
    """

    BUY_UNIFORM = "BUY_UNIFORM"
    """CalSim Difficulty2 ("inclusive") buy-phase signal.

    Redistributes uniform sell capital to hard genes the organism failed.
    Pair with ``GeneStrategyEnum.SELL_UNIFORM``.
    """

    BUY_EASY = "BUY_EASY"
    """CalSim Inverse buy-phase signal.

    Mirror of ``BUY_HARD`` — redistributes with inverted sign.
    Pair with ``GeneStrategyEnum.SELL_EASY``.
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
