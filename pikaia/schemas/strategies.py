"""Declare validated enums and configuration models for evolutionary strategies."""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class StrategyFormulation(str, Enum):
    """Mathematical formulation used by formulation-aware strategies.

    ``LEGACY`` preserves the original Python-package equations and remains the
    default. ``STANDARD`` names the revised, current equations previously
    exposed as ``MATH_PAPER``. The old member names remain aliases for source
    compatibility, while string inputs using their old serialized values are
    accepted by :meth:`_missing_` during the deprecation period.
    """

    LEGACY = "LEGACY"
    STANDARD = "STANDARD"

    # Deprecated source-compatible aliases. New code must use LEGACY/STANDARD.
    ORIGINAL = LEGACY
    MATH_PAPER = STANDARD

    @classmethod
    def _missing_(cls, value: object) -> "StrategyFormulation | None":
        """Accept serialized pre-0.4.2 values during the migration window."""
        legacy_values = {"ORIGINAL": cls.LEGACY, "MATH_PAPER": cls.STANDARD}
        return legacy_values.get(value) if isinstance(value, str) else None


class StrategyFormulationConfig(BaseModel):
    """Validated configuration shared by formulation-aware strategies."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    formulation: StrategyFormulation = StrategyFormulation.LEGACY


class KinRangeConfig(BaseModel):
    """Validate an optional positive kin-range request.

    The model applies the population-dependent upper bound separately because
    the number of organisms is unavailable while a strategy is constructed.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    kin_range: int | None = Field(default=None, ge=1)


class StrategyNormalizations(BaseModel):
    """Population-derived normalization values for STANDARD strategies."""

    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    gene_mean_pairwise_difference: float | None = Field(default=None, ge=0)
    harmonic_fitness_mean_pairwise_difference: float | None = Field(default=None, ge=0)

    def require_gene_mean_pairwise_difference(self) -> float:
        """Return the gene normalization or explain why it is unusable."""
        value = self.gene_mean_pairwise_difference
        if value is None or value == 0:
            raise ValueError(
                "STANDARD requires a positive gene_mean_pairwise_difference. "
                "Use a population with at least two distinct gene-column means."
            )
        return value

    def require_harmonic_fitness_mean_pairwise_difference(self) -> float:
        """Return the organism normalization or explain why it is unusable."""
        value = self.harmonic_fitness_mean_pairwise_difference
        if value is None or value == 0:
            raise ValueError(
                "STANDARD requires a positive "
                "harmonic_fitness_mean_pairwise_difference. Use a population "
                "with at least two distinct harmonic organism fitness values."
            )
        return value


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
    """Trading sell signal weighted by gene difficulty.

    Hard genes (low mean expression) lose more value per unit of performance.
    Pair with ``OrgStrategyEnum.BUY_HARD``.
    """

    SELL_UNIFORM = "SELL_UNIFORM"
    """Trading sell signal applied uniformly to all genes.

    All genes lose value at the same rate, independent of difficulty.
    Pair with ``OrgStrategyEnum.BUY_UNIFORM``.
    """

    SELL_EASY = "SELL_EASY"
    """Trading sell signal weighted by gene ease — the inverse of ``SELL_HARD``.

    Easy genes (high mean expression) lose more value.
    Pair with ``OrgStrategyEnum.BUY_EASY``.
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
    """Trading buy-phase paired with ``SELL_HARD``.

    Redistributes hard-gene sell capital to easy genes the organism failed.
    Pair with ``GeneStrategyEnum.SELL_HARD``.
    """

    BUY_UNIFORM = "BUY_UNIFORM"
    """Trading buy-phase paired with ``SELL_UNIFORM``.

    Redistributes uniform sell capital to hard genes the organism failed.
    Pair with ``GeneStrategyEnum.SELL_UNIFORM``.
    """

    BUY_EASY = "BUY_EASY"
    """Trading buy-phase paired with ``SELL_EASY`` — mirror of ``BUY_HARD``.

    Redistributes capital with inverted sign relative to ``BUY_HARD``.
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
