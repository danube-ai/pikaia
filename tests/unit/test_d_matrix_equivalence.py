"""Regression tests for exact D-matrix support in the LEGACY formulation."""

from collections.abc import Callable

import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.strategies.base_strategies import GeneStrategy, MixStrategy, OrgStrategy
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.entropy_max_strategy import EntropyMaxGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import (
    KinAltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.gs_strategies.orthogonality_strategy import OrthoGeneStrategy
from pikaia.strategies.gs_strategies.partial_corr_strategy import (
    PartialCorrGeneStrategy,
)
from pikaia.strategies.gs_strategies.redundancy_penalty_strategy import (
    RedundancyPenaltyGeneStrategy,
)
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.gs_strategies.sell_easy_strategy import SellEasyGeneStrategy
from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy
from pikaia.strategies.gs_strategies.sell_uniform_strategy import (
    SellUniformGeneStrategy,
)
from pikaia.strategies.gs_strategies.variance_strategy import VarianceGeneStrategy
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy
from pikaia.strategies.mix_strategies.self_consistent_strategy import (
    SelfConsistentMixStrategy,
)
from pikaia.strategies.os_strategies.altruistic_strategy import AltruisticOrgStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.kin_selfish_strategy import KinSelfishOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy


def _population() -> PikaiaPopulation:
    """Return the fixed non-degenerate population used for D-path regression."""
    return PikaiaPopulation(
        np.array(
            [
                [0.1, 0.6, 0.9],
                [0.8, 0.2, 0.4],
                [0.3, 0.9, 0.1],
                [0.7, 0.4, 0.6],
            ]
        )
    )


def _random_population(seed: int) -> PikaiaPopulation:
    """Return a deterministic non-degenerate population for broad equivalence checks."""
    return PikaiaPopulation(np.random.default_rng(seed).random((7, 4)))


def _fit(
    gene_strategies: list[GeneStrategy],
    org_strategies: list[OrgStrategy],
    *,
    use_d_matrix: bool,
    max_iter: int,
) -> PikaiaModel:
    """Fit one original-formulation model with the requested execution path."""
    model = PikaiaModel(
        population=_population(),
        gene_strategies=gene_strategies,
        org_strategies=org_strategies,
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=max_iter,
        use_d_matrix=use_d_matrix,
    )
    model.fit()
    return model


def _assert_fitness_histories_match(
    iterative: PikaiaModel,
    d_matrix: PikaiaModel,
    max_iter: int,
) -> None:
    """Assert that both public fitness histories agree through ``max_iter``.

    Organism fitness is deterministically derived from gene fitness as
    ``population.matrix @ gene_fitness``. Comparing it explicitly nevertheless
    protects the public output contract from storage, indexing, or execution-path
    regressions.

    Args:
        iterative: Model fitted through direct strategy evaluation.
        d_matrix: Model fitted through the reduced D-matrix path.
        max_iter: Final iteration included in the comparison.

    """
    history_slice = slice(0, max_iter + 1)
    np.testing.assert_allclose(
        iterative.gene_fitness_history[history_slice],
        d_matrix.gene_fitness_history[history_slice],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        iterative.organism_fitness_history[history_slice],
        d_matrix.organism_fitness_history[history_slice],
        rtol=1e-12,
        atol=1e-12,
    )


EXACT_GENE_STRATEGIES: list[tuple[str, Callable[[], GeneStrategy]]] = [
    ("dominant", DominantGeneStrategy),
    ("selfish", SelfishGeneStrategy),
    ("kin-altruistic-default", KinAltruisticGeneStrategy),
    ("altruistic", AltruisticGeneStrategy),
    ("sell-hard", SellHardGeneStrategy),
    ("sell-uniform", SellUniformGeneStrategy),
    ("sell-easy", SellEasyGeneStrategy),
    ("variance", VarianceGeneStrategy),
]


class CustomPassThroughMixStrategy(MixStrategy):
    """A custom mixer used to prove that D mode accepts only fixed mixing."""

    @property
    def name(self) -> str:
        """Return the test mixer's display name."""
        return "Custom pass-through"

    def __call__(
        self, delta: np.ndarray, mix_coeffs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Combine deltas while returning unchanged coefficients."""
        return np.einsum("ijk,k->ij", delta, mix_coeffs), mix_coeffs


class FixedMixSubclass(FixedMixStrategy):
    """A subclass used to prove that overridden fixed mixers are rejected."""


@pytest.mark.parametrize("max_iter", [1, 50, 100])
@pytest.mark.parametrize("_name,strategy_factory", EXACT_GENE_STRATEGIES)
def test_exact_original_gene_kernels_match_iterative_path(
    _name: str, strategy_factory: Callable[[], GeneStrategy], max_iter: int
) -> None:
    """Each retained original D kernel matches iteration through 1, 50, and 100 steps."""
    iterative = _fit(
        [strategy_factory()], [NoneOrgStrategy()], use_d_matrix=False, max_iter=max_iter
    )
    d_matrix = _fit(
        [strategy_factory()], [NoneOrgStrategy()], use_d_matrix=True, max_iter=max_iter
    )

    _assert_fitness_histories_match(iterative, d_matrix, max_iter)


@pytest.mark.parametrize("seed", [101, 102, 103])
@pytest.mark.parametrize("max_iter", [1, 50, 100])
@pytest.mark.parametrize("_name,strategy_factory", EXACT_GENE_STRATEGIES)
def test_exact_original_gene_kernels_match_across_populations(
    _name: str,
    strategy_factory: Callable[[], GeneStrategy],
    max_iter: int,
    seed: int,
) -> None:
    """Retained kernels remain exact across independent population fixtures."""
    shared_arguments = {
        "population": _random_population(seed),
        "gene_strategies": [strategy_factory()],
        "org_strategies": [NoneOrgStrategy()],
        "initial_gene_fitness": [0.4, 0.3, 0.2, 0.1],
        "max_iter": max_iter,
    }
    iterative = PikaiaModel(**shared_arguments, use_d_matrix=False)
    d_matrix = PikaiaModel(
        population=_random_population(seed),
        gene_strategies=[strategy_factory()],
        org_strategies=[NoneOrgStrategy()],
        initial_gene_fitness=[0.4, 0.3, 0.2, 0.1],
        max_iter=max_iter,
        use_d_matrix=True,
    )

    iterative.fit()
    d_matrix.fit()

    _assert_fitness_histories_match(iterative, d_matrix, max_iter)


UNSUPPORTED_GENE_STRATEGIES: list[tuple[str, Callable[[], GeneStrategy]]] = [
    ("entropy", EntropyMaxGeneStrategy),
    ("orthogonality", OrthoGeneStrategy),
    ("partial-correlation", PartialCorrGeneStrategy),
    ("redundancy-penalty", RedundancyPenaltyGeneStrategy),
    ("kin-altruistic-bounded", lambda: KinAltruisticGeneStrategy(kin_range=2)),
]


@pytest.mark.parametrize("_name,strategy_factory", UNSUPPORTED_GENE_STRATEGIES)
def test_inexact_original_gene_strategies_reject_d_matrix(
    _name: str, strategy_factory: Callable[[], GeneStrategy]
) -> None:
    """Strategies without a proven kernel reject D-matrix execution explicitly."""
    rejected = PikaiaModel(
        population=_population(),
        gene_strategies=[strategy_factory()],
        org_strategies=[NoneOrgStrategy()],
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="does not support use_d_matrix=True"):
        rejected.fit()


@pytest.mark.parametrize(
    "strategy_factory",
    [
        BalancedOrgStrategy,
        AltruisticOrgStrategy,
        KinSelfishOrgStrategy,
        SelfishOrgStrategy,
    ],
)
def test_inexact_original_organism_strategies_reject_d_matrix(
    strategy_factory: Callable[[], OrgStrategy],
) -> None:
    """Original organism strategies without a proven kernel reject D-matrix execution."""
    rejected = PikaiaModel(
        population=_population(),
        gene_strategies=[NoneGeneStrategy()],
        org_strategies=[strategy_factory()],
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="does not support use_d_matrix=True"):
        rejected.fit()


def test_d_matrix_rejects_a_mixed_supported_and_unsupported_combination() -> None:
    """A D-capable strategy cannot be combined with an iterative-only strategy.

    The model executes one complete update path per run. It therefore rejects
    this combination instead of calculating the dominant contribution from a
    D matrix and silently dropping the balanced organism contribution.
    """
    rejected = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy()],
        org_strategies=[BalancedOrgStrategy()],
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="BalancedOrgStrategy.*does not support"):
        rejected.fit()


def test_d_matrix_validates_before_zero_fitness_range_short_circuit() -> None:
    """A degenerate population must not hide an unsupported D configuration."""
    rejected = PikaiaModel(
        population=PikaiaPopulation(np.array([[0.2, 0.8], [0.8, 0.2]])),
        gene_strategies=[DominantGeneStrategy()],
        org_strategies=[BalancedOrgStrategy()],
        initial_gene_fitness=[0.5, 0.5],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="BalancedOrgStrategy.*does not support"):
        rejected.fit()


def test_d_matrix_rejects_self_consistent_mixing() -> None:
    """D-matrix mode rejects dynamic weights derived from per-organism deltas."""
    rejected = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy()],
        org_strategies=[NoneOrgStrategy()],
        gene_mix_strategy=SelfConsistentMixStrategy(),
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="requires FixedMixStrategy"):
        rejected.fit()


@pytest.mark.parametrize(
    ("gene_mix_strategy", "org_mix_strategy"),
    [
        (CustomPassThroughMixStrategy(), None),
        (FixedMixSubclass(), None),
        (None, CustomPassThroughMixStrategy()),
        (None, FixedMixSubclass()),
    ],
)
def test_d_matrix_rejects_custom_mixing(
    gene_mix_strategy: MixStrategy | None,
    org_mix_strategy: MixStrategy | None,
) -> None:
    """Neither strategy family permits custom or overridden mixers in D mode."""
    rejected = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy()],
        org_strategies=[NoneOrgStrategy()],
        gene_mix_strategy=gene_mix_strategy,
        org_mix_strategy=org_mix_strategy,
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="requires FixedMixStrategy"):
        rejected.fit()


@pytest.mark.parametrize("max_iter", [1, 50, 100])
def test_fixed_mixture_of_exact_kernels_matches_iterative_path(max_iter: int) -> None:
    """Fixed mixing preserves D-matrix equivalence when every term is exact."""
    strategies = [DominantGeneStrategy, AltruisticGeneStrategy, SellEasyGeneStrategy]
    iterative = PikaiaModel(
        population=_population(),
        gene_strategies=[strategy() for strategy in strategies],
        org_strategies=[NoneOrgStrategy()],
        gene_mixing_coeffs=[0.2, 0.3, 0.5],
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=max_iter,
        use_d_matrix=False,
    )
    d_matrix = PikaiaModel(
        population=_population(),
        gene_strategies=[strategy() for strategy in strategies],
        org_strategies=[NoneOrgStrategy()],
        gene_mixing_coeffs=[0.2, 0.3, 0.5],
        initial_gene_fitness=[0.6, 0.3, 0.1],
        max_iter=max_iter,
        use_d_matrix=True,
    )

    iterative.fit()
    d_matrix.fit()

    _assert_fitness_histories_match(iterative, d_matrix, max_iter)


@pytest.mark.parametrize("max_iter", [1, 50, 100])
def test_d_matrix_preserves_zero_gene_fitness(max_iter: int) -> None:
    """Zero-valued simplex components remain zero in both execution paths."""
    shared_arguments = {
        "population": _population(),
        "gene_strategies": [DominantGeneStrategy()],
        "org_strategies": [NoneOrgStrategy()],
        "initial_gene_fitness": [0.7, 0.3, 0.0],
        "max_iter": max_iter,
    }
    iterative = PikaiaModel(**shared_arguments, use_d_matrix=False)
    d_matrix = PikaiaModel(**shared_arguments, use_d_matrix=True)
    iterative.fit()
    d_matrix.fit()

    _assert_fitness_histories_match(iterative, d_matrix, max_iter)
    assert d_matrix.gene_fitness_history[max_iter, 2] == 0.0
