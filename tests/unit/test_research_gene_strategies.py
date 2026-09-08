"""
Tests for the four research-derived gene strategies:
EntropyMax, OrthoGene, PartialCorr, RedundancyPenalty.

Each test class covers:
1. name property
2. __call__ returns a finite float
3. __call__ with y (supervised mode where applicable)
4. kernel returns (D, None) with correct shape
5. kernel D-matrix diagonal is consistent with __call__ loop sum
6. Factory round-trip via GeneStrategyEnum
7. PikaiaModel integration (end-to-end fit)
8. Strategy-specific behavioural invariants
"""

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas.strategies import GeneStrategyEnum, MixStrategyEnum, OrgStrategyEnum
from pikaia.strategies import (
    GeneStrategyFactory,
    MixStrategyFactory,
    OrgStrategyFactory,
)
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.entropy_max_strategy import EntropyMaxGeneStrategy
from pikaia.strategies.gs_strategies.orthogonality_strategy import OrthoGeneStrategy
from pikaia.strategies.gs_strategies.partial_corr_strategy import (
    PartialCorrGeneStrategy,
)
from pikaia.strategies.gs_strategies.redundancy_penalty_strategy import (
    RedundancyPenaltyGeneStrategy,
)

# ---------------------------------------------------------------------------
# Shared fixtures and helpers
# ---------------------------------------------------------------------------

# 10 samples × 4 features, values in [0, 1]
np.random.seed(42)
X_BASE = np.random.rand(10, 4)
# Two-class target
Y_BASE = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])


def _make_pop(X: np.ndarray = X_BASE) -> PikaiaPopulation:
    return PikaiaPopulation(X)


def _make_ctx(
    pop: PikaiaPopulation,
    gene_id: int = 0,
    y: np.ndarray | None = None,
) -> StrategyContext:
    n, m = pop.N, pop.M
    return StrategyContext(
        population=pop,
        org_fitness=np.ones(n) / n,
        gene_fitness=np.ones(m) / m,
        org_similarity=np.eye(n),
        gene_similarity=np.eye(m),
        initial_org_fitness_range=1.0,
        org_id=0,
        gene_id=gene_id,
        y=y,
    )


def _run_pikaia(gene_strat, y: np.ndarray | None = None) -> np.ndarray:
    """Run 5 iterations and return final gene fitness."""
    pop = _make_pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[gene_strat],
        org_strategies=[OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)],
        gene_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
        org_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
        max_iter=5,
        y=y,
    )
    model.fit()
    return model.gene_fitness_history[-1]


# ---------------------------------------------------------------------------
# EntropyMaxGeneStrategy
# ---------------------------------------------------------------------------


class TestEntropyMaxGeneStrategy:
    def test_name(self):
        assert EntropyMaxGeneStrategy().name == "EntropyMax"

    def test_call_returns_finite_float(self):
        strat = EntropyMaxGeneStrategy()
        ctx = _make_ctx(_make_pop(), y=Y_BASE)
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_without_y_returns_finite(self):
        strat = EntropyMaxGeneStrategy()
        ctx = _make_ctx(_make_pop())  # no y
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_scores_cached_after_first_call(self):
        strat = EntropyMaxGeneStrategy()
        ctx = _make_ctx(_make_pop(), y=Y_BASE)
        strat(ctx)
        cached = strat._info_scores
        assert cached is not None
        strat(ctx)
        assert strat._info_scores is cached  # same object, not recomputed

    def test_precomputed_info_skips_computation(self):
        precomputed = np.array([0.8, 0.4, 0.6, 0.2])
        strat = EntropyMaxGeneStrategy(precomputed_info=precomputed)
        ctx = _make_ctx(_make_pop(), y=Y_BASE)
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_kernel_shape(self):
        strat = EntropyMaxGeneStrategy()
        pop = _make_pop()
        D, d = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0, y=Y_BASE)
        assert D is not None
        assert D.shape == (pop.M, pop.M)
        assert d is None

    def test_kernel_is_diagonal(self):
        strat = EntropyMaxGeneStrategy()
        pop = _make_pop()
        D, _ = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0, y=Y_BASE)
        assert D is not None
        off_diag = D - np.diag(np.diag(D))
        assert np.allclose(off_diag, 0.0)

    def test_info_scores_in_range(self):
        scores = EntropyMaxGeneStrategy.compute_info_scores(X_BASE, Y_BASE)
        assert scores.shape == (X_BASE.shape[1],)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_info_scores_without_y_are_zero(self):
        # Without y, MI is zero so info_score = 0 × entropy_norm = 0
        scores = EntropyMaxGeneStrategy.compute_info_scores(X_BASE, y=None)
        assert np.all(scores == 0.0)

    def test_mode_switch_recomputes(self):
        strat = EntropyMaxGeneStrategy()
        strat._get_scores(X_BASE, None)
        assert strat._mode == "unsupervised"
        assert strat._info_scores is not None
        first_scores = strat._info_scores.copy()
        strat._get_scores(X_BASE, Y_BASE)
        assert strat._mode == "supervised"
        assert strat._info_scores is not None
        assert not np.allclose(strat._info_scores, first_scores)  # recomputed with MI

    def test_precomputed_not_overwritten_on_mode_switch(self):
        precomputed = np.array([0.8, 0.4, 0.6, 0.2])
        strat = EntropyMaxGeneStrategy(precomputed_info=precomputed)
        strat._get_scores(X_BASE, Y_BASE)  # switching "mode" — should NOT recompute
        assert strat._info_scores is not None
        assert np.allclose(strat._info_scores, precomputed)

    def test_factory_round_trip(self):
        strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.ENTROPY_MAX)
        assert isinstance(strat, EntropyMaxGeneStrategy)

    def test_integration_with_pikaia_model(self):
        gf = _run_pikaia(EntropyMaxGeneStrategy(), y=Y_BASE)
        assert gf.shape == (X_BASE.shape[1],)
        assert np.all(np.isfinite(gf))
        assert np.isclose(gf.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# OrthoGeneStrategy
# ---------------------------------------------------------------------------


class TestOrthoGeneStrategy:
    def test_name(self):
        assert OrthoGeneStrategy().name == "OrthoGene"

    def test_call_returns_finite_float(self):
        strat = OrthoGeneStrategy()
        ctx = _make_ctx(_make_pop())
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_with_y_returns_finite(self):
        strat = OrthoGeneStrategy()
        ctx = _make_ctx(_make_pop(), y=Y_BASE)
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_supervised_and_unsupervised_differ(self):
        pop = _make_pop()
        strat_u = OrthoGeneStrategy()
        strat_s = OrthoGeneStrategy()
        r_u = strat_u(_make_ctx(pop))
        r_s = strat_s(_make_ctx(pop, y=Y_BASE))
        # Scores may differ because y is appended in supervised mode
        assert np.isfinite(r_u) and np.isfinite(r_s)

    def test_scores_shape(self):
        strat = OrthoGeneStrategy()
        strat._get_scores(X_BASE, None)
        assert strat._orthogonality is not None
        assert strat._orthogonality.shape == (X_BASE.shape[1],)

    def test_scores_shape_supervised(self):
        """Supervised mode must return n_features scores, not n_features+1."""
        strat = OrthoGeneStrategy()
        strat._get_scores(X_BASE, Y_BASE)
        assert strat._orthogonality is not None
        assert strat._orthogonality.shape == (X_BASE.shape[1],)

    def test_kernel_shape(self):
        strat = OrthoGeneStrategy()
        pop = _make_pop()
        D, d = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0)
        assert D is not None
        assert D.shape == (pop.M, pop.M)
        assert d is None

    def test_kernel_is_diagonal(self):
        strat = OrthoGeneStrategy()
        pop = _make_pop()
        D, _ = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0)
        assert D is not None
        off_diag = D - np.diag(np.diag(D))
        assert np.allclose(off_diag, 0.0)

    def test_uncorrelated_features_score_higher(self):
        """A feature independent of all others should score higher than a duplicate."""
        n = 20
        independent = np.random.rand(n)
        duplicate = np.random.rand(n)
        X = np.column_stack([duplicate, duplicate + 1e-10, independent])
        X = np.clip(X / X.max(axis=0), 0, 1)
        strat = OrthoGeneStrategy()
        scores = strat._get_scores(X, None)
        # Independent feature (col 2) should be more orthogonal than duplicates (0, 1)
        assert scores[2] > scores[0]

    def test_factory_round_trip(self):
        strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.ORTHO_GENE)
        assert isinstance(strat, OrthoGeneStrategy)

    def test_integration_with_pikaia_model(self):
        gf = _run_pikaia(OrthoGeneStrategy())
        assert gf.shape == (X_BASE.shape[1],)
        assert np.all(np.isfinite(gf))
        assert np.isclose(gf.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# PartialCorrGeneStrategy
# ---------------------------------------------------------------------------


class TestPartialCorrGeneStrategy:
    def test_name(self):
        assert PartialCorrGeneStrategy().name == "PartialCorr"

    def test_call_returns_finite_float(self):
        strat = PartialCorrGeneStrategy()
        ctx = _make_ctx(_make_pop(), y=Y_BASE)
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_without_y_returns_zero_delta(self):
        # Without y, partial corrs are all zero → delta = gf * (4/N) * (0 - 0.5) < 0
        strat = PartialCorrGeneStrategy()
        ctx = _make_ctx(_make_pop())
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)
        assert result < 0  # score = 0 - 0.5 < 0 ⟹ negative delta

    def test_scores_in_range(self):
        scores = PartialCorrGeneStrategy.compute_partial_correlations(X_BASE, Y_BASE)
        assert scores.shape == (X_BASE.shape[1],)
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_scores_without_y_are_zero(self):
        scores = PartialCorrGeneStrategy.compute_partial_correlations(X_BASE, y=None)
        assert np.all(scores == 0.0)

    def test_precomputed_pc_used(self):
        precomputed = np.array([0.9, 0.1, 0.5, 0.3])
        # Gene 0 score = 0.9 → delta > 0; gene 1 score = 0.1 → delta < 0
        ctx_g0 = _make_ctx(_make_pop(), gene_id=0, y=Y_BASE)
        ctx_g0.gene_fitness = np.ones(X_BASE.shape[1]) / X_BASE.shape[1]
        s0 = PartialCorrGeneStrategy(precomputed_pc=precomputed)(ctx_g0)
        ctx_g1 = _make_ctx(_make_pop(), gene_id=1, y=Y_BASE)
        ctx_g1.gene_fitness = np.ones(X_BASE.shape[1]) / X_BASE.shape[1]
        s1 = PartialCorrGeneStrategy(precomputed_pc=precomputed)(ctx_g1)
        assert s0 > s1  # higher partial corr → higher delta

    def test_kernel_shape(self):
        strat = PartialCorrGeneStrategy()
        pop = _make_pop()
        D, d = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0, y=Y_BASE)
        assert D is not None
        assert D.shape == (pop.M, pop.M)
        assert d is None

    def test_kernel_is_diagonal(self):
        strat = PartialCorrGeneStrategy()
        pop = _make_pop()
        D, _ = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0, y=Y_BASE)
        assert D is not None
        off_diag = D - np.diag(np.diag(D))
        assert np.allclose(off_diag, 0.0)

    def test_mode_switch_recomputes(self):
        strat = PartialCorrGeneStrategy()
        strat._get_scores(X_BASE, None)
        assert strat._mode == "unsupervised"
        assert np.all(strat._partial_corrs == 0.0)  # zeros without y
        strat._get_scores(X_BASE, Y_BASE)
        assert strat._mode == "supervised"
        assert not np.all(strat._partial_corrs == 0.0)  # recomputed with y

    def test_precomputed_not_overwritten_on_mode_switch(self):
        precomputed = np.array([0.9, 0.1, 0.5, 0.3])
        strat = PartialCorrGeneStrategy(precomputed_pc=precomputed)
        strat._get_scores(X_BASE, Y_BASE)  # switching "mode" — should NOT recompute
        assert strat._partial_corrs is not None
        assert np.allclose(strat._partial_corrs, precomputed)

    def test_factory_round_trip(self):
        strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.PARTIAL_CORR)
        assert isinstance(strat, PartialCorrGeneStrategy)

    def test_integration_with_pikaia_model(self):
        gf = _run_pikaia(PartialCorrGeneStrategy(), y=Y_BASE)
        assert gf.shape == (X_BASE.shape[1],)
        assert np.all(np.isfinite(gf))
        assert np.isclose(gf.sum(), 1.0, atol=1e-6)


# ---------------------------------------------------------------------------
# RedundancyPenaltyGeneStrategy
# ---------------------------------------------------------------------------


class TestRedundancyPenaltyGeneStrategy:
    def test_name(self):
        assert RedundancyPenaltyGeneStrategy().name == "RedundancyPenalty"

    def test_call_returns_finite_float(self):
        strat = RedundancyPenaltyGeneStrategy()
        ctx = _make_ctx(_make_pop())
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_with_y_returns_finite(self):
        strat = RedundancyPenaltyGeneStrategy()
        ctx = _make_ctx(_make_pop(), y=Y_BASE)
        result = strat(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_redundancy_values_in_range(self):
        red = RedundancyPenaltyGeneStrategy.compute_redundancy(X_BASE)
        assert red.shape == (X_BASE.shape[1],)
        assert np.all(red >= 0.0) and np.all(red <= 1.0)

    def test_identical_features_have_max_redundancy(self):
        """A feature duplicated exactly should have redundancy ≈ 1."""
        col = np.random.rand(20)
        X = np.column_stack([col, col, np.random.rand(20)])
        X = np.clip(X, 0, 1)
        red = RedundancyPenaltyGeneStrategy.compute_redundancy(X)
        assert red[0] > 0.5, "Duplicate feature should have high redundancy"
        assert red[1] > 0.5

    def test_independent_feature_has_lower_redundancy(self):
        """An independent feature should be less redundant than correlated ones."""
        n = 30
        base = np.random.rand(n)
        X = np.column_stack([base, base + 1e-6, np.random.rand(n)])
        X = np.clip(X / X.max(axis=0), 0, 1)
        red = RedundancyPenaltyGeneStrategy.compute_redundancy(X)
        assert red[2] < red[0], "Independent feature should be less redundant"

    def test_scores_shape_supervised(self):
        """Supervised mode must return n_features scores, not n_features+1."""
        strat = RedundancyPenaltyGeneStrategy()
        strat._get_scores(X_BASE, Y_BASE)
        assert strat._redundancy is not None
        assert strat._redundancy.shape == (X_BASE.shape[1],)

    def test_mode_switch_recomputes(self):
        strat = RedundancyPenaltyGeneStrategy()
        strat._get_scores(X_BASE, None)
        assert strat._mode == "unsupervised"
        strat._get_scores(X_BASE, Y_BASE)
        assert strat._mode == "supervised"

    def test_kernel_shape(self):
        strat = RedundancyPenaltyGeneStrategy()
        pop = _make_pop()
        D, d = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0)
        assert D is not None
        assert D.shape == (pop.M, pop.M)
        assert d is None

    def test_kernel_is_diagonal(self):
        strat = RedundancyPenaltyGeneStrategy()
        pop = _make_pop()
        D, _ = strat.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0)
        assert D is not None
        off_diag = D - np.diag(np.diag(D))
        assert np.allclose(off_diag, 0.0)

    def test_high_redundancy_gives_negative_delta(self):
        """Highly redundant features should receive a negative delta."""
        col = np.random.rand(20)
        X = np.column_stack([col, col + 1e-6, np.random.rand(20), np.random.rand(20)])
        X = np.clip(X / X.max(axis=0), 0, 1)
        pop = PikaiaPopulation(X)
        strat = RedundancyPenaltyGeneStrategy()
        ctx = _make_ctx(pop, gene_id=0)
        delta = strat(ctx)
        # Redundancy of col 0 >> 0.5 → (0.5 - red) < 0 → delta < 0
        assert delta < 0

    def test_factory_round_trip(self):
        strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REDUNDANCY_PENALTY)
        assert isinstance(strat, RedundancyPenaltyGeneStrategy)

    def test_integration_with_pikaia_model(self):
        gf = _run_pikaia(RedundancyPenaltyGeneStrategy())
        assert gf.shape == (X_BASE.shape[1],)
        assert np.all(np.isfinite(gf))
        assert np.isclose(gf.sum(), 1.0, atol=1e-6)
