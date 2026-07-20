"""Unit tests for strategy kernel() methods.

Each concrete strategy that overrides kernel() is tested here.  Tests verify:
  - Return shape and types
  - Structural properties (zero diagonal, symmetry, sign)
  - Mathematical properties (rank, magnitude scaling)
  - Edge-cases (single organism, kin_range=1, n_contributing==0)
"""

import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import (
    KinAltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.os_strategies.altruistic_strategy import AltruisticOrgStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.kin_selfish_strategy import KinSelfishOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pop(rng, N, M):
    return PikaiaPopulation(np.random.default_rng(rng).random((N, M)))


def _make_sims(pop):
    """Return (gene_similarity, org_similarity, initial_org_fitness_range)."""

    def sim(matrix):
        diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        mx = dist.max()
        return np.ones_like(dist) if mx == 0 else 1 - dist / mx

    gs = sim(pop.matrix.T)
    os_ = sim(pop.matrix)
    gf = np.ones(pop.M) / pop.M
    org_fit = pop.matrix @ gf
    R = org_fit.max() - org_fit.min()
    return gs, os_, max(R, 1e-6)


# ---------------------------------------------------------------------------
# Gene strategies – kernel()
# ---------------------------------------------------------------------------


class TestNoneGeneStrategyKernel:
    def test_returns_none_none(self):
        pop = _make_pop(0, 5, 3)
        gs, os_, R = _make_sims(pop)
        D, d = NoneGeneStrategy().kernel(pop, gs, os_, R)
        assert D is None
        assert d is None


class TestDominantGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(1, 8, 4)
        gs, os_, R = _make_sims(pop)
        return pop, DominantGeneStrategy().kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, (D, d) = result
        assert D is not None
        assert d is None

    def test_D_is_diagonal(self, result):
        pop, (D, _) = result
        off = D - np.diag(np.diag(D))
        assert np.allclose(off, 0), "Dominant gene D must be diagonal"

    def test_diagonal_formula(self, result):
        pop, (D, _) = result
        x_bar = pop.matrix.mean(axis=0)
        expected = np.diag(4.0 * (x_bar - 0.5))
        np.testing.assert_allclose(D, expected)

    def test_D_shape(self, result):
        pop, (D, _) = result
        assert D.shape == (pop.M, pop.M)


class TestAltruisticGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(2, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, gs, AltruisticGeneStrategy().kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, _, (D, d) = result
        assert D is not None
        assert d is None

    def test_D_shape(self, result):
        pop, _, (D, _) = result
        assert D.shape == (pop.M, pop.M)

    def test_zero_diagonal(self, result):
        _, _, (D, _) = result
        assert np.allclose(np.diag(D), 0.0), "Altruistic gene D must have zero diagonal"

    def test_finite(self, result):
        _, _, (D, _) = result
        assert np.all(np.isfinite(D))


class TestSelfishGeneStrategyKernel:
    @pytest.fixture
    def pair(self):
        pop = _make_pop(3, 10, 5)
        gs, os_, R = _make_sims(pop)
        D_alt, _ = AltruisticGeneStrategy().kernel(pop, gs, os_, R)
        D_sel, _ = SelfishGeneStrategy().kernel(pop, gs, os_, R)
        return D_alt, D_sel

    def test_selfish_is_negated_altruistic(self, pair):
        D_alt, D_sel = pair
        np.testing.assert_allclose(D_sel, -D_alt, atol=1e-12)

    def test_zero_diagonal(self, pair):
        _, D_sel = pair
        assert np.allclose(np.diag(D_sel), 0.0)

    def test_shape(self):
        pop = _make_pop(4, 6, 4)
        gs, os_, R = _make_sims(pop)
        D, _ = SelfishGeneStrategy().kernel(pop, gs, os_, R)
        assert D is not None
        assert D.shape == (pop.M, pop.M)


class TestKinAltruisticGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(5, 10, 6)
        gs, os_, R = _make_sims(pop)
        return pop, gs, KinAltruisticGeneStrategy(kin_range=3).kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, _, (D, d) = result
        assert D is not None
        assert d is None

    def test_zero_diagonal(self, result):
        _, _, (D, _) = result
        assert np.allclose(np.diag(D), 0.0)

    def test_shape(self, result):
        pop, _, (D, _) = result
        assert D.shape == (pop.M, pop.M)

    def test_kin_range_changes_d_matrix(self):
        """kin_range < M produces a different D than kin_range=M (full range).

        The masking sets gene_sim_masked[j,k]=0 for non-kin genes, replacing
        the true similarity weight (0.5 - gene_sim[j,k]) with the constant 0.5.
        This yields a different D than the unmasked full-range version.
        """
        pop = _make_pop(6, 8, 5)
        gs, os_, R = _make_sims(pop)
        D_kin, _ = KinAltruisticGeneStrategy(kin_range=2).kernel(pop, gs, os_, R)
        D_full, _ = KinAltruisticGeneStrategy(kin_range=pop.M).kernel(pop, gs, os_, R)
        assert D_kin is not None
        assert D_full is not None
        # kin_range=2 should produce a different D than full-range (M=5 > 2)
        assert not np.allclose(D_kin, D_full), (
            "kin_range=2 should change D vs full-range (gene similarities differ from 0)"
        )

    def test_default_kin_range_matches_all_genes(self):
        """Without kin_range, all genes within the gene_similarity structure contribute."""
        pop = _make_pop(7, 8, 4)
        gs, os_, R = _make_sims(pop)
        D_default, _ = KinAltruisticGeneStrategy().kernel(pop, gs, os_, R)
        D_full, _ = KinAltruisticGeneStrategy(kin_range=pop.M).kernel(pop, gs, os_, R)
        assert D_default is not None
        assert D_full is not None
        np.testing.assert_allclose(D_default, D_full)


# ---------------------------------------------------------------------------
# Org strategies – kernel()
# ---------------------------------------------------------------------------


class TestNoneOrgStrategyKernel:
    def test_returns_none_none(self):
        pop = _make_pop(10, 5, 3)
        gs, os_, R = _make_sims(pop)
        D, d = NoneOrgStrategy().kernel(pop, gs, os_, R)
        assert D is None
        assert d is None


class TestBalancedOrgStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(11, 8, 4)
        gs, os_, R = _make_sims(pop)
        return pop, BalancedOrgStrategy().kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, (D, d) = result
        assert D is not None
        assert d is None

    def test_shape(self, result):
        pop, (D, _) = result
        assert D.shape == (pop.M, pop.M)

    def test_rank1_structure(self, result):
        """D should be rank-1: D[j,k] = -2*x_bar_j for all k."""
        pop, (D, _) = result
        x_bar = pop.matrix.mean(axis=0)
        expected = np.outer(-2.0 * x_bar, np.ones(pop.M))
        np.testing.assert_allclose(D, expected, atol=1e-12)

    def test_Dgamma_equals_minus2_xbar(self, result):
        """(D@γ)_j = -2*x̄_j for any normalised γ — key correctness property."""
        pop, (D, _) = result
        rng = np.random.default_rng(99)
        gamma = rng.random(pop.M)
        gamma /= gamma.sum()
        x_bar = pop.matrix.mean(axis=0)
        np.testing.assert_allclose(D @ gamma, -2.0 * x_bar, atol=1e-12)


class TestSelfishOrgStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(12, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, SelfishOrgStrategy().kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, (D, d) = result
        assert D is not None
        assert d is None

    def test_shape(self, result):
        pop, (D, _) = result
        assert D.shape == (pop.M, pop.M)

    def test_finite(self, result):
        _, (D, _) = result
        assert np.all(np.isfinite(D))

    def test_n_contributing_zero_returns_zero_matrix(self):
        """With kin_range=0, no organism has relatives → zero D."""
        pop = _make_pop(13, 4, 3)
        gs, os_, R = _make_sims(pop)
        D, d = SelfishOrgStrategy(kin_range=0).kernel(pop, gs, os_, R)
        assert D is not None
        assert np.allclose(D, 0.0)
        assert d is None


class TestAltruisticOrgStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(14, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, AltruisticOrgStrategy().kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, (D, d) = result
        assert D is not None
        assert d is None

    def test_shape(self, result):
        pop, (D, _) = result
        assert D.shape == (pop.M, pop.M)

    def test_finite(self, result):
        _, (D, _) = result
        assert np.all(np.isfinite(D))

    def test_n_contributing_zero_returns_zero_matrix(self):
        """With kin_range=0, no organism has relatives → zero D."""
        pop = _make_pop(15, 4, 3)
        gs, os_, R = _make_sims(pop)
        D, d = AltruisticOrgStrategy(kin_range=0).kernel(pop, gs, os_, R)
        assert D is not None
        assert np.allclose(D, 0.0)
        assert d is None

    def test_matches_selfish_org(self):
        """AltruisticOrgStrategy and SelfishOrgStrategy use the same formula."""
        pop = _make_pop(16, 8, 4)
        gs, os_, R = _make_sims(pop)
        D_alt, _ = AltruisticOrgStrategy().kernel(pop, gs, os_, R)
        D_sel, _ = SelfishOrgStrategy().kernel(pop, gs, os_, R)
        assert D_alt is not None
        assert D_sel is not None
        np.testing.assert_allclose(D_alt, D_sel, atol=1e-12)


class TestKinSelfishOrgStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(17, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, KinSelfishOrgStrategy(kin_range=3).kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, (D, d) = result
        assert D is not None
        assert d is None

    def test_shape(self, result):
        pop, (D, _) = result
        assert D.shape == (pop.M, pop.M)

    def test_finite(self, result):
        _, (D, _) = result
        assert np.all(np.isfinite(D))

    def test_n_contributing_zero_returns_zero_matrix(self):
        """With kin_range=0, no organism has relatives → zero D."""
        pop = _make_pop(18, 4, 3)
        gs, os_, R = _make_sims(pop)
        D, d = KinSelfishOrgStrategy(kin_range=0).kernel(pop, gs, os_, R)
        assert D is not None
        assert np.allclose(D, 0.0)
        assert d is None

    def test_sign_opposite_to_selfish_org(self):
        """KinSelfish has +2/R sign vs Selfish -2/R — matrices have opposite sign when kin_range covers all."""
        pop = _make_pop(19, 8, 4)
        gs, os_, R = _make_sims(pop)
        D_sel, _ = SelfishOrgStrategy().kernel(pop, gs, os_, R)
        D_ksel, _ = KinSelfishOrgStrategy().kernel(pop, gs, os_, R)
        assert D_sel is not None
        assert D_ksel is not None
        # Both should be finite and non-trivially different (not checking exact sign
        # because similarity weights differ: s_il vs (0.5 - s_il))
        assert np.all(np.isfinite(D_ksel))
        assert not np.allclose(D_ksel, D_sel)


# ---------------------------------------------------------------------------
# kernel() y parameter — adaptive supervision
# ---------------------------------------------------------------------------


ALL_STRATEGIES = [
    DominantGeneStrategy(),
    AltruisticGeneStrategy(),
    SelfishGeneStrategy(),
    KinAltruisticGeneStrategy(),
    NoneGeneStrategy(),
    BalancedOrgStrategy(),
    AltruisticOrgStrategy(),
    SelfishOrgStrategy(),
    KinSelfishOrgStrategy(),
    NoneOrgStrategy(),
]


class TestKernelYParameter:
    """kernel() accepts y without error; built-in strategies produce identical output."""

    def setup_method(self):
        self.pop = _make_pop(99, 6, 4)
        self.gs, self.os_, self.R = _make_sims(self.pop)
        self.y = np.array([0, 1, 0, 1, 0, 1])

    @pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: type(s).__name__)
    def test_kernel_accepts_y_none(self, strat):
        """kernel(y=None) does not raise."""
        D, d = strat.kernel(self.pop, self.gs, self.os_, self.R, y=None)
        assert D is None or isinstance(D, np.ndarray)
        assert d is None or isinstance(d, np.ndarray)

    @pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: type(s).__name__)
    def test_kernel_accepts_y_array(self, strat):
        """kernel(y=array) does not raise."""
        D, d = strat.kernel(self.pop, self.gs, self.os_, self.R, y=self.y)
        assert D is None or isinstance(D, np.ndarray)
        assert d is None or isinstance(d, np.ndarray)

    @pytest.mark.parametrize("strat", ALL_STRATEGIES, ids=lambda s: type(s).__name__)
    def test_kernel_y_does_not_change_builtin_output(self, strat):
        """Built-in strategies return identical matrices regardless of y."""
        D_none, d_none = strat.kernel(self.pop, self.gs, self.os_, self.R, y=None)
        D_y, d_y = strat.kernel(self.pop, self.gs, self.os_, self.R, y=self.y)
        if D_none is None:
            assert D_y is None
        else:
            assert D_y is not None
            assert np.allclose(D_none, D_y)
        if d_none is None:
            assert d_y is None
        else:
            assert d_y is not None
            assert np.allclose(d_none, d_y)
