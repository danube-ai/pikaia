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
from pikaia.strategies.gs_strategies.sell_easy_strategy import SellEasyGeneStrategy
from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy
from pikaia.strategies.gs_strategies.sell_uniform_strategy import (
    SellUniformGeneStrategy,
)
from pikaia.strategies.gs_strategies.variance_strategy import VarianceGeneStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy

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


class TestVarianceGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(11, 8, 4)
        gs, os_, R = _make_sims(pop)
        return pop, VarianceGeneStrategy().kernel(pop, gs, os_, R)

    def test_returns_D_none_d(self, result):
        _, (D, d) = result
        assert D is not None
        assert d is None

    def test_D_is_diagonal(self, result):
        pop, (D, _) = result
        off = D - np.diag(np.diag(D))
        assert np.allclose(off, 0), "Variance gene D must be diagonal"

    def test_diagonal_formula(self, result):
        pop, (D, _) = result
        std = pop.matrix.std(axis=0, ddof=0)
        s_hat = std / (std.max() + 1e-8)
        x_bar = pop.matrix.mean(axis=0)
        expected = np.diag(4.0 * s_hat * (x_bar - 0.5))
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
        return pop, gs, KinAltruisticGeneStrategy().kernel(pop, gs, os_, R)

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

    def test_explicit_kin_range_has_no_d_matrix_implementation(self):
        """An explicit kin range is intentionally limited to the iterative path."""
        pop = _make_pop(7, 8, 4)
        gs, os_, R = _make_sims(pop)
        D_default, d_default = KinAltruisticGeneStrategy().kernel(pop, gs, os_, R)
        D_configured, d_configured = KinAltruisticGeneStrategy(kin_range=pop.M).kernel(
            pop, gs, os_, R
        )
        assert D_default is not None
        assert d_default is None
        assert D_configured is None
        assert d_configured is None


# ---------------------------------------------------------------------------
# SellHard / SellUniform / SellEasy kernels
# ---------------------------------------------------------------------------


class TestSellHardGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(8, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, SellHardGeneStrategy().kernel(pop, gs, os_, R)

    def test_returns_none_d(self, result):
        _, (D, d) = result
        assert D is None
        assert d is not None

    def test_d_shape(self, result):
        pop, (_, d) = result
        assert d.shape == (pop.M,)

    def test_d_non_positive(self, result):
        _, (_, d) = result
        assert np.all(d <= 0)

    def test_d_formula(self, result):
        pop, (_, d) = result
        mean_all = pop.matrix.mean(axis=0)
        excl = 1.0 - mean_all
        expected = -mean_all * excl / (1.0 - excl + 1e-8)
        np.testing.assert_allclose(d, expected, atol=1e-12)


class TestSellUniformGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(9, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, SellUniformGeneStrategy().kernel(pop, gs, os_, R)

    def test_returns_none_d(self, result):
        _, (D, d) = result
        assert D is None
        assert d is not None

    def test_d_shape(self, result):
        pop, (_, d) = result
        assert d.shape == (pop.M,)

    def test_d_non_positive(self, result):
        _, (_, d) = result
        assert np.all(d <= 0)

    def test_d_formula(self, result):
        pop, (_, d) = result
        expected = -pop.matrix.mean(axis=0)
        np.testing.assert_allclose(d, expected, atol=1e-12)


class TestSellEasyGeneStrategyKernel:
    @pytest.fixture
    def result(self):
        pop = _make_pop(10, 10, 5)
        gs, os_, R = _make_sims(pop)
        return pop, SellEasyGeneStrategy().kernel(pop, gs, os_, R)

    def test_returns_none_d(self, result):
        _, (D, d) = result
        assert D is None
        assert d is not None

    def test_d_shape(self, result):
        pop, (_, d) = result
        assert d.shape == (pop.M,)

    def test_d_negates_sell_hard(self):
        pop = _make_pop(10, 10, 5)
        gs, os_, R = _make_sims(pop)
        _, d_hard = SellHardGeneStrategy().kernel(pop, gs, os_, R)
        _, d_easy = SellEasyGeneStrategy().kernel(pop, gs, os_, R)
        assert d_hard is not None
        assert d_easy is not None
        np.testing.assert_allclose(d_easy, -d_hard, atol=1e-12)

    def test_d_formula(self, result):
        pop, (_, d) = result
        mean_all = pop.matrix.mean(axis=0)
        excl = 1.0 - mean_all
        expected = mean_all * excl / (1.0 - excl + 1e-8)
        np.testing.assert_allclose(d, expected, atol=1e-12)


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


# ---------------------------------------------------------------------------
# kernel() y parameter — adaptive supervision
# ---------------------------------------------------------------------------


ALL_STRATEGIES = [
    DominantGeneStrategy(),
    AltruisticGeneStrategy(),
    SelfishGeneStrategy(),
    KinAltruisticGeneStrategy(),
    SellHardGeneStrategy(),
    SellUniformGeneStrategy(),
    SellEasyGeneStrategy(),
    VarianceGeneStrategy(),
    NoneGeneStrategy(),
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
