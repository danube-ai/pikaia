"""
Tests for SellGeneStrategy and BuyOrgStrategy.

The pair is designed to reproduce CalSim's market recalibration round within
pikaia's replicator framework.  Tests verify:

1. Structural properties of each strategy in isolation.
2. Numerical direction match against CalSim on the canonical 3×4 dataset.
3. Kernel (d-vector) consistency with the __call__ loop sum.
4. Edge cases (perfect performers, all-zero columns).
"""

import sys

import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.sell_strategy import SellGeneStrategy
from pikaia.strategies.os_strategies.buy_strategy import BuyOrgStrategy

sys.path.insert(0, "/Users/uziel/Development/DanubeAI/experiments/tgeneticai")
calsim = pytest.importorskip(
    "calsim",
    reason="tgeneticai/calsim.py not on PYTHONPATH; skipping CalSim comparison tests",
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Canonical CalSim dataset (calsim.py __main__)
PERF = np.array(
    [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
)
START_VALUES = [5.0, 5.0, 5.0, 5.0]
N, M = PERF.shape
UNIFORM = 1.0 / M


def _pop():
    return PikaiaPopulation(PERF)


def _make_ctx(pop, org_id, gene_id=None, gene_fitness=None):
    if gene_fitness is None:
        gene_fitness = np.ones(pop.M) / pop.M
    return StrategyContext(
        population=pop,
        org_fitness=np.ones(pop.N) / pop.N,
        gene_fitness=gene_fitness,
        org_similarity=np.eye(pop.N),
        gene_similarity=np.eye(pop.M),
        initial_org_fitness_range=1.0,
        org_id=org_id,
        gene_id=gene_id,
    )


def _calsim_round(strategy: str) -> np.ndarray:
    """Run one CalSim recalibration round; return normalised gene values."""
    params = calsim.Params(sellStrategy=strategy, buyStrategy=strategy)
    exes = [
        calsim.Exercise(
            params=params,
            index=i,
            startvalue=START_VALUES[i],
            maxvalue=sum(START_VALUES),
            nprobs=N,
        )
        for i in range(M)
    ]
    probs = [calsim.Proband(params, i, START_VALUES, PERF[i]) for i in range(N)]
    test = calsim.Test(params, exes, probs)
    test.recalibrateAll()
    v = np.array(test.currentValues)
    return v / v.sum()


def _pikaia_round(gene_strat_enum, org_strat_enum, **gene_kwargs) -> np.ndarray:
    """Run one pikaia iteration; return normalised gene fitness."""
    pop = _pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[
            GeneStrategyFactory.get_strategy(gene_strat_enum, **gene_kwargs)
        ],
        org_strategies=[OrgStrategyFactory.get_strategy(org_strat_enum)],
        max_iter=1,
    )
    model.fit()
    return model.gene_fitness_history[1]


# ---------------------------------------------------------------------------
# 1. SellGeneStrategy structural tests
# ---------------------------------------------------------------------------


class TestSellGeneStrategy:
    def test_returns_nonpositive_for_solved_gene(self):
        """Organism that solved a gene (x=1) should produce a negative sell delta."""
        pop = _pop()
        strat = SellGeneStrategy()
        # org 0 solved gene 2 (x=1) — expect negative delta
        delta = strat(_make_ctx(pop, org_id=0, gene_id=2))
        assert delta < 0

    def test_returns_zero_for_unsolved_gene(self):
        """Organism that did not solve a gene (x=0) contributes nothing to sell."""
        pop = _pop()
        strat = SellGeneStrategy()
        # org 1 did NOT solve gene 3 (x=0)
        delta = strat(_make_ctx(pop, org_id=1, gene_id=3))
        assert np.isclose(delta, 0.0, atol=1e-12)

    def test_zero_exclusiveness_gives_zero_delta(self):
        """Gene solved by everyone (excl=0) has zero sell signal."""
        pop = _pop()
        strat = SellGeneStrategy()
        # gene 0 has mean=1.0, excl=0
        for org_id in range(pop.N):
            delta = strat(_make_ctx(pop, org_id=org_id, gene_id=0))
            assert np.isclose(delta, 0.0, atol=1e-12)

    def test_sum_over_organisms_matches_formula(self):
        """Sum over all organisms equals -mean_j * excl_j / (1-excl_j)."""
        pop = _pop()
        strat = SellGeneStrategy()
        mean_all = PERF.mean(axis=0)
        excl = 1.0 - mean_all
        expected_total = -mean_all * excl / (1.0 - excl + 1e-8)

        for j in range(M):
            total = sum(strat(_make_ctx(pop, org_id=i, gene_id=j)) for i in range(N))
            assert np.isclose(total, expected_total[j], atol=1e-10), (
                f"gene {j}: total={total:.8f} expected={expected_total[j]:.8f}"
            )

    def test_kernel_d_vector_matches_call_sum(self):
        """Kernel d-vector must equal the sum of __call__ over all organisms."""
        pop = _pop()
        strat = SellGeneStrategy()
        _, d = strat.kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert d is not None

        call_sum = np.array(
            [
                sum(strat(_make_ctx(pop, org_id=i, gene_id=j)) for i in range(N))
                for j in range(M)
            ]
        )
        np.testing.assert_allclose(d, call_sum, atol=1e-10)

    def test_kernel_D_is_none(self):
        """SellGeneStrategy has no bilinear D term."""
        pop = _pop()
        D, _ = SellGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None


# ---------------------------------------------------------------------------
# 2. BuyOrgStrategy structural tests
# ---------------------------------------------------------------------------


class TestBuyOrgStrategy:
    def test_returns_array_of_shape_M(self):
        """__call__ must return (M,) array."""
        pop = _pop()
        strat = BuyOrgStrategy()
        result = strat(_make_ctx(pop, org_id=0))
        assert result.shape == (M,)

    def test_buy_zero_for_solved_gene(self):
        """Organism does not buy a gene it already solved (x=1)."""
        pop = _pop()
        strat = BuyOrgStrategy()
        # org 0 solved gene 0 (x=1)
        result = strat(_make_ctx(pop, org_id=0))
        assert np.isclose(result[0], 0.0, atol=1e-12)

    def test_buy_positive_for_unsolved_gene(self):
        """Organism buys genes it did not solve (x=0), so delta must be ≥ 0."""
        pop = _pop()
        strat = BuyOrgStrategy()
        # org 1 did not solve gene 3
        result = strat(_make_ctx(pop, org_id=1))
        assert result[3] >= 0.0

    def test_kernel_d_vector_matches_call_sum(self):
        """Kernel d-vector must equal the sum of __call__ over all organisms."""
        pop = _pop()
        strat = BuyOrgStrategy()
        _, d = strat.kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert d is not None

        call_sum = np.zeros(M)
        for i in range(N):
            call_sum += strat(_make_ctx(pop, org_id=i))
        np.testing.assert_allclose(d, call_sum, atol=1e-10)

    def test_kernel_D_is_none(self):
        """BuyOrgStrategy has no bilinear D term."""
        pop = _pop()
        D, _ = BuyOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None

    def test_perfect_organism_contributes_nothing(self):
        """An organism that solved everything has zero excl_norm2 → zero buy."""
        data = np.array([[1.0, 1.0, 1.0], [0.5, 0.5, 0.5]])
        pop = PikaiaPopulation(data)
        strat = BuyOrgStrategy()
        result = strat(_make_ctx(pop, org_id=0))
        assert np.allclose(result, 0.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Sell+Buy together: capital conservation
# ---------------------------------------------------------------------------


def test_sell_plus_buy_net_delta_sums_to_zero():
    """
    In CalSim the total value is conserved (what is sold is bought back).
    The combined net delta summed over all genes should be ≈ 0.
    """
    pop = _pop()
    sell_strat = SellGeneStrategy()
    buy_strat = BuyOrgStrategy()

    sell_total = np.array(
        [
            sum(sell_strat(_make_ctx(pop, org_id=i, gene_id=j)) for i in range(N))
            for j in range(M)
        ]
    )
    buy_total = np.zeros(M)
    for i in range(N):
        buy_total += buy_strat(_make_ctx(pop, org_id=i))

    net = sell_total + buy_total
    assert np.isclose(net.sum(), 0.0, atol=1e-10), (
        f"Capital not conserved: net.sum()={net.sum():.2e}"
    )


# ---------------------------------------------------------------------------
# 4. Direction match vs CalSim
# ---------------------------------------------------------------------------


def test_direction_matches_calsim():
    """
    After one iteration, every gene that CalSim Difficulty1 moves up also goes
    up in pikaia, and every gene CalSim moves down also goes down.
    """
    cs_norm = _calsim_round("Difficulty1")
    pk_gf = _pikaia_round(GeneStrategyEnum.SELL, OrgStrategyEnum.BUY)

    for j in range(M):
        cs_dir = np.sign(cs_norm[j] - UNIFORM)
        pk_dir = np.sign(pk_gf[j] - UNIFORM)
        if cs_dir != 0:
            assert cs_dir == pk_dir, (
                f"Gene {j}: CalSim moved {'UP' if cs_dir > 0 else 'DN'} "
                f"but pikaia moved {'UP' if pk_dir > 0 else 'DN' if pk_dir < 0 else '='}.\n"
                f"CalSim norm: {cs_norm}\nPikaia gf:   {pk_gf}"
            )


def test_exact_numerical_match_calsim_difficulty1():
    """
    On the canonical 3×4 dataset starting from uniform gene fitness, pikaia
    SELL+BUY reproduces CalSim Difficulty1 with zero numerical error.

    This is a regression guard: if the formula changes, this test catches it.
    """
    cs_norm = _calsim_round("Difficulty1")
    pk_gf = _pikaia_round(GeneStrategyEnum.SELL, OrgStrategyEnum.BUY)
    np.testing.assert_allclose(pk_gf, cs_norm, atol=1e-8, rtol=0)


def test_sell_buy_ranking_closer_to_calsim_than_reward_hard():
    """
    Sell+Buy should produce a gene ranking closer to CalSim Difficulty1 than
    RewardHard alone (which misses the cross-organism buy redistribution).
    """
    cs_norm = _calsim_round("Difficulty1")
    pk_sell_buy = _pikaia_round(GeneStrategyEnum.SELL, OrgStrategyEnum.BUY)
    pk_hard = _pikaia_round(GeneStrategyEnum.REWARD_HARD, OrgStrategyEnum.BALANCED)

    # Kendall-tau distance: count agreeing pairs
    def rank_agreement(a, b):
        n = len(a)
        agree = sum(
            1
            for i in range(n)
            for j in range(i + 1, n)
            if np.sign(a[i] - a[j]) == np.sign(b[i] - b[j])
        )
        return agree

    sb_agree = rank_agreement(cs_norm, pk_sell_buy)
    rh_agree = rank_agreement(cs_norm, pk_hard)

    assert sb_agree >= rh_agree, (
        f"Sell+Buy rank agreement ({sb_agree}) should be ≥ RewardHard ({rh_agree})"
    )


# ---------------------------------------------------------------------------
# 5. Enum and factory round-trip
# ---------------------------------------------------------------------------


def test_enum_factory_roundtrip_sell():
    strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL)
    assert isinstance(strat, SellGeneStrategy)
    assert strat.name == "Sell"


def test_enum_factory_roundtrip_buy():
    strat = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY)
    assert isinstance(strat, BuyOrgStrategy)
    assert strat.name == "Buy"
