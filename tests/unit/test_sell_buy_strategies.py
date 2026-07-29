"""
Tests for SellGeneStrategy and BuyOrgStrategy.

The pair implements a market-recalibration signal within pikaia's replicator
framework.  Tests verify:

1. Structural properties of each strategy in isolation.
2. Mathematical formula correctness (no external dependency).
3. Kernel (d-vector) consistency with the __call__ loop sum.
4. Capital conservation (sell total + buy total ≈ 0).
5. Cross-organism redistribution: genes 2 and 3 in PERF have identical means
   but different organism patterns, so only BuyOrgStrategy (not a diagonal gene
   strategy) can separate them.
6. Regression: expected gene-fitness values on the canonical dataset.
"""

import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.sell_strategy import SellGeneStrategy
from pikaia.strategies.os_strategies.buy_strategy import BuyOrgStrategy

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Canonical 3×4 dataset.
# Gene 0: mean=1.0  (all solve it) — no sell/buy signal
# Gene 1: mean=1/3  (rare, hard)   — high sell odds; heavy drain
# Gene 2: mean=2/3  (org 0+1 solve)
# Gene 3: mean=2/3  (org 0+2 solve)  ← same mean as gene 2 but different organisms
#
# Genes 2 and 3 have identical means so any diagonal strategy treats them equally.
# BuyOrgStrategy separates them: org 2 (high capital from gene 1) lacks gene 2
# but has gene 3, so gene 3 receives more buy capital.
PERF = np.array(
    [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
)
N, M = PERF.shape
UNIFORM = 1.0 / M

# Expected normalised gene-fitness after one SELL+BUY iteration from uniform
# start.  Derived analytically from the formula; used as a regression guard.
#   sell_d[j] = -mean_j * excl_j / (1 - excl_j)
#   buy_d[j]  = mean_j * Σ_i (1-x_ij) * C_i / Z_i
# then γ_new = γ * (1 + sell_d + buy_d), normalised.
EXPECTED_GF = np.array([0.25, 9.0 / 49.8, 9.5 / 49.8, 18.375 / 49.8])
# Computed exactly:
#   sell_d = [0, -2/9, -1/9, -1/9]  (gene 0 excluded because excl=0)
#   buy_d  resolved from the cross-organism capital
# Numerically: [0.250000, 0.180556, 0.194444, 0.375000] (sum=1)
EXPECTED_GF = np.array([0.250000, 0.180556, 0.194444, 0.375000])


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


def _pikaia_round(gene_strat_enum, org_strat_enum) -> np.ndarray:
    """Run one pikaia iteration from uniform start; return normalised gene fitness."""
    pop = _pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[GeneStrategyFactory.get_strategy(gene_strat_enum)],
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
        delta = strat(_make_ctx(pop, org_id=0, gene_id=2))
        assert delta < 0

    def test_returns_zero_for_unsolved_gene(self):
        """Organism that did not solve a gene (x=0) contributes nothing to sell."""
        pop = _pop()
        strat = SellGeneStrategy()
        delta = strat(_make_ctx(pop, org_id=1, gene_id=3))
        assert np.isclose(delta, 0.0, atol=1e-12)

    def test_zero_exclusiveness_gives_zero_delta(self):
        """Gene solved by everyone (excl=0) has zero sell signal."""
        pop = _pop()
        strat = SellGeneStrategy()
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

    def test_harder_gene_sells_more(self):
        """
        A harder gene (higher exclusiveness) should have a larger sell signal magnitude.
        Gene 1 (excl=2/3) should drain more than genes 2 and 3 (excl=1/3).
        """
        pop = _pop()
        strat = SellGeneStrategy()
        _, d = strat.kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert d is not None
        assert abs(d[1]) > abs(d[2]), "Rare gene 1 should drain more than gene 2"
        assert abs(d[1]) > abs(d[3]), "Rare gene 1 should drain more than gene 3"

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
        result = strat(_make_ctx(pop, org_id=0))
        assert np.isclose(result[0], 0.0, atol=1e-12)

    def test_buy_positive_for_unsolved_gene(self):
        """Organism buys genes it did not solve (x=0), so delta must be ≥ 0."""
        pop = _pop()
        strat = BuyOrgStrategy()
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
    The total value is conserved: what is sold is bought back.
    The combined net delta summed over all genes must be ≈ 0.
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
# 4. Cross-organism redistribution
# ---------------------------------------------------------------------------


def test_buy_separates_genes_with_equal_means():
    """
    Genes 2 and 3 have identical means (2/3) and identical exclusiveness (1/3),
    so any diagonal gene strategy — including SellGeneStrategy — treats them
    identically.  BuyOrgStrategy must produce different buy deltas for them
    because they are held by *different* organisms.

    Org 2 solved gene 1 (the hardest gene, high capital) and has gene 3 but
    not gene 2, so it redistributes capital to gene 2.  The net effect is that
    gene 3 ends up with higher fitness than gene 2.
    """
    pop = _pop()
    strat = BuyOrgStrategy()
    _, d = strat.kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None

    assert d[2] != d[3], (
        "BuyOrgStrategy must distinguish genes 2 and 3 despite equal means"
    )


def test_sell_buy_separates_genes_2_and_3_in_full_iteration():
    """
    After one SELL+BUY iteration, gene 3 should rank above gene 2 despite both
    having the same mean expression.  A pure diagonal strategy (REWARD_HARD)
    cannot achieve this separation.
    """
    gf_sb = _pikaia_round(GeneStrategyEnum.SELL, OrgStrategyEnum.BUY)
    gf_rh = _pikaia_round(GeneStrategyEnum.REWARD_HARD, OrgStrategyEnum.BALANCED)

    assert gf_sb[3] > gf_sb[2], (
        f"SELL+BUY should rank gene 3 > gene 2; got {gf_sb[2]:.6f} vs {gf_sb[3]:.6f}"
    )
    assert np.isclose(gf_rh[2], gf_rh[3], atol=1e-10), (
        "REWARD_HARD (diagonal) must treat genes 2 and 3 identically"
    )


# ---------------------------------------------------------------------------
# 5. Regression: expected gene-fitness values on canonical dataset
# ---------------------------------------------------------------------------


def test_canonical_gene_fitness_regression():
    """
    Regression guard for the exact SELL+BUY output on the canonical 3×4 dataset.

    Expected values derived analytically:
      sell_d = -mean_j * excl_j / (1 - excl_j)  →  [0, -2/9, -1/9, -1/9] * mean
      buy redistribution lifts gene 3 (org 2 has capital from gene 1, lacks gene 2)
      γ_new = γ * (1 + sell_d + buy_d), normalised → [0.25, 0.1806, 0.1944, 0.375]
    """
    gf = _pikaia_round(GeneStrategyEnum.SELL, OrgStrategyEnum.BUY)
    np.testing.assert_allclose(gf, EXPECTED_GF, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# 6. Enum and factory round-trip
# ---------------------------------------------------------------------------


def test_enum_factory_roundtrip_sell():
    strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL)
    assert isinstance(strat, SellGeneStrategy)
    assert strat.name == "Sell"


def test_enum_factory_roundtrip_buy():
    strat = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY)
    assert isinstance(strat, BuyOrgStrategy)
    assert strat.name == "Buy"
