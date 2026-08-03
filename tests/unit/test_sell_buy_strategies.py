"""
Tests for SellOrgStrategy and BuyOrgStrategy.

The pair implements a market-recalibration signal within pikaia's replicator
framework, mirroring CalSim's proband.sellAll() / proband.buyAll() design.
Tests verify:

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
from pikaia.strategies.os_strategies.buy_strategy import BuyOrgStrategy
from pikaia.strategies.os_strategies.sell_strategy import SellOrgStrategy

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
# but has gene 3, so it redistributes capital to gene 2.
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
# start with both strategies as OrgStrategy (equal mixing weights 0.5 each).
#
# Derivation:
#   sell_d[j] = -mean_j * excl_j / (1 - excl_j)  → [0, -2/3, -1/3, -1/3]
#   buy_d[j]  = mean_j * Σ_i (1-x_ij) * C_i / Z_i → [0, 7/18, 1/9, 5/6]
#   net_d[j]  = 0.5*sell_d + 0.5*buy_d            → [0, -5/36, -1/9, 1/4]
#   γ_new = γ_0 * (1 + net_d), normalised
#         → [1/4, 31/144, 2/9, 5/16]
#         ≈ [0.2500, 0.2153, 0.2222, 0.3125]
EXPECTED_GF = np.array([0.25, 31.0 / 144, 2.0 / 9, 5.0 / 16])


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


def _pikaia_round_sell_buy() -> np.ndarray:
    """Run one SELL+BUY iteration from uniform start; return normalised gene fitness."""
    pop = _pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[],
        org_strategies=[
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELL),
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY),
        ],
        max_iter=1,
    )
    model.fit()
    return model.gene_fitness_history[1]


def _pikaia_round(gene_strat_enum, org_strat_enum) -> np.ndarray:
    """Run one iteration with the given GS+OS pair; return normalised gene fitness."""
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
# 1. SellOrgStrategy structural tests
# ---------------------------------------------------------------------------


class TestSellOrgStrategy:
    def test_returns_array_of_shape_M(self):
        """__call__ must return an (M,) array."""
        pop = _pop()
        strat = SellOrgStrategy()
        result = strat(_make_ctx(pop, org_id=0))
        assert result.shape == (M,)

    def test_returns_nonpositive_for_solved_gene(self):
        """Organism that solved a gene (x=1) produces a negative sell delta."""
        pop = _pop()
        strat = SellOrgStrategy()
        result = strat(_make_ctx(pop, org_id=0))
        # org 0: x=[1,0,1,1]; gene 0 excl=0 so delta=0; genes 2,3 solved → negative
        assert result[2] < 0
        assert result[3] < 0

    def test_returns_zero_for_unsolved_gene(self):
        """Organism that did not solve a gene (x=0) contributes nothing to sell."""
        pop = _pop()
        strat = SellOrgStrategy()
        result = strat(_make_ctx(pop, org_id=1))
        # org 1: x=[1,1,1,0]; gene 3 unsolved → zero
        assert np.isclose(result[3], 0.0, atol=1e-12)

    def test_zero_exclusiveness_gives_zero_delta(self):
        """Gene solved by everyone (excl=0) has zero sell signal for all organisms."""
        pop = _pop()
        strat = SellOrgStrategy()
        for org_id in range(pop.N):
            result = strat(_make_ctx(pop, org_id=org_id))
            assert np.isclose(result[0], 0.0, atol=1e-12)

    def test_sum_over_organisms_matches_formula(self):
        """Sum of __call__ over all organisms equals -mean_j * excl_j / (1-excl_j)."""
        pop = _pop()
        strat = SellOrgStrategy()
        mean_all = PERF.mean(axis=0)
        excl = 1.0 - mean_all
        expected_total = -mean_all * excl / (1.0 - excl + 1e-8)

        total = np.zeros(M)
        for i in range(N):
            total += strat(_make_ctx(pop, org_id=i))
        np.testing.assert_allclose(total, expected_total, atol=1e-10)

    def test_harder_gene_sells_more(self):
        """Harder gene (higher excl) has larger sell signal magnitude via kernel d."""
        pop = _pop()
        strat = SellOrgStrategy()
        _, d = strat.kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert d is not None
        assert abs(d[1]) > abs(d[2]), "Rare gene 1 should drain more than gene 2"
        assert abs(d[1]) > abs(d[3]), "Rare gene 1 should drain more than gene 3"

    def test_kernel_d_vector_matches_call_sum(self):
        """Kernel d-vector must equal the sum of __call__ over all organisms."""
        pop = _pop()
        strat = SellOrgStrategy()
        _, d = strat.kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert d is not None

        call_sum = np.zeros(M)
        for i in range(N):
            call_sum += strat(_make_ctx(pop, org_id=i))
        np.testing.assert_allclose(d, call_sum, atol=1e-10)

    def test_kernel_D_is_none(self):
        """SellOrgStrategy has no bilinear D term."""
        pop = _pop()
        D, _ = SellOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
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
    sell_strat = SellOrgStrategy()
    buy_strat = BuyOrgStrategy()

    sell_total = np.zeros(M)
    buy_total = np.zeros(M)
    for i in range(N):
        sell_total += sell_strat(_make_ctx(pop, org_id=i))
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
    so any diagonal strategy treats them identically.  BuyOrgStrategy must produce
    different buy deltas for them because they are held by *different* organisms.
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
    gf_sb = _pikaia_round_sell_buy()
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

    Both strategies run as OrgStrategy with equal mixing weights (0.5 each):
      net_d = 0.5*sell_d + 0.5*buy_d
      γ_new = γ_0 * (1 + net_d), normalised → [1/4, 31/144, 2/9, 5/16]
    """
    gf = _pikaia_round_sell_buy()
    np.testing.assert_allclose(gf, EXPECTED_GF, atol=1e-4, rtol=0)


# ---------------------------------------------------------------------------
# 6. Enum and factory round-trip
# ---------------------------------------------------------------------------


def test_enum_factory_roundtrip_sell():
    strat = OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELL)
    assert isinstance(strat, SellOrgStrategy)
    assert strat.name == "Sell"


def test_enum_factory_roundtrip_buy():
    strat = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY)
    assert isinstance(strat, BuyOrgStrategy)
    assert strat.name == "Buy"
