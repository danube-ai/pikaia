import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.sell_easy_strategy import SellEasyGeneStrategy
from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy
from pikaia.strategies.gs_strategies.sell_uniform_strategy import (
    SellUniformGeneStrategy,
)
from pikaia.strategies.os_strategies.buy_easy_strategy import BuyEasyOrgStrategy
from pikaia.strategies.os_strategies.buy_hard_strategy import BuyHardOrgStrategy
from pikaia.strategies.os_strategies.buy_uniform_strategy import BuyUniformOrgStrategy

# ---------------------------------------------------------------------------
# Canonical 3×4 dataset
# Gene 0: mean=1.0  (all solve it)
# Gene 1: mean=1/3  (hard)
# Gene 2: mean=2/3
# Gene 3: mean=2/3  (same mean as gene 2 but different organisms)
# ---------------------------------------------------------------------------

PERF = np.array(
    [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
)
N, M = PERF.shape


def _pop():
    return PikaiaPopulation(PERF)


def _make_ctx(pop, org_id, gene_fitness=None):
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
        gene_id=None,
    )


def _make_gene_ctx(pop, org_id, gene_id, gene_fitness=None):
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


# ---------------------------------------------------------------------------
# 1. Structural tests — shape, dtype, finite
# ---------------------------------------------------------------------------


class TestSellHardStructure:
    def test_name(self):
        assert SellHardGeneStrategy().name == "SellHard"

    def test_call_returns_float(self):
        pop = _pop()
        result = SellHardGeneStrategy()(_make_gene_ctx(pop, 0, 2))
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_finite_all_orgs_genes(self):
        pop = _pop()
        s = SellHardGeneStrategy()
        for i in range(N):
            for j in range(M):
                assert np.isfinite(s(_make_gene_ctx(pop, i, j)))

    def test_kernel_returns_none_d(self):
        pop = _pop()
        D, d = SellHardGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None
        assert d is not None
        assert d.shape == (M,)
        assert np.all(np.isfinite(d))


class TestSellUniformStructure:
    def test_name(self):
        assert SellUniformGeneStrategy().name == "SellUniform"

    def test_call_returns_float(self):
        pop = _pop()
        result = SellUniformGeneStrategy()(_make_gene_ctx(pop, 0, 2))
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_kernel_returns_none_d(self):
        pop = _pop()
        D, d = SellUniformGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None
        assert d is not None
        assert d.shape == (M,)
        assert np.all(np.isfinite(d))


class TestSellEasyStructure:
    def test_name(self):
        assert SellEasyGeneStrategy().name == "SellEasy"

    def test_call_returns_float(self):
        pop = _pop()
        result = SellEasyGeneStrategy()(_make_gene_ctx(pop, 0, 2))
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_kernel_returns_none_d(self):
        pop = _pop()
        D, d = SellEasyGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None
        assert d is not None
        assert d.shape == (M,)
        assert np.all(np.isfinite(d))


class TestBuyHardStructure:
    def test_name(self):
        assert BuyHardOrgStrategy().name == "BuyHard"

    def test_call_returns_array_M(self):
        pop = _pop()
        result = BuyHardOrgStrategy()(_make_ctx(pop, 0))
        assert result.shape == (M,)
        assert np.all(np.isfinite(result))

    def test_kernel_fallback_base_class(self):
        pop = _pop()
        D, d = BuyHardOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None
        assert d is None


class TestBuyUniformStructure:
    def test_name(self):
        assert BuyUniformOrgStrategy().name == "BuyUniform"

    def test_call_returns_array_M(self):
        pop = _pop()
        result = BuyUniformOrgStrategy()(_make_ctx(pop, 0))
        assert result.shape == (M,)
        assert np.all(np.isfinite(result))

    def test_kernel_fallback_base_class(self):
        pop = _pop()
        D, d = BuyUniformOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None
        assert d is None


class TestBuyEasyStructure:
    def test_name(self):
        assert BuyEasyOrgStrategy().name == "BuyEasy"

    def test_call_returns_array_M(self):
        pop = _pop()
        result = BuyEasyOrgStrategy()(_make_ctx(pop, 0))
        assert result.shape == (M,)
        assert np.all(np.isfinite(result))

    def test_kernel_fallback_base_class(self):
        pop = _pop()
        D, d = BuyEasyOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
        assert D is None
        assert d is None


# ---------------------------------------------------------------------------
# 2. Sign tests
# ---------------------------------------------------------------------------


def test_sell_hard_delta_nonpositive_when_xij_positive():
    pop = _pop()
    s = SellHardGeneStrategy()
    for i in range(N):
        for j in range(M):
            if PERF[i, j] > 0:
                assert s(_make_gene_ctx(pop, i, j)) <= 0


def test_sell_hard_delta_zero_when_xij_zero():
    pop = _pop()
    s = SellHardGeneStrategy()
    for i in range(N):
        for j in range(M):
            if PERF[i, j] == 0.0:
                assert np.isclose(s(_make_gene_ctx(pop, i, j)), 0.0, atol=1e-12)


def test_sell_uniform_delta_nonpositive_when_xij_positive():
    pop = _pop()
    s = SellUniformGeneStrategy()
    for i in range(N):
        for j in range(M):
            if PERF[i, j] > 0:
                assert s(_make_gene_ctx(pop, i, j)) <= 0


def test_buy_hard_delta_nonnegative():
    pop = _pop()
    s = BuyHardOrgStrategy()
    for i in range(N):
        result = s(_make_ctx(pop, i))
        assert np.all(result >= -1e-12)


def test_buy_uniform_delta_nonnegative():
    pop = _pop()
    s = BuyUniformOrgStrategy()
    for i in range(N):
        result = s(_make_ctx(pop, i))
        assert np.all(result >= -1e-12)


# ---------------------------------------------------------------------------
# 3. Kernel consistency: d == sum of __call__ over all organisms
# ---------------------------------------------------------------------------


def test_sell_hard_kernel_matches_call_sum():
    pop = _pop()
    s = SellHardGeneStrategy()
    _, d = s.kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    call_sum = np.zeros(M)
    for i in range(N):
        for j in range(M):
            call_sum[j] += s(_make_gene_ctx(pop, i, j))
    np.testing.assert_allclose(d, call_sum, atol=1e-10)


def test_sell_uniform_kernel_matches_call_sum():
    pop = _pop()
    s = SellUniformGeneStrategy()
    _, d = s.kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    call_sum = np.zeros(M)
    for i in range(N):
        for j in range(M):
            call_sum[j] += s(_make_gene_ctx(pop, i, j))
    np.testing.assert_allclose(d, call_sum, atol=1e-10)


def test_sell_easy_kernel_matches_call_sum():
    pop = _pop()
    s = SellEasyGeneStrategy()
    _, d = s.kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    call_sum = np.zeros(M)
    for i in range(N):
        for j in range(M):
            call_sum[j] += s(_make_gene_ctx(pop, i, j))
    np.testing.assert_allclose(d, call_sum, atol=1e-10)


# ---------------------------------------------------------------------------
# 4. Formula correctness
# ---------------------------------------------------------------------------


def test_sell_hard_kernel_formula():
    pop = _pop()
    mean_j = PERF.mean(axis=0)
    excl_j = 1.0 - mean_j
    expected = -mean_j * excl_j / (1.0 - excl_j + 1e-8)
    _, d = SellHardGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, expected, atol=1e-10)


def test_sell_uniform_kernel_formula():
    pop = _pop()
    mean_j = PERF.mean(axis=0)
    excl_j = 1.0 - mean_j
    # CalSim D2: sell=0 for trivially-solved (excl=0) or trivially-failed (excl=1) genes
    mask = (excl_j > 1e-6) & (excl_j < 1.0 - 1e-6)
    expected = -mean_j * mask.astype(float)
    _, d = SellUniformGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, expected, atol=1e-10)


def test_sell_easy_kernel_formula():
    pop = _pop()
    mean_j = PERF.mean(axis=0)
    excl_j = 1.0 - mean_j
    expected = mean_j * excl_j / (1.0 - excl_j + 1e-8)
    _, d = SellEasyGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, expected, atol=1e-10)


def test_buy_hard_no_d_matrix_kernel():
    # Buy strategies cannot be expressed in D-matrix form (capital depends on gamma).
    # kernel() falls back to base class returning (None, None).
    pop = _pop()
    D, d = BuyHardOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert D is None
    assert d is None


def test_buy_uniform_no_d_matrix_kernel():
    pop = _pop()
    D, d = BuyUniformOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert D is None
    assert d is None


def test_buy_easy_no_d_matrix_kernel():
    pop = _pop()
    D, d = BuyEasyOrgStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert D is None
    assert d is None


# ---------------------------------------------------------------------------
# 5. SELL_EASY = -SELL_HARD
# ---------------------------------------------------------------------------


def test_sell_easy_negates_sell_hard_kernel():
    pop = _pop()
    _, d_hard = SellHardGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    _, d_easy = SellEasyGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d_hard is not None
    assert d_easy is not None
    np.testing.assert_allclose(d_easy, -d_hard, atol=1e-10)


def test_sell_easy_negates_sell_hard_call():
    pop = _pop()
    sh = SellHardGeneStrategy()
    se = SellEasyGeneStrategy()
    for i in range(N):
        for j in range(M):
            ctx = _make_gene_ctx(pop, i, j)
            np.testing.assert_allclose(se(ctx), -sh(ctx), atol=1e-14)


# ---------------------------------------------------------------------------
# 7. BUY_EASY = -BUY_HARD
# ---------------------------------------------------------------------------


def test_buy_easy_negates_buy_hard_call():
    pop = _pop()
    bh = BuyHardOrgStrategy()
    be = BuyEasyOrgStrategy()
    for i in range(N):
        ctx = _make_ctx(pop, i)
        np.testing.assert_allclose(be(ctx), -bh(ctx), atol=1e-14)


# ---------------------------------------------------------------------------
# 8. End-to-end PikaiaModel run
# ---------------------------------------------------------------------------


def test_end_to_end_sell_hard_buy_hard():
    pop = _pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[SellHardGeneStrategy()],
        org_strategies=[BuyHardOrgStrategy()],
        max_iter=1,
    )
    model.fit()
    gf = model.gene_fitness_history[1]
    assert gf.shape == (M,)
    assert np.all(np.isfinite(gf))
    assert np.isclose(gf.sum(), 1.0, atol=1e-10)


def test_end_to_end_sell_uniform_buy_uniform():
    pop = _pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[SellUniformGeneStrategy()],
        org_strategies=[BuyUniformOrgStrategy()],
        max_iter=1,
    )
    model.fit()
    gf = model.gene_fitness_history[1]
    assert gf.shape == (M,)
    assert np.all(np.isfinite(gf))
    assert np.isclose(gf.sum(), 1.0, atol=1e-10)


def test_end_to_end_sell_easy_buy_easy():
    pop = _pop()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[SellEasyGeneStrategy()],
        org_strategies=[BuyEasyOrgStrategy()],
        max_iter=1,
    )
    model.fit()
    gf = model.gene_fitness_history[1]
    assert gf.shape == (M,)
    assert np.all(np.isfinite(gf))
    assert np.isclose(gf.sum(), 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Factory round-trips
# ---------------------------------------------------------------------------


def test_factory_roundtrip_sell_hard():
    s = GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL_HARD)
    assert isinstance(s, SellHardGeneStrategy)


def test_factory_roundtrip_sell_uniform():
    s = GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL_UNIFORM)
    assert isinstance(s, SellUniformGeneStrategy)


def test_factory_roundtrip_sell_easy():
    s = GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL_EASY)
    assert isinstance(s, SellEasyGeneStrategy)


def test_factory_roundtrip_buy_hard():
    s = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY_HARD)
    assert isinstance(s, BuyHardOrgStrategy)


def test_factory_roundtrip_buy_uniform():
    s = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY_UNIFORM)
    assert isinstance(s, BuyUniformOrgStrategy)


def test_factory_roundtrip_buy_easy():
    s = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY_EASY)
    assert isinstance(s, BuyEasyOrgStrategy)
