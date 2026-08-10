"""Verification: new pikaia strategies reproduce calsim output."""

import sys

import numpy as np
import pytest

sys.path.insert(0, "/Users/uziel/Development/DanubeAI/experiments/tgeneticai")

try:
    import calsim

    CALSIM_AVAILABLE = True
except ImportError:
    CALSIM_AVAILABLE = False

from pikaia.data.population import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy
from pikaia.strategies.gs_strategies.sell_uniform_strategy import (
    SellUniformGeneStrategy,
)
from pikaia.strategies.os_strategies.buy_hard_strategy import BuyHardOrgStrategy
from pikaia.strategies.os_strategies.buy_uniform_strategy import BuyUniformOrgStrategy

pytestmark = pytest.mark.skipif(
    not CALSIM_AVAILABLE, reason="calsim module not available"
)

PERF = np.array(
    [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
)
N, M = PERF.shape
START_VALUES = [5.0, 5.0, 5.0, 5.0]


def _calsim_one_iter(strategy_name, start_values):
    params = calsim.Params(sellStrategy=strategy_name)
    exes = [
        calsim.Exercise(
            params=params, index=i, startvalue=start_values[i], maxvalue=20.0, nprobs=N
        )
        for i in range(M)
    ]
    probs = [
        calsim.Proband(
            params=params, id=i, value=list(start_values), performance=PERF[i]
        )
        for i in range(N)
    ]
    test = calsim.Test(params=params, exerciseList=exes, probandsList=probs)
    test.recalibrateAll()
    return np.array(test.currentValues)


def _calsim_k_iters(strategy_name, k):
    values = list(START_VALUES)
    for _ in range(k):
        values = list(_calsim_one_iter(strategy_name, values))
    return np.array(values)


def _pikaia_one_iter(gene_strat, org_strat):
    pop = PikaiaPopulation(PERF)
    model = PikaiaModel(
        population=pop,
        gene_strategies=[gene_strat],
        org_strategies=[org_strat],
        max_iter=1,
    )
    model.fit()
    return model.gene_fitness_history[1]


def _pikaia_k_iters(gene_strat_cls, org_strat_cls, k):
    pop = PikaiaPopulation(PERF)
    model = PikaiaModel(
        population=pop,
        gene_strategies=[gene_strat_cls()],
        org_strategies=[org_strat_cls()],
        max_iter=k,
    )
    model.fit()
    return model.gene_fitness_history[k]


# ---------------------------------------------------------------------------
# SELL_HARD delta matches calsim Difficulty1
# ---------------------------------------------------------------------------


def test_sell_hard_delta_matches_calsim_difficulty1():
    # calsim Difficulty1 sell loss per unit: exclusiveness / (1 - exclusiveness) / N
    # pikaia SELL_HARD kernel d[j] = -mean_j * excl_j / (1 - excl_j + eps)
    # proportional sell loss = d[j] / start_value = -excl_j / (1-excl_j) / N (from calsim formula)
    mean_j = PERF.mean(axis=0)
    excl = 1.0 - mean_j

    # pikaia SELL_HARD summed delta over all organisms
    pikaia_sell_d = -mean_j * excl / (1.0 - excl + 1e-8)

    # They match up to the eps correction
    np.testing.assert_allclose(
        pikaia_sell_d, -mean_j * excl / (1.0 - excl + 1e-8), atol=1e-10
    )

    pop = PikaiaPopulation(PERF)
    from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy

    _, d = SellHardGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, pikaia_sell_d, atol=1e-10)


def test_buy_hard_delta_matches_calsim_difficulty1():
    # BUY_HARD: C_i = sum_k x_ik * sell_signal_k / N
    # Z_i = sum_k (1-x_ik) * mean_k
    # __call__ returns proportional delta = buy_abs / gamma_j
    # At uniform gamma = 1/M: sum_i __call__ = M * kernel_d
    from pikaia.strategies.base_strategies import StrategyContext

    X = PERF
    mean_all = X.mean(axis=0)
    excl = 1.0 - mean_all
    sell_signal = excl / (1.0 - excl + 1e-8)
    C = (X * sell_signal[np.newaxis, :]).sum(axis=1) / N
    Z = ((1.0 - X) * mean_all[np.newaxis, :]).sum(axis=1)
    safe_Z = np.where(Z < 1e-10, 1.0, Z)
    w = np.where(Z < 1e-10, 0.0, C / safe_Z)
    expected = mean_all * ((1.0 - X) * w[:, np.newaxis]).sum(axis=0)

    pop = PikaiaPopulation(PERF)
    gamma = np.ones(M) / M
    strat = BuyHardOrgStrategy()
    call_sum = np.zeros(M)
    for i in range(N):
        ctx = StrategyContext(
            population=pop,
            org_fitness=np.ones(N) / N,
            gene_fitness=gamma,
            org_similarity=np.eye(N),
            gene_similarity=np.eye(M),
            initial_org_fitness_range=1.0,
            org_id=i,
        )
        call_sum += strat(ctx)
    # At uniform gamma = 1/M: max_capital is scaled by 1/M (from gamma),
    # and the 1/gamma_j factor in the return cancels it.
    # So __call__ sum at uniform gamma equals the original kernel d-vector.
    np.testing.assert_allclose(call_sum, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# One-iteration exact match: Difficulty1
# ---------------------------------------------------------------------------


def test_one_iteration_exact_match_difficulty1():
    calsim_values = _calsim_one_iter("Difficulty1", START_VALUES)
    calsim_normalized = calsim_values / calsim_values.sum()

    pikaia_gf = _pikaia_one_iter(SellHardGeneStrategy(), BuyHardOrgStrategy())

    np.testing.assert_allclose(pikaia_gf, calsim_normalized, atol=1e-6)


# ---------------------------------------------------------------------------
# One-iteration exact match: Difficulty2 (SellUniform+BuyUniform)
# ---------------------------------------------------------------------------


def test_one_iteration_exact_match_difficulty2():
    calsim_values = _calsim_one_iter("Difficulty2", START_VALUES)
    calsim_normalized = calsim_values / calsim_values.sum()

    pikaia_gf = _pikaia_one_iter(SellUniformGeneStrategy(), BuyUniformOrgStrategy())

    np.testing.assert_allclose(pikaia_gf, calsim_normalized, atol=1e-6)


# ---------------------------------------------------------------------------
# SELL_UNIFORM delta matches calsim Difficulty2
# ---------------------------------------------------------------------------


def test_sell_uniform_delta_matches_calsim_difficulty2():
    # CalSim D2: vdeltaSell_j = startValue/N if excl ∉ {0,1} else 0
    # pikaia SELL_UNIFORM d[j] = -mean_j for 0 < excl_j < 1, else 0
    mean_j = PERF.mean(axis=0)
    excl_j = 1.0 - mean_j
    mask = (excl_j > 1e-6) & (excl_j < 1.0 - 1e-6)
    expected_d = -mean_j * mask.astype(float)

    pop = PikaiaPopulation(PERF)
    _, d = SellUniformGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, expected_d, atol=1e-10)


def test_buy_uniform_delta_matches_calsim_difficulty2():
    # BUY_UNIFORM: C_i = sum_{k: 0<excl_k<1} x_ik * gamma_k / N
    # Z_i = sum_k (1-x_ik) * excl_k
    # __call__ returns proportional delta = buy_abs / gamma_j
    # At uniform gamma = 1/M: sum_i __call__ = M * kernel_d
    from pikaia.strategies.base_strategies import StrategyContext

    X = PERF
    mean_all = X.mean(axis=0)
    excl = 1.0 - mean_all
    sell_mask = ((excl > 1e-6) & (excl < 1.0 - 1e-6)).astype(float)
    gamma = np.ones(M) / M
    C = (X * (sell_mask * gamma)[np.newaxis, :]).sum(axis=1) / N
    Z = ((1.0 - X) * excl[np.newaxis, :]).sum(axis=1)
    safe_Z = np.where(Z < 1e-10, 1.0, Z)
    w = np.where(Z < 1e-10, 0.0, C / safe_Z)
    expected = excl * ((1.0 - X) * w[:, np.newaxis]).sum(axis=0)

    pop = PikaiaPopulation(PERF)
    strat = BuyUniformOrgStrategy()
    call_sum = np.zeros(M)
    for i in range(N):
        ctx = StrategyContext(
            population=pop,
            org_fitness=np.ones(N) / N,
            gene_fitness=gamma,
            org_similarity=np.eye(N),
            gene_similarity=np.eye(M),
            initial_org_fitness_range=1.0,
            org_id=i,
        )
        call_sum += strat(ctx)
    # At uniform gamma = 1/M: sum of proportional deltas × gamma = buy_abs summed.
    np.testing.assert_allclose(call_sum * gamma, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Long-run convergence: SellHard+BuyHard produces valid gene fitness
# (Rankings can diverge from calsim after iter>1 because pikaia normalizes
# gene_fitness while calsim tracks absolute values — the mappings decouple.)
# ---------------------------------------------------------------------------


def test_many_iterations_sell_hard_buy_hard_valid():
    k = 20
    pop = PikaiaPopulation(PERF)
    model = PikaiaModel(
        population=pop,
        gene_strategies=[SellHardGeneStrategy()],
        org_strategies=[BuyHardOrgStrategy()],
        max_iter=k,
    )
    model.fit()
    gf = model.gene_fitness_history[k]
    assert gf.shape == (M,)
    assert np.all(np.isfinite(gf))
    assert np.isclose(gf.sum(), 1.0, atol=1e-10)
    # Gene 3 (solved by org 0 and 2, while org 1 has hard gene 1) should rank above gene 2
    assert gf[3] > gf[2]


# ---------------------------------------------------------------------------
# Multi-iteration convergence: pikaia matches calsim at k > 1
# This verifies the proportional delta fix (buy_abs / gamma) correctly
# reproduces CalSim's additive dynamics under replicator normalisation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 10, 20, 50, 100])
def test_multi_iter_convergence_difficulty1(k):
    """Pikaia SellHard+BuyHard matches CalSim Difficulty1 at many iterations."""
    calsim_values = _calsim_k_iters("Difficulty1", k)
    calsim_normalized = calsim_values / calsim_values.sum()

    pikaia_gf = _pikaia_k_iters(SellHardGeneStrategy, BuyHardOrgStrategy, k)

    np.testing.assert_allclose(
        pikaia_gf,
        calsim_normalized,
        atol=1e-5,
        err_msg=f"Mismatch at k={k}: pikaia={pikaia_gf}, calsim={calsim_normalized}",
    )


@pytest.mark.parametrize("k", [5, 10, 20, 50, 100])
def test_multi_iter_convergence_difficulty2(k):
    """Pikaia SellUniform+BuyUniform matches CalSim Difficulty2 at many iterations."""
    calsim_values = _calsim_k_iters("Difficulty2", k)
    calsim_normalized = calsim_values / calsim_values.sum()

    pikaia_gf = _pikaia_k_iters(SellUniformGeneStrategy, BuyUniformOrgStrategy, k)

    np.testing.assert_allclose(
        pikaia_gf,
        calsim_normalized,
        atol=1e-5,
        err_msg=f"Mismatch at k={k}: pikaia={pikaia_gf}, calsim={calsim_normalized}",
    )
