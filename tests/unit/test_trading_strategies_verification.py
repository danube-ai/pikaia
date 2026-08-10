"""Verification: the trading sell/buy strategy pairs reproduce the reference model.

The ground truth is :func:`fixtures.trading_reference.trading_recalibrate`, a
self-contained port of the original trading simulator (no external dependency).
Each pikaia pair is checked against its reference sell strategy:

    SellHard    + BuyHard    <->  "hard"
    SellUniform + BuyUniform <->  "uniform"
    SellEasy    + BuyEasy    <->  "easy"
"""

import numpy as np
import pytest
from fixtures.trading_reference import trading_recalibrate_k

from pikaia.data.population import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.strategies.gs_strategies.sell_easy_strategy import SellEasyGeneStrategy
from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy
from pikaia.strategies.gs_strategies.sell_uniform_strategy import (
    SellUniformGeneStrategy,
)
from pikaia.strategies.os_strategies.buy_easy_strategy import BuyEasyOrgStrategy
from pikaia.strategies.os_strategies.buy_hard_strategy import BuyHardOrgStrategy
from pikaia.strategies.os_strategies.buy_uniform_strategy import BuyUniformOrgStrategy

PERF = np.array(
    [
        [1.0, 0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ]
)
N, M = PERF.shape
START_VALUES = [5.0, 5.0, 5.0, 5.0]


def _reference_gene_fitness(sell_strategy, k):
    """Reference per-gene values after ``k`` rounds, normalised to sum 1."""
    values = trading_recalibrate_k(PERF, START_VALUES, sell_strategy, k)
    return values / values.sum()


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
# SELL_HARD delta matches the reference "hard" sell signal
# ---------------------------------------------------------------------------


def test_sell_hard_delta_formula():
    # Reference "hard" sell loss per unit: exclusiveness / (1 - exclusiveness) / N.
    # pikaia SELL_HARD kernel d[j] = -mean_j * excl_j / (1 - excl_j + eps).
    mean_j = PERF.mean(axis=0)
    excl = 1.0 - mean_j
    expected_d = -mean_j * excl / (1.0 - excl + 1e-8)

    pop = PikaiaPopulation(PERF)
    _, d = SellHardGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, expected_d, atol=1e-10)


def test_buy_hard_delta_formula():
    # BUY_HARD: C_i = sum_k x_ik * sell_signal_k / N,  Z_i = sum_k (1-x_ik) * mean_k.
    # __call__ returns a proportional delta = buy_abs / gamma_j; at uniform
    # gamma = 1/M the 1/gamma_j factor cancels the 1/M capital scaling, so the
    # summed __call__ equals the buy_abs vector.
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
    np.testing.assert_allclose(call_sum, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# One-iteration exact match: SellHard + BuyHard
# ---------------------------------------------------------------------------


def test_one_iteration_exact_match_hard():
    reference = _reference_gene_fitness("hard", 1)
    pikaia_gf = _pikaia_one_iter(SellHardGeneStrategy(), BuyHardOrgStrategy())
    np.testing.assert_allclose(pikaia_gf, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# One-iteration exact match: SellUniform + BuyUniform
# ---------------------------------------------------------------------------


def test_one_iteration_exact_match_uniform():
    reference = _reference_gene_fitness("uniform", 1)
    pikaia_gf = _pikaia_one_iter(SellUniformGeneStrategy(), BuyUniformOrgStrategy())
    np.testing.assert_allclose(pikaia_gf, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# SELL_UNIFORM delta matches the reference "uniform" sell signal
# ---------------------------------------------------------------------------


def test_sell_uniform_delta_formula():
    # Reference "uniform": vdeltaSell_j = startValue/N when 0 < excl_j < 1, else 0.
    # pikaia SELL_UNIFORM d[j] = -mean_j for 0 < excl_j < 1, else 0.
    mean_j = PERF.mean(axis=0)
    excl_j = 1.0 - mean_j
    mask = (excl_j > 1e-6) & (excl_j < 1.0 - 1e-6)
    expected_d = -mean_j * mask.astype(float)

    pop = PikaiaPopulation(PERF)
    _, d = SellUniformGeneStrategy().kernel(pop, np.eye(M), np.eye(N), 1.0)
    assert d is not None
    np.testing.assert_allclose(d, expected_d, atol=1e-10)


def test_buy_uniform_delta_formula():
    # BUY_UNIFORM: C_i = sum_{k: 0<excl_k<1} x_ik * gamma_k / N,
    # Z_i = sum_k (1-x_ik) * excl_k. __call__ returns proportional delta;
    # at uniform gamma = 1/M, sum_i __call__ * gamma = buy_abs.
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
    np.testing.assert_allclose(call_sum * gamma, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# Long-run validity: SellHard + BuyHard stays finite and well-formed
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
    # Gene 3 (solved by orgs 0 and 2) should rank above gene 2 (solved only by 0, 1).
    assert gf[3] > gf[2]


# ---------------------------------------------------------------------------
# Multi-iteration convergence: pikaia matches the reference at k > 1.
# This verifies the proportional buy delta (buy_abs / gamma) correctly
# reproduces the reference's additive dynamics under replicator normalisation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [5, 10, 20, 50, 100])
def test_multi_iter_convergence_hard(k):
    """SellHard + BuyHard matches the reference "hard" model at many iterations."""
    reference = _reference_gene_fitness("hard", k)
    pikaia_gf = _pikaia_k_iters(SellHardGeneStrategy, BuyHardOrgStrategy, k)
    np.testing.assert_allclose(
        pikaia_gf,
        reference,
        atol=1e-5,
        err_msg=f"Mismatch at k={k}: pikaia={pikaia_gf}, reference={reference}",
    )


@pytest.mark.parametrize("k", [5, 10, 20, 50, 100])
def test_multi_iter_convergence_uniform(k):
    """SellUniform + BuyUniform matches the reference "uniform" model."""
    reference = _reference_gene_fitness("uniform", k)
    pikaia_gf = _pikaia_k_iters(SellUniformGeneStrategy, BuyUniformOrgStrategy, k)
    np.testing.assert_allclose(
        pikaia_gf,
        reference,
        atol=1e-5,
        err_msg=f"Mismatch at k={k}: pikaia={pikaia_gf}, reference={reference}",
    )


# ---------------------------------------------------------------------------
# One-iteration exact match: SellEasy + BuyEasy
# ---------------------------------------------------------------------------


def test_one_iteration_exact_match_easy():
    reference = _reference_gene_fitness("easy", 1)
    pikaia_gf = _pikaia_one_iter(SellEasyGeneStrategy(), BuyEasyOrgStrategy())
    np.testing.assert_allclose(pikaia_gf, reference, atol=1e-6)


# ---------------------------------------------------------------------------
# Multi-iteration convergence: SellEasy + BuyEasy.
#
# The "easy" (inverse) pair is inherently divergent — easy genes grow without
# bound while hard genes go negative, so per-gene values grow exponentially and
# an absolute tolerance is meaningless at large k. We compare the normalised
# vectors with a relative tolerance and cap at k=10, where the dynamics are
# still numerically well-conditioned.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("k", [1, 5, 10])
def test_multi_iter_convergence_easy(k):
    """SellEasy + BuyEasy matches the reference "easy" model up to k=10 (rtol=1e-4)."""
    reference = _reference_gene_fitness("easy", k)
    pikaia_gf = _pikaia_k_iters(SellEasyGeneStrategy, BuyEasyOrgStrategy, k)
    np.testing.assert_allclose(
        pikaia_gf,
        reference,
        rtol=1e-4,
        err_msg=f"Mismatch at k={k}: pikaia={pikaia_gf}, reference={reference}",
    )
