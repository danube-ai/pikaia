"""
Deep verification: pikaia trading strategies vs tgeneticai/calsim.py

This module compares the pikaia implementations against the original
experiments/tgeneticai/calsim.py formulas to ensure the port is correct.

## calsim.py formulas (Exercise.calculateDelta)

    exclusiveness_j = (1/N) * sum(1 - performance_ij)   # fraction who didn't solve

    Difficulty1 (REWARD_HARD):
        vdeltaSell_j = startValue * exclusiveness_j / (1 - exclusiveness_j) / N

    Inverse (REWARD_EASY):
        vdeltaSell_j = startValue * -exclusiveness_j / (1 - exclusiveness_j) / N

    Mixed (VALUATION_BLEND):
        blends Difficulty1, Difficulty2, and Inverse with mixFactors

    DeltaFunction.linear(interval, value) = interval * value

## pikaia mapping

    mean_j = matrix.mean(axis=0)   # average expression of gene j
    exclusiveness_j = 1 - mean_j   # same concept: fraction who didn't "score high"
    odds_j = exclusiveness_j / (1 - exclusiveness_j + eps)
    difficulty_j = odds_j / max(odds_k)  # normalized odds ratio ∈ [0, 1]

    REWARD_HARD delta_ij = (16/N) * difficulty_j * gene_fitness_j * (x_ij - 0.5)
    REWARD_EASY delta_ij = -(16/N) * difficulty_j * gene_fitness_j * (x_ij - 0.5)
    VALUATION_BLEND delta_ij = (16/N) * sign * difficulty_j * ...
        where sign = 2*preference - 1  (0 → easy, 1 → hard)

## Key mathematical properties to verify

1. DIFFICULTY MONOTONICITY: pikaia difficulty and calsim difficulty have the
   same ordering (both increase monotonically with exclusiveness).

2. INVERSE RELATIONSHIP: REWARD_EASY = -REWARD_HARD exactly.

3. BLEND LINEARITY: VALUATION_BLEND(p) = p * REWARD_HARD + (1-p) * REWARD_EASY.

4. EDGE BEHAVIOR: at boundaries (all-same, all-zero, all-one), no NaN/Inf.

5. CROSS-STRATEGY CONSISTENCY: given the same difficulty signal, the signs
   are consistent with the calsim intent (hard → positive delta for REWARD_HARD).
"""

import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.schemas import (
    GeneStrategyEnum,
)
from pikaia.strategies import GeneStrategyFactory
from pikaia.strategies.base_strategies import StrategyContext

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _calsim_difficulty(exclusiveness, valid, eps=1e-8):
    """
    Original calsim difficulty formula (without startValue factor).

    calsim.py line ~318:
        self._dfunc.linear(startValue, exclusiveness/(1.0-self._exclusiveness)/valid)

    Linear() multiplies by startValue, so the per-unit formula is:
        difficulty = exclusiveness / (1 - exclusiveness) / valid
    """
    return exclusiveness / ((1.0 - exclusiveness) + eps) / valid


def _pikaia_difficulty(exclusiveness, eps=1e-8):
    """
    Pikaia difficulty formula: normalized odds ratio.

    odds_j = exclusiveness_j / (1 - exclusiveness_j + eps)
    difficulty_j = odds_j / max(odds_k)
    """
    odds = exclusiveness / (1.0 - exclusiveness + eps)
    return odds / (odds.max() + eps)


def _make_ctx(pop, org_id, gene_id, gene_fitness=None):
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
# Test 1: Difficulty formula matches calsim ordering (synthetic data)
# ---------------------------------------------------------------------------
def test_difficulty_ordering_synthetic():
    """
    Verify pikaia difficulty preserves the same gene ordering as calsim.

    Use 3 probands, 4 exercises with known exclusiveness values:
        [1,0,1,1]  → exercise 0: 2/3 solved → exclusiveness=1/3
        [1,1,1,0]  → exercise 1: 1/3 solved → exclusiveness=2/3
        [1,0,0,1]  → exercise 2: 2/3 solved → exclusiveness=1/3
    """
    # Performance matrix: 1=solved, 0=not solved
    performance = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    N = performance.shape[0]
    # In calsim, exclusiveness = fraction who DID NOT solve
    exclusiveness = (1.0 - performance).mean(axis=0)  # [1/3, 2/3, 2/3, 1/3]
    valid = N

    calsim_diff = _calsim_difficulty(exclusiveness, valid)
    pikaia_diff = _pikaia_difficulty(exclusiveness)

    # Both should rank exercise 1 (highest exclusiveness) as hardest
    assert calsim_diff[1] > calsim_diff[0]
    assert calsim_diff[1] > calsim_diff[2]
    assert pikaia_diff[1] > pikaia_diff[0]
    assert pikaia_diff[1] > pikaia_diff[2]

    # Orderings should be identical
    assert np.array_equal(np.argsort(calsim_diff), np.argsort(pikaia_diff))


# ---------------------------------------------------------------------------
# Test 2: Difficulty is monotonically increasing in exclusiveness
# ---------------------------------------------------------------------------
def test_difficulty_monotonicity():
    """
    Both calsim and pikaia difficulty should increase monotonically as
    exclusiveness increases from 0 to 1.
    """
    exclusivenesses = np.linspace(0.01, 0.99, 20)
    calsim_diffs = _calsim_difficulty(exclusivenesses, valid=1)
    pikaia_diffs = _pikaia_difficulty(exclusivenesses)

    # Both should be strictly increasing
    assert np.all(np.diff(calsim_diffs) > 0), "calsim difficulty not monotonic"
    assert np.all(np.diff(pikaia_diffs) > 0), "pikaia difficulty not monotonic"

    # Cross-mapping: higher exclusiveness should always rank the same
    calsim_order = np.argsort(calsim_diffs)
    pikaia_order = np.argsort(pikaia_diffs)
    assert np.array_equal(calsim_order, pikaia_order)


# ---------------------------------------------------------------------------
# Test 3: REWARD_EASY is exactly -REWARD_HARD at every point
# ---------------------------------------------------------------------------
def test_inverse_at_scale():
    """
    For every (organism, gene) pair in a population, REWARD_EASY should
    be exactly the negation of REWARD_HARD.
    """
    np.random.seed(42)
    data = np.random.rand(20, 8)
    pop = PikaiaPopulation(data)

    hard_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_HARD)
    easy_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_EASY)

    for org_id in range(pop.N):
        for gene_id in range(pop.M):
            h = hard_strat(_make_ctx(pop, org_id, gene_id))
            e = easy_strat(_make_ctx(pop, org_id, gene_id))
            assert np.isclose(h, -e, atol=1e-15), (
                f"org={org_id}, gene={gene_id}: {h} != -{e}"
            )


# ---------------------------------------------------------------------------
# Test 4: VALUATION_BLEND is linear interpolation of HARD and EASY
# ---------------------------------------------------------------------------
def test_blend_is_exact_interpolation():
    """
    For every (preference, organism, gene), VALUATION_BLEND should equal
    preference * REWARD_HARD + (1 - preference) * REWARD_EASY.
    """
    np.random.seed(42)
    data = np.random.rand(15, 6)
    pop = PikaiaPopulation(data)

    hard_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_HARD)
    easy_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_EASY)
    blend_strat = GeneStrategyFactory.get_strategy(
        GeneStrategyEnum.VALUATION_BLEND, preference=0.0
    )

    prefs = np.linspace(0.0, 1.0, 11)
    for pref in prefs:
        blend_strat.options["preference"] = pref
        for org_id in range(pop.N):
            for gene_id in range(pop.M):
                blend = blend_strat(_make_ctx(pop, org_id, gene_id))
                hard = hard_strat(_make_ctx(pop, org_id, gene_id))
                easy = easy_strat(_make_ctx(pop, org_id, gene_id))
                expected = pref * hard + (1.0 - pref) * easy
                assert np.isclose(blend, expected, atol=1e-14), (
                    f"pref={pref}, org={org_id}, gene={gene_id}: "
                    f"blend={blend:.10e} != expected={expected:.10e}"
                )


# ---------------------------------------------------------------------------
# Test 5: Kernel diagonal equals the difficulty signal times (16/M)
# ---------------------------------------------------------------------------
def test_kernel_diagonal_formula():
    """
    For REWARD_HARD: D[j,j] = (16/M) * difficulty_j
    For REWARD_EASY: D[j,j] = -(16/M) * difficulty_j
    For VALUATION_BLEND: D[j,j] = (16/M) * (2*preference - 1) * difficulty_j
    """
    np.random.seed(42)
    data = np.random.rand(10, 5)
    pop = PikaiaPopulation(data)
    M = pop.M

    # Compute expected difficulty (normalized odds ratio)
    mean_all = pop.matrix.mean(axis=0)
    excl = 1.0 - mean_all
    odds = excl / (1.0 - excl + 1e-8)
    expected_difficulty = odds / (odds.max() + 1e-8)

    for strat_enum, expected_sign in [
        (GeneStrategyEnum.REWARD_HARD, 1.0),
        (GeneStrategyEnum.REWARD_EASY, -1.0),
        (GeneStrategyEnum.VALUATION_BLEND, 2.0 * 0.7 - 1.0),
    ]:
        strat = GeneStrategyFactory.get_strategy(
            strat_enum,
            preference=0.7 if strat_enum == GeneStrategyEnum.VALUATION_BLEND else None,
        )
        D, d = strat.kernel(pop, np.eye(M), np.eye(pop.N), 1.0, y=None)

        assert D is not None
        assert d is None

        expected_diag = (16.0 / M) * expected_sign * expected_difficulty
        actual_diag = np.diag(D)
        assert np.allclose(actual_diag, expected_diag, atol=1e-14), (
            f"{strat_enum}: diag={actual_diag} != {expected_diag}"
        )


# ---------------------------------------------------------------------------
# Test 6: Kernel is diagonal (off-diagonal elements are zero)
# ---------------------------------------------------------------------------
def test_kernel_is_diagonal():
    """
    All three trading strategies should produce strictly diagonal D matrices
    (no cross-gene interaction in the kernel).
    """
    np.random.seed(42)
    data = np.random.rand(10, 5)
    pop = PikaiaPopulation(data)
    M = pop.M

    for strat_enum in [
        GeneStrategyEnum.REWARD_HARD,
        GeneStrategyEnum.REWARD_EASY,
        GeneStrategyEnum.VALUATION_BLEND,
    ]:
        strat = GeneStrategyFactory.get_strategy(
            strat_enum,
            preference=0.5 if strat_enum == GeneStrategyEnum.VALUATION_BLEND else None,
        )
        D, _ = strat.kernel(pop, np.eye(M), np.eye(pop.N), 1.0, y=None)

        off_diag = D - np.diag(np.diag(D))
        assert np.allclose(off_diag, 0, atol=1e-15), (
            f"{strat_enum} has non-zero off-diagonal: {np.max(np.abs(off_diag))}"
        )


# ---------------------------------------------------------------------------
# Test 7: Deltas are proportional to gene fitness
# ---------------------------------------------------------------------------
def test_deltas_scale_with_fitness():
    """
    The delta formula is multiplicative: delta ∝ gene_fitness.
    Doubling gene_fitness should double the delta.
    """
    np.random.seed(42)
    data = np.random.rand(8, 4)
    pop = PikaiaPopulation(data)

    for strat_enum in [
        GeneStrategyEnum.REWARD_HARD,
        GeneStrategyEnum.REWARD_EASY,
        GeneStrategyEnum.VALUATION_BLEND,
    ]:
        strat = GeneStrategyFactory.get_strategy(
            strat_enum,
            preference=0.3 if strat_enum == GeneStrategyEnum.VALUATION_BLEND else None,
        )

        for org_id in range(pop.N):
            for gene_id in range(pop.M):
                ctx_low = _make_ctx(
                    pop, org_id, gene_id, gene_fitness=np.ones(4) * 0.25
                )
                ctx_high = _make_ctx(
                    pop, org_id, gene_id, gene_fitness=np.ones(4) * 0.5
                )

                delta_low = strat(ctx_low)
                delta_high = strat(ctx_high)

                assert np.isclose(delta_high, 2.0 * delta_low, atol=1e-14), (
                    f"{strat_enum} org={org_id} gene={gene_id}: "
                    f"{delta_high} != 2*{delta_low}"
                )


# ---------------------------------------------------------------------------
# Test 8: Deltas are proportional to (x_ij - 0.5)
# ---------------------------------------------------------------------------
def test_deltas_scale_with_expression():
    """
    delta_ij = (16/N) * sign * difficulty_j * gf_j * (x_ij - 0.5)
    For fixed gene j and fitness, delta should scale with (x_ij - 0.5).
    """
    np.random.seed(42)
    data = np.random.rand(8, 4)
    pop = PikaiaPopulation(data)

    hard_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_HARD)

    gene_id = 0
    org_id = 0

    x_val = pop.matrix[org_id, gene_id]
    gf = np.ones(4) * 0.25

    ctx1 = _make_ctx(pop, org_id, gene_id, gene_fitness=gf)
    delta1 = hard_strat(ctx1)

    # For gene 0: difficulty is fixed, gf is fixed, only (x - 0.5) changes
    expected_scaling = x_val - 0.5
    all_excl = 1.0 - pop.matrix.mean(axis=0)
    all_odds = all_excl / (1.0 - all_excl + 1e-8)
    all_diff = all_odds / (all_odds.max() + 1e-8)
    expected_delta = (16.0 / pop.N) * all_diff[gene_id] * gf[gene_id] * expected_scaling

    assert np.isclose(delta1, expected_delta, atol=1e-10), (
        f"delta={delta1:.10e} != expected={expected_delta:.10e}"
    )


# ---------------------------------------------------------------------------
# Test 9: Preference=0.5 gives zero signal (balanced)
# ---------------------------------------------------------------------------
def test_preference_05_gives_zero_signal():
    """
    preference=0.5 → sign = 2*0.5 - 1 = 0 → zero signal.
    """
    np.random.seed(42)
    data = np.random.rand(10, 4)
    pop = PikaiaPopulation(data)

    balanced_strat = GeneStrategyFactory.get_strategy(
        GeneStrategyEnum.VALUATION_BLEND, preference=0.5
    )

    for org_id in range(pop.N):
        for gene_id in range(pop.M):
            delta = balanced_strat(_make_ctx(pop, org_id, gene_id))
            assert np.isclose(delta, 0.0, atol=1e-15), (
                f"preference=0.5 should give zero delta: {delta}"
            )


# ---------------------------------------------------------------------------
# Test 10: Preference=0.0 is pure REWARD_EASY
# ---------------------------------------------------------------------------
def test_preference_0_equals_reward_easy():
    """
    preference=0.0 → sign = -1 → equivalent to REWARD_EASY.
    """
    np.random.seed(42)
    data = np.random.rand(10, 4)
    pop = PikaiaPopulation(data)

    easy_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_EASY)
    blend_strat = GeneStrategyFactory.get_strategy(
        GeneStrategyEnum.VALUATION_BLEND, preference=0.0
    )

    for org_id in range(pop.N):
        for gene_id in range(pop.M):
            e = easy_strat(_make_ctx(pop, org_id, gene_id))
            b = blend_strat(_make_ctx(pop, org_id, gene_id))
            assert np.isclose(e, b, atol=1e-15), (
                f"preference=0 should equal REWARD_EASY: {e} vs {b}"
            )


# ---------------------------------------------------------------------------
# Test 11: Preference=1.0 is pure REWARD_HARD
# ---------------------------------------------------------------------------
def test_preference_1_equals_reward_hard():
    """
    preference=1.0 → sign = +1 → equivalent to REWARD_HARD.
    """
    np.random.seed(42)
    data = np.random.rand(10, 4)
    pop = PikaiaPopulation(data)

    hard_strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_HARD)
    blend_strat = GeneStrategyFactory.get_strategy(
        GeneStrategyEnum.VALUATION_BLEND, preference=1.0
    )

    for org_id in range(pop.N):
        for gene_id in range(pop.M):
            h = hard_strat(_make_ctx(pop, org_id, gene_id))
            b = blend_strat(_make_ctx(pop, org_id, gene_id))
            assert np.isclose(h, b, atol=1e-15), (
                f"preference=1 should equal REWARD_HARD: {h} vs {b}"
            )


# ---------------------------------------------------------------------------
# Test 12: Verify calsim formula arithmetic is correct
# ---------------------------------------------------------------------------
def test_calsim_difficulty_arithmetic():
    """
    Verify the calsim difficulty formula used in our test helpers is correct.

    calsim.py: vdeltaSell = startValue * exclusiveness / (1 - exclusiveness) / valid

    Using performanceMatrix = [[1,0,1,1], [1,1,1,0], [1,0,0,1]]:
    - exclusiveness = (1-perf).mean(axis=0) = [0, 2/3, 1/3, 1/3]
    - Difficulty1 vdelta (without startValue/valid):
        j=0: 0 / 1.0 = 0.0
        j=1: (2/3) / (1/3) = 2.0
        j=2: (1/3) / (2/3) = 0.5
        j=3: (1/3) / (2/3) = 0.5
    """
    performance = np.array(
        [
            [1.0, 0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 1.0],
        ]
    )
    exclusiveness = (1.0 - performance).mean(axis=0)  # [0.0, 2/3, 1/3, 1/3]
    valid = 3

    calsim_raw = _calsim_difficulty(exclusiveness, valid)
    # With valid=3: j=0: 0/1/3=0, j=1: (2/3)/(1/3)/3=2/3, j=2: (1/3)/(2/3)/3=1/6, j=3: same
    expected = np.array([0.0, 2.0 / 3.0, 0.5 / 3.0, 0.5 / 3.0])

    assert np.allclose(calsim_raw, expected, atol=1e-10), (
        f"calsim formula check: {calsim_raw} != {expected}"
    )

    # Also verify pikaia difficulty on same data
    pikaia_diff = _pikaia_difficulty(exclusiveness)
    # Both should rank exercise 1 as hardest
    assert pikaia_diff[1] >= pikaia_diff[0]
    assert pikaia_diff[1] >= pikaia_diff[2]
    assert pikaia_diff[1] >= pikaia_diff[3]
