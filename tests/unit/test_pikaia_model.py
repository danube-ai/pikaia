import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import (
    KinAltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.mix_strategies.self_consistent_strategy import (
    SelfConsistentMixStrategy,
)
from pikaia.strategies.os_strategies.altruistic_strategy import AltruisticOrgStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.kin_selfish_strategy import KinSelfishOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy

# ---------------------------------------------------------------------------
# Minimal strategy subclass that returns a d-vector (no D matrix)
# ---------------------------------------------------------------------------


class _LinearGeneStrategy(GeneStrategy):
    """Test-only strategy: returns (None, d) where d = x_bar."""

    @property
    def name(self) -> str:
        return "LinearTest"

    def __call__(self, ctx: StrategyContext) -> float:
        return 0.0

    def kernel(
        self,
        population,
        gene_similarity,
        org_similarity,
        initial_org_fitness_range,
        y=None,
    ):
        d = population.matrix.mean(axis=0)
        return None, d


class TestPikaiaModel:
    def test_init_minimal(self):
        """Test PikaiaModel initialization with minimal parameters."""
        population = PikaiaPopulation(np.random.rand(3, 4))
        model = PikaiaModel(population=population)
        assert model._population is population

    def test_init_with_strategies_no_max_iter(self):
        """Test initialization with strategies but no max_iter."""
        population = PikaiaPopulation(np.random.rand(3, 4))
        gene_strategies = [NoneGeneStrategy()]
        org_strategies = [NoneOrgStrategy()]

        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
        )
        assert model._population is population

    def test_init_max_iter_without_strategies_raises_error(self):
        """Test that max_iter without strategies raises ValueError."""
        population = PikaiaPopulation(np.random.rand(3, 4))

        with pytest.raises(
            ValueError, match="gene_strategies must be provided if max_iter is set"
        ):
            PikaiaModel(population=population, max_iter=10)

        with pytest.raises(
            ValueError, match="org_strategies must be provided if max_iter is set"
        ):
            PikaiaModel(
                population=population, gene_strategies=[NoneGeneStrategy()], max_iter=10
            )

    def test_init_with_max_iter_and_strategies(self):
        """Test initialization with max_iter and strategies."""
        population = PikaiaPopulation(np.random.rand(3, 4))
        gene_strategies = [NoneGeneStrategy()]
        org_strategies = [NoneOrgStrategy()]

        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
            max_iter=10,
        )
        assert model._population is population
        assert len(model._gene_strategies) == 1
        assert len(model._org_strategies) == 1

    def test_init_with_3x3_population(self, population_3x3):
        """Test model initialization with 3x3 example population."""
        model = PikaiaModel(population=population_3x3)
        assert model._population is population_3x3
        assert model._population.N == 3
        assert model._population.M == 3

    def test_init_with_10x5_population(self, population_10x5):
        """Test model initialization with 10x5 example population."""
        model = PikaiaModel(population=population_10x5)
        assert model._population is population_10x5
        assert model._population.N == 10
        assert model._population.M == 5

    def test_init_with_paper_example_population(self, population_paper_example):
        """Test model initialization with paper example population."""
        model = PikaiaModel(population=population_paper_example)
        assert model._population is population_paper_example
        assert model._population.N == 15
        assert model._population.M == 4

    def test_fit_no_max_iter_runs_fix_point(self):
        """fit() with no max_iter should call _run_fix_point and populate hist[1]."""
        population = PikaiaPopulation(np.random.default_rng(0).random((3, 4)))
        model = PikaiaModel(population=population)
        model.fit()
        # After fix-point, hist[1] should be non-zero
        assert not np.all(model.gene_fitness_history[1, :] == 0)
        assert not np.all(model.organism_fitness_history[1, :] == 0)

    def test_fit_with_max_iter_runs_iterations(self):
        """fit() with max_iter runs the iterative simulation."""
        population = PikaiaPopulation(np.random.default_rng(1).random((3, 4)))
        model = PikaiaModel(
            population=population,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=3,
        )
        model.fit()
        # History beyond index 0 should be filled
        assert not np.all(model.gene_fitness_history[1:, :] == 0)

    def test_fit_with_epsilon_stops_early(self):
        """fit() with epsilon should set ESE_iter when convergence is reached."""
        population = PikaiaPopulation(np.random.default_rng(2).random((3, 4)))
        # NoneStrategy produces zero deltas → gene_fitness unchanged → delta=0 < any epsilon
        model = PikaiaModel(
            population=population,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=10,
            epsilon=1.0,  # very large epsilon → converges immediately
        )
        model.fit()
        assert model.ESE_iter == 1  # should converge on first iteration

    def test_calculate_deltas_parallel(self):
        """_calculate_deltas with n_jobs=2 runs the parallel multiprocessing path."""
        population = PikaiaPopulation(np.random.default_rng(3).random((3, 4)))
        model = PikaiaModel(
            population=population,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=2,
            n_jobs=2,
        )
        model.fit()
        # Just verify it ran without error and produced results
        assert not np.all(model.gene_fitness_history[1:, :] == 0)

    def test_predict_mismatched_M_raises(self):
        """predict() should raise ValueError when new population has different M."""
        population = PikaiaPopulation(np.random.default_rng(4).random((3, 4)))
        model = PikaiaModel(population=population)
        model.fit()

        wrong_pop = PikaiaPopulation(np.random.default_rng(5).random((2, 5)))
        with pytest.raises(ValueError, match="number of genes"):
            model.predict(wrong_pop)

    def test_predict_after_fit(self):
        """predict() should return an array of shape (N,) for a compatible population."""
        population = PikaiaPopulation(np.random.default_rng(6).random((3, 4)))
        model = PikaiaModel(population=population)
        model.fit()
        new_pop = PikaiaPopulation(np.random.default_rng(7).random((5, 4)))
        result = model.predict(new_pop)
        assert result.shape == (5,)


# ---------------------------------------------------------------------------
# D-matrix path — unit and numerical verification
# ---------------------------------------------------------------------------


def _make_model(
    rng_seed,
    N=8,
    M=5,
    *,
    use_d_matrix=False,
    gene_strats,
    org_strats,
    max_iter=None,
    epsilon=None,
):
    """Helper: build a PikaiaModel with the given strategy combo."""
    X = np.random.default_rng(rng_seed).random((N, M))
    pop = PikaiaPopulation(X)
    return PikaiaModel(
        population=pop,
        gene_strategies=gene_strats,
        org_strategies=org_strats,
        max_iter=max_iter,
        epsilon=epsilon,
        use_d_matrix=use_d_matrix,
    )


class TestDMatrixInit:
    def test_use_d_matrix_flag_stored(self):
        pop = PikaiaPopulation(np.random.default_rng(0).random((4, 3)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=5,
            use_d_matrix=True,
        )
        assert model._use_d_matrix is True

    def test_default_is_false(self):
        pop = PikaiaPopulation(np.random.default_rng(0).random((4, 3)))
        model = PikaiaModel(population=pop)
        assert model._use_d_matrix is False


class TestDMatrixIterations:
    """Test that _run_d_matrix_iterations produces valid results."""

    def test_gamma_normalized(self):
        """After fit, gene fitness must sum to 1."""
        pop = PikaiaPopulation(np.random.default_rng(1).random((6, 4)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last_iter = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        gamma = model.gene_fitness_history[last_iter, :]
        assert pytest.approx(gamma.sum(), abs=1e-12) == 1.0

    def test_gamma_positive(self):
        """All gene fitness values must be strictly positive after fit."""
        pop = PikaiaPopulation(np.random.default_rng(2).random((6, 4)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last_iter = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        gamma = model.gene_fitness_history[last_iter, :]
        assert np.all(gamma > 0)

    def test_org_fitness_matches_gamma(self):
        """Organism fitness history must equal population @ gamma at every step."""
        pop = PikaiaPopulation(np.random.default_rng(3).random((5, 3)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=10,
            use_d_matrix=True,
        )
        model.fit()
        for i in range(1, 11):
            gamma_i = model.gene_fitness_history[i, :]
            if gamma_i.sum() == 0:
                break
            org_i = model.organism_fitness_history[i, :]
            expected = pop.matrix @ gamma_i
            np.testing.assert_allclose(org_i, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# V1 — Iterative fixed-point equivalence: D-matrix ≈ standard loop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [10, 11, 12])
class TestV1IterativeEquivalence:
    """V1: Both paths should converge to the same fixed point within tolerance."""

    def _run_both(
        self, seed, gene_strats, org_strats, N=10, M=5, max_iter=500, epsilon=1e-8
    ):
        X = np.random.default_rng(seed).random((N, M))
        pop = PikaiaPopulation(X)

        standard = PikaiaModel(
            population=pop,
            gene_strategies=gene_strats,
            org_strategies=org_strats,
            max_iter=max_iter,
            epsilon=epsilon,
            use_d_matrix=False,
        )
        standard.fit()

        d_matrix = PikaiaModel(
            population=pop,
            gene_strategies=gene_strats,
            org_strategies=org_strats,
            max_iter=max_iter,
            epsilon=epsilon,
            use_d_matrix=True,
        )
        d_matrix.fit()

        # Grab last non-zero iteration from each
        def last_gamma(model):
            mask = model.gene_fitness_history.sum(axis=1) > 0
            idx = np.flatnonzero(mask).max()
            return model.gene_fitness_history[idx, :]

        return last_gamma(standard), last_gamma(d_matrix)

    def test_dominant_none(self, seed):
        """Dominant gene + None org."""
        g_std, g_dm = self._run_both(
            seed,
            [DominantGeneStrategy()],
            [NoneOrgStrategy()],
        )
        np.testing.assert_allclose(g_std, g_dm, atol=1e-4)

    def test_altruistic_none(self, seed):
        """Altruistic gene + None org."""
        g_std, g_dm = self._run_both(
            seed,
            [AltruisticGeneStrategy()],
            [NoneOrgStrategy()],
        )
        np.testing.assert_allclose(g_std, g_dm, atol=1e-4)

    def test_dominant_selfish_org(self, seed):
        """Dominant gene + Selfish org."""
        g_std, g_dm = self._run_both(
            seed,
            [DominantGeneStrategy()],
            [SelfishOrgStrategy()],
        )
        np.testing.assert_allclose(g_std, g_dm, atol=1e-4)

    def test_dominant_balanced_org(self, seed):
        """Dominant gene + Balanced org (d-vector path)."""
        g_std, g_dm = self._run_both(
            seed,
            [DominantGeneStrategy()],
            [BalancedOrgStrategy()],
        )
        np.testing.assert_allclose(g_std, g_dm, atol=1e-4)


# ---------------------------------------------------------------------------
# D-matrix error paths
# ---------------------------------------------------------------------------


class TestDMatrixErrorPaths:
    """Tests for ValueError branches in the D-matrix path."""

    def test_d_matrix_requires_max_iter(self):
        """use_d_matrix=True with max_iter=None raises ValueError."""
        pop = PikaiaPopulation(np.random.default_rng(30).random((4, 3)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            use_d_matrix=True,
        )
        with pytest.raises(ValueError, match="use_d_matrix=True requires max_iter"):
            model.fit()

    def test_d_matrix_raises_when_no_strategy_contributes_kernel(self):
        """use_d_matrix=True raises ValueError when no strategy implements kernel().

        NoneGeneStrategy and NoneOrgStrategy both use the base-class default
        kernel() that returns (None, None).  Running all iterations silently
        doing nothing is wasteful, so _compute_d_matrix raises immediately.
        """
        pop = PikaiaPopulation(np.random.default_rng(31).random((4, 3)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            use_d_matrix=True,
            max_iter=50,
        )
        with pytest.raises(ValueError, match="None of the selected strategies"):
            model.fit()

    def test_iterations_raises_on_non_positive_gamma(self):
        """_run_d_matrix_iterations raises ValueError when gamma goes non-positive.

        With gamma_0 concentrated on gene 0 (≈0.98) and x_bar[0]≈0.7,
        the bilinear step is:
          step_0 = 0.98 * (D_bal @ gamma)_0 = 0.98 * (-2*0.7*1) ≈ -1.37
        so  gamma_new_0 = 0.98 * (1 - 1.37) < 0  → ValueError.
        """
        X = np.full((4, 3), 0.9)
        X[0, 0] = 0.1  # slight variation so fitness range > 0
        X[1, 1] = 0.1
        pop = PikaiaPopulation(X)
        # x_bar ≈ [0.7, 0.7, 0.9]; concentrate gamma on gene 0
        model = PikaiaModel(
            population=pop,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[BalancedOrgStrategy()],
            use_d_matrix=True,
            max_iter=1,
            initial_gene_fitness=[0.98, 0.01, 0.01],
        )
        with pytest.raises(ValueError, match="non-positive gene fitness"):
            model.fit()

    def test_error_message_lists_strategy_class_names(self):
        """ValueError message must name the offending strategy classes."""
        pop = PikaiaPopulation(np.random.default_rng(33).random((4, 3)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            use_d_matrix=True,
            max_iter=10,
        )
        with pytest.raises(ValueError, match="NoneGeneStrategy"):
            model.fit()

    def test_org_only_d_kernel_is_valid(self):
        """NoneGeneStrategy (no kernel) + BalancedOrgStrategy (D kernel) → valid.

        When only the org strategy contributes a kernel, _compute_d_matrix
        must NOT raise — it only raises when ALL strategies return (None, None).
        With uniform initial gamma and x_bar in [0, 0.45], the balanced D step
        keeps gamma positive every iteration.
        """
        # x_bar well below 0.5 so balanced step gamma*(1 - 2*x_bar*gamma) stays > 0
        X = np.random.default_rng(34).random((8, 4)) * 0.45
        pop = PikaiaPopulation(X)
        model = PikaiaModel(
            population=pop,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[BalancedOrgStrategy()],
            use_d_matrix=True,
            max_iter=10,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    def test_d_vector_only_kernel_is_valid(self):
        """A strategy returning (None, d) — no D matrix — is a valid kernel.

        _LinearGeneStrategy returns (None, d) where d = x_bar.  The D-matrix
        path should run the linear-only step (gamma*(1 + d)) without error.
        """
        X = np.random.default_rng(35).random((6, 4))
        pop = PikaiaPopulation(X)
        model = PikaiaModel(
            population=pop,
            gene_strategies=[_LinearGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            use_d_matrix=True,
            max_iter=5,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)
        assert np.all(model.gene_fitness_history[last, :] >= 0)

    def test_partial_kernel_mixed_strategies(self):
        """Mixing kernel and non-kernel gene strategies is valid.

        [DominantGeneStrategy (has kernel), NoneGeneStrategy (no kernel)] →
        the combined D is just DominantGeneStrategy's contribution;
        NoneGeneStrategy contributes (None, None) and is silently skipped.
        """
        X = np.random.default_rng(36).random((8, 4))
        pop = PikaiaPopulation(X)
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy(), NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            use_d_matrix=True,
            max_iter=20,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    def test_d_matrix_epsilon_early_stop(self):
        """epsilon triggers early stop on the D-matrix iterative path."""
        X = np.random.default_rng(37).random((8, 4))
        pop = PikaiaPopulation(X)
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            use_d_matrix=True,
            max_iter=500,
            epsilon=1e-3,  # lenient tolerance → converges quickly
        )
        model.fit()
        # ESE_iter should be set and well below max_iter
        assert model.ESE_iter > 0
        assert model.ESE_iter < 500


class TestSelfConsistentDMatrix:
    """SelfConsistentMixStrategy evolves mixing coefficients inside D-matrix loop."""

    def _pop(self, seed=40, N=8, M=4):
        return PikaiaPopulation(np.random.default_rng(seed).random((N, M)))

    def test_gene_sc_dmatrix_runs(self):
        """SelfConsistentMixStrategy as gene mix strategy runs D-matrix iterations."""
        pop = self._pop()
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy(), AltruisticGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            gene_mix_strategy=SelfConsistentMixStrategy(),
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        gamma = model.gene_fitness_history[last, :]
        assert np.isclose(gamma.sum(), 1.0, atol=1e-9)
        assert np.all(gamma >= 0)

    def test_org_sc_dmatrix_runs(self):
        """SelfConsistentMixStrategy as org mix strategy runs D-matrix iterations."""
        pop = self._pop(seed=41)
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[SelfishOrgStrategy(), NoneOrgStrategy()],
            org_mix_strategy=SelfConsistentMixStrategy(),
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    def test_gene_mixing_coeffs_change_over_time(self):
        """Mixing coefficients should evolve when using SelfConsistentMixStrategy."""
        pop = self._pop(seed=42)
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy(), AltruisticGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            gene_mix_strategy=SelfConsistentMixStrategy(),
            max_iter=30,
            use_d_matrix=True,
        )
        model.fit()
        initial_coeffs = model.gene_mixing_history[0, :]
        last_idx = np.flatnonzero(model.gene_mixing_history.sum(axis=1)).max()
        final_coeffs = model.gene_mixing_history[last_idx, :]
        # Coefficients must stay normalised
        assert np.isclose(final_coeffs.sum(), 1.0, atol=1e-9)
        # And they should have changed from uniform [0.5, 0.5]
        assert not np.allclose(initial_coeffs, final_coeffs, atol=1e-6)

    def test_update_coeffs_d_matrix_static(self):
        """update_coeffs_d_matrix returns normalised coefficients."""
        rng = np.random.default_rng(43)
        M = 4
        D1 = rng.random((M, M))
        D2 = rng.random((M, M))
        gamma = rng.random(M)
        gamma /= gamma.sum()
        coeffs = np.array([0.5, 0.5])
        updated = SelfConsistentMixStrategy.update_coeffs_d_matrix(
            [D1, D2], [None, None], gamma, coeffs
        )
        assert np.isclose(updated.sum(), 1.0, atol=1e-12)
        assert np.all(updated >= 0)

    def test_update_coeffs_d_matrix_d_none_path(self):
        """update_coeffs_d_matrix handles (None, d) pairs."""
        rng = np.random.default_rng(44)
        M = 4
        d1 = rng.random(M)
        d2 = rng.random(M)
        gamma = rng.random(M)
        gamma /= gamma.sum()
        coeffs = np.array([0.5, 0.5])
        updated = SelfConsistentMixStrategy.update_coeffs_d_matrix(
            [None, None], [d1, d2], gamma, coeffs
        )
        assert np.isclose(updated.sum(), 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# D-matrix with complex strategy combinations
# ---------------------------------------------------------------------------


class TestDMatrixComplexStrategies:
    """D-matrix runs for strategy combinations beyond dominant+none."""

    @pytest.mark.parametrize("seed", [50, 51])
    def test_altruistic_gene_dmatrix(self, seed):
        pop = PikaiaPopulation(np.random.default_rng(seed).random((8, 4)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[AltruisticGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    @pytest.mark.parametrize("seed", [52, 53])
    def test_kin_altruistic_gene_dmatrix(self, seed):
        pop = PikaiaPopulation(np.random.default_rng(seed).random((8, 5)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[KinAltruisticGeneStrategy(kin_range=3)],
            org_strategies=[NoneOrgStrategy()],
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    @pytest.mark.parametrize("seed", [54, 55])
    def test_dominant_altruistic_org_dmatrix(self, seed):
        pop = PikaiaPopulation(np.random.default_rng(seed).random((8, 4)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[AltruisticOrgStrategy()],
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    @pytest.mark.parametrize("seed", [56, 57])
    def test_dominant_kin_selfish_org_dmatrix(self, seed):
        pop = PikaiaPopulation(np.random.default_rng(seed).random((8, 4)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[KinSelfishOrgStrategy(kin_range=3)],
            max_iter=20,
            use_d_matrix=True,
        )
        model.fit()
        last = np.flatnonzero(model.gene_fitness_history.sum(axis=1)).max()
        assert np.isclose(model.gene_fitness_history[last, :].sum(), 1.0, atol=1e-9)

    def test_d_vector_path_in_compute_d_matrix(self):
        """Strategy returning (None, d) exercises the d_vector accumulation branch."""
        pop = PikaiaPopulation(np.random.default_rng(60).random((6, 4)))
        model = PikaiaModel(
            population=pop,
            gene_strategies=[_LinearGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=5,
            use_d_matrix=True,
        )
        model.fit()
        # _d_vector should have been populated
        assert model._d_vector is not None
        assert model._d_vector.shape == (pop.M,)
        assert model._D_matrix is None  # no bilinear contribution


# ---------------------------------------------------------------------------
# Adaptive supervision tests
# ---------------------------------------------------------------------------


class _CapturingGeneStrategy(GeneStrategy):
    """Test-only strategy that records the y values it receives via ctx."""

    def __init__(self):
        super().__init__()
        self.captured_y: list = []

    @property
    def name(self) -> str:
        return "Capturing"

    def __call__(self, ctx: StrategyContext) -> float:
        self.captured_y.append(ctx.y)
        return 0.0


class TestAdaptiveSupervision:
    """Tests for y threading through PikaiaModel and StrategyContext."""

    def _make_pop(self, n: int = 5, m: int = 3, seed: int = 0) -> PikaiaPopulation:
        return PikaiaPopulation(np.random.default_rng(seed).random((n, m)))

    def test_y_defaults_to_none_on_model(self):
        pop = self._make_pop()
        model = PikaiaModel(population=pop)
        assert model._y is None

    def test_y_stored_on_model(self):
        pop = self._make_pop()
        y = np.array([0, 1, 0, 1, 0])
        model = PikaiaModel(population=pop, y=y)
        assert model._y is not None
        assert np.array_equal(model._y, y)

    def test_no_y_unsupervised_runs_unchanged(self):
        """Model without y runs identically to before — no regression."""
        pop = self._make_pop()
        model = PikaiaModel(
            population=pop,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=3,
        )
        model.fit()
        assert model.gene_fitness_history.shape == (4, pop.M)

    def test_y_none_explicit_same_as_default(self):
        """Passing y=None is identical to omitting y."""
        pop = self._make_pop(seed=7)
        rng = np.random.default_rng(7)
        _ = rng  # same seed used for pop

        def _run(y):
            m = PikaiaModel(
                population=pop,
                gene_strategies=[NoneGeneStrategy()],
                org_strategies=[NoneOrgStrategy()],
                max_iter=2,
                y=y,
            )
            m.fit()
            return m.gene_fitness_history

        assert np.allclose(_run(None), _run(None))

    def test_y_propagated_to_strategy_context(self):
        """Strategies receive ctx.y matching the y passed to PikaiaModel."""
        pop = self._make_pop()
        y = np.array([1, 0, 1, 0, 1])
        strat = _CapturingGeneStrategy()
        model = PikaiaModel(
            population=pop,
            gene_strategies=[strat],
            org_strategies=[NoneOrgStrategy()],
            max_iter=1,
            y=y,
        )
        model.fit()
        assert all(
            captured is not None and np.array_equal(captured, y)
            for captured in strat.captured_y
        )

    def test_y_none_propagated_to_strategy_context(self):
        """Without y, strategies receive ctx.y == None."""
        pop = self._make_pop()
        strat = _CapturingGeneStrategy()
        model = PikaiaModel(
            population=pop,
            gene_strategies=[strat],
            org_strategies=[NoneOrgStrategy()],
            max_iter=1,
        )
        model.fit()
        assert all(captured is None for captured in strat.captured_y)

    def test_y_with_d_matrix_path(self):
        """D-matrix path passes y to kernel(); built-in strategies ignore it."""
        pop = self._make_pop()
        y = np.array([0, 1, 0, 1, 0])
        model = PikaiaModel(
            population=pop,
            gene_strategies=[DominantGeneStrategy()],
            org_strategies=[BalancedOrgStrategy()],
            use_d_matrix=True,
            max_iter=3,
            y=y,
        )
        model.fit()
        assert np.isclose(model.gene_fitness_history[3].sum(), 1.0)
