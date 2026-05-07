import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy


class TestPikaiaModel:
    """Test cases for PikaiaModel class."""

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
