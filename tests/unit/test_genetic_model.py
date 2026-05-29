"""Tests for GeneticModel property accessors and error paths.

GeneticModel is abstract; all tests use PikaiaModel as the concrete
implementation.
"""

import logging

import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy


def _make_population(n: int = 3, m: int = 4, seed: int = 42) -> PikaiaPopulation:
    rng = np.random.default_rng(seed)
    return PikaiaPopulation(rng.random((n, m)))


def _model_with_strategies(**kwargs) -> PikaiaModel:
    population = _make_population()
    return PikaiaModel(
        population=population,
        gene_strategies=[NoneGeneStrategy()],
        org_strategies=[NoneOrgStrategy()],
        max_iter=3,
        **kwargs,
    )


class TestGeneticModelProperties:
    """Exercises all public property accessors inherited from GeneticModel."""

    def test_population_property(self):
        pop = _make_population()
        model = PikaiaModel(population=pop)
        assert model.population is pop

    def test_gene_strategies_property(self):
        model = _model_with_strategies()
        assert len(list(model.gene_strategies)) == 1

    def test_org_strategies_property(self):
        model = _model_with_strategies()
        assert len(list(model.org_strategies)) == 1

    def test_gene_mixing_property(self):
        model = _model_with_strategies()
        mixing = list(model.gene_mixing)
        assert len(mixing) == 1
        assert abs(sum(mixing) - 1.0) < 1e-9

    def test_org_mixing_property(self):
        model = _model_with_strategies()
        mixing = list(model.org_mixing)
        assert len(mixing) == 1
        assert abs(sum(mixing) - 1.0) < 1e-9

    def test_initial_gene_fitness_property(self):
        model = _model_with_strategies()
        fitness = model.initial_gene_fitness
        assert fitness.ndim == 1
        assert len(fitness) == _make_population().M

    def test_initial_org_fitness_property(self):
        model = _model_with_strategies()
        fitness = model.initial_org_fitness
        assert fitness.ndim == 1
        assert len(fitness) == _make_population().N

    def test_initial_org_fitness_range_property(self):
        model = _model_with_strategies()
        assert model.initial_org_fitness_range > 0

    def test_gene_similarity_property(self):
        pop = _make_population(n=3, m=4)
        model = PikaiaModel(population=pop)
        sim = model.gene_similarity
        assert sim.shape == (pop.M, pop.M)

    def test_org_similarity_property(self):
        pop = _make_population(n=3, m=4)
        model = PikaiaModel(population=pop)
        sim = model.org_similarity
        assert sim.shape == (pop.N, pop.N)

    def test_gene_fitness_history_property(self):
        model = _model_with_strategies()
        hist = model.gene_fitness_history
        # shape: (max_iter+1, M)
        assert hist.shape[0] == 4
        assert hist.shape[1] == _make_population().M

    def test_organism_fitness_history_property(self):
        model = _model_with_strategies()
        hist = model.organism_fitness_history
        assert hist.shape[0] == 4
        assert hist.shape[1] == _make_population().N

    def test_gene_mixing_history_property(self):
        model = _model_with_strategies()
        hist = model.gene_mixing_history
        assert hist.shape == (4, 1)

    def test_organism_mixing_history_property(self):
        model = _model_with_strategies()
        hist = model.organism_mixing_history
        assert hist.shape == (4, 1)

    def test_max_iter_property(self):
        model = _model_with_strategies()
        assert model.max_iter == 3

    def test_max_iter_none_property(self):
        pop = _make_population()
        model = PikaiaModel(population=pop)
        assert model.max_iter is None

    def test_ese_iter_property_initial(self):
        model = _model_with_strategies()
        assert model.ESE_iter == -1


class TestGeneticModelMixingCoeffsValidation:
    """Tests for _init_and_validate_mixing_coeffs error paths."""

    def test_wrong_length_raises(self):
        pop = _make_population()
        with pytest.raises(ValueError, match="gene_mixing_coeffs must have length 1"):
            PikaiaModel(
                population=pop,
                gene_strategies=[NoneGeneStrategy()],
                org_strategies=[NoneOrgStrategy()],
                max_iter=3,
                gene_mixing_coeffs=[0.5, 0.5],  # length 2 but only 1 strategy
            )

    def test_negative_values_raises(self):
        pop = _make_population()
        with pytest.raises(ValueError, match="contains negative values"):
            PikaiaModel(
                population=pop,
                gene_strategies=[NoneGeneStrategy(), NoneGeneStrategy()],
                org_strategies=[NoneOrgStrategy()],
                max_iter=3,
                gene_mixing_coeffs=[-0.5, 0.5],
            )

    def test_zero_sum_raises(self):
        pop = _make_population()
        with pytest.raises(ValueError, match="sums to zero"):
            PikaiaModel(
                population=pop,
                gene_strategies=[NoneGeneStrategy(), NoneGeneStrategy()],
                org_strategies=[NoneOrgStrategy()],
                max_iter=3,
                gene_mixing_coeffs=[0.0, 0.0],
            )

    def test_non_normalized_warns_and_normalizes(self, caplog):
        pop = _make_population()
        with caplog.at_level(logging.WARNING):
            model = PikaiaModel(
                population=pop,
                gene_strategies=[NoneGeneStrategy(), NoneGeneStrategy()],
                org_strategies=[NoneOrgStrategy()],
                max_iter=3,
                gene_mixing_coeffs=[2.0, 1.0],  # sums to 3, not 1
            )
        assert "Normalized coefficients" in caplog.text
        mixing = list(model.gene_mixing)
        assert abs(sum(mixing) - 1.0) < 1e-9


class TestGeneticModelComputeSimilarity:
    """Tests for _compute_similarity error paths."""

    def test_unknown_mode_raises(self):
        pop = _make_population()
        model = PikaiaModel(population=pop)
        with pytest.raises(ValueError, match="Unknown mode"):
            model._compute_similarity(mode="xyz")

    def test_identical_gene_columns_raises(self):
        """A matrix where all gene columns are identical triggers the gene similarity error."""
        # All columns equal → _compute_similarity(mode="gene") raises
        # Rows differ so org_fitness_range > 0
        matrix = np.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9], [0.5, 0.5, 0.5]])
        pop = PikaiaPopulation(matrix)
        with pytest.raises(ValueError, match="All gene items are identical"):
            PikaiaModel(population=pop)

    def test_identical_org_rows_range_zero_raises(self):
        """Identical org rows give org_fitness_range == 0, raises before similarity."""
        identical_matrix = np.ones((3, 4)) * 0.5
        pop = PikaiaPopulation(identical_matrix)
        with pytest.raises(ValueError, match="All organism fitness values are 0"):
            PikaiaModel(population=pop)


class TestGeneticModelInitBranches:
    """Tests for initialisation branches not covered by the default path."""

    def test_n_jobs_minus_one_uses_cpu_count(self):
        import multiprocessing

        pop = _make_population()
        model = PikaiaModel(
            population=pop,
            gene_strategies=[NoneGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
            max_iter=2,
            n_jobs=-1,
        )
        assert model._n_jobs == multiprocessing.cpu_count()

    def test_gene_mix_strategy_with_max_iter_none_warns(self, caplog):
        from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy

        pop = _make_population()
        with caplog.at_level(logging.WARNING):
            PikaiaModel(population=pop, gene_mix_strategy=FixedMixStrategy())
        assert "gene_mix_strategy is ignored" in caplog.text

    def test_org_mix_strategy_with_max_iter_none_warns(self, caplog):
        from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy

        pop = _make_population()
        with caplog.at_level(logging.WARNING):
            PikaiaModel(population=pop, org_mix_strategy=FixedMixStrategy())
        assert "org_mix_strategy is ignored" in caplog.text

    def test_epsilon_with_max_iter_none_warns(self, caplog):
        pop = _make_population()
        with caplog.at_level(logging.WARNING):
            PikaiaModel(population=pop, epsilon=1e-6)
        assert "epsilon is ignored" in caplog.text

    def test_initial_gene_fitness_with_max_iter_none_warns(self, caplog):
        pop = _make_population()
        fitness = np.ones(pop.M) / pop.M
        with caplog.at_level(logging.WARNING):
            PikaiaModel(population=pop, initial_gene_fitness=fitness)
        assert "initial_gene_fitness has no effect" in caplog.text

    def test_initial_gene_fitness_with_max_iter_gt5_logs_info(self, caplog):
        pop = _make_population()
        fitness = np.ones(pop.M) / pop.M
        with caplog.at_level(logging.INFO):
            PikaiaModel(
                population=pop,
                gene_strategies=[NoneGeneStrategy()],
                org_strategies=[NoneOrgStrategy()],
                max_iter=6,
                initial_gene_fitness=fitness,
            )
        assert "initial_gene_fitness will likely have little effect" in caplog.text

    def test_all_zero_org_fitness_raises(self):
        """A population matrix that yields all-zero org fitness raises ValueError."""
        zero_pop = PikaiaPopulation(np.zeros((3, 4)))
        with pytest.raises(ValueError, match="All organism fitness values are 0"):
            PikaiaModel(population=zero_pop)
