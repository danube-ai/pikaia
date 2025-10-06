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
