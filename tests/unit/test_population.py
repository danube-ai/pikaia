import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation


class TestPikaiaPopulation:
    """Test cases for PikaiaPopulation class."""

    def test_init_valid_matrix(self):
        """Test initialization with a valid matrix."""
        matrix = np.array([[0.1, 0.5], [0.9, 0.3]])
        pop = PikaiaPopulation(matrix)
        assert np.array_equal(pop.matrix, matrix)
        assert pop.N == 2
        assert pop.M == 2

    def test_init_invalid_values_below_zero(self):
        """Test initialization with values below 0 raises ValueError."""
        matrix = np.array([[-0.1, 0.5], [0.9, 0.3]])
        with pytest.raises(
            ValueError,
            match="All values in the population matrix must be between 0 and 1",
        ):
            PikaiaPopulation(matrix)

    def test_init_invalid_values_above_one(self):
        """Test initialization with values above 1 raises ValueError."""
        matrix = np.array([[1.1, 0.5], [0.9, 0.3]])
        with pytest.raises(
            ValueError,
            match="All values in the population matrix must be between 0 and 1",
        ):
            PikaiaPopulation(matrix)

    def test_init_single_feature_requires_skip_correlation(self):
        """Test that single feature requires skipping correlation validation."""
        matrix = np.array([[0.1], [0.9]])
        with pytest.raises(
            ValueError, match="Population matrix must have at least two features"
        ):
            PikaiaPopulation(matrix, skip_correlation_validation=False)

    def test_init_single_feature_with_skip(self):
        """Test initialization with single feature when skipping correlation validation."""
        matrix = np.array([[0.1], [0.9]])
        pop = PikaiaPopulation(matrix, skip_correlation_validation=True)
        assert pop.N == 2
        assert pop.M == 1

    def test_init_correlation_validation_enabled_high_correlation(self, caplog):
        """Test that high correlation triggers warning when validation enabled."""
        # Create highly correlated features
        np.random.seed(42)
        base = np.random.rand(10)
        matrix = np.column_stack(
            [base, base + 0.01 * np.random.rand(10)]
        )  # High correlation
        pop = PikaiaPopulation(matrix, skip_correlation_validation=False)
        assert pop is not None  # Ensure object is created
        assert "Correlated feature pair" in caplog.text

    def test_init_correlation_validation_disabled(self, caplog):
        """Test that correlation validation is skipped when disabled."""
        np.random.seed(42)
        base = np.random.rand(10)
        matrix = np.column_stack([base, base + 0.01 * np.random.rand(10)])
        pop = PikaiaPopulation(matrix, skip_correlation_validation=True)
        assert pop is not None
        assert "Correlated feature pair" not in caplog.text

    def test_getitem_indexing(self):
        """Test indexing into the population matrix."""
        matrix = np.array([[0.1, 0.5], [0.9, 0.3]])
        pop = PikaiaPopulation(matrix)
        assert pop[0, 0] == 0.1
        assert np.array_equal(pop[0], np.array([0.1, 0.5]))
        assert np.array_equal(pop[:, 0], np.array([0.1, 0.9]))

    def test_properties(self):
        """Test N, M, and matrix properties."""
        matrix = np.random.rand(5, 3)
        pop = PikaiaPopulation(matrix)
        assert pop.N == 5
        assert pop.M == 3
        assert np.array_equal(pop.matrix, matrix)

    def test_empty_matrix(self):
        """Test initialization with empty matrix."""
        matrix = np.array([]).reshape(0, 2)
        pop = PikaiaPopulation(matrix)
        assert pop.N == 0
        assert pop.M == 2

    def test_single_organism(self):
        """Test with single organism."""
        matrix = np.array([[0.2, 0.8]])
        pop = PikaiaPopulation(matrix)
        assert pop.N == 1
        assert pop.M == 2

    def test_population_3x3_properties(self, population_3x3):
        """Test properties of 3x3 example population."""
        assert population_3x3.N == 3
        assert population_3x3.M == 3
        # Check all values are between 0 and 1
        assert np.all(population_3x3.matrix >= 0) and np.all(population_3x3.matrix <= 1)

    def test_population_10x5_properties(self, population_10x5):
        """Test properties of 10x5 example population."""
        assert population_10x5.N == 10
        assert population_10x5.M == 5
        # Check all values are between 0 and 1
        assert np.all(population_10x5.matrix >= 0) and np.all(
            population_10x5.matrix <= 1
        )

    def test_population_paper_example_properties(self, population_paper_example):
        """Test properties of paper example population."""
        assert population_paper_example.N == 15
        assert population_paper_example.M == 4
        # Check all values are between 0 and 1
        assert np.all(population_paper_example.matrix >= 0) and np.all(
            population_paper_example.matrix <= 1
        )
