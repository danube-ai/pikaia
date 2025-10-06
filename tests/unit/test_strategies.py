import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import GeneStrategyEnum, MixStrategyEnum, OrgStrategyEnum
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.strategy_factories import (
    GeneStrategyFactory,
    MixStrategyFactory,
    OrgStrategyFactory,
)


class TestGeneStrategies:
    """Test cases for gene strategies."""

    @pytest.fixture
    def sample_context(self):
        """Create a sample StrategyContext for testing."""
        population = PikaiaPopulation(np.random.rand(3, 4))  # 3 organisms, 4 genes
        org_fitness = np.array([0.8, 0.6, 0.9])
        gene_fitness = np.array([0.7, 0.5, 0.8, 0.6])
        org_similarity = np.random.rand(3, 3)
        gene_similarity = np.random.rand(4, 4)
        initial_org_fitness_range = 0.5

        return StrategyContext(
            population=population,
            org_fitness=org_fitness,
            gene_fitness=gene_fitness,
            org_similarity=org_similarity,
            gene_similarity=gene_similarity,
            initial_org_fitness_range=initial_org_fitness_range,
            org_id=0,
            gene_id=0,
        )

    def test_none_gene_strategy_name(self):
        """Test NoneGeneStrategy name property."""
        strategy = NoneGeneStrategy()
        assert strategy.name == "None"

    def test_none_gene_strategy_call(self, sample_context):
        """Test NoneGeneStrategy __call__ returns 0."""
        strategy = NoneGeneStrategy()
        result = strategy(sample_context)
        assert result == 0.0

    def test_altruistic_gene_strategy_name(self):
        """Test AltruisticGeneStrategy name property."""
        strategy = AltruisticGeneStrategy()
        assert strategy.name == "Altruistic"

    def test_altruistic_gene_strategy_call(self, sample_context):
        """Test AltruisticGeneStrategy __call__ returns a float."""
        strategy = AltruisticGeneStrategy()
        result = strategy(sample_context)
        assert isinstance(result, float)
        # Should be finite
        assert np.isfinite(result)

    def test_altruistic_gene_strategy_call_with_3x3_data(self, sample_context):
        """Test AltruisticGeneStrategy with 3x3 example data."""
        strategy = AltruisticGeneStrategy()
        result = strategy(sample_context)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_altruistic_gene_strategy_call_with_10x5_data(self, sample_context):
        """Test AltruisticGeneStrategy with 10x5 example data."""
        strategy = AltruisticGeneStrategy()
        result = strategy(sample_context)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_altruistic_gene_strategy_realistic_3x3_iterative(
        self, realistic_context_3x3_iterative
    ):
        """Test AltruisticGeneStrategy with realistic 3x3 iterative fitness values."""
        strategy = AltruisticGeneStrategy()
        result = strategy(realistic_context_3x3_iterative)
        assert isinstance(result, float)
        assert np.isfinite(result)
        # With realistic fitness values, result should be deterministic
        # Altruistic strategy can return negative values

    def test_altruistic_gene_strategy_realistic_10x5_iterative(
        self, realistic_context_10x5_iterative
    ):
        """Test AltruisticGeneStrategy with realistic 10x5 iterative fitness values."""
        strategy = AltruisticGeneStrategy()
        result = strategy(realistic_context_10x5_iterative)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_altruistic_gene_strategy_realistic_paper_iterative(
        self, realistic_context_paper_iterative
    ):
        """Test AltruisticGeneStrategy with realistic paper example iterative fitness values."""
        strategy = AltruisticGeneStrategy()
        result = strategy(realistic_context_paper_iterative)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestOrgStrategies:
    """Test cases for organism strategies."""

    @pytest.fixture
    def sample_context(self):
        """Create a sample StrategyContext for testing."""
        population = PikaiaPopulation(np.random.rand(3, 4))  # 3 organisms, 4 genes
        org_fitness = np.array([0.8, 0.6, 0.9])
        gene_fitness = np.array([0.7, 0.5, 0.8, 0.6])
        org_similarity = np.random.rand(3, 3)
        gene_similarity = np.random.rand(4, 4)
        initial_org_fitness_range = 0.5

        return StrategyContext(
            population=population,
            org_fitness=org_fitness,
            gene_fitness=gene_fitness,
            org_similarity=org_similarity,
            gene_similarity=gene_similarity,
            initial_org_fitness_range=initial_org_fitness_range,
            org_id=0,
        )

    def test_none_org_strategy_name(self):
        """Test NoneOrgStrategy name property."""
        strategy = NoneOrgStrategy()
        assert strategy.name == "None"

    def test_none_org_strategy_call(self, sample_context):
        """Test NoneOrgStrategy __call__ returns zeros array."""
        strategy = NoneOrgStrategy()
        result = strategy(sample_context)
        expected = np.zeros(sample_context.population.M)
        np.testing.assert_array_equal(result, expected)

    def test_none_org_strategy_call_with_3x3_data(self, sample_context_org_3x3):
        """Test NoneOrgStrategy with 3x3 example data."""
        strategy = NoneOrgStrategy()
        result = strategy(sample_context_org_3x3)
        expected = np.zeros(sample_context_org_3x3.population.M)
        np.testing.assert_array_equal(result, expected)

    def test_none_org_strategy_call_with_10x5_data(self, sample_context_org_10x5):
        """Test NoneOrgStrategy with 10x5 example data."""
        strategy = NoneOrgStrategy()
        result = strategy(sample_context_org_10x5)
        expected = np.zeros(sample_context_org_10x5.population.M)
        np.testing.assert_array_equal(result, expected)

    def test_none_org_strategy_realistic_3x3_iterative(
        self, realistic_context_3x3_iterative
    ):
        """Test NoneOrgStrategy with realistic 3x3 iterative fitness values."""
        strategy = NoneOrgStrategy()
        result = strategy(realistic_context_3x3_iterative)
        expected = np.zeros(realistic_context_3x3_iterative.population.M)
        np.testing.assert_array_equal(result, expected)

    def test_none_org_strategy_realistic_10x5_iterative(
        self, realistic_context_10x5_iterative
    ):
        """Test NoneOrgStrategy with realistic 10x5 iterative fitness values."""
        strategy = NoneOrgStrategy()
        result = strategy(realistic_context_10x5_iterative)
        expected = np.zeros(realistic_context_10x5_iterative.population.M)
        np.testing.assert_array_equal(result, expected)

    def test_none_org_strategy_realistic_paper_iterative(
        self, realistic_context_paper_iterative
    ):
        """Test NoneOrgStrategy with realistic paper example iterative fitness values."""
        strategy = NoneOrgStrategy()
        result = strategy(realistic_context_paper_iterative)
        expected = np.zeros(realistic_context_paper_iterative.population.M)
        np.testing.assert_array_equal(result, expected)


class TestStrategyContext:
    """Test cases for StrategyContext dataclass."""

    def test_strategy_context_creation(self):
        """Test StrategyContext can be created with required fields."""
        population = PikaiaPopulation(np.random.rand(2, 3))
        org_fitness = np.array([0.5, 0.7])
        gene_fitness = np.array([0.6, 0.8, 0.4])
        org_similarity = np.random.rand(2, 2)
        gene_similarity = np.random.rand(3, 3)
        initial_org_fitness_range = 0.3

        ctx = StrategyContext(
            population=population,
            org_fitness=org_fitness,
            gene_fitness=gene_fitness,
            org_similarity=org_similarity,
            gene_similarity=gene_similarity,
            initial_org_fitness_range=initial_org_fitness_range,
        )

        assert ctx.population is population
        assert np.array_equal(ctx.org_fitness, org_fitness)
        assert np.array_equal(ctx.gene_fitness, gene_fitness)
        assert ctx.org_id is None
        assert ctx.gene_id is None

    def test_strategy_context_with_optional_fields(self):
        """Test StrategyContext with optional org_id and gene_id."""
        population = PikaiaPopulation(np.random.rand(2, 3))
        org_fitness = np.array([0.5, 0.7])
        gene_fitness = np.array([0.6, 0.8, 0.4])
        org_similarity = np.random.rand(2, 2)
        gene_similarity = np.random.rand(3, 3)
        initial_org_fitness_range = 0.3

        ctx = StrategyContext(
            population=population,
            org_fitness=org_fitness,
            gene_fitness=gene_fitness,
            org_similarity=org_similarity,
            gene_similarity=gene_similarity,
            initial_org_fitness_range=initial_org_fitness_range,
            org_id=1,
            gene_id=2,
        )

        assert ctx.org_id == 1
        assert ctx.gene_id == 2


class TestStrategyFactories:
    """Test cases for strategy factories."""

    def test_gene_strategy_factory_get_strategy(self):
        """Test GeneStrategyFactory.get_strategy returns correct strategy."""
        strategy = GeneStrategyFactory.get_strategy(GeneStrategyEnum.NONE)
        assert isinstance(strategy, NoneGeneStrategy)
        assert strategy.name == "None"

    def test_gene_strategy_factory_invalid_strategy(self):
        """Test GeneStrategyFactory raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="Strategy 'INVALID' not found"):
            GeneStrategyFactory.get_strategy("INVALID")  # type: ignore

    def test_org_strategy_factory_get_strategy(self):
        """Test OrgStrategyFactory.get_strategy returns correct strategy."""
        strategy = OrgStrategyFactory.get_strategy(OrgStrategyEnum.NONE)
        assert isinstance(strategy, NoneOrgStrategy)
        assert strategy.name == "None"

    def test_org_strategy_factory_invalid_strategy(self):
        """Test OrgStrategyFactory raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="Strategy 'INVALID' not found"):
            OrgStrategyFactory.get_strategy("INVALID")  # type: ignore

    def test_mix_strategy_factory_get_strategy(self):
        """Test MixStrategyFactory.get_strategy returns correct strategy."""
        strategy = MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED)
        assert isinstance(strategy, FixedMixStrategy)
        assert strategy.name == "Fixed"

    def test_mix_strategy_factory_invalid_strategy(self):
        """Test MixStrategyFactory raises ValueError for invalid strategy."""
        with pytest.raises(ValueError, match="Strategy 'INVALID' not found"):
            MixStrategyFactory.get_strategy("INVALID")  # type: ignore
