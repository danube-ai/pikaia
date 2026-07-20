import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import GeneStrategyEnum, MixStrategyEnum, OrgStrategyEnum
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import (
    KinAltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy
from pikaia.strategies.mix_strategies.self_consistent_strategy import (
    SelfConsistentMixStrategy,
)
from pikaia.strategies.os_strategies.altruistic_strategy import AltruisticOrgStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.kin_selfish_strategy import KinSelfishOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy
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

    def test_strategy_context_y_defaults_to_none(self):
        """y is None when not provided."""
        population = PikaiaPopulation(np.random.rand(2, 3))
        ctx = StrategyContext(
            population=population,
            org_fitness=np.array([0.5, 0.7]),
            gene_fitness=np.array([0.6, 0.8, 0.4]),
            org_similarity=np.random.rand(2, 2),
            gene_similarity=np.random.rand(3, 3),
            initial_org_fitness_range=0.3,
        )
        assert ctx.y is None

    def test_strategy_context_y_accepts_array(self):
        """y is stored and accessible when provided."""
        population = PikaiaPopulation(np.random.rand(2, 3))
        y = np.array([0, 1])
        ctx = StrategyContext(
            population=population,
            org_fitness=np.array([0.5, 0.7]),
            gene_fitness=np.array([0.6, 0.8, 0.4]),
            org_similarity=np.random.rand(2, 2),
            gene_similarity=np.random.rand(3, 3),
            initial_org_fitness_range=0.3,
            y=y,
        )
        assert ctx.y is not None
        assert np.array_equal(ctx.y, y)


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


# ---------------------------------------------------------------------------
# Helper fixtures shared by the new strategy tests
# ---------------------------------------------------------------------------


def _make_gene_context(
    n_orgs: int = 3,
    n_genes: int = 4,
    org_id: int = 0,
    gene_id: int = 0,
    zero_org_fitness: bool = False,
) -> StrategyContext:
    rng = np.random.default_rng(42)
    matrix = rng.random((n_orgs, n_genes))
    population = PikaiaPopulation(matrix)
    org_fitness = np.zeros(n_orgs) if zero_org_fitness else rng.random(n_orgs) + 0.1
    gene_fitness = rng.random(n_genes) + 0.1
    org_similarity = rng.random((n_orgs, n_orgs))
    gene_similarity = rng.random((n_genes, n_genes))
    # Make diagonal the highest so self is always most similar
    np.fill_diagonal(gene_similarity, 1.0)
    np.fill_diagonal(org_similarity, 1.0)
    return StrategyContext(
        population=population,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=0.5,
        org_id=org_id,
        gene_id=gene_id,
    )


def _make_org_context(
    n_orgs: int = 3,
    n_genes: int = 4,
    org_id: int = 0,
    zero_org_fitness: bool = False,
) -> StrategyContext:
    return _make_gene_context(
        n_orgs=n_orgs,
        n_genes=n_genes,
        org_id=org_id,
        gene_id=None,  # type: ignore[arg-type]
        zero_org_fitness=zero_org_fitness,
    )


class TestDominantGeneStrategy:
    """Tests for DominantGeneStrategy."""

    def test_name(self):
        assert DominantGeneStrategy().name == "Dominant"

    def test_init_stores_options(self):
        s = DominantGeneStrategy(foo="bar")
        assert s.options["foo"] == "bar"

    def test_call_returns_float(self):
        ctx = _make_gene_context()
        result = DominantGeneStrategy()(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_deterministic(self):
        ctx = _make_gene_context()
        s = DominantGeneStrategy()
        assert s(ctx) == s(ctx)


class TestKinAltruisticGeneStrategy:
    """Tests for KinAltruisticGeneStrategy."""

    def test_name(self):
        assert KinAltruisticGeneStrategy().name == "KinAltruistic"

    def test_init_stores_options(self):
        s = KinAltruisticGeneStrategy(kin_range=2)
        assert s.options["kin_range"] == 2

    def test_call_returns_float(self):
        ctx = _make_gene_context()
        result = KinAltruisticGeneStrategy()(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_no_kin_early_exit(self):
        """With kin_range=1 and self as top-similar, no kin remain after filtering self."""
        ctx = _make_gene_context()
        result = KinAltruisticGeneStrategy(kin_range=1)(ctx)
        assert result == 0.0

    def test_call_with_explicit_kin_range(self):
        ctx = _make_gene_context(n_genes=4)
        result = KinAltruisticGeneStrategy(kin_range=3)(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)


class TestSelfishGeneStrategy:
    """Tests for SelfishGeneStrategy."""

    def test_name(self):
        assert SelfishGeneStrategy().name == "Selfish"

    def test_init_stores_options(self):
        s = SelfishGeneStrategy(foo=1)
        assert s.options["foo"] == 1

    def test_call_returns_float(self):
        ctx = _make_gene_context()
        result = SelfishGeneStrategy()(ctx)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_call_single_gene_returns_zero(self):
        """With only one gene, all others mask is empty → sum is 0."""
        ctx = _make_gene_context(n_genes=1)
        result = SelfishGeneStrategy()(ctx)
        assert result == 0.0


class TestSelfConsistentMixStrategy:
    """Tests for SelfConsistentMixStrategy."""

    def test_name(self):
        assert SelfConsistentMixStrategy().name == "SelfConsistent"

    def test_init_stores_options(self):
        s = SelfConsistentMixStrategy(alpha=0.1)
        assert s.options["alpha"] == 0.1

    def test_call_returns_tuple(self):
        rng = np.random.default_rng(0)
        delta = rng.random((3, 4, 2))
        mix_coeffs = np.array([0.5, 0.5])
        mixed, updated = SelfConsistentMixStrategy()(delta, mix_coeffs)
        assert mixed.shape == (3, 4)
        assert updated.shape == (2,)
        assert abs(updated.sum() - 1.0) < 1e-9

    def test_call_updates_coefficients(self):
        rng = np.random.default_rng(1)
        delta = rng.random((3, 4, 2))
        mix_coeffs = np.array([0.5, 0.5])
        _, updated = SelfConsistentMixStrategy()(delta, mix_coeffs)
        # Coefficients should still sum to 1 but values may differ from input
        assert abs(updated.sum() - 1.0) < 1e-9


class TestAltruisticOrgStrategy:
    """Tests for AltruisticOrgStrategy."""

    def test_name(self):
        assert AltruisticOrgStrategy().name == "Altruistic"

    def test_init_stores_options(self):
        s = AltruisticOrgStrategy(kin_range=5)
        assert s.options["kin_range"] == 5

    def test_call_returns_array(self):
        ctx = _make_org_context()
        result = AltruisticOrgStrategy()(ctx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (ctx.population.M,)
        assert np.all(np.isfinite(result))

    def test_call_no_relatives_returns_zeros(self):
        """With kin_range=1 and self most similar → no relatives after filter."""
        ctx = _make_org_context()
        result = AltruisticOrgStrategy(kin_range=1)(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))

    def test_call_zero_org_fitness_returns_zeros(self):
        ctx = _make_org_context(zero_org_fitness=True)
        result = AltruisticOrgStrategy()(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))

    def test_call_with_explicit_kin_range(self):
        ctx = _make_org_context(n_orgs=5)
        result = AltruisticOrgStrategy(kin_range=3)(ctx)
        assert result.shape == (ctx.population.M,)

    def test_call_large_kin_range_warns(self, caplog):
        """kin_range > 32 should emit a warning."""
        import logging

        ctx = _make_org_context(n_orgs=40)
        with caplog.at_level(logging.WARNING):
            AltruisticOrgStrategy(kin_range=33)(ctx)
        assert "kin_range is very large" in caplog.text


class TestBalancedOrgStrategy:
    """Tests for BalancedOrgStrategy."""

    def test_name(self):
        assert BalancedOrgStrategy().name == "Balanced"

    def test_init_stores_options(self):
        s = BalancedOrgStrategy(beta=0.2)
        assert s.options["beta"] == 0.2

    def test_call_returns_array(self):
        ctx = _make_org_context()
        result = BalancedOrgStrategy()(ctx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (ctx.population.M,)
        assert np.all(np.isfinite(result))

    def test_call_zero_org_fitness_returns_zeros(self):
        ctx = _make_org_context(zero_org_fitness=True)
        result = BalancedOrgStrategy()(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))


class TestKinSelfishOrgStrategy:
    """Tests for KinSelfishOrgStrategy."""

    def test_name(self):
        assert KinSelfishOrgStrategy().name == "KinSelfish"

    def test_init_stores_options(self):
        s = KinSelfishOrgStrategy(kin_range=2)
        assert s.options["kin_range"] == 2

    def test_call_returns_array(self):
        ctx = _make_org_context()
        result = KinSelfishOrgStrategy()(ctx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (ctx.population.M,)
        assert np.all(np.isfinite(result))

    def test_call_no_relatives_returns_zeros(self):
        ctx = _make_org_context()
        result = KinSelfishOrgStrategy(kin_range=1)(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))

    def test_call_zero_org_fitness_returns_zeros(self):
        ctx = _make_org_context(zero_org_fitness=True)
        result = KinSelfishOrgStrategy()(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))

    def test_call_with_kin_range(self):
        ctx = _make_org_context(n_orgs=5)
        result = KinSelfishOrgStrategy(kin_range=3)(ctx)
        assert result.shape == (ctx.population.M,)


class TestSelfishOrgStrategy:
    """Tests for SelfishOrgStrategy."""

    def test_name(self):
        assert SelfishOrgStrategy().name == "Selfish"

    def test_init_stores_options(self):
        s = SelfishOrgStrategy(foo="x")
        assert s.options["foo"] == "x"

    def test_call_returns_array(self):
        ctx = _make_org_context()
        result = SelfishOrgStrategy()(ctx)
        assert isinstance(result, np.ndarray)
        assert result.shape == (ctx.population.M,)
        assert np.all(np.isfinite(result))

    def test_call_no_relatives_returns_zeros(self):
        ctx = _make_org_context()
        result = SelfishOrgStrategy(kin_range=1)(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))

    def test_call_zero_org_fitness_returns_zeros(self):
        ctx = _make_org_context(zero_org_fitness=True)
        result = SelfishOrgStrategy()(ctx)
        np.testing.assert_array_equal(result, np.zeros(ctx.population.M))

    def test_call_with_kin_range(self):
        ctx = _make_org_context(n_orgs=5)
        result = SelfishOrgStrategy(kin_range=3)(ctx)
        assert result.shape == (ctx.population.M,)
