import numpy as np
import pytest

# New imports
from pikaia.data import PikaiaPopulation
from pikaia.strategies import StrategyContext
from pikaia.strategies.gs_strategies import (
    AltruisticGeneStrategy,
    DominantGeneStrategy,
    KinAltruisticGeneStrategy,
    NoneGeneStrategy,
    SelfishGeneStrategy,
)
from pikaia.strategies.os_strategies import (
    AltruisticOrgStrategy,
    BalancedOrgStrategy,
    KinSelfishOrgStrategy,
    NoneOrgStrategy,
    SelfishOrgStrategy,
)

# print
# Legacy imports
from tests.legacy.alg import (
    GSStrategy as LegacyGSStrategy,
)
from tests.legacy.alg import (
    OSStrategy as LegacyOSStrategy,
)
from tests.legacy.alg import (
    Population as LegacyPopulation,
)
from tests.legacy.alg import (
    Strategies as LegacyStrategies,
)
from tests.legacy.alg import (
    compute_all_organism_fitness,
)
from tests.legacy.alg import (
    compute_gene_deltas as legacy_compute_gene_deltas,
)
from tests.legacy.alg import (
    compute_organism_deltas as legacy_compute_organism_deltas,
)


@pytest.fixture
def setup_data():
    """Set up common data for tests."""
    rawdata = np.array([[0.1, 0.2, 0.7], [0.4, 0.5, 0.1], [0.8, 0.2, 0.0]])
    gvfitnessrule = [None, None, None]  # Simplified for testing
    kinrange = 2

    # Legacy population
    legacy_pop = LegacyPopulation(rawdata, gvfitnessrule, silent=True)
    legacy_pop._populationdata = rawdata  # Override with raw data for simplicity

    # New population
    pikaia_pop = PikaiaPopulation(rawdata)

    genefitness = np.array([0.3, 0.5, 0.2])
    genesimilarity = np.random.rand(3, 3)
    orgsimilarity = np.random.rand(3, 3)
    orgfitness = compute_all_organism_fitness(legacy_pop.data, genefitness)
    initialorgfitnessrange = np.max(orgfitness) - np.min(orgfitness)

    return {
        "legacy_pop": legacy_pop,
        "pikaia_pop": pikaia_pop,
        "genefitness": genefitness,
        "genesimilarity": genesimilarity,
        "orgsimilarity": orgsimilarity,
        "orgfitness": orgfitness,
        "initialorgfitnessrange": initialorgfitnessrange,
        "kinrange": kinrange,
    }


def test_dominant_gene_strategy_equivalence(setup_data):
    """Test equivalence for Dominant Gene Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.DOMINANT, osstrategy=LegacyOSStrategy.NONE
    )
    new_strategy = DominantGeneStrategy()

    for i in range(setup_data["legacy_pop"].n):
        for j in range(setup_data["legacy_pop"].m):
            # Legacy calculation
            legacy_delta = legacy_compute_gene_deltas(
                j,
                setup_data["legacy_pop"].data[i, j],
                setup_data["legacy_pop"].data[:, j],
                setup_data["legacy_pop"].data[i, :],
                setup_data["genefitness"],
                setup_data["genesimilarity"][j, :],
                legacy_strategies,
            )

            # New calculation
            ctx = StrategyContext(
                population=setup_data["pikaia_pop"],
                org_fitness=setup_data["orgfitness"],
                gene_fitness=setup_data["genefitness"],
                org_similarity=setup_data["orgsimilarity"],
                gene_similarity=setup_data["genesimilarity"],
                initial_org_fitness_range=setup_data["initialorgfitnessrange"],
                org_id=i,
                gene_id=j,
            )
            new_delta = new_strategy(ctx)

            # Compare results
            assert np.allclose(legacy_delta, new_delta), (
                f"Mismatch at organism {i}, gene {j}"
            )


def test_altruistic_gene_strategy_equivalence(setup_data):
    """Test equivalence for Altruistic Gene Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.ALTRUISTIC, osstrategy=LegacyOSStrategy.NONE
    )
    new_strategy = AltruisticGeneStrategy()

    for i in range(setup_data["legacy_pop"].n):
        for j in range(setup_data["legacy_pop"].m):
            legacy_delta = legacy_compute_gene_deltas(
                j,
                setup_data["legacy_pop"].data[i, j],
                setup_data["legacy_pop"].data[:, j],
                setup_data["legacy_pop"].data[i, :],
                setup_data["genefitness"],
                setup_data["genesimilarity"][j, :],
                legacy_strategies,
            )
            ctx = StrategyContext(
                population=setup_data["pikaia_pop"],
                org_fitness=setup_data["orgfitness"],
                gene_fitness=setup_data["genefitness"],
                org_similarity=setup_data["orgsimilarity"],
                gene_similarity=setup_data["genesimilarity"],
                initial_org_fitness_range=setup_data["initialorgfitnessrange"],
                org_id=i,
                gene_id=j,
            )
            new_delta = new_strategy(ctx)
            assert np.allclose(legacy_delta, new_delta), (
                f"Mismatch at organism {i}, gene {j}"
            )


def test_kin_altruistic_gene_strategy_equivalence(setup_data):
    """Test equivalence for Kin-Altruistic Gene Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.KIN_ALTRUISTIC,
        osstrategy=LegacyOSStrategy.NONE,
    )
    new_strategy = KinAltruisticGeneStrategy()

    for i in range(setup_data["legacy_pop"].n):
        for j in range(setup_data["legacy_pop"].m):
            legacy_delta = legacy_compute_gene_deltas(
                j,
                setup_data["legacy_pop"].data[i, j],
                setup_data["legacy_pop"].data[:, j],
                setup_data["legacy_pop"].data[i, :],
                setup_data["genefitness"],
                setup_data["genesimilarity"][j, :],
                legacy_strategies,
            )
            ctx = StrategyContext(
                population=setup_data["pikaia_pop"],
                org_fitness=setup_data["orgfitness"],
                gene_fitness=setup_data["genefitness"],
                org_similarity=setup_data["orgsimilarity"],
                gene_similarity=setup_data["genesimilarity"],
                initial_org_fitness_range=setup_data["initialorgfitnessrange"],
                org_id=i,
                gene_id=j,
            )
            new_delta = new_strategy(ctx)
            assert np.allclose(legacy_delta, new_delta), (
                f"Mismatch at organism {i}, gene {j}"
            )


def test_selfish_gene_strategy_equivalence(setup_data):
    """Test equivalence for Selfish Gene Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.SELFISH, osstrategy=LegacyOSStrategy.NONE
    )
    new_strategy = SelfishGeneStrategy()

    for i in range(setup_data["legacy_pop"].n):
        for j in range(setup_data["legacy_pop"].m):
            legacy_delta = legacy_compute_gene_deltas(
                j,
                setup_data["legacy_pop"].data[i, j],
                setup_data["legacy_pop"].data[:, j],
                setup_data["legacy_pop"].data[i, :],
                setup_data["genefitness"],
                setup_data["genesimilarity"][j, :],
                legacy_strategies,
            )
            ctx = StrategyContext(
                population=setup_data["pikaia_pop"],
                org_fitness=setup_data["orgfitness"],
                gene_fitness=setup_data["genefitness"],
                org_similarity=setup_data["orgsimilarity"],
                gene_similarity=setup_data["genesimilarity"],
                initial_org_fitness_range=setup_data["initialorgfitnessrange"],
                org_id=i,
                gene_id=j,
            )
            new_delta = new_strategy(ctx)
            assert np.allclose(legacy_delta, new_delta), (
                f"Mismatch at organism {i}, gene {j}"
            )


def test_none_gene_strategy_equivalence(setup_data):
    """Test equivalence for None Gene Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.NONE, osstrategy=LegacyOSStrategy.NONE
    )
    new_strategy = NoneGeneStrategy()

    for i in range(setup_data["legacy_pop"].n):
        for j in range(setup_data["legacy_pop"].m):
            legacy_delta = legacy_compute_gene_deltas(
                j,
                setup_data["legacy_pop"].data[i, j],
                setup_data["legacy_pop"].data[:, j],
                setup_data["legacy_pop"].data[i, :],
                setup_data["genefitness"],
                setup_data["genesimilarity"][j, :],
                legacy_strategies,
            )
            ctx = StrategyContext(
                population=setup_data["pikaia_pop"],
                org_fitness=setup_data["orgfitness"],
                gene_fitness=setup_data["genefitness"],
                org_similarity=setup_data["orgsimilarity"],
                gene_similarity=setup_data["genesimilarity"],
                initial_org_fitness_range=setup_data["initialorgfitnessrange"],
                org_id=i,
                gene_id=j,
            )
            new_delta = new_strategy(ctx)
            assert np.allclose(legacy_delta, new_delta), (
                f"Mismatch at organism {i}, gene {j}"
            )


def test_balanced_organism_strategy_equivalence(setup_data):
    """Test equivalence for Balanced Organism Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.NONE, osstrategy=LegacyOSStrategy.BALANCED
    )
    new_strategy = BalancedOrgStrategy()

    for i in range(setup_data["legacy_pop"].n):
        # Legacy calculation
        legacy_delta = legacy_compute_organism_deltas(
            i,
            setup_data["legacy_pop"].data,
            setup_data["genefitness"],
            setup_data["orgfitness"],
            setup_data["initialorgfitnessrange"],
            setup_data["orgsimilarity"][i, :],
            legacy_strategies,
        )

        # New calculation
        ctx = StrategyContext(
            population=setup_data["pikaia_pop"],
            org_fitness=setup_data["orgfitness"],
            gene_fitness=setup_data["genefitness"],
            org_similarity=setup_data["orgsimilarity"],
            gene_similarity=setup_data["genesimilarity"],
            initial_org_fitness_range=setup_data["initialorgfitnessrange"],
            org_id=i,
        )
        new_delta = new_strategy(ctx)

        # Compare results
        assert np.allclose(legacy_delta, new_delta), f"Mismatch at organism {i}"


def test_altruistic_organism_strategy_equivalence(setup_data):
    """Test equivalence for Altruistic Organism Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.NONE,
        osstrategy=LegacyOSStrategy.ALTRUISTIC,
        kinrange=setup_data["kinrange"],
    )
    new_strategy = AltruisticOrgStrategy(kin_range=setup_data["kinrange"])

    for i in range(setup_data["legacy_pop"].n):
        legacy_delta = legacy_compute_organism_deltas(
            i,
            setup_data["legacy_pop"].data,
            setup_data["genefitness"],
            setup_data["orgfitness"],
            setup_data["initialorgfitnessrange"],
            setup_data["orgsimilarity"][i, :],
            legacy_strategies,
        )
        ctx = StrategyContext(
            population=setup_data["pikaia_pop"],
            org_fitness=setup_data["orgfitness"],
            gene_fitness=setup_data["genefitness"],
            org_similarity=setup_data["orgsimilarity"],
            gene_similarity=setup_data["genesimilarity"],
            initial_org_fitness_range=setup_data["initialorgfitnessrange"],
            org_id=i,
        )
        new_delta = new_strategy(ctx)
        assert np.allclose(legacy_delta, new_delta), f"Mismatch at organism {i}"


def test_kin_selfish_organism_strategy_equivalence(setup_data):
    """Test equivalence for Kin-Selfish Organism Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.NONE,
        osstrategy=LegacyOSStrategy.KIN_SELFISH,
        kinrange=setup_data["kinrange"],
    )
    new_strategy = KinSelfishOrgStrategy(kin_range=setup_data["kinrange"])

    for i in range(setup_data["legacy_pop"].n):
        legacy_delta = legacy_compute_organism_deltas(
            i,
            setup_data["legacy_pop"].data,
            setup_data["genefitness"],
            setup_data["orgfitness"],
            setup_data["initialorgfitnessrange"],
            setup_data["orgsimilarity"][i, :],
            legacy_strategies,
        )
        ctx = StrategyContext(
            population=setup_data["pikaia_pop"],
            org_fitness=setup_data["orgfitness"],
            gene_fitness=setup_data["genefitness"],
            org_similarity=setup_data["orgsimilarity"],
            gene_similarity=setup_data["genesimilarity"],
            initial_org_fitness_range=setup_data["initialorgfitnessrange"],
            org_id=i,
        )
        new_delta = new_strategy(ctx)
        assert np.allclose(legacy_delta, new_delta), f"Mismatch at organism {i}"


def test_selfish_organism_strategy_equivalence(setup_data):
    """Test equivalence for Selfish Organism Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.NONE,
        osstrategy=LegacyOSStrategy.SELFISH,
        kinrange=setup_data["kinrange"],
    )
    new_strategy = SelfishOrgStrategy(kin_range=setup_data["kinrange"])

    for i in range(setup_data["legacy_pop"].n):
        legacy_delta = legacy_compute_organism_deltas(
            i,
            setup_data["legacy_pop"].data,
            setup_data["genefitness"],
            setup_data["orgfitness"],
            setup_data["initialorgfitnessrange"],
            setup_data["orgsimilarity"][i, :],
            legacy_strategies,
        )
        ctx = StrategyContext(
            population=setup_data["pikaia_pop"],
            org_fitness=setup_data["orgfitness"],
            gene_fitness=setup_data["genefitness"],
            org_similarity=setup_data["orgsimilarity"],
            gene_similarity=setup_data["genesimilarity"],
            initial_org_fitness_range=setup_data["initialorgfitnessrange"],
            org_id=i,
        )
        new_delta = new_strategy(ctx)
        assert np.allclose(legacy_delta, new_delta), f"Mismatch at organism {i}"


def test_none_organism_strategy_equivalence(setup_data):
    """Test equivalence for None Organism Strategy."""
    legacy_strategies = LegacyStrategies(
        gsstrategy=LegacyGSStrategy.NONE, osstrategy=LegacyOSStrategy.NONE
    )
    new_strategy = NoneOrgStrategy()

    for i in range(setup_data["legacy_pop"].n):
        legacy_delta = legacy_compute_organism_deltas(
            i,
            setup_data["legacy_pop"].data,
            setup_data["genefitness"],
            setup_data["orgfitness"],
            setup_data["initialorgfitnessrange"],
            setup_data["orgsimilarity"][i, :],
            legacy_strategies,
        )
        ctx = StrategyContext(
            population=setup_data["pikaia_pop"],
            org_fitness=setup_data["orgfitness"],
            gene_fitness=setup_data["genefitness"],
            org_similarity=setup_data["orgsimilarity"],
            gene_similarity=setup_data["genesimilarity"],
            initial_org_fitness_range=setup_data["initialorgfitnessrange"],
            org_id=i,
        )
        new_delta = new_strategy(ctx)
        assert np.allclose(legacy_delta, new_delta), f"Mismatch at organism {i}"
