import numpy as np
import pytest

from pikaia.data.population import PikaiaPopulation
from pikaia.preprocessing.pikaia_preprocessor import PikaiaPreprocessor
from pikaia.preprocessing.utils import max_scaler, min_max_scaler
from pikaia.schemas.preprocessing import FeatureType
from pikaia.strategies.base_strategies import StrategyContext
from tests.fixtures.realistic_fitness_values import REALISTIC_FITNESS_VALUES


@pytest.fixture
def data_3x3_raw():
    """Raw 3x3 dataset from examples.ipynb - flight data with price, time, stops."""
    return np.array([[300, 10, 2], [600, 5, 2], [1500, 4, 1]])


@pytest.fixture
def data_10x5_raw():
    """Raw 10x5 dataset from examples.ipynb - extended flight data."""
    return np.array(
        [
            [300, 10, 2, 0, 2.5],
            [600, 5, 2, 1, 3.0],
            [1500, 4, 1, 2, 4.0],
            [400, 8, 2, 0, 3.5],
            [500, 8, 2, 1, 3.0],
            [700, 5, 2, 1, 4.5],
            [900, 6, 1, 1, 4.0],
            [1100, 6, 1, 2, 3.5],
            [1300, 5, 2, 2, 5.0],
            [1700, 4, 1, 2, 5.0],
        ]
    )


@pytest.fixture
def paper_example_X():
    """Raw apartment data X from paper_example.ipynb."""
    return np.array(
        [
            [4348, 138, 3.0, 0],
            [2647, 133, 4.0, 0],
            [7413, 460, 7.0, 0],
            [5644, 329, 6.0, 0],
            [5979, 252, 6.0, 1],
            [5016, 219, 6.0, 0],
            [1106, 123, 2.0, 0],
            [4409, 175, 5.0, 0],
            [7708, 230, 8.0, 0],
            [5143, 159, 4.0, 0],
            [1650, 133, 3.0, 0],
            [7933, 383, 14.5, 1],
            [7912, 314, 7.0, 0],
            [8442, 335, 7.0, 0],
            [3218, 165, 3.0, 0],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def data_3x3_scaled(data_3x3_raw):
    """Scaled 3x3 dataset using the same preprocessing as examples."""
    feature_types = [FeatureType.COST, FeatureType.COST, FeatureType.COST]
    feature_transforms = [min_max_scaler] * data_3x3_raw.shape[1]
    preprocessor = PikaiaPreprocessor(
        num_features=data_3x3_raw.shape[1],
        feature_types=feature_types,
        feature_transforms=feature_transforms,
    )
    return preprocessor.fit_transform(data_3x3_raw)


@pytest.fixture
def data_10x5_scaled(data_10x5_raw):
    """Scaled 10x5 dataset using the same preprocessing as examples."""
    feature_types = [
        FeatureType.COST,  # price
        FeatureType.COST,  # time
        FeatureType.COST,  # stops
        FeatureType.GAIN,  # luggage
        FeatureType.GAIN,  # rating
    ]
    feature_transforms = [min_max_scaler] * data_10x5_raw.shape[1]
    preprocessor = PikaiaPreprocessor(
        num_features=data_10x5_raw.shape[1],
        feature_types=feature_types,
        feature_transforms=feature_transforms,
    )
    return preprocessor.fit_transform(data_10x5_raw)


@pytest.fixture
def paper_example_scaled(paper_example_X):
    """Scaled apartment data using the same preprocessing as paper_example."""
    feature_types = [
        FeatureType.COST,  # Rent
        FeatureType.GAIN,  # Size
        FeatureType.GAIN,  # Rooms
        FeatureType.GAIN,  # Balcony
    ]
    feature_transforms = [max_scaler] * paper_example_X.shape[1]
    preprocessor = PikaiaPreprocessor(
        num_features=paper_example_X.shape[1],
        feature_types=feature_types,
        feature_transforms=feature_transforms,
    )
    return preprocessor.fit_transform(paper_example_X)


@pytest.fixture
def population_3x3(data_3x3_scaled):
    """PikaiaPopulation from scaled 3x3 data."""
    return PikaiaPopulation(data_3x3_scaled)


@pytest.fixture
def population_10x5(data_10x5_scaled):
    """PikaiaPopulation from scaled 10x5 data."""
    return PikaiaPopulation(data_10x5_scaled)


@pytest.fixture
def population_paper_example(paper_example_scaled):
    """PikaiaPopulation from scaled paper example data."""
    return PikaiaPopulation(paper_example_scaled)


@pytest.fixture
def sample_context_3x3(population_3x3):
    """Create a sample StrategyContext using 3x3 example data."""
    org_fitness = np.array([0.8, 0.6, 0.9])
    gene_fitness = np.array([0.7, 0.5, 0.8])
    org_similarity = np.random.rand(3, 3)
    gene_similarity = np.random.rand(3, 3)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_3x3,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
        gene_id=0,
    )


@pytest.fixture
def sample_context_10x5(population_10x5):
    """Create a sample StrategyContext using 10x5 example data."""
    org_fitness = np.random.rand(10)
    gene_fitness = np.random.rand(5)
    org_similarity = np.random.rand(10, 10)
    gene_similarity = np.random.rand(5, 5)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_10x5,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
        gene_id=0,
    )


@pytest.fixture
def sample_context_org_3x3(population_3x3):
    """Create a sample StrategyContext for org strategies using 3x3 example data."""
    org_fitness = np.array([0.8, 0.6, 0.9])
    gene_fitness = np.array([0.7, 0.5, 0.8])
    org_similarity = np.random.rand(3, 3)
    gene_similarity = np.random.rand(3, 3)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_3x3,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
    )


@pytest.fixture
def sample_context_org_10x5(population_10x5):
    """Create a sample StrategyContext for org strategies using 10x5 example data."""
    org_fitness = np.random.rand(10)
    gene_fitness = np.random.rand(5)
    org_similarity = np.random.rand(10, 10)
    gene_similarity = np.random.rand(5, 5)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_10x5,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
    )


@pytest.fixture
def realistic_fitness_3x3_iterative():
    """Realistic fitness values from 3x3 example with 32 iterations."""
    return REALISTIC_FITNESS_VALUES["3x3_iterative_32"]


@pytest.fixture
def realistic_fitness_3x3_single_point():
    """Realistic fitness values from 3x3 example with single point (1 iteration)."""
    return REALISTIC_FITNESS_VALUES["3x3_single_point"]


@pytest.fixture
def realistic_fitness_10x5_iterative():
    """Realistic fitness values from 10x5 example with 32 iterations."""
    return REALISTIC_FITNESS_VALUES["10x5_iterative_32"]


@pytest.fixture
def realistic_fitness_10x5_single_point():
    """Realistic fitness values from 10x5 example with single point (1 iteration)."""
    return REALISTIC_FITNESS_VALUES["10x5_single_point"]


@pytest.fixture
def realistic_fitness_paper_iterative():
    """Realistic fitness values from paper example with 32 iterations."""
    return REALISTIC_FITNESS_VALUES["paper_iterative_32"]


@pytest.fixture
def realistic_fitness_paper_single_point():
    """Realistic fitness values from paper example with single point (1 iteration)."""
    return REALISTIC_FITNESS_VALUES["paper_single_point"]


@pytest.fixture
def realistic_context_3x3_iterative(population_3x3, realistic_fitness_3x3_iterative):
    """StrategyContext with realistic fitness values from 3x3 iterative run."""
    org_fitness = np.array(realistic_fitness_3x3_iterative["org_fitness"])
    gene_fitness = np.array(realistic_fitness_3x3_iterative["gene_fitness"])
    org_similarity = np.random.rand(3, 3)
    gene_similarity = np.random.rand(3, 3)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_3x3,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
        gene_id=0,
    )


@pytest.fixture
def realistic_context_10x5_iterative(population_10x5, realistic_fitness_10x5_iterative):
    """StrategyContext with realistic fitness values from 10x5 iterative run."""
    org_fitness = np.array(realistic_fitness_10x5_iterative["org_fitness"])
    gene_fitness = np.array(realistic_fitness_10x5_iterative["gene_fitness"])
    org_similarity = np.random.rand(10, 10)
    gene_similarity = np.random.rand(5, 5)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_10x5,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
        gene_id=0,
    )


@pytest.fixture
def realistic_context_paper_iterative(
    population_paper_example, realistic_fitness_paper_iterative
):
    """StrategyContext with realistic fitness values from paper example iterative run."""
    org_fitness = np.array(realistic_fitness_paper_iterative["org_fitness"])
    gene_fitness = np.array(realistic_fitness_paper_iterative["gene_fitness"])
    org_similarity = np.random.rand(15, 15)
    gene_similarity = np.random.rand(4, 4)
    initial_org_fitness_range = 0.5

    return StrategyContext(
        population=population_paper_example,
        org_fitness=org_fitness,
        gene_fitness=gene_fitness,
        org_similarity=org_similarity,
        gene_similarity=gene_similarity,
        initial_org_fitness_range=initial_org_fitness_range,
        org_id=0,
        gene_id=0,
    )
