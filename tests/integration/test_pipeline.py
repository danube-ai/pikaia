import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.preprocessing.pikaia_preprocessor import PikaiaPreprocessor
from pikaia.preprocessing.utils import max_scaler, min_max_scaler
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum
from pikaia.schemas.preprocessing import FeatureType
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory


class TestIntegration:
    """Integration tests for the full Pikaia pipeline."""

    def test_full_pipeline_3x3_example(self):
        """Test the complete pipeline with 3x3 flight data example."""
        # Raw data: [price, time, stops]
        raw_data = np.array([[300, 10, 2], [600, 5, 2], [1500, 4, 1]])

        # Preprocessing
        feature_types = [FeatureType.COST, FeatureType.COST, FeatureType.COST]
        feature_transforms = [min_max_scaler] * 3

        preprocessor = PikaiaPreprocessor(
            num_features=3,
            feature_types=feature_types,
            feature_transforms=feature_transforms,
        )

        scaled_data = preprocessor.fit_transform(raw_data)
        population = PikaiaPopulation(scaled_data)

        # Model setup
        gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.NONE)]
        org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.NONE)]

        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
            max_iter=5,
        )

        # Fit model
        model.fit()

        # Verify results
        assert model._gene_fitness_hist.shape[0] == 6  # initial + 5 iterations
        assert model._org_fitness_hist.shape[0] == 6
        assert len(model._gene_fitness_hist[-1]) == 3  # 3 genes
        assert len(model._org_fitness_hist[-1]) == 3  # 3 organisms

        # Check fitness values are reasonable
        gene_fitness = model._gene_fitness_hist[-1]
        org_fitness = model._org_fitness_hist[-1]

        assert np.all(np.isfinite(gene_fitness))
        assert np.all(np.isfinite(org_fitness))
        assert np.all((gene_fitness >= 0) & (gene_fitness <= 1))
        assert np.all((org_fitness >= 0) & (org_fitness <= 1))

    def test_full_pipeline_paper_example(self):
        """Test the complete pipeline with paper apartment data example."""
        # Raw apartment data: [rent, size, rooms, balcony]
        raw_data = np.array(
            [
                [4348, 138, 3.0, 0],
                [2647, 133, 4.0, 0],
                [7413, 460, 7.0, 0],
                [5644, 329, 6.0, 0],
                [5979, 252, 6.0, 1],
            ]
        )

        # Preprocessing
        feature_types = [
            FeatureType.COST,  # Rent
            FeatureType.GAIN,  # Size
            FeatureType.GAIN,  # Rooms
            FeatureType.GAIN,  # Balcony
        ]
        feature_transforms = [max_scaler] * 4

        preprocessor = PikaiaPreprocessor(
            num_features=4,
            feature_types=feature_types,
            feature_transforms=feature_transforms,
        )

        scaled_data = preprocessor.fit_transform(raw_data)
        population = PikaiaPopulation(scaled_data)

        # Model setup with different strategies
        gene_strategies = [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.ALTRUISTIC)
        ]
        org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)]

        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
            max_iter=3,
        )

        # Fit model
        model.fit()

        # Verify results
        assert model._gene_fitness_hist.shape[0] == 4  # initial + 3 iterations
        assert model._org_fitness_hist.shape[0] == 4
        assert len(model._gene_fitness_hist[-1]) == 4  # 4 genes
        assert len(model._org_fitness_hist[-1]) == 5  # 5 organisms

        # Check fitness values are reasonable
        gene_fitness = model._gene_fitness_hist[-1]
        org_fitness = model._org_fitness_hist[-1]

        assert np.all(np.isfinite(gene_fitness))
        assert np.all(np.isfinite(org_fitness))
        assert np.all((gene_fitness >= 0) & (gene_fitness <= 1))
        assert np.all((org_fitness >= 0) & (org_fitness <= 1))

    def test_pipeline_with_analytical_solution(self):
        """Test pipeline using analytical solution (max_iter=None)."""
        # Create test data that won't result in identical genes
        raw_data = np.array([[1, 8, 3], [4, 2, 6], [7, 9, 1], [3, 5, 7]])

        # Simple preprocessing
        feature_types = [FeatureType.GAIN] * 3
        feature_transforms = [max_scaler] * 3

        preprocessor = PikaiaPreprocessor(
            num_features=3,
            feature_types=feature_types,
            feature_transforms=feature_transforms,
        )

        scaled_data = preprocessor.fit_transform(raw_data)
        population = PikaiaPopulation(scaled_data)

        # Model with analytical solution
        gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.NONE)]
        org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.NONE)]

        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
            # max_iter=None for analytical solution
        )

        # Fit model
        model.fit()

        # Verify results - analytical solution should have only 2 history entries
        assert model._gene_fitness_hist.shape[0] == 2  # initial + final
        assert model._org_fitness_hist.shape[0] == 2
        assert len(model._gene_fitness_hist[-1]) == 3  # 3 genes
        assert len(model._org_fitness_hist[-1]) == 4  # 4 organisms

    def test_large_dataset_pipeline(self):
        """Test pipeline with larger dataset to check scalability."""
        # Create larger synthetic dataset
        np.random.seed(42)  # For reproducibility
        raw_data = np.random.rand(50, 10) * 100  # 50 organisms, 10 features

        # Preprocessing
        feature_types = [FeatureType.GAIN] * 10
        feature_transforms = [max_scaler] * 10

        preprocessor = PikaiaPreprocessor(
            num_features=10,
            feature_types=feature_types,
            feature_transforms=feature_transforms,
        )

        scaled_data = preprocessor.fit_transform(raw_data)
        population = PikaiaPopulation(scaled_data)

        # Model setup
        gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.NONE)]
        org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.NONE)]

        model = PikaiaModel(
            population=population,
            gene_strategies=gene_strategies,
            org_strategies=org_strategies,
            max_iter=2,  # Keep iterations low for speed
        )

        # Fit model
        model.fit()

        # Verify results
        assert model._gene_fitness_hist.shape[0] == 3  # initial + 2 iterations
        assert model._org_fitness_hist.shape[0] == 3
        assert len(model._gene_fitness_hist[-1]) == 10  # 10 genes
        assert len(model._org_fitness_hist[-1]) == 50  # 50 organisms

        # Check fitness values are reasonable
        gene_fitness = model._gene_fitness_hist[-1]
        org_fitness = model._org_fitness_hist[-1]

        assert np.all(np.isfinite(gene_fitness))
        assert np.all(np.isfinite(org_fitness))
        assert np.all((gene_fitness >= 0) & (gene_fitness <= 1))
        assert np.all((org_fitness >= 0) & (org_fitness <= 1))

    def test_preprocessing_model_integration(self):
        """Test that preprocessing and model work together correctly."""
        # Create raw data with mixed feature types
        raw_data = np.array(
            [
                [1000, 50, 3, 1],  # price, size, rooms, balcony
                [2000, 75, 4, 0],
                [1500, 60, 3, 1],
            ]
        )

        # Preprocessing with mixed types
        feature_types = [
            FeatureType.COST,  # price
            FeatureType.GAIN,  # size
            FeatureType.GAIN,  # rooms
            FeatureType.GAIN,  # balcony
        ]
        feature_transforms = [min_max_scaler] * 4

        preprocessor = PikaiaPreprocessor(
            num_features=4,
            feature_types=feature_types,
            feature_transforms=feature_transforms,
        )

        scaled_data = preprocessor.fit_transform(raw_data)

        # Verify scaling worked correctly
        assert scaled_data.shape == (3, 4)
        assert np.all((scaled_data >= 0) & (scaled_data <= 1))

        # Verify that scaled data is different (not all zeros)
        assert not np.allclose(scaled_data, 0)

        # Create population - this should work
        population = PikaiaPopulation(scaled_data)
        assert population.N == 3  # 3 organisms
        assert population.M == 4  # 4 genes
