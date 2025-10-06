import numpy as np
import pytest

from pikaia.preprocessing.pikaia_preprocessor import PikaiaPreprocessor
from pikaia.schemas.preprocessing import FeatureType


class TestPikaiaPreprocessor:
    """Test cases for PikaiaPreprocessor class."""

    def test_init_valid_parameters(self):
        """Test initialization with valid parameters."""
        num_features = 3
        feature_types = [FeatureType.GAIN, FeatureType.COST, FeatureType.GAIN]
        feature_transforms = [None, lambda x: x / 10, None]
        preprocessor = PikaiaPreprocessor(
            num_features, feature_types, feature_transforms
        )
        assert preprocessor.num_features == 3
        assert preprocessor.feature_types == feature_types
        assert preprocessor.feature_transforms == feature_transforms

    def test_init_mismatched_feature_types_length(self):
        """Test initialization with mismatched feature_types length."""
        with pytest.raises(
            ValueError, match="Length of feature_types .* must match num_features"
        ):
            PikaiaPreprocessor(
                3, [FeatureType.GAIN, FeatureType.COST], [None, None, None]
            )

    def test_init_mismatched_feature_transforms_length(self):
        """Test initialization with mismatched feature_transforms length."""
        with pytest.raises(
            ValueError, match="Length of feature_transforms .* must match num_features"
        ):
            PikaiaPreprocessor(
                3, [FeatureType.GAIN, FeatureType.COST, FeatureType.GAIN], [None, None]
            )

    def test_fit_valid_data(self):
        """Test fit method with valid data."""
        preprocessor = PikaiaPreprocessor(
            2, [FeatureType.GAIN, FeatureType.COST], [None, None]
        )
        X = np.array([[1, 2], [3, 4]])
        result = preprocessor.fit(X)
        assert result is preprocessor

    def test_fit_wrong_number_of_features(self):
        """Test fit method with wrong number of features."""
        preprocessor = PikaiaPreprocessor(
            2, [FeatureType.GAIN, FeatureType.COST], [None, None]
        )
        X = np.array([[1, 2, 3], [4, 5, 6]])
        with pytest.raises(
            ValueError, match="Number of features in X .* does not match num_features"
        ):
            preprocessor.fit(X)

    def test_transform_no_transforms(self):
        """Test transform with no transformations."""
        preprocessor = PikaiaPreprocessor(
            2, [FeatureType.GAIN, FeatureType.GAIN], [None, None]
        )
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        X_transformed = preprocessor.transform(X)
        np.testing.assert_array_equal(X_transformed, X)

    def test_transform_with_custom_transform(self):
        """Test transform with custom transformation function."""
        preprocessor = PikaiaPreprocessor(
            2, [FeatureType.GAIN, FeatureType.GAIN], [lambda x: x * 2, None]
        )
        X = np.array([[0.1, 0.2], [0.3, 0.4]])
        X_transformed = preprocessor.transform(X)
        expected = np.array([[0.2, 0.2], [0.6, 0.4]])
        np.testing.assert_array_equal(X_transformed, expected)

    def test_transform_cost_feature_inversion(self):
        """Test that COST features are inverted."""
        preprocessor = PikaiaPreprocessor(
            2, [FeatureType.GAIN, FeatureType.COST], [None, None]
        )
        X = np.array([[0.1, 0.8], [0.3, 0.6]])
        X_transformed = preprocessor.transform(X)
        # GAIN feature unchanged, COST feature inverted: max + min - value = 0.8 + 0.6 - value
        expected = np.array([[0.1, 0.8 + 0.6 - 0.8], [0.3, 0.8 + 0.6 - 0.6]])
        np.testing.assert_array_equal(X_transformed, expected)

    def test_transform_out_of_range_warning(self, caplog):
        """Test warning when transformed values are out of [0,1] range."""
        preprocessor = PikaiaPreprocessor(1, [FeatureType.GAIN], [lambda x: x * 2])
        X = np.array([[0.6], [0.7]])
        preprocessor.transform(X)
        assert "outside [0,1] range" in caplog.text

    def test_fit_transform(self):
        """Test fit_transform method."""
        preprocessor = PikaiaPreprocessor(
            2, [FeatureType.GAIN, FeatureType.COST], [None, None]
        )
        X = np.array([[0.1, 0.8], [0.3, 0.6]])
        X_transformed = preprocessor.fit_transform(X)
        # Same as transform test above
        expected = np.array([[0.1, 0.8 + 0.6 - 0.8], [0.3, 0.8 + 0.6 - 0.6]])
        np.testing.assert_array_equal(X_transformed, expected)

    def test_transform_preserves_shape(self):
        """Test that transform preserves the shape of input data."""
        preprocessor = PikaiaPreprocessor(3, [FeatureType.GAIN] * 3, [None] * 3)
        X = np.random.rand(10, 3)
        X_transformed = preprocessor.transform(X)
        assert X_transformed.shape == X.shape

    def test_preprocess_3x3_data(self, data_3x3_raw, data_3x3_scaled):
        """Test preprocessing of 3x3 example data."""
        # Verify the scaled data is properly normalized
        assert np.all(data_3x3_scaled >= 0) and np.all(data_3x3_scaled <= 1)
        assert data_3x3_scaled.shape == data_3x3_raw.shape

    def test_preprocess_10x5_data(self, data_10x5_raw, data_10x5_scaled):
        """Test preprocessing of 10x5 example data."""
        # Verify the scaled data is properly normalized
        assert np.all(data_10x5_scaled >= 0) and np.all(data_10x5_scaled <= 1)
        assert data_10x5_scaled.shape == data_10x5_raw.shape

    def test_preprocess_paper_example_data(self, paper_example_X, paper_example_scaled):
        """Test preprocessing of paper example data."""
        # Verify the scaled data is properly normalized
        assert np.all(paper_example_scaled >= 0) and np.all(paper_example_scaled <= 1)
        assert paper_example_scaled.shape == paper_example_X.shape
