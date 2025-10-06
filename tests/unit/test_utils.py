import numpy as np
import pytest

from pikaia.preprocessing.utils import (
    max_scaler,
    min_max_scaler,
    power_transform,
    robust_scaler,
    z_score_scaler,
)


class TestPreprocessingUtils:
    """Test cases for preprocessing utility functions."""

    def test_max_scaler_basic(self):
        """Test max_scaler with basic input."""
        arr = np.array([1, 2, 3, 4])
        result = max_scaler(arr)
        expected = np.array([0.25, 0.5, 0.75, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_max_scaler_with_numpy_array(self):
        """Test max_scaler with numpy array input."""
        arr = np.array([10, 20, 30])
        result = max_scaler(arr)
        expected = np.array([1 / 3, 2 / 3, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_max_scaler_empty_array(self):
        """Test max_scaler with empty array raises ValueError."""
        with pytest.raises(ValueError, match="Input array must not be empty"):
            max_scaler([])

    def test_max_scaler_zero_max(self):
        """Test max_scaler with all zeros raises ValueError."""
        with pytest.raises(ValueError, match="Maximum value must not be zero"):
            max_scaler([0, 0, 0])

    def test_min_max_scaler_basic(self):
        """Test min_max_scaler with basic input."""
        arr = np.array([1, 2, 3, 4])
        result = min_max_scaler(arr)
        expected = np.array([0.0, 1 / 3, 2 / 3, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_min_max_scaler_constant_values(self):
        """Test min_max_scaler with constant values."""
        arr = np.array([5, 5, 5])
        result = min_max_scaler(arr)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_min_max_scaler_empty_array(self):
        """Test min_max_scaler with empty array raises ValueError."""
        with pytest.raises(ValueError, match="Input array must not be empty"):
            min_max_scaler([])

    def test_robust_scaler_basic(self):
        """Test robust_scaler with basic input."""
        arr = np.array([1, 2, 3, 4, 5])
        result = robust_scaler(arr)
        # Should produce values in [0, 1]
        assert np.all(result >= 0) and np.all(result <= 1)
        assert result.shape == (5,)

    def test_robust_scaler_constant_values(self):
        """Test robust_scaler with constant values."""
        arr = np.array([2, 2, 2, 2])
        result = robust_scaler(arr)
        expected = np.array([0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_robust_scaler_empty_array(self):
        """Test robust_scaler with empty array raises ValueError."""
        with pytest.raises(ValueError, match="Input array must not be empty"):
            robust_scaler([])

    def test_power_transform_basic(self):
        """Test power_transform with basic input."""
        arr = np.array([1, 4, 9, 16])
        result = power_transform(arr, power=0.5)  # square root
        expected = np.array([0.0, 1 / 3, 2 / 3, 1.0])  # sqrt scaled to [0,1]
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_transform_different_power(self):
        """Test power_transform with different power."""
        arr = np.array([1, 8])
        result = power_transform(arr, power=1 / 3)  # cube root
        expected = np.array([0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_power_transform_negative_values(self):
        """Test power_transform with negative values raises ValueError."""
        with pytest.raises(
            ValueError, match="Power transform requires all values to be positive"
        ):
            power_transform([-1, 1, 2])

    def test_power_transform_zero_values(self):
        """Test power_transform with zero values raises ValueError."""
        with pytest.raises(
            ValueError, match="Power transform requires all values to be positive"
        ):
            power_transform([0, 1, 2])

    def test_power_transform_empty_array(self):
        """Test power_transform with empty array raises ValueError."""
        with pytest.raises(ValueError, match="Input array must not be empty"):
            power_transform([])

    def test_power_transform_constant_values(self):
        """Test power_transform with constant positive values."""
        arr = np.array([2, 2, 2])
        result = power_transform(arr)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_z_score_scaler_basic(self):
        """Test z_score_scaler with basic input."""
        arr = np.array([1, 2, 3, 4, 5])
        result = z_score_scaler(arr)
        # Should produce values in [0, 1]
        assert np.all(result >= 0) and np.all(result <= 1)
        assert result.shape == (5,)

    def test_z_score_scaler_constant_values(self):
        """Test z_score_scaler with constant values."""
        arr = np.array([3, 3, 3])
        result = z_score_scaler(arr)
        expected = np.array([0.0, 0.0, 0.0])
        np.testing.assert_array_equal(result, expected)

    def test_z_score_scaler_empty_array(self):
        """Test z_score_scaler with empty array raises ValueError."""
        with pytest.raises(ValueError, match="Input array must not be empty"):
            z_score_scaler([])

    def test_all_scalers_return_numpy_arrays(self):
        """Test that all scaler functions return numpy arrays."""
        arr = np.array([1, 2, 3])
        assert isinstance(max_scaler(arr), np.ndarray)
        assert isinstance(min_max_scaler(arr), np.ndarray)
        assert isinstance(robust_scaler(arr), np.ndarray)
        assert isinstance(power_transform(arr), np.ndarray)
        assert isinstance(z_score_scaler(arr), np.ndarray)

    def test_scalers_with_3x3_data(self, data_3x3_raw):
        """Test scalers work with 3x3 example data."""
        # Test each column with different scalers
        for col in range(data_3x3_raw.shape[1]):
            column_data = data_3x3_raw[:, col]

            # All scalers should return arrays in [0,1] range
            assert np.all(max_scaler(column_data) >= 0) and np.all(
                max_scaler(column_data) <= 1
            )
            assert np.all(min_max_scaler(column_data) >= 0) and np.all(
                min_max_scaler(column_data) <= 1
            )
            assert np.all(robust_scaler(column_data) >= 0) and np.all(
                robust_scaler(column_data) <= 1
            )
            assert np.all(power_transform(column_data) >= 0) and np.all(
                power_transform(column_data) <= 1
            )
            assert np.all(z_score_scaler(column_data) >= 0) and np.all(
                z_score_scaler(column_data) <= 1
            )

    def test_scalers_with_10x5_data(self, data_10x5_raw):
        """Test scalers work with 10x5 example data."""
        # Test each column with different scalers
        for col in range(data_10x5_raw.shape[1]):
            column_data = data_10x5_raw[:, col]

            # All scalers should return arrays in [0,1] range
            assert np.all(max_scaler(column_data) >= 0) and np.all(
                max_scaler(column_data) <= 1
            )
            assert np.all(min_max_scaler(column_data) >= 0) and np.all(
                min_max_scaler(column_data) <= 1
            )
            assert np.all(robust_scaler(column_data) >= 0) and np.all(
                robust_scaler(column_data) <= 1
            )
            # Skip power_transform for columns with zeros or negative values
            if np.all(column_data > 0):
                assert np.all(power_transform(column_data) >= 0) and np.all(
                    power_transform(column_data) <= 1
                )
            assert np.all(z_score_scaler(column_data) >= 0) and np.all(
                z_score_scaler(column_data) <= 1
            )

    def test_scalers_with_paper_example_data(self, paper_example_X):
        """Test scalers work with paper example data."""
        # Test each column with different scalers
        for col in range(paper_example_X.shape[1]):
            column_data = paper_example_X[:, col]

            # All scalers should return arrays in [0,1] range
            assert np.all(max_scaler(column_data) >= 0) and np.all(
                max_scaler(column_data) <= 1
            )
            assert np.all(min_max_scaler(column_data) >= 0) and np.all(
                min_max_scaler(column_data) <= 1
            )
            assert np.all(robust_scaler(column_data) >= 0) and np.all(
                robust_scaler(column_data) <= 1
            )
            # Skip power_transform for columns with zeros or negative values
            if np.all(column_data > 0):
                assert np.all(power_transform(column_data) >= 0) and np.all(
                    power_transform(column_data) <= 1
                )
            assert np.all(z_score_scaler(column_data) >= 0) and np.all(
                z_score_scaler(column_data) <= 1
            )
