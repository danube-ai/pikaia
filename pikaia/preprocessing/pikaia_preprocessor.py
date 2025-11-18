from typing import Callable, Sequence

import numpy as np

from pikaia.config.logger import logger
from pikaia.schemas.preprocessing import FeatureType


class PikaiaPreprocessor:
    """
    Preprocessor for Pikaia genetic algorithm data.

    This class preprocesses feature data for use with the Pikaia genetic algorithm.
    It applies specified transformation functions to each feature and checks whether
    the transformed data is suitable for the genetic algorithm by checking that all
    values fall within the [0, 1] range.

    The class follows the scikit-learn transformer interface, providing fit(),
    transform(), and fit_transform() methods for compatibility with ML pipelines.

    Attributes:
        num_features (int): The number of features expected in the input data.
    feature_types (Sequence[FeatureType]): A sequence of feature types for each feature.
    feature_transforms (Sequence[Callable[[np.ndarray], np.ndarray] | None]):
            A list where each element is either a transformation function to apply to
            the corresponding feature or None if no transformation is needed.
    """

    def __init__(
        self,
        num_features: int,
        feature_types: Sequence[FeatureType],
        feature_transforms: Sequence[Callable[[np.ndarray], np.ndarray] | None],
    ):
        """
        Initialize the PikaiaPreprocessor.

        Sets up the preprocessor with the specified number of features, their types,
        and the transformation functions to apply to each feature.

        Args:
            num_features (int): The number of features in the dataset. This must match
                the number of columns in the input data arrays passed to fit() and
                transform().
            feature_types (list[FeatureType]): A list of FeatureType enums, one for each
                feature. Each FeatureType indicates whether the feature represents a cost
                (lower values better) or gain (higher values better), though this info is
                stored for potential future use.
            feature_transforms (list[Callable[[NDArray], NDArray] | None]):
                A list of the same length as num_features. Each element is either a
                callable function that takes a 1D numpy array (a feature column) and
                returns a transformed 1D array, or None if no transformation should be
                applied to that feature.

        Raises:
            ValueError: If the lengths of feature_types or feature_transforms do not
                match num_features.
        """
        if len(feature_types) != num_features:
            raise ValueError(
                f"Length of feature_types ({len(feature_types)}) "
                f"must match num_features ({num_features})"
            )

        if len(feature_transforms) != num_features:
            raise ValueError(
                f"Length of feature_transforms ({len(feature_transforms)}) "
                f"must match num_features ({num_features})"
            )

        self.num_features = num_features
        self.feature_types = feature_types
        self.feature_transforms = feature_transforms

    def fit(self, X: np.ndarray) -> "PikaiaPreprocessor":
        """
        Fit the preprocessor to the input data.

        This method validates that the input data X has the correct number of features
        and that all values are numeric compatible (int, float, or boolean) as specified
        during initialization. No actual fitting (e.g., parameter estimation) is performed
        since transformations are predefined.

        Args:
            X (np.ndarray): The input data array with shape (n_samples, n_features).
                Must have exactly num_features columns.

        Returns:
            PikaiaPreprocessor: Returns self to allow method chaining.

        Raises:
            ValueError: If the number of features in X does not match num_features, or
                if any values in X are not numeric (int, float) or boolean, or if X
                contains NaN values.
        """
        if X.shape[1] != self.num_features:
            raise ValueError(
                f"Number of features in X ({X.shape[1]}) "
                f"does not match num_features ({self.num_features})"
            )

        if not (np.issubdtype(X.dtype, np.number) or X.dtype == np.bool_):
            raise ValueError("All values in X must be numeric (int, float) or boolean.")

        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data using the specified transformation functions.

        Applies the transformation function to each feature column if provided. For
        features marked as COST type, the values are inverted using the formula
        max_val + min_val - value to ensure higher values are more desirable for the
        genetic algorithm, regardless of the original data range. After all
        transformations, checks that all values in the transformed data are within the
        [0, 1] range. If not, logs a warning as this may indicate that the data is not
        suitable for the genetic algorithm.

        Args:
            X (np.ndarray): The input data array with shape (n_samples, n_features).
                Must have exactly num_features columns.

        Returns:
            np.ndarray: The transformed data array with the same shape as X, where each
                feature column has been processed according to the specified
                transformation, COST features have been inverted, and boolean values
                have been converted to int.

        Warns:
            Logs a warning if any values in the transformed data fall outside [0, 1].
        """
        if np.any(np.isnan(X)):
            raise ValueError("Input data contains NaN values.")

        X_transformed = X.copy()

        # Convert boolean values to int before transformations
        if X_transformed.dtype == np.bool_:
            X_transformed = X_transformed.astype(int)

        for i in range(self.num_features):
            transform_func = self.feature_transforms[i]
            if transform_func is not None:
                X_transformed[:, i] = transform_func(X_transformed[:, i])

            # Invert COST features
            if self.feature_types[i] == FeatureType.COST:
                min_val = np.min(X_transformed[:, i])
                max_val = np.max(X_transformed[:, i])
                X_transformed[:, i] = max_val + min_val - X_transformed[:, i]

            # Check if all feature values are in [0, 1]
            if not np.all((X_transformed[:, i] >= 0) & (X_transformed[:, i] <= 1)):
                logger.warning(
                    f"Some values in feature column {i} are outside [0,1] range "
                    "after transformation. This may not be valid input for the "
                    "genetic algorithm."
                )

        # Check for NaN values in transformed data
        if np.any(np.isnan(X_transformed)):
            raise ValueError("Transformed data contains NaN values.")

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the preprocessor and transform the data in one step.

        Equivalent to calling fit(X, y).transform(X). This is a convenience method
        for scikit-learn compatibility.

        Args:
            X (np.ndarray): The input data array with shape (n_samples, n_features).
                Must have exactly num_features columns.

        Returns:
            np.ndarray: The transformed data array with the same shape as X.
        """
        return self.fit(X).transform(X)
