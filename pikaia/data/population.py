import numpy as np

from pikaia.config.logger import logger


class PikaiaPopulation:
    """
    Represents a population matrix for genetic algorithms.

    Attributes:
        _matrix (np.ndarray): The population matrix of shape (N, M),
            where N is the number of organisms and M is the number of genes.

    """

    def __init__(self, matrix: np.ndarray, skip_correlation_validation: bool = True):
        """
        Initialize the Population with a matrix.

        Args:
            matrix (np.ndarray):
                A 2D numpy array of shape (N, M) representing the population.
                All values must be between 0 and 1. Higher values are treated as more
                desirable features, whereas lower values are less desirable.

            skip_correlation_validation (bool):
                If True, skips the correlation validation between features.

        Raises:
            ValueError: If the matrix does not meet the validation criteria.

        """
        self._matrix = matrix
        self._skip_correlation_validation = skip_correlation_validation
        self._validate_matrix()

    def _validate_matrix(self):
        """
        Validates the population matrix.

        Checks that all values are between 0 and 1, and that for high linear
        correlation between features (columns).

        Raises:
            ValueError: If validation fails.

        """
        # 1. Check all values are between 0 and 1
        if not np.all((self._matrix >= 0) & (self._matrix <= 1)):
            raise ValueError(
                "All values in the population matrix must be between 0 and 1."
            )

        # 2. Check for linear correlation between features (columns)
        if self._skip_correlation_validation:
            return
        if self._matrix.shape[1] > 1:
            corr_matrix = np.corrcoef(self._matrix, rowvar=False)
            # Ignore diagonal and lower triangle
            upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
            upper_corrs = np.abs(corr_matrix[upper_tri_indices])
            if upper_corrs.size > 0:
                # Find all pairs with correlation > 0.80
                warn_corr_indices = np.where(upper_corrs > 0.80)[0]
                if warn_corr_indices.size > 0:
                    for idx in warn_corr_indices:
                        i = upper_tri_indices[0][idx]
                        j = upper_tri_indices[1][idx]
                        logger.warning(
                            f"Correlated feature pair: (col {i}, col {j}) "
                            f"with correlation {corr_matrix[i, j]:.4f}"
                        )
        else:
            raise ValueError(
                "Population matrix must have at least two features (columns)."
            )

    def __getitem__(self, idx):
        """
        Allows direct indexing into the population.

        Example:
            population[i, j] or population[i].

        """
        return self._matrix[idx]

    @property
    def N(self) -> int:
        """
        Returns the number of organisms (rows) in the population matrix.

        """
        return self._matrix.shape[0]

    @property
    def M(self) -> int:
        """
        Returns the number of genes (columns) in the population matrix.

        """
        return self._matrix.shape[1]

    @property
    def matrix(self) -> np.ndarray:
        """
        Returns the underlying population matrix.

        """
        return self._matrix
