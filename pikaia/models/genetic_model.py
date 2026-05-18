import multiprocessing
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from pikaia.config.logger import logger
from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import (
    GeneStrategy,
    MixStrategy,
    OrgStrategy,
)
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy


class GeneticModel(ABC):
    """
    Abstract base class for Genetic AI models.

    This class handles input validation, preprocessing, and provides the interface
    for fitting and predicting with genetic models.
    """

    def __init__(
        self,
        population: PikaiaPopulation,
        gene_strategies: Iterable[GeneStrategy] | None = None,
        org_strategies: Iterable[OrgStrategy] | None = None,
        gene_mix_strategy: MixStrategy | None = None,
        org_mix_strategy: MixStrategy | None = None,
        gene_mixing_coeffs: list[float] | None = None,
        org_mixing_coeffs: list[float] | None = None,
        initial_gene_fitness: Iterable[float] | None = None,
        max_iter: int | None = None,
        epsilon: float | None = None,
        n_jobs: int = 1,
    ):
        """
        Initializes the GeneticModel.

        Args:
            population (Population):
                The population of organisms and genes.
            gene_strategies (Iterable[GeneStrategy] | None, optional):
                A list of strategies to update gene fitness.
            org_strategies (Iterable[OrgStrategy] | None, optional):
                A list of strategies to update organism fitness.
            gene_mix_strategy (MixStrategy | None, optional):
                The strategy for mixing gene strategies.
                Defaults to FixedMixStrategy.
            org_mix_strategy (MixStrategy | None, optional):
                The strategy for mixing organism strategies.
                Defaults to FixedMixStrategy.
            gene_mixing_coeffs (list[float] | None, optional):
                Initial coefficients for mixing gene strategies.
                Defaults to a uniform distribution.
            org_mixing_coeffs (list[float] | None, optional):
                Initial coefficients for mixing organism strategies.
                Defaults to a uniform distribution.
            initial_gene_fitness (Iterable[float] | None, optional):
                Initial fitness values for each gene.
                Defaults to a uniform distribution.
            max_iter (int, optional):
                The maximum number of iterations for the simulation. Defaults to None.
                When not set, strategies and initial gene fitness will be ignored and
                the optimal solution will be computed directly.
            epsilon (float, optional):
                The convergence threshold. If the L2 norm of the
                change in gene fitness between two consecutive iterations is less than
                this value, the simulation is considered to have reached an
                Evolutionarily Stable Equilibrium (ESE) and stops. If None, the
                simulation runs for `max_iter` iterations. Defaults to None.
            n_jobs (int):
                The number of parallel processes to use for strategy evaluations.
                Defaults to 1. If -1, all available CPUs are used.
        """
        # Population and strategies
        self._population = population

        if gene_strategies is None:
            self._gene_strategies = []
            if max_iter is not None:
                raise ValueError("gene_strategies must be provided if max_iter is set")
        else:
            if max_iter is None:
                logger.warning(
                    "gene_strategies is ignored when max_iter is None (default)"
                )
            self._gene_strategies = list(gene_strategies)

        if org_strategies is None:
            self._org_strategies = []
            if max_iter is not None:
                raise ValueError("org_strategies must be provided if max_iter is set")
        else:
            if max_iter is None:
                logger.warning(
                    "org_strategies is ignored when max_iter is None (default)"
                )
            self._org_strategies = list(org_strategies)

        if gene_mix_strategy is not None and max_iter is None:
            logger.warning(
                "gene_mix_strategy is ignored when max_iter is None (default)"
            )

        self._gene_mix_strategy = gene_mix_strategy or FixedMixStrategy()

        if org_mix_strategy is not None and max_iter is None:
            logger.warning(
                "org_mix_strategy is ignored when max_iter is None (default)"
            )
        self._org_mix_strategy = org_mix_strategy or FixedMixStrategy()

        # Initialize and validate mixing coefficients
        self._initial_gene_mixing_coeffs = self._init_and_validate_mixing_coeffs(
            gene_mixing_coeffs, self._gene_strategies, "gene_mixing_coeffs"
        )
        self._initial_org_mixing_coeffs = self._init_and_validate_mixing_coeffs(
            org_mixing_coeffs, self._org_strategies, "org_mixing_coeffs"
        )

        self._max_iter = max_iter

        if epsilon is not None and max_iter is None:
            logger.warning("epsilon is ignored when max_iter is None (default)")
        self._epsilon = epsilon

        # Initial fitness values
        if initial_gene_fitness is not None:
            if max_iter is None:
                logger.warning(
                    "initial_gene_fitness has no effect when max_iter is None; "
                    "the algorithm will reach convergence independent of initial "
                    "gene preferences."
                )
            elif max_iter > 5:
                logger.info(
                    "With max_iter > 5, initial_gene_fitness will likely have little "
                    "effect; gene preferences will vanish and the algorithm will "
                    "approximate convergence."
                )
        self._initial_gene_fitness = (
            np.array(initial_gene_fitness)
            if initial_gene_fitness is not None
            else np.ones(self._population.M) / self._population.M
        )
        self._initial_org_fitness = np.dot(
            self._population.matrix, self._initial_gene_fitness
        )

        self._initial_org_fitness_range = np.max(self._initial_org_fitness) - np.min(
            self._initial_org_fitness
        )
        if self._initial_org_fitness_range == 0:
            raise ValueError("All organism fitness values are 0.")

        # Similarity matrices
        self._gene_similarity = self._compute_similarity(mode="gene")
        self._org_similarity = self._compute_similarity(mode="org")

        # History containers
        self._gene_fitness_hist = np.zeros(
            [(self._max_iter or 1) + 1, self._population.M]
        )
        self._gene_fitness_hist[0, :] = self._initial_gene_fitness
        self._org_fitness_hist = np.zeros(
            [(self._max_iter or 1) + 1, self._population.N]
        )
        self._org_fitness_hist[0, :] = self._initial_org_fitness

        self._gene_mixing_coeffs_hist = np.zeros(
            [(self._max_iter or 1) + 1, len(self._gene_strategies)]
        )
        self._gene_mixing_coeffs_hist[0, :] = self._initial_gene_mixing_coeffs
        self._org_mixing_coeffs_hist = np.zeros(
            [(self._max_iter or 1) + 1, len(self._org_strategies)]
        )
        self._org_mixing_coeffs_hist[0, :] = self._initial_org_mixing_coeffs

        self._ESE_iter = -1
        if n_jobs == -1:
            self._n_jobs = multiprocessing.cpu_count()
        else:
            self._n_jobs = n_jobs

    @staticmethod
    def _init_and_validate_mixing_coeffs(
        coeffs: list[float] | None,
        strategies: list,
        param_name: str,
    ) -> list[float]:
        """
        Initializes, validates, and normalizes mixing coefficients to sum to 1.

        Args:
            coeffs (list[float] | None): User-provided coefficients or None.
            strategies (list): List of strategies.
            param_name (str): Parameter name for error messages.

        Returns:
            list[float]: Validated and normalized coefficients that sum to 1.

        Raises:
            ValueError: If the length doesn't match or all coefficients
                are zero/negative.
        """
        if not strategies:
            return []

        # Use provided coefficients or create uniform distribution
        if coeffs is None:
            return [1.0 / len(strategies)] * len(strategies)

        # Validate length
        if len(coeffs) != len(strategies):
            raise ValueError(
                f"{param_name} must have length {len(strategies)}, got {len(coeffs)}"
            )

        coeffs_array = np.array(coeffs)

        # Validate non-negativity
        if np.any(coeffs_array < 0):
            raise ValueError(
                f"{param_name} contains negative values. All coefficients "
                "must be non-negative."
            )

        # Validate non-zero sum
        total = np.sum(coeffs_array)
        if total == 0:
            raise ValueError(
                f"{param_name} sums to zero. At least one coefficient must be positive."
            )

        # Normalize to sum to 1
        normalized = coeffs_array / total

        # Log warning if normalization was needed
        if not np.isclose(total, 1.0, rtol=1e-9):
            logger.warning(
                f"{param_name} did not sum to 1 (sum={total:.6f}). "
                f"Normalized coefficients to sum to 1."
            )

        return normalized.tolist()

    def _compute_similarity(self, mode: str = "org") -> np.ndarray:
        """
        Computes the similarity/kinship matrix for organisms or genes.

        The similarity is defined as 1 minus the normalized Euclidean distance
        between the vectors representing each organism or gene.

        Args:
            mode (str): Specifies whether to compute similarity for 'org' (organisms)
                or 'gene' (genes). Defaults to "org".

        Returns:
            np.ndarray: A square matrix where the element (i, j) is the similarity
                between item i and item j. The shape is (N, N) for organisms or (M, M)
                for genes, where N is the number of organisms and M is the number
                of genes.

        Raises:
            ValueError: If an unknown mode is provided or if all items are identical.
        """
        if mode == "gene":
            matrix = self._population.matrix.T
        elif mode == "org":
            matrix = self._population.matrix
        else:
            raise ValueError(f"Unknown mode '{mode}'. Use 'org' or 'gene'.")

        diff = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
        distances = np.linalg.norm(diff, axis=2)
        max_dist = np.max(distances)

        if max_dist == 0:
            raise ValueError(f"All {mode} items are identical")

        return 1 - distances / max_dist

    @property
    def population(self) -> PikaiaPopulation:
        """The population used in the model."""
        return self._population

    @property
    def gene_strategies(self) -> Iterable[GeneStrategy]:
        """The gene strategies used in the model."""
        return self._gene_strategies

    @property
    def org_strategies(self) -> Iterable[OrgStrategy]:
        """The organism strategies used in the model."""
        return self._org_strategies

    def _compute_d_matrix(self) -> None:
        """Precompute the combined D matrix and d-vector for the D-matrix path.

        Calls ``strategy.kernel(...)`` on each active strategy and accumulates
        the weighted contributions.

        Populates:

        - ``self._D_matrix``: combined ``(M, M)`` bilinear matrix, or ``None``.
        - ``self._d_vector``: combined ``(M,)`` linear vector, or ``None``.
        - ``self._D_per_strategy``: per-strategy D matrices (unweighted).
        - ``self._d_per_strategy``: per-strategy d-vectors (unweighted).

        The per-strategy lists have length
        ``len(gene_strategies) + len(org_strategies)`` with gene entries first.
        """
        M = self._population.M
        D_total = np.zeros((M, M))
        d_total = np.zeros(M)
        has_D = False
        has_d = False

        D_per: list[np.ndarray | None] = []
        d_per: list[np.ndarray | None] = []

        all_pairs = list(
            zip(self._gene_strategies, self._initial_gene_mixing_coeffs)
        ) + list(zip(self._org_strategies, self._initial_org_mixing_coeffs))

        for strat, coeff in all_pairs:
            D_s, d_s = strat.kernel(
                self._population,
                self._gene_similarity,
                self._org_similarity,
                self._initial_org_fitness_range,
            )
            D_per.append(D_s)
            d_per.append(d_s)
            if D_s is not None:
                D_total += coeff * D_s
                has_D = True
            if d_s is not None:
                d_total += coeff * d_s
                has_d = True

        if not has_D and not has_d:
            strategy_names = [
                type(s).__name__
                for s, _ in all_pairs
            ]
            raise ValueError(
                "use_d_matrix=True requires at least one strategy to implement "
                "kernel(). None of the selected strategies contribute a D-matrix "
                f"or d-vector kernel: {strategy_names}. "
                "Use use_d_matrix=False for strategies without kernel support."
            )

        self._D_matrix = D_total if has_D else None
        self._d_vector = d_total if has_d else None
        self._D_per_strategy = D_per
        self._d_per_strategy = d_per

    @property
    def gene_mixing(self) -> Iterable[float]:
        """The initial gene mixing proportions."""
        return self._initial_gene_mixing_coeffs

    @property
    def org_mixing(self) -> Iterable[float]:
        """The initial organism mixing proportions."""
        return self._initial_org_mixing_coeffs

    @property
    def initial_gene_fitness(self) -> np.ndarray:
        """The initial gene fitness values."""
        return self._initial_gene_fitness

    @property
    def initial_org_fitness(self) -> np.ndarray:
        """The initial organism fitness values, derived from initial gene fitness."""
        return self._initial_org_fitness

    @property
    def initial_org_fitness_range(self) -> float:
        """The initial range of organism fitness values."""
        return self._initial_org_fitness_range

    @property
    def gene_similarity(self) -> np.ndarray:
        """The gene similarity matrix."""
        return self._gene_similarity

    @property
    def org_similarity(self) -> np.ndarray:
        """The organism similarity matrix."""
        return self._org_similarity

    @property
    def gene_fitness_history(self) -> np.ndarray:
        """The history of gene fitness values over iterations."""
        return self._gene_fitness_hist

    @property
    def organism_fitness_history(self) -> np.ndarray:
        """The history of organism fitness values over iterations."""
        return self._org_fitness_hist

    @property
    def gene_mixing_history(self) -> np.ndarray:
        """The history of gene mixing proportions over iterations."""
        return self._gene_mixing_coeffs_hist

    @property
    def organism_mixing_history(self) -> np.ndarray:
        """The history of organism mixing proportions over iterations."""
        return self._org_mixing_coeffs_hist

    @property
    def max_iter(self) -> int | None:
        """The maximum number of iterations for the model."""
        return self._max_iter

    @property
    def ESE_iter(self) -> int:
        """The iteration number at which the simulation converged (ESE).
        -1 if not converged."""
        return self._ESE_iter

    @abstractmethod
    def fit(self) -> None:
        """
        Fits the genetic model to the population data.

        This method should be implemented by subclasses to perform the fitting process.
        """
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, population: PikaiaPopulation) -> np.ndarray:
        """
        Predicts the organism fitness for a new population using the fitted model.

        Args:
            population (PikaiaPopulation): The new population for which to predict
                organism fitness.

        Returns:
            np.ndarray: A vector of predicted organism fitness values.
        """
        pass  # pragma: no cover
