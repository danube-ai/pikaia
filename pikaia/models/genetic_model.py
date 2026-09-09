"""Implement the stateful genetic-model core and its fitness calculations."""

import multiprocessing
from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np

from pikaia.config.logger import logger
from pikaia.data.population import PikaiaPopulation
from pikaia.schemas.strategies import (
    KinRangeConfig,
    StrategyFormulation,
    StrategyFormulationConfig,
    StrategyNormalizations,
)
from pikaia.strategies.base_strategies import (
    GeneStrategy,
    MixStrategy,
    OrgStrategy,
)
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy


def _mean_pairwise_absolute_difference(values: np.ndarray) -> float | None:
    """Return the mean absolute difference over distinct value pairs."""
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return None
    differences = np.abs(values[:, np.newaxis] - values[np.newaxis, :])
    return float(differences[np.triu_indices(values.size, k=1)].mean())


def _compute_standard_similarity(matrix: np.ndarray, divisor: int) -> np.ndarray:
    """Return the similarity scaling selected by ``STANDARD``.

    Args:
        matrix: Rows representing the items whose pairwise similarity is needed.
        divisor: STANDARD normalisation divisor: ``N`` for genes or ``M``
            for organisms, matching the historical Pikaia implementation.

    Returns:
        Square similarity matrix defined as one minus Euclidean distance divided
        by the formulation-specific divisor.

    """
    differences = matrix[:, np.newaxis, :] - matrix[np.newaxis, :, :]
    return 1.0 - np.linalg.norm(differences, axis=2) / divisor


class GeneticModel(ABC):
    """Abstract base class for Genetic AI models.

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
        y: np.ndarray | None = None,
        formulation: StrategyFormulation | str = StrategyFormulation.LEGACY,
    ):
        """Initialise the GeneticModel.

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
            y (np.ndarray | None, optional):
                Optional target values for supervised strategies.
            formulation (StrategyFormulation | str): Mathematical formulation
                selected for the complete simulation. Defaults to ``LEGACY``.

        """
        # Population and strategies
        self._population = population
        self._y = y
        self._formulation = StrategyFormulationConfig.model_validate(
            {"formulation": formulation}
        ).formulation

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

        self._apply_model_formulation()

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
            logger.warning(
                "All organisms have equal initial fitness (range = 0). "
                "No organism can be ranked, so the initial gene and organism "
                "fitness values will be returned unchanged."
            )

        # These fixed-population values are only used by STANDARD strategies.
        # Pydantic validates that any calculated values are finite and non-negative.
        self._strategy_normalizations = StrategyNormalizations(
            gene_mean_pairwise_difference=_mean_pairwise_absolute_difference(
                self._population.matrix.mean(axis=0)
            ),
            harmonic_fitness_mean_pairwise_difference=(
                _mean_pairwise_absolute_difference(self._population.matrix.mean(axis=1))
            ),
        )

        # Compute only the similarities selected for this run. This preserves
        # LEGACY behaviour while avoiding its max-distance preconditions in
        # STANDARD models, whose N/M-scaled similarities remain defined when
        # all compared vectors are identical.
        if self._formulation is StrategyFormulation.STANDARD:
            self._gene_similarity = _compute_standard_similarity(
                self._population.matrix.T, self._population.N
            )
            self._org_similarity = _compute_standard_similarity(
                self._population.matrix, self._population.M
            )
        else:
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
        """Initialise, validate, and normalise mixing coefficients to sum to 1.

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

    def _apply_model_formulation(self) -> None:
        """Validate and apply the formulation selected for this simulation.

        A formulation is model-owned because it defines shared quantities such
        as the similarity matrices and pairwise-difference normalisations. The
        method validates every strategy before mutating any instance, so a
        failed model construction cannot leave a partially reconfigured list.

        Raises:
            ValueError: If one or more selected strategies do not support the
                model's formulation.

        """
        strategies = [*self._gene_strategies, *self._org_strategies]
        incompatible = [
            type(strategy).__name__
            for strategy in strategies
            if self._formulation not in strategy.supported_formulations
        ]
        if incompatible:
            names = ", ".join(incompatible)
            raise ValueError(
                f"formulation {self._formulation.value} is not supported by "
                f"the selected strategies: {names}. "
                "Use LEGACY or choose only strategies that implement the "
                "requested formulation."
            )
        if self._formulation is StrategyFormulation.STANDARD:
            for strategy in strategies:
                if "kin_range" in strategy.options:
                    KinRangeConfig.model_validate(
                        {"kin_range": strategy.options["kin_range"]}
                    )
        for strategy in strategies:
            strategy.set_formulation(self._formulation)

    def _compute_similarity(self, mode: str = "org") -> np.ndarray:
        """Compute the similarity/kinship matrix for organisms or genes.

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
    def _active_gene_similarity(self) -> np.ndarray:
        """Return the similarity matrix selected by the model formulation."""
        return self._gene_similarity

    @property
    def _active_org_similarity(self) -> np.ndarray:
        """Return the similarity matrix selected by the model formulation."""
        return self._org_similarity

    @property
    def formulation(self) -> StrategyFormulation:
        """Return the formulation selected for the complete simulation."""
        return self._formulation

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
        """
        self._validate_d_matrix_configuration()
        M = self._population.M
        D_total = np.zeros((M, M))
        d_total = np.zeros(M)
        has_D = False
        has_d = False

        all_pairs = list(
            zip(self._gene_strategies, self._initial_gene_mixing_coeffs)
        ) + list(zip(self._org_strategies, self._initial_org_mixing_coeffs))

        for strat, coeff in all_pairs:
            kernel_kwargs = (
                {"normalizations": self._strategy_normalizations}
                if strat.requires_normalizations
                else {}
            )
            D_s, d_s = strat.kernel(
                self._population,
                self._active_gene_similarity,
                self._active_org_similarity,
                self._initial_org_fitness_range,
                self._y,
                **kernel_kwargs,
            )
            if D_s is not None:
                D_total += coeff * D_s
                has_D = True
            if d_s is not None:
                d_total += coeff * d_s
                has_d = True

        if not has_D and not has_d:
            strategy_names = [type(s).__name__ for s, _ in all_pairs]
            raise ValueError(
                "use_d_matrix=True requires at least one strategy to implement "
                "kernel(). None of the selected strategies contribute a D-matrix "
                f"or d-vector kernel: {strategy_names}. "
                "Use use_d_matrix=False for strategies without kernel support."
            )

        self._D_matrix = D_total if has_D else None
        self._d_vector = d_total if has_d else None

    def _validate_d_matrix_configuration(self) -> None:
        """Validate formulation-specific D-matrix configuration constraints.

        The historical reduced solver was derived for exactly one altruistic
        gene strategy and one selfish organism strategy, both with fixed unit
        coefficients. The STANDARD dominant strategy is independently exact
        when paired with the formulation-neutral no-op organism strategy.
        Every formulation requires the built-in ``FixedMixStrategy`` because
        the reduced equation does not represent coefficient updates performed
        by adaptive, custom, or overridden mixers.

        Raises:
            ValueError: If a strategy lacks D-matrix support, either mixer is
                not fixed, a STANDARD request is neither historical Alt-Sel
                nor isolated dominant gene, or isolated STANDARD dominant
                starts outside the gene-fitness simplex required by its
                row-constant kernel.

        """
        for strategy in (*self._gene_strategies, *self._org_strategies):
            if not strategy.supports_d_matrix:
                raise ValueError(
                    f"{type(strategy).__name__} in its selected formulation does "
                    "not support use_d_matrix=True. Use use_d_matrix=False."
                )

        has_fixed_mixing = all(
            type(strategy) is FixedMixStrategy
            for strategy in (self._gene_mix_strategy, self._org_mix_strategy)
        )
        if not has_fixed_mixing:
            raise ValueError(
                "use_d_matrix=True requires FixedMixStrategy for both gene and "
                "organism mixing. Use use_d_matrix=False with adaptive or "
                "custom mixing strategies, including overridden subclasses."
            )

        if self._formulation is StrategyFormulation.LEGACY:
            return

        from pikaia.strategies.gs_strategies.altruistic_strategy import (
            AltruisticGeneStrategy,
        )
        from pikaia.strategies.gs_strategies.dominant_strategy import (
            DominantGeneStrategy,
        )
        from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
        from pikaia.strategies.os_strategies.selfish_strategy import (
            SelfishOrgStrategy,
        )

        has_fixed_unit_mixing = self._initial_gene_mixing_coeffs == [
            1.0
        ] and self._initial_org_mixing_coeffs == [1.0]

        is_altsel = (
            len(self._gene_strategies) == 1
            and len(self._org_strategies) == 1
            and isinstance(self._gene_strategies[0], AltruisticGeneStrategy)
            and isinstance(self._org_strategies[0], SelfishOrgStrategy)
            and has_fixed_unit_mixing
        )
        is_isolated_dominant = (
            len(self._gene_strategies) == 1
            and len(self._org_strategies) == 1
            and isinstance(self._gene_strategies[0], DominantGeneStrategy)
            and isinstance(self._org_strategies[0], NoneOrgStrategy)
            and has_fixed_unit_mixing
        )
        if is_isolated_dominant:
            initial_gene_fitness = self._initial_gene_fitness
            is_on_simplex = (
                np.all(np.isfinite(initial_gene_fitness))
                and np.all(initial_gene_fitness >= 0)
                and np.isclose(
                    np.sum(initial_gene_fitness), 1.0, rtol=1e-12, atol=1e-12
                )
            )
            if not is_on_simplex:
                raise ValueError(
                    "STANDARD DominantGeneStrategy with use_d_matrix=True "
                    "requires initial_gene_fitness to contain finite, non-negative "
                    "values that sum to one because its exact row-constant D matrix "
                    "uses the normalized gene-fitness simplex."
                )
        if not (is_altsel or is_isolated_dominant):
            raise ValueError(
                "STANDARD use_d_matrix=True is available only for an "
                "unmixed DominantGeneStrategy + NoneOrgStrategy model or the "
                "unmixed AltruisticGeneStrategy + SelfishOrgStrategy (Alt-Sel) "
                "combination. Use use_d_matrix=False for other STANDARD "
                "configurations."
            )

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
        """Return the gene similarity matrix selected by the formulation."""
        return self._active_gene_similarity

    @property
    def org_similarity(self) -> np.ndarray:
        """Return the organism similarity matrix selected by the formulation."""
        return self._active_org_similarity

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
        """Return the iteration at which the simulation converged (ESE).

        Return ``-1`` when the simulation has not converged.
        """
        return self._ESE_iter

    @abstractmethod
    def fit(self) -> None:
        """Fits the genetic model to the population data.

        This method should be implemented by subclasses to perform the fitting process.
        """
        pass  # pragma: no cover

    @abstractmethod
    def predict(self, population: PikaiaPopulation) -> np.ndarray:
        """Predicts the organism fitness for a new population using the fitted model.

        Args:
            population (PikaiaPopulation): The new population for which to predict
                organism fitness.

        Returns:
            np.ndarray: A vector of predicted organism fitness values.

        """
        pass  # pragma: no cover
