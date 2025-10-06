import multiprocessing
from typing import Iterable

import numpy as np

from pikaia.config.logger import logger
from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import (
    GeneStrategy,
    MixStrategy,
    OrgStrategy,
    StrategyContext,
)
from pikaia.strategies.mix_strategies.fixed_strategy import FixedMixStrategy


class PikaiaModel:
    """
    Central organizing class for the Genetic AI model.

    This class orchestrates the evolutionary simulation. It takes a population,
    a set of gene and organism strategies, and runs a simulation over a specified
    number of iterations. It tracks the history of gene and organism fitness,
    as well as the mixing coefficients for the strategies.

    The model is fitted using the `fit` method, which iteratively updates the
    fitness values based on the provided strategies.
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
        Initializes the PikaiaModel.

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
                The maximum number of iterations for the simulation. Defaults to 1.
                When set to 1, strategies and initial gene fitness will be ignored and
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

        self._initial_gene_mixing_coeffs = (
            gene_mixing_coeffs
            if gene_mixing_coeffs is not None
            else (
                [1.0 / len(self._gene_strategies) for _ in self._gene_strategies]
                if self._gene_strategies
                else []
            )
        )
        self._initial_org_mixing_coeffs = (
            org_mixing_coeffs
            if org_mixing_coeffs is not None
            else (
                [1.0 / len(self._org_strategies) for _ in self._org_strategies]
                if self._org_strategies
                else []
            )
        )
        self._max_iter = max_iter

        if epsilon is not None and max_iter is None:
            logger.warning("epsilon is ignored when max_iter is None (default)")
        self._epsilon = epsilon

        # Initial fitness values
        if initial_gene_fitness is not None:
            if max_iter is None:
                logger.warning(
                    "initial_gene_fitness has no effect when max_iter is None (default); "
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
                for genes, where N is the number of organisms and M is the number of genes.

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
        """The iteration number at which the simulation converged (ESE). -1 if not converged."""
        return self._ESE_iter

    def fit(self) -> None:
        """
        Fits the genetic model to the population data by running the simulation.

        This method iteratively updates the gene and organism fitness values based on the
        provided strategies. The simulation runs for a maximum number of iterations as
        defined by `max_iter`. If an `epsilon` value is provided, the simulation will
        stop early if the change in gene fitness between iterations falls below this
        threshold, indicating convergence.
        """
        import time

        start_time = time.perf_counter()
        if self._max_iter is None:
            logger.info("No max iterations set, solving for optimal solution directly.")
            self._run_fix_point()
        else:
            logger.info(f"Running simulation for up to {self._max_iter} iterations.")
            self._run_iterations()
        total_time = time.perf_counter() - start_time
        logger.info(f"Total fit process time: {total_time:.4f} seconds.")

    def _run_fix_point(self):
        """
        Solves for the optimal gene fitness distribution directly, assuming a dominant
        gene strategy and a balanced organism strategy.
        """
        import time

        start_time = time.perf_counter()
        gene_means = np.mean(self._population.matrix, axis=0)  # (M,)
        denom = gene_means + 0.5  # (M,)
        sum_inv_denom = np.sum(1 / denom)  # scalar
        gene_fitness = 1 / (denom * sum_inv_denom)  # (M,)

        org_fitness = np.dot(self._population.matrix, gene_fitness)  # (N,)

        self._gene_fitness_hist[1, :] = gene_fitness
        self._org_fitness_hist[1, :] = org_fitness
        elapsed = time.perf_counter() - start_time
        logger.debug(f"_run_fix_point completed in {elapsed:.4f} seconds.")

    def _run_iterations(self):
        """
        Runs the evolutionary simulation for multiple iterations.

        This method iterates over the specified number of iterations, updating
        the gene and organism fitness values at each step. It checks for convergence
        based on the epsilon threshold if provided, and stops early if the simulation
        reaches an Evolutionarily Stable Equilibrium (ESE).
        """
        import time

        logger.info(
            f"Starting evolutionary simulation for up to {self._max_iter} iterations."
        )
        total_start = time.perf_counter()
        for i in range(1, (self._max_iter or 1) + 1):
            iter_start = time.perf_counter()
            logger.debug(f"Starting iteration {i}...")
            # 1. Run iteration
            (
                self._gene_fitness_hist[i, :],
                self._org_fitness_hist[i, :],
                self._gene_mixing_coeffs_hist[i, :],
                self._org_mixing_coeffs_hist[i, :],
            ) = self._run_iteration(i)

            # 2. Check for convergence
            delta = np.linalg.norm(
                self._gene_fitness_hist[i, :] - self._gene_fitness_hist[i - 1, :]
            )
            iter_elapsed = time.perf_counter() - iter_start
            logger.debug(
                f"Iteration {i} complete. Δgene_fitness = {delta:.6g}. "
                f"Iteration time: {iter_elapsed:.4f} seconds."
            )
            if self._epsilon is not None and delta < self._epsilon:
                self._ESE_iter = i
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    f"Reached ESE after {self._ESE_iter} iterations. "
                    f"Final delta = {delta}. "
                    f"Total time: {total_elapsed:.4f} seconds."
                )
                break
        else:
            total_elapsed = time.perf_counter() - total_start
            logger.info(
                f"Completed all {self._max_iter} iterations without reaching ESE. "
                f"Total time: {total_elapsed:.4f} seconds."
            )

    def _run_iteration(
        self, iter_num: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs a single evolutionary step of the simulation.

        This method calculates the change in gene fitness based on the defined gene and
        organism strategies. It then mixes these strategies and applies the updates to
        compute the new gene and organism fitness values for the current iteration.

        Args:
            iter_num (int): The current iteration number.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
                - The new gene fitness vector.
                - The new organism fitness vector.
                - The new gene mixing coefficients.
                - The new organism mixing coefficients.
        """
        current_org_fitness = self._org_fitness_hist[iter_num - 1, :]
        current_gene_fitness = self._gene_fitness_hist[iter_num - 1, :]

        delta_g, delta_o = self._calculate_deltas(
            current_org_fitness, current_gene_fitness
        )

        # 2. Mix evolutionary strategies
        mixed_delta_g, gene_mixing_coeffs = self._gene_mix_strategy(
            delta_g, self._gene_mixing_coeffs_hist[iter_num - 1, :]
        )
        mixed_delta_o, org_mixing_coeffs = self._org_mix_strategy(
            delta_o, self._org_mixing_coeffs_hist[iter_num - 1, :]
        )

        # 3. Apply delta in form of the central replicator equations
        gene_fitness = current_gene_fitness * (
            1 + np.sum(mixed_delta_g + mixed_delta_o, axis=0)
        )
        gene_fitness /= np.sum(gene_fitness)

        org_fitness = np.dot(self._population.matrix, gene_fitness)

        return (
            gene_fitness,
            org_fitness,
            gene_mixing_coeffs,
            org_mixing_coeffs,
        )

    def _calculate_deltas(
        self, current_org_fitness: np.ndarray, current_gene_fitness: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates the delta contributions for gene and organism fitness updates.

        This method can run in parallel or sequentially based on the `n_jobs` setting.

        Args:
            current_org_fitness (np.ndarray): The current organism fitness vector.
            current_gene_fitness (np.ndarray): The current gene fitness vector.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the delta_g and delta_o matrices.
        """
        delta_g = np.zeros(
            [self._population.N, self._population.M, len(self._gene_strategies)]
        )
        delta_o = np.zeros(
            [self._population.N, self._population.M, len(self._org_strategies)]
        )

        context_args = {
            "population": self._population,
            "org_fitness": current_org_fitness,
            "gene_fitness": current_gene_fitness,
            "initial_org_fitness_range": self._initial_org_fitness_range,
            "org_similarity": self._org_similarity,
            "gene_similarity": self._gene_similarity,
        }

        if self._n_jobs > 1:
            with multiprocessing.Pool(processes=self._n_jobs) as pool:
                # Organism strategies
                org_args_list = [
                    (strat, org_id, None, context_args)
                    for org_id in range(self._population.N)
                    for strat in self._org_strategies
                ]
                org_results = pool.starmap(
                    PikaiaModel._compute_single_delta, org_args_list
                )

                res_idx = 0
                for org_id in range(self._population.N):
                    for strat_idx in range(len(self._org_strategies)):
                        delta_o[org_id, :, strat_idx] = org_results[res_idx]
                        res_idx += 1

                # Gene strategies
                gene_args_list = [
                    (strat, org_id, gene_id, context_args)
                    for org_id in range(self._population.N)
                    for gene_id in range(self._population.M)
                    for strat in self._gene_strategies
                ]
                gene_results = pool.starmap(
                    PikaiaModel._compute_single_delta, gene_args_list
                )

                res_idx = 0
                for org_id in range(self._population.N):
                    for gene_id in range(self._population.M):
                        for strat_idx in range(len(self._gene_strategies)):
                            delta_g[org_id, gene_id, strat_idx] = gene_results[res_idx]
                            res_idx += 1
        else:
            # Naive loop-based calculation
            for org_id in range(self._population.N):
                for i, strat in enumerate(self._org_strategies):
                    delta_o[org_id, :, i] = PikaiaModel._compute_single_delta(
                        strat, org_id, None, context_args
                    )
                for gene_id in range(self._population.M):
                    for i, strat in enumerate(self._gene_strategies):
                        delta_g[org_id, gene_id, i] = PikaiaModel._compute_single_delta(
                            strat, org_id, gene_id, context_args
                        )
        return delta_g, delta_o

    @staticmethod
    def _compute_single_delta(
        strat: GeneStrategy | OrgStrategy,
        org_id: int,
        gene_id: int | None,
        context_args: dict,
    ) -> np.ndarray | float:
        """
        Computes a single delta contribution for either organism or gene strategies.

        Args:
            strat (GeneStrategy | OrgStrategy): The strategy to apply.
            org_id (int): The organism ID.
            gene_id (int | None): The gene ID, required for "gene" type.
            context_args (dict): A dictionary of common arguments for StrategyContext.

        Returns:
            np.ndarray | float: The computed delta value(s).
        """
        return strat(
            StrategyContext(
                org_id=org_id,
                gene_id=gene_id,
                **context_args,
            )
        )

    def predict(self, population: PikaiaPopulation) -> np.ndarray:
        """
        Predicts the organism fitness for a new population using the fitted model.

        This method computes the organism fitness values for a given population
        based on the final gene fitness distribution obtained from the last iteration
        of the fitted model.

        Args:
            population (PikaiaPopulation): The new population for which to predict
                organism fitness.
        Returns:
            np.ndarray: A vector of predicted organism fitness values of shape (N,),
                where N is the number of organisms in the provided population.
        """
        if population.M != self._population.M:
            raise ValueError(
                "The number of genes in the new population must match the fitted model."
                f" Got {population.M}, expected {self._population.M}."
            )

        return np.dot(population.matrix, self._gene_fitness_hist[1, :])
