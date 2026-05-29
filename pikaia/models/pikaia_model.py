import multiprocessing

import numpy as np

from pikaia.config.logger import logger
from pikaia.data.population import PikaiaPopulation
from pikaia.models.genetic_model import GeneticModel
from pikaia.strategies.base_strategies import (
    GeneStrategy,
    OrgStrategy,
    StrategyContext,
)
from pikaia.strategies.mix_strategies.self_consistent_strategy import (
    SelfConsistentMixStrategy,
)


class PikaiaModel(GeneticModel):
    """
    Central organizing class for the Genetic AI model.

    This class orchestrates the evolutionary simulation. It takes a population,
    a set of gene and organism strategies, and runs a simulation over a specified
    number of iterations. It tracks the history of gene and organism fitness,
    as well as the mixing coefficients for the strategies.

    The model is fitted using the `fit` method, which iteratively updates the
    fitness values based on the provided strategies.
    """

    def __init__(self, *args, use_d_matrix: bool = False, **kwargs):
        """
        Initialises the PikaiaModel.

        Accepts all arguments of :class:`GeneticModel` plus:

        Args:
            use_d_matrix (bool):
                When ``True``, the D-matrix fast path is used instead of the
                standard per-organism loop. Precomputes the ``(M, M)`` D matrix
                and ``(M,)`` d-vector once before the iteration loop, reducing
                per-step cost from ``O(N·M²)`` to ``O(M²)``.  All active
                strategies must have a registered D-matrix kernel; a
                ``ValueError`` is raised at fit time if any do not.
                Defaults to ``False``.
        """
        super().__init__(*args, **kwargs)
        self._use_d_matrix = use_d_matrix

        # D-matrix state — populated by _compute_d_matrix() inside fit().
        # Initialized as empty lists so type-checkers accept slicing.
        self._D_matrix: np.ndarray | None = None
        self._d_vector: np.ndarray | None = None
        self._D_per_strategy: list = []
        self._d_per_strategy: list = []

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
        if self._use_d_matrix:
            if self._max_iter is None:
                raise ValueError(
                    "use_d_matrix=True requires max_iter to be set. "
                    "There is no general closed-form fixed point for an arbitrary D matrix. "
                    "Use max_iter with a large value (e.g. max_iter=500) and optionally "
                    "epsilon for convergence detection, or use use_d_matrix=False for the "
                    "analytical Dominant+Balanced fixed point."
                )
            logger.info("D-matrix path selected. Precomputing D matrix...")
            self._compute_d_matrix()
            logger.info(f"Running D-matrix simulation for up to {self._max_iter} iterations.")
            self._run_d_matrix_iterations()
        elif self._max_iter is None:
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

    # ------------------------------------------------------------------
    # D-matrix fast paths
    # ------------------------------------------------------------------

    def _run_d_matrix_iterations(self, *, epsilon_override: float | None = None) -> None:
        """Run the D-matrix fast iteration loop.

        Uses the precomputed ``self._D_matrix`` (bilinear term) and
        ``self._d_vector`` (linear term from balanced org) to execute each
        step in ``O(M²)`` instead of ``O(N·M²)``.

        Args:
            epsilon_override: If provided, overrides ``self._epsilon`` as the
                convergence threshold.  Used internally by
                ``_run_d_matrix_fix_point()`` to apply a tight tolerance.
        """
        import time

        D = self._D_matrix   # (M, M) or None
        d = self._d_vector   # (M,)  or None
        epsilon = epsilon_override if epsilon_override is not None else self._epsilon

        is_sc_gene = isinstance(self._gene_mix_strategy, SelfConsistentMixStrategy)
        is_sc_org = isinstance(self._org_mix_strategy, SelfConsistentMixStrategy)
        K_g = len(self._gene_strategies)

        gene_mix_coeffs = np.array(self._initial_gene_mixing_coeffs)
        org_mix_coeffs = np.array(self._initial_org_mixing_coeffs)

        logger.info(
            f"Starting D-matrix iteration for up to {self._max_iter} iterations."
        )
        total_start = time.perf_counter()

        for i in range(1, (self._max_iter or 1) + 1):
            iter_start = time.perf_counter()
            logger.debug(f"D-matrix iteration {i}...")
            gamma = self._gene_fitness_hist[i - 1, :]

            # When SelfConsistentMixStrategy is active, the mixing coefficients
            # evolve and D_total must be recomputed each step.
            if is_sc_gene or is_sc_org:
                all_coeffs = np.concatenate([gene_mix_coeffs, org_mix_coeffs])
                D_active, d_active = self._recompute_combined_d(all_coeffs)
            else:
                D_active, d_active = D, d

            # Fast replicator step
            bilinear = gamma * (D_active @ gamma) if D_active is not None else 0.0
            linear = d_active if d_active is not None else 0.0
            step = linear + bilinear
            gamma_new = gamma * (1.0 + step)

            if np.any(gamma_new <= 0):
                raise ValueError(
                    f"D-matrix step produced non-positive gene fitness at iteration "
                    f"{i}. Population structure may be incompatible with the "
                    "D-matrix path. Check for gene columns with mean expression > 0.5 "
                    "when using BalancedOrgStrategy."
                )
            gamma_new /= gamma_new.sum()

            self._gene_fitness_hist[i, :] = gamma_new
            self._org_fitness_hist[i, :] = self._population.matrix @ gamma_new

            # Update mixing coefficients for SelfConsistent
            if is_sc_gene:
                D_gene = self._D_per_strategy[:K_g]
                d_gene = self._d_per_strategy[:K_g]
                gene_mix_coeffs = SelfConsistentMixStrategy.update_coeffs_d_matrix(
                    D_gene, d_gene, gamma_new, gene_mix_coeffs
                )
            if is_sc_org:
                D_org = self._D_per_strategy[K_g:]
                d_org = self._d_per_strategy[K_g:]
                org_mix_coeffs = SelfConsistentMixStrategy.update_coeffs_d_matrix(
                    D_org, d_org, gamma_new, org_mix_coeffs
                )

            self._gene_mixing_coeffs_hist[i, :] = gene_mix_coeffs
            self._org_mixing_coeffs_hist[i, :] = org_mix_coeffs

            delta_norm = np.linalg.norm(gamma_new - gamma)
            iter_elapsed = time.perf_counter() - iter_start
            logger.debug(
                f"D-matrix iteration {i} done. Δ={delta_norm:.6g}. "
                f"Time: {iter_elapsed:.4f}s."
            )
            if epsilon is not None and delta_norm < epsilon:
                self._ESE_iter = i
                total_elapsed = time.perf_counter() - total_start
                logger.info(
                    f"D-matrix reached ESE after {i} iterations. "
                    f"Δ={delta_norm}. Total: {total_elapsed:.4f}s."
                )
                break
        else:
            total_elapsed = time.perf_counter() - total_start
            logger.info(
                f"D-matrix completed {self._max_iter} iterations without ESE. "
                f"Total: {total_elapsed:.4f}s."
            )

    def _recompute_combined_d(
        self, all_coeffs: np.ndarray
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Recompute the combined D matrix and d-vector from current mixing coefficients.

        Used only by ``_run_d_matrix_iterations()`` when ``SelfConsistentMixStrategy``
        is active (coefficients evolve each step).

        Args:
            all_coeffs: Combined mixing coefficients for gene strategies followed by
                org strategies, shape ``(K_g + K_o,)``.

        Returns:
            ``(D_total, d_total)`` with current coefficients applied.
        """
        M = self._population.M
        D_total = np.zeros((M, M))
        d_total = np.zeros(M)
        has_D = False
        has_d = False
        for idx, (D_s, d_s) in enumerate(
            zip(self._D_per_strategy, self._d_per_strategy)
        ):
            c = all_coeffs[idx]
            if D_s is not None:
                D_total += c * D_s
                has_D = True
            if d_s is not None:
                d_total += c * d_s
                has_d = True
        return (D_total if has_D else None), (d_total if has_d else None)

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
