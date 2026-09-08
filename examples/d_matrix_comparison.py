#!/usr/bin/env python3
"""Compare every supported D-matrix configuration with iterative execution.

The D-matrix path is an exact reduced execution path, not a fallback for every
strategy. This example runs only configurations accepted by ``PikaiaModel``:

1. Each D-capable ``ORIGINAL`` gene strategy, isolated with
   ``NoneOrgStrategy``.
2. The ``MATH_PAPER`` dominant-gene strategy, isolated with
   ``NoneOrgStrategy``.
3. The ``MATH_PAPER`` altruistic-gene plus selfish-organism (Alt-Sel)
   configuration.

For each configuration, independently constructed iterative and D-matrix models
start from the same population and gene-fitness vector. The script reports the
largest absolute difference after 1, 50, and 100 iterations and the runtime of
both 100-iteration runs.
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from statistics import median
from time import perf_counter

import numpy as np

from pikaia.config.logger import logger
from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import StrategyFormulation
from pikaia.strategies.base_strategies import GeneStrategy, OrgStrategy
from pikaia.strategies.gs_strategies.altruistic_strategy import (
    AltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import (
    KinAltruisticGeneStrategy,
)
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.gs_strategies.sell_easy_strategy import SellEasyGeneStrategy
from pikaia.strategies.gs_strategies.sell_hard_strategy import SellHardGeneStrategy
from pikaia.strategies.gs_strategies.sell_uniform_strategy import (
    SellUniformGeneStrategy,
)
from pikaia.strategies.gs_strategies.variance_strategy import VarianceGeneStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy

logger.setLevel(logging.WARNING)

TIMING_REPEATS = 7


@dataclass(frozen=True)
class DMatrixConfiguration:
    """Describe one supported end-to-end D-matrix configuration."""

    name: str
    formulation: StrategyFormulation
    population_factory: Callable[[], PikaiaPopulation]
    gene_strategy_factory: Callable[[], GeneStrategy]
    org_strategy_factory: Callable[[], OrgStrategy]
    initial_gene_fitness: tuple[float, ...]


def original_population() -> PikaiaPopulation:
    """Return the fixed population used for original-formulation comparisons."""
    return PikaiaPopulation(
        np.array(
            [
                [0.1, 0.6, 0.9],
                [0.8, 0.2, 0.4],
                [0.3, 0.9, 0.1],
                [0.7, 0.4, 0.6],
            ]
        )
    )


def math_paper_population() -> PikaiaPopulation:
    """Return a population with non-zero math-paper normalization factors."""
    return PikaiaPopulation(
        np.array(
            [
                [0.59587318, 0.33885070, 0.17868147],
                [0.16156934, 0.71928917, 0.78892890],
                [0.57464504, 0.66064759, 0.53053749],
                [0.80455070, 0.72109749, 0.15191695],
                [0.75018299, 0.17350990, 0.66075881],
            ]
        )
    )


ORIGINAL_STRATEGIES: tuple[tuple[str, Callable[[], GeneStrategy]], ...] = (
    ("Dominant gene", DominantGeneStrategy),
    ("Selfish gene", SelfishGeneStrategy),
    ("Kin-altruistic gene, full neighbourhood", KinAltruisticGeneStrategy),
    ("Altruistic gene", AltruisticGeneStrategy),
    ("Sell hard gene", SellHardGeneStrategy),
    ("Sell uniform gene", SellUniformGeneStrategy),
    ("Sell easy gene", SellEasyGeneStrategy),
    ("Variance gene", VarianceGeneStrategy),
)

CONFIGURATIONS = tuple(
    DMatrixConfiguration(
        name=name,
        formulation=StrategyFormulation.ORIGINAL,
        population_factory=original_population,
        gene_strategy_factory=strategy_factory,
        org_strategy_factory=NoneOrgStrategy,
        initial_gene_fitness=(0.6, 0.3, 0.1),
    )
    for name, strategy_factory in ORIGINAL_STRATEGIES
) + (
    DMatrixConfiguration(
        name="Dominant gene",
        formulation=StrategyFormulation.MATH_PAPER,
        population_factory=math_paper_population,
        gene_strategy_factory=DominantGeneStrategy,
        org_strategy_factory=NoneOrgStrategy,
        initial_gene_fitness=(0.4, 0.35, 0.25),
    ),
    DMatrixConfiguration(
        name="Altruistic gene + Selfish organism (Alt-Sel)",
        formulation=StrategyFormulation.MATH_PAPER,
        population_factory=math_paper_population,
        gene_strategy_factory=AltruisticGeneStrategy,
        org_strategy_factory=SelfishOrgStrategy,
        initial_gene_fitness=(0.4, 0.35, 0.25),
    ),
)


def fit_configuration(
    configuration: DMatrixConfiguration,
    *,
    use_d_matrix: bool,
    iterations: int,
) -> tuple[np.ndarray, float]:
    """Fit one model and return its final gene fitness and elapsed seconds."""
    model = PikaiaModel(
        population=configuration.population_factory(),
        gene_strategies=[configuration.gene_strategy_factory()],
        org_strategies=[configuration.org_strategy_factory()],
        initial_gene_fitness=configuration.initial_gene_fitness,
        max_iter=iterations,
        use_d_matrix=use_d_matrix,
        formulation=configuration.formulation,
    )
    started = perf_counter()
    model.fit()
    elapsed = perf_counter() - started
    return model.gene_fitness_history[iterations], elapsed


def compare_configuration(
    configuration: DMatrixConfiguration,
) -> tuple[list[float], float, float]:
    """Return path errors and median 100-step runtimes.

    Correctness is checked independently at 1, 50, and 100 iterations. Runtime
    values are medians over ``TIMING_REPEATS`` complete 100-iteration fits to
    reduce one-off scheduling noise.
    """
    differences: list[float] = []

    for iterations in (1, 50, 100):
        iterative, _ = fit_configuration(
            configuration,
            use_d_matrix=False,
            iterations=iterations,
        )
        reduced, _ = fit_configuration(
            configuration,
            use_d_matrix=True,
            iterations=iterations,
        )
        np.testing.assert_allclose(iterative, reduced, rtol=1e-12, atol=1e-12)
        differences.append(float(np.max(np.abs(iterative - reduced))))

    iterative_runtimes = [
        fit_configuration(configuration, use_d_matrix=False, iterations=100)[1]
        for _ in range(TIMING_REPEATS)
    ]
    d_matrix_runtimes = [
        fit_configuration(configuration, use_d_matrix=True, iterations=100)[1]
        for _ in range(TIMING_REPEATS)
    ]

    return differences, median(iterative_runtimes), median(d_matrix_runtimes)


def main() -> None:
    """Run and print the supported-configuration comparison."""
    heading = (
        f"{'Formulation':<12} {'Configuration':<48} "
        f"{'1 iter':>10} {'50 iter':>10} {'100 iter':>10} "
        f"{'iter ms':>10} {'D ms':>10}"
    )
    print(heading)
    print("-" * len(heading))

    for configuration in CONFIGURATIONS:
        differences, iterative_runtime, d_matrix_runtime = compare_configuration(
            configuration
        )
        print(
            f"{configuration.formulation.value:<12} "
            f"{configuration.name:<48} "
            f"{differences[0]:>10.2e} "
            f"{differences[1]:>10.2e} "
            f"{differences[2]:>10.2e} "
            f"{iterative_runtime * 1000:>10.3f} "
            f"{d_matrix_runtime * 1000:>10.3f}"
        )

    print("\nAll differences satisfy rtol=1e-12 and atol=1e-12.")
    print(f"Runtimes are medians of {TIMING_REPEATS} complete 100-iteration fits.")


if __name__ == "__main__":
    main()
