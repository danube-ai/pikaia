"""Regression tests for the revised mathematical-paper strategy formulation."""

import numpy as np
import pytest
from pydantic import ValidationError

from pikaia.data.population import PikaiaPopulation
from pikaia.models.pikaia_model import PikaiaModel
from pikaia.schemas.strategies import (
    GeneStrategyEnum,
    OrgStrategyEnum,
    StrategyFormulation,
    StrategyNormalizations,
)
from pikaia.strategies.base_strategies import StrategyContext
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy
from pikaia.strategies.strategy_factories import GeneStrategyFactory, OrgStrategyFactory


def _population() -> PikaiaPopulation:
    """Return a population with non-zero math-paper normalizations."""
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


def _fit(
    *,
    gene_strategies,
    org_strategies,
    use_d_matrix,
    initial_gene_fitness,
    max_iter,
    formulation=StrategyFormulation.MATH_PAPER,
):
    """Run one formulation-controlled model for the requested iterations."""
    model = PikaiaModel(
        population=_population(),
        gene_strategies=gene_strategies,
        org_strategies=org_strategies,
        initial_gene_fitness=initial_gene_fitness,
        max_iter=max_iter,
        use_d_matrix=use_d_matrix,
        formulation=formulation,
    )
    model.fit()
    return model


@pytest.mark.parametrize(
    "initial_gene_fitness",
    [[0.4, 0.35, 0.25]],
)
@pytest.mark.parametrize("max_iter", [1, 50, 100])
def test_math_paper_altsel_d_matrix_matches_iterative_path(
    initial_gene_fitness, max_iter
):
    """The historical Alt-Sel pair is exact in both model paths."""
    iterative = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        use_d_matrix=False,
        initial_gene_fitness=initial_gene_fitness,
        max_iter=max_iter,
    )
    d_matrix = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        use_d_matrix=True,
        initial_gene_fitness=initial_gene_fitness,
        max_iter=max_iter,
    )

    np.testing.assert_allclose(
        iterative.gene_fitness_history[max_iter],
        d_matrix.gene_fitness_history[max_iter],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "gene_fitness",
    [np.array([0.4, 0.35, 0.25]), np.array([0.2, 0.3, 0.5])],
)
def test_math_paper_altruistic_kernel_matches_its_isolated_iterative_delta(
    gene_fitness: np.ndarray,
) -> None:
    """The gene component ``D^G`` is exact without relying on ``D^O``."""
    model = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        use_d_matrix=False,
        initial_gene_fitness=gene_fitness,
        max_iter=1,
    )
    population = model.population
    strategy = AltruisticGeneStrategy(formulation="MATH_PAPER")
    organism_fitness = population.matrix @ gene_fitness

    iterative_delta = np.array(
        [
            sum(
                strategy(
                    StrategyContext(
                        population=population,
                        org_fitness=organism_fitness,
                        gene_fitness=gene_fitness,
                        org_similarity=model._active_org_similarity,
                        gene_similarity=model._active_gene_similarity,
                        initial_org_fitness_range=model._initial_org_fitness_range,
                        org_id=organism_id,
                        gene_id=gene_id,
                        normalizations=model._strategy_normalizations,
                    )
                )
                for organism_id in range(population.N)
            )
            for gene_id in range(population.M)
        ]
    )
    d_matrix, linear_vector = strategy.kernel(
        population,
        model._active_gene_similarity,
        model._active_org_similarity,
        model._initial_org_fitness_range,
        normalizations=model._strategy_normalizations,
    )

    assert d_matrix is not None
    assert linear_vector is None
    np.testing.assert_allclose(
        iterative_delta,
        gene_fitness * (d_matrix @ gene_fitness),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "gene_fitness",
    [np.array([0.4, 0.35, 0.25]), np.array([0.2, 0.3, 0.5])],
)
def test_math_paper_selfish_kernel_matches_its_isolated_iterative_delta(
    gene_fitness: np.ndarray,
) -> None:
    """The organism component ``D^O`` is exact without relying on ``D^G``."""
    model = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        use_d_matrix=False,
        initial_gene_fitness=gene_fitness,
        max_iter=1,
    )
    population = model.population
    strategy = SelfishOrgStrategy(formulation="MATH_PAPER")
    organism_fitness = population.matrix @ gene_fitness

    iterative_delta = np.sum(
        [
            strategy(
                StrategyContext(
                    population=population,
                    org_fitness=organism_fitness,
                    gene_fitness=gene_fitness,
                    org_similarity=model._active_org_similarity,
                    gene_similarity=model._active_gene_similarity,
                    initial_org_fitness_range=model._initial_org_fitness_range,
                    org_id=organism_id,
                    normalizations=model._strategy_normalizations,
                )
            )
            for organism_id in range(population.N)
        ],
        axis=0,
    )
    d_matrix, linear_vector = strategy.kernel(
        population,
        model._active_gene_similarity,
        model._active_org_similarity,
        model._initial_org_fitness_range,
        normalizations=model._strategy_normalizations,
    )

    assert d_matrix is not None
    assert linear_vector is None
    np.testing.assert_allclose(
        iterative_delta,
        gene_fitness * (d_matrix @ gene_fitness),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "gene_fitness",
    [np.array([0.4, 0.35, 0.25]), np.array([0.2, 0.3, 0.5])],
)
def test_math_paper_dominant_kernel_matches_its_isolated_iterative_delta(
    gene_fitness: np.ndarray,
) -> None:
    """A row-constant D matrix exactly reproduces the linear dominant signal."""
    pop = _population()
    strategy = DominantGeneStrategy(formulation=StrategyFormulation.MATH_PAPER)
    D, d = strategy.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0)

    assert D is not None
    assert d is None
    expected_delta = gene_fitness * (pop.matrix.mean(axis=0) - 0.5)
    np.testing.assert_allclose(gene_fitness * (D @ gene_fitness), expected_delta)


@pytest.mark.parametrize("max_iter", [1, 50, 100])
def test_math_paper_dominant_d_matrix_matches_iterative_path(max_iter: int) -> None:
    """Isolated math-paper dominant agrees through short and long runs."""
    arguments = {
        "gene_strategies": [DominantGeneStrategy()],
        "org_strategies": [NoneOrgStrategy()],
        "initial_gene_fitness": [0.4, 0.35, 0.25],
        "max_iter": max_iter,
    }
    iterative = _fit(**arguments, use_d_matrix=False)
    d_matrix = _fit(**arguments, use_d_matrix=True)

    np.testing.assert_allclose(
        iterative.gene_fitness_history[max_iter],
        d_matrix.gene_fitness_history[max_iter],
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("seed", [7, 29, 101])
@pytest.mark.parametrize("max_iter", [1, 50, 100])
def test_math_paper_dominant_d_matrix_matches_on_deterministic_populations(
    seed: int, max_iter: int
) -> None:
    """Verify the reduction beyond the fixed documentation fixture."""
    population = PikaiaPopulation(np.random.default_rng(seed).random((7, 4)))
    shared_arguments = {
        "population": population,
        "gene_strategies": [DominantGeneStrategy()],
        "org_strategies": [NoneOrgStrategy()],
        "initial_gene_fitness": [0.4, 0.3, 0.2, 0.1],
        "max_iter": max_iter,
        "formulation": StrategyFormulation.MATH_PAPER,
    }
    iterative = PikaiaModel(**shared_arguments, use_d_matrix=False)
    d_matrix = PikaiaModel(**shared_arguments, use_d_matrix=True)
    iterative.fit()
    d_matrix.fit()

    np.testing.assert_allclose(
        iterative.gene_fitness_history[max_iter],
        d_matrix.gene_fitness_history[max_iter],
        rtol=1e-12,
        atol=1e-12,
    )


def test_original_formulation_is_the_default():
    assert DominantGeneStrategy().formulation is StrategyFormulation.ORIGINAL
    assert AltruisticGeneStrategy().formulation is StrategyFormulation.ORIGINAL
    assert SelfishOrgStrategy().formulation is StrategyFormulation.ORIGINAL


def test_every_gene_and_organism_strategy_defaults_to_original_formulation():
    for strategy_enum in GeneStrategyEnum:
        strategy = GeneStrategyFactory.get_strategy(strategy_enum)
        assert strategy.formulation is StrategyFormulation.ORIGINAL

    for strategy_enum in OrgStrategyEnum:
        strategy = OrgStrategyFactory.get_strategy(strategy_enum)
        assert strategy.formulation is StrategyFormulation.ORIGINAL


def test_only_math_paper_strategies_and_neutral_noops_accept_math_paper():
    supported_gene_strategies = {
        GeneStrategyEnum.DOMINANT,
        GeneStrategyEnum.ALTRUISTIC,
        GeneStrategyEnum.NONE,
    }
    supported_org_strategies = {OrgStrategyEnum.SELFISH, OrgStrategyEnum.NONE}

    for strategy_enum in GeneStrategyEnum:
        factory = GeneStrategyFactory.get_strategy
        if strategy_enum in supported_gene_strategies:
            assert (
                factory(strategy_enum, formulation="MATH_PAPER").formulation
                is StrategyFormulation.MATH_PAPER
            )
        else:
            with pytest.raises(ValueError, match="does not support MATH_PAPER"):
                factory(strategy_enum, formulation="MATH_PAPER")

    for strategy_enum in OrgStrategyEnum:
        factory = OrgStrategyFactory.get_strategy
        if strategy_enum in supported_org_strategies:
            assert (
                factory(strategy_enum, formulation="MATH_PAPER").formulation
                is StrategyFormulation.MATH_PAPER
            )
        else:
            with pytest.raises(ValueError, match="does not support MATH_PAPER"):
                factory(strategy_enum, formulation="MATH_PAPER")


def test_formulation_is_validated_by_pydantic():
    with pytest.raises(ValidationError):
        AltruisticGeneStrategy(formulation="not-a-formulation")


def test_math_paper_rejects_zero_gene_normalization():
    population = PikaiaPopulation(
        np.array(
            [
                [0.1, 0.2],
                [0.9, 0.8],
            ]
        )
    )
    model = PikaiaModel(
        population=population,
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        max_iter=1,
        formulation="MATH_PAPER",
    )

    with pytest.raises(ValueError, match="gene_mean_pairwise_difference"):
        model.fit()


@pytest.mark.parametrize(
    ("gene_strategy", "org_strategy", "unsupported_name"),
    [
        (SelfishGeneStrategy(), SelfishOrgStrategy(), "SelfishGeneStrategy"),
        (AltruisticGeneStrategy(), BalancedOrgStrategy(), "BalancedOrgStrategy"),
    ],
)
def test_math_paper_model_rejects_original_only_strategies(
    gene_strategy, org_strategy, unsupported_name: str
) -> None:
    """Fail at construction rather than silently retaining an original equation."""
    with pytest.raises(ValueError, match=unsupported_name):
        PikaiaModel(
            population=_population(),
            gene_strategies=[gene_strategy],
            org_strategies=[org_strategy],
            max_iter=1,
            formulation="MATH_PAPER",
        )


def test_math_paper_model_applies_its_formulation_to_supported_strategies():
    """Default-constructed compatible strategies follow the model formulation."""
    model = PikaiaModel(
        population=_population(),
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        max_iter=1,
        formulation="MATH_PAPER",
    )

    assert model.formulation is StrategyFormulation.MATH_PAPER
    assert (
        next(iter(model.gene_strategies)).formulation is StrategyFormulation.MATH_PAPER
    )
    assert (
        next(iter(model.org_strategies)).formulation is StrategyFormulation.MATH_PAPER
    )


def test_math_paper_uses_formulation_specific_similarity_scaling():
    """Select the old branch's N/M similarity denominators for math-paper runs."""
    model = PikaiaModel(
        population=_population(),
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        max_iter=1,
        formulation="MATH_PAPER",
    )
    X = model.population.matrix
    expected_gene = (
        1
        - np.linalg.norm(X.T[:, np.newaxis, :] - X.T[np.newaxis, :, :], axis=2)
        / X.shape[0]
    )
    expected_org = (
        1
        - np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2) / X.shape[1]
    )

    np.testing.assert_allclose(model._active_gene_similarity, expected_gene)
    np.testing.assert_allclose(model._active_org_similarity, expected_org)
    np.testing.assert_allclose(model.gene_similarity, expected_gene)
    np.testing.assert_allclose(model.org_similarity, expected_org)

    gene_distances = np.linalg.norm(
        X.T[:, np.newaxis, :] - X.T[np.newaxis, :, :], axis=2
    )
    org_distances = np.linalg.norm(X[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
    original_gene = 1 - gene_distances / np.max(gene_distances)
    original_org = 1 - org_distances / np.max(org_distances)
    assert not np.allclose(model.gene_similarity, original_gene)
    assert not np.allclose(model.org_similarity, original_org)


def test_math_paper_dominant_does_not_compute_unused_original_similarity():
    """Identical columns remain valid under the math-paper N-scaled equation."""
    population = PikaiaPopulation(np.array([[0.1, 0.1], [0.8, 0.8], [0.4, 0.4]]))
    model = PikaiaModel(
        population=population,
        gene_strategies=[DominantGeneStrategy()],
        org_strategies=[NoneOrgStrategy()],
        initial_gene_fitness=[0.5, 0.5],
        max_iter=1,
        use_d_matrix=True,
        formulation=StrategyFormulation.MATH_PAPER,
    )

    model.fit()

    np.testing.assert_allclose(model.gene_similarity, np.ones((2, 2)))
    np.testing.assert_allclose(model.gene_fitness_history[1], [0.5, 0.5])


def test_math_paper_clamps_kin_range_to_population_size():
    """Match the old ``min(kin_range, N)`` denominator convention."""
    iterative = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy(kin_range=999)],
        use_d_matrix=False,
        initial_gene_fitness=[0.4, 0.35, 0.25],
        max_iter=1,
    )
    d_matrix = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy(kin_range=999)],
        use_d_matrix=True,
        initial_gene_fitness=[0.4, 0.35, 0.25],
        max_iter=1,
    )

    np.testing.assert_allclose(
        iterative.gene_fitness_history[1], d_matrix.gene_fitness_history[1]
    )


def test_math_paper_explicit_none_kin_range_uses_the_population_size():
    """Treat an explicitly omitted kin range the same as the default."""
    default = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy()],
        use_d_matrix=True,
        initial_gene_fitness=[0.4, 0.35, 0.25],
        max_iter=1,
    )
    explicit_none = _fit(
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[SelfishOrgStrategy(kin_range=None)],
        use_d_matrix=True,
        initial_gene_fitness=[0.4, 0.35, 0.25],
        max_iter=1,
    )

    np.testing.assert_allclose(
        default.gene_fitness_history[1], explicit_none.gene_fitness_history[1]
    )


def test_math_paper_rejects_non_positive_kin_range():
    """Apply Pydantic kin-range validation at the math-paper model boundary."""
    with pytest.raises(ValidationError):
        PikaiaModel(
            population=_population(),
            gene_strategies=[AltruisticGeneStrategy()],
            org_strategies=[SelfishOrgStrategy(kin_range=0)],
            max_iter=1,
            formulation="MATH_PAPER",
        )


def test_explicit_original_matches_the_default_model_formulation():
    """Keep original-model results unchanged by the compatibility profile."""
    kwargs = {
        "population": _population(),
        "gene_strategies": [AltruisticGeneStrategy()],
        "org_strategies": [SelfishOrgStrategy()],
        "initial_gene_fitness": [0.4, 0.35, 0.25],
        "max_iter": 1,
    }
    default_model = PikaiaModel(**kwargs)
    explicit_model = PikaiaModel(**kwargs, formulation="ORIGINAL")
    default_model.fit()
    explicit_model.fit()

    np.testing.assert_allclose(
        default_model.gene_fitness_history, explicit_model.gene_fitness_history
    )


def test_math_paper_dominant_rejects_non_noop_organism_partner():
    """Keep the isolated dominant public contract explicit and fail fast."""
    model = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy(formulation="MATH_PAPER")],
        org_strategies=[SelfishOrgStrategy()],
        max_iter=1,
        use_d_matrix=True,
        formulation="MATH_PAPER",
    )

    with pytest.raises(ValueError, match="available only"):
        model.fit()


def test_math_paper_dominant_d_matrix_requires_normalized_initial_fitness():
    """Reject input outside the simplex required by the row-constant proof."""
    model = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy()],
        org_strategies=[NoneOrgStrategy()],
        initial_gene_fitness=[0.8, 0.7, 0.5],
        max_iter=1,
        use_d_matrix=True,
        formulation=StrategyFormulation.MATH_PAPER,
    )

    with pytest.raises(ValueError, match="initial_gene_fitness to sum to one"):
        model.fit()


def test_math_paper_rejects_original_only_analytical_fixed_point():
    """Do not label the original Dominant-Balanced solution as math-paper."""
    model = PikaiaModel(
        population=_population(),
        formulation=StrategyFormulation.MATH_PAPER,
    )

    with pytest.raises(ValueError, match="requires max_iter"):
        model.fit()


def test_model_skips_fit_when_uniform_gene_fitness_cannot_rank_organisms():
    """A zero initial organism-fitness range returns before selecting a path."""
    model = PikaiaModel(
        population=PikaiaPopulation(np.array([[0.2, 0.8], [0.8, 0.2]])),
        gene_strategies=[AltruisticGeneStrategy()],
        org_strategies=[NoneOrgStrategy()],
        initial_gene_fitness=[0.5, 0.5],
        max_iter=1,
    )

    model.fit()
    np.testing.assert_allclose(model.gene_fitness_history[0], [0.5, 0.5])
    np.testing.assert_allclose(model.gene_fitness_history[1], [0.0, 0.0])


def test_noop_strategies_are_formulation_neutral():
    """No-op strategies contribute exact zero in either formulation."""
    gene_strategy = NoneGeneStrategy()
    org_strategy = NoneOrgStrategy()

    gene_strategy.set_formulation(StrategyFormulation.MATH_PAPER)
    org_strategy.set_formulation(StrategyFormulation.MATH_PAPER)

    assert gene_strategy.formulation is StrategyFormulation.MATH_PAPER
    assert org_strategy.formulation is StrategyFormulation.MATH_PAPER


def test_strategy_set_formulation_rejects_unsupported_selection():
    """Original-only strategies retain validation after construction."""
    with pytest.raises(ValueError, match="does not support MATH_PAPER"):
        SelfishGeneStrategy().set_formulation(StrategyFormulation.MATH_PAPER)
    with pytest.raises(ValueError, match="does not support MATH_PAPER"):
        BalancedOrgStrategy().set_formulation(StrategyFormulation.MATH_PAPER)


def test_math_paper_kernel_guards_require_normalizations():
    """Paper-only kernels reject direct calls that omit their fixed normalizers."""
    population = _population()
    altruistic = AltruisticGeneStrategy(formulation="MATH_PAPER")
    selfish = SelfishOrgStrategy(formulation="MATH_PAPER")

    with pytest.raises(ValueError, match="requires population normalizations"):
        altruistic.kernel(population, np.eye(population.M), np.eye(population.N), 1.0)
    with pytest.raises(ValueError, match="requires population normalizations"):
        selfish.kernel(population, np.eye(population.M), np.eye(population.N), 1.0)


def test_math_paper_strategy_call_branches_are_explicit():
    """Paper strategies select their revised equation and enforce normalizers."""
    population = _population()
    context = StrategyContext(
        population=population,
        org_fitness=np.full(population.N, 0.5),
        gene_fitness=np.array([0.4, 0.35, 0.25]),
        org_similarity=np.eye(population.N),
        gene_similarity=np.eye(population.M),
        initial_org_fitness_range=1.0,
        org_id=0,
        gene_id=0,
    )

    assert np.isfinite(DominantGeneStrategy(formulation="MATH_PAPER")(context))
    with pytest.raises(ValueError, match="requires population normalizations"):
        AltruisticGeneStrategy(formulation="MATH_PAPER")(context)


def test_selfish_organism_kernel_is_paper_only_and_handles_no_relatives():
    """The removed original kernel is absent while the paper zero case is stable."""
    population = PikaiaPopulation(np.array([[0.4, 0.6]]))
    original = SelfishOrgStrategy()
    paper = SelfishOrgStrategy(formulation="MATH_PAPER")
    normalizations = StrategyNormalizations(
        harmonic_fitness_mean_pairwise_difference=1.0
    )

    assert original.kernel(population, np.eye(2), np.eye(1), 1.0) == (None, None)
    D, d = paper.kernel(
        population,
        np.eye(2),
        np.eye(1),
        1.0,
        normalizations=normalizations,
    )
    assert D is not None
    np.testing.assert_allclose(D, 0.0)
    assert d is None
