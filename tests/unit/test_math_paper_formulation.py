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


def test_math_paper_dominant_requires_the_iterative_path():
    pop = _population()
    strategy = DominantGeneStrategy(formulation=StrategyFormulation.MATH_PAPER)
    D, d = strategy.kernel(pop, np.eye(pop.M), np.eye(pop.N), 1.0)

    assert D is None
    assert d is None


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


def test_only_strategies_with_math_paper_equations_accept_math_paper():
    supported_gene_strategies = {
        GeneStrategyEnum.DOMINANT,
        GeneStrategyEnum.ALTRUISTIC,
    }
    supported_org_strategies = {OrgStrategyEnum.SELFISH}

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


def test_math_paper_model_rejects_original_only_strategies():
    """Fail at construction rather than silently retaining an original equation."""
    with pytest.raises(ValueError, match="NoneOrgStrategy"):
        PikaiaModel(
            population=_population(),
            gene_strategies=[AltruisticGeneStrategy()],
            org_strategies=[NoneOrgStrategy()],
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


def test_math_paper_uses_legacy_similarity_scaling():
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


def test_math_paper_dominant_rejects_d_matrix_path():
    model = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy(formulation="MATH_PAPER")],
        org_strategies=[SelfishOrgStrategy()],
        max_iter=1,
        use_d_matrix=True,
        formulation="MATH_PAPER",
    )

    with pytest.raises(ValueError, match="only for the unmixed"):
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


def test_strategy_set_formulation_rejects_unsupported_selection():
    """Both base strategy families retain formulation validation after construction."""
    with pytest.raises(ValueError, match="does not support MATH_PAPER"):
        NoneGeneStrategy().set_formulation(StrategyFormulation.MATH_PAPER)
    with pytest.raises(ValueError, match="does not support MATH_PAPER"):
        NoneOrgStrategy().set_formulation(StrategyFormulation.MATH_PAPER)


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
