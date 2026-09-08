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
)
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
                [0.10, 0.60, 0.90],
                [0.80, 0.20, 0.40],
                [0.30, 0.90, 0.10],
                [0.70, 0.40, 0.60],
            ]
        )
    )


def _fit_one_iteration(
    *, gene_strategies, org_strategies, use_d_matrix, initial_gene_fitness
):
    model = PikaiaModel(
        population=_population(),
        gene_strategies=gene_strategies,
        org_strategies=org_strategies,
        initial_gene_fitness=initial_gene_fitness,
        max_iter=1,
        use_d_matrix=use_d_matrix,
    )
    model.fit()
    return model


@pytest.mark.parametrize(
    ("gene_strategies", "org_strategies"),
    [
        (
            [AltruisticGeneStrategy(formulation="MATH_PAPER")],
            [NoneOrgStrategy()],
        ),
        (
            [NoneGeneStrategy()],
            [SelfishOrgStrategy(formulation="MATH_PAPER")],
        ),
    ],
)
@pytest.mark.parametrize(
    "initial_gene_fitness",
    [[1 / 3, 1 / 3, 1 / 3], [0.6, 0.3, 0.1], [0.1, 0.2, 0.7]],
)
def test_math_paper_d_matrix_matches_iterative_path(
    gene_strategies, org_strategies, initial_gene_fitness
):
    iterative = _fit_one_iteration(
        gene_strategies=gene_strategies,
        org_strategies=org_strategies,
        use_d_matrix=False,
        initial_gene_fitness=initial_gene_fitness,
    )
    d_matrix = _fit_one_iteration(
        gene_strategies=gene_strategies,
        org_strategies=org_strategies,
        use_d_matrix=True,
        initial_gene_fitness=initial_gene_fitness,
    )

    np.testing.assert_allclose(
        iterative.gene_fitness_history[1],
        d_matrix.gene_fitness_history[1],
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
        gene_strategies=[AltruisticGeneStrategy(formulation="MATH_PAPER")],
        org_strategies=[NoneOrgStrategy()],
        max_iter=1,
    )

    with pytest.raises(ValueError, match="gene_mean_pairwise_difference"):
        model.fit()


def test_math_paper_dominant_rejects_d_matrix_path():
    model = PikaiaModel(
        population=_population(),
        gene_strategies=[DominantGeneStrategy(formulation="MATH_PAPER")],
        org_strategies=[NoneOrgStrategy()],
        max_iter=1,
        use_d_matrix=True,
    )

    with pytest.raises(ValueError, match="does not support use_d_matrix=True"):
        model.fit()
