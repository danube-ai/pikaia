# API Reference

## Core

### Population

::: pikaia.data.population.PikaiaPopulation

### Model

::: pikaia.models.pikaia_model.PikaiaModel

### Preprocessor

::: pikaia.preprocessing.pikaia_preprocessor.PikaiaPreprocessor

### Plotter

::: pikaia.plotting.pikaia_plotter.PikaiaPlotter

::: pikaia.plotting.pikaia_plotter.PlotType

## Schemas

::: pikaia.schemas.strategies.GeneStrategyEnum

::: pikaia.schemas.strategies.OrgStrategyEnum

::: pikaia.schemas.strategies.MixStrategyEnum

## Strategies

### Base Classes

::: pikaia.strategies.base_strategies.GeneStrategy

::: pikaia.strategies.base_strategies.OrgStrategy

::: pikaia.strategies.base_strategies.MixStrategy

::: pikaia.strategies.base_strategies.StrategyContext

### Factories

::: pikaia.strategies.strategy_factories.GeneStrategyFactory

::: pikaia.strategies.strategy_factories.OrgStrategyFactory

::: pikaia.strategies.strategy_factories.MixStrategyFactory

### Gene Strategies

::: pikaia.strategies.gs_strategies.dominant_strategy.DominantGeneStrategy

::: pikaia.strategies.gs_strategies.reward_hard_strategy.RewardHardGeneStrategy

::: pikaia.strategies.gs_strategies.reward_easy_strategy.RewardEasyGeneStrategy

::: pikaia.strategies.gs_strategies.altruistic_strategy.AltruisticGeneStrategy

::: pikaia.strategies.gs_strategies.selfish_strategy.SelfishGeneStrategy

::: pikaia.strategies.gs_strategies.kin_altruistic_strategy.KinAltruisticGeneStrategy

::: pikaia.strategies.gs_strategies.valuation_blend_strategy.ValuationBlendGeneStrategy

::: pikaia.strategies.gs_strategies.entropy_max_strategy.EntropyMaxGeneStrategy

::: pikaia.strategies.gs_strategies.orthogonality_strategy.OrthoGeneStrategy

::: pikaia.strategies.gs_strategies.partial_corr_strategy.PartialCorrGeneStrategy

::: pikaia.strategies.gs_strategies.redundancy_penalty_strategy.RedundancyPenaltyGeneStrategy

### Organism Strategies

::: pikaia.strategies.os_strategies.balanced_strategy.BalancedOrgStrategy

::: pikaia.strategies.os_strategies.altruistic_strategy.AltruisticOrgStrategy

::: pikaia.strategies.os_strategies.selfish_strategy.SelfishOrgStrategy

::: pikaia.strategies.os_strategies.kin_selfish_strategy.KinSelfishOrgStrategy

::: pikaia.strategies.os_strategies.sell_strategy.SellOrgStrategy

::: pikaia.strategies.os_strategies.buy_strategy.BuyOrgStrategy

### Mix Strategies

::: pikaia.strategies.mix_strategies.fixed_strategy.FixedMixStrategy

::: pikaia.strategies.mix_strategies.self_consistent_strategy.SelfConsistentMixStrategy
