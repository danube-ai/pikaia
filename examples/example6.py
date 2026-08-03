#!/usr/bin/env python3
"""
Example 6: Valuation Strategies — Reward Hard vs Reward Easy vs Blend vs Sell+Buy

This script demonstrates the valuation strategies ported from the tgeneticai
CalSim framework (experiments/tgeneticai/calsim.py):

  - REWARD_HARD   — rewards features that are hard to achieve (high difficulty)
  - REWARD_EASY   — rewards features that are easy to achieve (low difficulty)
  - VALUATION_BLEND — interpolates between the two via a preference parameter
  - SELL + BUY    — full CalSim market recalibration round reproduced inside
                    the replicator framework.  Both are organism strategies:
                    SELL drains value from commonly-expressed genes;
                    BUY redistributes capital from high-performers to genes
                    they lack.

The original CalSim used these as "sellStrategy" settings:
  "Difficulty1" → SELL (org) + BUY (org)

We use a synthetic dataset with one easy, one hard, and two medium-difficulty
features so the strategies produce visibly different fitness trajectories.
"""

from pathlib import Path

import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.plotting import PikaiaPlotter, PlotType
from pikaia.preprocessing import PikaiaPreprocessor, min_max_scaler
from pikaia.schemas import (
    FeatureType,
    GeneStrategyEnum,
    MixStrategyEnum,
    OrgStrategyEnum,
)
from pikaia.strategies import (
    GeneStrategyFactory,
    MixStrategyFactory,
    OrgStrategyFactory,
)

print("=== Example 6: Valuation Strategies ===\n")

# ---------------------------------------------------------------------------
# Data — synthetic 10x4 with controlled difficulty levels
# ---------------------------------------------------------------------------
print("Data Processing")

np.random.seed(42)
data = np.zeros((10, 4))

# Feature 0: easy  — all organisms score high (near 1.0)
data[:, 0] = np.random.uniform(0.8, 1.0, 10)

# Feature 1: hard  — most organisms score low (near 0.0)
data[:, 1] = np.random.uniform(0.0, 0.2, 10)

# Features 2 & 3: medium — spread across [0, 1]
data[:, 2] = np.random.uniform(0.3, 0.7, 10)
data[:, 3] = np.random.uniform(0.2, 0.8, 10)

print("Synthetic data (columns: easy, hard, medium, medium):")
print(np.round(data, 2))
print()

feature_types = [FeatureType.GAIN] * 4
feature_transforms = [min_max_scaler] * 4
preprocessor = PikaiaPreprocessor(
    num_features=4,
    feature_types=feature_types,
    feature_transforms=feature_transforms,
)
data_scaled = preprocessor.fit_transform(data)

population = PikaiaPopulation(data_scaled)

gene_labels = [
    "gene 0 = easy (high values)",
    "gene 1 = hard  (low values)",
    "gene 2 = medium",
    "gene 3 = medium",
]
org_labels = [f"organism {i}" for i in range(population.N)]

print(f"Organisms: {population.N},  Genes: {population.M}")
print()

# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------
gene_mix_strategy = org_mix_strategy = MixStrategyFactory.get_strategy(
    MixStrategyEnum.FIXED
)

_balanced = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)]

runs = [
    {
        "label": "REWARD_HARD",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_HARD),
        ],
        "org_strategies": _balanced,
        "description": "Harder features gain fitness",
    },
    {
        "label": "REWARD_EASY",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.REWARD_EASY),
        ],
        "org_strategies": _balanced,
        "description": "Easier features gain fitness",
    },
    {
        "label": "VALUATION_BLEND (p=0.3)",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(
                GeneStrategyEnum.VALUATION_BLEND, preference=0.3
            ),
        ],
        "org_strategies": _balanced,
        "description": "Leans toward rewarding easy features",
    },
    {
        "label": "SELL + BUY (CalSim Difficulty1)",
        "gene_strategies": [],
        "org_strategies": [
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELL),
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY),
        ],
        "description": "Full CalSim market recalibration: sell drains common genes, "
        "buy redistributes capital from high-performers to genes they lack",
    },
]

# ---------------------------------------------------------------------------
# Fit all three and collect results
# ---------------------------------------------------------------------------
print("Model Setup and Fitting")

models = []
for run in runs:
    print(f"\n  Running: {run['label']}")
    print(f"  {run['description']}")

    model = PikaiaModel(
        population=PikaiaPopulation(data_scaled.copy()),
        gene_strategies=run["gene_strategies"],
        org_strategies=run["org_strategies"],
        gene_mix_strategy=gene_mix_strategy,
        org_mix_strategy=org_mix_strategy,
        max_iter=32,
    )
    model.fit()
    models.append(model)
    print(f"  Final gene fitness: {np.round(model.gene_fitness_history[-1], 4)}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
print("\nPlotting Results")
print("Saving plots...")

OUT_DIR = Path("artefacts/example6")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for model, run in zip(models, runs):
    plotter = PikaiaPlotter(model)
    safe_label = (
        run["label"]
        .lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("=", "")
        .replace(".", "")
    )
    plotter.plot(
        plot_type=PlotType.GENE_FITNESS_HISTORY,
        show=False,
        save_path=OUT_DIR / f"gene_fitness_{safe_label}.png",
        gene_labels=gene_labels,
        title=run["label"],
    )
    print(f"  Saved {run['label']} plot -> {OUT_DIR}/gene_fitness_{safe_label}.png")

    plotter.plot(
        plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
        show=False,
        save_path=OUT_DIR / f"org_fitness_{safe_label}.png",
        org_labels=org_labels,
        title=run["label"],
    )
    print(f"  Saved {run['label']} org plot -> {OUT_DIR}/org_fitness_{safe_label}.png")

print("\n=== Example 6 completed. Plots saved as PNG files. ===")
