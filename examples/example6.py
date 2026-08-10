#!/usr/bin/env python3
"""
Example 6: Trading Strategies — three matched sell+buy pairs

This script demonstrates the three trading strategy pairs:

  - SELL_HARD + BUY_HARD     — sell signal weighted by gene difficulty;
                                redistribute capital to easy genes the organism failed
  - SELL_UNIFORM + BUY_UNIFORM — uniform sell signal;
                                  redistribute capital to hard genes the organism failed
  - SELL_EASY + BUY_EASY     — sell signal weighted by gene ease (inverse of SELL_HARD);
                                redistribute capital mirroring BUY_HARD

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

print("=== Example 6: Trading Strategies ===\n")

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

runs = [
    {
        "label": "SELL_HARD + BUY_HARD",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL_HARD),
        ],
        "org_strategies": [
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY_HARD),
        ],
        "description": "Sell drains rare genes; buy redistributes capital to easy genes the organism failed",
    },
    {
        "label": "SELL_UNIFORM + BUY_UNIFORM",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL_UNIFORM),
        ],
        "org_strategies": [
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY_UNIFORM),
        ],
        "description": "Sell drains all genes uniformly; buy redistributes capital to hard genes the organism failed",
    },
    {
        "label": "SELL_EASY + BUY_EASY",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELL_EASY),
        ],
        "org_strategies": [
            OrgStrategyFactory.get_strategy(OrgStrategyEnum.BUY_EASY),
        ],
        "description": "Sell drains easy genes (inverse of SELL_HARD); buy redistributes capital mirroring BUY_HARD",
    },
]

# ---------------------------------------------------------------------------
# Fit all runs and collect results
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
        .replace("+", "plus")
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
