#!/usr/bin/env python3
"""
Example 10: Variance vs Dominant gene strategies

Demonstrates VarianceGeneStrategy, which weights features by cross-organism
dispersion (normalised column std). Compared with DominantGeneStrategy on a
synthetic dataset with controlled spread:

  - Feature 0: low variance, high mean — Dominant favours level of expression
  - Feature 1: high variance, moderate-high mean — Variance amplifies this column
  - Features 2 & 3: medium spread

Data is already in [0, 1]; the preprocessor validates without min-max so column
dispersion is preserved. Variance should favour the high-dispersion column more
strongly than Dominant, which keys off mean expression level rather than spread.
"""

from pathlib import Path

import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.plotting import PikaiaPlotter, PlotType
from pikaia.preprocessing import PikaiaPreprocessor
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

print("=== Example 10: Variance vs Dominant ===\n")

# ---------------------------------------------------------------------------
# Data — synthetic 10x4 with controlled column dispersion
# ---------------------------------------------------------------------------
print("Data Processing")

np.random.seed(42)
data = np.zeros((10, 4))

# Feature 0: low variance, high mean — Dominant favours level; Variance down-weights
# because the column barely separates organisms.
data[:, 0] = np.linspace(0.88, 0.96, 10)

# Feature 1: high variance, moderate-high mean — Variance amplifies this column.
data[:, 1] = np.array([0.05, 0.20, 0.40, 0.55, 0.70, 0.80, 0.88, 0.92, 0.96, 1.00])

# Features 2 & 3: medium spread / medium means
data[:, 2] = np.array([0.35, 0.42, 0.48, 0.52, 0.58, 0.62, 0.68, 0.72, 0.78, 0.85])
data[:, 3] = np.array([0.25, 0.35, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90])

print("Synthetic data (columns: low-var/high-mean, high-var, medium, medium):")
print(np.round(data, 2))
print()
print("Column means (raw):", np.round(data.mean(axis=0), 4))
print("Column stds  (raw):", np.round(data.std(axis=0, ddof=0), 4))
print()

# Data is already in [0, 1]; skip min-max so column dispersion is preserved.
# (Per-column min-max would remap every feature to full [0, 1] and erase the demo.)
preprocessor = PikaiaPreprocessor(
    num_features=4,
    feature_types=[FeatureType.GAIN] * 4,
    feature_transforms=[None] * 4,
)
data_scaled = preprocessor.fit_transform(data)

population = PikaiaPopulation(data_scaled)

gene_labels = [
    "gene 0 = low var, high mean",
    "gene 1 = high variance",
    "gene 2 = medium",
    "gene 3 = medium",
]

print(f"Organisms: {population.N},  Genes: {population.M}")
print("Column stds (scaled):", np.round(data_scaled.std(axis=0, ddof=0), 4))
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
        "label": "VARIANCE",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.VARIANCE),
        ],
        "org_strategies": _balanced,
        "description": "High-dispersion features gain fitness",
    },
    {
        "label": "DOMINANT",
        "gene_strategies": [
            GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT),
        ],
        "org_strategies": _balanced,
        "description": "Highly expressed features gain fitness",
    },
]

# ---------------------------------------------------------------------------
# Fit and collect results
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
    final = model.gene_fitness_history[-1]
    print(f"  Final gene fitness: {np.round(final, 4)}")
    for label, fitness in zip(gene_labels, final):
        print(f"    {label}: {fitness:.4f}")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
print("\nPlotting Results")
print("Saving plots...")

OUT_DIR = Path("artefacts/example10")
OUT_DIR.mkdir(parents=True, exist_ok=True)

for model, run in zip(models, runs):
    plotter = PikaiaPlotter(model)
    safe_label = run["label"].lower()
    plotter.plot(
        plot_type=PlotType.GENE_FITNESS_HISTORY,
        show=False,
        save_path=OUT_DIR / f"gene_fitness_{safe_label}.png",
        gene_labels=gene_labels,
        title=run["label"],
    )
    print(f"  Saved {run['label']} plot -> {OUT_DIR}/gene_fitness_{safe_label}.png")

print("\n=== Example 10 completed. Plots saved as PNG files. ===")
