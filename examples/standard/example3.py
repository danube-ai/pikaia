#!/usr/bin/env python3
"""
Example 3: Self-Consistency

This script demonstrates self-consistency using the 10x5 dataset with SELF_CONSISTENT mixing.
It prints results and saves plots for visualization.
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

print("=== Example 3: Self-Consistency ===\n")

# Data and Setup
print("Data and Setup")
data_10x5_raw = np.array(
    [
        [300, 10, 2, 0, 2.5],
        [600, 5, 2, 1, 3.0],
        [1500, 4, 1, 2, 4.0],
        [400, 8, 2, 0, 3.5],
        [500, 8, 2, 1, 3.0],
        [700, 5, 2, 1, 4.5],
        [900, 6, 1, 1, 4.0],
        [1100, 6, 1, 2, 3.5],
        [1300, 5, 2, 2, 5.0],
        [1700, 4, 1, 2, 5.0],
    ]
)

feature_types = [
    FeatureType.COST,
    FeatureType.COST,
    FeatureType.COST,
    FeatureType.GAIN,
    FeatureType.GAIN,
]
feature_transforms = [min_max_scaler] * 5
preprocessor = PikaiaPreprocessor(
    num_features=data_10x5_raw.shape[1],
    feature_types=feature_types,
    feature_transforms=feature_transforms,
)
data_10x5_scaled = preprocessor.fit_transform(data_10x5_raw)

population10x5 = PikaiaPopulation(data_10x5_scaled)

gene_labels = [
    "gene 1 = price",
    "gene 2 = time",
    "gene 3 = stops",
    "gene 4 = luggage",
    "gene 5 = rating",
]
org_labels = [f"flight {i}" for i in range(population10x5.N)]

population_3 = population10x5
print("Number of organisms:", population_3.N)
print("Number of genes:", population_3.M)
print()

# Model Setup and Fitting
print("Model Setup and Fitting")
gene_strategies = [
    GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT),
    GeneStrategyFactory.get_strategy(GeneStrategyEnum.ALTRUISTIC),
]
org_strategies = [
    OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED),
    OrgStrategyFactory.get_strategy(OrgStrategyEnum.ALTRUISTIC),
]
gene_mix_strategy = org_mix_strategy = MixStrategyFactory.get_strategy(
    MixStrategyEnum.SELF_CONSISTENT
)

model = PikaiaModel(
    population=population_3,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    gene_mix_strategy=gene_mix_strategy,
    org_mix_strategy=org_mix_strategy,
    max_iter=32,
)
print("Fitting model...")
model.fit()

print("Gene fitness history shape:", model.gene_fitness_history.shape)
print("Initial gene fitness:", model.gene_fitness_history[0])
print("Final gene fitness:", model.gene_fitness_history[-1])
print("Org fitness history shape:", model.organism_fitness_history.shape)
print("Initial org fitness:", model.organism_fitness_history[0])
print("Final org fitness:", model.organism_fitness_history[-1])
print()

# Plotting Results
print("Plotting Results")
print("Saving plots...")

plotter = PikaiaPlotter(model)
plotter.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=False,
    save_path=Path("artefacts/example3_gene_fitness.png"),
    gene_labels=gene_labels,
)
print("Gene fitness history plot saved to artefacts/example3_gene_fitness.png")

plotter.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=False,
    save_path=Path("artefacts/example3_org_fitness.png"),
    org_labels=org_labels,
)
print("Org fitness history plot saved to artefacts/example3_org_fitness.png")

print("\n=== Example 3 completed. Plots saved as PNG files. ===")
