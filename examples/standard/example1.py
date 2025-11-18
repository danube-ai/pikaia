#!/usr/bin/env python3
"""
Example 1: 3x3 Data - Balanced vs. Altruistic Selection

This script demonstrates using a small 3x3 dataset to compare BALANCED and ALTRUISTIC strategies.
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

print("=== Example 1: 3x3 Data - Balanced vs. Altruistic Selection ===\n")

# Data Processing
print("Data Processing")
data_3x3_raw = np.array([[300, 10, 2], [600, 5, 2], [1500, 4, 1]])
print("Raw data:")
print(data_3x3_raw)

feature_types = [FeatureType.COST, FeatureType.COST, FeatureType.COST]
feature_transforms = [min_max_scaler] * data_3x3_raw.shape[1]
preprocessor = PikaiaPreprocessor(
    num_features=data_3x3_raw.shape[1],
    feature_types=feature_types,
    feature_transforms=feature_transforms,
)
data_3x3_scaled = preprocessor.fit_transform(data_3x3_raw)

population3x3 = PikaiaPopulation(data_3x3_scaled)

gene_labels = ["gene 1 = price", "gene 2 = time", "gene 3 = stops"]
org_labels = [f"flight {i}" for i in range(population3x3.N)]

population_1 = population3x3
print("Number of organisms:", population_1.N)
print("Number of genes:", population_1.M)
print()

# Model Setup and Fitting
print("Model Setup and Fitting")
gene_strategies = [
    GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT),
    GeneStrategyFactory.get_strategy(GeneStrategyEnum.ALTRUISTIC),
]
org_strategies = [
    OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED),
    OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELFISH),
]
gene_mix_strategy = org_mix_strategy = MixStrategyFactory.get_strategy(
    MixStrategyEnum.FIXED
)

model = PikaiaModel(
    population=population_1,
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
    save_path=Path("artefacts/example1_gene_fitness.png"),
    gene_labels=gene_labels,
)
print("Gene fitness history plot saved to artefacts/example1_gene_fitness.png")

plotter.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=False,
    save_path=Path("artefacts/example1_org_fitness.png"),
    org_labels=org_labels,
)
print("Org fitness history plot saved to artefacts/example1_org_fitness.png")

print("\n=== Example 1 completed. Plots saved as PNG files. ===")
