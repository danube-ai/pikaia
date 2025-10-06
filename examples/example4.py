#!/usr/bin/env python3
"""
Example 4: Search with Movie Data

This script demonstrates using movie data for evolutionary search and recommendation.
It prints results and saves plots for visualization.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

print("=== Example 4: Search with Movie Data ===\n")

# Data Processing
print("Data Processing")
movie_df = pd.read_csv("examples/data/movie_matrix.csv")

print("Movie data shape:", movie_df.shape)
print("Columns:", list(movie_df.columns))

# Preprocess data
gene_labels = list(movie_df.columns)
org_labels = movie_df.iloc[:, 0].tolist()
movie_raw = movie_df.iloc[:, 1:].values

# Shuffle rows and columns and select a subset of features for quicker execution
movie_raw = movie_raw[:, :16]

print("Using first 16 features for quicker execution.")

# Define feature types: all movie features default to GAIN
feature_types = [FeatureType.GAIN] * 16
feature_transforms = [min_max_scaler] * 16
preprocessor = PikaiaPreprocessor(
    num_features=16, feature_types=feature_types, feature_transforms=feature_transforms
)
movie_scaled = preprocessor.fit_transform(movie_raw)

# Create PikaiaPopulation
population_movie = PikaiaPopulation(movie_scaled)

population_4 = population_movie
print("Number of organisms:", population_4.N)
print("Number of genes:", population_4.M)
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
    population=population_4,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    gene_mix_strategy=gene_mix_strategy,
    org_mix_strategy=org_mix_strategy,
    max_iter=16,
)
print("Fitting model (this may take a while)...")
model.fit()

print("Gene fitness history shape:", model.gene_fitness_history.shape)
print("Final gene fitness:", model.gene_fitness_history[-1][:5], "...")  # Show first 5
print("Org fitness history shape:", model.organism_fitness_history.shape)
print("Final org fitness:", model.organism_fitness_history[-1][:5], "...")
print()

# Plotting and Results
print("Plotting and Results")
print("Saving plots...")

plotter = PikaiaPlotter(model)
plotter.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=False,
    gene_labels=gene_labels,
)
plt.savefig("examples/artefacts/example4_gene_fitness.png", dpi=300)
print("Gene fitness history plot saved to examples/artefacts/example4_gene_fitness.png")

plotter.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=False,
    org_labels=org_labels,
)
plt.savefig("examples/artefacts/example4_org_fitness.png", dpi=300)
print("Org fitness history plot saved to examples/artefacts/example4_org_fitness.png")

sorted_indices = np.argsort(model.organism_fitness_history[-1, :])[::-1]

print("\nTop 5 Movies:")
for i in range(5):
    print(
        f"{i + 1}. "
        f"{org_labels[sorted_indices[i]]} "
        f"(Fitness: {model.organism_fitness_history[-1, sorted_indices[i]]:.4f})"
    )

print("\n=== Example 4 completed. Plots saved as PNG files. ===")
