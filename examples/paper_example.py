#!/usr/bin/env python3
"""
Paper Example Script

This script demonstrates the core concepts of the Genetic AI algorithm using a real-world dataset.
It runs the same analysis as the paper_example.ipynb notebook, printing results and saving plots.
"""

import matplotlib.pyplot as plt
import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.plotting import PikaiaPlotter, PlotType
from pikaia.preprocessing import PikaiaPreprocessor, max_scaler
from pikaia.schemas import (
    FeatureType,
    GeneStrategyEnum,
    OrgStrategyEnum,
)
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory

print("=== Paper Example: Genetic AI Algorithm Demo ===\n")

# 2. Data Definition
print("2. Data Definition")
print("Raw data (Rent, Size, Rooms, Balcony):")
X = np.array(
    [
        [4348, 138, 3.0, 0],
        [2647, 133, 4.0, 0],
        [7413, 460, 7.0, 0],
        [5644, 329, 6.0, 0],
        [5979, 252, 6.0, 1],
        [5016, 219, 6.0, 0],
        [1106, 123, 2.0, 0],
        [4409, 175, 5.0, 0],
        [7708, 230, 8.0, 0],
        [5143, 159, 4.0, 0],
        [1650, 133, 3.0, 0],
        [7933, 383, 14.5, 1],
        [7912, 314, 7.0, 0],
        [8442, 335, 7.0, 0],
        [3218, 165, 3.0, 0],
    ],
    dtype=np.float64,
)
gene_labels = ["Rent", "Size", "Rooms", "Balcony"]
print(X)
print(f"Gene labels: {gene_labels}\n")

# 3. Data Preprocessing
print("3. Data Preprocessing")
feature_types = [
    FeatureType.COST,  # Rent
    FeatureType.GAIN,  # Size
    FeatureType.GAIN,  # Rooms
    FeatureType.GAIN,  # Balcony
]
feature_transforms = [max_scaler] * 4
preprocessor = PikaiaPreprocessor(
    num_features=X.shape[1],
    feature_types=feature_types,
    feature_transforms=feature_transforms,
)

# Create PikaiaPopulation
phi = PikaiaPopulation(preprocessor.fit_transform(X))
print("Preprocessed population matrix:")
print(phi.matrix)
print()

# 4. Model Setup and Fitting
print("4. Model Setup and Fitting")
gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)]
org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)]

# Iterative model
print("Fitting iterative model (max_iter=10)...")
model_iterative = PikaiaModel(
    population=phi,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    max_iter=10,
)
model_iterative.fit()

gene_fitness_iterative_initial = model_iterative._gene_fitness_hist[0, :]
org_fitness_iterative_initial = model_iterative._org_fitness_hist[0, :]
gene_fitness_iterative = model_iterative._gene_fitness_hist[-1, :]
org_fitness_iterative = model_iterative._org_fitness_hist[-1, :]

print(
    "Iterative model fitness values:"
    f"\nInitial Gene Fitness:\n\t{gene_fitness_iterative_initial}"
    f"\nInitial Organism Fitness:\n\t{org_fitness_iterative_initial}"
    f"\nFinal Gene Fitness:\n\t{gene_fitness_iterative}"
    f"\nFinal Organism Fitness:\n\t{org_fitness_iterative}"
)

# Analytical model
print("\nFitting analytical model...")
model_analytical = PikaiaModel(
    population=phi,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
)
model_analytical.fit()

gene_fitness_analytical_initial = model_analytical._gene_fitness_hist[0, :]
org_fitness_analytical_initial = model_analytical._org_fitness_hist[0, :]
gene_fitness_analytical = model_analytical._gene_fitness_hist[-1, :]
org_fitness_analytical = model_analytical._org_fitness_hist[-1, :]

print(
    "Analytical model fitness values:"
    f"\nInitial Gene Fitness:\n\t{gene_fitness_analytical_initial}"
    f"\nInitial Organism Fitness:\n\t{org_fitness_analytical_initial}"
    f"\nFinal Gene Fitness:\n\t{gene_fitness_analytical}"
    f"\nFinal Organism Fitness:\n\t{org_fitness_analytical}"
)

# Difference plot
print("\nSaving difference plot...")
diff = gene_fitness_analytical - gene_fitness_iterative
plt.figure(figsize=(10, 6))
plt.bar(gene_labels, diff)
plt.ylabel("Analytical - Iterative Gene Fitness")
plt.title("Difference in Gene Fitness (Analytical vs Iterative)")
plt.savefig("examples/artefacts/paper_example_difference.png", dpi=300)
print("Plot saved to examples/artefacts/paper_example_difference.png")

# 5. Results and Visualization
print("\n5. Results and Visualization")
print("Saving gene fitness history plots...")

plotter_iterative = PikaiaPlotter(model_iterative)
plotter_iterative.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=False,
    gene_labels=gene_labels,
)
plt.savefig("examples/artefacts/paper_example_iterative_fitness.png", dpi=300)
print(
    "Iterative model plot saved to examples/artefacts/paper_example_iterative_fitness.png"
)

plotter_analytical = PikaiaPlotter(model_analytical)
plotter_analytical.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=False,
    gene_labels=gene_labels,
)
plt.savefig("examples/artefacts/paper_example_analytical_fitness.png", dpi=300)
print(
    "Analytical model plot saved to examples/artefacts/paper_example_analytical_fitness.png"
)

# 7. Impact Norms
print("\n7. Impact Norms")
n = phi.N
m = phi.M
uniform = np.ones(m) / m

# Overall impact norm
col_means = phi.matrix.mean(axis=0)
impact_norm = (
    m / (m - 1) * np.sum(np.abs(gene_fitness_analytical - uniform) * col_means)
)

# Qualified impact norm
row_means = phi.matrix.mean(axis=1)
n_top = max(1, int(np.ceil(0.1 * len(row_means))))
top_indices = np.argsort(-row_means)[:n_top]
Phi_top = phi[top_indices]
Phi_top_means = Phi_top.mean(axis=0)
qualified_impact_norm = (
    m / (m - 1) * np.sum(np.abs(gene_fitness_analytical - uniform) * Phi_top_means)
)

print(f"Impact norm        ||Φ||    = {round(impact_norm, 6)}")
print(f"Qualified impact   ||Φ||_*  = {round(qualified_impact_norm, 6)}")
print(f"Column means: {col_means}")

# 8. Feature Impacts
print("\n8. Feature Impacts")
feature_ranges = phi.matrix.max(axis=0) - phi.matrix.min(axis=0)
feature_impacts = feature_ranges * gene_fitness_analytical

print("Feature impacts (ζ_j):")
for j, zeta in enumerate(feature_impacts):
    print(f"ζ_{j + 1} = {round(zeta, 6)}")

# 9. Population Ranking
print("\n9. Population Ranking")
population_initial = np.sum(phi.matrix * uniform, axis=1)
population_evolved = np.sum(phi.matrix * gene_fitness_analytical, axis=1)
idx_initial = np.argsort(population_initial)[::-1]
idx_evolved = np.argsort(population_evolved)[::-1]
complete_initial = np.zeros([n, m + 1])
complete_evolved = np.zeros([n, m + 1])
complete_initial[:, 0] = population_initial[idx_initial]
complete_evolved[:, 0] = population_evolved[idx_evolved]
for i in range(0, n):
    complete_initial[i, 1:] = X[idx_initial[i]]
    complete_evolved[i, 1:] = X[idx_evolved[i]]

print("Initial agent ranking (fitness + original features):")
print(complete_initial)
print("\nEvolved agent ranking (fitness + original features):")
print(complete_evolved)

print("\n=== Script completed. Plots saved as PNG files. ===")
