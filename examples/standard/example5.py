#!/usr/bin/env python3
"""
Example 5: 3x3 Data - Single Point Prediction

This script demonstrates predicting fitness for a new data point using the 3x3 dataset.
It prints results and saves plots for visualization.
"""

from pathlib import Path

import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.plotting import PikaiaPlotter, PlotType
from pikaia.preprocessing import PikaiaPreprocessor, min_max_scaler
from pikaia.schemas import FeatureType

print("=== Example 5: 3x3 Data - Single Point Prediction ===\n")

# Data Processing
print("Data Processing")
data_3x3_raw = np.array(
    [
        [300, 10, 2],
        [600, 5, 2],
        [1500, 4, 1],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
    ]
)
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

population_5 = population3x3
print("Number of organisms:", population_5.N)
print("Number of genes:", population_5.M)
print()

# Model Setup and Fitting
print("Model Setup and Fitting")
# Using default strategies
model = PikaiaModel(
    population=population_5,
)
print("Fitting model with default strategies...")
model.fit()

print("Gene fitness history shape:", model.gene_fitness_history.shape)
print("Initial gene fitness:", model.gene_fitness_history[0])
print("Final gene fitness:", model.gene_fitness_history[-1])
print("Org fitness history shape:", model.organism_fitness_history.shape)
print("Initial org fitness:", model.organism_fitness_history[0])
print("Final org fitness:", model.organism_fitness_history[-1])

# Single point prediction
new_point = np.array([[400, 7, 1.5]])
new_point_scaled = preprocessor.transform(new_point)
predicted_fitness = np.sum(new_point_scaled * model.gene_fitness_history[-1])
print(f"Predicted fitness for new point {new_point[0]}: {predicted_fitness}")
print()

# Plotting Results
print("Plotting Results")
print("Saving plots...")

plotter = PikaiaPlotter(model)
plotter.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=False,
    save_path=Path("artefacts/example5_gene_fitness.png"),
)
print("Gene fitness history plot saved to artefacts/example5_gene_fitness.png")

plotter.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=False,
    save_path=Path("artefacts/example5_org_fitness.png"),
    org_labels=org_labels,
)
print("Org fitness history plot saved to artefacts/example5_org_fitness.png")

print("\n=== Example 5 completed. Plots saved as PNG files. ===")
