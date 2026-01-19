# import matplotlib.pyplot as plt
# import matplotlib
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

# setup numeric printout format
float_formatter = "{:.4f}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

# set this to the directory where the images should be stored
relpath = "artefacts/arxiv_example/"
Path(relpath).mkdir(parents=True, exist_ok=True)

# Data for 3x3 example
data_3x3_raw = np.array([[300, 10, 2], [600, 5, 2], [1500, 4, 1]])
feature_types_3x3 = [FeatureType.COST, FeatureType.COST, FeatureType.COST]
feature_transforms_3x3 = [min_max_scaler] * data_3x3_raw.shape[1]
preprocessor_3x3 = PikaiaPreprocessor(
    num_features=data_3x3_raw.shape[1],
    feature_types=feature_types_3x3,
    feature_transforms=feature_transforms_3x3,
)
data_3x3_scaled = preprocessor_3x3.fit_transform(data_3x3_raw)
population3x3 = PikaiaPopulation(data_3x3_scaled)
initialgenefitness3x3 = np.ones(population3x3.M) / population3x3.M

# names for output images
filename3x3gene = relpath + "Smallexample_3x3_genefitness"
filename3x3orgs = relpath + "Smallexample_3x3_orgfitness"

# Data for 10x5 example
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
feature_types_10x5 = [FeatureType.COST] * data_10x5_raw.shape[1]
feature_transforms_10x5 = [min_max_scaler] * data_10x5_raw.shape[1]
preprocessor_10x5 = PikaiaPreprocessor(
    num_features=data_10x5_raw.shape[1],
    feature_types=feature_types_10x5,
    feature_transforms=feature_transforms_10x5,
)
data_10x5_scaled = preprocessor_10x5.fit_transform(data_10x5_raw)
population10x5 = PikaiaPopulation(data_10x5_scaled)
initialgenefitness10x5 = np.ones(population10x5.M) / population10x5.M

# names for output images
filename10x5gene = relpath + "Realworldexample_10x5_genefitness"
filename10x5orgs = relpath + "Realworldexample_10x5_orgfitness"

iterations3x3 = 30
iterations10x5 = 60
# Using epsilon allows to stop the simulation at ESE
epsilon = None  # 0.00005

# define used gene and organism strategies
gene_strategy_dom = GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)
org_strategy_bal = OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)
gene_strategy_alt = GeneStrategyFactory.get_strategy(
    GeneStrategyEnum.KIN_ALTRUISTIC, kin_range=10
)
org_strategy_sel = OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELFISH)
mix_strategy = MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED)

# Labels
gene_labels_3x3 = ["gene 1 = price", "gene 2 = time", "gene 3 = stops"]
org_labels_3x3 = [f"flight {i}" for i in range(population3x3.N)]
gene_labels_10x5 = [f"gene {i + 1}" for i in range(population10x5.M)]
org_labels_10x5 = [f"flight {i}" for i in range(population10x5.N)]

# create models and converge them for simple example 3x3
dombal_small = PikaiaModel(
    population=population3x3,
    gene_strategies=[gene_strategy_dom],
    org_strategies=[org_strategy_bal],
    gene_mix_strategy=mix_strategy,
    org_mix_strategy=mix_strategy,
    initial_gene_fitness=initialgenefitness3x3,
    max_iter=iterations3x3,
    epsilon=epsilon,
)
dombal_small.fit()

altsel_small = PikaiaModel(
    population=population3x3,
    gene_strategies=[gene_strategy_alt],
    org_strategies=[org_strategy_sel],
    gene_mix_strategy=mix_strategy,
    org_mix_strategy=mix_strategy,
    initial_gene_fitness=initialgenefitness3x3,
    max_iter=iterations3x3,
    epsilon=epsilon,
)
altsel_small.fit()

# TODO: Update plotting to match old API - plot multiple models in one figure
plotter_dombal_small = PikaiaPlotter(dombal_small)
plotter_dombal_small.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename3x3gene + "_dombal.png"),
    gene_labels=gene_labels_3x3,
)
plotter_altsel_small = PikaiaPlotter(altsel_small)
plotter_altsel_small.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename3x3gene + "_altsel.png"),
    gene_labels=gene_labels_3x3,
)
# Old: pikaia.plot.plot_gene_fitness([dombal_small, altsel_small], 1, show=True, savename=filename3x3gene)

# TODO: Update plotting for organism fitness
plotter_dombal_small.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename3x3orgs + "_dombal.png"),
    org_labels=org_labels_3x3,
)
plotter_altsel_small.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename3x3orgs + "_altsel.png"),
    org_labels=org_labels_3x3,
)
# Old: pikaia.plot.plot_organism_fitness([dombal_small, altsel_small], 2, None, show=True, savename=filename3x3orgs)

# create models and converge them for real-world example 10x5
dombal_large = PikaiaModel(
    population=population10x5,
    gene_strategies=[gene_strategy_dom],
    org_strategies=[org_strategy_bal],
    gene_mix_strategy=mix_strategy,
    org_mix_strategy=mix_strategy,
    initial_gene_fitness=initialgenefitness10x5,
    max_iter=iterations10x5,
    epsilon=epsilon,
)
dombal_large.fit()

# import pdb; pdb.set_trace()
altsel_large = PikaiaModel(
    population=population10x5,
    gene_strategies=[gene_strategy_alt],
    org_strategies=[org_strategy_sel],
    gene_mix_strategy=mix_strategy,
    org_mix_strategy=mix_strategy,
    initial_gene_fitness=initialgenefitness10x5,
    max_iter=iterations10x5,
    epsilon=epsilon,
)
altsel_large.fit()

# TODO: Update plotting to match old API
plotter_dombal_large = PikaiaPlotter(dombal_large)
plotter_dombal_large.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename10x5gene + "_dombal.png"),
    gene_labels=gene_labels_10x5,
)
plotter_altsel_large = PikaiaPlotter(altsel_large)
plotter_altsel_large.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename10x5gene + "_altsel.png"),
    gene_labels=gene_labels_10x5,
)
# Old: pikaia.plot.plot_gene_fitness([dombal_large, altsel_large], 1, show=True, savename=filename10x5gene)

maxitershown = 64
# TODO: Update plotting for organism fitness with maxitershown
plotter_dombal_large.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename10x5orgs + "_dombal.png"),
    org_labels=org_labels_10x5,
)
plotter_altsel_large.plot(
    plot_type=PlotType.ORGANISM_FITNESS_HISTORY,
    show=True,
    save_path=Path(filename10x5orgs + "_altsel.png"),
    org_labels=org_labels_10x5,
)
# Old: pikaia.plot.plot_organism_fitness([dombal_large], 2, maxitershown, show=True, savename=filename10x5orgs+"_dombal")
# Old: pikaia.plot.plot_organism_fitness([altsel_large], 2, None, show=True, savename=filename10x5orgs+"_altsel")
