
pikaia: Genetic AI Documentation
===============================

Welcome to the documentation for **pikaia**, a Python package for data analysis using evolutionary simulation (Genetic AI).

**pikaia** enables you to analyze tabular data using evolutionary strategies inspired by genetics and game theory. It is designed for researchers, data scientists, and engineers who want to explore, rank, and understand data without the need for supervised learning or labeled datasets.

**Key Features:**

- No training data required—fully unsupervised
- Flexible evolutionary strategies (dominant, altruistic, selfish, balanced, etc.)
- Works on any tabular data (normalized to [0, 1])
- Visualizes gene and organism fitness over time
- Extensible and modular design

For a detailed introduction, see the :doc:`overview <overview>`.

**Quick Start:**

.. code-block:: python

   import numpy as np
   from pikaia.data import PikaiaPopulation
   from pikaia.models import PikaiaModel
   from pikaia.plotting import PikaiaPlotter, PlotType
   from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum, MixStrategyEnum
   from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory, MixStrategyFactory

   data = np.array([[0.1, 0.5, 0.9], [0.2, 0.3, 0.7], [0.8, 0.2, 0.4]])
   population = PikaiaPopulation(data)
   gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)]
   org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)]
   gene_mix_strategy = org_mix_strategy = MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED)
   model = PikaiaModel(
       population=population,
       gene_strategies=gene_strategies,
       org_strategies=org_strategies,
       gene_mix_strategy=gene_mix_strategy,
       org_mix_strategy=org_mix_strategy,
       max_iter=16,
   )
   model.fit()
   plotter = PikaiaPlotter(model)
   plotter.plot(plot_type=PlotType.GENE_FITNESS_HISTORY, show=True)

**Navigation:**

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview

