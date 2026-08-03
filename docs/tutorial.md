# Tutorial: Your first pikaia analysis

This tutorial walks you through a complete pikaia analysis from raw data to ranked results. By the end you will have run an evolutionary simulation, interpreted the gene-fitness output, and produced a plot.

**Prerequisites:** pikaia installed (`pip install pikaia`), basic Python familiarity.

---

## The scenario

You have data about 5 candidates evaluated on 4 criteria:

| Candidate | Speed | Accuracy | Cost efficiency | Reliability |
|-----------|-------|----------|-----------------|-------------|
| A         | 300   | 0.91     | 80              | 0.95        |
| B         | 600   | 0.76     | 55              | 0.88        |
| C         | 150   | 0.95     | 90              | 0.97        |
| D         | 900   | 0.60     | 40              | 0.72        |
| E         | 450   | 0.83     | 70              | 0.84        |

You want to know: **which criteria actually drive differentiation between candidates?**

---

## Step 1 — Prepare your data

pikaia expects values in **[0, 1]** where higher means better. Scale each column using min-max normalisation.

```python
import numpy as np
from pikaia.preprocessing import PikaiaPreprocessor, min_max_scaler
from pikaia.schemas import FeatureType

raw = np.array([
    [300,  0.91, 80, 0.95],
    [600,  0.76, 55, 0.88],
    [150,  0.95, 90, 0.97],
    [900,  0.60, 40, 0.72],
    [450,  0.83, 70, 0.84],
])

preprocessor = PikaiaPreprocessor(
    num_features=4,
    feature_types=[FeatureType.GAIN] * 4,   # higher is better for all columns
    feature_transforms=[min_max_scaler] * 4,
)
data = preprocessor.fit_transform(raw)
print(data)
```

The result is a 5×4 matrix with all values in [0, 1].

---

## Step 2 — Create a population

```python
from pikaia.data import PikaiaPopulation

population = PikaiaPopulation(data)
print(f"Organisms (candidates): {population.N}")
print(f"Genes (criteria):       {population.M}")
```

In pikaia's language, each **row is an organism** (candidate) and each **column is a gene** (criterion).

---

## Step 3 — Choose strategies

Strategies control how the evolutionary simulation evolves gene fitness. Start with the most common combination:

```python
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum, MixStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory, MixStrategyFactory

gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)]
org_strategies  = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)]
mix_strategy    = MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED)
```

- **DOMINANT** rewards genes that are highly expressed across the population.
- **BALANCED** keeps organisms from being purely selfish or purely altruistic.

See the [overview](overview.md) for a conceptual explanation of what strategies do, and the [reference](reference.md) for the full list.

---

## Step 4 — Fit the model

```python
from pikaia.models import PikaiaModel

model = PikaiaModel(
    population=population,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    gene_mix_strategy=mix_strategy,
    org_mix_strategy=mix_strategy,
    max_iter=32,
)
model.fit()
```

`max_iter=32` runs 32 evolutionary iterations. For most datasets 16–64 iterations is sufficient to reach a stable ranking.

---

## Step 5 — Read the results

Gene fitness converges to a vector that sums to 1. Higher values mean that criterion drove more differentiation.

```python
final_fitness = model.gene_fitness_history[-1]
gene_labels = ["Speed", "Accuracy", "Cost efficiency", "Reliability"]

for label, fitness in zip(gene_labels, final_fitness):
    print(f"  {label:20s}: {fitness:.4f}")
```

Example output:
```
  Speed               : 0.2766
  Accuracy            : 0.2368
  Cost efficiency     : 0.2517
  Reliability         : 0.2349
```

Speed emerged as the most differentiating criterion here — it spans the widest range across candidates (150 → 900). The other three criteria are closer together, reflecting the more uniform spread of Accuracy, Cost efficiency, and Reliability in this dataset.

---

## Step 6 — Plot the fitness trajectory

```python
from pikaia.plotting import PikaiaPlotter, PlotType

plotter = PikaiaPlotter(model)
plotter.plot(
    plot_type=PlotType.GENE_FITNESS_HISTORY,
    gene_labels=gene_labels,
    title="Criterion fitness over iterations",
    show=True,
)
```

The plot shows how each gene's fitness evolves. A steep early trajectory means the criterion differentiates candidates strongly.

---

## What's next

- **Compare strategies** — try `GeneStrategyEnum.REWARD_HARD` to favour criteria that are rare across candidates, or `GeneStrategyEnum.REWARD_EASY` for the opposite. See `examples/example6.py`.
- **Speed up large datasets** — pass `use_d_matrix=True` to `PikaiaModel` for a 30–80× speedup. See [overview](overview.md#d-matrix-accelerated-mode).
- **Mix multiple strategies** — pass a list to `gene_strategies` and use `MixStrategyEnum.SELF_CONSISTENT` to let the model self-select weights.
- **Explore more examples** — the `examples/` directory contains scripts for real-world movie ranking, self-consistency, and a full strategy comparison grid.
- **Extend pikaia** — add your own strategy by following the [contributor guide](contributing.md).
