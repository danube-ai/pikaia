# pikaia: Genetic AI Documentation

Welcome to the documentation for **pikaia**, a Python package for data analysis using evolutionary simulation (Genetic AI). This guide will help you understand what pikaia is, how to get started, and how to use its main features.

---

## What is pikaia?

**pikaia** is a library for analyzing data using evolutionary simulation. It models your data as a population of organisms (rows) and genes (columns), then applies evolutionary strategies to uncover patterns, rank features, and solve decision problems. Unlike traditional machine learning, pikaia does not require training data or labels—it autonomously explores your data using strategies inspired by genetics and game theory.

- **Key features:**
  - No training data required
  - Flexible evolutionary strategies (dominant, altruistic, selfish, etc.)
  - Works on any tabular data (normalized to [0, 1])
  - Visualizes gene and organism fitness over time
  - Extensible and modular design

For the scientific background, see the [Genetic AI preprint](http://arxiv.org/abs/2501.19113).

---

## How to Get Started

### Installation

Install pikaia from PyPI:

```bash
pip install pikaia
```

### Basic Example

Here's a minimal example to get you started:

```python
import numpy as np
from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.plotting import PikaiaPlotter, PlotType
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum, MixStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory, MixStrategyFactory

# Prepare your data (rows: organisms, columns: genes, values in [0, 1])
data = np.array([[0.1, 0.5, 0.9], [0.2, 0.3, 0.7], [0.8, 0.2, 0.4]])
population = PikaiaPopulation(data)

# Choose strategies
gene_strategies = [GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)]
org_strategies = [OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)]
gene_mix_strategy = org_mix_strategy = MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED)

# Create and fit the model
model = PikaiaModel(
    population=population,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    gene_mix_strategy=gene_mix_strategy,
    org_mix_strategy=org_mix_strategy,
    max_iter=16,
)
model.fit()

# Plot results
plotter = PikaiaPlotter(model)
plotter.plot(plot_type=PlotType.GENE_FITNESS_HISTORY, show=True)
```

---

## Main Concepts & Classes

- **PikaiaPopulation**: Wraps your data matrix. Each row is an organism, each column is a gene/feature. Data must be normalized to [0, 1].
- **PikaiaModel**: The core evolutionary simulation. Takes a population and strategies, runs the simulation, and stores fitness histories.
- **Strategies**: Control how genes and organisms evolve. Choose from Dominant, Altruistic, Selfish, Balanced, etc. (see `pikaia.schemas.strategies`).
- **PikaiaPlotter**: Visualizes fitness histories and other results.

---

## Examples

Extensive examples are provided in [`examples/examples.ipynb`](../examples/examples.ipynb):

### 1. Small Decision Problem (3x3)

- Compares different selection strategies (Balanced vs. Altruistic)
- Shows how to preprocess data, set up the model, and plot results

### 2. Larger Problem (10x5)

- Demonstrates performance on more complex data
- Shows how strategies affect convergence and fitness

### 3. Self-Consistency

- Runs the model multiple times to average results for stability

### 4. Real-World Search (Movie Data)

- Uses a real dataset to rank movies based on multiple criteria
- Demonstrates how to use pikaia for search and recommendation

See the notebook for code and output.

---

## How to Use pikaia for Your Data

1. **Prepare your data**: Organize your data as a 2D NumPy array (organisms × genes), and scale all values to [0, 1].
2. **Create a PikaiaPopulation**: `population = PikaiaPopulation(data)`
3. **Choose strategies**: Use the factories and enums to select gene and organism strategies.
4. **Create and fit a PikaiaModel**: Pass your population and strategies, set `max_iter` for the number of evolutionary steps.
5. **Analyze results**: Use `PikaiaPlotter` to visualize gene/organism fitness, mixing coefficients, and similarities.

---

## Advanced Usage

- **Custom strategies**: Implement your own by subclassing `GeneStrategy` or `OrgStrategy`.
- **Mixing strategies**: Control how multiple strategies are combined (e.g., fixed weights, self-consistent mixing).
- **Similarity analysis**: Explore gene and organism similarity matrices to understand relationships in your data.

---

## Where to Find More

- **Examples**: See [`examples/examples.ipynb`](../examples/examples.ipynb) for detailed walkthroughs.
- **API Reference**: Auto-generated from the codebase (see left sidebar or [API docs](autoapi/index.html) if enabled).
- **Source code**: [GitHub repository](https://github.com/danube-ai/pikaia)
- **Issues & Questions**: [GitHub Issues](https://github.com/danube-ai/pikaia/issues)

---

## License

MIT License. See [LICENSE](../LICENSE) for details.
