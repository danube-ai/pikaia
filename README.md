<p align="center">
    <img src="https://img.shields.io/pypi/v/pikaia?color=blue" alt="PyPI version">
    <img src="https://img.shields.io/pypi/pyversions/pikaia" alt="Python version">
    <img src="https://img.shields.io/github/license/danube-ai/pikaia" alt="License">
    <img src="https://img.shields.io/github/issues/danube-ai/pikaia" alt="GitHub issues">
</p>

# 🧬 Pikaia

Welcome to **Pikaia** — a Python package for evolutionary algorithms, genetic programming, and AI-driven optimization. This package is designed for researchers, students, and practitioners interested in evolutionary computation and data analysis.

---

## ✨ Key Features

- 🧬 Evolutionary simulation for data analysis
- 📊 Built-in plotting and visualization
- 🧩 Modular, extensible strategy system
- 📝 Jupyter notebook examples included
- 🔬 Scientific approach, ready for research and teaching

---

---

## 📚 Table of Contents

- [🧬 Pikaia](#-pikaia)
  - [✨ Key Features](#-key-features)
  - [📚 Table of Contents](#-table-of-contents)
  - [🚀 Installation](#-installation)
  - [🛠️ Local Development](#️-local-development)
    - [Prerequisites](#prerequisites)
    - [Install UV](#install-uv)
    - [Set up a Local Environment](#set-up-a-local-environment)
  - [📝 Quickstart](#-quickstart)
  - [🧬 Scientific Background](#-scientific-background)
  - [👥 Authors \& Contact](#-authors--contact)
  - [📄 License](#-license)
  - [📚 How to Cite](#-how-to-cite)

---

## 🚀 Installation

Install the package using pip:

```bash
pip install pikaia
```

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 🛠️ Local Development

For local development, we recommend using [UV](https://astral.sh/uv), a fast Python package installer and resolver.

### Prerequisites

Clone the repository and navigate to the project directory:

```bash
git clone https://github.com/danube-ai/pikaia.git
cd pikaia
```

### Install UV

Install UV using the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For more installation options, visit the [UV installation guide](https://astral.sh/uv/installation).

### Set up a Local Environment

1. Create a virtual environment:

   ```bash
   uv venv
   ```

2. Sync the dependencies (including development and notebook extras):

   ```bash
   uv sync --extra dev --extra notebooks
   ```

3. Activate the virtual environment:

   ```bash
   source .venv/bin/activate
   ```

   This installs the package in editable mode along with tools for development (e.g., testing with pytest, linting with ruff) and Jupyter notebooks. The `uv sync` command ensures reproducible installations using the locked dependencies in `uv.lock`.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 📝 Quickstart

Here's a minimal example to get you started:

```python
import numpy as np
from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import GeneStrategyEnum, OrgStrategyEnum, MixStrategyEnum
from pikaia.strategies import GeneStrategyFactory, OrgStrategyFactory, MixStrategyFactory

# Prepare a small dataset (3 samples, 3 features)
data_3x3_raw = np.array([[300, 10, 2], [600, 5, 2], [1500, 4, 1]])
data_min = data_3x3_raw.min(axis=0)
data_max = data_3x3_raw.max(axis=0)
data_3x3_scaled = (data_3x3_raw - data_min) / (data_max - data_min)
population = PikaiaPopulation(data_3x3_scaled)

# Define strategies
gene_strategies = [
    GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT),
    GeneStrategyFactory.get_strategy(GeneStrategyEnum.ALTRUISTIC),
]
org_strategies = [
    OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED),
    OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELFISH),
]
gene_mix_strategy = org_mix_strategy = MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED)

# Create and fit the model
model = PikaiaModel(
    population=population,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    gene_mix_strategy=gene_mix_strategy,
    org_mix_strategy=org_mix_strategy,
    max_iter=32,
)
model.fit()

print("Gene fitness history:", model.gene_fitness_history())
```

- Explore the `examples/` directory for Jupyter notebooks, Python scripts, and data files.
- See `examples/examples.ipynb` for a hands-on walkthrough or run individual example scripts like `python examples/example1.py`.
- See `examples/paper_example.py` for the paper example script.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 🧬 Scientific Background

Genetic AI is a framework for evolutionary simulation and data analysis. In Genetic AI, a data problem is converted into a model of genes and organisms, and evolutionary simulations are run to gain insight into the input data.

- Genetic AI does **not** use training data to 'learn', but instead autonomously analyzes a problem using evolutionary strategies that capture behaviors and correlations in the data.
- This approach is useful for understanding complex datasets, optimization, and exploring emergent properties in data-driven systems.

**Preprint:** [Genetic AI (arXiv)](http://arxiv.org/abs/2501.19113)

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 👥 Authors & Contact

- Philipp Wissgott (<philipp@danube.ai>)
- Andreas Roschal (<andreas@danube.ai>)
- Martin Bär (<martin@danube.ai>)
- Carlos U. Pérez Malla (<carlos@danube.ai>)

For questions, suggestions, or contributions, please feel free to open an issue.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 📄 License

This project is licensed under the terms of the MIT License. See the [LICENSE](../LICENSE) file for details.

<p align="right">(<a href="#top">back to top</a>)</p>

---

## 📚 How to Cite

If you use Pikaia in your research, please cite our preprint:

```bibtex
@misc{wissgott2025geneticaievolutionarygames,
            title={Genetic AI: Evolutionary Games for ab initio dynamic Multi-Objective Optimization},
            author={Philipp Wissgott},
            year={2025},
            eprint={2501.19113},
            archivePrefix={arXiv},
            primaryClass={cs.NE},
            url={https://arxiv.org/abs/2501.19113},
}
```

**Preprint:** [Genetic AI (arXiv)](https://arxiv.org/abs/2501.19113)

<p align="right">(<a href="#top">back to top</a>)</p>
