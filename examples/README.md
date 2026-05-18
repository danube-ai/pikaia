# Examples

This directory contains example scripts and notebooks demonstrating the capabilities of **pikaia**.

---

## Quick Start

Install the `examples` extras before running any script:

```bash
uv sync --extra examples
```

---

## Scripts

| File | Description |
|------|-------------|
| [`example1.py`](example1.py) | **3×3 decision problem** — Balanced vs. Altruistic gene selection on a small cost-minimisation dataset. |
| [`example2.py`](example2.py) | **10×5 dataset** — Dominant + Balanced vs. alternating selection; demonstrates convergence on more complex data. |
| [`example3.py`](example3.py) | **Self-consistency** — Runs the model multiple times with `SelfConsistentMixStrategy` to average results for stability. |
| [`example4.py`](example4.py) | **Movie search** — Real-world ranking and recommendation using a movie feature matrix (`data/movie_matrix.csv`). |
| [`example5.py`](example5.py) | **Single-point prediction** — Predicts fitness for a new unseen data point injected into the population. |
| [`paper_example.py`](paper_example.py) | Reference implementation matching the results reported in the Genetic AI preprint. |
| [`arxiv_example.py`](arxiv_example.py) | Standalone script reproducing figures from the arXiv paper. |
| [`d_matrix_comparison.py`](d_matrix_comparison.py) | **All 25 strategy combinations** (5 gene × 5 org strategies) with runtime benchmarks comparing standard iterative vs. D-matrix accelerated modes and an analytical fix-point baseline. |

---

## Notebooks

| File | Description |
|------|-------------|
| [`examples.ipynb`](examples.ipynb) | Interactive walkthrough of examples 1–4 with live output and plots. |
| [`paper_example.ipynb`](paper_example.ipynb) | Notebook version of the paper reference implementation. |

---

## Data

| File | Description |
|------|-------------|
| [`data/movie_matrix.csv`](data/movie_matrix.csv) | Movie feature matrix used by `example4.py`. |

---

## Artefacts

The `artefacts/` directory is used as the default output location for generated plots and saved figures.

---

## Strategy Combinations Benchmark

`d_matrix_comparison.py` is the most comprehensive example. It covers all 25 combinations of:

| Gene strategies | Org strategies |
|-----------------|---------------|
| `DominantGeneStrategy` | `BalancedOrgStrategy` |
| `AltruisticGeneStrategy` | `AltruisticOrgStrategy` |
| `SelfishGeneStrategy` | `SelfishOrgStrategy` |
| `KinAltruisticGeneStrategy` | `KinSelfishOrgStrategy` |
| `NoneGeneStrategy` | `NoneOrgStrategy` |

Three fit modes are compared for each valid combination:

1. **Analytical fix-point** (`use_d_matrix=False, max_iter=None`) — instant, Dominant + Balanced only.
2. **Standard iterative** (`use_d_matrix=False, max_iter=500`) — general-purpose, O(N·M²) per step.
3. **D-matrix iterative** (`use_d_matrix=True, max_iter=500`) — O(M²) per step, typically **30–80× faster**.

Run it with:

```bash
uv run python examples/d_matrix_comparison.py
```
