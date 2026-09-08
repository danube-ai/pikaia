# Examples

This directory contains example scripts and notebooks demonstrating the capabilities of **pikaia**.

---

## 1. Quick Start

Install the `examples` extras before running any script:

```bash
uv sync --extra examples
```

---

## 2. Scripts

| File | Description |
|------|-------------|
| [`example1.py`](example1.py) | **3×3 decision problem** — Balanced vs. Altruistic gene selection on a small cost-minimisation dataset. |
| [`example2.py`](example2.py) | **10×5 dataset** — Dominant + Balanced vs. alternating selection; demonstrates convergence on more complex data. |
| [`example3.py`](example3.py) | **Self-consistency** — Runs the model multiple times with `SelfConsistentMixStrategy` to average results for stability. |
| [`example4.py`](example4.py) | **Movie search** — Real-world ranking and recommendation using a movie feature matrix (`data/movie_matrix.csv`). |
| [`example5.py`](example5.py) | **Single-point prediction** — Predicts fitness for a new unseen data point injected into the population. |
| [`example6.py`](example6.py) | **Trading strategies** — three matched sell+buy pairs (SELL_HARD+BUY_HARD, SELL_UNIFORM+BUY_UNIFORM, SELL_EASY+BUY_EASY) on a synthetic 4-gene dataset with controlled difficulty levels. |
| [`example7_stability.py`](example7_stability.py) | **Gene fitness stability** — Bootstrap Jaccard similarity comparison: Pikaia-DOM-BAL vs Mutual Information on the Wine dataset. Reproduces the K3 audit result (Pikaia Jaccard 0.950 vs MI 0.793). |
| [`example8_entropymax_fast_selection.py`](example8_entropymax_fast_selection.py) | **EntropyMax fast supervised selection** — Shows Pikaia-ENTR-BAL (supervised) converging to MI-level feature selection within 5 iterations on the Wine dataset. Reproduces the Phase 7 audit result (0.985 accuracy agreement). |
| [`example9_archetypal_organisms.py`](example9_archetypal_organisms.py) | **Archetypal organism detection** — Pikaia-SELFISH organism fitness retrieving dominant/archetypal samples from a synthetic dataset, with recall comparison against mean-row and random baselines. |
| [`paper_example.py`](paper_example.py) | Reference implementation matching the results reported in the Genetic AI preprint. |
| [`arxiv_example.py`](arxiv_example.py) | Standalone script reproducing figures from the arXiv paper. |
| [`d_matrix_comparison.py`](d_matrix_comparison.py) | **All 40 strategy combinations** (8 gene × 5 org strategies) with runtime benchmarks comparing standard iterative vs. D-matrix accelerated modes and an analytical fix-point baseline. Only D-matrix-capable strategies are benchmarked; the trading buy strategies are excluded (they require standard iterative mode). |

---

## 3. Notebooks

| File | Description |
|------|-------------|
| [`examples.ipynb`](examples.ipynb) | Interactive walkthrough of examples 1–4 with live output and plots. |
| [`paper_example.ipynb`](paper_example.ipynb) | Notebook version of the paper reference implementation. |

---

## 4. Data

| File | Description |
|------|-------------|
| [`data/movie_matrix.csv`](data/movie_matrix.csv) | Movie feature matrix used by `example4.py`. |

---

## 5. Artefacts

The `artefacts/` directory is used as the default output location for generated plots and saved figures.

---

## 6. Strategy Combinations Benchmark

`d_matrix_comparison.py` is the most comprehensive example. It benchmarks every combination of these two independent lists (8 gene × 5 org = 40 combinations):

- **Gene strategies:** `DominantGeneStrategy`, `AltruisticGeneStrategy`, `SelfishGeneStrategy`, `KinAltruisticGeneStrategy`, `SellHardGeneStrategy`, `SellUniformGeneStrategy`, `SellEasyGeneStrategy`, `NoneGeneStrategy`
- **Org strategies:** `BalancedOrgStrategy`, `AltruisticOrgStrategy`, `KinSelfishOrgStrategy`, `SelfishOrgStrategy`, `NoneOrgStrategy`

The grid deliberately mixes gene and org strategies freely to benchmark the D-matrix mechanism — the rows above are **not** matched pairs. Only strategies that implement `kernel()` are included, so the trading buy strategies (`BuyHardOrgStrategy`, `BuyUniformOrgStrategy`, `BuyEasyOrgStrategy`) are excluded; they run only under the standard iterative loop. To see the full trading pairs (each sell strategy with its matching buy strategy), run [`example6.py`](example6.py).

Three fit modes are compared for each valid combination:

1. **Analytical fix-point** (`use_d_matrix=False, max_iter=None`) — instant, Dominant + Balanced only.
2. **Standard iterative** (`use_d_matrix=False, max_iter=500`) — general-purpose, O(N·M²) per step.
3. **D-matrix iterative** (`use_d_matrix=True, max_iter=500`) — O(M²) per step, typically **30–80× faster**.

Run it with:

```bash
uv run python examples/d_matrix_comparison.py
```
