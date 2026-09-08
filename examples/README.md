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
| [`d_matrix_comparison.py`](d_matrix_comparison.py) | Compares every supported D-matrix configuration with the corresponding iterative path at 1, 50, and 100 iterations. |

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

## 6. D-matrix Equivalence Comparison

`d_matrix_comparison.py` evaluates the configurations for which the package has
an exact reduced implementation:

- `ORIGINAL`: each supported gene strategy paired with `NoneOrgStrategy`, which
  isolates its contribution.
- `MATH_PAPER`: dominant gene paired with `NoneOrgStrategy`, plus the supported
  altruistic-gene and selfish-organism Alt-Sel configuration.

For every row, it independently runs the ordinary and D-matrix paths at 1, 50,
and 100 iterations and reports the largest absolute difference between their
final gene-fitness vectors. It also reports median runtimes over seven complete
100-iteration fits. Unsupported combinations are not benchmarked as if they
had a valid D-matrix implementation; requesting one in the package raises
`ValueError`.

Run it with:

```bash
uv run python examples/d_matrix_comparison.py
```
