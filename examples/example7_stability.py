#!/usr/bin/env python3
"""
Example 7: Gene Fitness Bootstrap Stability

Demonstrates that Pikaia-DOM-BAL gene fitness rankings are highly reproducible
under bootstrap resampling, outperforming Mutual Information on this metric.

Key result from the 65-experiment audit (K3):
  - Pikaia stability: Jaccard 0.950 vs MI stability: 0.793 (8/10 datasets)

This example reproduces the comparison on the Wine dataset (13 features, 3 classes):
  1. Draw N_BOOTSTRAP bootstrap samples from the data.
  2. Compute gene fitness (Pikaia-DOM-BAL) and MI scores on each sample.
  3. Select the top-k features per sample.
  4. Compute pairwise Jaccard similarity of the top-k sets across bootstrap draws.
  5. Report mean Jaccard for both methods.

Note: gene fitness stability is a consequence of the MinMax row-sum mechanism,
not a signal of predictive relevance.  It measures reproducibility, not accuracy.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import GeneStrategyEnum, MixStrategyEnum, OrgStrategyEnum
from pikaia.strategies import (
    GeneStrategyFactory,
    MixStrategyFactory,
    OrgStrategyFactory,
)

print("=== Example 7: Gene Fitness Bootstrap Stability ===\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = load_wine()
X_raw, y = data.data, data.target
feature_names = data.feature_names

scaler = MinMaxScaler()
X = np.clip(scaler.fit_transform(X_raw), 0.0, 1.0)

N_SAMPLES, N_FEATURES = X.shape
print(
    f"Dataset: Wine — {N_SAMPLES} samples, {N_FEATURES} features, {len(np.unique(y))} classes\n"
)

# ---------------------------------------------------------------------------
# Bootstrap stability parameters
# ---------------------------------------------------------------------------
N_BOOTSTRAP = 20
TOP_K = N_FEATURES // 2  # top-half features
MAX_ITER = 100
RNG = np.random.RandomState(42)

print(f"Bootstrap draws : {N_BOOTSTRAP}")
print(f"Top-k features  : {TOP_K}")
print(f"Iterations      : {MAX_ITER}\n")


def compute_gene_fitness(X_boot: np.ndarray) -> np.ndarray:
    pop = PikaiaPopulation(X_boot)
    model = PikaiaModel(
        population=pop,
        gene_strategies=[GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)],
        org_strategies=[OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)],
        gene_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
        org_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
        max_iter=MAX_ITER,
    )
    model.fit()
    return model.gene_fitness_history[-1]


def pairwise_jaccard(topk_sets: list[set]) -> float:
    """Mean Jaccard similarity over all pairs of top-k sets."""
    scores = []
    n = len(topk_sets)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = topk_sets[i], topk_sets[j]
            scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# Run bootstrap
# ---------------------------------------------------------------------------
print("Running bootstrap draws...")
pikaia_topk_sets: list[set] = []
mi_topk_sets: list[set] = []

for draw in range(N_BOOTSTRAP):
    idx = RNG.choice(N_SAMPLES, size=N_SAMPLES, replace=True)
    X_boot = X[idx]
    y_boot = y[idx]

    # Pikaia-DOM-BAL gene fitness
    gf = compute_gene_fitness(X_boot)
    pikaia_topk_sets.append(set(np.argsort(gf)[-TOP_K:]))

    # Mutual Information (supervised baseline)
    mi = mutual_info_classif(X_boot, y_boot, random_state=42)
    mi_topk_sets.append(set(np.argsort(mi)[-TOP_K:]))

    if (draw + 1) % 5 == 0:
        print(f"  Bootstrap draw {draw + 1}/{N_BOOTSTRAP} done")

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
pikaia_jaccard = pairwise_jaccard(pikaia_topk_sets)
mi_jaccard = pairwise_jaccard(mi_topk_sets)

print("\n--- Bootstrap Stability Results (Wine dataset) ---")
print(f"  Pikaia-DOM-BAL  Jaccard: {pikaia_jaccard:.3f}")
print(f"  Mutual Info     Jaccard: {mi_jaccard:.3f}")
winner = "Pikaia" if pikaia_jaccard >= mi_jaccard else "MI"
print(f"  More stable: {winner}  (margin: {abs(pikaia_jaccard - mi_jaccard):.3f})")
print()
print("Interpretation:")
print("  Higher Jaccard = more reproducible top-k selections across bootstrap draws.")
print("  Pikaia stability arises from the MinMax row-sum mechanism (not predictive")
print("  relevance): once features are normalised to [0,1], the replicator dynamics")
print("  consistently amplify the same skewness-driven ordering.")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    OUT_DIR = Path("artefacts/example7")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ["Pikaia-DOM-BAL", "Mutual Info"]
    values = [pikaia_jaccard, mi_jaccard]
    colours = ["#2c7bb6", "#d7191c"]
    bars = ax.bar(methods, values, color=colours, width=0.4)
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean pairwise Jaccard similarity")
    ax.set_title(
        f"Gene fitness bootstrap stability — Wine dataset\n"
        f"({N_BOOTSTRAP} draws, top-{TOP_K} of {N_FEATURES} features)"
    )
    ax.axhline(0.9, color="grey", linestyle="--", linewidth=0.8, label="0.9 threshold")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = OUT_DIR / "stability_bar.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {path}")
except ImportError:
    print("\n(matplotlib not available — skipping plot)")

print("\n=== Example 7 completed. ===")
