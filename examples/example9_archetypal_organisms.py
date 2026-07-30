#!/usr/bin/env python3
"""
Example 9: Archetypal Organism Detection with Pikaia-SELFISH

Demonstrates that Pikaia-SELFISH organism fitness reliably surfaces archetypal
samples (those that embody the dominant pattern in the data) when a single
strong structure exists.

Key result from the 65-experiment audit (recommender experiments):
  - Pikaia-SELFISH organism fitness outperforms MMR and Popularity baselines
    on 12/14 recommender datasets for identifying dominant/archetypal samples.

This example uses a synthetic 40×5 matrix with two groups:
  - 10 "archetype" samples that score high on all features (the dominant pattern)
  - 30 "background" samples with lower, noisier scores

Pikaia-SELFISH organism fitness is expected to rank the archetype group at the
top, since it selects organisms with high aggregate feature magnitude weighted
by gene fitness.

Comparison baselines:
  - Random ranking (expected ~25% recall of top-10 archetypes)
  - Mean-row baseline (direct row sum — equivalent to uniform gene fitness)
"""

from pathlib import Path

import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import GeneStrategyEnum, MixStrategyEnum, OrgStrategyEnum
from pikaia.strategies import (
    GeneStrategyFactory,
    MixStrategyFactory,
    OrgStrategyFactory,
)

print("=== Example 9: Archetypal Organism Detection ===\n")

# ---------------------------------------------------------------------------
# Synthetic data — one dominant cluster of archetypes
# ---------------------------------------------------------------------------
np.random.seed(7)

N_ARCHETYPES = 10
N_BACKGROUND = 30
N_FEATURES = 5

# Archetype group: consistently high scores across all features
X_arch = np.random.uniform(0.75, 1.0, (N_ARCHETYPES, N_FEATURES))

# Background group: lower, noisier scores
X_bg = np.random.uniform(0.0, 0.55, (N_BACKGROUND, N_FEATURES))

X = np.vstack([X_arch, X_bg])
# Ground truth: first N_ARCHETYPES rows are archetypes
archetype_idx = set(range(N_ARCHETYPES))

print(
    f"Organisms : {X.shape[0]}  ({N_ARCHETYPES} archetypes + {N_BACKGROUND} background)"
)
print(f"Features  : {N_FEATURES}")
print(f"Archetype mean score per feature : {X_arch.mean(axis=0).round(3)}")
print(f"Background mean score per feature: {X_bg.mean(axis=0).round(3)}\n")

# ---------------------------------------------------------------------------
# Pikaia-SELFISH organism fitness
# ---------------------------------------------------------------------------
MAX_ITER = 100
pop = PikaiaPopulation(X)
model = PikaiaModel(
    population=pop,
    gene_strategies=[GeneStrategyFactory.get_strategy(GeneStrategyEnum.SELFISH)],
    org_strategies=[OrgStrategyFactory.get_strategy(OrgStrategyEnum.SELFISH)],
    gene_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
    org_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
    max_iter=MAX_ITER,
)
model.fit()

org_fitness = model.organism_fitness_history[-1]
gene_fitness = model.gene_fitness_history[-1]

# ---------------------------------------------------------------------------
# Mean-row baseline (equivalent to Pikaia with uniform gene fitness)
# ---------------------------------------------------------------------------
mean_row_scores = X.mean(axis=1)

# ---------------------------------------------------------------------------
# Evaluation: recall of archetypes in top-k
# ---------------------------------------------------------------------------
TOP_K = N_ARCHETYPES  # retrieve exactly as many as there are archetypes


def recall_at_k(scores: np.ndarray, k: int, true_set: set) -> float:
    top_k = set(np.argsort(scores)[-k:])
    return len(top_k & true_set) / len(true_set)


pikaia_recall = recall_at_k(org_fitness, TOP_K, archetype_idx)
mean_row_recall = recall_at_k(mean_row_scores, TOP_K, archetype_idx)
random_recall = TOP_K / X.shape[0]  # expected recall for random ranking

print(f"--- Top-{TOP_K} Archetype Recall ---")
print(
    f"  Pikaia-SELFISH organism fitness : {pikaia_recall:.2f}  "
    f"({int(pikaia_recall * TOP_K)}/{TOP_K} archetypes retrieved)"
)
print(
    f"  Mean-row baseline               : {mean_row_recall:.2f}  "
    f"({int(mean_row_recall * TOP_K)}/{TOP_K} archetypes retrieved)"
)
print(f"  Random baseline (expected)      : {random_recall:.2f}")
print()

# Detailed ranking
pikaia_ranked = np.argsort(org_fitness)[::-1]
print(f"Pikaia organism ranking (top-{TOP_K}):")
print(f"  Ranks  : {pikaia_ranked[:TOP_K].tolist()}")
print(
    f"  Labels : {['ARCHETYPE' if i in archetype_idx else 'background' for i in pikaia_ranked[:TOP_K]]}"
)
print()

print("Final gene fitness (which features Pikaia weighted most):")
for j, gf in enumerate(gene_fitness):
    print(f"  feature {j}: {gf:.4f}")

print()
print("Interpretation:")
print("  Pikaia-SELFISH organism fitness acts as a weighted row-sum: samples")
print("  with high values across genes with high fitness score highest.")
print("  When a dominant pattern exists (archetypes score high on all features),")
print("  this mechanism reliably surfaces them with near-perfect recall.")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    OUT_DIR = Path("artefacts/example9")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: organism fitness scatter
    ax = axes[0]
    colours = [
        "#e74c3c" if i in archetype_idx else "#95a5a6" for i in range(X.shape[0])
    ]
    ax.scatter(range(X.shape[0]), org_fitness, c=colours, s=50, alpha=0.8)
    ax.axhline(
        np.sort(org_fitness)[-TOP_K],
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"top-{TOP_K} threshold",
    )
    ax.set_xlabel("Organism index")
    ax.set_ylabel("Organism fitness")
    ax.set_title("Pikaia-SELFISH organism fitness")
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#e74c3c",
            markersize=8,
            label="Archetype",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="#95a5a6",
            markersize=8,
            label="Background",
        ),
    ]
    ax.legend(handles=legend_elements, fontsize=9)

    # Right: recall comparison
    ax = axes[1]
    methods = ["Pikaia-SELFISH", "Mean-row", "Random"]
    recalls = [pikaia_recall, mean_row_recall, random_recall]
    bar_colours = ["#2c7bb6", "#fdae61", "#d7191c"]
    bars = ax.bar(methods, recalls, color=bar_colours, width=0.45)
    ax.bar_label(bars, fmt="%.2f", padding=4, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(f"Recall @ {TOP_K}")
    ax.set_title(f"Archetype retrieval recall\n(top-{TOP_K} of {X.shape[0]} organisms)")

    fig.suptitle("Archetypal Organism Detection — Synthetic dataset", fontsize=12)
    fig.tight_layout()
    path = OUT_DIR / "archetypal_organisms.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {path}")
except ImportError:
    print("\n(matplotlib not available — skipping plot)")

print("\n=== Example 9 completed. ===")
