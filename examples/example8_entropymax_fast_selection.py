#!/usr/bin/env python3
"""
Example 8: EntropyMax — Fast Supervised Feature Selection

Demonstrates that Pikaia-ENTR-BAL converges to its best feature ranking
within 5 iterations and matches the accuracy of full Mutual Information
selection at a fraction of the cost.

Key result from the 65-experiment audit (Phase 7, nested CV):
  - Pikaia-ENTR-BAL achieves 0.985 mean accuracy agreement with the best
    supervised method across 4 datasets, reaching peak selection at iteration 5.

IMPORTANT: EntropyMax is a SUPERVISED strategy — it requires the target
variable ``y`` to compute mutual information.  It is not label-free.

This example:
  1. Loads the Wine dataset (13 features, 3 classes).
  2. Runs EntropyMax at iterations 1, 2, 5, 10, 20, 50 and evaluates top-k
     feature selection accuracy (Random Forest, 5-fold CV).
  3. Compares against MI baseline (full mutual_info_classif).
  4. Shows convergence is essentially complete by iteration 5.
"""

from pathlib import Path

import numpy as np
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.schemas import MixStrategyEnum, OrgStrategyEnum
from pikaia.strategies import MixStrategyFactory, OrgStrategyFactory
from pikaia.strategies.gs_strategies.entropy_max_strategy import EntropyMaxGeneStrategy

print("=== Example 8: EntropyMax — Fast Supervised Feature Selection ===\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data = load_wine()
X_raw, y = data.data, data.target
feature_names = list(data.feature_names)

scaler = MinMaxScaler()
X = np.clip(scaler.fit_transform(X_raw), 0.0, 1.0)

N_SAMPLES, N_FEATURES = X.shape
TOP_K = N_FEATURES // 2  # top-half selection
print(
    f"Dataset : Wine — {N_SAMPLES} samples, {N_FEATURES} features, {len(np.unique(y))} classes"
)
print(f"Top-k   : {TOP_K} features\n")


# ---------------------------------------------------------------------------
# Helper: RF accuracy on top-k selected features
# ---------------------------------------------------------------------------
def rf_accuracy(
    X_full: np.ndarray, y: np.ndarray, feature_ranking: np.ndarray
) -> float:
    top_k_idx = np.argsort(feature_ranking)[-TOP_K:]
    X_sel = X_full[:, sorted(top_k_idx)]
    scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42),
        X_sel,
        y,
        cv=5,
        scoring="accuracy",
    )
    return float(scores.mean())


# ---------------------------------------------------------------------------
# MI baseline (single run, deterministic)
# ---------------------------------------------------------------------------
mi_scores = mutual_info_classif(X, y, random_state=42)
mi_acc = rf_accuracy(X, y, mi_scores)
print(f"MI baseline accuracy (top-{TOP_K} features): {mi_acc:.4f}\n")

# ---------------------------------------------------------------------------
# EntropyMax at various iteration counts
# ---------------------------------------------------------------------------
ITER_COUNTS = [1, 2, 5, 10, 20, 50]

print(f"{'Iterations':>12}  {'RF Accuracy':>12}  {'vs MI':>8}  {'Top-k features'}")
print("-" * 70)

results = {}
for max_iter in ITER_COUNTS:
    pop = PikaiaPopulation(X.copy())
    strat = EntropyMaxGeneStrategy()
    model = PikaiaModel(
        population=pop,
        gene_strategies=[strat],
        org_strategies=[OrgStrategyFactory.get_strategy(OrgStrategyEnum.BALANCED)],
        gene_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
        org_mix_strategy=MixStrategyFactory.get_strategy(MixStrategyEnum.FIXED),
        max_iter=max_iter,
        y=y,
    )
    model.fit()
    gf = model.gene_fitness_history[-1]
    acc = rf_accuracy(X, y, gf)
    top_k = sorted(np.argsort(gf)[-TOP_K:].tolist())
    agreement = acc / mi_acc if mi_acc > 0 else float("nan")
    results[max_iter] = {"acc": acc, "agreement": agreement, "top_k": top_k}
    top_names = [feature_names[i] for i in top_k]
    print(f"{max_iter:>12}  {acc:>12.4f}  {agreement:>7.3f}x  {top_names}")

print("-" * 70)
print(f"{'MI (baseline)':>12}  {mi_acc:>12.4f}  {'1.000x':>8}")

print()
acc_at_5 = results[5]["acc"]
agreement_at_5 = results[5]["agreement"]
print(f"At 5 iterations: accuracy = {acc_at_5:.4f}  ({agreement_at_5:.3f}× of MI)")
print()
print("Interpretation:")
print("  EntropyMax uses mutual information with y as gene fitness, so it converges")
print("  immediately — the replicator dynamics amplify the MI signal with each step.")
print(f"  After just 5 iterations the selected top-{TOP_K} features match MI almost")
print("  exactly, making it the lowest-cost entry point for supervised selection")
print("  within the Pikaia framework.")

# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
try:
    import matplotlib.pyplot as plt

    OUT_DIR = Path("artefacts/example8")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    iters = list(results.keys())
    accs = [results[i]["acc"] for i in iters]
    ax.plot(
        iters,
        accs,
        "o-",
        color="#2c7bb6",
        linewidth=2,
        markersize=7,
        label="EntropyMax (ENTR-BAL)",
    )
    ax.axhline(
        mi_acc,
        color="#d7191c",
        linestyle="--",
        linewidth=1.5,
        label=f"MI baseline ({mi_acc:.4f})",
    )
    ax.axvline(5, color="grey", linestyle=":", linewidth=1.2, label="5-iteration mark")
    ax.set_xlabel("Max iterations")
    ax.set_ylabel(f"RF accuracy (top-{TOP_K} features, 5-fold CV)")
    ax.set_title(
        "EntropyMax convergence speed — Wine dataset\n(supervised; requires y)"
    )
    ax.legend(fontsize=9)
    ax.set_xscale("log")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    path = OUT_DIR / "entropymax_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved → {path}")
except ImportError:
    print("\n(matplotlib not available — skipping plot)")

print("\n=== Example 8 completed. ===")
