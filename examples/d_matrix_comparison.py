#!/usr/bin/env python3
"""D-Matrix Comparison Example

Demonstrates all three fit modes of PikaiaModel on the apartment dataset
(15 apartments x 4 features: Rent, Size, Rooms, Balcony).

Three fit modes
---------------
  1. Analytical fix-point   -- use_d_matrix=False, max_iter=None
     Formula: gamma*_j ~ 1 / (x_bar_j + 0.5).  Only valid for Dominant+Balanced.
     Instant: no iterations.

  2. Standard iterative     -- use_d_matrix=False, max_iter=N
     Full replicator update: O(N*M^2) per iteration.
     Works with any strategy combination.

  3. D-matrix iterative     -- use_d_matrix=True,  max_iter=N
     Precomputes D from strategy kernels; O(M^2) per iteration.
     Requires at least one strategy to implement kernel().
     Fails for NoneGeneStrategy + NoneOrgStrategy (no kernel).

All 5 x 5 = 25 combinations are run:
  Gene strategies : Dominant, Altruistic, Selfish, KinAltruistic, None
  Org  strategies : Balanced, Altruistic, KinSelfish, Selfish,   None

D-matrix is skipped for NoneGene+NoneOrg (raises ValueError -- no kernel).
Fix-point is the same analytical baseline for all combos (Dom+Bal formula).

Runtime comparison: fix-point (baseline) vs std iterative vs D-matrix iterative.
Consistency check:  cosine(std iter, D-matrix) should be ~= 1.0 for all combos.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np

from pikaia.data import PikaiaPopulation
from pikaia.models import PikaiaModel
from pikaia.preprocessing import PikaiaPreprocessor, max_scaler
from pikaia.schemas import FeatureType
from pikaia.strategies.gs_strategies.altruistic_strategy import AltruisticGeneStrategy
from pikaia.strategies.gs_strategies.dominant_strategy import DominantGeneStrategy
from pikaia.strategies.gs_strategies.kin_altruistic_strategy import KinAltruisticGeneStrategy
from pikaia.strategies.gs_strategies.none_strategy import NoneGeneStrategy
from pikaia.strategies.gs_strategies.selfish_strategy import SelfishGeneStrategy
from pikaia.strategies.os_strategies.altruistic_strategy import AltruisticOrgStrategy
from pikaia.strategies.os_strategies.balanced_strategy import BalancedOrgStrategy
from pikaia.strategies.os_strategies.kin_selfish_strategy import KinSelfishOrgStrategy
from pikaia.strategies.os_strategies.none_strategy import NoneOrgStrategy
from pikaia.strategies.os_strategies.selfish_strategy import SelfishOrgStrategy

# ---------------------------------------------------------------------------
# 0. Output directory
# ---------------------------------------------------------------------------
OUT_DIR = "artefacts/d_matrix_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 75)
print("D-Matrix Comparison: All 25 Strategy Combinations x Three Fit Modes")
print("=" * 75)

# ---------------------------------------------------------------------------
# 1. Data (apartment dataset)
# ---------------------------------------------------------------------------
X_raw = np.array(
    [
        [4348, 138, 3.0, 0],
        [2647, 133, 4.0, 0],
        [7413, 460, 7.0, 0],
        [5644, 329, 6.0, 0],
        [5979, 252, 6.0, 1],
        [5016, 219, 6.0, 0],
        [1106, 123, 2.0, 0],
        [4409, 175, 5.0, 0],
        [7708, 230, 8.0, 0],
        [5143, 159, 4.0, 0],
        [1650, 133, 3.0, 0],
        [7933, 383, 14.5, 1],
        [7912, 314, 7.0, 0],
        [8442, 335, 7.0, 0],
        [3218, 165, 3.0, 0],
    ],
    dtype=np.float64,
)
gene_labels = ["Rent (cost)", "Size (gain)", "Rooms (gain)", "Balcony (gain)"]
short_labels = ["Rent", "Size", "Rooms", "Balcony"]
bar_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

feature_types = [FeatureType.COST, FeatureType.GAIN, FeatureType.GAIN, FeatureType.GAIN]
preprocessor = PikaiaPreprocessor(
    num_features=X_raw.shape[1],
    feature_types=feature_types,
    feature_transforms=[max_scaler] * 4,
)
X_pre = preprocessor.fit_transform(X_raw)
phi = PikaiaPopulation(X_pre)
x_bar = phi.matrix.mean(axis=0)
N, M = phi.N, phi.M

print(f"\nDataset: {N} apartments x {M} features")
print("Column means after preprocessing (x_bar_j):")
for lbl, val in zip(gene_labels, x_bar):
    print(f"  {lbl:<22}: {val:.4f}")

# ---------------------------------------------------------------------------
# 2. Strategy grid (5 gene x 5 org = 25 combinations)
# ---------------------------------------------------------------------------
GENE_STRATEGIES = [
    ("Dominant",     DominantGeneStrategy),
    ("Altruistic",   AltruisticGeneStrategy),
    ("Selfish",      SelfishGeneStrategy),
    ("KinAltruistic",KinAltruisticGeneStrategy),
    ("None",         NoneGeneStrategy),
]
ORG_STRATEGIES = [
    ("Balanced",    BalancedOrgStrategy),
    ("Altruistic",  AltruisticOrgStrategy),
    ("KinSelfish",  KinSelfishOrgStrategy),
    ("Selfish",     SelfishOrgStrategy),
    ("None",        NoneOrgStrategy),
]
GENE_NAMES = [g for g, _ in GENE_STRATEGIES]
ORG_NAMES  = [o for o, _ in ORG_STRATEGIES]

# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
MAX_ITER = 500
EPSILON = 1e-8


def last_gamma(model: PikaiaModel) -> np.ndarray:
    mask = model.gene_fitness_history.sum(axis=1) > 0
    return model.gene_fitness_history[np.flatnonzero(mask).max(), :]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def run_iterative_timed(pop, gene_strats, org_strats):
    m = PikaiaModel(
        population=pop,
        gene_strategies=gene_strats,
        org_strategies=org_strats,
        max_iter=MAX_ITER,
        epsilon=EPSILON,
        use_d_matrix=False,
    )
    t0 = time.perf_counter()
    m.fit()
    return m, time.perf_counter() - t0


def run_d_matrix_timed(pop, gene_strats, org_strats):
    m = PikaiaModel(
        population=pop,
        gene_strategies=gene_strats,
        org_strategies=org_strats,
        max_iter=MAX_ITER,
        epsilon=EPSILON,
        use_d_matrix=True,
    )
    t0 = time.perf_counter()
    m.fit()
    return m, time.perf_counter() - t0


def run_fix_point_timed(pop):
    m = PikaiaModel(population=pop)
    t0 = time.perf_counter()
    m.fit()
    return m, time.perf_counter() - t0


# ---------------------------------------------------------------------------
# 4. Section A -- Dominant+Balanced: all three modes + timing
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION A -- DominantGeneStrategy + BalancedOrgStrategy (all 3 modes)")
print("=" * 75)

m_fix, t_fix = run_fix_point_timed(phi)
m_iter_A, t_iter_A = run_iterative_timed(phi, [DominantGeneStrategy()], [BalancedOrgStrategy()])
m_dm_A,   t_dm_A   = run_d_matrix_timed(phi, [DominantGeneStrategy()], [BalancedOrgStrategy()])

gf_fix    = m_fix.gene_fitness_history[1, :]
gf_iter_A = last_gamma(m_iter_A)
gf_dm_A   = last_gamma(m_dm_A)

print(f"\n  {'Feature':<22} {'Fix-point':>12} {'Std iter':>12} {'D-matrix':>12}")
print("  " + "-" * 62)
for j, lbl in enumerate(gene_labels):
    print(f"  {lbl:<22} {gf_fix[j]:>12.6f} {gf_iter_A[j]:>12.6f} {gf_dm_A[j]:>12.6f}")

cos_fix_iter  = cosine(gf_fix, gf_iter_A)
cos_fix_dm    = cosine(gf_fix, gf_dm_A)
cos_iter_dm_A = cosine(gf_iter_A, gf_dm_A)

print(f"\n  Cosine  fix-point vs std-iter : {cos_fix_iter:.6f}")
print(f"  Cosine  fix-point vs d-matrix : {cos_fix_dm:.6f}")
print(f"  Cosine  std-iter  vs d-matrix : {cos_iter_dm_A:.6f}")
print(f"\n  Runtime  fix-point : {t_fix*1000:>8.4f} ms  (baseline)")
print(f"  Runtime  std-iter  : {t_iter_A*1000:>8.4f} ms  "
      f"(x{t_iter_A/t_fix:>6.1f} fix-point)")
print(f"  Runtime  D-matrix  : {t_dm_A*1000:>8.4f} ms  "
      f"(x{t_dm_A/t_fix:>6.1f} fix-point, x{t_dm_A/t_iter_A:.2f} std-iter)")
print(f"  ESE  std-iter={m_iter_A.ESE_iter}  D-matrix={m_dm_A.ESE_iter}")

# ---------------------------------------------------------------------------
# 5. Section B -- ALL 25 combos: std iterative vs D-matrix, with timing
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("SECTION B -- All 25 combinations: Standard iterative vs D-matrix")
print("             (D-matrix skipped for NoneGene+NoneOrg -- no kernel)")
print("=" * 75)

# 5x5 result arrays (rows=gene, cols=org)
iter_times  = np.full((5, 5), np.nan)
dm_times    = np.full((5, 5), np.nan)
cosines     = np.full((5, 5), np.nan)
speedups    = np.full((5, 5), np.nan)   # t_iter / t_dm

combo_results = {}  # (gi, oi) -> dict

for gi, (gname, GCls) in enumerate(GENE_STRATEGIES):
    for oi, (oname, OCls) in enumerate(ORG_STRATEGIES):
        m_iter, t_iter = run_iterative_timed(phi, [GCls()], [OCls()])
        gf_i = last_gamma(m_iter)
        iter_times[gi, oi] = t_iter

        dm_ok = not (gname == "None" and oname == "None")
        if dm_ok:
            try:
                m_dm, t_dm = run_d_matrix_timed(phi, [GCls()], [OCls()])
                gf_d = last_gamma(m_dm)
                cos_val = cosine(gf_i, gf_d)
                dm_times[gi, oi]  = t_dm
                cosines[gi, oi]   = cos_val
                speedups[gi, oi]  = t_iter / t_dm
                combo_results[(gi, oi)] = {
                    "gf_iter": gf_i, "gf_dm": gf_d, "cosine": cos_val,
                    "t_iter": t_iter, "t_dm": t_dm,
                    "ese_iter": m_iter.ESE_iter, "ese_dm": m_dm.ESE_iter,
                }
            except ValueError as e:
                combo_results[(gi, oi)] = {
                    "gf_iter": gf_i, "gf_dm": None, "cosine": None,
                    "t_iter": t_iter, "t_dm": None,
                    "ese_iter": m_iter.ESE_iter, "ese_dm": None,
                    "error": str(e),
                }
        else:
            combo_results[(gi, oi)] = {
                "gf_iter": gf_i, "gf_dm": None, "cosine": None,
                "t_iter": t_iter, "t_dm": None,
                "ese_iter": m_iter.ESE_iter, "ese_dm": None,
            }

# Print timing table
print(f"\n  {'Combination':<32} {'t_iter(ms)':>10} {'t_dm(ms)':>10} "
      f"{'speedup':>8} {'cosine':>9} {'ESE_i':>6} {'ESE_d':>6}")
print("  " + "-" * 85)
for gi, (gname, _) in enumerate(GENE_STRATEGIES):
    for oi, (oname, _) in enumerate(ORG_STRATEGIES):
        r = combo_results[(gi, oi)]
        name = f"{gname} + {oname}"
        t_i_ms = r["t_iter"] * 1000
        t_d_ms = r["t_dm"] * 1000 if r["t_dm"] is not None else float("nan")
        sp = speedups[gi, oi]
        cos_v = r["cosine"]
        ese_i = r["ese_iter"]
        ese_d = r["ese_dm"]

        t_d_str  = f"{t_d_ms:>10.3f}" if not np.isnan(t_d_ms) else "       n/a"
        sp_str   = f"{sp:>8.1f}"       if not np.isnan(sp)     else "     n/a"
        cos_str  = f"{cos_v:>9.6f}"    if cos_v is not None     else "      n/a"
        ese_d_str= f"{ese_d:>6}"       if ese_d is not None     else "   n/a"

        print(f"  {name:<32} {t_i_ms:>10.3f}{t_d_str}{sp_str}{cos_str}{ese_i:>6}{ese_d_str}")

# ---------------------------------------------------------------------------
# 6. Runtime summary across all combos
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("Runtime Summary")
print("=" * 75)
valid_sp = speedups[~np.isnan(speedups)]
print(f"\n  Fix-point baseline       : {t_fix*1000:.4f} ms")
print(f"  Std iterative  -- mean   : {np.nanmean(iter_times)*1000:.3f} ms  "
      f"min={np.nanmin(iter_times)*1000:.3f}  max={np.nanmax(iter_times)*1000:.3f}")
print(f"  D-matrix iter  -- mean   : {np.nanmean(dm_times)*1000:.3f} ms  "
      f"min={np.nanmin(dm_times)*1000:.3f}  max={np.nanmax(dm_times)*1000:.3f}")
print("\n  D-matrix speedup vs std-iter:")
print(f"    mean={np.mean(valid_sp):.1f}x   median={np.median(valid_sp):.1f}x   "
      f"min={np.min(valid_sp):.1f}x   max={np.max(valid_sp):.1f}x")
print(f"\n  D-matrix vs fix-point (Dom+Bal only, N={N}, M={M}):")
print(f"    fix-point={t_fix*1000:.4f} ms  D-mat={t_dm_A*1000:.4f} ms  "
      f"(D-mat is {t_dm_A/t_fix:.1f}x slower than fix-point)")
print("  => Fix-point wins when Dominant+Balanced is sufficient;")
print(f"     D-matrix is O(M^2)/iter vs O(N*M^2)/iter for std -- "
      f"speedup grows with N (here N={N}).")

# ---------------------------------------------------------------------------
# 7. Consistency check
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("Consistency summary (cosine: std-iterative vs D-matrix)")
print("=" * 75)
for gi, (gname, _) in enumerate(GENE_STRATEGIES):
    for oi, (oname, _) in enumerate(ORG_STRATEGIES):
        cos_v = cosines[gi, oi]
        if np.isnan(cos_v):
            flag = "--"
            cos_str = "  n/a (no kernel)"
        elif cos_v > 0.9999:
            flag = "OK"
            cos_str = f"  {cos_v:.6f}"
        elif cos_v > 0.999:
            flag = "~="
            cos_str = f"  {cos_v:.6f}"
        else:
            flag = "!!"
            cos_str = f"  {cos_v:.6f}"
        print(f"  [{flag}] {gname:<14} + {oname:<12}:{cos_str}")

# ---------------------------------------------------------------------------
# 8. Plots
# ---------------------------------------------------------------------------

# -- Plot 1: Section A bar chart (all three modes) --
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
x = np.arange(M)
w = 0.6
for ax, (gf, title) in zip(
    axes,
    [
        (gf_fix,    f"Fix-point\n({t_fix*1000:.3f} ms)"),
        (gf_iter_A, f"Std iterative\n({t_iter_A*1000:.3f} ms, ESE={m_iter_A.ESE_iter})"),
        (gf_dm_A,   f"D-matrix iterative\n({t_dm_A*1000:.3f} ms, ESE={m_dm_A.ESE_iter})"),
    ],
):
    bars = ax.bar(x, gf, w, color=bar_colors, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, rotation=15, ha="right")
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Gene fitness weight")
    ax.set_ylim(0, max(gf) * 1.35)
    for bar, val in zip(bars, gf):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
fig.suptitle(
    f"Section A: Dominant+Balanced -- all three fit modes\n"
    f"cosine: fp/iter={cos_fix_iter:.4f}  fp/dm={cos_fix_dm:.4f}  iter/dm={cos_iter_dm_A:.4f}",
    fontsize=11,
)
plt.tight_layout()
out1 = f"{OUT_DIR}/section_a_three_modes.png"
plt.savefig(out1, dpi=150)
plt.close()
print(f"\nSection A chart saved to {out1}")

# -- Plot 2: 5×5 Heatmaps (cosine, speedup, iter time, dm time) --
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

cosine_masked   = np.where(np.isnan(cosines),   0.0, cosines)
speedup_masked  = np.where(np.isnan(speedups),  0.0, speedups)
iter_ms         = iter_times * 1000
dm_ms           = np.where(np.isnan(dm_times), 0.0, dm_times * 1000)

hmap_data = [
    (axes[0, 0], cosine_masked,  "Cosine (iter vs D-mat)",  "RdYlGn", 0.998, 1.0),
    (axes[0, 1], speedup_masked, "Speedup (t_iter / t_dm)", "YlOrRd", None,  None),
    (axes[1, 0], iter_ms,        "Std iter time (ms)",      "Blues",  None,  None),
    (axes[1, 1], dm_ms,          "D-matrix time (ms)",      "Oranges",None,  None),
]
for ax, data, title, cmap, vmin, vmax in hmap_data:
    kwargs = {"vmin": vmin, "vmax": vmax} if vmin is not None else {}
    im = ax.imshow(data, cmap=cmap, aspect="auto", **kwargs)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(ORG_NAMES, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(GENE_NAMES, fontsize=9)
    ax.set_xlabel("Org strategy", fontsize=9)
    ax.set_ylabel("Gene strategy", fontsize=9)
    ax.set_title(title, fontsize=11)
    for gi in range(5):
        for oi in range(5):
            v = data[gi, oi]
            if v != 0 or (gi == 4 and oi == 4):
                ax.text(oi, gi, f"{v:.3f}" if v < 10 else f"{v:.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if (title.startswith("D-mat") and v > dm_ms.max() * 0.6) else "black")

fig.suptitle(
    f"All 25 combinations -- Cosine, Speedup, and Runtime Heatmaps\n"
    f"(fix-point baseline = {t_fix*1000:.4f} ms, N={N}, M={M})",
    fontsize=12,
)
plt.tight_layout()
out2 = f"{OUT_DIR}/heatmaps_25_combos.png"
plt.savefig(out2, dpi=150)
plt.close()
print(f"Heatmap chart saved to {out2}")

# -- Plot 3: Runtime bar chart -- all 25 combos --
combo_labels = [f"{gn[:3]}\n+{on[:3]}"
                for gn, _ in GENE_STRATEGIES for on, _ in ORG_STRATEGIES]
iter_ms_flat = iter_times.flatten() * 1000
dm_ms_flat   = np.where(np.isnan(dm_times.flatten()), np.nan, dm_times.flatten() * 1000)

idx = np.arange(25)
fig, ax = plt.subplots(figsize=(18, 5))
ax.bar(idx - 0.2, iter_ms_flat, 0.4, label="Std iterative", color="#4C72B0", alpha=0.85)
ax.bar(idx + 0.2, dm_ms_flat,   0.4, label="D-matrix iterative", color="#DD8452", alpha=0.85)
ax.axhline(t_fix * 1000, color="#55A868", linewidth=1.5, linestyle="--",
           label=f"Fix-point ({t_fix*1000:.3f} ms)")
ax.set_xticks(idx)
ax.set_xticklabels(combo_labels, fontsize=6.5)
ax.set_ylabel("Time (ms)")
ax.set_title(f"Runtime: Standard iterative vs D-matrix iterative -- all 25 combinations\n"
             f"(max_iter={MAX_ITER}, epsilon={EPSILON}, N={N}, M={M})")
ax.legend(fontsize=9)
ax.set_xlim(-0.7, 24.7)
plt.tight_layout()
out3 = f"{OUT_DIR}/runtime_all_combos.png"
plt.savefig(out3, dpi=150)
plt.close()
print(f"Runtime bar chart saved to {out3}")

# -- Plot 4: Convergence (Dom+Bal, iter vs d-mat) --
fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
for ax, model, title in [
    (axes[0], m_iter_A, f"Std iterative  (Dom+Bal, ESE={m_iter_A.ESE_iter})"),
    (axes[1], m_dm_A,   f"D-matrix iter  (Dom+Bal, ESE={m_dm_A.ESE_iter})"),
]:
    hist   = model.gene_fitness_history
    filled = hist[hist.sum(axis=1) > 0]
    for j in range(M):
        ax.plot(filled[:, j], label=short_labels[j], color=bar_colors[j])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gene fitness weight")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 0.7)
plt.suptitle("Convergence: Standard iterative vs D-matrix iterative (Dominant+Balanced)")
plt.tight_layout()
out4 = f"{OUT_DIR}/convergence_comparison.png"
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"Convergence chart saved to {out4}")

# ---------------------------------------------------------------------------
# 9. Top-5 rankings (Section A, Dominant+Balanced)
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("Top-5 Apartment Rankings -- Dominant+Balanced, all three modes")
print("=" * 75)


def rank_apartments(pop, gf, x_raw, k=5):
    scores = pop.matrix @ gf
    top_idx = np.argsort(scores)[::-1][:k]
    return [(i + 1, scores[i], x_raw[i]) for i in top_idx]


for label, gf in [("Fix-point", gf_fix), ("Std iter", gf_iter_A), ("D-matrix", gf_dm_A)]:
    print(f"\n  {label}:")
    print(f"  {'Rk':<4} {'Apt':>4} {'Score':>8}  {'Rent':>6} {'Size':>6} {'Rooms':>6} {'Balcony':>8}")
    for rank, (apt_idx, score, raw) in enumerate(rank_apartments(phi, gf, X_raw), 1):
        print(f"  {rank:<4} {apt_idx:>4} {score:>8.4f}  "
              f"{raw[0]:>6.0f} {raw[1]:>6.0f} {raw[2]:>6.1f} {raw[3]:>8.0f}")

# ---------------------------------------------------------------------------
# 10. Final summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 75)
print("Summary")
print("=" * 75)
print(f"\n  Fix-point  : {t_fix*1000:.4f} ms  -- analytical, instant, Dom+Bal only")
print(f"  Std iter   : mean {np.nanmean(iter_times)*1000:.3f} ms over 25 combos")
print(f"  D-matrix   : mean {np.nanmean(dm_times)*1000:.3f} ms over 24 combos")
print(f"\n  D-matrix speedup over std-iter: mean={np.mean(valid_sp):.1f}x, "
      f"max={np.max(valid_sp):.1f}x")
print(f"  (speedup = O(N*M^2) / O(M^2) = N={N}x theoretical; "
      f"observed lower due to Python overhead)")
valid_cosines = cosines[~np.isnan(cosines)]
n_perfect = np.sum(valid_cosines > 0.9999)
n_close   = np.sum((valid_cosines > 0.999) & (valid_cosines <= 0.9999))
print("\n  Cosine(std-iter, D-matrix) across 24 combos:")
print(f"    Perfect (>0.9999): {n_perfect}   Near-perfect (>0.999): {n_close}   "
      f"Other: {len(valid_cosines) - n_perfect - n_close}")
print(f"    Min cosine: {valid_cosines.min():.6f}   Max: {valid_cosines.max():.6f}")
print()
print(f"  Output artefacts saved to: {OUT_DIR}/")
print("=== Example completed. ===")
