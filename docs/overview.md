# pikaia: Genetic AI — Overview

**pikaia** is a Python library for data analysis using evolutionary simulation. It models tabular data as a population of organisms (rows) and genes (columns), then applies strategies inspired by evolutionary biology and game theory to uncover which features drive differentiation in your data.

For a step-by-step first run, see the [Tutorial](tutorial.md). For adding new strategies, see the [Contributor Guide](contributing.md).

---

## 1. The replicator equation

pikaia evolves a gene-fitness vector **γ** (one value per feature, summing to 1) using the **replicator equation**:

```text
γ_j(t+1) = γ_j(t) · (1 + Σ_i Δ(i, j))
```

At each iteration, every organism *i* contributes a delta Δ(i, j) to gene *j*. The result is multiplied element-wise and re-normalised. Genes with consistently positive deltas grow in fitness; genes with negative deltas shrink.

This differs from gradient-based optimisation: there is no loss function, no training data, and no labels. The simulation explores the population's internal structure through the strategy rules.

---

## 2. Organisms, genes, and the population matrix

| Concept | Meaning | Representation |
|---------|---------|----------------|
| **Gene** | A feature or criterion in your data | Column of the matrix |
| **Organism** | A sample, candidate, or entity | Row of the matrix |
| **Gene fitness γ_j** | How much gene *j* drives differentiation | Scalar in [0, 1], Σ = 1 |
| **Organism fitness** | How well organism *i* expresses the current gene-fitness weighting | Dot product of row *i* with **γ** |

Data must be normalised to [0, 1] (higher = better) before passing to `PikaiaPopulation`. The `PikaiaPreprocessor` handles this for common cases.

---

## 3. Strategies

Strategies are the rules that determine how organisms and genes interact each iteration. They produce the delta values that feed the replicator equation.

### 3.1. Gene strategies (`GeneStrategy`)

Called once per *(organism i, gene j)* pair. Returns a scalar delta for gene *j* based on how organism *i* expressed it.

| Strategy | Effect |
|----------|--------|
| `DOMINANT` | Rewards genes that are highly and broadly expressed |
| `ALTRUISTIC` | Gene donates fitness to dissimilar genes |
| `SELFISH` | Gene takes fitness from similar genes |
| `KIN_ALTRUISTIC` | Altruistic within a similarity neighbourhood |
| `SELL_HARD` | Drains value from rare genes (high exclusiveness); pair with `BUY_HARD` for full trading behaviour |
| `SELL_UNIFORM` | Drains value uniformly regardless of gene difficulty; pair with `BUY_UNIFORM` for full trading behaviour |
| `SELL_EASY` | Drains value from common genes — inverse of `SELL_HARD`; pair with `BUY_EASY` for full trading behaviour |
| `VARIANCE` | Rewards genes with high cross-organism dispersion |
| `ENTROPY_MAX` | Rewards genes with high entropy; supports supervised mode |
| `ORTHO_GENE` | Rewards genes that are uncorrelated with each other; supports supervised mode |
| `PARTIAL_CORR` | Rewards genes with low partial correlation to others; supports supervised mode |
| `REDUNDANCY_PENALTY` | Penalises genes that are redundant with the rest; supports supervised mode |
| `NONE` | No contribution |

### 3.2. Organism strategies (`OrgStrategy`)

Called once per organism *i*. Returns an array of shape (M,) — the delta for every gene in one shot. Organism strategies can express **cross-gene** interactions that a per-gene strategy cannot.

| Strategy | Effect |
|----------|--------|
| `BALANCED` | Organism balances mean expression against current gene fitness |
| `ALTRUISTIC` | Redistributes fitness toward dissimilar organisms |
| `SELFISH` | Takes fitness from similar organisms |
| `KIN_SELFISH` | Selfish within a similarity neighbourhood |
| `BUY_HARD` | Redistributes capital to easy genes the organism failed; pair with `SELL_HARD` |
| `BUY_UNIFORM` | Redistributes capital to hard genes the organism failed; pair with `SELL_UNIFORM` |
| `BUY_EASY` | Inverse redistribution — mirror of `BUY_HARD`; pair with `SELL_EASY` |
| `NONE` | No contribution |

### 3.3. Supervised mode

Four gene strategies (`ENTROPY_MAX`, `ORTHO_GENE`, `PARTIAL_CORR`, `REDUNDANCY_PENALTY`) can optionally incorporate a target variable. Pass `y` to `PikaiaModel` and they blend it into their signal automatically — without `y` they run fully unsupervised:

```python
model = PikaiaModel(population=population, gene_strategies=gene_strategies, y=labels)
```

### 3.4. Mixing strategies (`MixStrategy`)

When multiple gene or organism strategies are active, a mixing strategy determines how their deltas are combined each iteration.

| Strategy | Effect |
|----------|--------|
| `FIXED` | Fixed equal weights across strategies |
| `SELF_CONSISTENT` | Weights adapt each iteration based on strategy performance |

---

## 4. D-matrix accelerated mode

For compatible strategy combinations, pikaia precomputes a compact kernel `(D, d)` once before the iteration loop and then runs cheap `O(M²)` updates:

```text
γ_new = γ * (1 + d + γ * (D @ γ))
```

instead of the full `O(N·M²)` per-organism loop. The practical speed-up depends on population size, gene count, and the selected strategies.

Enable it with:

```python
model = PikaiaModel(
    population=population,
    gene_strategies=gene_strategies,
    org_strategies=org_strategies,
    use_d_matrix=True,
    max_iter=500,
)
```

In `ORIGINAL`, D-matrix execution is limited to the built-in `FixedMixStrategy` and strategies whose kernels are regression-tested as exact; every selected non-no-op strategy must support it. In `MATH_PAPER`, the exact public configurations are dominant gene paired with a no-op organism strategy and the unmixed altruistic-gene plus selfish-organism (Alt-Sel) pair. Adaptive, custom, and otherwise unsupported requests raise `ValueError` instead of silently omitting a contribution or using an approximation.

See the [D-matrix formulation](d-matrix.md) for the exact equations, historical similarity scaling, compatibility table, and limits.

---

## 5. Key classes

| Class | Role |
|-------|------|
| `PikaiaPopulation` | Wraps the (N, M) data matrix |
| `PikaiaModel` | Orchestrates the simulation; records fitness histories |
| `PikaiaPreprocessor` | Scales raw data to [0, 1] per feature |
| `PikaiaPlotter` | Plots gene/organism fitness trajectories |
| `GeneStrategyFactory` | Instantiates gene strategies by enum |
| `OrgStrategyFactory` | Instantiates organism strategies by enum |
| `MixStrategyFactory` | Instantiates mixing strategies by enum |

---

## 6. Where to go next

- [Tutorial](tutorial.md) — run your first analysis end to end
- [Contributor Guide](contributing.md) — add new strategies and extend pikaia
- [Reference](reference.md) — full auto-generated reference documentation
- [Examples](https://github.com/danube-ai/pikaia/tree/main/examples) — runnable scripts for real-world and synthetic datasets
- [Preprint](https://arxiv.org/abs/2501.19113) — scientific background (Genetic AI)
