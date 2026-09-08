# Contributing to pikaia

This guide explains the codebase structure and walks through adding new features — in particular new strategies, which are the most common extension point.

---

## 1. Repository layout

```text
pikaia/
├── pikaia/                   # Library source
│   ├── data/                 # PikaiaPopulation — wraps the (N, M) matrix
│   ├── models/               # PikaiaModel + GeneticModel (simulation loop)
│   ├── preprocessing/        # Scalers and PikaiaPreprocessor
│   ├── plotting/             # PikaiaPlotter
│   ├── schemas/
│   │   └── strategies.py     # GeneStrategyEnum / OrgStrategyEnum — add names here
│   └── strategies/
│       ├── base_strategies.py         # GeneStrategy, OrgStrategy, MixStrategy ABCs
│       ├── strategy_factories.py      # Enum → class mapping — register here
│       ├── gs_strategies/             # Gene strategy implementations
│       ├── os_strategies/             # Organism strategy implementations
│       └── mix_strategies/            # Mixing strategy implementations
├── tests/
│   └── unit/                 # One file per strategy family
├── examples/                 # Runnable scripts (example1.py … example6.py)
└── docs/                     # MkDocs source (this file lives here)
```

---

## 2. Core concepts

### 2.1. The replicator equation

pikaia evolves a gene-fitness vector **γ** of shape `(M,)` (one value per gene/feature). Each iteration applies:

```text
γ_j(t+1) = γ_j(t) · (1 + Σ_i Δ(i, j))
```

then normalises **γ** to sum to 1. The deltas `Δ(i, j)` come from two complementary strategy types.

### 2.2. Gene strategies (`GeneStrategy`)

Called once per *(organism i, gene j)* pair. Return a **scalar** delta that drives gene *j*'s fitness up or down based on how organism *i* expressed it.

**When to use:** the effect of organism *i* on gene *j* depends only on *x_ij* and population-level statistics (e.g. gene means). Examples: `DominantGeneStrategy`, `SellHardGeneStrategy`.

### 2.3. Organism strategies (`OrgStrategy`)

Called once per **organism i**. Return an `(M,)` array — the delta for every gene in one shot.

**When to use:** organism *i*'s contribution to gene *j* depends on *i*'s performance on other genes (cross-gene interaction). Examples: `BalancedOrgStrategy`, `BuyHardOrgStrategy`.

### 2.4. `StrategyContext`

Both strategy types receive a `StrategyContext` dataclass:

| Field | Type | Description |
|---|---|---|
| `population` | `PikaiaPopulation` | Full population; `population.matrix` is the `(N, M)` data array |
| `gene_fitness` | `ndarray (M,)` | Current gene-fitness vector **γ** |
| `org_fitness` | `ndarray (N,)` | Current organism-fitness vector |
| `gene_similarity` | `ndarray (M, M)` | Pairwise gene similarities |
| `org_similarity` | `ndarray (N, N)` | Pairwise organism similarities |
| `initial_org_fitness_range` | `float` | Range of organism fitness at t=0 |
| `org_id` | `int` | Index of current organism (gene strategies only) |
| `gene_id` | `int` | Index of current gene (gene strategies only) |
| `y` | `ndarray \| None` | Optional supervised target |

---

## 3. Adding a new gene strategy

We'll implement a toy `BiasGeneStrategy` that gives a fixed positive boost to genes above a threshold and penalises the rest.

### 3.1. Step 1 — Implement the class

Create `pikaia/strategies/gs_strategies/bias_strategy.py`:

```python
import numpy as np

from pikaia.data.population import PikaiaPopulation
from pikaia.strategies.base_strategies import GeneStrategy, StrategyContext


class BiasGeneStrategy(GeneStrategy):
    """
    Boosts genes whose mean expression exceeds a threshold, penalises the rest.

    Args:
        threshold: Mean-expression cutoff in [0, 1].  Defaults to 0.5.
    """

    def __init__(self, threshold: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.options["threshold"] = threshold

    @property
    def name(self) -> str:
        return "Bias"

    def __call__(self, ctx: StrategyContext) -> float:
        mean_j = ctx.population.matrix[:, ctx.gene_id].mean()
        sign = 1.0 if mean_j >= self.options["threshold"] else -1.0
        return float((4.0 / ctx.population.N) * sign * ctx.gene_fitness[ctx.gene_id])

    def kernel(
        self,
        population: PikaiaPopulation,
        gene_similarity: np.ndarray,
        org_similarity: np.ndarray,
        initial_org_fitness_range: float,
        y: np.ndarray | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Diagonal D-matrix: D[j,j] = 4 * sign_j."""
        threshold = self.options["threshold"]
        mean_all = population.matrix.mean(axis=0)
        signs = np.where(mean_all >= threshold, 1.0, -1.0)
        D = np.diag(4.0 * signs)
        return D, None
```

**Key rules:**

- `__call__` must return a Python `float`.
- `kernel()` returns `(D, d)` where `D` is `(M, M)` (bilinear term) and `d` is `(M,)` (linear term). Return `None` for the term your strategy doesn't use. The default base-class implementation already returns `(None, None)`, so you can skip `kernel()` entirely if you don't need the D-matrix fast path.

### 3.2. Step 2 — Add an enum value

In `pikaia/schemas/strategies.py`, add to `GeneStrategyEnum`:

```python
BIAS = "BIAS"
```

Include a short docstring line in the class docstring:

```text
BIAS: Boosts genes above a mean-expression threshold, penalises the rest.
```

### 3.3. Step 3 — Register in the factory

In `pikaia/strategies/strategy_factories.py`, import the class and add it to `GeneStrategyFactory._strategies`:

```python
from pikaia.strategies.gs_strategies.bias_strategy import BiasGeneStrategy

# inside GeneStrategyFactory._strategies:
GeneStrategyEnum.BIAS: BiasGeneStrategy,
```

### 3.4. Step 4 — Export from the package

In `pikaia/strategies/gs_strategies/__init__.py`:

```python
from .bias_strategy import BiasGeneStrategy

__all__ = [
    ...
    "BiasGeneStrategy",
]
```

### 3.5. Step 5 — Write tests

Add `tests/unit/test_bias_strategy.py`. At minimum cover:

- **Sign correctness**: genes above threshold should produce positive deltas.
- **Kernel consistency**: `D[j,j]` matches what `__call__` produces when summed over all organisms.
- **Edge cases**: threshold at 0, threshold at 1, all-same population.
- **Enum/factory round-trip**: `GeneStrategyFactory.get_strategy(GeneStrategyEnum.BIAS)` returns a `BiasGeneStrategy`.

```python
def test_kernel_diagonal_matches_call_sum():
    pop = PikaiaPopulation(np.random.default_rng(0).random((6, 4)))
    strat = BiasGeneStrategy(threshold=0.5)
    D, d = strat.kernel(pop, np.eye(4), np.eye(6), 1.0)
    assert d is None
    # sum of __call__ over all organisms per gene
    gf = np.ones(4) / 4
    call_sum = np.array([
        sum(strat(StrategyContext(..., org_id=i, gene_id=j)) for i in range(6))
        for j in range(4)
    ])
    np.testing.assert_allclose(np.diag(D) @ gf, call_sum @ gf, atol=1e-10)
```

---

## 4. Adding a new organism strategy

The pattern is identical; the only differences are:

- Subclass `OrgStrategy` instead of `GeneStrategy`.
- `__call__` receives a context with `org_id` set and returns `np.ndarray` of shape `(M,)`.
- File goes in `pikaia/strategies/os_strategies/`.
- Enum value goes in `OrgStrategyEnum`.
- Factory registration goes in `OrgStrategyFactory._strategies`.

**Example skeleton:**

```python
class MyOrgStrategy(OrgStrategy):
    @property
    def name(self) -> str:
        return "MyOrg"

    def __call__(self, ctx: StrategyContext) -> np.ndarray:
        # ctx.org_id is the current organism
        # return shape (M,)
        ...

    def kernel(self, population, gene_similarity, org_similarity,
               initial_org_fitness_range, y=None):
        # return (D, d) or (None, None) if no kernel
        return None, None
```

See `BuyHardOrgStrategy` in `pikaia/strategies/os_strategies/buy_hard_strategy.py` for a real example where the kernel computes a linear `d`-vector from a cross-organism redistribution.

---

## 5. The D-matrix fast path

When `PikaiaModel(use_d_matrix=True)`, the model precomputes kernels once and then runs cheap `O(M²)` updates each iteration instead of the full `O(N·M²)` loop. This is typically 30–80× faster for large populations.

For your strategy to support this path:

1. Override `kernel()` to return `(D, d)` where at least one is not `None`.
2. The update applied is `γ_new = γ * (1 + d + γ * (D @ γ))`, then re-normalised.
3. `D` captures deltas of the form `γ_j * sum_k D_jk * γ_k`; `d` captures fixed delta offsets.
4. If your delta does not depend on **γ** at all, return `D=None` and a precomputed `d`.
5. If your delta scales with `γ_j`, express it as `D[j,j]` on the diagonal.

**Test your kernel** by verifying that summing `__call__` over all organisms produces the same result as `d + γ * (D @ γ)` for a few random **γ** vectors. See the [D-matrix formulation](d-matrix.md) for the derivation procedure and `tests/unit/test_strategy_kernels.py` for examples.

---

## 6. Development workflow

```bash
# Install all dependencies (editable mode, dev + examples extras)
uv sync --extra dev --extra examples

# Run the full test suite
uv run pytest tests/unit/ --cov=pikaia --cov-report=term-missing

# Run only the strategy-related tests
uv run pytest tests/unit/test_strategy_kernels.py \
              tests/unit/test_trading_strategies_verification.py \
              tests/unit/test_sell_buy_strategies.py -v

# Run an example
uv run python examples/example6.py
```

All CI checks (ruff lint, ruff format, import ordering) run automatically on commit via pre-commit hooks. Fix issues flagged by the hook and re-commit.

---

## 7. Branching and releases

pikaia uses a **single-branch (trunk-based)** model:

- **`main`** is the only long-lived branch. Branch off it, open a PR into it, and merge with **squash or rebase** (`main` requires linear history).
- Every merge to `main` publishes to **TestPyPI** and updates the `dev` docs automatically — no version bump needed for regular PRs.
- **Production PyPI releases are cut by pushing a `v*` git tag**, gated by a manual approval on the `pypi` environment.

The full step-by-step release process lives in the [Releasing guide](releasing.md).

---

## 8. Pre-PR checklist

Before opening a pull request, verify:

- [ ] `uv lock` run after **any** change to `pyproject.toml` (deps, version, extras) and the updated `uv.lock` committed — CI runs `uv lock --locked` and will fail if the lockfile is stale
- [ ] All unit tests pass: `uv run pytest tests/unit/`
- [ ] Docs build cleanly: `uv run --extra docs mkdocs build`

> Bumping `project.version` is a **release** step, not a per-PR requirement — see the [Releasing guide](releasing.md).

---

## 9. Checklist for a new strategy

- [ ] Implementation file in `gs_strategies/` or `os_strategies/`
- [ ] Enum value added to `GeneStrategyEnum` or `OrgStrategyEnum` with docstring
- [ ] Registered in `GeneStrategyFactory` or `OrgStrategyFactory`
- [ ] Exported from the package `__init__.py`
- [ ] `kernel()` implemented if D-matrix acceleration is desired
- [ ] Unit tests covering sign, formula, kernel consistency, and enum/factory round-trip
- [ ] Added to `ALL_STRATEGIES` in `tests/unit/test_strategy_kernels.py` for y-parameter coverage
