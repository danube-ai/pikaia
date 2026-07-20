# Adaptive Supervision for Pikaia Strategies

## Goal

Enable every Pikaia strategy to operate in **both supervised and unsupervised modes** on demand, depending on the nature of the task. This allows strategies to be reused flexibly — no need for separate supervised/unsupervised variants.

### What this means

A strategy like OrthoGene can:
- **Unsupervised mode** (default): compute orthogonality from the population matrix `X` alone (current behavior)
- **Supervised mode** (when `y` is available): combine orthogonality with MI with the target

The strategy detects the mode automatically. No API change required for the user — `y=None` → unsupervised, `y=...` → supervised.

## Why This Matters

Currently, strategies are inconsistently wired for supervision:

| Strategy | Supervised? | How `y` is handled |
|----------|-------------|-------------------|
| DOMINANT | No | Never uses `y` (inherently unsupervised) |
| SELFISH | No | Never uses `y` (inherently unsupervised) |
| BALANCED | No | Never uses `y` (inherently unsupervised) |
| EntropyMax | Yes | Requires `y`, no fallback |
| OrthoGene | No | No `y` parameter at all |
| RedPenalty | No | No `y` parameter at all |
| PartialCorr | Yes/No | Has `y` param but unsupervised fallback is buggy (precision matrix issues) |
| MultiMoment | Yes | Hack: `strategy._y = y` set after instantiation |

This inconsistency makes strategy composition confusing and forces experiment authors to write custom code for each strategy.

## Design Principles

1. **Zero breaking changes** — all existing code must continue to work without modification
2. **Minimal pikaia-core changes** — only touch what's necessary in the core library
3. **Strategy-level responsibility** — each strategy decides how to use `y`, not the framework
4. **Optional by design** — `y` is always `None` by default, strategies opt-in to supervised mode

## Proposed Changes

### 1. Add `y` to `StrategyContext` (pikaia core)

**File:** `pikaia/strategies/base_strategies.py`

```python
@dataclass(slots=True)
class StrategyContext:
    """Holds the context for a strategy calculation."""

    population: PikaiaPopulation
    org_fitness: np.ndarray
    gene_fitness: np.ndarray
    org_similarity: np.ndarray
    gene_similarity: np.ndarray
    initial_org_fitness_range: float
    org_id: Optional[int] = None
    gene_id: Optional[int] = None
    y: Optional[np.ndarray] = None          # NEW: target variable
```

This is a frozen dataclass with `slots=True`. Adding an optional field with a default is backward-compatible — callers that don't pass `y` get `None`.

### 2. Thread `y` through `PikaiaModel.fit()` (pikaia core)

**File:** `pikaia/models/pikaia_model.py`

The `y` value is stored on `GeneticModel.__init__` and passed through to `_calculate_deltas()`:

```python
# In GeneticModel.__init__ (pikaia/models/genetic_model.py):
def __init__(self, population, gene_strategies=None, org_strategies=None,
             ...
             y: Optional[np.ndarray] = None,      # NEW
             **kwargs):
    ...
    self._y = y
```

Then in `PikaiaModel._calculate_deltas()`:

```python
def _calculate_deltas(self, current_org_fitness, current_gene_fitness) -> tuple:
    ...
    context_args = {
        "population": self._population,
        "org_fitness": current_org_fitness,
        "gene_fitness": current_gene_fitness,
        "initial_org_fitness_range": self._initial_org_fitness_range,
        "org_similarity": self._org_similarity,
        "gene_similarity": self._gene_similarity,
        "y": self._y,      # NEW: pass y to strategies
    }
```

### 3. Strategies check `ctx.y` for supervised mode (experiment repo)

Each strategy in the experiments repo checks `ctx.y` to decide which mode to use:

```python
class OrthoGeneStrategy(GeneStrategy):
    def __call__(self, ctx: StrategyContext) -> float:
        if self._orthogonality is None:
            X = ctx.population.matrix
            if ctx.y is not None:
                # Supervised: orthogonality + MI with target
                self._orthogonality = self._compute_supervised(X, ctx.y)
            else:
                # Unsupervised: pure orthogonality (existing behavior)
                self._orthogonality = self._compute_unsupervised(X)

        score = self._orthogonality[ctx.gene_id] - 0.5
        return float((4 / ctx.population.N) * ctx.gene_fitness[ctx.gene_id] * score)
```

### 4. Runner passes `y` when available (experiment repo)

The experiment runner (e.g., `pikaia_helpers_new.py`) accepts an optional `y` parameter and passes it to the model:

```python
def run_pikaia(X, gene_strat=GeneStrategyEnum.DOMINANT,
               org_strat=OrgStrategyEnum.BALANCED, n_seeds=5, max_iter=3,
               fixed_point=False, dataset_name=None, y=None):  # NEW
    ...
    model = PikaiaModel(
        population=population,
        gene_strategies=[gene_obj],
        org_strategies=[org_obj],
        gene_mix_strategy=MixStrategyFactory.get_strategy(mix),
        org_mix_strategy=MixStrategyFactory.get_strategy(mix),
        max_iter=max_iter,
        y=y,  # NEW: pass y when available
    )
```

## Impact on Built-in Strategies

### DOMINANT, SELFISH, BALANCED, ALTRUISTIC, KIN_*
No changes needed. These strategies have no natural supervised interpretation. They check `ctx.y` → it's `None` → they behave exactly as before.

### Mixed-mode strategies (to be implemented)

#### OrthoGene
- **Unsupervised**: orthogonality score from correlation matrix of `X`
- **Supervised**: orthogonality * (1 + MI(j, y) factor)

#### RedPenalty
- **Unsupervised**: redundancy penalty from pairwise correlations of `X`
- **Supervised**: redundancy * (1 + MI(j, y) factor) — penalize redundant features that are also predictive

#### PartialCorr
- **Unsupervised**: partial correlation from precision matrix of `X` (fix: increase regularization)
- **Supervised**: partial correlation from precision matrix of `[X, y_encoded]` — controls for target to find direct relationships

#### MultiMoment
- **Unsupervised**: skewness + kurtosis terms only (MI term = 0)
- **Supervised**: full score with MI(j, y) term

#### EntropyMax
- **Unsupervised**: entropy of `X` only (MI term = 0)
- **Supervised**: entropy + MI(j, y) (current behavior)

## Testing Strategy

### Unit tests (pikaia core)
1. `StrategyContext` accepts `y=None` and `y=array` — no constructor errors
2. `PikaiaModel` without `y` — behaves identically to before
3. `PikaiaModel` with `y=None` — behaves identically to before
4. `PikaiaModel` with `y=array` — context receives `y`, strategies that don't use it continue to work

### Integration tests (experiment repo)
1. Run existing experiments without `y` — all 64 experiments pass (backward compat)
2. Run same experiments with `y` — supervised variants produce results
3. Compare supervised vs unsupervised for strategies that support both

## Migration Path

### Phase 1: Pikaia core changes (this PR)
- Add `y` to `StrategyContext`
- Thread `y` through `GeneticModel.__init__` and `PikaiaModel._calculate_deltas`
- Add unit tests
- No behavior change for existing users

### Phase 2: Experiment repo updates (next PR)
- Update all custom strategies to check `ctx.y` for mode switching
- Fix PartialCorr precision matrix (increase regularization)
- Add supervised variants of OrthoGene and RedPenalty
- Re-run comparison experiments

### Phase 3: Validation
- Re-run full experiment suite (64 experiments)
- Document which strategies support supervised mode
- Add to strategy documentation

## Files Modified (Pikaia Core)

| File | Change |
|------|--------|
| `pikaia/strategies/base_strategies.py` | Add `y: Optional[np.ndarray] = None` to `StrategyContext` |
| `pikaia/models/genetic_model.py` | Add `y: Optional[np.ndarray] = None` to `__init__`, store as `self._y` |
| `pikaia/models/pikaia_model.py` | Pass `self._y` in `context_args` in `_calculate_deltas` |

## Files Modified (Experiment Repo)

| File | Change |
|------|--------|
| `genetic_importance_scores/experiments/enhanced_strategies/test_strategy2_orthogonality_gene.py` | Add supervised mode to OrthoGene |
| `genetic_importance_scores/experiments/enhanced_strategies/test_strategy3_partialcorr_gene.py` | Add supervised mode + fix precision matrix |
| `genetic_importance_scores/experiments/enhanced_strategies/test_strategy4_redundancy_gene.py` | Add supervised mode to RedPenalty |
| `genetic_importance_scores/experiments/enhanced_strategies/test_strategy5_entropy_gene.py` | Refactor to use `ctx.y` instead of `self._y` |
| `genetic_importance_scores/experiments/enhanced_strategies/test_priority2_multi_moment_gene.py` | Refactor to use `ctx.y` instead of `strategy._y` hack |
| `genetic_importance_scores/pikaia_helpers_new.py` | Add `y` parameter to `run_pikaia` |

## Backward Compatibility Guarantee

```python
# ALL of these continue to work exactly as before:

# No y passed — unsupervised
model = PikaiaModel(population=pop, gene_strategies=[...], max_iter=3)
model.fit()

# y explicitly None — unsupervised
model = PikaiaModel(population=pop, gene_strategies=[...], max_iter=3, y=None)
model.fit()

# Factory still works (no y argument)
strat = GeneStrategyFactory.get_strategy(GeneStrategyEnum.DOMINANT)

# StrategyContext still works (y defaults to None)
ctx = StrategyContext(population=pop, org_fitness=f, gene_fitness=g, ...)
assert ctx.y is None  # True
```

No existing caller needs to change. The feature is purely opt-in.
