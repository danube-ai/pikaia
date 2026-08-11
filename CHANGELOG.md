# Changelog

All notable changes to **pikaia** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.3] - 2026-08-11

### Fixed

- **Zero org-fitness range no longer raises.** When features are perfectly
  inversely symmetric (balanced anti-correlations that cancel under uniform
  gene weighting), all organisms receive equal initial fitness and the model
  cannot distinguish them. Previously this raised `ValueError("All organism
  fitness values are 0")` — a misleading message, since the values are not
  zero, just equal. Now a `WARNING` is logged instead and `fit()` returns
  early, leaving scores at the initial uniform values. This matches the
  behaviour intended: no ranking is attempted when no ranking is possible.
  The corner case is extremely unlikely in practice for large populations but
  can appear with small, carefully constructed examples such as two items
  with perfectly anti-correlated features.

## [0.3.2] - 2026-08-10

### Fixed

- **Trading strategy multi-iteration accuracy.** All three sell/buy pairs now
  hold their intended dynamics across many iterations, not just the first:
  - **BuyHard / BuyUniform / BuyEasy** now return a *proportional* delta
    (`buy_abs / γ_j`) instead of an absolute one.  Without this, the
    multiplicative replicator diverges from the intended additive dynamics after
    a single iteration, collapsing genes to extrema by k ≈ 10.
  - **SellUniform** now zeroes its sell signal for genes where `excl_j ≈ 0`
    (all organisms solved it) or `excl_j ≈ 1` (none solved it), so trivially
    solved/failed genes no longer drain value.
  - **BuyUniform** capital now excludes genes with trivial exclusiveness (same
    mask as SellUniform), keeping per-organism capital ratios correct.

### Added

- **100-iteration convergence tests** for all three trading pairs:
  - `SellHard + BuyHard`: stable match at k = 1 … 100 (atol 1e-5).
  - `SellUniform + BuyUniform`: stable match at k = 1 … 100 (atol 1e-5).
  - `SellEasy + BuyEasy`: match at k = 1 … 10 (rtol 1e-4); capped at k = 10
    because this inverse pair is inherently divergent (values grow
    exponentially).

## [0.3.1] - 2026-08-07

### Added

- **`VarianceGeneStrategy`** (`VARIANCE`) — rewards genes with high
  cross-organism dispersion (variance of expression across the population).
  Registered in `GeneStrategyEnum` and `GeneStrategyFactory`; unit tests and
  `kernel()` included.

## [0.3.0] - 2026-08-04

### Changed

- **Release process moved to a single-branch (trunk-based) model.** `main` is
  now the only long-lived branch. Every merge to `main` publishes to TestPyPI
  and updates the `dev` docs; **production PyPI releases are cut by pushing a
  `v*` git tag**, gated by a manual approval on the `pypi` environment. This
  replaces the previous GitFlow-style `develop → main` sync flow. See
  [`docs/releasing.md`](docs/releasing.md).

### Added

- **`docs/releasing.md`** — maintainer release guide covering the day-to-day
  contribution flow, cutting a release via a version tag, the `pypi` approval
  gate, and the rationale for trunk-based over GitFlow. Added to the docs nav.

### Removed

- Retired the `develop` branch and the `develop → main` sync workflow.
- Removed the CI `version-check` job (a per-PR version bump is no longer
  required) and the `tag-release` job (tags are now created by maintainers, not
  auto-generated), plus the now-unused `check_version.py`.

> This release contains no functional changes to the `pikaia` package itself; it
> marks the release-infrastructure overhaul.

## [0.2.10] - 2026-08-03

### Changed

- Documentation deploy now redirects the site root URL to the latest released
  version on `main` deploys (#29).

## [0.2.9] - 2026-08-03

### Changed

- Cleaned up supervision documentation, corrected the strategy tables and
  README, and added a supervised-mode note (#27).

### Removed

- `adaptive_supervision.md`, superseded by the consolidated supervision docs
  (#26).

## [0.2.8] - 2026-08-03

### Changed

- **Sell strategy relocated** from a gene strategy to an organism strategy
  (`pikaia/strategies/os_strategies/sell_strategy.py`); strategy docstrings and
  type annotations cleaned up across the package (#24).

## [0.2.7] - 2026-08-03

### Added

- **MkDocs Material documentation site** with `mike` version management and
  `mkdocstrings`-generated API reference, replacing the Sphinx build (#24).

## [0.2.6] - 2026-07-30

### Added

- **Information-theoretic and redundancy-aware gene strategies** — four new
  `GeneStrategy` implementations validated across 65 experiments on 18
  datasets (see `genetic_importance_scores` research report):
  - **EntropyMax** (`ENTROPY_MAX`) — supervised strategy combining mutual
    information with the target and differential entropy.  Converges within
    5 iterations and matches the best supervised feature selectors (0.985
    mean accuracy agreement under nested cross-validation).  Requires ``y``.
  - **OrthoGene** (`ORTHO_GENE`) — promotes features with low pairwise
    correlation (high orthogonality).  Works unsupervised; in supervised mode
    appends ``y`` to the correlation matrix to bias towards features
    uncorrelated with each other *and* the target.
  - **PartialCorr** (`PARTIAL_CORR`) — rewards features whose relationship
    with the target survives controlling for all other features, estimated
    via a shrinkage precision matrix.  Supervised; falls back to zeros
    without ``y``.
  - **RedundancyPenalty** (`REDUNDANCY_PENALTY`) — suppresses features highly
    correlated with their peers; unsupervised by default, supervised when
    ``y`` is provided.
- All four strategies registered in `GeneStrategyEnum` and
  `GeneStrategyFactory`, with unit tests and `kernel()` implementations.
- **Three new examples** showcasing validated results from the audit:
  - `example7_stability.py` — bootstrap Jaccard stability (Pikaia-DOM-BAL vs MI)
  - `example8_entropymax_fast_selection.py` — EntropyMax convergence at 5 iterations
  - `example9_archetypal_organisms.py` — SELFISH organism archetype detection

## [0.2.5] - 2026-07-28

### Added

- **Trading strategies** (RewardHard, RewardEasy, ValuationBlend) — three new
  `GeneStrategy` implementations:
  - **RewardHard** (`REWARD_HARD`) — rewards features that are hard to achieve
    (low mean expression)
  - **RewardEasy** (`REWARD_EASY`) — the exact inverse of RewardHard, rewards
    easy-to-express features
  - **ValuationBlend** (`VALUATION_BLEND`) — blends between RewardHard and
    RewardEasy via a `preference` parameter in `[0, 1]`
- Registered all three strategies in `GeneStrategyEnum` and
  `GeneStrategyFactory`, with unit tests (name, call, kernel, edge cases,
  integration) and example in `examples/example6.py`
- **D-matrix comparison benchmark** updated to cover all 8 gene strategies
  (40 strategy combinations instead of 25)

### Changed

- `d_matrix_comparison.py` and strategy documentation now reflect 8 gene
  strategies / 40 combinations (was 5 strategies / 25 combinations)

## [0.2.4] - 2026-07-20

### Added

- **Adaptive supervision** — an optional target `y` is threaded through
  `StrategyContext` and every strategy `kernel()`, enabling supervised strategy
  behaviour while remaining fully optional for unsupervised use (#18).

## [0.2.3] - 2026-07-20

### Added

- Groundwork for adaptive supervision, released together with 0.2.4 (#18).

## [0.2.2] - 2026-05-29

### Changed

- Updated README status shields; version bump (#12).

## [0.2.1] - 2026-05-29

### Changed

- Synced `main` with `develop`; version bump (#10).

## [0.2.0] - 2026-05-29

### Added

- **D-matrix acceleration path** for `PikaiaModel` (`use_d_matrix=True`), with
  `kernel()` methods implemented across every strategy in `pikaia.strategies`.
- **`GeneticModel` base class** and a `DanubeModel` stub
  ([`pikaia/models/danube_model.py`](pikaia/models/danube_model.py)) to formalise
  the hierarchy of evolutionary models.
- **Genetic-attention research track** under
  [`research/hybrid_ai/genetic_attention/`](research/hybrid_ai/genetic_attention/):
  six iterative attempts at attaching the genetic kernel to encoder-only
  transformers (BERT), culminating in a full MS-MARCO ablation (2048 steps)
  with results reported in `ABLATION_REPORT.md`.
- **Hybrid-AI white paper** under
  [`research/hybrid_ai/_white_paper/`](research/hybrid_ai/_white_paper/),
  including verified DOI links and scientific-style prose.
- **D-matrix comparison benchmark** ([`examples/d_matrix_comparison.py`](examples/d_matrix_comparison.py))
  and dedicated **arXiv example** ([`examples/arxiv_example.py`](examples/arxiv_example.py)).
- **CI/CD pipeline** (`.github/workflows/`):
  - PR gate: unit tests with ≥95% coverage, semver version-bump check, and
    pre-merge query against PyPI/TestPyPI to refuse duplicate versions.
  - Publish pipeline: SHA-pinned actions, `twine check --strict`, GitHub
    Environments for `testpypi` (auto) and `pypi` (manual approval),
    artifact handoff between build and publish jobs, auto-tagging of
    releases, `print-hash` for auditable upload logs, and per-branch
    concurrency control.
  - `uv lock --locked` check to prevent unsynced lockfile drift.
  - Dependabot configuration for weekly grouped GitHub-Actions SHA bumps.

### Changed

- **Python requirement bumped to `>=3.14`**; all dependencies upgraded.
- Restructured workspace: research experiments moved from `examples/` to
  [`research/hybrid_ai/`](research/hybrid_ai/); standard examples flattened
  from `examples/standard/` to [`examples/`](examples/); per-example artefacts
  isolated under `examples/artefacts/{example_name}/`.
- Improved input validation in `Population` (numeric + NaN guards) and in
  `PikaiaModel` (stricter mixing-coefficient checks).
- Sphinx documentation build is functional with AutoAPI auto-generation.
- Documentation overhaul: every Markdown file in the repository reviewed
  for accuracy, cross-references fixed, README updated with index links to
  `examples/` and `research/`, and reference notes cleaned of HTML artefacts
  (`&nbsp;` entities, broken external paths).

### Fixed

- Broken module imports across the package after the examples/research split.

### Testing

- Unit-test coverage expanded to **≥99%** across the `pikaia` package;
  strategy-kernel tests added; model and population tests updated.

## [0.1.0] - 2025-10-06

Consolidated release migrating internal development work into the public
repository. Includes the genetic-layer architecture with input projection,
refreshed examples, updated README with preprint links, and improved mixing-
coefficient validation in `PikaiaModel`.

## [0.0.3] - 2025-05-05

Maintenance release.

## [0.0.2] - 2025-02-06

Initial public release.

[0.3.3]: https://github.com/danube-ai/pikaia/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/danube-ai/pikaia/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/danube-ai/pikaia/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/danube-ai/pikaia/compare/v0.2.10...v0.3.0
[0.2.10]: https://github.com/danube-ai/pikaia/compare/v0.2.6...v0.2.10
[0.2.6]: https://github.com/danube-ai/pikaia/compare/v0.2.4...v0.2.6
[0.2.4]: https://github.com/danube-ai/pikaia/compare/v0.2.2...v0.2.4
[0.2.2]: https://github.com/danube-ai/pikaia/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/danube-ai/pikaia/compare/v0.1.0...v0.2.1
[0.1.0]: https://github.com/danube-ai/pikaia/releases/tag/v0.1.0
[0.0.3]: https://github.com/danube-ai/pikaia/releases/tag/v0.0.3
[0.0.2]: https://github.com/danube-ai/pikaia/releases/tag/v0.0.2
