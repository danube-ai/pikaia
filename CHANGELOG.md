# Changelog

All notable changes to **pikaia** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Slated for release as `0.2.0`.

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
- Sphinx documentation build now produces **zero warnings, zero errors**.
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

[Unreleased]: https://github.com/danube-ai/pikaia/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/danube-ai/pikaia/releases/tag/v0.1.0
[0.0.3]: https://github.com/danube-ai/pikaia/releases/tag/v0.0.3
[0.0.2]: https://github.com/danube-ai/pikaia/releases/tag/v0.0.2
