# Contributing to pikaia

Thank you for your interest in contributing! All kinds of contributions are welcome — bug reports, documentation improvements, new strategies, and feature ideas.

Please read our [Code of Conduct](CODE_OF_CONDUCT.md) before participating.

---

## Ways to contribute

- **Report a bug** — open a [GitHub issue](https://github.com/danube-ai/pikaia/issues) with a minimal reproducible example.
- **Suggest a feature** — open an issue describing the use case.
- **Fix a bug or add a feature** — fork the repo, make your changes on a branch, and open a pull request.
- **Improve documentation** — typos, clarifications, and new examples are always appreciated.

## Development setup

```bash
git clone https://github.com/danube-ai/pikaia.git
cd pikaia
uv sync --extra dev --extra examples
```

Run the test suite:

```bash
uv run pytest tests/unit/ --cov=pikaia --cov-report=term-missing
```

## Adding a new strategy

The most common extension point is a new evolutionary strategy. See the detailed step-by-step guide in [docs/contributing.md](docs/contributing.md), which covers implementation, registration, testing, and the D-matrix kernel interface.

## Pull request guidelines

- Keep PRs focused — one logical change per PR.
- Add or update tests for any code you change.
- Run `uv run pytest` and fix any failures before opening a PR.
- Pre-commit hooks (ruff lint, ruff format) run automatically on commit.

## Questions?

Open a [GitHub issue](https://github.com/danube-ai/pikaia/issues) or contact us at the addresses listed in the README.
