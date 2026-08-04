# Releasing pikaia

pikaia follows a **single-branch (trunk-based)** model:

- **`main`** is the one long-lived branch. All work lands here via pull requests from short-lived feature branches.
- Every push to `main` (i.e. every merged PR) publishes to **[TestPyPI](https://test.pypi.org/project/pikaia/)** and deploys the `dev` docs — continuous validation that the package builds, installs, and documents cleanly.
- **Production [PyPI](https://pypi.org/project/pikaia/) releases are triggered by pushing a `v*` git tag** — never by a branch merge. The tag is the deliberate, explicit release gate.

There is no `develop` branch. There is no `develop → main` sync. Releases are tags on `main`.

---

## Day-to-day contribution flow

```
feature branch  ──PR──▶  main   ──▶  TestPyPI + dev docs   (automatic, every merge)
```

1. Branch off `main`: `git checkout -b feat/my-change`
2. Open a PR into `main`. CI runs the test suite.
3. Merge (squash or rebase — `main` requires linear history).
4. On merge, the package is published to TestPyPI (skipped automatically if the current version already exists there) and the `dev` docs are updated.

No version bump is required for regular PRs. Bumping the version is a **release** action (below).

---

## Cutting a production release

```
git tag v0.3.0  ──push──▶  PyPI + versioned docs   (manual approval gate)
```

### 1. Bump the version

Open a PR that bumps `project.version` in `pyproject.toml` and run `uv lock` so the lockfile stays in sync (CI enforces `uv lock --locked`).

```bash
git checkout -b release/0.3.0
# edit pyproject.toml: version = "0.3.0"
uv lock
git commit -am "chore: bump version to 0.3.0"
```

Open the PR, get it reviewed, and merge into `main`.

### 2. Verify on TestPyPI

Merging the bump publishes `0.3.0` to TestPyPI automatically. Confirm it installs cleanly before releasing to production:

```bash
pip install -i https://test.pypi.org/simple/ \
            --extra-index-url https://pypi.org/simple/ \
            pikaia==0.3.0
```

### 3. Tag the release

Tag the merge commit on `main` (make sure your local `main` is up to date first):

```bash
git checkout main
git pull
git tag v0.3.0
git push origin v0.3.0
```

The tag **must** point at a commit whose `pyproject.toml` already contains the matching version — the publish and docs jobs read the version from `pyproject.toml` at the tagged commit.

### 4. Approve the deployment

The tag push starts the **Publish** workflow. The production `pypi` environment has a **required-reviewer gate**: the workflow pauses at the "Publish to PyPI" job until a maintainer approves it in the GitHub Actions run. This is the last checkpoint before an irreversible upload — nothing reaches production PyPI without it.

Approve it, and the workflow:

- publishes `0.3.0` to production PyPI, and
- deploys the versioned docs (`mike deploy 0.3.0 latest`) and sets `latest` as the default.

### 5. Verify production

```bash
pip install pikaia==0.3.0
```

---

## Version numbers and tags

- The tag name is `v` + the `pyproject.toml` version, e.g. version `0.3.0` → tag `v0.3.0`.
- `check_pypi_version.py` runs in the build step and **refuses to publish a version that already exists** on the target index — PyPI uploads are irreversible, so this guards against accidental duplicates.
- To release a new version, always bump `pyproject.toml` first (step 1), then tag (step 3).

---

## Summary

| Action | Trigger | Target | Gate |
|---|---|---|---|
| Merge PR to `main` | push to `main` | TestPyPI + `dev` docs | CI (tests) |
| Push `v*` tag | tag push | PyPI + versioned docs | Manual approval on `pypi` environment |
