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

Open a PR that:

1. Bumps `project.version` in `pyproject.toml`.
2. Runs `uv lock` so the lockfile stays in sync (CI enforces `uv lock --locked`).
3. Adds a `## [X.Y.Z] - YYYY-MM-DD` section to `CHANGELOG.md` summarising what changed, and appends a comparison link at the bottom (e.g. `[X.Y.Z]: https://github.com/danube-ai/pikaia/compare/vX.Y.(Z-1)...vX.Y.Z`).

```bash
git checkout -b release/0.3.0
# edit pyproject.toml: version = "0.3.0"
# edit CHANGELOG.md: add [0.3.0] section and comparison link
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

- publishes `0.3.0` to production PyPI,
- deploys the versioned docs (`mike deploy 0.3.0 latest`) and sets `latest` as the default, and
- **creates the GitHub Release automatically** — a `Create GitHub Release` job reads the matching `## [0.3.0]` section from `CHANGELOG.md` and publishes it as the release notes for the `v0.3.0` tag.

> **Do not create the GitHub Release by hand.** Pushing the tag is the single trigger; the release is generated from the changelog so every release is consistent. If a release object already exists for the tag (e.g. someone created it in the UI), the job **reconciles** its notes from the changelog rather than failing — but the tag push, not the UI, is the canonical way to cut a release.

### 5. Verify production

```bash
pip install pikaia==0.3.0
```

Confirm the [GitHub Releases page](https://github.com/danube-ai/pikaia/releases) shows `v0.3.0` with the changelog notes.

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
| Push `v*` tag | tag push | PyPI + versioned docs + GitHub Release | Manual approval on `pypi` environment |

The GitHub Release is created automatically from the `CHANGELOG.md` section once the
PyPI publish succeeds — there is no manual "draft a release" step.

---

## Why trunk-based, and not GitFlow?

pikaia previously used a **GitFlow**-style model: a long-lived `develop` branch for
integration and a separate `main` branch for releases, with periodic `develop → main`
"sync" pull requests. We deliberately moved away from it. Here's the reasoning.

### The problem with the two-branch model

GitFlow keeps two permanent branches in sync by merging one into the other. In
practice this is fragile:

- **Persistent divergence.** Every `develop → main` sync merged through the GitHub UI
  creates a *new* merge/squash commit on `main` with a SHA that `develop` has never
  seen. The moment the sync lands, the branches have diverged again — so the *next*
  sync starts with a conflict, and it compounds over time.
- **Recurring merge conflicts.** Because the histories never truly converge, routine
  syncs repeatedly conflict on the same files (e.g. docs, changelogs), and resolving
  them on a protected, linear-history branch often forces awkward workarounds.
- **Release ≠ merge.** Tying "publish to PyPI" to "merge into `main`" means the branch
  topology *is* the release mechanism. Any merge accident becomes a publish accident,
  and PyPI uploads are irreversible.
- **Overhead with little benefit.** For a library with a linear release cadence (no
  parallel maintenance of many released versions), the second long-lived branch adds
  ceremony without buying isolation you actually use.

Notably, GitFlow's own author added a
[reflection note](https://nvie.com/posts/a-successful-git-branching-model/) recommending
*against* it for teams doing continuous delivery of a single versioned product —
exactly pikaia's situation.

### The trunk-based model we adopted

- **One long-lived branch (`main`).** Short-lived feature branches merge into it and are
  deleted. There is no second branch to keep in sync, so the divergence/conflict cycle
  simply cannot happen.
- **Releases are tags, not merges.** A production release is an explicit, intentional act
  — pushing a `v*` tag — decoupled from day-to-day merges. The tag is an immutable
  pointer to an exact commit, which is a natural fit for "this is version X.Y.Z".
- **Continuous validation.** Every merge to `main` still exercises the full build and
  publishes to TestPyPI, so integration problems surface immediately rather than at
  release time.
- **A real gate where it matters.** The one irreversible step — uploading to production
  PyPI — sits behind a manual approval on the `pypi` environment, instead of being an
  implicit side effect of a branch merge.

### This is the mainstream approach for Python libraries

The Python libraries pikaia takes as models all release from a **single branch + tags**,
not GitFlow:

- [**pydantic**](https://github.com/pydantic/pydantic) — trunk on `main`
- [**FastAPI**](https://github.com/fastapi/fastapi) — trunk on `master`
- [**Typer**](https://github.com/fastapi/typer) — trunk on `master`
- [**SQLModel**](https://github.com/fastapi/sqlmodel) — trunk on `main`

None of them keep a long-lived `develop` integration branch: feature branches merge
straight into the trunk and releases are cut from tags. Where long-lived side branches
exist at all, they are `x.y`-style **maintenance** branches for backporting fixes to
*already-released* versions — not a parallel integration branch. Tag-triggered publishing
(`on: push: tags: ['v*']`) is likewise the pattern used by tooling such as
[uv](https://github.com/astral-sh/uv) and [ruff](https://github.com/astral-sh/ruff).

If pikaia ever needs to support multiple released major versions simultaneously, the
right addition is a `x.y`-maintenance branch for backports — **not** a return to a
`develop` integration branch.
