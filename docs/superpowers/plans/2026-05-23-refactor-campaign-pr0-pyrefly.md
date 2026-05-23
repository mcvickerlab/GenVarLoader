# Refactor Campaign PR0 — Pyrefly Strict Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Install pyrefly as the project's sole Python type checker (strict preset), wire it into pre-commit + CI, drop basedpyright, and establish a passing baseline across `python/genvarloader/` and `tests/`.

**Architecture:** Pyrefly is configured in `pyproject.toml` under `[tool.pyrefly]` with `preset = "strict"`. The `facebook/pyrefly-pre-commit` hook runs it on every commit (system-language install — uses the pyrefly already installed in the pixi env). A new `lint.yaml` GitHub Actions workflow runs `pyrefly check` on PRs alongside ruff. Errors that strict surfaces are either fixed in this PR (cheap fixes only) or relaxed in `[tool.pyrefly.errors]` with a one-line comment giving the reason; the result is a deliberately-curated baseline that subsequent refactor PRs must continue to satisfy.

**Tech Stack:** pyrefly (via pypi/pixi), prek (the pixi-installed pre-commit runner already in use), GitHub Actions, ruff.

**Campaign context:** This is PR0 of a 9-PR campaign described in `docs/superpowers/specs/2026-05-23-refactor-campaign-design.md`. PRs 1–8 will get their own plan documents drafted as we approach each one — file references in those PRs will shift as the campaign progresses, so writing them all now would produce stale code.

---

## File Structure

**Modified files:**
- `pyproject.toml` — add `[tool.pyrefly]` block; delete `[tool.basedpyright]` block
- `pixi.toml` — add `pyrefly` to dev/test environments as a pypi dependency; add a `typecheck` task
- `.pre-commit-config.yaml` — add the `pyrefly-check` hook
- `.github/workflows/lint.yaml` — **new file** — runs `ruff check` and `pyrefly check` on PRs
- `CLAUDE.md` — replace the `basedpyright` reference with `pyrefly`
- Possibly a small number of `python/genvarloader/**/*.py` and `tests/**/*.py` files where a strict error is cheaper to fix than to relax in config (discovered in Task 5)

**Out of scope:**
- Editing existing plan docs under `docs/superpowers/plans/` that reference basedpyright commands (historical record; do not touch)
- Any source-level refactoring (that's PR1+)

---

## Task 1: Install pyrefly in the dev environment

**Files:**
- Modify: `pixi.toml`

- [ ] **Step 1: Verify pyrefly is available via pypi**

Run:

```bash
pixi run -e dev python -c "import urllib.request, json; print(json.loads(urllib.request.urlopen('https://pypi.org/pypi/pyrefly/json').read())['info']['version'])"
```

Expected: A version string like `0.x.y` or `1.x.y` is printed (confirming pyrefly is installable from pypi). If this fails, stop and report — pyrefly is required for this plan.

- [ ] **Step 2: Add pyrefly to the py310 feature's pypi-dependencies**

In `pixi.toml`, locate the `[feature.py310.pypi-dependencies]` table (around line 83) and add `pyrefly = "*"` to it. The block should look like:

```toml
[feature.py310.pypi-dependencies]
pyarrow = ">=21"
hirola = "==0.3"
seqpro = "==0.11.0"
genoray = "==2.4.0"
polars = "==1.37.1"
loguru = "*"
# ... (other existing deps unchanged)
polars-bio = ">=0.20,<0.21"
pyrefly = "*"
```

`pyrefly` is in py310's pypi-deps (rather than the workspace-level `[dependencies]`) because the workspace deps are conda-only; the py310 feature is inherited by every env that needs typechecking (`default`, `dev`, `py310`–`py313`, `notebook`, `splice`).

- [ ] **Step 3: Refresh the pixi lockfile**

Run:

```bash
pixi lock
```

Expected: lockfile updates with no errors. If pixi reports a resolver conflict, capture the error and stop.

- [ ] **Step 4: Verify pyrefly is now callable**

Run:

```bash
pixi run -e dev pyrefly --version
```

Expected: a version string prints. If "command not found", investigate before proceeding.

- [ ] **Step 5: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "build(deps): add pyrefly to dev environments"
```

---

## Task 2: Add the `[tool.pyrefly]` config with strict preset

**Files:**
- Modify: `pyproject.toml:52-72` (the `[tool.basedpyright]` block — we add pyrefly *above* it for now; basedpyright is removed in Task 7 after pyrefly is fully green)

- [ ] **Step 1: Add the pyrefly config block**

In `pyproject.toml`, insert the following block immediately **before** the existing `[tool.basedpyright]` block (between line 51 `lint.ignore = ["E501"]` and line 52 `[tool.basedpyright]`):

```toml
[tool.pyrefly]
project-includes = ["python/genvarloader", "tests"]
project-excludes = ["**/data/**", "**/__pycache__/**"]
preset = "strict"

# Per-error relaxations are added in Task 5 once the strict baseline is run.
# Each entry below MUST carry a one-line comment giving the reason for relaxation.
[tool.pyrefly.errors]
```

The empty `[tool.pyrefly.errors]` table is a deliberate placeholder — Task 5 populates it.

- [ ] **Step 2: Verify pyrefly reads the config**

Run:

```bash
pixi run -e dev pyrefly check --help
```

Expected: help text prints with no config-related errors.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "build(typecheck): add [tool.pyrefly] strict config"
```

---

## Task 3: Add a `typecheck` task to pixi

**Files:**
- Modify: `pixi.toml:118-127` (the `[tasks]` table)

- [ ] **Step 1: Add the typecheck task**

In `pixi.toml`, locate the workspace `[tasks]` block (line 118) and add a `typecheck` task. The block should become:

```toml
[tasks]
prek-install = "prek install --hook-type commit-msg --hook-type pre-push"
gen = { cmd = "python tests/data/generate_ground_truth.py" }
gen-1kg = { cmd = "python tests/data/generate_1kg_ground_truth.py", depends-on = [
    "gen",
] }
test = { cmd = "pytest tests && cargo test --release", depends-on = [
    "gen",
    "gen-1kg",
] }
typecheck = { cmd = "pyrefly check" }
```

- [ ] **Step 2: Verify the task is registered (without expecting it to pass yet)**

Run:

```bash
pixi run -e dev typecheck 2>&1 | head -20
```

Expected: pyrefly runs and prints errors (because PR0 hasn't established the baseline yet). The errors themselves are fine at this point — we just want to confirm the task is wired up. If the command itself is "not found" or "no such task", fix the config before proceeding.

- [ ] **Step 3: Commit**

```bash
git add pixi.toml
git commit -m "build(typecheck): add `typecheck` pixi task"
```

---

## Task 4: Wire the pre-commit hook

**Files:**
- Modify: `.pre-commit-config.yaml:14-16` (insert after the ruff block, before the local `pixi-lock` block)

- [ ] **Step 1: Add the pyrefly-check hook**

In `.pre-commit-config.yaml`, insert a new repo block immediately after the `ruff-pre-commit` block (line 16) and before the `repo: local` block (line 17). The hook uses `language: system` so it picks up the pyrefly installed by pixi in Task 1:

```yaml
- repo: https://github.com/facebook/pyrefly-pre-commit
  rev: 1.0.0
  hooks:
  - id: pyrefly-check
    name: Pyrefly (type checking)
    pass_filenames: false
    language: system
```

- [ ] **Step 2: Verify the hook is registered**

Run:

```bash
pixi run -e dev prek run pyrefly-check --all-files 2>&1 | tail -10
```

Expected: prek invokes pyrefly. It will currently fail (errors exist) — that is OK; we just need to confirm the hook is wired. Capture roughly how many errors and which files dominate; you'll use this in Task 5.

- [ ] **Step 3: Commit**

```bash
git add .pre-commit-config.yaml
git commit -m "build(pre-commit): wire pyrefly-check hook"
```

---

## Task 5: Establish the strict baseline (the hard part)

This task is **discovery-driven** rather than TDD-shaped. The work is: run pyrefly, classify each error, and decide per-error whether to fix-in-this-PR or relax-in-config.

**Files:**
- Modify: `pyproject.toml` — populate `[tool.pyrefly.errors]` with reasoned relaxations
- Possibly modify: a small number of `python/genvarloader/**/*.py` and `tests/**/*.py` files where the fix is one or two lines

**Decision rule for each error category:**

| If the fix is... | Action |
|---|---|
| One annotation, one cast, or one `assert isinstance(...)` (≤ ~5 lines, no behavior change) | Fix in source. |
| Genuinely structural (would require touching design — e.g. `Ragged` operations, `evolve()` chains, awkward array protocols, numba kernels) | Relax in `[tool.pyrefly.errors]` with a one-line comment. |
| A pre-existing bug | Stop. Surface it to the user. Do **not** silently fix bugs in PR0. |
| Bulk-affects ≥10 sites of the same kind | Relax in config; PR8 will revisit. |

**Forbidden:** Blanket `# type: ignore` sprinkling across the codebase. The whole point of PR0 is to make suppressions explicit and auditable in one config block.

- [ ] **Step 1: Capture the baseline error list**

Run:

```bash
pixi run -e dev pyrefly check 2>&1 | tee /tmp/pyrefly-baseline.txt
```

This produces a snapshot of every strict-mode error pyrefly currently raises. Keep `/tmp/pyrefly-baseline.txt` for reference throughout this task.

- [ ] **Step 2: Group errors by category**

Pyrefly prefixes each error with its category code (e.g. `bad-assignment`, `implicit-any`, `missing-attribute`). Group the baseline output by category:

```bash
grep -oE '\[[a-z-]+\]' /tmp/pyrefly-baseline.txt | sort | uniq -c | sort -rn
```

You will get a table like:
```
142 [implicit-any]
 31 [bad-assignment]
 12 [missing-attribute]
  ...
```

This is your worklist.

- [ ] **Step 3: For each category, apply the decision rule above**

For categories with **many sites** (typically `implicit-any`, `unknown-argument-type`, `unknown-variable-type` if pyrefly emits those — names may differ from basedpyright's), relax in config. Add to `[tool.pyrefly.errors]` in `pyproject.toml`:

```toml
[tool.pyrefly.errors]
# Numpy/awkward/numba interop produces many implicit-Any values that strict
# would flag; tightening these is deferred to PR8 after the campaign settles
# the surrounding code.
implicit-any = "ignore"

# Bulk-affects Ragged operations and evolve() chains where the structural
# type isn't expressible until PRs 1, 3, 5 redesign the relevant classes.
bad-assignment = "warn"
```

Use category names **as pyrefly actually emits them** in your baseline output — do not transcribe the names above blindly; they are illustrative.

For categories with **a small number of sites** where the fix is local and obvious (e.g. missing `-> None` returns, missing `Optional[...]`, a forgotten `cast(...)`), fix them in the relevant source file. **Each such fix MUST be its own commit** with a message naming the file and category, so the history stays auditable.

Example of an in-source fix commit message:

```
fix(types): annotate return -> None on Dataset._validate_*
```

- [ ] **Step 4: Re-run pyrefly until clean**

After applying relaxations and fixes, re-run:

```bash
pixi run -e dev pyrefly check
```

Expected: exit code 0, no errors printed.

- [ ] **Step 5: Sanity-check that the relaxations are minimal**

For each entry you added to `[tool.pyrefly.errors]`, ask: would removing this entry produce more than ~20 errors? If no, remove the entry and either fix the few errors or convert from `"ignore"` to `"warn"`. Goal: the relaxation list is as short as is feasible without expanding PR0's blast radius.

- [ ] **Step 6: Commit the final config**

If you made source fixes earlier (already committed one-by-one), commit the final pyproject.toml relaxations separately:

```bash
git add pyproject.toml
git commit -m "build(typecheck): populate pyrefly strict baseline relaxations"
```

- [ ] **Step 7: Confirm pre-commit now passes end-to-end**

Run:

```bash
pixi run -e dev prek run --all-files
```

Expected: all hooks pass (or only the known pixi-lock pre-push hook is skipped because it's `stages: [pre-push]`).

---

## Task 6: Add the lint CI workflow

**Files:**
- Create: `.github/workflows/lint.yaml`

This implements the lint workflow anticipated by the release-pipeline spec (`docs/superpowers/specs/2026-05-21-release-pipeline-design.md:134`).

**Approach:** The CI job invokes `prek run --all-files` rather than calling ruff / pyrefly individually. This keeps CI mechanically in sync with `.pre-commit-config.yaml` — adding, removing, or reconfiguring a hook locally automatically updates what CI runs. `prek` only runs hooks whose stages include the default `pre-commit` stage, so the existing `pixi-lock` (pre-push) and `commitizen` (commit-msg / pre-push) hooks are naturally skipped.

- [ ] **Step 1: Create `.github/workflows/lint.yaml`**

Write the following content:

```yaml
name: Lint

on:
  pull_request:
    branches:
      - main
  push:
    branches:
      - main
      - stable
  workflow_dispatch:

jobs:
  lint:
    runs-on: ubuntu-latest
    name: "pre-commit hooks"
    steps:
      - name: Check out
        uses: actions/checkout@v4
        with:
          fetch-depth: 0
      - name: Setup pixi
        uses: prefix-dev/setup-pixi@v0.9.5
        with:
          pixi-version: v0.68.1
          cache: true
          environments: dev
          locked: false
      - name: Run pre-commit hooks
        run: pixi run -e dev prek run --all-files --show-diff-on-failure
```

- [ ] **Step 2: Validate the YAML locally**

Run:

```bash
pixi run -e dev python -c "import yaml; yaml.safe_load(open('.github/workflows/lint.yaml'))"
```

Expected: no exception.

- [ ] **Step 3: Confirm `prek run --all-files` matches what CI will execute**

Run the same command the workflow will run:

```bash
pixi run -e dev prek run --all-files --show-diff-on-failure
```

Expected: exit code 0. The hooks that should execute at default (pre-commit) stage are the pre-commit-hooks suite, ruff-check, ruff-format, and pyrefly-check. The `pixi-lock` and `commitizen*` hooks should be silently skipped because their stages don't include `pre-commit`.

If any hook fails, fix the underlying issue (do **not** add `--hook-stage` overrides to dodge it). If a stage-gated hook unexpectedly runs, fix the stage config in `.pre-commit-config.yaml`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/lint.yaml
git commit -m "ci: add lint workflow (delegates to prek/.pre-commit-config.yaml)"
```

---

## Task 7: Remove basedpyright

basedpyright is only removed *after* pyrefly is fully green and in CI — this keeps PR0 reversible if pyrefly turns out to be intolerable.

**Files:**
- Modify: `pyproject.toml:52-72` — delete the `[tool.basedpyright]` block
- Modify: `CLAUDE.md:161` — replace the `basedpyright` command line with `pyrefly`

- [ ] **Step 1: Delete the basedpyright config**

In `pyproject.toml`, delete the entire block from line 52 (`[tool.basedpyright]`) through line 72 (`reportUninitializedInstanceVariable = false`). After deletion, the section between `[tool.ruff]` and `[tool.maturin]` should contain only the pyrefly config added in Task 2.

- [ ] **Step 2: Update CLAUDE.md**

In `CLAUDE.md`, find the line (currently around line 161):

```
pixi run -e dev basedpyright python/
```

Replace with:

```
pixi run -e dev typecheck
```

Also scan the surrounding paragraph for any other references to basedpyright and replace with pyrefly (the section header is likely "# Lint" or similar — update wording for consistency).

- [ ] **Step 3: Confirm the project still type-checks**

Run:

```bash
pixi run -e dev typecheck
```

Expected: exit code 0.

- [ ] **Step 4: Run the full test suite as a sanity check**

Run:

```bash
pixi run -e dev test
```

Expected: all tests pass. (This guards against any source-level changes from Task 5 having introduced a regression. It will take several minutes.)

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml CLAUDE.md
git commit -m "build(typecheck): drop basedpyright (replaced by pyrefly)"
```

---

## Task 8: Final verification

- [ ] **Step 1: Confirm all gates green**

Run these in order:

```bash
pixi run -e dev prek run --all-files --show-diff-on-failure
pixi run -e dev test
```

Each command must exit 0 (the pre-push `pixi-lock` and `commitizen*` hooks are stage-gated and will be skipped by `prek run --all-files`). `prek` covers ruff and pyrefly. If anything fails, stop and investigate before opening the PR.

- [ ] **Step 2: Open the PR**

Use the project's normal PR conventions (conventional-commit title, brief body summarizing PR0's scope, link to the campaign spec). Do **not** ask the agent to push or open the PR — the user opens PRs themselves.

---

## Self-Review Checklist

Before declaring PR0 complete:

- [ ] `[tool.pyrefly]` is the only type-checker config in `pyproject.toml`
- [ ] `[tool.pyrefly.errors]` only contains entries that are each accompanied by a one-line `# comment` giving the reason
- [ ] No new `# type: ignore` lines were sprinkled into source files; any added ones (≤5 across the whole PR) have a `# type: ignore[<code>]` form with a comment
- [ ] No basedpyright references remain in `pyproject.toml`, `pixi.toml`, `CLAUDE.md`, `.github/workflows/`, or `.pre-commit-config.yaml`
- [ ] (Historical plan docs under `docs/superpowers/plans/` are untouched — they're frozen records)
- [ ] CI workflow `.github/workflows/lint.yaml` runs both ruff and pyrefly
- [ ] `pixi run -e dev typecheck` exits 0 locally
- [ ] `pixi run -e dev test` exits 0 locally

---

## What comes next

After PR0 merges, the next plan is **PR1: `DatasetSettings` value object**. It will be drafted in `docs/superpowers/plans/YYYY-MM-DD-refactor-campaign-pr1-dataset-settings.md` at that time, when `_dataset/_impl.py` line numbers are known to be current. Same pattern for PR2–PR8.
