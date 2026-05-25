# Test Suite Overhaul — Phase 7 CI Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a coverage CI job that publishes the `htmlcov/` HTML report (and `coverage.xml`) as a workflow artifact on every PR and main-branch push.

**Architecture:** A new job `coverage` is added to `.github/workflows/test.yaml`, running in parallel with the existing pytest matrix. It pins to one Python environment (`py312`), runs the existing `test-cov` pixi task (which produces term + HTML + XML reports), and uploads `htmlcov/` and `coverage.xml` via `actions/upload-artifact@v4`. The existing pytest matrix (py310–py313) is untouched.

**Tech Stack:** GitHub Actions, prefix-dev/setup-pixi, actions/upload-artifact, existing `test-cov` pixi task.

---

## Existing state

- `.github/workflows/test.yaml` runs pytest + cargo test across py310/py311/py312/py313 via `pixi run -e ${{ matrix.environment }} test`.
- `pixi.toml` defines `test-cov` as `pytest tests --cov --cov-report=term-missing --cov-report=html --cov-report=xml`, which depends-on `gen` and `gen-1kg` (matches the existing `test` task's dependencies).
- `pytest-cov` is in the top-level `[dependencies]` of `pixi.toml` — available in every environment.
- `[tool.coverage.run]` already targets `python/genvarloader` with `branch = true` and omits `tests/` and `_builders/`.

## File Structure

- **Modify:** `.github/workflows/test.yaml` — add a new `coverage` job after the existing `pytest` job.
- **No source changes.** No `pixi.toml` changes. No `pyproject.toml` changes.

---

### Task 1: Add the coverage job

**Files:**
- Modify: `.github/workflows/test.yaml`

- [ ] **Step 1: Read the current workflow**

Confirm the current file matches expectations (one `pytest` job, no `coverage` job yet):
```bash
cat .github/workflows/test.yaml
```

Expected: 36 lines, single `pytest` job with py310/py311/py312/py313 matrix.

- [ ] **Step 2: Append the coverage job**

Edit `.github/workflows/test.yaml`. After the final line of the existing `pytest` job (after the `Test` step that runs `pixi run -e ${{ matrix.environment }} test`), append the following job (a peer of `pytest`, indented at the same level under `jobs:`):

```yaml
  coverage:
    runs-on: ubuntu-latest
    name: "coverage (py312)"
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
          environments: py312
          locked: false
      - name: Run pytest with coverage
        run: pixi run -e py312 test-cov
      - name: Upload HTML coverage report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: htmlcov
          path: htmlcov/
          if-no-files-found: error
      - name: Upload coverage.xml
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: coverage.xml
          if-no-files-found: error
```

Notes on design choices:
- **Pinned to py312** — most modern stable Python; matches the `docs` env's Python pin. Single-version coverage is sufficient; coverage results don't meaningfully differ across Python versions for this codebase.
- **`if: always()` on both upload steps** — even if pytest fails, we want to inspect the coverage artifact to understand what *did* run. If test-cov errors out before producing artifacts, `if-no-files-found: error` makes the step fail loudly rather than silently uploading nothing.
- **Two separate artifacts (`htmlcov` and `coverage-xml`)** — kept separate so consumers can fetch the cheaper XML without downloading the full HTML tree, and so a future Codecov integration (out of scope here) can pull just the XML.
- **`name: "coverage (py312)"`** — display name matches the existing `"pytest on Python ${{ matrix.environment }}"` pattern.
- **`fetch-depth: 0`** — matches the existing pytest job; safe default for any downstream tooling that wants full history.
- **No `cargo test`** — coverage is for Python only; the Rust tests run in the existing pytest job.

- [ ] **Step 3: Verify the YAML parses**

Run:
```bash
pixi run -e dev python -c "import yaml; yaml.safe_load(open('.github/workflows/test.yaml')); print('ok')"
```

Expected: `ok`. If you get a YAML parse error, the indentation of the appended job is wrong — both `pytest:` and `coverage:` must be at the same indent level under `jobs:`.

- [ ] **Step 4: Lint with actionlint (if available); else skip**

Run:
```bash
which actionlint && actionlint .github/workflows/test.yaml || echo "actionlint not installed; skipping (CI's own validation will catch issues)"
```

Expected: either passes or skips. If actionlint reports issues, fix them inline.

- [ ] **Step 5: Local smoke test of the test-cov task**

Run:
```bash
pixi run -e dev test-cov 2>&1 | tail -20
```

Expected: pytest run completes; the last lines should show coverage summary (~63% per the Phase 3 baseline). Confirm `htmlcov/` and `coverage.xml` were created:
```bash
ls -d htmlcov coverage.xml
```

Expected: both exist. (If `htmlcov/` was created by a prior local run, this just confirms it's still being produced.)

Note: this runs in the `dev` environment locally (which is py310) rather than `py312` to avoid forcing a fresh env install in the local worktree. The CI run will exercise the actual `py312` path.

- [ ] **Step 6: Commit**

Run:
```bash
git add .github/workflows/test.yaml
git status --short
git commit -m "$(cat <<'EOF'
ci(test): add coverage job uploading htmlcov + coverage.xml

Runs the existing test-cov pixi task on py312 in parallel with the
pytest matrix. Publishes htmlcov/ and coverage.xml as separate
GitHub Actions artifacts; both upload with if-no-files-found=error
so a silently-empty artifact never lands.

Closes the test-suite-overhaul Phase 7 (CI report).
EOF
)"
```

Expected: commit succeeds. (Pre-commit hooks shouldn't touch workflow files; verify the only change is `.github/workflows/test.yaml`.)

---

### Task 2: Update the status doc — declare the overhaul complete

**Files:**
- Modify: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md`

- [ ] **Step 1: Update header `As of` line**

Run:
```bash
git log -1 --format=%h
```

Replace the hash after `committed through` with this value.

- [ ] **Step 2: Add Phase 7 plan to "Authoritative reference files"**

After the `phase5-dataset-polymorphism.md` line, append:
```
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase7-ci-coverage.md`
```

- [ ] **Step 3: Add a row to "What shipped"**

Append after the Phase 6 row:
```
| 7 CI coverage | new `coverage` job in .github/workflows/test.yaml | Runs `test-cov` on py312 in parallel with the pytest matrix; uploads `htmlcov/` and `coverage.xml` as GHA artifacts. Closes Phase 7. |
```

- [ ] **Step 4: Replace the "Recommended next plan" section with a closeout**

The overhaul is complete. Replace the whole `## Recommended next plan` section with:

```markdown
## Status: complete

All seven phases of the test-suite overhaul have shipped on this branch. Summary:

- **Tiers:** `tests/unit/` (167 fast tests, no project fixtures) and `tests/integration/` (184 tests using real on-disk artifacts).
- **Builders:** `tests/_builders/` provides `make_ragged_seqs`, `make_ragged_intervals`, `make_tracks` for unit-tier scaffolding.
- **Conftest:** centralized path fixtures + a session-scoped `reference` fixture (metadata-only Reference is cheap; see gotcha 9).
- **CI:** htmlcov + coverage.xml artifacts published on every PR and main push (Phase 7).
- **Deferred:** `make_dataset` builder + `test_ds_indexing` port (no consumer beyond one test; YAGNI).

This document remains a useful resume note for anyone touching the test suite. Re-read it before adding new tests so you place them in the right tier and reuse the conftest fixtures / builders rather than re-inventing them.
```

- [ ] **Step 5: Commit**

Run:
```bash
git add docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase7-ci-coverage.md
git status --short
git commit -m "docs(status): test-suite overhaul complete (Phase 7 landed)"
```

Expected: commit succeeds.

---

## Self-Review Checklist

- **Spec coverage:** Phase 7 per the status doc was "Wire `htmlcov/` upload into CI." Task 1 does exactly that, plus the trivially-free `coverage.xml` upload that the existing `test-cov` task already produces. ✅
- **Existing pytest matrix untouched:** The new `coverage` job is a peer, not a modification. ✅
- **No placeholders:** Every step shows exact YAML/commands; one local smoke test step verifies the pixi task still works before CI runs it. ✅
- **CI safety:** Uses pinned action versions (`@v4`, `@v0.9.5`) matching the existing job. No new secrets required. `if-no-files-found: error` prevents silent empty-artifact uploads.
- **Reversibility:** A bad job definition is reverted with a single-file revert. The change does not gate merge on coverage thresholds or block any existing path. The existing `pytest` job remains the source-of-truth for pass/fail.
- **Closeout doc:** Task 2 declares the overhaul done, summarizes the final state, and records the deferral. Future readers picking up the test suite know where things land and why.
