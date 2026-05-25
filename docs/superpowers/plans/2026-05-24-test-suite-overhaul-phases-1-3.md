# Test Suite Overhaul (Phases 1–3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the foundational pieces of the test-suite overhaul — coverage tooling, two-tier directory layout, centralized conftest fixtures, and the test audit deliverable that drives subsequent rewrite phases.

**Architecture:** Restructure `tests/` into `unit/`, `integration/`, and a shared `_builders/` package (empty at this stage). Move existing tests under `integration/` wholesale, centralize on-disk `.gvl` and reference paths into `tests/conftest.py`, configure `pytest-cov` for branch coverage reporting (no failing gate), and produce a per-test audit document classifying every existing test as delete/port/keep.

**Tech Stack:** pytest, pytest-cov, pytest-cases (already in `pixi.toml`), pixi tasks, Python 3.10+ paths.

**Scope:** This plan covers spec Phases 1–3 only. Phases 4–7 (delete pass, builders + unit tier, integration trim, CI report) get their own plans informed by the audit produced here.

**Reference spec:** `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`

---

## Conventions used throughout this plan

- All commands assume working directory `/Users/david/projects/GenVarLoader`.
- Run pytest via `pixi run -e dev pytest ...` unless noted.
- The current pixi `test` task depends on `gen` and `gen-1kg`, which materialize `tests/data/*.gvl` directories. If those directories already exist, you can run pytest directly without regenerating.
- Commits use Conventional Commits. The project uses commitizen — no `BREAKING CHANGE:` footers for refactors.
- The suite must remain green after every commit in this plan.

---

## Task 1: Add pytest-cov configuration to pyproject.toml

**Files:**
- Modify: `pyproject.toml` (append new `[tool.coverage.*]` sections after the existing `[tool.pytest.ini_options]` block)

- [ ] **Step 1: Read current pyproject sections**

Run: `grep -n "^\[tool\." pyproject.toml`
Expected: shows existing `[tool.ruff]`, `[tool.pyrefly]`, `[tool.pytest.ini_options]`, `[tool.commitizen]`, `[tool.maturin]` headers.

- [ ] **Step 2: Append coverage config**

After the `markers = [...]` line in `[tool.pytest.ini_options]` (around line 126), insert two new top-level sections **before** `[tool.commitizen]`:

```toml
[tool.coverage.run]
source = ["python/genvarloader"]
branch = true
omit = [
    "*/tests/*",
    "*/_builders/*",
    "python/genvarloader/_cli.py",
]

[tool.coverage.report]
show_missing = true
skip_covered = false
exclude_lines = [
    "pragma: no cover",
    "if TYPE_CHECKING:",
    "raise NotImplementedError",
    "\\.\\.\\.",
]
```

Notes for the implementer:
- `omit` excludes the test tree and builder helpers (the latter does not exist yet — the entry is preemptive). `_cli.py` is excluded because the CLI is delegated to a separate package per `pyproject.toml` (`genvarloader-cli`).
- `exclude_lines` includes `\\.\\.\\.` to skip protocol/stub `...` bodies common in `_types.py`.

- [ ] **Step 3: Verify pyproject still parses**

Run: `pixi run -e dev python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"`
Expected: no output, exit 0.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml
git commit -m "test: add pytest-cov configuration"
```

---

## Task 2: Add `test-cov` pixi task

**Files:**
- Modify: `pixi.toml` (append to `[tasks]` block, around line 124)

- [ ] **Step 1: Add the task**

In `pixi.toml`, immediately after the existing `test = { ... }` task (around line 127), add:

```toml
test-cov = { cmd = "pytest tests --cov --cov-report=term-missing --cov-report=html --cov-report=xml", depends-on = [
    "gen",
    "gen-1kg",
] }
```

- [ ] **Step 2: Verify the task exists**

Run: `pixi task list -e dev | grep test-cov`
Expected: a `test-cov` entry appears.

- [ ] **Step 3: Smoke-run the task against one fast file**

Run: `pixi run -e dev pytest tests/test_utils.py --cov --cov-report=term-missing -q`
Expected: tests pass; a "TOTAL" coverage row prints; no `htmlcov/` generation noise (omitted `--cov-report=html` deliberately for smoke).

- [ ] **Step 4: Commit**

```bash
git add pixi.toml
git commit -m "test: add test-cov pixi task"
```

---

## Task 3: Create empty `unit/`, `integration/`, `_builders/` directories

**Files:**
- Create: `tests/unit/.gitkeep`
- Create: `tests/integration/.gitkeep`
- Create: `tests/_builders/__init__.py`

- [ ] **Step 1: Make the directories**

```bash
mkdir -p tests/unit tests/integration tests/_builders
touch tests/unit/.gitkeep tests/integration/.gitkeep
```

- [ ] **Step 2: Create `_builders/__init__.py` with a docstring**

Write `tests/_builders/__init__.py`:

```python
"""In-memory builders for tests.

Builders take plain Python/numpy/pyarrow inputs and return real internal
GenVarLoader types. Mocks are reserved for the ``Reader`` protocol boundary.
See docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md.

This package is intentionally importable on its own (no test fixtures here)
so it can be reused from ``tests/conftest.py`` and from unit tests directly.
"""
```

- [ ] **Step 3: Verify directories are picked up by pytest collection**

Run: `pixi run -e dev pytest tests/unit tests/integration --collect-only -q`
Expected: `no tests ran` (or `0 tests collected`); no errors.

- [ ] **Step 4: Commit**

```bash
git add tests/unit/.gitkeep tests/integration/.gitkeep tests/_builders/__init__.py
git commit -m "test: scaffold unit/, integration/, _builders/ directories"
```

---

## Task 4: Move existing tests under `tests/integration/` (preserving structure)

**Goal:** Bulk-move every existing test file under `tests/integration/`, preserving the current `dataset/`, `tracks/`, `variants/` subdirectory structure. This is a pure relocation — no test logic changes. The suite must stay green.

**Files:**
- Move: every file under `tests/` matching `test_*.py` or the subdirectories `dataset/`, `tracks/`, `variants/`
- Keep in place: `tests/data/` (shared inputs), `tests/test_bigwig.rs` (Rust test), `tests/_builders/`, `tests/unit/`, `tests/integration/`

- [ ] **Step 1: List files to move**

Run: `ls tests/ && echo --- && ls tests/dataset && echo --- && ls tests/tracks && echo --- && ls tests/variants`
Expected: confirms the current layout per the spec ("Directory layout — current state" at top of this plan).

- [ ] **Step 2: Move the files with `git mv`**

```bash
git mv tests/test_fasta.py tests/integration/
git mv tests/test_interval_track.py tests/integration/
git mv tests/test_ref_ds_splicing.py tests/integration/
git mv tests/test_ref_ds.py tests/integration/
git mv tests/test_table.py tests/integration/
git mv tests/test_utils.py tests/integration/
git mv tests/dataset tests/integration/dataset
git mv tests/tracks tests/integration/tracks
git mv tests/variants tests/integration/variants
```

- [ ] **Step 3: Check that imports still resolve**

Each moved test file uses path constructs like `Path(__file__).resolve().parents[1] / "data"` or `parents[2] / "data"`. After the move, the resolution depth changes by one. Verify:

Run: `grep -rn "parents\[" tests/integration | head -40`

For each match:
- Files now at `tests/integration/test_*.py` previously at `tests/test_*.py` used `parents[1]` to reach `tests/`. After the move they are one level deeper, so `parents[1]` now points to `tests/integration/`, not `tests/`. They need `parents[2]`.
- Files now at `tests/integration/dataset/test_*.py` previously at `tests/dataset/test_*.py` used `parents[1]` to reach `tests/`. After the move they need `parents[2]`.
- Files now at `tests/integration/tracks/test_*.py` and `tests/integration/variants/test_*.py` follow the same pattern: increment the `parents[N]` index by 1.

**Do not skip this step** — failure to update path depth will be the most common cause of green→red transitions in this task.

- [ ] **Step 4: Patch path depths**

For each file found in Step 3, increment the `parents[N]` index by exactly 1. The mechanical rule: every test file moved deeper by one directory needs `parents[N]` → `parents[N+1]` wherever the path expression was reaching `tests/data/` or `tests/`.

After patching, run: `grep -rn "Path(__file__)" tests/integration` and visually confirm each path expression resolves to `tests/data/...`.

- [ ] **Step 5: Run the full suite**

Run: `pixi run -e dev pytest tests -m "not slow" -q`
Expected: same pass/fail count as before the move (zero new failures). If failures appear that look like `FileNotFoundError` on `tests/data/...`, you missed a `parents[N]` increment in Step 4.

- [ ] **Step 6: Commit**

```bash
git add -A tests/
git commit -m "test: relocate existing tests under tests/integration/"
```

---

## Task 5: Create `tests/conftest.py` with centralized path fixtures

**Files:**
- Create: `tests/conftest.py`

The goal here is to define a *single source of truth* for on-disk artifacts (`tests/data/*.gvl`, reference FASTAs, BED files). Test files will migrate to these fixtures in Task 6. The conftest at the `tests/` level is automatically discovered by pytest for everything under `tests/`.

- [ ] **Step 1: Inventory the on-disk artifacts referenced by tests**

Run: `grep -rohE "data_dir / \"[^\"]+\"|DATA_DIR / \"[^\"]+\"" tests/integration | sort -u`
Expected output is a list of relative paths under `tests/data/` that tests load. Example entries (your output may include more):
```
data_dir / "phased_dataset.vcf.gvl"
data_dir / "phased_dataset.pgen.gvl"
data_dir / "phased_dataset.svar.gvl"
data_dir / "fasta" / "hg38.fa.bgz"
data_dir / "1kg" / "phased_1kg.bcf.gvl"
data_dir / "1kg" / "phased_1kg.pgen.gvl"
data_dir / "1kg" / "phased_1kg.svar.gvl"
data_dir / "issue_153.bed"
data_dir / "issue_153.vcf"
data_dir / "source.bed"
data_dir / "source.vcf"
```

- [ ] **Step 2: Write `tests/conftest.py`**

Write `tests/conftest.py`:

```python
"""Shared fixtures for the GenVarLoader test suite.

Centralizes paths to on-disk artifacts under ``tests/data/``. Test files
should import these via fixture injection rather than constructing paths
or opening datasets at module scope.

Fixtures here intentionally yield *paths*, not opened Datasets. Opening
a Dataset costs real time; pushing that decision to the test (or to a
small, locally scoped fixture in the test file) keeps fixture cost
predictable. Where a session-scoped opened Dataset is genuinely useful,
prefer adding it inside the integration test file that needs it.
"""

from pathlib import Path

import pytest


# --- root paths --------------------------------------------------------------

@pytest.fixture(scope="session")
def tests_dir() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def data_dir(tests_dir: Path) -> Path:
    return tests_dir / "data"


# --- reference -------------------------------------------------------------

@pytest.fixture(scope="session")
def ref_fasta(data_dir: Path) -> Path:
    """bgzipped hg38 reference used by the default toy datasets."""
    return data_dir / "fasta" / "hg38.fa.bgz"


# --- toy phased datasets (one per variant source) ----------------------------

@pytest.fixture(scope="session")
def phased_vcf_gvl(data_dir: Path) -> Path:
    return data_dir / "phased_dataset.vcf.gvl"


@pytest.fixture(scope="session")
def phased_pgen_gvl(data_dir: Path) -> Path:
    return data_dir / "phased_dataset.pgen.gvl"


@pytest.fixture(scope="session")
def phased_svar_gvl(data_dir: Path) -> Path:
    return data_dir / "phased_dataset.svar.gvl"


# --- 1kg datasets (slow tier) ------------------------------------------------

@pytest.fixture(scope="session")
def kg_dir(data_dir: Path) -> Path:
    return data_dir / "1kg"


@pytest.fixture(scope="session")
def kg_bcf_gvl(kg_dir: Path) -> Path:
    return kg_dir / "phased_1kg.bcf.gvl"


@pytest.fixture(scope="session")
def kg_pgen_gvl(kg_dir: Path) -> Path:
    return kg_dir / "phased_1kg.pgen.gvl"


@pytest.fixture(scope="session")
def kg_svar_gvl(kg_dir: Path) -> Path:
    return kg_dir / "phased_1kg.svar.gvl"


# --- raw inputs / regression artifacts --------------------------------------

@pytest.fixture(scope="session")
def source_bed(data_dir: Path) -> Path:
    return data_dir / "source.bed"


@pytest.fixture(scope="session")
def source_vcf(data_dir: Path) -> Path:
    return data_dir / "source.vcf"


@pytest.fixture(scope="session")
def issue_153_bed(data_dir: Path) -> Path:
    return data_dir / "issue_153.bed"


@pytest.fixture(scope="session")
def issue_153_vcf(data_dir: Path) -> Path:
    return data_dir / "issue_153.vcf"
```

If your Step 1 output revealed paths not in the list above, add a fixture for each one following the same shape (one fixture per artifact, session-scoped, returns a `Path`).

- [ ] **Step 3: Verify conftest is discovered without altering collection**

Run: `pixi run -e dev pytest tests --collect-only -q | tail -5`
Expected: same number of items collected as before. The conftest is loaded but no test consumes its fixtures yet.

- [ ] **Step 4: Verify fixtures resolve to existing paths**

Add a temporary smoke test:

Create `tests/integration/test_conftest_smoke.py`:

```python
"""Smoke test for tests/conftest.py path fixtures.

Deleted at the end of this task — it exists only to prove the fixtures
resolve. Real coverage of these paths comes from the migrated tests in
the next task.
"""

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "fixture_name",
    [
        "data_dir",
        "ref_fasta",
        "phased_vcf_gvl",
        "phased_pgen_gvl",
        "phased_svar_gvl",
        "source_bed",
        "source_vcf",
        "issue_153_bed",
        "issue_153_vcf",
    ],
)
def test_path_exists(request: pytest.FixtureRequest, fixture_name: str):
    path: Path = request.getfixturevalue(fixture_name)
    assert path.exists(), f"{fixture_name} -> {path} does not exist"
```

Run: `pixi run -e dev pytest tests/integration/test_conftest_smoke.py -v`
Expected: all parametrized cases PASS. If `phased_*.gvl` is missing, run `pixi run -e dev gen` first.

- [ ] **Step 5: Remove the smoke test**

```bash
rm tests/integration/test_conftest_smoke.py
```

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py
git commit -m "test: add centralized path fixtures in tests/conftest.py"
```

---

## Task 6: Migrate test files to use conftest fixtures

**Goal:** Replace module-level `DATA_DIR = Path(__file__)...`, `REF = ...`, and `DATASET = gvl.Dataset.open(...)` constants in each integration test file with fixture-injected parameters. Do this *one file at a time, with a green-suite checkpoint after each*, so any breakage is bisectable.

**Files (one per sub-task):**
- Migrate: every file under `tests/integration/` that references `Path(__file__)` for paths.

**General migration recipe** (apply per file):

1. Remove `DATA_DIR = Path(__file__).resolve().parents[N] / "data"` and `REF = ...` constants.
2. Remove module-level `DATASET = gvl.Dataset.open(...)` constants.
3. For each test or fixture that used those constants, add the appropriate path fixture (`data_dir`, `ref_fasta`, `phased_vcf_gvl`, etc.) to its signature.
4. If `DATASET` was used at module scope to derive `REGIONS`/`SAMPLES` constants for case generators, convert those case generators into fixtures or pass the dataset through `parametrize_with_cases` indirection (pytest-cases supports fixture-aware cases via `@fixture` decorator).
5. After editing the file, run only that file: `pixi run -e dev pytest tests/integration/<path>/<file>.py -v -q`.
6. After it passes, run the whole non-slow suite to make sure nothing else regressed: `pixi run -e dev pytest tests -m "not slow" -q`.

**Note on `test_subset.py`:** This file is unusually entangled — it computes `REGIONS = DATASET._idxer.full_region_idxs` at module scope and uses it inside `idx_*` case generators. Migration requires moving the per-case `desired` computation inside the test function (or a fixture-scoped case), accepting the `dataset` fixture. Plan to spend extra time here.

- [ ] **Step 1: Enumerate files to migrate**

Run: `grep -rln "Path(__file__)" tests/integration`
Expected: a list. Tackle them in this order (simplest → most complex):

1. `tests/integration/test_utils.py`
2. `tests/integration/test_fasta.py`
3. `tests/integration/test_interval_track.py`
4. `tests/integration/test_ref_ds.py`
5. `tests/integration/test_ref_ds_splicing.py`
6. `tests/integration/test_table.py`
7. `tests/integration/variants/test_sites.py`
8. `tests/integration/variants/test_variant_utils.py`
9. `tests/integration/tracks/test_annot_tracks.py`
10. `tests/integration/tracks/test_i2t_t2i.py`
11. `tests/integration/tracks/test_random_nonoverlapping.py`
12. `tests/integration/dataset/test_dataset.py`
13. `tests/integration/dataset/test_ds_haps.py`
14. `tests/integration/dataset/test_ds_haps_1kg.py`
15. `tests/integration/dataset/test_write.py`
16. `tests/integration/dataset/test_write_tracks.py`
17. `tests/integration/dataset/test_build_reconstructor.py`
18. `tests/integration/dataset/test_indexing.py`
19. `tests/integration/dataset/test_insertion_fill.py`
20. `tests/integration/dataset/test_issue_153.py`
21. `tests/integration/dataset/test_issue_191_var_fields.py`
22. `tests/integration/dataset/test_jitter.py`
23. `tests/integration/dataset/test_open_vs_settings_parity.py`
24. `tests/integration/dataset/test_rc_packing.py`
25. `tests/integration/dataset/test_realign.py`
26. `tests/integration/dataset/test_splice_plan.py`
27. `tests/integration/dataset/test_svar_link.py`
28. `tests/integration/dataset/test_with_settings_var_filter.py`
29. `tests/integration/dataset/test_get_splice_bed.py`
30. `tests/integration/dataset/test_subset.py`  ← most complex; do last

- [ ] **Step 2: Migrate each file using the recipe above, one commit per file.**

After each file:
```bash
pixi run -e dev pytest tests/integration/<that-file>.py -q
pixi run -e dev pytest tests -m "not slow" -q
git add tests/integration/<that-file>.py
git commit -m "test: use conftest fixtures in <basename>"
```

**Worked example — `test_dataset.py` (current state shown for reference):**

```python
# BEFORE (tests/integration/dataset/test_dataset.py top of file):
from pathlib import Path
from typing import Literal

import genvarloader as gvl
import numpy as np
from pytest_cases import fixture, parametrize_with_cases

data_dir = Path(__file__).resolve().parents[2] / "data"   # parents[2] after Task 4
ref = data_dir / "fasta" / "hg38.fa.bgz"


def ds_phased():
    return gvl.Dataset.open(data_dir / "phased_dataset.vcf.gvl", ref)
# ...
```

```python
# AFTER:
from pathlib import Path
from typing import Literal

import genvarloader as gvl
import numpy as np
from pytest_cases import fixture, parametrize_with_cases


@fixture(scope="session")
def ds_phased(phased_vcf_gvl: Path, ref_fasta: Path):
    return gvl.Dataset.open(phased_vcf_gvl, ref_fasta)


@fixture(scope="session")
@parametrize_with_cases("ds", prefix="ds_", cases=".")
def dataset(ds: gvl.Dataset, seq_type: Literal["reference", "haplotypes", "annotated"]):
    return ds.with_seqs(seq_type)
# ...
```

Note: pytest-cases lets fixtures themselves be cases (`@fixture` + `prefix="ds_"`), so the conversion is small.

**Worked example — `test_subset.py` (the complex one):**

The module-level `REGIONS` and `SAMPLES` constants are used inside `idx_*` and `smp_*` case generators. The cleanest fix: instead of constants, derive them inside each test from the fixture-provided dataset. Convert case functions from `() -> (idx, desired)` to `(idx) -> idx_only`, and compute `desired` inside the test body using the injected `dataset` fixture.

```python
# AFTER (sketch):
@fixture(scope="session")
def dataset(phased_vcf_gvl: Path, ref_fasta: Path):
    return gvl.Dataset.open(phased_vcf_gvl, ref_fasta)


def idx_none(): return None
def idx_scalar(): return 0
def idx_neg_scalar(): return -1
def idx_slice_none(): return slice(None)
def idx_slice_start_none(): return slice(1, None)
def idx_slice_none_stop(): return slice(None, 2)
def idx_list(): return [0, 1, 2]
def idx_array(): return np.arange(3)


def smp_none(): return None
def smp_scalar(): return 0
def smp_neg_scalar(): return -1
def smp_slice_none(): return slice(None)
def smp_slice_start_none(): return slice(1, None)
def smp_slice_none_stop(): return slice(None, 2)
def smp_list(): return [2, 0, 1]
def smp_array(): return np.arange(3)


@parametrize_with_cases("regions", cases=".", prefix="idx_")
@parametrize_with_cases("samples", cases=".", prefix="smp_")
def test_subset(dataset, regions, samples):
    full_regions = dataset._idxer.full_region_idxs
    full_samples = dataset._idxer.full_samples
    desired_regions = full_regions if regions is None else full_regions[regions]
    desired_samples = full_samples if samples is None else full_samples[samples]

    sub = dataset.subset_to(regions, samples)
    np.testing.assert_equal(sub._idxer._r_idx, desired_regions)
    np.testing.assert_equal(np.atleast_1d(sub.samples), desired_samples)
```

The bool-array, string, and series cases need extra thought because they depend on dataset size; either keep them as standalone non-parametrized tests or guard them inside the test body. Use your judgment — keep behavior identical to before.

- [ ] **Step 3: Final suite check after all files migrated**

Run: `pixi run -e dev pytest tests -m "not slow" -q`
Expected: green. Same pass count as at the start of Task 6.

Run: `pixi run -e dev pytest tests -q`
Expected: green (now including `slow`-marked tests). Same pass count as the original suite. If slow tests were passing before the overhaul they should still pass now.

---

## Task 7: Generate coverage baseline

**Files:**
- Create: `docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt`

- [ ] **Step 1: Run full coverage**

Run: `pixi run -e dev pytest tests --cov --cov-report=term-missing --cov-report=xml --cov-report=html -q`
Expected: tests pass; a `coverage.xml`, `.coverage`, and `htmlcov/` directory are produced.

- [ ] **Step 2: Capture a per-file summary**

Run:
```bash
pixi run -e dev pytest tests --cov --cov-report=term -q 2>&1 | tee /tmp/coverage-baseline.txt
```

Then extract the per-file table:
```bash
awk '/^---* coverage/,/^TOTAL/' /tmp/coverage-baseline.txt > docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt
```

If the `awk` pattern doesn't match (formatting varies between coverage versions), open `/tmp/coverage-baseline.txt`, locate the coverage table (a block starting with `Name`, `Stmts`, `Miss`, `Branch`, `BrPart`, `Cover`, `Missing` and ending at `TOTAL`), and copy that block verbatim into the baseline file.

- [ ] **Step 3: Capture per-test durations**

Run: `pixi run -e dev pytest tests --durations=0 -q 2>&1 | tail -200 > /tmp/test-durations.txt`
Expected: a `slowest durations` block. Used by the next task as input.

- [ ] **Step 4: Commit the baseline**

```bash
git add docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt
git commit -m "test: capture coverage baseline for audit"
```

---

## Task 8: Write the test audit document

**Files:**
- Create: `docs/superpowers/specs/2026-05-24-test-audit.md`

This is the deliverable that gates Phases 4+. Produce a Markdown document with three sections (Delete, Port, Keep) plus a polymorphism-gap section, all driven by:

- The coverage baseline produced in Task 7
- The per-test durations from Task 7
- A read-through of every test function under `tests/integration/`

- [ ] **Step 1: Build the audit skeleton**

Write `docs/superpowers/specs/2026-05-24-test-audit.md`:

```markdown
# Test Audit (post-refactor)

**Date:** 2026-05-24
**Spec:** docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md
**Baseline:** docs/superpowers/specs/2026-05-24-test-audit-coverage-baseline.txt

This audit classifies every test function in `tests/integration/` (the
sole test tier as of this audit) into one of three buckets:

- **Delete** — tautological, fully duplicated by another test, or
  testing behavior removed in the refactor campaign.
- **Port** — valuable behavior, but the test is unnecessarily E2E and
  should move to `tests/unit/` once the relevant builder exists.
- **Keep-as-integration** — true regression coverage (write/read
  roundtrips, variant-source parity, golden 1kg checks).

A fourth section lists **polymorphism gaps**: code paths the existing
suite doesn't exercise, organized by axis (output mode, with_settings,
subset_to, __getitem__, splicing, VCF/PGEN/SVAR parity, insertion-fill).

## Method

For each file under `tests/integration/`:
1. Read every test function.
2. Cross-reference against the coverage baseline to identify what lines
   it uniquely covers vs. what's already covered by another test.
3. Cross-reference against `--durations` to flag candidates where E2E
   cost is not justified.
4. Assign one bucket. Note rationale in one line.

## Per-file classification

<one subsection per file under tests/integration/, formatted as the
example below>

## Polymorphism gaps

<one subsection per axis>

## Summary

| Bucket | Count |
|---|---|
| Delete | TBD |
| Port | TBD |
| Keep-as-integration | TBD |
```

- [ ] **Step 2: Fill in per-file classification**

For **each** file in this list, add a subsection in the audit doc. Use the template below. The implementer must read each test function to make the call — this is the analytical core of the audit, do not skim.

Files to audit:
- `tests/integration/test_fasta.py`
- `tests/integration/test_interval_track.py`
- `tests/integration/test_ref_ds_splicing.py`
- `tests/integration/test_ref_ds.py`
- `tests/integration/test_table.py`
- `tests/integration/test_utils.py`
- `tests/integration/dataset/test_build_reconstructor.py`
- `tests/integration/dataset/test_dataset.py`
- `tests/integration/dataset/test_ds_haps.py`
- `tests/integration/dataset/test_ds_haps_1kg.py`
- `tests/integration/dataset/test_get_splice_bed.py`
- `tests/integration/dataset/test_indexing.py`
- `tests/integration/dataset/test_insertion_fill.py`
- `tests/integration/dataset/test_issue_153.py`
- `tests/integration/dataset/test_issue_191_var_fields.py`
- `tests/integration/dataset/test_jitter.py`
- `tests/integration/dataset/test_open_vs_settings_parity.py`
- `tests/integration/dataset/test_rc_packing.py`
- `tests/integration/dataset/test_realign.py`
- `tests/integration/dataset/test_splice_plan.py`
- `tests/integration/dataset/test_subset.py`
- `tests/integration/dataset/test_svar_link.py`
- `tests/integration/dataset/test_with_settings_var_filter.py`
- `tests/integration/dataset/test_write_tracks.py`
- `tests/integration/dataset/test_write.py`
- `tests/integration/tracks/test_annot_tracks.py`
- `tests/integration/tracks/test_i2t_t2i.py`
- `tests/integration/tracks/test_random_nonoverlapping.py`
- `tests/integration/variants/test_sites.py`
- `tests/integration/variants/test_variant_utils.py`

Template per file:

```markdown
### `tests/integration/<path>/<file>.py`

| Test | Bucket | Rationale |
|---|---|---|
| `test_foo` | Port | Exercises kernel X end-to-end; no Dataset required once `make_haps` lands. |
| `test_bar` | Delete | Asserts only that `Dataset.open` doesn't raise; covered by `test_write_read_roundtrip`. |
| `test_baz` | Keep-as-integration | Validates VCF/PGEN parity at write-time. |
```

Apply the following classification heuristics:

- **Delete if** the test:
  - asserts `Dataset.open(...)` doesn't raise and nothing else,
  - asserts state that's set up by a fixture (`assert ds.n_samples == 3` where the fixture built a 3-sample dataset),
  - tests behavior that issue / refactor commits removed,
  - is fully redundant with another file (e.g. `test_ds_haps.py` and `test_ds_haps_1kg.py` exercising the same code path at different scales — keep the smaller; mark the larger as `Keep-as-integration` if it's the actual regression net, else Delete).
- **Port if** the test:
  - calls a kernel or component method directly and only uses `Dataset` as scaffolding,
  - parametrizes over polymorphism axes (`with_seqs`, `with_settings`, output modes) — these become unit tests against in-memory builders,
  - tests `Ragged*` operations, `SplicePlan`, `_Variants.from_table`, insertion-fill strategies, or reconstruction algorithms.
- **Keep-as-integration if** the test:
  - performs an end-to-end write→read roundtrip,
  - validates VCF/PGEN/SVAR parity (cross-source comparisons require real data sources),
  - is a regression fixture for a filed issue (`test_issue_153.py`, `test_issue_191_var_fields.py`) — keep but consider whether the regression check itself could be ported once builders allow it,
  - exercises the 1kg pipeline (`test_ds_haps_1kg.py`) — keep one canonical case, mark others Delete.

- [ ] **Step 3: Fill in polymorphism-gap section**

Use the coverage baseline (`coverage-baseline.txt`) to identify under-covered files. For each axis below, walk the source and note which code paths lack coverage.

Polymorphism axes (from spec Section 3 / spec lines 113-126):

- **Output mode matrix** — `python/genvarloader/_dataset/_impl.py`: walk every `with_seqs(...)` branch and confirm coverage across `haplotypes`, `reference`, `annotated`, `variants` × ragged/padded × jitter × rc_neg. Note missing combinations.
- **`with_settings` lazy reload** — `_dataset/_impl.py`: which kwargs trigger a reload? Are all branches covered? Specifically check `var_fields` (recently added per commit `f6da36e`).
- **`subset_to`** — `_dataset/_impl.py`: scalar / slice / list / array / bool / str / series index types, regions × samples combinations.
- **`__getitem__` polymorphism** — return type per (output mode × len mode) combination.
- **Splicing / GTF** — `_dataset/_reference.py`, `_dataset/_impl.py`, `SplicePlan`: edge cases for multi-exon, single-exon, negative-strand spliced regions, exonic variant filter.
- **Insertion-fill strategies** — `_dataset/_tracks.py`: each strategy as documented in the spec.
- **VCF/PGEN/SVAR parity** — pairs that should produce identical output across all dataset modes; gap exists wherever a parity test doesn't cover a given mode.

Format:

```markdown
### Output mode matrix

Missing combinations (verified against coverage baseline):

- `haplotypes × padded × jitter=5 × rc_neg=True` — not exercised by any current test.
- `annotated × ragged × jitter=0 × rc_neg=False` — covered only via `test_ds_haps.py`, no assertion on return type.
- ...

Files with low line coverage in this area: `_dataset/_impl.py` lines NNN-NNN (with_seqs branch for `variants` mode).
```

- [ ] **Step 4: Fill in the summary count**

Replace the `TBD` cells in the summary table with the actual counts derived from your per-file tables.

- [ ] **Step 5: Self-review the audit doc**

Run through this checklist on your audit:
1. Every file in the audit list has a subsection.
2. Every test function in every file appears in exactly one row.
3. No row says "TBD" or "TODO".
4. Polymorphism-gap section has at least one entry per axis (an empty axis is suspicious — verify with the coverage report before claiming "no gaps").
5. The summary count matches the row count across the three buckets.

- [ ] **Step 6: Commit the audit**

```bash
git add docs/superpowers/specs/2026-05-24-test-audit.md
git commit -m "docs(audit): classify existing tests for overhaul (phase 3)"
```

---

## Task 9: End-of-plan verification and handoff

- [ ] **Step 1: Run full suite (including slow)**

Run: `pixi run -e dev pytest tests -q`
Expected: green, same pass count as before this plan started.

- [ ] **Step 2: Run coverage one more time and confirm parity with baseline**

Run: `pixi run -e dev pytest tests --cov --cov-report=term -q | tail -30`
Expected: TOTAL coverage roughly matches the baseline (small drift acceptable from fixture-induced ordering). Significant drops in coverage indicate a test broke silently during migration — bisect against the per-file commits from Task 6.

- [ ] **Step 3: Confirm the audit doc is ready for review**

Open `docs/superpowers/specs/2026-05-24-test-audit.md` and read it end to end. The user will review it before Phase 4 (delete pass) is planned.

- [ ] **Step 4: Final state check**

Run: `git log --oneline -20`
Expected: roughly 30+ commits since the start of this plan (one per migrated file plus the structural commits), all green, conventional-commit-formatted.

Run: `ls tests/`
Expected:
```
_builders/  conftest.py  data/  integration/  test_bigwig.rs  unit/
```

(No top-level `test_*.py` files left; everything except `test_bigwig.rs` and `conftest.py` has been moved to `integration/`.)

---

## Out of scope (deferred to follow-up plans)

- **Phase 4 — delete pass.** Driven by the audit produced in Task 8.
- **Phase 5 — builders + unit tier.** Per-component PRs: ragged → reconstruct → variants → haps → tracks → splice → dataset polymorphism.
- **Phase 6 — integration trim.** Triggered per-area as Phase 5 lands coverage.
- **Phase 7 — CI report.** Wire `htmlcov/` upload into CI.

Each of these gets its own plan informed by Phase 3 audit output.
