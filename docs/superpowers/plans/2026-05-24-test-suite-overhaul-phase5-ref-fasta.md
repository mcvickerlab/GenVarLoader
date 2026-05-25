# Test Suite Overhaul — Phase 5 Ref / FASTA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the remaining Ref/FASTA-component tests to the unit tier and promote the duplicated `reference` fixture to `tests/conftest.py`. Concretely: whole-file move of `test_fasta.py` (3 ports), whole-file move of `test_ref_ds.py` (2 ports — both remaining tests are audit-Port after the Phase 4 deletion of `test_refdataset_unspliced_defaults`), and shared-fixture cleanup that touches `test_ref_ds_splicing.py` (integration) and `test_ref_ds_splice_settings.py` (unit).

**Architecture:** Three call sites currently duplicate `reference = gvl.Reference.from_path(ref_fasta, in_memory=False)` as a local fixture. Two are pytest fixtures; one uses `pytest_cases.fixture` (test_ref_ds.py). Promoting the fixture to conftest is safe because the conftest convention only forbids opened *Datasets* (which are expensive due to genotype/track loading) — `Reference.from_path(..., in_memory=False)` reads only metadata and is cheap. After promotion all four files (the two moved unit files plus the two existing files) share a single session-scoped fixture.

**Tech Stack:** pytest, pytest-cases, numpy, polars, pysam, seqpro, genvarloader internals (`_fasta`, `_dataset._utils`).

---

## File Structure

- **Modify:** `tests/conftest.py` — add session-scoped `reference` fixture under a new `# --- reference (opened) ---` section, after the `ref_fasta` block.
- **Modify:** `tests/integration/test_ref_ds_splicing.py` — drop the local `reference` fixture (lines 15–17).
- **Modify:** `tests/unit/splice/test_ref_ds_splice_settings.py` — drop the local `reference` fixture (lines 17–19).
- **Move + drop local fixture:** `tests/integration/test_ref_ds.py` → `tests/unit/dataset/test_ref_ds.py`. Remove the local `@fixture\ndef reference(...)` block (lines 10–12) and remove now-unused `fixture` from the `pytest_cases` import.
- **Move (whole-file):** `tests/integration/test_fasta.py` → `tests/unit/test_fasta.py`. No fixture changes (this file uses `ref_fasta` only, not `reference`).
- **No production source changes.**
- **No new builder modules.**
- **Basename collision check:** no existing `tests/unit/test_fasta.py` or `tests/unit/dataset/test_ref_ds.py` — clean moves.

---

### Task 1: Promote `reference` fixture to `tests/conftest.py`; remove duplicates from the two existing files

**Files:**
- Modify: `tests/conftest.py:38` (insert after `ref_fasta` fixture)
- Modify: `tests/integration/test_ref_ds_splicing.py:15-17` (remove local fixture)
- Modify: `tests/unit/splice/test_ref_ds_splice_settings.py:17-19` (remove local fixture)

**Why session scope:** `Reference.from_path(path, in_memory=False)` reads only contig metadata from the FAI/GZI sidecar files — it doesn't pull bytes into memory. Repeated per-test instantiation is wasteful for ~zero benefit. Session scope is consistent with the other conftest fixtures (all path fixtures are session-scoped).

**Why the conftest docstring still holds:** The existing docstring says fixtures here yield *paths*, not opened Datasets, because Datasets are expensive. `Reference` (with `in_memory=False`) is metadata-only and effectively free. I'm adding a short note in the new section header to explain the carve-out.

- [ ] **Step 1: Pre-change baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/test_ref_ds_splicing.py tests/unit/splice/test_ref_ds_splice_settings.py -q
```

Expected: all pass. Note the exact count.

- [ ] **Step 2: Add the `reference` fixture to conftest**

Edit `tests/conftest.py`. Find this block (lines 32–38):
```python
# --- reference -------------------------------------------------------------


@pytest.fixture(scope="session")
def ref_fasta(data_dir: Path) -> Path:
    """bgzipped hg38 reference used by the default toy datasets."""
    return data_dir / "fasta" / "hg38.fa.bgz"
```

Replace with (preserving the path fixture, adding the opened-Reference fixture below):
```python
# --- reference -------------------------------------------------------------


@pytest.fixture(scope="session")
def ref_fasta(data_dir: Path) -> Path:
    """bgzipped hg38 reference used by the default toy datasets."""
    return data_dir / "fasta" / "hg38.fa.bgz"


@pytest.fixture(scope="session")
def reference(ref_fasta: Path):
    """Opened ``gvl.Reference`` (metadata-only, ``in_memory=False``).

    Carve-out vs. the "yield paths, not Datasets" convention: opening a
    Reference with ``in_memory=False`` is metadata-only (FAI/GZI read),
    not the expensive bytes-loading path. Three tests (now four after
    the Phase 5 ref/fasta extraction) need this exact construction, so
    centralizing it here removes duplication without violating the
    spirit of the convention.
    """
    import genvarloader as gvl

    return gvl.Reference.from_path(ref_fasta, in_memory=False)
```

(The local `import genvarloader as gvl` keeps the conftest's module-level imports light — `gvl` is a heavy import, so deferring it to fixture-call time is the right move for files that don't need a Reference.)

- [ ] **Step 3: Remove the duplicate fixture from `test_ref_ds_splicing.py`**

Edit `tests/integration/test_ref_ds_splicing.py`. Delete lines 15–17 (the local `reference` fixture):

```python
@pytest.fixture
def reference(ref_fasta) -> gvl.Reference:
    return gvl.Reference.from_path(ref_fasta, in_memory=False)
```

After the edit, the file should jump straight from the module docstring + imports to the `two_transcript_bed` fixture. Verify `pytest` is still imported at top (it's used by `@pytest.fixture` on `two_transcript_bed` — wait, that's now the only remaining `@pytest.fixture`, still needs the import).

Also: `gvl` is still used by every test body (`gvl.RefDataset(...)`), so that import stays.

- [ ] **Step 4: Remove the duplicate fixture from `test_ref_ds_splice_settings.py`**

Edit `tests/unit/splice/test_ref_ds_splice_settings.py`. Delete lines 17–19 (the local `reference` fixture):

```python
@pytest.fixture
def reference(ref_fasta) -> gvl.Reference:
    return gvl.Reference.from_path(ref_fasta, in_memory=False)
```

Verify `pytest` is still imported (used by `@pytest.fixture` on `two_transcript_bed` and possibly elsewhere). `gvl` stays (used in test bodies).

- [ ] **Step 5: Run both affected files**

Run:
```bash
pixi run -e dev pytest tests/integration/test_ref_ds_splicing.py tests/unit/splice/test_ref_ds_splice_settings.py -v
```

Expected: same count as Step 1, all passing. The session-scoped conftest fixture is now resolving `reference` for both files.

- [ ] **Step 6: Ruff check**

Run:
```bash
pixi run -e dev ruff check tests/conftest.py tests/integration/test_ref_ds_splicing.py tests/unit/splice/test_ref_ds_splice_settings.py
```

Expected: no findings. If ruff flags an unused import (e.g. `gvl.Reference` no longer used in a type hint after fixture removal — unlikely since `gvl.RefDataset(reference: gvl.Reference, ...)` no longer has the local fixture's type hint but the test signatures still use `gvl.Reference`), apply autofix:

```bash
pixi run -e dev ruff check --fix tests/integration/test_ref_ds_splicing.py tests/unit/splice/test_ref_ds_splice_settings.py
```

Then re-run tests to confirm.

- [ ] **Step 7: Full suite + commit**

Run:
```bash
pixi run -e dev pytest -q
```

Expected: 351 passed (test count unchanged — fixture relocation does not add or remove tests).

Then commit:
```bash
git add tests/conftest.py tests/integration/test_ref_ds_splicing.py tests/unit/splice/test_ref_ds_splice_settings.py
git status --short
git commit -m "test(conftest): promote reference fixture; drop duplicates

Add session-scoped \`reference\` fixture to tests/conftest.py. Two
existing files (test_ref_ds_splicing, test_ref_ds_splice_settings)
duplicated the same Reference.from_path(..., in_memory=False)
construction; both now consume the conftest fixture. Convention
carve-out documented in the fixture docstring — metadata-only
construction is cheap enough to warrant centralization.

Phase 5 ref/fasta component — fixture pre-work."
```

Expected: commit succeeds.

---

### Task 2: Whole-file move of `test_fasta.py`

**Files:**
- Move: `tests/integration/test_fasta.py` → `tests/unit/test_fasta.py`

**Why a whole-file move:** All 3 audit-classified tests are Port. The only fixture used is `ref_fasta` (already in conftest, unchanged by Task 1). No `reference` fixture used. No basename collision.

- [ ] **Step 1: Pre-move baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/test_fasta.py -v
```

Expected: 3 passed (`test_pad_right`, `test_pad_left`, `test_no_pad`).

- [ ] **Step 2: Verify no basename collision**

Run:
```bash
ls tests/unit/test_fasta.py 2>&1 || echo "no collision (expected)"
```

Expected: file does not exist.

- [ ] **Step 3: Move**

Run:
```bash
git mv tests/integration/test_fasta.py tests/unit/test_fasta.py
```

- [ ] **Step 4: Verify clean rename**

Run:
```bash
git status --short
git diff --cached --stat
```

Expected:
```
R  tests/integration/test_fasta.py -> tests/unit/test_fasta.py
```
with `0 insertions(+), 0 deletions(-)`.

- [ ] **Step 5: Run from new location**

Run:
```bash
pixi run -e dev pytest tests/unit/test_fasta.py -v
```

Expected: 3 passed.

- [ ] **Step 6: Ruff**

Run:
```bash
pixi run -e dev ruff check tests/unit/test_fasta.py
```

Expected: no findings.

- [ ] **Step 7: Commit**

Run:
```bash
git commit -m "test: move test_fasta.py to unit tier (ref/fasta component)

3 Fasta.read pad/no-pad boundary tests. Audit-Port whole-file move;
only ref_fasta fixture used (already in conftest); no basename
collision.

Phase 5 ref/fasta component."
```

Expected: commit succeeds.

---

### Task 3: Whole-file move of `test_ref_ds.py` + drop local `reference` fixture

**Files:**
- Move: `tests/integration/test_ref_ds.py` → `tests/unit/dataset/test_ref_ds.py`
- Modify (after move): drop local `reference` fixture (lines 10–12) + tidy `pytest_cases` import.

**Why this works post-Task-1:** With `reference` promoted to conftest, the moved file no longer needs its local definition. The two remaining tests (`test_getitem`, `test_padded_slice`) are both audit-Port. The third audit entry (`test_refdataset_unspliced_defaults`) was already deleted in Phase 4 — verify before moving.

**Why drop the local fixture rather than leaving it:** A local `@pytest_cases.fixture` defined on `reference` would shadow the conftest fixture, but only for tests in this file. Keeping both creates a maintenance hazard: someone updates conftest but not this file (or vice-versa). One source of truth.

- [ ] **Step 1: Pre-move baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/test_ref_ds.py -v
```

Expected: 6 passed — `test_getitem` parametrized over 2 cases (1 xfail + 1 pass), `test_padded_slice` over 4 cases. Actual collection: 6 items (the `case_no_regions` case is decorated `@pytest.mark.xfail(strict=True, raises=ValueError)` so it should xpass→fail strict if the behavior changes; verify it shows up as "xfailed" in the output, not as a failure). If your count differs, note it and proceed.

- [ ] **Step 2: Confirm the audit-Delete test is already gone**

Run:
```bash
grep -n "test_refdataset_unspliced_defaults" tests/integration/test_ref_ds.py 2>&1 || echo "already deleted (expected)"
```

Expected: not found. If found, STOP — Phase 4 was supposed to delete this; do not re-delete as part of this plan.

- [ ] **Step 3: Move the file**

Run:
```bash
git mv tests/integration/test_ref_ds.py tests/unit/dataset/test_ref_ds.py
```

Expected: no output.

- [ ] **Step 4: Drop the local `reference` fixture from the moved file**

Edit `tests/unit/dataset/test_ref_ds.py`. Find lines 10–12:

```python
@fixture
def reference(ref_fasta):
    return gvl.Reference.from_path(ref_fasta, in_memory=False)
```

Delete those three lines (plus the blank line that follows, if it would leave two consecutive blank lines).

- [ ] **Step 5: Tidy the `pytest_cases` import**

Find the import line:
```python
from pytest_cases import fixture, parametrize_with_cases
```

`fixture` was used only by the now-deleted `reference` fixture. Change to:
```python
from pytest_cases import parametrize_with_cases
```

- [ ] **Step 6: Verify the test signatures still resolve**

The two tests reference the fixture by parameter name:
```python
def test_getitem(
    reference: gvl.Reference, regions: pl.DataFrame, desired: gvl.Ragged[np.bytes_]
):
```

pytest's fixture-injection-by-name will find `reference` in conftest. No code changes needed.

- [ ] **Step 7: Run the moved file**

Run:
```bash
pixi run -e dev pytest tests/unit/dataset/test_ref_ds.py -v
```

Expected: same count as Step 1 (6 items, with `case_no_regions` still xfailing strict). If you see `fixture 'reference' not found`, the conftest fixture from Task 1 didn't land — STOP and revisit Task 1.

- [ ] **Step 8: Ruff**

Run:
```bash
pixi run -e dev ruff check tests/unit/dataset/test_ref_ds.py
```

Expected: no findings. If ruff flags unused imports (e.g. `import genvarloader as gvl` if Steps 4–5 somehow left a stranded reference), apply autofix and verify the file still passes.

- [ ] **Step 9: Full suite**

Run:
```bash
pixi run -e dev pytest -q
```

Expected: 351 passed (test counts unchanged through this whole plan — same tests, different homes).

- [ ] **Step 10: Commit**

Run:
```bash
git add tests/unit/dataset/test_ref_ds.py
git status --short
git commit -m "test: move test_ref_ds.py to unit tier; drop local reference fixture

2 remaining audit-Port tests (test_getitem, test_padded_slice) move
whole-file to tests/unit/dataset/. The local reference fixture is
deleted in favor of the conftest fixture promoted in the previous
commit. Tidy the pytest_cases import (fixture no longer needed).

Phase 5 ref/fasta component — final move."
```

Expected: commit succeeds.

---

### Task 4: Update the status doc

**Files:**
- Modify: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md`

- [ ] **Step 1: Re-read live counts**

Run:
```bash
pixi run -e dev pytest tests/unit -q 2>&1 | tail -3
pixi run -e dev pytest -q 2>&1 | tail -3
git log -1 --format=%h
```

Note unit-tier count, full-suite count, and latest hash.

- [ ] **Step 2: Update header `As of` line**

Replace the hash after `committed through` with the value from Step 1.

- [ ] **Step 3: Add plan to "Authoritative reference files"**

After the `phase5-tracks-broader.md` line, append:
```
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-ref-fasta.md`
```

- [ ] **Step 4: Update unit-tier test count**

Change `158 passed, 1 xfailed (...)` to the value from Step 1.

- [ ] **Step 5: Update file-layout tree**

- Under `tests/unit/`: add `test_fasta.py` (alphabetical among the top-level files: between any existing top-level files; ends up between `test_fasta.py` going before `test_table.py`).
- Under `tests/unit/dataset/`: add `test_ref_ds.py`.
- Under `tests/integration/`: remove both `test_fasta.py` and `test_ref_ds.py`.
- Bump the unit-tier comment count.
- Lower the integration-tier comment count by the same delta.

- [ ] **Step 6: Add row to "What shipped"**

Append after the tracks-broader row:
```
| 5 ref/fasta | reference fixture → conftest; test_fasta.py + test_ref_ds.py whole-file moves | Fixture promoted with carve-out docstring (metadata-only Reference is cheap); 2 duplicate fixtures dropped; 2 whole-file moves; no basename collisions |
```

- [ ] **Step 7: Remove the Ref / FASTA section from "What's left"**

Delete the entire `#### Ref / FASTA (~10 ports across 3 files)` subsection (the heading + all 4 bullets).

- [ ] **Step 8: Update "Recommended next plan"**

Replace the body so that **Dataset polymorphism / `make_dataset`** becomes the recommended next plan. New text:

```markdown
## Recommended next plan

**Dataset polymorphism / `make_dataset`** — Last component plan. Two specific audit-Port tests gated on a `make_dataset` builder: `test_dataset.py:test_ds_indexing` and `test_dummy_dataset_insertion_fill.py:test_with_insertion_fill_rejects_when_no_tracks_active`. The builder needs to synthesize Datasets with selectable seq/track/variant configurations to stand in for the toy-dataset fixtures these tests currently depend on.

Once that lands:
- **Phase 6 (integration trim)** — As outlined in design spec. Review each integration-tier file post-overhaul; remove redundancies where unit coverage now subsumes them.
- **Phase 7 (CI report)** — Wire `htmlcov/` upload into CI per design spec.
```

Keep the trailing `---` separator before "How to resume".

- [ ] **Step 9: Update "Notable decisions / gotchas" with the conftest carve-out**

Append a new numbered item:

```markdown
9. **Conftest `reference` fixture carve-out** — The original conftest convention (docstring) said fixtures yield *paths*, not opened Datasets, because Datasets are expensive. Phase 5 ref/fasta added a `reference` fixture to conftest as a deliberate exception: `Reference.from_path(path, in_memory=False)` reads only FAI/GZI metadata, so session-scoped centralization is cheap. The fixture's own docstring records the rationale. Future work that wants to add another "opened object" fixture should explicitly justify the same way or stick to paths.
```

- [ ] **Step 10: Commit**

Run:
```bash
git add docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-ref-fasta.md
git status --short
git commit -m "docs(status): record ref/fasta component; recommend make_dataset next"
```

Expected: commit succeeds.

---

## Self-Review Checklist

- **Spec coverage:** The status doc identifies Ref/FASTA as `test_fasta.py` (3 ports), `test_ref_ds.py` (2 remaining ports), plus `test_ref_ds_splicing.py` mentioned as part of the fixture-promotion deduplication target. Tasks 1–3 cover all three; the splicing file is touched only for fixture cleanup (no test moved). ✅
- **Audit alignment:** Task 2 follows the audit's "all three belong together in tests/unit/fasta/test_fasta.py" recommendation but places the file at `tests/unit/test_fasta.py` per the established convention (component subdir only when ≥2 files; YAGNI). Task 3 covers the 2 remaining audit-Port tests after Phase 4 already deleted the third. ✅
- **No placeholders:** Every code edit shows the exact before/after; every command has expected output. ✅
- **Type/name consistency:** Conftest fixture name `reference` matches all three consumer files' parameter names. No new types introduced. ✅
- **Basename collision:** No collisions for either move (verified by name lookups; targets don't exist pre-move). ✅
- **Fixture-promotion safety:**
  - `test_ref_ds_splicing.py` (Task 1 Step 3) and `test_ref_ds_splice_settings.py` (Step 4) use `@pytest.fixture` for the local `reference` — straightforward to remove.
  - `test_ref_ds.py` (Task 3 Step 4) uses `@pytest_cases.fixture` for the local `reference`. Removing it works because the test still injects `reference` by name and pytest finds the conftest fixture via the same name resolution. `pytest_cases.fixture` and `pytest.fixture` are interchangeable for injection (the difference matters only when *parametrizing* the fixture itself, which the conftest version doesn't do).
  - Session scope is correct: the underlying `ref_fasta` path is already session-scoped; the constructed `Reference` is metadata-only and immutable.
- **Conftest convention update:** New fixture has an explicit docstring justifying the carve-out; status doc gotcha 9 records the policy.
