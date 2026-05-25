# Test Suite Overhaul — Phase 4 (Delete Pass) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the audit-approved deletions: remove 5 tests (one entirely-skipped file, two whole files with single tests, two single-function deletions inside otherwise-kept files) plus any newly-unused fixtures/imports that fall out of the removals.

**Architecture:** Five surgical deletions ordered by risk. Each deletion is one commit. Suite must remain at baseline (500 passed, 6 skipped, 3 deselected, 2 xfailed for the non-slow tier; 3 passed for slow tier) after every commit — except that the totals will shift down as tests are removed (track the new expected count per task).

**Tech Stack:** pytest, conventional commits, no code changes outside `tests/`.

**Source authority:** `docs/superpowers/specs/2026-05-24-test-audit.md` Recommendations section (lines 536–560 of the audit).

**Scope discipline:** This plan ONLY deletes. It does NOT port tests, does NOT add builders, does NOT touch `tests/unit/`. Port tests get their own plan (Phase 5).

---

## Pre-flight baseline (already established on this branch)

- Non-slow tier: **500 passed, 6 skipped, 3 deselected, 2 xfailed**
- Slow tier (`-m slow`): **3 passed**
- Total tests collected (`pytest --collect-only`): 511 items / 508 deselected, 3 selected for slow

Track the running counts as deletions land:

| After task | Non-slow expected |
|---|---|
| Baseline (start) | 500 passed, 6 skipped, 3 deselected, 2 xfailed |
| Task 1 (`test_filter_af` file delete) | 500 passed, **5 skipped**, 3 deselected, 2 xfailed |
| Task 2 (`test_rs_indexing` delete, 147 cases) | **353 passed**, 5 skipped, 3 deselected, 2 xfailed |
| Task 3 (`test_interval_track.py` file delete, 1 test) | **352 passed**, 5 skipped, 3 deselected, 2 xfailed |
| Task 4 (`test_refdataset_unspliced_defaults` delete) | **351 passed**, 5 skipped, 3 deselected, 2 xfailed |
| Task 5 (`test_write` (skip) + dead fixtures) | 351 passed, **4 skipped**, 3 deselected, 2 xfailed |

The "deselected" count (3) is the slow tests being deselected by `-m "not slow"`; that stays constant. The "xfailed" (2) likewise.

Each task includes its expected count change; do not advance to the next task if the count doesn't match.

---

## Task 1: Delete `tests/integration/dataset/genotypes/test_filter_af.py`

**Rationale (from audit):** The entire file is `@pytest.mark.skip`. The case generators (`case_filter_af_*`) referenced by the parametrize decorator are NOT defined in the file. If the skip were ever removed, parametrize would fail with `pytest.cases collection error`. Dead code.

**Files:**
- Delete: `tests/integration/dataset/genotypes/test_filter_af.py`

- [ ] **Step 1: Verify the file is exactly as described**

Run: `cat tests/integration/dataset/genotypes/test_filter_af.py`

Expected: a 35-line file containing one `@pytest.mark.skip`-decorated `test_filter_af` function and a `@parametrize_with_cases(..., prefix="case_filter_af")` decorator. No `case_filter_af_*` functions defined in the file.

If the file looks materially different (e.g., the skip has been removed, or case generators have been added since the audit), STOP and escalate to the controller. Do not delete.

- [ ] **Step 2: Delete the file**

```bash
git rm tests/integration/dataset/genotypes/test_filter_af.py
```

- [ ] **Step 3: Run the genotypes test directory to confirm it still collects clean**

Run: `pixi run -e dev pytest tests/integration/dataset/genotypes/ --collect-only -q`
Expected: collection succeeds; no error about missing `test_filter_af`. The other `genotypes/test_*.py` files still collect.

- [ ] **Step 4: Run the non-slow suite and confirm count**

Run: `pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2`
Expected: **`500 passed, 5 skipped, 3 deselected, 2 xfailed`** (one fewer skipped — the removed test).

If pass count drops or anything else changes unexpectedly, STOP and investigate.

- [ ] **Step 5: Commit**

```bash
git commit -m "test: delete dead test_filter_af.py (entirely skipped, missing cases)"
```

Verify `git status` clean.

---

## Task 2: Delete `test_rs_indexing` from `tests/integration/dataset/test_dataset.py`

**Rationale (from audit):** 147-case cross-product (7 r_idx × 7 s_idx × 3 seq_type) that only asserts indexing doesn't raise. Strictly weaker than `test_ds_indexing` (which exercises 21 cases of the same code path) plus `test_subset.py` (108 cases asserting on actual return values).

**Files:**
- Modify: `tests/integration/dataset/test_dataset.py` — remove lines 65-68 (the `test_rs_indexing` function and its two `@parametrize_with_cases` decorators).

- [ ] **Step 1: Read the current file**

Run: `cat tests/integration/dataset/test_dataset.py`
Expected: ends with these lines (around lines 60-68):

```python
@parametrize_with_cases("idx", prefix="idx_", cases=".")
def test_ds_indexing(dataset, idx):
    dataset[idx]


@parametrize_with_cases("r_idx", prefix="idx_", cases=".")
@parametrize_with_cases("s_idx", prefix="idx_", cases=".")
def test_rs_indexing(dataset, r_idx, s_idx):
    dataset[r_idx, s_idx]
```

- [ ] **Step 2: Remove `test_rs_indexing` and its decorators**

The file should end after the `test_ds_indexing` function. Delete:

```python


@parametrize_with_cases("r_idx", prefix="idx_", cases=".")
@parametrize_with_cases("s_idx", prefix="idx_", cases=".")
def test_rs_indexing(dataset, r_idx, s_idx):
    dataset[r_idx, s_idx]
```

(Note the leading blank lines — strip them as well so the file ends with `def test_ds_indexing(...)` body and a trailing newline.)

The `idx_*` case generator functions stay in place — they are consumed by `test_ds_indexing` (which we keep). Do NOT remove them.

- [ ] **Step 3: Run the file alone**

Run: `pixi run -e dev pytest tests/integration/dataset/test_dataset.py -q`
Expected: `test_ds_indexing` passes (21 cases: 7 idx × 3 seq_type); no `test_rs_indexing` collected. Roughly: `21 passed`.

- [ ] **Step 4: Run the full non-slow suite**

Run: `pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2`
Expected: **`353 passed, 5 skipped, 3 deselected, 2 xfailed`** (147 fewer passes than Task 1's count — `500 - 147 = 353`).

- [ ] **Step 5: Commit**

```bash
git add tests/integration/dataset/test_dataset.py
git commit -m "test: delete test_rs_indexing (subsumed by test_ds_indexing + test_subset)"
```

Verify `git status` clean.

---

## Task 3: Delete `tests/integration/test_interval_track.py`

**Rationale (from audit):** Only one test in the file (`test_bigwigs_satisfies_interval_track_protocol`). That test is a tautological `hasattr` check on a class we control; it provides no regression value beyond what static type-checking (`pyrefly`) already guarantees via the `IntervalTrack: bw` assignment line that's already in the test body. With this test removed, the file would be empty of tests — delete the whole file.

**Files:**
- Delete: `tests/integration/test_interval_track.py`

- [ ] **Step 1: Confirm the file contains exactly the one test**

Run: `cat tests/integration/test_interval_track.py`
Expected: a 21-line file with one function `test_bigwigs_satisfies_interval_track_protocol(bigwig_dir)` and no other test functions or fixtures.

If a second test has been added since the audit, STOP and escalate.

- [ ] **Step 2: Delete the file**

```bash
git rm tests/integration/test_interval_track.py
```

- [ ] **Step 3: Run the non-slow suite**

Run: `pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2`
Expected: **`352 passed, 5 skipped, 3 deselected, 2 xfailed`** (one fewer pass than Task 2's count).

- [ ] **Step 4: Commit**

```bash
git commit -m "test: delete test_interval_track.py (only test was a tautological hasattr check)"
```

Verify `git status` clean.

---

## Task 4: Delete `test_refdataset_unspliced_defaults` from `tests/integration/test_ref_ds.py`

**Rationale (from audit):** Asserts that `ds.is_spliced is False` and `ds.splice_info is None` when `RefDataset` is constructed with no splice arguments. This is fixture-setup state, not behavior — the constructor literally sets these defaults with no logic. Zero regression value.

**Files:**
- Modify: `tests/integration/test_ref_ds.py` — remove lines 97-107 (the `test_refdataset_unspliced_defaults` function).

- [ ] **Step 1: Read the current file**

Run: `sed -n '94,108p' tests/integration/test_ref_ds.py`
Expected (last block of the file):

```python


def test_refdataset_unspliced_defaults(reference: gvl.Reference):
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [0, 100],
            "chromEnd": [100, 150],
        }
    )
    ds = gvl.RefDataset(reference, bed)
    assert ds.is_spliced is False
    assert ds.splice_info is None
```

- [ ] **Step 2: Remove the function**

Delete the function definition AND the preceding blank line so the file ends with the `test_padded_slice` function body. The file's tail should look like:

```python
@parametrize_with_cases(
    "arr, start, stop, pad_val, desired", cases=".", prefix="padded_slice_"
)
def test_padded_slice(
    arr: np.ndarray, start: int, stop: int, pad_val: int, desired: np.ndarray
):
    actual = np.empty_like(desired)
    padded_slice(arr, start, stop, pad_val, actual)
    np.testing.assert_equal(actual, desired)
```

(Ends with a trailing newline; nothing after the `np.testing.assert_equal` line except the EOF newline.)

- [ ] **Step 3: Check for newly-unused imports**

After the deletion, run: `grep -n "^import\|^from" tests/integration/test_ref_ds.py`

The deleted function used `pl.DataFrame` and `gvl.Reference`. Both are still used elsewhere in the file (the `reference` fixture uses `gvl.Reference`, and `case_ragged_regions`/`case_no_regions` use `pl.DataFrame`). Do not remove any imports.

- [ ] **Step 4: Run the file**

Run: `pixi run -e dev pytest tests/integration/test_ref_ds.py -q`
Expected: 6 passed, 1 xfailed (was 7 passed, 1 xfailed; `test_refdataset_unspliced_defaults` is gone).

- [ ] **Step 5: Run the full non-slow suite**

Run: `pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2`
Expected: **`351 passed, 5 skipped, 3 deselected, 2 xfailed`** (one fewer pass than Task 3's count).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/test_ref_ds.py
git commit -m "test: delete test_refdataset_unspliced_defaults (fixture-state assertion only)"
```

Verify `git status` clean.

---

## Task 5: Delete `test_write` (skipped) and dead fixtures from `tests/integration/dataset/test_write.py`

**Rationale (from audit):** `test_write` is `@pytest.mark.skip`. If revived, it would need to be rewritten against the current data layout. Two case functions (`reader_vcf`, `reader_pgen`) and two local fixtures (`bed`, `ref`) exist solely to feed this test — they're dead code once `test_write` is gone. The remaining live tests (`test_write_errors_when_post_index_budget_too_small`, `test_write_loads_lazy_vcf_index`, `test_write_loads_lazy_pgen_index`) use the conftest fixture `source_bed` instead.

**Files:**
- Modify: `tests/integration/dataset/test_write.py` — remove the skip-marked `test_write`, the two case generators, the local `bed` and `ref` fixtures, and any now-unused imports.

- [ ] **Step 1: Read the file**

Run: `cat tests/integration/dataset/test_write.py | head -50`

Confirm lines 16-47 (approximately) contain `reader_vcf`, `reader_pgen`, `bed`, `ref`, and the start of `test_write`. The deletions below assume that layout.

- [ ] **Step 2: Identify what stays**

The following functions/fixtures stay (verify by reading lines 105+):

- `test_write_errors_when_post_index_budget_too_small(tmp_path, monkeypatch, vcf_dir, source_bed)` — uses `vcf_dir` and `source_bed` from conftest, plus inline-imports `pytest`.
- `test_write_loads_lazy_vcf_index(tmp_path, vcf_dir, source_bed)` — uses conftest fixtures.
- `test_write_loads_lazy_pgen_index(tmp_path, pgen_dir, source_bed)` — uses conftest fixtures.

None of the live tests use the local `bed`, `ref`, `reader_vcf`, or `reader_pgen` definitions. After deletion, those tests must remain intact.

- [ ] **Step 3: Rewrite the file**

After deletion, the file should look exactly like this:

```python
from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
import seqpro as sp
from genoray import PGEN, VCF


def test_write_errors_when_post_index_budget_too_small(
    tmp_path, monkeypatch, vcf_dir: Path, source_bed: Path
):
    """If max_mem minus the variant index leaves no room for even one
    variant chunk, gvl.write raises ValueError instead of silently
    blowing the budget."""
    import pytest

    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()

    bed = sp.bed.read(source_bed)

    # Force nbytes large enough that effective_max_mem < bytes_per_var.
    # bytes_per_var = n_samples * ploidy (VCF, Genos8 = 1 byte).
    # Set nbytes = max_mem so effective_max_mem == 0.
    max_mem = 4 * 1024 * 1024
    monkeypatch.setattr(type(vcf), "nbytes", property(lambda self: max_mem))

    out = tmp_path / "test.gvl"
    with pytest.raises(ValueError, match="max_mem"):
        gvl.write(out, bed, vcf, max_mem=max_mem)


def test_write_loads_lazy_vcf_index(tmp_path, vcf_dir: Path, source_bed: Path):
    """gvl.write should load the index itself when given a VCF constructed
    with with_gvi_index=False, and produce a valid dataset."""
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz", with_gvi_index=False)
    assert vcf._index is None

    bed = sp.bed.read(source_bed)
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, vcf)

    assert (out / "metadata.json").exists()
    assert (out / "genotypes" / "variants.arrow").exists()


def test_write_loads_lazy_pgen_index(tmp_path, pgen_dir: Path, source_bed: Path):
    """gvl.write should load the index itself when given a PGEN constructed
    with load_index=False, and produce a valid dataset."""
    pgen = PGEN(pgen_dir / "filtered_source.pgen", load_index=False)
    assert pgen._index is None

    bed = sp.bed.read(source_bed)
    out = tmp_path / "test.gvl"
    gvl.write(out, bed, pgen)

    assert (out / "metadata.json").exists()
    assert (out / "genotypes" / "variants.arrow").exists()
```

Key import changes from the original:
- **Removed:** `awkward as ak`, `seqpro.rag.Ragged`, `genvarloader._utils.lengths_to_offsets`, `polars.testing.asserts.assert_frame_equal`, `Reader` from genoray, `pytest.fixture, mark`, `pytest_cases.parametrize_with_cases` — all consumed only by the deleted `test_write` and its case generators / local fixtures.
- **Kept:** `Path`, `genvarloader as gvl`, `numpy as np` (verify it's still used; if not, remove), `polars as pl` (verify), `seqpro as sp`, `PGEN, VCF` from genoray.

After writing, re-check usage:
- `np` — used? Search the file: not used in the remaining tests. **Remove `import numpy as np`.**
- `pl` — used? Not used in the remaining tests (`source_bed` is a `Path`, `sp.bed.read` returns a polars DataFrame but no `pl.` reference). **Remove `import polars as pl`.**

So the final import block should be:

```python
from pathlib import Path

import genvarloader as gvl
import seqpro as sp
from genoray import PGEN, VCF
```

- [ ] **Step 4: Run the file**

Run: `pixi run -e dev pytest tests/integration/dataset/test_write.py -q`
Expected: **3 passed**. No `test_write` collected, no skipped tests in this file anymore.

If a `ModuleNotFoundError` or `NameError` appears, you removed an import that's actually still used. Re-add it.

- [ ] **Step 5: Run the full non-slow suite**

Run: `pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2`
Expected: **`351 passed, 4 skipped, 3 deselected, 2 xfailed`** (one fewer skipped than Task 4's count — the removed `test_write` was the only skipped item in this file).

- [ ] **Step 6: Commit**

```bash
git add tests/integration/dataset/test_write.py
git commit -m "test: delete skipped test_write and its dead-only fixtures/imports"
```

Verify `git status` clean.

---

## Task 6: End-of-plan verification

- [ ] **Step 1: Full non-slow suite**

Run: `pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2`
Expected: **`351 passed, 4 skipped, 3 deselected, 2 xfailed`** — matches Task 5's final count.

- [ ] **Step 2: Slow tier still passes**

Run: `pixi run -e dev pytest tests -m slow -q 2>&1 | tail -2`
Expected: **3 passed, 508 deselected**. (Adjust 508 if needed — the deselected count is "everything not slow"; with 5 deletions it's now 511 − 159 = 352 non-slow tests + 4 skipped... actually the deselection count is what pytest reports; just verify all 3 slow tests pass and no slow tests broke.)

If running this in a worktree where `tests/data/1kg/` was not generated, the slow tests will fail with `FileNotFoundError`. That's an environment limitation, not a regression. Skip this step in that case and note it in the report.

- [ ] **Step 3: Coverage parity check**

Run: `pixi run -e dev pytest tests -m "not slow" --cov --cov-report=term -q 2>&1 | grep "^TOTAL"`
Expected: `TOTAL  ...  63%` (same as Phase 3 baseline, give or take 1 percentage point). Significant coverage drop indicates we deleted live coverage we shouldn't have — bisect against the 5 commits above.

- [ ] **Step 4: Confirm commit graph**

Run: `git log --oneline -7`

Expected:
```
<sha> test: delete skipped test_write and its dead-only fixtures/imports
<sha> test: delete test_refdataset_unspliced_defaults (fixture-state assertion only)
<sha> test: delete test_interval_track.py (only test was a tautological hasattr check)
<sha> test: delete test_rs_indexing (subsumed by test_ds_indexing + test_subset)
<sha> test: delete dead test_filter_af.py (entirely skipped, missing cases)
14ee713 style: apply ruff format and lint fixes
1facf12 docs(audit): correct spec reference in audit header
```

(The 5 new commits sit on top of the Phase 1-3 work.)

- [ ] **Step 5: Final state check**

Run: `git status`
Expected: `clean — nothing to commit`.

Run: `ls tests/integration/dataset/genotypes/ tests/integration/`
Expected: `test_filter_af.py` and `test_interval_track.py` are gone; everything else intact.

---

## Out of scope (deferred to Phase 5+)

- **Phase 5 — builders + unit tier.** 77 port-bucket tests need builders before they can move to `tests/unit/`. Per-component PRs (ragged → reconstruct → variants → haps → tracks → splice → dataset polymorphism).
- **Phase 6 — integration trim.** Triggered per-area as Phase 5 lands coverage.
- **Phase 7 — CI report.** Wire `htmlcov/` upload into CI.

The audit lists 5 "portable today" candidates (`test_build_reconstructor.py`, `test_indexing.py`, `test_splice_plan.py` 6/8, `test_realign.py`, `genotypes/test_reconstruct.py`, `test_variant_utils.py`) that could move without any builder work. Those are Phase 5 material — NOT this plan.
