# Test Suite Overhaul — Phase 5 Dataset Polymorphism Implementation Plan (minimal scope)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out the dataset-polymorphism component of the test suite overhaul with a minimal (YAGNI) split. Move the one test that can move without a new builder. Defer the other and record the deferral.

**Architecture:** Of the two audit-Port tests in this component, only `test_with_insertion_fill_rejects_when_no_tracks_active` is movable without new scaffolding — it uses `gvl.get_dummy_dataset()` (a gvl-shipped helper, not a project fixture). The other, `test_ds_indexing`, uses the project's `phased_vcf_gvl` toy-dataset path fixture; moving it would require a `make_dataset` builder that wraps `gvl.write(...)` with synthetic inputs — a meaningful new piece of test infrastructure with no other consumer. Deferred per YAGNI.

**Tech Stack:** pytest, genvarloader internals (`_dataset._insertion_fill.Repeat5p`).

---

## Caveat on "unit" purity

`gvl.get_dummy_dataset()` loads a small canned Dataset from a gvl-shipped on-disk artifact (genotypes + tracks may be populated). It is heavier than a pure-kernel unit test but lighter than the project's toy-dataset path fixtures (`phased_vcf_gvl` etc.). Placing this test in `tests/unit/dataset/` is consistent with the conftest-carve-out precedent (status doc gotcha 9): a gvl-shipped helper that's stable, fast, and not project-test-tier scaffolding is acceptable in the unit tier.

## File Structure

- **Create:** `tests/unit/dataset/test_with_insertion_fill.py` (1 test extracted)
- **Modify:** `tests/integration/dataset/test_dummy_dataset_insertion_fill.py` — remove the extracted function. The 2 remaining tests (e2e reconstruction calls) stay in integration.
- **No production source changes.**
- **No builder modules.**
- **Basename collision check:** No `tests/unit/dataset/test_with_insertion_fill.py` exists. The integration file (`test_dummy_dataset_insertion_fill.py`) keeps its name. No collision.

---

### Task 1: Atomic split — extract `test_with_insertion_fill_rejects_when_no_tracks_active` to unit

**Files:**
- Create: `tests/unit/dataset/test_with_insertion_fill.py`
- Modify: `tests/integration/dataset/test_dummy_dataset_insertion_fill.py:42-50` (remove the function)

- [ ] **Step 1: Pre-split baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/dataset/test_dummy_dataset_insertion_fill.py -v
```

Expected: 3 passed (`test_end_to_end_set_insertion_fill`, `test_dummy_dataset_with_default_insertion_fill_does_not_crash`, `test_with_insertion_fill_rejects_when_no_tracks_active`). If any skip, note it — `get_dummy_dataset()` could have a shape that triggers the `pytest.skip` guards in tests 1 and 2.

- [ ] **Step 2: Create the unit-side file**

Write `tests/unit/dataset/test_with_insertion_fill.py` with this exact content (lifted from the integration file lines 42–50, with imports trimmed to what this test alone needs):

```python
"""Dataset-level ``with_insertion_fill`` API tests.

Single API rejection check extracted from
``tests/integration/dataset/test_dummy_dataset_insertion_fill.py``.
Uses ``gvl.get_dummy_dataset()`` (a gvl-shipped helper) rather than a
project toy-dataset path fixture; the two remaining tests in the
integration file exercise the full reconstruction call path and stay
in the integration tier.
"""

import pytest

import genvarloader as gvl
from genvarloader._dataset._insertion_fill import Repeat5p


def test_with_insertion_fill_rejects_when_no_tracks_active():
    """A dataset with tracks disabled should reject with_insertion_fill."""
    ds = gvl.get_dummy_dataset()
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Disable tracks: view-state no longer has active tracks.
    ds_no_tracks = ds.with_tracks(False)
    with pytest.raises(ValueError, match="with_tracks"):
        ds_no_tracks.with_insertion_fill(Repeat5p())
```

- [ ] **Step 3: Remove the extracted function from the integration file**

Edit `tests/integration/dataset/test_dummy_dataset_insertion_fill.py`. Delete lines 42–50 (the `test_with_insertion_fill_rejects_when_no_tracks_active` function and the preceding blank line):

```python


def test_with_insertion_fill_rejects_when_no_tracks_active():
    """A dataset with tracks disabled should reject with_insertion_fill."""
    ds = gvl.get_dummy_dataset()
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Disable tracks: view-state no longer has active tracks.
    ds_no_tracks = ds.with_tracks(False)
    with pytest.raises(ValueError, match="with_tracks"):
        ds_no_tracks.with_insertion_fill(Repeat5p())
```

After the edit, verify:
- `Repeat5p` is no longer imported (the remaining tests only use `Constant`). The import `from genvarloader._dataset._insertion_fill import Constant, Repeat5p` should become `from genvarloader._dataset._insertion_fill import Constant`.
- Remaining tests untouched.

- [ ] **Step 4: Run both files**

Run:
```bash
pixi run -e dev pytest tests/unit/dataset/test_with_insertion_fill.py tests/integration/dataset/test_dummy_dataset_insertion_fill.py -v
```

Expected: 3 passed total (1 unit + 2 integration). If any test fails with `ImportError`, revisit Step 3's import cleanup.

- [ ] **Step 5: Ruff + pyrefly check**

Run:
```bash
pixi run -e dev ruff check tests/unit/dataset/test_with_insertion_fill.py tests/integration/dataset/test_dummy_dataset_insertion_fill.py
```

Expected: no findings. If unused imports detected in the integration file, apply autofix:
```bash
pixi run -e dev ruff check --fix tests/integration/dataset/test_dummy_dataset_insertion_fill.py
```

- [ ] **Step 6: Full suite**

Run:
```bash
pixi run -e dev pytest -q
```

Expected: 351 passed (test count unchanged: -1 from integration, +1 to unit; same pre-existing 3 errors from `test_ds_haps_1kg` per status doc gotcha 6).

- [ ] **Step 7: Commit**

Run:
```bash
git add tests/unit/dataset/test_with_insertion_fill.py tests/integration/dataset/test_dummy_dataset_insertion_fill.py
git status --short
git commit -m "$(cat <<'EOF'
test: extract with_insertion_fill rejection test to unit tier

Move test_with_insertion_fill_rejects_when_no_tracks_active to
tests/unit/dataset/test_with_insertion_fill.py. The test uses
gvl.get_dummy_dataset() (gvl-shipped helper, not a project toy-
dataset fixture) and only checks API-level rejection behavior.
The 2 remaining tests in the integration file exercise the full
reconstruction call path and stay in integration.

Phase 5 dataset polymorphism component — minimal scope (test_ds_indexing
deferred pending a make_dataset consumer; see status doc).
EOF
)"
```

Expected: commit succeeds.

---

### Task 2: Update status doc — close out the dataset polymorphism component

**Files:**
- Modify: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md`

- [ ] **Step 1: Re-read live counts**

Run:
```bash
pixi run -e dev pytest tests/unit -q 2>&1 | tail -3
pixi run -e dev pytest -q 2>&1 | tail -3
git log -1 --format=%h
```

Note the unit-tier count, full-suite count, and latest commit hash.

- [ ] **Step 2: Update header `As of` line and unit-tier count**

- Replace the hash after `committed through` with Step 1's hash.
- Update the `Unit tier alone:` line with the new count.

- [ ] **Step 3: Add plan to "Authoritative reference files"**

After the `phase5-ref-fasta.md` line, append:
```
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-dataset-polymorphism.md`
```

- [ ] **Step 4: Update file-layout tree**

- Under `tests/unit/dataset/`: add `test_with_insertion_fill.py` (alphabetical position).
- The integration file (`test_dummy_dataset_insertion_fill.py`) stays — no rename.
- Bump unit-tier count comment; lower integration-tier count comment.

- [ ] **Step 5: Add row to "What shipped"**

Append after the ref/fasta row:
```
| 5 dataset polymorphism (minimal) | atomic split of test_dummy_dataset_insertion_fill.py | 1 test (test_with_insertion_fill_rejects_when_no_tracks_active) extracted to tests/unit/dataset/test_with_insertion_fill.py; test_ds_indexing deferred (would require a `make_dataset` builder wrapping gvl.write — speculative scaffolding per YAGNI) |
```

- [ ] **Step 6: Update "What's left — Components with NO port-bucket tests remaining"**

Find the `- **Dataset polymorphism**` bullet and rewrite it to reflect the closeout:

```markdown
- **Dataset polymorphism (closed, minimal scope)** — `test_with_insertion_fill_rejects_when_no_tracks_active` moved to unit. `test_dataset.py:test_ds_indexing` deferred: porting requires a `make_dataset` builder that wraps `gvl.write()` with synthetic BED + variants + tracks + reference inputs. That builder has no other consumer; per YAGNI, build it when a real need surfaces.
```

(The Haps bullet stays unchanged.)

- [ ] **Step 7: Update "Recommended next plan"**

All component plans are now done. Replace the entire Recommended next plan section:

```markdown
## Recommended next plan

All component-level plans are complete. The next two phases (per design spec):

1. **Phase 6 (integration trim)** — Review each remaining integration-tier file. Where unit coverage now strictly subsumes an integration test, delete the redundancy. Candidates worth examining first: integration files where the unit-tier extraction left a thin shell (e.g. `test_dummy_dataset_insertion_fill.py`, `test_ref_ds_splicing.py`, `test_write_tracks_e2e.py`, `test_dataset.py`).
2. **Phase 7 (CI report)** — Wire `htmlcov/` upload into CI per the design spec.

Phase 6 should land as its own plan; Phase 7 is a small CI-config change that can probably ride along with whatever PR completes the overhaul.
```

- [ ] **Step 8: Commit**

Run:
```bash
git add docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-dataset-polymorphism.md
git status --short
git commit -m "docs(status): close dataset polymorphism (minimal); recommend Phase 6 next"
```

Expected: commit succeeds.

---

## Self-Review Checklist

- **Spec coverage:** The status doc identifies two audit-Port tests gated on `make_dataset`. Task 1 moves the one that doesn't actually need the builder; Task 2 records the deferral of the other and explains the rationale (no consumer for the builder beyond one test). ✅
- **YAGNI alignment:** The user explicitly chose "minimal" scope over building speculative test infrastructure. No new builder modules, no scaffolding. ✅
- **No placeholders:** Every code edit shows exact before/after; every command has expected output. ✅
- **Basename collision:** No `tests/unit/dataset/test_with_insertion_fill.py` pre-move; integration file's name is unchanged. ✅
- **Import hygiene:** Step 3 explicitly addresses the now-unused `Repeat5p` import in the trimmed integration file.
- **Status doc:** The closeout documents both what shipped (1 test moved) and what didn't ship (test_ds_indexing deferred with reason). Readers picking this up later will know why.
