# Test Suite Overhaul — Phase 5 Utility Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port `tests/integration/test_utils.py` to the unit tier as a whole-file move — 5 pure-utility tests with no fixture dependencies.

**Architecture:** The audit classifies every test in this file as Port. The tests exercise three pure helpers (`bed_to_regions`, `splits_sum_le_value`, `normalize_contig_name`) using only stdlib + `numpy`/`polars`/`pytest_cases`. No `tmp_path`, no session fixtures, no `make_*` builder dependency. The cleanest move is `git mv tests/integration/test_utils.py tests/unit/test_utils.py` — file contents stay byte-identical.

**Tech Stack:** pytest, pytest-cases, numpy, polars, genvarloader internals (`_dataset._utils`, `_utils`).

---

## File Structure

- **Move:** `tests/integration/test_utils.py` → `tests/unit/test_utils.py` (byte-identical)
- **No source changes.** Tested functions (`bed_to_regions`, `splits_sum_le_value`, `normalize_contig_name`) stay where they are.
- **No new builder.** The tests construct `polars.DataFrame` and `numpy` arrays inline; no shared scaffolding to extract.
- **No basename collision risk.** `tests/unit/` does not currently contain a `test_utils.py`, and `tests/integration/tracks/utils.py` is unrelated (sibling helper for `test_random_nonoverlapping.py`, not a test module).

---

### Task 1: Whole-file move and verification

**Files:**
- Move: `tests/integration/test_utils.py` → `tests/unit/test_utils.py`

- [ ] **Step 1: Pre-move sanity — confirm baseline passes**

Run:
```bash
pixi run -e dev pytest tests/integration/test_utils.py -v
```

Expected: 7 passed (3 `bed_to_regions` + 1 `splits_sum_le_value` + 3 parametrized `normalize_contig_name` cases — `pytest_cases` may report 5 depending on how it counts; the audit lists 5 cases for `normalize_contig_name` but `contig_list` is one case that internally tests a list, so total collected should be 7). If the number diverges from the previous test count audit, note it but do not block — the goal is "same count before and after move".

- [ ] **Step 2: Perform the move with `git mv`**

Run:
```bash
git mv tests/integration/test_utils.py tests/unit/test_utils.py
```

Expected: no output; `git status` shows a single rename (R) entry.

- [ ] **Step 3: Verify it's a clean rename (no content drift)**

Run:
```bash
git status --short
git diff --cached --stat
```

Expected output includes a line like:
```
R  tests/integration/test_utils.py -> tests/unit/test_utils.py
```
with `0 insertions(+), 0 deletions(-)` in the stat (pure rename).

- [ ] **Step 4: Run the moved tests in their new location**

Run:
```bash
pixi run -e dev pytest tests/unit/test_utils.py -v
```

Expected: same number of `passed` as Step 1, 0 failures, 0 errors. No `ImportError` (the test only imports from `genoray`, `genvarloader`, `numpy`, `polars`, `pytest_cases` — none of which depend on the file's location).

- [ ] **Step 5: Verify the integration file is gone and full suite still passes**

Run:
```bash
ls tests/integration/test_utils.py 2>&1 || echo "gone (expected)"
pixi run -e dev pytest -q
```

Expected:
- `ls` reports the file does not exist.
- Full non-slow suite shows **351 passed, 3 skipped, 3 deselected, 2 xfailed** (matches current baseline from the status doc — no test count change, since the file just moved).

- [ ] **Step 6: Verify unit tier still runs cleanly in isolation**

Run:
```bash
pixi run -e dev pytest tests/unit -q
```

Expected: 134 passed, 1 xfailed (129 + 5 new from the move). Wall time should still be ~1–2s for the unit tier alone — these are pure-function tests, no I/O.

- [ ] **Step 7: Ruff check on the moved file (defensive, even though contents are unchanged)**

Run:
```bash
pixi run -e dev ruff check tests/unit/test_utils.py
```

Expected: no findings. (Imports were already minimal and used; the move doesn't change them.)

- [ ] **Step 8: Commit**

Run:
```bash
git add tests/unit/test_utils.py tests/integration/test_utils.py
git commit -m "test: move test_utils.py to unit tier (utility component)

5 pure-utility tests (bed_to_regions, splits_sum_le_value,
normalize_contig_name) with no fixture dependencies. Whole-file
move per audit classification — all Port, zero Keep.

Phase 5 utility component of the test suite overhaul."
```

Expected: commit succeeds; pre-commit hooks (ruff) pass.

---

### Task 2: Update the status doc

**Files:**
- Modify: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md`

- [ ] **Step 1: Add the utility row to the "What shipped" table**

Open `docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md` and find the `## What shipped` table. Add a new row at the bottom:

```markdown
| 5 utility | test_utils.py whole-file move | 5 tests moved to `unit/test_utils.py`; no source changes; no builder needed |
```

- [ ] **Step 2: Update test-count line under "Current state"**

Find the line:
```markdown
- **Unit tier alone:** 129 passed, 1 xfailed (~12s combined, 1.3s unit alone)
```
Change `129 passed` to `134 passed` (5 new tests landed in the unit tier). Leave the xfailed count and timing notes alone; rerun timing locally if curious but don't block on it.

- [ ] **Step 3: Update the file-layout tree**

In the `### File layout` code block, add `│   ├── test_utils.py` under `tests/unit/` (alphabetical placement: after the `splice/` dir block and before `tracks/`, or wherever it sorts — match the existing tree's ordering convention, which is dirs-then-files alphabetical). Remove `│   ├── test_utils.py` from the `tests/integration/` block.

- [ ] **Step 4: Update "What's left" — strike the Utility section**

In `## What's left (by remaining component)`, remove the `#### Utility (~5 ports — likely fastest next plan)` subsection (the whole bullet under it too). The Tracks/Ref-FASTA/Dataset-polymorphism subsections stay.

- [ ] **Step 5: Update "Recommended next plan"**

Replace the body of `## Recommended next plan` so that **Tracks (broader)** is now the recommended next component (it's currently item 1 in the "After utility" list). Move the existing item-1 bullet up into the recommendation slot and renumber the follow-on list. Keep Phase 6 and Phase 7 notes intact.

- [ ] **Step 6: Bump the plans list under "Authoritative reference files"**

Add to the bulleted plan list:
```markdown
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-utility.md`
```
in alphabetical/chronological position (after `phase5-svar-link.md`).

- [ ] **Step 7: Update the header**

Change the `**As of:**` line at the top of the status doc to reflect the new commit hash from Task 1 Step 8. Run `git log -1 --format=%h` to get the hash. Update the parenthetical `(committed through ...)` to match.

- [ ] **Step 8: Commit the doc update**

Run:
```bash
git add docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md
git commit -m "docs(status): record utility component move; recommend tracks next"
```

Expected: commit succeeds; hooks pass.

---

## Self-Review Checklist

- **Spec coverage:** The status doc identifies Utility as the next plan with "5 ports, whole-file move candidate, very low risk." Task 1 performs exactly that. Task 2 keeps the status doc current. ✅
- **No placeholders:** Every step contains exact commands and expected output. ✅
- **Type/name consistency:** No new types or symbols introduced — pure file move. ✅
- **Basename collision check:** Verified `tests/unit/test_utils.py` does not exist pre-move; integration file is removed in the same rename, so the post-move state has exactly one `test_utils.py`. ✅
- **Sibling-file gotcha (per status doc note 1):** N/A — `test_utils.py` has no sibling `utils.py` in `tests/integration/`. The only `tests/integration/tracks/utils.py` belongs to `test_random_nonoverlapping.py`, which this plan does not touch.
- **Ruff cleanup gotcha (per status doc note 3):** Defensively run in Task 1 Step 7. Imports are minimal and all used; should be a no-op.
