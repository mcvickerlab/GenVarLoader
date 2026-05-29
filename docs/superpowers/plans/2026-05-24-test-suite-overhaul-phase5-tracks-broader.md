# Test Suite Overhaul — Phase 5 Tracks (broader) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the remaining Tracks-component tests from integration to unit per the audit: a whole-file move of `test_random_nonoverlapping.py` (with its sibling `utils.py`), an atomic split of `test_write_tracks.py` (1 Port / 3 Keep), and a whole-file move of `test_table.py` (12 Ports, 0 Keeps).

**Architecture:** Three independent file operations, each isolated in its own task. The first preserves a sibling-import relationship by moving both files together. The second handles a basename collision by renaming the integration file (precedent from the reconstruct plan). The third is a clean whole-file move. No production source changes. No new builder is required — `tmp_path` is the only fixture in the entire scope, and only as a working directory.

**Tech Stack:** pytest, pytest-cases, numpy, polars, genvarloader internals (`_table`, `_utils`).

---

## File Structure

- **Move + sibling-import preservation:**
  - `tests/integration/tracks/test_random_nonoverlapping.py` → `tests/unit/tracks/test_random_nonoverlapping.py`
  - `tests/integration/tracks/utils.py` → `tests/unit/tracks/utils.py`
- **Atomic split with integration rename (basename collision):**
  - `tests/integration/dataset/test_write_tracks.py` → `tests/integration/dataset/test_write_tracks_e2e.py` (3 Keeps)
  - New file `tests/unit/dataset/test_write_tracks.py` (1 Port: `test_write_duplicate_track_names_rejected`)
- **Whole-file move:**
  - `tests/integration/test_table.py` → `tests/unit/test_table.py`
- **No production source changes.** The functions under test (`Table`, `gvl.write`, helper `nonoverlapping_intervals`) stay where they are.
- **No new builder modules.** Test inputs are constructed inline with `polars.DataFrame` / `numpy` literals.

---

### Task 1: Move `test_random_nonoverlapping.py` + sibling `utils.py`

**Files:**
- Move: `tests/integration/tracks/test_random_nonoverlapping.py` → `tests/unit/tracks/test_random_nonoverlapping.py`
- Move: `tests/integration/tracks/utils.py` → `tests/unit/tracks/utils.py`

**Why both files move together:** The test does `from utils import nonoverlapping_intervals` which relies on pytest's prepend-mode `sys.path` injection adding the test's own directory. If `utils.py` stays in `tests/integration/tracks/` while the test moves to `tests/unit/tracks/`, the import breaks. Moving both together preserves the import. The audit explicitly recommends this.

- [ ] **Step 1: Pre-move baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/tracks/test_random_nonoverlapping.py -v
```

Expected: 4 passed (one parametrized test with 4 cases: `gaps_no_max_width`, `gaps_with_max_width`, `no_gaps_at_min_width`, `no_gaps_at_max_width`).

- [ ] **Step 2: Verify nothing else imports the sibling `utils.py`**

Run:
```bash
grep -rn "from utils import\|import utils" tests/integration tests/unit 2>&1 | grep -v "Binary file"
```

Expected: exactly one match — `tests/integration/tracks/test_random_nonoverlapping.py:3:from utils import nonoverlapping_intervals`. If any other file imports `utils`, STOP and reassess: the move could break that file.

- [ ] **Step 3: Confirm `tests/unit/tracks/` already exists**

Run:
```bash
ls tests/unit/tracks/
```

Expected: directory exists, contains e.g. `test_i2t_t2i.py`, `test_insertion_fill.py`, `test_realign.py`, `test_tracks_splice.py`. (It was created by prior Phase 5 work.)

- [ ] **Step 4: Move both files**

Run:
```bash
git mv tests/integration/tracks/test_random_nonoverlapping.py tests/unit/tracks/test_random_nonoverlapping.py
git mv tests/integration/tracks/utils.py tests/unit/tracks/utils.py
```

Expected: no output. `git status --short` should show two `R` entries (renames).

- [ ] **Step 5: Verify the renames are clean (no content drift)**

Run:
```bash
git status --short
git diff --cached --stat
```

Expected:
```
R  tests/integration/tracks/test_random_nonoverlapping.py -> tests/unit/tracks/test_random_nonoverlapping.py
R  tests/integration/tracks/utils.py -> tests/unit/tracks/utils.py
```
with `0 insertions(+), 0 deletions(-)` (pure renames).

- [ ] **Step 6: Run the moved test in its new location**

Run:
```bash
pixi run -e dev pytest tests/unit/tracks/test_random_nonoverlapping.py -v
```

Expected: 4 passed. No `ImportError: No module named 'utils'`. (If this fails with an import error, the prepend-mode injection is not finding the new `tests/unit/tracks/utils.py`; reassess pytest config before proceeding.)

- [ ] **Step 7: Confirm the now-empty integration tracks dir still works**

Run:
```bash
ls tests/integration/tracks/
pixi run -e dev pytest tests/integration/tracks -v
```

Expected:
- `ls` shows the directory still contains `test_annot_tracks.py` and possibly a `__pycache__`. (No `__init__.py` should exist — the suite has no per-dir packages.)
- pytest run on the remaining file passes its tests.

- [ ] **Step 8: Full suite + ruff**

Run:
```bash
pixi run -e dev pytest -q
pixi run -e dev ruff check tests/unit/tracks/test_random_nonoverlapping.py tests/unit/tracks/utils.py
```

Expected:
- 351 passed (matches current baseline; 3 errors from `test_ds_haps_1kg` in this worktree are pre-existing per status doc note 6 — ignore).
- Ruff: no findings.

- [ ] **Step 9: Commit**

Run:
```bash
git add tests/unit/tracks/test_random_nonoverlapping.py tests/unit/tracks/utils.py
git status --short  # confirm the renames are staged
git commit -m "test: move test_random_nonoverlapping + utils.py to unit tier

Both files move together to preserve the prepend-mode sibling import
(\`from utils import nonoverlapping_intervals\`). 4 parametrized cases;
no fixture dependencies; the tested helper is itself a test utility,
not library code.

Phase 5 tracks (broader) component."
```

Expected: commit succeeds. Pre-commit hooks (ruff, pyrefly, commitizen) pass.

---

### Task 2: Atomic split of `test_write_tracks.py` (rename integration to avoid basename collision)

**Files:**
- Rename (integration side, 3 Keeps): `tests/integration/dataset/test_write_tracks.py` → `tests/integration/dataset/test_write_tracks_e2e.py`
- Create (unit side, 1 Port extracted): `tests/unit/dataset/test_write_tracks.py`

**Why rename the integration file:** Per status doc gotcha 1, pytest's default `--import-mode=prepend` treats same-basename files in different dirs as the same module, causing collection errors. The reconstruct plan established the precedent: rename the integration file when the unit-tier file naturally wants the same name. The 3 remaining tests are all end-to-end `gvl.write()` roundtrips, so `_e2e` suffix is apt.

**Why this is an atomic split, not a copy:** The unit-tier test (`test_write_duplicate_track_names_rejected`) goes through `gvl.write()` but the duplicate-name `ValueError` is raised by the input-validation layer before any disk I/O. The audit classifies it Port on that basis. `tmp_path` is used purely as a never-touched path argument; this is acceptable for unit tests (built-in pytest fixture, no project-specific scaffolding).

- [ ] **Step 1: Pre-split baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/dataset/test_write_tracks.py -v
```

Expected: 4 passed (`test_write_with_table_only_roundtrip`, `test_write_with_mixed_bigwigs_and_table`, `test_write_with_variants_and_tracks`, `test_write_duplicate_track_names_rejected`).

- [ ] **Step 2: Rename integration file first**

Run:
```bash
git mv tests/integration/dataset/test_write_tracks.py tests/integration/dataset/test_write_tracks_e2e.py
```

- [ ] **Step 3: Remove the ported test from the integration file**

Open `tests/integration/dataset/test_write_tracks_e2e.py` and delete the entire `test_write_duplicate_track_names_rejected` function (lines 133–162 of the original file — the function body + the trailing blank line). Verify the remaining file structure: imports at top, `_make_bed` and `_make_table_df` helpers, then the three remaining tests (`test_write_with_table_only_roundtrip`, `test_write_with_mixed_bigwigs_and_table`, `test_write_with_variants_and_tracks`).

After the edit, the file should NOT import `pytest` (the deleted function had a local `import pytest`); confirm `import pytest` is not at the top either. None of the 3 remaining tests use pytest directly — they use `tmp_path`/`bigwig_dir`/`vcf_dir` fixtures via positional injection only.

Expected final imports block at the top of `test_write_tracks_e2e.py`:
```python
from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
from genvarloader._table import Table
```

- [ ] **Step 4: Create the unit-side file**

Create `tests/unit/dataset/test_write_tracks.py` with this exact content (verbatim from the integration file, with the local `import pytest` lifted to the module level):

```python
import genvarloader as gvl
import polars as pl
import pytest


def test_write_duplicate_track_names_rejected(tmp_path):
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    t1 = gvl.Table(
        "dup",
        pl.DataFrame({
            "sample_id": ["s0"],
            "chrom": ["chr1"],
            "start": [0],
            "end": [10],
            "value": [1.0],
        }),
    )
    t2 = gvl.Table(
        "dup",
        pl.DataFrame({
            "sample_id": ["s0"],
            "chrom": ["chr1"],
            "start": [50],
            "end": [60],
            "value": [2.0],
        }),
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        gvl.write(path=tmp_path / "x.gvl", bed=bed, tracks=[t1, t2])
```

(Note: the body is byte-equivalent to the original except the local `import pytest` has been hoisted to the module level — the canonical Python style. No behavioral change.)

- [ ] **Step 5: Run the unit-side test**

Run:
```bash
pixi run -e dev pytest tests/unit/dataset/test_write_tracks.py -v
```

Expected: 1 passed (`test_write_duplicate_track_names_rejected`). If you get an `ImportError` complaining about basename collision, STOP — the rename from Step 2 didn't take effect. Reassess.

- [ ] **Step 6: Run the integration-side renamed file**

Run:
```bash
pixi run -e dev pytest tests/integration/dataset/test_write_tracks_e2e.py -v
```

Expected: 3 passed (`test_write_with_table_only_roundtrip`, `test_write_with_mixed_bigwigs_and_table`, `test_write_with_variants_and_tracks`).

- [ ] **Step 7: Ruff and pyrefly check on both files**

Run:
```bash
pixi run -e dev ruff check tests/unit/dataset/test_write_tracks.py tests/integration/dataset/test_write_tracks_e2e.py
```

Expected: no findings. If ruff complains about an unused import in the integration file (e.g. if `Path` or `np` or `Table` becomes unused after removing the duplicate-track test), apply the suggested fix:

```bash
pixi run -e dev ruff check tests/integration/dataset/test_write_tracks_e2e.py --fix
```

Then re-run the integration tests to confirm nothing broke:
```bash
pixi run -e dev pytest tests/integration/dataset/test_write_tracks_e2e.py -v
```

(Check manually: `Path` is only used in the type hint of `_make_bed`'s `tmp_path: Path` parameter — still used. `np` is used by `test_write_with_table_only_roundtrip` — still used. `Table` is used by `test_write_with_table_only_roundtrip` — still used. So no unused imports expected, but ruff-fix is harmless.)

- [ ] **Step 8: Full suite**

Run:
```bash
pixi run -e dev pytest -q
```

Expected: 351 passed (test count unchanged: -1 from integration, +1 to unit). Same pre-existing 3 errors from `test_ds_haps_1kg`.

- [ ] **Step 9: Commit**

Run:
```bash
git add tests/unit/dataset/test_write_tracks.py tests/integration/dataset/test_write_tracks_e2e.py
git status --short
git commit -m "test: split test_write_tracks atomically — extract validation to unit

Move test_write_duplicate_track_names_rejected to
tests/unit/dataset/test_write_tracks.py (port: input validation
fires before disk I/O). Rename the integration file to
test_write_tracks_e2e.py to avoid the pytest prepend-mode basename
collision (precedent: reconstruct plan).

Phase 5 tracks (broader) component."
```

Expected: commit succeeds.

---

### Task 3: Whole-file move of `test_table.py`

**Files:**
- Move: `tests/integration/test_table.py` → `tests/unit/test_table.py`

**Why a whole-file move:** All 12 audit-classified tests are Port. No Keeps. Fixtures used: `tmp_path` (built-in) for the file-format dispatch tests and `long_df` (a local pytest fixture defined inline at file scope; not a session/shared fixture). No project fixtures. No basename collision (no existing `tests/unit/test_table.py`).

- [ ] **Step 1: Pre-move baseline**

Run:
```bash
pixi run -e dev pytest tests/integration/test_table.py -v
```

Expected: 15 collected, 15 passed. (12 tests in the audit; `test_table_from_path_long_form` is parametrized over 4 extensions, adding 3 to the collected count — so 12 + 3 = 15. Verify the exact count empirically.)

- [ ] **Step 2: Verify no basename collision**

Run:
```bash
ls tests/unit/test_table.py 2>&1 || echo "no collision (expected)"
```

Expected: file does not exist.

- [ ] **Step 3: Perform the move**

Run:
```bash
git mv tests/integration/test_table.py tests/unit/test_table.py
```

- [ ] **Step 4: Verify it's a clean rename**

Run:
```bash
git status --short
git diff --cached --stat
```

Expected:
```
R  tests/integration/test_table.py -> tests/unit/test_table.py
```
with `0 insertions(+), 0 deletions(-)` (pure rename).

- [ ] **Step 5: Run the moved tests**

Run:
```bash
pixi run -e dev pytest tests/unit/test_table.py -v
```

Expected: same passing count as Step 1 (15 passed). No ImportError. No fixture-not-found errors. (Imports are `numpy`, `polars`, `pytest`, `genvarloader._table.Table`, `genvarloader._utils.lengths_to_offsets` — none position-dependent.)

- [ ] **Step 6: Ruff check**

Run:
```bash
pixi run -e dev ruff check tests/unit/test_table.py
```

Expected: no findings.

- [ ] **Step 7: Full suite + unit tier isolation check**

Run:
```bash
pixi run -e dev pytest -q
pixi run -e dev pytest tests/unit -q
```

Expected:
- Full suite: 351 passed (unchanged total — same tests, different homes).
- Unit tier: previous count + ~15 (the `test_table.py` move). Whatever the unit tier counted at after Task 2 + 15 from Task 3.

- [ ] **Step 8: Commit**

Run:
```bash
git add tests/unit/test_table.py
git status --short
git commit -m "test: move test_table.py to unit tier (tracks broader component)

12 Table tests (init, column_map, from_path format dispatch,
count_intervals kernel, contig normalization). All audit-Port, zero
Keep. tmp_path is the only fixture used; no project scaffolding.

Phase 5 tracks (broader) component — final file."
```

Expected: commit succeeds.

---

### Task 4: Update the status doc

**Files:**
- Modify: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md`

- [ ] **Step 1: Re-read the live test counts**

Run:
```bash
pixi run -e dev pytest tests/unit -q 2>&1 | tail -3
pixi run -e dev pytest -q 2>&1 | tail -3
git log -1 --format=%h
```

Note the unit-tier passed count, the full-suite passed count, and the latest commit hash. Use these exact numbers in the updates below; **do not assume them**.

- [ ] **Step 2: Update header "As of" line**

Edit the line:
```
**As of:** 2026-05-24 (committed through `<old-hash>`)
```
to reference the new commit hash from Step 1.

- [ ] **Step 3: Add the new plan to the "Authoritative reference files" list**

After the line:
```
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-utility.md`
```
add:
```
  - `docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-tracks-broader.md`
```

- [ ] **Step 4: Update test-count line under "Current state"**

Change:
```
- **Unit tier alone:** 138 passed, 1 xfailed (...)
```
to the new unit-tier count from Step 1. Leave the wall-time note as-is unless it changed dramatically.

- [ ] **Step 5: Update the file-layout tree**

In the `### File layout` code block:
- Under `tests/unit/`: add (alphabetically among top-level files) `test_table.py`. Under `tests/unit/dataset/`: add `test_write_tracks.py`.
- Under `tests/integration/`: remove `test_table.py`. Under `tests/integration/dataset/`: rename `test_write_tracks.py` → `test_write_tracks_e2e.py`. Under `tests/integration/tracks/`: remove `test_random_nonoverlapping.py` and the (uncommitted) `utils.py` reference if present.
- Bump the comment header on `tests/unit/` from `# ← 138 tests` to the new count.
- Bump the comment header on `tests/integration/` proportionally (`# ← 222 tests` → -16 since we moved 1+1+15... actually it's -1 (test_random_nonoverlapping was 4 cases but 1 test) -1 (validation extract) -12 (test_table.py audit count = 12, but pytest collects 15 due to parametrize). Use the live count delta from Step 1 — don't hand-compute.

- [ ] **Step 6: Add a row to "What shipped"**

Append after the `5 utility` row:
```
| 5 tracks (broader) | test_random_nonoverlapping + utils.py whole-move; test_write_tracks atomic split (integration renamed to _e2e); test_table.py whole-file move | 3 files relocated, 1 atomic split, 1 integration rename for basename-collision avoidance |
```

- [ ] **Step 7: Remove the Tracks (broader) section from "What's left"**

Delete the entire `#### Tracks (broader) — ~14 ports across 3 files` subsection (header + the three bullets for `test_random_nonoverlapping.py`, `test_write_tracks.py`, `test_table.py`).

- [ ] **Step 8: Update "Recommended next plan"**

Replace the body of `## Recommended next plan` so that **Ref / FASTA + reference-fixture promotion** is now the top recommendation (it was item 1 in the post-tracks list). Demote / renumber the follow-on items so the new order is:

```markdown
## Recommended next plan

**Ref / FASTA + reference-fixture promotion** — One combined plan: move `test_fasta.py` (3 ports) and the 2 remaining `test_ref_ds.py` ports to the unit tier, and promote the `reference = gvl.Reference.from_path(ref_fasta, in_memory=False)` fixture to `tests/conftest.py`, deduplicating across `test_ref_ds_splicing.py` (integration), `test_ref_ds_splice_settings.py` (unit), and the new unit ref tests.

Subsequent order:

1. **Dataset polymorphism / `make_dataset`** — Last component plan. Requires the most builder work but only has 2 specific tests gated on it.
```

Keep the Phase 6 (integration trim) and Phase 7 (CI report) notes that follow.

- [ ] **Step 9: Add a "Notable decisions / gotchas" entry for the second basename-collision instance**

Append a new numbered item to the `## Notable decisions / gotchas` section (or extend gotcha 1):

```markdown
7. **Second basename-collision case (tracks-broader plan)** — `test_write_tracks.py` collided between `tests/integration/dataset/` (3 Keeps) and the new `tests/unit/dataset/` (1 Port). Resolved by renaming the integration file to `test_write_tracks_e2e.py`, consistent with the reconstruct-plan precedent (gotcha 1). Watch the same pattern for any future split that wants the natural integration filename.
```

- [ ] **Step 10: Commit the doc update**

Run:
```bash
git add docs/superpowers/specs/2026-05-24-test-suite-overhaul-status.md docs/superpowers/plans/2026-05-24-test-suite-overhaul-phase5-tracks-broader.md
git status --short
git commit -m "docs(status): record tracks (broader) component; recommend ref/fasta next"
```

Expected: commit succeeds; hooks pass.

---

## Self-Review Checklist

- **Spec coverage:** The status doc identifies Tracks (broader) as 3 files (`test_random_nonoverlapping.py`, `test_write_tracks.py`, `test_table.py`). Each gets its own task. ✅
- **Audit alignment:** Task 1 follows the audit's explicit recommendation to "move together with `utils.py`." Task 2's atomic split matches the audit's 1-Port / 3-Keep classification. Task 3 matches the audit's 12-Port / 0-Keep classification (note: pytest collects 15 due to a 4-way parametrize on `test_table_from_path_long_form`). ✅
- **No placeholders:** Every step contains exact commands and expected output. Task 2 Step 4 quotes the full file contents verbatim. ✅
- **Type/name consistency:** No new types or symbols introduced; pure file relocations. ✅
- **Basename collision check:**
  - Task 1: no collisions — no `tests/unit/tracks/test_random_nonoverlapping.py` or `tests/unit/tracks/utils.py` pre-move. ✅
  - Task 2: collision exists (`test_write_tracks.py` in both tiers) — handled by renaming the integration file to `_e2e.py`. ✅
  - Task 3: no collision — no `tests/unit/test_table.py` pre-move. ✅
- **Sibling-file gotcha (status doc note 1):** Task 1 specifically addresses this by moving both files together. ✅
- **Ruff cleanup gotcha (status doc note 3):** Defensive ruff checks in Tasks 1, 2 (with autofix), and 3. Task 2 Step 7 explicitly handles potential unused-import fallout. ✅
- **Plan-body verbatim quoting (status doc note 4):** Task 2 Step 4's unit-side file body was lifted directly from the read of `test_write_tracks.py` lines 133–162 (re-verified by direct file read at planning time). ✅
- **Order:** Tasks 1–3 are independent file operations and could run in any order, but the linear ordering (move → split → move) is simplest and gives clear commit boundaries. Task 4 must run last.
