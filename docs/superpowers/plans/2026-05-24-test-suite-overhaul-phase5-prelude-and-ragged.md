# Test Suite Overhaul — Phase 5 Prelude + Ragged Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish `tests/unit/` as a real tier (not just an empty scaffold) by (a) relocating the audit's "portable today" tests (no builders needed) and (b) landing the first real builder module (`tests/_builders/ragged.py`) along with the ragged-specific tests that exercise it.

**Architecture:** Pure relocations come first (zero risk — `git mv` + import-path fixes). Then a small builder module that provides `make_ragged_intervals` and `make_ragged_seqs` helpers. Then extract three ragged-specific test functions (`test_rc_returns_packed_buffer`, `test_rc`, and the whole of `tracks/test_i2t_t2i.py`) into `tests/unit/`. Suite count stays consistent at every commit.

**Tech Stack:** pytest, pytest-cases, numpy, awkward, seqpro's `Ragged`. No production code changes — this plan only adds test infra and moves tests.

**Scope:** Covers spec Phase 5's "prelude" (audit's portable-today list) plus the first builder component (ragged). Phases 5-remaining components (reconstruct, variants, haps, tracks-broader, splice-broader, dataset-polymorphism) and Phases 6-7 get separate plans.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`
- Audit: `docs/superpowers/specs/2026-05-24-test-audit.md` (Recommendations section + per-file classification)

---

## Pre-flight baseline (Phase 4 final state)

- Non-slow tier: **351 passed, 3 skipped, 3 deselected, 2 xfailed**
- Slow tier (`-m slow`, where 1kg data is generated): **3 passed**
- Coverage: **63%**

Track count drift per task — every relocation MUST keep the pass count constant (a test moved from integration/ to unit/ runs in exactly one place now, not both).

---

## File destination map

The audit recommends specific destinations for each port; these tasks land them in the canonical structure.

| Source (integration) | Destination (unit) | Notes |
|---|---|---|
| `tests/integration/dataset/test_build_reconstructor.py` | `tests/unit/dataset/test_build_reconstructor.py` | Whole-file move |
| `tests/integration/dataset/test_indexing.py` | `tests/unit/dataset/test_indexing.py` | Whole-file move |
| `tests/integration/dataset/test_realign.py` | `tests/unit/tracks/test_realign.py` | Whole-file move (relocates across subdir) |
| `tests/integration/dataset/genotypes/test_reconstruct.py` | `tests/unit/dataset/genotypes/test_reconstruct.py` | Whole-file move |
| `tests/integration/variants/test_variant_utils.py` | `tests/unit/variants/test_variant_utils.py` | Whole-file move |
| `tests/integration/dataset/test_splice_plan.py` (8 of 9 tests) | `tests/unit/splice/test_splice_plan.py` | Partial move; one test goes to `tracks/` (next row) |
| `tests/integration/dataset/test_splice_plan.py::test_tracks_call_float32_splice_plan` | `tests/unit/tracks/test_tracks_splice.py` | One-test extraction |
| `tests/integration/tracks/test_i2t_t2i.py` | `tests/unit/tracks/test_i2t_t2i.py` | Whole-file move |
| `tests/integration/dataset/test_rc_packing.py::test_rc_returns_packed_buffer` (+ its 3 `case_*` helpers) | `tests/unit/ragged/test_ragged_rc_packing.py` | Extraction (file retains 9 Keep tests) |
| `tests/integration/dataset/genotypes/test_rag_variants.py` | `tests/unit/ragged/test_rag_variants.py` | Whole-file move (both tests are Port) |

Builders introduced:
- `tests/_builders/ragged.py` — `make_ragged_seqs(rows: list[bytes]) -> Ragged`, `make_ragged_intervals(per_region: list[list[tuple[int, int, float]]]) -> RaggedIntervals`. Minimal scope — only what the ragged-component tests would actually consume going forward.

---

## Conventions used throughout this plan

- All commands assume working directory `/Users/david/projects/GenVarLoader/.claude/worktrees/test-suite-overhaul`.
- Run pytest via `pixi run -e dev pytest ...`.
- Use `git mv` for renames so history follows; never `cp + rm`.
- One commit per task. Suite stays green at every commit.
- For partial extractions: cut the test function block (and its case helpers if private to that test) and paste into the new unit file with original semantics; remove from integration file. Import adjustments happen in the same commit.

---

## Task 1: Scaffold `tests/unit/` subdirectories

**Files:**
- Create: `tests/unit/dataset/.gitkeep`
- Create: `tests/unit/dataset/genotypes/.gitkeep`
- Create: `tests/unit/tracks/.gitkeep`
- Create: `tests/unit/variants/.gitkeep`
- Create: `tests/unit/ragged/.gitkeep`
- Create: `tests/unit/splice/.gitkeep`
- Delete: `tests/unit/.gitkeep` (replaced by deeper structure)

- [ ] **Step 1: Create the directories with markers**

```bash
mkdir -p tests/unit/dataset/genotypes tests/unit/tracks tests/unit/variants tests/unit/ragged tests/unit/splice
touch tests/unit/dataset/.gitkeep tests/unit/dataset/genotypes/.gitkeep \
      tests/unit/tracks/.gitkeep tests/unit/variants/.gitkeep \
      tests/unit/ragged/.gitkeep tests/unit/splice/.gitkeep
git rm tests/unit/.gitkeep
```

- [ ] **Step 2: Verify collection is clean**

```
pixi run -e dev pytest tests/unit --collect-only -q
```

Expected: 0 tests collected; no errors.

- [ ] **Step 3: Run full non-slow suite (sanity check — no regression from scaffolding)**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: **`351 passed, 3 skipped, 3 deselected, 2 xfailed`** (matches Phase 4 final).

- [ ] **Step 4: Commit**

```bash
git add tests/unit
git commit -m "test: scaffold unit/ subdirectories by component"
```

---

## Task 2: Move `test_build_reconstructor.py` to unit tier

**Audit row:** 12 Port tests, 0 Keep. Whole-file move.

- [ ] **Step 1: Move the file**

```bash
git mv tests/integration/dataset/test_build_reconstructor.py tests/unit/dataset/test_build_reconstructor.py
```

- [ ] **Step 2: Verify the moved file still runs (no path-depth dependencies expected)**

```
pixi run -e dev pytest tests/unit/dataset/test_build_reconstructor.py -q
```

Expected: 12 passed (the audit lists 12 logical tests in this file).

If failures appear that look like missing fixtures: check whether the file imports anything from `tests.conftest` or uses `tests/data/`. Per the audit it uses only `Mock` — no fixtures, no data deps. Failures here mean the audit was wrong or the file changed since; STOP and report BLOCKED.

- [ ] **Step 3: Run the full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: **`351 passed, 3 skipped, 3 deselected, 2 xfailed`** (count unchanged — the test now runs from unit/ instead of integration/, but runs exactly once).

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move test_build_reconstructor to unit/dataset/"
```

---

## Task 3: Move `test_indexing.py` to unit tier

**Audit row:** 9 Port tests, 0 Keep. Whole-file move; uses only synthetic `DatasetIndexer`.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/test_indexing.py tests/unit/dataset/test_indexing.py
```

- [ ] **Step 2: Run the moved file**

```
pixi run -e dev pytest tests/unit/dataset/test_indexing.py -q
```

Expected: roughly 9 passed (the audit counts 9 logical tests; some parametrize so the raw pass count may be higher).

- [ ] **Step 3: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move test_indexing to unit/dataset/"
```

---

## Task 4: Move `test_realign.py` to unit/tracks/

**Audit row:** 1 Port test (`test_sparse`, 4 parametrized cases). Direct kernel invocation. Move to `unit/tracks/` because it tests `shift_and_realign_track_sparse` — a track kernel.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/test_realign.py tests/unit/tracks/test_realign.py
```

- [ ] **Step 2: Run the moved file**

```
pixi run -e dev pytest tests/unit/tracks/test_realign.py -q
```

Expected: 4 passed (4 parametrized cases of `test_sparse`).

- [ ] **Step 3: Full suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move test_realign to unit/tracks/"
```

---

## Task 5: Move `genotypes/test_reconstruct.py` to unit tier

**Audit row:** 1 Port test (`test_sparse`, 4 cases). Direct numba kernel.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/genotypes/test_reconstruct.py tests/unit/dataset/genotypes/test_reconstruct.py
```

- [ ] **Step 2: Run**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_reconstruct.py -q
```

Expected: 4 passed.

- [ ] **Step 3: Full suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move genotypes/test_reconstruct to unit/dataset/genotypes/"
```

---

## Task 6: Move `variants/test_variant_utils.py` to unit tier

**Audit row:** 2 Port tests (`test_path_is_pgen`, `test_path_is_vcf`). Pure path-string predicates.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/variants/test_variant_utils.py tests/unit/variants/test_variant_utils.py
```

- [ ] **Step 2: Run**

```
pixi run -e dev pytest tests/unit/variants/test_variant_utils.py -q
```

Expected: 2 passed.

- [ ] **Step 3: Full suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move variants/test_variant_utils to unit/variants/"
```

---

## Task 7: Split `test_splice_plan.py` — extract `test_tracks_call_float32_splice_plan`

This file has 9 tests, all Port-classified, but the audit recommends 8 go to `unit/splice/` and 1 goes to `unit/tracks/`. We do the extraction first (so the file is then 8 tests), then move the remainder in Task 8.

**Audit row:**
- `test_tracks_call_float32_splice_plan` → `tests/unit/tracks/test_tracks_splice.py` (tests `Tracks._call_float32` with a `SplicePlan`)

- [ ] **Step 1: Inspect source**

```
sed -n '230,290p' tests/integration/dataset/test_splice_plan.py
```

Locate the `test_tracks_call_float32_splice_plan` function (starts at line ~235 per `grep "^def "`). Note its full body, any helper functions it uses, and its imports.

- [ ] **Step 2: Create `tests/unit/tracks/test_tracks_splice.py`**

Create the new file containing:
- A module docstring: `"""Unit tests for Tracks._call_float32 with SplicePlan inputs."""`
- ALL imports the extracted test needs (copy from the source file, prune to what the test actually uses — `numpy as np`, `pytest`, the relevant `genvarloader._dataset` symbols, etc.).
- The `test_tracks_call_float32_splice_plan` function body, copied verbatim.
- Any test-private helper functions it depends on (likely none for this test — verify by reading the source).

Reference the source file at `tests/integration/dataset/test_splice_plan.py` (still present at this step) and copy from there.

- [ ] **Step 3: Run the new file**

```
pixi run -e dev pytest tests/unit/tracks/test_tracks_splice.py -q
```

Expected: 1 passed. If `ImportError`/`NameError`: missing import or missing helper; fix.

- [ ] **Step 4: Remove the test from the source file**

Edit `tests/integration/dataset/test_splice_plan.py`: delete the entire `def test_tracks_call_float32_splice_plan(...): ...` function, including its decorators and the blank lines that frame it. Do NOT remove any helpers it shared with other tests — only remove imports that became completely unused.

- [ ] **Step 5: Run both files**

```
pixi run -e dev pytest tests/integration/dataset/test_splice_plan.py tests/unit/tracks/test_tracks_splice.py -q
```

Expected: total = 9 (was 9 in the original file; now 8 in source + 1 in new file).

- [ ] **Step 6: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 7: Commit**

```bash
git add tests/integration/dataset/test_splice_plan.py tests/unit/tracks/test_tracks_splice.py
git commit -m "test: extract test_tracks_call_float32_splice_plan to unit/tracks/"
```

---

## Task 8: Move remaining `test_splice_plan.py` to unit/splice/

After Task 7, the file has 8 tests (7 `test_plan_*` + `test_ref_call_with_plan_writes_per_element_layout`). All belong in `unit/splice/`.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/test_splice_plan.py tests/unit/splice/test_splice_plan.py
```

- [ ] **Step 2: Run the moved file**

```
pixi run -e dev pytest tests/unit/splice/test_splice_plan.py -q
```

Expected: 8 passed.

`test_ref_call_with_plan_writes_per_element_layout` uses the `ref_fasta` fixture from `tests/conftest.py` (path-only, session-scoped). The conftest is at `tests/conftest.py`, so it applies to anything under `tests/`. No change needed.

- [ ] **Step 3: Full suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move remaining test_splice_plan to unit/splice/"
```

---

## Task 9: Move `tracks/test_i2t_t2i.py` to unit tier

**Audit row:** 2 Port tests (`test_intervals_to_tracks`, `test_tracks_to_intervals`). Synthetic `RaggedIntervals` inputs. Whole-file move.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/tracks/test_i2t_t2i.py tests/unit/tracks/test_i2t_t2i.py
```

- [ ] **Step 2: Run**

```
pixi run -e dev pytest tests/unit/tracks/test_i2t_t2i.py -q
```

Expected: 4 passed (2 tests × 2 parametrized cases each per the audit row).

- [ ] **Step 3: Full suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move tracks/test_i2t_t2i to unit/tracks/"
```

---

## Task 10: Move `genotypes/test_rag_variants.py` to unit/ragged/

**Audit row:** 2 Port tests (`test_infer_germ_ccfs`, `test_rc`). Both use awkward-array constructors directly to build `RaggedVariants` — no `Dataset`.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/genotypes/test_rag_variants.py tests/unit/ragged/test_rag_variants.py
```

- [ ] **Step 2: Run**

```
pixi run -e dev pytest tests/unit/ragged/test_rag_variants.py -q
```

Expected: 9 passed (6 cases of `test_infer_germ_ccfs` + 3 cases of `test_rc`).

- [ ] **Step 3: Full suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move genotypes/test_rag_variants to unit/ragged/"
```

---

## Task 11: Add `tests/_builders/ragged.py` with minimal helpers

Now we shift from "moves" to "builds the first builder". This is the smallest useful version — only the helpers that future ragged tests would naturally consume. No over-engineering.

**Files:**
- Create: `tests/_builders/ragged.py`

- [ ] **Step 1: Create `tests/_builders/ragged.py`** with this exact content:

```python
"""In-memory ragged-array builders for tests.

Two helpers cover the two shapes that ragged-component tests need:

- ``make_ragged_seqs`` — wrap a list of byte rows as a ``Ragged[S1]``.
- ``make_ragged_intervals`` — wrap a list-of-lists of (start, end, value)
  triples as a ``RaggedIntervals`` (the structured-dtype ragged used by
  BigWig output and track plumbing).

Neither helper opens a Dataset or touches disk. They exist so unit tests
can construct synthetic ragged inputs without re-implementing the
``Ragged.from_lengths`` / structured-dtype boilerplate inline.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import Ragged

from genvarloader._ragged import INTERVAL_DTYPE, RaggedIntervals


def make_ragged_seqs(rows: list[bytes]) -> Ragged:
    """Build a ``Ragged[S1]`` from a list of byte-string rows.

    Each row becomes one ragged element; lengths are derived from each
    row's byte length.

    Example::

        rag = make_ragged_seqs([b"ACGT", b"NN", b"GGGG"])
        rag.lengths  # array([4, 2, 4], dtype=int32)
        rag.data     # ascii bytes for "ACGTNNGGGG" as S1
    """
    if not rows:
        data = np.empty(0, dtype="S1")
        lengths = np.empty(0, dtype=np.int32)
        return Ragged.from_lengths(data, lengths)

    joined = b"".join(rows)
    data = np.frombuffer(joined, dtype="S1")
    lengths = np.array([len(r) for r in rows], dtype=np.int32)
    return Ragged.from_lengths(data, lengths)


def make_ragged_intervals(
    per_region: list[list[tuple[int, int, float]]],
) -> RaggedIntervals:
    """Build a ``RaggedIntervals`` from a per-region list of intervals.

    ``per_region[i]`` is the list of ``(start, end, value)`` triples for
    region ``i``. Empty inner lists are allowed (region with no
    intervals).

    Example::

        rag = make_ragged_intervals([
            [(0, 10, 1.0), (10, 20, 2.0)],
            [],
            [(5, 15, 0.5)],
        ])
        rag.lengths  # array([2, 0, 1], dtype=int32)
    """
    rows: list[NDArray] = []
    lengths = np.array([len(r) for r in per_region], dtype=np.int32)
    for region in per_region:
        if not region:
            continue
        arr = np.empty(len(region), dtype=INTERVAL_DTYPE)
        for i, (start, end, value) in enumerate(region):
            arr["start"][i] = start
            arr["end"][i] = end
            arr["value"][i] = value
        rows.append(arr)

    if not rows:
        data = np.empty(0, dtype=INTERVAL_DTYPE)
    else:
        data = np.concatenate(rows)
    return RaggedIntervals.from_lengths(data, lengths)
```

- [ ] **Step 2: Smoke-test the builders interactively**

Run:

```
pixi run -e dev python -c "
from tests._builders.ragged import make_ragged_seqs, make_ragged_intervals

rag = make_ragged_seqs([b'ACGT', b'NN', b'GGGG'])
assert rag.lengths.tolist() == [4, 2, 4]
assert bytes(rag.data.tobytes()) == b'ACGTNNGGGG'

itv = make_ragged_intervals([[(0, 10, 1.0), (10, 20, 2.0)], [], [(5, 15, 0.5)]])
assert itv.lengths.tolist() == [2, 0, 1]
assert itv.data['start'].tolist() == [0, 10, 5]
assert itv.data['end'].tolist() == [10, 20, 15]
assert abs(itv.data['value'][0] - 1.0) < 1e-6
print('builders OK')
"
```

Expected output: `builders OK`. If `ImportError`, check that `tests` is importable (the directory needs `__init__.py`-style discovery — pytest handles this, but `python -c` may not). If that fails, run the same checks inside a pytest test instead by adding a temporary `tests/unit/ragged/test_builders_smoke.py` with `from _builders.ragged import ...` and pytest's import-mode handling.

If the temporary smoke test path is needed: write it, run with pytest, then DELETE it before commit. The commit must NOT include any smoke test file.

- [ ] **Step 3: Full suite (no actual changes to consumed code, but verify nothing broke)**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add tests/_builders/ragged.py
git commit -m "test(builders): add ragged in-memory builders (make_ragged_seqs, make_ragged_intervals)"
```

Verify `git status` clean. If the temporary smoke test from Step 2 is present, remove it before commit.

---

## Task 12: Extract `test_rc_returns_packed_buffer` to unit/ragged/

This is the only Port test in `tests/integration/dataset/test_rc_packing.py` (the other 9 tests in the file are Keep-as-integration). We extract surgically: move the test function and its three `case_*` helpers into a new unit file; leave the rest of the integration file intact.

**Files:**
- Create: `tests/unit/ragged/test_ragged_rc_packing.py`
- Modify: `tests/integration/dataset/test_rc_packing.py` (remove the extracted test, its 3 case functions, and any newly-unused imports)

- [ ] **Step 1: Read the source area**

```
sed -n '1,76p' tests/integration/dataset/test_rc_packing.py
```

Expected lines 1-76 contain the file docstring, imports, the `_buffer_matches_lengths` helper, three case functions (`case_all_false`, `case_all_true`, `case_mixed`), and `test_rc_returns_packed_buffer`.

- [ ] **Step 2: Create `tests/unit/ragged/test_ragged_rc_packing.py`**

Write exactly:

```python
"""Unit tests for the ``_rc`` packing invariant on ``Ragged``.

Pins the behavior of ``ak.to_packed(ak.where(...))`` wrapping inside
``Dataset._rc``: the resulting ``Ragged`` must have its content buffer
length equal to the sum of its logical lengths (no doubled-buffer leak
from ``ak.where``).

Originally lived in ``tests/integration/dataset/test_rc_packing.py``;
extracted to the unit tier because it constructs synthetic ``Ragged``
inputs and exercises only the in-memory packing path — no Dataset, no
disk I/O.
"""

import awkward as ak
import numpy as np
from pytest_cases import parametrize_with_cases

from genvarloader._ragged import Ragged, reverse_complement


def _buffer_matches_lengths(rag: Ragged) -> bool:
    """Packed invariant: raw content equals the sum of the logical lengths."""
    return len(rag.data) == int(rag.lengths.sum())


def case_all_false():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([False, False, False])


def case_all_true():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([True, True, True])


def case_mixed():
    data = np.frombuffer(b"ATGCCC" * 3, dtype="S1")
    rag = Ragged.from_lengths(data, np.array([6, 6, 6]))
    return rag, np.array([False, True, False])


@parametrize_with_cases("rag, to_rc", cases=".", prefix="case_")
def test_rc_returns_packed_buffer(rag: Ragged, to_rc: np.ndarray):
    # mimic Dataset._rc exactly
    packed = Ragged(
        ak.to_packed(ak.where(to_rc, reverse_complement(rag.to_ak()), rag.to_ak()))
    )
    assert _buffer_matches_lengths(packed), (
        f"buffer doubled (len={len(packed.data)}, expected={int(packed.lengths.sum())})"
    )
    # and the content is correct for each row
    original = rag.to_ak().to_list()
    rc = reverse_complement(rag.to_ak()).to_list()
    got = packed.to_ak().to_list()
    for i, flip in enumerate(to_rc):
        expected = rc[i] if flip else original[i]
        assert got[i] == expected
```

Note: we do NOT use the new `make_ragged_seqs` builder here — the original construction is small (one line) and idiomatic. The builder will earn its place in future tests where it eliminates more boilerplate.

- [ ] **Step 3: Run the new unit test**

```
pixi run -e dev pytest tests/unit/ragged/test_ragged_rc_packing.py -q
```

Expected: 3 passed (3 parametrized cases).

- [ ] **Step 4: Remove the extracted block from the source file**

Edit `tests/integration/dataset/test_rc_packing.py`:

1. Delete lines containing the comment banner `# ---` separating the "Unit" section (the comment block around line 37-39 saying "Unit: _rc produces a packed layout regardless of the to_rc mask").
2. Delete `_buffer_matches_lengths`, `case_all_false`, `case_all_true`, `case_mixed`, and `test_rc_returns_packed_buffer`.
3. Audit imports: `awkward as ak` and `reverse_complement` were used both by the extracted test AND by other tests in the file. Grep the remaining file: `grep -n "ak\.\|reverse_complement\|Ragged\b" tests/integration/dataset/test_rc_packing.py`. Keep an import if it's still referenced anywhere else; remove it otherwise. (Realistically, `ak` and `reverse_complement` and `Ragged` are likely all still used by the integration-side tests — only the `Ragged` from `genvarloader._ragged` import line and `parametrize_with_cases` may now be unused depending on the rest of the file. Verify, don't guess.)
4. The first comment banner `# ---` immediately above the deleted section can stay (it's a section header for what follows); the second `# ---` banner below the deleted section was the start of the "Integration fixtures" section and should also stay.

- [ ] **Step 5: Confirm the integration file still passes**

```
pixi run -e dev pytest tests/integration/dataset/test_rc_packing.py -q
```

Expected: 9 passed or 9 passed with skips (the other 9 tests in the file — `test_unspliced_single_item_buffer_packed`, etc., some of which are env-gated `@pytest.mark.skipif`).

Some tests in this file (`test_cds_*`) are gated by env vars (`GVL_TEST_CDS_DS` / `GVL_TEST_CDS_REF`) and will be skipped when those aren't set. Skips here are expected; do not treat them as failures.

- [ ] **Step 6: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed` (the 3 extracted parametrized cases ran in unit/ instead of integration/ — same total).

- [ ] **Step 7: Commit**

```bash
git add tests/integration/dataset/test_rc_packing.py tests/unit/ragged/test_ragged_rc_packing.py
git commit -m "test: extract test_rc_returns_packed_buffer to unit/ragged/"
```

Verify `git status` clean.

---

## Task 13: End-of-plan verification

- [ ] **Step 1: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: **`351 passed, 3 skipped, 3 deselected, 2 xfailed`** (same as Phase 4 baseline — no tests gained or lost, only moved).

- [ ] **Step 2: Unit tier collection check**

```
pixi run -e dev pytest tests/unit -q --collect-only 2>&1 | tail -5
```

Expected: a non-zero number of tests collected from each of `tests/unit/dataset/`, `tests/unit/tracks/`, `tests/unit/variants/`, `tests/unit/ragged/`, `tests/unit/splice/`, `tests/unit/dataset/genotypes/`. Roughly: 50+ tests across the unit tier (12 build_reconstructor + 9+ indexing + 4 realign + 4 reconstruct + 2 variant_utils + 8 splice_plan + 1 tracks_splice + 4 i2t_t2i + 9 rag_variants + 3 ragged_rc_packing ≈ 56+).

If any subdirectory shows zero collected tests, the moves missed it — investigate.

- [ ] **Step 3: Unit tier runs green standalone**

```
pixi run -e dev pytest tests/unit -q 2>&1 | tail -2
```

Expected: all unit tests pass. No skips, no xfailed, no errors. (Unit tier should be fast and deterministic by design.)

- [ ] **Step 4: Coverage parity**

```
pixi run -e dev pytest tests -m "not slow" --cov --cov-report=term -q 2>&1 | grep "^TOTAL"
```

Expected: `TOTAL ... 63%` (give or take 1pp). Significant coverage drift would mean a test was lost in the move — bisect.

- [ ] **Step 5: Confirm commit graph**

```
git log --oneline -15
```

Expected: 12 new commits on top of Phase 4 (scaffold + 9 moves/extractions + 1 builder + 1 final). The exact list:

```
<sha> test: extract test_rc_returns_packed_buffer to unit/ragged/
<sha> test(builders): add ragged in-memory builders (make_ragged_seqs, make_ragged_intervals)
<sha> test: move genotypes/test_rag_variants to unit/ragged/
<sha> test: move tracks/test_i2t_t2i to unit/tracks/
<sha> test: move remaining test_splice_plan to unit/splice/
<sha> test: extract test_tracks_call_float32_splice_plan to unit/tracks/
<sha> test: move variants/test_variant_utils to unit/variants/
<sha> test: move genotypes/test_reconstruct to unit/dataset/genotypes/
<sha> test: move test_realign to unit/tracks/
<sha> test: move test_indexing to unit/dataset/
<sha> test: move test_build_reconstructor to unit/dataset/
<sha> test: scaffold unit/ subdirectories by component
bebfa0f style: strip trailing blank lines from end-of-file edits
b001d6d test: delete skipped test_write and its dead-only fixtures/imports
...
```

- [ ] **Step 6: Final tree shape**

```
/bin/ls tests/unit
```

Expected:
```
dataset/
ragged/
splice/
tracks/
variants/
```

(The original `tests/unit/.gitkeep` is gone; subdirectory `.gitkeep` markers may or may not remain depending on whether each subdir now contains real test files — pytest doesn't care either way.)

```
/bin/ls tests/integration
```

Expected: still has `dataset/`, `tracks/`, `variants/`, and the top-level `test_*.py` files. The moved/extracted files are gone from their original locations.

```
/bin/ls tests/integration/dataset/genotypes
```

Expected: `test_choose_exonic_variants.py` (still here — not in this plan's scope). `test_rag_variants.py` and `test_reconstruct.py` are gone (moved to unit/).

---

## Out of scope (deferred to Phase 5-remaining)

The audit lists 77 Port-bucket tests; this plan moves a subset (~56 tests). The remainder require real builders for components beyond ragged:

- **reconstruct component** — beyond `test_reconstruct.py` already moved here, the haplotype reconstruction kernel tests dispersed across other files.
- **variants component** — `_Variants.from_table`, info_fields filter, dosage gating. Needs `make_variants` / `make_variants_table` builders.
- **haps component** — `Haps.from_path` and `var_fields` plumbing. Needs `make_haps` builder.
- **tracks component (broader)** — insertion-fill strategies, write-pipeline validation. Needs richer track builders.
- **splice component (broader)** — splice-related dataset polymorphism. Needs `make_dataset` integration.
- **dataset polymorphism** — `__getitem__`, `subset_to`, `with_settings`, `with_seqs`. Needs `make_dataset` / `make_haps_dataset` builders.

Each becomes a separate plan after this one ships, in roughly the order listed above (matches the spec's component sequence: ragged → reconstruct → variants → haps → tracks → splice → dataset polymorphism).
