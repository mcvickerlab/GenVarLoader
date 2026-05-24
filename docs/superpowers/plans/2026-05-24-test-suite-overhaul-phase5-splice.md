# Test Suite Overhaul — Phase 5 Splice Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the splice component of Phase 5 — move `test_get_splice_bed.py` whole-file (11 ports) to `tests/unit/splice/`, and split `test_ref_ds_splicing.py` into 5 unit tests + 4 integration tests.

**Architecture:** Both target files are splice-related. `test_get_splice_bed.py` exercises `gvl.get_splice_bed(gtf_path)` — a public function that converts a GTF to a splice BED — using a local synthetic-GTF fixture that writes to `tmp_path`; no Dataset needed. `test_ref_ds_splicing.py` mixes 5 settings/validation tests (Port) with 4 byte-level roundtrip tests (Keep). The 5 Port tests use `gvl.RefDataset` constructed from an in-memory `Reference` plus a small polars DataFrame; they assert on state transitions and validation errors, not on output bytes.

**Tech Stack:** pytest, polars, genvarloader. No production code changes. No new builders this round — the `SplicePlan` builder envisioned in the spec design isn't needed because both targets construct their inputs inline (GTF text in one file, polars DataFrame in the other).

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`
- Audit: `docs/superpowers/specs/2026-05-24-test-audit.md` (per-file rows for `test_get_splice_bed.py` and `test_ref_ds_splicing.py`)

---

## Pre-flight baseline (after Phase 5 variants)

- Non-slow tier: **351 passed, 3 skipped, 3 deselected, 2 xfailed**
- Coverage: **63%**
- Unit tier: **106 passed, 1 xfailed**

Counts unchanged after this plan (relocations only).

---

## Test classification (from audit)

### `tests/integration/dataset/test_get_splice_bed.py` — 11 tests total

All 11 are **Port**. Whole-file move to `tests/unit/splice/test_get_splice_bed.py`. Each test takes the local `gtf_path` fixture (which `p.write_text(GTF_TEXT)`-s a synthetic GTF to `tmp_path`) and calls `gvl.get_splice_bed(gtf_path)`. Pure public-API unit tests; no Dataset, no genome data.

### `tests/integration/test_ref_ds_splicing.py` — 9 tests total

**Port (5) → `tests/unit/splice/test_ref_ds_splice_settings.py`:**

| Test | Why portable |
|---|---|
| `test_with_settings_disable_splice` | State transition: `splice_info=False` clears spliced state |
| `test_with_settings_enable_splice` | State transition: `splice_info="transcript_id"` enables splicing |
| `test_with_settings_validation` | `RuntimeError` raises for incompatible jitter+splice and deterministic+splice |
| `test_spliced_output_length_variable` | Validates output shape on `with_len("variable")` — no byte comparison |
| `test_spliced_rejects_fixed_length` | `RuntimeError` raises for splice + fixed-length combination |

**Keep-as-integration (4) → stay in `tests/integration/test_ref_ds_splicing.py`:**

| Test | Why keep |
|---|---|
| `test_spliced_single_col` | Byte-level concat verification across exons |
| `test_spliced_two_col_reorders_exons` | Byte-level: `exon_number` drives sort order |
| `test_spliced_mixed_strand` | Negative-strand RC byte-for-byte check |
| `test_subset_to_transcripts` | Cross-cutting state + byte-level output verification |

Both the Port and Keep tests use the same two fixtures (`reference` from `Reference.from_path(ref_fasta, in_memory=False)`, and `two_transcript_bed` — an inline polars DataFrame). The fixtures get duplicated into both files; that's fine because they're small and self-contained. The `_as_s1` helper (line 27 of source) is only used by Keep tests — it does NOT need to move to the unit file.

---

## Task 1: Move `test_get_splice_bed.py` to unit/splice/

Whole-file move. The audit row says 11 ports, 0 keeps.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/test_get_splice_bed.py tests/unit/splice/test_get_splice_bed.py
```

- [ ] **Step 2: Run the moved file alone**

```
pixi run -e dev pytest tests/unit/splice/test_get_splice_bed.py -q 2>&1 | tail -3
```

Expected: **14 passed** (1 of the 11 tests is `test_default_keeps_only_t1` parametrized over 4 csv/tsv/parquet/arrow... actually re-check the audit — line 28 of source is `test_default_keeps_only_t1(gtf_path)` which is NOT parametrized. Most tests are single-case; the test count should be roughly 11). If pass count differs significantly, double-check that no test imports something only available in the integration tier (none expected).

Reasonable acceptable range: 11–15 passed.

- [ ] **Step 3: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed` (unchanged — test count for the moved file is preserved).

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move test_get_splice_bed to unit/splice/"
```

Verify `git status` clean.

---

## Task 2: Split `test_ref_ds_splicing.py` (atomic create + trim)

Same atomic-commit pattern used in previous component plans.

**Files:**
- Create: `tests/unit/splice/test_ref_ds_splice_settings.py` (5 extracted Port tests + fixtures)
- Modify: `tests/integration/test_ref_ds_splicing.py` (keep 4 Keep tests + fixtures + `_as_s1` helper)

### Step 1: Create the unit file

Write `tests/unit/splice/test_ref_ds_splice_settings.py` with this exact content (bodies copied verbatim from source lines 91-140):

- [ ] **Write the file**

```python
"""Unit tests for ``RefDataset`` splice-related settings and validation.

Originally lived in tests/integration/test_ref_ds_splicing.py;
extracted to the unit tier because every test here exercises only
in-memory state transitions or validation errors — no byte-level output
comparison against the reference genome. The 4 byte-comparison tests
(``test_spliced_single_col``, ``test_spliced_two_col_reorders_exons``,
``test_spliced_mixed_strand``, ``test_subset_to_transcripts``) remain in
the original file as integration regression tests.
"""

import genvarloader as gvl
import polars as pl
import pytest


@pytest.fixture
def reference(ref_fasta) -> gvl.Reference:
    return gvl.Reference.from_path(ref_fasta, in_memory=False)


@pytest.fixture
def two_transcript_bed() -> pl.DataFrame:
    # Two transcripts, both '+' strand. T1 has 2 exons; T2 has 1 exon.
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1000, 2000, 5000],
            "chromEnd": [1010, 2010, 5010],
            "strand": [1, 1, 1],
            "transcript_id": ["T1", "T1", "T2"],
            "exon_number": [1, 2, 1],
        }
    )


def test_with_settings_disable_splice(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
    assert ds.is_spliced
    plain = ds.with_settings(splice_info=False)
    assert plain.is_spliced is False
    assert len(plain) == 3  # back to per-exon row count


def test_with_settings_enable_splice(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed)
    assert not ds.is_spliced
    sp = ds.with_settings(splice_info="transcript_id")
    assert sp.is_spliced
    assert len(sp) == 2


def test_with_settings_validation(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed, jitter=0)
    with pytest.raises(RuntimeError, match="Jitter is not supported"):
        ds.with_settings(splice_info="transcript_id", jitter=1)

    with pytest.raises(RuntimeError, match="Non-deterministic"):
        ds.with_settings(splice_info="transcript_id", deterministic=False)


def test_spliced_output_length_variable(reference, two_transcript_bed):
    ds = gvl.RefDataset(
        reference, two_transcript_bed, splice_info="transcript_id"
    ).with_len("variable")
    out = ds[:]
    # variable-length pads to the longest transcript in the batch.
    assert out.shape[0] == 2
    assert out.shape[1] == 20  # T1 has 2 × 10 = 20; T2 padded to 20.


def test_spliced_rejects_fixed_length(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed)
    with pytest.raises(RuntimeError, match="Splicing requires output_length"):
        ds.with_settings(splice_info="transcript_id").with_len(5)
```

Note: `test_spliced_output_length_variable` does invoke `ds[:]` which triggers actual reference reading — it's not strictly "no byte comparison". It only asserts on `.shape`, though, so it's still unit-tier appropriate. The `reference` fixture's `in_memory=False` mode keeps it cheap.

### Step 2: Run the new unit file alone

```
pixi run -e dev pytest tests/unit/splice/test_ref_ds_splice_settings.py -q 2>&1 | tail -3
```

Expected: **5 passed**.

If failures appear: most likely cause is import-path drift or a fixture incompatibility. Investigate, do not paper over.

### Step 3: Trim the integration file

Rewrite `tests/integration/test_ref_ds_splicing.py` to keep ONLY the 4 Keep tests + the two fixtures + `_as_s1` helper + imports they need.

- [ ] **Rewrite the file**

```python
"""End-to-end splice tests requiring real reference data + byte comparisons.

The settings/validation unit tests for ``RefDataset`` splice handling live
in ``tests/unit/splice/test_ref_ds_splice_settings.py``. The 4 tests here
all compare reconstructed sequence bytes against expected concatenations
of unspliced slices — they need the actual reference genome.
"""

import genvarloader as gvl
import numpy as np
import polars as pl
import pytest


@pytest.fixture
def reference(ref_fasta) -> gvl.Reference:
    return gvl.Reference.from_path(ref_fasta, in_memory=False)


@pytest.fixture
def two_transcript_bed() -> pl.DataFrame:
    # Two transcripts, both '+' strand. T1 has 2 exons; T2 has 1 exon.
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1"],
            "chromStart": [1000, 2000, 5000],
            "chromEnd": [1010, 2010, 5010],
            "strand": [1, 1, 1],
            "transcript_id": ["T1", "T1", "T2"],
            "exon_number": [1, 2, 1],
        }
    )


def _as_s1(x) -> np.ndarray:
    """Convert bytes or S1 array to a 1-D S1 numpy array."""
    if isinstance(x, (bytes, bytearray)):
        return np.frombuffer(x, dtype="S1")
    return np.asarray(x, dtype="S1").ravel()


def test_spliced_single_col(reference: gvl.Reference, two_transcript_bed: pl.DataFrame):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
    assert ds.is_spliced is True
    assert len(ds) == 2  # two transcripts

    spliced = ds[:]  # ragged: (2, ~l)

    unsp = gvl.RefDataset(reference, two_transcript_bed)[:]
    expected_t1 = np.concatenate([_as_s1(unsp[0]), _as_s1(unsp[1])])
    expected_t2 = _as_s1(unsp[2])

    np.testing.assert_equal(_as_s1(spliced[0]), expected_t1)
    np.testing.assert_equal(_as_s1(spliced[1]), expected_t2)
```

**Wait** — the file has 4 Keep tests, not 1. The block above shows only `test_spliced_single_col`. The implementer must include the other 3 Keep tests verbatim from the source: `test_spliced_two_col_reorders_exons` (line 49), `test_spliced_mixed_strand` (line 70), `test_subset_to_transcripts` (line 116). Read those tests from `tests/integration/test_ref_ds_splicing.py` and append them after `test_spliced_single_col`, preserving their exact bodies. Do NOT include any of the 5 Port tests.

The complete trimmed integration file should have:
- Module docstring (shown above)
- Imports: `genvarloader as gvl`, `numpy as np`, `polars as pl`, `pytest`
- `reference` fixture
- `two_transcript_bed` fixture
- `_as_s1` helper
- 4 test functions: `test_spliced_single_col`, `test_spliced_two_col_reorders_exons`, `test_spliced_mixed_strand`, `test_subset_to_transcripts`

### Step 4: Run the trimmed integration file

```
pixi run -e dev pytest tests/integration/test_ref_ds_splicing.py -q 2>&1 | tail -3
```

Expected: **4 passed** (was 9 passed before; lost 5 to the unit tier).

### Step 5: Full non-slow suite

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed` (unchanged — 5 tests moved tier, total preserved).

### Step 6: Commit atomically

```bash
git add tests/integration/test_ref_ds_splicing.py tests/unit/splice/test_ref_ds_splice_settings.py
git commit -m "test: extract RefDataset splice settings tests to unit/splice/"
```

Verify `git status` clean.

---

## Task 3: End-of-plan verification

- [ ] **Step 1: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: **`351 passed, 3 skipped, 3 deselected, 2 xfailed`**.

- [ ] **Step 2: Unit tier collection**

```
pixi run -e dev pytest tests/unit -q 2>&1 | tail -2
```

Expected: roughly **122 passed, 1 xfailed** (was 106/1; +11 from get_splice_bed move, +5 from ref_ds_splice_settings extraction).

- [ ] **Step 3: Coverage parity**

```
pixi run -e dev pytest tests -m "not slow" --cov --cov-report=term -q 2>&1 | grep "^TOTAL"
```

Expected: `TOTAL ... 63%` (±1pp).

- [ ] **Step 4: File structure**

```
/bin/ls tests/unit/splice/
```

Expected: `test_get_splice_bed.py`, `test_ref_ds_splice_settings.py`, `test_splice_plan.py` (existed). Plus possibly a leftover `.gitkeep`.

```
/bin/ls tests/integration/dataset/test_get_splice_bed.py
```

Expected: file does NOT exist (`No such file or directory`).

```
/bin/ls tests/integration/test_ref_ds_splicing.py
```

Expected: file exists, contains 4 Keep tests.

- [ ] **Step 5: Commit graph**

```
git log --oneline -4
```

Expected:
```
<sha> test: extract RefDataset splice settings tests to unit/splice/
<sha> test: move test_get_splice_bed to unit/splice/
<previous Phase 5 variants head>
```

---

## Out of scope (deferred to subsequent component plans)

- **tracks (broader)** — `test_random_nonoverlapping.py` (1 port), `test_write_tracks.py:test_write_duplicate_track_names_rejected` (1 port), `test_table.py` (12 ports).
- **ref/fasta** — `test_fasta.py` (3 ports), `test_ref_ds.py` (2 remaining ports). Also worth considering a shared `reference` fixture in conftest to deduplicate the local fixtures that appear in `test_ref_ds_splicing.py` (both halves) and `test_ref_ds.py`.
- **dataset polymorphism** — `test_svar_link.py` Pydantic ports (8), `test_with_insertion_fill_rejects_when_no_tracks_active` (once `make_dataset` exists).
- **utility** — `test_utils.py` (5 ports).
- **haps** — Builder-only; no port tests remain. Skipped per user direction; revisit when later component plans actually need synthetic `_Variants` / `Haps`.
