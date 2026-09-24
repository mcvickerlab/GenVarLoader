# Test Suite Overhaul — Phase 5 Variants Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the variants component of Phase 5 — extract 5 `_Variants`-class unit tests from `test_issue_191_var_fields.py` into `tests/unit/variants/`, and move the whole-file `test_choose_exonic_variants.py` (2 tests) into `tests/unit/dataset/genotypes/`.

**Architecture:** Both target files contain pure component-level tests of the variants subsystem (`_Variants.from_table`, `_Variants.available_info_fields`, `_Variants.load_info`, and the `choose_exonic_variants` numba kernel) — no `Dataset.open`, no write pipeline. The 5 Port tests in `test_issue_191_var_fields.py` use only the existing `filtered_svar` path fixture; the 2 Port tests in `test_choose_exonic_variants.py` use pure numpy inputs. No new builder needed — `make_variants_table` is deferred to the haps component plan, where unit tests will actually consume it.

**Tech Stack:** pytest, numpy, polars. No production code changes.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`
- Audit: `docs/superpowers/specs/2026-05-24-test-audit.md` (per-file rows for `test_issue_191_var_fields.py` and `genotypes/test_choose_exonic_variants.py`)

---

## Pre-flight baseline (after Phase 5 reconstruct)

- Non-slow tier: **351 passed, 3 skipped, 3 deselected, 2 xfailed**
- Coverage: **63%**
- Unit tier: **99 passed, 1 xfailed**

Counts unchanged after this plan (relocations only).

---

## Test classification (from audit)

### `tests/integration/dataset/test_issue_191_var_fields.py` — 15 tests total

**Port (5) → `tests/unit/variants/test_variants_info_fields.py`:**

| Test | Why portable |
|---|---|
| `test_available_info_fields_lists_numeric_columns_without_loading` | Calls `_Variants.available_info_fields(path)` on the `filtered_svar` path; pure schema peek |
| `test_from_table_info_fields_filter` | Calls `_Variants.from_table(path, info_fields=set)`; checks the info dict |
| `test_from_table_info_fields_none_loads_all` | Same call with `info_fields=None`; back-compat path |
| `test_load_info_extends_info_dict` | Constructs `_Variants` with empty info, then `.load_info([field])` |
| `test_load_info_idempotent_for_already_loaded_fields` | `.load_info([already_loaded])` must not reload arrays |

All 5 take only the `filtered_svar` path fixture (from `tests/conftest.py`) and import `_Variants` from `genvarloader._dataset._haps`. No `Dataset.open`, no write, no `gvl.write`.

**Keep-as-integration (10) → stay in `tests/integration/dataset/test_issue_191_var_fields.py`:**

- `test_dosage_absent_when_not_requested`
- `test_dosage_present_when_requested`
- `test_available_var_fields_includes_dosage_when_present`
- `test_available_var_fields_excludes_dosage_when_absent`
- `test_haps_from_path_filters_info_loading`
- `test_haps_available_var_fields_from_schema`
- `test_dataset_open_accepts_var_fields`
- `test_dataset_open_default_var_fields_is_minimum_useful_set`
- `test_with_settings_lazily_loads_new_info_field`
- `test_with_settings_lazily_loads_dosages`

All 10 use `gvl.Dataset.open(...)` and/or the `svar_with_dosages_ds` fixture (writes a GVL dataset to `tmp_path`). They stay because they exercise the full open/write pipeline.

The module-scoped `svar_with_dosages_ds` fixture (lines 15-36) stays — only the 10 Keep tests use it.

### `tests/integration/dataset/genotypes/test_choose_exonic_variants.py` — 2 tests total

**Port (2) → `tests/unit/dataset/genotypes/test_choose_exonic_variants.py`:**

Whole-file move. Both tests use pure numpy inputs (`np.asarray(...)` for starts/ends/geno_offsets/v_starts/ilens) and call `choose_exonic_variants` directly. No fixtures, no I/O.

---

## Task 1: Split `test_issue_191_var_fields.py` (atomic create + trim)

Same atomic-commit pattern used in the reconstruct plan: create the new unit file AND trim the integration file in ONE commit so total test count doesn't drift between commits.

**Files:**
- Create: `tests/unit/variants/test_variants_info_fields.py` (5 extracted tests)
- Modify: `tests/integration/dataset/test_issue_191_var_fields.py` (keep 10 Dataset-dependent tests + module-scoped fixture + necessary imports)

### Step 1: Create `tests/unit/variants/test_variants_info_fields.py`

Write the new unit file with this exact content. Bodies are verbatim from the existing integration source (lines 87-144):

- [ ] **Write the file**

```python
"""Unit tests for ``_Variants`` info-field plumbing.

Originally lived in tests/integration/dataset/test_issue_191_var_fields.py;
extracted to the unit tier because every test here calls ``_Variants``
class methods (``from_table``, ``available_info_fields``, ``load_info``)
against the canonical ``filtered_svar`` path fixture — no Dataset, no
write pipeline.

Dataset-level tests (``Dataset.open(var_fields=...)``, ``with_settings``
lazy reload, dosage gating end-to-end) remain in the original file
because they require the full open/write path.
"""

import pytest

from genvarloader._dataset._haps import _Variants


def test_available_info_fields_lists_numeric_columns_without_loading(filtered_svar):
    """Schema peek: returns numeric columns from the variants table without
    materializing data."""
    fields = _Variants.available_info_fields(filtered_svar / "index.arrow")
    # Canonical SVAR has at least AF in its index.
    # POS and ILEN must NOT appear — they're treated as positional, not info.
    assert "POS" not in fields
    assert "ILEN" not in fields
    # All returned names are strings
    assert all(isinstance(f, str) for f in fields)
    # Non-empty for a real SVAR
    assert len(fields) >= 0  # may be 0 if no extra numeric columns; that's fine


def test_from_table_info_fields_filter(filtered_svar):
    """When info_fields is a set, only those numeric columns are loaded."""
    available = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    if not available:
        pytest.skip("No numeric info columns in canonical SVAR; cannot exercise filter")

    pick = {next(iter(available))}
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=pick)
    assert set(v.info.keys()) == pick


def test_from_table_info_fields_none_loads_all(filtered_svar):
    """Back-compat: info_fields=None loads every numeric column (current behavior)."""
    available = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=None)
    assert set(v.info.keys()) == available


def test_load_info_extends_info_dict(filtered_svar):
    """load_info reads only the missing fields from disk and merges them."""
    available = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    if not available:
        pytest.skip(
            "No numeric info columns in canonical SVAR; cannot exercise load_info"
        )

    pick = next(iter(available))
    # Start with empty info
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=set())
    assert pick not in v.info

    v.load_info([pick])
    assert pick in v.info


def test_load_info_idempotent_for_already_loaded_fields(filtered_svar):
    v = _Variants.from_table(filtered_svar / "index.arrow", info_fields=None)
    already = list(v.info.keys())
    if not already:
        pytest.skip("No numeric info columns in canonical SVAR")
    # Snapshot one array; load_info shouldn't reload it.
    arr0 = v.info[already[0]]
    v.load_info(already)
    assert v.info[already[0]] is arr0
```

`filtered_svar` is the existing session-scoped path fixture in `tests/conftest.py` — no fixture work needed.

### Step 2: Run the new unit file in isolation

```
pixi run -e dev pytest tests/unit/variants/test_variants_info_fields.py -q 2>&1 | tail -3
```

Expected: 5 passed (assuming the canonical SVAR has at least one numeric info column — if it has zero, some tests will skip; that's acceptable).

If failures appear: most likely cause is a `from genvarloader._dataset._haps import _Variants` issue (the class moved or got renamed). Check the source and adjust.

### Step 3: Trim the integration file

Replace the contents of `tests/integration/dataset/test_issue_191_var_fields.py` with this exact file (keeping the 10 Dataset-dependent tests + the `svar_with_dosages_ds` fixture + necessary imports). The unit-tier tests at lines 87-144 of the original are removed.

- [ ] **Rewrite the file**

```python
"""Tests for issue #191 — var_fields loading and dosage gating.

See docs/superpowers/specs/2026-05-24-issue-191-var-fields-loading-design.md.

The unit-tier ``_Variants``-class tests live in
``tests/unit/variants/test_variants_info_fields.py``. The tests here all
exercise the full ``Dataset.open`` / write pipeline.
"""

import shutil

import genvarloader as gvl
import numpy as np
import pytest
from genoray._types import DOSAGE_TYPE

from genvarloader._dataset._haps import _Variants


@pytest.fixture
def svar_with_dosages_ds(tmp_path, filtered_svar, source_bed):
    """Copy the canonical SVAR to tmp_path, add a synthetic dosages.npy, then
    write a fresh GVL dataset pointing at it. Returns the GVL path."""
    svar_copy = tmp_path / "filtered.svar"
    shutil.copytree(filtered_svar, svar_copy)

    v_idxs_bytes = (svar_copy / "variant_idxs.npy").stat().st_size
    n_records = v_idxs_bytes // np.dtype(np.uint32).itemsize
    mm = np.memmap(
        svar_copy / "dosages.npy",
        dtype=DOSAGE_TYPE,
        mode="w+",
        shape=(n_records,),
    )
    mm[:] = np.arange(n_records, dtype=DOSAGE_TYPE)
    mm.flush()
    del mm

    gvl_path = tmp_path / "ds.gvl"
    gvl.write(path=gvl_path, bed=source_bed, variants=svar_copy, overwrite=True)
    return gvl_path


def test_dosage_absent_when_not_requested(svar_with_dosages_ds, ref_fasta):
    """Regression: bug from issue #191.

    Dataset has dosages on disk; user did NOT request dosage in var_fields.
    The output RaggedVariants must not contain a `dosage` field.
    """
    ds = (
        gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" not in batch.fields, (
        f"dosage leaked into output despite not being in var_fields; got fields={batch.fields}"
    )


def test_dosage_present_when_requested(svar_with_dosages_ds, ref_fasta):
    """Sanity: opting in adds the field."""
    ds = (
        gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start", "dosage"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" in batch.fields


def test_available_var_fields_includes_dosage_when_present(
    svar_with_dosages_ds, ref_fasta
):
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert "dosage" in ds.available_var_fields


def test_available_var_fields_excludes_dosage_when_absent(phased_svar_gvl, ref_fasta):
    # The canonical SVAR (no dosages.npy) — opening the existing fixture
    # should not list dosage as available.
    ds = gvl.Dataset.open(
        phased_svar_gvl,
        ref_fasta,
        rc_neg=False,
    )
    assert "dosage" not in ds.available_var_fields


def test_haps_from_path_filters_info_loading(svar_with_dosages_ds, ref_fasta):
    """from_path(var_fields=[default]) does not load extra numeric info columns
    or open the dosages memmap."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    haps = ds._seqs  # type: ignore[attr-defined]
    # Default var_fields: only alt/ilen/start. info dict must not contain extras.
    assert haps.var_fields == ["alt", "ilen", "start"]
    assert set(haps.variants.info.keys()) == set()
    # Dosages file exists on disk but should not be loaded.
    assert haps.dosages is None


def test_haps_available_var_fields_from_schema(svar_with_dosages_ds, ref_fasta):
    """available_var_fields reflects the file's schema + dosage presence,
    not what was actually loaded."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert "dosage" in ds.available_var_fields
    # ref is also discoverable because the SVAR has a REF column
    assert "ref" in ds.available_var_fields


def test_dataset_open_accepts_var_fields(svar_with_dosages_ds, ref_fasta):
    """Dataset.open(var_fields=...) routes through to Haps.from_path so the
    requested fields are loaded eagerly at open time."""
    ds = gvl.Dataset.open(
        svar_with_dosages_ds,
        ref_fasta,
        rc_neg=False,
        var_fields=["alt", "ilen", "start", "dosage"],
    )
    haps = ds._seqs  # type: ignore[attr-defined]
    assert haps.var_fields == ["alt", "ilen", "start", "dosage"]
    assert haps.dosages is not None


def test_dataset_open_default_var_fields_is_minimum_useful_set(
    svar_with_dosages_ds, ref_fasta
):
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert ds.active_var_fields == ["alt", "ilen", "start"]


def test_with_settings_lazily_loads_new_info_field(
    svar_with_dosages_ds, filtered_svar, ref_fasta
):
    """Opening with default var_fields does not load AF (or other info columns).
    with_settings(var_fields=[..., 'AF']) should lazily extend the info dict."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    available_info = set(_Variants.available_info_fields(filtered_svar / "index.arrow"))
    if not available_info:
        pytest.skip("No numeric info columns; cannot test lazy info expansion")

    new_field = next(iter(available_info))
    haps_before = ds._seqs  # type: ignore[attr-defined]
    assert new_field not in haps_before.variants.info

    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", new_field])
    haps_after = ds2._seqs  # type: ignore[attr-defined]
    assert new_field in haps_after.variants.info


def test_with_settings_lazily_loads_dosages(svar_with_dosages_ds, ref_fasta):
    """Opening with default var_fields does not memmap dosages.
    with_settings(var_fields=[..., 'dosage']) should memmap them."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, ref_fasta, rc_neg=False)
    assert ds._seqs.dosages is None  # type: ignore[attr-defined]

    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", "dosage"])
    assert ds2._seqs.dosages is not None  # type: ignore[attr-defined]
```

**CRITICAL** — Before pasting, read lines 208-215 of the existing integration source to confirm the body of `test_with_settings_lazily_loads_dosages` is correct. The last statement above is reconstructed from context; if the source differs, use the source verbatim.

### Step 4: Run the trimmed integration file

```
pixi run -e dev pytest tests/integration/dataset/test_issue_191_var_fields.py -q 2>&1 | tail -3
```

Expected: **10 passed** (same as before minus the 5 extracted tests).

### Step 5: Run the full non-slow suite

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed` (totals unchanged — 5 tests moved tier).

### Step 6: Commit atomically

```bash
git add tests/integration/dataset/test_issue_191_var_fields.py tests/unit/variants/test_variants_info_fields.py
git commit -m "test: extract _Variants info-field unit tests to unit/variants/"
```

Verify `git status` clean.

---

## Task 2: Move `test_choose_exonic_variants.py` to unit tier

Whole-file move. The audit row classifies both tests as Port; they use pure numpy inputs and call `choose_exonic_variants` directly.

- [ ] **Step 1: Move**

```bash
git mv tests/integration/dataset/genotypes/test_choose_exonic_variants.py tests/unit/dataset/genotypes/test_choose_exonic_variants.py
```

- [ ] **Step 2: Run the moved file**

```
pixi run -e dev pytest tests/unit/dataset/genotypes/test_choose_exonic_variants.py -q 2>&1 | tail -2
```

Expected: 2 passed.

- [ ] **Step 3: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add -A tests/
git commit -m "test: move genotypes/test_choose_exonic_variants to unit/dataset/genotypes/"
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

Expected: roughly **106 passed, 1 xfailed** (was 99/1; +5 from variants extraction, +2 from choose_exonic_variants move).

- [ ] **Step 3: Coverage parity**

```
pixi run -e dev pytest tests -m "not slow" --cov --cov-report=term -q 2>&1 | grep "^TOTAL"
```

Expected: `TOTAL ... 63%` (±1pp).

- [ ] **Step 4: File structure**

```
/bin/ls tests/unit/variants/ tests/unit/dataset/genotypes/
```

Expected:
- `tests/unit/variants/`: `test_variant_utils.py` (existed), `test_variants_info_fields.py` (new), plus possibly a leftover `.gitkeep`.
- `tests/unit/dataset/genotypes/`: `test_choose_exonic_variants.py` (new), `test_reconstruct.py` (existed).

```
/bin/ls tests/integration/dataset/genotypes/
```

Expected: ALMOST empty — only `__pycache__` may remain (pytest leftover). All test files for this directory have now moved to `tests/unit/dataset/genotypes/`. If the directory contains only `__pycache__`, leave it; git doesn't track empty directories. No action needed.

- [ ] **Step 5: Commit graph**

```
git log --oneline -4
```

Expected:
```
<sha> test: move genotypes/test_choose_exonic_variants to unit/dataset/genotypes/
<sha> test: extract _Variants info-field unit tests to unit/variants/
<previous Phase 5 reconstruct head>
```

---

## Out of scope (deferred to subsequent component plans)

- **haps component** — `Haps.from_path` + `var_fields` plumbing tests. Will introduce `make_variants_table` / `make_variants` builders to construct synthetic `_Variants` without a real SVAR. Likely small file count but real builder work.
- **tracks (broader)** — `test_random_nonoverlapping.py` (1 port), `test_write_tracks.py:test_write_duplicate_track_names_rejected` (1 port), `test_table.py` (12 ports).
- **splice (broader)** — `test_get_splice_bed.py` (11 ports), `test_ref_ds_splicing.py` (5 ports).
- **ref/fasta** — `test_fasta.py` (3 ports), `test_ref_ds.py` (2 remaining ports).
- **dataset polymorphism** — `test_svar_link.py` Pydantic ports (8), `test_with_insertion_fill_rejects_when_no_tracks_active` (once `make_dataset` exists).
- **utility** — `test_utils.py` (5 ports).
