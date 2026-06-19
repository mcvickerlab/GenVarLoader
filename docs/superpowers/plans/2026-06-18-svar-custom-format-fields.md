# Surface genoray Custom FORMAT Fields Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let `Dataset.open(..., var_fields=["<name>"])` surface genoray's arbitrary per-call (`Number=G`) FORMAT fields — discovered from `<svar>/metadata.json["fields"]`, memmapped from `<svar>/<name>.npy` — in the variants, variant-windows, and flat output modes, generalizing the existing hardcoded `dosage` handling.

**Architecture:** Custom FORMAT fields are per-call (one value per genotype call, sharing the genotype `offsets`) — byte-identical on disk to `dosages.npy`. They therefore follow the existing `dosage` code path: discovered in `Haps.__post_init__`, memmapped in `Haps.from_path` into a new `var_field_data` dict, gathered per-row in `get_variants_flat` via `_gather_rows` (with the same AF `_compact_keep`), and added to the `fields` dict. Because `_FlatVariants.to_ragged()`, the variant-windows `wfields` filter, and `DummyVariant.scalar_for` all iterate `fields` generically, all three output modes and dummy-fill work with no further changes.

**Tech Stack:** Python, numpy (memmap), `seqpro.rag.Ragged`, awkward (`RaggedVariants`), numba (gather kernels), pytest, pixi.

## Global Constraints

- Run all dev tasks via pixi: `pixi run -e dev <cmd>`. Platform is linux-64.
- Field dtype MUST come from `metadata.json["fields"]` — never hardcoded.
- `dosage` stays a named special-case (own block, `DOSAGE_TYPE`) — do NOT unify it into the generic path; preserve its byte-identical guarantee.
- Per-call FORMAT fields take precedence over a same-named INFO column (excluded from `info_fields`); `available_var_fields` must list each name once.
- E501 (line length) is ignored by ruff; otherwise code must pass `ruff check` AND `ruff format`.
- Before pushing, run the full tree: `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`).
- Per `CLAUDE.md`: any public-API/`var_fields`-behavior change must update `skills/genvarloader/SKILL.md`.

---

## File Structure

- **Modify** `python/genvarloader/_dataset/_haps.py`
  - Add module-level helper `_svar_format_fields(svar_dir) -> dict[str, np.dtype]`.
  - `Haps.__post_init__` (~line 293): extend `available_var_fields` with custom field names (de-duped).
  - `Haps` dataclass (~line 251, next to `dosages`): add `var_field_data: dict[str, Ragged] = field(default_factory=dict)`.
  - `Haps.from_path` (~lines 345, 397, 426): exclude custom fields from `info_fields`; memmap requested custom fields into `var_field_data`; pass to constructor.
- **Modify** `python/genvarloader/_dataset/_flat_variants.py`
  - `get_variants_flat` (~lines 699–762): gather each requested custom field like dosage (incl. AF `_compact_keep`); add to `fields`; skip them in the "other info fields" loop.
- **Create** `tests/integration/dataset/test_issue_231_custom_format_fields.py`
  - Fixture that copies `filtered_svar`, writes a custom field `.npy` + patches svar `metadata.json["fields"]`, then `gvl.write`s a dataset linking to it. Tests discovery, load, and all three output modes.
- **Modify** `skills/genvarloader/SKILL.md` — document custom FORMAT fields in the `var_fields` discussion.

---

## Task 1: Discover custom FORMAT fields in `available_var_fields`

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (add helper; `__post_init__` ~line 293)
- Create: `tests/integration/dataset/test_issue_231_custom_format_fields.py`

**Interfaces:**
- Produces: `_svar_format_fields(svar_dir: Path) -> dict[str, np.dtype]` (module-level in `_haps.py`) — reads `<svar_dir>/metadata.json`, returns `{name: np.dtype(dtype_str)}` from its `"fields"` map, or `{}` if the file is absent.
- Produces: a shared pytest fixture `custom_field_ds` (in the new test file) returning `(gvl_path, field_name, n_records)` for later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/dataset/test_issue_231_custom_format_fields.py`:

```python
"""Tests for issue #231 — surfacing genoray custom per-call FORMAT fields.

A .svar can register arbitrary Number=G FORMAT fields in
<svar>/metadata.json["fields"] (name -> numpy dtype) stored as <svar>/<name>.npy,
sharing the genotype offsets (byte-identical layout to dosages.npy). These tests
exercise the full Dataset.open path with a hand-crafted custom field.
"""

import json
import shutil

import awkward as ak
import genvarloader as gvl
import numpy as np
import pytest
from genoray._types import DOSAGE_TYPE

FIELD_NAME = "mutcat"
FIELD_DTYPE = "int16"


@pytest.fixture
def custom_field_ds(tmp_path, filtered_svar, source_bed):
    """Copy the canonical SVAR to tmp_path, register a custom int16 FORMAT field
    `mutcat` (values = arange over the genotype calls) AND a parallel dosages.npy
    with identical arange values, then write a fresh GVL dataset linking to it.

    Returns (gvl_path, field_name, n_records).
    """
    svar_copy = tmp_path / "filtered.svar"
    shutil.copytree(filtered_svar, svar_copy)

    # One value per genotype call: length == n variant_idxs (V_IDX_TYPE = uint32).
    v_idxs_bytes = (svar_copy / "variant_idxs.npy").stat().st_size
    n_records = v_idxs_bytes // np.dtype(np.uint32).itemsize

    # Custom int16 field, values 0..n-1.
    mm = np.memmap(
        svar_copy / f"{FIELD_NAME}.npy", dtype=FIELD_DTYPE, mode="w+", shape=(n_records,)
    )
    mm[:] = np.arange(n_records, dtype=FIELD_DTYPE)
    mm.flush()
    del mm

    # Parallel dosages with the same arange values, to prove gather-equivalence.
    dmm = np.memmap(
        svar_copy / "dosages.npy", dtype=DOSAGE_TYPE, mode="w+", shape=(n_records,)
    )
    dmm[:] = np.arange(n_records, dtype=DOSAGE_TYPE)
    dmm.flush()
    del dmm

    # Register the custom field in the SVAR metadata.
    meta_path = svar_copy / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["fields"] = {FIELD_NAME: FIELD_DTYPE}
    meta_path.write_text(json.dumps(meta))

    gvl_path = tmp_path / "ds.gvl"
    gvl.write(path=gvl_path, bed=source_bed, variants=svar_copy, overwrite=True)
    return gvl_path, FIELD_NAME, n_records


def test_available_var_fields_lists_custom_field(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    ds = gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
    assert field_name in ds.available_var_fields
    # Listed exactly once.
    assert ds.available_var_fields.count(field_name) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py::test_available_var_fields_lists_custom_field -v`
Expected: FAIL — `mutcat` is not in `available_var_fields` (discovery not implemented).

- [ ] **Step 3: Add the `_svar_format_fields` helper**

In `python/genvarloader/_dataset/_haps.py`, add a module-level function (place it just above the `Haps` class definition, ~line 240). `json`, `np`, and `Path` are already imported:

```python
def _svar_format_fields(svar_dir: Path) -> dict[str, np.dtype]:
    """genoray custom per-call FORMAT fields: name -> dtype, from <svar>/metadata.json.

    Returns {} when the metadata file is absent (non-SVAR / synthetic datasets).
    """
    meta = svar_dir / "metadata.json"
    if not meta.is_file():
        return {}
    fields = json.loads(meta.read_text()).get("fields", {})
    return {name: np.dtype(dt) for name, dt in fields.items()}
```

- [ ] **Step 4: Extend `available_var_fields` in `Haps.__post_init__`**

In `python/genvarloader/_dataset/_haps.py`, replace the `self.available_var_fields = (...)` assignment at ~line 293 with a de-duped version that appends custom field names. `self.variants.path` is `<svar>/index.arrow`, so `.parent` is the svar dir:

```python
        custom_fmt = _svar_format_fields(self.variants.path.parent)
        base = (
            ["alt", "ilen", "start"]
            + schema_info_fields
            + (["ref"] if self.variants.ref is not None else [])
            + (["dosage"] if has_dosage_file else [])
        )
        # Per-call FORMAT fields win over a same-named INFO column; list each once.
        self.available_var_fields = base + [f for f in custom_fmt if f not in base]
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py::test_available_var_fields_lists_custom_field -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_haps.py tests/integration/dataset/test_issue_231_custom_format_fields.py
git commit -m "feat: discover genoray custom FORMAT fields in available_var_fields (#231)"
```

---

## Task 2: Load custom fields into `Haps.var_field_data`

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`Haps` dataclass ~line 251; `from_path` ~lines 345, 397–401, 426–437)
- Test: `tests/integration/dataset/test_issue_231_custom_format_fields.py`

**Interfaces:**
- Consumes: `_svar_format_fields` (Task 1).
- Produces: `Haps.var_field_data: dict[str, Ragged]` — populated only with the requested custom fields (names in `var_fields` that are registered FORMAT fields); empty otherwise. Each value is `Ragged.from_offsets(memmap, rag_shape, offsets.reshape(2, -1))` on the genotype offsets.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/dataset/test_issue_231_custom_format_fields.py`:

```python
def test_var_field_data_loaded_only_when_requested(custom_field_ds, ref_fasta):
    gvl_path, field_name, n_records = custom_field_ds

    # Not requested (default var_fields) -> not memmapped, not in info dict.
    ds = gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
    haps = ds._seqs  # type: ignore[attr-defined]
    assert field_name not in haps.var_field_data
    assert field_name not in haps.variants.info  # not loaded as an INFO column

    # Requested -> memmapped into var_field_data with the registered dtype.
    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", field_name])
    haps2 = ds2._seqs  # type: ignore[attr-defined]
    assert field_name in haps2.var_field_data
    rag = haps2.var_field_data[field_name]
    assert np.asarray(rag.data).dtype == np.dtype(FIELD_DTYPE)
    assert np.asarray(rag.data).shape[0] == n_records
    # Still must not have been loaded as an INFO column.
    assert field_name not in haps2.variants.info
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py::test_var_field_data_loaded_only_when_requested -v`
Expected: FAIL — `AttributeError: 'Haps' object has no attribute 'var_field_data'`.

- [ ] **Step 3: Add the `var_field_data` field to `Haps`**

In `python/genvarloader/_dataset/_haps.py`, in the `Haps` dataclass, add a field next to `dosages` (~line 251). Place it among the `field(default_factory=...)` entries near `var_fields` (~line 260) so it has a default and does not break positional construction:

```python
    var_field_data: dict[str, Ragged] = field(default_factory=dict)
    """Custom per-call (Number=G) FORMAT fields requested via ``var_fields``,
    memmapped on the genotype offsets. Parallel to ``dosages``. See issue #231."""
```

- [ ] **Step 4: Exclude custom fields from INFO loading and memmap them in `from_path`**

In `python/genvarloader/_dataset/_haps.py`, `Haps.from_path`:

(a) Keep the existing `info_fields = {f for f in var_fields if f not in builtin}` at ~line 346 unchanged. Just below the `svar_meta_path = ...` / `dosages = None` lines (~line 349), add the new accumulator (and keep the existing `dosages = None`):

```python
        var_field_data: dict[str, Ragged] = {}
```

(b) Inside the `if svar_meta_path.exists():` branch, after `svar_path` is resolved and `offsets`/`rag_shape` are built and right after the existing `if "dosage" in var_fields and dosage_path.exists():` block (~line 401), discover the custom fields, exclude them from the INFO set (they are not columns in `index.arrow`), and memmap the requested ones:

```python
            custom_fmt = _svar_format_fields(svar_path)
            info_fields = info_fields - set(custom_fmt)
            for name in var_fields:
                if name in custom_fmt:
                    field_mm = np.memmap(
                        svar_path / f"{name}.npy", dtype=custom_fmt[name], mode="r"
                    )
                    var_field_data[name] = Ragged.from_offsets(
                        field_mm, rag_shape, offsets.reshape(2, -1)
                    )
```

This re-binds `info_fields` (a `set`) before the `_Variants.from_table(svar_path / "index.arrow", info_fields=info_fields)` call at ~line 404. The legacy (non-SVAR) `else` branch is untouched — no custom fields are possible there.

(c) Pass the new dict to the constructor at ~line 426:

```python
        return cls(
            path=path,
            reference=reference,
            variants=variants,
            genotypes=genotypes,
            dosages=dosages,
            var_field_data=var_field_data,
            kind=RaggedVariants,
            filter=filter,
            min_af=min_af,
            max_af=max_af,
            var_fields=var_fields,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py::test_var_field_data_loaded_only_when_requested -v`
Expected: PASS.

- [ ] **Step 6: Run the existing var_fields/dosage regression suite (no regressions)**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_191_var_fields.py -q`
Expected: PASS (the `info_fields` / `from_path` restructure must not break dosage gating).

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_haps.py tests/integration/dataset/test_issue_231_custom_format_fields.py
git commit -m "feat: memmap requested custom FORMAT fields into Haps.var_field_data (#231)"
```

---

## Task 3: Surface custom fields in RaggedVariants (variants mode)

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (`get_variants_flat` ~lines 699–762)
- Test: `tests/integration/dataset/test_issue_231_custom_format_fields.py`

**Interfaces:**
- Consumes: `Haps.var_field_data` (Task 2), `_gather_rows`, `_compact_keep` (existing in `_flat_variants.py`).
- Produces: each requested custom field added to the `fields` dict in `get_variants_flat` as `_Flat.from_offsets(gathered, shape, row_offsets)`, so it flows into `RaggedVariants` via `to_ragged()`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/integration/dataset/test_issue_231_custom_format_fields.py`:

```python
def _open_variants(gvl_path, ref_fasta, field_name, **settings):
    return (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ilen", "start", "dosage", field_name],
                       **settings)
    )


def test_custom_field_present_in_ragged_variants(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    ds = _open_variants(gvl_path, ref_fasta, field_name)
    batch = ds[0, ds.samples[0]]
    assert field_name in batch.fields
    # dtype is the registered int16, not coerced.
    flat = ak.to_numpy(ak.flatten(batch[field_name], axis=None))
    assert flat.dtype == np.dtype(FIELD_DTYPE)
    # Per-cell variant counts equal `start`'s (call-aligned with the genotypes).
    assert ak.num(batch[field_name], -1).to_list() == ak.num(batch["start"], -1).to_list()


def test_custom_field_matches_dosage_gather(custom_field_ds, ref_fasta):
    """Custom field and dosages were written with identical arange values and
    share the genotype offsets, so the gathered output must be elementwise equal."""
    gvl_path, field_name, _ = custom_field_ds
    ds = _open_variants(gvl_path, ref_fasta, field_name)
    batch = ds[0, ds.samples[0]]
    custom = ak.to_numpy(ak.flatten(batch[field_name], axis=None)).astype(np.float64)
    dosage = ak.to_numpy(ak.flatten(batch["dosage"], axis=None)).astype(np.float64)
    np.testing.assert_array_equal(custom, dosage)


def test_custom_field_af_compaction_matches_dosage(custom_field_ds, ref_fasta):
    """Under AF filtering the custom field is compacted with the SAME keep mask
    as dosage, so they stay elementwise equal."""
    gvl_path, field_name, _ = custom_field_ds
    ds = _open_variants(gvl_path, ref_fasta, field_name, max_af=0.5)
    batch = ds[0, ds.samples[0]]
    custom = ak.to_numpy(ak.flatten(batch[field_name], axis=None)).astype(np.float64)
    dosage = ak.to_numpy(ak.flatten(batch["dosage"], axis=None)).astype(np.float64)
    np.testing.assert_array_equal(custom, dosage)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py -k custom_field_present_in_ragged -v`
Expected: FAIL — `mutcat` not in `batch.fields` (gather not implemented).

- [ ] **Step 3: Gather custom fields in `get_variants_flat`**

In `python/genvarloader/_dataset/_flat_variants.py`, `get_variants_flat`:

(a) After the dosage block that ends at the `fields["dosage"] = _Flat.from_offsets(dosage_data, shape, row_offsets)` line (~line 755), add a generic per-call gather for the requested custom fields. Gather against the UNFILTERED offsets, then apply the same AF `_compact_keep`:

```python
    # Custom per-call FORMAT fields (issue #231): same gather/compaction as dosage.
    for name, rag in haps.var_field_data.items():
        if name not in haps.var_fields:
            continue
        cf_off = np.asarray(rag.offsets, np.int64)
        cf_all = np.asarray(rag.data)
        cf_data, _ = _gather_rows(geno_offset_idx, cf_off, cf_all)
        if keep is not None:
            cf_data, _ = _compact_keep(cf_data, unfiltered_row_offsets, keep)
        fields[name] = _Flat.from_offsets(cf_data, shape, row_offsets)
```

(b) In the "other info fields" loop (~line 758), skip names already handled as custom fields so they are not looked up in `haps.variants.info` (they are not columns there):

```python
    # other info fields
    for k in haps.var_fields:
        if k in {"alt", "start", "ref", "ilen", "dosage"} or k in haps.var_field_data:
            continue
        info_data = np.asarray(haps.variants.info[k])[v_idxs]
        fields[k] = _Flat.from_offsets(info_data, shape, row_offsets)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py -k "custom_field_present_in_ragged or custom_field_matches_dosage or af_compaction" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_flat_variants.py tests/integration/dataset/test_issue_231_custom_format_fields.py
git commit -m "feat: gather custom FORMAT fields into RaggedVariants like dosage (#231)"
```

---

## Task 4: Custom fields in flat-mode and variant-windows output

**Files:**
- Test: `tests/integration/dataset/test_issue_231_custom_format_fields.py`

**Interfaces:**
- Consumes: the `fields`-dict surfacing from Task 3. No production code change expected — `_FlatVariants.to_ragged()`, the `flat` output format, and the variant-windows `wfields` filter all iterate `fields` generically. This task is a guard that proves it.

- [ ] **Step 1: Write the failing/guard tests**

Append to `tests/integration/dataset/test_issue_231_custom_format_fields.py`:

```python
def test_custom_field_present_in_flat_mode(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    ds = (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ilen", "start", field_name])
        .with_output_format("flat")
    )
    flat = ds[0, ds.samples[0]]  # _FlatVariants
    assert field_name in flat.fields
    # Re-wrapping to ragged keeps the field.
    assert field_name in flat.to_ragged().fields


def test_custom_field_present_in_variant_windows(custom_field_ds, ref_fasta):
    gvl_path, field_name, _ = custom_field_ds
    opt = gvl.VarWindowOpt(flank_length=3, token_alphabet=b"ACGT", unknown_token=4)
    ds = (
        gvl.Dataset.open(gvl_path, ref_fasta, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variant-windows", opt)
        .with_settings(var_fields=["alt", "ilen", "start", field_name])
    )
    batch = ds[0, ds.samples[0]]
    assert field_name in batch.fields
```

- [ ] **Step 2: Run the tests**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py -k "flat_mode or variant_windows" -v`
Expected: PASS if Task 3's generic surfacing is correct. If either FAILS, the failure pinpoints a mode that filters `fields` non-generically — fix that mode's field handling in `_flat_variants.py` (flat) or `_flat_flanks.py` / the `wfields` construction (windows) so custom fields ride along, then re-run.

- [ ] **Step 3: Run the whole new test file**

Run: `pixi run -e dev pytest tests/integration/dataset/test_issue_231_custom_format_fields.py -v`
Expected: PASS (all tests).

- [ ] **Step 4: Commit**

```bash
git add tests/integration/dataset/test_issue_231_custom_format_fields.py
git commit -m "test: custom FORMAT fields surface in flat + variant-windows modes (#231)"
```

---

## Task 5: Documentation — SKILL.md + changelog note

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

**Interfaces:**
- Consumes: the shipped behavior from Tasks 1–4. No code.

- [ ] **Step 1: Locate the `var_fields` discussion in the skill**

Run: `pixi run -e dev grep -n "var_fields\|available_var_fields\|dosage" skills/genvarloader/SKILL.md`
Expected: one or more sections describing `var_fields` / `available_var_fields`.

- [ ] **Step 2: Add a custom-FORMAT-fields note**

In the `var_fields` section of `skills/genvarloader/SKILL.md`, add a sentence (match the file's existing prose style and heading depth) covering:

> Beyond the built-ins (`alt`, `start`, `ref`, `ilen`, `dosage`) and per-variant INFO columns, a genoray `.svar` may register arbitrary per-call (`Number=G`) FORMAT fields in `<svar>/metadata.json["fields"]`. These appear in `Dataset.available_var_fields` and can be requested via `Dataset.open(..., var_fields=[...])` or `with_settings(var_fields=[...])`; each surfaces in the `variants`, `variant-windows`, and `flat` outputs as a per-call ragged field aligned with the genotypes. A FORMAT field shadows a same-named INFO column.

- [ ] **Step 3: Verify the skill still reads cleanly**

Run: `pixi run -e dev grep -n "Number=G\|FORMAT field" skills/genvarloader/SKILL.md`
Expected: the new note is present.

- [ ] **Step 4: Commit**

```bash
git add skills/genvarloader/SKILL.md
git commit -m "docs: document genoray custom FORMAT fields in genvarloader skill (#231)"
```

---

## Final Verification (run before declaring complete)

- [ ] **Lint + format**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/`
Expected: no errors. (If format reports changes, run `pixi run -e dev ruff format python/ tests/` and amend.)

- [ ] **Type check**

Run: `pixi run -e dev typecheck`
Expected: no new errors.

- [ ] **Full test tree** (scoped runs skip `tests/unit/`; custom-field code touches shared paths)

Run: `pixi run -e dev pytest tests -q`
Expected: PASS, including the new `test_issue_231_custom_format_fields.py` and the existing `test_issue_191_var_fields.py`.

- [ ] **Acceptance criteria (issue #231) confirmed:**
  - `available_var_fields` lists the custom field (Task 1).
  - `Dataset.open(..., var_fields=["f"])[0]` returns `RaggedVariants` with `f` as its registered dtype, per-cell length-equal to `start`/the genotypes (Task 3).
  - `f` present in flat-mode and variant-windows output (Task 4).
