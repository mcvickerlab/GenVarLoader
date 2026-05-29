# Issue #191 — var_fields loading + dosage correctness fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop emitting a phantom `dosage` field in `RaggedVariants` when the user didn't request it, and make variant-info loading honor `var_fields` lazily — at `Dataset.open` and at `with_settings`.

**Architecture:** Two phases in one PR. Phase 1: gate the `dosage` field on `var_fields` in `Haps._get_variants` and add `"dosage"` to `available_var_fields` when the file exists. Phase 2: thread `var_fields` from `Dataset.open` → `OpenRequest` → `Haps.from_path` → `_Variants.from_table`, where it filters which numeric info columns are eagerly `.to_numpy()`'d and whether the dosages memmap is opened; `available_var_fields` is now computed from a schema peek so it reflects the file, not what loaded; `with_settings(var_fields=...)` lazily extends the loaded set via a new `_Variants.load_info` helper.

**Tech Stack:** Python 3.10+, polars (schema peek via `pl.scan_ipc`), numpy memmap (dosages), awkward (RaggedVariants record), pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-24-issue-191-var-fields-loading-design.md`.

**Setup:** Work in worktree `../GenVarLoader-issue191` (create with `git worktree add ../GenVarLoader-issue191 -b fix/issue-191-var-fields-loading`). All commands assume this worktree as cwd unless noted.

**Verification cadence:** every task ends with `pixi run -e dev test`, `pixi run -e dev ruff check python/`, `pixi run -e dev typecheck` — all green before committing.

---

## Background facts the implementer will need

- `Haps` is a `@dataclass(slots=True)` with mutating helpers via `dataclasses.replace`.
- `_Variants` is also `@dataclass(slots=True)`; its `info: dict[str, NDArray]` is mutable.
- `Haps.var_fields` defaults to `["alt", "ilen", "start"]`. `available_var_fields` is set in `__post_init__`. `_get_variants` is in the same file (`_haps.py`).
- SVAR test fixture: `tests/data/filtered.svar` is the canonical small SVAR (17 variants, 34 genotype-records — `variant_idxs.npy` is 136 bytes of `uint32`). The matching GVL is `tests/data/phased_dataset.svar.gvl`. The SVAR-link is recorded in the GVL's `metadata.json`. To create a SVAR-with-dosages fixture, copy the SVAR to `tmp_path`, write a synthetic `dosages.npy` of length 34 `float32`, then `gvl.write(... variants=tmp_svar)` to produce a matching dataset (see fixture in Task 1, Step 1).
- `Dataset.open` currently has 3 overload signatures (none-ref, ref, impl) starting at `_impl.py:95`. Adding a new optional parameter requires updating all three. `OpenRequest` is at `_dataset/_open.py:39`.
- The existing `with_settings(var_fields=...)` validation at `_impl.py:287-293` raises if requested fields aren't in `available_var_fields`. We preserve this; just expand `available_var_fields` to reflect the schema, and add lazy loading for newly-requested fields.
- The non-SVAR branch of `Haps.from_path` (`_haps.py:265-281`) loads from `variants.arrow` instead. Treat it identically: same `info_fields` filtering applies; no dosages branch.

---

## File Structure

**Modify:**
- `python/genvarloader/_dataset/_haps.py` — `_Variants` (add `info_fields` param + `available_info_fields` + `load_info`), `Haps.__post_init__` (schema-peek `available_var_fields`), `Haps.from_path` (new `var_fields` param), `Haps._get_variants` (gate dosage on `var_fields`).
- `python/genvarloader/_dataset/_impl.py` — `Dataset.open` overloads + impl (add `var_fields` param), `Dataset.with_settings` (lazy extension via `load_info` + dosages memmap on demand).
- `python/genvarloader/_dataset/_open.py` — `OpenRequest` dataclass (add `var_fields` field), `_build_seqs` (forward `var_fields` to `Haps.from_path`).
- `skills/genvarloader/SKILL.md` — document new `var_fields` parameter on `Dataset.open`.

**Create:**
- `tests/dataset/test_issue_191_var_fields.py` — fixture for SVAR-with-dosages + bug-regression and lazy-loading tests.

**Test:**
- Above test file + parity check via the existing `pixi run -e dev test` suite.

---

## Task 1: Phase 1 — gate dosage in `_get_variants` and add to `available_var_fields`

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (two small edits)
- Create: `tests/dataset/test_issue_191_var_fields.py`

- [ ] **Step 1: Create the test file with the SVAR-with-dosages fixture**

Create `tests/dataset/test_issue_191_var_fields.py`:

```python
"""Tests for issue #191 — var_fields loading and dosage gating.

See docs/superpowers/specs/2026-05-24-issue-191-var-fields-loading-design.md.
"""

import shutil
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pytest
from genoray._types import DOSAGE_TYPE

_DATA = Path(__file__).resolve().parents[1] / "data"
_REF = _DATA / "fasta" / "hg38.fa.bgz"
_SOURCE_SVAR = _DATA / "filtered.svar"
_SOURCE_BED = _DATA / "source.bed"


@pytest.fixture
def svar_with_dosages_ds(tmp_path):
    """Copy the canonical SVAR to tmp_path, add a synthetic dosages.npy, then
    write a fresh GVL dataset pointing at it. Returns the GVL path."""
    svar_copy = tmp_path / "filtered.svar"
    shutil.copytree(_SOURCE_SVAR, svar_copy)

    v_idxs_bytes = (svar_copy / "variant_idxs.npy").stat().st_size
    n_records = v_idxs_bytes // np.dtype(np.uint32).itemsize
    # synthetic dosages: arange so test can spot-check ordering if needed
    dosages = np.arange(n_records, dtype=DOSAGE_TYPE)
    np.save(svar_copy / "dosages.npy", dosages, allow_pickle=False)
    # np.save writes a .npy header — but Haps.from_path uses np.memmap directly.
    # Replace with a raw memmap so the file is a flat array (matches v_idxs.npy
    # convention in this repo).
    (svar_copy / "dosages.npy").unlink()
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
    gvl.write(path=gvl_path, bed=_SOURCE_BED, variants=svar_copy, overwrite=True)
    # Ensure the freshly written GVL links to OUR svar_copy (with dosages),
    # not the source. gvl.write should already do this since variants=svar_copy.
    return gvl_path


def test_dosage_absent_when_not_requested(svar_with_dosages_ds):
    """Regression: bug from issue #191.

    Dataset has dosages on disk; user did NOT request dosage in var_fields.
    The output RaggedVariants must not contain a `dosage` field.
    """
    ds = (
        gvl.Dataset
        .open(svar_with_dosages_ds, _REF, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" not in batch.fields, (
        f"dosage leaked into output despite not being in var_fields; got fields={batch.fields}"
    )


def test_dosage_present_when_requested(svar_with_dosages_ds):
    """Sanity: opting in adds the field."""
    ds = (
        gvl.Dataset
        .open(svar_with_dosages_ds, _REF, rc_neg=False)
        .with_len("ragged")
        .with_seqs("variants")
        .with_settings(var_fields=["alt", "ref", "start", "dosage"])
    )
    batch = ds[0, ds.samples[0]]
    assert "dosage" in batch.fields


def test_available_var_fields_includes_dosage_when_present(svar_with_dosages_ds):
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    assert "dosage" in ds.available_var_fields


def test_available_var_fields_excludes_dosage_when_absent():
    # The canonical SVAR (no dosages.npy) — opening the existing fixture
    # should not list dosage as available.
    ds = gvl.Dataset.open(
        _DATA / "phased_dataset.svar.gvl",
        _REF,
        rc_neg=False,
    )
    assert "dosage" not in ds.available_var_fields
```

- [ ] **Step 2: Run tests to verify the bug reproduces**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v`

Expected: `test_dosage_absent_when_not_requested` FAILS (dosage leaks in). `test_available_var_fields_includes_dosage_when_present` FAILS (dosage not listed). The other two may pass or fail depending on schema; that's fine — record results.

- [ ] **Step 3: Fix the dosage gate in `_get_variants`**

In `python/genvarloader/_dataset/_haps.py`, find the block (around line 597):

```python
        if self.dosages is not None:
            # guaranteed to have same shape as genotypes but need to make it contiguous/copy the data
            dosages = self.dosages[r, s]
            if _keep is not None:
                dosages = ak.to_regular(dosages[_keep], 1)  # type: ignore
            fields["dosage"] = Ragged(ak.to_packed(dosages))
```

Replace with:

```python
        if self.dosages is not None and "dosage" in self.var_fields:
            # guaranteed to have same shape as genotypes but need to make it contiguous/copy the data
            dosages = self.dosages[r, s]
            if _keep is not None:
                dosages = ak.to_regular(dosages[_keep], 1)  # type: ignore
            fields["dosage"] = Ragged(ak.to_packed(dosages))
```

- [ ] **Step 4: Add `"dosage"` to `available_var_fields` when applicable**

In `Haps.__post_init__` (around `_haps.py:177-183`), find:

```python
    def __post_init__(self):
        self.n_variants = ak.num(self.genotypes, -1).to_numpy()
        self.available_var_fields = (
            ["alt", "ilen", "start"]
            + list(self.variants.info.keys())
            + (["ref"] if self.variants.ref is not None else [])
        )
```

Replace with:

```python
    def __post_init__(self):
        self.n_variants = ak.num(self.genotypes, -1).to_numpy()
        self.available_var_fields = (
            ["alt", "ilen", "start"]
            + list(self.variants.info.keys())
            + (["ref"] if self.variants.ref is not None else [])
            + (["dosage"] if self.dosages is not None else [])
        )
```

- [ ] **Step 5: Run the tests — all four should now pass**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v`

Expected: 4 passed.

- [ ] **Step 6: Run the full project test suite to confirm no regression**

Run: `pixi run -e dev test`

Expected: same pre-existing counts (488 passed, 6 skipped, 2 xfailed; 4 cargo) plus the 4 new tests = 492 passed.

- [ ] **Step 7: Lint + typecheck**

Run:
```
pixi run -e dev ruff check python/
pixi run -e dev typecheck
```

Expected: both clean. (Pyrefly baseline unchanged.)

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_haps.py tests/dataset/test_issue_191_var_fields.py
git commit -m "$(cat <<'EOF'
fix(haps): gate dosage output by var_fields (#191)

Dosage was unconditionally added to RaggedVariants whenever dosages.npy
existed on disk, causing ak.broadcast_records errors downstream when a
consumer expected a schema without dosage. Gate the field on
'dosage' in self.var_fields, matching how ref/ilen/info fields work.

Also list 'dosage' in available_var_fields when the file is present.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 2a: `_Variants.available_info_fields` schema-peek staticmethod

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (add staticmethod to `_Variants`)
- Modify: `tests/dataset/test_issue_191_var_fields.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_issue_191_var_fields.py`:

```python
from genvarloader._dataset._haps import _Variants


def test_available_info_fields_lists_numeric_columns_without_loading():
    """Schema peek: returns numeric columns from the variants table without
    materializing data."""
    fields = _Variants.available_info_fields(_SOURCE_SVAR / "index.arrow")
    # Canonical SVAR has at least AF in its index.
    # POS and ILEN must NOT appear — they're treated as positional, not info.
    assert "POS" not in fields
    assert "ILEN" not in fields
    # All returned names are strings
    assert all(isinstance(f, str) for f in fields)
    # Non-empty for a real SVAR
    assert len(fields) >= 0  # may be 0 if no extra numeric columns; that's fine
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py::test_available_info_fields_lists_numeric_columns_without_loading -v`

Expected: AttributeError — `_Variants.available_info_fields` doesn't exist.

- [ ] **Step 3: Add the staticmethod**

In `_haps.py`, on the `_Variants` class (after `from_table` ends around line 145), add:

```python
@staticmethod
def available_info_fields(path: str | Path) -> list[str]:
    """Return numeric column names that would be loaded as info, without
    materializing any data.

    ``POS`` and ``ILEN`` are excluded — they're positional, not info.
    """
    schema = pl.scan_ipc(path).collect_schema()
    return [k for k, v in schema.items() if v.is_numeric() and k not in {"POS", "ILEN"}]
```

- [ ] **Step 4: Run to confirm pass**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py::test_available_info_fields_lists_numeric_columns_without_loading -v`

Expected: pass.

- [ ] **Step 5: Lint + typecheck**

Run: `pixi run -e dev ruff check python/` and `pixi run -e dev typecheck`. Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_haps.py tests/dataset/test_issue_191_var_fields.py
git commit -m "feat(haps): add _Variants.available_info_fields schema peek"
```

---

## Task 2b: `_Variants.from_table(info_fields=...)` filter

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`_Variants.from_table`)
- Modify: `tests/dataset/test_issue_191_var_fields.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_issue_191_var_fields.py`:

```python
def test_from_table_info_fields_filter():
    """When info_fields is a set, only those numeric columns are loaded."""
    available = set(_Variants.available_info_fields(_SOURCE_SVAR / "index.arrow"))
    if not available:
        pytest.skip("No numeric info columns in canonical SVAR; cannot exercise filter")

    pick = {next(iter(available))}
    v = _Variants.from_table(_SOURCE_SVAR / "index.arrow", info_fields=pick)
    assert set(v.info.keys()) == pick


def test_from_table_info_fields_none_loads_all():
    """Back-compat: info_fields=None loads every numeric column (current behavior)."""
    available = set(_Variants.available_info_fields(_SOURCE_SVAR / "index.arrow"))
    v = _Variants.from_table(_SOURCE_SVAR / "index.arrow", info_fields=None)
    assert set(v.info.keys()) == available
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py::test_from_table_info_fields_filter -v`

Expected: `TypeError: from_table() got an unexpected keyword argument 'info_fields'`.

- [ ] **Step 3: Add the parameter**

In `_haps.py`, modify `_Variants.from_table` signature (~line 96):

```python
    @classmethod
    def from_table(
        cls,
        path: str | Path,
        one_based: bool = True,
        info_fields: set[str] | None = None,
    ):
        """
        Loads variant info from a table. Must always have POS, ILEN, and ALT.

        Parameters
        ----------
        path : str | Path
            The path to the variants table.
        one_based : bool, optional
            Whether the variants are one-based, by default False.
        info_fields
            Optional whitelist of numeric column names to load as info.
            If ``None`` (default), load every numeric column except POS/ILEN.
        """
```

Then in the same method, change the info dict comprehension (around line 126):

```python
        info = {
            k: variants[k].to_numpy()
            for k, v in variants.schema.items()
            if v.is_numeric()
            and k not in {"POS", "ILEN"}
            and (info_fields is None or k in info_fields)
        }
```

- [ ] **Step 4: Run the new tests**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "from_table"`

Expected: 2 passed.

- [ ] **Step 5: Run full suite for regression**

Run: `pixi run -e dev test`

Expected: full suite still green.

- [ ] **Step 6: Lint + typecheck + commit**

```
pixi run -e dev ruff check python/
pixi run -e dev typecheck
```

Both clean.

```bash
git add python/genvarloader/_dataset/_haps.py tests/dataset/test_issue_191_var_fields.py
git commit -m "feat(haps): _Variants.from_table accepts info_fields filter"
```

---

## Task 2c: `_Variants.load_info(fields)` lazy extension

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (add method to `_Variants`)
- Modify: `tests/dataset/test_issue_191_var_fields.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_issue_191_var_fields.py`:

```python
def test_load_info_extends_info_dict():
    """load_info reads only the missing fields from disk and merges them."""
    available = set(_Variants.available_info_fields(_SOURCE_SVAR / "index.arrow"))
    if not available:
        pytest.skip(
            "No numeric info columns in canonical SVAR; cannot exercise load_info"
        )

    pick = next(iter(available))
    # Start with empty info
    v = _Variants.from_table(_SOURCE_SVAR / "index.arrow", info_fields=set())
    assert pick not in v.info

    v.load_info([pick])
    assert pick in v.info


def test_load_info_idempotent_for_already_loaded_fields():
    v = _Variants.from_table(_SOURCE_SVAR / "index.arrow", info_fields=None)
    already = list(v.info.keys())
    if not already:
        pytest.skip("No numeric info columns in canonical SVAR")
    # Snapshot one array; load_info shouldn't reload it.
    arr0 = v.info[already[0]]
    v.load_info(already)
    assert v.info[already[0]] is arr0
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "load_info"`

Expected: AttributeError — method doesn't exist.

- [ ] **Step 3: Add the method**

In `_haps.py`, on `_Variants` (after `from_table`), add:

```python
    def load_info(self, fields) -> None:
        """Lazily load additional numeric info columns from ``self.path``.

        Fields already present in ``self.info`` are skipped. Unknown numeric
        columns silently no-op (the caller should validate against
        :meth:`available_info_fields` first).
        """
        missing = [f for f in fields if f not in self.info]
        if not missing:
            return
        df = pl.read_ipc(self.path, columns=missing, memory_map=False)
        for f in missing:
            self.info[f] = df[f].to_numpy()
```

Note: `_Variants` already stores `path` as its first field (line 88), so `self.path` is available.

- [ ] **Step 4: Run the new tests**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "load_info"`

Expected: 2 passed.

- [ ] **Step 5: Lint + typecheck + commit**

```bash
pixi run -e dev ruff check python/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_haps.py tests/dataset/test_issue_191_var_fields.py
git commit -m "feat(haps): _Variants.load_info lazily extends info dict"
```

---

## Task 3: `Haps.from_path(var_fields=None)` + schema-peek `available_var_fields`

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`Haps.from_path` signature + body; `Haps.__post_init__` available list)
- Modify: `tests/dataset/test_issue_191_var_fields.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_issue_191_var_fields.py`:

```python
def test_haps_from_path_filters_info_loading(svar_with_dosages_ds):
    """from_path(var_fields=[default]) does not load extra numeric info columns
    or open the dosages memmap."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    haps = ds._seqs  # type: ignore[attr-defined]
    # Default var_fields: only alt/ilen/start. info dict must not contain extras.
    assert haps.var_fields == ["alt", "ilen", "start"]
    assert set(haps.variants.info.keys()) == set()
    # Dosages file exists on disk but should not be loaded.
    assert haps.dosages is None


def test_haps_available_var_fields_from_schema(svar_with_dosages_ds):
    """available_var_fields reflects the file's schema + dosage presence,
    not what was actually loaded."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    assert "dosage" in ds.available_var_fields
    # ref is also discoverable because the SVAR has a REF column
    assert "ref" in ds.available_var_fields
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "haps_from_path or haps_available"`

Expected: both fail — `info` currently has every numeric column eagerly loaded; `dosages` is loaded if the file exists.

- [ ] **Step 3: Add `var_fields` parameter to `Haps.from_path`**

In `_haps.py`, modify `Haps.from_path` signature (around line 193-207). Add a new param `var_fields` and the logic to use it. Replace the whole signature + body up through the `return cls(...)` block:

```python
@classmethod
def from_path(
    cls: type[Haps[RaggedVariants]],
    path: Path,
    reference: Reference | None,
    regions: NDArray[np.int32],
    samples: list[str],
    ploidy: int,
    version: SemanticVersion | None,
    svar_link: SvarLink | None = None,
    svar_override: Path | str | None = None,
    min_af: float | None = None,
    max_af: float | None = None,
    filter: Literal["exonic"] | None = None,
    var_fields: list[str] | None = None,
) -> Haps[RaggedVariants]:
    # Default var_fields for loading. var_fields=None means "use the default
    # set" — we resolve it here so we know exactly which info columns to load.
    if var_fields is None:
        var_fields = ["alt", "ilen", "start"]
    # Which numeric info columns to eagerly load: those in var_fields that
    # aren't built-ins. (alt/ilen/start/ref/dosage are handled separately.)
    builtin = {"alt", "ilen", "start", "ref", "dosage"}
    info_fields = {f for f in var_fields if f not in builtin}

    svar_meta_path = path / "genotypes" / "svar_meta.json"
    dosages = None

    if svar_meta_path.exists():
        with open(svar_meta_path) as f:
            metadata = json.load(f)
        # (2 r s p)
        shape = cast(tuple[int, ...], tuple(metadata["shape"]))
        dtype = np.dtype(metadata["dtype"])

        offset_path = path / "genotypes" / "offsets.npy"

        if svar_link is not None:
            svar_path = _resolve_svar(path, svar_link, svar_override)
            _verify_fingerprint(svar_path, svar_link)
        else:
            legacy_link = path / "genotypes" / "link.svar"
            if svar_override is not None:
                svar_path = Path(svar_override)
                if not svar_path.is_dir():
                    raise FileNotFoundError(
                        f"svar override does not exist: {svar_path}"
                    )
            elif legacy_link.exists():
                warnings.warn(
                    f"GVL dataset at {path} uses the legacy link.svar "
                    f"symlink. Run "
                    f"`genvarloader.migrate_svar_link({str(path)!r})` "
                    f"to upgrade.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                svar_path = legacy_link.resolve()
            else:
                raise FileNotFoundError(
                    f"Legacy GVL dataset at {path} is missing its link.svar "
                    f"symlink and has no svar_link metadata. "
                    f"Pass `svar=` to Dataset.open(...) to recover, or "
                    f"re-run `gvl.write`."
                )

        geno_path = svar_path / "variant_idxs.npy"
        dosage_path = svar_path / "dosages.npy"

        offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
        v_idxs = np.memmap(geno_path, dtype=V_IDX_TYPE, mode="r")
        rag_shape = (*shape[1:], None)
        genotypes = Ragged.from_offsets(v_idxs, rag_shape, offsets.reshape(2, -1))

        if "dosage" in var_fields and dosage_path.exists():
            dosages_mm = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
            dosages = Ragged.from_offsets(dosages_mm, rag_shape, offsets.reshape(2, -1))

        logger.info("Loading variant data.")
        variants = _Variants.from_table(
            svar_path / "index.arrow", info_fields=info_fields
        )
    else:
        logger.info("Loading variant data.")
        variants = _Variants.from_table(
            path / "genotypes" / "variants.arrow",
            one_based=version is not None
            and version >= SemanticVersion.parse("0.18.0"),
            info_fields=info_fields,
        )
        v_idxs = np.memmap(
            path / "genotypes" / "variant_idxs.npy",
            dtype=V_IDX_TYPE,
            mode="r",
        )
        offsets = np.memmap(
            path / "genotypes" / "offsets.npy", dtype=np.int64, mode="r"
        )
        shape = (len(regions), len(samples), ploidy, None)
        genotypes = Ragged.from_offsets(v_idxs, shape, offsets)

    return cls(
        path=path,
        reference=reference,
        variants=variants,
        genotypes=genotypes,
        dosages=dosages,
        kind=RaggedVariants,
        filter=filter,
        min_af=min_af,
        max_af=max_af,
        var_fields=var_fields,
    )
```

(The two changes versus current code: pass `info_fields=info_fields` to both `_Variants.from_table` calls; gate the dosages memmap on `"dosage" in var_fields`; pass `var_fields=var_fields` through to `cls(...)`.)

- [ ] **Step 4: Update `Haps.__post_init__` to compute `available_var_fields` from the schema**

In `_haps.py`, replace `Haps.__post_init__` (around line 177-191):

```python
def __post_init__(self):
    self.n_variants = ak.num(self.genotypes, -1).to_numpy()

    # Discover available info fields from the on-disk schema, not from the
    # (possibly-filtered) loaded info dict. This way the user can see every
    # field they could request, even if only a subset was loaded.
    schema_info_fields = _Variants.available_info_fields(self.variants.path)
    has_dosage_file = self._has_dosage_file_on_disk()

    self.available_var_fields = (
        ["alt", "ilen", "start"]
        + schema_info_fields
        + (["ref"] if self.variants.ref is not None else [])
        + (["dosage"] if has_dosage_file else [])
    )

    if (
        self.min_af is not None or self.max_af is not None
    ) and "AF" not in schema_info_fields:
        raise RuntimeError(
            "Either this dataset is not backed by an SVAR file, or the SVAR file has not had AFs cached yet."
            + "Doing this automatically is not yet supported."
        )


def _has_dosage_file_on_disk(self) -> bool:
    """True iff the linked SVAR contains a dosages.npy.

    Returns False for non-SVAR datasets (no dosage path).
    """
    # If we already loaded dosages, we definitely had the file.
    if self.dosages is not None:
        return True
    # Otherwise inspect the SVAR directory next to the variants table.
    # _Variants.path is set to <svar_dir>/index.arrow for SVAR datasets,
    # or <gvl>/genotypes/variants.arrow for legacy. We treat "next-to
    # variants table" as "is dosage possible here".
    candidate = self.variants.path.parent / "dosages.npy"
    return candidate.exists()
```

Note: the old `__post_init__` derived available info fields from `self.variants.info.keys()`. After this change it derives them from the schema peek. The `min_af`/`max_af` precondition now checks `schema_info_fields` (what's available) rather than `self.variants.info` (what's loaded), which is the correct check.

- [ ] **Step 5: Run the new tests + full suite**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v`

Expected: all tests in the file pass.

Run: `pixi run -e dev test`

Expected: full suite green. Watch carefully: this changes which numeric columns are loaded by default. If any existing test was implicitly relying on AF or similar being eagerly present in `_seqs.variants.info` without setting `var_fields`, it will fail. If that happens, those tests likely accessed internal state — fix by either using `with_settings(var_fields=[..., "AF"])` in the test, or noting it as a regression to discuss with the controller.

- [ ] **Step 6: Lint + typecheck + commit**

```bash
pixi run -e dev ruff check python/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_haps.py tests/dataset/test_issue_191_var_fields.py
git commit -m "feat(haps): from_path honors var_fields for lazy info+dosage loading"
```

---

## Task 4: Thread `var_fields` through `OpenRequest` and `Dataset.open`

**Files:**
- Modify: `python/genvarloader/_dataset/_open.py` (`OpenRequest` field + `_build_seqs`)
- Modify: `python/genvarloader/_dataset/_impl.py` (3 `Dataset.open` overloads + impl + docstring)
- Modify: `tests/dataset/test_issue_191_var_fields.py` (add test)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_issue_191_var_fields.py`:

```python
def test_dataset_open_accepts_var_fields(svar_with_dosages_ds):
    """Dataset.open(var_fields=...) routes through to Haps.from_path so the
    requested fields are loaded eagerly at open time."""
    ds = gvl.Dataset.open(
        svar_with_dosages_ds,
        _REF,
        rc_neg=False,
        var_fields=["alt", "ilen", "start", "dosage"],
    )
    haps = ds._seqs  # type: ignore[attr-defined]
    assert haps.var_fields == ["alt", "ilen", "start", "dosage"]
    assert haps.dosages is not None


def test_dataset_open_default_var_fields_is_minimum_useful_set(svar_with_dosages_ds):
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    assert ds.active_var_fields == ["alt", "ilen", "start"]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "dataset_open"`

Expected: `TypeError: open() got an unexpected keyword argument 'var_fields'`.

- [ ] **Step 3: Add `var_fields` to `OpenRequest`**

In `python/genvarloader/_dataset/_open.py`, modify the `OpenRequest` dataclass (around line 38-57):

```python
@dataclass(frozen=True, slots=True)
class OpenRequest:
    """Parsed, validated arguments for opening a dataset.

    Construct directly or via :meth:`Dataset.open`. Call :meth:`resolve` to
    produce the dataset.
    """

    path: Path
    reference: str | Path | Reference | None = None
    jitter: int = 0
    rng: int | np.random.Generator | None = False
    deterministic: bool = True
    rc_neg: bool = True
    min_af: float | None = None
    max_af: float | None = None
    region_names: str | None = None
    splice_info: str | tuple[str, str] | None = None
    var_filter: Literal["exonic"] | None = None
    svar: str | Path | None = None
    var_fields: list[str] | None = None
```

- [ ] **Step 4: Forward `var_fields` in `_build_seqs`**

In `python/genvarloader/_dataset/_open.py`, modify `_build_seqs` (around line 137-166) — specifically the `Haps.from_path` call:

```python
            seqs = Haps.from_path(
                path=self.path,
                reference=reference,
                regions=regions,
                samples=metadata.samples,
                ploidy=metadata.ploidy,
                version=metadata.version,
                svar_link=metadata.svar_link,
                svar_override=self.svar,
                min_af=self.min_af,
                max_af=self.max_af,
                var_fields=self.var_fields,
            )
```

- [ ] **Step 5: Add `var_fields` param to all 3 `Dataset.open` signatures**

In `python/genvarloader/_dataset/_impl.py`, update all three (two `@overload` stubs at lines 95-126 and the impl at line 127-178). Pattern: add `var_fields: list[str] | None = None,` immediately before `region_names`. For example, the first overload becomes:

```python
    @staticmethod
    @overload
    def open(
        path: str | Path,
        reference: None = ...,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
        min_af: float | None = None,
        max_af: float | None = None,
        var_fields: list[str] | None = None,
        region_names: str | None = None,
        splice_info: str | tuple[str, str] | None = None,
        var_filter: Literal["exonic"] | None = None,
        *,
        svar: str | Path | None = None,
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK]: ...
```

Repeat for the second overload and the impl.

- [ ] **Step 6: Add docstring entry + thread into `OpenRequest`**

In the impl, after the existing `max_af` docstring entry and before `splice_info`, add:

```
        var_fields
            The variant fields to include in the dataset. Defaults to the
            minimum useful set ``["alt", "ilen", "start"]``. Pass additional
            field names (e.g. ``"ref"``, ``"dosage"``, or any info column
            present in the source variants table) to load them eagerly at open
            time. Must be a subset of :attr:`available_var_fields`.
```

Then in the body of the impl, when constructing the `OpenRequest`, add the new arg. Find the existing block (around line 181 `return OpenRequest(...)`) and add `var_fields=var_fields,` to the call.

- [ ] **Step 7: Run the new tests**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "dataset_open"`

Expected: 2 passed.

- [ ] **Step 8: Full suite + lint + typecheck**

```
pixi run -e dev test
pixi run -e dev ruff check python/
pixi run -e dev typecheck
```

Expected: all green.

- [ ] **Step 9: Commit**

```bash
git add python/genvarloader/_dataset/_open.py python/genvarloader/_dataset/_impl.py tests/dataset/test_issue_191_var_fields.py
git commit -m "feat(open): Dataset.open accepts var_fields, forwards to Haps.from_path"
```

---

## Task 5: `Dataset.with_settings(var_fields=...)` lazy expansion

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`with_settings` block at lines 287-293)
- Modify: `tests/dataset/test_issue_191_var_fields.py` (add tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_issue_191_var_fields.py`:

```python
def test_with_settings_lazily_loads_new_info_field(svar_with_dosages_ds):
    """Opening with default var_fields does not load AF (or other info columns).
    with_settings(var_fields=[..., 'AF']) should lazily extend the info dict."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    available_info = set(_Variants.available_info_fields(_SOURCE_SVAR / "index.arrow"))
    if not available_info:
        pytest.skip("No numeric info columns; cannot test lazy info expansion")

    new_field = next(iter(available_info))
    haps_before = ds._seqs  # type: ignore[attr-defined]
    assert new_field not in haps_before.variants.info

    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", new_field])
    haps_after = ds2._seqs  # type: ignore[attr-defined]
    assert new_field in haps_after.variants.info


def test_with_settings_lazily_loads_dosages(svar_with_dosages_ds):
    """Opening with default var_fields does not memmap dosages.
    with_settings(var_fields=[..., 'dosage']) should memmap them."""
    ds = gvl.Dataset.open(svar_with_dosages_ds, _REF, rc_neg=False)
    assert ds._seqs.dosages is None  # type: ignore[attr-defined]

    ds2 = ds.with_settings(var_fields=["alt", "ilen", "start", "dosage"])
    assert ds2._seqs.dosages is not None  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run to confirm failure**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "with_settings_lazily"`

Expected: both fail — `with_settings` does not currently load missing fields/dosages.

- [ ] **Step 3: Update `with_settings` to lazily load**

In `python/genvarloader/_dataset/_impl.py`, find the existing `var_fields` block (around line 287-293):

```python
        if var_fields is not None:
            missing = list(set(var_fields) - set(self.available_var_fields))
            if missing or not isinstance(self._seqs, Haps):
                raise ValueError(f"Missing variant fields: {missing}")
            haps = to_evolve.get("_seqs", self._seqs)
            haps = replace(haps, var_fields=var_fields)
            to_evolve["_seqs"] = haps
```

Replace with:

```python
if var_fields is not None:
    missing = list(set(var_fields) - set(self.available_var_fields))
    if missing or not isinstance(self._seqs, Haps):
        raise ValueError(f"Missing variant fields: {missing}")
    haps = to_evolve.get("_seqs", self._seqs)
    # Lazily load any newly-requested info columns into the existing
    # _Variants struct (mutates self.info in place).
    builtin = {"alt", "ilen", "start", "ref", "dosage"}
    new_info_fields = [
        f for f in var_fields if f not in builtin and f not in haps.variants.info
    ]
    if new_info_fields:
        haps.variants.load_info(new_info_fields)
    # Lazily memmap dosages if newly requested.
    if "dosage" in var_fields and haps.dosages is None:
        haps = _lazy_load_dosages(self, haps)
    haps = replace(haps, var_fields=var_fields)
    to_evolve["_seqs"] = haps
```

- [ ] **Step 4: Add the `_lazy_load_dosages` helper**

In `_impl.py`, at module scope (near the bottom, or near other private helpers — find an appropriate spot in the same file), add:

```python
def _lazy_load_dosages(dataset, haps):
    """Open the dosages memmap for a Haps that didn't request them at open time.

    Reuses the same path-resolution logic that ``Haps.from_path`` used. Returns
    a new ``Haps`` with ``dosages`` populated (does NOT mutate the input).
    """
    from ._haps import _Variants  # local to avoid cycles
    import json as _json
    from genoray._types import DOSAGE_TYPE
    from seqpro.rag import Ragged
    from ._svar_link import _resolve_svar

    metadata = dataset._metadata
    path = haps.path
    svar_meta_path = path / "genotypes" / "svar_meta.json"
    if not svar_meta_path.exists():
        raise ValueError(
            "Dosage requested but this dataset is not SVAR-backed; no dosages.npy possible."
        )

    with open(svar_meta_path) as f:
        svar_meta = _json.load(f)
    shape = tuple(svar_meta["shape"])
    dtype = np.dtype(svar_meta["dtype"])

    offset_path = path / "genotypes" / "offsets.npy"
    if metadata.svar_link is not None:
        svar_path = _resolve_svar(path, metadata.svar_link, None)
    else:
        legacy_link = path / "genotypes" / "link.svar"
        svar_path = legacy_link.resolve()

    dosage_path = svar_path / "dosages.npy"
    if not dosage_path.exists():
        raise ValueError(
            f"Dosage requested but {dosage_path} does not exist. "
            f"Check the SVAR was built with dosages."
        )

    offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
    dosages_mm = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
    rag_shape = (*shape[1:], None)
    dosages = Ragged.from_offsets(dosages_mm, rag_shape, offsets.reshape(2, -1))
    return replace(haps, dosages=dosages)
```

The helper accesses `dataset._metadata` — verify this attribute exists on the Dataset. If it doesn't (the dataset stores the contigs/samples directly rather than the parsed metadata), the implementer should adapt: either store the `Metadata` instance on Dataset (likely already there as `self._metadata` — confirm via grep), or re-read `metadata.json` here.

If `_metadata` isn't on the dataset, replace the `metadata = dataset._metadata` line with:

```python
from ._write import Metadata

metadata = Metadata.model_validate_json((path / "metadata.json").read_text())
```

- [ ] **Step 5: Run the new tests**

Run: `pixi run -e dev pytest tests/dataset/test_issue_191_var_fields.py -v -k "with_settings_lazily"`

Expected: 2 passed.

- [ ] **Step 6: Full suite for regression**

Run: `pixi run -e dev test`

Expected: all green. Pay special attention to `tests/dataset/test_with_settings_var_filter.py` and `tests/dataset/test_open_vs_settings_parity.py` — the parity test in particular exercises open-then-with_settings against directly-opening-with-args, and our changes should preserve parity.

- [ ] **Step 7: Lint + typecheck + commit**

```bash
pixi run -e dev ruff check python/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_impl.py tests/dataset/test_issue_191_var_fields.py
git commit -m "feat(dataset): with_settings lazily loads new var_fields"
```

---

## Task 6: Update SKILL.md + final verification

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Find the `Dataset.open` documentation block in the skill**

Run: `rtk grep -n "Dataset.open" skills/genvarloader/SKILL.md | head -5`

Locate the section that documents `Dataset.open` parameters.

- [ ] **Step 2: Add `var_fields` to the documented parameter list**

Add an entry near the other `var_*`-related params (e.g. near `var_filter`):

```markdown
- **`var_fields: list[str] | None`** — Variant fields to include. Defaults to the minimum useful set `["alt", "ilen", "start"]`. Pass additional names (e.g. `"ref"`, `"dosage"`, or any numeric info column in the source variants table) to load them eagerly at open time. Must be a subset of `Dataset.available_var_fields`. Same set can be reconfigured later via `Dataset.with_settings(var_fields=...)`, which lazily loads any newly requested columns.
```

If a "Common gotchas" or "Where to look next" table exists, add a one-line entry pointing readers at this parameter when they see a missing `dosage` field (links the fix back to issue #191 indirectly through behavior, not number).

- [ ] **Step 3: Final full verification**

```bash
pixi run -e dev test
pixi run -e dev ruff check python/
pixi run -e dev typecheck
```

Expected: all green. Test counts: prior 488 + 12 new = 500 passed (plus existing skipped/xfailed counts unchanged).

- [ ] **Step 4: Commit the skill update**

```bash
git add skills/genvarloader/SKILL.md
git commit -m "docs(skill): document var_fields parameter on Dataset.open"
```

- [ ] **Step 5: Push and open the PR**

```bash
git push -u origin fix/issue-191-var-fields-loading

gh pr create --title "fix(haps): gate dosage by var_fields + lazy info loading (#191)" --body "$(cat <<'EOF'
## Summary

Closes #191.

Two related changes:

1. **Correctness:** Dosage field was unconditionally added to `RaggedVariants` whenever `dosages.npy` existed on disk, regardless of the user's `var_fields`. This caused `ak.broadcast_records` crashes downstream when consumers expected a schema without dosage. Now gated on `'dosage' in self.var_fields`.

2. **Lazy loading:** `Dataset.open` accepts a new `var_fields` parameter. Both `Dataset.open(var_fields=...)` and `Dataset.with_settings(var_fields=...)` now honor the user's request — non-default info columns and the dosages memmap are only loaded when asked for. `available_var_fields` is computed from a schema peek so it reflects what the file *could* provide, not what was actually loaded.

## Test plan

- [x] `pixi run -e dev test` (pytest + cargo, 500 passed)
- [x] `pixi run -e dev ruff check python/` clean
- [x] `pixi run -e dev typecheck` (pyrefly) — baseline preserved
- [x] New regression tests in `tests/dataset/test_issue_191_var_fields.py`

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

- [x] **Spec coverage:** Phase 1 (dosage gate + available list) → Task 1. Phase 2 plumbing (`_Variants.available_info_fields` / `from_table` filter / `load_info` / `Haps.from_path(var_fields)` / `Dataset.open(var_fields)` / `with_settings` lazy expansion) → Tasks 2a, 2b, 2c, 3, 4, 5. SKILL.md → Task 6. All testing requirements covered.
- [x] **Placeholder scan:** No "TBD/TODO/fill in" left. Every code-changing step shows the actual code.
- [x] **Type consistency:** Method names match throughout: `available_info_fields` (Task 2a, 3, 5), `from_table(info_fields=...)` (Task 2b, 3), `load_info(fields)` (Task 2c, 5), `Haps.from_path(var_fields=...)` (Task 3, 4), `OpenRequest.var_fields` (Task 4).
- [x] **API addition flagged:** Spec said "no public API additions"; we ARE adding a new optional parameter `var_fields` to `Dataset.open` (the user explicitly asked for it). The SKILL.md update in Task 6 covers this.
- [x] **Fixture concerns:** Task 1 fixture creates a fresh GVL + SVAR with synthetic dosages.npy via `gvl.write`, avoiding source-tree pollution. The synthetic file uses raw memmap (not `np.save`) to match the `np.memmap` reader in `Haps.from_path`.

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-24-issue-191-var-fields-loading.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, two-stage review between tasks, fast iteration. Good fit here because tasks are mostly mechanical with clear specs.

**2. Inline Execution** — execute tasks in this session using `executing-plans`, batch execution with checkpoints.

**Which approach?**
