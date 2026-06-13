# Variant assembly (B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `get_variants_flat` the single variant decode (retiring the awkward `_get_variants`) and add an optional empty-group dummy-variant fill to gvl's variant-record output, working identically in flat and ragged modes.

**Architecture:** B1 routes both `Haps.__call__` and the `get_haps_and_shifts` variants branch through `get_variants_flat`; ragged mode converts at the `_query.py` boundary via `_FlatVariants.to_ragged()` (already wired for `_Flat`/`_FlatAnnotatedHaps` in A). B2 adds two numba kernels (`_fill_empty_scalar`, `_fill_empty_seq`) and `_FlatVariants.fill_empty_groups(dummy)`, applied as the final step inside `get_variants_flat` when a `DummyVariant` is configured via `with_settings`. The dummy spec lives on the `Haps` reconstructor (like `min_af`/`max_af`/`var_fields`), not on `QueryView`.

**Tech Stack:** Python, numba (`@nb.njit`), numpy, awkward (only for `to_ragged()` + the snapshot reference), seqpro `Ragged`, pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-06-13-variant-assembly-B-design.md`

**Run tests with:** `pixi run -e dev pytest <path> -v` (test data already generated; if not, `pixi run -e dev gen` once).

---

## File Structure

- **Modify** `python/genvarloader/_dataset/_haps.py` — `Haps.__call__` + `get_haps_and_shifts` variants branch route to `get_variants_flat`; **delete** `_get_variants`, `_get_alleles`, `_get_info`; add a `dummy_variant: "DummyVariant | None" = None` field to the `Haps` dataclass (TYPE_CHECKING import).
- **Modify** `python/genvarloader/_dataset/_query.py` — add `_FlatVariants` to the boundary `to_ragged()` conversion in `getitem`.
- **Modify** `python/genvarloader/_dataset/_flat_variants.py` — add `DummyVariant` dataclass, the `_fill_empty_scalar`/`_fill_empty_seq` numba kernels, `_FlatVariants.fill_empty_groups(dummy)`, and apply the fill at the end of `get_variants_flat`.
- **Modify** `python/genvarloader/_dataset/_impl.py` — add `dummy_variant` parameter to `with_settings`; add a guard in `__getitem__` (raise if `dummy_variant` set and output kind ≠ variants).
- **Modify** `python/genvarloader/__init__.py` — export `DummyVariant`.
- **Modify** `tests/dataset/test_flat_mode_equivalence.py` — flat↔ragged parity with dummy fill.
- **Modify** `tests/unit/dataset/test_flat_variants_type.py` — kernel + `fill_empty_groups` + `DummyVariant` unit tests.
- **Modify** `tests/dataset/test_no_awkward_in_hotpath.py` — assert the dummy-fill path stays awkward-free.
- **Modify** `tests/dataset/test_output_format.py` (or `test_with_methods.py`) — `with_settings(dummy_variant=...)` + validation tests.
- **Modify** `skills/genvarloader/SKILL.md` — document `with_settings(dummy_variant=...)` + `DummyVariant`.

---

## Task 1: B1 — always-flat variant decode (retire the awkward path)

This is a behavior-preserving refactor. The regression oracle is the committed snapshot `tests/dataset/_snapshots/variants_ragged.npz` (generated from the old awkward path); it must stay green and **must not be regenerated**.

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (`__call__` ~514-525; `get_haps_and_shifts` variants branch ~569-580; delete `_get_variants`/`_get_alleles`/`_get_info`)
- Modify: `python/genvarloader/_dataset/_query.py` (`getitem` ~110-113)

- [ ] **Step 1: Confirm the baseline is green**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py tests/dataset/test_flat_mode_equivalence.py tests/dataset/test_flat_variants.py -q`
Expected: PASS. This is the pre-refactor reference (esp. `test_flat_getitem_snapshot.py::test_getitem_snapshot[variants_ragged-*]`, which pins the awkward output via `_snapshots/variants_ragged.npz`).

- [ ] **Step 2: Route `Haps.__call__` variants branch to the flat decode**

In `python/genvarloader/_dataset/_haps.py`, replace the variants branch of `__call__` (currently the `if flat: ... else: ragv = self._get_variants(...)` block) with an unconditional flat decode:

```python
        if issubclass(self.kind, RaggedVariants):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for RaggedVariants."
                )
            from ._flat_variants import get_variants_flat

            return cast(_H, get_variants_flat(self, idx))
```

(The `flat` parameter on `__call__` is retained for signature stability across reconstructors but is no longer read for variants — the `_query.py` boundary decides ragged-vs-flat via `view.flat_output`.)

- [ ] **Step 3: Route the `get_haps_and_shifts` variants branch to the flat decode**

In `get_haps_and_shifts`, replace the `elif issubclass(self.kind, RaggedVariants):` branch (which calls `self._get_variants(... keep=..., keep_offsets=...)`) with:

```python
        elif issubclass(self.kind, RaggedVariants):
            if splice_plan is not None:
                raise NotImplementedError(
                    "Spliced output is not supported for RaggedVariants."
                )
            from ._flat_variants import get_variants_flat

            out = get_variants_flat(self, idx)
```

- [ ] **Step 4: Delete the awkward decode helpers**

Delete the methods `_get_variants`, `_get_alleles`, and `_get_info` from `_haps.py` (they are now unused — verify with `rtk grep "_get_variants\|_get_alleles\|_get_info" python/genvarloader/` returning only references inside docstrings/comments of `_flat_variants.py`). Do NOT delete `_allele_bytes_sum` or `_build_allele_layout` (still used by `_FlatAlleles.to_ragged` / elsewhere — verify before removing anything else).

- [ ] **Step 5: Add `_FlatVariants` to the boundary `to_ragged()` conversion**

In `python/genvarloader/_dataset/_query.py`, in `getitem`, extend the conversion tuple (currently converts `_Flat`/`_FlatAnnotatedHaps`):

```python
        recon = tuple(
            o.to_ragged() if isinstance(o, (_Flat, _FlatAnnotatedHaps, _FlatVariants)) else o
            for o in recon
        )
```

`_FlatVariants` is already imported at module top (from A). In flat mode this block is skipped (the `if not view.flat_output:` guard), so `_FlatVariants` passes through; in ragged mode it now converts.

- [ ] **Step 6: Run the regression gate**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py tests/dataset/test_flat_mode_equivalence.py tests/dataset/test_flat_variants.py tests/dataset/test_no_awkward_in_hotpath.py -q`
Expected: PASS, with `_snapshots/variants_ragged.npz` unchanged (do NOT pass `--snapshot-update` or regenerate). Then run the broader suite:
Run: `pixi run -e dev pytest tests/ -q`
Expected: PASS (same skip/xfail counts as before).

- [ ] **Step 7: Confirm the awkward helpers are gone**

Run: `rtk grep "def _get_variants\|def _get_alleles\|def _get_info" python/genvarloader/`
Expected: no matches.

- [ ] **Step 8: Commit**

```bash
cd /root/GenVarLoader/.claude/worktrees/flat-output-impl
rtk git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_query.py
rtk git commit -m "refactor(flat): retire awkward _get_variants; ragged variants decode via flat path"
```

---

## Task 2: `DummyVariant` type + empty-group fill kernels + `fill_empty_groups`

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py`
- Test: `tests/unit/dataset/test_flat_variants_type.py` (append)

- [ ] **Step 1: Write the failing tests (append)**

```python
# append to tests/unit/dataset/test_flat_variants_type.py
def test_fill_empty_scalar_kernel():
    from genvarloader._dataset._flat_variants import _fill_empty_scalar

    data = np.array([10, 11, 20], np.int32)
    offsets = np.array([0, 0, 2, 2, 3], np.int64)  # rows: empty, [10,11], empty, [20]
    new_data, new_off = _fill_empty_scalar(data, offsets, np.int32(-1))
    assert new_off.tolist() == [0, 1, 3, 4, 5]
    assert new_data.tolist() == [-1, 10, 11, -1, 20]


def test_fill_empty_seq_kernel():
    from genvarloader._dataset._flat_variants import _fill_empty_seq

    # 3 rows: empty, ["AC","G"], empty
    data = np.frombuffer(b"ACG", np.uint8).copy()
    var_off = np.array([0, 0, 2, 2], np.int64)      # per-row variant boundaries
    seq_off = np.array([0, 2, 3], np.int64)         # per-variant byte boundaries
    dummy = np.frombuffer(b"N", np.uint8).copy()
    nd, nvar, nseq = _fill_empty_seq(data, var_off, seq_off, dummy)
    assert nvar.tolist() == [0, 1, 3, 4]            # each empty row gains 1 variant
    assert nseq.tolist() == [0, 1, 3, 4, 5]         # dummy(1) AC(2) G(1) dummy(1)
    assert bytes(nd) == b"NACGN"


def test_fill_empty_groups_roundtrip():
    import awkward as ak

    from genvarloader._dataset._flat_variants import DummyVariant, _FlatAlleles, _FlatVariants
    from genvarloader._flat import _Flat

    # b*p = 3 rows: row0 empty, row1 has [b"AC", b"G"], row2 empty
    group_off = np.array([0, 0, 2, 2], np.int64)
    alt = _FlatAlleles(
        byte_data=np.frombuffer(b"ACG", np.uint8).copy(),
        seq_offsets=np.array([0, 2, 3], np.int64),
        var_offsets=group_off.copy(),
        shape=(3, None),
    )
    start = _Flat.from_offsets(np.array([5, 9], np.int32), (3, None), group_off.copy())
    fv = _FlatVariants(fields={"alt": alt, "start": start})
    filled = fv.fill_empty_groups(DummyVariant(start=-1, alt=b"N"))
    rv = filled.to_ragged()
    # empty rows now hold exactly the dummy; non-empty row unchanged
    assert ak.to_list(rv["alt"]) == [[b"N"], [b"AC", b"G"], [b"N"]]
    assert ak.to_list(rv["start"]) == [[-1], [5, 9], [-1]]


def test_fill_empty_groups_noop_when_no_empties():
    from genvarloader._dataset._flat_variants import DummyVariant, _FlatAlleles, _FlatVariants
    from genvarloader._flat import _Flat
    import awkward as ak

    group_off = np.array([0, 1, 2], np.int64)  # every row has 1 variant
    alt = _FlatAlleles(np.frombuffer(b"AG", np.uint8).copy(),
                       np.array([0, 1, 2], np.int64), group_off.copy(), (2, None))
    start = _Flat.from_offsets(np.array([3, 7], np.int32), (2, None), group_off.copy())
    fv = _FlatVariants(fields={"alt": alt, "start": start})
    filled = fv.fill_empty_groups(DummyVariant())
    assert ak.to_list(filled.to_ragged()["alt"]) == [[b"A"], [b"G"]]
    assert ak.to_list(filled.to_ragged()["start"]) == [[3], [7]]
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py -k "fill or dummy" -v`
Expected: FAIL with `ImportError`/`AttributeError` (`DummyVariant`, `_fill_empty_scalar`, `_fill_empty_seq`, `fill_empty_groups` not defined).

- [ ] **Step 3: Implement `DummyVariant`, kernels, and `fill_empty_groups`**

Add to `python/genvarloader/_dataset/_flat_variants.py`. First the dataclass (place near the top, after the imports):

```python
@dataclass(frozen=True)
class DummyVariant:
    """Per-field values for the dummy variant inserted into empty
    (region, sample, ploid) groups. Unspecified info fields default to ``0``
    for integer columns and ``NaN`` for float columns."""

    start: int = -1
    ilen: int = 0
    dosage: float = 0.0
    ref: bytes = b"N"
    alt: bytes = b"N"
    info: dict[str, Any] = field(default_factory=dict)

    def scalar_for(self, name: str, dtype: np.dtype):
        """Return the dummy fill value for a scalar field, as a numpy scalar of ``dtype``."""
        dt = np.dtype(dtype)
        if name == "start":
            return dt.type(self.start)
        if name == "ilen":
            return dt.type(self.ilen)
        if name == "dosage":
            return dt.type(self.dosage)
        if name in self.info:
            return dt.type(self.info[name])
        if np.issubdtype(dt, np.floating):
            return dt.type(np.nan)
        return dt.type(0)
```

Then the kernels (place near the other `@nb.njit` kernels):

```python
@nb.njit(nogil=True, cache=True)
def _fill_empty_scalar(data, offsets, fill):  # pragma: no cover - njit
    """Insert one ``fill`` element into each empty row; copy non-empty rows
    through. Returns ``(new_data, new_offsets)``."""
    n_rows = offsets.shape[0] - 1
    new_offsets = np.empty(n_rows + 1, np.int64)
    new_offsets[0] = 0
    for i in range(n_rows):
        ln = offsets[i + 1] - offsets[i]
        new_offsets[i + 1] = new_offsets[i] + (ln if ln > 0 else 1)
    new_data = np.empty(new_offsets[n_rows], data.dtype)
    for i in range(n_rows):
        s = offsets[i]
        e = offsets[i + 1]
        d = new_offsets[i]
        if e == s:
            new_data[d] = fill
        else:
            for k in range(s, e):
                new_data[d] = data[k]
                d += 1
    return new_data, new_offsets


@nb.njit(nogil=True, cache=True)
def _fill_empty_seq(data, var_offsets, seq_offsets, dummy):  # pragma: no cover - njit
    """Two-level analogue of ``_fill_empty_scalar`` for allele bytestrings.
    Empty variant-rows receive one dummy allele of ``dummy`` bytes. Returns
    ``(new_data, new_var_offsets, new_seq_offsets)``."""
    n_rows = var_offsets.shape[0] - 1
    L = dummy.shape[0]
    new_var = np.empty(n_rows + 1, np.int64)
    new_var[0] = 0
    for i in range(n_rows):
        nv = var_offsets[i + 1] - var_offsets[i]
        new_var[i + 1] = new_var[i] + (nv if nv > 0 else 1)
    total_vars = new_var[n_rows]
    new_seq = np.empty(total_vars + 1, np.int64)
    new_seq[0] = 0
    vptr = 0
    for i in range(n_rows):
        vs = var_offsets[i]
        ve = var_offsets[i + 1]
        if ve == vs:
            new_seq[vptr + 1] = new_seq[vptr] + L
            vptr += 1
        else:
            for v in range(vs, ve):
                vlen = seq_offsets[v + 1] - seq_offsets[v]
                new_seq[vptr + 1] = new_seq[vptr] + vlen
                vptr += 1
    new_data = np.empty(new_seq[total_vars], np.uint8)
    vptr = 0
    dptr = 0
    for i in range(n_rows):
        vs = var_offsets[i]
        ve = var_offsets[i + 1]
        if ve == vs:
            for k in range(L):
                new_data[dptr] = dummy[k]
                dptr += 1
            vptr += 1
        else:
            for v in range(vs, ve):
                bs = seq_offsets[v]
                be = seq_offsets[v + 1]
                for k in range(bs, be):
                    new_data[dptr] = data[k]
                    dptr += 1
                vptr += 1
    return new_data, new_var, new_seq
```

Then `fill_empty_groups` on `_FlatVariants` (add as a method):

```python
    def fill_empty_groups(self, dummy: "DummyVariant") -> "_FlatVariants":
        """Insert one dummy variant into each empty (b*p) group; non-empty
        groups are unchanged. Every field shares the same empty-row pattern, so
        the rebuilt offsets stay consistent across fields."""
        from .._flat import _Flat

        new_fields: dict[str, Any] = {}
        for name, f in self.fields.items():
            if isinstance(f, _FlatAlleles):
                db = np.frombuffer(dummy.alt if name == "alt" else dummy.ref, np.uint8).copy()
                nd, nvar, nseq = _fill_empty_seq(f.byte_data, f.var_offsets, f.seq_offsets, db)
                new_fields[name] = _FlatAlleles(nd, nseq, nvar, f.shape)
            else:
                fill = dummy.scalar_for(name, f.data.dtype)
                nd, noff = _fill_empty_scalar(f.data, f.offsets, fill)
                new_fields[name] = _Flat.from_offsets(nd, f.shape, noff)
        return _FlatVariants(new_fields)
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py -k "fill or dummy" -v`
Expected: PASS. Then ruff: `pixi run -e dev ruff check python/genvarloader/_dataset/_flat_variants.py tests/unit/dataset/test_flat_variants_type.py`

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/unit/dataset/test_flat_variants_type.py
rtk git commit -m "feat(flat): DummyVariant + empty-group fill kernels + fill_empty_groups"
```

---

## Task 3: Apply the fill in `get_variants_flat` (Haps.dummy_variant field)

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (Haps dataclass field)
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (`get_variants_flat`)
- Test: `tests/unit/dataset/test_flat_variants_type.py` (append)

- [ ] **Step 1: Write the failing test (append)**

This test constructs a `Haps` via the public dataset and sets `dummy_variant` directly on the reconstructor (the user-facing `with_settings` path is Task 4):

```python
# append to tests/unit/dataset/test_flat_variants_type.py
def test_get_variants_flat_fills_empty_groups(snap_dataset):
    import awkward as ak
    from dataclasses import replace

    from genvarloader._dataset._flat_variants import DummyVariant, get_variants_flat

    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    haps = ds._seqs  # the Haps reconstructor
    idx = np.arange(min(6, snap_dataset.shape[0]) * snap_dataset.shape[1])

    plain = get_variants_flat(haps, idx).to_ragged()
    has_empty = any(len(g) == 0 for row in ak.to_list(plain["start"]) for g in [row])
    # (snap_dataset has regions with no variants for some samples)

    haps_d = replace(haps, dummy_variant=DummyVariant(start=-1, alt=b"N", ref=b"N"))
    filled = get_variants_flat(haps_d, idx).to_ragged()

    # no group is empty after filling
    for ploid_groups in ak.to_list(filled["start"]):
        for g in ploid_groups:
            assert len(g) >= 1
    # non-empty groups are unchanged vs plain (dummy only added to empties)
    plain_starts = ak.to_list(plain["start"])
    filled_starts = ak.to_list(filled["start"])
    for pr, fr in zip(plain_starts, filled_starts):
        for pg, fg in zip(pr, fr):
            if len(pg) > 0:
                assert fg == pg
            else:
                assert fg == [-1]
```

`snap_dataset` is in `tests/dataset/conftest.py`; if this unit test can't see it (different tree), reference it via the integration test in Task 5 instead and keep this as a fixture-built `Haps`. If `ds._seqs` is not the right attribute, inspect `Dataset` for the reconstructor handle (it is `_seqs`, a `Haps`, per `_impl.py`).

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py::test_get_variants_flat_fills_empty_groups -v`
Expected: FAIL with `TypeError` (`replace` rejects unknown field `dummy_variant`) — the Haps field doesn't exist yet.

- [ ] **Step 3: Add the `dummy_variant` field to `Haps`**

In `python/genvarloader/_dataset/_haps.py`, add a TYPE_CHECKING import and the field. In the existing `if TYPE_CHECKING:` block (add one if absent) include:

```python
if TYPE_CHECKING:
    from ._flat_variants import DummyVariant
```

In the `Haps` dataclass, add the field immediately after `var_fields` (which has a default) and before the `available_var_fields: ... = field(init=False)` line:

```python
    var_fields: list[str] = field(default_factory=lambda: ["alt", "ilen", "start"])
    dummy_variant: "DummyVariant | None" = None
    available_var_fields: list[str] = field(init=False)
```

(The string annotation avoids a runtime import cycle; the default `None` keeps it optional so existing `Haps(...)` construction is unaffected. Verify `Haps.to_kind` / `_build_reconstructor` preserve it — they use `replace`, so it carries through.)

- [ ] **Step 4: Apply the fill at the end of `get_variants_flat`**

In `python/genvarloader/_dataset/_flat_variants.py`, change the final `return _FlatVariants(fields)` to:

```python
    result = _FlatVariants(fields)
    if haps.dummy_variant is not None:
        result = result.fill_empty_groups(haps.dummy_variant)
    return result
```

(The fill runs AFTER AF/exonic compaction — i.e. a group emptied by filtering is also padded — because it is the last step.)

- [ ] **Step 5: Run to verify pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py::test_get_variants_flat_fills_empty_groups -v`
Expected: PASS. Then ruff on the two modified files.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_flat_variants.py tests/unit/dataset/test_flat_variants_type.py
rtk git commit -m "feat(flat): Haps.dummy_variant + apply empty-group fill in get_variants_flat"
```

---

## Task 4: Public API — `with_settings(dummy_variant=...)` + export + guard

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (`with_settings`; `__getitem__` guard)
- Modify: `python/genvarloader/__init__.py` (export `DummyVariant`)
- Test: `tests/dataset/test_output_format.py` (append)

- [ ] **Step 1: Write the failing tests (append)**

```python
# append to tests/dataset/test_output_format.py
import pytest


def test_with_settings_dummy_variant_sets_field(snap_dataset):
    import genvarloader as gvl

    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    dv = gvl.DummyVariant(start=-1, alt=b"N", ref=b"N")
    ds2 = ds.with_settings(dummy_variant=dv)
    assert ds2._seqs.dummy_variant == dv
    # original unchanged
    assert ds._seqs.dummy_variant is None
    # disable with False
    ds3 = ds2.with_settings(dummy_variant=False)
    assert ds3._seqs.dummy_variant is None


def test_dummy_variant_rejected_on_non_variant_kind(snap_dataset):
    import genvarloader as gvl

    ds = snap_dataset.with_seqs("haplotypes").with_settings(
        dummy_variant=gvl.DummyVariant()
    )
    # guard fires at access time when the output kind is not variants
    with pytest.raises(ValueError):
        _ = ds[0]


def test_dummy_variant_export():
    import genvarloader as gvl

    from genvarloader._dataset._flat_variants import DummyVariant

    assert gvl.DummyVariant is DummyVariant
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_output_format.py -k dummy -v`
Expected: FAIL (`with_settings` has no `dummy_variant` kwarg; `gvl.DummyVariant` missing).

- [ ] **Step 3: Add the `dummy_variant` parameter to `with_settings`**

In `python/genvarloader/_dataset/_impl.py`, add to the `with_settings` signature (after `var_filter`):

```python
        dummy_variant: "DummyVariant | Literal[False] | None" = None,
```

Add a runtime import at the top of `_impl.py` (no cycle — `_flat_variants` imports only `_flat` at runtime):

```python
from ._flat_variants import DummyVariant
```

Add a handling block inside `with_settings` (alongside the `min_af`/`var_filter` blocks, before the `if "_seqs" in to_evolve ...` rebuild):

```python
        if dummy_variant is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "dummy_variant requires a dataset with variants/genotypes."
                )
            dv = None if dummy_variant is False else dummy_variant
            haps = to_evolve.get("_seqs", self._seqs)
            to_evolve["_seqs"] = replace(haps, dummy_variant=dv)
```

Add a `dummy_variant` entry to the `with_settings` docstring Parameters section:

```
        dummy_variant
            A :class:`DummyVariant` to insert into empty (region, sample, ploid) variant
            groups so every group has at least one variant. Only valid for the variants
            output (:meth:`with_seqs("variants") <genvarloader.Dataset.with_seqs>`). Pass
            :code:`False` to disable.
```

- [ ] **Step 4: Add the guard in `__getitem__`**

Find `Dataset.__getitem__` in `_impl.py`. Near its top (before constructing `QueryView`), add:

```python
        if (
            isinstance(self._seqs, Haps)
            and self._seqs.dummy_variant is not None
            and self._seqs_kind != "variants"
        ):
            raise ValueError(
                "dummy_variant is only valid for the variants output; "
                "call with_seqs('variants') (got output kind "
                f"{self._seqs_kind!r})."
            )
```

(Validating here — rather than in `with_settings` — keeps `dummy_variant` order-independent with `with_seqs`, and catches switching away from variants after setting it.)

- [ ] **Step 5: Export `DummyVariant`**

In `python/genvarloader/__init__.py`, add the import grouped with the other `_dataset._flat_variants` imports:

```python
from ._dataset._flat_variants import _FlatAlleles as FlatAlleles
from ._dataset._flat_variants import _FlatVariants as FlatVariants
from ._dataset._flat_variants import DummyVariant
```

Add `"DummyVariant",` to `__all__` in its correct alphabetical position (between `"Dataset"`/`"DatasetWithSites"` and `"FlankSample"`).

- [ ] **Step 6: Run to verify pass**

Run: `pixi run -e dev pytest tests/dataset/test_output_format.py -k dummy -v`
Expected: PASS. Then:
Run: `pixi run -e dev python -c "import genvarloader as gvl; print(gvl.DummyVariant())"`
Run: `pixi run -e dev ruff check python/genvarloader/_dataset/_impl.py python/genvarloader/__init__.py tests/dataset/test_output_format.py`

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py python/genvarloader/__init__.py tests/dataset/test_output_format.py
rtk git commit -m "feat(flat): with_settings(dummy_variant=...) + export DummyVariant + non-variant guard"
```

---

## Task 5: Flat↔ragged parity + no-awkward guard for the dummy fill

**Files:**
- Modify: `tests/dataset/test_flat_mode_equivalence.py` (append)
- Modify: `tests/dataset/test_no_awkward_in_hotpath.py` (append)

- [ ] **Step 1: Write the parity test (append to `test_flat_mode_equivalence.py`)**

```python
# append to tests/dataset/test_flat_mode_equivalence.py
import genvarloader as gvl


@pytest.mark.parametrize("idx", IDX)
def test_b_dummy_fill_flat_to_ragged_matches_ragged(snap_dataset, idx):
    dv = gvl.DummyVariant(start=-1, alt=b"N", ref=b"N", ilen=0)
    ds = snap_dataset.with_seqs("variants").with_tracks(False).with_settings(dummy_variant=dv)
    ragged = ds[idx]                                   # ragged mode (now flat decode + to_ragged)
    rewrapped = ds.with_output_format("flat")[idx].to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)


def test_b_dummy_fill_no_empty_groups(snap_dataset):
    import awkward as ak

    dv = gvl.DummyVariant(start=-1, alt=b"N", ref=b"N")
    ds = snap_dataset.with_seqs("variants").with_tracks(False).with_settings(dummy_variant=dv)
    idx = (np.arange(min(6, snap_dataset.shape[0])),)
    rv = ds[idx]
    for ploid_groups in ak.to_list(rv["start"]):
        for g in ploid_groups:
            assert len(g) >= 1
```

(`_rv_to_lists` and `IDX` already exist in this file from A.)

- [ ] **Step 2: Run to verify pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat_mode_equivalence.py -k "dummy or b_" -v`
Expected: PASS for all index shapes (scalar, scalar-scalar squeeze, 1-D, 2-D from `IDX`), exercising the fill in both flat and ragged modes byte-identically.

- [ ] **Step 3: Write the no-awkward test (append to `test_no_awkward_in_hotpath.py`)**

Reuse the file's existing `_install_ak_counters(monkeypatch)` helper and `guard_dataset` fixture (inspect the file to confirm names):

```python
# append to tests/dataset/test_no_awkward_in_hotpath.py
def test_flat_variants_dummy_fill_has_no_awkward(monkeypatch, guard_dataset):
    import genvarloader as gvl

    calls = _install_ak_counters(monkeypatch)
    ds = (
        guard_dataset.with_seqs("variants")
        .with_tracks(False)
        .with_output_format("flat")
        .with_settings(dummy_variant=gvl.DummyVariant(start=-1, alt=b"N", ref=b"N"))
    )
    n_regions = min(4, ds.shape[0])
    regions = list(range(n_regions))
    samples = [i % ds.shape[1] for i in range(n_regions)]
    _ = ds[regions, samples]
    assert calls["n"] == 0, (
        f"flat dummy-fill decode dispatched {calls['n']} awkward kernel(s)"
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/dataset/test_no_awkward_in_hotpath.py -k dummy -v`
Expected: PASS (the fill is pure numba/numpy). Then the whole no-awkward file:
Run: `pixi run -e dev pytest tests/dataset/test_no_awkward_in_hotpath.py -v`
Expected: PASS.

- [ ] **Step 5: Full-suite regression**

Run: `pixi run -e dev pytest tests/ -q`
Expected: PASS (snapshot `variants_ragged.npz` still unchanged).

- [ ] **Step 6: Commit**

```bash
rtk git add tests/dataset/test_flat_mode_equivalence.py tests/dataset/test_no_awkward_in_hotpath.py
rtk git commit -m "test(flat): dummy-fill flat<->ragged parity + no-awkward guard"
```

---

## Task 6: Update the genvarloader skill

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Document the new API**

Read `skills/genvarloader/SKILL.md` and add, matching its existing conventions:
- A `with_settings(dummy_variant=...)` entry: inserts a dummy variant into empty `(region, sample, ploid)` groups so every group has ≥1 variant; only valid for the variants output (raises otherwise); `False` disables; default `None`. Note it fills **only** empty groups (not every group).
- `gvl.DummyVariant` in the public-types / symbol list with its fields (`start=-1`, `ref=b"N"`, `alt=b"N"`, `ilen=0`, `dosage=0.0`, `info={}`) and per-field defaults, and a pointer to `python/genvarloader/_dataset/_flat_variants.py`.
- A "Common gotchas" note: dummy padding is variant-output-only; a non-`N` dummy allele is reverse-complemented on negative-strand regions like any allele (the default `b"N"` is rc-invariant).

- [ ] **Step 2: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): document with_settings(dummy_variant) + DummyVariant"
```

---

## Self-Review

**Spec coverage:**
- §1 B1 always-flat decode (retire `_get_variants`) → Task 1.
- §1 B2 empty-group fill; §2 fill-only-empty semantics → Tasks 2, 3.
- §3 public API (`with_settings(dummy_variant=...)`, `DummyVariant`, defaults, `False` disables, raise on non-variant kind) → Task 4.
- §4 boundary `to_ragged` for `_FlatVariants`; snapshot oracle → Task 1.
- §5 kernels + `fill_empty_groups`, applied after filters → Tasks 2, 3.
- §6 data flow / rc interaction (dummy before rc; `b"N"` rc-invariant) → Tasks 3, 6 (documented).
- §7 error handling (raise on non-variant kind, bad info key handled by `scalar_for` defaults) → Task 4.
- §8 testing (snapshot regression, flat↔ragged parity, fill golden units, no-awkward, validation) → Tasks 1, 2, 3, 4, 5.
- §9 deferred merge/sort → out of scope (no task), rationale in spec.
- §10 gvf consumer thinning → cross-repo, not in this gvl plan.
- Skill update (CLAUDE.md mandate) → Task 6.

**Placeholder scan:** No TBD/TODO. Kernel and `DummyVariant` code is complete. The one verification note (Task 3 `ds._seqs` attribute / fixture visibility) names the concrete fallback, not a placeholder.

**Type consistency:** `DummyVariant(start, ilen, dosage, ref, alt, info)` + `scalar_for(name, dtype)`; `_fill_empty_scalar(data, offsets, fill) -> (data, offsets)`; `_fill_empty_seq(data, var_offsets, seq_offsets, dummy) -> (data, var_offsets, seq_offsets)`; `_FlatVariants.fill_empty_groups(dummy) -> _FlatVariants`; `Haps.dummy_variant`; `with_settings(dummy_variant=...)`; `gvl.DummyVariant`. Names are used consistently across tasks. `get_variants_flat(haps, idx)` signature unchanged (reads `haps.dummy_variant`).

**Known verification points (forced by tests):** snapshot `variants_ragged.npz` unchanged after retiring the awkward path (Task 1); flat↔ragged parity of the fill across scalar/scalar-scalar/1-D/2-D indices + empty/ploidy (Task 5); fill-only-empty (non-empty groups untouched) (Tasks 2, 3); no-awkward in the fill path (Task 5); raise on non-variant kind (Task 4).
