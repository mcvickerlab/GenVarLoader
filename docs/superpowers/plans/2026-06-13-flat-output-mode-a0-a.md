# Flat output mode (A0 + A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flat-buffer output mode to gvl so `dataset.with_output_format("flat")[idx]` returns pure-numpy `(data, offsets, shape)` containers with zero awkward on the hot path, byte-identical (when re-wrapped) to today's output — for the seqs/haps/annotated-haps/reference outputs (A0) and the variants output (A).

**Architecture:** A0 is a pure boundary change in `_dataset/_query.py`: the seqs/haps/annot/ref reconstructors already return `_Flat`/`_FlatAnnotatedHaps`, so flat mode skips the `to_ragged()` conversion (and the `output_length` pad/`to_fixed` massaging) at the return boundary. A introduces a new `_FlatVariants`/`_FlatAlleles` type and a numba reimplementation of the variant decode (`_get_variants_flat`) that produces those buffers directly from the sparse genotype + variant store with no awkward; a `flat` flag is threaded from the public `Dataset` through `QueryView` into the reconstructor so the variant path returns `_FlatVariants` in flat mode (ragged mode is left completely unchanged). The byte-identity equivalence test (`flat.to_ragged() == dataset[idx]`) is the correctness gate that also catches any reshape/squeeze/rc_neg interaction.

**Tech Stack:** Python, numba (`@nb.njit`), numpy, awkward (only for `to_ragged()` rebuild + equivalence reference), seqpro `Ragged`, pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-06-13-flat-output-mode-design.md`

**Run tests with:** `pixi run -e dev pytest <path> -v` (generate test data once with `pixi run -e dev gen` if not already present).

---

## File Structure

- **Create** `python/genvarloader/_dataset/_flat_variants.py` — `_FlatAlleles`, `_FlatVariants` dataclasses (with `to_ragged()`, `reshape`, `squeeze`, `reverse_masked`) + the numba gather kernels (`_gather_v_idxs`, `_gather_alleles`, `_compact_keep`) and the `get_variants_flat(...)` builder.
- **Modify** `python/genvarloader/_flat.py` — nothing structural; `_FlatVariants` lives in its own module to avoid bloating `_flat.py`, but re-export `_FlatAlleles`/`_FlatVariants` here for a single import surface if convenient. (Optional; default: import from `_flat_variants`.)
- **Modify** `python/genvarloader/_dataset/_haps.py` — add a `flat: bool = False` parameter to `Haps.__call__` / `Haps.get_haps_and_shifts`; when `flat` and `kind is RaggedVariants`, call the new `get_variants_flat`.
- **Modify** `python/genvarloader/_dataset/_reconstruct.py` (and any other `Reconstructor` implementors — `Ref`, `Tracks`, `HapsTracks`) — add/accept `flat: bool = False` in `__call__` and thread it to the `Haps` sub-component; ignore elsewhere.
- **Modify** `python/genvarloader/_dataset/_query.py` — add `flat_output: bool` to `QueryView`; thread `flat=view.flat_output` into the `view.recon(...)` calls; add `_FlatVariants` to `reverse_complement_ragged`; in `getitem`, when `flat_output`, skip the `output_length` massaging + `to_ragged()` and return the public flat wrappers.
- **Modify** `python/genvarloader/_dataset/_impl.py` — add the `output_format` field (default `"ragged"`) to the `Dataset` frozen dataclass, the `with_output_format` method, and pass `flat_output` into `QueryView` in `__getitem__`.
- **Modify** `python/genvarloader/__init__.py` — export `FlatRagged`, `FlatAnnotatedHaps`, `FlatVariants`, `FlatAlleles`.
- **Create** `tests/dataset/test_flat_mode_equivalence.py` — the byte-identity gate.
- **Create** `tests/unit/dataset/test_flat_variants_type.py` — unit tests for `_FlatVariants.to_ragged()` round-trip.
- **Modify** `tests/dataset/test_no_awkward_in_hotpath.py` — add the flat-variants-mode no-awkward assertion.
- **Modify** `skills/genvarloader/SKILL.md` — document `with_output_format` + flat types.

---

## Task 1: `_FlatAlleles` + `_FlatVariants` types with `to_ragged()`

**Files:**
- Create: `python/genvarloader/_dataset/_flat_variants.py`
- Test: `tests/unit/dataset/test_flat_variants_type.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dataset/test_flat_variants_type.py
from __future__ import annotations

import awkward as ak
import numpy as np
from seqpro.rag import Ragged

from genvarloader import RaggedVariants
from genvarloader._dataset._flat_variants import _FlatAlleles, _FlatVariants


def _alleles(rows, group_off):
    """rows: list[bytes] per variant; group_off: per-(b*p)-row variant boundaries."""
    data = np.frombuffer(b"".join(rows), np.uint8).copy()
    seq_off = np.concatenate([[0], np.cumsum([len(r) for r in rows])]).astype(np.int64)
    return _FlatAlleles(
        byte_data=data,
        seq_offsets=seq_off,
        var_offsets=np.asarray(group_off, np.int64),
        shape=(len(group_off) - 1, None),
    )


def test_flat_variants_to_ragged_matches_handbuilt():
    # b*p = 2 rows: row0 has 2 variants, row1 has 1 variant
    group_off = [0, 2, 3]
    alt = _alleles([b"ACG", b"T", b"GG"], group_off)
    ref = _alleles([b"A", b"CC", b"T"], group_off)
    start = _FlatAlleles  # placeholder to avoid unused import lint; replaced below
    from genvarloader._flat import _Flat

    start = _Flat.from_offsets(
        np.array([1, 5, 9], np.int32), (2, None), np.asarray(group_off, np.int64)
    )
    fv = _FlatVariants(fields={"alt": alt, "start": start, "ref": ref})
    rv = fv.to_ragged()

    assert isinstance(rv, RaggedVariants)
    assert ak.to_list(rv["alt"]) == [[b"ACG", b"T"], [b"GG"]]
    assert ak.to_list(rv["ref"]) == [[b"A", b"CC"], [b"T"]]
    assert ak.to_list(rv["start"]) == [[1, 5], [9]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'genvarloader._dataset._flat_variants'`.

- [ ] **Step 3: Write minimal implementation**

```python
# python/genvarloader/_dataset/_flat_variants.py
"""Flat-buffer analog of RaggedVariants: pure-numpy (data, offsets) per field,
no awkward on the hot path. Converts to RaggedVariants only via to_ragged()."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numba as nb
import numpy as np
from numpy.typing import NDArray

from .._flat import _Flat


@dataclass(slots=True)
class _FlatAlleles:
    """Two-level flat bytestring for an alt/ref allele field, shape (b, p, ~v, ~l).

    Layout matches genvarformer's `_bpvl`/`_decompose_bytestring` (inner-before-outer):
    - byte_data:   uint8 contiguous allele bytes
    - seq_offsets: per-variant byte boundaries, len n_variants + 1
    - var_offsets: per-(b*p)-row variant boundaries, len b*p + 1
    - shape:       outer fixed dims with exactly one None (the ragged variant axis)
    """

    byte_data: NDArray[np.uint8]
    seq_offsets: NDArray[np.int64]
    var_offsets: NDArray[np.int64]
    shape: tuple[int | None, ...]

    @property
    def ploidy(self) -> int:
        # shape is (b, p, None) for variants; ploidy is the last fixed dim.
        fixed = [d for d in self.shape if d is not None]
        return fixed[-1] if len(fixed) >= 2 else 1

    def to_ragged(self):
        from ._haps import _build_allele_layout

        return _build_allele_layout(
            np.ascontiguousarray(self.byte_data).view(np.uint8),
            np.asarray(self.seq_offsets, np.int64),
            np.asarray(self.var_offsets, np.int64),
            self.ploidy,
        )

    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatAlleles":
        """DNA reverse-complement the mask-selected (b*p) rows' alleles, in place.
        ``mask`` is one entry per (b*p) row; broadcast across that row's variants."""
        from .._ragged import _COMP, reverse_complement_masked
        from seqpro.rag import Ragged

        m = np.ascontiguousarray(mask, np.bool_).reshape(-1)
        # per-allele mask: repeat each row's flag across its variant count
        per_allele = np.repeat(m, np.diff(self.var_offsets))
        view = Ragged.from_offsets(
            self.byte_data.view("S1"),
            (per_allele.size, None),
            np.asarray(self.seq_offsets, np.int64),
        )
        reverse_complement_masked(view, per_allele)  # mutates byte_data in place
        return self

    def reshape(self, shape: tuple[int | None, ...]) -> "_FlatAlleles":
        return _FlatAlleles(self.byte_data, self.seq_offsets, self.var_offsets, shape)


@dataclass(slots=True)
class _FlatVariants:
    """Flat analog of RaggedVariants. `fields` maps field name -> _Flat (scalar
    fields: start/ilen/dosage/info) or _FlatAlleles (alt/ref)."""

    fields: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.fields["start"].shape

    def to_ragged(self):
        from ._rag_variants import RaggedVariants

        kw = {}
        for name, f in self.fields.items():
            kw[name] = f.to_ragged()
        return RaggedVariants(**kw)

    def reshape(self, shape) -> "_FlatVariants":
        return _FlatVariants({k: v.reshape(shape) for k, v in self.fields.items()})

    def squeeze(self, axis: int | None = None) -> "_FlatVariants":
        return _FlatVariants(
            {k: v.squeeze(axis) for k, v in self.fields.items()}
        )

    def reverse_masked(self, mask: NDArray[np.bool_]) -> "_FlatVariants":
        # Only alt/ref alleles are reverse-complemented; scalar fields unchanged
        # (matches RaggedVariants.rc_ which only touches alt/ref).
        for name in ("alt", "ref"):
            if name in self.fields:
                self.fields[name] = self.fields[name].reverse_masked(mask)
        return self
```

Note: `_Flat` already has `.reshape`/`.squeeze`/`.to_ragged`; `_FlatVariants.to_ragged` calls them for scalar fields. `start`/`ilen`/`info` `_Flat`s must carry shape `(b, p, None)` so `.to_ragged()` yields `(b, p, ~v)`. `_FlatVariants.squeeze` delegates to each field — `_Flat.squeeze(0)` drops the leading axis (matches `RaggedVariants.squeeze` → `self[0]`); verify in the equivalence test.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /root/GenVarLoader
rtk git add python/genvarloader/_dataset/_flat_variants.py tests/unit/dataset/test_flat_variants_type.py
rtk git commit -m "feat(flat): _FlatVariants/_FlatAlleles types with to_ragged()"
```

---

## Task 2: Public exports (`FlatRagged`, `FlatAnnotatedHaps`, `FlatVariants`, `FlatAlleles`)

**Files:**
- Modify: `python/genvarloader/__init__.py:18-61`
- Test: `tests/unit/dataset/test_flat_variants_type.py` (append)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_public_flat_exports():
    import genvarloader as gvl

    assert gvl.FlatRagged is not None
    assert gvl.FlatAnnotatedHaps is not None
    assert gvl.FlatVariants is not None
    assert gvl.FlatAlleles is not None
    # aliases point at the existing internals
    from genvarloader._flat import _Flat, _FlatAnnotatedHaps
    from genvarloader._dataset._flat_variants import _FlatVariants, _FlatAlleles

    assert gvl.FlatRagged is _Flat
    assert gvl.FlatAnnotatedHaps is _FlatAnnotatedHaps
    assert gvl.FlatVariants is _FlatVariants
    assert gvl.FlatAlleles is _FlatAlleles
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py::test_public_flat_exports -v`
Expected: FAIL with `AttributeError: module 'genvarloader' has no attribute 'FlatRagged'`.

- [ ] **Step 3: Write minimal implementation**

In `python/genvarloader/__init__.py`, after the existing imports (around line 27) add:

```python
from ._flat import _Flat as FlatRagged
from ._flat import _FlatAnnotatedHaps as FlatAnnotatedHaps
from ._dataset._flat_variants import _FlatVariants as FlatVariants
from ._dataset._flat_variants import _FlatAlleles as FlatAlleles
```

And add the four names to `__all__` (keep alphabetical ordering used in the file):

```python
    "FlatAlleles",
    "FlatAnnotatedHaps",
    "FlatRagged",
    "FlatVariants",
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_flat_variants_type.py::test_public_flat_exports -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/__init__.py tests/unit/dataset/test_flat_variants_type.py
rtk git commit -m "feat(flat): export FlatRagged/FlatAnnotatedHaps/FlatVariants/FlatAlleles"
```

---

## Task 3: `output_format` field + `with_output_format` + thread into `QueryView`

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (add field near line 653; add method near the other `with_*` methods ~line 485; modify `__getitem__` ~line 1588)
- Modify: `python/genvarloader/_dataset/_query.py` (add `flat_output` to `QueryView` ~line 50)
- Test: `tests/unit/dataset/test_with_methods.py` (append) or a new `tests/unit/dataset/test_output_format.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/dataset/test_output_format.py
import pytest


def test_with_output_format_sets_field(snap_dataset):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    assert ds.output_format == "ragged"
    flat = ds.with_output_format("flat")
    assert flat.output_format == "flat"
    # original is unchanged (frozen dataclass / replace semantics)
    assert ds.output_format == "ragged"


def test_with_output_format_rejects_bad_value(snap_dataset):
    with pytest.raises(ValueError):
        snap_dataset.with_output_format("nope")
```

Add a `conftest.py`-visible `snap_dataset` or import the fixture. The `snap_dataset` fixture lives in `tests/dataset/test_flat_getitem_snapshot.py`; move it to `tests/dataset/conftest.py` (or `tests/conftest.py`) so multiple test modules can use it. **Do this move as the first sub-step** (cut the `snap_dataset` fixture + its imports into `tests/dataset/conftest.py`, leave the rest of the snapshot test importing it implicitly via pytest fixture resolution).

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_output_format.py -v`
Expected: FAIL with `AttributeError: 'Dataset' object has no attribute 'output_format'`.

- [ ] **Step 3: Write minimal implementation**

In `_impl.py`, add the field to the `Dataset` frozen dataclass alongside `output_length` (after line 656):

```python
    output_format: Literal["ragged", "flat"] = "ragged"
    """Container format for eager indexing. ``"ragged"`` (default) returns awkward-backed
    seqpro ``Ragged`` / ``RaggedVariants``; ``"flat"`` returns pure-numpy ``FlatRagged`` /
    ``FlatVariants`` with zero awkward on the hot path. See ``with_output_format``."""
```

> If the dataclass has other non-default fields declared *after* this point, give `output_format` no default and instead set it in `open()`/`replace` defaults. Check field ordering: dataclass fields with defaults cannot precede fields without. The simplest safe move is to add `output_format` as the LAST declared field with a default, after `_rng` (line 700). Place it there.

Add the method near the other `with_*` methods (e.g. after `with_len`, ~line 483):

```python
    def with_output_format(self, fmt: Literal["ragged", "flat"]) -> "Dataset":
        """Return a copy that yields ``fmt`` containers from eager indexing.

        Parameters
        ----------
        fmt
            ``"ragged"`` for awkward-backed ``Ragged``/``RaggedVariants`` (default),
            or ``"flat"`` for pure-numpy ``FlatRagged``/``FlatVariants``.
        """
        if fmt not in ("ragged", "flat"):
            raise ValueError(f"output_format must be 'ragged' or 'flat', got {fmt!r}.")
        return replace(self, output_format=fmt)
```

In `__getitem__` (~line 1588) pass it into `QueryView`:

```python
        view = QueryView(
            idxer=self._idxer,
            sp_idxer=self._sp_idxer,
            full_regions=self._full_regions,
            rng=self._rng,
            recon=self._recon,
            output_length=self.output_length,
            jitter=self.jitter,
            deterministic=self.deterministic,
            rc_neg=self.rc_neg,
            flat_output=self.output_format == "flat",
        )
```

In `_query.py`, add to the `QueryView` dataclass (after `rc_neg: bool`, ~line 53):

```python
    flat_output: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_output_format.py -v`
Expected: PASS.

- [ ] **Step 5: Run the existing suite to confirm no regression**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -v`
Expected: PASS (the fixture move didn't break it; behavior unchanged since `flat_output` is unused so far).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_query.py tests/unit/dataset/test_output_format.py tests/dataset/conftest.py tests/dataset/test_flat_getitem_snapshot.py
rtk git commit -m "feat(flat): output_format field + with_output_format + QueryView.flat_output"
```

---

## Task 4: A0 — flat passthrough for non-variant outputs (boundary change)

**Files:**
- Modify: `python/genvarloader/_dataset/_query.py` — `getitem` (~lines 92-122)
- Create: `tests/dataset/test_flat_mode_equivalence.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/dataset/test_flat_mode_equivalence.py
"""flat-mode output, re-wrapped via .to_ragged(), must be byte-identical to ragged mode."""
from __future__ import annotations

import numpy as np
import pytest
from seqpro.rag import Ragged

from genvarloader._ragged import RaggedAnnotatedHaps

IDX = [0, (np.array([0, 1, 2]),)]  # scalar-ish and a list index


def _to_plain(obj):
    """Normalize a ragged/annot/flat object into dict of ndarrays for comparison."""
    if isinstance(obj, RaggedAnnotatedHaps):
        return {
            "haps": np.asarray(obj.haps.data), "haps_off": np.asarray(obj.haps.offsets),
            "vidx": np.asarray(obj.var_idxs.data), "pos": np.asarray(obj.ref_coords.data),
        }
    if isinstance(obj, Ragged):
        return {"data": np.asarray(obj.data), "off": np.asarray(obj.offsets)}
    raise TypeError(type(obj))


@pytest.mark.parametrize("seqs", ["haplotypes", "reference", "annotated"])
@pytest.mark.parametrize("idx", IDX)
def test_a0_flat_to_ragged_matches_ragged(snap_dataset, seqs, idx):
    ds = snap_dataset.with_seqs(seqs).with_tracks(False)
    ragged = ds[idx]
    flat = ds.with_output_format("flat")[idx]
    rewrapped = flat.to_ragged()
    r, f = _to_plain(ragged), _to_plain(rewrapped)
    assert r.keys() == f.keys()
    for k in r:
        np.testing.assert_array_equal(r[k], f[k], err_msg=f"field {k}")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_mode_equivalence.py -v`
Expected: FAIL — in flat mode the boundary still calls `to_ragged()`, so `flat` is already a `Ragged` and `.to_ragged()` raises `AttributeError`, OR (after we wire it) the wrapper type lacks `.to_ragged`. Confirm it errors before the fix.

- [ ] **Step 3: Write minimal implementation**

In `_query.py` `getitem`, gate the `output_length` massaging and the `to_ragged()` conversion on `flat_output`. Replace the block at lines ~92-110 with:

```python
    if not view.flat_output:
        if view.output_length == "variable":
            recon = tuple(
                r if isinstance(r, (RaggedVariants, RaggedIntervals)) else pad(r)
                for r in recon
            )
        elif isinstance(view.output_length, int):
            recon = tuple(
                r
                if isinstance(r, (RaggedVariants, RaggedIntervals))
                else r.to_fixed(view.output_length)
                for r in recon
            )
        # Convert any still-flat elements to their public Ragged types before
        # reshape/squeeze apply the existing logic.
        recon = tuple(
            o.to_ragged() if isinstance(o, (_Flat, _FlatAnnotatedHaps)) else o
            for o in recon
        )
    # flat_output: leave _Flat / _FlatAnnotatedHaps (and, after Task 5,
    # _FlatVariants) unconverted. reshape/squeeze below still apply via their
    # flat methods; the consumer densifies via .to_fixed/.to_padded if desired.
```

The subsequent `out_reshape` (line ~112) and `squeeze` (line ~115) blocks already call `.reshape`/`.squeeze`, which `_Flat`/`_FlatAnnotatedHaps` implement, so they work unchanged in flat mode.

`rc_neg` (in `_getitem_unspliced`, lines 158-160) already runs `reverse_complement_ragged` on the `_Flat`/`_FlatAnnotatedHaps` recon (see the overloads at `_query.py:322-344`), so negative-strand handling is already correct for A0.

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_flat_mode_equivalence.py -v`
Expected: PASS for all `seqs` in `{haplotypes, reference, annotated}`.

- [ ] **Step 5: Confirm ragged path is untouched**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -v`
Expected: PASS (byte-identical snapshots unchanged).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_query.py tests/dataset/test_flat_mode_equivalence.py
rtk git commit -m "feat(flat): A0 flat passthrough for seqs/haps/annot/reference outputs"
```

---

## Task 5: A — flat variant decode (`get_variants_flat`) + thread `flat` flag

This is the core task. It has several sub-steps; commit once at the end.

**Files:**
- Modify: `python/genvarloader/_dataset/_flat_variants.py` (add kernels + `get_variants_flat`)
- Modify: `python/genvarloader/_dataset/_haps.py` (`__call__` / `get_haps_and_shifts` gain `flat: bool = False`)
- Modify: `python/genvarloader/_dataset/_reconstruct.py` + other `Reconstructor` impls (thread `flat`)
- Modify: `python/genvarloader/_dataset/_query.py` (pass `flat=view.flat_output` to `view.recon(...)`; add `_FlatVariants` to `reverse_complement_ragged`; allow `_FlatVariants` to skip pad/to_ragged in flat mode)
- Test: `tests/dataset/test_flat_mode_equivalence.py` (extend with variants cases)

- [ ] **Step 1: Write the failing test (extend)**

```python
# append to tests/dataset/test_flat_mode_equivalence.py
import awkward as ak
from genvarloader import RaggedVariants


def _rv_to_lists(rv: RaggedVariants) -> dict:
    out = {"alt": ak.to_list(rv["alt"]), "start": ak.to_list(rv["start"])}
    for f in ("ref", "ilen", "dosage"):
        if f in rv.fields:
            out[f] = ak.to_list(rv[f])
    return out


@pytest.mark.parametrize("idx", IDX)
def test_a_flat_variants_to_ragged_matches_ragged(snap_dataset, idx):
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    ragged = ds[idx]                                  # RaggedVariants (current path)
    flat = ds.with_output_format("flat")[idx]         # _FlatVariants
    rewrapped = flat.to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)


def test_a_flat_variants_empty_region_and_ploidy(snap_dataset):
    # exercise an index range likely to include empty variant groups; the
    # equivalence must still hold element-for-element.
    ds = snap_dataset.with_seqs("variants").with_tracks(False)
    idx = (np.arange(min(6, snap_dataset.shape[0])),)
    ragged = ds[idx]
    rewrapped = ds.with_output_format("flat")[idx].to_ragged()
    assert _rv_to_lists(rewrapped) == _rv_to_lists(ragged)
```

> If the dataset supports AF / exonic filters, add a parametrization that applies `with_settings(min_af=...)` / the exonic filter and re-asserts equivalence. Mirror whatever the existing variant tests use to enable those (search `tests/integration/dataset/test_query_filters.py`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_mode_equivalence.py -k variants -v`
Expected: FAIL — flat mode still returns a `RaggedVariants` from the reconstructor (no `flat` plumbing yet), so `.to_ragged()` errors, or the variant path isn't flat.

- [ ] **Step 3a: Add the numba gather kernels + builder to `_flat_variants.py`**

```python
# add to python/genvarloader/_dataset/_flat_variants.py

@nb.njit(nogil=True, cache=True)
def _gather_v_idxs(geno_offset_idx, geno_offsets, geno_v_idxs):
    """Concatenate the per-(b*p)-row sparse variant-index slices into one flat
    buffer + per-row offsets. Replaces `genotypes[r, s].to_packed()`."""
    n_rows = geno_offset_idx.shape[0]
    out_offsets = np.empty(n_rows + 1, np.int64)
    out_offsets[0] = 0
    for i in range(n_rows):
        goi = geno_offset_idx[i]
        out_offsets[i + 1] = out_offsets[i] + (geno_offsets[goi + 1] - geno_offsets[goi])
    total = out_offsets[n_rows]
    v_idxs = np.empty(total, geno_v_idxs.dtype)
    dst = 0
    for i in range(n_rows):
        goi = geno_offset_idx[i]
        s = geno_offsets[goi]
        e = geno_offsets[goi + 1]
        for k in range(s, e):
            v_idxs[dst] = geno_v_idxs[k]
            dst += 1
    return v_idxs, out_offsets


@nb.njit(nogil=True, cache=True)
def _gather_alleles(v_idxs, allele_bytes, allele_offsets):
    """Gather selected variants' allele bytes into a contiguous buffer + per-variant
    byte offsets. Replaces `variants.<kind>[v_idxs].to_packed()`."""
    n = v_idxs.shape[0]
    seq_offsets = np.empty(n + 1, np.int64)
    seq_offsets[0] = 0
    for i in range(n):
        v = v_idxs[i]
        seq_offsets[i + 1] = seq_offsets[i] + (allele_offsets[v + 1] - allele_offsets[v])
    data = np.empty(seq_offsets[n], np.uint8)
    dst = 0
    for i in range(n):
        v = v_idxs[i]
        s = allele_offsets[v]
        e = allele_offsets[v + 1]
        for k in range(s, e):
            data[dst] = allele_bytes[k]
            dst += 1
    return data, seq_offsets


@nb.njit(nogil=True, cache=True)
def _compact_keep(v_idxs, row_offsets, keep):
    """Drop masked variants and recompute per-row offsets. `keep` is per-variant
    (aligned to v_idxs)."""
    n_rows = row_offsets.shape[0] - 1
    new_offsets = np.empty(n_rows + 1, np.int64)
    new_offsets[0] = 0
    n_keep = 0
    for i in range(n_rows):
        for j in range(row_offsets[i], row_offsets[i + 1]):
            if keep[j]:
                n_keep += 1
        new_offsets[i + 1] = n_keep
    new_v = np.empty(n_keep, v_idxs.dtype)
    dst = 0
    for j in range(v_idxs.shape[0]):
        if keep[j]:
            new_v[dst] = v_idxs[j]
            dst += 1
    return new_v, new_offsets


def get_variants_flat(haps, idx) -> "_FlatVariants":
    """Flat (no-awkward) reimplementation of Haps._get_variants. Mirrors the field
    set and filter semantics of _get_variants in _haps.py.

    `haps` is the Haps reconstructor (provides .genotypes, .variants, .dosages,
    .var_fields, .min_af, .max_af). `idx` is the flat (region*sample) index array.
    """
    from .._flat import _Flat
    from ._haps import Haps  # for _get_geno_offset_idx (static method)

    genotypes = haps.genotypes
    ploidy = int(genotypes.shape[-2])
    geno_offset_idx = Haps._get_geno_offset_idx(idx, genotypes)  # (b, p)
    goi_flat = np.ascontiguousarray(geno_offset_idx).reshape(-1)

    v_idxs, row_offsets = _gather_v_idxs(
        goi_flat,
        np.asarray(genotypes.offsets),
        np.asarray(genotypes.data),
    )

    # ---- filters (match _haps._get_variants) ----
    if haps.min_af is not None or haps.max_af is not None:
        afs = haps.variants.info["AF"][v_idxs]
        keep = np.ones(v_idxs.shape[0], np.bool_)
        if haps.min_af is not None:
            keep &= afs >= haps.min_af
        if haps.max_af is not None:
            keep &= afs <= haps.max_af
        v_idxs, row_offsets = _compact_keep(v_idxs, row_offsets, keep)
    # NOTE: exonic filter (haps.filter == "exonic") is applied UPSTREAM via
    # req.keep/req.keep_offsets in get_haps_and_shifts. For the variants kind the
    # current code path passes keep/keep_offsets into _get_variants only in
    # get_haps_and_shifts (not the bare __call__). Replicate: if keep is supplied,
    # compact with it here. See Step 3b for how keep is threaded.

    n_rows = row_offsets.shape[0]                       # b*p + 1
    b = goi_flat.shape[0] // ploidy
    shape = (b, ploidy, None)

    def _scalar(arr_full):
        data = np.ascontiguousarray(arr_full[v_idxs])
        return _Flat.from_offsets(data, shape, row_offsets)

    fields: dict = {}
    # alt is required
    alt_bytes, alt_seq_off = _gather_alleles(
        v_idxs,
        np.asarray(haps.variants.alt.data).view(np.uint8),
        np.asarray(haps.variants.alt.offsets),
    )
    fields["alt"] = _FlatAlleles(alt_bytes, alt_seq_off, row_offsets, shape)
    fields["start"] = _scalar(np.asarray(haps.variants.start))

    if "ref" in haps.var_fields:
        ref_bytes, ref_seq_off = _gather_alleles(
            v_idxs,
            np.asarray(haps.variants.ref.data).view(np.uint8),
            np.asarray(haps.variants.ref.offsets),
        )
        fields["ref"] = _FlatAlleles(ref_bytes, ref_seq_off, row_offsets, shape)
    if "ilen" in haps.var_fields:
        fields["ilen"] = _scalar(np.asarray(haps.variants.ilen))

    if haps.dosages is not None and "dosage" in haps.var_fields:
        # dosage is parallel to genotypes (gathered by the same offset ranges, not v_idxs)
        dos, _ = _gather_v_idxs(
            goi_flat, np.asarray(genotypes.offsets), np.asarray(haps.dosages.data)
        )
        # if AF/exonic filtered, dosage must be compacted with the same keep mask
        # — keep dosage gather BEFORE compaction OR carry keep; simplest: gather
        # dosage parallel to the UNfiltered v_idxs then apply the same _compact_keep.
        fields["dosage"] = _Flat.from_offsets(np.ascontiguousarray(dos), shape, row_offsets)

    # remaining info fields
    for k in haps.var_fields:
        if k in ("alt", "start", "ref", "ilen", "dosage"):
            continue
        fields[k] = _scalar(np.asarray(haps.variants.info[k]))

    return _FlatVariants(fields)
```

> **Dosage + filter ordering caveat:** the snippet gathers dosage AFTER computing the
> (possibly filtered) `row_offsets`, which is wrong if a filter compacted `v_idxs`.
> Fix during implementation: capture the UNfiltered `row_offsets` and the `keep` mask,
> gather dosage against the unfiltered offsets, then `_compact_keep` dosage with the same
> mask. The equivalence test with `min_af` set is the gate that forces this correct.

- [ ] **Step 3b: Thread `flat` through `Haps` and the reconstructors**

In `_haps.py` `Haps.__call__` (line 502), add `flat: bool = False`:

```python
    def __call__(self, idx, r_idx, regions, output_length, jitter, rng, deterministic,
                 splice_plan=None, flat: bool = False):
        if issubclass(self.kind, RaggedVariants):
            if splice_plan is not None:
                raise NotImplementedError("Spliced output is not supported for RaggedVariants.")
            if flat:
                from ._flat_variants import get_variants_flat
                return cast(_H, get_variants_flat(self, idx))
            ragv = self._get_variants(idx=idx, regions=None, shifts=None)
            return cast(_H, ragv)
        else:
            haps, *_ = self.get_haps_and_shifts(idx=idx, regions=regions,
                output_length=output_length, rng=rng, deterministic=deterministic,
                splice_plan=splice_plan)
            return haps
```

(For exonic filter parity, also branch in `get_haps_and_shifts` where `_get_variants` is called with `keep`/`keep_offsets`: when `flat`, call `get_variants_flat(self, idx, keep=req.keep, keep_offsets=req.keep_offsets)` and add those optional params to `get_variants_flat` applying `_compact_keep`. Only needed if the variants-kind path flows through `get_haps_and_shifts` — verify which call site the variants output uses; the bare `__call__` path is the primary one.)

Add `flat: bool = False` to the `Reconstructor` protocol `__call__` and to each implementor (`Ref`, `Tracks`, `HapsTracks`, etc.) in `_reconstruct.py`. Non-variant reconstructors accept and ignore it (they already return `_Flat`); `HapsTracks` forwards `flat` to its `Haps` sub-call.

In `_query.py` `_getitem_unspliced` (line 145) and `_getitem_spliced`, pass `flat=view.flat_output`:

```python
    recon = view.recon(
        idx=ds_idx, r_idx=r_idx, regions=regions, output_length=view.output_length,
        jitter=view.jitter, rng=view.rng, deterministic=view.deterministic,
        flat=view.flat_output,
    )
```

- [ ] **Step 3c: Teach `_query.py` to carry `_FlatVariants` through rc / boundary**

In `reverse_complement_ragged` (line 335), add a branch before `RaggedVariants`:

```python
    from ._flat_variants import _FlatVariants
    if isinstance(rag, _FlatVariants):
        return rag.reverse_masked(to_rc)
```

Add the overload signature too. In `getitem`, the flat branch (Task 4) already leaves non-`(_Flat, _FlatAnnotatedHaps)` objects alone — but `_FlatVariants` must also bypass `to_ragged()` in ragged mode? No: in **ragged** mode the variant recon returns `RaggedVariants` (flat=False), so nothing changes. In **flat** mode the recon returns `_FlatVariants` and the `if not view.flat_output:` guard skips conversion — so `_FlatVariants` flows straight out. `reshape`/`squeeze` apply via `_FlatVariants` methods. Confirm `squeeze` matches `RaggedVariants.squeeze` (→ `self[0]`): `_FlatVariants.squeeze(0)` delegates to `_Flat.squeeze(0)` / `_FlatAlleles` (add a `squeeze` to `_FlatAlleles` that drops the leading fixed axis). The equivalence test on a scalar `idx` (which triggers squeeze) is the gate.

- [ ] **Step 3d: Add `squeeze` to `_FlatAlleles`**

```python
    def squeeze(self, axis: int | None = None) -> "_FlatAlleles":
        fixed = [d for d in self.shape if d is not None]
        if axis is None:
            fixed = [d for d in fixed if d != 1]
        else:
            del fixed[axis]
        return _FlatAlleles(self.byte_data, self.seq_offsets, self.var_offsets,
                            (*fixed, None))
```

- [ ] **Step 4: Run the variant equivalence tests**

Run: `pixi run -e dev pytest tests/dataset/test_flat_mode_equivalence.py -v`
Expected: PASS for all cases (haps/ref/annot from Task 4 + variants, empty regions, ploidy, and any AF/exonic parametrization).

- [ ] **Step 5: Confirm ragged path untouched + full suite**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py tests/dataset/test_flat_variants.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_flat_variants.py python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_query.py tests/dataset/test_flat_mode_equivalence.py
rtk git commit -m "feat(flat): A flat variant decode (get_variants_flat) with no awkward"
```

---

## Task 6: No-awkward-in-hot-path guard for flat variants mode

**Files:**
- Modify: `tests/dataset/test_no_awkward_in_hotpath.py`

- [ ] **Step 1: Inspect the existing guard mechanism**

Run: `pixi run -e dev pytest tests/dataset/test_no_awkward_in_hotpath.py -v` and read the file to learn how it detects awkward calls (it likely patches `awkward.highlevel.Array.__getitem__` or uses a profiler/import guard).

- [ ] **Step 2: Write the failing test (add a case)**

Add a test that runs `ds.with_seqs("variants").with_tracks(False).with_output_format("flat")[idx]` under the same awkward-detection harness the file already uses, asserting **zero** `awkward.highlevel.__getitem__` calls during the decode. Reuse the existing fixture and detection helper — do not invent a new mechanism. Example shape (adapt to the file's actual helper name):

```python
def test_flat_variants_decode_has_no_awkward(snap_dataset, awkward_getitem_counter):
    ds = snap_dataset.with_seqs("variants").with_tracks(False).with_output_format("flat")
    with awkward_getitem_counter() as count:
        _ = ds[(np.arange(4),)]
    assert count.value == 0, f"awkward.__getitem__ called {count.value}x in flat variant decode"
```

- [ ] **Step 3: Run to verify it fails (or passes)**

Run: `pixi run -e dev pytest tests/dataset/test_no_awkward_in_hotpath.py -k flat -v`
Expected: PASS if `get_variants_flat` is genuinely awkward-free; FAIL pinpoints a residual awkward call (e.g. `genotypes.offsets`/`.data` returning an awkward-backed array — convert with `np.asarray` as the kernel wrappers already do). Fix any residual until it passes.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/dataset/test_no_awkward_in_hotpath.py
rtk git commit -m "test(flat): assert flat variant decode runs zero awkward getitem"
```

---

## Task 7: Update the genvarloader skill + docs

**Files:**
- Modify: `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Document the new API**

Add to the skill:
- `Dataset.with_output_format("ragged" | "flat")` — what it does, default `"ragged"`, that flat mode returns `FlatRagged`/`FlatVariants`/`FlatAnnotatedHaps`/`FlatAlleles` with `.to_ragged()`/`.to_fixed()`/`.to_padded()` escape hatches and zero awkward on the hot path.
- The new public types in the symbol list / "Where to look next" table.
- A one-line note: flat offsets are int64; cast to int32 for torch nested tensors.

- [ ] **Step 2: Verify the doc builds (if applicable) and commit**

Run: `pixi run -e docs doc` (only if docs reference the skill / API; otherwise skip).

```bash
rtk git add skills/genvarloader/SKILL.md
rtk git commit -m "docs(skill): document with_output_format + flat output types"
```

---

## Task 8 (cross-repo, genvarformer): thin-wrapper validation

> This task lives in the **genvarformer** repo (`/root/genvarformer`), not gvl. It validates the gvl change end-to-end and is tracked as the consumer half of issue #214 §8. Do it after Tasks 1-7 land and gvl is installed/linked into the genvarformer env. It gets its own commit/PR in that repo.

**Files (genvarformer):**
- Modify: `src/genvarformer/data/sources/_helpers.py` (`rag_to_nested`), `src/genvarformer/data/sources/tokens.py` (`RefSeq._call_one_to_many`, `Variants._fetch` step 1)

- [ ] **Step 1:** Switch the gvl dataset construction in genvarformer to `.with_output_format("flat")` for the sources that immediately tensorize.
- [ ] **Step 2:** In `rag_to_nested` / `RefSeq`, consume `FlatRagged` directly → `Nested` (`torch.from_numpy(flat.data)`, `torch.from_numpy(flat.offsets.astype(np.int32))`), dropping the `np.asarray(rag.data)` round-trip.
- [ ] **Step 3:** In `Variants._fetch`, take `FlatVariants` and feed its buffers into PR #19's merge/sort/dummy kernels without the `_decompose_bytestring` re-boxing (inputs are already flat).
- [ ] **Step 4:** Run genvarformer's existing batch-equality guardrail; assert outputs unchanged.
- [ ] **Step 5:** Re-profile the production data path; confirm `awkward.highlevel.__getitem__` is gone from the top cumulative functions and member-seqs/s improves.

---

## Self-Review

**Spec coverage:**
- §2 goal (flat mode, byte-identical) → Tasks 3-5 + equivalence test.
- §3 public API (`with_output_format`, `FlatRagged`/`FlatAnnotatedHaps`/`FlatVariants`/`FlatAlleles`) → Tasks 1-3.
- §4 boundary change → Tasks 3-5.
- §5 A flat variant decode (v_idx gather, scalar fields, allele gather, AF/exonic filters, dosage parallel-gather) → Task 5.
- §6 A0 passthrough (incl. deferred tracks/intervals — left ragged, not in scope) → Task 4. *(Tracks/intervals intentionally not flattened; documented as deferred in the spec.)*
- §7 equivalence + no-awkward gates → Tasks 4, 5, 6.
- §8 genvarformer thinning → Task 8.
- Skill update mandated by CLAUDE.md → Task 7.

**Placeholder scan:** The dosage/filter ordering and the exonic-keep threading are flagged as explicit implementation caveats with the equivalence test (`min_af` parametrization) as the forcing gate — not hand-waved; the fix is described. The no-awkward harness reuses the existing file's mechanism (Task 6 Step 1 inspects it first) rather than inventing one.

**Type consistency:** `_FlatVariants(fields=dict)`, `_FlatAlleles(byte_data, seq_offsets, var_offsets, shape)`, `get_variants_flat(haps, idx)`, `_gather_v_idxs`/`_gather_alleles`/`_compact_keep`, `with_output_format`, `QueryView.flat_output`, `Haps.__call__(..., flat=False)` are used consistently across tasks. `to_ragged()` is the conversion method on every flat type.

**Known verification points (forced by tests, not left open):** scalar-idx `squeeze` parity (`_FlatVariants.squeeze`/`_FlatAlleles.squeeze`), `rc_neg` on negative-strand variants (snapshot fixture has `rc_neg=True`), empty variant groups, multi-ploidy, AF/exonic filters.
