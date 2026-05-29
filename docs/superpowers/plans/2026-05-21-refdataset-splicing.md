# RefDataset Splicing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add transcript-level splicing to `gvl.RefDataset` and refactor `Dataset`'s splicing internals so both datasets share one sample-agnostic `SpliceMap` abstraction, while keeping `Dataset`'s public behavior bit-identical.

**Architecture:** Extract a sample-agnostic `SpliceMap` (and the post-fetch `_cat_length` concatenation helpers) into a new module `_dataset/_splice.py`. Refactor `SpliceIndexer` to compose `SpliceMap` + `DatasetIndexer`. Add `splice_info`, `_splice_map`, `_spliced_bed`, validation, properties, and a `_getitem_spliced` path to `RefDataset`. `RefDataset`'s spliced path reuses its own unspliced `__getitem__` via `evolve(self, _splice_map=None)`, exactly mirroring `Dataset`'s strategy of using `with_len("ragged")` + `_recon` to fetch the flat exon rows, then `_cat_length` to join.

**Tech Stack:** Python 3.10+, attrs, polars, numpy, awkward (`ak`), seqpro (`Ragged`), hirola (`HashTable`), pytest, pytest-cases.

---

## File Structure

**New file:**
- `python/genvarloader/_dataset/_splice.py` — owns `SpliceMap`, `_cat_length`, `_cat_length_inner`. Sample-agnostic.

**Modified:**
- `python/genvarloader/_dataset/_indexing.py` — `SpliceIndexer` becomes a thin composition of `SpliceMap` + `DatasetIndexer`.
- `python/genvarloader/_dataset/_impl.py` — drop `_parse_splice_info`/`_cat_length`/`_cat_length_inner`; update `Dataset.open`, `Dataset.with_settings`, `Dataset._getitem_spliced` to go through `SpliceMap.from_bed` and `splice_idxer.map`.
- `python/genvarloader/_dataset/_reference.py` — `RefDataset` gains `splice_info`, `_splice_map`, `_spliced_bed`, `is_spliced`, `spliced_regions`, validation, splice-aware `with_settings`/`subset_to`/`__getitem__`.
- `python/genvarloader/_dummy.py` — switch from `_parse_splice_info` to `SpliceMap.from_bed`.
- `tests/dataset/test_rc_packing.py` — update `_cat_length` import path.
- `skills/genvarloader/SKILL.md` — extend "Spliced haplotypes" section to mention `RefDataset(splice_info=...)`.
- `docs/source/splicing.ipynb` — add a brief `RefDataset` splicing example.

**New tests:**
- `tests/test_ref_ds_splicing.py` — `RefDataset` splice tests (kept separate from the broader `test_ref_ds.py` for clarity).

---

## Task 1: Create `_splice.py` skeleton with `SpliceMap` and move `_cat_length`

**Files:**
- Create: `python/genvarloader/_dataset/_splice.py`
- Modify: `python/genvarloader/_dataset/_impl.py` (delete moved helpers, add import)
- Modify: `tests/dataset/test_rc_packing.py` (update import)

- [ ] **Step 1: Run the existing splice tests to capture the baseline**

Run: `pixi run -e dev pytest tests/dataset/test_rc_packing.py -v`
Expected: PASS (all current splice + RC packing tests).

- [ ] **Step 2: Create `_splice.py` with imports and `_cat_length`/`_cat_length_inner` moved verbatim**

Create `python/genvarloader/_dataset/_splice.py`:

```python
from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast, overload

import awkward as ak
import numpy as np
import polars as pl
from attrs import define, evolve, field
from awkward.contents import ListOffsetArray, NumpyArray
from awkward.index import Index64
from hirola import HashTable
from numpy.typing import NDArray
from seqpro.rag import Ragged
from typing_extensions import Self, assert_never

from .._ragged import RaggedAnnotatedHaps
from .._types import Idx, StrIdx
from .._utils import is_rag_dtype, lengths_to_offsets
from ._indexing import s2i

DTYPE = TypeVar("DTYPE", bound=np.generic)


@overload
def _cat_length(rag: Ragged[DTYPE], offsets: NDArray[np.integer]) -> Ragged[DTYPE]: ...
@overload
def _cat_length(
    rag: RaggedAnnotatedHaps, offsets: NDArray[np.integer]
) -> RaggedAnnotatedHaps: ...
def _cat_length(
    rag: Ragged | RaggedAnnotatedHaps, offsets: NDArray[np.integer]
) -> Ragged | RaggedAnnotatedHaps:
    """Concatenate the lengths of the ragged data.

    ``offsets`` groups contiguous ranges along the outer (batch) axis. Every
    other fixed dimension (e.g. ploidy, tracks) is preserved; we merge
    bytes/elements across the grouped batch slots within each inner index.

    Branch on ``shape[:-1]`` (the fixed axes; ``shape[-1]`` is always ``None``
    for the ragged axis) instead of ``Ragged.ndim``, which counts bytestring
    dtypes differently from numeric dtypes.

    The fast ``np.add.reduceat`` path only works when the data buffer is
    already laid out in batch order — i.e. a single fixed dim, or every inner
    fixed dim == 1. Otherwise the buffer is interleaved by the inner axes and
    walking it sequentially produces wrong bytes (pre-fix ploid-1 corruption).
    """
    if isinstance(rag, Ragged):
        fixed = rag.shape[:-1]
        inner_is_trivial = all(s == 1 for s in fixed[1:])
        if len(fixed) == 1 or inner_is_trivial:
            new_lengths = np.add.reduceat(rag.lengths, offsets[:-1], 0)
            cat = Ragged.from_lengths(rag.data, new_lengths)
        elif len(fixed) == 2:
            cat = _cat_length_inner(
                rag, offsets, is_bytestring=is_rag_dtype(rag, np.bytes_)
            )
        else:
            raise NotImplementedError(
                f"Splicing with shape {rag.shape} (≥3 fixed axes) is not implemented."
            )

        if is_rag_dtype(rag, np.bytes_):
            cat = cat.view("S1")  # type: ignore
        return cast(Ragged, cat)
    elif isinstance(rag, RaggedAnnotatedHaps):
        haps = _cat_length(rag.haps, offsets)
        var_idxs = _cat_length(rag.var_idxs, offsets)
        ref_coords = _cat_length(rag.ref_coords, offsets)
        return RaggedAnnotatedHaps(haps, var_idxs, ref_coords)
    else:
        assert_never(rag)


def _cat_length_inner(
    rag: Ragged, offsets: NDArray[np.integer], is_bytestring: bool
) -> Ragged:
    """Per-inner-axis bytes concatenation for (b, inner, ~l) Ragged arrays."""
    inner = rag.shape[1]
    grouped = ak.Array(ListOffsetArray(Index64(offsets), rag.to_ak().layout))
    parts = []
    for i in range(inner):  # type: ignore
        sel = grouped[:, :, i]
        if is_bytestring:
            sel_raw = ak.without_parameters(sel)
            flat = ak.flatten(sel_raw, -1)
            inner_np = flat.layout.content  # type: ignore
            inner_np = NumpyArray(
                inner_np.data.view(np.uint8),  # type: ignore
                parameters={"__array__": "byte"},
            )
            wrapped = ListOffsetArray(
                flat.layout.offsets,  # type: ignore
                inner_np,
                parameters={"__array__": "bytestring"},
            )
            part = ak.Array(wrapped)
        else:
            part = ak.flatten(sel, -1)
        parts.append(part[:, None])
    return Ragged(ak.concatenate(parts, 1))  # type: ignore
```

Verify imports `is_rag_dtype` and `lengths_to_offsets` exist where stated:

Run: `pixi run -e dev python -c "from genvarloader._utils import is_rag_dtype, lengths_to_offsets"`
Expected: no output (success). If `is_rag_dtype` lives in `_ragged` instead, adjust the import accordingly — check first with `grep -rn 'def is_rag_dtype' python/genvarloader/`.

- [ ] **Step 3: Delete `_cat_length` and `_cat_length_inner` from `_impl.py`**

In `python/genvarloader/_dataset/_impl.py`, delete lines 1762–1850 (the two `@overload` decls, the `_cat_length` definition, and `_cat_length_inner`).

Replace the existing `_impl.py` `_cat_length(...)` usage at line 1704 by importing from `_splice`. Near the top of `_impl.py`, add:

```python
from ._splice import _cat_length
```

- [ ] **Step 4: Update `tests/dataset/test_rc_packing.py` import**

Change:
```python
from genvarloader._dataset._impl import _cat_length
```
to:
```python
from genvarloader._dataset._splice import _cat_length
```

- [ ] **Step 5: Run all splice tests; expect green**

Run: `pixi run -e dev pytest tests/dataset/test_rc_packing.py -v`
Expected: PASS — behavior is unchanged; this is a pure relocation.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_splice.py python/genvarloader/_dataset/_impl.py tests/dataset/test_rc_packing.py
rtk git commit -m "refactor(splice): move _cat_length helpers to _splice module"
```

---

## Task 2: Add `SpliceMap` to `_splice.py` (sample-agnostic core)

**Files:**
- Modify: `python/genvarloader/_dataset/_splice.py`

- [ ] **Step 1: Append `SpliceMap` class to `_splice.py`**

Add to `python/genvarloader/_dataset/_splice.py` (after `_cat_length_inner`):

```python
@define
class SpliceMap:
    """Sample-agnostic mapping from splice row → ordered region indices.

    Owns the parsed splice BED, the name → row hash table, the awkward
    splice-map (rows → list[region_idx]), and any active row subset. Used by
    both `Dataset` (via `SpliceIndexer`) and `RefDataset`.
    """

    names: HashTable
    splice_map: ak.Array
    full_splice_map: ak.Array
    row_idxs: NDArray[np.intp]
    row_subset_idxs: NDArray[np.intp] | None = None

    @classmethod
    def from_bed(
        cls,
        splice_info: str | tuple[str, str],
        full_bed: pl.DataFrame,
    ) -> tuple["SpliceMap", pl.DataFrame]:
        """Parse splice_info into a (SpliceMap, spliced_bed) pair. Pure — no sampler.

        `splice_info` is either a column name (rows already in order) or a
        ``(group_col, sort_col)`` pair. The returned `spliced_bed` is the
        per-group aggregation of `full_bed`, ordered as in `splice_map`.
        """
        if isinstance(splice_info, str):
            sp_bed = (
                full_bed
                .rename({splice_info: "splice_id"})
                .with_row_index()
                .group_by("splice_id", maintain_order=True)
                .agg(pl.all())
            )
        elif isinstance(splice_info, tuple):
            if len(splice_info) != 2:
                raise ValueError(
                    "Splice info tuple must be of length 2, corresponding to columns "
                    "names for splice IDs and element ordering."
                )
            sp_bed = (
                full_bed
                .rename({splice_info[0]: "splice_id"})
                .with_row_index()
                .group_by("splice_id", maintain_order=True)
                .agg(pl.all().sort_by(splice_info[1]))
            )
        else:
            assert_never(splice_info)

        names = sp_bed["splice_id"].to_numpy().astype(np.str_)
        lengths = sp_bed["index"].list.len().to_numpy()
        splice_map = Ragged.from_lengths(
            sp_bed["index"].explode().to_numpy(), lengths
        ).to_ak()
        splice_map = cast(ak.Array, splice_map)

        rows = HashTable(max=len(names) * 2, dtype=names.dtype)  # type: ignore
        rows.add(names)

        return (
            cls(
                names=rows,
                splice_map=splice_map,
                full_splice_map=splice_map,
                row_idxs=np.arange(len(splice_map), dtype=np.intp),
                row_subset_idxs=None,
            ),
            sp_bed,
        )

    @property
    def n_rows(self) -> int:
        return len(self.splice_map)

    @property
    def _r_idx(self) -> NDArray[np.intp]:
        if self.row_subset_idxs is None:
            return self.row_idxs
        return self.row_subset_idxs

    def row2idx(self, rows: StrIdx) -> Idx:
        """Convert row names (or already-int indices) to int indices."""
        return s2i(rows, self.names)

    def subset_to(self, rows: StrIdx | None) -> Self:
        """Return a new SpliceMap restricted to the given rows."""
        if rows is None:
            return self
        row_idxs = self._r_idx[self.row2idx(rows)]
        splice_map = cast(ak.Array, self.full_splice_map[row_idxs])
        return evolve(self, splice_map=splice_map, row_subset_idxs=row_idxs)

    def to_full(self) -> Self:
        """Reset to the un-subsetted splice map."""
        return evolve(self, splice_map=self.full_splice_map, row_subset_idxs=None)

    def parse_rows(
        self, rows: Idx | StrIdx
    ) -> tuple[NDArray[np.intp], NDArray[np.int64], tuple[int, ...] | None, bool]:
        """Parse a row index into the inputs needed for a per-region fetch.

        Returns
        -------
        flat_region_idxs
            1-D region indices to feed the unspliced reader.
        offsets
            For ``np.add.reduceat``-style concat (len == n_selected_rows + 1).
        out_reshape
            Target shape for fancy/combo indexing, or ``None``.
        squeeze
            Whether to squeeze the row dim out (scalar index).
        """
        out_reshape: tuple[int, ...] | None = None
        squeeze = False

        r_idx_raw = self.row2idx(rows)
        if isinstance(r_idx_raw, (int, np.integer)):
            squeeze = True
            r_idx = np.atleast_1d(np.asarray(r_idx_raw, dtype=np.intp))
        elif isinstance(r_idx_raw, slice):
            r_idx = np.atleast_1d(self._r_idx[r_idx_raw])
        else:
            r_idx_arr = np.asarray(r_idx_raw)
            if r_idx_arr.ndim > 1:
                out_reshape = r_idx_arr.shape
            r_idx = self._r_idx[r_idx_arr.ravel()]

        # When row indices came in as ints/slices we already advanced through _r_idx
        # for slices; for the scalar/array cases, fold _r_idx in here for non-slice paths.
        # For slices, r_idx is already absolute; for arrays, fold _r_idx.
        if not isinstance(r_idx_raw, slice):
            r_idx = self._r_idx[
                np.asarray(r_idx_raw).ravel()
                if not isinstance(r_idx_raw, (int, np.integer))
                else np.atleast_1d(np.asarray(r_idx_raw, dtype=np.intp))
            ]

        sel = cast(ak.Array, self.full_splice_map[r_idx])
        lengths = ak.count(sel, -1)
        if not isinstance(lengths, np.integer):
            lengths = lengths.to_numpy()
        lengths = cast(NDArray[np.int64], lengths)
        offsets = lengths_to_offsets(lengths)
        flat_region_idxs = ak.flatten(sel, -1).to_numpy().astype(np.intp)

        return flat_region_idxs, offsets, out_reshape, squeeze
```

Note: the double-resolution in the draft above is a hazard — simplify by always folding `_r_idx` exactly once. The simpler, correct version:

```python
    def parse_rows(self, rows):
        out_reshape: tuple[int, ...] | None = None
        squeeze = False

        r_idx_raw = self.row2idx(rows)
        if isinstance(r_idx_raw, (int, np.integer)):
            squeeze = True
            local = np.atleast_1d(np.asarray(r_idx_raw, dtype=np.intp))
        elif isinstance(r_idx_raw, slice):
            local = np.arange(self.n_rows, dtype=np.intp)[r_idx_raw]
        else:
            local = np.asarray(r_idx_raw)
            if local.ndim > 1:
                out_reshape = local.shape
            local = local.ravel().astype(np.intp)

        abs_idx = self._r_idx[local]
        sel = cast(ak.Array, self.full_splice_map[abs_idx])
        lengths = ak.count(sel, -1)
        if not isinstance(lengths, np.integer):
            lengths = lengths.to_numpy()
        lengths = cast(NDArray[np.int64], lengths)
        offsets = lengths_to_offsets(lengths)
        flat_region_idxs = ak.flatten(sel, -1).to_numpy().astype(np.intp)

        return flat_region_idxs, offsets, out_reshape, squeeze
```

Use the simpler version. Delete the first draft of `parse_rows` and keep only this one.

- [ ] **Step 2: Smoke-test `SpliceMap.from_bed` against the existing splice fixture**

Add a temporary script `/tmp/smoke_splicemap.py`:

```python
import polars as pl
from genvarloader._dataset._splice import SpliceMap

bed = pl.DataFrame({
    "chrom": ["chr1"] * 4,
    "chromStart": [0, 100, 200, 300],
    "chromEnd": [10, 110, 210, 310],
    "transcript_id": ["T1", "T1", "T2", "T2"],
    "exon_number": [1, 2, 1, 2],
})
sm, sp_bed = SpliceMap.from_bed("transcript_id", bed)
assert sm.n_rows == 2
print("ok", sm.n_rows, sp_bed.height)

sm2, sp_bed2 = SpliceMap.from_bed(("transcript_id", "exon_number"), bed)
assert sm2.n_rows == 2
print("ok2", sm2.n_rows)

flat, offsets, reshape, squeeze = sm.parse_rows([0, 1])
print("parse", flat, offsets, reshape, squeeze)
```

Run: `pixi run -e dev python /tmp/smoke_splicemap.py`
Expected: prints `ok 2 2`, `ok2 2`, and a `parse [...] [...] None False` line where flat == `[0,1,2,3]` and offsets == `[0,2,4]`.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_dataset/_splice.py
rtk git commit -m "feat(splice): add sample-agnostic SpliceMap"
```

---

## Task 3: Refactor `SpliceIndexer` to compose `SpliceMap` + `DatasetIndexer`

**Files:**
- Modify: `python/genvarloader/_dataset/_indexing.py`
- Modify: `python/genvarloader/_dataset/_impl.py` (call sites in `Dataset.open`, `Dataset.with_settings`, `Dataset._getitem_spliced`)
- Modify: `python/genvarloader/_dummy.py`

- [ ] **Step 1: Rewrite `SpliceIndexer` to delegate row state to `SpliceMap`**

In `python/genvarloader/_dataset/_indexing.py`, replace the entire `SpliceIndexer` class (lines 313–508) with:

```python
@define
class SpliceIndexer:
    """Splice-aware indexer = sample-agnostic SpliceMap + sample-aware DatasetIndexer."""

    map: "SpliceMap"
    dsi: DatasetIndexer

    @classmethod
    def _init(
        cls,
        names: Collection[str] | NDArray[np.str_],
        splice_map: ak.Array,
        dsi: DatasetIndexer,
    ) -> "SpliceIndexer":
        # Kept for backward-compat callers; prefer SpliceMap.from_bed.
        from ._splice import SpliceMap

        _names = np.asarray(names, dtype=np.str_)
        if (
            ak.max(splice_map, None) >= dsi.n_regions
            or ak.min(splice_map, None) < -dsi.n_regions
        ):
            raise ValueError(
                "Found indices in the splice map that are out of bounds for the dataset."
            )
        rows = HashTable(max=len(_names) * 2, dtype=_names.dtype)  # type: ignore
        rows.add(_names)
        sm = SpliceMap(
            names=rows,
            splice_map=splice_map,
            full_splice_map=splice_map,
            row_idxs=np.arange(len(splice_map), dtype=np.intp),
            row_subset_idxs=None,
        )
        return cls(map=sm, dsi=dsi)

    @property
    def n_rows(self) -> int:
        return self.map.n_rows

    @property
    def n_samples(self) -> int:
        return self.dsi.n_samples

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_samples

    @property
    def full_shape(self) -> tuple[int, int]:
        return len(self.map.full_splice_map), len(self.dsi.full_samples)

    def __len__(self):
        return self.n_rows * self.n_samples

    def subset_to(
        self,
        rows: StrIdx | None = None,
        samples: StrIdx | None = None,
    ) -> tuple[Self, DatasetIndexer]:
        if rows is None and samples is None:
            return self, self.dsi

        new_map = self.map.subset_to(rows) if rows is not None else self.map
        sub_dsi = self.dsi.subset_to(samples=samples)
        region_idxs = ak.flatten(new_map.splice_map, None).to_numpy()  # type: ignore
        eff_dsi = self.dsi.subset_to(regions=region_idxs, samples=samples)

        return evolve(self, map=new_map, dsi=sub_dsi), eff_dsi

    def to_full_dataset(self) -> Self:
        return evolve(self, map=self.map.to_full(), dsi=self.dsi.to_full_dataset())

    def parse_idx(self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]):
        """See historical docstring — semantics unchanged."""
        out_reshape = None
        squeeze = False

        if not isinstance(idx, tuple):
            rows = idx
            samples = slice(None)
        elif len(idx) == 1:
            rows = idx[0]
            samples = slice(None)
        else:
            rows, samples = idx

        r_idx = self.map.row2idx(rows)
        s_idx = self.sample2idx(samples)
        idx = (r_idx, s_idx)
        idx_t = idx_type(idx)
        if idx_t == "basic":
            if all(isinstance(i, (int, np.integer)) for i in idx):
                squeeze = True
            r_idx = np.atleast_1d(self.map._r_idx[r_idx])
            s_idx = np.atleast_1d(self.dsi._s_idx[s_idx])
            idx = np.ravel_multi_index(np.ix_(r_idx, s_idx), self.full_shape)
            if isinstance(rows, slice) and isinstance(samples, slice):
                out_reshape = (len(r_idx), len(s_idx))
        elif idx_t == "adv":
            r_idx = self.map._r_idx[r_idx]
            s_idx = self.dsi._s_idx[s_idx]
            idx = np.ravel_multi_index((r_idx, s_idx), self.full_shape)
        elif idx_t == "combo":
            r_idx = self.map._r_idx[r_idx]
            s_idx = self.dsi._s_idx[s_idx]
            idx = np.ravel_multi_index(
                np.ix_(r_idx.ravel(), s_idx.ravel()), self.full_shape
            )
            if r_idx.ndim > 1 or s_idx.ndim > 1:
                out_reshape = (*r_idx.shape, *s_idx.shape)
            elif idx.ndim > 1:
                out_reshape = idx.shape
        else:
            assert_never(idx_t)

        if idx_t == "adv" and idx.ndim > 1:
            out_reshape = idx.shape
        idx = idx.ravel()

        (r_idx, s_idx) = np.unravel_index(idx, self.full_shape)
        r_idx = self.map.full_splice_map[r_idx]
        lengths = ak.count(r_idx, -1)
        if not isinstance(lengths, np.integer):
            lengths = lengths.to_numpy()
        lengths = cast(NDArray[np.int64], lengths)
        offsets = lengths_to_offsets(lengths)
        r_idx = ak.flatten(r_idx, -1).to_numpy()
        s_idx = s_idx.repeat(lengths)

        ds_idx, *_ = self.dsi.parse_idx((r_idx, s_idx))

        return ds_idx, squeeze, out_reshape, offsets

    @property
    def splice_map(self) -> ak.Array:
        # Back-compat shim for any direct readers; prefer .map.splice_map.
        return self.map.splice_map

    @property
    def full_splice_map(self) -> ak.Array:
        return self.map.full_splice_map

    @property
    def row_subset_idxs(self):
        return self.map.row_subset_idxs

    def sample2idx(self, samples: StrIdx) -> Idx:
        return self.dsi.sample2idx(samples)
```

Note the `from ._splice import SpliceMap` inside `_init` — keep it local to avoid a circular import (`_splice.py` imports `s2i` from `_indexing`).

- [ ] **Step 2: Update `Dataset.open` to use `SpliceMap.from_bed`**

In `python/genvarloader/_dataset/_impl.py` around line 265–269, replace:

```python
        if splice_info is not None:
            splice_idxer, spliced_bed = _parse_splice_info(splice_info, bed, idxer)
        else:
            splice_idxer = None
            spliced_bed = None
```

with:

```python
        if splice_info is not None:
            sm, spliced_bed = SpliceMap.from_bed(splice_info, bed)
            splice_idxer = SpliceIndexer(map=sm, dsi=idxer)
        else:
            splice_idxer = None
            spliced_bed = None
```

Add to imports at the top of `_impl.py`:

```python
from ._splice import SpliceMap, _cat_length
```

(Keep `_cat_length` here — already added in Task 1.)

- [ ] **Step 3: Update `Dataset.with_settings` to use `SpliceMap.from_bed`**

In `_impl.py` around lines 427–436, replace the `_parse_splice_info` call:

```python
        if splice_info is not None:
            if splice_info is False:
                splice_idxer = None
                spliced_bed = None
            else:
                sm, spliced_bed = SpliceMap.from_bed(splice_info, self._full_bed)
                splice_idxer = SpliceIndexer(map=sm, dsi=self._idxer)
            to_evolve["_sp_idxer"] = splice_idxer
            to_evolve["_spliced_bed"] = spliced_bed
```

- [ ] **Step 4: Update `Dataset.spliced_regions` to use `splice_idxer.map.row_subset_idxs`**

Around lines 920–928, the existing code reads `self._sp_idxer.row_subset_idxs` — that still works thanks to the back-compat shim in step 1, so no change is needed. Verify by inspection.

- [ ] **Step 5: Delete `_parse_splice_info` from `_impl.py`**

Delete lines 1883–1932 (the entire `_parse_splice_info` function).

- [ ] **Step 6: Update `_dummy.py`**

In `python/genvarloader/_dummy.py`:

Change line 11 from:

```python
from ._dataset._impl import RaggedDataset, _parse_splice_info
```

to:

```python
from ._dataset._impl import RaggedDataset
from ._dataset._indexing import SpliceIndexer
from ._dataset._splice import SpliceMap
```

Change line ~186 (inside `get_dummy_dataset`) from:

```python
        dummy_spi, sp_bed = _parse_splice_info(("gene", "exon"), dummy_bed, dummy_idxer)
```

to:

```python
        sm, sp_bed = SpliceMap.from_bed(("gene", "exon"), dummy_bed)
        dummy_spi = SpliceIndexer(map=sm, dsi=dummy_idxer)
```

- [ ] **Step 7: Run the full existing test suite for splice + dataset behavior**

Run: `pixi run -e dev pytest tests/dataset/ -v -x`
Expected: PASS. Any regression here means the `SpliceIndexer` refactor changed observable behavior — fix before continuing.

Run (broader sanity): `pixi run -e dev pytest tests/ -v -x -k "not slow"`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
rtk git add python/genvarloader/_dataset/_indexing.py python/genvarloader/_dataset/_impl.py python/genvarloader/_dummy.py python/genvarloader/_dataset/_splice.py
rtk git commit -m "refactor(splice): compose SpliceIndexer from SpliceMap + DatasetIndexer"
```

---

## Task 4: Add splice fields and `is_spliced`/`spliced_regions` to `RefDataset` (no behavior yet)

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py`

- [ ] **Step 1: Write a failing test for the new fields and properties**

Append to `tests/test_ref_ds.py`:

```python
def test_refdataset_unspliced_defaults(reference: gvl.Reference):
    bed = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [0, 100],
        "chromEnd": [100, 150],
    })
    ds = gvl.RefDataset(reference, bed)
    assert ds.is_spliced is False
    assert ds.splice_info is None
```

Run: `pixi run -e dev pytest tests/test_ref_ds.py::test_refdataset_unspliced_defaults -v`
Expected: FAIL with `AttributeError` on `is_spliced` or `splice_info`.

- [ ] **Step 2: Add `splice_info`, `_splice_map`, `_spliced_bed`, `is_spliced`, `spliced_regions`**

In `python/genvarloader/_dataset/_reference.py`:

Add to the imports near the top (after the existing `from .._ragged import ...` line):

```python
from ._splice import SpliceMap, _cat_length
```

Inside `class RefDataset(Generic[T])`, after the existing `region_names: str | None = None` and `_region_map` fields (around line 201), add:

```python
    splice_info: str | tuple[str, str] | None = None
    """If set, the dataset is spliced. The value is the column name (rows already in order)
    or a (group_col, sort_col) pair, applied against ``full_bed``."""
    _splice_map: SpliceMap | None = field(init=False, alias="_splice_map", default=None)
    _spliced_bed: pl.DataFrame | None = field(
        init=False, alias="_spliced_bed", default=None
    )
```

In `__attrs_post_init__`, after the existing region-name initialization, add:

```python
        if self.splice_info is not None:
            sm, sp_bed = SpliceMap.from_bed(self.splice_info, self.full_bed)
            self._splice_map = sm
            self._spliced_bed = sp_bed
            self._check_valid_state()
        else:
            self._splice_map = None
            self._spliced_bed = None
```

Add `_check_valid_state` as a method on `RefDataset` (place after `__attrs_post_init__`):

```python
    def _check_valid_state(self):
        if self._splice_map is None:
            return
        if self.jitter > 0:
            raise RuntimeError(
                "Jitter is not supported with splicing. Please set jitter to 0."
            )
        if not self.deterministic:
            raise RuntimeError(
                "Non-deterministic algorithms are not supported with splicing."
                " Please set deterministic to True."
            )
        if isinstance(self.output_length, int):
            raise RuntimeError(
                "Splicing requires output_length='ragged' or 'variable',"
                " not a fixed integer length."
            )
```

Add the two properties after the existing `regions` property:

```python
@property
def is_spliced(self) -> bool:
    """Whether the dataset is spliced."""
    return self._splice_map is not None


@property
def spliced_regions(self) -> pl.DataFrame:
    """The spliced BED, subset to the current row subset."""
    if self._spliced_bed is None or self._splice_map is None:
        raise ValueError("Dataset does not have splice information.")
    subset = self._splice_map.row_subset_idxs
    if subset is None:
        return self._spliced_bed
    return self._spliced_bed[subset]
```

Update `shape` and `__len__` to reflect splice rows when spliced:

```python
@property
def shape(self) -> tuple[int]:
    """Shape of the dataset."""
    if self._splice_map is not None:
        return (self._splice_map.n_rows,)
    return (self.regions.height,)


def __len__(self) -> int:
    if self._splice_map is not None:
        return self._splice_map.n_rows
    return self.regions.height
```

- [ ] **Step 3: Run the new test plus the broader RefDataset tests**

Run: `pixi run -e dev pytest tests/test_ref_ds.py -v`
Expected: PASS — including the newly-added `test_refdataset_unspliced_defaults`.

- [ ] **Step 4: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py tests/test_ref_ds.py
rtk git commit -m "feat(refds): add splice_info field and is_spliced/spliced_regions"
```

---

## Task 5: Implement `_getitem_spliced` and dispatch in `__getitem__`

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py`
- Create: `tests/test_ref_ds_splicing.py`

- [ ] **Step 1: Write a failing test for single-column splice on the positive strand**

Create `tests/test_ref_ds_splicing.py`:

```python
from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
import pytest
import seqpro as sp

DDIR = Path(__file__).parent / "data"
REF = DDIR / "fasta" / "hg38.fa.bgz"


@pytest.fixture
def reference() -> gvl.Reference:
    return gvl.Reference.from_path(REF, in_memory=False)


@pytest.fixture
def two_transcript_bed() -> pl.DataFrame:
    # Two transcripts, both on '+' strand. T1 has 2 exons; T2 has 1 exon.
    return pl.DataFrame({
        "chrom": ["chr1", "chr1", "chr1"],
        "chromStart": [1000, 2000, 5000],
        "chromEnd": [1010, 2010, 5010],
        "strand": [1, 1, 1],
        "transcript_id": ["T1", "T1", "T2"],
        "exon_number": [1, 2, 1],
    })


def test_spliced_single_col(reference: gvl.Reference, two_transcript_bed: pl.DataFrame):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
    assert ds.is_spliced is True
    assert len(ds) == 2  # two transcripts

    spliced = ds[:]  # ragged: (2, ~l)
    # Compare against a manual concatenation of unspliced reads in BED order.
    unsp = gvl.RefDataset(reference, two_transcript_bed)[:]
    expected_t1 = np.concatenate([unsp[0], unsp[1]])
    expected_t2 = unsp[2]

    np.testing.assert_equal(spliced[0], expected_t1)
    np.testing.assert_equal(spliced[1], expected_t2)
```

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py::test_spliced_single_col -v`
Expected: FAIL — `__getitem__` still uses the unspliced path; indexing with `idx=0` returns a single region of length 10 instead of the spliced length 20.

- [ ] **Step 2: Add `_getitem_spliced` and dispatch**

In `python/genvarloader/_dataset/_reference.py`, replace the existing `__getitem__` (around line 339) with a dispatcher and add `_getitem_spliced`:

```python
    def __getitem__(self, idx: Idx) -> T:
        if self._splice_map is not None:
            return self._getitem_spliced(idx)
        return self._getitem_unspliced(idx)

    def _getitem_spliced(self, idx: Idx) -> T:
        assert self._splice_map is not None
        # Splicing forbids fixed-length output — enforced in _check_valid_state.
        assert not isinstance(self.output_length, int)

        flat_r_idx, offsets, out_reshape, squeeze = self._splice_map.parse_rows(idx)

        # Build a ragged, unspliced inner view that reads the flat exon list.
        inner = evolve(
            self,
            output_length="ragged",
            splice_info=None,
            _splice_map=None,
            _spliced_bed=None,
        )
        ref = inner._getitem_unspliced(flat_r_idx)  # Ragged[S1]
        ref = _cat_length(ref, offsets)

        if out_reshape is not None:
            ref = ref.reshape(out_reshape)  # type: ignore

        if self.output_length == "ragged":
            out = ref
        elif self.output_length == "variable":
            out = to_padded(ref, pad_value=self.reference.pad_char)  # type: ignore
        else:
            raise AssertionError("splice + fixed-length output should be blocked earlier")

        if squeeze:
            out = out.squeeze(0)  # type: ignore

        return cast(T, out)

    def _getitem_unspliced(self, idx: Idx) -> T:
```

Now rename the body of the previous `__getitem__` to live under `_getitem_unspliced` (everything from the original `regions = self._subset_regions[idx].copy()` line through the final `return cast(T, out)` is unchanged — just re-indented under the new method name).

- [ ] **Step 3: Run the test, expect PASS**

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py::test_spliced_single_col -v`
Expected: PASS.

- [ ] **Step 4: Add tests for two-col splice (out-of-order exons) and mixed strand**

Append to `tests/test_ref_ds_splicing.py`:

```python
def test_spliced_two_col_reorders_exons(reference: gvl.Reference):
    # Exons stored out-of-order; exon_number column dictates splice order.
    bed = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [2000, 1000],
        "chromEnd": [2010, 1010],
        "strand": [1, 1],
        "transcript_id": ["T1", "T1"],
        "exon_number": [2, 1],
    })

    ds = gvl.RefDataset(reference, bed, splice_info=("transcript_id", "exon_number"))
    spliced = ds[0]

    # Manual order: exon 1 then exon 2 — i.e. read row 1 then row 0 of full_bed.
    unsp = gvl.RefDataset(reference, bed)[:]
    expected = np.concatenate([unsp[1], unsp[0]])
    np.testing.assert_equal(spliced, expected)


def test_spliced_mixed_strand(reference: gvl.Reference):
    # T1 has both exons on '-' strand; rc_neg=True means per-exon RC, then concat.
    bed = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [1000, 2000],
        "chromEnd": [1010, 2010],
        "strand": [-1, -1],
        "transcript_id": ["T1", "T1"],
        "exon_number": [1, 2],
    })

    ds = gvl.RefDataset(reference, bed, splice_info="transcript_id")
    spliced = ds[0]

    # Per-exon RC, then concat in BED order.
    unsp = gvl.RefDataset(reference, bed, rc_neg=True)[:]
    expected = np.concatenate([unsp[0], unsp[1]])
    np.testing.assert_equal(spliced, expected)
```

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py -v`
Expected: PASS for all three tests.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py tests/test_ref_ds_splicing.py
rtk git commit -m "feat(refds): implement spliced __getitem__ via SpliceMap + _cat_length"
```

---

## Task 6: Add `with_settings(splice_info=...)`, `subset_to`, and `to_full_dataset` splice support

**Files:**
- Modify: `python/genvarloader/_dataset/_reference.py`
- Modify: `tests/test_ref_ds_splicing.py`

- [ ] **Step 1: Write failing tests for `with_settings` and `subset_to`**

Append to `tests/test_ref_ds_splicing.py`:

```python
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


def test_subset_to_transcripts(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed, splice_info="transcript_id")
    sub = ds.subset_to(["T2"])
    assert len(sub) == 1
    spliced = sub[0]
    unsp = gvl.RefDataset(reference, two_transcript_bed)[:]
    np.testing.assert_equal(spliced, unsp[2])
```

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py -v -k "with_settings or subset_to"`
Expected: FAIL (with_settings does not accept `splice_info`, and subset_to does not understand splice rows).

- [ ] **Step 2: Extend `with_settings` signature and behavior**

In `python/genvarloader/_dataset/_reference.py`, replace the existing `with_settings` signature with:

```python
    def with_settings(
        self,
        jitter: int | None = None,
        deterministic: bool | None = None,
        rc_neg: bool | None = None,
        seed: int | np.random.Generator | None = None,
        splice_info: str | tuple[str, str] | Literal[False] | None = None,
    ) -> Self:
```

After the existing `if seed is not None:` block (before the final `return evolve(...)`), add:

```python
        if splice_info is not None:
            if splice_info is False:
                to_evolve["splice_info"] = None
                to_evolve["_splice_map"] = None
                to_evolve["_spliced_bed"] = None
            else:
                sm, sp_bed = SpliceMap.from_bed(splice_info, self.full_bed)
                to_evolve["splice_info"] = splice_info
                to_evolve["_splice_map"] = sm
                to_evolve["_spliced_bed"] = sp_bed
```

Change the final `return evolve(self, **to_evolve)` to:

```python
        out = evolve(self, **to_evolve)
        out._check_valid_state()
        return out
```

Add `from typing import Literal` to the existing typing imports at the top if not already present (it is — line 5).

- [ ] **Step 3: Extend `subset_to` to handle splice rows**

In `python/genvarloader/_dataset/_reference.py`, replace the existing `subset_to` body (around line 306–331) with:

```python
def subset_to(self, regions: StrIdx):
    """Subset the dataset to a subset of regions (or transcripts, when spliced)."""
    if self._splice_map is not None:
        new_map = self._splice_map.subset_to(regions)
        # Subset full_bed to the flat region rows referenced by the new map.
        flat = ak.flatten(new_map.splice_map, None).to_numpy()
        self._splice_map = new_map
        self._subset_bed = self.full_bed[flat]
        self._subset_regions = bed_to_regions(self._subset_bed, self.reference.c_map)
        return self

    if self._region_map is not None:
        regions = s2i(regions, self._region_map)
    elif is_str_arr(regions):
        raise ValueError(
            "Cannot subset to regions by name because no region name was set."
        )

    if (
        isinstance(regions, (int, np.integer, slice))
        or is_dtype(regions, np.integer)
        or (isinstance(regions, Sequence) and isinstance(regions[0], int))
    ):
        self._subset_bed = self.full_bed[regions]  # type: ignore
    else:
        self._subset_bed = self.full_bed.filter(regions)  # type: ignore

    self._subset_regions = bed_to_regions(self._subset_bed, self.reference.c_map)
    return self
```

- [ ] **Step 4: Extend `to_full_dataset` to reset the splice map**

In `python/genvarloader/_dataset/_reference.py`, replace `to_full_dataset` with:

```python
    def to_full_dataset(self) -> Self:
        """Reset the dataset to the full dataset."""
        if self._splice_map is not None:
            self._splice_map = self._splice_map.to_full()
        self._subset_bed = self.full_bed
        self._subset_regions = bed_to_regions(self._subset_bed, self.reference.c_map)
        return self
```

- [ ] **Step 5: Run the splice tests**

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py -v`
Expected: PASS for all tests.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py tests/test_ref_ds_splicing.py
rtk git commit -m "feat(refds): support with_settings(splice_info) and spliced subset_to"
```

---

## Task 7: Test `output_length='variable'` and fixed-length rejection

**Files:**
- Modify: `tests/test_ref_ds_splicing.py`

- [ ] **Step 1: Write failing test for variable-length pad and fixed-length rejection**

Append to `tests/test_ref_ds_splicing.py`:

```python
def test_spliced_output_length_variable(reference, two_transcript_bed):
    ds = gvl.RefDataset(
        reference, two_transcript_bed, splice_info="transcript_id"
    ).with_len("variable")
    out = ds[:]
    # variable-length pads to the longest transcript in the batch.
    assert out.shape == (2, 20)  # T1 = 10 + 10 = 20; T2 = 10 (padded)


def test_spliced_rejects_fixed_length(reference, two_transcript_bed):
    ds = gvl.RefDataset(reference, two_transcript_bed)
    with pytest.raises(RuntimeError, match="Splicing requires output_length"):
        ds.with_settings(splice_info="transcript_id").with_len(5)
```

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py -v -k "variable or fixed"`
Expected: PASS (`_check_valid_state` already blocks fixed length; `_getitem_spliced` already calls `to_padded` for `variable`). If the variable-length assertion fails because `to_padded` chooses a different shape, adjust the assertion to match `out.shape[0] == 2` and `out.shape[1] == max transcript length`, then verify the per-row equality against the unspliced concatenation.

- [ ] **Step 2: Also verify `with_len` itself routes through `_check_valid_state`**

`RefDataset.with_len` currently does `return evolve(self, output_length=output_length)`. After evolve, splice validation must run. Update `with_len` to:

```python
def with_len(self, output_length: Literal["ragged", "variable"] | int) -> RefDataset:
    if isinstance(output_length, int):
        if output_length < 1:
            raise ValueError(
                f"Output length ({output_length}) must be a positive integer."
            )
        min_r_len: int = (self._subset_regions[:, 2] - self._subset_regions[:, 1]).min()
        max_output_length = min_r_len
        eff_length = output_length + 2 * self.jitter

        if eff_length > max_output_length:
            raise ValueError(
                f"Jitter-expanded output length (out_len={self.output_length}) + 2 * ({self.jitter=}) = {eff_length} must be less"
                f" than or equal to the maximum output length of the dataset ({max_output_length})."
                f" The maximum output length is the minimum region length ({min_r_len})."
            )

    out = evolve(self, output_length=output_length)
    out._check_valid_state()
    return out
```

(Body unchanged except for `out._check_valid_state()`.)

- [ ] **Step 3: Run all splice tests**

Run: `pixi run -e dev pytest tests/test_ref_ds_splicing.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/test_ref_ds_splicing.py python/genvarloader/_dataset/_reference.py
rtk git commit -m "test(refds): cover spliced variable-length output and fixed-length rejection"
```

---

## Task 8: Full regression run + docs/skill updates

**Files:**
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `docs/source/splicing.ipynb`
- Modify: `python/genvarloader/_dataset/_reference.py` (docstring)

- [ ] **Step 1: Run the full pytest suite**

Run: `pixi run -e dev test`
Expected: PASS. Investigate and fix any regression before continuing.

- [ ] **Step 2: Type-check**

Run: `pixi run -e dev basedpyright python/genvarloader/_dataset/_splice.py python/genvarloader/_dataset/_reference.py python/genvarloader/_dataset/_indexing.py python/genvarloader/_dataset/_impl.py`
Expected: no new errors beyond the pre-existing baseline.

Run: `pixi run -e dev ruff check python/`
Expected: clean.

- [ ] **Step 3: Update the `RefDataset` docstring**

In `python/genvarloader/_dataset/_reference.py`, update the class docstring on `RefDataset` to mention `splice_info`:

```python
class RefDataset(Generic[T]):
    """A reference dataset for pulling out sequences from a reference genome.

    When ``splice_info`` is provided, the dataset returns per-transcript
    concatenated reference sequence, with one row per splice group instead of
    one row per BED region. Same semantics as
    :meth:`Dataset.open(splice_info=...) <genvarloader.Dataset.open>`.
    """
```

Also update the `splice_info` field docstring (already added in Task 4) to describe the accepted forms.

- [ ] **Step 4: Update the gvl skill**

In `skills/genvarloader/SKILL.md`, find the "Spliced haplotypes" section. Append (or merge into the existing prose):

> `RefDataset` accepts the same `splice_info` argument as `Dataset.open`. Pass either a transcript-ID column name (rows already in splice order) or a `(group_col, sort_col)` tuple to reorder exons. `with_settings(splice_info=False)` disables splicing on an existing `RefDataset`; pass a new value to re-enable. Splicing requires `output_length` in `{"ragged", "variable"}`, `jitter=0`, and `deterministic=True`. `subset_to(transcript_ids)` works the same as for `Dataset`.

- [ ] **Step 5: Add a brief example to `docs/source/splicing.ipynb`**

Open `docs/source/splicing.ipynb`. Append a new section "Spliced reference-only datasets" with:

```python
import genvarloader as gvl

ref = gvl.Reference.from_path("hg38.fa.bgz")
bed = gvl.get_splice_bed("annotations.gtf")
ref_ds = gvl.RefDataset(ref, bed, splice_info="transcript_id")
seqs = ref_ds[:]  # Ragged[S1], one row per transcript
```

Keep it under ~10 lines; mirror the tone of the existing splicing example.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reference.py skills/genvarloader/SKILL.md docs/source/splicing.ipynb
rtk git commit -m "docs(refds): document RefDataset splicing in skill + notebook"
```

---

## Self-Review Notes

**Spec coverage check:**
- New `SpliceMap` in `_splice.py` — Task 2.
- `SpliceIndexer` refactor — Task 3.
- `_cat_length`/`_cat_length_inner` move — Task 1.
- `RefDataset` splice fields + post-init — Task 4.
- `_check_valid_state` invariants — Task 4 + Task 7 (`with_len` routing).
- `is_spliced`, `spliced_regions` properties — Task 4.
- `shape`/`__len__` reflect splice rows — Task 4.
- `with_settings(splice_info=...)` — Task 6.
- `subset_to` (spliced rows) — Task 6.
- `__getitem__` dispatch + `_getitem_spliced` — Task 5.
- `Dataset.open`, `Dataset.with_settings`, `Dataset._getitem_spliced` touch points — Task 3.
- `_dummy.py` update — Task 3.
- Splice tests `test_spliced_single_col`, `test_spliced_two_col`, `test_spliced_mixed_strand`, `test_spliced_subset_to`, `test_spliced_with_settings_disable`, `test_spliced_validation`, `test_spliced_output_length_*` — Tasks 5/6/7. (Note: `test_*_two_col` is named `test_spliced_two_col_reorders_exons` and `test_spliced_subset_to` is named `test_subset_to_transcripts`; semantics are equivalent.)
- Skill + docstring + notebook updates — Task 8.

**Risks (from spec):**
- `SpliceIndexer` back-compat shim properties (`splice_map`, `full_splice_map`, `row_subset_idxs`) are intentionally preserved in Task 3 to keep `Dataset.spliced_regions` and other existing readers working without further edits.
- Circular-import hazard between `_splice.py` (uses `s2i` from `_indexing`) and `_indexing.py` (uses `SpliceMap` only inside `_init`) is handled by a function-local import in `SpliceIndexer._init`.

