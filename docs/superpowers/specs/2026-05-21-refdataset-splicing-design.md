# Splicing support for `RefDataset`

**Date:** 2026-05-21
**Status:** Draft

## Goal

Add transcript-level splicing to `gvl.RefDataset` (reference-only dataset, no
genotypes), and in the same change refactor `Dataset`'s splicing internals so
both datasets share a single sample-agnostic splice-map abstraction. Net result:

- `RefDataset(splice_info=..., region_names=..., ...)` works the same as
  `Dataset.open(splice_info=...)`, returning per-transcript concatenated
  reference sequence.
- `with_settings(splice_info=...)` works on `RefDataset` (enable / change /
  disable splicing), mirroring `Dataset.with_settings`.
- The existing public API of `Dataset` does not change. Behavior is
  bit-identical.

## Non-goals

- No new variant-related knobs on `RefDataset` (e.g. `var_filter="exonic"`).
  `RefDataset` is variant-free; splice_info is the only spliced kwarg added.
- No on-disk format change. `RefDataset` is purely a runtime construct.
- No change to `gvl.write` or `Dataset.open`'s splicing behavior.

## Current state

Splicing in `Dataset` is implemented in three pieces:

1. `_parse_splice_info(splice_info, full_bed, idxer)` (`_dataset/_impl.py`)
   builds an `ak.Array` `splice_map` (rows → ordered list of region indices)
   and a `spliced_bed` DataFrame, then constructs a `SpliceIndexer`.
2. `SpliceIndexer` (`_dataset/_indexing.py`) wraps a sample-aware
   `DatasetIndexer` plus the splice map. Its `parse_idx` flattens
   `(transcript, sample)` indices into per-region, per-sample indices plus
   `offsets` for later concatenation.
3. `Dataset._getitem_spliced` (`_dataset/_impl.py`) fetches per-region ragged
   data via `_recon`, reverse-complements negative-strand regions, then
   `_cat_length` concatenates within each transcript.

Splicing is logically **orthogonal to samples and variants** — it is "fetch
these rows in this order, concatenate them per group". The current code
implements it inside `SpliceIndexer`, which is sample-aware, so reusing it for
`RefDataset` (which has no sample dimension) would require either an awkward
phantom-sample adapter or duplication.

## Approach: extract a sample-agnostic `SpliceMap`

A new module `python/genvarloader/_dataset/_splice.py` owns the
splice-map type and the post-fetch concatenation helpers. Both `Dataset`
and `RefDataset` depend on this module.

### `SpliceMap` (new)

```python
@define
class SpliceMap:
    names: HashTable                   # row name → row idx
    splice_map: ak.Array               # rows → list[region_idx] (current view)
    full_splice_map: ak.Array          # pre-subset map
    row_idxs: NDArray[np.intp]
    row_subset_idxs: NDArray[np.intp] | None = None

    @classmethod
    def from_bed(
        cls,
        splice_info: str | tuple[str, str],
        full_bed: pl.DataFrame,
    ) -> tuple["SpliceMap", pl.DataFrame]:
        """Parse splice_info into (SpliceMap, spliced_bed). Pure; no sampler."""

    @property
    def n_rows(self) -> int: ...
    @property
    def _r_idx(self) -> NDArray[np.intp]: ...
    def row2idx(self, rows: StrIdx) -> Idx: ...
    def subset_to(self, rows: StrIdx | None) -> Self: ...
    def to_full(self) -> Self: ...

    # Parse rows into:
    #   flat_region_idxs: 1-D region indices to feed the unspliced reader
    #   offsets:          for np.add.reduceat-style concat (len = n_rows + 1)
    #   out_reshape:      target shape for fancy/combo indexing, or None
    #   squeeze:          whether to squeeze the row dim out (scalar idx)
    def parse_rows(
        self, rows: StrIdx
    ) -> tuple[
        NDArray[np.intp], NDArray[np.int64], tuple[int, ...] | None, bool
    ]: ...
```

The body of `SpliceMap.from_bed` is the body of today's `_parse_splice_info`
minus the `idxer` arg and final `SpliceIndexer._init` call.

`_cat_length` and `_cat_length_inner` move out of `_impl.py` into `_splice.py`,
unchanged in behavior. They are now reachable from both datasets without an
`_impl ↔ _reference` import cycle.

### `SpliceIndexer` (refactor)

```python
@define
class SpliceIndexer:
    map: SpliceMap
    dsi: DatasetIndexer

    @property
    def n_rows(self) -> int: ...
    @property
    def n_samples(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def full_shape(self) -> tuple[int, int]: ...
    def subset_to(self, rows=None, samples=None
                 ) -> tuple[Self, DatasetIndexer]: ...
    def to_full_dataset(self) -> Self: ...
    def parse_idx(self, idx) -> tuple[Idx, bool, tuple[int,...] | None,
                                       NDArray[np.int64]]: ...
```

All row-side state and logic delegates to `self.map`; all sample-side state to
`self.dsi`. No `Dataset`-facing behavior change.

## `RefDataset` changes (`_dataset/_reference.py`)

### New fields

```python
@define
class RefDataset(Generic[T]):
    ...
    splice_info: str | tuple[str, str] | None = None
    _splice_map: SpliceMap | None = field(init=False, alias="_splice_map",
                                           default=None)
    _spliced_bed: pl.DataFrame | None = field(init=False, alias="_spliced_bed",
                                               default=None)
```

`region_names` already exists; when spliced, it identifies the *transcript*
column. (Today it's only consulted for the unspliced row-name lookup.)

### `__attrs_post_init__`

After existing validation, if `self.splice_info is not None`:

- `splice_map, spliced_bed = SpliceMap.from_bed(self.splice_info, self.full_bed)`
- assign `_splice_map`, `_spliced_bed`
- call `_check_valid_state()` (below) to enforce splice invariants.

### Validation (`_check_valid_state`)

When spliced, enforce the same invariants as `Dataset`:

- `jitter == 0` → `RuntimeError`
- `deterministic is True` → `RuntimeError`
- `output_length` may not be `int` (only `"ragged"` / `"variable"`) →
  `RuntimeError` (consistent with `_getitem_spliced` assert in `Dataset`)

Error messages mirror the existing ones in `Dataset._check_valid_state`.

### New properties

```python
@property
def is_spliced(self) -> bool: return self._splice_map is not None

@property
def spliced_regions(self) -> pl.DataFrame:
    """The spliced BED, subset to the current row subset."""
```

When spliced, `shape` and `__len__` return `(n_splice_rows,)`.

### `with_settings`

Extend with:

```python
def with_settings(
    self, ...,
    splice_info: str | tuple[str, str] | Literal[False] | None = None,
) -> Self:
```

- `None` → no change.
- `False` → disable splicing (`_splice_map = None`, `_spliced_bed = None`).
- value → re-parse via `SpliceMap.from_bed(value, self.full_bed)`.

After `evolve(...)`, call `_check_valid_state()` (defense-in-depth, matches
`Dataset.with_settings`).

### `subset_to`

When spliced, `subset_to(rows)` delegates to `_splice_map.subset_to(rows)` and
recomputes the flat region BED from the flattened region indices in the
subset map. This keeps the unspliced `__getitem__` path operating on the
already-subset regions array — no second indirection.

When unspliced, behavior is unchanged.

### `__getitem__`

Dispatch:

```python
def __getitem__(self, idx: Idx) -> T:
    if self._splice_map is not None:
        return self._getitem_spliced(idx)
    # ... existing unspliced path ...
```

`_getitem_spliced`:

1. `flat_r_idx, offsets, out_reshape, squeeze = self._splice_map.parse_rows(idx)`
2. Build an unspliced view and reuse the existing per-region fetch:
   ```python
   inner = evolve(
       self, output_length="ragged",
       _splice_map=None, _spliced_bed=None,
   )
   ref: Ragged[np.bytes_] = inner[flat_r_idx]
   ```
   This is the DRY win — exactly one place fetches reference bytes for a list
   of region rows. RC handling sits inside that path and runs per-exon based
   on each exon's strand, matching the chosen "per-exon RC, then concat"
   semantics.
3. `ref = _cat_length(ref, offsets)` — joins exons per transcript.
4. Materialize per `self.output_length`:
   - `"ragged"` → reshape `ref` via `out_reshape` if non-None, return.
   - `"variable"` → `to_padded(ref, self.reference.pad_char)`.
   - `int` → asserted out earlier; safety assert here.
5. Squeeze if `squeeze`.

### Invariants enforced

- Padding under `"variable"` is to the longest **transcript** in the batch
  (natural fallout of pad-after-concat). Matches `Dataset` behavior.
- Fixed-length output is rejected under splicing.
- Jitter and non-determinism are rejected under splicing.

## Touch points in `_impl.py` (behavior-preserving)

- Delete `_parse_splice_info`, `_cat_length`, `_cat_length_inner`. They now
  live in `_splice.py`.
- `Dataset.open`: replace `_parse_splice_info(splice_info, bed, idxer)` with
  `map, spliced_bed = SpliceMap.from_bed(splice_info, bed)` and
  `splice_idxer = SpliceIndexer(map=map, dsi=idxer)`.
- `Dataset.with_settings`: same substitution.
- `Dataset._getitem_spliced`: read `splice_idxer.map` where it currently reads
  `splice_idxer.splice_map`/`row_subset_idxs`/etc. Update `_cat_length`
  import.

## Touch points in `_dummy.py`

`get_dummy_dataset(spliced=True)` currently imports `_parse_splice_info` from
`_impl.py`. Switch to `SpliceMap.from_bed`.

## Tests

New tests in `tests/test_ref_ds.py`:

- `test_spliced_single_col` — `splice_info="transcript_id"` matches manual
  concat of per-row `RefDataset` reads in BED order.
- `test_spliced_two_col` — `splice_info=("transcript_id","exon_number")` with
  shuffled exon rows reorders correctly.
- `test_spliced_mixed_strand` — per-exon RC then concat for `-` exons.
- `test_spliced_subset_to` — subsetting by transcript ID yields a proper
  spliced view.
- `test_spliced_with_settings_disable` — `with_settings(splice_info=False)`
  reverts to unspliced; `with_settings(splice_info=...)` re-enables.
- `test_spliced_validation` — setting `splice_info` with `jitter>0`,
  `deterministic=False`, or `output_length=int` raises.
- `test_spliced_output_length_ragged_variable` — both modes produce correct
  shapes.

Existing `Dataset` splice tests are the contract for the refactor — they must
pass unchanged.

## Docs / skill maintenance (per CLAUDE.md)

- Update `skills/genvarloader/SKILL.md`: extend the "Spliced haplotypes"
  section to mention `RefDataset(splice_info=...)`, same semantics.
- `docs/source/splicing.ipynb`: add a brief example using `RefDataset`
  splicing (reference-only transcript sequence).
- Update `RefDataset` docstring with the new `splice_info` arg.

## Risks

- The `SpliceIndexer` refactor changes a class layout that is currently
  serialized into `Dataset` via attrs `evolve()`. Need to verify no external
  pickling/serialization depends on the old field names (project does not
  pickle datasets across versions, so low risk).
- `_dummy.py` and any other module reaching into the private
  `_parse_splice_info` symbol must move to `SpliceMap.from_bed`. Grep for
  callers as part of the change.

## Rollout

Single PR:

1. Add `_splice.py` with `SpliceMap`, `_cat_length`, `_cat_length_inner`.
2. Refactor `SpliceIndexer` to compose `SpliceMap` + `DatasetIndexer`. Update
   callers in `_impl.py` and `_dummy.py`. All existing splice tests must
   pass.
3. Add splice fields, validation, properties, `with_settings`, `subset_to`,
   and `_getitem_spliced` to `RefDataset`.
4. Add `RefDataset` splice tests.
5. Update skill + docstring + notebook.
