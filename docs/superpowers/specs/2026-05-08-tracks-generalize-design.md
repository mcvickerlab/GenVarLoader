# Generalize `gvl.write()` to accept `Table` tracks alongside `BigWigs`

**Date:** 2026-05-08
**Status:** Approved (brainstorming)

## Summary

Generalize the dataset-writing API so `gvl.write()` can ingest interval data from sources other than bigWig files. A new `Table` class accepts long-form DataFrames (or per-sample DataFrames, or files) of `(sample_id, chrom, start, end, value)` rows and exposes the same interval-extraction surface that `BigWigs` already provides. The two are unified behind an `IntervalTrack` Protocol so the existing `_write_bigwigs` routine generalizes to `_write_track` with no behavioral change. Mixed sequences (`BigWigs + Table` together in one write) are supported.

## Motivation

`gvl.write()` currently only accepts bigWig files for non-sequence track data. Users with intervals in tabular form (BED-like files, parquet exports from peak callers, joined-sample tables) must convert to bigWig before they can build a GVL dataset, even though the on-disk format GVL writes is just `(start, end, value)` intervals — exactly what their tables already contain. This adds a lossy round-trip and a tooling dependency for no functional gain.

## Public API

### `gvl.write()` signature change

```python
def write(
    path,
    bed,
    variants=None,
    tracks: IntervalTrack | Sequence[IntervalTrack] | None = None,  # was: bigwigs
    samples=None,
    max_jitter=None,
    overwrite=False,
    max_mem="4g",
    extend_to_length=True,
):
```

- The `bigwigs` keyword is **renamed and removed** — no back-compat shim.
- A single instance or a heterogeneous sequence is accepted (e.g. `[bigwigs_obj, table_obj]`).
- Each element must have a unique `name` (validated up front; the on-disk layout is `intervals/<track.name>/`).

### `Table`

New module `python/genvarloader/_table.py`:

```python
class Table:
    name: str
    samples: list[str]
    contigs: dict[str, int]

    def __init__(
        self,
        name: str,
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None = None,
    ) -> None: ...

    @classmethod
    def from_path(
        cls,
        name: str,
        path: str | Path | Mapping[str, str | Path],
        column_map: Mapping[str, str] | None = None,
    ) -> "Table": ...

    def count_intervals(
        self, contig: str, starts: ArrayLike, ends: ArrayLike,
        sample: str | list[str] | None = None, **kwargs,
    ) -> NDArray[np.int32]: ...

    def _intervals_from_offsets(
        self, contig: str, starts: ArrayLike, ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None, **kwargs,
    ) -> RaggedIntervals: ...
```

**`data`**:
- `pl.DataFrame` — long-form, must have canonical columns `sample_id, chrom, start, end, value` (after applying `column_map`).
- `Mapping[str, pl.DataFrame]` — keys are sample IDs, values omit `sample_id` and have only `chrom, start, end, value`. Mirrors the ergonomics of `BigWigs(paths=dict[str, str])`.

**`column_map`**: maps **canonical → actual** column name, e.g. `{"sample_id": "donor", "start": "chromStart"}`. Only canonical names that need renaming need to appear. Unmapped columns are dropped after selection.

**`from_path`**: dispatches on file extension:
- `.csv` → `pl.read_csv`
- `.tsv`, `.txt` → `pl.read_csv(separator="\t")`
- `.parquet` → `pl.read_parquet`
- `.arrow`, `.ipc` → `pl.read_ipc`

Single path → expected to be long-form. `Mapping[str, path]` → per-sample, no `sample_id` column expected.

### `IntervalTrack` Protocol

Defined in `python/genvarloader/_types.py` alongside `Reader`:

```python
@runtime_checkable
class IntervalTrack(Protocol):
    name: str
    samples: list[str]
    contigs: dict[str, int]

    def count_intervals(
        self, contig: str, starts: ArrayLike, ends: ArrayLike,
        sample: str | list[str] | None = None, **kwargs,
    ) -> NDArray[np.int32]: ...   # shape (regions, samples)

    def _intervals_from_offsets(
        self, contig: str, starts: ArrayLike, ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None, **kwargs,
    ) -> RaggedIntervals: ...
```

`BigWigs` already conforms structurally — no changes to that class. The Protocol is used purely for typing the `tracks` parameter in `write()` and the `track` argument in `_write_track`.

## Internals

### `Table` storage and queries

On construction:
- Normalize input to a single eager `pl.DataFrame` with canonical columns.
- Cast `start`/`end` to `Int64`, `value` to `Float32`, `sample_id`/`chrom` to `Utf8`.
- Sort by `(chrom, sample_id, start)`.
- Compute `samples = sorted(unique(sample_id))`, `contigs = {chrom: int(group_max(end))}`.

`count_intervals(contig, starts, ends, sample)`:
1. Subset stored frame to rows where `chrom == contig` and `sample_id ∈ samples`.
2. Build queries frame: cross-join `(_q, chrom, start, end)` rows with the requested sample list.
3. `counts = polars_bio.count_overlaps(queries, subset, cols1=["chrom","start","end"], cols2=["chrom","start","end"], on_cols=["sample_id"], output_type="polars.DataFrame")`.
4. Sort/pivot to a dense `(len(starts), len(samples))` `NDArray[np.int32]`. Unmatched query rows (no overlap) are zero-filled.

`_intervals_from_offsets(contig, starts, ends, offsets, sample)`:
1. Subset + queries identical to above.
2. `joined = polars_bio.overlap(queries, subset, on_cols=["sample_id"], output_type="polars.DataFrame")` — one row per (query, matching interval).
3. Sort by `(_q, sample_id, table_start)` so intervals within each `(region, sample)` cell are start-ordered, matching `BigWigs`.
4. Extract flat `starts: i32`, `ends: i32`, `values: f32` numpy arrays. Assert their length equals `offsets[-1]`.
5. Construct `RaggedIntervals(Ragged.from_offsets(starts, shape, offsets), …)` with `shape = (len(starts), len(samples), None)`, exactly mirroring `BigWigs._intervals_from_offsets`.

**Dependency**: add `polars-bio` to project dependencies. Import lazily inside `Table` methods so users who never construct a `Table` don't pay the import cost (and a missing optional install can produce a clear error).

**Risks to verify during implementation** (small spike before wiring into `_write_track`):
- Exact output column names from `polars_bio.overlap` after default suffixing.
- Whether `count_overlaps` zero-fills query rows that have no matches, or omits them (the implementation must zero-fill either way).
- Behavior of `on_cols` with both `overlap` and `count_overlaps`.

### `write()` adjustments

In `python/genvarloader/_dataset/_write.py`:

1. Rename param `bigwigs` → `tracks`. Update the early `isinstance(bigwigs, BigWigs)` normalization to handle either single track or sequence; produce `tracks: list[IntervalTrack]`.
2. Validate `len({t.name for t in tracks}) == len(tracks)` — duplicate names would clobber each other on disk.
3. The existing contig/sample-availability loop (`_write.py:151–165`) becomes a loop over `tracks`. Same logic — uses `track.contigs` keys and `track.samples`.
4. The "at least one of variants or tracks" precondition replaces the current bigwigs-specific message.
5. The write loop calls `_write_track(path, gvl_bed, track, samples, max_mem)` for each.

### `_write_track` (renamed from `_write_bigwigs`)

Body is unchanged from the current `_write_bigwigs`. It already only references `bw.name`, `bw.contigs`, `bw.samples`, `bw.count_intervals`, `bw._intervals_from_offsets` — all on the Protocol. Just rename the parameter and update its type annotation to `IntervalTrack`.

### On-disk layout

Unchanged: `intervals/<track.name>/intervals.npy` + `offsets.npy`. `Dataset.open` reads by directory name and is agnostic to which track type produced the data. No reader-side changes required.

## Tests

New tests under `tests/`:

- `tests/test_table.py`
  - Construct `Table` from long-form `pl.DataFrame`.
  - Construct from `dict[str, pl.DataFrame]`.
  - Construct from each file format (`.csv`, `.tsv`, `.parquet`, `.arrow`) via `from_path`.
  - `column_map` rename works end-to-end.
  - `count_intervals` matches a brute-force Python reference on a small fixture (multi-sample, multi-contig, partial overlaps, zero overlaps, exact-edge cases).
  - `_intervals_from_offsets` returns flat arrays in the documented `(region, sample, by-start)` order, with lengths matching the supplied offsets.

- `tests/dataset/test_write_tracks.py`
  - `gvl.write(..., tracks=Table(...))` round-trip: open the dataset and confirm the materialized intervals match the input table values.
  - `gvl.write(..., tracks=[bigwigs, table])` produces both `intervals/<bw_name>/` and `intervals/<table_name>/` and reads back correctly.
  - Duplicate-name validation raises before any disk writes.
  - Sample/contig intersection across mixed track types is computed correctly.

## Out of scope

- Reader-side support for `Table` (queries against a written dataset go through the existing `intervals/` reader and don't care about origin).
- Streaming / chunked Table construction for tables that don't fit in memory. Initial version assumes the table fits.
- Backwards-compatible `bigwigs=` keyword. Clean rename per project convention.

## Worktree and execution

Per the brainstorming workflow, the implementation plan (next step, via `writing-plans`) will execute on a fresh git worktree (`using-git-worktrees`). The current `_write.py` has uncommitted whitespace edits that should be stashed or included as the worktree base — to be confirmed before worktree creation.
