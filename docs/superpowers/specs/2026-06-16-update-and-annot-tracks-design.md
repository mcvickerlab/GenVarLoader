# Design: `gvl.update()`, `annot_tracks` in `gvl.write()`, and parallel writing

Date: 2026-06-16

## Goal

Three related changes to the dataset writing/updating surface:

1. Add a unified `gvl.update(path_or_dataset, ...)` function for adding tracks to an
   existing dataset on disk, analogous to `gvl.write()`. It replaces
   `Dataset.write_annot_tracks` and adds post-hoc per-sample track addition.
2. Make `gvl.write()` accept `annot_tracks` (sample-independent annotation tracks)
   in addition to the existing `tracks` (per-sample).
3. Parallelize `gvl.write()` over its `variants`, `tracks`, and `annot_tracks`
   inputs using background processes for a speedup when they are supplied together.
4. Rework annotation-track extraction to be DRY with `gvl.Table`: use **polars-bio**
   overlap for table/DataFrame sources (retiring the pyranges `_annot_to_intervals`
   path).

Out of scope:
- Transformed-track writing (`Dataset.write_transformed_track` / `Tracks.write_transformed_track`)
  remains a `NotImplementedError` stub, untouched.
- Live (already-open) `Dataset` objects discovering that the on-disk dataset gained
  a new track. Updates are written atomically so a live dataset can keep reading,
  but it will not see new tracks until reopened.

## Background (current state)

- `gvl.write(path, bed, variants=, tracks=, ...)` writes a *complete* dataset into a
  private temp dir via `atomic_dir` and publishes it with `os.replace`. `tracks` is
  an `IntervalTrack` (`BigWigs`/`Table`) or a sequence; each is written per-sample to
  `intervals/<name>/` by `_write_track`.
- `Dataset.write_annot_tracks(tracks: dict[str, str|Path|pl.DataFrame])` adds
  *sample-independent* BED annotations *post-hoc*, writing **directly into the live
  dataset dir** (`annot_intervals/<name>/`) — not atomic. It uses the bespoke
  `_annot_to_intervals` (seqpro/pyranges join), a different code path from how
  `Table`/`BigWigs` extract intervals.
- `metadata.json` does **not** enumerate tracks. Tracks are discovered by
  `Tracks.from_path` scanning `intervals/` (kind `SAMPLE`) and `annot_intervals/`
  (kind `ANNOT`). Adding a track is therefore purely additive — no metadata mutation.
- On-disk offset shapes: sample tracks are offset over `(regions × samples)`; annot
  tracks over `(regions)` only.
- The variant pass mutates `gvl_bed.chromEnd`: each region's end is recomputed to the
  furthest retained variant's end (`_region_end` / `_region_ends_from_list`), which can
  exceed the input region end whenever a deletion straddles the edge. This happens
  **regardless of `extend_to_length`**. Tracks are then written against the recomputed
  bed.

## Decisions

### Source types

- `gvl.write(..., tracks=)` and `gvl.update(..., tracks=)`: per-sample sources, i.e.
  `IntervalTrack` objects (`BigWigs`, `Table`). Unchanged.
- `annot_tracks` (in both `write` and `update`): a `dict[str, source]` where each
  `source` is one of:
  - a path to an interval table (`.csv/.tsv/.txt/.parquet/.arrow/.ipc`),
  - a path to a bigwig,
  - a `pl.DataFrame` or `pl.LazyFrame` interpreted as an interval table.

  `annot_tracks` does **not** accept `Table`/`BigWigs` objects: those force a sample
  name per source, which is redundant with the track `name` and confusing for a
  sample-less annotation.

### Annotation extraction (DRY with `gvl.Table`)

A new leaf writer `_write_annot_track` produces a sample-less `RaggedIntervals`
(`(regions, None)`) and writes `intervals.npy` + `offsets.npy`:

- table/DataFrame/LazyFrame source → **polars-bio** overlap against the dataset's
  region bed (the same machinery `gvl.Table` uses). Replaces and retires
  `_annot_to_intervals` (pyranges).
- bigwig path source → **Rust per-region interval extraction** via the existing
  `BigWigs` backend (bigwig queries are already region-scoped, so no join is needed),
  collapsed into a single sample-less interval set.

### `gvl.write()` parallelization

- New parameter `annot_tracks: dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None = None`.
- Orchestration over up to 3 category jobs (one for variants, one running all
  `tracks` sequentially, one running all `annot_tracks` sequentially):
  - **If `variants` is provided:** run the variants job **first** (serially) so it
    finalizes `gvl_bed.chromEnd`, then run **tracks ∥ annot_tracks** in parallel.
  - **If `variants` is not provided:** run **tracks ∥ annot_tracks** in parallel.
  - `extend_to_length` does **not** affect orchestration (decision B): the variant
    pass can extend `chromEnd` even when `extend_to_length=False`, so tracks must
    always be written against the finalized bed to preserve current coverage
    semantics.
- `max_mem` is divided across the jobs running concurrently so total peak stays under
  the user's budget. (Variants runs alone in its phase and gets the full budget;
  tracks+annot split the budget when they run together.)
- Backend: joblib with the loky process backend (already the de-facto backend; the
  existing `os.fork` polars `RuntimeWarning` suppression remains). All jobs write into
  the single `atomic_dir` temp dataset directory (distinct subdirs: `genotypes/`,
  `intervals/<name>/`, `annot_intervals/<name>/` — no write conflicts), and the entire
  temp dir is published atomically at the end. No per-track atomic rename inside
  `write`.

### `gvl.update()`

Signature (public, exported as `gvl.update`):

```python
def update(
    dataset: str | Path | Dataset,
    tracks: IntervalTrack | Sequence[IntervalTrack] | None = None,
    annot_tracks: dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None = None,
    *,
    overwrite: bool = False,
    max_mem: int | str = "4g",
) -> None:
```

- Accepts a path **or** a `Dataset` (extracts `.path`); does not require a live
  `Dataset` to be opened for use.
- Reads the dataset's already-finalized region ends from disk (`input_regions.arrow` /
  `regions.npy`). There is **no variant pass**, so there is no chromEnd coupling:
  `tracks ∥ annot_tracks` run fully in parallel, `max_mem` divided across concurrent
  jobs. Same loky backend as `write`.
- `tracks` (per-sample): errors unless the track's sample set is in **exact agreement**
  with the dataset's samples — no missing, no extra. Reordering the track's samples to
  match the dataset order is allowed and performed automatically.
- `annot_tracks`: sample-less, same source types and extraction as `write`.
- `overwrite`: if `False`, adding a track whose name already exists raises; if `True`,
  the existing track is replaced.
- **Atomic per-track publish:** each track is built into a private temp dir and
  `atomic_dir`-renamed into its final `intervals/<name>/` or `annot_intervals/<name>/`
  location. This lets a dataset be read while it is being updated without contention.
  (`atomic_dir` already implements temp-build → `os.replace`, including the
  move-aside-then-rename path for `overwrite=True`.)

### Removed / unchanged API

- **Removed:** `Dataset.write_annot_tracks` (subsumed by `gvl.update(..., annot_tracks=)`).
- **Unchanged / out of scope:** `Dataset.write_transformed_track` and
  `Tracks.write_transformed_track` remain as the existing `NotImplementedError` stub.

## Shared structure

Leaf writers shared by `write` and `update`:
- `_write_track(out_dir, bed, track, samples, max_mem)` — per-sample, existing.
- `_write_annot_track(out_dir, region_bed, source, max_mem)` — new, sample-less.

`write` calls these into its `atomic_dir` temp dataset; `update` calls them into a
per-track temp dir and atomic-renames into the live dataset. A small orchestration
helper encapsulates "run these category jobs with loky, dividing `max_mem`" and is
shared by both.

## Affected files

- `python/genvarloader/_dataset/_write.py` — `write` gains `annot_tracks` + parallel
  orchestration; new `_write_annot_track`; new `update`; shared job-runner helper.
- `python/genvarloader/_dataset/_impl.py` — remove `Dataset.write_annot_tracks`;
  retire `_annot_to_intervals` (moved/replaced by polars-bio extraction).
- `python/genvarloader/__init__.py` — export `update` (add to `__all__`).
- `python/genvarloader/_table.py` — reuse/extract its polars-bio overlap helper for
  the annot table path (avoid duplicating overlap setup).
- `skills/genvarloader/SKILL.md` — document `gvl.update`, `annot_tracks`, removal of
  `Dataset.write_annot_tracks`.
- `docs/source/changelog.md` — note the API change.
- `tests/integration/tracks/test_annot_tracks.py` — migrate to `gvl.update(...)`.

## Testing

- annot extraction via polars-bio matches the previous pyranges output on the existing
  fixtures (coordinate alignment assertion in `test_annot_tracks`).
- `gvl.update` adds per-sample tracks and annot tracks to a written dataset; reopening
  shows them via `with_tracks`.
- `gvl.update` sample-set agreement: missing/extra samples raise; reordered samples are
  accepted and aligned.
- `gvl.update` atomicity: a track is built and published atomically (no partial
  `intervals/<name>/` visible; no leftover temp dirs); `overwrite` semantics.
- `gvl.write` with `variants` + `tracks` + `annot_tracks` together produces a dataset
  byte-identical (or equivalent) to the pre-parallel sequential path, and track spans
  match the variant-finalized bed.
- `gvl.write` parallel `max_mem` division stays within budget.
- bigwig-as-annot-source produces a sample-less annot track.

## Risks

- polars-bio overlap is the same backend flagged as intermittently segfaulting
  (issue #395, tracked in memory). Annot extraction in `write`/`update` now depends on
  it for table/DataFrame sources. The bigwig and per-sample (`BigWigs`) paths do not.
  Consider whether annot table extraction should be behind the same opt-in/experimental
  guard as `gvl.Table`, or gated to the `[table]` extra.
- loky + memmap writing across processes: each job writes its own subdir, so there is
  no shared-state hazard, but child processes must reconstruct readers (genoray /
  pyBigWig / polars-bio) independently. Confirm picklability of the job closures and
  reader objects.
