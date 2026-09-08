# Interval (BigWigs/Table) streaming + mixed variant+interval scheduler

**Issue:** [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279) — spec C, the last
major backend family for the write-free `StreamingDataset`.
**Roadmap:** `docs/roadmaps/streaming-dataset.md` (fills the `_TBD_` row in the Specs table).
**Date:** 2026-09-08

## Goal

`StreamingDataset` streams variants only. Extend it to read intervals directly from
`BigWigs` / `Table` sources and reconstruct tracks on the fly — including tracks
re-aligned to haplotype coordinates when indels are present — with **byte-identical
parity** against `gvl.write(bed, variants, tracks)` + `Dataset.open()[r, s]` at
`jitter=0`.

## Summary of decisions

| Decision | Choice |
|---|---|
| Constructor argument | `tracks=` accepting `IntervalTrack \| Sequence[IntervalTrack]` (**not** the issue's `bigwigs=`) |
| Number of new backends | **One** — `_TrackBackend`, keyed on the `IntervalTrack` Protocol, not one per class |
| New Rust reconstruction kernel | **None** — `intervals_and_realign_track_fused` already takes the arrays streaming holds |
| `iteration_order` | `Literal["auto", "regions", "samples"] = "auto"` |
| `auto` resolution | variants-only → `regions`; tracks-only → `samples`; mixed → `regions` (documented non-optimal) |
| Mixed-source composition | Both backends driven off the **same** `_plan()` step; a track window cursor tracks the variant engine's batch counter |
| Blocking prerequisite | `RustTable` must cache COITrees per contig (see "The `Table` rebuild defect") |

## 1. `tracks=`, not `bigwigs=`

The issue names the new argument `bigwigs=`. Use `tracks=` instead.

`IntervalTrack` (`python/genvarloader/_types.py:119`) is already a structural Protocol —
`count_intervals` + `_intervals_from_offsets` + `name` / `samples` / `contigs` — and both
`BigWigs` and `Table` implement it. `gvl.write` already unified them behind
`tracks: IntervalTrack | Sequence[IntervalTrack]`. Naming the streaming argument `bigwigs=`
would (a) diverge from the write path users already know, (b) misdescribe a `Table`
argument, and (c) force a rename the first time someone passes a `Table`.

`tracks=` accepts a single `IntervalTrack` or a sequence of them, exactly like `gvl.write`.

## 2. One backend, not two

"BigWigs/Table streaming" is a single backend. `_TrackBackend` holds a
`list[IntervalTrack]` and calls only the two Protocol methods. Nothing in the read path
branches on the concrete class.

It follows the same `read_window` / `generate_batch` split as `_Svar1Backend`,
`_Svar2Backend`, `_VcfBackend`, and `_PgenBackend`:

```
read_window(r_idx, s_idx) -> RaggedIntervals            # window = READ granularity
generate_batch(..., lo, hi, output_length) -> Ragged    # batch = GENERATION granularity
```

`read_window` is exactly the two Protocol calls, mirroring `BigWigs.intervals()`:

```python
counts  = track.count_intervals(contig, starts, ends, sample=window_samples)
offsets = lengths_to_offsets(counts)
itvs    = track._intervals_from_offsets(contig, starts, ends, offsets, sample=window_samples)
# RaggedIntervals, shape (n_regions, n_samples, None)
```

Single-contig per call, matching the invariant every variant backend already enforces
(`_Svar1Backend.read_window` raises on a multi-contig window). `_plan()`'s contig-run
outer loop already guarantees this.

**Output shape.** Matching the written path exactly: one track yields a
`(batch, None)` Ragged; more than one yields `(batch, n_tracks, None)`, stacked along a
new track axis by `_ragged_stack_tracks` (`_dataset/_tracks.py:91`, `:419`). Track-axis
order is the order tracks were passed to `tracks=`, mirroring the written
`Tracks.active_tracks` dict-insertion order. Under `with_len(L)` tracks take the same
fixed length `L` as sequences; under `with_len("ragged")` a track cell's length is its
reconstructed length, which under realignment is the haplotype length, not the reference
length.

## 3. Reconstruction: no new Rust kernel

Two cases, both already implemented:

- **Tracks without variants** — `intervals_to_tracks`, no realignment. This is what
  `Tracks._call_float32` (`_dataset/_tracks.py`) does today.
- **Tracks with variants** — `intervals_and_realign_track_fused`
  (`_dataset/_reconstruct.py:257`). Its genotype inputs are `geno_offset_idx`,
  `geno_v_idxs`, `geno_offsets`, `v_starts`, `ilens`. **The streaming variant backends
  already hold every one of these** (`_streaming.py:1788-89` holds `_v_starts` / `_ilens`;
  `read_window` returns the CSR offsets). So the mixed path is a wiring problem, not a
  kernel problem.

`offset_idxs` follows the written path's rule: `TrackType.SAMPLE` → per-cell index;
otherwise → per-region index.

**Two settings change track bytes and must therefore be part of v1**, because parity is
undefined without them:

- **`realign_tracks`** (`Dataset` default `True`). When `False`, the mixed case degrades
  to `intervals_to_tracks` — reference-coordinate track values alongside haplotype
  sequences. Exposed on `StreamingDataset.with_settings(realign_tracks=)` with the same
  default and the same meaning.
- **`insertion_fill`** (per track, default `Repeat5p()`). It selects what a realigned
  track emits across an insertion, so it is a byte-level input to
  `intervals_and_realign_track_fused` via `strategy_id` / `params`. Exposed as
  `with_insertion_fill(...)` with the written path's signature and defaults. The written
  path raises when `with_insertion_fill` is combined with `realign_tracks=False`;
  streaming raises the same error.

**Parity of unsupported modes.** `RaggedTracks` splicing raises
`NotImplementedError("Splicing of RaggedIntervals tracks is not supported.")` in the
written path. Streaming raises the same error with the same message — parity includes
parity of failure.

## 4. The scheduler

### 4.1 What `_plan()` does today

`_plan()` (`_streaming.py:566`) nests **contig-run → region-window → sample-chunk**. That
is region-major. Sample-major is the same three loops with the inner two swapped:

```python
for r_lo, r_hi in contig_runs:              # UNCHANGED in both orders
    if order == "regions":
        for w in region_windows(r_lo, r_hi):
            for s in sample_chunks():
                yield w, s
    else:                                    # order == "samples"
        for s in sample_chunks():
            for w in region_windows(r_lo, r_hi):
                yield w, s
```

The contig-run outer loop stays outermost in **both** orders. It is required by the
single-contig Rust invariant, and it is what makes the per-contig `Table` tree cache
(§5) effective.

The set of windows is identical between the two orders. Only the visit order changes.

### 4.2 `iteration_order`

```python
StreamingDataset(..., iteration_order: Literal["auto", "regions", "samples"] = "auto")
```

`auto` resolves at construction from the source mix:

| Sources | `auto` resolves to | Why |
|---|---|---|
| variants only | `regions` | Variant stores are position-major; a region window is one contiguous CSR span |
| tracks only | `samples` | `BigWigs` holds **one file per sample**; sample-major keeps a single file's B-tree and zoom index hot and walks it front-to-back |
| variants + tracks | `regions` | The roadmap's decision. Documented as non-optimal for the track axis; `iteration_order="samples"` is the escape hatch |

Explicit `"regions"` / `"samples"` always override. Any other value raises `ValueError`
at construction with the accepted literals in the message.

`auto` keys off the **source mix**, never off per-class micro-properties. That keeps the
rule predictable and stateable in one sentence of documentation.

### 4.3 Why `Table` does not change the rule

`Table` is fully in memory. `Table.from_path` (`_table.py:98`) uses eager
`pl.read_csv` / `read_parquet` / `read_ipc` — no `scan_*`, no mmap, no retained file
handle. The DataFrame is then copied into `RustTable`, which owns
`Vec<ContigStore> → Vec<SampleIntervals> → Vec<i32>/Vec<f32>` (`src/tables.rs:11-27`).

`Table.__init__` sorts by `("chrom", "sample_id", "start")`, which looks like a
sample-major locality hint. It is not one for iteration order. That row order is
scattered at build time into per-`(contig, sample)` contiguous vectors; only the `start`
component survives as a property the query code depends on (`tables.rs:137`:
*"ascending stored index == ascending start"*). Once `RustTable` is built the
DataFrame's row order is dead, and each `(contig, sample)` cell is its own contiguous
array reachable in O(1).

So `Table` is genuinely order-indifferent, and `auto`'s "tracks only → samples" rule is
justified by `BigWigs`' per-sample files alone. `Table` is never worse under it.

## 5. The `Table` rebuild defect (blocking, in scope)

`RustTable::build_trees(chrom)` (`src/tables.rs:57`) constructs a COITree **for every
sample on the contig**, and both `count` and `intervals_from_offsets` call it on **every
invocation**. Its cost is O(all intervals on that contig across *all* samples),
independent of how many samples the window actually selects.

Under a streaming sweep with `_window_samples = 1`, that is
`ceil(n_regions / window_regions) × n_samples` full all-sample rebuilds per contig —
quadratic in cohort size. Changing iteration order does **not** help: the window count is
identical in both orders.

This has not bitten anyone yet because `gvl.write` never calls these methods —
`_write.py:1567` dispatches `Table` to `_rust.write_track`, and `write_track_impl`
already caches trees across a contig run (`tables.rs:170-180`, the `cur_chrom` guard).
**Streaming would be the first real consumer of `Table.count_intervals` /
`Table._intervals_from_offsets`.**

**Requirement:** `RustTable` memoizes trees for the most recently queried contig
(interior mutability, one contig resident at a time), mirroring `write_track_impl`'s
existing guard. Because `_plan()` is contig-run-major in both iteration orders, that
yields exactly one rebuild per contig per sweep.

Rejected alternative: building all contigs' trees eagerly at `Table` construction. It
makes construction cost and resident memory scale with the whole table rather than with
the contig being read, and penalizes callers who query one contig.

## 6. Mixed variant + interval composition

Both backends are driven off the **same** `_plan()` step with the same `(r_idx, s_idx)`.
Composition, not a new scheduler.

The complication is the `"engine"` prefetch drive. `_iter_batches` materializes the
entire plan up front into `engine_jobs` and hands it to `backend.build_engine(...)`; the
Rust producer thread then owns the window loop and Python only drains `.next_batch()`.
There is no per-window Python callback inside that drive.

**Approach:** a track window cursor on the Python side. The batch spans are already
derivable without reconstructing anything — `_iter_batch_spans` walks `_plan()` and
yields each batch's row count. The track backend keeps its own cursor over the same
plan and issues a `read_window` when the drained batch counter crosses a window
boundary. The engine's batch order is deterministic and derived from that same plan, so
the two stay in lockstep by construction.

This keeps the measured engine win for the variant axis and adds tracks without a new
Rust seam. The track read holds the GIL, so it serializes against the consumer;
**a track producer thread is a measured follow-up, not v1** — the same shape as how
SVAR2's PR-3 engine shipped gated off pending measurement.

## 7. Order invariants

These already hold and become gated invariants:

1. **Jitter is keyed by region index, not plan step.** One vectorized draw per `to_iter`
   call, indexed by each region's absolute sweep index. A region gets the same offset
   however many times the plan revisits it, so jitter output is independent of
   iteration order, `max_mem`, and cohort-driven chunking.
2. **`return_indices` yields original BED-row order.** `_sort_order` maps sorted position
   back to the original row, so consumers never see iteration order leak into indices.
3. **`n_batches(batch_size)` is order-invariant.** The window set is identical between
   orders, so the multiset of batch spans is identical; only their sequence changes.

## 8. Parity oracle

Byte-identical against `gvl.write(bed, variants=..., tracks=...)` +
`Dataset.open()[r, s]` at `jitter=0`.

Because iteration order differs from the written dataset's index order **by design**,
the comparison is over the **set** of `(region, sample)` cells — collect the streamed
cells keyed by their returned indices and compare each against `Dataset[r, s]`.

Matrix:

| Axis | Cases |
|---|---|
| Track source | `BigWigs`, `Table` |
| Variants | absent (`intervals_to_tracks`); present without indels; **present with indels** (the realignment case #279 calls out) |
| Iteration order | `regions`, `samples` — both must produce the same cell set |
| Track count | one track (`(batch, None)`); a sequence of tracks (`(batch, n_tracks, None)`, track-axis order = argument order) |
| Output length | `with_len("ragged")`; `with_len(L)` |
| Track settings | `realign_tracks=True` / `False`; default `Repeat5p()` insertion fill and at least one non-default fill |

Plus: `auto` resolves as specified for each of the three source mixes; an invalid
`iteration_order` raises at construction; `RaggedTracks` splicing raises the written
path's exact `NotImplementedError`.

## 9. Out of scope

- **A track producer thread** (GIL-free prefetch of interval windows). Follow-up, gated
  on measurement, mirroring SVAR2 PR-3.
- **Per-hap within-window sub-shifts** (`deterministic=False`). Already a documented
  Wave A deferral on the variant axis; tracks inherit it unchanged.
- **`Table` streaming from disk.** `Table` is in-memory by construction (§4.3). A
  `scan_parquet`-backed lazy `Table` is a separate feature with its own parity story.
- **Track *subsetting*** (`with_tracks(names)`-style activation on an already-constructed
  `StreamingDataset`). Streaming reads exactly the tracks passed to `tracks=`; to read a
  different set, construct with a different set. Multi-track *output* is in scope (§2);
  only post-hoc selection is deferred. `realign_tracks` and `insertion_fill` are **in**
  scope (§3) because they change bytes.

## 10. Documentation

Per `CLAUDE.md`'s public-API rules, the implementation PR must update:

- `skills/genvarloader/SKILL.md` — the new `tracks=` and `iteration_order=` arguments.
- `docs/source/*.md` — `dataset.md` / `api.md` as applicable; `api.md` stays in sync
  with `__all__`.
- `docs/roadmaps/streaming-dataset.md` — replace the `_TBD_` Specs row with a pointer to
  this file, and mirror status on the StreamingDataset project board.
