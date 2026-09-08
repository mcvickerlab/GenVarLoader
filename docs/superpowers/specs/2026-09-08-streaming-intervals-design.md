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
| New Rust reconstruction kernel | **None**, but the mixed path needs a Python-side `variant_idxs` memmap and a window-local index re-base (§3) |
| Mixed variants + tracks | **SVAR1 only in v1.** VCF / PGEN / SVAR2 raise `NotImplementedError`. Tracks *without* variants work on every backend |
| Track axis order | Lexicographic by `track.name` — matching the written path's sorted `available_tracks`, **not** `tracks=` argument order |
| `iteration_order` | `Literal["auto", "regions", "samples"] = "auto"`; a no-op unless the sample axis is actually chunked (§4.4) |
| `auto` resolution | variants-only → `regions`; tracks-only → `samples`; mixed → `regions` (documented non-optimal) |
| Mixed-source composition | The track read goes in the per-window Python loop every drive already has (§6) — no cursor, no new seam |
| Blocking prerequisite | `RustTable` must cache COITrees per contig behind a `Mutex` (§5) |

## 1. `tracks=`, not `bigwigs=`

The issue names the new argument `bigwigs=`. Use `tracks=` instead.

`IntervalTrack` (`python/genvarloader/_types.py:119`) is already a structural Protocol —
`count_intervals` + `_intervals_from_offsets` + `name` / `samples` / `contigs` — and both
`BigWigs` and `Table` implement it. `gvl.write` already unified them behind
`tracks: IntervalTrack | Sequence[IntervalTrack]`. Naming the streaming argument `bigwigs=`
would (a) diverge from the write path users already know, (b) misdescribe a `Table`
argument, and (c) force a rename the first time someone passes a `Table`.

`tracks=` accepts a single `IntervalTrack` or a sequence of them, exactly like `gvl.write`.

Note the asymmetry: `gvl.write` accepts the Protocol in its *signature* but `_write_track`
`isinstance`-dispatches to `BigWigs` / `Table` and raises `TypeError` for anything else
(`_write.py:1594-1616`). Streaming calls only the Protocol methods, so it admits sources
`gvl.write` rejects — for which there is **no parity oracle**. v1 keeps the same two
concrete sources in the parity matrix and documents that a third-party `IntervalTrack`
is unvalidated.

## 2. One backend, not two

"BigWigs/Table streaming" is a single backend. `_TrackBackend` holds a
`list[IntervalTrack]` and calls only the two Protocol methods. Nothing in the read path
branches on the concrete class.

It follows the same `read_window` / `generate_batch` split as `_Svar1Backend`:

```
read_window(r_idx, s_idx) -> RaggedIntervals            # window = READ granularity
generate_batch(..., lo, hi, output_length) -> Ragged    # batch = GENERATION granularity
```

`read_window` is exactly the two Protocol calls, mirroring `BigWigs.intervals()`:

```python
names   = [self._samples[i] for i in s_idx]          # query by NAME, never by index
counts  = track.count_intervals(contig, starts, ends, sample=names)
offsets = lengths_to_offsets(counts.ravel())          # ravel first, cf. _bigwig.py:221
itvs    = track._intervals_from_offsets(contig, starts, ends, offsets, sample=names)
# RaggedIntervals, shape (n_regions, n_samples, None)
```

Single-contig per call, matching the invariant every variant backend already enforces
(`_Svar1Backend.read_window` raises on a multi-contig window, `_streaming.py:2213-2218`).
`_plan()`'s contig-run outer loop already guarantees this.

### 2.1 Sample identity

`s_idx` is a public index into `StreamingDataset.samples` (lexicographically sorted).
Tracks are queried **by name**, never by index. This is the track-side analogue of
`_Svar1Backend._phys_sample_idx` (`_streaming.py:1739-48`), and it is load-bearing
because the two implementations disagree about their own sample order:

- `BigWigs.samples = list(self.paths)` — **dict insertion order** (`_bigwig.py:41`).
- `Table.samples = sorted(...)` — sorted (`_table.py:58`).

Querying by name is also what `gvl.write` does (`_write.py:1542`:
`paths = [track.paths[s] for s in samples]`).

**Sample-set reconciliation.** With variants present the dataset's sample list is the
variant source's, and each track must cover it. Streaming adopts `gvl.write(...)`'s
**intersection** rule (`_write.py:341-346`) — a track with extra samples is fine; a track
missing a dataset sample raises `ValueError` naming the missing names — and *not* the
add-tracks path's exact-equality rule (`_write.py:536-544`). Parity fixtures must include
a track whose `samples` is a strict superset of the dataset's, and one in non-sorted order.

### 2.2 Contig coverage

The two sources fail differently today: `BigWigs` raises
`ValueError("Contig … not found")` (`_bigwig.py:157-161`) while `Table` maps an unknown
contig to code `-1` and silently returns zeros (`src/tables.rs:93-95`). `gvl.write` only
warns (`_write.py:340-346`).

`StreamingDataset` normalizes every BED contig against every track's `contigs` **once in
`__init__`** (via `normalize_contig_name`, which already handles UCSC `chr1` vs Ensembl
`1`) and raises one `ValueError` naming the track and the uncovered contigs.

`BigWigs.contigs` is the intersection across sample files (`_bigwig.py:43-55`) and
`Table.contigs` is `{chrom: max observed end}` (`_table.py:59-64`), not true contig
lengths. **Neither may be used as a coordinate bound.**

### 2.3 Output shape

The track axis is **unconditional**, including for a single track. `docs/source/dataset.md`
states the contract: *"tracks … have shape `(regions, samples, tracks, [ploidy], length)`"*.

| Mode | Streamed shape | Written analogue |
|---|---|---|
| tracks, no variants | `(batch, n_tracks, None)` | `Tracks._call_float32`, `_tracks.py:419` |
| tracks + variants, `realign_tracks=True` (default) | `(batch, n_tracks, ploidy, None)` | `HapsTracks.__call__`, `_reconstruct.py:292-297` |
| tracks + variants, `realign_tracks=False` | `(batch, n_tracks, None)` | `SeqsTracks` → `_call_float32`, `_reconstruct.py:545` |

A single track is `(batch, 1, …)`, never squeezed: `_ragged_stack_tracks`
(`_tracks.py:125-134`) *promotes* one track to `(n_batch, 1, None)` rather than collapsing
it, and no `.squeeze` on a track axis exists anywhere in the read path.

`_ragged_stack_tracks` itself is only used for the `RaggedIntervals` output kind
(`_tracks.py:527-529`). The float-track interleave streaming must reproduce is the
`repeat(out_lengths, "b -> b t")` + `lengths_to_offsets` pattern
(`_tracks.py:412-414`, `_reconstruct.py:200-203`).

### 2.4 Track axis order

Track-axis order is the **lexicographic sort of `track.name`**, not `tracks=` argument
order. The written path's default `active_tracks` comes from `Tracks.from_path`'s
`available_tracks.sort()` (`_tracks.py:283`) — a sorted directory listing — not from
`gvl.write`'s argument order. `gvl.write(tracks=[bw_zeta, bw_alpha])` therefore yields
track-axis order `[alpha, zeta]`.

Streaming sorts by name at construction so the two agree. Passing tracks in
non-alphabetical order is legal and has no effect on the output axis. Duplicate
`track.name`s raise `ValueError` at construction — the written layout `intervals/<name>/`
makes them indistinguishable anyway.

### 2.5 `max_mem` and the track window

`max_mem` currently bounds only the genotype offsets buffer (`cell_bytes = ploidy * 16`,
`_streaming.py:476`). A track window at default settings is `_window_regions` regions ×
**all** samples of intervals, which is unbounded for a dense source, and for `BigWigs`
means `n_samples` concurrently-open readers per thread (the reader cache at
`src/bigwig.rs:18-21` is unbounded and per-rayon-worker).

v1 folds an estimated per-cell interval cost into the `max_cells` derivation — a measured
bytes-per-(region, sample) from a first `count_intervals`, or a conservative constant — so
`_window_samples` shrinks for track-bearing datasets. Without this, `max_mem` stops
meaning anything the moment tracks are attached, **and** `iteration_order` stays a no-op
because `_window_samples` never drops below `n_samples` (§4.4). The two issues resolve
together.

## 3. Reconstruction

**Tracks without variants** — `intervals_to_tracks`, no realignment. This is what
`Tracks._call_float32` (`_dataset/_tracks.py`) does today. Works on every backend.

**Tracks with variants** — `intervals_and_realign_track_fused` (`_reconstruct.py:257`).
**v1 supports this for the SVAR1 backend only.** All five genotype inputs are obtainable
there without a new kernel, but two are not simply lying around as the earlier draft of
this spec claimed:

| Kernel arg | Written path | Streaming (SVAR1) |
|---|---|---|
| `v_starts` / `ilens` | `Haps.variants` | `_Svar1Backend._v_starts` / `._ilens` (`_streaming.py:1788-89`) — same global tables, `int32` |
| `geno_v_idxs` | `haps.genotypes.data` (memmap) | **NEW**: Python must `np.memmap(svar/variant_idxs.npy, V_IDX_TYPE)`. Today the backend only `stat()`s this file (`_streaming.py:2083-86`); the array itself stays a zero-copy mmap *inside* the Rust `Svar1Store` (`_streaming.py:2035-39`) |
| `geno_offsets` (2, N) | `_as_starts_stops(genotypes.offsets)`, dataset-global | `np.stack([o_starts, o_stops])` from `read_window` — already a starts/stops pair, but **window-local** |
| `geno_offset_idx` (b, p) | `ravel_multi_index((r,s,ploid), (R,S,P))`, dataset-global (`_haps.py:776`) | `arange(lo*P, hi*P).reshape(hi-lo, P)` — **re-based to the window**, since `geno_offsets` is window-local |
| `diffs` (for `track_lengths`) | `get_diffs_sparse` via `_haplotype_ilens` (`_haps.py:540`) | the **same** `get_diffs_sparse`, called with the re-based arrays above |
| `shifts` (b, p) | `_prepare_request` | all-zeros (see below) |

So: no new *kernel*, but a Python-side `variant_idxs` memmap, a window-local index
re-base, and a second existing kernel (`get_diffs_sparse`).

`shifts` is all-zeros for every combination v1 supports — `_haps.py:740-742` returns zeros
whenever `deterministic` is `True` (streaming's default) **or** output is ragged. This is
exactly why `with_len(L)` costs no new Rust seam. It is also why `deterministic=False`
(§9) is not merely a deferred feature but a hard blocker for this design: enabling it
would require the streaming engine to return per-hap `shifts` to Python, which no engine
does today.

`offset_idxs` is always the per-cell (region, sample) index, re-based to the window:
`arange(lo, hi)` over the window's C-order rows. The written path's `TrackType.ANNOT` →
per-region branch (`_tracks.py:403`, `:516`) is unreachable here — `ANNOT` is produced only
by `gvl.write(annot_tracks=)` (`_tracks.py:311-13`), never by `tracks=`.

**VCF / PGEN / SVAR2 mixed variants+tracks is out of scope for v1.** None of those
backends surfaces genotypes to Python: `_VcfBackend` / `_PgenBackend` have no
`read_window` / CSR seam at all (`_streaming.py:2624-2634` says so explicitly), and
`_Svar2Backend` reconstructs read-bound inside Rust and uses a *different* written-path
track kernel (`_reconstruct.py::_call_svar2`, which splits into `intervals_to_tracks` +
`shift_and_realign_tracks_from_svar2_readbound`). Combining `tracks=` with a non-SVAR1
variant source raises `NotImplementedError` at `to_iter` time, in the style of the
existing SVAR2/VCF output-mode guards (`_streaming.py:735-747`, `:750-762`).

### 3.1 Settings that change track bytes

Two are in scope because parity is undefined without them:

- **`realign_tracks`** (`Dataset` default `True`). When `False`, the mixed case degrades
  to `intervals_to_tracks` and the output loses its ploidy axis (§2.3). Exposed on
  `with_settings(realign_tracks=)` with the same default and meaning.
- **`insertion_fill`** (per track, default `Repeat5p()`). Selects what a realigned track
  emits across an insertion, so it is a byte-level input to the fused kernel via
  `strategy_id` / `params`. Exposed as `with_insertion_fill(...)` with the written path's
  signature and defaults. The written path raises
  `ValueError("with_insertion_fill has no effect when realign_tracks=False…")`
  (`_impl.py:884-889`); streaming raises the same.

**`jitter`** is not a byte-parity setting but *is* a correctness hazard for tracks: the
jittered bounds are applied only when building `engine_jobs` (`_streaming.py:806-816`),
while `plan_jobs` keeps unjittered `r_idx` (`:785`). The track read must translate its
query bounds with the **same** per-region offsets via `_jitter_region_bounds`
(`_streaming.py:614-660`) — those offsets are keyed by absolute region index, so this is a
lookup, not a redraw. A test must assert a jittered track window's values equal the
unjittered values of the translated region; otherwise tracks and haplotypes silently
disagree by up to `jitter` bases.

### 3.2 Which reference span the track read covers

`gvl.write` does **not** store intervals over the user's BED. It stores them over
`gvl_bed` *after* the variant step has extended each `chromEnd` to cover spanning
deletions (`_write.py:409-431` reassigns `gvl_bed`; `_write.py:1167` / `:1349` apply
`max_horizontal(max_ends, chromEnd)`; `extend_to_length` defaults to `True`,
`_write.py:194`), and after `max_jitter` widening (`_write.py:673-677`).
`StreamingDataset._regions` is the raw sorted input BED (`_streaming.py:411-416`).

This is observable in the output bytes, because the realign kernel's scratch buffer runs
past the region end by the max deletion length
(`track_lengths = lengths - diffs.clip(max=0).min(1)`, `_reconstruct.py:189`): positions
the written dataset covered with real interval values would read as zeros from an
unextended streaming read.

v1 resolution: the track `read_window` queries `[chromStart, chromEnd + max_del)` where
`max_del` comes from the same `get_diffs_sparse` result used for `track_lengths`, and
**v1 parity is gated on `gvl.write(..., extend_to_length=False, max_jitter=None)`**.
Parity against the default `extend_to_length=True` requires reproducing write-time
`max_ends`, which is a whole-cohort scan — precisely what a write-free design exists to
avoid — and is deferred with its own issue.

### 3.3 Guards mirrored from `_build_reconstructor`

`with_seqs` accepts `"variants"` and `"variant-windows"` (`_streaming.py:1368-70`), so
these combinations are reachable and must raise the written path's exact `ValueError`s
(`_reconstruct.py:522-546`):

| Combination | Written behaviour |
|---|---|
| `with_seqs("variant-windows")` + tracks + `realign_tracks=True` | `ValueError`, `_reconstruct.py:538-543` |
| `with_seqs("variants")` + tracks + `realign_tracks=True` | `ValueError`, same message shape |
| interval-kind tracks + `realign_tracks=True` | `ValueError`, `_reconstruct.py:527-533` |

v1 emits only float (`RaggedTracks`) output; a `kind="intervals"` streaming option is
deferred (§9), so the third row is a construction-time rejection.

**Splicing is out of scope, not "parity of failure".** `StreamingDataset` has no splicing
support at all — zero occurrences of `splice` in `_streaming.py` — so there is no
written-path error to mirror.

## 4. The scheduler

### 4.1 What `_plan()` does today

`_plan()` (`_streaming.py:566`) nests **contig-run → region-window → sample-chunk**. That
is region-major. Sample-major swaps the inner two — but not naively:

```python
for r_lo, r_hi in contig_runs:              # UNCHANGED in both orders
    # Hoisted so BOTH orders share one `r_idx` array object per region window.
    # `_plan` today allocates `r_idx` once per region window and re-yields it for
    # every sample chunk (`_streaming.py:584`); `plan_jobs` stores that same object
    # n_sample_chunks times, which is what keeps job residency at
    # O(n_region_windows x window_regions). A naive loop swap that allocates
    # `arange` inside the inner loop multiplies that by n_sample_chunks.
    windows = [np.arange(lo, min(lo + W, r_hi), dtype=np.intp)
               for lo in range(r_lo, r_hi, W)]
    if order == "regions":
        for w in windows:
            for s in sample_chunks(): yield w, s
    else:
        for s in sample_chunks():
            for w in windows: yield w, s
```

The contig-run loop stays outermost in **both** orders: it is required by the
single-contig Rust invariant, and it is what makes the per-contig `Table` tree cache (§5)
effective.

The set of windows is identical between the two orders; only the visit order changes. The
`engine_jobs` region-bounds arrays (`_streaming.py:806-822`) are allocated per job in both
orders and are transient (`del engine_jobs`), so they are unaffected.

### 4.2 `iteration_order`

```python
StreamingDataset(..., iteration_order: Literal["auto", "regions", "samples"] = "auto")
```

`auto` resolves at construction from the source mix:

| Sources | `auto` resolves to | Why |
|---|---|---|
| variants only | `regions` | Variant stores are position-major; a region window is one contiguous CSR span |
| tracks only | `samples` | `BigWigs` holds one file per sample; sample-major walks a single file front-to-back |
| variants + tracks | `regions` | The roadmap's decision. Documented as non-optimal for the track axis; `iteration_order="samples"` is the escape hatch |

Explicit `"regions"` / `"samples"` always override. Any other value raises `ValueError` at
construction with the accepted literals in the message.

`auto` keys off the **source mix**, never off per-class micro-properties.

### 4.3 Why `Table` does not change the rule

`Table` is fully in memory. `Table.from_path` (`_table.py:98`) uses eager `pl.read_csv` /
`read_parquet` / `read_ipc` — no `scan_*`, no mmap, no retained file handle. The DataFrame
is then copied into `RustTable`, which owns
`Vec<ContigStore> → Vec<SampleIntervals> → Vec<i32>/Vec<f32>` (`src/tables.rs:11-27`).
(`Table.__init__` also *retains* `self._df` (`_table.py:56`), so a `Table` holds the polars
frame **and** the Rust copy — roughly 2× resident.)

`Table.__init__` sorts by `("chrom", "sample_id", "start")`, which looks like a
sample-major locality hint. It is not one for iteration order. That row order is scattered
at build time into per-`(contig, sample)` contiguous vectors; only the `start` component
survives as a property the query code depends on (`tables.rs:137`: *"ascending stored index
== ascending start"*). Once `RustTable` is built the DataFrame's row order is dead, and
each `(contig, sample)` cell is its own contiguous array reachable in O(1).

So `Table` is order-indifferent, and `auto`'s tracks-only rule rests on `BigWigs`'
per-sample files alone. `Table` is never worse under it.

### 4.4 `iteration_order` only bites when the sample axis is chunked

`_window_samples` is **derived from `max_mem`**, not from the field default of 1
(`_streaming.py:468-483`):

```python
max_cells      = max(1, max_mem_bytes // (cell_bytes * n_slots))  # 512MB/(2*16*2) ≈ 8.4e6
window_samples = max(1, min(int(n_samples), max_cells))
window_regions = max(1, min(region_target, max_cells // window_samples))
```

At the default `max_mem="512MB"` with `ploidy=2`, `max_cells ≈ 8.4e6`, so
**`_window_samples == n_samples` for any cohort below ~8.4 M**. The sample loop then runs
exactly once and the two orders emit the *identical* plan — `iteration_order` is a no-op.

It becomes meaningful only at cohort scale, with a deliberately small `max_mem`, or once
§2.5's track-memory accounting lands and starts shrinking `_window_samples` for
track-bearing datasets. **This must be stated in the `iteration_order` docstring**;
otherwise a user setting `"samples"` on a 100-sample dataset, measuring no change, and
filing a bug is the expected outcome.

### 4.5 Constructing a tracks-only `StreamingDataset`

Not possible today; v1 must change the constructor:

- `__init__` raises when `variants is None` (`_streaming.py:400-405`); it must accept
  `tracks=` alone.
- `reference` becomes optional — it is only needed to reconstruct sequence.
- `contigs` falls back to the intersection of the tracks' `contigs`; `samples` /
  `n_samples` to the sorted intersection of the tracks' `samples` (mirroring
  `gvl.write`'s `available_samples.intersection_update`, `_write.py:346`); `ploidy` is
  undefined and must not be read — `shape` stays `(n_regions, n_samples)` and the output
  carries no ploidy axis.
- `cell_bytes` (`_streaming.py:476`) is a genotype-offsets model; a tracks-only dataset
  must declare a track-side per-cell cost instead, or `max_mem` bounds nothing (§2.5).
- There is no way to *disable* sequence output today — `with_seqs` accepts only
  `"haplotypes" | "annotated" | "variants" | "variant-windows"` (`_streaming.py:1368-70`),
  no `None` / `"reference"`. A tracks-only dataset yields tracks alone from `to_iter`
  without a `with_seqs` call, and calling `with_seqs(...)` on one raises.

## 5. The `Table` rebuild defect (blocking, in scope)

`RustTable::build_trees(chrom)` (`src/tables.rs:57`) constructs a COITree **for every
sample on the contig**, and both `count` (`tables.rs:96`) and `intervals_from_offsets`
(`tables.rs:126`) call it unconditionally on **every** invocation. Its cost is O(all
intervals on that contig across *all* samples), independent of how many samples the window
selects.

Under a streaming sweep the rebuild count is
`ceil(n_regions / _window_regions) × ceil(n_samples / _window_samples)` per contig. Per
§4.4, `_window_samples == n_samples` at default `max_mem`, so the practical cost is
`ceil(n_regions / _window_regions)` all-sample rebuilds per contig — **linear in region
count, not quadratic in cohort size**. It is worst at cohort scale precisely because
`_window_regions` falls toward 1 there (one full all-sample rebuild per region).

This has not bitten anyone because `gvl.write` never calls these methods —
`_write.py:1606-1609` dispatches `Table` to `_write_track_table`, which calls
`track._rust.write_track` (`_write.py:1583`), and `write_track_impl` already caches trees
across a contig run (`tables.rs:169-181`, the `cur_chrom` guard). A repo-wide grep for
`count_intervals` / `_intervals_from_offsets` outside `_bigwig.py` / `_table.py` /
`_types.py` returns **zero** hits: **streaming would be the first real consumer.**

**Requirement:** `RustTable` memoizes the most recently queried contig's trees in a
`std::sync::Mutex<Option<(usize, Vec<BasicCOITree<u32, u32>>)>>`, mirroring
`write_track_impl`'s `cur_chrom` guard and the codebase's existing pyclass-mutable-state
pattern (`src/ffi/stream_core.rs:124`, `:195`; `svar2_stream_engine.rs:170`) — **not** a
`RefCell`: `#[pyclass]` on pyo3 0.29 plus the free-threaded story make `Sync` the safer
bound, and the only `RefCell` in `src/` is a `thread_local!` (`bigwig.rs:18-21`).

Contention is a non-issue today because `py_count` / `py_intervals` (`tables.rs:260-296`)
never call `py.detach`, so calls are already fully serialized — which is also why §6's GIL
note matters. If a track producer thread ever lands (§9), the single-slot cache becomes a
thrash point for two threads on different contigs and must grow to a small per-contig map.

Because `_plan()` is contig-run-major in both iteration orders, this yields exactly one
rebuild per contig per sweep.

Rejected alternative: building all contigs' trees eagerly at `Table` construction. It
makes construction cost and resident memory scale with the whole table rather than with
the contig being read.

## 6. Mixed variant + interval composition

Both backends are driven off the **same** `_plan()` step. There is no scheduling problem
to solve, because every drive already has a per-window Python loop.

The `"engine"` drive materializes the plan up front (`plan_jobs`, `_streaming.py:783-788`)
and hands region bounds to `backend.build_engine(...)`, but Python still walks the plan
window by window to pack each batch:

```python
# _streaming.py:875
for _contig_idx, r_idx, s_lo, s_hi in plan_jobs:
    ...
    for lo in range(0, n_rows, batch_size):
        nxt = next_batch()
```

Every drive has this shape — `"engine"` (`:875`), `"readahead"` (`:1129`), `"sync"`
(`:1167`), `"svar2_engine"` (`:1236`), and the injected-callback test path (`:1274`). The
track `read_window(r_idx, arange(s_lo, s_hi))` is issued at the top of that existing loop
body, where the window's identity is already in hand, and its rows are sliced `[lo:hi]`
alongside the variant batch. **No cursor, no batch counter, no lockstep assumption.**

A cursor keyed on a drained-batch counter — the earlier draft's design — would also be
*wrong* for `_Svar2Backend`, whose default drive is `"sync"` (`_streaming.py:2318`) and
whose batches are nested inside super-batches (`:1174-1188`): the batch counter has no
fixed relationship to window boundaries there.

Batches never span windows, so a window's last batch is partial by construction and the
track slice is just `[lo:hi]` of that window's rows.

The track read holds the GIL — `RustTable::py_count` / `py_intervals` do not call
`py.detach` (`src/tables.rs:260-296`) — so it serializes against the variant engine's
consumer for the full duration of a COITree rebuild. **A track producer thread is a
measured follow-up, not v1**, mirroring how SVAR2's PR-3 engine shipped gated off pending
measurement.

## 7. Order invariants

These already hold and become gated invariants:

1. **Jitter is keyed by region index, not plan step** (`_streaming.py:594-600`). One
   vectorized draw per `to_iter` call, indexed by each region's absolute sweep index. A
   region gets the same offset however many times the plan revisits it, so jitter output
   is independent of iteration order, `max_mem`, and cohort-driven chunking.
2. **`return_indices` yields original BED-row order** —
   `flat_r = np.repeat(self._sort_order[r_idx], n_s)` (`_streaming.py:877`).
3. **`n_batches(batch_size)` is order-invariant** — `_iter_batch_spans`
   (`_streaming.py:1361`) walks `_plan()`, so the batch-span multiset is identical between
   orders; only their sequence changes.

## 8. Parity oracle

Byte-identical against `gvl.write(bed, variants=..., tracks=..., extend_to_length=False,
max_jitter=None)` + `Dataset.open()[r, s]` at `jitter=0` (see §3.2 for why the write flags
are pinned).

Because iteration order differs from the written dataset's index order **by design**, the
comparison is over the **set** of `(region, sample)` cells — collect the streamed cells
keyed by their returned indices and compare each against `Dataset[r, s]`. Assert
`set(seen) == full cartesian product`, as the existing streaming parity test does
(`tests/dataset/test_streaming_parity.py:44-56`).

Three axes are not covered by "same set of cells" and need explicit assertions, because
they are where this design is most likely to be wrong:

1. **Track axis** — the fixture must use ≥2 tracks whose names are passed in
   **non-alphabetical** order, and must assert the streamed track axis matches the written
   dataset's sorted order (§2.4). An alphabetical fixture cannot distinguish the two rules.
2. **Length axis** — assert per-cell `len()` before comparing values. The written track
   length is `lengths - diffs.clip(max=0).min(1)` over the stored region
   (`_reconstruct.py:189`), so a length mismatch is the signature of §3.2's span bug and
   must fail loudly rather than as a value diff.
3. **Shape rank** — assert `ndim`, so a spurious squeeze or a missing ploidy axis fails as
   a shape error rather than silently broadcasting.

Matrix:

| Axis | Cases |
|---|---|
| Track source | `BigWigs`, `Table` |
| Variant source | SVAR1 (mixed supported); VCF / PGEN / SVAR2 (tracks+variants must raise `NotImplementedError`; tracks-only must work) |
| Variants | absent (`intervals_to_tracks`); present without indels; **present with indels** (the realignment case #279 calls out) |
| Iteration order | `regions`, `samples` — same cell set; and at default `max_mem` the two plans are provably identical (§4.4) |
| Track count | one track (`(batch, 1, …)` — axis NOT squeezed); a sequence passed in non-alphabetical order |
| Output length | `with_len("ragged")`; `with_len(L)` |
| Track settings | `realign_tracks=True` / `False` (note the rank change, §2.3); default `Repeat5p()` insertion fill and at least one non-default fill |
| Sample sets | a track whose `samples` is a strict superset of the dataset's; a track in non-sorted sample order |

Plus: `auto` resolves as specified for each source mix; an invalid `iteration_order`
raises at construction; the §3.3 guards raise the written path's exact `ValueError`s.

## 9. Out of scope

- **Mixed variants + tracks on VCF / PGEN / SVAR2** (§3). Follow-up issue per backend.
- **A track producer thread** (GIL-free prefetch of interval windows), gated on
  measurement, mirroring SVAR2 PR-3.
- **Parity against `extend_to_length=True` / `max_jitter>0`** (§3.2) — needs write-time
  `max_ends`, a whole-cohort scan. Its own issue.
- **`annot_tracks=`** (sample-independent annotation tracks). `gvl.write` supports them,
  stored under `annot_intervals/` with `TrackType.ANNOT` (`_tracks.py:285-293`). They need
  a per-region rather than per-cell `offset_idxs` and a separate source type (a path or
  DataFrame, not an `IntervalTrack`). Follow-up issue.
- **`kind="intervals"` streaming output** (`RaggedIntervals` rather than float tracks).
- **Spliced streaming** — `StreamingDataset` has no splicing at all (§3.3).
- **Per-hap within-window sub-shifts** (`deterministic=False`). Already a documented Wave
  A deferral; §3 explains why it is a hard blocker for the fused-kernel wiring, not just a
  missing feature.
- **`Table` streaming from disk.** `Table` is in-memory by construction (§4.3). A
  `scan_parquet`-backed lazy `Table` is a separate feature with its own parity story.
- **Track subsetting** (`with_tracks(names)` on a constructed `StreamingDataset`).
  Streaming reads exactly the tracks passed to `tracks=`. Multi-track *output* is in scope
  (§2.3); only post-hoc selection is deferred.

## 10. Documentation

Per `CLAUDE.md`'s public-API rules, the implementation PR must update:

- `skills/genvarloader/SKILL.md` — the new `tracks=` and `iteration_order=` arguments,
  including §4.4's caveat that `iteration_order` is a no-op unless the sample axis chunks.
- `docs/source/*.md` — `dataset.md` / `api.md` as applicable; `api.md` stays in sync with
  `__all__`.
- `docs/roadmaps/streaming-dataset.md` — replace the `_TBD_` Specs row with a pointer to
  this file, and mirror status on the StreamingDataset project board.

## Appendix: side finding to file separately

`n_batches` / `_iter_batch_spans` (`_streaming.py:1352-1366`) compute a flat
`range(0, n_rows, batch_size)` per window, but the `"sync"` and `"svar2_engine"` drives
nest batching inside super-batches (`:1174-1188`, `:1243-1250`). Whenever
`super_batch_rows` does not divide `n_rows`, `len(dl)` under-reports the batches actually
yielded for SVAR2 — the same class of bug `test_dataloader_len_matches_batches_yielded`
was written to catch for SVAR1. Not this spec's work; file as a `type: bug` issue on the
StreamingDataset board.
