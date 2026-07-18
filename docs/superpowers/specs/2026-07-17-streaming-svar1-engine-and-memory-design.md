# Design: `StreamingDataset` — wire the double-buffer engine + bound peak memory in cohort size

**Date:** 2026-07-17
**Status:** design (pending spec review)
**Roadmap:** `docs/roadmaps/streaming-dataset.md` → Plan 2
**Issues:** [#283](https://github.com/mcvickerlab/GenVarLoader/issues/283) (engine wiring — the pre-authorized split-out of #275 Task 5 Step 5), [#284](https://github.com/mcvickerlab/GenVarLoader/issues/284) (peak memory unbounded in cohort size)
**Follows:** [#275](https://github.com/mcvickerlab/GenVarLoader/issues/275) (Plan 2 — window reads + generic engine landed)
**Supersedes parts of:** `docs/superpowers/specs/2026-07-16-streaming-svar1-window-engine-design.md` (the "SVAR1 buffer is degenerate offsets *borrowed from the mmap*, nothing materialized" framing — see "Supersession" below)

## Summary

Two coupled follow-ups to Plan 2, folded into one design because they interact along a
single seam (the engine doubles resident windows; the window is what must become
cohort-independent):

1. **#284 — cohort-independent peak memory.** Today `reconstruct_haplotypes_svar1`
   materializes a *whole window's* haplotypes (`regions × ALL samples × ploidy × len`)
   in one allocation, so peak RAM scales with cohort size and OOMs at ~100k samples.
   Fix: **split the window read from generation** — read window offsets once (window
   granularity), then **generate per batch** (output ∝ `batch_size`) — and add a
   **sample-chunk window axis under a `max_mem` byte budget** so the offset buffer
   (∝ `window_samples`) is bounded too.

2. **#283 — overlap producer I/O with consumer generation.** Wire the landed generic
   engine (`StreamBackend` + `run_windows`, `src/stream/mod.rs`) so window N+1's read
   happens on a background thread while the consumer generates batches from window N.

The two ship as **two PRs along the read/generate seam**: PR 1 (#284) does the split +
budget single-threaded and is the parity-critical, independently-valuable memory fix;
PR 2 (#283) moves the read ahead onto a producer thread, **measurement-gated**.

Byte-identical parity with `gvl.write()` + `Dataset.open()[r, s]` (jitter=0) is the
control throughout, exactly as in Plan 2.

## Guiding principle: share the *mechanism*, not the *representation*

The landed `StreamBackend` trait already gets the format-generality boundary right:

```rust
pub trait StreamBackend: Sync {
    type Buffer: Send;                                             // per-backend
    fn fill(&self, window: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()>;
}
pub fn run_windows<B: StreamBackend, F>(backend: &B, windows: &[WindowSpec],
                                        n_slots: usize, consume: F) -> anyhow::Result<()>;
```

The **only** thing shared across formats is the *mechanism*: the producer/consumer
plumbing (`run_windows` — named thread, `crossbeam_channel::bounded`, slot recycling,
close-by-`Sender`-drop shutdown, join-then-classify-panics) plus the **traversal model**
(fixed cartesian region × sample sweep, byte-budget chunking). The **representation and
kernels stay per-backend**, via the associated `type Buffer`:

| Backend | `fill` (read) | `Buffer` | generate |
|---|---|---|---|
| **SVAR1** (this design) | offsets + `madvise(WILLNEED)` | two `Vec<i64>` (offsets into the shared `variant_idxs` mmap) | sparse → haplotype, **per batch** |
| **VCF / PGEN** ([#276](https://github.com/mcvickerlab/GenVarLoader/issues/276)) | read + **decode** | *its own* — decided on #276's benchmarks | sparse→hap (reuse) **or** a new dense→hap kernel — **not prejudged here** |

**Explicit non-goal / deferred to #276.** VCF/PGEN store *dense* genotypes; feeding
today's `reconstruct_haplotypes_from_sparse` requires a dense→sparse conversion, and a
new benchmarked dense→haplotype kernel may win instead. **This design does not choose,
and does not force a shared data representation.** That is #276's call, made on its own
evidence. `type Buffer` being associated is precisely what keeps that door open — the
engine is a shared harness that per-format pipelines plug into, not a shared pipeline.

## Supersession

`2026-07-16-streaming-svar1-window-engine-design.md` framed SVAR1's engine buffer as
"**degenerate — offsets only, `geno_v_idxs` *borrowed from the mmap*, nothing
materialized**," and left the buffer split (offsets-only vs. fully-reconstructed) as an
open question for #283. This design settles it:

| Prior framing | This design |
|---|---|
| Buffer split is an open question, re-decide first | **Settled: offsets-only, but the offsets buffer is an *owned* recycled slot** (`Vec<i64>`), while the *variant data* stays in the shared page cache. |
| "borrow offsets from the mmap" (lifetime-tangled with the store) | The buffer carries no borrow — it holds absolute indices; `geno_v_idxs` is fetched from the shared `Arc<Svar1Store>` at generate time. This is what lets the buffer be a `Send + Default` recycled slot. |
| Window = `regions × ALL samples`; window buffer holds the whole window's output | Window read is offsets-only and **sample-chunked**; **generation is per batch**, so output is never a whole-window allocation. |

Unchanged and still in force: region-major fixed traversal, reuse of
`reconstruct_haplotypes_from_sparse`, byte-identical parity as the oracle, `std::thread`
+ `crossbeam_channel::bounded` (not tokio), `variant_idxs` stays mmap'd (never
materialized).

## Why the gather is redundant for SVAR1 (the RAM floor)

An earlier draft proposed gathering each window's variant runs into an owned contiguous
`Vec<i32>`. **That copy does not exist today and is redundant for SVAR1:** the landed
path already passes `geno_v_idxs` as the `variant_idxs` **mmap slice itself**
(`Svar1Store::geno_v_idxs() -> &[i32]`), with `find_ranges` returning *absolute*
`(o_start, o_stop)` index pairs into it. The kernel indexes the mmap directly.

The mmap's backing is the OS **page cache — a process-wide, cross-thread shared heap.**
Once the producer faults window N+1's pages, they are resident and the consumer reads
them zero-copy. So:

- The buffer crossing the channel carries **only offsets** (`o_starts`, `o_stops`).
  `geno_v_idxs` stays the shared mmap, reached via the `Arc<Svar1Store>` both threads
  hold. This is `Svar1Window` minus its `geno_offset_idx` (which is the pure identity
  map `bi*ploidy + p` and is *derived*, not stored or sent).
- **Prefetch ≠ `find_ranges`.** `find_ranges`'s `partition_point`s touch only the
  binary-search path (a few pages per run); the consumer reads the *full*
  `[o_start, o_stop)` runs. To warm the exact pages the consumer needs, `fill` issues
  **`madvise(MADV_WILLNEED)`** over those byte ranges (memmap2's `advise_range`; a
  read-through loop is the portable fallback). Async kernel readahead, no copy, pages
  land in the shared, reclaimable page cache.

### Peak RAM accounting (allocated, non-reclaimable)

```
  n_slots × sizeof(offset buffer)   ← o_starts + o_stops, ∝ window_regions × window_samples × ploidy × 8B
+ 1       × batch output            ← ∝ batch_size × ploidy × mean_len   (per-batch generation)
+ static tables + ref               ← loaded once, shared (Arc)
──────────────────────────────────
+ shared page-cache working set     ← ∝ live windows' variant footprint; RECLAIMABLE, NOT a hard alloc,
                                      NOT multiplied by n_slots (same physical mmap pages)
```

**Cohort dependence lives entirely in the offset buffer** (∝ `window_samples`). It is
*not* negligible — at 100k samples × 64 regions × ploidy 2 it is ~200 MB — so
sample-chunking under `max_mem` genuinely earns its keep. The big variant data is never
copied or doubled; the big haplotype output is `batch_size`-bounded. This is the RAM
floor: it cannot be beaten without giving up overlap.

## Iteration & traversal model

Unchanged from Plan 2 in spirit — a fixed cartesian sweep of BED × samples,
region-major, single-contig per window — with **one addition: a sample-chunk axis.**

- **Window** = `window_regions` regions × `window_samples` samples × ploidy, cartesian,
  single-contig. One `read_window` (offsets) per window.
- **Batch** = a slice of a window, `batch_size` `(region, sample)` cells. Generation
  granularity. Never its own read.
- **Sample-chunking:** when a contig's regions are windowed, the sample axis is *also*
  chunked into `window_samples`-sized runs. `find_ranges` already accepts
  `samples: Option<&[usize]>`; `var_ranges` (region-only, sample-independent) is computed
  once per region-chunk and reused across that chunk's sample sub-windows.

`WindowSpec` gains a sample sub-range:

```rust
pub struct WindowSpec {
    pub contig_idx: usize,
    pub r_lo: usize, pub r_hi: usize,   // half-open region span on the contig
    pub s_lo: usize, pub s_hi: usize,   // half-open sample sub-range  (NEW)
}
```

### The `max_mem` byte budget

New public knob on `StreamingDataset`, replacing the `_window_regions = 64` placeholder
that Plan 2 shipped explicitly for this purpose:

```python
gvl.StreamingDataset(regions, reference=..., variants="x.svar", max_mem="512MB")
```

- Matches gvl's / genoray's existing `max_mem` idiom (`str | int`, e.g. `"512MB"`).
- `window_regions` and `window_samples` are **derived** from it so that
  `n_slots × offset_buffer_bytes ≤ max_mem`, where `n_slots = 1` in PR 1 (single window
  resident) and `n_slots = 2` in PR 2 (ping-pong). Sizing accounts for the resident-slot
  multiplier so the budget bounds *actual* peak, not per-window.
- The **default is measured, not guessed** (project perf convention): swept on the scale
  fixture, the knee recorded in the roadmap. The batch output (∝ `batch_size`, a
  `to_iter` argument) is a separate, user-controlled term and is not part of this budget.

## Part 1 — PR 1: cohort-independent memory (#284), single-threaded

The decisive change: **decouple read granularity (window) from generation granularity
(batch).** Read a window's offsets once; generate batch-by-batch off the shared mmap.
This caps output at `batch_size` while keeping the read amortized over the whole window —
and it is *exactly* the `fill` / `consume` seam PR 2 needs.

### Rust

- **`src/svar1/store.rs`** — `read_window` already returns offsets (`Svar1Window`). Drop
  the stored `geno_offset_idx` from the hot path (derive identity in the kernel call);
  keep `o_starts`/`o_stops`. Add `advise_range(WILLNEED)` over each run's byte span
  (feature-gated / portable fallback). `Svar1Store` is already `Send + Sync`.
- **`src/ffi/mod.rs`** — split the single `reconstruct_haplotypes_svar1` (read + whole-
  window generate) into two entry points:
  - `svar1_read_window(store, contig, v_starts_c, v_ends_c, region_bounds, sample_idx) -> (o_starts, o_stops)`
    — the offsets read (window granularity), inside `py.detach`, `store: PyRef<'py>`.
  - `svar1_generate_batch(store, <offset slice for the batch's rows>, region_bounds_batch,
    static tables, ref, pad_char, parallel) -> (data, offsets)` — generation for a batch,
    reading `geno_v_idxs` from the shared mmap, calling the unchanged
    `reconstruct_haplotypes_from_sparse`. Output ∝ `batch_size`.

  Static tables (`v_starts`/`ilens`/`alt_alleles`/`alt_offsets`) and ref bytes still
  cross as zero-copy `PyReadonlyArray1`; nothing sample-scale is copied.

### Python (`python/genvarloader/_dataset/_streaming.py`)

- `_plan` yields sample-chunked windows: `(region_idxs, sample_idxs)` where
  `sample_idxs` is now a *chunk*, not always all samples.
- `_iter_batches` calls `svar1_read_window` once per window, then loops
  `svar1_generate_batch` per `batch_size` slice — the window loop stays synchronous in
  PR 1.
- `max_mem` field + derivation of `window_regions` / `window_samples`; delete
  `_window_regions`.
- `_Svar1Backend` moves the static tables + ref into the store/engine ownership so PR 2
  can share them across threads without per-call marshalling.
- Parity test needs **no** behavioral change (only the same mechanical surface Plan 2
  already settled).

### Scope note (PR 1)

`num_workers > 0` stays deferred, `jitter = 0` only, `haplotypes` only, ragged only,
SVAR1 only — unchanged from Plan 2.

## Part 2 — PR 2: overlap producer I/O with consumer generation (#283)

Move the window read ahead of generation. **Which vehicle ships is measurement-gated.**

- **Design A — landed producer-thread engine.** Implement `StreamBackend for Svar1Store`
  (`Buffer = Svar1Window`-offsets, needs `Default`; `fill` = `read_window` +
  `madvise`). A new `#[pyclass] StreamEngine` owns `Arc<Svar1Store>` + the static tables
  + the `Vec<WindowSpec>` plan, spawns the producer via `run_windows`, and exposes
  `__next__` to the consumer: `recv()` a filled offset slot (GIL released), generate the
  next `batch_size` rows off the shared mmap (GIL released), wrap as `PyArray` (GIL held),
  recycle the slot when its window is exhausted. `_iter_batches` drives the pyclass
  instead of the synchronous loop.
- **Design C — single-thread `madvise` pipelining.** No producer thread: before
  generating window N, compute window N+1's offsets and issue `MADV_WILLNEED`, letting
  kernel readahead overlap with N's generate + the user's train step. Captures most of
  A's I/O win with **zero threading** (there is currently no `std::thread` in `src/`;
  the spec flags threading as the single highest-risk part).

**The measurement gate (cold page cache):** on a store larger than RAM (or with caches
dropped between runs), compare A vs C — wall-clock and overlap fraction. Ship whichever
wins for SVAR1 and **record the result**, because:

- If A ≫ C, the producer thread pays for itself on I/O alone.
- If A ≈ C, the thread's value for SVAR1 is marginal and is really about **#276's decode
  overlap** (`madvise` cannot hide decode CPU) — a finding worth pinning, since the
  landed engine is then validated/reserved for #276 rather than justified by SVAR1.

Either outcome serves #283's intent (overlap I/O for streaming SVAR1); the landed
`run_windows` is not wasted regardless (it is #276's mechanism).

### Generation stays per-backend

Generation is **not** lifted into the `StreamBackend` trait speculatively (YAGNI). SVAR1's
generate lives with SVAR1's backend code; #276 decides whether to generalize a `generate`
method when it has a second implementation to factor against.

## Testing

- **Parity is the control, byte-identical, untouched semantics.**
  `tests/dataset/test_streaming_parity.py` and `test_scale_parity_still_byte_identical`
  must stay green through both PRs. Any non-mechanical change to them means behavior
  changed — stop and investigate.
- **#284 gate — cohort-scale RSS is flat in `n_samples`** (deterministic, not
  wall-clock; this node is too noisy for absolute timings). A new scale test sweeps
  `n_samples` and asserts peak `ru_maxrss` growth stays bounded and does *not* track the
  cohort. Complements the existing entries-touched counter gate, which stays.
- **`max_mem` sizing** — assert derived `window_regions`/`window_samples` keep
  `n_slots × offset_bytes ≤ budget`; assert batch output is independent of `window_*`.
- **Engine (PR 2)** — reuse the 5 landed adversarial `run_windows` tests (plan-order,
  slot-cap recycling, producer-error, consumer-error, empty-plan; each run 20× in
  `--release` for races). Add a `StreamBackend for Svar1Store` unit test.
- **Overlap measurement (PR 2)** — cold-cache A-vs-C, reported as secondary color per the
  perf-gate convention, never as the pass/fail gate.

## Docs / skill / roadmap gates

Public-API change (`max_mem` on `StreamingDataset`) — the docs-audit gate fires:

- `docs/source/api.md` — `StreamingDataset` autoclass members (add `max_mem` if surfaced
  as a documented parameter); keep the `__all__` sync check green.
- `docs/source/dataset.md`, `docs/source/faq.md` — the `StreamingDataset` sections: note
  cohort-independent memory and the `max_mem` knob; refresh the limitations list.
- `skills/genvarloader/SKILL.md` — `StreamingDataset` section + gotchas: `max_mem`,
  cohort-independent peak memory, per-batch generation.
- `docs/roadmaps/streaming-dataset.md` — tick Plan 2's Task 5 Step 5 (#283) and the
  #284 memory concern; record the measured `max_mem` default (the knee), the
  cohort-scale RSS result, and the A-vs-C overlap finding + its #276 implication.

## Implementation chunks

1. **PR 1 (#284)** — read/generate split; per-batch generation; sample-chunk axis;
   `max_mem` budget + derivation; measured default; cohort-scale RSS gate; docs. Parity
   green, single-threaded. **Gates PR 2.**
2. **PR 2 (#283)** — `StreamBackend for Svar1Store`; `StreamEngine` pyclass (Design A);
   Design C pipelining; cold-cache A-vs-C measurement; ship the winner; roadmap record.

## Deferred / open questions

- **VCF/PGEN pipeline (dense→sparse conversion vs. new dense→hap kernel)** — #276,
  explicitly not prejudged here.
- **N-slot ring vs. 2-slot ping-pong** — start with 2; promote only on profiling
  evidence (unchanged from Plan 2).
- **`num_workers` sharding** — deferred; the skeleton's guard stays.
- **`max_mem` default** — measured on the scale fixture; recorded in the roadmap.
- **Whether SVAR1 ships Design A or C** — decided by the PR 2 cold-cache measurement.
