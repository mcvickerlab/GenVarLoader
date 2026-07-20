# Streaming SVAR2 backend — Phase 2 PR 3 (read↔reconstruct pipeline engine) — design

**Date:** 2026-07-19
**Status:** ✅ implemented — `Svar2StreamEngine` landed behind `_prefetch_strategy="svar2_engine"`.
Cold-cache A/B (500 and 2000 samples, best-of-3) did not clear the non-overlapping-ranges ship bar
(§"Gating") — `_Svar2Backend._default_strategy` stays `"sync"`; the engine ships off-by-default.
See `docs/roadmaps/streaming-dataset.md` → Plans (spec A), PR-3 row, for the raw numbers.
**Issue:** [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) — spec B of the write-free
`StreamingDataset` effort. **Target branch:** `streaming` (stacks on the merged PR-2 work on
`278-svar2-streaming-backend`), not `main`.
**Roadmap:** `docs/roadmaps/streaming-dataset.md` → Plan 2 (SVAR2), PR 3 row.
**Refines:** `docs/superpowers/specs/2026-07-18-streaming-svar2-phase2-design.md` → **Lever 3 (PR 3)**.
This doc supersedes that spec's PR-3 sketch on two points (see "Deviations from the parent spec").
**Follows:** PR 1 (fast synchronous path — [#301](https://github.com/mcvickerlab/GenVarLoader/pull/301)/
[#302](https://github.com/mcvickerlab/GenVarLoader/pull/302)) and PR 2 (super-batch parallel
reconstruct — [#303](https://github.com/mcvickerlab/GenVarLoader/pull/303), merged).

## Summary

PR 2 shipped the super-batch's multi-core reconstruct dispatch and **measured** two things that decide
what PR 3 must attack:

1. The reconstruct kernel is **memory-bandwidth-bound** — only ~1.1–1.3× on 8 cores, and larger
   super-batches *hurt* (cache locality). More reconstruct parallelism is spent.
2. End-to-end `to_iter` wall (~0.42 s on the 2000×20000 cohort store) is **dominated by the serial,
   GIL-held Python drive**, not reconstruct (end-to-end cpu/wall ≤ 1.24). That drive is the glue
   *around* the one GIL-free rayon call (`_Svar2Backend._fill_super_batch`): `read_window`'s reshape
   (`_streaming.py:1096-1099`), `_gather_rows`' per-row fancy-indexing (`:1132-1156`), and
   `_drain`/`yield` marshaling (`:1219-1227`).

So the lever is **not** more reconstruct cores — it is getting that serial Python glue **off the GIL and
off the critical path**, overlapping it with the GIL-free reconstruct. That is exactly Lever 3 of the
parent spec. PR 3 realizes it as a **Rust producer-thread engine** for SVAR2, mirroring the landed
SVAR1 `Svar1StreamEngine` one format down.

Byte-identical parity with `gvl.write()` + `Dataset.open()[r, s]` under `.with_seqs("haplotypes")`
(jitter=0) remains the hard correctness gate.

## Architecture — mirror `Svar1StreamEngine`, one format down

Add a `Svar2StreamEngine` pyclass (new `src/ffi/svar2_stream_engine.rs`, sibling to
`src/ffi/stream_engine.rs`) that **copies the SVAR1 engine's concurrency discipline exactly** — the same
discipline already hardened by SVAR1's adversarial suite and #296:

- **Detached `std::thread` producer** owning an `Arc<Svar2Store>` (Arc-shares the same mmap the
  Python-owned `Svar2Store` pyclass holds — not the pyclass instance), the region-scale plan, and the
  channel `Sender`s. Detached (not `thread::scope`) because it must outlive individual `next_batch()`
  FFI calls, exactly as SVAR1's producer does.
- **`crossbeam_channel::bounded(2)` ping-pong** — a `filled` channel (producer→consumer) and a `free`
  channel (consumer→producer, recycles drained buffers). Only two reconstruct buffers exist total,
  ping-ponging, so peak memory is `2 × super-batch` regardless of plan length (preserves #284's
  cohort-independent bound; the two buffers are both super-batch-sized, `max_mem`-derived).
- **Shutdown by dropping the producer's filled `Sender`** when the job loop ends; the consumer's
  `recv()` then observes channel close.
- **Join-then-classify-panics**: on channel close the consumer joins the producer and only then
  surfaces a producer error/panic (never a silent wedge). `Drop for EngineState` joins rather than
  leaking a detached thread.
- **`next_batch()` blocks on `recv` under `py.detach`** (GIL released while waiting), rebuilds the
  output `PyArray`s under the GIL from the reconstructed buffer.

**The decisive difference from the current `"sync"` drive:** the producer runs the **entire fill chain
in Rust, GIL-free** — `find_ranges` → per-row index expansion (today's Python `_gather_rows`) → gather /
decode → super-batch reconstruct into a recycled buffer. Python never sees the intermediate window
arrays; main-thread work shrinks to draining `batch_size` slices and marshaling output, **concurrently
with the producer building window N+1**. This moves the `_gather_rows`/reshape glue off the GIL *and*
overlaps it with reconstruct in one move.

### Where the pieces already exist

- **Reconstruct + gather kernel:** the PR-2 chain `svar2_reconstruct_super_batch` + the row expansion
  in `_gather_rows` is the reference for what the producer must do in Rust. The genoray gather/decode
  (`gather_haps_readbound` / `decode_hap`) and gvl `split_to_flat`/`hap_diffs_svar2` /
  `reconstruct_haplotypes_from_svar2` already run inside `py.detach`; PR 3 hoists the *row index
  expansion* (currently Python fancy-indexing of the window dict) into the same GIL-free region.
- **Range query:** `Svar2Store` already holds a per-contig `ContigReader`; `svar2_read_window`
  (`genoray_core::query::find_ranges`) is the GIL-free range producer (landed in PR 1b). The producer
  calls it directly and keeps the ranges in Rust structures instead of reshaping to numpy.
- **Buffer:** `Svar2ReconBuf` (PR 2) is the recycled owned output buffer; the engine holds two and
  recycles them through the `free` channel (SVAR1 recycles offset buffers the same way).

### Python wiring

- `_Svar2Backend` gains `build_engine(...)` (mirrors `_Svar1Backend.build_engine`) constructing a
  `Svar2StreamEngine` from the cached per-contig meta + region-scale plan.
- `StreamingDataset._iter_batches` gains an SVAR2 engine branch, selected by `_prefetch_strategy`. It
  drives `engine.next_batch(batch_size)` in lockstep with the **same** compact region-scale plan the
  `"sync"` branch builds, so job order and per-window batching cannot diverge (the exact invariant
  SVAR1's 8b wiring holds).
- The `"sync"` branch stays as the parity-anchored fallback and the A/B baseline.

## Deviations from the parent spec (Lever 3)

1. **Deterministic order kept — the "relaxed / completion-order iteration" requirement is dropped from
   PR 3 scope.** The parent spec made iteration completion-order so multiple windows could be in flight
   and drain out of order without a reordering barrier, and anticipated faq/dataset/skill doc edits to
   declare streaming order non-deterministic. **With a single producer feeding an ordered
   `bounded(2)` channel, windows are emitted in plan order for free** — no barrier, no reordering, and
   parity assertions stay position-based. Completion-order would only buy something with *multiple
   parallel reconstruct workers*, and PR 2 proved reconstruct is memory-bandwidth-bound (rayon-internal
   already takes what the bus allows), so more reconstruct workers is precisely what we do **not** want.
   Keeping deterministic order is a YAGNI simplification and **avoids a public iteration-order contract
   change** (no faq/dataset/skill edit for order). If a future design ever adds parallel reconstruct
   workers, completion-order can be revisited then, on its own evidence.

2. **Build-then-measure, not measure-then-build.** The parent spec left PR 3 "gated: ship only on a
   clear cold-cache win." PR 3 still ships gated (below), but the engine is **built first** and the
   cold-cache A/B is run against the completed engine — matching how SVAR1 shipped its producer-thread
   engine (Design A) after building it and measuring A-vs-C (Task 9). The measurement is the ship/no-ship
   gate on the *built* engine, not a prerequisite that blocks the build.

## Gating (ship/no-ship) — same discipline as SVAR1 Design-A-vs-C

The engine is added **behind the existing `_prefetch_strategy` seam**; `_Svar2Backend._default_strategy`
stays `"sync"` until measured. After the engine is parity-green:

- Run a **cold-cache A/B** (engine vs `"sync"`) on the vcfixture cohort store (fresh store per run,
  page-evicted), best-of-N, on the shared dev node, extending
  `benchmarking/streaming/svar2_cold_cache.py` / the super-batch sweep harness.
- **Ship the engine as the SVAR2 default only if it beats `"sync"` outside node noise** (non-overlapping
  run ranges, the SVAR1 Task-9 bar). Otherwise keep it off-by-default behind the toggle and **record the
  reason** in the roadmap. Either outcome is a valid PR-3 landing — the engine + harness are kept as
  measured infrastructure regardless (they additionally matter for the #276 VCF/PGEN decode path, which
  `madvise`/read-through cannot hide, exactly as SVAR1 concluded).
- **Perf is secondary color, never pass/fail.** Parity is the hard gate.

## Testing

- **Parity is the control, byte-identical, position-based** (order preserved — deviation 1).
  `tests/dataset/test_streaming_parity_svar2.py` (multi-contig, unsorted bed, mixed contig-naming,
  sample-order) stays green under the engine strategy, asserted the same way as under `"sync"`.
- **Cross-strategy identity:** a test asserting the engine strategy yields **byte-identical** batches to
  the `"sync"` strategy for the same plan/batch_size (the toggle changes execution, never output) —
  the SVAR2 analog of SVAR1's engine-vs-readahead parity.
- **#284 cohort-scale gate** — the reconstruct buffer's byte count stays `max_mem`-bounded and
  **identical across cohort sizes**; the engine holds exactly two super-batch buffers, so peak output is
  cohort-independent. Generalize the existing PR-2 gate to run under the engine strategy.
- **Engine adversarial suite** — reuse SVAR1's `run_windows`/engine test shapes: plan-order yield,
  slot-cap recycling, producer-error propagation, consumer-error, empty-plan; each run **20× in
  `--release`** for hang/race stability (the SVAR1 bar).
- **Counter-blind-to-producer-thread regression** — SVAR1's #296 showed a `thread_local!` gate goes
  blind once `read_window` runs on the producer thread. Any SVAR2 scale/entries counter touched here
  must be process-wide (`AtomicUsize`), verified meaningful under the engine (not a `0==0` vacuum pass).

## Docs / skill / roadmap gates

- **No new public `__all__` symbol; no signature change.** The engine is internal, selected by the
  existing `_prefetch_strategy` seam. `docs/source/api.md` `__all__`-sync stays `MISSING: none`.
- **No iteration-order doc change** (deviation 1 keeps order deterministic) — the parent spec's
  anticipated faq/dataset/skill "completion-order" edits are **not** made.
- `docs/roadmaps/streaming-dataset.md` — add the PR-3 plan row + the cold-cache engine-vs-sync
  measurement + the ship/no-ship decision (mirroring the SVAR1 Task-9 entry); tick Phase 2 further.
- New plan doc: `docs/superpowers/plans/2026-07-19-streaming-svar2-phase2-pr3.md`.

## Implementation chunks (for the plan, not this spec)

1. **Rust `Svar2StreamEngine`** (`src/ffi/svar2_stream_engine.rs`) — port the SVAR1 engine structure:
   detached producer, `bounded(2)` ping-pong, shutdown-by-`Sender`-drop, join-then-classify,
   `next_batch()` under `py.detach`. Producer runs `find_ranges` → row expansion → gather/decode →
   super-batch reconstruct into a recycled `Svar2ReconBuf`, all GIL-free. Register the pyclass in
   `src/lib.rs`.
2. **Python wiring** — `_Svar2Backend.build_engine`; `StreamingDataset._iter_batches` SVAR2 engine
   branch driven in lockstep with the same region-scale plan as `"sync"`. Parity + cross-strategy
   identity + #284 green.
3. **Adversarial engine tests** (chunk 1's Rust side) + Python cross-strategy/parity tests (chunk 2).
4. **Measurement + decision** — cold-cache A/B harness; record the number; set the SVAR2 default; write
   the roadmap entry. Ships gated either way.

## Deferred / open questions

- **Two-stage (separate read + reconstruct threads)** — a further refinement overlapping `find_ranges`
  of window N+1 with reconstruct of window N *within* the producer. Deferred: `find_ranges` is cheap
  binary searches; the dominant serial cost is the Python gather/marshal glue, which the single-producer
  engine already removes/overlaps. Revisit only if a profiled single-window fill shows `find_ranges`
  dominating at super-batch scale.
- **Parallelizing the serial fill stages** (`split_to_flat`/`hap_diffs` in gvl; `find_ranges`/`gather`
  in genoray) — unchanged from the parent spec: not pursued unless a profiled single-window fill shows a
  serial stage dominating where rayon would amortize.
- **Whether the engine clears its gate** — the cold-cache A/B decides; either outcome lands (default
  flip or off-by-default with reason recorded).
- **Splicing / other output modes / intervals** — unchanged, out of scope (#277, #279).

## Pointers

- **SVAR1 engine template (copied):** `src/ffi/stream_engine.rs` (`Svar1StreamEngine` — detached
  producer, `bounded(2)` ping-pong, shutdown/join/classify discipline, `next_batch` under `py.detach`);
  `_dataset/_streaming.py` (`_Svar1Backend.build_engine`, the `"engine"` `_iter_batches` branch and its
  in-lockstep plan drive); #296 process-wide counter fix.
- **SVAR2 PR-2 chain (hoisted into the producer):** `_Svar2Backend.read_window` (`_streaming.py:1071`),
  `_gather_rows` (`:1102`), `_fill_super_batch` (`:1172`), `_drain` (`:1219`);
  `svar2_reconstruct_super_batch` FFI; `Svar2ReconBuf`; `Svar2Store` (`src/svar2/store.rs`).
- **Rust range query:** `svar2_read_window` / `genoray_core::query::find_ranges` (pinned rev; no bump).
- **Parity contract & conventions:** `docs/archive/roadmaps/rust-migration.md` (Phase 6a — SVAR2
  read-bound precedent).
- **Parent spec (Lever 3):** `docs/superpowers/specs/2026-07-18-streaming-svar2-phase2-design.md`.
