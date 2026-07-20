# Streaming SVAR2 backend — Phase 2 (measurement-gated) — design

**Date:** 2026-07-18
**Status:** design (pending spec review)
**Issue:** [#278](https://github.com/mcvickerlab/GenVarLoader/issues/278) — spec B of the write-free
`StreamingDataset` effort. **Target branch:** `streaming` (not `main`).
**Roadmap:** `docs/roadmaps/streaming-dataset.md` → Plan 2 (SVAR2).
**Follows:** `docs/superpowers/specs/2026-07-17-streaming-svar2-backend-design.md` (Phase 1 — parity,
landed) and `docs/superpowers/plans/2026-07-17-streaming-svar2-backend.md` (§"Phase 2 — producer-thread
engine (DEFERRED, gated)").

## Summary

Phase 1 shipped a **synchronous** SVAR2 `StreamBackend` (byte-identical parity vs `gvl.write()` +
`Dataset[r,s]`). Phase 2 was deferred behind a measurement gate: *profile the synchronous fill first,
then choose the lever*. **That gate has now been run** (it never had — the Task-4 harness reported
`vcfixture` "absent" only because the binary wasn't on `PATH`; it is present on the dev box). The
measurement redirects Phase 2 away from the plan's presumed lever (a single producer-thread engine, by
analogy to SVAR1) toward **many-core parallel reconstruction in Rust**.

**What the measurement found (details in "Measurement gate"):**

1. The SVAR2 streaming fill is **CPU-bound, not IO-bound** (cold ≈ warm page cache). SVAR1's
   producer-thread win (1.46× hiding page faults, #283) **does not transfer**.
2. The read-bound kernel's `parallel=True` is **actively harmful at streaming *batch* granularity**: it
   forks the global ~96-thread rayon pool for only `batch_size × ploidy ≈ 64` tiny (~1 kb) haplotypes
   per call, and the scheduler/work-steal overhead dwarfs the reconstruction (`perf`: ~30%
   `__sched_yield`, ~10%+ crossbeam steal/epoch; reconstruct closure 0.77% self). `parallel=False` is
   **1.2–1.8× faster and uses one core instead of two-to-three**, byte-parity-identical.
3. ~25–30% of the synchronous pass is **GIL-held Python** (`SparseVar2._find_ranges` ~17% + numpy
   reshape/marshal glue), reclaimable via GIL-free Rust — reachable through the already-linked, public
   `genoray_core::query::find_ranges` with **no genoray rev bump**.

**The reconstruction is embarrassingly parallel** — every `(region, sample)` cell in a window is
independent (queries are sample-independent; regions in a window don't interact). The streaming path is
single-core only because the rayon *dispatch granularity* (one user batch) is far too small. **The fix
is to decouple the reconstruct granularity from the user's `batch_size` and dispatch one coarse,
core-saturating parallel reconstruction per super-batch — in Rust, not via Python multiprocessing.**

Byte-identical parity with `gvl.write()` + `Dataset.open()[r, s]` under `.with_seqs("haplotypes")`
(jitter=0) remains the hard correctness gate. **Iteration order is relaxed to completion-order** (a
product decision, below): each batch carries its `(region, sample)` indices, and parity is matched by
index, so relaxed order is parity-safe.

## Measurement gate (the evidence base)

All measured this session on the shared dev node (96 cores), 2000 samples × 20 000 records, vcfixture
bulk store, `.with_seqs("haplotypes")`, best-of-N.
`VCFIXTURE_BIN=/carter/users/dlaub/projects/vcfixture-rs/target/release/vcfixture`.

**(1) IO vs CPU — cold-vs-warm wall clock.** Build a cohort-scale store, evict its pages with
`posix_fadvise(DONTNEED)`, time a cold pass then an immediate warm pass. `cold − warm` = page-fault (IO)
cost; warm = pure CPU.

| n_samples | cold (best-of-3) | warm | IO delta | verdict |
|---|---|---|---|---|
| 2000 | 4.41 s | 4.71 s | **−0.30 s (−6.7%)** | **no IO to hide — CPU-bound** |

Negative delta ⇒ within noise of zero: after eviction the cold pass is no slower than warm. Architectural
(SVAR2 stores are small vs RAM; the fill is decode-heavy), not a fixture artifact — more samples = more
decode CPU, never IO-bound.

**(2) Core utilization — `cpu_secs(all threads) / wall`.** Despite `parallel=True`, ~2 of 96 cores
(stable across 3 runs at bs=32): bs=8 ~2.1–2.7×, bs=32 ~2.0×, bs=128 ~1.7×.

**(3) Where the CPU goes — `perf record`, 269 k samples, steady-state loop.** Dominated by **rayon
scheduling overhead, not work**: ~38% kernel of which **~30% `__sched_yield`** (idle rayon workers);
~10%+ `crossbeam_deque::Stealer::steal` / `crossbeam_epoch` / `WorkerThread::find_work`; useful work tiny
(`SearchTree::build` inside `find_ranges` 2.78%; `reconstruct_haplotypes_from_svar2::{closure#0}`
**0.77% self**).

**(4) `parallel=True` vs `parallel=False` A/B (per-batch reconstruct).** False wins everywhere, one core:

| batch_size | parallel=True | parallel=False | speedup | CPU (True→False) |
|---|---|---|---|---|
| 8 | 3.44 s | **1.92 s** | 1.79× | 2.7→1.0 cores |
| 32 | 1.18 s | **0.82 s** | 1.44× | 2.4→1.0 cores |
| 128 | 0.63 s | **0.52 s** | 1.22× | 1.8→1.0 cores |

Parity under `parallel=False` **verified** (`test_streaming_parity_svar2.py` passes — the flag changes
iteration strategy, never output). The crossover shrinks as per-call work grows (bs=8 1.79× → bs=128
1.22×) — i.e. rayon *would* pay off if the dispatch had far more work; that is exactly the super-batch
lever below.

**(5) Stage structure (code map).** Of the fill chain only the final reconstruct uses rayon; the rest are
serial: `find_ranges` (genoray), `gather_haps_readbound` (genoray), `split_to_flat` (gvl
`src/svar2/mod.rs`), `hap_diffs_svar2` (gvl); `reconstruct_haplotypes_from_svar2` (gvl
`src/reconstruct/mod.rs:620`, rayon over `(query,hap)` rows). The gather→reconstruct body already runs
inside `py.detach` (`src/ffi/mod.rs:1392`); the **GIL-held** cost is the *separate*
`SparseVar2._find_ranges` pymethod + its numpy reshape glue.

## Product decision: relaxed (completion-order) iteration

`StreamingDataset` is an **`IterableDataset`**; there is no random access and every batch already carries
its `(region, sample)` indices (`to_iter(..., return_indices=True)`). Phase 2 makes iteration order
**non-deterministic (completion-order)**: the engine emits each batch as soon as it is reconstructed.

- **Parity-safe.** The parity oracle matches each cell by its returned `(r, s)` index, not by position,
  so completion-order emission is byte-identical to the written dataset. (Any consumer that needs a
  specific order sorts by the emitted indices — the same contract as a shuffled `DataLoader`.)
- **Why relax:** it lets multiple windows be in flight (hide window N+1's serial fill behind window N's
  reconstruction) and drain on completion, without a cross-window reordering barrier. It is *not* what
  gives us multi-core (that is the super-batch dispatch, which is order-preserving on its own) — it is
  the extra lever on top.

## Why not `num_workers` (DataLoader multiprocessing)

The obvious way to use 96 cores — `DataLoader(num_workers=N)` — is the wrong tool for a Rust-first
`IterableDataset` and is **explicitly rejected** here:

- **Peak RAM inflates.** Each worker is a separate process with its own Python heap, store handles, and
  buffers; the resident set multiplies roughly with `N`, defeating #284's cohort-independent memory bound.
- **IPC tax.** Reconstructed batches are pickled and copied over a pipe back to the main process — pure
  overhead on top of the reconstruction we already did.
- **Leaves speed on the table.** Worker startup, GIL-bound collation in the main process, and the
  inability to share one GIL-free Rust thread pool across processes all waste cores.

For a high-performance Rust-backed library the parallelism belongs **inside Rust**, over the store's
shared mmap, with output bounded by `max_mem`. This design does that; `num_workers > 0` stays guarded/
deferred and is documented as *not* the scaling path (so a future contributor does not "fix" single-core
streaming by re-enabling per-batch kernel rayon or reaching for MP).

## Design — the three decoupled granularities

The core idea: **read window ⊃ reconstruct super-batch ⊃ user batch**, three independently-sized units.

| Unit | What it bounds | Sizing | Today |
|---|---|---|---|
| **Read window** | `find_ranges`/gather amortization; offsets/ranges buffer | `max_mem` (region + sample chunk) | exists (#284) |
| **Reconstruct super-batch** | the *rayon dispatch* — must be large enough to saturate cores; output buffer | `max_mem` output budget, ≫ `batch_size`, cohort-independent | **new** (== batch today) |
| **User `batch_size`** | the yield/drain unit off the reconstructed buffer | user arg to `to_iter` | conflated with reconstruct |

Reconstructing a **super-batch** of thousands of independent cells in **one** `parallel=True` dispatch
amortizes the rayon fork/join, saturating cores; the buffer is drained to the consumer `batch_size` at a
time. Output stays `max_mem`-bounded and cohort-independent, so the #284 guarantee holds.

### Lever 1 (PR 1): fast synchronous path — immediate, parity-safe, no rev bump

**1a. `parallel=False` for the *current* per-batch reconstruct.** In `_Svar2Backend.generate_batch`
(`_streaming.py:1098-1115`), pass `parallel=False`. Immediate 1.2–1.8×, halves CPU, verified
parity-identical. **Streaming-only** — the written-`Dataset` path (`_svar2_haps.py`) keeps `parallel=True`
(its `__getitem__` chunks are large enough to amortize rayon). This is a per-call flag, not a kernel
default. *(Superseded by PR 2's super-batch dispatch, which re-enables `parallel=True` at the right
granularity; 1a is the correct choice for the pre-restructure structure and a fast standalone win.)*

**1b. GIL-free Rust `Svar2Store.read_window`.** Replace the Python `SparseVar2._find_ranges` call in
`_Svar2Backend.read_window` (`_streaming.py:1008-1047`) with a new `Svar2Store.read_window` pymethod that
computes the four range arrays in Rust under `py.detach` and returns them, deleting the ~17% GIL-held call
+ ~10% Python reshape glue. **No rev bump:** `genoray_core::query::find_ranges(reader, regions, samples)
-> RangesBundle` is public at the pinned rev (`Cargo.toml` `e07477e…`; genoray
`src/query/gather.rs:338`), and the Python `_find_ranges` is a thin wrapper over it
(`genoray src/py_query_ranges.rs:231`). `Svar2Store` already holds a `ContigReader` per contig
(`src/svar2/store.rs`). Mechanical `RangesBundle` (`Vec<Range<usize>>`) → the `i64` `(…,2)` arrays the
read-bound FFI already consumes; parity is the oracle.

**PR 1 is parity-critical and independently valuable** (the SVAR2 analog of SVAR1's #284 memory PR). It
gates PR 2. 1a can ship as its own commit/PR ahead of 1b if the immediate win is wanted first.

### Lever 2 (PR 2): many-core parallel reconstruction — the real scaling lever

Decouple the reconstruct super-batch from `batch_size` and dispatch one coarse `parallel=True`
reconstruction per super-batch, draining to the consumer per `batch_size`. This is where the 94 idle
cores come back — **in Rust, order-preserving within the super-batch**.

- **Rust:** a `Svar2Store` method (or an extension of the read-bound FFI) that, given a window's ranges,
  gathers + decodes + reconstructs a `max_mem`-sized super-batch of rows in one `parallel=True` call
  (rayon over the super-batch's `rows × ploidy`), writing to a recycled owned output buffer. Output
  `(hi−lo)`-bounded per drained batch is preserved by draining the buffer, not by shrinking the dispatch.
- **Python:** `_iter_batches`' SVAR2 branch reads a window's ranges (Rust, GIL-free), drives one
  super-batch reconstruction, then yields `batch_size` slices with their `(r, s)` indices.
- **`max_mem` sizes the super-batch** so it saturates cores while the output buffer fits the budget
  (cohort-independent). The super-batch size is **measured** (a core-utilization + wall-clock sweep), not
  guessed — record the knee in the roadmap. Expected: cores_used ≫ 2 and a multi-× wall-clock win over
  PR 1's single-core path.

### Lever 3 (PR 3, gated): relaxed-order multi-window pipeline

On top of Lever 2, keep multiple windows in flight so window N+1's serial fill (`find_ranges` + gather +
`split_to_flat`, ~20% after 1b) hides behind window N's reconstruction. Structure: a bounded
`crossbeam_channel` output queue (backpressure = memory bound) fed by the reconstruction, drained by a
Python-visible iterator whose blocking `recv` runs under `py.detach` — reusing `Svar1StreamEngine`'s
discipline (`src/ffi/stream_engine.rs`: named thread, shutdown-by-`Sender`-drop, join-then-classify).
Completion-order emission (the product decision) is what makes this simple — no reordering barrier.

**Gated:** ship as default only if a cold-cache run beats PR 2's synchronous super-batch path outside
node noise. Prior: a ~1.2× fill-hiding gain (the fill is a minority of CPU after 1b, and reconstruction
already saturates cores, so a fill thread mostly competes). If it does not clear, keep it behind the
`_prefetch_strategy` toggle and record the reason — the same evidence-based ship/no-ship SVAR1 used.

## Testing

- **Parity is the control, byte-identical, order-independent** (matched by returned `(r,s)`).
  `test_streaming_parity_svar2.py` (multi-contig, unsorted bed, mixed contig-naming, sample-order) stays
  green through all PRs. Add: parity under `parallel=False` (PR 1a); Rust-`read_window` ranges byte-
  identical to the Python `_find_ranges` path (PR 1b); parity under super-batch reconstruction + relaxed
  emission — assert the *set* of `(r,s)→haplotype` pairs matches, since order is no longer fixed (PR 2/3).
- **#284 cohort-scale gate** — generalize `test_svar2_generate_batch_output_is_flat_in_cohort_size` to the
  super-batch: assert the reconstruct buffer's byte count is `max_mem`-bounded and **identical across
  cohort sizes** (cohort-independent), not tracking `n_samples`. The property is unchanged; the unit
  becomes the super-batch.
- **Core-utilization gate (PR 2)** — deterministic-ish: assert `cpu_secs/wall` rises materially above the
  ~1× single-core PR-1 baseline on the scale fixture (secondary color on wall-clock per the shared-node
  convention, but core-count is a robust signal the restructure engaged rayon).
- **Perf is secondary color, never pass/fail.** Record in the roadmap: the `parallel=False` speedup, the
  super-batch knee + its core-utilization/wall win, and (PR 3) the cold-cache pipeline vs super-batch
  result with the ship/no-ship decision.
- **Engine (PR 3, if built)** — reuse SVAR1's adversarial `run_windows`/engine tests (plan-order,
  slot-cap recycling, producer-error, consumer-error, empty-plan; 20× in `--release`).

## Docs / skill / roadmap gates

No new public `__all__` symbol; but iteration order becomes an explicit part of the contract:

- `docs/source/faq.md` / `dataset.md` + `skills/genvarloader/SKILL.md` — document that streaming
  iteration is **completion-order (non-deterministic)**; batches carry `(r,s)` indices; sort by them if a
  fixed order is needed. State that streaming parallelism is internal (Rust) and `num_workers` is *not*
  the scaling path.
- `docs/roadmaps/streaming-dataset.md` — record the measurement gate result (CPU-bound; per-batch
  `parallel=True` harmful; the A/B table), the super-batch knee, the core-utilization win, and the PR-3
  decision + its SVAR1 contrast. Tick Phase 2; add PR pointers.
- `docs/source/api.md` `__all__`-sync check stays green (expected `MISSING: none`).

## Implementation chunks (stacked PRs; for the plan, not this spec)

1. **PR 1a — `parallel=False` streaming reconstruct.** One-line flag + parity re-verify + the quick sweep
   confirming small-batch streaming never wants `parallel=True`. Fast standalone win.
2. **PR 1b — GIL-free Rust `Svar2Store.read_window`** via `genoray_core::query::find_ranges`; delete the
   Python `_find_ranges` glue. Parity + #284 green. Stacks on 1a.
3. **PR 2 — super-batch parallel reconstruction.** Decouple reconstruct granularity from `batch_size`;
   coarse `parallel=True` dispatch; `max_mem` super-batch sizing (measured knee); core-utilization gate;
   generalized #284 gate. The real multi-core lever. Stacks on 1b.
4. **PR 3 (gated) — relaxed-order multi-window pipeline.** Bounded-queue producer/consumer over the
   super-batch reconstruction; cold-cache vs PR-2 measurement; ship as default only on a clear win, else
   off-by-default with the reason recorded. Stacks on 2.

## Deferred / open questions

- **Super-batch size default** — measured knee (core-utilization × wall × `max_mem` output budget); do
  not ship a guess.
- **`parallel` hard-off vs work-size gate at the *batch* level (PR 1a)** — a quick sweep; PR 2 supersedes
  it with the super-batch dispatch, so keep 1a minimal.
- **Whether PR 3 ships as default** — cold-cache pipeline-vs-super-batch measurement; prior says marginal.
- **`num_workers` sharding** — rejected as the scaling path (RAM/IPC/idle cores); documented, not built.
- **Adding rayon to the serial fill stages** (`split_to_flat`/`hap_diffs` in gvl; `find_ranges`/`gather`
  in genoray) — not pursued unless a profiled *single-window* fill (not the whole loop) shows a serial
  stage dominating at super-batch scale where rayon would amortize.
- **Splicing / other output modes / intervals** — unchanged, out of scope (#277, #279).

## Pointers

- **SVAR1 engine template (for PR 3):** `src/ffi/stream_engine.rs` (`Svar1StreamEngine`),
  `_dataset/_streaming.py` (`_Svar1Backend.build_engine`, the `"engine"` `_iter_batches` branch).
- **SVAR2 read-bound FFI (reused/extended):** `reconstruct_haplotypes_from_svar2_readbound`
  (`src/ffi/mod.rs:1321-1482`; body under `py.detach` at `:1392`), `split_to_flat` / `hap_diffs_svar2`
  (`src/svar2/mod.rs`), `reconstruct_haplotypes_from_svar2` (`src/reconstruct/mod.rs:620`, `parallel`
  flag, rayon over `(query,hap)` rows).
- **Rust range query (for PR 1b):** `genoray_core::query::find_ranges` + `RangesBundle` (pinned rev
  `e07477e`, genoray `src/query/gather.rs:338, 311`); `Svar2Store` (`src/svar2/store.rs`).
- **Phase-1 backend:** `_Svar2Backend` (`_dataset/_streaming.py`), `read_window` (`:1008`),
  `generate_batch` (`:1049`, parallel flag at `:1113`), `_plan` region-major sweep (`:285`).
- **Measurement harness:** `benchmarking/streaming/svar2_cold_cache.py` (extend with cold-vs-warm split,
  core-utilization, and a super-batch/pipeline strategy).
- **Parity contract & conventions:** `docs/archive/roadmaps/rust-migration.md` (Phase 6a — SVAR2
  read-bound precedent).
