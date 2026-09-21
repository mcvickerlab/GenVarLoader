# Fold the record-backend mixed track window decode into the producer

**Issue:** [#400](https://github.com/mcvickerlab/GenVarLoader/issues/400)
**Roadmap:** `docs/roadmaps/streaming-dataset.md` → Plans table
**Date:** 2026-09-21
**Branch:** `feat/400-mixed-record-decode-fold` (targets `streaming`)

## Goal

Mixed variants+tracks streams on the VCF/BCF and PGEN backends decode every window **twice**:
once synchronously on the consumer thread, to size the deletion-extended track query
(`t_ends_ext`) before the window's track read, and once in the engine's producer thread, which
fills the ping-pong slot the batches are generated from.

Remove the second decode by reading the window the producer already decoded, from the engine
being driven. Byte-identical parity against `gvl.write()` + `Dataset[r, s]` is preserved on
both halves of the `(haplotypes, tracks)` tuple, and there is no public API change.

## Summary of decisions

| Decision | Choice |
|---|---|
| Source of the window's variant data | The drive engine's **current window slot** — no ad-hoc decode |
| New Rust core primitive | `StreamEngineCore::ensure_current_window` (extracted from `advance`) + `with_current_window` |
| New FFI surface | `RecordStreamEngine.current_window_realign_inputs(contig_idx, region_starts, region_ends, s_lo, s_hi)` |
| Window identity | **Validated in Rust** against the current job; mismatch raises `PyValueError` |
| Python seam | `_mixed_track_window(..., engine)` → `_MixedTracksBackend.mixed_realign_window(..., engine=None)` |
| Second engine object | Deleted — `_VcfBackend._mixed_engine` / `_PgenBackend._mixed_engine` go away |
| `window_realign_inputs` | **Kept, documented test-only** — the only independent oracle for the CSR-replication test |
| SVAR1 / SVAR2 | Not folded (SVAR1's re-read is offsets-only, no decode; SVAR2 never had a second decode) |
| Gate | Deterministic decode counters, not wall-clock |

## 1. The duplication

In the `"engine"` drive (`python/genvarloader/_dataset/_streaming.py:2270-2287`) each window
composes in this order:

1. `_mixed_track_window` (`:1677`) calls `backend.mixed_realign_window(...)` (`:1758`), which
   for a record backend reaches `_record_mixed_realign_window` (`:735`).
2. That function calls `backend._mixed_engine().window_realign_inputs(...)` (`:812-819`) —
   `_mixed_engine` (`:4816`, `:5072`) is a cached, **plan-less** second engine
   (`build_engine([], 1, 1)`), and `window_realign_inputs`
   (`src/record_stream/engine.rs:1086`) decodes the window it is handed via
   `RecordBackend::debug_fill` into a throwaway slot.
3. The decode yields `(v_starts, ilens, geno_v_idxs, geno_offsets)`, from which the mixed path
   computes `diffs` → `region_max_del` → `t_ends_ext`, then reads the window's tracks.
4. The batch loop then pulls batches with `next_batch()`, whose producer thread decodes the
   **same window** again into the ping-pong slot.

So a mixed VCF/PGEN sweep pays 2× window decode plus a second engine object per backend (for
PGEN a second `PgenWindowFiller`, hence a second pgenlib reader; for VCF a second
`VcfRecordSource` open per window), and — because the consumer's decode is
`py.detach`'d but still serial with the track read — no overlap between decode and the track
read it feeds.

## 2. Rust: `ensure_current_window` + `with_current_window`

`StreamEngineCore::advance` (`src/ffi/stream_core.rs:324-399`) already owns the window
lifecycle: start the producer once, recycle the spent window, `recv` the next, then yield the
next `batch_size` row slice. Split it in two, with no control-flow change:

- `fn ensure_current_window(&self, state: &mut EngineState<B::Slot>) -> Result<(), NextSlice>`
  — the start/recycle/`recv` loop, returning `Ok(())` only when
  `state.current` exists **and** has rows left (`next_row < n_batch_rows`), leaving
  `next_row` untouched. The `Err` side carries only `NextSlice::Done`/`NextSlice::Failed`;
  `Ready` is constructed by `advance` alone.
- `fn advance(...) -> NextSlice` — `ensure_current_window`, then the existing slice-and-return
  from `current`. Byte-identical to today, including the zero-row-window behavior (such a
  window is skipped by the loop, exactly as now).

Then the accessor primitive:

```rust
pub(crate) fn with_current_window<R>(
    &self,
    f: impl FnOnce(usize, &B::Slot) -> anyhow::Result<R>,
) -> Option<anyhow::Result<R>>
```

It locks the state mutex, calls `ensure_current_window`, and on success invokes `f(job_idx,
&current.filled)` **while holding the lock** — the same discipline `next_batch_core` already
uses for `generate` (`:421-422`). `None` means plan exhaustion; `Some(Err(_))` is the
producer's error/panic, already join-then-classified. No GIL is needed inside `f`; the
blocking `recv` happens under the caller's `py.detach`.

## 3. Rust: `RecordStreamEngine.current_window_realign_inputs`

A new `#[pymethods]` entry on `RecordStreamEngine` (`src/record_stream/engine.rs:622`),
mirroring `window_realign_inputs`'s signature and return shape so the Python side is unchanged:

```rust
fn current_window_realign_inputs<'py>(
    &self, py: Python<'py>,
    contig_idx: usize,
    region_starts: Vec<u32>, region_ends: Vec<u32>,
    s_lo: usize, s_hi: usize,
) -> PyResult<Option<(
    Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>,
)>>
```

Behavior, all inside `py.detach(|| self.core.with_current_window(...))`:

1. **Identity validation first.** The closure compares the current job
   (`RecordBackend::jobs[job_idx]`, fields `contig_idx` / `regions` / `s_lo` / `s_hi`;
   `src/record_stream/engine.rs:60-65`) against the requested
   `(contig_idx, region_starts|region_ends, s_lo, s_hi)` and bails with a descriptive message
   on any difference. This is the change that turns "pair window N's tracks with window M's
   variants" from a silent parity break into a loud error, and it is cheap (one region-scale
   `Vec` compare per window). It is only sound because the drive builds the engine's jobs and
   the track window's bounds from the same `region_offsets` translation
   (`_streaming.py:2200-2219` vs `:1715-1725`).
2. **No decode.** A new inherent `RecordBackend::realign_inputs(&self, slot: &DecodedWindow)`
   clones the slot's four arrays into `Array1`s — `v_starts`/`ilens`/`geno_v_idxs`/
   `geno_offsets` — exactly the values `window_realign_inputs` returns. The arrays must be
   copied rather than moved: the slot is still needed to generate this window's batches
   (`EngineBackend::generate` reads `v_starts`/`geno_v_idxs`). The copy is window-scale
   (`geno_v_idxs` is the window's CSR entries), i.e. strictly cheaper than the decode that
   produced them, let alone a second decode.
3. **Exhaustion** returns `None`; `_record_mixed_realign_window` raises the drive's existing
   `"streaming engine exhausted before the plan did"` message, matching the batch loop.

`RecordBackend::debug_fill` and the existing `window_realign_inputs` pymethod are untouched
(§5).

## 4. Python: one seam, one engine

- `_MixedTracksBackend.mixed_realign_window` (`:488`) gains a trailing
  `engine: "object | None" = None`. `object` matches this file's existing engine-handle typing
  (`_mixed_engine() -> object`, `build_engine(...) -> object`), and the drive's engine class is
  imported lazily inside functions.
- `StreamingDataset._mixed_track_window` (`:1677`) gains `engine: "object | None" = None` and
  forwards it. Its two call
  sites: `:2277` (the `"engine"` drive) passes `engine=engine`; `:2657` (the SVAR2 `"sync"`
  drive, no engine exists) passes nothing.
- `_record_mixed_realign_window` (`:735`) takes `engine` and calls
  `engine.current_window_realign_inputs(contig_idx, starts, ends, s_lo, s_hi)` where it used to
  call `backend._mixed_engine().window_realign_inputs(...)`. With `engine is None` it raises a
  `ValueError` naming the required argument — a direct caller (a test) must supply a plan
  engine; there is no second-engine fallback, which is the point of the change.
- `_Svar1Backend.mixed_realign_window` (`:4020`) and `_Svar2Backend.mixed_realign_window`
  (`:4573`) accept and ignore `engine`, with a one-line docstring note (`SVAR1` reads its own
  store's offsets; SVAR2 reads its `read_window` bundle).
- Delete `_VcfBackend._mixed_engine` (`:4816`) and `_PgenBackend._mixed_engine` (`:5072`) plus
  their `_mixed_engine_obj` fields (`:4747`, `:5062`), and the `#307` comments that mention
  `_mixed_engine()`'s `build_engine([], 1, 1)` (`:4920`, `:5162`) — every remaining
  `build_engine` call now carries a real plan, so the `touched_contigs` optimization is
  unchanged but its motivating note needs rewording.

Ordering is preserved without restructuring the drive: the peek itself starts the producer and
receives the window (`ensure_current_window`), leaving `next_row == 0`; the existing
`for lo, hi in _batch_bounds(0, n_rows, batch_size)` loop then drains it. The engine's
`batch_size` is the same value passed to `build_engine`, so the row slices line up.

## 5. What stays, and why

- `window_realign_inputs` (`src/record_stream/engine.rs:1086`) **stays**, re-documented as
  test-only (precedent: `debug_decode_window`, `:1012`). It is the only independent oracle for
  `tests/dataset/test_streaming_tracks_record.py::test_record_window_csr_replicates_across_regions`:
  it decodes a caller-named window on a plan-less engine, so the test can keep comparing the
  mixed path's replicated CSR against a decode the mixed path did not produce. Deleting it
  would force that test to become a self-consistency check, which a consistently-wrong CSR
  replica can pass.
- `StreamEngineCore::backend()`'s M1 doc (`src/ffi/stream_core.rs:228-261`) and the
  `PgenWindowFiller::reader_lock` doc (`src/record_stream/pgen.rs:415-435`) both describe the
  consolidation as a future step and name the production caller of the ad-hoc decode. Both are
  now stale: the fold happened, and the remaining second callers are the two test-only decode
  accessors (`debug_decode_window` and `window_realign_inputs` themselves). The lock **stays**
  (the issue's own instruction): it is what keeps a shared filler correct, and those accessors
  still exercise it.

## 6. Non-goals

- **SVAR1 is not folded.** Its `mixed_realign_window` calls `self.read_window(r_idx, s_idx)`,
  which returns offsets borrowed from the store's `variant_idxs` mmap — no decode, so the fold
  would save nothing measurable while touching the parity-critical SVAR1 path. The fold is
  record-backend-only by construction: only those backends have a decode inside the engine.
- **The `jitter>0` + `realign_tracks=True` guard is not lifted** (`:2054-2066`). Its stated
  rationale (`_Svar1Backend.read_window` re-reads raw `_regions` bounds) survives for SVAR1,
  and the record-backend half is untested either way; that is
  [#383](https://github.com/mcvickerlab/GenVarLoader/issues/383)'s scope. The guard's comment
  gets one added sentence recording that record backends' decode is now jitter-translated.
- No new public symbol, so no `__all__`, `docs/source/api.md`, skill, or FAQ changes. This is a
  pure perf/correctness-neutral change to an existing path.
- `_streaming.py` is 5.2k lines and the mixed-tracks seam (`_MixedRealign`, `_RealignWindow`,
  `_Svar2Realign`, `_record_mixed_realign_window`, `_TrackWindow`, `_mixed_track_window`,
  `_mixed_tracks_batch`) is a natural module-extraction candidate. Deliberately not done here;
  it would swamp a review whose whole value is "same bytes, half the decode".

## 7. Gate

Standing convention: deterministic counters, never wall-clock on this shared node.

**Primary gate (new tests, `tests/dataset/test_streaming_tracks_record.py`).** For VCF and
PGEN, drive the same mixed fixture twice — once `realign_tracks=True`, once `False` — with
`transpose_word_reads_reset()` (both) and `pgen_variants_decoded_reset()` (PGEN) around each
sweep, and require the totals be **equal** and non-zero. `realign_tracks` cannot change the
engine's job list, so the decode totals must match; before the fold the realign run decodes
each window twice and the equality fails. The `> 0` half keeps it from passing vacuously.

**Parity (the safety net, unchanged):** the existing byte-identical suites on the touched paths
— `test_streaming_tracks_record.py`'s mixed parity cases, `test_streaming_tracks.py`,
`test_streaming_tracks_svar2.py`, `test_streaming_parity.py`, and the abandoned-iterator tests
(`test_streaming_abandoned_iter{,_mixed}.py`, #399) since the producer now starts earlier in
each window's composition.

**New guard test:** a `current_window_realign_inputs` call whose region bounds (or `s_lo`/`s_hi`)
do not match the current job raises `ValueError` — the deliberate-failure half of §3's identity
check, which no parity test can reach.

**Before/after numbers** are color, captured by running the same counter sweep against the
parent commit; the gate itself is the equality above.

Full verification: `pixi run -e dev maturin develop --release` before any Python test run, then
`pytest tests/dataset tests/unit -q`, `cargo test`, `cargo clippy`, `ruff check python/ tests/`,
`pyrefly`, and `python scripts/docstring_style.py --check python/genvarloader`.

Closing bookkeeping: the change is `perf(streaming):` (so it gets a changelog entry), and
`docs/roadmaps/streaming-dataset.md`'s Plans table row for #400 gets its status marker, PR
pointer, and the measured before/after counter numbers.

## 8. Risks

1. **Locking.** `with_current_window` holds the engine's state mutex while cloning
   window-scale arrays, and the blocking `recv` inside it must stay under `py.detach`
   (the #399 deadlock discipline). Any marshaling happens after the lock is released.
2. **Producer start time moves earlier** in each window's composition (before the track read
   instead of at the first `next_batch`). Semantically it is the same sequence position; the
   abandonment tests are the regression net.
3. **Identity validation is deliberately strict.** A future change that legitimately decodes a
   wider window than the track query must relax it explicitly rather than silently; the error
   message must name both sides to make that obvious.
4. **Zero-row windows** are skipped by `ensure_current_window` exactly as `advance` skips them
   today, so the fold cannot introduce a new skip — and an unexpected skip now trips the
   identity check instead of silently mispairing windows.
