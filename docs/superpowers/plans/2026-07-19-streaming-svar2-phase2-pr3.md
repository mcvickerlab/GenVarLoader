# SVAR2 Phase-2 PR-3 — read↔reconstruct pipeline engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the SVAR2 streaming backend a Rust producer-thread engine (`Svar2StreamEngine`) that runs the full read→gather→reconstruct chain GIL-free on a background thread, overlapping it with the consumer's cheap drain/marshal, so the serial GIL-held Python glue that dominated PR-2's wall comes off the critical path.

**Architecture:** Mirror the landed SVAR1 `Svar1StreamEngine` (`src/ffi/stream_engine.rs`) one format down. A detached `std::thread` producer, fed a region-scale plan, reconstructs one super-batch at a time into a recycled buffer and ping-pongs two buffers through `crossbeam_channel::bounded(2)` channels; a Python-visible `next_batch()` blocks on `recv` under `py.detach` and slices `batch_size` rows out of the reconstructed buffer. The engine lands behind the existing `_prefetch_strategy` seam as a new `"svar2_engine"` strategy; the SVAR2 default stays `"sync"` until the cold-cache A/B in the final task decides whether to flip it.

**Tech Stack:** Rust (PyO3/`pyo3`, `ndarray`/`numpy`, `crossbeam-channel`, `rayon`, `anyhow`), Python (numpy, `seqpro.rag.Ragged`), genoray_core (`query::find_ranges`, `query::ContigReader`), maturin, pixi, pytest, cargo test.

## Global Constraints

- **Target branch `streaming`**, not `main`. This branch (`278-svar2-phase2-pr3`) stacks on the merged PR-2 work on `278-svar2-streaming-backend`. Open the PR as a draft against `streaming`.
- **Byte-identical parity** with `gvl.write()` + `Dataset.open()[r, s]` under `.with_seqs("haplotypes")`, jitter=0, is the hard correctness gate. Perf is secondary color, never pass/fail (shared-node convention).
- **Deterministic iteration order is preserved** — the engine emits windows in plan order. Do NOT introduce relaxed/completion-order emission and do NOT change any iteration-order docs. (See the design doc "Deviations from the parent spec".)
- **No genoray rev bump.** `genoray_core::query::find_ranges` and `ContigReader` are public at the pinned rev.
- **No new public `__all__` symbol; no public signature change.** `docs/source/api.md` `__all__`-sync stays `MISSING: none`.
- **Rebuild Rust before running Python tests after ANY `src/` edit:** `pixi run -e dev maturin develop --release`. Skipping this silently imports the stale extension (CLAUDE.md).
- **`#284` memory bound holds:** peak reconstructed output is `2 × super-batch` (two ping-pong buffers), cohort-independent — never `O(n_samples)` output.
- **Process-wide counters only.** Any scale/entries counter this PR touches must be `AtomicUsize` (not `thread_local!`), because the read/reconstruct now runs on the producer thread (SVAR1 #296).

---

## Reference: the exact signatures this plan builds on

Copy-paste anchors (all under the worktree root). Read these files while implementing.

**SVAR1 engine template — `src/ffi/stream_engine.rs`** (mirror its structure):
- Structs `EngineState:125-135`, `CurrentWindow:116-122`, `Svar1StreamEngine:171-191`; `impl EngineState::new/Drop:138-168`; `build:197-228`; `ensure_started:232-304` (producer closure `:262-296`); `next_batch_core:314-378`; `generate_from_current:383-440`; `#[new]:456-490`; `next_batch:587-599`; inline `#[cfg(test)] mod tests:602-1135`.
- Channels: `let n_slots = 2usize; let (tx_filled, rx_filled) = bounded::<FilledWindow>(n_slots); let (tx_free, rx_free) = bounded::<FilledWindow>(n_slots);` then prefill `free` with 2 `FilledWindow::default()`.

**SVAR2 reconstruct chain — `src/svar2/mod.rs:495`:**
```rust
pub(crate) fn svar2_readbound_chain(
    reader: &genoray_core::query::ContigReader,
    region_starts_v: &[u32],
    orig_samples_v: &[usize],
    vk_snp_range_v: &[Range<usize>],
    vk_indel_range_v: &[Range<usize>],
    dense_snp_range_v: &[Range<usize>],
    dense_indel_range_v: &[Range<usize>],
    regions: numpy::ndarray::ArrayView2<i32>,   // per-row region bounds (m,2)
    shifts_a: numpy::ndarray::ArrayView2<i32>,   // (m,P) zeros (jitter=0)
    ref_a: numpy::ndarray::ArrayView1<u8>,
    ref_offsets_a: numpy::ndarray::ArrayView1<i64>,
    pad_char: u8,
    output_length: i64,                          // -1
    parallel: bool,
    filter_exonic: bool,                         // false
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<i64>,
)
```

**Range query — `genoray_core::query::find_ranges`** (called at `src/ffi/mod.rs:952`):
```rust
let rb = genoray_core::query::find_ranges(reader, &regions_v, Some(&samples_v));
// rb fields (Vec<Range<usize>>): vk_snp_range (len n_reg*n_s*P), vk_indel_range (same),
//   dense_snp_range (len n_reg), dense_indel_range (len n_reg); rb.sample_cols: Vec<usize> (len n_s).
```

**Store / buffer — `src/svar2/store.rs`:** `Svar2Store::new(store_path, contigs: Vec<String>, n_samples, ploidy)`; `Svar2Store::reader(&self, contig) -> Option<&ContigReader>`; `Svar2Store::store_path(&self) -> &str`. `Svar2ReconBuf::new(ploidy)`; `.set(data, offsets, n_rows)` (pub(crate)); `.batch(py, lo, hi)`; `.n_rows()`; `.total_bytes()`.

**Python gather to port — `_Svar2Backend._gather_rows` (`python/genvarloader/_dataset/_streaming.py:1102-1170`)** and its callers `read_window:1071`, `_fill_super_batch:1172`, `_drain:1219`. The row→(region,sample) expansion is `ri = rows // n_s; si = rows % n_s` in C-order.

**lib.rs registration — `src/lib.rs`:** `m.add_class::<ffi::stream_engine::Svar1StreamEngine>()?;` (`:28`); `m.add_function(wrap_pyfunction!(ffi::svar2_reconstruct_super_batch, m)?)?;` (`:56`).

---

## Task 1: GIL-free `svar2_fill_super_batch` core (find_ranges → gather → reconstruct)

The producer thread cannot call a pyfunction (no GIL), so the whole chain must exist as a **pub(crate) Rust core** `fill_super_batch_rs`. We also expose a thin GIL-holding pyfunction `svar2_fill_super_batch` wrapping it, used to prove byte-identity against the current `"sync"` path before the engine is built. This is the "move gather+read+reconstruct into one GIL-free call" primitive the engine reuses.

**Files:**
- Create: `src/svar2/window.rs` (the `Svar2WindowRanges` struct + `fill_super_batch_rs` core)
- Modify: `src/svar2/mod.rs` (`pub(crate) mod window;` + re-export)
- Modify: `src/ffi/mod.rs` (add `svar2_fill_super_batch` pyfunction)
- Modify: `src/lib.rs:56` area (register the pyfunction)
- Test: `tests/dataset/test_streaming_phase2_pr3.py` (new)

**Interfaces:**
- Produces (used by Task 2's engine):
  ```rust
  // src/svar2/window.rs
  pub(crate) struct Svar2WindowRanges {
      pub n_reg: usize,
      pub n_s: usize,
      pub ploidy: usize,
      pub vk_snp: Vec<std::ops::Range<usize>>,    // len n_reg*n_s*ploidy, C-order (reg,sample,ploid)
      pub vk_indel: Vec<std::ops::Range<usize>>,  // len n_reg*n_s*ploidy
      pub dense_snp: Vec<std::ops::Range<usize>>, // len n_reg
      pub dense_indel: Vec<std::ops::Range<usize>>,
      pub sample_cols: Vec<usize>,                // len n_s
  }
  impl Svar2WindowRanges {
      // Compute the window's ranges once (genoray find_ranges). `regions`/`samples` are physical.
      pub(crate) fn compute(
          reader: &genoray_core::query::ContigReader,
          regions: &[(u32, u32)],
          samples: &[usize],
          ploidy: usize,
      ) -> Self;
  }
  // Reconstruct C-order rows [sb_lo, sb_hi) of the window into out buffers (GIL-free).
  // `region_bounds` is the window's per-region (start,end) i32 pairs, len n_reg.
  #[allow(clippy::too_many_arguments)]
  pub(crate) fn fill_super_batch_rs(
      reader: &genoray_core::query::ContigReader,
      ranges: &Svar2WindowRanges,
      region_bounds: &[(i32, i32)],
      ref_bytes: &[u8],
      pad_char: u8,
      sb_lo: usize,
      sb_hi: usize,
      parallel: bool,
      out_data: &mut Vec<u8>,
      out_offsets: &mut Vec<i64>,
  );
  ```
- Consumes: `svar2_readbound_chain` (`src/svar2/mod.rs:495`), `find_ranges` (genoray).

- [ ] **Step 1: Write the failing parity test**

Create `tests/dataset/test_streaming_phase2_pr3.py`. It compares the new one-call Rust chain to the *existing* `"sync"` fill+drain, per plan window, byte-for-byte. Reuse the PR-2 fixture builder pattern (`tests/dataset/test_streaming_phase2_pr2.py` imports its fixture from `conftest`/a shared fixture — mirror however pr2 obtains `svar2_multicontig_fixture`).

```python
import numpy as np
import genvarloader as gvl
from genvarloader._dataset._streaming import _Svar2Backend


def _sync_reference(backend: _Svar2Backend, r_idx, s_idx, window, lo, hi):
    """Existing sync path: gather rows [lo,hi) + reconstruct into a fresh buffer, drain."""
    from genvarloader.genvarloader import Svar2ReconBuf
    buf = Svar2ReconBuf(backend.ploidy)
    backend._fill_super_batch(r_idx, s_idx, window, lo, hi, buf, parallel=False)
    return backend._drain(buf, 0, hi - lo)


def test_fill_super_batch_matches_sync_path(svar2_multicontig_fixture):
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert isinstance(backend, _Svar2Backend)
    from genvarloader.genvarloader import svar2_fill_super_batch

    for r_idx, s_idx in sds._plan():
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)
        # one-call Rust chain over the whole window
        contig_idx, contig = backend._contig_of(r_idx)
        rb = backend._regions[r_idx, 1:3]
        starts = np.ascontiguousarray(rb[:, 0], np.uint32)
        ends = np.ascontiguousarray(rb[:, 1], np.uint32)
        phys = np.ascontiguousarray(backend._phys_sample_idx[s_idx], np.int64)
        ref_, ref_offsets = backend._ref._contig_slice(contig_idx)
        data, offsets = svar2_fill_super_batch(
            backend._store, contig, starts, ends, phys,
            np.ascontiguousarray(rb, np.int32),
            ref_, ref_offsets, np.uint8(backend._ref.pad_char),
            0, n_rows, False,
        )
        # sync reference over the same rows
        ref_rag = _sync_reference(backend, r_idx, s_idx, window, 0, n_rows)
        np.testing.assert_array_equal(np.asarray(data).view("S1"), ref_rag.data)
        np.testing.assert_array_equal(np.asarray(offsets, np.int64), ref_rag.offsets)
```

- [ ] **Step 2: Run it to verify it fails (import error)**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr3.py -v`
Expected: FAIL — `ImportError: cannot import name 'svar2_fill_super_batch'`.

- [ ] **Step 3: Write `src/svar2/window.rs`**

Model `fill_super_batch_rs` on `svar2_reconstruct_super_batch`'s body (`src/ffi/mod.rs:1517-1576`) for the reconstruct call, and on `_gather_rows` (`_streaming.py:1132-1170`) for the row expansion. `Svar2WindowRanges::compute` calls `find_ranges` exactly as `svar2_read_window` does (`src/ffi/mod.rs:952`) but keeps `Vec<Range<usize>>` (no i64 flatten).

```rust
//! GIL-free SVAR2 window range computation + super-batch reconstruction.
//! The producer thread in `Svar2StreamEngine` runs this without the GIL; the
//! `svar2_fill_super_batch` pyfunction (src/ffi/mod.rs) is a thin GIL-holding wrapper
//! used to prove byte-identity vs the Python "sync" fill+drain path.
use std::ops::Range;

pub(crate) struct Svar2WindowRanges {
    pub n_reg: usize,
    pub n_s: usize,
    pub ploidy: usize,
    pub vk_snp: Vec<Range<usize>>,
    pub vk_indel: Vec<Range<usize>>,
    pub dense_snp: Vec<Range<usize>>,
    pub dense_indel: Vec<Range<usize>>,
    pub sample_cols: Vec<usize>,
}

impl Svar2WindowRanges {
    pub(crate) fn compute(
        reader: &genoray_core::query::ContigReader,
        regions: &[(u32, u32)],
        samples: &[usize],
        ploidy: usize,
    ) -> Self {
        let regions_v: Vec<(u32, u32)> = regions.to_vec();
        let samples_v: Vec<usize> = samples.to_vec();
        let rb = genoray_core::query::find_ranges(reader, &regions_v, Some(&samples_v));
        // field names verbatim from svar2_read_window (src/ffi/mod.rs:962-966)
        Self {
            n_reg: regions.len(),
            n_s: samples.len(),
            ploidy,
            vk_snp: rb.vk_snp_range,
            vk_indel: rb.vk_indel_range,
            dense_snp: rb.dense_snp_range,
            dense_indel: rb.dense_indel_range,
            sample_cols: rb.sample_cols,
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_super_batch_rs(
    reader: &genoray_core::query::ContigReader,
    ranges: &Svar2WindowRanges,
    region_bounds: &[(i32, i32)],
    ref_bytes: &[u8],
    pad_char: u8,
    sb_lo: usize,
    sb_hi: usize,
    parallel: bool,
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<i64>,
) {
    use numpy::ndarray::{Array2, ArrayView1};
    let n_s = ranges.n_s;
    let p = ranges.ploidy;
    let m = sb_hi - sb_lo;

    // Per-row expanded inputs (C-order (region, sample)), mirroring `_gather_rows`.
    let mut region_starts_v: Vec<u32> = Vec::with_capacity(m);
    let mut orig_samples_v: Vec<usize> = Vec::with_capacity(m);
    let mut vk_snp_v: Vec<Range<usize>> = Vec::with_capacity(m * p);
    let mut vk_indel_v: Vec<Range<usize>> = Vec::with_capacity(m * p);
    let mut dense_snp_v: Vec<Range<usize>> = Vec::with_capacity(m);
    let mut dense_indel_v: Vec<Range<usize>> = Vec::with_capacity(m);
    let mut region_bounds_a = Array2::<i32>::zeros((m, 2));
    let shifts_a = Array2::<i32>::zeros((m, p)); // jitter=0

    for (j, row) in (sb_lo..sb_hi).enumerate() {
        let ri = row / n_s;
        let si = row % n_s;
        let (rs, re) = region_bounds[ri];
        region_starts_v.push(rs as u32);
        region_bounds_a[[j, 0]] = rs;
        region_bounds_a[[j, 1]] = re;
        orig_samples_v.push(ranges.sample_cols[si]);
        let base = (ri * n_s + si) * p;
        for pp in 0..p {
            vk_snp_v.push(ranges.vk_snp[base + pp].clone());
            vk_indel_v.push(ranges.vk_indel[base + pp].clone());
        }
        dense_snp_v.push(ranges.dense_snp[ri].clone());
        dense_indel_v.push(ranges.dense_indel[ri].clone());
    }

    let ref_a: ArrayView1<u8> = ArrayView1::from(ref_bytes);
    let ref_offsets = [0i64, ref_bytes.len() as i64];
    let ref_offsets_a = ArrayView1::from(&ref_offsets[..]);

    out_data.clear();
    out_offsets.clear();
    crate::svar2::svar2_readbound_chain(
        reader,
        &region_starts_v,
        &orig_samples_v,
        &vk_snp_v,
        &vk_indel_v,
        &dense_snp_v,
        &dense_indel_v,
        region_bounds_a.view(),
        shifts_a.view(),
        ref_a,
        ref_offsets_a,
        pad_char,
        -1,
        parallel,
        false,
        out_data,
        out_offsets,
    );
}
```

- [ ] **Step 4: Wire the module + add the pyfunction wrapper**

In `src/svar2/mod.rs` add near the top: `pub(crate) mod window;`.

In `src/ffi/mod.rs` add the wrapper (model the signature/`py.detach` on `svar2_reconstruct_super_batch` at `:1497`, and the region/regions setup on its body). `region_bounds` in is the window's per-region (start,end) i32 pairs `(n_reg,2)`:
```rust
#[allow(clippy::too_many_arguments)]
#[pyfunction]
pub fn svar2_fill_super_batch<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar2::store::Svar2Store>,
    contig: &str,
    starts: PyReadonlyArray1<u32>,
    ends: PyReadonlyArray1<u32>,
    sample_idx: PyReadonlyArray1<i64>,
    region_bounds: PyReadonlyArray2<i32>,   // (n_reg, 2)
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,     // accepted for signature symmetry; ref slice is contiguous
    pad_char: u8,
    sb_lo: usize,
    sb_hi: usize,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
    use numpy::IntoPyArray;
    let starts_v: Vec<u32> = starts.as_slice()?.to_vec();
    let ends_v: Vec<u32> = ends.as_slice()?.to_vec();
    let regions_v: Vec<(u32, u32)> =
        starts_v.iter().zip(&ends_v).map(|(&s, &e)| (s, e)).collect();
    let samples_v: Vec<usize> = sample_idx.as_slice()?.iter().map(|&x| x as usize).collect();
    let rbnd = region_bounds.as_array();
    let region_bnd_v: Vec<(i32, i32)> =
        (0..rbnd.nrows()).map(|i| (rbnd[[i, 0]], rbnd[[i, 1]])).collect();
    let ref_v: Vec<u8> = ref_.as_slice()?.to_vec();
    let _ = ref_offsets; // ref slice is a single contiguous contig; offsets are [0, len]
    let (data, offsets) = py.detach(move || -> PyResult<(Vec<u8>, Vec<i64>)> {
        let reader = store
            .reader(contig)
            .ok_or_else(|| PyValueError::new_err(format!("contig {contig} not in store")))?;
        let ranges = crate::svar2::window::Svar2WindowRanges::compute(
            reader, &regions_v, &samples_v, store_ploidy(&store),
        );
        let mut out_data = Vec::new();
        let mut out_offsets = Vec::new();
        crate::svar2::window::fill_super_batch_rs(
            reader, &ranges, &region_bnd_v, &ref_v, pad_char, sb_lo, sb_hi, parallel,
            &mut out_data, &mut out_offsets,
        );
        Ok((out_data, out_offsets))
    })?;
    Ok((data.into_pyarray(py), offsets.into_pyarray(py)))
}
```
> Note: `store_ploidy(&store)` — read the store's ploidy. If `Svar2Store` has no ploidy getter, add a `pub fn ploidy(&self) -> usize` to `src/svar2/store.rs` returning the value passed at construction (store it in the struct if not already), and call `store.ploidy()`. Prefer that over threading ploidy through the signature.

Register in `src/lib.rs` next to `:56`:
```rust
m.add_function(wrap_pyfunction!(ffi::svar2_fill_super_batch, m)?)?;
```
Ensure `svar2_fill_super_batch` is re-exported from `ffi` the same way `svar2_reconstruct_super_batch` is (check the `pub use`/module layout in `src/ffi/mod.rs` and match it).

- [ ] **Step 5: Rebuild Rust + run the test**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr3.py::test_fill_super_batch_matches_sync_path -v`
Expected: PASS (byte-identical data + offsets for every plan window).

- [ ] **Step 6: cargo + lint + typecheck**

Run: `cargo build 2>&1 | tail -5 && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: builds clean; no new lint/type errors.

- [ ] **Step 7: Commit**

```bash
git add src/svar2/window.rs src/svar2/mod.rs src/ffi/mod.rs src/lib.rs tests/dataset/test_streaming_phase2_pr3.py
git commit -m "feat(streaming): GIL-free svar2_fill_super_batch core (find_ranges->gather->reconstruct) (#278)"
```

---

## Task 2: `Svar2StreamEngine` pyclass (producer-thread pipeline)

Mirror `Svar1StreamEngine` (`src/ffi/stream_engine.rs`). **Copy its concurrency scaffolding verbatim** — the only substantive changes are (a) the `FilledWindow` carries a *reconstructed super-batch* (`data`/`offsets`/`n_rows`), (b) the producer calls `fill_super_batch_rs` per super-batch job instead of `read_window`, and (c) the consumer *slices* the buffer instead of reconstructing.

**Files:**
- Create: `src/ffi/svar2_stream_engine.rs`
- Modify: `src/ffi/mod.rs` (`pub mod svar2_stream_engine;`)
- Modify: `src/lib.rs:28` area (register the pyclass)

**Interfaces:**
- Produces (used by Task 3):
  ```rust
  #[pyclass]
  pub struct Svar2StreamEngine { /* ... */ }
  #[pymethods] impl Svar2StreamEngine {
      #[new]
      fn new(
          store_path: &str,
          store_contigs: Vec<String>,   // ALL store contigs, for Svar2Store::new
          n_samples: usize,             // store's full sample count
          ploidy: usize,
          // per-contig reference bytes, indexed by job_contig_idx:
          contig_names: Vec<String>,
          contig_ref_bytes: Vec<Vec<u8>>,
          phys_sample_idx: Vec<usize>,  // public-order -> physical column, len == this backend's n_samples
          // per-job (window) plan, region-scale:
          job_contig_idx: Vec<usize>,   // index into contig_names/contig_ref_bytes
          job_region_starts: Vec<Vec<u32>>,
          job_region_ends: Vec<Vec<u32>>,
          job_s_lo: Vec<usize>,
          job_s_hi: Vec<usize>,
          pad_char: u8,
          super_batch_rows: usize,
          batch_size: usize,
      ) -> PyResult<Self>;
      fn next_batch<'py>(&self, py: Python<'py>)
          -> PyResult<Option<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>>;
  }
  ```
- Consumes: `fill_super_batch_rs`, `Svar2WindowRanges` (Task 1); `Svar2Store` (`src/svar2/store.rs`).

- [ ] **Step 1: Write the failing Rust unit test (plan-order yield)**

Add an inline `#[cfg(test)] mod tests` at the bottom of `src/ffi/svar2_stream_engine.rs`, mirroring `stream_engine.rs:602-1135`. Reuse that module's fixture-building approach for a small `.svar2` store (build a fixture store on a tempdir; if the SVAR1 test builds its store from raw bytes, build an SVAR2 store analogously, or open a checked-in tiny fixture — match whatever `stream_engine.rs` tests do). First test asserts plan-order byte-equality vs a direct `fill_super_batch_rs`:

```rust
#[test]
fn svar2_stream_engine_yields_windows_in_plan_order() {
    // Build a >=2-window plan over a small store; run the engine at batch_size = window rows.
    // Assert: batch k byte-equals fill_super_batch_rs over window k's rows; exhaustion -> None (idempotent).
    // (Mirror stream_engine.rs::svar1_stream_engine_yields_windows_in_plan_order:716.)
}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cargo test svar2_stream_engine_yields_windows_in_plan_order 2>&1 | tail -15`
Expected: FAIL to compile — `Svar2StreamEngine` does not exist yet.

- [ ] **Step 3: Write the engine — structs + scaffolding**

Create `src/ffi/svar2_stream_engine.rs`. Copy the imports and the `EngineState`/`Drop`/channel discipline from `stream_engine.rs` verbatim. The SVAR2-specific structs:

```rust
//! SVAR2 read↔reconstruct pipeline engine — the SVAR2 analog of `Svar1StreamEngine`.
//! Producer thread runs find_ranges -> gather -> reconstruct (GIL-free) per super-batch,
//! ping-ponging two reconstructed buffers through bounded(2) channels; the consumer
//! slices `batch_size` rows out of the current buffer under `py.detach`. Concurrency
//! discipline (detached producer, shutdown-by-Sender-drop, join-then-classify) is copied
//! from `stream_engine.rs`.
use std::ops::Range;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use crate::svar2::store::Svar2Store;
use crate::svar2::window::{fill_super_batch_rs, Svar2WindowRanges};

// Per-contig reference bytes, indexed by job.contig_idx.
struct ContigRef { name: String, ref_bytes: Vec<u8> }

// One super-batch job: a row span [sb_lo, sb_hi) within a window's cartesian rows.
struct SbJob {
    contig_idx: usize,
    regions: Vec<(u32, u32)>,   // window regions (start,end)
    s_lo: usize,                // into phys_sample_idx
    s_hi: usize,
    sb_lo: usize,               // row span within the window (0..n_reg*n_s)
    sb_hi: usize,
}

#[derive(Default)]
struct FilledWindow { data: Vec<u8>, offsets: Vec<i64>, n_rows: usize, job_idx: usize }

struct CurrentWindow { filled: FilledWindow, next_row: usize }

// EngineState, impl new/Drop: COPY from stream_engine.rs:125-168 (FilledWindow type differs only).
```

Engine struct + `build`/`new`/`ensure_started`. The producer **precomputes super-batch jobs** by expanding each window job into `ceil(n_rows / super_batch_rows)` `SbJob`s, and computes `find_ranges` **once per window** (cache the `Svar2WindowRanges` across a window's super-batches):

```rust
#[pyclass]
pub struct Svar2StreamEngine {
    store: Arc<Svar2Store>,
    contig_refs: Arc<Vec<ContigRef>>,
    jobs: Arc<Vec<SbJob>>,          // super-batch-granular, plan order
    phys_sample_idx: Arc<Vec<usize>>,
    ploidy: usize,
    pad_char: u8,
    parallel_threshold_rows: usize, // super_batch large enough -> parallel=true (mirror should_parallelize intent)
    batch_size: usize,
    state: Mutex<EngineState>,
}
```

Producer closure (inside `ensure_started`, mirroring `stream_engine.rs:262-296`) — note the per-window `find_ranges` cache keyed by consecutive jobs sharing `(contig_idx, regions, s_lo, s_hi)`:

```rust
let mut cached: Option<((usize, usize, usize), Svar2WindowRanges, Vec<(i32,i32)>)> = None;
for (job_idx, job) in jobs.iter().enumerate() {
    let Ok(mut slot) = rx_free.recv() else { return Ok(()); };
    let cref = &contig_refs[job.contig_idx];               // panics on OOB (tested)
    let reader = store.reader(&cref.name)
        .ok_or_else(|| anyhow::anyhow!("contig {} not in store", cref.name))?;
    let phys: Vec<usize> = phys_sample_idx[job.s_lo..job.s_hi].to_vec();
    let key = (job.contig_idx, job.s_lo, job.s_hi);
    // recompute ranges only when the window (contig+sample span+regions) changes
    let need = cached.as_ref().map(|(k, _, rb)| *k != key
        || rb.len() != job.regions.len()
        || rb.iter().zip(&job.regions).any(|((rs,_),(js,_))| *rs as u32 != *js)).unwrap_or(true);
    if need {
        let region_bnd: Vec<(i32,i32)> =
            job.regions.iter().map(|&(s,e)| (s as i32, e as i32)).collect();
        let ranges = Svar2WindowRanges::compute(reader, &job.regions, &phys, ploidy);
        cached = Some((key, ranges, region_bnd));
    }
    let (_, ranges, region_bnd) = cached.as_ref().unwrap();
    let n_rows = job.sb_hi - job.sb_lo;
    let parallel = n_rows * ploidy >= parallel_threshold_rows;
    fill_super_batch_rs(reader, ranges, region_bnd, &cref.ref_bytes, pad_char,
        job.sb_lo, job.sb_hi, parallel, &mut slot.data, &mut slot.offsets);
    slot.n_rows = n_rows;
    slot.job_idx = job_idx;
    if tx_filled.send(slot).is_err() { return Ok(()); }
}
Ok(())
```
> The window/region cache key is a simplification — because `_plan` yields one window per (contig, region-chunk, sample-chunk) and a window's super-batches are consecutive jobs, keying on `(contig_idx, s_lo, s_hi)` + region-start match is sufficient. If simpler, carry a `window_idx: usize` on `SbJob` (assigned in Python) and key on that; either is fine. Prefer whichever the Task-2 implementer finds least error-prone — correctness (a fresh `find_ranges` when the window changes) is the invariant.

- [ ] **Step 4: Write `next_batch_core` + `next_batch` (consumer slices the buffer)**

Copy `next_batch_core`'s loop/recv/recycle/join-then-classify from `stream_engine.rs:314-378`. The difference: there is **no reconstruct on the consumer** — slice the pre-reconstructed buffer:

```rust
fn slice_current(&self, cur: &mut CurrentWindow) -> (Array1<u8>, Array1<i64>) {
    let p = self.ploidy;
    let row_lo = cur.next_row;
    let row_hi = (row_lo + self.batch_size).min(cur.filled.n_rows);
    cur.next_row = row_hi;
    let o_lo = row_lo * p;
    let o_hi = row_hi * p;
    let byte_lo = cur.filled.offsets[o_lo] as usize;
    let byte_hi = cur.filled.offsets[o_hi] as usize;
    let data = Array1::from(cur.filled.data[byte_lo..byte_hi].to_vec());
    let offsets: Vec<i64> =
        cur.filled.offsets[o_lo..=o_hi].iter().map(|&o| o - byte_lo as i64).collect();
    (data, Array1::from(offsets))
}
```
`next_batch_core`: while `cur.next_row < cur.filled.n_rows` → `Some(Ok(self.slice_current(cur)))`; else recycle spent buffer to `tx_free`, `rx_filled.recv()` → new `CurrentWindow { filled, next_row: 0 }`, loop; on `Err(_)` join-then-classify exactly as SVAR1. `next_batch<'py>` copies `stream_engine.rs:587-599` verbatim (`py.detach` + `into_pyarray`).

- [ ] **Step 5: Register + write remaining adversarial tests**

`src/ffi/mod.rs`: add `pub mod svar2_stream_engine;`. `src/lib.rs` near `:28`: `m.add_class::<ffi::svar2_stream_engine::Svar2StreamEngine>()?;`.

Add the remaining inline tests, one per SVAR1 engine test (mirror names/asserts from the D.2 list):
- `svar2_stream_engine_splits_windows_into_bounded_batches` (batch_size=1 splits; concat == full-window fill).
- `svar2_stream_engine_empty_plan_is_none` (no jobs → `None`, no hang).
- `svar2_stream_engine_producer_error_surfaces_not_eof` (bad contig name → `Some(Err)` containing the error text, not "panicked").
- `svar2_stream_engine_producer_panic_surfaces_not_hang` (OOB `contig_idx` → `Err` join branch, msg contains "panicked").
- `svar2_stream_engine_drop_midstream_joins_cleanly` (drop engine mid-stream, producer parked on `rx_free.recv()` → clean join).

- [ ] **Step 6: Run the Rust tests 20× in release (race stability)**

Run: `for i in $(seq 20); do cargo test --release svar2_stream_engine 2>&1 | grep -E "test result|panicked|FAILED" ; done`
Expected: every iteration `test result: ok`; no `FAILED`, no hang.

- [ ] **Step 7: Commit**

```bash
git add src/ffi/svar2_stream_engine.rs src/ffi/mod.rs src/lib.rs
git commit -m "feat(streaming): Svar2StreamEngine producer-thread pipeline (#278)"
```

---

## Task 3: Python wiring — `build_engine` + `"svar2_engine"` strategy + parity

Wire the engine behind the `_prefetch_strategy` seam. Keep `_Svar2Backend._default_strategy = "sync"` (the A/B baseline; Task 4 decides the flip). Add byte-identical parity under the engine strategy and cross-strategy identity vs `"sync"`.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_Svar2Backend.build_engine`; `"svar2_engine"` branch in `_iter_batches`; a test seam to select the strategy)
- Test: `tests/dataset/test_streaming_phase2_pr3.py` (extend), `tests/dataset/test_streaming_parity_svar2.py` (extend)

**Interfaces:**
- Consumes: `Svar2StreamEngine` (Task 2), `_Svar2Backend` fields (`_store`, `_regions`, `_phys_sample_idx`, `_contigs`, `_ref`, `ploidy`, `n_samples`, `_super_batch_rows`, `_sv`).
- Produces: `_Svar2Backend.build_engine(engine_jobs, batch_size) -> object`; a constructed `StreamingDataset` selecting `"svar2_engine"`.

- [ ] **Step 1: Write the failing cross-strategy identity test**

Add to `tests/dataset/test_streaming_phase2_pr3.py`. Needs a way to select the engine strategy. Add a tiny test seam first (Step 3 implements it): `_with_strategy(sds, "svar2_engine")` using `object.__setattr__` (StreamingDataset is a frozen dataclass; the codebase already uses `object.__setattr__` — `_streaming.py:249`). But the engine also needs `_super_batch_rows` on the backend and the strategy honored in `_iter_batches`.

```python
def _collect(sds, batch_size):
    """All (r, s) -> haplotype rows, keyed by cell, as a dict for order-independent compare."""
    out = {}
    for data, r_idx, s_idx in sds.to_iter(batch_size=batch_size, return_indices=True):
        for i in range(len(r_idx)):
            out[(int(r_idx[i]), int(s_idx[i]))] = np.asarray(data[i]).copy()
    return out


def test_svar2_engine_matches_sync_bytewise(svar2_multicontig_fixture):
    fx = svar2_multicontig_fixture
    base = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    sync = _collect(base, batch_size=4)  # default "sync"
    eng_sds = _with_strategy(base, "svar2_engine")
    eng = _collect(eng_sds, batch_size=4)
    assert set(sync) == set(eng)
    for cell in sync:
        for p in range(base._backend.ploidy):
            np.testing.assert_array_equal(sync[cell][p], eng[cell][p])
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr3.py::test_svar2_engine_matches_sync_bytewise -v`
Expected: FAIL — `_with_strategy` undefined / `"svar2_engine"` unhandled `ValueError`.

- [ ] **Step 3: Add `build_engine` + the `"svar2_engine"` branch + the seam**

`_Svar2Backend.build_engine` (mirror `_Svar1Backend.build_engine`, `_streaming.py:822`). `engine_jobs` are the same region-scale jobs the SVAR1 `"engine"` branch builds `(contig_idx, region_starts, region_ends, s_lo, s_hi)`:

```python
def build_engine(
    self,
    jobs: list[tuple[int, NDArray[np.uint32], NDArray[np.uint32], int, int]],
    batch_size: int,
) -> object:
    from ..genvarloader import Svar2StreamEngine

    contig_names = list(self._contigs)
    contig_ref_bytes = [
        np.asarray(self._ref._contig_slice(i)[0], np.uint8).tobytes()
        for i in range(len(contig_names))
    ]
    job_contig_idx = [int(j[0]) for j in jobs]
    job_region_starts = [np.ascontiguousarray(j[1], np.uint32) for j in jobs]
    job_region_ends = [np.ascontiguousarray(j[2], np.uint32) for j in jobs]
    job_s_lo = [int(j[3]) for j in jobs]
    job_s_hi = [int(j[4]) for j in jobs]
    phys = self._phys_sample_idx.astype(np.int64, copy=False).tolist()
    return Svar2StreamEngine(
        str(self._sv.path) if hasattr(self._sv, "path") else self._store_path,
        list(self._sv.contigs),
        int(self._sv.n_samples),
        self.ploidy,
        contig_names,
        contig_ref_bytes,
        phys,
        job_contig_idx,
        job_region_starts,
        job_region_ends,
        job_s_lo,
        job_s_hi,
        int(self._ref.pad_char),
        int(self._super_batch_rows),
        batch_size,
    )
```
> `Svar2StreamEngine.new` takes `contig_ref_bytes: Vec<Vec<u8>>`; pass `bytes` objects (pyo3 maps `bytes` → `Vec<u8>`). Store the `.svar2` path on the backend at construction as `self._store_path = str(svar2_path)` (add it in `__init__` if absent) so `build_engine` doesn't depend on a `SparseVar2.path` attribute.

The `"svar2_engine"` branch in `_iter_batches` mirrors the SVAR1 `"engine"` branch (`:339-403`) — same `flat_r`/`flat_s`, same lockstep drive; only the isinstance check + `build_engine` call differ. Insert it after the `"sync"` branch (before the final `else` `ValueError`), and add `"svar2_engine"` to the error message's accepted list:

```python
elif self._prefetch_strategy == "svar2_engine":
    assert isinstance(self._backend, _Svar2Backend), (
        '"svar2_engine" prefetch strategy requires the SVAR2 backend'
    )
    backend = self._backend
    plan_jobs: list[tuple[int, NDArray[np.intp], int, int]] = []
    for r_idx, s_idx in self._plan():
        contig_idx = int(self._regions[r_idx[0], 0])
        plan_jobs.append((contig_idx, r_idx, int(s_idx[0]), int(s_idx[-1]) + 1))
    engine_jobs = [
        (
            c_idx,
            np.ascontiguousarray(self._regions[r_idx, 1], np.uint32),
            np.ascontiguousarray(self._regions[r_idx, 2], np.uint32),
            s_lo,
            s_hi,
        )
        for (c_idx, r_idx, s_lo, s_hi) in plan_jobs
    ]
    engine = backend.build_engine(engine_jobs, batch_size)
    del engine_jobs
    for _c_idx, r_idx, s_lo, s_hi in plan_jobs:
        n_s = s_hi - s_lo
        flat_r = np.repeat(self._sort_order[r_idx], n_s)
        flat_s = np.tile(np.arange(s_lo, s_hi, dtype=np.intp), len(r_idx))
        n_rows = len(flat_r)
        for lo in range(0, n_rows, batch_size):
            hi = min(lo + batch_size, n_rows)
            nxt = engine.next_batch()
            if nxt is None:
                raise RuntimeError("Svar2StreamEngine exhausted before the plan did")
            data, offsets = nxt
            yield (
                Ragged.from_offsets(
                    np.asarray(data).view("S1"),
                    (hi - lo, backend.ploidy, None),
                    np.asarray(offsets, np.int64),
                ),
                flat_r[lo:hi],
                flat_s[lo:hi],
            )
    if engine.next_batch() is not None:
        raise RuntimeError("Svar2StreamEngine had extra batches beyond the plan")
```

Add the test seam near the top of `tests/dataset/test_streaming_phase2_pr3.py`:
```python
def _with_strategy(sds, strategy):
    import copy
    clone = copy.copy(sds)
    object.__setattr__(clone, "_prefetch_strategy", strategy)
    return clone
```

- [ ] **Step 4: Rebuild + run the identity + parity tests**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr3.py tests/dataset/test_streaming_parity_svar2.py -v`
Expected: PASS. (`test_streaming_parity_svar2.py` still green under default `"sync"`; the new identity test green under `"svar2_engine"`.)

- [ ] **Step 5: Add engine-strategy parity (position-based) test**

Add a test that runs the *existing* parity assertion under the engine strategy — the strongest guarantee (byte-identical to the written dataset, not just to sync):

```python
def test_svar2_engine_matches_written(svar2_multicontig_fixture):
    fx = svar2_multicontig_fixture
    written = gvl.Dataset.open(fx.written_path, fx.reference_path).with_seqs("haplotypes")
    base = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    sds = _with_strategy(base, "svar2_engine")
    seen = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=3, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            for p in range(base._backend.ploidy):
                np.testing.assert_array_equal(data[i][p], written[r, s][p])
            seen += 1
    assert seen == fx.bed.height * base.n_samples
```
> If `svar2_multicontig_fixture` doesn't expose a `written_path`, build the written dataset in the test via `gvl.write(...)` the same way `test_streaming_parity_svar2.py` obtains its written oracle (follow that file's pattern exactly).

- [ ] **Step 6: Run full scoped suite + lint/type**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q && pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck`
Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_phase2_pr3.py tests/dataset/test_streaming_parity_svar2.py
git commit -m "feat(streaming): wire Svar2StreamEngine behind svar2_engine strategy + parity (#278)"
```

---

## Task 4: #284 gate under engine, cold-cache A/B measurement, ship decision, docs

Prove the engine preserves the cohort-independent memory bound, measure it against `"sync"` cold-cache, decide the default, and update the roadmap. Ship gated either way.

**Files:**
- Test: `tests/dataset/test_streaming_scale.py` (extend)
- Modify: `benchmarking/streaming/svar2_cold_cache.py` (add engine-vs-sync A/B) — or add a sibling `svar2_engine_ab.py` following `benchmarking/streaming/cold_cache_overlap.py`'s structure
- Modify (only if measurement says flip): `python/genvarloader/_dataset/_streaming.py` (`_Svar2Backend._default_strategy`)
- Modify: `docs/roadmaps/streaming-dataset.md`

**Interfaces:** consumes everything from Tasks 1-3. No new public symbols.

- [ ] **Step 1: Write the #284 cohort-scale gate under the engine**

Add to `tests/dataset/test_streaming_scale.py`, mirroring `test_svar2_super_batch_buffer_is_flat_in_cohort_size:379` but driving the engine strategy. Assert the drained batch output byte count at a fixed `batch_size` is **identical across two cohort sizes** (cohort-independent) while the plan still covers every sample:

```python
def test_svar2_engine_output_is_flat_in_cohort_size(svar2_scale_fixture_factory):
    def one(n_samples):
        fx = svar2_scale_fixture_factory(n_samples=n_samples)
        base = gvl.StreamingDataset(
            fx.bed, reference=fx.reference_path, variants=fx.svar2_path, max_mem="64MB"
        ).with_seqs("haplotypes")
        sds = _with_strategy(base, "svar2_engine")
        total = 0
        for data, _r, _s in sds.to_iter(batch_size=4, return_indices=True):
            total += int(np.asarray(data.data).nbytes)
        return total, base.n_samples
    small, n_small = one(50)
    large, n_large = one(400)
    assert n_small == 50 and n_large == 400        # plan covers the whole cohort
    assert small == large and small > 0            # per-batch output is cohort-independent
```
> Use whatever the existing SVAR2 scale tests use to build cohort-scale fixtures (`test_svar2_generate_batch_output_is_flat_in_cohort_size:264` shows the builder). Import `_with_strategy` from the pr3 test module or duplicate the 4-line helper.

- [ ] **Step 2: Rebuild + run the gate**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_scale.py -k svar2 -v`
Expected: PASS — `small == large`, both non-zero.

- [ ] **Step 3: Commit the gate**

```bash
git add tests/dataset/test_streaming_scale.py
git commit -m "test(streaming): SVAR2 engine cohort-independence gate (#284) (#278)"
```

- [ ] **Step 4: Add the cold-cache A/B harness**

Extend `benchmarking/streaming/svar2_cold_cache.py` (or add `svar2_engine_ab.py`) to time `to_iter` under `"sync"` vs `"svar2_engine"` on the vcfixture cohort store, fresh store per run, page-evicted (`posix_fadvise(DONTNEED)` — copy the eviction helper already in `svar2_cold_cache.py`), best-of-N, reporting each strategy's wall range. Follow the module-docstring + `VCFIXTURE_BIN` skip convention of the sibling benchmarks; select the engine via the same `object.__setattr__` seam. Header comment must state: perf is secondary color; ship rule = engine beats sync outside node noise (non-overlapping ranges).

- [ ] **Step 5: Run the A/B (measurement, not a gate)**

Run:
```bash
VCFIXTURE_BIN=/carter/users/dlaub/projects/vcfixture-rs/target/release/vcfixture \
  pixi run -e dev python benchmarking/streaming/svar2_cold_cache.py
```
Record both strategies' best-of-N wall ranges. **Decision rule:** if `"svar2_engine"`'s range is entirely below `"sync"`'s (non-overlapping, the SVAR1 Task-9 bar), it ships as the SVAR2 default → do Step 6. Otherwise keep `"sync"` as default and record the reason → skip Step 6.

- [ ] **Step 6 (conditional): Flip the SVAR2 default**

Only if Step 5 cleared the bar. Change `_Svar2Backend._default_strategy = "sync"` → `"svar2_engine"` (`_streaming.py:1008`). Re-run `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py tests/dataset/test_streaming_phase2_pr3.py -q` (now exercising the engine by default) — expect all green. Commit:
```bash
git add python/genvarloader/_dataset/_streaming.py
git commit -m "perf(streaming): default SVAR2 streaming to the pipeline engine (measured) (#278)"
```

- [ ] **Step 7: Update the roadmap + the design's status**

In `docs/roadmaps/streaming-dataset.md`: add a PR-3 plan row under "Plans (spec A)" with the cold-cache engine-vs-sync numbers + the ship/no-ship decision (mirror the SVAR1 Task-9 entry's format), and tick Phase 2's PR-3 as done. Flip the design doc's status line to reflect the landed decision. Commit:
```bash
git add docs/roadmaps/streaming-dataset.md docs/superpowers/specs/2026-07-19-streaming-svar2-phase2-pr3-design.md
git commit -m "docs(streaming): record SVAR2 PR-3 engine measurement + decision (#278)"
```

- [ ] **Step 8: Full-tree sanity before the PR**

Run: `pixi run -e dev pytest tests -q && python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: full tree green; `MISSING: none`.

- [ ] **Step 9: Push + open the draft PR against `streaming`**

```bash
git push -u origin 278-svar2-phase2-pr3
gh pr create --draft --base streaming --title "streaming(svar2): Phase-2 PR 3 — read↔reconstruct pipeline engine (#278)" --body "$(cat <<'EOF'
Realizes the parent Phase-2 spec's Lever 3 as a Rust producer-thread engine (`Svar2StreamEngine`),
mirroring the landed SVAR1 `Svar1StreamEngine`. The producer runs find_ranges → gather → reconstruct
GIL-free per super-batch, ping-ponging two buffers through bounded(2) channels; the consumer slices
`batch_size` rows under `py.detach`. This moves PR-2's dominant serial GIL-held glue off the critical
path and overlaps it with reconstruct.

Deterministic order preserved (single ordered producer) — no iteration-order contract change.
Shipped gated behind the `_prefetch_strategy` seam; the cold-cache A/B decision is recorded in the roadmap.

Closes #278 (Phase 2 PR 3). Part of the StreamingDataset effort; targets `streaming`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review

**Spec coverage** (design doc → task):
- Rust producer-thread engine mirroring SVAR1 discipline → Tasks 1-2.
- Full chain GIL-free on producer (gather off the GIL) → Task 1 (`fill_super_batch_rs`) + Task 2 (producer calls it).
- `bounded(2)` ping-pong, shutdown-by-Sender-drop, join-then-classify, `next_batch` under `py.detach` → Task 2 (copied from `stream_engine.rs`).
- Deterministic order kept, no relaxed-order, no iteration-order doc change → Global Constraints + Task 3 (mirrors the in-order `"engine"` branch) + no doc edits in Task 4.
- Gated behind `_prefetch_strategy`, default stays `"sync"` until measured → Task 3 (keeps default) + Task 4 (measures, conditional flip).
- Parity byte-identical, position-based → Task 3 (`test_svar2_engine_matches_written`); cross-strategy identity → Task 3 (`test_svar2_engine_matches_sync_bytewise`).
- #284 cohort-independence under the engine → Task 4 Step 1.
- Adversarial engine suite 20× release → Task 2 Steps 1,5,6.
- Process-wide counters (no `thread_local!`) → Global Constraints (this PR adds no new counter; the constraint stands as a guard).
- Cold-cache A/B + ship decision recorded → Task 4 Steps 4-7.
- No `__all__`/api change → Task 4 Step 8 check.

**Placeholder scan:** no "TBD"/"handle edge cases"/"similar to Task N" — the mechanical mirror steps name exact `stream_engine.rs` line ranges to copy and state the specific delta; all novel code is written out.

**Type consistency:** `FilledWindow{data,offsets,n_rows,job_idx}` used identically in Task 2 Steps 3-4; `fill_super_batch_rs`/`Svar2WindowRanges` signatures in Task 1 match their call sites in Task 2's producer and the Task 1 pyfunction; `build_engine`'s argument order matches `Svar2StreamEngine::new`'s parameter list; `"svar2_engine"` is the single strategy string across Tasks 3-4.

## Execution notes

This is a **dependency chain** (Task 1 → 2 → 3 → 4), not a parallel fan-out — each task consumes the prior task's Rust/Python surface, so tasks run sequentially with a review gate between them (no parallel dispatch). Per project convention, use **subagent-driven-development** with a fresh Sonnet subagent per task and two-stage review; escalate to a stronger model only for a second-pass fix if an implementer critically fails a task. Every task rebuilds Rust (`maturin develop --release`) before its Python tests — the stale-extension trap (CLAUDE.md) is the most likely silent failure here.
