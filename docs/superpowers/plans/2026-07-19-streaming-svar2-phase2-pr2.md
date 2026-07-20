# SVAR2 streaming Phase-2 PR 2 (super-batch parallel reconstruction) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. **Implementers: use Sonnet or a weaker model** (per project convention); reserve stronger models for second-pass fixes only.

**Goal:** Make SVAR2 streaming reconstruction multi-core by decoupling the reconstruct granularity from the user's `batch_size` — reconstruct one coarse, `should_parallelize`-gated **super-batch** of rows per call into a recycled owned buffer, then drain `batch_size` slices from it — with byte-identical parity preserved.

**Architecture:** Today the "sync" SVAR2 drive reconstructs one FFI call *per `batch_size` slice* with a hard-coded `parallel=False` (`_streaming.py:1111`), so the rayon dispatch grain (`~batch_size × ploidy` tiny haplotypes) is far too small to saturate cores. PR 2 inserts a **super-batch** between the read window and the user batch: `read window ⊃ reconstruct super-batch ⊃ user batch`. A new Rust `Svar2ReconBuf` pyclass owns the output buffer (allocated once, refilled in place across super-batches); a new `svar2_reconstruct_super_batch` FFI runs the *unchanged* gather→split→hap_diffs→reconstruct chain (lifted into a shared helper) into that buffer with `parallel=should_parallelize(total_out_bytes)`; a `Svar2ReconBuf.batch(lo, hi)` accessor copies out one drained batch. The Rust kernel already writes each `(query, hap)` row into its own disjoint output slice regardless of `parallel` (`reconstruct/mod.rs:751-801`), so output order — and therefore parity — is unchanged. The recycled buffer is exactly the ping-pong slot PR 3's pipeline will reuse.

**Tech Stack:** Python 3.10+ (abi3), Rust via PyO3/maturin, `genoray_core::query` (Rust, pinned rev, already linked), `numpy`, `seqpro.rag.Ragged`, `should_parallelize` (`_threads.py`), pytest, the vcfixture bulk-cohort bench harness.

## Global Constraints

- **Target branch:** `streaming` (not `main`). Stacks on `278-svar2-streaming-backend` (PR #298's head, where PR 1a/1b landed). Streaming PRs merge into `streaming` per `CLAUDE.md`.
- **Correctness oracle (hard gate):** byte-identical parity vs `gvl.write()` + `Dataset.open()[r, s]` under `.with_seqs("haplotypes")`, jitter=0, **matched by returned `(r, s)` index** (order-independent). `tests/dataset/test_streaming_parity_svar2.py` (both tests) and `tests/dataset/test_streaming_scale.py::test_svar2_generate_batch_output_is_flat_in_cohort_size` must stay green. A faster variant that fails parity is a bug, not a feature.
- **No genoray rev bump.** `Cargo.toml` stays pinned at `rev = e07477e687c913f9605fc79ea251f1bb3b177aa9`.
- **Streaming-only.** The written-`Dataset` path (`_svar2_haps.py`) is untouched. Do **not** change any kernel default or any `reconstruct::*` signature.
- **`parallel` decision = `should_parallelize(total_out_bytes)`** (`_threads.py:122`) — the codebase-wide gate every other reconstruct path uses (`_haps.py`, `_reconstruct.py`, `_genotypes.py`, …). This *subsumes* PR 1a: a tiny tail super-batch stays serial, a core-saturating super-batch parallelizes. `total_out_bytes` is the reconstructed buffer's byte length (`offsets[-1]`).
- **Iteration order stays deterministic in PR 2.** Relaxed/completion-order emission is PR 3. The super-batch reconstruct is order-preserving on its own.
- **`max_mem` / #284 invariant:** the super-batch output buffer must be `max_mem`-bounded and **cohort-independent** (bytes identical across cohort sizes at fixed super-batch size). The super-batch is a chunk of a window's rows; it never scales with cohort.
- **Sample-index convention:** public `sample_idx` indexes the lexicographically-sorted sample-name order; `_phys_sample_idx` (already on `_Svar2Backend`, added in PR 1b) translates to physical store columns. Row-gather uses window arrays already in sorted order — no new translation.
- **Rebuild Rust before testing Rust changes:** after editing `src/`, run `pixi run -e dev maturin develop --release` **before** `pixi run -e dev pytest`, or pytest imports the stale extension (`CLAUDE.md`).
- **Commit hooks:** ensure `prek install` has run in this worktree before the first commit (`.pre-commit-config.yaml` present).
- **Testing commands:** `pixi run -e dev pytest <path> -v`. Before pushing: `pixi run -e dev pytest tests/dataset tests/unit -q`. Lint/format/types: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `src/svar2/mod.rs` | Add `pub(crate) fn svar2_readbound_chain(...)` — the shared gather→split→hap_diffs→offsets→reconstruct chain lifted verbatim from `reconstruct_haplotypes_from_svar2_readbound`'s `py.detach` body, writing into caller `&mut Vec<u8>`/`&mut Vec<i64>` (capacity reused). | 1 |
| `src/svar2/store.rs` | Add `Svar2ReconBuf` pyclass (owns `data`/`offsets`/`n_rows`/`ploidy`, reused across fills) with a `batch(lo, hi)` drain accessor. | 2 |
| `src/ffi/mod.rs` | (1) Refactor `reconstruct_haplotypes_from_svar2_readbound` to call `svar2_readbound_chain` (behavior-preserving; parity is the guard). (2) Add `pub fn svar2_reconstruct_super_batch(...)` that preps inputs like the existing fn, then fills a `&mut Svar2ReconBuf` via the chain with `parallel` supplied by Python. | 1, 2 |
| `src/lib.rs` | Register `Svar2ReconBuf` (pyclass) and `svar2_reconstruct_super_batch` (pyfunction), after the PR-1b `svar2_read_window` registration. | 2 |
| `python/genvarloader/_dataset/_streaming.py` | `_Svar2Backend`: add `_super_batch_rows` sizing in `__init__`; replace `generate_batch` with `reconstruct_super_batch(r_idx, s_idx, window, sb_lo, sb_hi, buf)` + a `drain_batch(buf, m, lo, hi)` helper. Restructure `_iter_batches` "sync" branch into window → super-batch → drain, holding one `Svar2ReconBuf` per iterator. | 3 |
| `tests/dataset/test_streaming_phase2_pr2.py` | Rust super-batch smoke (fill+drain shape/parity vs the per-batch FFI); core-utilization gate; generalized #284 super-batch flatness gate. | 2, 4 |
| `benchmarking/streaming/svar2_superbatch_sweep.py` | Super-batch-size sweep harness (core-util × wall) → knee. | 5 |
| `docs/roadmaps/streaming-dataset.md`, `docs/source/faq.md`/`dataset.md`, `skills/genvarloader/SKILL.md` | Record the measurement (knee, core-util win); document that streaming parallelism is internal Rust and `num_workers` is *not* the scaling path. | 5 |

**Parallelism (dispatch with `superpowers:dispatching-parallel-agents` + `subagent-driven-development`, Sonnet-or-weaker):**

- **Tasks 1 → 2 → 3 are a hard serial chain** (Python drive needs the compiled `Svar2ReconBuf` + fill fn; the fill fn needs the shared chain helper). No way around it — each builds on the prior's compiled symbol.
- **Tasks 4a (core-util gate) and 4b (#284 super-batch flatness gate) ARE parallelizable** once Task 3 lands: disjoint test functions in the same new test file, no shared state. Dispatch them concurrently.
- **Task 5 (sweep + docs)** depends on Task 3 but is independent of Task 4 — it can run concurrently with Tasks 4a/4b.

---

## Task 1: Shared read-bound reconstruct chain helper (Rust, behavior-preserving refactor)

**Files:**
- Modify: `src/svar2/mod.rs` (add `pub(crate) fn svar2_readbound_chain`)
- Modify: `src/ffi/mod.rs:1462-1549` (route `reconstruct_haplotypes_from_svar2_readbound`'s `py.detach` through the helper)

**Interfaces:**
- Consumes: `genoray_core::query::{HapRanges, gather_haps_readbound, ContigReader}`; `svar2::{split_to_flat, hap_diffs_svar2}`; `reconstruct::{bounds_from_offsets, reconstruct_haplotypes_from_svar2}`; `numpy::ndarray::{Array1, ArrayView1, ArrayView2}`.
- Produces (relied on by Tasks 1-refactor and Task 2):
  ```rust
  // src/svar2/mod.rs
  /// Fill `out_data`/`out_offsets` (cleared + refilled, capacity reused) with the
  /// read-bound reconstruction of `n_q = region_starts_v.len()` regions x `ploidy`
  /// haplotypes. Identical logic to the original `reconstruct_haplotypes_from_svar2_readbound`
  /// py.detach body; `parallel` is passed straight to the kernel. Runs GIL-free —
  /// callers wrap in `py.detach`. `out_offsets` has length `n_q*ploidy + 1`.
  pub(crate) fn svar2_readbound_chain(
      reader: &genoray_core::query::ContigReader,
      region_starts_v: &[u32],
      orig_samples_v: &[i64],
      vk_snp_range_v: &[std::ops::Range<usize>],
      vk_indel_range_v: &[std::ops::Range<usize>],
      dense_snp_range_v: &[std::ops::Range<usize>],
      dense_indel_range_v: &[std::ops::Range<usize>],
      regions: numpy::ndarray::ArrayView2<i32>,
      shifts_a: numpy::ndarray::ArrayView2<i32>,
      ref_a: numpy::ndarray::ArrayView1<u8>,
      ref_offsets_a: numpy::ndarray::ArrayView1<i64>,
      pad_char: u8,
      output_length: i64,
      parallel: bool,
      filter_exonic: bool,
      out_data: &mut Vec<u8>,
      out_offsets: &mut Vec<i64>,
  );
  ```

- [ ] **Step 1: Baseline — confirm SVAR2 parity is green before touching anything**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py -v`
Expected: PASS (2 tests). This is the invariant the refactor must not break.

- [ ] **Step 2: Add `svar2_readbound_chain` to `src/svar2/mod.rs`**

Lift the body of the `py.detach(move || { ... })` closure at `src/ffi/mod.rs:1462-1549` **verbatim** into this new function, with exactly two changes: (a) the trailing `(out_data, out_offsets_vec)` tuple return is removed; (b) `out_offsets_vec` and `out_data` are written into the caller's `&mut Vec` params instead of being locals. Concretely:

```rust
pub(crate) fn svar2_readbound_chain(
    reader: &genoray_core::query::ContigReader,
    region_starts_v: &[u32],
    orig_samples_v: &[i64],
    vk_snp_range_v: &[std::ops::Range<usize>],
    vk_indel_range_v: &[std::ops::Range<usize>],
    dense_snp_range_v: &[std::ops::Range<usize>],
    dense_indel_range_v: &[std::ops::Range<usize>],
    regions: numpy::ndarray::ArrayView2<i32>,
    shifts_a: numpy::ndarray::ArrayView2<i32>,
    ref_a: numpy::ndarray::ArrayView1<u8>,
    ref_offsets_a: numpy::ndarray::ArrayView1<i64>,
    pad_char: u8,
    output_length: i64,
    parallel: bool,
    filter_exonic: bool,
    out_data: &mut Vec<u8>,
    out_offsets: &mut Vec<i64>,
) {
    use numpy::ndarray::{ArrayView1, ArrayView2};
    let ploidy = shifts_a.ncols();
    let n_q = regions.nrows();

    let rb = genoray_core::query::HapRanges::new(
        region_starts_v, orig_samples_v,
        vk_snp_range_v, vk_indel_range_v, dense_snp_range_v, dense_indel_range_v, ploidy,
    );
    let br = genoray_core::query::gather_haps_readbound(reader, &rb);
    let (lut_bytes, lut_off_u64) = reader.lut_arrays();
    let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();
    let flat = crate::svar2::split_to_flat(&br);
    let dense_range_a = ArrayView2::from_shape((n_q, 2), &flat.dense_range).unwrap();

    let diffs = crate::svar2::hap_diffs_svar2(
        regions, ploidy,
        &flat.vk_pos, &flat.vk_key, &flat.vk_off,
        &flat.dense_pos, &flat.dense_key, dense_range_a,
        &flat.dense_present, &flat.dense_present_off,
        &lut_bytes, &lut_off, filter_exonic,
    );

    // Offsets (prefix sum) into the reused buffer.
    let n_work = n_q * ploidy;
    out_offsets.clear();
    out_offsets.resize(n_work + 1, 0);
    let mut acc: i64 = 0;
    out_offsets[0] = 0;
    for k in 0..n_work {
        let query = k / ploidy;
        let hap = k % ploidy;
        let len: i64 = if output_length >= 0 {
            output_length
        } else {
            let ref_len = (regions[[query, 2]] - regions[[query, 1]]) as i64;
            let diff = diffs[[query, hap]] as i64;
            (ref_len + diff).max(0)
        };
        acc += len;
        out_offsets[k + 1] = acc;
    }

    // Output buffer (reused capacity; fully overwritten by reconstruct).
    let total = out_offsets[n_work] as usize;
    out_data.clear();
    out_data.resize(total, 0u8);

    let out_offsets_view = ArrayView1::from(out_offsets.as_slice());
    let out_bounds = crate::reconstruct::bounds_from_offsets(out_offsets_view);
    let out_data_view = numpy::ndarray::ArrayViewMut1::from(out_data.as_mut_slice());
    crate::reconstruct::reconstruct_haplotypes_from_svar2(
        out_data_view,
        out_bounds.view(),
        regions,
        shifts_a,
        ArrayView1::from(flat.vk_pos.as_slice()),
        ArrayView1::from(flat.vk_key.as_slice()),
        ArrayView1::from(flat.vk_off.as_slice()),
        ArrayView1::from(flat.dense_pos.as_slice()),
        ArrayView1::from(flat.dense_key.as_slice()),
        dense_range_a,
        ArrayView1::from(flat.dense_present.as_slice()),
        ArrayView1::from(flat.dense_present_off.as_slice()),
        ArrayView1::from(lut_bytes.as_slice()),
        ArrayView1::from(lut_off.as_slice()),
        ref_a,
        ref_offsets_a,
        pad_char,
        parallel,
        filter_exonic,
    );
}
```

> Note: the original allocated with `uninit_output(total)`; the reused buffer uses `resize(total, 0)`. This is safe because `reconstruct_haplotypes_from_svar2` writes exactly `total` bytes (offsets are sized from `hap_diffs_svar2`) — every byte is overwritten. Parity (Step 5) is the proof.

- [ ] **Step 3: Route the existing FFI through the helper**

In `src/ffi/mod.rs`, replace the `py.detach(move || { ... })` block at `:1462-1549` (up to and including the closure) with:

```rust
    let (out_data, out_offsets_vec) = py.detach(move || {
        let mut out_data: Vec<u8> = Vec::new();
        let mut out_offsets: Vec<i64> = Vec::new();
        crate::svar2::svar2_readbound_chain(
            reader,
            &region_starts_v,
            &orig_samples_v,
            &vk_snp_range_v,
            &vk_indel_range_v,
            &dense_snp_range_v,
            &dense_indel_range_v,
            regions.view(),
            shifts_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            output_length,
            parallel,
            filter_exonic,
            &mut out_data,
            &mut out_offsets,
        );
        (Array1::from_vec(out_data), Array1::from_vec(out_offsets))
    });

    Ok((out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py)))
```

Keep the pre-detach prep (`:1413-1461`: `reader`, `regions`, `arr2_to_ranges`, `require_contiguous_1d`, `shifts_a`/`ref_a`/`ref_offsets_a` views, the `region_starts_v`/`orig_samples_v`/`*_range_v` bindings) exactly as-is. Delete the now-unused `uninit_output` import from this function's path only if the compiler flags it (it is used elsewhere — check before removing).

- [ ] **Step 4: Rebuild**

Run: `pixi run -e dev maturin develop --release`
Expected: builds cleanly. If `svar2_readbound_chain` is unresolved, confirm it's `pub(crate)` and that `src/svar2/mod.rs` is the module `crate::svar2` re-exports from (the existing `svar2::split_to_flat` call in `ffi/mod.rs` proves the path).

- [ ] **Step 5: Parity must still be byte-identical (the refactor guard)**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py tests/dataset/test_streaming_phase2_pr1.py -v`
Expected: PASS (all). Behavior-preserving refactor — same chain, same order. If parity breaks here, the lift changed logic (compare `svar2_readbound_chain` line-by-line against the original `:1463-1548`).

- [ ] **Step 6: Commit**

```bash
git add src/svar2/mod.rs src/ffi/mod.rs
git commit -m "refactor(streaming): extract svar2_readbound_chain shared reconstruct helper (#278)"
```

---

## Task 2: `Svar2ReconBuf` recycled buffer + `svar2_reconstruct_super_batch` FFI + drain

**Files:**
- Modify: `src/svar2/store.rs` (add `Svar2ReconBuf` pyclass)
- Modify: `src/ffi/mod.rs` (add `pub fn svar2_reconstruct_super_batch`)
- Modify: `src/lib.rs` (register both)
- Create: `tests/dataset/test_streaming_phase2_pr2.py` (fill+drain smoke + per-batch-FFI equivalence)

**Interfaces:**
- Consumes: `Svar2Store` (`store.reader(contig) -> Option<&ContigReader>`); `svar2::svar2_readbound_chain` (Task 1); the pre-detach prep pattern from `reconstruct_haplotypes_from_svar2_readbound` (`arr2_to_ranges`, `require_contiguous_1d`, view builders).
- Produces (relied on by Task 3):
  ```rust
  #[pyclass] pub struct Svar2ReconBuf { /* data, offsets, n_rows, ploidy */ }
  #[pymethods] impl Svar2ReconBuf {
      #[new] fn new(ploidy: usize) -> Self;                 // empty; grows on first fill
      #[getter] fn n_rows(&self) -> usize;                  // rows (queries) in current fill
      /// Copy out rows [lo, hi): returns (data u8 flat, offsets i64 len (hi-lo)*ploidy+1, rebased to 0).
      fn batch<'py>(&self, py: Python<'py>, lo: usize, hi: usize)
          -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>;
      #[getter] fn total_bytes(&self) -> usize;             // offsets[-1]; for the #284 gate
  }

  // src/ffi/mod.rs — fills `buf` in place; parallel supplied by Python (should_parallelize).
  #[pyfunction]
  pub fn svar2_reconstruct_super_batch(
      py, store: PyRef<Svar2Store>, contig: &str,
      region_starts: PyReadonlyArray1<u32>, orig_samples: PyReadonlyArray1<i64>,
      vk_snp_range: PyReadonlyArray2<i64>, vk_indel_range: PyReadonlyArray2<i64>,
      dense_snp_range: PyReadonlyArray2<i64>, dense_indel_range: PyReadonlyArray2<i64>,
      region_bounds: PyReadonlyArray2<i32>, shifts: PyReadonlyArray2<i32>,
      ref_: PyReadonlyArray1<u8>, ref_offsets: PyReadonlyArray1<i64>,
      pad_char: u8, parallel: bool,
      buf: PyRefMut<Svar2ReconBuf>,
  ) -> PyResult<()>;
  ```

- [ ] **Step 1: Write the failing smoke + equivalence test**

Create `tests/dataset/test_streaming_phase2_pr2.py`:

```python
"""PR 2: Svar2ReconBuf super-batch fill + drain — byte-identical to the per-batch
read-bound FFI, and self-consistent across drain boundaries."""

from __future__ import annotations

import numpy as np


def _plan_windows(sds):
    """(r_idx, s_idx) windows exactly as the sync drive sees them."""
    return list(sds._plan())


def test_super_batch_fill_drain_matches_per_batch_ffi(svar2_multicontig_fixture) -> None:
    """Reconstructing a whole super-batch then draining batch_size slices is
    byte-identical to reconstructing each batch_size slice on its own via the
    Phase-1/PR-1 per-batch FFI (parallel=False)."""
    import genvarloader as gvl
    from genvarloader.genvarloader import Svar2ReconBuf, svar2_reconstruct_super_batch

    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    backend = sds._backend
    assert backend is not None
    P = backend.ploidy

    for r_idx, s_idx in _plan_windows(sds):
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)

        # (a) super-batch: one fill over ALL window rows, drain per batch_size.
        buf = Svar2ReconBuf(P)
        backend._fill_super_batch(  # thin wrapper added in Task 3 helper; see below
            r_idx, s_idx, window, 0, n_rows, buf, parallel=False
        )
        assert buf.n_rows == n_rows
        drained = []
        for lo in range(0, n_rows, 4):
            hi = min(lo + 4, n_rows)
            rag = backend._drain(buf, lo, hi)
            drained.append(rag.data.view("S1").copy())
        got = np.concatenate(drained) if drained else np.empty(0, "S1")

        # (b) reference: per-batch FFI, parallel=False (the PR-1 path).
        ref_parts = []
        for lo in range(0, n_rows, 4):
            hi = min(lo + 4, n_rows)
            rag = backend._reconstruct_batch_reference(r_idx, s_idx, window, lo, hi)
            ref_parts.append(np.asarray(rag.data).view("S1").copy())
        ref = np.concatenate(ref_parts) if ref_parts else np.empty(0, "S1")

        np.testing.assert_array_equal(got, ref)
```

> `_fill_super_batch`, `_drain`, and `_reconstruct_batch_reference` are small `_Svar2Backend` helpers finalized in Task 3; for this task, add minimal versions (below) so the test drives the Rust surface. `_reconstruct_batch_reference` is the *old* `generate_batch` body kept as a parity reference (delete it at the end of Task 3 once the drive is switched).

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr2.py::test_super_batch_fill_drain_matches_per_batch_ffi -v`
Expected: FAIL — `ImportError: cannot import name 'Svar2ReconBuf'`.

- [ ] **Step 3: Add the `Svar2ReconBuf` pyclass to `src/svar2/store.rs`**

```rust
use numpy::{PyArray1, IntoPyArray};
use numpy::ndarray::Array1;
use pyo3::prelude::*;

#[pyclass]
pub struct Svar2ReconBuf {
    data: Vec<u8>,      // reconstructed bytes for the current super-batch (capacity reused)
    offsets: Vec<i64>,  // len n_rows*ploidy + 1
    n_rows: usize,      // queries (region,sample cells) in the current fill
    ploidy: usize,
}

impl Svar2ReconBuf {
    // Rust-only: the FFI fill fn writes results here (moves the chain's Vecs in).
    pub(crate) fn set(&mut self, data: Vec<u8>, offsets: Vec<i64>, n_rows: usize) {
        self.data = data;
        self.offsets = offsets;
        self.n_rows = n_rows;
    }
    pub(crate) fn ploidy(&self) -> usize { self.ploidy }
}

#[pymethods]
impl Svar2ReconBuf {
    #[new]
    fn new(ploidy: usize) -> Self {
        Svar2ReconBuf { data: Vec::new(), offsets: vec![0], n_rows: 0, ploidy }
    }

    #[getter]
    fn n_rows(&self) -> usize { self.n_rows }

    #[getter]
    fn total_bytes(&self) -> usize {
        *self.offsets.last().unwrap_or(&0) as usize
    }

    /// Copy out rows [lo, hi): flat data bytes + offsets (len (hi-lo)*ploidy+1) rebased to 0.
    fn batch<'py>(
        &self,
        py: Python<'py>,
        lo: usize,
        hi: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
        if hi < lo || hi > self.n_rows {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "batch [{lo},{hi}) out of range for n_rows={}",
                self.n_rows
            )));
        }
        let p = self.ploidy;
        let o_lo = lo * p; // first offset index for the row block
        let o_hi = hi * p; // last row's last hap offset index (inclusive end at o_hi)
        let byte_lo = self.offsets[o_lo] as usize;
        let byte_hi = self.offsets[o_hi] as usize;
        let data = self.data[byte_lo..byte_hi].to_vec();
        let base = self.offsets[o_lo];
        let offsets: Vec<i64> = self.offsets[o_lo..=o_hi].iter().map(|&x| x - base).collect();
        Ok((
            Array1::from_vec(data).into_pyarray(py),
            Array1::from_vec(offsets).into_pyarray(py),
        ))
    }
}
```

- [ ] **Step 4: Add `svar2_reconstruct_super_batch` to `src/ffi/mod.rs`**

Place adjacent to `reconstruct_haplotypes_from_svar2_readbound`. It reuses that function's pre-detach prep (copy the `arr2_to_ranges` conversions + view builders), then fills `buf` via the shared chain:

```rust
/// Fill a recycled `Svar2ReconBuf` with the read-bound reconstruction of a super-batch
/// (region,sample) rows. `parallel` is supplied by Python (`should_parallelize(total_out_bytes)`);
/// output is always ragged (`output_length = -1`). GIL released for the whole chain.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn svar2_reconstruct_super_batch<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar2::store::Svar2Store>,
    contig: &str,
    region_starts: PyReadonlyArray1<u32>,
    orig_samples: PyReadonlyArray1<i64>,
    vk_snp_range: PyReadonlyArray2<i64>,
    vk_indel_range: PyReadonlyArray2<i64>,
    dense_snp_range: PyReadonlyArray2<i64>,
    dense_indel_range: PyReadonlyArray2<i64>,
    region_bounds: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    parallel: bool,
    mut buf: PyRefMut<'py, crate::svar2::store::Svar2ReconBuf>,
) -> PyResult<()> {
    require_contiguous_1d(&ref_, "ref_")?;
    let reader = store
        .reader(contig)
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("no reader for contig {contig}")))?;
    let ploidy = shifts.as_array().ncols();
    let n_q = region_bounds.as_array().nrows();

    // regions (n_q,3): col0 unused, col1=start, col2=end — same layout as the read-bound FFI.
    let rbnd = region_bounds.as_array();
    let mut regions: Array2<i32> = Array2::zeros((n_q, 3));
    for i in 0..n_q {
        regions[[i, 1]] = rbnd[[i, 0]];
        regions[[i, 2]] = rbnd[[i, 1]];
    }
    let region_starts_v: Vec<u32> = region_starts.as_array().iter().copied().collect();
    let orig_samples_v: Vec<i64> = orig_samples.as_array().iter().copied().collect();
    let vk_snp_range_v = arr2_to_ranges(&vk_snp_range);
    let vk_indel_range_v = arr2_to_ranges(&vk_indel_range);
    let dense_snp_range_v = arr2_to_ranges(&dense_snp_range);
    let dense_indel_range_v = arr2_to_ranges(&dense_indel_range);
    let shifts_a = shifts.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();

    let (data, offsets) = py.detach(move || {
        let mut out_data: Vec<u8> = Vec::new();
        let mut out_offsets: Vec<i64> = Vec::new();
        crate::svar2::svar2_readbound_chain(
            reader,
            &region_starts_v, &orig_samples_v,
            &vk_snp_range_v, &vk_indel_range_v, &dense_snp_range_v, &dense_indel_range_v,
            regions.view(), shifts_a, ref_a, ref_offsets_a, pad_char,
            -1, parallel, false,
            &mut out_data, &mut out_offsets,
        );
        (out_data, out_offsets)
    });
    buf.set(data, offsets, n_q);
    Ok(())
}
```

> Match imports/helpers to the existing file: `arr2_to_ranges`, `require_contiguous_1d`, `Array2`, `PyReadonlyArray1/2` are already imported and used by `reconstruct_haplotypes_from_svar2_readbound`. `filter_exonic` is hard-`false` (splicing out of scope). This mirrors the existing regions build at `ffi/mod.rs:1413-1461`.

- [ ] **Step 5: Register both in `src/lib.rs`**

After the PR-1b `m.add_function(wrap_pyfunction!(ffi::svar2_read_window, m)?)?;` line, add:

```rust
    m.add_class::<crate::svar2::store::Svar2ReconBuf>()?;
    m.add_function(wrap_pyfunction!(ffi::svar2_reconstruct_super_batch, m)?)?;
```

(If `Svar2Store` is registered via `m.add_class::<...Svar2Store>()` nearby, place the `Svar2ReconBuf` line beside it for readability.)

- [ ] **Step 6: Add the minimal `_Svar2Backend` test helpers (finalized in Task 3)**

In `python/genvarloader/_dataset/_streaming.py`, add three methods to `_Svar2Backend` (Task 3 wires them into the drive; here they only need to exist for the test):

```python
    def _fill_super_batch(self, r_idx, s_idx, window, sb_lo, sb_hi, buf, parallel):
        """Reconstruct C-order rows [sb_lo, sb_hi) of the window into `buf`."""
        from ..genvarloader import svar2_reconstruct_super_batch

        (region_starts, orig_samples, vk_snp, vk_indel, dense_snp, dense_indel,
         region_bounds, shifts, ref_, ref_offsets) = self._gather_rows(
            r_idx, s_idx, window, sb_lo, sb_hi
        )
        svar2_reconstruct_super_batch(
            self._store, self._contigs[cast(int, window["contig_idx"])],
            region_starts, orig_samples, vk_snp, vk_indel, dense_snp, dense_indel,
            region_bounds, shifts, ref_, ref_offsets,
            np.uint8(self._ref.pad_char), bool(parallel), buf,
        )

    def _drain(self, buf, lo, hi) -> Ragged:
        data, offsets = buf.batch(int(lo), int(hi))
        m = hi - lo
        return Ragged.from_offsets(
            np.asarray(data).view("S1"), (m, self.ploidy, None),
            np.asarray(offsets, np.int64),
        )
```

`_gather_rows` and `_reconstruct_batch_reference` come from Task 3 Step 1 (extract the per-row gather out of the old `generate_batch`). To keep Task 2 self-contained, add `_gather_rows` now (it is a pure refactor of `generate_batch`'s lines 1068-1094 — see Task 3 Step 1 for the exact body) and a temporary `_reconstruct_batch_reference` = the *unchanged* old `generate_batch` renamed. Both are deleted/finalized in Task 3.

- [ ] **Step 7: Rebuild and run the smoke test to green**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr2.py::test_super_batch_fill_drain_matches_per_batch_ffi -v`
Expected: PASS — super-batch fill+drain is byte-identical to per-batch FFI across every window of the multi-contig fixture.

- [ ] **Step 8: Commit**

```bash
git add src/svar2/store.rs src/ffi/mod.rs src/lib.rs tests/dataset/test_streaming_phase2_pr2.py python/genvarloader/_dataset/_streaming.py
git commit -m "feat(streaming): Svar2ReconBuf recycled buffer + super-batch reconstruct FFI (#278)"
```

---

## Task 3: Wire the super-batch drive into `_iter_batches`; size `_super_batch_rows`

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_Svar2Backend.__init__` sizing; extract `_gather_rows`; replace `generate_batch` usage; `_iter_batches` "sync" branch)

**Interfaces:**
- Consumes: `Svar2ReconBuf` (Task 2), `_fill_super_batch`/`_drain` (Task 2), `should_parallelize` (`_threads.py`).
- Produces: `_Svar2Backend._super_batch_rows: int`; `_gather_rows(r_idx, s_idx, window, lo, hi) -> tuple[NDArray, ...]` (10-tuple of FFI-ready arrays); the super-batch "sync" drive in `_iter_batches`. `generate_batch` and `_reconstruct_batch_reference` are removed at the end.

- [ ] **Step 1: Extract `_gather_rows` from `generate_batch`**

Move the per-row gather (current `generate_batch` lines 1068-1094) into a standalone method so both the super-batch fill and the parity reference use one copy:

```python
    def _gather_rows(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        window: dict[str, object],
        lo: int,
        hi: int,
    ) -> tuple[
        NDArray[np.uint32], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64],
        NDArray[np.int64], NDArray[np.int64], NDArray[np.int32], NDArray[np.int32],
        NDArray[np.uint8], NDArray[np.int64],
    ]:
        """FFI-ready per-row inputs for C-order (region, sample) rows [lo, hi).
        Mirrors `_svar2_haps.py:_gather_inputs`; used by both the super-batch fill and
        the per-batch parity reference."""
        r_idx = np.asarray(r_idx, np.intp)
        n_s = len(np.asarray(s_idx))
        P = self.ploidy
        contig_idx = cast(int, window["contig_idx"])
        ref_, ref_offsets = self._ref._contig_slice(contig_idx)

        rows = np.arange(lo, hi)
        ri = rows // n_s
        si = rows % n_s
        region_bounds = np.ascontiguousarray(
            cast("NDArray[np.int32]", window["region_bounds"])[ri], np.int32
        )
        region_starts = np.ascontiguousarray(region_bounds[:, 0], np.uint32)
        orig_samples = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["orig_samples"])[si], np.int64
        )
        vk_snp = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["vk_snp"])[ri, si].reshape(-1, 2), np.int64
        )
        vk_indel = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["vk_indel"])[ri, si].reshape(-1, 2), np.int64
        )
        dense_snp = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["dense_snp"])[ri], np.int64
        )
        dense_indel = np.ascontiguousarray(
            cast("NDArray[np.int64]", window["dense_indel"])[ri], np.int64
        )
        m = hi - lo
        shifts = np.zeros((m, P), np.int32)  # jitter out of scope (jitter=0)
        return (
            region_starts, orig_samples, vk_snp, vk_indel, dense_snp, dense_indel,
            region_bounds, shifts, ref_, ref_offsets,
        )
```

Update `_fill_super_batch` (added in Task 2 Step 6) to call `self._gather_rows(...)` — it already does per the Task 2 code.

- [ ] **Step 2: Add `_super_batch_rows` sizing to `_Svar2Backend.__init__`**

After `self._regions = bed_to_regions(...)` (`:1005`), add:

```python
        # Reconstruct super-batch: the rayon dispatch grain. Sized to saturate cores
        # (n_work = rows*ploidy must be >> num_threads) while the output buffer stays
        # max_mem-bounded and cohort-independent (#284). SUPERBATCH_TARGET_ROWS is the
        # measured knee (benchmarking/streaming/svar2_superbatch_sweep.py; recorded in
        # docs/roadmaps/streaming-dataset.md); the max_mem cap keeps output bounded.
        SUPERBATCH_TARGET_ROWS = 4096  # confirmed by the Task-5 sweep
        widths = self._regions[:, 2] - self._regions[:, 1]
        mean_width = int(max(1, widths.mean())) if len(widths) else 1
        bytes_per_row = self.ploidy * mean_width  # ~ ragged hap length
        max_mem_out = getattr(self, "_max_mem_bytes", 512 * 1024 * 1024)
        self._super_batch_rows = max(
            1, min(SUPERBATCH_TARGET_ROWS, max_mem_out // max(1, bytes_per_row))
        )
```

> `_max_mem_bytes` is set on `StreamingDataset`, not the backend. If the backend cannot see it at construction, pass it through: read it in `_iter_batches` (which has `self._max_mem_bytes`) and set `backend._super_batch_rows` there, or thread `max_mem` into `_Svar2Backend.__init__`. Prefer threading it into `__init__` (constructor at `_streaming.py:192` — add the `max_mem_bytes` arg). Confirm the constructor call site and keep the fallback default.

- [ ] **Step 3: Replace the "sync" drive with the super-batch drive**

In `_iter_batches`, replace the `elif self._prefetch_strategy == "sync":` body (`:440-461`) with:

```python
            elif self._prefetch_strategy == "sync":
                # SVAR2 super-batch drive: read each window's ranges once, reconstruct a
                # coarse super-batch (the rayon dispatch grain -- should_parallelize gates
                # it), then drain user batch_size slices. Output stays (hi-lo)-bounded per
                # drained batch (#284); iteration order is deterministic (relaxed order is
                # PR 3). See docs/superpowers/specs/2026-07-18-streaming-svar2-phase2-design.md.
                from .._threads import should_parallelize
                from ..genvarloader import Svar2ReconBuf

                assert isinstance(self._backend, _Svar2Backend), (
                    '"sync" prefetch strategy requires the SVAR2 backend'
                )
                backend = self._backend
                buf = Svar2ReconBuf(backend.ploidy)  # one recycled buffer per iterator
                sb_rows = backend._super_batch_rows
                for r_idx, s_idx in self._plan():
                    window = backend.read_window(r_idx, s_idx)
                    n_s = len(s_idx)
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(s_idx, len(r_idx))
                    n_rows = len(flat_r)
                    for sb_lo in range(0, n_rows, sb_rows):
                        sb_hi = min(sb_lo + sb_rows, n_rows)
                        # Estimated output bytes gate parallel *before* the fill (the fill
                        # IS the reconstruct, so exact total_bytes is only known after):
                        # tiny tail stays serial (PR-1a), a core-saturating super-batch
                        # parallelizes (PR-2). Overestimate only flips the decision, never
                        # correctness.
                        backend._fill_super_batch(
                            r_idx, s_idx, window, sb_lo, sb_hi, buf,
                            parallel=should_parallelize(
                                backend._est_out_bytes(r_idx, sb_hi - sb_lo)
                            ),
                        )
                        for lo in range(sb_lo, sb_hi, batch_size):
                            hi = min(lo + batch_size, sb_hi)
                            data = backend._drain(buf, lo - sb_lo, hi - sb_lo)
                            yield data, flat_r[lo:hi], flat_s[lo:hi]
```

Add the `_est_out_bytes` helper to `_Svar2Backend` (a cheap pre-fill estimate so `parallel` is decided *before* reconstruction; the buffer's exact `total_bytes` is only known after):

```python
    def _est_out_bytes(self, r_idx: NDArray[np.intp], n_rows: int) -> int:
        """Estimate reconstructed super-batch bytes (~ rows*ploidy*mean_region_width)
        to gate should_parallelize before the fill. Overestimate is harmless (only
        flips the parallel decision)."""
        r_idx = np.asarray(r_idx, np.intp)
        widths = self._regions[r_idx, 2] - self._regions[r_idx, 1]
        mean_width = int(max(1, widths.mean())) if len(widths) else 1
        return int(n_rows) * self.ploidy * mean_width
```

- [ ] **Step 4: Delete the dead per-batch path**

Remove `_Svar2Backend.generate_batch` (its only caller was the old sync loop) and the temporary `_reconstruct_batch_reference` — **but keep `_reconstruct_batch_reference` until Task 4's tests are green if they reference it**; the Task 2 smoke test uses it. Sequence: land Task 4 tests first, then delete in a cleanup step, OR keep `_reconstruct_batch_reference` as a private test-only helper. Decision: **keep `_reconstruct_batch_reference` deleted from production and move its body into the test file** as a local `_per_batch_reference(backend, ...)` function, so production has one obvious path. Update `tests/dataset/test_streaming_phase2_pr2.py` accordingly.

- [ ] **Step 5: Parity — byte-identical through the super-batch drive**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py tests/dataset/test_streaming_phase2_pr2.py -v`
Expected: PASS. `test_streaming_svar2_matches_written_all_cells` and `..._covers_every_cell_once` prove the super-batch drive is byte-identical, order-independent (matched by `(r,s)`).

- [ ] **Step 6: The #284 per-batch bound still holds**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scale.py::test_svar2_generate_batch_output_is_flat_in_cohort_size -v`
Expected: PASS. (Drained batches are still `(hi-lo)`-bounded; Task 4b adds the *super-batch* flatness gate.)

- [ ] **Step 7: Lint/format/type gate**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_phase2_pr2.py
git commit -m "feat(streaming): SVAR2 super-batch reconstruct drive (parallel gated, drained per batch) (#278)"
```

---

## Task 4a: Core-utilization gate (parallelizable with 4b, 5)

**Files:**
- Modify: `tests/dataset/test_streaming_phase2_pr2.py` (add `test_super_batch_engages_multiple_cores`)

**Interfaces:**
- Consumes: the super-batch drive (Task 3); `GVL_FORCE_PARALLEL` (forces `should_parallelize` True on small inputs, `_threads.py:123`); `os.times()` / `time.process_time()` for CPU-seconds.

**Why:** the robust signal that the restructure engaged rayon is `cpu_secs/wall > 1` (multi-core), whereas the PR-1 single-core baseline sits at ~1×. Wall-clock is secondary color on this shared node (per the perf-gate convention); core-count is the deterministic-ish gate.

- [ ] **Step 1: Write the core-utilization test**

```python
def test_super_batch_engages_multiple_cores(svar2_scale_fixture, monkeypatch) -> None:
    """With a core-saturating super-batch, cpu_secs/wall rises materially above the
    single-core (~1x) PR-1 baseline. GVL_FORCE_PARALLEL removes the size gate so the
    modest-size fixture still dispatches rayon; the signal is threads engaged, not speed."""
    import os
    import time

    import genvarloader as gvl

    monkeypatch.setenv("GVL_FORCE_PARALLEL", "1")

    fx = svar2_scale_fixture  # 200 variants x >=50 samples, single contig (Task 4 fixture)
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")

    # Warm import/JIT of the rust path.
    for _ in sds.to_iter(batch_size=8):
        pass

    cpu0, wall0 = time.process_time(), time.perf_counter()
    n = 0
    for data, _, _ in sds.to_iter(batch_size=8):
        n += 1
    cpu1, wall1 = time.process_time(), time.perf_counter()

    wall = wall1 - wall0
    cpu = cpu1 - cpu0
    ratio = cpu / wall if wall > 0 else 0.0
    print(f"[svar2 core-util] cpu={cpu:.3f}s wall={wall:.3f}s ratio={ratio:.2f} batches={n}")
    # Multi-core engaged: comfortably above 1.0. Loose bound for shared-node noise.
    assert ratio > 1.3, (
        f"super-batch reconstruct did not engage multiple cores (cpu/wall={ratio:.2f}); "
        "expected >1.3 with GVL_FORCE_PARALLEL"
    )
```

> If `svar2_scale_fixture` does not yet exist, reuse the store builder from `test_streaming_scale.py::test_svar2_generate_batch_output_is_flat_in_cohort_size` (the `build(n_samples)` helper at `:316-320`) as a `pytest.fixture` returning `(bed, reference_path, svar2_path)` with ~200 records × 100 samples, single contig. Add it to `tests/dataset/conftest.py` or the test module.

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr2.py::test_super_batch_engages_multiple_cores -v -s`
Expected: PASS, with the printed ratio comfortably > 1.3. If it hovers near 1.0, the super-batch is too small to saturate — confirm `_super_batch_rows` covers the whole window on this fixture and `GVL_FORCE_PARALLEL` is set.

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_streaming_phase2_pr2.py tests/dataset/conftest.py
git commit -m "test(streaming): SVAR2 super-batch core-utilization gate (#278)"
```

---

## Task 4b: Generalized #284 super-batch flatness gate (parallelizable with 4a, 5)

**Files:**
- Modify: `tests/dataset/test_streaming_scale.py` (add `test_svar2_super_batch_buffer_is_flat_in_cohort_size`)

**Interfaces:**
- Consumes: `Svar2ReconBuf.total_bytes` (Task 2); the `build(n_samples)` store builder in `test_streaming_scale.py`; `_Svar2Backend._fill_super_batch`.

**Why:** #284 requires the reconstruct buffer to be `max_mem`-bounded and cohort-independent. The unit generalizes from "per-batch output" to "super-batch buffer": at a fixed `_super_batch_rows`, the filled buffer's `total_bytes` must be **identical** across cohort sizes (the super-batch is a chunk of window rows, never scales with cohort), while the read window still covers the whole cohort.

- [ ] **Step 1: Write the super-batch flatness test**

```python
def test_svar2_super_batch_buffer_is_flat_in_cohort_size(tmp_path):
    """The super-batch reconstruct buffer's byte count is IDENTICAL between a 50- and a
    400-sample cohort at a fixed super-batch size (cohort-independent, #284), while the
    read window covers the whole cohort. Generalizes the per-batch flatness gate."""
    import numpy as np

    import genvarloader as gvl
    from genvarloader.genvarloader import Svar2ReconBuf

    # `build` is the module-local store builder used by
    # test_svar2_generate_batch_output_is_flat_in_cohort_size (see that test, ~:316).
    def measure(n_samples: int) -> int:
        svar2, ref = build(n_samples)  # reuse the existing builder in this module
        bed = _scale_bed()             # reuse the existing scale bed helper in this module
        sds = gvl.StreamingDataset(bed, reference=ref, variants=svar2).with_seqs(
            "haplotypes"
        )
        backend = sds._backend
        assert backend is not None
        # Force a small, FIXED super-batch so the buffer size cannot track cohort.
        backend._super_batch_rows = 16
        r_idx, s_idx = next(iter(sds._plan()))
        # The window covers the whole cohort (sample axis not chunked at this scale).
        assert len(s_idx) == n_samples, (
            f"expected window to cover the whole cohort ({n_samples}), got {len(s_idx)}"
        )
        window = backend.read_window(r_idx, s_idx)
        n_rows = len(r_idx) * len(s_idx)
        buf = Svar2ReconBuf(backend.ploidy)
        backend._fill_super_batch(
            r_idx, s_idx, window, 0, min(16, n_rows), buf, parallel=False
        )
        return int(buf.total_bytes)

    bytes_50 = measure(50)
    bytes_400 = measure(400)
    print(f"[svar2 super-batch flatness] n=50 {bytes_50}B n=400 {bytes_400}B")
    assert bytes_50 > 0
    assert bytes_50 == bytes_400, (
        f"super-batch buffer scaled with cohort size (50->{bytes_50}B, 400->{bytes_400}B); "
        "the super-batch must be a fixed chunk of window rows, cohort-independent (#284)"
    )
```

> Adapt `build` / `_scale_bed` names to whatever the existing `test_svar2_generate_batch_output_is_flat_in_cohort_size` uses (read `:264-368` first). The two rows chosen (first 16 of the window, contiguous in C-order) are the **same regions** across cohorts because region_bounds are identical; only the cohort's sample count differs — so a fixed-row-count super-batch must produce identical bytes.

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scale.py::test_svar2_super_batch_buffer_is_flat_in_cohort_size -v -s`
Expected: PASS — identical byte counts at n=50 and n=400.

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_streaming_scale.py
git commit -m "test(streaming): SVAR2 super-batch buffer cohort-independence gate (#284) (#278)"
```

---

## Task 5: Super-batch-size sweep + roadmap/docs (parallelizable with 4a/4b)

**Files:**
- Create: `benchmarking/streaming/svar2_superbatch_sweep.py`
- Modify: `docs/roadmaps/streaming-dataset.md`, `docs/source/faq.md`, `docs/source/dataset.md`, `skills/genvarloader/SKILL.md`

**Interfaces:**
- Consumes: the super-batch drive (Task 3); the vcfixture bulk-cohort builder (`tests/benchmarks/data/build_svar2_stream_bulk.py`, referenced by the existing scale test); `VCFIXTURE_BIN` env.

**Why:** the spec requires the super-batch default to be a **measured knee**, not a guess (matching how `window_regions=64` and the PR-1 gate were justified). Record the sweep in the roadmap and confirm/adjust `SUPERBATCH_TARGET_ROWS` (Task 3 Step 2).

- [ ] **Step 1: Write the sweep harness**

```python
# benchmarking/streaming/svar2_superbatch_sweep.py
"""Sweep _super_batch_rows on a vcfixture cohort-scale SVAR2 store; report core-util
(cpu/wall) and best-of-N wall per setting to locate the knee. Perf is secondary color
on this shared node -- the reported knee is the point past which cpu/wall and wall both
plateau. Run: VCFIXTURE_BIN=... pixi run -e dev python benchmarking/streaming/svar2_superbatch_sweep.py
"""
from __future__ import annotations

import os
import time

import numpy as np

import genvarloader as gvl

SUPERBATCH_GRID = [256, 1024, 4096, 16384, 65536]
REPEATS = 3


def build_bulk_store(n_samples: int, n_records: int):
    # Reuse tests/benchmarks/data/build_svar2_stream_bulk.py's builder (import it).
    from tests.benchmarks.data.build_svar2_stream_bulk import build  # adjust to real API
    return build(n_samples=n_samples, n_records=n_records)


def bench(sds, sb_rows: int) -> tuple[float, float]:
    sds._backend._super_batch_rows = sb_rows
    best_wall = float("inf")
    best_ratio = 0.0
    for _ in range(REPEATS):
        cpu0, wall0 = time.process_time(), time.perf_counter()
        for _ in sds.to_iter(batch_size=32):
            pass
        cpu1, wall1 = time.process_time(), time.perf_counter()
        wall = wall1 - wall0
        if wall < best_wall:
            best_wall = wall
            best_ratio = (cpu1 - cpu0) / wall if wall > 0 else 0.0
    return best_wall, best_ratio


def main() -> None:
    bed, ref, svar2 = build_bulk_store(n_samples=2000, n_records=20000)
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar2).with_seqs("haplotypes")
    for _ in sds.to_iter(batch_size=32):  # warm
        break
    print(f"{'sb_rows':>10} {'best_wall_s':>12} {'cpu/wall':>10}")
    for sb in SUPERBATCH_GRID:
        wall, ratio = bench(sds, sb)
        print(f"{sb:>10} {wall:>12.3f} {ratio:>10.2f}")


if __name__ == "__main__":
    main()
```

> Adjust `build_bulk_store` to the actual `build_svar2_stream_bulk.py` API (read it first). If `VCFIXTURE_BIN` is unset the builder fails fast — set it (`/carter/users/dlaub/projects/vcfixture-rs/target/release/vcfixture`).

- [ ] **Step 2: Run the sweep (best-of-3), record numbers**

Run: `VCFIXTURE_BIN=/carter/users/dlaub/projects/vcfixture-rs/target/release/vcfixture pixi run -e dev python benchmarking/streaming/svar2_superbatch_sweep.py`
Expected: a table of `(sb_rows, best_wall, cpu/wall)`. Identify the knee — the smallest `sb_rows` where `cpu/wall` plateaus near its max and `best_wall` stops improving beyond node noise. If the knee differs from 4096, update `SUPERBATCH_TARGET_ROWS` in `_streaming.py` (Task 3 Step 2) to the measured value and re-run Task 4a to confirm the core-util gate still passes.

- [ ] **Step 3: Record the measurement in the roadmap**

In `docs/roadmaps/streaming-dataset.md`, under the SVAR2 Plan 2 row, add a Phase-2 PR-2 bullet: the super-batch knee (grid + best-of-3 cpu/wall + wall), the core-utilization win over PR-1's ~1× single-core baseline, and the PR pointer. Tick Phase-2 PR 2. Keep it brief (roadmap convention: summary + pointer, detail lives here).

- [ ] **Step 4: Document the iteration/parallelism contract**

In `docs/source/faq.md` and `docs/source/dataset.md`, and `skills/genvarloader/SKILL.md`, state: SVAR2 streaming reconstruction is parallelized **internally in Rust** (super-batch dispatch, `max_mem`-bounded); `num_workers > 0` is **not** the scaling path (RAM/IPC/idle cores — see the spec's "Why not num_workers"); batches carry `(r, s)` indices. (Iteration order stays deterministic in PR 2; the completion-order note lands with PR 3.)

- [ ] **Step 5: `__all__`/api.md sync check (no new public symbol expected)**

Run: `python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none`. (`Svar2ReconBuf`/`svar2_reconstruct_super_batch` are internal — not in `__all__`.)

- [ ] **Step 6: Commit**

```bash
git add benchmarking/streaming/svar2_superbatch_sweep.py docs/roadmaps/streaming-dataset.md docs/source/faq.md docs/source/dataset.md skills/genvarloader/SKILL.md python/genvarloader/_dataset/_streaming.py
git commit -m "perf(streaming): SVAR2 super-batch size sweep + measured knee; doc parallelism contract (#278)"
```

---

## Final verification (before opening the PR)

- [ ] **Full parity + scale + unit sweep** (shared code — cover both trees):

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: green, including all SVAR1 + SVAR2 parity/scale tests (no regression). The SVAR1 path is untouched but shares `_iter_batches`/`_streaming.py` — confirm it stays green.

- [ ] **Lint/format/type gate:**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Rebuild sanity (Rust changed):**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev pytest tests/dataset/test_streaming_phase2_pr2.py -q`
Expected: clean build, all PR-2 tests pass against the freshly-built extension.

- [ ] **Open the draft PR into `streaming`** (not `main`), stacked on `278-svar2-streaming-backend`:
  - `streaming: SVAR2 super-batch parallel reconstruction (PR 2)` — Tasks 1-5 (relates to #278). Add it to the StreamingDataset project board. Reference the spec and this plan.

---

## Out of scope (separate plan)

- **PR 3 (gated) — relaxed-order multi-window pipeline**: bounded-queue crossbeam producer/consumer over the super-batch reconstruction (the `Svar2ReconBuf` becomes the ping-pong slot), completion-order emission, cold-cache vs PR-2 measurement; ships as default only on a clear win, else off-by-default with the reason recorded (SVAR1's evidence-based ship/no-ship convention). Its own plan **after** PR 2's numbers land — the PR-2 super-batch is PR-3's measured baseline.
- Design reference: `docs/superpowers/specs/2026-07-18-streaming-svar2-phase2-design.md` (Lever 3).
