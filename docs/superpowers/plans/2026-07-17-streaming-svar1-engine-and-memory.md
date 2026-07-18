# StreamingDataset — SVAR1 engine wiring + cohort-independent memory — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** The plan is two PRs. **PR 1 (Tasks 1-6, issue #284) gates PR 2 (Tasks 7-10, issue #283).** Within PR 1, Task 1 (Rust FFI split) and Task 2 (Python `max_mem`/plan) touch disjoint files and can be dispatched concurrently via superpowers:dispatching-parallel-agents; Task 3 needs both. Within PR 2, Task 7 (Rust engine) gates Tasks 8-9.
>
> **Model policy:** use Sonnet (or weaker) for implementation subagents. Escalate to a stronger model only for a second-pass fix where the implementer critically failed. **Task 7 (the engine wiring) is the exception** — it introduces gvl's first *production* threading; consider a stronger model and extra review there.

**Goal:** Make `StreamingDataset` peak memory independent of cohort size (#284) by splitting the SVAR1 window read from per-batch generation under a `max_mem` byte budget, then wire the landed double-buffer engine so producer I/O overlaps consumer generation (#283) — shipping whichever of a producer thread (Design A) or single-thread read-through prefetch (Design C) a cold-cache measurement favors.

**Architecture:** The window (regions × sample-chunk × ploidy, cartesian, single-contig) is the **read** granularity — one offsets read per window; a **batch** is the **generation** granularity — `reconstruct_haplotypes_from_sparse` runs per `batch_size` slice, off the shared `variant_idxs` mmap, so output is `batch_size`-bounded. The offsets buffer (∝ window samples) is bounded by `max_mem`. PR 2 moves the read ahead of generation.

**Tech Stack:** Rust 2021 (PyO3 0.29, ndarray 0.17, crossbeam-channel 0.5, anyhow), Python 3.10+ (numpy, polars, seqpro, torch optional), pixi, maturin, pytest.

**Spec:** `docs/superpowers/specs/2026-07-17-streaming-svar1-engine-and-memory-design.md` (✅ approved)
**Issues:** [#284](https://github.com/mcvickerlab/GenVarLoader/issues/284) (PR 1), [#283](https://github.com/mcvickerlab/GenVarLoader/issues/283) (PR 2). Both target the long-lived `streaming` branch.

## Global Constraints

- **Byte-identical parity is the control and is non-negotiable.** `tests/dataset/test_streaming_parity.py` (`test_streaming_matches_written_all_cells`, `test_scale_parity_still_byte_identical`) must stay green. If a parity test needs more than a mechanical edit, behavior changed — STOP and investigate.
- **Rebuild Rust before pytest.** `pixi run -e dev pytest` does **not** rebuild the extension. After any `src/` change run `pixi run -e dev maturin develop --release` first, or pytest imports the **stale** binary. (`pixi run -e dev cargo-test` compiles from source and is unaffected.)
- **Commands:**
  - `pixi run -e dev maturin develop --release`
  - `pixi run -e dev pytest tests/dataset tests/unit -q`
  - `pixi run -e dev cargo-test`
  - `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev ruff format python/ tests/`
  - `pixi run -e dev typecheck`
  - **Full tree before pushing:** `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`).
- **Coordinates:** 0-based half-open `[start, end)` everywhere. `SparseVar.index.POS` is 1-based.
- **Never materialize a sample-scale array.** `variant_idxs` stays mmap'd; `geno_v_idxs()` MUST stay a borrow of the reader's mmap (pinned by `geno_v_idxs_borrows_the_mmap_not_a_copy`). `v_starts_c`/`v_ends_c` cross as zero-copy `PyReadonlyArray1`.
- **Share the mechanism, not the representation.** `type Buffer` is per-backend. Do **not** generalize a generation kernel across formats or touch VCF/PGEN — that is #276, out of scope, and its dense-vs-sparse pipeline is not prejudged here.
- **Scope:** SVAR1 only, `haplotypes` only, `jitter=0`, ragged output only. `num_workers>0` stays deferred (its guard stays). `conversion` stays enabled in `Cargo.toml`.
- **Commits are slow** — prek runs a full `pyrefly`. Use a ≥10-minute timeout. Ensure hooks are installed: `pixi run -e dev prek-install`.
- **Work in the worktree:** `/carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory` (branch `spec/streaming-svar1-engine-memory`, based on `origin/streaming`).

---

# PR 1 — cohort-independent peak memory (#284)

The one decisive change: **decouple read granularity (window) from generation granularity (batch).** `reconstruct_haplotypes_svar1` today materializes the whole window's haplotypes in one `out_data` alloc — the OOM. Split it into `svar1_read_window` (offsets, per window) + `svar1_generate_batch` (per batch), and sample-chunk the window under `max_mem`.

---

### Task 1: Split the FFI into `svar1_read_window` + `svar1_generate_batch`

**Files:**
- Modify: `src/ffi/mod.rs:862-1041` (replace `reconstruct_haplotypes_svar1` with the two functions)
- Modify: `src/lib.rs:51` (register both new functions in the `#[pymodule]`)

**Interfaces:**
- Consumes: `Svar1Store::read_window` (unchanged), `Svar1Store::geno_v_idxs`, `genotypes::get_diffs_sparse`, `reconstruct::reconstruct_haplotypes_from_sparse`, the `uninit_output`/`require_contiguous_1d` helpers already in `src/ffi/mod.rs`.
- Produces:
  ```rust
  // Read one window's offsets. Returns (o_starts, o_stops), each length
  // n_regions * n_samples * ploidy, C-order (region, sample, ploid), ABSOLUTE indices
  // into the store's variant_idxs mmap. Runs inside py.detach; store is PyRef<'py>.
  pub fn svar1_read_window<'py>(
      py: Python<'py>,
      store: PyRef<'py, crate::svar1::store::Svar1Store>,
      contig: &str,
      v_starts_c: PyReadonlyArray1<u32>,
      v_ends_c: PyReadonlyArray1<u32>,
      region_bounds: PyReadonlyArray2<i32>,   // (n_regions, 2)
      sample_idx: PyReadonlyArray1<i64>,      // (n_samples,)
  ) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)>;

  // Generate haplotypes for ONE batch of rows. `o_starts_b`/`o_stops_b` are the CSR-row
  // slices for this batch (length n_rows * ploidy, absolute indices into variant_idxs);
  // `region_bounds_b` is (n_rows, 2), already expanded per (region, sample). Output is
  // batch-sized — NEVER whole-window. geno_v_idxs is the shared mmap (zero copy).
  pub fn svar1_generate_batch<'py>(
      py: Python<'py>,
      store: PyRef<'py, crate::svar1::store::Svar1Store>,
      o_starts_b: PyReadonlyArray1<i64>,
      o_stops_b: PyReadonlyArray1<i64>,
      region_bounds_b: PyReadonlyArray2<i32>,  // (n_rows, 2)
      v_starts: PyReadonlyArray1<i32>,
      ilens: PyReadonlyArray1<i32>,
      alt_alleles: PyReadonlyArray1<u8>,
      alt_offsets: PyReadonlyArray1<i64>,
      ref_: PyReadonlyArray1<u8>,
      ref_offsets: PyReadonlyArray1<i64>,
      pad_char: u8,
      parallel: bool,
  ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>;
  ```

**Background:** `read_window` already returns `Svar1Window { o_starts, o_stops, geno_offset_idx }` — `geno_offset_idx` is the pure identity map and is NOT needed downstream (the kernel's per-batch `geno_offset_idx` is a *local* identity over the batch's rows). `svar1_read_window` therefore returns only `o_starts`/`o_stops`. `svar1_generate_batch` is the second half of the old `py.detach` body (lines 924-1034), re-parameterized to run over a batch's rows rather than the whole window.

- [ ] **Step 1: Write the failing Rust test**

Append to `src/svar1/store.rs`'s `#[cfg(test)] mod tests` (these exercise the offsets contract the FFI relies on; the FFI functions themselves are covered by the Python parity + scale tests in Tasks 3-4):

```rust
    #[test]
    fn read_window_offsets_are_absolute_and_row_major() {
        // Same 4-hap fixture as read_window_is_cartesian_and_borrows_the_mmap.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 0, 1, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 3, 4]);
        let mut store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        store.set_contig_meta_rs("chr1", 0, 2, 1);
        let w = store
            .read_window("chr1", &[10, 20], &[11, 21], &[(0, 30)], &[0, 1])
            .unwrap();
        // A batch [lo=1, hi=2) (row 1 only) selects CSR rows [1*2 .. 2*2) = [2, 4):
        // o_starts[2..4] = [1, 3], o_stops[2..4] = [3, 4]  -> hap2, hap3.
        assert_eq!(&w.o_starts[2..4], &[1, 3]);
        assert_eq!(&w.o_stops[2..4], &[3, 4]);
    }
```

- [ ] **Step 2: Run to verify it passes (read_window already satisfies this)**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev cargo-test 2>&1 | grep -E "read_window_offsets|test result" | head`
Expected: `read_window_offsets_are_absolute_and_row_major ... ok`. (This pins the slicing contract Task 3's Python relies on before we split the FFI.)

- [ ] **Step 3: Replace `reconstruct_haplotypes_svar1` with `svar1_read_window`**

In `src/ffi/mod.rs`, replace the `reconstruct_haplotypes_svar1` function (lines 862-1041) with two functions. First, `svar1_read_window` — it is the current function's pre-detach setup + `read_window` call, returning offsets:

```rust
/// Read ONE cartesian window's sparse-genotype offsets from a live `.svar` store via
/// genoray's ungated `svar1_query` (two binary-search stages, no record walk). Returns
/// `(o_starts, o_stops)`, each `n_regions * n_samples * ploidy` long in C-order
/// `(region, sample, ploid)` — ABSOLUTE indices into the store's `variant_idxs` mmap.
/// Generation is a SEPARATE call (`svar1_generate_batch`) so output is batch-bounded,
/// never whole-window (issue #284). Runs inside `py.detach`; `store` is `PyRef<'py>`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn svar1_read_window<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar1::store::Svar1Store>,
    contig: &str,
    v_starts_c: PyReadonlyArray1<u32>,
    v_ends_c: PyReadonlyArray1<u32>,
    region_bounds: PyReadonlyArray2<i32>,
    sample_idx: PyReadonlyArray1<i64>,
) -> PyResult<(Bound<'py, PyArray1<i64>>, Bound<'py, PyArray1<i64>>)> {
    require_contiguous_1d(&v_starts_c, "v_starts_c")?;
    require_contiguous_1d(&v_ends_c, "v_ends_c")?;

    let rb = region_bounds.as_array();
    let n_regions = rb.nrows();
    let regions_v: Vec<(u32, u32)> = (0..n_regions)
        .map(|i| (rb[[i, 0]].max(0) as u32, rb[[i, 1]].max(0) as u32))
        .collect();
    let samples_v: Vec<usize> = sample_idx.as_array().iter().map(|&s| s as usize).collect();

    let v_starts_c_a = v_starts_c.as_array();
    let v_ends_c_a = v_ends_c.as_array();
    let v_starts_c_s: &[u32] = v_starts_c_a
        .as_slice()
        .expect("contiguity checked by require_contiguous_1d above");
    let v_ends_c_s: &[u32] = v_ends_c_a
        .as_slice()
        .expect("contiguity checked by require_contiguous_1d above");

    let store_ref: &crate::svar1::store::Svar1Store = &store;

    let result = py.detach(move || -> anyhow::Result<(Array1<i64>, Array1<i64>)> {
        let w = store_ref.read_window(contig, v_starts_c_s, v_ends_c_s, &regions_v, &samples_v)?;
        Ok((Array1::from_vec(w.o_starts), Array1::from_vec(w.o_stops)))
    });

    let (o_starts, o_stops) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((o_starts.into_pyarray(py), o_stops.into_pyarray(py)))
}
```

- [ ] **Step 4: Add `svar1_generate_batch`**

Immediately after `svar1_read_window`, add the generation half. This is the old detach body's lines 924-1034 with `batch` = `region_bounds_b.nrows()`, `regions_arr` built from `region_bounds_b` (already per-row, so a straight copy), and `o_starts`/`o_stops` taken from the passed arrays:

```rust
/// Generate haplotypes for ONE batch of window rows. `o_starts_b`/`o_stops_b` are the
/// CSR-row offsets for exactly this batch (length `n_rows * ploidy`, ABSOLUTE indices
/// into `variant_idxs`); `region_bounds_b` is `(n_rows, 2)`, already expanded per
/// (region, sample). Output is `n_rows`-bounded — the #284 fix. `geno_v_idxs` is the
/// shared `variant_idxs` mmap (zero copy). Ragged output only, jitter=0.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn svar1_generate_batch<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar1::store::Svar1Store>,
    o_starts_b: PyReadonlyArray1<i64>,
    o_stops_b: PyReadonlyArray1<i64>,
    region_bounds_b: PyReadonlyArray2<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
    use crate::genotypes;
    use crate::reconstruct;

    require_contiguous_1d(&ref_, "ref_")?;
    require_contiguous_1d(&o_starts_b, "o_starts_b")?;
    require_contiguous_1d(&o_stops_b, "o_stops_b")?;

    let rb = region_bounds_b.as_array();
    let batch = rb.nrows();
    let ploidy = store.ploidy();
    let n_work = batch * ploidy;

    // Per-row region bounds (already (region, sample)-expanded on the Python side).
    let mut regions_arr = Array2::<i32>::zeros((batch, 3));
    for bi in 0..batch {
        regions_arr[[bi, 1]] = rb[[bi, 0]];
        regions_arr[[bi, 2]] = rb[[bi, 1]];
    }
    let shifts_arr = Array2::<i32>::zeros((batch, ploidy)); // jitter=0

    // Local identity map over THIS batch's rows: batch row bi, hap p -> local CSR row
    // bi*ploidy + p, indexing o_starts_b/o_stops_b (which are already the batch slice).
    let mut geno_offset_idx = Array2::<i64>::zeros((batch, ploidy));
    for bi in 0..batch {
        for p in 0..ploidy {
            geno_offset_idx[[bi, p]] = (bi * ploidy + p) as i64;
        }
    }

    let o_starts_a = o_starts_b.as_array();
    let o_stops_a = o_stops_b.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();

    let store_ref: &crate::svar1::store::Svar1Store = &store;

    let result = py.detach(move || -> anyhow::Result<(Array1<u8>, Array1<i64>)> {
        // ZERO COPY: kernel sparse input IS the store's mmap (see geno_v_idxs contract).
        let geno_v_idxs = store_ref.geno_v_idxs();
        let geno_v_idxs_view = numpy::ndarray::ArrayView1::from(geno_v_idxs);

        let q_starts_owned: Array1<i32> = regions_arr.column(1).to_owned();
        let q_ends_owned: Array1<i32> = regions_arr.column(2).to_owned();
        let diffs = genotypes::get_diffs_sparse(
            geno_offset_idx.view(),
            geno_v_idxs_view,
            o_starts_a,
            o_stops_a,
            ilens_a,
            None,
            None,
            Some(q_starts_owned.view()),
            Some(q_ends_owned.view()),
            Some(v_starts_a),
            parallel,
        );

        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let ref_len = (regions_arr[[query, 2]] - regions_arr[[query, 1]]) as i64;
                let diff = diffs[[query, hap]] as i64;
                acc += (ref_len + diff).max(0);
                out_offsets_vec[k + 1] = acc;
            }
        }

        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<u8> = uninit_output(total);

        reconstruct::reconstruct_haplotypes_from_sparse(
            out_data.view_mut(),
            out_offsets_vec.view(),
            regions_arr.view(),
            shifts_arr.view(),
            geno_offset_idx.view(),
            o_starts_a,
            o_stops_a,
            geno_v_idxs_view,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            None, // keep
            None, // keep_offsets
            None, // annot_v_idxs
            None, // annot_ref_pos
            parallel,
        );

        Ok((out_data, out_offsets_vec))
    });

    let (out_data, out_offsets_vec) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py)))
}
```

- [ ] **Step 5: Register both functions**

In `src/lib.rs`, replace the line `m.add_function(wrap_pyfunction!(ffi::reconstruct_haplotypes_svar1, m)?)?;` (line 51) with:

```rust
    m.add_function(wrap_pyfunction!(ffi::svar1_read_window, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::svar1_generate_batch, m)?)?;
```

- [ ] **Step 6: Build and run the Rust tests**

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev cargo-test 2>&1 | tail -6
```
Expected: all cargo tests pass. (Python parity is red until Task 3 rewires `_Svar1Backend` — that is expected; do not touch Python here.)

- [ ] **Step 7: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add src/ffi/mod.rs src/lib.rs src/svar1/store.rs
git commit -m "refactor(streaming): split svar1 window read from per-batch generation

svar1_read_window returns offsets only; svar1_generate_batch reconstructs one
batch off the shared variant_idxs mmap, so output is batch-bounded not
whole-window (#284). Replaces reconstruct_haplotypes_svar1."
```

---

### Task 2: `max_mem` budget + sample-chunked window plan

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`StreamingDataset` fields, `__init__`, `_plan`, `_iter_batch_spans`)
- Test: `tests/dataset/test_streaming_scheduler.py` (add plan-shape tests)

**Interfaces:**
- Consumes: nothing new (pure Python).
- Produces:
  ```python
  StreamingDataset(..., max_mem: str | int = "512MB")   # new keyword
  StreamingDataset._window_regions: int                  # derived (was a fixed field)
  StreamingDataset._window_samples: int                  # derived (NEW)
  StreamingDataset._plan() -> Iterator[tuple[NDArray[intp], NDArray[intp]]]
      # now yields (region_idxs, sample_chunk) — sample_chunk is a SUBSET, not all samples
  ```

**Background:** Today `_plan` yields `(region_idxs, all_samples)` and `_window_regions=64` is a fixed field. The offsets buffer is `window_regions × window_samples × ploidy × 16 B` (o_starts + o_stops, i64 each); at 100k samples this alone is hundreds of MB, so the sample axis must be chunkable and bounded by a byte budget. The per-batch output (Task 3) is separately `batch_size`-bounded and is NOT part of this budget.

Sizing rule (offsets only; `n_slots=1` for PR 1, becomes 2 in PR 2 Task 7):

```
cell_bytes = ploidy * 16                       # o_start + o_stop per (region,sample,ploid)
max_cells  = max(1, max_mem_bytes // (cell_bytes * n_slots))
window_samples = max(1, min(n_samples, max_cells))
REGION_TARGET  = 64                            # measured read-amortization knee (Task 6 confirms)
window_regions = max(1, min(REGION_TARGET, max_cells // window_samples))
```

Keeps whole sample sets when they fit, shrinks samples only under a tight budget, and holds regions at the amortization knee. Parse `max_mem` with the existing helper if gvl has one; otherwise a small local parser (accept `int` bytes or `str` like `"512MB"`, `"1g"`).

- [ ] **Step 1: Write the failing test**

Add to `tests/dataset/test_streaming_scheduler.py`:

```python
import numpy as np
import polars as pl

import genvarloader as gvl


def _injected_sds(n_regions, n_samples, ploidy=2, max_mem="512MB", contigs=("chr1",)):
    """Build a StreamingDataset via the injected-callback path (no store) with all
    regions on one contig, so we can inspect the window plan in isolation."""
    bed = pl.DataFrame(
        {
            "chrom": [contigs[0]] * n_regions,
            "chromStart": list(range(0, 100 * n_regions, 100)),
            "chromEnd": list(range(100, 100 * n_regions + 100, 100)),
        }
    )
    return gvl.StreamingDataset(
        bed,
        contigs=list(contigs),
        n_samples=n_samples,
        ploidy=ploidy,
        max_mem=max_mem,
        _reconstruct_window=lambda r, s: None,
    )


def test_plan_chunks_samples_under_a_tight_budget():
    # ploidy=2 -> cell_bytes=32. max_mem tiny (2 KB) -> max_cells small, so a
    # 1000-sample cohort must be split into multiple sample chunks per region window.
    sds = _injected_sds(n_regions=4, n_samples=1000, max_mem=2048)
    windows = list(sds._plan())
    sample_chunk_sizes = {len(s) for _, s in windows}
    assert max(sample_chunk_sizes) < 1000, "samples must be chunked under a tight budget"
    # Every (region, sample) cell appears exactly once across the plan.
    seen = set()
    for r_idx, s_idx in windows:
        for r in r_idx.tolist():
            for s in s_idx.tolist():
                assert (r, s) not in seen
                seen.add((r, s))
    assert len(seen) == 4 * 1000


def test_plan_keeps_all_samples_when_they_fit():
    # Generous budget -> a small cohort stays as one sample chunk per window.
    sds = _injected_sds(n_regions=4, n_samples=8, max_mem="512MB")
    for _, s_idx in sds._plan():
        assert len(s_idx) == 8, "all samples should fit in one chunk under a big budget"


def test_plan_is_single_contig_per_window():
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "chromStart": [0, 100, 0],
            "chromEnd": [100, 200, 100],
        }
    )
    sds = gvl.StreamingDataset(
        bed, contigs=["chr1", "chr2"], n_samples=4, ploidy=2,
        _reconstruct_window=lambda r, s: None,
    )
    for r_idx, _ in sds._plan():
        contigs = sds._regions[r_idx, 0]
        assert len(set(contigs.tolist())) == 1
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev pytest tests/dataset/test_streaming_scheduler.py -q 2>&1 | tail -12`
Expected: FAIL — `StreamingDataset() got an unexpected keyword argument 'max_mem'`.

- [ ] **Step 3: Add `max_mem` + derived window fields**

In `_streaming.py`, replace the `_window_regions: int = 64` field (lines 58-67) with:

```python
    # Read-window sizing, DERIVED from `max_mem` in __init__ (not user-set directly).
    # The window (regions x sample-chunk x ploidy) is the READ granularity; its offsets
    # buffer is what `max_mem` bounds. Per-batch generation (Task 3) bounds OUTPUT
    # separately by batch_size, so neither term scales with cohort size.
    _window_regions: int = 64
    _window_samples: int = 1
    _max_mem_bytes: int = 512 * 1024 * 1024
```

Add a `max_mem` parameter to `__init__` (after `jitter`):

```python
        max_mem: str | int = "512MB",
```

Add a module-level parser near the top of the file (after imports):

```python
def _parse_max_mem(max_mem: str | int) -> int:
    """Bytes from an int or a size string like '512MB' / '1g' / '2GiB'."""
    if isinstance(max_mem, int):
        return int(max_mem)
    s = str(max_mem).strip().lower().replace("ib", "b")
    units = {"b": 1, "kb": 1024, "mb": 1024**2, "gb": 1024**3, "tb": 1024**4}
    for suffix in ("tb", "gb", "mb", "kb", "b"):
        if s.endswith(suffix):
            return int(float(s[: -len(suffix)]) * units[suffix])
    return int(float(s))  # bare number = bytes
```

At the end of `__init__` (replacing the `_window_regions` `object.__setattr__` block at lines 169-173), derive all three:

```python
        max_mem_bytes = _parse_max_mem(max_mem)
        n_slots = 1  # PR 1 is single-window-resident; PR 2 (engine) sets 2 (ping-pong).
        cell_bytes = int(ploidy) * 16  # o_start + o_stop, i64 each, per (region,sample,ploid)
        max_cells = max(1, max_mem_bytes // (cell_bytes * n_slots))
        window_samples = max(1, min(int(n_samples), max_cells))
        region_target = 64  # measured read-amortization knee; see roadmap Plan 2.
        window_regions = max(1, min(region_target, max_cells // window_samples))
        object.__setattr__(self, "_max_mem_bytes", max_mem_bytes)
        object.__setattr__(self, "_window_samples", int(window_samples))
        object.__setattr__(self, "_window_regions", int(window_regions))
```

- [ ] **Step 4: Chunk the sample axis in `_plan` and `_iter_batch_spans`**

Replace `_plan` (lines 194-215) with:

```python
    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield one WINDOW per step: (region_idxs, sample_chunk), cartesian,
        single-contig. Both the region axis (`_window_regions`) and the sample axis
        (`_window_samples`) are chunked so the offsets buffer stays within `max_mem`
        regardless of cohort size. NOT pairwise: the traversal is a fixed cartesian
        sweep and the window is the read granularity.
        """
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        for r_lo, r_hi in zip(run_starts, run_ends):
            for w_lo in range(int(r_lo), int(r_hi), self._window_regions):
                w_hi = min(w_lo + self._window_regions, int(r_hi))
                r_idx = np.arange(w_lo, w_hi, dtype=np.intp)
                for s_lo in range(0, n_samples, self._window_samples):
                    s_hi = min(s_lo + self._window_samples, n_samples)
                    yield r_idx, np.arange(s_lo, s_hi, dtype=np.intp)
```

`_iter_batch_spans` (lines 272-277) already uses `len(r_idx) * len(s_idx)` and needs **no change** — it now naturally counts sample-chunked windows.

- [ ] **Step 5: Run the tests**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev pytest tests/dataset/test_streaming_scheduler.py -q 2>&1 | tail -12`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_scheduler.py
git commit -m "feat(streaming): max_mem byte budget + sample-chunked window plan (#284)

Window plan now chunks the sample axis so the offsets buffer is bounded by
max_mem independent of cohort size. _window_regions/_window_samples are derived,
not fixed."
```

---

### Task 3: Per-batch generation in `_Svar1Backend` + `_iter_batches` — parity green

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_Svar1Backend.reconstruct_window` → `read_window` + `generate_batch`; `StreamingDataset._iter_batches`; wire `_backend`)
- Test: `tests/dataset/test_streaming_parity.py` (must stay green), `tests/dataset/test_svar1_window.py` (mechanical update)

**Interfaces:**
- Consumes: `svar1_read_window`, `svar1_generate_batch` (Task 1).
- Produces:
  ```python
  class _Svar1Backend:
      ploidy: int
      def read_window(self, r_idx, s_idx) -> tuple[NDArray[int64], NDArray[int64]]  # (o_starts, o_stops)
      def generate_batch(self, r_idx, s_idx, o_starts, o_stops, lo, hi) -> Ragged   # rows [lo:hi]
  # StreamingDataset gains an optional `_backend` field; _iter_batches uses it when set,
  # else falls back to the injected whole-window `_reconstruct_window` (test seam).
  ```

**Background:** The real SVAR1 path now reads a window's offsets once, then generates each `batch_size` slice separately (output batch-bounded). The injected `_reconstruct_window` callback stays as a **test-only seam** (whole-window; used by scheduler/window tests where memory is irrelevant), so `_iter_batches` branches on which is present. `_phys_sample_idx` translation and the ref/contig derivation (`_streaming.py:526-558`) move unchanged into the two new methods.

- [ ] **Step 1: Write the failing test**

Add to `tests/dataset/test_streaming_parity.py`:

```python
def test_backend_read_then_generate_matches_whole_window(svar1_multicontig_fixture):
    """read_window + per-batch generate must reproduce the same bytes a single
    whole-window reconstruction would, for every batch slice."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(f.gvl_path, reference=f.reference_path).with_seqs("haplotypes")

    for data, r_idx, s_idx in sds.to_iter(batch_size=3):
        for i in range(len(r_idx)):
            expected = written[int(r_idx[i]), int(s_idx[i])]
            for h in range(sds.ploidy):
                np.testing.assert_array_equal(np.asarray(data[i][h]), np.asarray(expected[h]))
```

(If `svar1_multicontig_fixture` does not already expose `gvl_path`, reuse whatever the existing `test_streaming_matches_written_all_cells` uses to build the written dataset — mirror that fixture exactly.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev maturin develop --release 2>&1 | tail -2 && pixi run -e dev pytest tests/dataset/test_streaming_parity.py -q 2>&1 | tail -15`
Expected: FAIL — `_iter_batches` still calls `self._reconstruct_window`, which no longer exists on the backend (renamed), or `ImportError: cannot import name 'reconstruct_haplotypes_svar1'`.

- [ ] **Step 3: Split `_Svar1Backend.reconstruct_window` into `read_window` + `generate_batch`**

Replace `reconstruct_window` (lines 517-579) with the two methods. `read_window` is the current pre-FFI setup returning offsets; `generate_batch` slices and calls the generation FFI:

```python
    def read_window(
        self, r_idx: NDArray[np.intp], s_idx: NDArray[np.intp]
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Read one window's CSR offsets: every region in `r_idx` x every sample in
        `s_idx`, single-contig. Returns (o_starts, o_stops), each
        `len(r_idx) * len(s_idx) * ploidy`, C-order (region, sample, ploid) -- absolute
        indices into the store's variant_idxs mmap. No haplotypes are generated here.
        """
        from ..genvarloader import svar1_read_window

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_Svar1Backend.read_window: window spans multiple contigs; "
                "every Rust call must be single-contig."
            )
        contig_name = self._contigs[contig_idx]
        vs_c, ve_c = self._contig_arrays[contig_name]
        region_bounds = np.ascontiguousarray(self._regions[r_idx, 1:3], np.int32)
        phys_s_idx = self._phys_sample_idx[s_idx]

        o_starts, o_stops = svar1_read_window(
            self._store,
            contig_name,
            vs_c,
            ve_c,
            region_bounds,
            np.ascontiguousarray(phys_s_idx, np.int64),
        )
        return np.asarray(o_starts, np.int64), np.asarray(o_stops, np.int64)

    def generate_batch(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        o_starts: NDArray[np.int64],
        o_stops: NDArray[np.int64],
        lo: int,
        hi: int,
    ) -> Ragged:
        """Generate haplotypes for window rows [lo:hi] (C-order (region, sample)).
        Output is (hi-lo)-bounded -- NEVER the whole window (issue #284). `o_starts`/
        `o_stops` are the whole window's offsets (from `read_window`); this slices the
        CSR rows [lo*ploidy : hi*ploidy] and the matching per-row region bounds.
        """
        from ..genvarloader import svar1_generate_batch

        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)
        n_s = len(s_idx)
        contig_idx = int(self._regions[r_idx[0], 0])
        ref_bytes, ref_offsets = self._ref._contig_slice(contig_idx)

        # Per (region, sample) row bounds for rows [lo:hi], C-order (region, sample):
        # window row bi = ri*n_s + si -> region r_idx[bi // n_s].
        rows = np.arange(lo, hi)
        region_bounds_b = np.ascontiguousarray(
            self._regions[r_idx[rows // n_s], 1:3], np.int32
        )
        o_lo, o_hi = lo * self.ploidy, hi * self.ploidy

        data, offsets = svar1_generate_batch(
            self._store,
            np.ascontiguousarray(o_starts[o_lo:o_hi], np.int64),
            np.ascontiguousarray(o_stops[o_lo:o_hi], np.int64),
            region_bounds_b,
            self._v_starts,
            self._ilens,
            self._alt_alleles,
            self._alt_offsets,
            ref_bytes,
            ref_offsets,
            self._ref.pad_char,
            True,
        )
        n_rows = hi - lo
        return Ragged.from_offsets(
            data.view("S1"), (n_rows, self.ploidy, None), np.asarray(offsets, np.int64)
        )
```

- [ ] **Step 4: Wire `_backend` and per-batch `_iter_batches`**

In `StreamingDataset`: add an optional `_backend` field **among the trailing defaulted
fields** (right after `_max_mem_bytes` from Task 2) — with `slots=True` + a user-defined
`__init__`, defaulted fields must follow the non-default ones (same reason
`_window_regions` already sits at the end):

```python
    # The split read/generate backend (real SVAR1 path). When set, _iter_batches
    # generates per batch (output bounded by batch_size). The injected
    # `_reconstruct_window` remains a whole-window TEST seam used when `_backend` is None.
    _backend: object = None
```

In `__init__`, initialize `_backend_obj = None` at the very top (before any branch) so
every construction path defines it. The `.svar` branch (lines 113-121) then sets it and
clears `_reconstruct_window`:

```python
                backend = _Svar1Backend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                _reconstruct_window = None
                _backend_obj = backend
```

and after the branch, set it (add alongside the other `object.__setattr__` calls, and default `_backend_obj = None` before the branch so the injected path leaves it None):

```python
        object.__setattr__(self, "_backend", _backend_obj)
```

Replace `_iter_batches` (lines 217-233) with:

```python
    def _iter_batches(self, batch_size: int) -> Iterator[tuple]:
        """Drive the plan; generate each window PER BATCH so output is batch-bounded.

        The window is the READ granularity; a batch is the GENERATION granularity. For
        the real SVAR1 backend this reads a window's offsets once, then reconstructs
        each batch_size slice separately (issue #284). The injected `_reconstruct_window`
        path (tests) reconstructs the whole window and slices -- memory-unbounded, but
        only ever used with tiny fixtures.
        """
        for r_idx, s_idx in self._plan():
            n_s = len(s_idx)
            flat_r = np.repeat(self._sort_order[r_idx], n_s)
            flat_s = np.tile(s_idx, len(r_idx))
            n_rows = len(flat_r)
            if self._backend is not None:
                o_starts, o_stops = self._backend.read_window(r_idx, s_idx)
                for lo in range(0, n_rows, batch_size):
                    hi = min(lo + batch_size, n_rows)
                    data = self._backend.generate_batch(
                        r_idx, s_idx, o_starts, o_stops, lo, hi
                    )
                    yield data, flat_r[lo:hi], flat_s[lo:hi]
            else:
                data = self._reconstruct_window(r_idx, s_idx)
                for lo in range(0, n_rows, batch_size):
                    hi = min(lo + batch_size, n_rows)
                    yield data[lo:hi], flat_r[lo:hi], flat_s[lo:hi]
```

- [ ] **Step 5: Mechanical update to `test_svar1_window.py`**

`tests/dataset/test_svar1_window.py` calls `_Svar1Backend.reconstruct_window` and/or expects `reconstruct_haplotypes_svar1`. Update those call sites: replace a single `reconstruct_window(r_idx, s_idx)` assertion with `read_window` + `generate_batch(..., 0, n_rows)` producing the same bytes, OR drive through `sds.to_iter(...)`. Keep every existing byte-level assertion; only the call surface changes.

- [ ] **Step 6: Rebuild and run parity + window tests**

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev maturin develop --release 2>&1 | tail -2
pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_svar1_window.py tests/dataset/test_streaming_scheduler.py -q 2>&1 | tail -15
```
Expected: **`test_streaming_matches_written_all_cells` PASSES** (the control), plus the new `test_backend_read_then_generate_matches_whole_window`. If parity fails, STOP — the split changed behavior.

- [ ] **Step 7: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add python/genvarloader/_dataset/_streaming.py tests/dataset/
git commit -m "feat(streaming): per-batch SVAR1 generation, output bounded by batch_size (#284)

_Svar1Backend splits into read_window (offsets, per window) + generate_batch
(per batch). _iter_batches reads a window once then generates each batch slice,
so peak output no longer scales with window size. Parity green."
```

---

### Task 4: Cohort-scale RSS gate + entries-touched still green

**Files:**
- Create/Modify: `tests/dataset/test_streaming_scale.py` (add the cohort-scale RSS test; keep the existing entries-touched/batch-invariance tests green)

**Interfaces:**
- Consumes: `svar1_csr_entries_touched` (unchanged), `sds.to_iter`.

**Background:** Per-batch generation (Task 3) should make peak RSS growth flat as `n_samples` grows at fixed `batch_size`. This is the #284 gate. Deterministic-ish (RSS is coarse) — assert growth stays well below the whole-window output size, and does not track the cohort. The existing `test_entries_touched_*` tests must still pass unchanged (the read path's asymptotics are untouched).

- [ ] **Step 1: Write the failing test**

Add to `tests/dataset/test_streaming_scale.py` (reuse its existing `scale_fixture`/`_make_vcf` helpers; if the file builds a fixed 20-sample fixture, add a parametrizable builder or a second fixture with a `n_samples` knob — mirror the existing one):

```python
def test_peak_rss_is_flat_in_cohort_size(tmp_path):
    """THE #284 GATE. At fixed batch_size, peak RSS growth across a full sweep must NOT
    scale with the number of samples: per-batch generation caps output at batch_size,
    and the offsets buffer is max_mem-bounded. Whole-window generation (the old path)
    would grow output ~linearly in n_samples and blow this."""
    import resource
    import subprocess

    import numpy as np
    import polars as pl
    from genoray import VCF, SparseVar

    def build(n_samples: int):
        d = tmp_path / f"n{n_samples}"
        d.mkdir()
        ref = d / "ref.fa"
        rng = np.random.default_rng(1)
        seq = "".join(rng.choice(list("ACGT"), 4000))
        ref.write_text(f">chr1\n{seq}\n")
        subprocess.run(["samtools", "faidx", str(ref)], check=True)
        vcf = d / "in.vcf"
        lines = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr1,length=4000>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(f"S{i}" for i in range(n_samples)),
        ]
        pos = np.sort(rng.choice(np.arange(2, 3998), 200, replace=False))
        for p in pos:
            gts = "\t".join(f"{rng.integers(0,2)}|{rng.integers(0,2)}" for _ in range(n_samples))
            lines.append(f"chr1\t{p}\t.\tA\tG\t.\t.\t.\tGT\t{gts}")
        vcf.write_text("\n".join(lines) + "\n")
        bcf = d / "in.bcf"
        subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
        subprocess.run(["bcftools", "index", str(bcf)], check=True)
        svar = d / "store.svar"
        SparseVar.from_vcf(svar, VCF(bcf), max_mem="1g",
                           samples=[f"S{i}" for i in range(n_samples)], overwrite=True)
        return svar, ref

    def peak_growth(n_samples: int) -> int:
        svar, ref = build(n_samples)
        bed = pl.DataFrame({"chrom": ["chr1"] * 4, "chromStart": [0, 100, 200, 300],
                            "chromEnd": [100, 200, 300, 400]})
        sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")
        list(sds.to_iter(batch_size=4))  # warm up allocator
        before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        for _ in range(3):
            list(sds.to_iter(batch_size=4))
        after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return (after - before) * 1024  # KiB -> bytes on Linux

    small = peak_growth(50)
    large = peak_growth(400)
    # 8x the cohort must NOT produce ~8x the peak growth; per-batch output is flat.
    assert large < max(small, 8 * 1024 * 1024) * 2, (
        f"peak RSS growth scaled with cohort (50->{small}B, 400->{large}B) -- "
        "whole-window output materialization has returned"
    )
```

- [ ] **Step 2: Run to verify it passes (behavior already correct after Task 3)**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev pytest tests/dataset/test_streaming_scale.py -q 2>&1 | tail -12`
Expected: PASS (RSS-flat + the pre-existing entries-touched/batch-invariance tests). If the new test is flaky at these sizes, widen the sample gap (50 vs 800) rather than loosening the ratio — the signal is flat-vs-linear.

- [ ] **Step 3: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add tests/dataset/test_streaming_scale.py
git commit -m "test(streaming): cohort-scale peak-RSS gate for #284"
```

---

### Task 5: Measure the `max_mem` / region-window default; docs + roadmap

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (confirm/adjust `region_target`, `max_mem` default)
- Modify: `docs/source/api.md`, `docs/source/dataset.md`, `docs/source/faq.md`, `skills/genvarloader/SKILL.md`
- Modify: `docs/roadmaps/streaming-dataset.md`

**Interfaces:** none (docs + measured constant).

**Background:** The public API gained `max_mem` — the docs-audit gate fires. The `region_target=64` knee and `max_mem="512MB"` default are carried from Plan 2's sweep; re-confirm on the scale fixture and record.

- [ ] **Step 1: Sweep and confirm the region knee**

Run (point at the scale fixture built by Task 4, or rebuild inline):
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev python - <<'PY'
import time
import numpy as np, polars as pl
import genvarloader as gvl
# ... build/point svar, ref (mirror test_streaming_scale.build) ...
svar, ref = "<svar path>", "<ref path>"
bed = pl.DataFrame({"chrom": ["chr1"]*20,
                    "chromStart": list(range(0,2000,100)),
                    "chromEnd": list(range(100,2100,100))})
for rt in (1, 4, 16, 64, 256):
    sds = gvl.StreamingDataset(bed, reference=ref, variants=svar).with_seqs("haplotypes")
    object.__setattr__(sds, "_window_regions", rt)
    t0 = time.perf_counter(); n = sum(1 for _ in sds.to_iter(batch_size=8)); dt = time.perf_counter()-t0
    print(f"window_regions={rt:4d} batches={n:4d} {dt:.3f}s")
PY
```
Pick the knee (smallest `region_target` past which wall-clock flattens; best of ≥3 runs; this node is noisy — use the curve shape, not one number). If flat everywhere, keep 64 and say so. Update `region_target` in `__init__` only if the knee moved.

- [ ] **Step 2: Update docs**

- `docs/source/api.md` — add `max_mem` to the `StreamingDataset` autoclass `:members:` list (and keep `__all__` sync green).
- `docs/source/dataset.md` — the `StreamingDataset` section: add a sentence that peak memory is bounded by `max_mem` (offsets) + `batch_size` (output), independent of cohort size; show `max_mem=` in the example.
- `docs/source/faq.md` — the streaming entry: note the `max_mem` knob and cohort-independent memory.
- `skills/genvarloader/SKILL.md` — the `StreamingDataset` section + gotchas: document `max_mem`, per-batch generation, cohort-independent peak memory.

- [ ] **Step 3: Verify the api.md/`__all__` sync gate**

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`.

- [ ] **Step 4: Update the roadmap**

In `docs/roadmaps/streaming-dataset.md`, under Plan 2: tick the #284 memory concern as resolved; record the read/generate split, the `max_mem` budget, the measured `region_target`, and the cohort-scale RSS result. Note PR 2 (#283 engine) is next.

- [ ] **Step 5: Full verification**

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev pytest tests -q 2>&1 | tail -5
pixi run -e dev cargo-test 2>&1 | tail -3
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck 2>&1 | tail -3
```
Expected: all green.

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add docs/ skills/ python/genvarloader/_dataset/_streaming.py
git commit -m "docs(streaming): document max_mem + cohort-independent memory (#284)"
```

---

### Task 6: Open PR 1

**Files:** none (git/gh).

- [ ] **Step 1: Push and open the draft PR into `streaming`**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev prek-install
git push -u origin HEAD
gh pr create --draft --base streaming \
  --title "perf(streaming): cohort-independent peak memory via per-batch SVAR1 generation (#284)" \
  --body "Closes #284. Splits the SVAR1 window read (offsets, window-granular) from generation (per batch, output bounded by batch_size) and adds a max_mem byte budget that sample-chunks the offsets buffer. Peak memory is now independent of cohort size. Spec: \`docs/superpowers/specs/2026-07-17-streaming-svar1-engine-and-memory-design.md\`. Sets up the read/generate seam #283 needs.

Parity (\`test_streaming_matches_written_all_cells\`) green; new cohort-scale RSS gate in \`test_streaming_scale.py\`.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 2: Add the PR to the StreamingDataset project board** (per the project workflow), cross-linked to #284.

---

# PR 2 — overlap producer I/O with consumer generation (#283)

Move the window read ahead of generation. **Which vehicle ships is decided by the Task 9 cold-cache measurement** (Design A: producer thread via the landed engine; Design C: single-thread read-through prefetch). Build both, measure, keep the winner. Prefetch is a **read-through over the existing `geno_v_idxs()` slice** — no genoray API change (`madvise` is an optional genoray-side follow-up, out of scope).

---

### Task 7: `StreamBackend for Svar1Store` + `WindowSpec` sample fields + `Svar1Window: Default`

**Files:**
- Modify: `src/stream/mod.rs` (`WindowSpec` gains `s_lo`/`s_hi`; add a `#[cfg(test)]` `Svar1`-shaped backend test if useful — keep the 5 existing engine tests untouched)
- Modify: `src/svar1/mod.rs` (`impl Default for Svar1Window`)
- Modify: `src/svar1/store.rs` (`impl StreamBackend for Svar1Store` — but it needs per-window contig-local arrays that `WindowSpec` lacks; see Background) 

**Interfaces:**
- Produces:
  ```rust
  pub struct WindowSpec { pub contig_idx: usize, pub r_lo: usize, pub r_hi: usize,
                          pub s_lo: usize, pub s_hi: usize }   // s_lo/s_hi NEW
  impl Default for Svar1Window
  ```

**Background:** `StreamBackend::fill(&self, &WindowSpec, &mut Buffer)` gets only a `WindowSpec` (contig + region + sample spans), but `Svar1Store::read_window` also needs the contig-local `v_starts_c`/`v_ends_c` slices, the contig name, and the mapped sample indices — which live on the Python `_Svar1Backend`. So the production engine is driven from a **new pyclass** (Task 8) that owns those arrays and closes over them; the `StreamBackend` impl here is the thin adapter the pyclass uses. Keep this task to the type-level plumbing (`WindowSpec` fields, `Svar1Window: Default`) plus a unit test that a `Svar1Store`-backed `fill` populates offsets; the full drive is Task 8.

- [ ] **Step 1: Write the failing test**

Add to `src/stream/mod.rs`'s test module — a `WindowSpec` with sample fields must construct, and `run_windows` must still pass its 5 existing tests with the new fields (update the test `windows()` helper to set `s_lo: 0, s_hi: 1`):

```rust
    #[test]
    fn window_spec_carries_sample_span() {
        let w = WindowSpec { contig_idx: 0, r_lo: 0, r_hi: 10, s_lo: 5, s_hi: 12 };
        assert_eq!(w.s_hi - w.s_lo, 7);
    }
```

And in `src/svar1/mod.rs`'s tests (or a new `#[cfg(test)]` there):

```rust
#[cfg(test)]
mod window_default_tests {
    #[test]
    fn svar1_window_default_is_empty() {
        let w = super::Svar1Window::default();
        assert!(w.o_starts.is_empty() && w.o_stops.is_empty());
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev cargo-test 2>&1 | tail -15`
Expected: FAIL — `WindowSpec` has no field `s_lo`; `Svar1Window: Default` not satisfied. (The 5 existing engine tests also fail to compile until their `windows()` helper adds `s_lo`/`s_hi` — update it.)

- [ ] **Step 3: Add the fields + `Default`**

In `src/stream/mod.rs`, extend `WindowSpec` (lines 39-44):

```rust
#[derive(Clone, Debug)]
pub struct WindowSpec {
    pub contig_idx: usize,
    pub r_lo: usize,
    pub r_hi: usize,
    pub s_lo: usize,
    pub s_hi: usize,
}
```

Update the test-module `windows(n)` helper (in the same file) to set `s_lo: 0, s_hi: 1` on each `WindowSpec` it builds, so the 5 existing engine tests compile and still pass.

In `src/svar1/mod.rs`, add after the `Svar1Window` struct:

```rust
impl Default for Svar1Window {
    fn default() -> Self {
        Svar1Window {
            o_starts: Vec::new(),
            o_stops: Vec::new(),
            geno_offset_idx: ndarray::Array2::zeros((0, 0)),
        }
    }
}
```

Update `src/svar1/mod.rs`'s doc comment on `Svar1Window` to drop the stale "borrowed straight from the mmap ... degenerate" framing per the superseding spec (the buffer is offsets; the mmap is the shared page cache, reached at generate time — not "borrowed into the buffer").

- [ ] **Step 4: Run the tests**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev cargo-test 2>&1 | grep -E "window_spec|svar1_window_default|stream::|test result" | head`
Expected: new tests + the 5 engine tests pass.

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add src/stream/mod.rs src/svar1/mod.rs
git commit -m "feat(streaming): WindowSpec sample span + Svar1Window Default for the engine"
```

---

### Task 8: `StreamEngine` pyclass (Design A) — producer thread drives read; consumer generates

**Files:**
- Create: `src/ffi/stream_engine.rs` (or add to `src/ffi/mod.rs`) — `#[pyclass] Svar1StreamEngine`
- Modify: `src/lib.rs` (register the pyclass)
- Modify: `python/genvarloader/_dataset/_streaming.py` (`_iter_batches` uses the engine for the real path)

**Interfaces:**
- Consumes: `run_windows`, `Svar1Store::read_window`, `Svar1Store::geno_v_idxs`, the generation core.
- Produces:
  ```rust
  #[pyclass]
  pub struct Svar1StreamEngine { /* owns Arc<Svar1Store>, static tables, ref, plan */ }
  // Python-facing: constructed with the plan + all static arrays; yields (data, offsets)
  // per BATCH via a `next_batch()` method (producer thread reads window N+1 offsets +
  // reads-through the runs while the consumer generates batches from window N).
  ```

**Background:** This is the highest-risk task (first production threading). The engine owns `Arc<Svar1Store>` (already `Send+Sync`), the static tables (moved in once), and the `Vec<WindowSpec>` plan. A producer thread runs `run_windows`, whose `fill` = `read_window` (offsets) + a **read-through** over each run's `geno_v_idxs()[o_start..o_stop]` slice (faults the pages the consumer will read into the shared page cache; use `std::hint::black_box` on a fold so it isn't optimized away). The consumer side (`next_batch`, GIL held only to build `PyArray`s) generates each `batch_size` slice off the shared mmap. Because `run_windows` is a blocking driver (it calls `consume` inline), expose it to Python as an **iterator that pulls**: the cleanest shape is a channel where the producer sends filled offset buffers and Python's `next_batch` receives + generates. Reuse `run_windows` if its `consume` closure can push into a Python-visible queue; otherwise a thin bespoke producer using the same crossbeam pattern is acceptable — keep the shutdown/panic discipline identical (join-then-classify; shutdown by `Sender` drop).

**Design note (keep it honest):** if wiring `run_windows` directly proves to fight the pull-based `next_batch` shape, a small purpose-built producer loop (same channels, same shutdown rules) is fine and should be documented as such — do not contort the generic engine. The generic engine remains #276's mechanism regardless.

- [ ] **Step 1: Write the failing test (Rust producer/consumer smoke)**

Add a `#[cfg(test)]` test next to the engine that drives a 2-window plan through the producer/consumer path with a stub store fixture (reuse `store.rs`'s `write_raw` pattern via a shared test helper), asserting every batch arrives in plan order and offsets match a direct `read_window`. (Parity is the real gate — Step 5 — but this pins the threading in isolation first.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev cargo-test 2>&1 | tail -10`
Expected: FAIL — engine type does not exist.

- [ ] **Step 3: Implement `Svar1StreamEngine`**

First, factor the `svar1_generate_batch` detach body (Task 1) into a shared, GIL-free
`fn generate_batch_core(store, o_starts, o_stops, region_bounds, static tables, ref, pad_char, parallel) -> (Array1<u8>, Array1<i64>)` in `src/ffi/mod.rs`, and have the Task-1
`svar1_generate_batch` call it — so the FFI path and the engine share one implementation
(DRY). Then implement the pyclass. Concrete starting skeleton (the exact producer shape
is deliberately flexible — see the Design note; keep the shutdown/panic discipline
identical to `run_windows`):

```rust
/// Producer/consumer SVAR1 streamer (Design A): a background thread reads window N+1's
/// offsets and reads-through its runs (warming the shared page cache) while the consumer
/// generates batches from window N. `next_batch()` pulls one batch (GIL held only to
/// build PyArrays). Prefetch is a read-through over `geno_v_idxs()` — no genoray change.
#[pyclass]
pub struct Svar1StreamEngine {
    store: std::sync::Arc<crate::svar1::store::Svar1Store>,
    // static tables + ref (owned, moved in once), the Vec<WindowSpec> plan, the batch
    // size, and the per-contig (v_starts_c, v_ends_c, contig_name, phys_sample map).
    // Producer handle + crossbeam Receiver<FilledWindow> lazily created on first pull.
    inner: std::sync::Mutex<EngineState>,
}

struct FilledWindow { o_starts: Vec<i64>, o_stops: Vec<i64>, r_idx: Vec<usize>, s_idx: Vec<usize> }

#[pymethods]
impl Svar1StreamEngine {
    // #[new] — build from the store + plan + static arrays (mirror _Svar1Backend's
    // construction inputs, crossing as PyReadonlyArray1 and cloned into owned Vecs so
    // the producer thread can hold them).

    /// Return the next batch's (data, offsets), or None when the plan is exhausted.
    /// Drains the current filled window batch-by-batch; when a window is spent, recv()s
    /// the next prefetched window (GIL released) and recycles the slot. Producer errors
    /// surface here; a producer panic is join-then-classified, never a hang.
    fn next_batch<'py>(&self, py: Python<'py>)
        -> PyResult<Option<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>> {
        // 1. ensure producer spawned (run_windows with fill = read_window + read-through,
        //    OR a bespoke producer loop using the same crossbeam bounded(2) + Sender-drop
        //    shutdown — whichever the Design note settles on).
        // 2. if current window has rows left, generate_batch_core on the next slice,
        //    into_pyarray, return Some(...).
        // 3. else recv() next FilledWindow (GIL released via py.detach), recycle, recurse;
        //    Err/closed => join producer, classify, return None or propagate.
        todo!("see Step 3 body")
    }
}
```

`fill` = `read_window` + a read-through over each run: `let s = store.geno_v_idxs(); let _ = std::hint::black_box(s[o_lo..o_hi].iter().fold(0i64, |a, &x| a ^ x as i64));` for every `(o_lo, o_hi)` in the window (this faults the exact pages the consumer will read into the shared page cache; `black_box` stops the compiler eliding it). Follow `reconstruct_haplotypes_from_svar2_readbound`'s `PyRef<'py>` + `py.detach` discipline for GIL handling on the consumer side. If reusing `run_windows` fights the pull-based `next_batch`, a small bespoke producer loop with the same channels and shutdown rules is acceptable and should be commented as such.

- [ ] **Step 4: Run the Rust test**

Run: `cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory && pixi run -e dev cargo-test 2>&1 | grep -E "engine|stream::|test result" | head`
Expected: PASS. Then run 20× for races:
```bash
for i in $(seq 1 20); do cargo test --release svar1_stream 2>&1 | grep -c "test result: ok"; done
```
Expected: consistent, no hangs.

- [ ] **Step 5: Wire the engine into `_iter_batches` and run parity**

In `_streaming.py`, the real backend path in `_iter_batches` constructs/drives the engine instead of the read/generate two-call loop. Keep the injected `_reconstruct_window` test path unchanged. Set `n_slots = 2` in the `max_mem` derivation (Task 2 Step 3) so the budget accounts for ping-pong residency.

Run:
```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev maturin develop --release 2>&1 | tail -2
pixi run -e dev pytest tests -q 2>&1 | tail -6
```
Expected: **parity green** across the full tree. If red, STOP.

- [ ] **Step 6: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git add src/ python/genvarloader/_dataset/_streaming.py Cargo.toml
git commit -m "feat(streaming): Svar1StreamEngine producer/consumer overlap (Design A, #283)

Producer thread reads window N+1 offsets and reads-through the runs (warming the
shared page cache) while the consumer generates batches from window N. First
production threading in src/. Parity green."
```

---

### Task 9: Design C (single-thread read-through) + cold-cache A-vs-C measurement; ship the winner

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (a Design-C code path, selectable)
- Create: `benchmarking/streaming/cold_cache_overlap.py` (measurement harness)
- Modify: `docs/roadmaps/streaming-dataset.md`

**Interfaces:** none public (an internal strategy toggle, removed once the winner is chosen).

**Background:** Design C: no producer thread — before generating window N, compute window N+1's offsets and read-through its runs, letting the current-window generation + user compute overlap the readahead. Measure A vs C on a **cold page cache** (store > RAM, or drop caches between runs if permitted). This node is noisy — report the curve/overlap fraction, not a single wall-clock, as secondary color; the ship decision is "which reliably overlaps I/O for SVAR1."

- [ ] **Step 1: Implement Design C behind an internal toggle**

Add a single-thread prefetch path (compute next window's offsets + read-through) as an alternative to the engine in `_iter_batches`, selected by an internal attribute (e.g. `_prefetch_strategy in {"engine", "readahead"}`), defaulting to `"engine"`. No public API.

- [ ] **Step 2: Write the measurement harness**

`benchmarking/streaming/cold_cache_overlap.py`: build/point at a large store (> RAM if possible), drop caches if permitted (`echo 3 > /proc/sys/vm/drop_caches` needs root — otherwise use a fresh large store per run), time a full `to_iter` sweep under `"engine"` vs `"readahead"`, report wall-clock (best of ≥3) and, if cheap, an overlap estimate (producer-busy fraction). Print a clear table.

- [ ] **Step 3: Run the measurement and decide**

Run the harness. Record numbers. **Decide:** if Design A reliably beats C on cold cache, keep the engine as default and delete the Design-C toggle (or keep it documented as an option). If A ≈ C for SVAR1, ship Design C (simpler, no threading) as the SVAR1 default, keep the engine (Task 8) landed and validated for #276, and record that the thread's value is in the decode path #276 will exercise.

- [ ] **Step 4: Record in the roadmap**

Add a bullet under Plan 2: the cold-cache A-vs-C result (overlap fraction / wall-clock as secondary color, per the perf-gate convention), the shipped SVAR1 default, and the #276 implication. Remove the internal toggle if only one strategy ships.

- [ ] **Step 5: Full verification + commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev maturin develop --release 2>&1 | tail -2
pixi run -e dev pytest tests -q 2>&1 | tail -5
pixi run -e dev cargo-test 2>&1 | tail -3
pixi run -e dev typecheck 2>&1 | tail -3
git add python/ benchmarking/ docs/
git commit -m "perf(streaming): cold-cache A-vs-C overlap measurement; ship SVAR1 default (#283)"
```

---

### Task 10: Open PR 2

**Files:** none (git/gh).

- [ ] **Step 1: Final full verification**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
pixi run -e dev maturin develop --release 2>&1 | tail -2
pixi run -e dev pytest tests -q 2>&1 | tail -5
pixi run -e dev cargo-test 2>&1 | tail -3
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: all green, `MISSING: none`.

- [ ] **Step 2: Push and open the draft PR into `streaming`**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/streaming-svar1-engine-memory
git push
gh pr create --draft --base streaming \
  --title "perf(streaming): overlap producer I/O with consumer generation (#283)" \
  --body "Closes #283. Wires the landed double-buffer engine (Design A) and a single-thread read-through prefetch (Design C); ships whichever the cold-cache measurement favors for SVAR1. Builds on the read/generate split from #284. Spec: \`docs/superpowers/specs/2026-07-17-streaming-svar1-engine-and-memory-design.md\`.

Prefetch is a read-through over the existing geno_v_idxs() slice — no genoray change. Parity green; cold-cache overlap reported as secondary color per the perf-gate convention.

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
```

- [ ] **Step 3: Add the PR to the StreamingDataset project board**, cross-linked to #283.

---

## Notes for the reviewer

- **Parity is the control** (both PRs). It should need only mechanical edits. Anything more means behavior changed.
- **PR 1 is the memory fix and is independently valuable** — the read/generate split caps output at `batch_size`; the `max_mem` budget bounds the offsets buffer. No threading.
- **Task 8 is where the risk is** — gvl's first *production* threading. Shutdown ordering (`Sender` drop) and join-then-classify are the parts that hang or lie if wrong; keep them identical to `run_windows`'s landed discipline.
- **Do not touch VCF/PGEN or generalize a generation kernel** — that is #276, and its dense-vs-sparse pipeline is deliberately not prejudged here.
- **Prefetch is read-through, not `madvise`** — `madvise` needs a genoray-side `Svar1Reader` method and is an optional follow-up, off this plan's critical path.
