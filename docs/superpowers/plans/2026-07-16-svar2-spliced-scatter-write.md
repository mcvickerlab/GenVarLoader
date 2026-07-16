# SVAR2 Spliced Scatter-Write Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Tasks 1, 2, and 3 are independent (different files, no shared symbols) — dispatch them concurrently with superpowers:dispatching-parallel-agents. Task 4 needs 1+2. Task 5 needs 3+4. Task 6 needs 5. Use Sonnet or weaker for implementation.
>
> **Spec:** `docs/superpowers/specs/2026-07-16-svar2-spliced-scatter-write-design.md` · **Issue:** #273

**Goal:** Make SVAR2 spliced haplotype reads write kernel output directly into final spliced positions, deleting the 13.7 ms Python re-order pass.

**Architecture:** SVAR1's "fused" spliced kernel permutes *metadata* (O(rows)) rather than *bytes* and hands the kernel `permuted_out_offsets`. SVAR2 can't reuse that directly because it calls the kernel once per contig group, so a group's rows land at non-contiguous destinations. We generalize the SVAR2 core from a gap-free `out_offsets` array to explicit per-row `(start, end)` bounds, and add an out-param FFI entry so every contig group scatters into one shared, Python-allocated buffer.

**Tech Stack:** Rust (PyO3, ndarray, rayon) + Python (numpy), pixi, maturin.

## Global Constraints

- **Byte-identity is the contract.** Output must stay byte-identical to SVAR1 and to the current SVAR2 path. See `docs/roadmaps/rust-migration.md`.
- **Rebuild the extension before running any Python test:** `pixi run -e dev maturin develop --release`. `pytest` does NOT rebuild; it silently imports the stale `.so` and would validate the old binary.
- **Run the full tree before pushing** (`pixi run -e dev pytest tests -q`), not just `tests/dataset` — shared code changes and `tests/unit/` is skipped by scoped runs.
- **Lint covers both trees:** `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev typecheck`.
- **Cargo tests need libpython on the loader path:** `export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH` or the test binary can't load libpython.
- **prek hooks must be installed** before committing: `prek install`. First commit is slow (hook env build) — allow ~2 min.
- Conventional commits (commitizen `check` runs as a hook).
- Do NOT touch the unspliced path, `_assemble_haps`, annotated haps, tracks, or variants splicing.

---

### Task 1: Generalize the SVAR2 core to per-row destination bounds

The core currently derives each row's output range from a gap-free offsets array (`(out_offsets[k], out_offsets[k+1])`). That makes interleaved destinations unrepresentable. Replace it with explicit `(start, end)` bounds per row.

The parallel path carves disjoint `&mut [u8]` chunks with a `split_at_mut` chain that walks a cursor forward, so it requires rows in **ascending start order**. It already tolerates *gaps* (it skips `s - cursor` bytes) — the only thing tying it to contiguity is that bounds come from an offsets array. Since a contig group's rows arrive in query order but scatter to non-monotonic destinations, sort row indices by start before carving (~6600 rows → tens of µs, negligible).

**Files:**
- Modify: `src/reconstruct/mod.rs:616-802` (`reconstruct_haplotypes_from_svar2` signature + both dispatch paths)
- Modify: `src/ffi/mod.rs:893` and `src/ffi/mod.rs:1085` (the two callers)
- Modify: `src/reconstruct/mod.rs:1570`, `:1622` (two existing tests)
- Test: `src/reconstruct/mod.rs` (`mod tests`, ~line 805)

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `pub fn reconstruct_haplotypes_from_svar2(out: ArrayViewMut1<u8>, out_bounds: ArrayView2<i64>, regions: ArrayView2<i32>, shifts: ArrayView2<i32>, vk_pos: ArrayView1<i32>, vk_key: ArrayView1<i32>, vk_off: ArrayView1<i64>, dense_pos: ArrayView1<i32>, dense_key: ArrayView1<i32>, dense_range: ArrayView2<i32>, dense_present: ArrayView1<u8>, dense_present_off: ArrayView1<i64>, lut_bytes: ArrayView1<u8>, lut_off: ArrayView1<i64>, ref_: ArrayView1<u8>, ref_offsets: ArrayView1<i64>, pad_char: u8, parallel: bool, filter_exonic: bool)` — `out_bounds` is `(n_work, 2)`; `n_work = out_bounds.nrows()` (no longer `regions.nrows() * shifts.ncols()`).
  - `pub fn bounds_from_offsets(out_offsets: ArrayView1<i64>) -> Array2<i64>`

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/reconstruct/mod.rs`. This is the case a single-contig test can never reach: two rows whose destinations are interleaved with a gap between them, written out of order.

```rust
/// Scatter write: rows given non-monotonic, gapped destinations must land exactly
/// where `out_bounds` says, leaving the gap untouched. Row 0 is written AFTER
/// row 1 in the buffer, mirroring a multi-contig spliced read where one group's
/// rows interleave with another's.
#[test]
fn svar2_scatter_write_honors_out_bounds() {
    for parallel in [false, true] {
        let reference = b"ACGT";
        let ref_ = arr1(reference.as_ref());
        let ref_offsets = arr1(&[0i64, 4]);
        // Two identical single-hap queries on contig 0.
        let regions = ndarray::arr2(&[[0i32, 0, 4], [0i32, 0, 4]]);
        let shifts = ndarray::arr2(&[[0i32], [0i32]]);

        // No variants at all -> each row is the bare reference "ACGT".
        let vk_pos = arr1::<i32>(&[]);
        let vk_key = arr1::<i32>(&[]);
        let vk_off = arr1(&[0i64, 0, 0]);
        let dense_pos = arr1::<i32>(&[]);
        let dense_key = arr1::<i32>(&[]);
        let dense_range = ndarray::arr2(&[[0i32, 0], [0i32, 0]]);
        let dense_present = arr1::<u8>(&[]);
        let dense_present_off = arr1(&[0i64, 0, 0]);
        let lut_bytes = arr1::<u8>(&[]);
        let lut_off = arr1(&[0i64]);

        let pad_char = b'N';
        // Buffer: [row1 @ 0..4][gap 4..6 = another group's rows][row0 @ 6..10]
        let mut out = Array1::<u8>::from_elem(10, b'-');
        let out_bounds = ndarray::arr2(&[[6i64, 10], [0i64, 4]]);

        super::reconstruct_haplotypes_from_svar2(
            out.view_mut(),
            out_bounds.view(),
            regions.view(),
            shifts.view(),
            vk_pos.view(),
            vk_key.view(),
            vk_off.view(),
            dense_pos.view(),
            dense_key.view(),
            dense_range.view(),
            dense_present.view(),
            dense_present_off.view(),
            lut_bytes.view(),
            lut_off.view(),
            ref_.view(),
            ref_offsets.view(),
            pad_char,
            parallel,
            false,
        );

        assert_eq!(
            out.as_slice().unwrap(),
            b"ACGT--ACGT",
            "parallel={parallel}: rows must land at their out_bounds, gap untouched"
        );
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH
cargo test svar2_scatter_write_honors_out_bounds 2>&1 | tail -20
```
Expected: FAIL to compile — `reconstruct_haplotypes_from_svar2` takes `ArrayView1<i64>`, got `ArrayView2<i64>`.

- [ ] **Step 3: Change the core signature and both dispatch paths**

In `src/reconstruct/mod.rs`, change the parameter (line ~618) from `out_offsets: ArrayView1<i64>` to `out_bounds: ArrayView2<i64>`.

Then at lines ~637-639, which currently read:

```rust
    let batch_size = regions.nrows();
    let ploidy = shifts.ncols();
    let n_work = batch_size * ploidy;
```

**Keep the `batch_size` and `ploidy` bindings** — `ploidy` is used by `do_work` (`k / ploidy`, `k % ploidy`) and removing it breaks the closure. Replace only the `n_work` line:

```rust
    let batch_size = regions.nrows();
    let ploidy = shifts.ncols();
    // n_work comes from out_bounds: the caller owns the row->destination mapping.
    let n_work = out_bounds.nrows();
    debug_assert_eq!(
        n_work,
        batch_size * ploidy,
        "out_bounds must have one row per (query, hap)"
    );
```

Replace the parallel carve block (the `if parallel {` branch, lines ~740-777) with:

```rust
    if parallel {
        // Build disjoint per-k mutable slices for all active buffers using the
        // proven split_at_mut chain idiom (mirrors get_reference in reference/mod.rs).
        // &mut [_] slices are Send, unlike raw *mut pointers — safe for rayon closures.
        let bounds: Vec<(usize, usize)> = (0..n_work)
            .map(|k| (out_bounds[[k, 0]] as usize, out_bounds[[k, 1]] as usize))
            .collect();

        // Destinations are caller-supplied and may be scattered (a spliced read
        // interleaves this contig group's rows with other groups'), so carve in
        // ascending-start order rather than row order. The chain's cursor only
        // moves forward; gaps are bytes owned by another call and are skipped.
        let mut order: Vec<usize> = (0..n_work).collect();
        order.sort_unstable_by_key(|&k| bounds[k].0);

        let out_slice = out.as_slice_mut().unwrap();
        let mut out_chunks: Vec<(usize, &mut [u8])> = Vec::with_capacity(n_work);
        {
            let mut rest = &mut out_slice[..];
            let mut cursor = 0usize;
            for &k in &order {
                let (s, e) = bounds[k];
                // Contract: `out_bounds` rows are pairwise disjoint. Walking them in
                // ascending-start order then guarantees `s - cursor` does not
                // underflow and the carved slices are disjoint.
                debug_assert!(
                    s >= cursor && e >= s,
                    "out_bounds must be pairwise disjoint and non-empty (got s={s}, e={e}, cursor={cursor})"
                );
                let (_, tail) = rest.split_at_mut(s - cursor);
                let (mid, tail2) = tail.split_at_mut(e - s);
                out_chunks.push((k, mid));
                rest = tail2;
                cursor = e;
            }
        }

        // SVAR2 read-bound haps are never annotated, so dispatch the un-annotated
        // work items straight across rayon.
        out_chunks.into_par_iter().for_each(|(k, out_chunk)| {
            do_work(k, ArrayViewMut1::from(out_chunk));
        });
    } else {
```

Replace the serial loop body (lines ~783-799) with:

```rust
        for k in 0..n_work {
            let out_s = out_bounds[[k, 0]] as usize;
            let out_e = out_bounds[[k, 1]] as usize;
            debug_assert!(
                out_e >= out_s,
                "out_bounds rows must be non-empty (got out_s={out_s}, out_e={out_e})"
            );

            // SAFETY: `out_bounds` rows are required by the calling contract to be
            // pairwise disjoint address ranges within the same allocation. Because the
            // loop is serial there are no concurrent borrows, so constructing a
            // `&mut [u8]` from each disjoint sub-range is free of aliasing UB.
            let out_chunk =
                unsafe { std::slice::from_raw_parts_mut(out_raw.add(out_s), out_e - out_s) };
            do_work(k, ArrayViewMut1::from(out_chunk));
        }
```

- [ ] **Step 4: Add the offsets→bounds helper**

Add near `reconstruct_haplotypes_from_svar2` in `src/reconstruct/mod.rs`:

```rust
/// Per-row `(start, end)` destination bounds from a gap-free `(n_work + 1,)` offsets
/// array — the adapter for callers that let the kernel size its own contiguous output.
pub fn bounds_from_offsets(out_offsets: ArrayView1<i64>) -> Array2<i64> {
    let n = out_offsets.len() - 1;
    let mut bounds = Array2::<i64>::zeros((n, 2));
    for k in 0..n {
        bounds[[k, 0]] = out_offsets[k];
        bounds[[k, 1]] = out_offsets[k + 1];
    }
    bounds
}
```

Ensure `Array2` is imported in that module (`use ndarray::{Array2, ...}`); add it to the existing `use` if missing.

- [ ] **Step 5: Update the two FFI callers**

In `src/ffi/mod.rs` at line ~893 (union path) and line ~1085 (read-bound path), both currently pass `out_offsets_vec.view()`. Change each to pass bounds. Insert before each call:

```rust
        let out_bounds = reconstruct::bounds_from_offsets(out_offsets_vec.view());
```

and change the argument `out_offsets_vec.view(),` to `out_bounds.view(),`.

- [ ] **Step 6: Update the two existing core tests**

In `src/reconstruct/mod.rs` at lines ~1567 and ~1618, replace:

```rust
        let out_offsets = arr1(&[0i64, 8]);
```
with
```rust
        let out_bounds = ndarray::arr2(&[[0i64, 8]]);
```
(and `arr1(&[0i64, 4])` → `ndarray::arr2(&[[0i64, 4]])` in the second test), then change each call's `out_offsets.view(),` argument to `out_bounds.view(),`.

- [ ] **Step 7: Run the tests to verify they pass**

```bash
export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH
cargo test 2>&1 | tail -20
```
Expected: PASS, including `svar2_scatter_write_honors_out_bounds` and both pre-existing `svar2_*` tests. All 125 cargo tests green.

- [ ] **Step 8: Commit**

```bash
git add src/reconstruct/mod.rs src/ffi/mod.rs
git commit -m "refactor(svar2): per-row destination bounds in the reconstruct core

Generalizes out_offsets to (n_work, 2) out_bounds so callers can scatter
rows into non-contiguous destinations; carve in ascending-start order.
No behavior change — both FFI callers adapt via bounds_from_offsets."
```

---

### Task 2: Scattered-row reverse-complement helper

SVAR1's fused spliced entry RCs negative-strand rows in a separate pass after reconstruct (`rc_flat_rows_inplace`, `src/ffi/mod.rs:1783`). That helper indexes rows via a gap-free offsets array. Add the bounds-addressed counterpart so the SVAR2 scatter path can mirror it. This replaces the Python `reverse_masked` post-pass (1.70 ms measured).

**Files:**
- Modify: `src/reverse.rs` (add fn after `rc_flat_rows_inplace`, line ~69)
- Test: `src/reverse.rs` (`mod tests`, ~line 71)

**Interfaces:**
- Consumes: `rc_row` (existing, `pub(crate)`, `src/reverse.rs:44`).
- Produces: `pub fn rc_bounded_rows_inplace(data: &mut [u8], bounds: ArrayView2<i64>, to_rc: ArrayView1<bool>)`

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/reverse.rs`:

```rust
#[test]
fn rc_bounded_rows_handles_scattered_rows() {
    use ndarray::array;
    // Two rows at scattered destinations with an untouched gap between them.
    // Layout: [row0 "ACGT" @ 0..4]["--" gap 4..6][row1 "AACC" @ 6..10]
    let mut data = b"ACGT--AACC".to_vec();
    let bounds = array![[6i64, 10], [0i64, 4]];
    let to_rc = array![true, false];

    super::rc_bounded_rows_inplace(&mut data, bounds.view(), to_rc.view());

    // Row 0 of the mask addresses bytes 6..10 ("AACC" -> RC -> "GGTT").
    // Row 1 is unmasked; the gap must be untouched.
    assert_eq!(&data, b"ACGT--GGTT");
}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH
cargo test rc_bounded_rows_handles_scattered_rows 2>&1 | tail -10
```
Expected: FAIL to compile — `cannot find function rc_bounded_rows_inplace`.

- [ ] **Step 3: Write the implementation**

Add to `src/reverse.rs` after `rc_flat_rows_inplace`, and extend the imports to `use ndarray::{ArrayView1, ArrayView2};`:

```rust
/// Reverse AND complement bytes within each masked row, addressed by explicit
/// `(start, end)` bounds. The scattered-destination counterpart of
/// [`rc_flat_rows_inplace`]: rows need not be contiguous or in ascending order,
/// which is what a spliced SVAR2 scatter write produces.
pub fn rc_bounded_rows_inplace(data: &mut [u8], bounds: ArrayView2<i64>, to_rc: ArrayView1<bool>) {
    for i in 0..to_rc.len() {
        if !to_rc[i] {
            continue;
        }
        let s = bounds[[i, 0]] as usize;
        let e = bounds[[i, 1]] as usize;
        rc_row(&mut data[s..e]);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

```bash
export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH
cargo test reverse:: 2>&1 | tail -10
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/reverse.rs
git commit -m "feat(reverse): rc_bounded_rows_inplace for scattered rows"
```

---

### Task 3: Multi-contig spliced parity test (regression guard)

**Write this first and independently: it must pass on the CURRENT code**, then keep passing after Task 5. The chr22 benchmark is single-contig, so nothing in the existing suite exercises a spliced read whose rows scatter across contig groups — exactly the case the new carve handles.

Reuses the existing two-contig fixtures (`_src2`, `svar_fixture2`, `svar2_fixture2`, `tests/dataset/test_svar2_dataset.py:647-692`) and the spliced-test shape at `:169`.

**Files:**
- Modify: `tests/dataset/test_svar2_dataset.py` (add after `test_svar2_haplotypes_match_svar1_multicontig`, ~line 731)

**Interfaces:**
- Consumes: fixtures `svar_fixture2`, `svar2_fixture2`, `_src2` (module-scoped, already defined).
- Produces: nothing.

- [ ] **Step 1: Write the test**

```python
def test_svar2_spliced_haplotypes_match_svar1_multicontig(
    tmp_path, svar_fixture2, svar2_fixture2, _src2
):
    """Spliced haplotypes byte-identical to SVAR1 when transcripts span TWO contigs.

    Splice order is (splice_row, sample, ploid, element), but SVAR2 reconstructs
    per contig group — so each group's rows land at NON-contiguous destinations
    interleaved with the other group's. Single-contig spliced tests (and the chr22
    benchmark) never produce that layout, so this is the only guard on the
    scattered-destination path. Minus strand is included so the RC pass is
    exercised on scattered rows too.
    """
    from genoray import SparseVar, SparseVar2

    _bcf, ref = _src2
    # Interleaved contigs, out of sorted order; Tb is minus-strand and multi-exon.
    splice_bed = pl.DataFrame(
        {
            "chrom": ["chr2", "chr1", "chr2", "chr1"],
            "chromStart": [0, 0, 20, 5],
            "chromEnd": [13, 13, 40, 20],
            "strand": ["+", "-", "+", "-"],
            "transcript_id": ["Ta", "Tb", "Ta", "Tb"],
            "exon_number": [1, 1, 2, 2],
        }
    )
    d1 = tmp_path / "mcs1.gvl"
    d2 = tmp_path / "mcs2.gvl"
    gvl.write(
        d1, splice_bed, variants=SparseVar(svar_fixture2), samples=None, overwrite=True
    )
    gvl.write(
        d2,
        splice_bed,
        variants=SparseVar2(svar2_fixture2),
        samples=None,
        overwrite=True,
    )
    settings = {"splice_info": ("transcript_id", "exon_number"), "var_filter": "exonic"}
    ds1 = gvl.Dataset.open(d1, reference=ref, **settings).with_seqs("haplotypes")
    ds2 = gvl.Dataset.open(d2, reference=ref, **settings).with_seqs("haplotypes")

    a = ds1[:, :]
    b = ds2[:, :]
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets)), (
        f"offsets differ: svar1={np.asarray(a.offsets).tolist()} "
        f"svar2={np.asarray(b.offsets).tolist()}"
    )
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))
```

- [ ] **Step 2: Run it to verify it passes on current code**

```bash
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py::test_svar2_spliced_haplotypes_match_svar1_multicontig -v
```
Expected: PASS. If it FAILS, **stop and report** — that is a pre-existing multi-contig spliced bug, not something this plan introduces, and it must be triaged before continuing. Check the transcripts really do split across contig groups (`Ta` on chr2, `Tb` on chr1).

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_svar2_dataset.py
git commit -m "test(svar2): multi-contig spliced parity guard

The chr22 benchmark and existing spliced tests are single-contig, so no
test covers a spliced read whose rows scatter across contig groups."
```

---

### Task 4: Out-param FFI entry (`..._readbound_into`)

Mirrors `reconstruct_haplotypes_from_svar2_readbound` (`src/ffi/mod.rs:951-1112`) but writes into a caller-supplied buffer at caller-supplied bounds instead of allocating.

**It also skips `hap_diffs_svar2` entirely.** In the allocating entry, the diffs pass exists *only* to size `out_offsets` (`src/ffi/mod.rs:1058-1077`); the reconstruct core takes sizes from the bounds and pads/truncates itself. The caller now supplies bounds, so that whole merge pass disappears from this path — savings on top of the re-order.

The out-param + `py.detach` shape is established: `reconstruct_haplotypes_from_sparse` (`src/ffi/mod.rs:560`) already does exactly this.

**Files:**
- Modify: `src/ffi/mod.rs` (add after the allocating entry, ~line 1112)
- Modify: `src/lib.rs:50` (register the pyfunction)

**Interfaces:**
- Consumes: `reconstruct::reconstruct_haplotypes_from_svar2` with `out_bounds` (Task 1); `crate::reverse::rc_bounded_rows_inplace` (Task 2).
- Produces: Python symbol `reconstruct_haplotypes_from_svar2_readbound_into(out, out_bounds, store, contig, region_starts, orig_samples, vk_snp_range, vk_indel_range, dense_snp_range, dense_indel_range, region_bounds, shifts, ref_, ref_offsets, pad_char, to_rc, parallel, filter_exonic=False) -> None`. `out` is `uint8 (total,)`, `out_bounds` is `int64 (n_q*ploidy, 2)` in row order `q*ploidy + p`, `to_rc` is `bool (n_q*ploidy,)` or `None`.

- [ ] **Step 1: Write the implementation**

Add to `src/ffi/mod.rs` after `reconstruct_haplotypes_from_svar2_readbound`:

```rust
/// Scatter-write variant of [`reconstruct_haplotypes_from_svar2_readbound`]: writes
/// each (query, hap) row into `out` at the caller-supplied `out_bounds[k] = (start, end)`
/// instead of allocating a contiguous buffer and returning it.
///
/// This is how the SVAR2 spliced read reaches SVAR1's "fused" behavior: the Python
/// splice plan already knows every row's final address, so each contig group scatters
/// straight into the shared output buffer — no post-kernel re-order, no extra copy.
/// `out_bounds` rows are pairwise disjoint but NOT contiguous or ordered: a group's
/// rows interleave with the other contig groups' rows.
///
/// Unlike the allocating entry, this skips `hap_diffs_svar2` — that pass exists only
/// to size the output, and sizes come from `out_bounds` here.
///
/// `to_rc` (per row, kernel row order) reverse-complements negative-strand rows in
/// place after reconstruction, mirroring `reconstruct_haplotypes_spliced_fused`.
#[pyfunction(signature = (
    out,
    out_bounds,
    store,
    contig,
    region_starts,
    orig_samples,
    vk_snp_range,
    vk_indel_range,
    dense_snp_range,
    dense_indel_range,
    region_bounds,
    shifts,
    ref_,
    ref_offsets,
    pad_char,
    to_rc,
    parallel,
    filter_exonic = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_svar2_readbound_into<'py>(
    py: Python<'py>,
    mut out: PyReadwriteArray1<u8>,
    out_bounds: PyReadonlyArray2<i64>,
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
    to_rc: Option<PyReadonlyArray1<bool>>,
    parallel: bool,
    filter_exonic: bool,
) -> PyResult<()> {
    use crate::reconstruct;
    use crate::svar2;

    let reader = store.reader(contig).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("contig {contig} not in store"))
    })?;

    let shifts_a = shifts.as_array();
    let ploidy = shifts_a.ncols();
    let region_bounds_a = region_bounds.as_array();
    let n_q = region_bounds_a.nrows();

    if out_bounds.as_array().nrows() != n_q * ploidy {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "out_bounds must have n_q*ploidy = {} rows, got {}",
            n_q * ploidy,
            out_bounds.as_array().nrows()
        )));
    }

    // Build `regions` (n_q, 3) as [contig_idx=0, start, end) — `ref_` is the
    // single contig slice the caller passed in (ref_offsets = [0, len]).
    let mut regions = Array2::<i32>::zeros((n_q, 3));
    for q in 0..n_q {
        regions[[q, 1]] = region_bounds_a[[q, 0]];
        regions[[q, 2]] = region_bounds_a[[q, 1]];
    }

    let region_starts_v: Vec<u32> = region_starts.as_array().to_vec();
    let orig_samples_v: Vec<usize> = orig_samples
        .as_array()
        .iter()
        .map(|&x| x as usize)
        .collect();
    let vk_snp_range_v = arr2_to_ranges(vk_snp_range.as_array());
    let vk_indel_range_v = arr2_to_ranges(vk_indel_range.as_array());
    let dense_snp_range_v = arr2_to_ranges(dense_snp_range.as_array());
    let dense_indel_range_v = arr2_to_ranges(dense_indel_range.as_array());

    // See the allocating entry: `ref_` is sliced then `.as_slice().unwrap()`'d inside
    // the kernel, so a non-contiguous view would panic there.
    require_contiguous_1d(&ref_, "ref_")?;

    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let out_bounds_a = out_bounds.as_array();
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());
    let out_a = out.as_array_mut();

    py.detach(move || {
        let rb = genoray_core::query::HapRanges::new(
            &region_starts_v,
            &orig_samples_v,
            &vk_snp_range_v,
            &vk_indel_range_v,
            &dense_snp_range_v,
            &dense_indel_range_v,
            ploidy,
        );
        let br = genoray_core::query::gather_haps_readbound(reader, &rb);

        let (lut_bytes, lut_off_u64) = reader.lut_arrays();
        let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();

        let flat = svar2::split_to_flat(&br);
        let dense_range_a =
            numpy::ndarray::ArrayView2::from_shape((n_q, 2), &flat.dense_range).unwrap();

        // No sizing pass: `out_bounds` already carries every row's destination, so
        // `hap_diffs_svar2` (needed only to build out_offsets) is skipped entirely.
        let mut out_a = out_a;
        reconstruct::reconstruct_haplotypes_from_svar2(
            out_a.view_mut(),
            out_bounds_a,
            regions.view(),
            shifts_a,
            numpy::ndarray::ArrayView1::from(flat.vk_pos.as_slice()),
            numpy::ndarray::ArrayView1::from(flat.vk_key.as_slice()),
            numpy::ndarray::ArrayView1::from(flat.vk_off.as_slice()),
            numpy::ndarray::ArrayView1::from(flat.dense_pos.as_slice()),
            numpy::ndarray::ArrayView1::from(flat.dense_key.as_slice()),
            dense_range_a,
            numpy::ndarray::ArrayView1::from(flat.dense_present.as_slice()),
            numpy::ndarray::ArrayView1::from(flat.dense_present_off.as_slice()),
            numpy::ndarray::ArrayView1::from(lut_bytes.as_slice()),
            numpy::ndarray::ArrayView1::from(lut_off.as_slice()),
            ref_a,
            ref_offsets_a,
            pad_char,
            parallel,
            filter_exonic,
        );

        // In-place RC of negative-strand rows, mirroring the SVAR1 fused splice entry.
        if let Some(to_rc) = to_rc_a.as_ref() {
            crate::reverse::rc_bounded_rows_inplace(
                out_a.as_slice_mut().unwrap(),
                out_bounds_a,
                *to_rc,
            );
        }
    });

    Ok(())
}
```

- [ ] **Step 2: Register the pyfunction**

In `src/lib.rs`, after the existing registration at line ~50:

```rust
    m.add_function(wrap_pyfunction!(
        ffi::reconstruct_haplotypes_from_svar2_readbound_into,
        m
    )?)?;
```

- [ ] **Step 3: Build and verify the symbol exists**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -5
pixi run -e dev python -c "
from genvarloader.genvarloader import reconstruct_haplotypes_from_svar2_readbound_into as f
print('symbol ok:', f)
"
```
Expected: build succeeds; prints `symbol ok: <builtin_function ...>`.

If the borrow checker rejects `out_a` inside `py.detach` (the `Ungil` bound), take a raw slice before the closure instead — `let out_slice: &mut [u8] = out_a.as_slice_mut().unwrap();` — and rebuild the `ArrayViewMut1` inside via `ArrayViewMut1::from(out_slice)`. `&mut [u8]` is `Send`.

- [ ] **Step 4: Verify nothing regressed**

```bash
export LD_LIBRARY_PATH=$PWD/.pixi/envs/dev/lib:$LD_LIBRARY_PATH
cargo test 2>&1 | tail -5
```
Expected: PASS (125 tests).

- [ ] **Step 5: Commit**

```bash
git add src/ffi/mod.rs src/lib.rs
git commit -m "feat(svar2): scatter-write FFI entry for read-bound haplotypes

Writes rows at caller-supplied destinations and skips the diffs sizing
pass (sizes come from out_bounds). Unused until the Python path lands."
```

---

### Task 5: Route the spliced Python path through the scatter write

Replaces the re-order + RC post-pass with a single scatter write. `_getitem_spliced` asserts `jitter == 0` and `deterministic`, so shifts are always zero here — no rng, no diffs needed for shift bounding.

`to_rc` arrives in **permuted** (spliced) order — `_getitem_spliced` builds it as `to_rc_flat[plan.permutation]` (`_query.py:270`) — so it must be un-permuted into kernel row order.

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_haps.py:42-49` (import), `:318-393` (`__call__`), add `_reconstruct_spliced` method
- Test: `tests/dataset/test_svar2_dataset.py` (Task 3's test + existing spliced tests)

**Interfaces:**
- Consumes: `reconstruct_haplotypes_from_svar2_readbound_into` (Task 4); `self._contig_groups` (`:1022`), `self._gather_inputs` (`:1035`), `self._ref_for_contig` (`:1081`).
- Produces: `Svar2Haps._reconstruct_spliced(self, idx, regions, splice_plan, to_rc) -> _Flat[np.bytes_]`.

- [ ] **Step 1: Run the spliced tests to confirm the baseline is green**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -2
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -k spliced -v
```
Expected: PASS (including Task 3's multi-contig test). These are the byte-identity contract — they must stay green through this task.

- [ ] **Step 2: Add the import**

In `python/genvarloader/_dataset/_svar2_haps.py`, add to the `from ..genvarloader import (...)` block (line ~43, keep alphabetical):

```python
    reconstruct_haplotypes_from_svar2_readbound,
    reconstruct_haplotypes_from_svar2_readbound_into,
```

- [ ] **Step 3: Add the `_reconstruct_spliced` method**

Insert after `__call__` (before `haplotype_lengths_for_plan`, line ~396):

```python
    def _reconstruct_spliced(
        self,
        idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        splice_plan: "SplicePlan",
        to_rc: "NDArray[np.bool_] | None",
    ) -> _Flat[np.bytes_]:
        """Reconstruct spliced haplotypes directly into spliced layout (no re-order).

        The splice plan already knows every element's final address, so instead of
        reconstructing in region order and permuting the OUTPUT BYTES afterwards, we
        permute the per-row METADATA (O(rows)) and let each contig group's kernel call
        scatter straight into the shared buffer — the same trick SVAR1's fused spliced
        entry uses (``reconstruct_haplotypes_spliced_fused``).

        The plan's k-index (``k = query * E + e`` with ``E = ploidy`` for haplotypes,
        see ``_splice.build_splice_plan``) is exactly the kernel's row index
        ``k = q * P + p``, so ``plan.permutation`` indexes hap rows with no translation.

        Callers reach this only via ``_getitem_spliced``, which asserts ``jitter == 0``
        and ``deterministic`` — hence zero shifts.
        """
        assert self.store is not None
        regions = np.asarray(regions, np.int32)
        P = int(self.genotypes.shape[-2])
        b = len(idx)
        R_all, S_all = int(self.genotypes.shape[0]), int(self.genotypes.shape[1])
        r_q, si_q = np.unravel_index(np.asarray(idx), (R_all, S_all))

        perm = np.asarray(splice_plan.permutation, np.intp)
        off = np.asarray(splice_plan.permuted_out_offsets, np.int64)
        n_work = b * P
        if len(perm) != n_work:
            raise AssertionError(
                f"splice permutation length {len(perm)} != n_queries*ploidy {n_work}"
            )

        # dest_rank[k] = position of kernel row k within the permuted (spliced) layout.
        dest_rank = np.empty(n_work, np.intp)
        dest_rank[perm] = np.arange(n_work, dtype=np.intp)
        bounds_all = np.empty((n_work, 2), np.int64)
        bounds_all[:, 0] = off[dest_rank]
        bounds_all[:, 1] = off[dest_rank + 1]

        # to_rc arrives in permuted order (_getitem_spliced builds it as
        # to_rc_flat[plan.permutation]); the kernel wants it per row.
        rc_all: NDArray[np.bool_] | None = None
        if to_rc is not None and bool(np.asarray(to_rc).any()):
            rc_all = np.empty(n_work, np.bool_)
            rc_all[perm] = np.asarray(to_rc, np.bool_)

        out = np.empty(int(off[-1]), np.uint8)
        shifts_all = np.zeros((b, P), np.int32)
        p_range = np.arange(P, dtype=np.intp)

        for ci, qsel in self._contig_groups(regions[:, 0].astype(np.int64)):
            gi = self._gather_inputs(r_q[qsel], si_q[qsel], regions[qsel], P)
            ref_, ref_offsets = self._ref_for_contig(ci)
            rows = (qsel[:, None] * P + p_range).ravel()
            g_bounds = np.ascontiguousarray(bounds_all[rows], np.int64)
            g_rc = (
                None
                if rc_all is None
                else np.ascontiguousarray(rc_all[rows], np.bool_)
            )
            g_total = int((g_bounds[:, 1] - g_bounds[:, 0]).sum())
            reconstruct_haplotypes_from_svar2_readbound_into(
                out,
                g_bounds,
                self.store,
                self.ds_contigs[ci],
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                np.ascontiguousarray(shifts_all[qsel], np.int32),
                ref_,
                ref_offsets,
                np.uint8(self.reference.pad_char),  # type: ignore[union-attr]  # reference guaranteed for haplotypes
                g_rc,
                should_parallelize(g_total),
                self.filter == "exonic",
            )

        return _Flat.from_offsets(out, (len(perm), None), off).view("S1")
```

- [ ] **Step 4: Route `__call__` to it**

In `__call__`, immediately after the `RaggedAnnotatedHaps` guard (line ~360, before the `haps, *_ = self.get_haps_and_shifts(` call at line ~365), insert:

```python
        if splice_plan is not None:
            return cast(
                _H,
                self._reconstruct_spliced(
                    idx=idx,
                    regions=np.asarray(regions, np.int32),
                    splice_plan=splice_plan,
                    to_rc=to_rc,
                ),
            )
```

Then delete the now-dead splice branch. Replace lines ~376-393 (from `if splice_plan is None and (to_rc is None` through `return cast(_H, flat)`) with:

```python
        if to_rc is None or not bool(np.asarray(to_rc).any()):
            return cast(_H, haps)

        flat = _Flat.from_offsets(
            np.asarray(haps.data), haps.shape, np.asarray(haps.offsets, np.int64)
        ).view("S1")
        flat = flat.reverse_masked(np.asarray(to_rc, np.bool_), comp=_COMP)
        return cast(_H, flat)
```

- [ ] **Step 5: Run the spliced tests**

```bash
pixi run -e dev maturin develop --release 2>&1 | tail -2
pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -k spliced -v
```
Expected: PASS — byte-identical to SVAR1 on both the single-contig and multi-contig spliced tests.

If bytes differ, the likely cause is `to_rc` permutation direction. `rc_all[perm] = to_rc` maps permuted→row order; the inverse (`rc_all = to_rc[dest_rank]`) is the same mapping — verify against `_query.py:270` rather than flipping it blind.

- [ ] **Step 6: Verify `_ragged_arange_gather` is no longer used by the splice path**

```bash
grep -n "_ragged_arange_gather" python/genvarloader/_dataset/_svar2_haps.py
```
Expected: definitions (~119, 143) plus uses at ~668, ~826, ~999, ~1143 (unspliced + variants paths — these STAY). The former splice-path call inside `__call__` must be gone.

- [ ] **Step 7: Run the full tree + lint**

```bash
pixi run -e dev pytest tests -q 2>&1 | tail -5
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck 2>&1 | tail -3
```
Expected: all green (~1030 pytest). The full tree matters — `tests/unit/dataset/test_build_reconstructor.py` and friends are skipped by scoped runs.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_svar2_haps.py
git commit -m "perf(svar2): scatter-write spliced haplotypes, no Python re-order

Permutes per-row metadata and lets each contig group's kernel call write
straight into the final spliced buffer, deleting the O(bytes) concatenate
and the RC post-pass. Byte-identical to SVAR1."
```

---

### Task 6: Measure, gate, and update the roadmap

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`

**Interfaces:**
- Consumes: everything above.
- Produces: nothing (docs only).

- [ ] **Step 1: Measure before/after in one session**

The node is too noisy for cross-session wall-clock or medians (during the spike svar1's median swung to 45 ms against a 25 ms min). Compare **minimums**, same session, both backends in one process.

```bash
pixi run -e dev pytest tests/benchmarks/test_e2e_svar_splice.py -v 2>&1 | tail -20
```
Expected: parity test PASSES; svar2 spliced benchmark improves against its own baseline.

Baseline to beat (spike, 2026-07-16, minimums of 25 reps): svar1 **25.15 ms**, svar2 **35.33 ms** (1.41×). Target ≈1.0× or better; projection ≈20 ms (≈0.8×).

If the benchmark fixture skips (missing `hg38.fa.bgz` / `plink2` / the uncommitted `chr22_5s_hapsafe.pgen`), say so plainly rather than reporting an unmeasured win.

- [ ] **Step 2: Record results and update the roadmap**

Per the roadmap gate: tick the task, record before/after measurements under the relevant checkpoint, set the phase status marker (⬜/🚧/✅) and PR link. Read `docs/roadmaps/rust-migration.md` first and follow its existing structure — do not invent a new section shape.

- [ ] **Step 3: Commit**

```bash
git add docs/roadmaps/rust-migration.md
git commit -m "docs(roadmap): record svar2 spliced scatter-write results (#273)"
```

- [ ] **Step 4: Push and open a draft PR**

```bash
git push -u origin HEAD
gh pr create --draft --base main \
  --title "perf(svar2): scatter-write spliced haplotypes (#273)" \
  --body "$(cat <<'EOF'
Closes #273.

Gives SVAR2 spliced reads SVAR1's behavior: the kernel writes bytes at their
final spliced address instead of Python re-ordering them afterwards.

## What the spike found

The Python re-order (`_ragged_arange_gather`) was **13.7 ms of a 10.2 ms**
svar2-vs-svar1 gap — and at ~9x the memcpy floor for the same 13.2 MB, it was
per-row numpy dispatch overhead across 6600 rows, not bandwidth. #272 removed the
index materialization; per-row dispatch was what remained.

## Approach

SVAR1's "fused" spliced kernel doesn't know about splicing — it permutes per-element
*metadata* and passes `permuted_out_offsets`. SVAR2 couldn't reuse that because it
reconstructs per contig group, so a group's rows scatter to non-contiguous
destinations. So the core now takes per-row `(start, end)` bounds instead of a
gap-free offsets array, and a new out-param FFI entry scatters each group into one
shared buffer. It also skips the diffs sizing pass (sizes come from the bounds).

## Verification

- Byte-identical to SVAR1: existing spliced parity tests + a **new multi-contig
  spliced test** (the chr22 benchmark is single-contig, so nothing previously
  covered the scattered-destination path).
- Full pytest tree + cargo tests green.
- Perf: same-session minimums, both backends in one process.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the implementer

- **Do not "fix" the double gather.** The spliced path gathers twice (once to size the plan, once to reconstruct). That is a known, measured 3.44 ms, deliberately out of scope (spec §4) — it needs an opaque gather-handle type across the FFI and may be moot after this change. If you think it matters, say so; don't build it.
- **Do not touch the unspliced path.** Its ragged reads let the kernel self-size; preallocating would force an extra diffs gather and could regress. `_assemble_haps` stays (0.06 ms measured, thanks to #272's single-contig fast path).
- **The `.so` is the trap.** Every Python test run after a Rust edit needs `maturin develop --release` first, or you are testing the old binary.
