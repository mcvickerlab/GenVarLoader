# Phase 3 — Reconstruction + Track Realignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the 8 numba-only read-path kernel groups (reference fetch, haplotype reconstruction, track realignment + insertion-fill, track→interval RLE) to Rust as byte-identical 1:1 parity twins behind dispatch, then fuse the haplotypes and tracks `__getitem__` read paths into single Rust boundary crossings.

**Architecture:** Strangler-fig, identical to Phase 2. Each kernel becomes a pure-`ndarray`/`rayon` core in a new `src/` domain module, wrapped by a `#[pyfunction]` in `src/ffi/mod.rs`, registered in `src/lib.rs`, and wired into the existing `genvarloader._dispatch` registry (default `rust`; numba retained as parity reference). Parity is hard-gated (byte-identical); throughput is recorded only.

**Tech Stack:** Rust (ndarray 0.17, rayon 1.12, pyo3 0.28 abi3-py310, numpy 0.28), maturin build, Python 3.10–3.13, numba (reference impls), hypothesis + pytest (parity), pixi (`-e dev`).

## Global Constraints

- **Parity is the hard gate.** Every ported kernel must be **byte-identical** (dtype + shape + values via `np.testing.assert_array_equal`) to its numba twin across hypothesis-generated inputs before it lands. Throughput is **recorded only** — no throughput gate this phase (per the 2026-06-24 decision; the throughput gate lives in Phase 5).
- **Dispatch contract:** new kernels register via `genvarloader._dispatch.register(name, numba=<fn>, rust=<fn>, default="rust")`. `GVL_BACKEND=numba|rust` force-overrides all kernels (used by parity sweeps). Numba impls stay as the registered reference; they are deleted wholesale in Phase 5, **not** this phase.
- **Type floors (confirmed at runtime in Phase 2):** `OFFSET_TYPE` = `int64`, genoray `V_IDX_TYPE` = `int32`, `DOSAGE_TYPE` = `float32`. Reference/haplotype bytes are `uint8` (viewed `S1`). Track values are `float32`. Insertion-fill `params` are `float64`; `strategy_ids` are `int8`; PRNG seeds are `uint64`.
- **Numba-fidelity rule:** accumulate length sums in a wider int (`i64`) and truncate on store to mirror numpy's `int32`-slot assignment (Phase 2 precedent in `src/genotypes/mod.rs`). For unsigned PRNG arithmetic, use **wrapping** `u64` ops to mirror numba's `np.uint64` overflow semantics exactly.
- **Offset normalization:** offsets may arrive 1-D `(n+1,)` or 2-D `(2, n)`. Reuse the established `_as_starts_stops` helper (`_genotypes.py:112`) so both backends consume the single `(2, n)` int64 form.
- **abi3 wheels must keep building** across py310–313 × linux/macOS (standing CI invariant).
- **Out of scope this phase:** `_insertion_fill.py:lower` and `_splice.py:build_splice_plan` stay plain Python; variant-flat/flank kernels (done Phase 2); wholesale numba deletion + crate consolidation (Phase 5); genoray IO (Phase 6).
- **Test tmp filesystem:** dataset tests need pytest's tmp on the same filesystem as `tests/data` — run with `--basetemp=<repo>/.pytest_tmp` or the write-path `os.link` hardlink fails cross-device (Errno 18).
- **Branch:** all work lands incrementally on `phase-3-reconstruction` (off `rust-migration`); the phase merges to `rust-migration` as ONE bundled PR. Commit after every kernel.

---

## The porting recipe (every kernel task in §3a–§3c follows this)

This is the invariant mechanical loop. Each task below supplies only the parts that differ (numba source reference, Rust core signature, ffi signature, dispatch name + wiring location, cargo tests, parity strategy + assertion). The 9 steps are always:

1. **Write the failing parity test** — add a hypothesis strategy to `tests/parity/strategies.py` and a `test_<name>_parity.py` under `tests/parity/` using the harness (`assert_kernel_parity` / `assert_kernel_parity_tuple` / `assert_inplace_kernel_parity`). Import the owning `_dataset` module so `register()` runs.
2. **Run it, verify it FAILS** — `pixi run -e dev pytest tests/parity/test_<name>_parity.py -v`. Expected: `KeyError: no kernel registered as '<name>'` (rust not wired yet) or a `register()`-time failure. (Numba-only kernels aren't registered yet, so the test fails until both backends exist.)
3. **Write the Rust core** in `src/<module>/mod.rs` (pure ndarray, no PyO3) translating the numba source **line-by-line**, honoring the numba-fidelity rule. Add `#[cfg(test)] mod tests` cargo unit tests covering the empty/boundary/typical cases listed in the task.
4. **Run cargo tests** — `pixi run -e dev cargo-test` (or `cargo test -p genvarloader <name>`). Expected: PASS.
5. **Add the ffi wrapper** — a `#[pyfunction] pub fn <name>` in `src/ffi/mod.rs` (`PyReadonlyArray*::as_array()` in, `Array::into_pyarray(py)` out, `as_array_mut()` for in-place buffers, `.row(0)/.row(1)` to split normalized offsets).
6. **Register** in `src/lib.rs` — `m.add_function(wrap_pyfunction!(ffi::<name>, m)?)?;`.
7. **Wire dispatch** in the owning `_dataset` module — add `_<name>_rust` thin binding calling `_gvl_rust.<name>(...)`, and a `register("<name>", numba=<numba_fn>, rust=_<name>_rust, default="rust")` call. Route the production call site through `get("<name>")(...)` (or keep the existing wrapper and add the rust branch).
8. **Build + run parity on BOTH backends** — `pixi run -e dev maturin develop` then `GVL_BACKEND=rust pytest tests/parity/test_<name>_parity.py -v` and `GVL_BACKEND=numba …`. Expected: PASS both.
9. **Commit** — `rtk git add … && rtk git commit -m "perf(<area>): port <name> numba->rust (parity)"`.

The Phase 2 reference implementations to mirror for shape/idiom: `src/genotypes/mod.rs` (core), `src/ffi/mod.rs` (boundary), `tests/parity/_harness.py` + `tests/parity/test_get_diffs_sparse_parity.py` (tests), `_genotypes.py:112-167` (`_as_starts_stops` + wrapper + `register`).

---

## File structure

**New Rust modules (created):**
- `src/reference/mod.rs` — `padded_slice`, `get_reference` (par/ser selection inside the core via a `parallel: bool` flag).
- `src/reconstruct/mod.rs` — `reconstruct_haplotype_from_sparse` (singular) + `reconstruct_haplotypes_from_sparse` (batch, rayon), with the optional annotation outputs.
- `src/tracks/mod.rs` — `xorshift64`, `hash4`, `apply_insertion_fill`, `shift_and_realign_track_sparse` (singular) + `shift_and_realign_tracks_sparse` (batch, rayon), `tracks_to_intervals` (+ `scanned_mask`/`compact_mask`).

**Modified:**
- `src/ffi/mod.rs` — one `#[pyfunction]` per ported entry kernel.
- `src/lib.rs` — `pub mod reference; pub mod reconstruct; pub mod tracks;` + `add_function` lines.
- `python/genvarloader/_dataset/_reference.py`, `_genotypes.py`, `_tracks.py`, `_intervals.py` — `_<name>_rust` bindings + `register(...)` + call-site routing.
- `python/genvarloader/_dataset/_utils.py` — `padded_slice` stays (numba reference) but its production callers move behind dispatch via `get_reference`.

**New tests:**
- `tests/parity/strategies.py` — extend with reference/reconstruct/track input strategies.
- `tests/parity/test_get_reference_parity.py`, `test_reconstruct_haplotypes_parity.py`, `test_shift_and_realign_tracks_parity.py`, `test_tracks_to_intervals_parity.py`.
- `tests/parity/test_dataset_parity.py` — extend the existing spy-guarded backstop with haplotypes-mode and tracks-mode (realign) `ds[:, :]` byte-identical checks + fused-path assertions.

---

# Sub-unit 3a — Reference path (warm-up, low parity risk)

### Task 1: `padded_slice` Rust core

Port the leaf used by all reference fetches. It is njit-internal (not a Python entry), so it gets **no** dispatch registration of its own — it is exercised through `get_reference` (Task 2). This task lands the Rust core + cargo tests only.

**Files:**
- Create: `src/reference/mod.rs`
- Modify: `src/lib.rs` (add `pub mod reference;`)

**Numba source to mirror:** `python/genvarloader/_dataset/_utils.py:14-48` (`padded_slice`).

**Interfaces:**
- Produces (consumed by Task 2): `pub fn padded_slice(arr: ArrayView1<u8>, start: i64, stop: i64, pad_val: u8, out: ArrayViewMut1<u8>)` — writes into `out` in place, mirroring the numba semantics: `start >= stop` → no-op; `stop < 0` → fill `pad_val`; otherwise copy `arr[start:stop]` with left/right padding where the slice runs past `[0, len(arr))`.

- [ ] **Step 1: Write the Rust core + cargo tests**

```rust
//! Reference sequence assembly cores (pure ndarray). PyO3 lives in `crate::ffi`.
use ndarray::{ArrayView1, ArrayViewMut1};

/// Copy `arr[start:stop]` into `out`, padding with `pad_val` where the slice
/// runs past `[0, arr.len())`. Mirrors numba `padded_slice`
/// (`_dataset/_utils.py`). `out.len()` MUST equal `stop - start` for the
/// in-bounds case (the caller guarantees this via out_offsets).
pub fn padded_slice(
    arr: ArrayView1<u8>,
    start: i64,
    stop: i64,
    pad_val: u8,
    mut out: ArrayViewMut1<u8>,
) {
    if start >= stop {
        return;
    }
    if stop < 0 {
        out.fill(pad_val);
        return;
    }
    let len = arr.len() as i64;
    let pad_left = (-start).max(0);
    let pad_right = (stop - len).max(0);
    if pad_left == 0 && pad_right == 0 {
        // out[:] = arr[start:stop]
        out.assign(&arr.slice(ndarray::s![start as usize..stop as usize]));
        return;
    }
    let out_len = out.len() as i64;
    if pad_left > 0 && pad_right > 0 {
        let out_stop = out_len - pad_right;
        out.slice_mut(ndarray::s![..pad_left as usize]).fill(pad_val);
        out.slice_mut(ndarray::s![pad_left as usize..out_stop as usize])
            .assign(&arr);
        out.slice_mut(ndarray::s![out_stop as usize..]).fill(pad_val);
    } else if pad_left > 0 {
        // out[:pad_left] = pad; out[pad_left:] = arr[:stop]
        out.slice_mut(ndarray::s![..pad_left as usize]).fill(pad_val);
        out.slice_mut(ndarray::s![pad_left as usize..])
            .assign(&arr.slice(ndarray::s![..stop as usize]));
    } else {
        // pad_right > 0: out[:out_stop] = arr[start:]; out[out_stop:] = pad
        let out_stop = out_len - pad_right;
        out.slice_mut(ndarray::s![..out_stop as usize])
            .assign(&arr.slice(ndarray::s![start as usize..]));
        out.slice_mut(ndarray::s![out_stop as usize..]).fill(pad_val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, Array1};

    fn run(arr: &[u8], start: i64, stop: i64, pad: u8) -> Vec<u8> {
        let a = arr1(arr);
        let mut out = Array1::<u8>::zeros((stop - start).max(0) as usize);
        padded_slice(a.view(), start, stop, pad, out.view_mut());
        out.to_vec()
    }

    #[test]
    fn in_bounds() {
        assert_eq!(run(&[1, 2, 3, 4, 5], 1, 4, 0), vec![2, 3, 4]);
    }
    #[test]
    fn pad_left_only() {
        assert_eq!(run(&[1, 2, 3], -2, 2, 9), vec![9, 9, 1, 2]);
    }
    #[test]
    fn pad_right_only() {
        assert_eq!(run(&[1, 2, 3], 1, 5, 9), vec![2, 3, 9, 9]);
    }
    #[test]
    fn pad_both() {
        assert_eq!(run(&[1, 2], -1, 3, 9), vec![9, 1, 2, 9]);
    }
    #[test]
    fn empty_when_start_ge_stop() {
        assert_eq!(run(&[1, 2, 3], 2, 2, 9), Vec::<u8>::new());
    }
    #[test]
    fn all_pad_when_stop_negative() {
        let a = arr1(&[1u8, 2, 3]);
        let mut out = Array1::<u8>::zeros(3);
        padded_slice(a.view(), -5, -1, 7, out.view_mut());
        // stop < 0 → numba returns early after filling pad_val on the whole out
        assert_eq!(out.to_vec(), vec![7, 7, 7]);
    }
}
```

- [ ] **Step 2: Declare the module** — add `pub mod reference;` to the module list at the top of `src/lib.rs`.

- [ ] **Step 3: Run cargo tests, verify PASS**

Run: `pixi run -e dev cargo-test`
Expected: the 6 `reference::tests::*` cases PASS (and the existing suite stays green).

- [ ] **Step 4: Commit**

```bash
rtk git add src/reference/mod.rs src/lib.rs
rtk git commit -m "perf(reference): port padded_slice numba->rust core (cargo-tested)"
```

---

### Task 2: `get_reference` entry kernel (core + ffi + dispatch + parity)

**Files:**
- Modify: `src/reference/mod.rs` (add `get_reference`), `src/ffi/mod.rs`, `src/lib.rs`
- Modify: `python/genvarloader/_dataset/_reference.py` (`_get_reference_rust` + `register` + route `get_reference`)
- Create: `tests/parity/test_get_reference_parity.py`; extend `tests/parity/strategies.py`

**Numba source to mirror:** `_reference.py:685-723` (`_get_reference_par/_ser`, `_get_reference_row`) + `get_reference` Python entry. The kernel writes `out[out_offsets[i]:out_offsets[i+1]] = padded_slice(ref[c_s:c_e], start, end, pad_char)` for each region `i`, where `regions[i] = (c_idx, start, end)` and `c_s,c_e = ref_offsets[c_idx], ref_offsets[c_idx+1]`. Parallel vs serial is a pure scheduling choice (disjoint out-slices) selected by `should_parallelize(out_offsets[-1])` — **byte-identical regardless of scheduling**, so the Rust core takes a `parallel: bool` flag and uses rayon when true.

**Interfaces:**
- Produces: `pub fn get_reference(regions: ArrayView2<i32>, out_offsets: ArrayView1<i64>, reference: ArrayView1<u8>, ref_offsets: ArrayView1<i64>, pad_char: u8, parallel: bool) -> Array1<u8>` (length `out_offsets[-1]`).
- ffi: `#[pyfunction] pub fn get_reference(py, regions: PyReadonlyArray2<i32>, out_offsets: PyReadonlyArray1<i64>, reference: PyReadonlyArray1<u8>, ref_offsets: PyReadonlyArray1<i64>, pad_char: u8, parallel: bool) -> Bound<PyArray1<u8>>`.
- dispatch name: `"get_reference"`.

- [ ] **Step 1: Add hypothesis strategy** to `tests/parity/strategies.py`

```python
@st.composite
def get_reference_inputs(draw):
    """Generate (regions, out_offsets, reference, ref_offsets, pad_char, parallel)
    with regions whose [start,end) windows may run off either contig edge."""
    import numpy as np
    n_contigs = draw(st.integers(1, 3))
    contig_lens = [draw(st.integers(1, 40)) for _ in range(n_contigs)]
    ref_offsets = np.concatenate([[0], np.cumsum(contig_lens)]).astype(np.int64)
    reference = draw(
        arrays(np.uint8, int(ref_offsets[-1]), elements=st.integers(0, 255))
    )
    n_regions = draw(st.integers(1, 6))
    regions = np.empty((n_regions, 3), np.int32)
    lengths = []
    for i in range(n_regions):
        c = draw(st.integers(0, n_contigs - 1))
        clen = contig_lens[c]
        start = draw(st.integers(-5, clen + 5))
        length = draw(st.integers(0, clen + 5))
        regions[i] = (c, start, start + length)
        lengths.append(length)
    out_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    pad_char = draw(st.integers(0, 255))
    parallel = draw(st.booleans())
    return regions, out_offsets, reference, ref_offsets, np.uint8(pad_char), parallel
```

- [ ] **Step 2: Write the failing parity test** — `tests/parity/test_get_reference_parity.py`

```python
import pytest
from hypothesis import given, settings

from genvarloader._dataset import _reference  # noqa: F401  (triggers register())
from tests.parity._harness import assert_kernel_parity
from tests.parity.strategies import get_reference_inputs

pytestmark = pytest.mark.parity


@settings(deadline=None)
@given(get_reference_inputs())
def test_get_reference_parity(inputs):
    regions, out_offsets, reference, ref_offsets, pad_char, parallel = inputs
    assert_kernel_parity(
        "get_reference", regions, out_offsets, reference, ref_offsets, pad_char, parallel
    )
```

- [ ] **Step 3: Run it, verify FAIL**

Run: `pixi run -e dev pytest tests/parity/test_get_reference_parity.py -q`
Expected: FAIL — `KeyError: no kernel registered as 'get_reference'`.

- [ ] **Step 4: Add the Rust core** to `src/reference/mod.rs`

```rust
use ndarray::{Array1, ArrayView1, ArrayView2};
use rayon::prelude::*;

/// Fetch padded reference rows for each region into one flat buffer.
/// `regions[i] = (contig_idx, start, end)`. Mirrors numba
/// `_get_reference_par/_ser` + `_get_reference_row`. Scheduling (rayon vs
/// serial) does not affect output — out-slices are disjoint.
pub fn get_reference(
    regions: ArrayView2<i32>,
    out_offsets: ArrayView1<i64>,
    reference: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    parallel: bool,
) -> Array1<u8> {
    let total = out_offsets[out_offsets.len() - 1] as usize;
    let mut out = Array1::<u8>::zeros(total);
    let n = regions.nrows();

    // Build disjoint mutable row slices so we can fill each region independently.
    let row = |i: usize, dst: &mut [u8]| {
        let c_idx = regions[[i, 0]] as usize;
        let start = regions[[i, 1]] as i64;
        let end = regions[[i, 2]] as i64;
        let c_s = ref_offsets[c_idx] as usize;
        let c_e = ref_offsets[c_idx + 1] as usize;
        let contig = reference.slice(ndarray::s![c_s..c_e]);
        let mut dst_view = ndarray::ArrayViewMut1::from(dst);
        padded_slice(contig, start, end, pad_char, dst_view.view_mut());
    };

    // Partition `out` into per-region chunks by out_offsets, then fill.
    let bounds: Vec<(usize, usize)> = (0..n)
        .map(|i| (out_offsets[i] as usize, out_offsets[i + 1] as usize))
        .collect();
    let out_slice = out.as_slice_mut().unwrap();
    if parallel {
        // split_at_mut chain over sorted disjoint bounds via chunks_by indices
        let mut chunks: Vec<&mut [u8]> = Vec::with_capacity(n);
        let mut rest = out_slice;
        let mut cursor = 0usize;
        for &(s, e) in &bounds {
            let (_, tail) = rest.split_at_mut(s - cursor);
            let (mid, tail2) = tail.split_at_mut(e - s);
            chunks.push(mid);
            rest = tail2;
            cursor = e;
        }
        chunks
            .into_par_iter()
            .enumerate()
            .for_each(|(i, dst)| row(i, dst));
    } else {
        for (i, &(s, e)) in bounds.iter().enumerate() {
            row(i, &mut out_slice[s..e]);
        }
    }
    out
}
```

Add cargo tests covering: a fully in-bounds region; a region straddling the left edge (`start < 0`); a region straddling the right edge (`end > contig_len`); two contigs with a region in each; `parallel=true` vs `false` produce identical buffers.

- [ ] **Step 5: Run cargo tests, verify PASS** — `pixi run -e dev cargo-test`.

- [ ] **Step 6: Add the ffi wrapper** to `src/ffi/mod.rs`

```rust
use crate::reference;

#[pyfunction]
pub fn get_reference<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    out_offsets: PyReadonlyArray1<i64>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    parallel: bool,
) -> Bound<'py, PyArray1<u8>> {
    let out = reference::get_reference(
        regions.as_array(),
        out_offsets.as_array(),
        reference.as_array(),
        ref_offsets.as_array(),
        pad_char,
        parallel,
    );
    out.into_pyarray(py)
}
```

- [ ] **Step 7: Register** in `src/lib.rs` — add `m.add_function(wrap_pyfunction!(ffi::get_reference, m)?)?;`.

- [ ] **Step 8: Wire dispatch** in `_reference.py`. Add the rust binding + registration and route the existing `get_reference` entry through dispatch:

```python
from genvarloader import _genvarloader as _gvl_rust  # match existing import alias
from genvarloader._dispatch import register, get


def _get_reference_numba(regions, out_offsets, reference, ref_offsets, pad_char, parallel):
    out = np.empty(out_offsets[-1], np.uint8)
    kernel = _get_reference_par if parallel else _get_reference_ser
    return kernel(regions, out_offsets, reference, ref_offsets, pad_char, out)


def _get_reference_rust(regions, out_offsets, reference, ref_offsets, pad_char, parallel):
    return _gvl_rust.get_reference(
        np.ascontiguousarray(regions, np.int32),
        np.ascontiguousarray(out_offsets, np.int64),
        np.ascontiguousarray(reference, np.uint8),
        np.ascontiguousarray(ref_offsets, np.int64),
        int(pad_char),
        bool(parallel),
    )


register("get_reference", numba=_get_reference_numba, rust=_get_reference_rust, default="rust")


def get_reference(regions, out_offsets, reference, ref_offsets, pad_char):
    parallel = should_parallelize(int(out_offsets[-1]))
    return get("get_reference")(regions, out_offsets, reference, ref_offsets, pad_char, parallel)
```

Note: `parallel` is computed in the Python entry (not inside the kernels) so both backends receive the identical flag — this keeps the numba twin byte-identical to today's behavior and makes the strategy's `parallel` field meaningful.

- [ ] **Step 9: Build + run parity on both backends**

Run:
```bash
pixi run -e dev maturin develop
pixi run -e dev pytest tests/parity/test_get_reference_parity.py -q
GVL_BACKEND=numba pixi run -e dev pytest tests/parity/test_get_reference_parity.py -q
```
Expected: PASS (default rust) and PASS (forced numba).

- [ ] **Step 10: Commit**

```bash
rtk git add src/reference/mod.rs src/ffi/mod.rs src/lib.rs \
  python/genvarloader/_dataset/_reference.py \
  tests/parity/test_get_reference_parity.py tests/parity/strategies.py
rtk git commit -m "perf(reference): port get_reference numba->rust (parity, default rust)"
```

---

### Task 3: spliced-reference parity backstop

`_fetch_spliced_ref` (`_reference.py:728-755`) is plain Python that permutes regions via `SplicePlan` then calls `get_reference`. It needs **no** new kernel — Task 2 already covers its hot call. This task adds a dataset-level backstop proving the rust `get_reference` is byte-identical through the splice path.

**Files:**
- Modify: `tests/parity/test_dataset_parity.py`

**Interfaces:**
- Consumes: the `get_reference` dispatch from Task 2; the existing dataset fixtures + backend-forcing helper used by the Phase 0/2 backstops.

- [ ] **Step 1: Add a spy-guarded reference-mode backstop test**

Add a test that opens a reference-bearing dataset (reuse the existing parity fixtures), spies on `genvarloader._genvalloader.get_reference` (or the `_get_reference_rust` binding) to assert it is invoked, materializes `ds[:, :]` for a reference/spliced query under `GVL_BACKEND=rust` and `GVL_BACKEND=numba`, and asserts the two are byte-identical and non-trivially non-zero (the Phase 0 spy lesson — a vacuous pass must be impossible).

```python
def test_reference_mode_dataset_parity(parity_ref_dataset, force_backend, kernel_spy):
    with kernel_spy("get_reference") as spy:
        rust = materialize(parity_ref_dataset, backend="rust")
    assert spy.called
    numba = materialize(parity_ref_dataset, backend="numba")
    assert_ragged_byte_identical(rust, numba)
    assert rust.data.size > 0 and (rust.data != 0).any()
```

(Use the existing helpers in `test_dataset_parity.py`; the names above mirror its Phase 2 patterns — adapt to the actual fixture/spy utilities in that file.)

- [ ] **Step 2: Run, verify PASS** — `pixi run -e dev pytest tests/parity/test_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp`.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/parity/test_dataset_parity.py
rtk git commit -m "test(parity): reference-mode + spliced dataset backstop (spy-guarded)"
```

---

# Sub-unit 3b — Haplotype reconstruction (core)

### Task 4: `reconstruct_haplotype_from_sparse` (singular) Rust core

The ~190-line workhorse. Port it first in isolation with exhaustive cargo tests **before** the batch driver, because every parity edge case lives here (negative `ref_start` padding, DEL spanning start, overlapping ALTs, shift consumption across ref+allele, right-pad with `pad_char`, and the annotation arrays `annot_v_idxs`/`annot_ref_pos`).

**Files:**
- Create: `src/reconstruct/mod.rs`
- Modify: `src/lib.rs` (`pub mod reconstruct;`)

**Numba source to mirror EXACTLY (line-by-line):** `_genotypes.py:277-465` (`reconstruct_haplotype_from_sparse`). Preserve every branch, including the `allele_start_idx == v_len` early-`continue`, the `out_idx + ref_len >= length` break, and the final unfilled/right-pad clause. Annotation writes: reference runs write `annot_v_idxs = -1` and `annot_ref_pos = arange(ref_idx, ref_idx+ref_len)`; allele runs write `annot_v_idxs = variant` and `annot_ref_pos = v_pos`; trailing pad writes `annot_v_idxs = -1` and `annot_ref_pos = i32::MAX` (note: the **leading** pad uses `-1` for ref_pos, the **trailing** pad uses `i32::MAX` — they differ; replicate exactly).

**Interfaces:**
- Produces: `pub fn reconstruct_haplotype_from_sparse(v_idxs: ArrayView1<i32>, v_starts: ArrayView1<i32>, ilens: ArrayView1<i32>, shift: i64, alt_alleles: ArrayView1<u8>, alt_offsets: ArrayView1<i64>, ref_: ArrayView1<u8>, ref_start: i64, out: ArrayViewMut1<u8>, pad_char: u8, keep: Option<ArrayView1<bool>>, annot_v_idxs: Option<ArrayViewMut1<i32>>, annot_ref_pos: Option<ArrayViewMut1<i32>>)`.

- [ ] **Step 1: Port the core** to `src/reconstruct/mod.rs`, translating `_genotypes.py:277-465` statement-by-statement. Keep `ref_idx`, `out_idx`, `shifted` as `i64`/`usize` mirroring the numba ints; use `slice`/`assign`/`fill` for the block writes. Thread the two optional annotation views through with `if let Some(..)` guards at each write site.

- [ ] **Step 2: Add cargo unit tests** covering, each as a named case with hand-computed expected bytes:
  - No variants, `shift=0`, in-bounds → `out == ref[ref_start:ref_start+len]`.
  - Negative `ref_start` → leading pad of `pad_char`, `annot_ref_pos == -1` over the pad.
  - A single SNP (ilen 0) → one byte replaced, `annot_v_idxs == variant` at that base.
  - A 2bp insertion (ilen +2) → allele bytes spliced in, downstream ref shifted.
  - A deletion (ilen −2) → ref skipped, `ref_idx` advances to `v_ref_end`.
  - DEL spanning `ref_start` (`v_pos < ref_start`, `v_diff < 0`, `v_ref_end >= ref_start`) → `ref_idx = v_ref_end`, variant not emitted.
  - Overlapping ALTs at the same pos → only the first applied.
  - `shift` consumed partly by ref + partly by allele (`allele = allele[allele_start_idx:]`).
  - Right-pad clause: `out` longer than ref+variants → trailing `pad_char`, trailing `annot_ref_pos == i32::MAX`.
  - Annotated vs non-annotated calls produce identical `out` bytes.

- [ ] **Step 3: Run cargo tests, verify PASS** — `pixi run -e dev cargo-test`.

- [ ] **Step 4: Commit**

```bash
rtk git add src/reconstruct/mod.rs src/lib.rs
rtk git commit -m "perf(reconstruct): port reconstruct_haplotype_from_sparse core (cargo-tested)"
```

---

### Task 5: `reconstruct_haplotypes_from_sparse` (batch) + ffi + dispatch + parity

**Files:**
- Modify: `src/reconstruct/mod.rs` (batch driver), `src/ffi/mod.rs`, `src/lib.rs`
- Modify: `python/genvarloader/_dataset/_genotypes.py` (binding + `register`), `python/genvarloader/_dataset/_haps.py` (route both reconstruct methods through dispatch)
- Create: `tests/parity/test_reconstruct_haplotypes_parity.py`; extend `strategies.py`

**Numba source to mirror:** `_genotypes.py:158-275` (`reconstruct_haplotypes_from_sparse`). The batch driver loops `(query, hap)`, slices each region's reference (`ref[ref_offsets[c_idx]:ref_offsets[c_idx+1]]`), genotype variant indices (`geno_v_idxs[o_s:o_e]` via normalized offsets), per-(query,hap) keep slice, and the out / annotation sub-slices by `out_offsets[k_idx]:out_offsets[k_idx+1]`, then calls the singular kernel. Per-(query,hap) out-slices are disjoint → rayon-parallelizable, byte-identical to numba's `prange`.

**Interfaces:**
- Produces: `pub fn reconstruct_haplotypes_from_sparse(out: ArrayViewMut1<u8>, out_offsets, regions: ArrayView2<i32>, shifts: ArrayView2<i32>, geno_offset_idx: ArrayView2<i64>, geno_o_starts: ArrayView1<i64>, geno_o_stops: ArrayView1<i64>, geno_v_idxs: ArrayView1<i32>, v_starts, ilens, alt_alleles, alt_offsets, ref_, ref_offsets, pad_char, keep: Option<...>, keep_offsets: Option<...>, annot_v_idxs: Option<ArrayViewMut1<i32>>, annot_ref_pos: Option<ArrayViewMut1<i32>>)` — writes `out` (and optional annotation buffers) in place.
- ffi: `#[pyfunction] pub fn reconstruct_haplotypes_from_sparse(...)` — takes the normalized `(2,n)` geno_offsets and splits with `.row(0)/.row(1)`; out + annotation buffers via `PyReadwriteArray1`; the two annotation params are `Option<PyReadwriteArray1<i32>>`.
- dispatch name: `"reconstruct_haplotypes_from_sparse"`.

> **Rayon + in-place annotation note:** because three buffers (`out`, `annot_v_idxs`, `annot_ref_pos`) are written by disjoint per-(query,hap) slices, parallelize by pre-splitting each buffer into disjoint chunks (same `split_at_mut` chaining as Task 2) and zipping the three chunk-vectors per work item. Keep a serial path for the non-annotated common case and verify both produce identical output in cargo tests.

- [ ] **Step 1: Add the batch strategy** to `strategies.py` — `reconstruct_haplotypes_inputs()` generating a small reference (1–2 contigs), a handful of variants (SNP/ins/del mix) with `v_starts`/`ilens`/`alt_alleles`/`alt_offsets`, sparse genotype offsets, `regions`, `shifts` (0 and small positive), optional `keep`/`keep_offsets`, and out_offsets sized to the query windows. Yield the inputs in **both** annotated and non-annotated variants (a `annotate: bool` field), with the out + annotation buffers built by an `out_factory` for the in-place harness.

- [ ] **Step 2: Write the failing parity test** — `tests/parity/test_reconstruct_haplotypes_parity.py` using `assert_inplace_kernel_parity("reconstruct_haplotypes_from_sparse", inputs, out_factory, out_index)` for the non-annotated case, plus a tuple variant asserting all three buffers (out + annot_v + annot_pos) byte-identical for the annotated case (build a small helper mirroring `assert_inplace_kernel_parity` that compares all three written buffers).

- [ ] **Step 3: Run it, verify FAIL** — `KeyError: no kernel registered as 'reconstruct_haplotypes_from_sparse'`.

- [ ] **Step 4: Implement the batch driver** in `src/reconstruct/mod.rs` (serial + rayon paths) calling the Task 4 singular kernel.

- [ ] **Step 5: Run cargo tests, verify PASS** — include a cargo test asserting serial == parallel on a multi-region input.

- [ ] **Step 6: Add the ffi wrapper** + register in `src/lib.rs`.

- [ ] **Step 7: Wire dispatch** in `_genotypes.py` (mirror the `get_diffs_sparse` wrapper: a `register(...)` plus a public `reconstruct_haplotypes_from_sparse` wrapper that normalizes offsets via `_as_starts_stops` and dispatches). Update `_haps.py:_reconstruct_haplotypes` and `_reconstruct_annotated_haplotypes` to call the dispatched wrapper (they already pass the exact kwargs; only the import/callee changes — keep the `_Flat.from_offsets(...).view("S1")` wrapping unchanged).

- [ ] **Step 8: Build + parity both backends** — `maturin develop`; run the parity test under default and `GVL_BACKEND=numba`. Expected PASS both.

- [ ] **Step 9: Commit**

```bash
rtk git add src/reconstruct/mod.rs src/ffi/mod.rs src/lib.rs \
  python/genvarloader/_dataset/_genotypes.py python/genvarloader/_dataset/_haps.py \
  tests/parity/test_reconstruct_haplotypes_parity.py tests/parity/strategies.py
rtk git commit -m "perf(reconstruct): port reconstruct_haplotypes_from_sparse batch (parity, default rust)"
```

---

### Task 6: haplotypes-mode dataset backstop

**Files:**
- Modify: `tests/parity/test_dataset_parity.py`

- [ ] **Step 1: Add a spy-guarded haplotypes-mode backstop** — spy on the `reconstruct_haplotypes_from_sparse` rust binding, materialize `ds[:, :]` for a haplotypes query (and a spliced-haplotypes query) under both backends, assert byte-identical haplotype bytes **and** (for the annotated path) the variant-index + ref-coord arrays. Assert non-trivial output.

- [ ] **Step 2: Run, verify PASS** — `pytest tests/parity/test_dataset_parity.py -q --basetemp=$(pwd)/.pytest_tmp`.

- [ ] **Step 3: Commit** — `test(parity): haplotypes + spliced-haps dataset backstop (spy-guarded)`.

---

# Sub-unit 3c — Track realignment + RLE (hairiest; parity risks live here)

### Task 7: PRNG (`xorshift64`, `hash4`) Rust core + direct parity

The FlankSample fill is the highest parity risk. Lock the PRNG **before** the kernel that uses it, with a direct numba-vs-rust sequence comparison.

**Files:**
- Create: `src/tracks/mod.rs`
- Modify: `src/lib.rs` (`pub mod tracks;`), `src/ffi/mod.rs` (temporary debug export, see below)
- Create: `tests/parity/test_prng_parity.py`; expose a tiny numba helper in `_tracks.py`

**Numba source to mirror:** `_tracks.py:37-53` (`_xorshift64`, `_hash4`). All ops are on `np.uint64` → use Rust `u64` **wrapping** shifts/xors: `x ^= x.wrapping_shl(13)` etc. (shifts by 13/7/17). `hash4(a,b,c,d) = xorshift64(xorshift64(xorshift64(a^b)^c)^d)`.

**Interfaces:**
- Produces: `pub fn xorshift64(x: u64) -> u64`, `pub fn hash4(a: u64, b: u64, c: u64, d: u64) -> u64`.

- [ ] **Step 1: Implement + cargo-test** the two functions in `src/tracks/mod.rs` with a hardcoded expected vector (compute the first few outputs by hand / from the numba definition and assert).

```rust
/// One round of xorshift64 (wrapping, mirrors numba `_xorshift64` on np.uint64).
#[inline(always)]
pub fn xorshift64(mut x: u64) -> u64 {
    x ^= x.wrapping_shl(13);
    x ^= x >> 7;
    x ^= x.wrapping_shl(17);
    x
}

/// Hash four u64 into one (mirrors numba `_hash4`).
#[inline(always)]
pub fn hash4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    let mut h = a;
    h = xorshift64(h ^ b);
    h = xorshift64(h ^ c);
    h = xorshift64(h ^ d);
    h
}
```

- [ ] **Step 2: Add a direct numba-vs-rust PRNG parity test.** Temporarily expose the rust `hash4` via a `#[pyfunction]` (e.g. `ffi::_debug_hash4`) and a numba `_hash4` accessor in `_tracks.py`, then over a hypothesis grid of `(a,b,c,d)` `uint64` quadruples assert `rust_hash4(a,b,c,d) == int(_hash4(a,b,c,d))`. This is the single most important guard for FlankSample byte-identity.

```python
@given(st.integers(0, 2**64 - 1), st.integers(0, 2**64 - 1),
       st.integers(0, 2**64 - 1), st.integers(0, 2**64 - 1))
def test_hash4_parity(a, b, c, d):
    from genvarloader._dataset._tracks import _hash4
    import numpy as np
    exp = int(_hash4(np.uint64(a), np.uint64(b), np.uint64(c), np.uint64(d)))
    assert _gvl_rust._debug_hash4(a, b, c, d) == exp
```

- [ ] **Step 3: Run both (cargo + pytest), verify PASS.**

- [ ] **Step 4: Commit** — `perf(tracks): port xorshift64/hash4 PRNG (direct numba parity)`.

---

### Task 8: `apply_insertion_fill` (4 strategies) Rust core

**Files:**
- Modify: `src/tracks/mod.rs`

**Numba source to mirror:** `_tracks.py:56-139` (`_apply_insertion_fill`). Strategy IDs (`src/tracks` mirrors `_insertion_fill.py`): `REPEAT_5P=0`, `REPEAT_5P_NORM=1`, `CONSTANT=2`, `FLANK_SAMPLE=3`, `INTERPOLATE=4`. **Float-parity risk lives in INTERPOLATE** — replicate the Lagrange evaluation in the *exact same operation order*: anchors built 5′ side first (`xs[j] = -j`, `ys[j] = track[max(v_rel_pos-j,0)]`) then 3′ side (`xs[k+j] = v_len + j`, `ys[k+j] = track[min(v_rel_pos+1+j, track_len-1)]`), and the per-output accumulation `acc += ys[a] * Π_{b≠a} (x - xs[b])/(xs[a] - xs[b])` with `x = i as f64`, looping `a` outer, `b` inner, skipping `b==a`. Keep all interpolation math in `f64` and store the final `acc` into the `f32` out (matching numba, where `out` is float32 and the arithmetic is float64).

**Interfaces:**
- Produces: `pub fn apply_insertion_fill(out: &mut ArrayViewMut1<f32>, out_idx: usize, writable_length: usize, v_len: i64, track: ArrayView1<f32>, v_rel_pos: i64, strategy_id: i64, params: ArrayView1<f64>, base_seed: u64, query: u64, hap: u64)`. FlankSample uses `hash4(base_seed, query, hap, (out_idx+i) as u64) % pool_size` for each position `i` (note: `query`/`hap` and `out_idx+i` are the per-position seed components — replicate the cast order exactly).

- [ ] **Step 1: Implement** the four branches in `src/tracks/mod.rs`. For `REPEAT_5P_NORM` divide `track[v_rel_pos]` by `v_len as f32`... — **match the numba dtype**: numba computes `track[v_rel_pos] / v_len` where `track` is f32 and `v_len` is a python int → numpy promotes to f32 result? Confirm by reading the numba: the value is stored into f32 `out`; compute in the same precision numba uses (f32/f32 or f64). Mirror exactly; cargo-test against hand values.

- [ ] **Step 2: Cargo-test each strategy** with a fixed `track`, `params`, `base_seed`: Repeat5pNorm (sum-preserving), Constant (params[0]), FlankSample (deterministic given seed — assert exact indices chosen), Interpolate order 1/2/3 (assert against hand-computed Lagrange values; order-1 endpoints must equal the two flanking track values).

- [ ] **Step 3: Run cargo tests, verify PASS.**

- [ ] **Step 4: Commit** — `perf(tracks): port apply_insertion_fill (4 strategies) core (cargo-tested)`.

---

### Task 9: `shift_and_realign_track[s]_sparse` + ffi + dispatch + parity

**Files:**
- Modify: `src/tracks/mod.rs` (singular + batch), `src/ffi/mod.rs`, `src/lib.rs`
- Modify: `python/genvarloader/_dataset/_tracks.py` (binding + `register`), `python/genvarloader/_dataset/_reconstruct.py` (route the call site at `_reconstruct.py:210-227`)
- Create: `tests/parity/test_shift_and_realign_tracks_parity.py`; extend `strategies.py`

**Numba source to mirror:** singular `_tracks.py:230-401`, batch `_tracks.py:141-228`. The singular kernel mirrors the haplotype reconstruct shift logic but on f32 track values, with three key differences: SNPs (`v_diff == 0`) are skipped (tracks match ref there); insertions route to `apply_insertion_fill` unless `strategy_id == REPEAT_5P` (which repeats `track[v_rel_pos]`); deletions/Repeat5p repeat `track[v_rel_pos]`; trailing fill pads with `0` (not `pad_char`). Batch driver loops `(query, hap)` with disjoint out-slices (rayon-safe) and passes `query`/`hap` indices through for the FlankSample seed.

**Interfaces:**
- Produces: `pub fn shift_and_realign_tracks_sparse(out: ArrayViewMut1<f32>, out_offsets, regions: ArrayView2<i32>, shifts: ArrayView2<i32>, geno_offset_idx: ArrayView2<i64>, geno_v_idxs: ArrayView1<i32>, geno_o_starts: ArrayView1<i64>, geno_o_stops: ArrayView1<i64>, v_starts, ilens, tracks: ArrayView1<f32>, track_offsets: ArrayView1<i64>, params: ArrayView1<f64>, keep: Option<...>, keep_offsets: Option<...>, strategy_id: i64, base_seed: u64)`.
- ffi `#[pyfunction] pub fn shift_and_realign_tracks_sparse(...)` — `out` via `PyReadwriteArray1<f32>`; normalized `(2,n)` geno_offsets split with `.row()`; `params` is a 1-D `f64` slice (the per-track row already indexed Python-side as `strat_params[track_ofst]`).
- dispatch name: `"shift_and_realign_tracks_sparse"`.

- [ ] **Step 1: Add the batch strategy** to `strategies.py` — generate a track (f32), variants (SNP/ins/del mix), sparse genos, regions, shifts, optional keep, and for the fill strategy sample `strategy_id ∈ {0,1,2,3,4}` with matching `params` (Constant value; FlankSample width≥0; Interpolate order∈{1,2,3}) and a random `base_seed`. Provide an `out_factory` building the f32 out buffer.

- [ ] **Step 2: Write the failing parity test** using `assert_inplace_kernel_parity("shift_and_realign_tracks_sparse", inputs, out_factory, out_index)`. Ensure the strategy exercises **all five** strategy IDs (especially FlankSample + Interpolate) so byte-identity is proven on the risky paths.

- [ ] **Step 3: Run, verify FAIL** — kernel not registered.

- [ ] **Step 4: Implement** singular + batch in `src/tracks/mod.rs` (calling Task 8's `apply_insertion_fill` and Task 7's `hash4`).

- [ ] **Step 5: Cargo-test** singular kernel cases (no variants → `out = track[:length]`; deletion; insertion under each strategy; shift) + serial==parallel batch.

- [ ] **Step 6: ffi wrapper + register** in `src/lib.rs`.

- [ ] **Step 7: Wire dispatch** in `_tracks.py` (`register(...)` + a wrapper normalizing offsets) and route the `_reconstruct.py:210-227` call site through the dispatched wrapper (kwargs already match; keep the `_Flat.from_offsets(out, out_shape, out_offsets)` wrapping unchanged).

- [ ] **Step 8: Build + parity both backends.** If Interpolate float-parity fails byte-identity after honest operation-order matching, apply the documented fallback: register a strategy-dispatched rust core that handles Repeat5p/Constant/FlankSample/Repeat5pNorm and falls back to numba for `INTERPOLATE` only — and record this in the roadmap decisions log. Attempt strict byte-identity first.

- [ ] **Step 9: Commit** — `perf(tracks): port shift_and_realign_tracks_sparse (parity, default rust)`.

---

### Task 10: `tracks_to_intervals` RLE + ffi + dispatch + parity

**Files:**
- Modify: `src/tracks/mod.rs` (`tracks_to_intervals`, `scanned_mask`, `compact_mask`), `src/ffi/mod.rs`, `src/lib.rs`
- Modify: `python/genvarloader/_dataset/_intervals.py` (binding + `register` + route)
- Create: `tests/parity/test_tracks_to_intervals_parity.py`; extend `strategies.py`

**Numba source to mirror:** `_intervals.py:129-220` (`tracks_to_intervals`, `_scanned_mask`, `_compact_mask`). Returns `(all_starts: i32, all_ends: i32, all_values: f32, interval_offsets: i64)`. RLE: per query, `scanned_mask` = cumulative count of value changes (`backward_mask[0]=True`, `backward_mask[i] = track[i-1] != track[i]`); `compact_mask` recovers run-boundary indices; values are `track[boundaries[:-1]]`; starts/ends are boundaries shifted by `regions[query,1]`. Note `0`-value intervals **are** included (matches numba comment). Per-query work over disjoint output ranges → rayon-safe (but the two-pass cumsum/offsets must mirror numba's `n_intervals.cumsum()`).

**Interfaces:**
- Produces: `pub fn tracks_to_intervals(regions: ArrayView2<i32>, tracks: ArrayView1<f32>, track_offsets: ArrayView1<i64>) -> (Array1<i32>, Array1<i32>, Array1<f32>, Array1<i64>)`.
- ffi returns a 4-tuple of `Bound<PyArray*>`.
- dispatch name: `"tracks_to_intervals"`.

- [ ] **Step 1: Strategy** — generate `regions` + a piecewise-constant `tracks` f32 buffer (draw run lengths + values so RLE has interesting structure, including a single all-constant query and an empty query) + `track_offsets`.

- [ ] **Step 2: Failing parity test** with `assert_kernel_parity_tuple("tracks_to_intervals", regions, tracks, track_offsets)`.

- [ ] **Step 3: Run, verify FAIL.**

- [ ] **Step 4: Implement** in `src/tracks/mod.rs` (two-pass: count intervals per query → cumsum offsets → fill starts/ends/values). Cargo-test against a hand-built RLE example.

- [ ] **Step 5: cargo-test, verify PASS.**

- [ ] **Step 6: ffi + register.**

- [ ] **Step 7: Wire dispatch** in `_intervals.py`; route the production call site through `get("tracks_to_intervals")`.

- [ ] **Step 8: Build + parity both backends.**

- [ ] **Step 9: Commit** — `perf(intervals): port tracks_to_intervals RLE numba->rust (parity, default rust)`.

---

### Task 11: tracks-mode dataset backstop

**Files:**
- Modify: `tests/parity/test_dataset_parity.py`

- [ ] **Step 1: Add a spy-guarded tracks-mode backstop** — spy on `shift_and_realign_tracks_sparse`, materialize `ds[:, :]` for a tracks query that triggers realignment (indel-bearing regions) under both backends across **each** insertion-fill strategy, assert byte-identical realigned tracks + non-trivial output. Include a tracks_to_intervals round-trip check if a public path exercises it.

- [ ] **Step 2: Run, verify PASS** — `--basetemp=$(pwd)/.pytest_tmp`.

- [ ] **Step 3: Commit** — `test(parity): tracks-realign dataset backstop across fill strategies (spy-guarded)`.

---

# Sub-unit 3d — Consolidation (fuse hot read paths; throughput recorded, not gated)

> Goal: collapse the per-kernel boundary crossings + redundant `np.ascontiguousarray` coercions Phase 2 profiling pinned at 62% of the variants loop, for the **haplotypes** and **tracks** read paths. Parity is still hard-gated (dataset-level, byte-identical); throughput is **recorded** in the roadmap.

### Task 12: Audit the haplotypes + tracks `__getitem__` glue

**Files:**
- Create: `docs/roadmaps/phase-3-getitem-glue-audit.md` (scratch findings; can be deleted before merge or folded into the roadmap)

- [ ] **Step 1: Trace + list** every `np.ascontiguousarray` / boundary crossing / intermediate numpy alloc on the live haplotypes path (`__getitem__` → `_haps._reconstruct_haplotypes` → `get_diffs_sparse` → `reconstruct_haplotypes_from_sparse`) and the tracks path (`__getitem__` → `_reconstruct` → `get_diffs_sparse` → `shift_and_realign_tracks_sparse` → `intervals_to_tracks`). Use `cProfile` on `chr22_geuv` (haplotypes + tracks modes, `NUMBA_NUM_THREADS=1`) per the Phase 0 `profile.py` to confirm the coercion hotspots.

- [ ] **Step 2: Decide the fusion seam** per path — the minimal single ffi entry that takes the already-available arrays once and returns the final ragged buffers, dropping intermediate Python coercions. Document the chosen signatures.

- [ ] **Step 3: Commit** the audit doc — `docs(phase-3): getitem glue audit for haps/tracks fusion`.

### Task 13: Fused haplotypes `__getitem__` kernel

**Files:**
- Modify: `src/reconstruct/mod.rs` (or new `src/reconstruct/fused.rs`), `src/ffi/mod.rs`, `src/lib.rs`
- Modify: `python/genvarloader/_dataset/_haps.py` (call the fused entry on the default path)
- Modify: `tests/parity/test_dataset_parity.py`

**Interfaces:**
- Produces: a fused ffi entry (e.g. `reconstruct_haps_fused`) that computes diffs → out_offsets → reconstruction in one crossing from the raw genotype/variant/reference arrays, returning `(out_data, out_offsets)` (and optional annotation buffers) without Python-side coercions between sub-steps.

- [ ] **Step 1: Write a dataset-level parity test FIRST** — assert the fused-path `ds[:, :]` haplotype output is byte-identical to the current composed path under `GVL_BACKEND=numba` (the numba composed pipeline remains the oracle). This is the gate.

- [ ] **Step 2: Run, verify FAIL** (fused entry not yet implemented / not wired).

- [ ] **Step 3: Implement** the fused entry reusing the Task 4/5 cores (call `get_diffs_sparse` core + `reconstruct_haplotypes_from_sparse` core internally; allocate `out` from computed offsets in Rust). No new algorithm — pure plumbing of existing cores.

- [ ] **Step 4: Wire** `_haps._reconstruct_haplotypes` (non-splice default path) to call the fused entry; keep the unfused dispatched kernels for the splice path and as the numba oracle.

- [ ] **Step 5: Build + run dataset parity** both backends; verify PASS + spy confirms the fused entry ran.

- [ ] **Step 6: Record throughput** — re-run `profile.py --mode haps` on `chr22_geuv`, capture batch/s + peak RSS, confirm via cProfile the `np.ascontiguousarray` glue is gone from the fused path. Note the numbers for the roadmap (Task 15).

- [ ] **Step 7: Commit** — `perf(reconstruct): fused haplotypes __getitem__ kernel (dataset parity; throughput recorded)`.

### Task 14: Fused tracks `__getitem__` kernel

**Files:**
- Modify: `src/tracks/mod.rs` (or `src/tracks/fused.rs`), `src/ffi/mod.rs`, `src/lib.rs`
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (tracks path)
- Modify: `tests/parity/test_dataset_parity.py`

**Interfaces:**
- Produces: a fused ffi entry chaining `get_diffs_sparse` → `shift_and_realign_tracks_sparse` → `intervals_to_tracks` cores in one crossing, returning the final realigned ragged tracks buffer + offsets.

- [ ] **Step 1: Dataset-level parity test FIRST** — fused tracks `ds[:, :]` byte-identical to the composed numba pipeline, across fill strategies. Verify FAIL.

- [ ] **Step 2: Implement** the fused entry from the existing cores (plumbing only).

- [ ] **Step 3: Wire** the tracks default path to the fused entry.

- [ ] **Step 4: Build + dataset parity** both backends; spy confirms fused entry ran. PASS.

- [ ] **Step 5: Record throughput** — `profile.py --mode tracks` on `chr22_geuv`; capture batch/s + peak RSS.

- [ ] **Step 6: Commit** — `perf(tracks): fused tracks __getitem__ kernel (dataset parity; throughput recorded)`.

---

# Phase close-out

### Task 15: Full-tree verification, roadmap update, skill check

**Files:**
- Modify: `docs/roadmaps/rust-migration.md`
- Modify (if public API changed): `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Full tree, both backends.** Run, all green:
```bash
pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp
pixi run -e dev cargo-test
```
Expected: PASS (rust default) and PASS (numba forced); cargo green.

- [ ] **Step 2: Lint + types + build.**
```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
pixi run -e dev maturin build   # confirm abi3 wheel builds
```
Expected: clean.

- [ ] **Step 3: Update the roadmap** (`docs/roadmaps/rust-migration.md`):
  - Fix the stale Phase 3 `Gate:` line → "parity hard-gate; throughput recorded only".
  - Tick all Phase 3 checkboxes; set the phase marker ⬜→✅ + the bundled PR link.
  - Record the fused haplotypes + tracks throughput / peak RSS (Tasks 13–14) in a Phase 3 measurement block.
  - Add a Notes & decisions log entry mirroring the Phase 2 entry (kernels ported, fusion seams, any Interpolate-fallback decision, env notes).

- [ ] **Step 4: Skill check.** Phase 3 is internal (no public API change expected). Confirm `python/genvarloader/__init__.py:__all__`, `gvl.write`, `Dataset.open`, and `Dataset.with_*` signatures/defaults are unchanged; if anything public shifted, update `skills/genvarloader/SKILL.md` per CLAUDE.md. State the result explicitly.

- [ ] **Step 5: Commit + open the bundled PR** into `rust-migration`.
```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): Phase 3 complete — reconstruction+tracks ported, fused paths, throughput recorded"
rtk git push -u origin phase-3-reconstruction
rtk gh pr create --base rust-migration --title "Phase 3: reconstruction + track realignment (Rust)" --body "..."
```

---

## Self-review notes (author)

- **Spec coverage:** 3a reference (Tasks 1–3), 3b reconstruction incl. annotated (Tasks 4–6), 3c tracks realign + 4 fill strategies + RLE (Tasks 7–11), 3d fuse both haplotypes+tracks (Tasks 12–14), parity-hard/throughput-recorded gate + roadmap fix (Task 15). All spec sections mapped.
- **Parity risks** (FlankSample PRNG, Interpolate float) are isolated to their own tasks (7, 8/9) with direct guards + a documented numba fallback for Interpolate only.
- **Type consistency:** offsets normalized via `_as_starts_stops` everywhere; `i64`-accumulate-truncate for length sums; `u64` wrapping for PRNG; f64 interpolation stored to f32; annotation leading-pad ref_pos `-1` vs trailing-pad `i32::MAX` called out explicitly.
- **njit-internal leaves** (`padded_slice`, `_get_reference_row`, `xorshift64`, `hash4`, `apply_insertion_fill`, `scanned_mask`, `compact_mask`) get **no** dispatch registration — they land inside their entry kernel's task and are covered through it, per the Phase 0 dispatch rule.
