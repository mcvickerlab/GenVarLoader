//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use ndarray::{Array1, Array2};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::ops::Range;

use crate::variants::windows::{assemble_variants_mode, assemble_windows_mode, VariantBufs};

pub(crate) mod stream_engine;

use crate::genotypes;
use crate::intervals;
use crate::reference;
use crate::variants;

/// Allocate an output buffer of `len` elements WITHOUT zero-initialization.
///
/// SAFETY/INVARIANT: every element is fully overwritten by the reconstruct/track
/// core before it is read. For in-contract inputs the core writes every output
/// position; out-of-contract inputs (e.g. a deletion driving `ref_idx` past the
/// contig end) are already undefined and excluded from the parity oracle by the
/// overshoot/double-init guards in
/// tests/parity/test_reconstruct_haplotypes_parity.py, so skipping the zero-init
/// adds no new observable exposure. `T` is a plain numeric type (u8/i32/f32) with
/// no invalid bit patterns.
#[allow(clippy::uninit_vec)]
fn uninit_output<T: Copy>(len: usize) -> Array1<T> {
    let mut v: Vec<T> = Vec::with_capacity(len);
    // SAFETY: see function-level invariant — every element is written before read.
    unsafe {
        v.set_len(len);
    }
    Array1::from_vec(v)
}

/// Marshal a `(n, 2)` int64 array of `[start, end)` pairs into `Range<usize>`s.
fn arr2_to_ranges(a: numpy::ndarray::ArrayView2<i64>) -> Vec<Range<usize>> {
    a.rows()
        .into_iter()
        .map(|r| (r[0] as usize)..(r[1] as usize))
        .collect()
}

/// Validate that a Python-supplied 1-D array is C-contiguous, or return a Python
/// `ValueError`. Some SVAR2 readbound kernels below slice these arrays and then
/// call `.as_slice()`/`.as_slice_mut()` on the result; a non-contiguous view (e.g.
/// a strided `a[::2]` slice) makes that call panic — a Rust panic surfaces to
/// Python as an uncatchable `pyo3_runtime.PanicException`, not a normal exception
/// — instead of raising. Validate at the FFI boundary so a bad input surfaces as a
/// clean `ValueError`. Only apply this to arrays actually consumed via
/// `.as_slice()` on a stride-preserving view downstream; arrays consumed via
/// `.as_array()` + direct ndarray indexing, `arr2_to_ranges`, or `.to_vec()` are
/// already stride-safe and must NOT be gated here (doing so would reject
/// currently-valid strided inputs).
fn require_contiguous_1d<T: numpy::Element>(
    arr: &PyReadonlyArray1<T>,
    name: &str,
) -> PyResult<()> {
    arr.as_slice().map(|_| ()).map_err(|_| {
        pyo3::exceptions::PyValueError::new_err(format!("`{name}` must be C-contiguous"))
    })
}

/// Mutable-array counterpart of [`require_contiguous_1d`] — same rationale, for a
/// `PyReadwriteArray1` that is consumed via `.as_slice_mut()` downstream. Kept as a
/// separate function (rather than a generic over read/write) because `numpy`'s
/// `PyReadonlyArray1`/`PyReadwriteArray1` don't share a trait for `.as_slice()`.
fn require_contiguous_1d_mut<T: numpy::Element>(
    arr: &mut PyReadwriteArray1<T>,
    name: &str,
) -> PyResult<()> {
    arr.as_array_mut().as_slice_mut().map(|_| ()).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("`{name}` must be C-contiguous"))
    })
}

/// Validate a caller-supplied `out_bounds` (scatter-write destinations) before it is
/// handed to `reconstruct::reconstruct_haplotypes_from_svar2`, whose contract
/// requires rows to be in-range, pairwise-disjoint `(start, end)` byte ranges within
/// `out`. That kernel does not check this itself (its serial path does
/// `std::slice::from_raw_parts_mut(out_raw.add(out_s), out_e - out_s)` with no bound
/// against `out.len()`, and its parallel `split_at_mut` chain only `debug_assert!`s
/// disjointness) — this is the ONLY read-bound FFI entry whose `out_bounds` values
/// come straight from Python rather than being computed by Rust itself
/// (`reconstruct::bounds_from_offsets` over a kernel-sized offsets array), so unlike
/// every sibling entry they cannot be trusted by construction. An out-of-range row is
/// a silent out-of-bounds write; an overlapping row is, under the parallel path with
/// the GIL released (`py.detach`), a silent aliasing race — both genuine UB, not a
/// clean error, if left unchecked.
///
/// Checks, in order:
/// 1. every row satisfies `0 <= start <= end <= out_len` (cheap, O(n));
/// 2. rows are pairwise disjoint, via a sort-by-start + running-max-end sweep
///    (O(n log n) — tens of microseconds at the ~thousands-of-rows scale this path
///    runs at, negligible next to the reconstruct work it guards).
///
/// Pure-Rust core (no `pyo3` error type) so it is unit-testable without a GIL /
/// initialized interpreter; the `#[pyfunction]` call site maps `Err(String)` to a
/// `PyValueError`.
fn check_disjoint_bounds_within(
    out_bounds: numpy::ndarray::ArrayView2<i64>,
    out_len: usize,
) -> Result<(), String> {
    let out_len = out_len as i64;
    for k in 0..out_bounds.nrows() {
        let s = out_bounds[[k, 0]];
        let e = out_bounds[[k, 1]];
        if s < 0 || s > e || e > out_len {
            return Err(format!(
                "out_bounds[{k}] = ({s}, {e}) is invalid for out.len() = {out_len}; every row must satisfy 0 <= start <= end <= out.len()"
            ));
        }
    }

    // Tie-break on `(start, end)`, not `start` alone — see the matching comment on
    // the carve's `order.sort_unstable_by_key` in `reconstruct::reconstruct_haplotypes_from_svar2`
    // for why (a zero-length row shares its start with the following row in a
    // gap-free layout) and why the two sweeps' key ordering must stay identical.
    let mut order: Vec<usize> = (0..out_bounds.nrows()).collect();
    order.sort_unstable_by_key(|&k| (out_bounds[[k, 0]], out_bounds[[k, 1]]));
    let mut max_end: i64 = i64::MIN;
    let mut max_end_k: usize = usize::MAX;
    for &k in &order {
        let s = out_bounds[[k, 0]];
        let e = out_bounds[[k, 1]];
        if s < max_end {
            return Err(format!(
                "out_bounds rows must be pairwise disjoint: row {k} = ({s}, {e}) overlaps row {max_end_k} which ends at {max_end}"
            ));
        }
        if e > max_end {
            max_end = e;
            max_end_k = k;
        }
    }

    Ok(())
}

/// Per-(query, hap) reference-length diffs (see `genotypes::get_diffs_sparse`).
/// `geno_offsets` is the normalized (2, n) int64 starts/stops array.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn get_diffs_sparse<'py>(
    py: Python<'py>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    geno_offsets: PyReadonlyArray2<i64>,
    ilens: PyReadonlyArray1<i32>,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    q_starts: Option<PyReadonlyArray1<i32>>,
    q_ends: Option<PyReadonlyArray1<i32>>,
    v_starts: Option<PyReadonlyArray1<i32>>,
    parallel: bool,
) -> Bound<'py, PyArray2<i32>> {
    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);
    let geno_offset_idx_a = geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let ilens_a = ilens.as_array();
    let keep_a = keep.as_ref().map(|a| a.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|a| a.as_array());
    let q_starts_a = q_starts.as_ref().map(|a| a.as_array());
    let q_ends_a = q_ends.as_ref().map(|a| a.as_array());
    let v_starts_a = v_starts.as_ref().map(|a| a.as_array());
    let diffs = py.detach(move || {
        genotypes::get_diffs_sparse(
            geno_offset_idx_a,
            geno_v_idxs_a,
            go_starts,
            go_stops,
            ilens_a,
            keep_a,
            keep_offsets_a,
            q_starts_a,
            q_ends_a,
            v_starts_a,
            parallel,
        )
    });
    diffs.into_pyarray(py)
}

/// Paint base-pair-resolution tracks from intervals (writes `out` in place).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn intervals_to_tracks(
    py: Python<'_>,
    offset_idxs: PyReadonlyArray1<i64>,
    starts: PyReadonlyArray1<i32>,
    itv_starts: PyReadonlyArray1<i32>,
    itv_ends: PyReadonlyArray1<i32>,
    itv_values: PyReadonlyArray1<f32>,
    itv_offsets: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray1<f32>,
    out_offsets: PyReadonlyArray1<i64>,
    parallel: bool,
) {
    let offset_idxs_a = offset_idxs.as_array();
    let starts_a = starts.as_array();
    let itv_starts_a = itv_starts.as_array();
    let itv_ends_a = itv_ends.as_array();
    let itv_values_a = itv_values.as_array();
    let itv_offsets_a = itv_offsets.as_array();
    let out_a = out.as_array_mut();
    let out_offsets_a = out_offsets.as_array();
    py.detach(move || {
        intervals::intervals_to_tracks(
            offset_idxs_a,
            starts_a,
            itv_starts_a,
            itv_ends_a,
            itv_values_a,
            itv_offsets_a,
            out_a,
            out_offsets_a,
            parallel,
        );
    });
}

/// Exonic keep-mask (see `genotypes::choose_exonic_variants`). Returns
/// `(keep: bool[n], keep_offsets: i64[n_groups+1])`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn choose_exonic_variants<'py>(
    py: Python<'py>,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    geno_offsets: PyReadonlyArray2<i64>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
) -> (Bound<'py, PyArray1<bool>>, Bound<'py, PyArray1<i64>>) {
    let go = geno_offsets.as_array();
    let (keep, koff) = genotypes::choose_exonic_variants(
        starts.as_array(),
        ends.as_array(),
        geno_offset_idx.as_array(),
        geno_v_idxs.as_array(),
        go.row(0),
        go.row(1),
        v_starts.as_array(),
        ilens.as_array(),
    );
    (keep.into_pyarray(py), koff.into_pyarray(py))
}

/// Per-row i32 gather — variant indices (see `variants::gather_rows_i32`).
#[pyfunction]
pub fn gather_rows_i32<'py>(
    py: Python<'py>,
    geno_offset_idx: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    data: PyReadonlyArray1<i32>,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>) {
    let go = geno_offsets.as_array();
    let (v, off) = variants::gather_rows_i32(
        geno_offset_idx.as_array(),
        go.row(0),
        go.row(1),
        data.as_array(),
    );
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Per-row f32 gather — dosage values (see `variants::gather_rows_f32`).
#[pyfunction]
pub fn gather_rows_f32<'py>(
    py: Python<'py>,
    geno_offset_idx: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    data: PyReadonlyArray1<f32>,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
    let go = geno_offsets.as_array();
    let (v, off) = variants::gather_rows_f32(
        geno_offset_idx.as_array(),
        go.row(0),
        go.row(1),
        data.as_array(),
    );
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Gather allele bytestrings (see `variants::gather_alleles`).
#[pyfunction]
pub fn gather_alleles<'py>(
    py: Python<'py>,
    v_idxs: PyReadonlyArray1<i32>,
    allele_bytes: PyReadonlyArray1<u8>,
    allele_offsets: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>) {
    let (data, seq) = variants::gather_alleles(
        v_idxs.as_array(),
        allele_bytes.as_array(),
        allele_offsets.as_array(),
    );
    (data.into_pyarray(py), seq.into_pyarray(py))
}

/// Compact i32 values under keep mask, rebuilding row offsets
/// (see `variants::compact_keep_i32`).
#[pyfunction]
pub fn compact_keep_i32<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    keep: PyReadonlyArray1<bool>,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>) {
    let (v, off) =
        variants::compact_keep_i32(values.as_array(), row_offsets.as_array(), keep.as_array());
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Compact f32 values under keep mask, rebuilding row offsets
/// (see `variants::compact_keep_f32`).
#[pyfunction]
pub fn compact_keep_f32<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<f32>,
    row_offsets: PyReadonlyArray1<i64>,
    keep: PyReadonlyArray1<bool>,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
    let (v, off) =
        variants::compact_keep_f32(values.as_array(), row_offsets.as_array(), keep.as_array());
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Fill empty rows with one scalar sentinel (i32). Returns `(new_data, new_offsets)`.
/// (see `variants::fill_empty_scalar_i32`).
#[pyfunction]
pub fn fill_empty_scalar_i32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<i32>,
    offsets: PyReadonlyArray1<i64>,
    fill: i32,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>) {
    let (v, off) = variants::fill_empty_scalar_i32(data.as_array(), offsets.as_array(), fill);
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Fill empty rows with one scalar sentinel (f32). Returns `(new_data, new_offsets)`.
/// (see `variants::fill_empty_scalar_f32`).
#[pyfunction]
pub fn fill_empty_scalar_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f32>,
    offsets: PyReadonlyArray1<i64>,
    fill: f32,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
    let (v, off) = variants::fill_empty_scalar_f32(data.as_array(), offsets.as_array(), fill);
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Fill empty rows with `inner` copies of sentinel (i32, fixed-stride).
/// Returns `(new_data, new_offsets)`. (see `variants::fill_empty_fixed_i32`).
#[pyfunction]
pub fn fill_empty_fixed_i32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<i32>,
    offsets: PyReadonlyArray1<i64>,
    inner: i64,
    fill: i32,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>) {
    let (v, off) = variants::fill_empty_fixed_i32(data.as_array(), offsets.as_array(), inner, fill);
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Fill empty rows with `inner` copies of sentinel (f32, fixed-stride).
/// Returns `(new_data, new_offsets)`. (see `variants::fill_empty_fixed_f32`).
#[pyfunction]
pub fn fill_empty_fixed_f32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<f32>,
    offsets: PyReadonlyArray1<i64>,
    inner: i64,
    fill: f32,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
    let (v, off) = variants::fill_empty_fixed_f32(data.as_array(), offsets.as_array(), inner, fill);
    (v.into_pyarray(py), off.into_pyarray(py))
}

/// Two-level dummy-fill for allele bytestrings (uint8).
/// Returns `(new_data, new_var_offsets, new_seq_offsets)`.
/// (see `variants::fill_empty_seq_u8`).
#[pyfunction]
pub fn fill_empty_seq_u8<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<u8>,
    var_offsets: PyReadonlyArray1<i64>,
    seq_offsets: PyReadonlyArray1<i64>,
    dummy: PyReadonlyArray1<u8>,
) -> (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
) {
    let (nd, nvar, nseq) = variants::fill_empty_seq_u8(
        data.as_array(),
        var_offsets.as_array(),
        seq_offsets.as_array(),
        dummy.as_array(),
    );
    (
        nd.into_pyarray(py),
        nvar.into_pyarray(py),
        nseq.into_pyarray(py),
    )
}

/// Two-level dummy-fill for token windows (int32).
/// Returns `(new_data, new_var_offsets, new_seq_offsets)`.
/// (see `variants::fill_empty_seq_i32`).
#[pyfunction]
pub fn fill_empty_seq_i32<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<i32>,
    var_offsets: PyReadonlyArray1<i64>,
    seq_offsets: PyReadonlyArray1<i64>,
    dummy: PyReadonlyArray1<i32>,
) -> (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
) {
    let (nd, nvar, nseq) = variants::fill_empty_seq_i32(
        data.as_array(),
        var_offsets.as_array(),
        seq_offsets.as_array(),
        dummy.as_array(),
    );
    (
        nd.into_pyarray(py),
        nvar.into_pyarray(py),
        nseq.into_pyarray(py),
    )
}

/// Build the `{name: (data, seq_offsets)}` dict from assembled buffers.
fn bufs_to_pydict<'py, Tok: numpy::Element + Copy>(
    py: Python<'py>,
    bufs: VariantBufs<Tok>,
) -> Bound<'py, PyDict> {
    let d = PyDict::new(py);
    for (name, data, off) in bufs.byte_bufs {
        d.set_item(name, (data.into_pyarray(py), off.into_pyarray(py)))
            .unwrap();
    }
    for (name, data, off) in bufs.tok_bufs {
        d.set_item(name, (data.into_pyarray(py), off.into_pyarray(py)))
            .unwrap();
    }
    d
}

/// Monomorphized assembly entry. `Tok` is the token dtype; `mode` selects
/// variants (0) vs windows (1). See module docs in `variants::windows`.
#[allow(clippy::too_many_arguments)]
fn assemble_variant_buffers_impl<'py, Tok: numpy::Element + Copy>(
    py: Python<'py>,
    mode: i64,
    v_idxs: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    alt_global: PyReadonlyArray1<u8>,
    alt_off_global: PyReadonlyArray1<i64>,
    ref_global: Option<PyReadonlyArray1<u8>>,
    ref_off_global: Option<PyReadonlyArray1<i64>>,
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: i64,
    alt_mode: i64,
    flank_len: i64,
    lut: Option<PyReadonlyArray1<Tok>>,
    v_contigs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
) -> Bound<'py, PyDict> {
    let rg = ref_global.as_ref().map(|a| a.as_array());
    let ro = ref_off_global.as_ref().map(|a| a.as_array());
    let lut_v = lut.as_ref().map(|a| a.as_array());
    let bufs = if mode == 0 {
        assemble_variants_mode::<Tok>(
            v_idxs.as_array(),
            row_offsets.as_array(),
            alt_global.as_array(),
            alt_off_global.as_array(),
            if want_ref_bytes { rg } else { None },
            if want_ref_bytes { ro } else { None },
            want_flank,
            flank_len,
            lut_v,
            v_contigs.as_array(),
            v_starts.as_array(),
            ilens.as_array(),
            reference.as_array(),
            ref_offsets.as_array(),
            pad_char,
        )
    } else {
        assemble_windows_mode::<Tok>(
            v_idxs.as_array(),
            row_offsets.as_array(),
            ref_mode,
            alt_mode,
            alt_global.as_array(),
            alt_off_global.as_array(),
            rg,
            ro,
            flank_len,
            lut_v.expect("windows mode requires a token LUT"),
            v_contigs.as_array(),
            v_starts.as_array(),
            ilens.as_array(),
            reference.as_array(),
            ref_offsets.as_array(),
            pad_char,
        )
    };
    bufs_to_pydict(py, bufs)
}

/// u8-token assembly (token_dtype == uint8). See `assemble_variant_buffers_impl`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn assemble_variant_buffers_u8<'py>(
    py: Python<'py>,
    mode: i64,
    v_idxs: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    alt_global: PyReadonlyArray1<u8>,
    alt_off_global: PyReadonlyArray1<i64>,
    ref_global: Option<PyReadonlyArray1<u8>>,
    ref_off_global: Option<PyReadonlyArray1<i64>>,
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: i64,
    alt_mode: i64,
    flank_len: i64,
    lut: Option<PyReadonlyArray1<u8>>,
    v_contigs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
) -> Bound<'py, PyDict> {
    assemble_variant_buffers_impl::<u8>(
        py,
        mode,
        v_idxs,
        row_offsets,
        alt_global,
        alt_off_global,
        ref_global,
        ref_off_global,
        want_ref_bytes,
        want_flank,
        ref_mode,
        alt_mode,
        flank_len,
        lut,
        v_contigs,
        v_starts,
        ilens,
        reference,
        ref_offsets,
        pad_char,
    )
}

/// i32-token assembly (token_dtype == int32). See `assemble_variant_buffers_impl`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn assemble_variant_buffers_i32<'py>(
    py: Python<'py>,
    mode: i64,
    v_idxs: PyReadonlyArray1<i32>,
    row_offsets: PyReadonlyArray1<i64>,
    alt_global: PyReadonlyArray1<u8>,
    alt_off_global: PyReadonlyArray1<i64>,
    ref_global: Option<PyReadonlyArray1<u8>>,
    ref_off_global: Option<PyReadonlyArray1<i64>>,
    want_ref_bytes: bool,
    want_flank: bool,
    ref_mode: i64,
    alt_mode: i64,
    flank_len: i64,
    lut: Option<PyReadonlyArray1<i32>>,
    v_contigs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
) -> Bound<'py, PyDict> {
    assemble_variant_buffers_impl::<i32>(
        py,
        mode,
        v_idxs,
        row_offsets,
        alt_global,
        alt_off_global,
        ref_global,
        ref_off_global,
        want_ref_bytes,
        want_flank,
        ref_mode,
        alt_mode,
        flank_len,
        lut,
        v_contigs,
        v_starts,
        ilens,
        reference,
        ref_offsets,
        pad_char,
    )
}

/// Reconstruct haplotypes for a batch of (query, hap) pairs in place (writes `out`).
///
/// `geno_offsets` is the normalized (2, n) int64 starts/stops array.
/// `keep_offsets` is the 1-D (batch*ploidy + 1) offsets array for the keep mask, or None.
/// `parallel` enables rayon batch parallelism (caller computes `should_parallelize`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_sparse(
    py: Python<'_>,
    mut out: PyReadwriteArray1<u8>,
    out_offsets: PyReadonlyArray1<i64>,
    regions: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    mut annot_v_idxs: Option<PyReadwriteArray1<i32>>,
    mut annot_ref_pos: Option<PyReadwriteArray1<i32>>,
    parallel: bool,
) {
    use crate::reconstruct;
    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);
    let out_a = out.as_array_mut();
    let out_offsets_a = out_offsets.as_array();
    let regions_a = regions.as_array();
    let shifts_a = shifts.as_array();
    let geno_offset_idx_a = geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let keep_a = keep.as_ref().map(|k| k.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|ko| ko.as_array());
    let annot_v_idxs_a = annot_v_idxs.as_mut().map(|a| a.as_array_mut());
    let annot_ref_pos_a = annot_ref_pos.as_mut().map(|a| a.as_array_mut());
    py.detach(move || {
        reconstruct::reconstruct_haplotypes_from_sparse(
            out_a,
            out_offsets_a,
            regions_a,
            shifts_a,
            geno_offset_idx_a,
            go_starts,
            go_stops,
            geno_v_idxs_a,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            keep_a,
            keep_offsets_a,
            annot_v_idxs_a,
            annot_ref_pos_a,
            parallel,
        );
    });
}

/// Fused haplotypes __getitem__ kernel.
///
/// Collapses two FFI crossings into one:
///   1. Compute per-haplotype length diffs (``get_diffs_sparse`` logic).
///   2. Allocate the output buffer and offset array in Rust from the computed diffs.
///   3. Run ``reconstruct_haplotypes_from_sparse`` logic.
///   4. Return ``(out_data: Array1<u8>, out_offsets: Array1<i64>)`` — ready for
///      wrapping into ``_Flat.from_offsets(...).view("S1")`` with no further coercions.
///
/// ``output_length``:
///   - ``-1`` → ragged mode (each haplotype gets its natural length = ref_len + diff).
///   - ``>= 0`` → fixed-length mode (every haplotype is padded/truncated to this length).
///
/// ``geno_offsets`` is the normalized ``(2, n)`` int64 starts/stops array (same
/// layout as the existing ``reconstruct_haplotypes_from_sparse`` FFI entry).
///
/// Annotation buffers are not supported in the fused entry (annotated path
/// remains on the unfused dispatch wrappers, which carry the extra
/// `annot_*` output buffers this fused entry does not allocate).
/// `parallel` enables rayon batch parallelism (caller computes `should_parallelize`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_fused<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    output_length: i64,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    to_rc: Option<PyReadonlyArray1<bool>>,
    parallel: bool,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>) {
    use crate::genotypes;
    use crate::reconstruct;

    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);

    let regions_a = regions.as_array();
    let shifts_a = shifts.as_array();
    let geno_offset_idx_a = geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let keep_a = keep.as_ref().map(|a| a.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|a| a.as_array());
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());

    let (batch_size, ploidy) = geno_offset_idx_a.dim();
    let n_work = batch_size * ploidy;

    let (out_data, out_offsets_vec) = py.detach(move || {
        // Step 1: compute per-haplotype length diffs (reuses get_diffs_sparse core).
        // Mirrors _haps.py _haplotype_ilens exactly: pass q_starts/q_ends/v_starts so
        // partial deletions that span a query boundary are correctly clipped.
        // q_starts = regions[:, 1], q_ends = regions[:, 2] (both already in regions_a).
        // v_starts is the same array passed in — it is the per-variant genomic start.
        let q_starts_owned: ndarray::Array1<i32> = regions_a.column(1).to_owned();
        let q_ends_owned: ndarray::Array1<i32> = regions_a.column(2).to_owned();
        let diffs = genotypes::get_diffs_sparse(
            geno_offset_idx_a,
            geno_v_idxs_a,
            go_starts,
            go_stops,
            ilens_a,
            keep_a,
            keep_offsets_a,
            Some(q_starts_owned.view()), // q_starts = regions[:, 1]
            Some(q_ends_owned.view()),   // q_ends   = regions[:, 2]
            Some(v_starts_a),            // v_starts = per-variant genomic starts
            parallel,
        );

        // Step 2: compute per-haplotype output lengths and prefix-sum offsets.
        // Mirrors the Python side: out_lengths = hap_lengths (or fixed output_length).
        // hap_lengths = regions[:, 2] - regions[:, 1] + diffs  (end - start + diff)
        // out_offsets shape: (n_work + 1,)
        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            out_offsets_vec[0] = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let len: i64 = if output_length >= 0 {
                    output_length
                } else {
                    let ref_len = (regions_a[[query, 2]] - regions_a[[query, 1]]) as i64;
                    let diff = diffs[[query, hap]] as i64;
                    (ref_len + diff).max(0)
                };
                acc += len;
                out_offsets_vec[k + 1] = acc;
            }
        }

        // Step 3: allocate the output buffer in Rust — Python never calls np.empty.
        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<u8> = uninit_output(total);

        // Step 4: reconstruct all haplotypes into the owned buffer (reuses batch core).
        reconstruct::reconstruct_haplotypes_from_sparse(
            out_data.view_mut(),
            out_offsets_vec.view(),
            regions_a,
            shifts_a,
            geno_offset_idx_a,
            go_starts,
            go_stops,
            geno_v_idxs_a,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            keep_a,
            keep_offsets_a,
            None, // annot_v_idxs — not supported in fused plain path
            None, // annot_ref_pos — not supported in fused plain path
            parallel,
        );

        // Step 4b: optional in-kernel reverse-complement (one bool per (query, hap) work item).
        if let Some(to_rc) = to_rc_a.as_ref() {
            debug_assert_eq!(
                to_rc.len(),
                out_offsets_vec.len() - 1,
                "to_rc mask length must equal number of output rows (offsets.len() - 1)"
            );
            crate::reverse::rc_flat_rows_inplace(
                out_data.as_slice_mut().unwrap(),
                out_offsets_vec.view(),
                *to_rc,
            );
        }

        (out_data, out_offsets_vec)
    });

    // Step 5: return owned arrays — Python wraps them with no further coercions.
    (out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py))
}

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

/// Compute a window's SVAR2 read-bound ranges in Rust (GIL released), replacing the
/// Python `SparseVar2._find_ranges` call. `sample_idx` are PHYSICAL store columns
/// (public sorted-name -> physical translation happens Python-side). Returns flat i64
/// range arrays: vk_snp/vk_indel are `[start, stop, ...]` in (region, sample, ploid)
/// C-order (len n_reg*n_s*P*2); dense_snp/dense_indel per region (len n_reg*2);
/// sample_cols len n_s. No genoray rev bump: `find_ranges` is public at the pinned rev.
#[pyfunction]
pub fn svar2_read_window<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar2::store::Svar2Store>,
    contig: &str,
    starts: PyReadonlyArray1<u32>,
    ends: PyReadonlyArray1<u32>,
    sample_idx: PyReadonlyArray1<i64>,
) -> PyResult<(
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
)> {
    require_contiguous_1d(&starts, "starts")?;
    require_contiguous_1d(&ends, "ends")?;

    let starts_a = starts.as_array();
    let ends_a = ends.as_array();
    let regions_v: Vec<(u32, u32)> = starts_a
        .iter()
        .zip(ends_a.iter())
        .map(|(&s, &e)| (s, e))
        .collect();
    let samples_v: Vec<usize> = sample_idx.as_array().iter().map(|&s| s as usize).collect();

    let store_ref: &crate::svar2::store::Svar2Store = &store;

    let result = py.detach(
        move || -> anyhow::Result<(Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>, Vec<i64>)> {
            let reader = store_ref
                .reader(contig)
                .ok_or_else(|| anyhow::anyhow!("no reader for contig {contig}"))?;
            let rb = genoray_core::query::find_ranges(reader, &regions_v, Some(&samples_v));
            let flat = |v: &[std::ops::Range<usize>]| -> Vec<i64> {
                let mut out = Vec::with_capacity(v.len() * 2);
                for r in v {
                    out.push(r.start as i64);
                    out.push(r.end as i64);
                }
                out
            };
            Ok((
                flat(&rb.vk_snp_range),
                flat(&rb.vk_indel_range),
                flat(&rb.dense_snp_range),
                flat(&rb.dense_indel_range),
                rb.sample_cols.iter().map(|&c| c as i64).collect(),
            ))
        },
    );

    let (vk_snp, vk_indel, dense_snp, dense_indel, sample_cols) =
        result.map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))?;
    Ok((
        Array1::from_vec(vk_snp).into_pyarray(py),
        Array1::from_vec(vk_indel).into_pyarray(py),
        Array1::from_vec(dense_snp).into_pyarray(py),
        Array1::from_vec(dense_indel).into_pyarray(py),
        Array1::from_vec(sample_cols).into_pyarray(py),
    ))
}

/// Read-through prefetch: fault the EXACT pages a later `generate_batch` call will
/// read, warming the shared OS page cache (and triggering kernel readahead of
/// subsequent pages that overlaps with the caller's own CPU work). `o_starts`/
/// `o_stops` are ABSOLUTE indices into `vidx` (the store's `variant_idxs` mmap,
/// zero-copy — see `Svar1Store::geno_v_idxs`'s contract). `black_box` stops the
/// compiler eliding the fold. Shared by the streaming engine's producer
/// (`stream_engine.rs`) AND the standalone [`svar1_prefetch_runs`] FFI entry (Design
/// C's single-thread read-ahead drive, issue #283) — ONE implementation (DRY).
pub(crate) fn prefetch_runs_core(vidx: &[i32], o_starts: &[i64], o_stops: &[i64]) {
    for (&lo, &hi) in o_starts.iter().zip(o_stops.iter()) {
        let (lo, hi) = (lo as usize, hi as usize);
        let _ = std::hint::black_box(vidx[lo..hi].iter().fold(0i64, |a, &x| a ^ x as i64));
    }
}

/// Standalone read-through prefetch FFI (Design C, issue #283): the SAME read-through
/// the engine's producer does (`prefetch_runs_core`, shared), exposed directly so a
/// single-thread Python drive can prefetch the NEXT window's runs before generating
/// the CURRENT one, overlapping kernel readahead with CPU-side generation without a
/// background thread. `o_starts`/`o_stops` come from `svar1_read_window`. Runs inside
/// `py.detach`; `store` is `PyRef<'py>`. No return value — this only warms pages.
#[pyfunction]
pub fn svar1_prefetch_runs<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar1::store::Svar1Store>,
    o_starts: PyReadonlyArray1<i64>,
    o_stops: PyReadonlyArray1<i64>,
) -> PyResult<()> {
    require_contiguous_1d(&o_starts, "o_starts")?;
    require_contiguous_1d(&o_stops, "o_stops")?;

    let o_starts_a = o_starts.as_array();
    let o_stops_a = o_stops.as_array();
    let o_starts_s: &[i64] = o_starts_a
        .as_slice()
        .expect("contiguity checked by require_contiguous_1d above");
    let o_stops_s: &[i64] = o_stops_a
        .as_slice()
        .expect("contiguity checked by require_contiguous_1d above");

    let store_ref: &crate::svar1::store::Svar1Store = &store;

    py.detach(move || {
        let vidx = store_ref.geno_v_idxs();
        prefetch_runs_core(vidx, o_starts_s, o_stops_s);
    });

    Ok(())
}

/// GIL-free core that generates haplotypes for ONE batch of window rows. Shared by the
/// [`svar1_generate_batch`] FFI entry AND the streaming engine's consumer
/// ([`crate::ffi::stream_engine::Svar1StreamEngine`]) so there is ONE implementation
/// (DRY). Both call it inside a GIL-released context — the FFI wraps it in `py.detach`,
/// the engine calls it from `next_batch_core` (itself run under `py.detach`).
///
/// `o_starts_b`/`o_stops_b` are the CSR-row offsets for exactly this batch (length
/// `n_rows * ploidy`, ABSOLUTE indices into `variant_idxs`); `region_bounds_b` is
/// `(n_rows, 2)`, already expanded per (region, sample). Output is `n_rows`-bounded —
/// the #284 fix. Sparse input IS the store's `variant_idxs` mmap (zero copy). `ref_` /
/// `ref_offsets` are the ACTIVE contig's slice (`ref_offsets = [0, contig_len]`), since
/// the per-row `regions` this builds carry `contig_idx = 0`. Ragged output only,
/// jitter=0.
///
/// This function does not touch Python and takes no GIL token; callers own the
/// `into_pyarray` marshaling. It is infallible (all inputs are pre-validated by the
/// caller / the store contract), so it returns the owned arrays directly.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate_batch_core(
    store: &crate::svar1::store::Svar1Store,
    o_starts_b: &[i64],
    o_stops_b: &[i64],
    region_bounds_b: ndarray::ArrayView2<i32>,
    v_starts: ndarray::ArrayView1<i32>,
    ilens: ndarray::ArrayView1<i32>,
    alt_alleles: ndarray::ArrayView1<u8>,
    alt_offsets: ndarray::ArrayView1<i64>,
    ref_: ndarray::ArrayView1<u8>,
    ref_offsets: ndarray::ArrayView1<i64>,
    pad_char: u8,
    parallel: bool,
) -> (Array1<u8>, Array1<i64>) {
    use crate::genotypes;
    use crate::reconstruct;

    let batch = region_bounds_b.nrows();
    let ploidy = store.ploidy();
    let n_work = batch * ploidy;

    // Per-row region bounds (already (region, sample)-expanded by the caller).
    let mut regions_arr = Array2::<i32>::zeros((batch, 3));
    for bi in 0..batch {
        regions_arr[[bi, 1]] = region_bounds_b[[bi, 0]];
        regions_arr[[bi, 2]] = region_bounds_b[[bi, 1]];
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

    let o_starts_a = ndarray::ArrayView1::from(o_starts_b);
    let o_stops_a = ndarray::ArrayView1::from(o_stops_b);

    // ZERO COPY: kernel sparse input IS the store's mmap (see geno_v_idxs contract).
    let geno_v_idxs = store.geno_v_idxs();
    let geno_v_idxs_view = numpy::ndarray::ArrayView1::from(geno_v_idxs);

    let q_starts_owned: Array1<i32> = regions_arr.column(1).to_owned();
    let q_ends_owned: Array1<i32> = regions_arr.column(2).to_owned();
    let diffs = genotypes::get_diffs_sparse(
        geno_offset_idx.view(),
        geno_v_idxs_view,
        o_starts_a,
        o_stops_a,
        ilens,
        None,
        None,
        Some(q_starts_owned.view()),
        Some(q_ends_owned.view()),
        Some(v_starts),
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
        v_starts,
        ilens,
        alt_alleles,
        alt_offsets,
        ref_,
        ref_offsets,
        pad_char,
        None, // keep
        None, // keep_offsets
        None, // annot_v_idxs
        None, // annot_ref_pos
        parallel,
    );

    (out_data, out_offsets_vec)
}

/// Generate haplotypes for ONE batch of window rows. `o_starts_b`/`o_stops_b` are the
/// CSR-row offsets for exactly this batch (length `n_rows * ploidy`, ABSOLUTE indices
/// into `variant_idxs`); `region_bounds_b` is `(n_rows, 2)`, already expanded per
/// (region, sample). Output is `n_rows`-bounded — the #284 fix. `geno_v_idxs` is the
/// shared `variant_idxs` mmap (zero copy). Ragged output only, jitter=0.
///
/// Thin FFI wrapper over [`generate_batch_core`] (which the streaming engine also uses).
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
    require_contiguous_1d(&ref_, "ref_")?;
    require_contiguous_1d(&o_starts_b, "o_starts_b")?;
    require_contiguous_1d(&o_stops_b, "o_stops_b")?;

    let o_starts_arr = o_starts_b.as_array();
    let o_stops_arr = o_stops_b.as_array();
    let o_starts_s: &[i64] = o_starts_arr
        .as_slice()
        .expect("contiguity checked by require_contiguous_1d above");
    let o_stops_s: &[i64] = o_stops_arr
        .as_slice()
        .expect("contiguity checked by require_contiguous_1d above");
    let rb = region_bounds_b.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();

    let store_ref: &crate::svar1::store::Svar1Store = &store;

    let (out_data, out_offsets_vec) = py.detach(move || {
        generate_batch_core(
            store_ref,
            o_starts_s,
            o_stops_s,
            rb,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            parallel,
        )
    });
    Ok((out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py)))
}

/// Test hook: CSR entries spanned by SVAR1 window reads on this thread. See
/// `crate::svar1::store::csr_entries_touched`.
#[pyfunction]
pub fn svar1_csr_entries_touched() -> usize {
    crate::svar1::store::csr_entries_touched()
}

/// Fused SVAR2 two-source haplotype reconstruction: merge each hap's `var_key` ⋈
/// `dense` channels and decode via `svar2-codec` inline (no materialized global
/// variant table), sizing and allocating the output buffer in Rust — one FFI
/// crossing, mirrors `reconstruct_haplotypes_fused` above.
///
/// `output_length`:
///   - ``-1`` → ragged mode (each haplotype gets its natural length = ref_len + diff).
///   - ``>= 0`` → fixed-length mode (every haplotype is padded/truncated to this length).
///
/// No annotation, no to_rc; mirrors the plain fused path.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_svar2<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    vk_pos: PyReadonlyArray1<i32>,
    vk_key: PyReadonlyArray1<i32>,
    vk_off: PyReadonlyArray1<i64>,
    dense_pos: PyReadonlyArray1<i32>,
    dense_key: PyReadonlyArray1<i32>,
    dense_range: PyReadonlyArray2<i32>,
    dense_present: PyReadonlyArray1<u8>,
    dense_present_off: PyReadonlyArray1<i64>,
    lut_bytes: PyReadonlyArray1<u8>,
    lut_off: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    output_length: i64,
    parallel: bool,
) -> (Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>) {
    use crate::reconstruct;
    use crate::svar2;

    let regions_a = regions.as_array();
    let shifts_a = shifts.as_array();
    let vk_pos_a = vk_pos.as_array();
    let vk_key_a = vk_key.as_array();
    let vk_off_a = vk_off.as_array();
    let dense_pos_a = dense_pos.as_array();
    let dense_key_a = dense_key.as_array();
    let dense_range_a = dense_range.as_array();
    let dense_present_a = dense_present.as_array();
    let dense_present_off_a = dense_present_off.as_array();
    let lut_bytes_a = lut_bytes.as_array();
    let lut_off_a = lut_off.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();

    let ploidy = shifts_a.ncols();
    let n_q = regions_a.nrows();
    let n_work = n_q * ploidy;

    let (out_data, out_offsets_vec) = py.detach(move || {
        // Step 1: compute per-haplotype length diffs via the two-source diff core.
        let vk_pos_s: &[i32] = vk_pos_a.as_slice().unwrap();
        let vk_key_s: &[i32] = vk_key_a.as_slice().unwrap();
        let vk_off_s: &[i64] = vk_off_a.as_slice().unwrap();
        let dense_pos_s: &[i32] = dense_pos_a.as_slice().unwrap();
        let dense_key_s: &[i32] = dense_key_a.as_slice().unwrap();
        let dense_present_s: &[u8] = dense_present_a.as_slice().unwrap();
        let dense_present_off_s: &[i64] = dense_present_off_a.as_slice().unwrap();
        let lut_bytes_s: &[u8] = lut_bytes_a.as_slice().unwrap();
        let lut_off_s: &[i64] = lut_off_a.as_slice().unwrap();

        let diffs = svar2::hap_diffs_svar2(
            regions_a,
            ploidy,
            vk_pos_s,
            vk_key_s,
            vk_off_s,
            dense_pos_s,
            dense_key_s,
            dense_range_a,
            dense_present_s,
            dense_present_off_s,
            lut_bytes_s,
            lut_off_s,
            false,
        );

        // Step 2: compute per-haplotype output lengths and prefix-sum offsets.
        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            out_offsets_vec[0] = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let len: i64 = if output_length >= 0 {
                    output_length
                } else {
                    let ref_len = (regions_a[[query, 2]] - regions_a[[query, 1]]) as i64;
                    let diff = diffs[[query, hap]] as i64;
                    (ref_len + diff).max(0)
                };
                acc += len;
                out_offsets_vec[k + 1] = acc;
            }
        }

        // Step 3: allocate the output buffer in Rust — Python never calls np.empty.
        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<u8> = uninit_output(total);

        // Step 4: reconstruct all haplotypes into the owned buffer.
        let out_bounds = reconstruct::bounds_from_offsets(out_offsets_vec.view());
        reconstruct::reconstruct_haplotypes_from_svar2(
            out_data.view_mut(),
            out_bounds.view(),
            regions_a,
            shifts_a,
            vk_pos_a,
            vk_key_a,
            vk_off_a,
            dense_pos_a,
            dense_key_a,
            dense_range_a,
            dense_present_a,
            dense_present_off_a,
            lut_bytes_a,
            lut_off_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            parallel,
            false,
        );

        (out_data, out_offsets_vec)
    });

    (out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py))
}

/// Read-bound SVAR2 haplotype reconstruction: gather off a query-only genoray
/// `Svar2Store` reader with NO interval-search-tree rebuild and NO dense-union
/// rebuild (`genoray_core::query::gather_haps_readbound`), marshal the split
/// result into the flat layout via [`crate::svar2::split_to_flat`], then reuse
/// the byte-validated [`reconstruct_haplotypes_from_svar2`] kernel unchanged —
/// one FFI crossing, byte-identical to the union-path oracle.
///
/// `region_starts`/`orig_samples`/`vk_snp_range`/`vk_indel_range`/
/// `dense_snp_range`/`dense_indel_range` are the per-query outputs of
/// `SparseVar2.find_ranges` (flattened region-major, sample-minor); see
/// `python/genvarloader/_dataset/_svar2_store_py.py::build_readbound_haps`.
#[pyfunction(signature = (
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
    output_length,
    parallel,
    filter_exonic = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_svar2_readbound<'py>(
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
    output_length: i64,
    parallel: bool,
    filter_exonic: bool,
) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
    let reader = store.reader(contig).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("contig {contig} not in store"))
    })?;

    let shifts_a = shifts.as_array();
    let region_bounds_a = region_bounds.as_array();
    let n_q = region_bounds_a.nrows();

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

    // `ref_` is sliced (`ref_.slice(s![c_s..c_e])`) and then `.as_slice().unwrap()`'d
    // inside `reconstruct::reconstruct_haplotypes_from_svar2` (src/reconstruct/mod.rs) —
    // a non-contiguous `ref_` (e.g. `ref_[::2]`) panics there. `ref_offsets` is only
    // ever indexed directly (`ref_offsets[c_idx]`), which is stride-safe, so it is not
    // gated here.
    require_contiguous_1d(&ref_, "ref_")?;

    // NOTE: the pure-DEL anchor-base read inside
    // `reconstruct::reconstruct_haplotypes_from_svar2`
    // (`&contig_ref_s[pos as usize..pos as usize + 1]`, src/reconstruct/mod.rs) is
    // in-bounds for all valid input: gathered variants come from within-contig records
    // so `pos < contig_ref_len` always holds. It is NOT bounded here by the query
    // window: gvl regions legitimately extend past the contig end (jitter / max_jitter
    // padding, right-padded with `pad_char`), so `region_bounds[q, 1] > contig_ref_len`
    // is a normal, valid read — rejecting it would break SVAR1 parity. The only way to
    // reach the anchor OOB is a corrupt store (a variant `pos >= contig_len`); that is
    // caught by a `debug_assert!` at the read site (fires in test/debug builds).

    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();

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
}

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

    // `out` is consumed via `.as_slice_mut()` in the kernel's parallel carve
    // (`reconstruct::reconstruct_haplotypes_from_svar2`) and in the RC pass
    // (`crate::reverse::rc_bounded_rows_inplace`) below; a non-contiguous view (e.g.
    // a strided `out[::2]`) would panic there instead of raising. Gate it here per
    // this file's stated `.as_slice()`-consumer policy (see `require_contiguous_1d`).
    require_contiguous_1d_mut(&mut out, "out")?;

    // `out_bounds` values come straight from Python (unlike every sibling entry's
    // Rust-computed offsets) — validate range + disjointness before `py.detach`, or
    // a bad row is a silent OOB write / aliasing race instead of a clean error.
    // See `check_disjoint_bounds_within` for the full rationale.
    check_disjoint_bounds_within(out_bounds.as_array(), out.as_array().len())
        .map_err(pyo3::exceptions::PyValueError::new_err)?;

    if let Some(to_rc) = to_rc.as_ref() {
        if to_rc.as_array().len() != n_q * ploidy {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "to_rc must have n_q*ploidy = {} rows, got {}",
                n_q * ploidy,
                to_rc.as_array().len()
            )));
        }
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

    // NOTE: same reasoning as the allocating entry's NOTE above the pure-DEL anchor
    // read applies unchanged here — this entry calls the identical
    // `reconstruct::reconstruct_haplotypes_from_svar2` kernel, fed by the same
    // `gather_haps_readbound(reader, &rb)` gather off the same `store`/`contig`, so
    // gathered variants are the same within-contig records (`pos < contig_ref_len`
    // always holds) and `region_bounds` carries the same jitter-past-contig-end
    // semantics. The anchor-base read is in-bounds for all valid input; a corrupt
    // store is caught by the same `debug_assert!` at the read site.

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

/// Read-bound SVAR2 per-hap ilen diffs: the same gather
/// (`genoray_core::query::gather_haps_readbound` + [`crate::svar2::split_to_flat`]) as
/// [`reconstruct_haplotypes_from_svar2_readbound`], but stops after
/// [`crate::svar2::hap_diffs_svar2`] and returns just the `(n_q, ploidy)` diffs —
/// no reconstruct sizing/allocation/kernel pass. Used by the dataset read path to
/// compute random jitter shifts from diffs BEFORE reconstructing (mirrors how the
/// SVAR1 path derives shifts from diffs in `_prepare_request`).
///
/// See [`reconstruct_haplotypes_from_svar2_readbound`] for the shared
/// `region_starts`/`orig_samples`/`vk_*_range`/`dense_*_range`/`region_bounds`
/// argument semantics (the per-query outputs of `SparseVar2.find_ranges`,
/// flattened region-major, sample-minor); see
/// `python/genvarloader/_dataset/_svar2_store_py.py::build_readbound_diffs`.
#[pyfunction(signature = (
    store,
    contig,
    region_starts,
    orig_samples,
    vk_snp_range,
    vk_indel_range,
    dense_snp_range,
    dense_indel_range,
    region_bounds,
    ploidy,
    filter_exonic = false,
))]
#[allow(clippy::too_many_arguments)]
pub fn hap_diffs_from_svar2_readbound<'py>(
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
    ploidy: usize,
    filter_exonic: bool,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    use crate::svar2;

    let reader = store.reader(contig).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("contig {contig} not in store"))
    })?;

    let region_bounds_a = region_bounds.as_array();
    let n_q = region_bounds_a.nrows();

    // Build `regions` (n_q, 3) as [contig_idx=0, start, end) — matches the
    // reconstruct-readbound FFI's convention.
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

    let diffs = py.detach(move || {
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

        svar2::hap_diffs_svar2(
            regions.view(),
            ploidy,
            &flat.vk_pos,
            &flat.vk_key,
            &flat.vk_off,
            &flat.dense_pos,
            &flat.dense_key,
            dense_range_a,
            &flat.dense_present,
            &flat.dense_present_off,
            &lut_bytes,
            &lut_off,
            filter_exonic,
        )
    });

    Ok(diffs.into_pyarray(py))
}

/// Read-bound SVAR2 track re-alignment: gather off a query-only genoray
/// `Svar2Store` reader with NO interval-search-tree rebuild and NO dense-union
/// rebuild (`genoray_core::query::gather_haps_readbound`), marshal the split
/// result into the flat layout via [`crate::svar2::split_to_flat`], then reuse
/// the byte-validated [`shift_and_realign_tracks_from_svar2`] kernel unchanged —
/// one FFI crossing, byte-identical to the union-path oracle.
///
/// See [`reconstruct_haplotypes_from_svar2_readbound`] for the shared
/// `region_starts`/`orig_samples`/`vk_*_range`/`dense_*_range`/`region_bounds`
/// argument semantics (the per-query outputs of `SparseVar2.find_ranges`,
/// flattened region-major, sample-minor); see
/// `python/genvarloader/_dataset/_svar2_store_py.py::build_readbound_tracks`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn shift_and_realign_tracks_from_svar2_readbound<'py>(
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
    tracks: PyReadonlyArray1<f32>,
    track_offsets: PyReadonlyArray1<i64>,
    params: PyReadonlyArray1<f64>,
    strategy_id: i64,
    base_seed: u64,
    global_query: PyReadonlyArray1<i64>,
    parallel: bool,
) -> PyResult<(Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>)> {
    use crate::svar2;
    use crate::tracks;

    let reader = store.reader(contig).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("contig {contig} not in store"))
    })?;

    let shifts_a = shifts.as_array();
    let ploidy = shifts_a.ncols();
    let region_bounds_a = region_bounds.as_array();
    let n_q = region_bounds_a.nrows();

    // Build `regions` (n_q, 3) as [contig_idx=0, start, end).
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

    // `tracks` is `.as_slice().expect("tracks must be contiguous (C-order)")`'d inside
    // `tracks::shift_and_realign_tracks_from_svar2` (src/tracks/mod.rs) — a
    // non-contiguous `tracks` (e.g. `tracks[::2]`) panics there. `track_offsets` and
    // `params` are only ever indexed directly (`track_offsets[query]`, `params[0]`),
    // which is stride-safe, so neither is gated here.
    require_contiguous_1d(&tracks, "tracks")?;

    let tracks_a = tracks.as_array();
    let track_offsets_a = track_offsets.as_array();
    let params_a = params.as_array();
    // (n_q,) LOCAL query -> GLOBAL batch row map for the FlankSample fill seed;
    // the read-bound path calls this kernel once per contig group, so the local
    // `k / ploidy` index would otherwise diverge from the single fused SVAR1 call
    // (issue #267).
    let global_query_a = global_query.as_array();

    let (out_data, out_offsets_vec) = py.detach(move || {
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

        // Step 1: size via the same two-source diff core the union path uses.
        let diffs = svar2::hap_diffs_svar2(
            regions.view(),
            ploidy,
            &flat.vk_pos,
            &flat.vk_key,
            &flat.vk_off,
            &flat.dense_pos,
            &flat.dense_key,
            dense_range_a,
            &flat.dense_present,
            &flat.dense_present_off,
            &lut_bytes,
            &lut_off,
            false,
        );

        // Step 2: per-haplotype output lengths and prefix-sum offsets — tracks
        // always size to ref_len + diff (no `output_length` override).
        let n_work = n_q * ploidy;
        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            out_offsets_vec[0] = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let ref_len = (regions[[query, 2]] - regions[[query, 1]]) as i64;
                let diff = diffs[[query, hap]] as i64;
                let len: i64 = (ref_len + diff).max(0);
                acc += len;
                out_offsets_vec[k + 1] = acc;
            }
        }

        // Step 3: allocate the output buffer in Rust.
        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<f32> = Array1::<f32>::zeros(total);

        // Step 4: realign — reuse the byte-validated union-path kernel unchanged,
        // now fed the read-bound gather's flat channels.
        tracks::shift_and_realign_tracks_from_svar2(
            out_data.view_mut(),
            out_offsets_vec.view(),
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
            tracks_a,
            track_offsets_a,
            params_a,
            strategy_id,
            base_seed,
            Some(global_query_a),
            parallel,
        );

        (out_data, out_offsets_vec)
    });

    Ok((out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py)))
}

/// Read-bound SVAR2 variants decode: gather off a query-only genoray `Svar2Store`
/// reader with NO interval-search-tree rebuild and NO dense-union rebuild
/// (`genoray_core::query::gather_haps_readbound`), then decode each hap's merged
/// `var_key ⋈ dense` keys via [`crate::svar2::decode_variants_from_split`] — one
/// FFI crossing, mirroring genoray's `decode_hap` (no overlap/clip filter; the
/// gather already restricts to overlapping variants).
///
/// See [`reconstruct_haplotypes_from_svar2_readbound`] for the shared
/// `region_starts`/`orig_samples`/`vk_*_range`/`dense_*_range` argument semantics
/// (the per-query outputs of `SparseVar2.find_ranges`, flattened region-major,
/// sample-minor). `ploidy` is passed explicitly (there is no `shifts` array to
/// infer it from here).
///
/// `fields` is a list of `(category, name, dtype_str)` triples (`category` is
/// `"info"` or `"format"`; `dtype_str` is genoray's `StorageDtype` meta string,
/// e.g. `"i32"`), and may be empty. When non-empty, this opens the four
/// `FieldSub`-keyed `FieldView`s per field and gathers their bytes alongside the
/// variant decode via the var_key-provenance-tracking
/// `genoray_core::query::gather_haps_readbound_src` (plain `gather_haps_readbound`
/// is used when `fields` is empty, since it doesn't need that provenance).
///
/// Returns the `RaggedVariants` SoA `(pos, ilen, alt_bytes, str_off, var_off)`
/// plus, per requested field in `fields` order, a flat `u8` byte buffer and its
/// per-value itemsize (`field_bufs`, `field_itemsizes`); see
/// `python/genvarloader/_dataset/_svar2_store_py.py::build_readbound_variants`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn decode_variants_from_svar2_readbound<'py>(
    py: Python<'py>,
    store: PyRef<'py, crate::svar2::store::Svar2Store>,
    contig: &str,
    region_starts: PyReadonlyArray1<u32>,
    orig_samples: PyReadonlyArray1<i64>,
    vk_snp_range: PyReadonlyArray2<i64>,
    vk_indel_range: PyReadonlyArray2<i64>,
    dense_snp_range: PyReadonlyArray2<i64>,
    dense_indel_range: PyReadonlyArray2<i64>,
    ploidy: usize,
    fields: Vec<(String, String, String)>,
) -> PyResult<(
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i64>>,
    Bound<'py, PyArray1<i64>>,
    Vec<Bound<'py, PyArray1<u8>>>,
    Vec<usize>,
)> {
    use crate::svar2;
    use genoray_core::field::StorageDtype;
    use genoray_core::layout::{ContigPaths, FieldSub};
    use genoray_core::query::FieldView;

    let reader = store.reader(contig).ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("contig {contig} not in store"))
    })?;

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

    let n_samples = reader.n_samples();
    let paths = ContigPaths::new(store.store_path(), contig);

    let gathers: Vec<svar2::FieldGather> = fields
        .iter()
        .map(|(cat, name, dtype_str)| {
            let dtype = StorageDtype::from_meta_str(dtype_str).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "field {name}: unknown storage dtype {dtype_str:?}"
                ))
            })?;
            let width = dtype.width_bytes().ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "field {name}: unresolved dtype"
                ))
            })?;
            // views MUST be in FieldSub::all() order — FieldGather indexes them by sub_ix.
            let mut views = Vec::with_capacity(4);
            for sub in FieldSub::all() {
                views.push(
                    FieldView::open(&paths, cat, name, sub, dtype, n_samples).map_err(|e| {
                        pyo3::exceptions::PyIOError::new_err(format!("open field {name}: {e}"))
                    })?,
                );
            }
            let views: [FieldView; 4] = views
                .try_into()
                .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err("expected 4 field views"))?;
            Ok(svar2::FieldGather {
                views,
                is_format: cat == "format",
                width,
                cohort_n_samples: n_samples,
            })
        })
        .collect::<PyResult<Vec<_>>>()?;

    let itemsizes: Vec<usize> = gathers.iter().map(|g| g.width).collect();
    let has_fields = !gathers.is_empty();

    let (soa, field_bufs) = py.detach(move || {
        let rb = genoray_core::query::HapRanges::new(
            &region_starts_v,
            &orig_samples_v,
            &vk_snp_range_v,
            &vk_indel_range_v,
            &dense_snp_range_v,
            &dense_indel_range_v,
            ploidy,
        );
        // Field gather needs var_key provenance (vk_src), which ONLY the _src
        // variant populates.
        let br = if has_fields {
            genoray_core::query::gather_haps_readbound_src(reader, &rb)
        } else {
            genoray_core::query::gather_haps_readbound(reader, &rb)
        };

        let (lut_bytes, lut_off_u64) = reader.lut_arrays();
        let lut_off: Vec<i64> = lut_off_u64.iter().map(|&x| x as i64).collect();

        svar2::decode_variants_from_split(
            &br,
            &lut_bytes,
            &lut_off,
            &gathers,
            // ON-DISK dense windows (from find_ranges / HapRanges), NOT br's output windows.
            &dense_snp_range_v,
            &dense_indel_range_v,
            &orig_samples_v,
        )
    });

    let field_out: Vec<Bound<'py, PyArray1<u8>>> = field_bufs
        .into_iter()
        .map(|b| Array1::from_vec(b).into_pyarray(py))
        .collect();

    Ok((
        Array1::from_vec(soa.pos).into_pyarray(py),
        Array1::from_vec(soa.ilen).into_pyarray(py),
        Array1::from_vec(soa.alt_bytes).into_pyarray(py),
        Array1::from_vec(soa.str_off).into_pyarray(py),
        Array1::from_vec(soa.var_off).into_pyarray(py),
        field_out,
        itemsizes,
    ))
}

/// Fused SVAR2 two-source track shift+realign: merge each hap's `var_key` ⋈ `dense`
/// channels and decode via `svar2-codec` inline, sizing and allocating the output
/// buffer in Rust — one FFI crossing, mirrors `reconstruct_haplotypes_from_svar2`
/// above but for f32 tracks (see `tracks::shift_and_realign_tracks_from_svar2`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn shift_and_realign_tracks_from_svar2<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    vk_pos: PyReadonlyArray1<i32>,
    vk_key: PyReadonlyArray1<i32>,
    vk_off: PyReadonlyArray1<i64>,
    dense_pos: PyReadonlyArray1<i32>,
    dense_key: PyReadonlyArray1<i32>,
    dense_range: PyReadonlyArray2<i32>,
    dense_present: PyReadonlyArray1<u8>,
    dense_present_off: PyReadonlyArray1<i64>,
    lut_bytes: PyReadonlyArray1<u8>,
    lut_off: PyReadonlyArray1<i64>,
    tracks: PyReadonlyArray1<f32>,
    track_offsets: PyReadonlyArray1<i64>,
    params: PyReadonlyArray1<f64>,
    strategy_id: i64,
    base_seed: u64,
    parallel: bool,
) -> (Bound<'py, PyArray1<f32>>, Bound<'py, PyArray1<i64>>) {
    use crate::svar2;
    use crate::tracks;

    let regions_a = regions.as_array();
    let shifts_a = shifts.as_array();
    let vk_pos_a = vk_pos.as_array();
    let vk_key_a = vk_key.as_array();
    let vk_off_a = vk_off.as_array();
    let dense_pos_a = dense_pos.as_array();
    let dense_key_a = dense_key.as_array();
    let dense_range_a = dense_range.as_array();
    let dense_present_a = dense_present.as_array();
    let dense_present_off_a = dense_present_off.as_array();
    let lut_bytes_a = lut_bytes.as_array();
    let lut_off_a = lut_off.as_array();
    let tracks_a = tracks.as_array();
    let track_offsets_a = track_offsets.as_array();
    let params_a = params.as_array();

    let ploidy = shifts_a.ncols();
    let n_q = regions_a.nrows();
    let n_work = n_q * ploidy;

    let (out_data, out_offsets_vec) = py.detach(move || {
        // Step 1: compute per-haplotype length diffs via the two-source diff core
        // (a realigned track has haplotype length = ref_len + diff, same as reconstruct).
        let vk_pos_s: &[i32] = vk_pos_a.as_slice().unwrap();
        let vk_key_s: &[i32] = vk_key_a.as_slice().unwrap();
        let vk_off_s: &[i64] = vk_off_a.as_slice().unwrap();
        let dense_pos_s: &[i32] = dense_pos_a.as_slice().unwrap();
        let dense_key_s: &[i32] = dense_key_a.as_slice().unwrap();
        let dense_present_s: &[u8] = dense_present_a.as_slice().unwrap();
        let dense_present_off_s: &[i64] = dense_present_off_a.as_slice().unwrap();
        let lut_bytes_s: &[u8] = lut_bytes_a.as_slice().unwrap();
        let lut_off_s: &[i64] = lut_off_a.as_slice().unwrap();

        let diffs = svar2::hap_diffs_svar2(
            regions_a,
            ploidy,
            vk_pos_s,
            vk_key_s,
            vk_off_s,
            dense_pos_s,
            dense_key_s,
            dense_range_a,
            dense_present_s,
            dense_present_off_s,
            lut_bytes_s,
            lut_off_s,
            false,
        );

        // Step 2: compute per-haplotype output lengths and prefix-sum offsets.
        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            out_offsets_vec[0] = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let ref_len = (regions_a[[query, 2]] - regions_a[[query, 1]]) as i64;
                let diff = diffs[[query, hap]] as i64;
                let len: i64 = (ref_len + diff).max(0);
                acc += len;
                out_offsets_vec[k + 1] = acc;
            }
        }

        // Step 3: allocate the output buffer in Rust — Python never calls np.empty.
        // f32 track fill writes every position it needs; zeros is a safe default.
        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<f32> = Array1::<f32>::zeros(total);

        // Step 4: realign all tracks into the owned buffer.
        tracks::shift_and_realign_tracks_from_svar2(
            out_data.view_mut(),
            out_offsets_vec.view(),
            regions_a,
            shifts_a,
            vk_pos_a,
            vk_key_a,
            vk_off_a,
            dense_pos_a,
            dense_key_a,
            dense_range_a,
            dense_present_a,
            dense_present_off_a,
            lut_bytes_a,
            lut_off_a,
            tracks_a,
            track_offsets_a,
            params_a,
            strategy_id,
            base_seed,
            // Single fused call over the whole batch: `k / ploidy` IS the global
            // row, so the FlankSample seed needs no remap (issue #267).
            None,
            parallel,
        );

        (out_data, out_offsets_vec)
    });

    (out_data.into_pyarray(py), out_offsets_vec.into_pyarray(py))
}

/// Fused spliced-haplotype reconstruction: reconstruct in one FFI crossing using
/// precomputed output offsets.
///
/// Unlike ``reconstruct_haplotypes_fused``, the Python splice path already computes
/// the permutation and output offsets (``splice_plan.permuted_out_offsets``), so
/// this kernel takes ``out_offsets`` as a direct parameter and skips Steps 1-2
/// (no ``get_diffs_sparse``, no offset loop). This makes it simpler than the
/// plain fused entry.
///
/// ``permuted_regions`` is shape ``(n_perm, 3)`` where each row is
/// ``[contig_idx, start, end]`` after splice permutation.
/// ``out_offsets`` is ``permuted_out_offsets`` from the Python splice plan
/// (length ``n_perm + 1``).
/// ``geno_offsets`` is the normalized ``(2, n)`` int64 starts/stops array.
///
/// Returns ``out_data`` (u8 flat buffer). The caller already holds ``out_offsets``
/// so it is NOT returned — Python wraps with ``_Flat.from_offsets``.
/// `parallel` enables rayon batch parallelism (caller computes `should_parallelize`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_spliced_fused<'py>(
    py: Python<'py>,
    permuted_regions: PyReadonlyArray2<i32>,
    flat_shifts: PyReadonlyArray2<i32>,
    flat_geno_offset_idx: PyReadonlyArray2<i64>,
    out_offsets: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    to_rc: Option<PyReadonlyArray1<bool>>,
    parallel: bool,
) -> Bound<'py, PyArray1<u8>> {
    use crate::reconstruct;

    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);

    // out_offsets are precomputed by the Python splice plan — use them directly.
    let out_offsets_a = out_offsets.as_array();
    let permuted_regions_a = permuted_regions.as_array();
    let flat_shifts_a = flat_shifts.as_array();
    let flat_geno_offset_idx_a = flat_geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let keep_a = keep.as_ref().map(|k| k.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|ko| ko.as_array());
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());

    let out_data = py.detach(move || {
        let total = out_offsets_a[out_offsets_a.len() - 1] as usize;

        // Allocate output buffer.
        let mut out_data: Array1<u8> = uninit_output(total);

        // Reconstruct all haplotypes into the owned buffer (reuses batch core).
        reconstruct::reconstruct_haplotypes_from_sparse(
            out_data.view_mut(),
            out_offsets_a,
            permuted_regions_a,
            flat_shifts_a,
            flat_geno_offset_idx_a,
            go_starts,
            go_stops,
            geno_v_idxs_a,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            keep_a,
            keep_offsets_a,
            None, // annot_v_idxs — not used in splice path
            None, // annot_ref_pos — not used in splice path
            parallel,
        );

        // Optional in-place RC per permuted element (negative-strand haplotypes).
        // out_offsets_a is the permuted per-element offsets array (splice_plan.permuted_out_offsets),
        // so each masked element is RC'd in its own byte range — matching the to_rc_per_elem post-pass.
        if let Some(to_rc) = to_rc_a.as_ref() {
            debug_assert_eq!(
                to_rc.len(),
                out_offsets_a.len() - 1,
                "to_rc mask length must equal number of output rows (offsets.len() - 1)"
            );
            crate::reverse::rc_flat_rows_inplace(
                out_data.as_slice_mut().unwrap(),
                out_offsets_a,
                *to_rc,
            );
        }

        out_data
    });

    // Return out_data only — Python already holds out_offsets (no round-trip).
    out_data.into_pyarray(py)
}

/// Fused annotated spliced-haplotype reconstruction: the annotated counterpart of
/// `reconstruct_haplotypes_spliced_fused`. Reconstructs in one FFI crossing using
/// precomputed splice output offsets AND fills the two per-nucleotide annotation
/// arrays (variant index, reference coordinate).
///
/// Like the non-annotated splice entry, the Python splice plan already computes the
/// permutation and `out_offsets` (`splice_plan.permuted_out_offsets`), so this kernel
/// takes `out_offsets` directly and skips `get_diffs_sparse` / the offset loop.
///
/// On `to_rc`, each masked permuted element is reverse-complemented in place
/// (`rc_flat_rows_inplace` on the sequence bytes) and its annotation rows are reversed
/// in place (`reverse_flat_rows_inplace`, no complement) — byte-identical to
/// `_FlatAnnotatedHaps.reverse_masked(mask, _COMP)`.
///
/// Returns `(out_data, annot_v, annot_pos)`. `out_offsets` is held by the caller and
/// not returned (matches `reconstruct_haplotypes_spliced_fused`).
/// `parallel` enables rayon batch parallelism (caller computes `should_parallelize`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_annotated_haplotypes_spliced_fused<'py>(
    py: Python<'py>,
    permuted_regions: PyReadonlyArray2<i32>,
    flat_shifts: PyReadonlyArray2<i32>,
    flat_geno_offset_idx: PyReadonlyArray2<i64>,
    out_offsets: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    to_rc: Option<PyReadonlyArray1<bool>>,
    parallel: bool,
) -> (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
) {
    use crate::reconstruct;

    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);

    // out_offsets are precomputed by the Python splice plan — use them directly.
    let out_offsets_a = out_offsets.as_array();
    let permuted_regions_a = permuted_regions.as_array();
    let flat_shifts_a = flat_shifts.as_array();
    let flat_geno_offset_idx_a = flat_geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let keep_a = keep.as_ref().map(|k| k.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|ko| ko.as_array());
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());

    let (out_data, annot_v, annot_pos) = py.detach(move || {
        let total = out_offsets_a[out_offsets_a.len() - 1] as usize;

        // Allocate the sequence + annotation buffers.
        let mut out_data: Array1<u8> = uninit_output(total);
        let mut annot_v: Array1<i32> = uninit_output(total);
        let mut annot_pos: Array1<i32> = uninit_output(total);

        // Reconstruct all haplotypes + annotations into the owned buffers (reuses batch core).
        reconstruct::reconstruct_haplotypes_from_sparse(
            out_data.view_mut(),
            out_offsets_a,
            permuted_regions_a,
            flat_shifts_a,
            flat_geno_offset_idx_a,
            go_starts,
            go_stops,
            geno_v_idxs_a,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            keep_a,
            keep_offsets_a,
            Some(annot_v.view_mut()), // annot_v_idxs — variant index per nucleotide
            Some(annot_pos.view_mut()), // annot_ref_pos — reference coordinate per nucleotide
            parallel,
        );

        // Optional in-place RC per permuted element. Sequence bytes are reverse-complemented;
        // annotation rows are reversed only (no complement) — matching
        // _FlatAnnotatedHaps.reverse_masked. out_offsets_a is the permuted per-element
        // offsets array, so each masked element is transformed in its own byte range.
        if let Some(to_rc) = to_rc_a.as_ref() {
            let m = *to_rc;
            debug_assert_eq!(
                m.len(),
                out_offsets_a.len() - 1,
                "to_rc mask length must equal number of output rows (offsets.len() - 1)"
            );
            crate::reverse::rc_flat_rows_inplace(
                out_data.as_slice_mut().unwrap(),
                out_offsets_a,
                m,
            );
            crate::reverse::reverse_flat_rows_inplace(
                annot_v.as_slice_mut().unwrap(),
                out_offsets_a,
                m,
            );
            crate::reverse::reverse_flat_rows_inplace(
                annot_pos.as_slice_mut().unwrap(),
                out_offsets_a,
                m,
            );
        }

        (out_data, annot_v, annot_pos)
    });

    (
        out_data.into_pyarray(py),
        annot_v.into_pyarray(py),
        annot_pos.into_pyarray(py),
    )
}

/// Fused annotated-haplotype reconstruction: diffs + offsets + reconstruct in one FFI crossing.
///
/// Identical to ``reconstruct_haplotypes_fused`` but ALSO fills per-nucleotide
/// annotation arrays (variant indices and reference coordinates), returning them
/// alongside the haplotype bytes and offsets.
///
/// Steps:
///   1. Compute per-haplotype length diffs via ``get_diffs_sparse``.
///   2. Compute output-length prefix-sum offsets.
///   3. Allocate ``out_data`` (u8), ``annot_v`` (i32), ``annot_pos`` (i32).
///   4. Run ``reconstruct_haplotypes_from_sparse`` with ``Some(annot_v)``, ``Some(annot_pos)``.
///   5. Return ``(out_data, annot_v, annot_pos, out_offsets)`` — Python builds three
///      ``Ragged`` arrays from the shared offsets with no further coercions.
///
/// ``output_length``:
///   - ``-1`` → ragged mode (each haplotype gets its natural length = ref_len + diff).
///   - ``>= 0`` → fixed-length mode (every haplotype is padded/truncated to this length).
///
/// ``geno_offsets`` is the normalized ``(2, n)`` int64 starts/stops array (same
/// layout as the existing ``reconstruct_haplotypes_from_sparse`` FFI entry).
///
/// Annotation buffers are not supported in the plain ``reconstruct_haplotypes_fused``
/// entry; this function is its annotated counterpart.
/// `parallel` enables rayon batch parallelism (caller computes `should_parallelize`).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_annotated_haplotypes_fused<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    alt_alleles: PyReadonlyArray1<u8>,
    alt_offsets: PyReadonlyArray1<i64>,
    ref_: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    output_length: i64,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    to_rc: Option<PyReadonlyArray1<bool>>,
    parallel: bool,
) -> (
    Bound<'py, PyArray1<u8>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i64>>,
) {
    use crate::genotypes;
    use crate::reconstruct;

    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);

    let regions_a = regions.as_array();
    let shifts_a = shifts.as_array();
    let geno_offset_idx_a = geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let alt_alleles_a = alt_alleles.as_array();
    let alt_offsets_a = alt_offsets.as_array();
    let ref_a = ref_.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let keep_a = keep.as_ref().map(|a| a.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|a| a.as_array());
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());

    let (batch_size, ploidy) = geno_offset_idx_a.dim();
    let n_work = batch_size * ploidy;

    let (out_data, annot_v, annot_pos, out_offsets_vec) = py.detach(move || {
        // Step 1: compute per-haplotype length diffs (reuses get_diffs_sparse core).
        // Mirrors _haps.py _haplotype_ilens exactly: pass q_starts/q_ends/v_starts so
        // partial deletions that span a query boundary are correctly clipped.
        // q_starts = regions[:, 1], q_ends = regions[:, 2] (both already in regions_a).
        // v_starts is the same array passed in — it is the per-variant genomic start.
        let q_starts_owned: ndarray::Array1<i32> = regions_a.column(1).to_owned();
        let q_ends_owned: ndarray::Array1<i32> = regions_a.column(2).to_owned();
        let diffs = genotypes::get_diffs_sparse(
            geno_offset_idx_a,
            geno_v_idxs_a,
            go_starts,
            go_stops,
            ilens_a,
            keep_a,
            keep_offsets_a,
            Some(q_starts_owned.view()), // q_starts = regions[:, 1]
            Some(q_ends_owned.view()),   // q_ends   = regions[:, 2]
            Some(v_starts_a),            // v_starts = per-variant genomic starts
            parallel,
        );

        // Step 2: compute per-haplotype output lengths and prefix-sum offsets.
        // Mirrors the Python side: out_lengths = hap_lengths (or fixed output_length).
        // hap_lengths = regions[:, 2] - regions[:, 1] + diffs  (end - start + diff)
        // out_offsets shape: (n_work + 1,)
        let mut out_offsets_vec: Array1<i64> = Array1::zeros(n_work + 1);
        {
            let mut acc: i64 = 0;
            out_offsets_vec[0] = 0;
            for k in 0..n_work {
                let query = k / ploidy;
                let hap = k % ploidy;
                let len: i64 = if output_length >= 0 {
                    output_length
                } else {
                    let ref_len = (regions_a[[query, 2]] - regions_a[[query, 1]]) as i64;
                    let diff = diffs[[query, hap]] as i64;
                    (ref_len + diff).max(0)
                };
                acc += len;
                out_offsets_vec[k + 1] = acc;
            }
        }

        // Step 3: allocate the output buffer and annotation buffers in Rust.
        let total = out_offsets_vec[n_work] as usize;
        let mut out_data: Array1<u8> = uninit_output(total);
        let mut annot_v: Array1<i32> = uninit_output(total);
        let mut annot_pos: Array1<i32> = uninit_output(total);

        // Step 4: reconstruct all haplotypes into the owned buffers (reuses batch core).
        reconstruct::reconstruct_haplotypes_from_sparse(
            out_data.view_mut(),
            out_offsets_vec.view(),
            regions_a,
            shifts_a,
            geno_offset_idx_a,
            go_starts,
            go_stops,
            geno_v_idxs_a,
            v_starts_a,
            ilens_a,
            alt_alleles_a,
            alt_offsets_a,
            ref_a,
            ref_offsets_a,
            pad_char,
            keep_a,
            keep_offsets_a,
            Some(annot_v.view_mut()), // annot_v_idxs — variant index per nucleotide
            Some(annot_pos.view_mut()), // annot_ref_pos — reference coordinate per nucleotide
            parallel,
        );

        if let Some(to_rc) = to_rc_a.as_ref() {
            let m = *to_rc;
            debug_assert_eq!(
                m.len(),
                out_offsets_vec.len() - 1,
                "to_rc mask length must equal number of output rows (offsets.len() - 1)"
            );
            crate::reverse::rc_flat_rows_inplace(
                out_data.as_slice_mut().unwrap(),
                out_offsets_vec.view(),
                m,
            );
            crate::reverse::reverse_flat_rows_inplace(
                annot_v.as_slice_mut().unwrap(),
                out_offsets_vec.view(),
                m,
            );
            crate::reverse::reverse_flat_rows_inplace(
                annot_pos.as_slice_mut().unwrap(),
                out_offsets_vec.view(),
                m,
            );
        }

        (out_data, annot_v, annot_pos, out_offsets_vec)
    });

    // Step 5: return owned arrays — Python wraps them with no further coercions.
    (
        out_data.into_pyarray(py),
        annot_v.into_pyarray(py),
        annot_pos.into_pyarray(py),
        out_offsets_vec.into_pyarray(py),
    )
}

/// Fetch padded reference rows for each region into one flat buffer.
/// `regions[i] = (contig_idx, start, end)`. Mirrors numba `_get_reference_par/_ser`.
#[pyfunction]
pub fn get_reference<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    out_offsets: PyReadonlyArray1<i64>,
    reference: PyReadonlyArray1<u8>,
    ref_offsets: PyReadonlyArray1<i64>,
    pad_char: u8,
    parallel: bool,
    to_rc: Option<PyReadonlyArray1<bool>>,
) -> Bound<'py, PyArray1<u8>> {
    let regions_a = regions.as_array();
    let out_offsets_a = out_offsets.as_array();
    let reference_a = reference.as_array();
    let ref_offsets_a = ref_offsets.as_array();
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());
    let out = py.detach(move || {
        reference::get_reference(
            regions_a,
            out_offsets_a,
            reference_a,
            ref_offsets_a,
            pad_char,
            parallel,
            to_rc_a,
        )
    });
    out.into_pyarray(py)
}

/// Shift and realign tracks for a batch of (query, hap) pairs in place (writes `out`).
///
/// `geno_offsets` is the normalized (2, n) int64 starts/stops array;
/// internally split into `.row(0)` (starts) and `.row(1)` (stops).
/// `keep_offsets` stays 1-D (batch*ploidy + 1) offsets array for the keep mask, or None.
/// `params` is a 1-D f64 parameter array (one entry per track, indexed Python-side).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn shift_and_realign_tracks_sparse(
    py: Python<'_>,
    mut out: PyReadwriteArray1<f32>,
    out_offsets: PyReadonlyArray1<i64>,
    regions: PyReadonlyArray2<i32>,
    shifts: PyReadonlyArray2<i32>,
    geno_offset_idx: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
    geno_offsets: PyReadonlyArray2<i64>,
    v_starts: PyReadonlyArray1<i32>,
    ilens: PyReadonlyArray1<i32>,
    tracks: PyReadonlyArray1<f32>,
    track_offsets: PyReadonlyArray1<i64>,
    params: PyReadonlyArray1<f64>,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    strategy_id: i64,
    base_seed: u64,
    parallel: bool,
) {
    use crate::tracks;
    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);
    let out_a = out.as_array_mut();
    let out_offsets_a = out_offsets.as_array();
    let regions_a = regions.as_array();
    let shifts_a = shifts.as_array();
    let geno_offset_idx_a = geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let tracks_a = tracks.as_array();
    let track_offsets_a = track_offsets.as_array();
    let params_a = params.as_array();
    let keep_a = keep.as_ref().map(|k| k.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|ko| ko.as_array());
    py.detach(move || {
        tracks::shift_and_realign_tracks_sparse(
            out_a,
            out_offsets_a,
            regions_a,
            shifts_a,
            geno_offset_idx_a,
            geno_v_idxs_a,
            go_starts,
            go_stops,
            v_starts_a,
            ilens_a,
            tracks_a,
            track_offsets_a,
            params_a,
            keep_a,
            keep_offsets_a,
            strategy_id,
            base_seed,
            parallel,
        );
    });
}

/// RLE-encode a ragged f32 track buffer into (starts, ends, values, offsets).
///
/// Mirrors numba `tracks_to_intervals` in `_intervals.py` lines 129-195.
/// Returns a 4-tuple `(all_starts: i32, all_ends: i32, all_values: f32, interval_offsets: i64)`.
#[pyfunction]
pub fn tracks_to_intervals<'py>(
    py: Python<'py>,
    regions: PyReadonlyArray2<i32>,
    tracks: PyReadonlyArray1<f32>,
    track_offsets: PyReadonlyArray1<i64>,
    parallel: bool,
) -> (
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<i32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray1<i64>>,
) {
    use crate::tracks;
    let regions_a = regions.as_array();
    let tracks_a = tracks.as_array();
    let track_offsets_a = track_offsets.as_array();
    let (starts, ends, values, offsets) = py.detach(move || {
        tracks::tracks_to_intervals(regions_a, tracks_a, track_offsets_a, parallel)
    });
    (
        starts.into_pyarray(py),
        ends.into_pyarray(py),
        values.into_pyarray(py),
        offsets.into_pyarray(py),
    )
}

/// Fused per-track __getitem__ kernel.
///
/// Collapses two FFI crossings into one per track:
///   1. ``intervals_to_tracks`` core: fills a Rust-side scratch buffer from
///      stored intervals (replacing the Python ``_tracks = np.empty(...)``
///      intermediate, audit T2).
///   2. ``shift_and_realign_tracks_sparse`` core: reads the scratch and writes
///      the caller's pre-allocated ``out`` slice.
///
/// The outer Python loop over n_tracks remains (bounded by track count, small).
/// Each loop iteration now makes ONE FFI crossing instead of two, and allocates
/// ZERO Python-side intermediates.
///
/// ``out`` is the per-track slice of the caller's pre-allocated output buffer
/// (shape ``(b*p*l,)`` f32).  ``out_offsets`` gives ragged lengths into that
/// slice for each (query, hap) pair.
///
/// ``offset_idxs`` is the per-query index array into ``itv_offsets`` (shape
/// ``(b,)``); ``itv_offsets`` is 1-D ``(n_samples*n_regions + 1)`` int64.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn intervals_and_realign_track_fused(
    py: Python<'_>,
    mut out: PyReadwriteArray1<f32>, // (b*p*l) — caller's per-track slice
    out_offsets: PyReadonlyArray1<i64>, // (b*p + 1)
    regions: PyReadonlyArray2<i32>,  // (b, 3)
    shifts: PyReadonlyArray2<i32>,   // (b, p)
    geno_offset_idx: PyReadonlyArray2<i64>, // (b, p)
    geno_v_idxs: PyReadonlyArray1<i32>, // (r*s*p*v)
    geno_offsets: PyReadonlyArray2<i64>, // (2, r*s*p)
    v_starts: PyReadonlyArray1<i32>, // (tot_v)
    ilens: PyReadonlyArray1<i32>,    // (tot_v)
    // intervals (reference-coordinate, for this track)
    offset_idxs: PyReadonlyArray1<i64>, // (b) — per-query index into itv_offsets
    itv_starts: PyReadonlyArray1<i32>,  // (n_intervals)
    itv_ends: PyReadonlyArray1<i32>,    // (n_intervals)
    itv_values: PyReadonlyArray1<f32>,  // (n_intervals)
    itv_offsets: PyReadonlyArray1<i64>, // (n_samples*n_regions + 1)
    track_offsets: PyReadonlyArray1<i64>, // (b+1) — out_offsets for scratch buffer
    // insertion-fill strategy
    params: PyReadonlyArray1<f64>,
    strategy_id: i64,
    base_seed: u64,
    keep: Option<PyReadonlyArray1<bool>>,
    keep_offsets: Option<PyReadonlyArray1<i64>>,
    to_rc: Option<PyReadonlyArray1<bool>>,
    parallel: bool,
) -> PyResult<()> {
    use crate::intervals;
    use crate::tracks;

    let go = geno_offsets.as_array();
    let go_starts = go.row(0);
    let go_stops = go.row(1);

    let out_offsets_a = out_offsets.as_array();
    let regions_a = regions.as_array();
    let track_offsets_a = track_offsets.as_array();
    let mut out_a = out.as_array_mut();
    let shifts_a = shifts.as_array();
    let geno_offset_idx_a = geno_offset_idx.as_array();
    let geno_v_idxs_a = geno_v_idxs.as_array();
    let v_starts_a = v_starts.as_array();
    let ilens_a = ilens.as_array();
    let offset_idxs_a = offset_idxs.as_array();
    let itv_starts_a = itv_starts.as_array();
    let itv_ends_a = itv_ends.as_array();
    let itv_values_a = itv_values.as_array();
    let itv_offsets_a = itv_offsets.as_array();
    let params_a = params.as_array();
    let keep_a = keep.as_ref().map(|k| k.as_array());
    let keep_offsets_a = keep_offsets.as_ref().map(|ko| ko.as_array());
    let to_rc_a = to_rc.as_ref().map(|a| a.as_array());

    py.detach(move || {
        // Determine scratch buffer size from track_offsets.
        let scratch_len = track_offsets_a[track_offsets_a.len() - 1] as usize;

        // Allocate Rust-side scratch buffer — replaces Python `_tracks = np.empty(...)`.
        // intervals_to_tracks calls out.fill(0.0) as its first step, so full-write is
        // guaranteed; uninit_output is safe here.
        let mut scratch = uninit_output::<f32>(scratch_len);

        // Extract query starts (regions[:, 1]) as a contiguous owned array.
        // regions_a.column(1) is a non-contiguous view (row-major storage); we
        // must own/contiguify it before passing to intervals_to_tracks which
        // expects a contiguous ArrayView1<i32>.
        let q_starts: ndarray::Array1<i32> = regions_a.column(1).to_owned();

        // Step 1: paint reference-coordinate intervals into scratch (reuses intervals core).
        intervals::intervals_to_tracks(
            offset_idxs_a,
            q_starts.view(),
            itv_starts_a,
            itv_ends_a,
            itv_values_a,
            itv_offsets_a,
            scratch.view_mut(),
            track_offsets_a,
            parallel,
        );

        // Step 2: shift and realign into caller's out slice (reuses tracks core).
        tracks::shift_and_realign_tracks_sparse(
            out_a.view_mut(),
            out_offsets_a,
            regions_a,
            shifts_a,
            geno_offset_idx_a,
            geno_v_idxs_a,
            go_starts,
            go_stops,
            v_starts_a,
            ilens_a,
            scratch.view(),
            track_offsets_a,
            params_a,
            keep_a,
            keep_offsets_a,
            strategy_id,
            base_seed,
            parallel,
        );

        // Step 3: optional in-place reverse for negative-strand tracks (reverse only, no complement).
        if let Some(to_rc) = to_rc_a.as_ref() {
            debug_assert_eq!(
                to_rc.len(),
                out_offsets_a.len() - 1,
                "to_rc mask length must equal number of output rows (offsets.len() - 1)"
            );
            crate::reverse::reverse_flat_rows_inplace(
                out_a.as_slice_mut().unwrap(),
                out_offsets_a,
                *to_rc,
            );
        }
    });

    Ok(())
}

// ── guard test — drives rc_flat_rows_inplace on a synthetic hap buffer ─
// ── guard test — drives reverse_flat_rows_inplace::<f32> (reverse only) ─
// ── guard test — proves per-element masking over permuted offsets ────────
#[cfg(test)]
mod tests {
    #[test]
    fn haplotype_buffer_rc_is_revcomp_of_forward() {
        let mut out = b"ACGTA".to_vec(); // pretend reconstructed forward bytes
        let offsets = ndarray::array![0i64, 5];
        let to_rc = ndarray::array![true];
        crate::reverse::rc_flat_rows_inplace(&mut out, offsets.view(), to_rc.view());
        assert_eq!(&out, b"TACGT"); // revcomp(ACGTA)
    }

    #[test]
    fn track_buffer_rc_is_reverse_only() {
        let mut out = vec![1.0f32, 2.0, 3.0];
        let offsets = ndarray::array![0i64, 3];
        let to_rc = ndarray::array![true];
        crate::reverse::reverse_flat_rows_inplace(&mut out, offsets.view(), to_rc.view());
        assert_eq!(out, vec![3.0, 2.0, 1.0]); // no value transform
    }

    #[test]
    fn spliced_rc_applies_per_element_over_permuted_offsets() {
        // two permuted elements: "ACG" (rc) and "TTT" (not rc)
        let mut out = b"ACGTTT".to_vec();
        let offsets = ndarray::array![0i64, 3, 6];
        let to_rc = ndarray::array![true, false];
        crate::reverse::rc_flat_rows_inplace(&mut out, offsets.view(), to_rc.view());
        assert_eq!(&out[0..3], b"CGT"); // revcomp(ACG)
        assert_eq!(&out[3..6], b"TTT"); // untouched
    }

    #[test]
    fn annotated_rc_complements_bytes_reverses_indices() {
        let mut bytes = b"ACG".to_vec(); // revcomp -> "CGT"
        let mut vidx = vec![5i32, 6, 7]; // reverse -> [7,6,5]
        let mut rpos = vec![100i32, 101, 102]; // reverse -> [102,101,100]
        let offsets = ndarray::array![0i64, 3];
        let m = ndarray::array![true];
        crate::reverse::rc_flat_rows_inplace(&mut bytes, offsets.view(), m.view());
        crate::reverse::reverse_flat_rows_inplace(&mut vidx, offsets.view(), m.view());
        crate::reverse::reverse_flat_rows_inplace(&mut rpos, offsets.view(), m.view());
        assert_eq!(&bytes, b"CGT");
        assert_eq!(vidx, vec![7, 6, 5]);
        assert_eq!(rpos, vec![102, 101, 100]);
    }

    // ── guard tests — check_disjoint_bounds_within (Finding 1, PR #273 review) ──

    #[test]
    fn disjoint_bounds_accepts_in_range_nonoverlapping_rows() {
        // scattered (not row-ordered), matches the spliced scatter-write shape.
        let bounds = ndarray::arr2(&[[6i64, 10], [0, 4], [4, 6]]);
        assert!(super::check_disjoint_bounds_within(bounds.view(), 10).is_ok());
    }

    #[test]
    fn disjoint_bounds_accepts_adjacent_zero_length_rows() {
        // two empty rows at the same offset write zero bytes each — no aliasing.
        let bounds = ndarray::arr2(&[[3i64, 3], [3, 3], [0, 3]]);
        assert!(super::check_disjoint_bounds_within(bounds.view(), 5).is_ok());
    }

    #[test]
    fn disjoint_bounds_rejects_end_past_out_len() {
        let bounds = ndarray::arr2(&[[0i64, 4], [4, 11]]);
        let msg = super::check_disjoint_bounds_within(bounds.view(), 10).unwrap_err();
        assert!(msg.contains("out_bounds[1] = (4, 11)"), "{msg}");
        assert!(msg.contains("out.len() = 10"), "{msg}");
    }

    #[test]
    fn disjoint_bounds_rejects_negative_start() {
        let bounds = ndarray::arr2(&[[-1i64, 4]]);
        let msg = super::check_disjoint_bounds_within(bounds.view(), 10).unwrap_err();
        assert!(msg.contains("out_bounds[0] = (-1, 4)"));
    }

    #[test]
    fn disjoint_bounds_rejects_start_after_end() {
        let bounds = ndarray::arr2(&[[5i64, 2]]);
        let msg = super::check_disjoint_bounds_within(bounds.view(), 10).unwrap_err();
        assert!(msg.contains("out_bounds[0] = (5, 2)"));
    }

    #[test]
    fn disjoint_bounds_rejects_overlapping_rows() {
        // row 0 = [0, 6), row 1 = [4, 8): overlap in [4, 6), a Python-side indexing
        // bug that would otherwise race under py.detach in the parallel path.
        let bounds = ndarray::arr2(&[[0i64, 6], [4, 8]]);
        let msg = super::check_disjoint_bounds_within(bounds.view(), 10).unwrap_err();
        assert!(msg.contains("pairwise disjoint"), "{msg}");
        assert!(msg.contains("row 1 = (4, 8)"), "{msg}");
    }

    /// Guard for Finding 1 (PR #273 review): sorting the overlap sweep by `start`
    /// alone ties on equal starts, and `sort_unstable`'s tie-break follows original
    /// row order. When a non-empty row sits at a lower row index than a zero-length
    /// row sharing its start, the non-empty row is swept first, its end becomes
    /// `max_end`, and the zero-length row then spuriously fails `s < max_end` even
    /// though it writes zero bytes and cannot alias anything. Row order here
    /// (non-empty row 0, zero-length row 1, both starting at 4) is the ordering that
    /// reproduces the bug pre-fix.
    #[test]
    fn disjoint_bounds_accepts_zero_length_row_tied_with_nonempty_start() {
        let bounds = ndarray::arr2(&[[4i64, 8], [4, 4]]);
        assert!(super::check_disjoint_bounds_within(bounds.view(), 8).is_ok());
    }

    /// Guard for Finding 4 (PR #273 review): pins the "running max end, not just
    /// previous row's end" property. Row 1 = [2, 4) is nested entirely inside row 0
    /// = [0, 10), so a weaker check that only compares each row's start against the
    /// PREVIOUS row's end (rather than the running max) would miss the overlap
    /// between row 0 and row 2 = [6, 8).
    #[test]
    fn disjoint_bounds_rejects_nested_interval() {
        let bounds = ndarray::arr2(&[[0i64, 10], [2, 4], [6, 8]]);
        let msg = super::check_disjoint_bounds_within(bounds.view(), 10).unwrap_err();
        assert!(msg.contains("pairwise disjoint"), "{msg}");
    }
}

// ── DEBUG exports for PRNG parity tests ─────────────────────────────
// These thin wrappers exist solely to make the Rust PRNG functions callable from
// Python tests. Decision: KEEP permanently as the direct
// PRNG parity guard. The njit-internal xorshift64/hash4 leaves have no other
// Python entry point, so these are the only way to assert byte-identity of the
// PRNG core from test_prng_parity.py. Do NOT remove.

/// In-place reverse-complement of the alleles of mask-selected `(b*p)` rows.
/// See `crate::variants::rc_alleles_inplace`.
#[pyfunction]
pub fn rc_alleles(
    mut byte_data: PyReadwriteArray1<u8>,
    seq_offsets: PyReadonlyArray1<i64>,
    var_offsets: PyReadonlyArray1<i64>,
    to_rc_row: PyReadonlyArray1<bool>,
) {
    crate::variants::rc_alleles_inplace(
        byte_data.as_slice_mut().unwrap(),
        seq_offsets.as_array(),
        var_offsets.as_array(),
        to_rc_row.as_array(),
    );
}

/// [DEBUG] Rust xorshift64 — callable from Python for parity testing.
/// Mirrors numba `_xorshift64` on `np.uint64`.
#[pyfunction]
pub fn _debug_xorshift64(x: u64) -> u64 {
    crate::tracks::xorshift64(x)
}

/// [DEBUG] Rust hash4 — callable from Python for parity testing.
/// Mirrors numba `_hash4` on `np.uint64`.
#[pyfunction]
pub fn _debug_hash4(a: u64, b: u64, c: u64, d: u64) -> u64 {
    crate::tracks::hash4(a, b, c, d)
}
