//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;

use crate::genotypes;
use crate::intervals;
use crate::reference;
use crate::variants;

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
) -> Bound<'py, PyArray2<i32>> {
    let go = geno_offsets.as_array();
    let diffs = genotypes::get_diffs_sparse(
        geno_offset_idx.as_array(),
        geno_v_idxs.as_array(),
        go.row(0),
        go.row(1),
        ilens.as_array(),
        keep.as_ref().map(|a| a.as_array()),
        keep_offsets.as_ref().map(|a| a.as_array()),
        q_starts.as_ref().map(|a| a.as_array()),
        q_ends.as_ref().map(|a| a.as_array()),
        v_starts.as_ref().map(|a| a.as_array()),
    );
    diffs.into_pyarray(py)
}

/// Paint base-pair-resolution tracks from intervals (writes `out` in place).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn intervals_to_tracks(
    offset_idxs: PyReadonlyArray1<i64>,
    starts: PyReadonlyArray1<i32>,
    itv_starts: PyReadonlyArray1<i32>,
    itv_ends: PyReadonlyArray1<i32>,
    itv_values: PyReadonlyArray1<f32>,
    itv_offsets: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray1<f32>,
    out_offsets: PyReadonlyArray1<i64>,
) {
    intervals::intervals_to_tracks(
        offset_idxs.as_array(),
        starts.as_array(),
        itv_starts.as_array(),
        itv_ends.as_array(),
        itv_values.as_array(),
        itv_offsets.as_array(),
        out.as_array_mut(),
        out_offsets.as_array(),
    );
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
    let (v, off) = variants::compact_keep_i32(
        values.as_array(),
        row_offsets.as_array(),
        keep.as_array(),
    );
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
    let (v, off) = variants::compact_keep_f32(
        values.as_array(),
        row_offsets.as_array(),
        keep.as_array(),
    );
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
    let (v, off) = variants::fill_empty_scalar_i32(
        data.as_array(),
        offsets.as_array(),
        fill,
    );
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
    let (v, off) = variants::fill_empty_scalar_f32(
        data.as_array(),
        offsets.as_array(),
        fill,
    );
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
    let (v, off) = variants::fill_empty_fixed_i32(
        data.as_array(),
        offsets.as_array(),
        inner,
        fill,
    );
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
    let (v, off) = variants::fill_empty_fixed_f32(
        data.as_array(),
        offsets.as_array(),
        inner,
        fill,
    );
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
    (nd.into_pyarray(py), nvar.into_pyarray(py), nseq.into_pyarray(py))
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
    (nd.into_pyarray(py), nvar.into_pyarray(py), nseq.into_pyarray(py))
}

/// Reconstruct haplotypes for a batch of (query, hap) pairs in place (writes `out`).
///
/// `geno_offsets` is the normalized (2, n) int64 starts/stops array.
/// `keep_offsets` is the 1-D (batch*ploidy + 1) offsets array for the keep mask, or None.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_sparse(
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
) {
    use crate::reconstruct;
    let go = geno_offsets.as_array();
    reconstruct::reconstruct_haplotypes_from_sparse(
        out.as_array_mut(),
        out_offsets.as_array(),
        regions.as_array(),
        shifts.as_array(),
        geno_offset_idx.as_array(),
        go.row(0),
        go.row(1),
        geno_v_idxs.as_array(),
        v_starts.as_array(),
        ilens.as_array(),
        alt_alleles.as_array(),
        alt_offsets.as_array(),
        ref_.as_array(),
        ref_offsets.as_array(),
        pad_char,
        keep.as_ref().map(|k| k.as_array()),
        keep_offsets.as_ref().map(|ko| ko.as_array()),
        annot_v_idxs.as_mut().map(|a| a.as_array_mut()),
        annot_ref_pos.as_mut().map(|a| a.as_array_mut()),
    );
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

/// Shift and realign tracks for a batch of (query, hap) pairs in place (writes `out`).
///
/// `geno_offsets` is the normalized (2, n) int64 starts/stops array;
/// internally split into `.row(0)` (starts) and `.row(1)` (stops).
/// `keep_offsets` stays 1-D (batch*ploidy + 1) offsets array for the keep mask, or None.
/// `params` is a 1-D f64 parameter array (one entry per track, indexed Python-side).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn shift_and_realign_tracks_sparse(
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
) {
    use crate::tracks;
    let go = geno_offsets.as_array();
    tracks::shift_and_realign_tracks_sparse(
        out.as_array_mut(),
        out_offsets.as_array(),
        regions.as_array(),
        shifts.as_array(),
        geno_offset_idx.as_array(),
        geno_v_idxs.as_array(),
        go.row(0),
        go.row(1),
        v_starts.as_array(),
        ilens.as_array(),
        tracks.as_array(),
        track_offsets.as_array(),
        params.as_array(),
        keep.as_ref().map(|k| k.as_array()),
        keep_offsets.as_ref().map(|ko| ko.as_array()),
        strategy_id,
        base_seed,
    );
}

// ── DEBUG exports for PRNG parity tests (Task 7) ─────────────────────────────
// These thin wrappers exist solely to make the Rust PRNG functions callable from
// Python tests. They may be kept or removed after Task 8/9 review.

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
