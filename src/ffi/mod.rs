//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::variants::windows::{assemble_variants_mode, assemble_windows_mode, VariantBufs};

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
        py, mode, v_idxs, row_offsets, alt_global, alt_off_global, ref_global,
        ref_off_global, want_ref_bytes, want_flank, ref_mode, alt_mode, flank_len,
        lut, v_contigs, v_starts, ilens, reference, ref_offsets, pad_char,
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
        py, mode, v_idxs, row_offsets, alt_global, alt_off_global, ref_global,
        ref_off_global, want_ref_bytes, want_flank, ref_mode, alt_mode, flank_len,
        lut, v_contigs, v_starts, ilens, reference, ref_offsets, pad_char,
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

/// Fused haplotypes __getitem__ kernel (Task 13).
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
/// remains on the unfused dispatch wrappers — see Task 13 report for rationale).
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
            Some(annot_v.view_mut()),   // annot_v_idxs — variant index per nucleotide
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
            crate::reverse::rc_flat_rows_inplace(out_data.as_slice_mut().unwrap(), out_offsets_a, m);
            crate::reverse::reverse_flat_rows_inplace(annot_v.as_slice_mut().unwrap(), out_offsets_a, m);
            crate::reverse::reverse_flat_rows_inplace(annot_pos.as_slice_mut().unwrap(), out_offsets_a, m);
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
            Some(annot_v.view_mut()),   // annot_v_idxs — variant index per nucleotide
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
            crate::reverse::rc_flat_rows_inplace(out_data.as_slice_mut().unwrap(), out_offsets_vec.view(), m);
            crate::reverse::reverse_flat_rows_inplace(annot_v.as_slice_mut().unwrap(), out_offsets_vec.view(), m);
            crate::reverse::reverse_flat_rows_inplace(annot_pos.as_slice_mut().unwrap(), out_offsets_vec.view(), m);
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

/// Fused per-track __getitem__ kernel (Task 14).
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
    mut out: PyReadwriteArray1<f32>,          // (b*p*l) — caller's per-track slice
    out_offsets: PyReadonlyArray1<i64>,       // (b*p + 1)
    regions: PyReadonlyArray2<i32>,           // (b, 3)
    shifts: PyReadonlyArray2<i32>,            // (b, p)
    geno_offset_idx: PyReadonlyArray2<i64>,   // (b, p)
    geno_v_idxs: PyReadonlyArray1<i32>,       // (r*s*p*v)
    geno_offsets: PyReadonlyArray2<i64>,      // (2, r*s*p)
    v_starts: PyReadonlyArray1<i32>,          // (tot_v)
    ilens: PyReadonlyArray1<i32>,             // (tot_v)
    // intervals (reference-coordinate, for this track)
    offset_idxs: PyReadonlyArray1<i64>,       // (b) — per-query index into itv_offsets
    itv_starts: PyReadonlyArray1<i32>,         // (n_intervals)
    itv_ends: PyReadonlyArray1<i32>,           // (n_intervals)
    itv_values: PyReadonlyArray1<f32>,         // (n_intervals)
    itv_offsets: PyReadonlyArray1<i64>,        // (n_samples*n_regions + 1)
    track_offsets: PyReadonlyArray1<i64>,      // (b+1) — out_offsets for scratch buffer
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

// ── Task 3: guard test — drives rc_flat_rows_inplace on a synthetic hap buffer ─
// ── Task 4: guard test — drives reverse_flat_rows_inplace::<f32> (reverse only) ─
// ── Task 6: guard test — proves per-element masking over permuted offsets ────────
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
        let mut bytes = b"ACG".to_vec();          // revcomp -> "CGT"
        let mut vidx = vec![5i32, 6, 7];          // reverse -> [7,6,5]
        let mut rpos = vec![100i32, 101, 102];    // reverse -> [102,101,100]
        let offsets = ndarray::array![0i64, 3];
        let m = ndarray::array![true];
        crate::reverse::rc_flat_rows_inplace(&mut bytes, offsets.view(), m.view());
        crate::reverse::reverse_flat_rows_inplace(&mut vidx, offsets.view(), m.view());
        crate::reverse::reverse_flat_rows_inplace(&mut rpos, offsets.view(), m.view());
        assert_eq!(&bytes, b"CGT");
        assert_eq!(vidx, vec![7, 6, 5]);
        assert_eq!(rpos, vec![102, 101, 100]);
    }
}

// ── DEBUG exports for PRNG parity tests (Task 7) ─────────────────────────────
// These thin wrappers exist solely to make the Rust PRNG functions callable from
// Python tests. Decision (final-review, Task 15): KEEP permanently as the direct
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
