//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyReadwriteArray1};
use pyo3::prelude::*;

use crate::genotypes;
use crate::intervals;
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

/// Per-row variant-index gather (see `variants::gather_rows`).
#[pyfunction]
pub fn gather_rows<'py>(
    py: Python<'py>,
    geno_offset_idx: PyReadonlyArray1<i64>,
    geno_offsets: PyReadonlyArray2<i64>,
    geno_v_idxs: PyReadonlyArray1<i32>,
) -> (Bound<'py, PyArray1<i32>>, Bound<'py, PyArray1<i64>>) {
    let go = geno_offsets.as_array();
    let (v, off) = variants::gather_rows(
        geno_offset_idx.as_array(),
        go.row(0),
        go.row(1),
        geno_v_idxs.as_array(),
    );
    (v.into_pyarray(py), off.into_pyarray(py))
}
