//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::prelude::*;

use crate::intervals;

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
