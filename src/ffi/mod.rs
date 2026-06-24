//! PyO3 boundary for migrated core kernels. The ONLY place new kernels touch Python.
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;

use crate::utils;

/// Greedy split offsets for groups summing to no more than `max_value`.
#[pyfunction]
pub fn splits_sum_le_value<'py>(
    py: Python<'py>,
    arr: PyReadonlyArray1<i64>,
    max_value: f64,
) -> Bound<'py, PyArray1<i64>> {
    utils::splits_sum_le_value(arr.as_array(), max_value).into_pyarray(py)
}
