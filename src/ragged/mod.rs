use numpy::{PyReadonlyArray1, PyReadwriteArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Copy each ragged row into a pre-filled (n_rows, out_len) uint8 buffer via
/// the shared seqpro-core kernel. Mirrors seqpro's `_ragged_to_padded`.
#[pyfunction]
pub fn ragged_to_padded(
    data: PyReadonlyArray1<u8>,
    offsets: PyReadonlyArray1<i64>,
    mut out: PyReadwriteArray1<u8>,
    itemsize: usize,
    out_len: usize,
) -> PyResult<()> {
    let data = data.as_slice()?;
    let offsets = offsets.as_slice()?;
    let out = out
        .as_slice_mut()
        .map_err(|_| PyValueError::new_err("out must be contiguous"))?;
    seqpro_core::Ragged::new(offsets, data, itemsize)
        .to_padded_into(out, itemsize, out_len)
        .map_err(PyValueError::new_err)
}
