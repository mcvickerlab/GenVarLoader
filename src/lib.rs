mod bigwig;
use numpy::{prelude::*, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::path::PathBuf;

#[pymodule]
fn genvarloader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(intervals, m)?)?;
    Ok(())
}

/// Load intervals from BigWig files.
///
/// Parameters
/// ----------
/// paths : List[str | Path]
///     Paths to BigWig files.
/// contig : str
///     Contig name.
/// starts : NDArray[int32]
///     Start positions.
/// ends : NDArray[int32]
///     End positions.
///
/// Returns
/// -------
/// coordinates : NDArray[uint32]
///     Shape = (intervals) Coordinates.
/// values : NDArray[float32]
///     Shape = (intervals) Values.
/// n_per_query : NDArray[int32]
///     Shape = (samples, regions) Number of intervals per query.
#[pyfunction]
fn intervals<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    contig: &str,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
) -> (
    Bound<'py, PyArray2<u32>>,
    Bound<'py, PyArray1<f32>>,
    Bound<'py, PyArray2<i32>>,
) {
    let (itvs, vals, n_per_query) =
        bigwig::intervals(paths, contig, starts.as_array(), ends.as_array()).unwrap();
    (
        itvs.into_pyarray_bound(py),
        vals.into_pyarray_bound(py),
        n_per_query.into_pyarray_bound(py),
    )
}
