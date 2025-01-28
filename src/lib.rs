pub mod bigwig;
use numpy::{prelude::*, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::path::PathBuf;

#[pymodule]
fn genvarloader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(intervals, m)?)?;
    Ok(())
}

/// Count intervals from BigWig files.
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
/// n_per_query : NDArray[int32]
///     Shape = (samples, regions) Number of intervals per query.
#[pyfunction]
fn count_intervals<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    contig: &str,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
) -> Bound<'py, PyArray2<i32>> {
    let n_per_query =
        bigwig::count_intervals(&paths, contig, starts.as_array(), ends.as_array()).unwrap();

    n_per_query.into_pyarray(py)
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
/// offsets : NDArray[int64]
///     Offsets corresponding to the returned interval data of shape (regions, samples). Can be
///     computed from the number of intervals per query, e.g. with the count_intervals function.
///
/// Returns
/// -------
/// coordinates : NDArray[uint32]
///     Shape = (intervals, 2) Coordinates.
/// values : NDArray[float32]
///     Shape = (intervals) Values.
#[pyfunction]
fn intervals<'py>(
    py: Python<'py>,
    paths: Vec<PathBuf>,
    contig: &str,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
    offsets: PyReadonlyArray1<i64>,
) -> (Bound<'py, PyArray2<u32>>, Bound<'py, PyArray1<f32>>) {
    let (itvs, vals) = unsafe {
        bigwig::intervals(
            &paths,
            contig,
            starts.as_array(),
            ends.as_array(),
            offsets.as_array(),
        )
    }
    .unwrap();
    (itvs.into_pyarray(py), vals.into_pyarray(py))
}
