pub mod bigwig;
pub mod ffi;
pub mod genotypes;
pub mod intervals;
pub mod ragged;
pub mod reconstruct;
pub mod reference;
pub mod reverse;
pub mod svar2;
pub mod tables;
pub mod tracks;
pub mod variants;
use numpy::{prelude::*, PyArray1, PyArray2, PyReadonlyArray1};
use pyo3::prelude::*;
use std::path::PathBuf;

#[pymodule]
fn genvarloader(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(count_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(bigwig_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(bigwig_write_track, m)?)?;
    m.add_class::<tables::RustTable>()?;
    m.add_function(wrap_pyfunction!(ragged::ragged_to_padded, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::intervals_to_tracks, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::get_diffs_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::choose_exonic_variants, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gather_rows_i32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gather_rows_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::gather_alleles, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::compact_keep_i32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::compact_keep_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fill_empty_scalar_i32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fill_empty_scalar_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fill_empty_fixed_i32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fill_empty_fixed_f32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fill_empty_seq_u8, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::fill_empty_seq_i32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::assemble_variant_buffers_u8, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::assemble_variant_buffers_i32, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::rc_alleles, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::get_reference, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::reconstruct_haplotypes_from_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::reconstruct_haplotypes_fused, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::reconstruct_annotated_haplotypes_fused, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::reconstruct_haplotypes_spliced_fused, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::reconstruct_annotated_haplotypes_spliced_fused, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::shift_and_realign_tracks_sparse, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::tracks_to_intervals, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::intervals_and_realign_track_fused, m)?)?;
    // DEBUG: PRNG parity exports (Task 7) — keep or remove after Task 8/9 review
    m.add_function(wrap_pyfunction!(ffi::_debug_xorshift64, m)?)?;
    m.add_function(wrap_pyfunction!(ffi::_debug_hash4, m)?)?;
    Ok(())
}

/// Write SoA starts/ends/values.npy + offsets.npy for a bigWig track directly to `out_dir`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn bigwig_write_track(
    paths: Vec<PathBuf>,
    contigs: Vec<String>,
    starts: PyReadonlyArray1<i32>,
    ends: PyReadonlyArray1<i32>,
    max_mem: usize,
    out_dir: PathBuf,
    sample_less: bool,
) -> PyResult<()> {
    bigwig::write_track(
        &paths,
        &contigs,
        starts.as_array(),
        ends.as_array(),
        max_mem,
        &out_dir,
        sample_less,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e.to_string()))
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
fn bigwig_intervals<'py>(
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
