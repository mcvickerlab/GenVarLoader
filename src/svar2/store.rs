use std::collections::HashMap;

use genoray_core::query::ContigReader;
use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// Opened once at Dataset.open; holds one query-only ContigReader per contig for
/// the store's lifetime (SVAR2 analog of SVAR1's cached _HapsFfiStatic).
#[pyclass]
pub struct Svar2Store {
    readers: HashMap<String, ContigReader>,
    store_path: String,
}

impl Svar2Store {
    /// Opens one query-only `ContigReader` per contig under `store_path`. Rust-side
    /// constructor (the `#[pymethods]` `new` below delegates here) so callers that build
    /// their OWN store instance without going through Python's `__new__` -- the streaming
    /// engine (`Svar2StreamEngine::ensure_started`'s producer opens its own `Arc<Svar2Store>`,
    /// same convention as `Svar1Store::open_meta`) and Rust unit tests -- have a `pub`
    /// entry point.
    pub fn open(
        store_path: &str,
        contigs: Vec<String>,
        n_samples: usize,
        ploidy: usize,
    ) -> PyResult<Self> {
        let mut readers = HashMap::with_capacity(contigs.len());
        for c in contigs {
            let r = ContigReader::open(store_path, &c, n_samples, ploidy)
                .map_err(|e| PyIOError::new_err(format!("open contig {c}: {e}")))?;
            readers.insert(c, r);
        }
        let store_path = store_path.to_string();
        Ok(Self {
            readers,
            store_path,
        })
    }

    /// Returns the cached `ContigReader` for `contig`, if one was opened.
    pub fn reader(&self, contig: &str) -> Option<&ContigReader> {
        self.readers.get(contig)
    }
    /// Returns the filesystem path the store was opened from.
    pub fn store_path(&self) -> &str {
        &self.store_path
    }
}

#[pymethods]
impl Svar2Store {
    /// Opens one query-only `ContigReader` per contig under `store_path`.
    #[new]
    fn new(
        store_path: &str,
        contigs: Vec<String>,
        n_samples: usize,
        ploidy: usize,
    ) -> PyResult<Self> {
        Self::open(store_path, contigs, n_samples, ploidy)
    }

    /// Returns the sorted list of contigs with an opened reader.
    fn contigs(&self) -> Vec<String> {
        let mut v: Vec<String> = self.readers.keys().cloned().collect();
        v.sort();
        v
    }
}

/// Recycled output buffer for a SVAR2 streaming "super-batch" reconstruction: one
/// [`crate::svar2::svar2_readbound_chain`] fill covers many (region, sample) rows at
/// once (coarser than the Phase-1 per-`batch_size` FFI call), and the sync/async drive
/// then drains `batch_size`-sized C-order row slices out of it via [`Svar2ReconBuf::batch`].
/// Reused across super-batches (capacity kept, contents replaced by `set`) to avoid a
/// per-super-batch allocation.
#[pyclass]
pub struct Svar2ReconBuf {
    data: Vec<u8>,     // reconstructed bytes for the current super-batch (capacity reused)
    offsets: Vec<i64>, // len n_rows*ploidy + 1
    n_rows: usize,     // queries (region,sample cells) in the current fill
    ploidy: usize,
}

impl Svar2ReconBuf {
    /// Rust-only: the FFI fill fn writes results here (moves the chain's Vecs in).
    pub(crate) fn set(&mut self, data: Vec<u8>, offsets: Vec<i64>, n_rows: usize) {
        self.data = data;
        self.offsets = offsets;
        self.n_rows = n_rows;
    }
}

#[pymethods]
impl Svar2ReconBuf {
    #[new]
    fn new(ploidy: usize) -> Self {
        Svar2ReconBuf {
            data: Vec::new(),
            offsets: vec![0],
            n_rows: 0,
            ploidy,
        }
    }

    #[getter]
    fn n_rows(&self) -> usize {
        self.n_rows
    }

    #[getter]
    fn total_bytes(&self) -> usize {
        *self.offsets.last().unwrap_or(&0) as usize
    }

    /// Copy out rows `[lo, hi)`: flat data bytes + offsets (len `(hi-lo)*ploidy+1`)
    /// rebased to 0. Always a copy (never a view into the recycled buffer), since the
    /// buffer is refilled in place across super-batches and a view would dangle.
    #[allow(clippy::type_complexity)]
    fn batch<'py>(
        &self,
        py: Python<'py>,
        lo: usize,
        hi: usize,
    ) -> PyResult<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)> {
        if hi < lo || hi > self.n_rows {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "batch [{lo},{hi}) out of range for n_rows={}",
                self.n_rows
            )));
        }
        let p = self.ploidy;
        let o_lo = lo * p; // first offset index for the row block
        let o_hi = hi * p; // last row's last hap offset index (inclusive end at o_hi)
        let byte_lo = self.offsets[o_lo] as usize;
        let byte_hi = self.offsets[o_hi] as usize;
        let data = self.data[byte_lo..byte_hi].to_vec();
        let base = self.offsets[o_lo];
        let offsets: Vec<i64> = self.offsets[o_lo..=o_hi]
            .iter()
            .map(|&x| x - base)
            .collect();
        Ok((
            Array1::from_vec(data).into_pyarray(py),
            Array1::from_vec(offsets).into_pyarray(py),
        ))
    }
}
