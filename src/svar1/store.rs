use std::collections::HashMap;
use std::path::Path;

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// Per-contig static variant table + geometry supplied by Python at construction.
pub struct ContigTable {
    pub contig_start: usize, // index of this contig's first variant in the global table
    pub n_local: usize,      // variants on this contig
    pub pos: Vec<u32>,       // 0-based POS
    pub ref_bytes: Vec<u8>,
    pub ref_offsets: Vec<i64>,
    pub alt_bytes: Vec<u8>,
    pub alt_offsets: Vec<i64>,
}

#[pyclass]
pub struct Svar1Store {
    store_path: String,
    n_samples: usize,
    ploidy: usize,
    contigs: Vec<String>,
    tables: HashMap<String, ContigTable>, // filled via `set_contig_table` in Task 4
}

impl Svar1Store {
    /// Metadata-only constructor (validates the store dir); used by tests + `#[new]`.
    pub fn open_meta(
        store_path: &str,
        contigs: Vec<String>,
        n_samples: usize,
        ploidy: usize,
    ) -> PyResult<Self> {
        if !Path::new(store_path).is_dir() {
            return Err(PyIOError::new_err(format!(
                "svar store not found: {store_path}"
            )));
        }
        Ok(Self {
            store_path: store_path.to_string(),
            n_samples,
            ploidy,
            contigs,
            tables: HashMap::new(),
        })
    }

    pub fn store_path(&self) -> &str {
        &self.store_path
    }
    pub fn table(&self, contig: &str) -> Option<&ContigTable> {
        self.tables.get(contig)
    }
    pub fn n_samples(&self) -> usize {
        self.n_samples
    }
    pub fn ploidy(&self) -> usize {
        self.ploidy
    }
}

#[pymethods]
impl Svar1Store {
    #[new]
    fn new(store_path: &str, contigs: Vec<String>, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        Self::open_meta(store_path, contigs, n_samples, ploidy)
    }

    fn contigs(&self) -> Vec<String> {
        self.contigs.clone()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn store_path_roundtrips() {
        // Construction records metadata; a missing dir errors.
        let err = super::Svar1Store::open_meta("/no/such/svar", vec!["chr1".into()], 2, 2);
        assert!(err.is_err());
    }
}
