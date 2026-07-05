use std::collections::HashMap;

use genoray_core::query::ContigReader;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// Opened once at Dataset.open; holds one query-only ContigReader per contig for
/// the store's lifetime (SVAR2 analog of SVAR1's cached _HapsFfiStatic).
#[pyclass]
pub struct Svar2Store {
    readers: HashMap<String, ContigReader>,
}

impl Svar2Store {
    pub fn reader(&self, contig: &str) -> Option<&ContigReader> {
        self.readers.get(contig)
    }
}

#[pymethods]
impl Svar2Store {
    #[new]
    fn new(
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
        Ok(Self { readers })
    }

    fn contigs(&self) -> Vec<String> {
        let mut v: Vec<String> = self.readers.keys().cloned().collect();
        v.sort();
        v
    }
}
