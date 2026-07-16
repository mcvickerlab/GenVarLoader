use std::collections::HashMap;
use std::path::Path;

use genoray_core::record_source::RecordSource;
use genoray_core::svar1_reader::Svar1RecordSource;
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

    /// Read one contig's window into CSR sparse arrays for a batch of
    /// `(region, sample)` pairs. `region_bounds` and `samples` are PARALLEL
    /// arrays of the same length `batch`: batch row `bi` is the pair
    /// `(region_bounds[bi], samples[bi])` (NOT a cartesian grid — the caller,
    /// `reconstruct_haplotypes_svar1`, flattens `(region, sample)` instances,
    /// so a region can repeat across rows with different samples and vice
    /// versa). `region_bounds` are 0-based half-open `(start, end)` pairs on
    /// `contig`; `samples` are absolute sample indices. Walks
    /// `Svar1RecordSource::next_record` once (in file/position order) and, per
    /// record, buckets the GLOBAL variant index into every `(batch-row, hap)`
    /// row whose bounds contain the record's position and whose genotype
    /// carries a non-REF allele. Returns row-major (batch-row-major, then hap)
    /// CSR — see `Sparse` doc comment.
    pub fn read_window(
        &self,
        contig: &str,
        region_bounds: &[(i32, i32)],
        samples: &[usize],
    ) -> anyhow::Result<super::Sparse> {
        let t = self
            .table(contig)
            .ok_or_else(|| anyhow::anyhow!("no contig table registered for {contig}"))?;
        let mut src = Svar1RecordSource::new(
            self.store_path(),
            t.contig_start,
            t.n_local,
            self.n_samples,
            self.ploidy,
            t.pos.clone(),
            t.ref_bytes.clone(),
            t.ref_offsets.clone(),
            t.alt_bytes.clone(),
            t.alt_offsets.clone(),
            &[],
            &[],
        )
        .map_err(|e| anyhow::anyhow!("svar1 open (contig {contig}): {e:?}"))?;

        let ploidy = self.ploidy;
        if region_bounds.len() != samples.len() {
            anyhow::bail!(
                "read_window: region_bounds ({}) and samples ({}) must be parallel arrays of equal length",
                region_bounds.len(),
                samples.len()
            );
        }
        let batch = region_bounds.len();
        let n_rows = batch * ploidy;
        let mut buckets: Vec<Vec<i32>> = vec![Vec::new(); n_rows];

        let mut local_i: i32 = 0;
        loop {
            let rec = src
                .next_record()
                .map_err(|e| anyhow::anyhow!("svar1 read (contig {contig}): {e:?}"))?;
            let Some(rec) = rec else { break };
            let global_v = t.contig_start as i32 + local_i;
            local_i += 1;
            let pos = rec.pos as i32;
            for (bi, (&(lo, hi), &s)) in region_bounds.iter().zip(samples).enumerate() {
                if pos < lo || pos >= hi {
                    continue;
                }
                let row_base = bi * ploidy;
                for h in 0..ploidy {
                    let col = s * ploidy + h;
                    if rec.gt[col] != 0 {
                        buckets[row_base + h].push(global_v);
                    }
                }
            }
        }

        let mut geno_v_idxs: Vec<i32> = Vec::new();
        let mut o_starts: Vec<i64> = Vec::with_capacity(n_rows);
        let mut o_stops: Vec<i64> = Vec::with_capacity(n_rows);
        let mut geno_offset_idx = ndarray::Array2::<i64>::zeros((batch, ploidy));
        for row in 0..batch {
            for h in 0..ploidy {
                let idx = row * ploidy + h;
                let start = geno_v_idxs.len() as i64;
                geno_v_idxs.extend_from_slice(&buckets[idx]);
                let stop = geno_v_idxs.len() as i64;
                o_starts.push(start);
                o_stops.push(stop);
                geno_offset_idx[[row, h]] = idx as i64;
            }
        }

        Ok(super::Sparse {
            geno_v_idxs,
            o_starts,
            o_stops,
            geno_offset_idx,
        })
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

    /// Register the per-contig static variant table (local 0-based `pos` +
    /// REF/ALT CSR, keyed by local variant index) read from `SparseVar.index`
    /// on the Python side. `contig_start` is this contig's first variant's
    /// index in the GLOBAL variant table; `n_local` is its variant count.
    #[allow(clippy::too_many_arguments)]
    fn set_contig_table(
        &mut self,
        contig: &str,
        contig_start: usize,
        n_local: usize,
        pos: Vec<u32>,
        ref_bytes: Vec<u8>,
        ref_offsets: Vec<i64>,
        alt_bytes: Vec<u8>,
        alt_offsets: Vec<i64>,
    ) {
        self.tables.insert(
            contig.to_string(),
            ContigTable {
                contig_start,
                n_local,
                pos,
                ref_bytes,
                ref_offsets,
                alt_bytes,
                alt_offsets,
            },
        );
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
