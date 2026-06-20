use anyhow::Result;
use ndarray::prelude::*;
use pyo3::prelude::*;

/// One sample's intervals on one contig, sorted by start.
#[derive(Default, Clone)]
struct SampleIntervals {
    starts: Vec<i32>,
    ends: Vec<i32>,
    values: Vec<f32>,
}

/// All samples' intervals on one contig.
#[derive(Clone)]
struct ContigStore {
    samples: Vec<SampleIntervals>, // indexed by sample_code
}

#[pyclass]
pub struct RustTable {
    n_samples: usize,
    store: Vec<ContigStore>, // indexed by chrom_code (0..n_contigs)
}

impl RustTable {
    pub fn build(
        sample_codes: ArrayView1<i32>,
        chrom_codes: ArrayView1<i32>,
        starts: ArrayView1<i32>,
        ends: ArrayView1<i32>,
        values: ArrayView1<f32>,
        n_samples: usize,
        n_contigs: usize,
    ) -> RustTable {
        let mut store: Vec<ContigStore> = (0..n_contigs)
            .map(|_| ContigStore {
                samples: vec![SampleIntervals::default(); n_samples],
            })
            .collect();
        for i in 0..sample_codes.len() {
            let c = chrom_codes[i] as usize;
            let s = sample_codes[i] as usize;
            let cell = &mut store[c].samples[s];
            cell.starts.push(starts[i]);
            cell.ends.push(ends[i]);
            cell.values.push(values[i]);
        }
        RustTable { n_samples, store }
    }

    #[cfg(test)]
    fn n_in_cell(&self, chrom: usize, sample: usize) -> usize {
        self.store[chrom].samples[sample].starts.len()
    }
}

// (test module below)

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// Columns pre-sorted by (chrom_code, sample_code, start), as Python guarantees.
    fn toy() -> RustTable {
        // sample 0: chr0 [0,10),[50,60); sample 1: chr0 [10,20); sample 0: chr1 [0,5)
        let sample_codes = array![0i32, 0, 1, 0];
        let chrom_codes = array![0i32, 0, 0, 1];
        let starts = array![0i32, 50, 10, 0];
        let ends = array![10i32, 60, 20, 5];
        let values = array![1.0f32, 2.0, 3.0, 4.0];
        RustTable::build(
            sample_codes.view(),
            chrom_codes.view(),
            starts.view(),
            ends.view(),
            values.view(),
            2,
            2,
        )
    }

    #[test]
    fn store_groups_by_contig_then_sample() {
        let t = toy();
        // chr0 sample0 has 2 intervals; chr0 sample1 has 1; chr1 sample0 has 1; chr1 sample1 has 0
        assert_eq!(t.n_in_cell(0, 0), 2);
        assert_eq!(t.n_in_cell(0, 1), 1);
        assert_eq!(t.n_in_cell(1, 0), 1);
        assert_eq!(t.n_in_cell(1, 1), 0);
    }
}
