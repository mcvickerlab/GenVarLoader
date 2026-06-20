use anyhow::Result;
use coitrees::{BasicCOITree, Interval, IntervalTree};
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

    /// Build one COITree per sample for `chrom`. Intervals are stored half-open
    /// [start, end); coitrees is inclusive, so we store [start, end-1] and query
    /// [qs, qe-1]. Metadata = index into the sample's sorted arrays.
    fn build_trees(&self, chrom: usize) -> Vec<BasicCOITree<u32, u32>> {
        self.store[chrom]
            .samples
            .iter()
            .map(|cell| {
                let ivs: Vec<Interval<u32>> = (0..cell.starts.len())
                    .map(|k| Interval::new(cell.starts[k], cell.ends[k] - 1, k as u32))
                    .collect();
                BasicCOITree::new(&ivs)
            })
            .collect()
    }

    /// Count interval overlaps for each (region, sample) pair.
    ///
    /// Returns a 2-D array of shape `(n_regions, n_sel)` where entry `[ri, sj]`
    /// is the number of stored intervals that overlap `[q_starts[ri], q_ends[ri])`.
    ///
    /// # Panics
    ///
    /// Panics if `chrom_code >= n_contigs` (negative codes are treated as an
    /// unknown contig and return zeros) or if any element of `sel_samples` is
    /// `>= n_samples`.  Callers (the Python `Table`) must pass factor-encoded
    /// codes derived from the Table's own sample / contig lists to satisfy
    /// these preconditions.
    pub fn count(
        &self,
        chrom_code: i32,
        q_starts: &[i32],
        q_ends: &[i32],
        sel_samples: &[i32],
    ) -> Array2<i32> {
        let n_regions = q_starts.len();
        let n_sel = sel_samples.len();
        let mut out = Array2::<i32>::zeros((n_regions, n_sel));
        if chrom_code < 0 {
            return out;
        }
        let trees = self.build_trees(chrom_code as usize);
        for (sj, &s) in sel_samples.iter().enumerate() {
            debug_assert!(
                (s as usize) < trees.len(),
                "sample code {s} out of range (n_samples={})",
                trees.len()
            );
            let tree = &trees[s as usize];
            for ri in 0..n_regions {
                let c = tree.query_count(q_starts[ri], q_ends[ri] - 1) as i32;
                out[[ri, sj]] = c;
            }
        }
        out
    }

    pub fn intervals_from_offsets(
        &self,
        chrom_code: i32,
        q_starts: &[i32],
        q_ends: &[i32],
        sel_samples: &[i32],
        offsets: &[i64],
    ) -> (Array2<i32>, Array1<f32>) {
        let n_total = *offsets.last().unwrap() as usize;
        let mut coords = Array2::<i32>::zeros((n_total, 2));
        let mut values = Array1::<f32>::zeros(n_total);
        if chrom_code < 0 || n_total == 0 {
            return (coords, values);
        }
        let chrom = chrom_code as usize;
        let trees = self.build_trees(chrom);
        let n_sel = sel_samples.len();
        for ri in 0..q_starts.len() {
            for (sj, &s) in sel_samples.iter().enumerate() {
                let cell_idx = ri * n_sel + sj;
                let base = offsets[cell_idx] as usize;
                let tree = &trees[s as usize];
                let mut idxs: Vec<u32> = Vec::new();
                tree.query(q_starts[ri], q_ends[ri] - 1, |iv| idxs.push(iv.metadata));
                // ascending stored index == ascending start (Table pre-sorted by start)
                idxs.sort_unstable();
                let cell = &self.store[chrom].samples[s as usize];
                for (j, &k) in idxs.iter().enumerate() {
                    let k = k as usize;
                    coords[[base + j, 0]] = cell.starts[k];
                    coords[[base + j, 1]] = cell.ends[k];
                    values[base + j] = cell.values[k];
                }
            }
        }
        (coords, values)
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

    fn brute_count(t: &RustTable, chrom: usize, qs: &[i32], qe: &[i32], sel: &[i32]) -> Array2<i32> {
        let mut out = Array2::<i32>::zeros((qs.len(), sel.len()));
        for (sj, &s) in sel.iter().enumerate() {
            let cell = &t.store[chrom].samples[s as usize];
            for (ri, (&rs, &re)) in qs.iter().zip(qe).enumerate() {
                let mut n = 0;
                for k in 0..cell.starts.len() {
                    if cell.starts[k] < re && cell.ends[k] > rs {
                        n += 1;
                    }
                }
                out[[ri, sj]] = n;
            }
        }
        out
    }

    #[test]
    fn count_matches_brute_force() {
        let t = toy();
        let qs = [0i32, 55, 5];
        let qe = [15i32, 65, 55];
        let sel = [0i32, 1];
        let got = t.count(0, &qs, &qe, &sel);
        let exp = brute_count(&t, 0, &qs, &qe, &sel);
        assert_eq!(got, exp);
    }

    #[test]
    fn count_unknown_contig_is_zeros() {
        let t = toy();
        let got = t.count(-1, &[0i32], &[10i32], &[0i32]);
        assert_eq!(got, Array2::<i32>::zeros((1, 1)));
    }

    fn offsets_from_count(counts: &Array2<i32>) -> Vec<i64> {
        let mut v = vec![0i64];
        let mut acc = 0i64;
        for r in 0..counts.nrows() {
            for s in 0..counts.ncols() {
                acc += counts[[r, s]] as i64;
                v.push(acc);
            }
        }
        v
    }

    #[test]
    fn intervals_from_offsets_ordered_and_correct() {
        let t = toy();
        let qs = [0i32, 5];
        let qe = [60i32, 55];
        let sel = [0i32, 1];
        let counts = t.count(0, &qs, &qe, &sel);
        let offsets = offsets_from_count(&counts);
        let (coords, vals) = t.intervals_from_offsets(0, &qs, &qe, &sel, &offsets);

        // cell (region0, sample0): intervals [0,10) v1 and [50,60) v2, sorted by start
        assert_eq!(coords[[0, 0]], 0);
        assert_eq!(coords[[0, 1]], 10);
        assert_eq!(vals[0], 1.0);
        assert_eq!(coords[[1, 0]], 50);
        assert_eq!(coords[[1, 1]], 60);
        assert_eq!(vals[1], 2.0);
        // total length equals sum of counts
        assert_eq!(vals.len(), counts.sum() as usize);
    }
}
