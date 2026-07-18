use std::collections::HashMap;

use genoray_core::svar1_query::{Svar1Reader, find_ranges, var_ranges};
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

static CSR_ENTRIES_TOUCHED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Test/observability hook: total CSR entries spanned by window reads, process-wide
/// (the streaming engine reads windows on a producer thread, not the caller's thread,
/// so this must accumulate across threads to stay visible). The #275 throughput gate
/// asserts this scales with the WINDOW, not the store — the pre-rewrite path inverted
/// the whole contig CSR per batch. Mirrors genoray's `search::search_tree_build_count`.
#[doc(hidden)]
pub fn csr_entries_touched() -> usize {
    CSR_ENTRIES_TOUCHED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Per-contig scalars. Three numbers — the big `v_starts`/`v_ends` arrays stay on the
/// Python side and cross per call as zero-copy `PyReadonlyArray1` borrows, so nothing
/// sample- or variant-scale is duplicated into Rust residency.
pub struct ContigMeta {
    /// This contig's first variant's GLOBAL id (contigs are contiguous in id space).
    pub contig_start: u32,
    pub n_local: usize,
    /// `max(v_ends - v_starts)` over the contig — genoray `var_ranges`'s convention.
    /// An over-estimate of `overlap_range`'s `>=` bound, which is overshoot-safe.
    pub max_v_len: u32,
}

/// Opened once; holds ONE `Svar1Reader` for the store's lifetime (an SVAR1 store is a
/// single flat directory — no per-contig readers, unlike `Svar2Store`) plus per-contig
/// scalars. Converges on the `Svar2Store` shape and ends up smaller.
#[pyclass]
pub struct Svar1Store {
    reader: Svar1Reader,
    contigs: HashMap<String, ContigMeta>,
}

impl Svar1Store {
    /// Opens the store's mmap'd CSR. Used by tests + `#[new]`.
    pub fn open_meta(store_path: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let reader = Svar1Reader::open(store_path, n_samples, ploidy)
            .map_err(|e| PyIOError::new_err(format!("open svar store {store_path}: {e}")))?;
        Ok(Self {
            reader,
            contigs: HashMap::new(),
        })
    }

    pub fn reader(&self) -> &Svar1Reader {
        &self.reader
    }

    /// Zero-copy borrow of the sparse genotype variant-index array -- the
    /// `geno_v_idxs` handed to the reconstruction kernel by
    /// `svar1_generate_batch` (`src/ffi/mod.rs`). MUST stay a borrow of the
    /// reader's mmap'd `variant_idxs`, never an owned copy: a per-window `.to_vec()`
    /// here would silently reintroduce the sample-scale materialization the #275
    /// rewrite exists to avoid. That regression is invisible to an RSS high-water-mark
    /// test (the copy is a few KB -- far below `ru_maxrss`'s page-granularity noise
    /// floor) but caught deterministically by `geno_v_idxs_borrows_the_mmap_not_a_copy`
    /// below via pointer identity.
    pub fn geno_v_idxs(&self) -> &[i32] {
        self.reader.variant_idxs()
    }

    pub fn ploidy(&self) -> usize {
        self.reader.ploidy()
    }

    pub fn meta(&self, contig: &str) -> Option<&ContigMeta> {
        self.contigs.get(contig)
    }

    /// Rust-side setter (the `#[pymethods]` one delegates here) so unit tests can
    /// register a contig without a Python interpreter.
    pub fn set_contig_meta_rs(
        &mut self,
        contig: &str,
        contig_start: u32,
        n_local: usize,
        max_v_len: u32,
    ) {
        self.contigs.insert(
            contig.to_string(),
            ContigMeta {
                contig_start,
                n_local,
                max_v_len,
            },
        );
    }

    /// Read ONE window: `regions x samples x ploidy`, cartesian.
    ///
    /// `v_starts_c`/`v_ends_c` are this contig's LOCAL 0-based starts (ascending) and
    /// exclusive ends (`v_end = POS - min(ILEN, 0)`), borrowed from the caller's numpy
    /// arrays. `regions` are 0-based half-open on `contig`; `samples` are absolute
    /// sample indices.
    ///
    /// Two binary-search stages, no walk: `var_ranges` (POS -> global variant ids, one
    /// search tree for the whole window) then `find_ranges` (ids -> absolute CSR index
    /// pairs, two `partition_point`s per hap). Returns offsets only —
    /// `geno_v_idxs` is `self.reader().variant_idxs()`, borrowed by the caller.
    pub fn read_window(
        &self,
        contig: &str,
        v_starts_c: &[u32],
        v_ends_c: &[u32],
        regions: &[(u32, u32)],
        samples: &[usize],
    ) -> anyhow::Result<super::Svar1Window> {
        let m = self
            .contigs
            .get(contig)
            .ok_or_else(|| anyhow::anyhow!("no contig metadata registered for {contig}"))?;

        if v_starts_c.len() != m.n_local || v_ends_c.len() != m.n_local {
            anyhow::bail!(
                "read_window: contig {contig} has n_local={} but got v_starts={} v_ends={}",
                m.n_local,
                v_starts_c.len(),
                v_ends_c.len()
            );
        }

        let ranges = var_ranges(v_starts_c, v_ends_c, m.max_v_len, m.contig_start, regions);
        let b = find_ranges(&self.reader, &ranges, Some(samples));

        // Observability: entries this window actually spans. See `csr_entries_touched`.
        let spanned: usize = b
            .starts
            .iter()
            .zip(&b.stops)
            .map(|(s, e)| (e - s) as usize)
            .sum();
        CSR_ENTRIES_TOUCHED.fetch_add(spanned, std::sync::atomic::Ordering::Relaxed);

        // `find_ranges` emits C-order (region, sample, ploid), so batch row
        // bi = ri * n_samples + si and CSR row = bi * ploidy + p — an identity map.
        // Callers rebuild this locally over their batch slice (see
        // `svar1_generate_batch`); no need to materialize it window-scale here.

        Ok(super::Svar1Window {
            o_starts: b.starts,
            o_stops: b.stops,
        })
    }
}

#[pymethods]
impl Svar1Store {
    /// Open the SVAR1 store at `store_path`. `n_samples`/`ploidy` must match the
    /// store's `offsets.npy` length (`n_samples * ploidy + 1`) — a mismatch errors here
    /// rather than indexing out of bounds later.
    #[new]
    fn new(store_path: &str, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        Self::open_meta(store_path, n_samples, ploidy)
    }

    fn n_samples(&self) -> usize {
        self.reader.n_samples()
    }

    #[pyo3(name = "ploidy")]
    fn ploidy_py(&self) -> usize {
        self.reader.ploidy()
    }

    /// Register a contig's scalars: its first variant's GLOBAL id, its variant count,
    /// and `max(v_ends - v_starts)`. Three numbers — no arrays cross here.
    #[pyo3(name = "set_contig_meta")]
    fn set_contig_meta(&mut self, contig: &str, contig_start: u32, n_local: usize, max_v_len: u32) {
        self.set_contig_meta_rs(contig, contig_start, n_local, max_v_len);
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    #[test]
    fn open_missing_store_is_err() {
        let err = super::Svar1Store::open_meta("/no/such/svar", 2, 2);
        assert!(err.is_err());
    }

    #[test]
    fn geno_v_idxs_borrows_the_mmap_not_a_copy() {
        // The scale-guard defect this catches: `Svar1Store::geno_v_idxs()` (or the
        // `svar1_generate_batch` call site that uses it) reintroducing an
        // owned copy via `.to_vec()`. Pointer identity fails deterministically on
        // that regression; an RSS high-water-mark test cannot -- the copy is a few
        // KB, far below `ru_maxrss`'s page-granularity noise floor, so it never moves
        // the measurement (see the deleted Python RSS test this replaces).
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 0, 1, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 3, 4]);
        let store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();

        let a = store.geno_v_idxs();
        let b = store.reader().variant_idxs();
        assert_eq!(
            a.as_ptr(),
            b.as_ptr(),
            "geno_v_idxs() must be the SAME allocation as reader().variant_idxs() \
             (pointer-identical), not a value-equal copy"
        );
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn read_window_is_cartesian_and_borrows_the_mmap() {
        // 2 samples x ploidy 2 = 4 haps. Global ids 0..2 on one contig at
        // contig_start 0. Per-hap sorted global ids:
        //   hap0: [0]   hap1: []   hap2: [0, 1]   hap3: [1]
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 0, 1, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 3, 4]);

        let mut store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        // var0 SNP@10 (v_end 11), var1 SNP@20 (v_end 21); max_v_len = 1
        store.set_contig_meta_rs("chr1", 0, 2, 1);

        let v_starts_c: Vec<u32> = vec![10, 20];
        let v_ends_c: Vec<u32> = vec![11, 21];

        // One region [0, 30) covering both variants, both samples.
        let w = store
            .read_window("chr1", &v_starts_c, &v_ends_c, &[(0, 30)], &[0, 1])
            .unwrap();

        // batch = 1 region * 2 samples = 2 rows; 2 rows * ploidy 2 = 4 CSR rows.
        assert_eq!(w.o_starts.len(), 4);
        // hap0 -> [0,1); hap1 -> [1,1) empty; hap2 -> [1,3); hap3 -> [3,4)
        assert_eq!(w.o_starts, vec![0, 1, 1, 3]);
        assert_eq!(w.o_stops, vec![1, 1, 3, 4]);
    }

    #[test]
    fn read_window_unknown_contig_is_err() {
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 1, 1]);
        let store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        assert!(store.read_window("nope", &[], &[], &[(0, 10)], &[0]).is_err());
    }

    #[test]
    fn read_window_empty_contig_yields_all_empty_rows() {
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[] as &[i32]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 0, 0, 0, 0]);
        let mut store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        store.set_contig_meta_rs("chr1", 0, 0, 0);
        let w = store.read_window("chr1", &[], &[], &[(0, 30)], &[0, 1]).unwrap();
        for (s, e) in w.o_starts.iter().zip(&w.o_stops) {
            assert_eq!(s, e, "empty contig must give in-bounds zero-length rows");
        }
    }

    #[test]
    fn read_window_offsets_are_absolute_and_row_major() {
        // Same 4-hap fixture as read_window_is_cartesian_and_borrows_the_mmap.
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 0, 1, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 3, 4]);
        let mut store = super::Svar1Store::open_meta(tmp.path().to_str().unwrap(), 2, 2).unwrap();
        store.set_contig_meta_rs("chr1", 0, 2, 1);
        let w = store
            .read_window("chr1", &[10, 20], &[11, 21], &[(0, 30)], &[0, 1])
            .unwrap();
        // A batch [lo=1, hi=2) (row 1 only) selects CSR rows [1*2 .. 2*2) = [2, 4):
        // o_starts[2..4] = [1, 3], o_stops[2..4] = [3, 4]  -> hap2, hap3.
        assert_eq!(&w.o_starts[2..4], &[1, 3]);
        assert_eq!(&w.o_stops[2..4], &[3, 4]);
    }
}
