use std::sync::atomic::{AtomicUsize, Ordering};

use genoray_core::types::DenseChunk;

/// Process-wide count of genotype-grid word reads performed by the transpose
/// (see `crate::svar1::store::csr_entries_touched` for the pattern). Deterministic
/// given the input grid; the perf gate on this noisy node, not wall-clock.
static WORD_READS: AtomicUsize = AtomicUsize::new(0);

/// Process-wide, test-only lock serializing every test (in this module and in
/// `crate::record_stream::vcf` / `crate::record_stream::pgen`) that calls a filler's
/// `fill` (hence `fill_decoded_window`, which bumps `WORD_READS`) or
/// `PgenWindowFiller::fill` (which bumps `crate::record_stream::pgen::VARIANTS_DECODED`).
/// Both counters are process-wide `AtomicUsize`s and cargo runs same-binary tests
/// concurrently by default, so the tests asserting on their absolute values
/// (`word_reads_counter_tracks_transpose_work`, `variants_decoded_counter_tracks_range_width`)
/// need exclusive access to *every* path that can bump either counter, not just the ones
/// in their own module — hence one shared lock rather than one per module/counter.
#[cfg(test)]
pub(crate) static FILLER_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub fn word_reads() -> usize {
    WORD_READS.load(Ordering::Relaxed)
}
pub fn word_reads_reset() {
    WORD_READS.store(0, Ordering::Relaxed);
}

/// Per-variant staged INFO values for one requested field. genoray's `StagedColumn`
/// has exactly two variants, so "arbitrary dtype" collapses to these two.
#[derive(Debug, Clone, PartialEq)]
pub enum InfoVals {
    I32(Vec<i32>),
    F32(Vec<f32>),
}

impl InfoVals {
    pub fn len(&self) -> usize {
        match self {
            InfoVals::I32(v) => v.len(),
            InfoVals::F32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named per-variant INFO column carried on the window (Wave B PR-B3a).
#[derive(Debug, Clone, PartialEq)]
pub struct InfoCol {
    pub name: String,
    pub values: InfoVals,
}

#[derive(Default)]
pub struct DecodedWindow {
    pub v_starts: Vec<i32>,
    pub ilens: Vec<i32>,
    pub alt_alleles: Vec<u8>,
    pub alt_offsets: Vec<i64>,
    pub geno_v_idxs: Vec<i32>,
    pub geno_offsets: Vec<i64>,
    /// Per-variant dataset-global id, parallel to `v_starts` (i.e. `global_v_idxs[i]`
    /// is the global id of the variant at window-local column `i`). Copied verbatim
    /// from `chunk.global_idx` by `fill_decoded_window` (every call, including the
    /// empty-window path, so this recycled slot never leaks a stale value). Consumed
    /// by `generate_batch_core` (via `RecordBackend::generate`), which gathers each
    /// window-local annotation id through this array to its dataset-global id — see
    /// [`crate::ffi::remap_annot_local_to_global`].
    pub global_v_idxs: Vec<i32>,
    /// Per-variant AF, parallel to `v_starts` (Wave B PR-B2, #319). Copied from
    /// `chunk.info_staged[0]` (the single requested `AF` INFO field) when present,
    /// else left EMPTY. `afs.is_empty()` ⇒ no AF filter (matches SVAR1's `afs: None`
    /// no-op and the PGEN path, whose `info_staged` is always empty).
    pub afs: Vec<f32>,
    /// Requested non-AF INFO columns, one per name passed to `fill_decoded_window`,
    /// in request order. Each has one value per window variant. Empty when no
    /// `var_fields` INFO column was requested (Wave B PR-B3a).
    pub info_cols: Vec<InfoCol>,
}

/// Fill `slot` (reusing its allocations) from a window's `DenseChunk`. The static table
/// copies straight across; the genotype transpose walks haps in C-order and pushes each
/// carried variant's COLUMN INDEX (into the static table), building a per-hap CSR.
///
/// `want_af` and `info_names` describe how `chunk.info_staged` was populated: AF (when
/// requested at all) is always staged first (`info_staged[0]`), and the `info_names`
/// fields follow it in request order (see the `afs`/`info_cols` field docs above).
pub fn fill_decoded_window(
    chunk: &DenseChunk,
    n_samples: usize,
    ploidy: usize,
    want_af: bool,
    info_names: &[String],
    slot: &mut DecodedWindow,
) {
    let n_var = chunk.pos.len();

    slot.v_starts.clear();
    slot.v_starts.extend(chunk.pos.iter().map(|&p| p as i32));
    slot.ilens.clear();
    slot.ilens.extend_from_slice(&chunk.ilens);
    slot.alt_alleles.clear();
    slot.alt_alleles.extend_from_slice(&chunk.alt);
    slot.alt_offsets.clear();
    slot.alt_offsets
        .extend(chunk.alt_offsets.iter().map(|&o| o as i64));
    slot.global_v_idxs.clear();
    slot.global_v_idxs.extend_from_slice(&chunk.global_idx);

    // AF occupies info_staged[0] iff it was requested (PR-B2); the var_fields INFO
    // columns follow it in request order (PR-B3a).
    slot.afs.clear();
    let af_offset = if want_af {
        if let Some(genoray_core::types::StagedColumn::Float(v)) = chunk.info_staged.first() {
            slot.afs.extend_from_slice(v);
        }
        1
    } else {
        0
    };

    slot.info_cols.clear();
    for (i, name) in info_names.iter().enumerate() {
        let values = match chunk.info_staged.get(af_offset + i) {
            Some(genoray_core::types::StagedColumn::Int(v)) => InfoVals::I32(v.clone()),
            Some(genoray_core::types::StagedColumn::Float(v)) => InfoVals::F32(v.clone()),
            None => continue,
        };
        slot.info_cols.push(InfoCol {
            name: name.clone(),
            values,
        });
    }

    // Word-level two-pass counting-sort transpose. Instead of calling get_bit once per
    // (v,s,p) cell (V*S*P reads), walk the packed `words` in flat v-major order — this is
    // sequential/cache-friendly and, since genotypes are sparse, skips whole empty words
    // via `trailing_zeros`. Two passes over `words` (count per hap, then fill at per-hap
    // cursors) replace the single hap-major cell scan.
    let plane = n_samples * ploidy; // hap stride; hap = s*ploidy + p = flat % plane
    let n_haps = plane;
    let total_bits = n_var * plane;
    let words = &chunk.genos.words;

    // Pass 1: count set bits per hap (offsets[hap+1] holds the count, then prefix-sum).
    slot.geno_offsets.clear();
    slot.geno_offsets.resize(n_haps + 1, 0);
    for (w_idx, &word) in words.iter().enumerate() {
        WORD_READS.fetch_add(1, Ordering::Relaxed);
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let flat = w_idx * 64 + b;
            if flat >= total_bits {
                continue; // padding bit in the final word (genoray leaves these 0, defensive)
            }
            let hap = flat % plane;
            slot.geno_offsets[hap + 1] += 1;
        }
    }
    for h in 0..n_haps {
        slot.geno_offsets[h + 1] += slot.geno_offsets[h];
    }
    let total = slot.geno_offsets[n_haps] as usize;

    // Pass 2: fill geno_v_idxs at per-hap cursors. v is the slowest-varying flat axis,
    // so iterating words in order yields ascending v per hap — the CSR contract.
    slot.geno_v_idxs.clear();
    slot.geno_v_idxs.resize(total, 0);
    let mut cursor: Vec<usize> = slot.geno_offsets[..n_haps]
        .iter()
        .map(|&o| o as usize)
        .collect();
    for (w_idx, &word) in words.iter().enumerate() {
        WORD_READS.fetch_add(1, Ordering::Relaxed);
        let mut bits = word;
        while bits != 0 {
            let b = bits.trailing_zeros() as usize;
            bits &= bits - 1;
            let flat = w_idx * 64 + b;
            if flat >= total_bits {
                continue;
            }
            let v = flat / plane;
            let hap = flat % plane;
            slot.geno_v_idxs[cursor[hap]] = v as i32;
            cursor[hap] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use genoray_core::types::{BitGrid3, StagedColumn};

    fn dense_fixture() -> DenseChunk {
        // 2 variants, 2 samples, ploidy 2. BitGrid3 dims (V, S, P), C-order flat index
        // is v*S*P + s*P + p.
        let mut genos = BitGrid3::zeros(2, 2, 2);
        genos.or_bit(0 * 2 * 2 + 0 * 2 + 0, true); // v0, sample0, ploid0 -> hap0
        genos.or_bit(0 * 2 * 2 + 1 * 2 + 0, true); // v0, sample1, ploid0 -> hap2
        genos.or_bit(1 * 2 * 2 + 1 * 2 + 0, true); // v1, sample1, ploid0 -> hap2
        genos.or_bit(1 * 2 * 2 + 1 * 2 + 1, true); // v1, sample1, ploid1 -> hap3
        DenseChunk {
            chunk_id: 0,
            pos: vec![10, 20],
            global_idx: vec![5, 8],
            ilens: vec![0, 0],
            alt: vec![b'A', b'C'],
            alt_offsets: vec![0, 1, 2],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
            carriers: None,
            format_by_carrier: None,
        }
    }

    #[test]
    fn transpose_emits_variant_indices_hap_major() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let chunk = dense_fixture();
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, false, &[], &mut slot);

        assert_eq!(slot.v_starts, vec![10, 20]);
        assert_eq!(slot.ilens, vec![0, 0]);
        assert_eq!(slot.alt_alleles, vec![b'A', b'C']);
        assert_eq!(slot.alt_offsets, vec![0, 1, 2]);
        // CSR over 4 haps: hap0=[0], hap1=[], hap2=[0,1], hap3=[1]
        assert_eq!(slot.geno_offsets, vec![0, 1, 1, 3, 4]);
        assert_eq!(slot.geno_v_idxs, vec![0, 0, 1, 1]);
        assert_eq!(slot.global_v_idxs, vec![5, 8]);
    }

    #[test]
    fn transpose_empty_chunk_is_all_zero_csr() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let genos = BitGrid3::zeros(0, 2, 2);
        let chunk = DenseChunk {
            chunk_id: 0,
            pos: vec![],
            global_idx: Vec::new(),
            ilens: vec![],
            alt: vec![],
            alt_offsets: vec![0],
            genos,
            info_staged: vec![],
            format_staged: vec![],
            carriers: None,
            format_by_carrier: None,
        };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, false, &[], &mut slot);
        assert!(slot.geno_v_idxs.is_empty());
        assert_eq!(slot.geno_offsets, vec![0, 0, 0, 0, 0]); // 4 haps, all empty
    }

    #[test]
    fn transpose_monomorphic_hap_all_bits_on_one_variant() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        // 1 variant, 3 samples, ploidy 2 = 6 haps. All 6 haps carry the single variant.
        let mut genos = BitGrid3::zeros(1, 3, 2);
        for s in 0..3 {
            for p in 0..2 {
                genos.or_bit(0 * 3 * 2 + s * 2 + p, true);
            }
        }
        let chunk = DenseChunk {
            chunk_id: 0,
            pos: vec![42],
            global_idx: vec![17],
            ilens: vec![0],
            alt: vec![b'T'],
            alt_offsets: vec![0, 1],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
            carriers: None,
            format_by_carrier: None,
        };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 3, 2, false, &[], &mut slot);

        assert_eq!(slot.v_starts, vec![42]);
        // Every hap carries variant 0.
        assert_eq!(slot.geno_offsets, vec![0, 1, 2, 3, 4, 5, 6]);
        assert_eq!(slot.geno_v_idxs, vec![0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn word_reads_counter_tracks_transpose_work() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        word_reads_reset();
        assert_eq!(word_reads(), 0);
        let chunk = dense_fixture(); // 2 var, 2 samples, ploidy 2
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, false, &[], &mut slot);
        // Metric changed from per-cell get_bit reads (V*S*P = 8) to per-word reads: the
        // word-level two-pass transpose walks chunk.genos.words twice (count pass +
        // fill pass), so the count is 2 * n_words. dense_fixture has 8 bits = 1 word.
        assert_eq!(word_reads(), 2 * chunk.genos.words.len());
    }

    #[test]
    fn word_transpose_matches_naive_on_sparse_grid() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let (n_var, n_samples, ploidy) = (37usize, 11usize, 2usize);
        let mut genos = BitGrid3::zeros(n_var, n_samples, ploidy);
        // Deterministic sparse pattern (~5% density).
        for v in 0..n_var {
            for s in 0..n_samples {
                for p in 0..ploidy {
                    if (v * 7 + s * 3 + p) % 19 == 0 {
                        genos.or_bit(v * n_samples * ploidy + s * ploidy + p, true);
                    }
                }
            }
        }
        // Reference: the old hap-major scan.
        let mut ref_idxs: Vec<i32> = Vec::new();
        let mut ref_offsets: Vec<i64> = vec![0];
        for s in 0..n_samples {
            for p in 0..ploidy {
                for v in 0..n_var {
                    if genos.get_bit(v * n_samples * ploidy + s * ploidy + p) {
                        ref_idxs.push(v as i32);
                    }
                }
                ref_offsets.push(ref_idxs.len() as i64);
            }
        }
        let chunk = DenseChunk {
            chunk_id: 0,
            pos: (0..n_var as u32).collect(),
            global_idx: (0..n_var as i32).collect(),
            ilens: vec![0; n_var],
            alt: vec![b'A'; n_var],
            alt_offsets: (0..=n_var as u32).collect(),
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
            carriers: None,
            format_by_carrier: None,
        };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, n_samples, ploidy, false, &[], &mut slot);
        assert_eq!(slot.geno_offsets, ref_offsets);
        assert_eq!(slot.geno_v_idxs, ref_idxs);
    }

    /// `info_staged[0]` (a `StagedColumn::Float`, the sole requested `AF` field)
    /// is copied verbatim into `slot.afs`, parallel to `v_starts`.
    #[test]
    fn afs_copied_from_info_staged_float_column() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut chunk = dense_fixture();
        chunk.info_staged = vec![StagedColumn::Float(vec![0.1f32, 0.9])];
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, true, &[], &mut slot);
        assert_eq!(slot.afs, vec![0.1f32, 0.9]);
    }

    /// No AF requested (or PGEN, which never stages INFO) ⇒ `info_staged` is
    /// empty ⇒ `slot.afs` stays empty (the no-filter no-op contract).
    #[test]
    fn afs_empty_when_info_staged_empty() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let chunk = dense_fixture(); // info_staged: Vec::new()
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, false, &[], &mut slot);
        assert!(slot.afs.is_empty());
    }

    /// Recycling a slot across windows must not leak a stale AF vector: filling
    /// once with a Float `info_staged`, then again with an empty one on the SAME
    /// slot, must clear `afs` (catches a missing `.clear()`).
    #[test]
    fn afs_cleared_on_recycle_across_windows() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut with_af = dense_fixture();
        with_af.info_staged = vec![StagedColumn::Float(vec![0.1f32, 0.9])];
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&with_af, 2, 2, true, &[], &mut slot);
        assert_eq!(slot.afs, vec![0.1f32, 0.9]);

        let without_af = dense_fixture(); // info_staged: Vec::new()
        fill_decoded_window(&without_af, 2, 2, false, &[], &mut slot);
        assert!(slot.afs.is_empty());
    }

    /// Requested non-AF INFO fields land in `info_cols` in request order, with the
    /// dtype genoray staged them as. `info_staged[0]` stays reserved for AF (PR-B2),
    /// so a chunk with AF + 2 extra fields yields `afs` plus 2 `info_cols`.
    #[test]
    fn info_cols_copied_from_staged_columns_in_request_order() {
        let mut chunk = dense_fixture();
        chunk.info_staged = vec![
            StagedColumn::Float(vec![0.1f32, 0.9]),
            StagedColumn::Int(vec![7i32, 8]),
            StagedColumn::Float(vec![1.5f32, 2.5]),
        ];
        let names = vec!["DP".to_string(), "QUAL2".to_string()];
        let mut w = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, true, &names, &mut w);
        assert_eq!(w.afs, vec![0.1f32, 0.9]);
        assert_eq!(w.info_cols.len(), 2);
        assert_eq!(w.info_cols[0].name, "DP");
        assert_eq!(w.info_cols[0].values, InfoVals::I32(vec![7, 8]));
        assert_eq!(w.info_cols[1].name, "QUAL2");
        assert_eq!(w.info_cols[1].values, InfoVals::F32(vec![1.5, 2.5]));
    }

    /// No AF requested: `info_staged[0]` is the first *requested* field, not AF.
    #[test]
    fn info_cols_start_at_zero_when_af_not_requested() {
        let mut chunk = dense_fixture();
        chunk.info_staged = vec![StagedColumn::Int(vec![3i32, 4])];
        let names = vec!["DP".to_string()];
        let mut w = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, false, &names, &mut w);
        assert!(w.afs.is_empty());
        assert_eq!(w.info_cols.len(), 1);
        assert_eq!(w.info_cols[0].values, InfoVals::I32(vec![3, 4]));
    }

    /// Slot recycling: a window with info columns then one without must not leak the
    /// previous window's columns (same defect class as the existing `afs` reuse test).
    #[test]
    fn info_cols_cleared_on_slot_reuse() {
        let mut with_cols = dense_fixture();
        with_cols.info_staged = vec![StagedColumn::Int(vec![3i32, 4])];
        let mut w = DecodedWindow::default();
        fill_decoded_window(&with_cols, 2, 2, false, &["DP".to_string()], &mut w);
        assert_eq!(w.info_cols.len(), 1);
        let without = dense_fixture();
        fill_decoded_window(&without, 2, 2, false, &[], &mut w);
        assert!(w.info_cols.is_empty());
    }
}
