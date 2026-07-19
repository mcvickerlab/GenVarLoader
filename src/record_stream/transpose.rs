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

#[derive(Default)]
pub struct DecodedWindow {
    pub v_starts: Vec<i32>,
    pub ilens: Vec<i32>,
    pub alt_alleles: Vec<u8>,
    pub alt_offsets: Vec<i64>,
    pub geno_v_idxs: Vec<i32>,
    pub geno_offsets: Vec<i64>,
}

/// Fill `slot` (reusing its allocations) from a window's `DenseChunk`. The static table
/// copies straight across; the genotype transpose walks haps in C-order and pushes each
/// carried variant's COLUMN INDEX (into the static table), building a per-hap CSR.
pub fn fill_decoded_window(
    chunk: &DenseChunk,
    n_samples: usize,
    ploidy: usize,
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
    use genoray_core::types::BitGrid3;

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
            ilens: vec![0, 0],
            alt: vec![b'A', b'C'],
            alt_offsets: vec![0, 1, 2],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
        }
    }

    #[test]
    fn transpose_emits_variant_indices_hap_major() {
        let _guard = FILLER_TEST_LOCK
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let chunk = dense_fixture();
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, &mut slot);

        assert_eq!(slot.v_starts, vec![10, 20]);
        assert_eq!(slot.ilens, vec![0, 0]);
        assert_eq!(slot.alt_alleles, vec![b'A', b'C']);
        assert_eq!(slot.alt_offsets, vec![0, 1, 2]);
        // CSR over 4 haps: hap0=[0], hap1=[], hap2=[0,1], hap3=[1]
        assert_eq!(slot.geno_offsets, vec![0, 1, 1, 3, 4]);
        assert_eq!(slot.geno_v_idxs, vec![0, 0, 1, 1]);
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
            ilens: vec![],
            alt: vec![],
            alt_offsets: vec![0],
            genos,
            info_staged: vec![],
            format_staged: vec![],
        };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 2, 2, &mut slot);
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
            ilens: vec![0],
            alt: vec![b'T'],
            alt_offsets: vec![0, 1],
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
        };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, 3, 2, &mut slot);

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
        fill_decoded_window(&chunk, 2, 2, &mut slot);
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
            ilens: vec![0; n_var],
            alt: vec![b'A'; n_var],
            alt_offsets: (0..=n_var as u32).collect(),
            genos,
            info_staged: Vec::new(),
            format_staged: Vec::new(),
        };
        let mut slot = DecodedWindow::default();
        fill_decoded_window(&chunk, n_samples, ploidy, &mut slot);
        assert_eq!(slot.geno_offsets, ref_offsets);
        assert_eq!(slot.geno_v_idxs, ref_idxs);
    }
}
