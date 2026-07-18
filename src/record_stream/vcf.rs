//! `VcfWindowFiller` — decode one VCF window into a `DecodedWindow` (issue #276 task 4).
//!
//! Drives genoray's `VcfRecordSource -> ChunkAssembler` pipeline for a single job's
//! region set on one contig, then runs the Task 2 transpose (`fill_decoded_window`).
//! VCF and PGEN (Task 10) differ ONLY in this filler; `engine.rs`'s `RecordBackend`
//! is source-agnostic (see that module's `WindowFiller` trait).
//!
//! ## Normalization config (parity with `gvl.write`'s VCF path)
//!
//! `gvl.write` reads VCF genotypes through genoray's *Python* `VCF` class
//! (`python/genvarloader/_dataset/_write.py::_write_from_vcf` ->
//! `_vcf_region_chunks` -> `VCF.chunk` / `VCF._chunk_ranges_with_length`,
//! `.pixi/envs/dev/lib/python3.10/site-packages/genoray/_vcf.py`), a thin
//! `cyvcf2` wrapper whose `__init__` (`_vcf.py:188`) takes no reference/FASTA
//! parameter at all. `gvl.write` constructs it as plain `VCF(variants)`
//! (`_write.py:228`) — no left-alignment or REF/FASTA check happens at read
//! time; users are expected to pre-normalize upstream with
//! `bcftools norm -f ref --normalize --ref-from-fa` (`docs/format.md`,
//! `docs/write.md`). This filler matches that: **`fasta_path: None`** by
//! default (a real path may be threaded through later, e.g. if Task 5/6 wants
//! to opt into read-time left-alignment for un-normalized inputs — that would
//! be a deliberate deviation from today's write-path parity, not the default).
//! `ChunkAssembler::new`'s `has_reference` gate (`genoray_core::chunk_assembler
//! ::decompose_raw_record`) skips BOTH `left_align` and `apply_check_ref` when
//! `fasta_path` is `None`, so `check_ref` is inert in the default config; a
//! value is still required by the signature, so `CheckRef::Exclude` is passed
//! (the more permissive of the two — matches `bcftools norm --check-ref x`,
//! the exclude-not-abort convention `docs/write.md` points users toward).
//!
//! **`OverlapMode::Variant`** (extent-overlap, no POS filter) matches htslib's
//! standard tabix/CSI region-fetch semantics that cyvcf2's
//! `vcf(f"{c}:{s}-{e}")` region query uses under the hood: a deletion whose
//! POS is upstream of the query start but whose extent reaches into it is
//! still returned. This is also exactly what `engine.rs`'s
//! `RecordBackend::generate` doc comment requires — a spanning upstream DEL
//! must be present in the decoded table for the kernel to clip it per-row.
//!
//! No INFO/FORMAT fields are requested (`fields: &[]`) — haplotype
//! reconstruction only needs POS/ILEN/ALT/GT.
//!
//! ## Single-chunk invariant
//!
//! A window's variant count is bounded (region size x local variant density),
//! so `chunk_size` is sized generously (`DEFAULT_CHUNK_SIZE`) to make one
//! `read_next_chunk` call cover the whole window; `fill` asserts this by
//! checking a second call drains to `None` with no variants, and returns an
//! error (not a silent truncation) if it doesn't. This lets `fill` skip the
//! `concat_dense`-across-chunks machinery the sketch anticipated.

use anyhow::ensure;

use genoray_core::chunk_assembler::ChunkAssembler;
use genoray_core::field::FieldSpec;
use genoray_core::normalize::CheckRef;
use genoray_core::svar2_view::OverlapMode;
use genoray_core::types::{BitGrid3, DenseChunk};
use genoray_core::vcf_reader::VcfRecordSource;

use crate::record_stream::engine::{ContigRef, RecordJob, WindowFiller};
use crate::record_stream::{fill_decoded_window, DecodedWindow};

/// Upper bound on a single window's variant count. See the module doc's
/// "Single-chunk invariant" section.
const DEFAULT_CHUNK_SIZE: usize = 1 << 20;

/// Decodes one VCF window via genoray's record-stream pipeline. Holds only a
/// path/strings/config (no file handles), so it is trivially `Send + Sync` —
/// `fill` opens a fresh `VcfRecordSource` per call.
pub struct VcfWindowFiller {
    vcf_path: String,
    /// Full cohort sample names, in `s_lo`/`s_hi`-indexable order. A job's
    /// `fill` call slices `[job.s_lo..job.s_hi]` out of this list.
    samples: Vec<String>,
    ploidy: usize,
    fasta_path: Option<String>,
    check_ref: CheckRef,
    overlap: OverlapMode,
    fields: Vec<FieldSpec>,
    htslib_threads: usize,
    chunk_size: usize,
}

impl VcfWindowFiller {
    /// `fasta_path: None` matches `gvl.write`'s VCF normalization (no
    /// read-time left-alignment) — see the module doc. Pass `Some(path)` only
    /// to deliberately deviate from that parity contract.
    pub fn new(
        vcf_path: &str,
        samples: &[&str],
        ploidy: usize,
        fasta_path: Option<&str>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            vcf_path: vcf_path.to_string(),
            samples: samples.iter().map(|s| s.to_string()).collect(),
            ploidy,
            fasta_path: fasta_path.map(str::to_string),
            check_ref: CheckRef::Exclude,
            overlap: OverlapMode::Variant,
            fields: Vec::new(),
            htslib_threads: 1,
            chunk_size: DEFAULT_CHUNK_SIZE,
        })
    }

    /// Override the single-chunk size bound (tests only need a handful of
    /// variants; production callers may want to tune this against expected
    /// window density).
    #[cfg(test)]
    fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}

impl WindowFiller for VcfWindowFiller {
    fn fill(
        &self,
        job: &RecordJob,
        contig: &ContigRef,
        slot: &mut DecodedWindow,
    ) -> anyhow::Result<()> {
        let local_samples = &self.samples[job.s_lo..job.s_hi];
        let sample_refs: Vec<&str> = local_samples.iter().map(String::as_str).collect();
        let n_local_samples = sample_refs.len();

        let source = VcfRecordSource::new(
            &self.vcf_path,
            &contig.name,
            &sample_refs,
            self.htslib_threads,
            self.ploidy,
            &self.fields,
            job.regions.clone(),
            self.overlap,
        )?;
        let mut asm = ChunkAssembler::new(
            Box::new(source),
            n_local_samples,
            self.ploidy,
            self.fasta_path.as_deref(),
            &contig.name,
            /* skip_out_of_scope */ true,
            self.check_ref,
            &self.fields,
        )?;

        match asm.read_next_chunk(self.chunk_size, 0, None)? {
            Some(chunk) => {
                // Single-chunk invariant (see module doc): a window must fit in one
                // `read_next_chunk` call. If it didn't, the chunk above is a silent
                // truncation of the real window — fail loudly instead of decoding a
                // partial table.
                ensure!(
                    asm.read_next_chunk(self.chunk_size, 1, None)?.is_none(),
                    "VcfWindowFiller single-chunk invariant violated: window on {} \
                     regions {:?} has more variants than chunk_size ({}) — raise \
                     DEFAULT_CHUNK_SIZE or shrink the job's regions",
                    contig.name,
                    job.regions,
                    self.chunk_size,
                );
                fill_decoded_window(&chunk, n_local_samples, self.ploidy, slot);
            }
            None => {
                // Empty window (no variants in range): decode an empty `DenseChunk`
                // through the same transpose path, so the "empty" contract stays in
                // exactly one place (`fill_decoded_window`'s all-zero-CSR behavior,
                // covered by `transpose::tests::transpose_empty_chunk_is_all_zero_csr`).
                let empty = DenseChunk {
                    chunk_id: 0,
                    pos: Vec::new(),
                    ilens: Vec::new(),
                    alt: Vec::new(),
                    alt_offsets: vec![0],
                    genos: BitGrid3::zeros(0, n_local_samples, self.ploidy),
                    info_staged: Vec::new(),
                    format_staged: Vec::new(),
                };
                fill_decoded_window(&empty, n_local_samples, self.ploidy, slot);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_path() -> String {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/streaming/two_var_two_sample.vcf.gz")
            .to_str()
            .unwrap()
            .to_string()
    }

    /// Fixture: `tests/data/streaming/two_var_two_sample.vcf.gz` (+ `.tbi`), 2 samples
    /// (s1, s2) on `chr1`:
    ///   chr1  POS=11 (1-based)  REF=A     ALT=G     GT: s1=0|1  s2=0|0   (SNP)
    ///   chr1  POS=21 (1-based)  REF=ACGT  ALT=A     GT: s1=0|0  s2=1|1   (DEL)
    ///
    /// Expected atomization (`genoray_core::normalize::atomize_biallelic`):
    ///   - SNP: 0-based pos = 10, ilen = 0, alt = 'G' (no boundary/anchor games for
    ///     a pure 1bp substitution).
    ///   - DEL: REF=ACGT (rlen=4) vs ALT=A (alen=1); shared-suffix trim can't apply
    ///     (both ends already differ), so k = min(rlen,alen)-1 = 0. Boundary at k=0:
    ///     alen<rlen => deletion branch, del_ref_len = rlen-k = 4,
    ///     ilen = 1-4 = -3. r[0]='A' == a[0]='A' => clean anchor (pure DEL):
    ///     pos = POS+k = 20 (0-based), alt = [r[0]] = 'A'.
    ///   Atoms are heap-ordered by (pos, seq), so v0=SNP@10, v1=DEL@20.
    ///
    /// Expected per-hap CSR (hap-major s1p0,s1p1,s2p0,s2p1; `fill_decoded_window`
    /// scans variants ascending per hap): s1 GT for v0 is 0|1 (ALT on p1 only), 0|0
    /// for v1 (neither hap); s2 GT is 0|0 for v0 (neither hap), 1|1 for v1 (both
    /// haps). So hap0(s1p0)=[], hap1(s1p1)=[0], hap2(s2p0)=[1], hap3(s2p1)=[1].
    #[test]
    fn vcf_filler_decodes_window_to_local_table() {
        let filler = VcfWindowFiller::new(&fixture_path(), &["s1", "s2"], 2, None).unwrap();
        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 100)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 100],
        };
        let mut slot = DecodedWindow::default();
        filler.fill(&job, &contig, &mut slot).unwrap();

        assert_eq!(slot.v_starts, vec![10, 20]);
        assert_eq!(slot.ilens, vec![0, -3]);
        assert_eq!(slot.alt_alleles, vec![b'G', b'A']);
        assert_eq!(slot.alt_offsets, vec![0, 1, 2]);
        assert_eq!(slot.geno_v_idxs, vec![0, 1, 1]);
        assert_eq!(slot.geno_offsets, vec![0, 0, 1, 2, 3]);
    }

    /// A region with no variants must decode to the empty-CSR state (all-zero
    /// offsets over `n_samples*ploidy` haps, no variants), not an error.
    #[test]
    fn vcf_filler_empty_window_is_all_zero_csr() {
        let filler = VcfWindowFiller::new(&fixture_path(), &["s1", "s2"], 2, None).unwrap();
        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(500, 600)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 1000],
        };
        let mut slot = DecodedWindow::default();
        filler.fill(&job, &contig, &mut slot).unwrap();

        assert!(slot.v_starts.is_empty());
        assert!(slot.ilens.is_empty());
        assert!(slot.alt_alleles.is_empty());
        assert_eq!(slot.alt_offsets, vec![0]);
        assert!(slot.geno_v_idxs.is_empty());
        assert_eq!(slot.geno_offsets, vec![0, 0, 0, 0, 0]);
    }

    /// A job that only selects a sample sub-range (`s_lo`/`s_hi` narrower than the
    /// full cohort) must decode ONLY those samples, in `s_lo`-order — proving the
    /// filler honors sample-scoping generally, not just the full-cohort case the
    /// other tests exercise.
    #[test]
    fn vcf_filler_honors_sample_subrange() {
        let filler = VcfWindowFiller::new(&fixture_path(), &["s1", "s2"], 2, None).unwrap();
        // s_lo=1,s_hi=2 selects only s2.
        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 100)],
            s_lo: 1,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 100],
        };
        let mut slot = DecodedWindow::default();
        filler.fill(&job, &contig, &mut slot).unwrap();

        assert_eq!(slot.v_starts, vec![10, 20]);
        // s2 alone: hap0(p0)=[1] hap1(p1)=[1] (GT 0|0 for v0, 1|1 for v1).
        assert_eq!(slot.geno_v_idxs, vec![1, 1]);
        assert_eq!(slot.geno_offsets, vec![0, 1, 2]);
    }

    /// The single-chunk invariant: a `chunk_size` too small for the window's
    /// variant count must fail loudly (an error), never silently truncate.
    #[test]
    fn vcf_filler_errors_when_window_exceeds_chunk_size() {
        let filler = VcfWindowFiller::new(&fixture_path(), &["s1", "s2"], 2, None)
            .unwrap()
            .with_chunk_size(1);
        let job = RecordJob {
            contig_idx: 0,
            regions: vec![(0, 100)],
            s_lo: 0,
            s_hi: 2,
        };
        let contig = ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'A'; 100],
        };
        let mut slot = DecodedWindow::default();
        let err = filler.fill(&job, &contig, &mut slot).unwrap_err();
        assert!(
            format!("{err:?}").contains("single-chunk invariant"),
            "expected the single-chunk guard to fire, got: {err:?}"
        );
    }
}
