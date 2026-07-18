//! `RecordStreamEngine` — record-stream (VCF/PGEN) producer/consumer engine (issue #276
//! task 3b).
//!
//! This is the second [`crate::ffi::stream_core::EngineBackend`] implementation, alongside
//! `Svar1Backend` (`crate::ffi::stream_engine`). Where SVAR1's producer thread reads a
//! pre-decoded on-disk CSR layout, this engine's producer *decodes* a window from a live
//! source (a VCF or PGEN reader) via a caller-supplied [`WindowFiller`] — the actual
//! genoray record-stream reading (Task 4 for VCF, Task 10 for PGEN) plugs in here without
//! this module knowing anything about either format. The filled window is a
//! [`DecodedWindow`] (`crate::record_stream::transpose`): a window-local static table
//! (`v_starts`/`ilens`/`alt_alleles`/`alt_offsets`) plus a window-local per-hap CSR
//! (`geno_v_idxs`/`geno_offsets`) built by `fill_decoded_window`'s hap-major transpose.
//!
//! The threading/shutdown/panic discipline (two bounded(2) channels, shutdown-by-drop,
//! join-then-classify) is entirely `stream_core`'s — read that module's doc comment. This
//! module supplies only the two divergence points: `fill` delegates to the `WindowFiller`,
//! and `generate` assembles a `generate_batch_core` call from the slot's window-local table
//! (mirroring `Svar1Backend::generate`, but sourcing `geno_v_idxs`/CSR offsets from the
//! *window* rather than a whole-store mmap — see the CSR-offset note on
//! [`RecordBackend::generate`]).
//!
//! **`WindowFiller: Send + Sync`, not `Send`-only.** `EngineBackend: Send + Sync` (see
//! `stream_core`'s doc comment: `Arc<B>` is shared between the producer and consumer
//! threads), so `RecordBackend` — which owns the filler — must be `Sync`, which requires
//! the boxed filler itself to be `Sync`. Both real fillers satisfy this: a VCF filler holds
//! a path/strings, and a PGEN filler's `Py<PyAny>` is `Send + Sync` (PyO3 GIL-guarded
//! handle).

use std::sync::Arc;

use ndarray::{Array1, Array2};

use crate::ffi::stream_core::{EngineBackend, StreamEngineCore};
use crate::record_stream::DecodedWindow;

/// Decode job `job`'s window (on `contig`) into `slot`, reusing its allocations. VCF/PGEN
/// implementors (Task 4/10) do the actual genoray record-stream reading + the hap-major
/// transpose (`fill_decoded_window`); this trait is the engine's only view of the source.
pub trait WindowFiller: Send + Sync {
    fn fill(
        &self,
        job: &RecordJob,
        contig: &ContigRef,
        slot: &mut DecodedWindow,
    ) -> anyhow::Result<()>;
}

/// One pre-expanded window of the fixed traversal: `regions` (0-based half-open, on
/// `contig_idx`) crossed with the contiguous sample sub-range `[s_lo, s_hi)`. Mirrors
/// SVAR1's `WindowJob` shape (see `stream_engine.rs`); sample-selection semantics are the
/// filler's responsibility (a VCF/PGEN filler resolves `[s_lo, s_hi)` against whatever
/// sample-index map it holds).
pub struct RecordJob {
    pub contig_idx: usize,
    pub regions: Vec<(u32, u32)>,
    pub s_lo: usize,
    pub s_hi: usize,
}

/// Per-contig data needed for generation: name (for the filler to resolve, e.g. VCF region
/// queries) and reference bytes (offsets are implicitly `[0, ref_bytes.len()]`, consumed by
/// `generate_batch_core` — see `Svar1Backend`'s per-contig-reference note for why this must
/// be per-contig, not a single engine-level reference).
pub struct ContigRef {
    pub name: String,
    pub ref_bytes: Vec<u8>,
}

/// The record-stream [`EngineBackend`]: owns the filler (the decode), the plan
/// (jobs/contigs), and the reconstruction knobs. Wrapped in an `Arc` by `StreamEngineCore`
/// and shared between the producer and consumer threads.
struct RecordBackend {
    filler: Box<dyn WindowFiller + Send + Sync>,
    contigs: Vec<ContigRef>,
    jobs: Vec<RecordJob>,
    /// Cohort sample count. Not read by `fill`/`generate` (a job's local sample count is
    /// `s_hi - s_lo`); kept for construction-time validation (`s_hi <= n_samples`) and for
    /// fillers that need the cohort size to interpret `[s_lo, s_hi)`.
    #[allow(dead_code)]
    n_samples: usize,
    ploidy: usize,
    pad_char: u8,
    parallel: bool,
}

impl EngineBackend for RecordBackend {
    type Slot = DecodedWindow;

    fn n_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// PRODUCER: decode this window via the filler.
    fn fill(&self, job_idx: usize, slot: &mut DecodedWindow) -> anyhow::Result<()> {
        let job = &self.jobs[job_idx];
        let c = &self.contigs[job.contig_idx];
        self.filler.fill(job, c, slot)
    }

    fn n_batch_rows(&self, job_idx: usize, _slot: &DecodedWindow) -> usize {
        let job = &self.jobs[job_idx];
        job.regions.len() * (job.s_hi - job.s_lo)
    }

    /// CONSUMER: generate the `[row_lo, row_hi)` slice of this window's rows. Delegates to
    /// the shared [`crate::ffi::generate_batch_core`], mirroring `Svar1Backend::generate`'s
    /// row->region mapping exactly (C-order (region, sample): window row
    /// `bi = ri*n_samples + si` -> `regions[bi/n_samples]`).
    ///
    /// **CSR-offset note.** SVAR1's `o_starts`/`o_stops` are absolute indices sliced out of
    /// a whole-store mmap. `DecodedWindow`'s per-hap CSR is instead a single window-local
    /// `geno_offsets` (length `n_haps + 1`): hap `h`'s run is
    /// `geno_v_idxs[geno_offsets[h]..geno_offsets[h+1]]`. For haps `[o_lo, o_hi)` that's
    /// `o_starts_b = geno_offsets[o_lo..o_hi]` and `o_stops_b = geno_offsets[o_lo+1..=o_hi]`
    /// — both built here as owned `Vec<i64>` since `generate_batch_core` wants matching
    /// start/stop slices, not one shared array.
    fn generate(
        &self,
        job_idx: usize,
        slot: &DecodedWindow,
        row_lo: usize,
        row_hi: usize,
    ) -> anyhow::Result<(Array1<u8>, Array1<i64>)> {
        let job = &self.jobs[job_idx];
        let c = &self.contigs[job.contig_idx];
        let n_samples = job.s_hi - job.s_lo;
        let n_rows = row_hi - row_lo;

        // Per (region, sample) row bounds for rows [row_lo, row_hi).
        let mut rb = Array2::<i32>::zeros((n_rows, 2));
        for (i, bi) in (row_lo..row_hi).enumerate() {
            let ri = bi / n_samples;
            let (s, e) = job.regions[ri];
            rb[[i, 0]] = s as i32;
            rb[[i, 1]] = e as i32;
        }

        // CSR rows for this batch slice: haps [row_lo*ploidy, row_hi*ploidy).
        let o_lo = row_lo * self.ploidy;
        let o_hi = row_hi * self.ploidy;
        let o_starts_b: Vec<i64> = slot.geno_offsets[o_lo..o_hi].to_vec();
        let o_stops_b: Vec<i64> = slot.geno_offsets[o_lo + 1..=o_hi].to_vec();

        // This contig's reference slice (offsets [0, len]).
        let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);

        Ok(crate::ffi::generate_batch_core(
            &slot.geno_v_idxs,
            self.ploidy,
            &o_starts_b,
            &o_stops_b,
            rb.view(),
            ndarray::ArrayView1::from(slot.v_starts.as_slice()),
            ndarray::ArrayView1::from(slot.ilens.as_slice()),
            ndarray::ArrayView1::from(slot.alt_alleles.as_slice()),
            ndarray::ArrayView1::from(slot.alt_offsets.as_slice()),
            ndarray::ArrayView1::from(c.ref_bytes.as_slice()),
            ref_offsets.view(),
            self.pad_char,
            self.parallel,
        ))
    }
}

/// Producer/consumer record-stream engine (issue #276 task 3b). See the module docs. No
/// Python surface yet — `#[pyclass]`/`#[pymethods]`/a `#[new]` constructor and lib.rs
/// registration land in Task 5, once a real `WindowFiller` (Task 4 VCF / Task 10 PGEN)
/// exists to construct one from Python.
pub struct RecordStreamEngine {
    core: StreamEngineCore<RecordBackend>,
}

impl RecordStreamEngine {
    /// Rust-facing constructor (tests; Task 5 wraps this for `#[new]`).
    #[allow(clippy::too_many_arguments)]
    pub fn new_rs(
        filler: Box<dyn WindowFiller + Send + Sync>,
        contigs: Vec<ContigRef>,
        jobs: Vec<RecordJob>,
        n_samples: usize,
        ploidy: usize,
        pad_char: u8,
        parallel: bool,
        batch_size: usize,
    ) -> Self {
        let backend = RecordBackend {
            filler,
            contigs,
            jobs,
            n_samples,
            ploidy,
            pad_char,
            parallel,
        };
        Self {
            core: StreamEngineCore::new(Arc::new(backend), batch_size),
        }
    }

    /// Return the next batch's `(data, offsets)`, or `None` when the plan is exhausted.
    /// Delegates to the shared core; see `stream_core`'s doc comment for the
    /// exhaustion/error/panic classification contract.
    pub fn next_batch_core(&self) -> Option<anyhow::Result<(Array1<u8>, Array1<i64>)>> {
        self.core.next_batch_core()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Filler that ignores the source entirely and decodes each window to a fixed
    /// 2-variant table where every hap carries variant 0 only — so the engine's plumbing
    /// (job dispatch, CSR-offset slicing, per-contig reference selection), not a real VCF
    /// or PGEN source, is under test. Mirrors `Svar1StreamEngine`'s fixture shape.
    struct StubFiller;
    impl WindowFiller for StubFiller {
        fn fill(
            &self,
            job: &RecordJob,
            _c: &ContigRef,
            slot: &mut DecodedWindow,
        ) -> anyhow::Result<()> {
            let n_haps = (job.s_hi - job.s_lo) * 2; // ploidy 2
            slot.v_starts = vec![10, 20];
            slot.ilens = vec![0, 0];
            slot.alt_alleles = vec![b'A', b'C'];
            slot.alt_offsets = vec![0, 1, 2];
            // every hap carries v0 only
            slot.geno_v_idxs = vec![0; n_haps];
            slot.geno_offsets = (0..=n_haps as i64).collect();
            Ok(())
        }
    }

    /// A filler whose `fill` always errors, to exercise the producer-error surfacing path.
    struct FailingFiller;
    impl WindowFiller for FailingFiller {
        fn fill(
            &self,
            _job: &RecordJob,
            _c: &ContigRef,
            _slot: &mut DecodedWindow,
        ) -> anyhow::Result<()> {
            Err(anyhow::anyhow!("stub filler deliberately fails"))
        }
    }

    fn chr1() -> ContigRef {
        ContigRef {
            name: "chr1".into(),
            ref_bytes: vec![b'T'; 30],
        }
    }

    fn two_job_plan() -> Vec<RecordJob> {
        vec![
            RecordJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
            },
            RecordJob {
                contig_idx: 0,
                regions: vec![(0, 15)],
                s_lo: 0,
                s_hi: 2,
            },
        ]
    }

    /// Oracle: fill a scratch `DecodedWindow` with the same `StubFiller` logic + run the
    /// same `generate_batch_core` assembly `RecordBackend::generate` uses, independent of
    /// the engine, for direct comparison against what the engine produced.
    fn expected_window(job: &RecordJob, c: &ContigRef) -> (Vec<u8>, Vec<i64>) {
        let mut slot = DecodedWindow::default();
        StubFiller.fill(job, c, &mut slot).unwrap();

        let ploidy = 2usize;
        let n_samples = job.s_hi - job.s_lo;
        let n_rows = job.regions.len() * n_samples;

        let mut rb = Array2::<i32>::zeros((n_rows, 2));
        for bi in 0..n_rows {
            let ri = bi / n_samples;
            rb[[bi, 0]] = job.regions[ri].0 as i32;
            rb[[bi, 1]] = job.regions[ri].1 as i32;
        }
        let o_lo = 0usize;
        let o_hi = n_rows * ploidy;
        let o_starts_b: Vec<i64> = slot.geno_offsets[o_lo..o_hi].to_vec();
        let o_stops_b: Vec<i64> = slot.geno_offsets[o_lo + 1..=o_hi].to_vec();
        let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);

        let (data, offs) = crate::ffi::generate_batch_core(
            &slot.geno_v_idxs,
            ploidy,
            &o_starts_b,
            &o_stops_b,
            rb.view(),
            ndarray::ArrayView1::from(slot.v_starts.as_slice()),
            ndarray::ArrayView1::from(slot.ilens.as_slice()),
            ndarray::ArrayView1::from(slot.alt_alleles.as_slice()),
            ndarray::ArrayView1::from(slot.alt_offsets.as_slice()),
            ndarray::ArrayView1::from(c.ref_bytes.as_slice()),
            ref_offsets.view(),
            b'N',
            false,
        );
        (data.to_vec(), offs.to_vec())
    }

    /// The structural gate: a >=2-job plan flows through the producer/consumer path in
    /// plan order; every batch byte-equals the direct fill+generate oracle for that job's
    /// window; the plan exhausts and further pulls are clean, idempotent `None`s.
    #[test]
    fn record_stream_engine_yields_windows_in_plan_order() {
        let jobs = two_job_plan();
        let c = chr1();
        let exp0 = expected_window(&jobs[0], &c);
        let exp1 = expected_window(&jobs[1], &c);

        let engine = RecordStreamEngine::new_rs(
            Box::new(StubFiller),
            vec![chr1()],
            jobs,
            2, // n_samples
            2, // ploidy
            b'N',
            false,
            1000, // batch_size larger than any window -> one batch per window
        );

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, o) = r.expect("no producer error");
            batches.push((d.to_vec(), o.to_vec()));
        }
        assert_eq!(batches.len(), 2, "one batch per window, in plan order");
        assert_eq!(batches[0], exp0, "window 0 must match direct fill+generate");
        assert_eq!(batches[1], exp1, "window 1 must match direct fill+generate");

        // Exhaustion is clean and idempotent (must not hang or re-join).
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }

    /// An empty plan must not hang: the producer spawns, finds no jobs, drops its filled
    /// `Sender`; the consumer sees the channel close, joins cleanly, and returns `None`.
    #[test]
    fn record_stream_engine_empty_plan_is_none() {
        let engine = RecordStreamEngine::new_rs(
            Box::new(StubFiller),
            vec![chr1()],
            Vec::new(),
            2,
            2,
            b'N',
            false,
            8,
        );
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }

    /// A producer `fill` error must surface through the consumer's join-then-classify path
    /// as `Some(Err(_))` — NOT a clean/empty EOF and NOT a hang. Panic/drop discipline
    /// itself is shared `stream_core` machinery, already covered by
    /// `Svar1StreamEngine`'s tests (`stream_engine.rs`); this only proves `RecordBackend`
    /// propagates a `WindowFiller` error correctly through that shared path.
    #[test]
    fn record_stream_engine_producer_error_surfaces_not_eof() {
        let jobs = vec![RecordJob {
            contig_idx: 0,
            regions: vec![(0, 30)],
            s_lo: 0,
            s_hi: 2,
        }];
        let engine = RecordStreamEngine::new_rs(
            Box::new(FailingFiller),
            vec![chr1()],
            jobs,
            2,
            2,
            b'N',
            false,
            8,
        );

        match engine.next_batch_core() {
            Some(Err(e)) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("deliberately fails"),
                    "expected the filler's error to propagate, got: {msg}"
                );
            }
            other => panic!(
                "producer error must surface as Some(Err), not {:?}",
                other.map(|r| r.is_ok())
            ),
        }
        // After the error the engine is `done`: further pulls are clean None, never a hang.
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }
}
