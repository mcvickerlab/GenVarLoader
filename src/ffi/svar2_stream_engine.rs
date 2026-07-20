//! `Svar2StreamEngine` — the SVAR2 read↔reconstruct pipeline engine (PR-3 Task 2,
//! issue #278). The SVAR2 analog of `Svar1StreamEngine` (`src/ffi/stream_engine.rs`), one
//! format down: a detached producer thread runs `fill_super_batch_rs` (Task 1's GIL-free
//! `find_ranges -> gather -> reconstruct` primitive) per super-batch job, ping-ponging two
//! reconstructed buffers through `crossbeam_channel::bounded(2)`; the consumer's
//! `next_batch()` blocks on `recv` under `py.detach` and SLICES `batch_size` rows out of the
//! pre-reconstructed buffer — no reconstruct on the consumer thread.
//!
//! **Concurrency discipline is copied EXACTLY from `stream_engine.rs`** — read that file's
//! module doc for the full rationale (it is not repeated here): two `bounded(2)` channels
//! (`filled` producer→consumer, `free` consumer→producer) with `free` prefilled with 2
//! default slots so no `send` ever blocks; shutdown by dropping the producer's filled
//! `Sender` when its job loop ends; join-then-classify on channel close (the consumer never
//! returns with the producer live and unjoined); `Drop for EngineState` closes both channels
//! first (unblocking a parked producer) then joins, so a mid-stream drop can't hang or leak
//! a detached thread.
//!
//! **What differs from SVAR1's engine:**
//!   * SVAR1's producer overlaps I/O *latency* (reads CSR offsets, faults pages) while the
//!     consumer reconstructs on its own thread; here the producer does the WHOLE
//!     reconstruct up front (SVAR2's decode is not split the same way) and the consumer only
//!     slices bytes already sitting in memory.
//!   * Jobs are **super-batch-granular**, not window-granular: each window (contig + region
//!     set + physical sample span) is pre-expanded in `#[new]` into
//!     `ceil(n_rows / super_batch_rows)` `SbJob`s, so `next_batch` drains many small
//!     `CurrentWindow`s per plan window rather than one.
//!   * A window's `find_ranges` output (`Svar2WindowRanges`) is expensive enough to be worth
//!     caching across that window's consecutive super-batch jobs (see the `cached` local in
//!     `ensure_started`'s producer closure) — recomputed only when the window (contig +
//!     sample span + regions, full start/end match) changes from the previous job.
//!
//! **Memory** stays the #284 cohort-independent bound: exactly 2 reconstructed super-batch
//! buffers live at once, never `O(n_samples)` — the same `bounded(2)` ping-pong contract as
//! SVAR1's engine, just carrying reconstructed bytes instead of CSR offsets.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::svar2::store::Svar2Store;
use crate::svar2::window::{fill_super_batch_rs, Svar2WindowRanges};

/// Row-count knee below which rayon fork/join overhead outweighs the parallel win for a
/// super-batch fill — measured in `benchmarking/streaming/svar2_superbatch_sweep.py`
/// (`sb<=256` rows went *slower* parallel; see commit `c05e7914`). The byte-based Python
/// gate `_Svar2Backend`'s own "sync" drive uses (`should_parallelize`, gated on an
/// output-size estimate) isn't reachable from the producer thread without an equivalent
/// estimator, so the engine gates on row count directly using this measured knee instead —
/// same intent ("skip parallel dispatch for a batch too small to pay for it"), cheaper to
/// evaluate, and correctness-neutral (Task 1 confirmed `parallel=true`/`false` are
/// byte-identical; this only affects speed).
const PARALLEL_THRESHOLD_ROWS: usize = 256;

/// Per-contig reference bytes, indexed by a job's `contig_idx`. Mirrors the SVAR1 engine's
/// `ContigData::ref_bytes` / its module note on per-contig reference: `fill_super_batch_rs`
/// always reconstructs against whatever contig slice its caller hands it, so a multi-contig
/// plan needs one slice per contig, never a single engine-level `ref_`.
struct ContigRef {
    name: String,
    ref_bytes: Vec<u8>,
}

/// One super-batch job: a window's `regions` (physical, half-open `[start, end)`) crossed
/// with the contiguous physical-sample sub-range `[s_lo, s_hi)`, further sliced to the row
/// span `[sb_lo, sb_hi)` within that window's `n_reg * (s_hi - s_lo)` C-order
/// `(region, sample)` rows. A window whose row count exceeds `super_batch_rows` expands
/// into multiple consecutive `SbJob`s (built in `#[new]`) that all share the same
/// `contig_idx`/`regions`/`s_lo`/`s_hi` — exactly what the producer's per-window
/// `find_ranges` cache keys on (see `ensure_started`).
struct SbJob {
    contig_idx: usize,
    regions: Vec<(u32, u32)>,
    s_lo: usize,
    s_hi: usize,
    sb_lo: usize,
    sb_hi: usize,
}

/// Key: `(contig_idx, s_lo, s_hi)`. Cache: the window's `Svar2WindowRanges` plus the
/// `(i32, i32)` region bounds `fill_super_batch_rs` needs, kept alongside so a cache hit
/// avoids re-deriving them from `job.regions` too.
type WindowRangesCache = Option<((usize, usize, usize), Svar2WindowRanges, Vec<(i32, i32)>)>;

/// One recycled slot: a fully reconstructed super-batch (`fill_super_batch_rs`'s
/// `(data, offsets)` output), its row count, and the `job_idx` that produced it. Unlike
/// SVAR1's `FilledWindow` (CSR offsets only — the consumer still has to generate bytes from
/// them), this buffer already IS the batch data; the consumer only slices it.
#[derive(Default)]
struct FilledWindow {
    data: Vec<u8>,
    offsets: Vec<i64>,
    n_rows: usize,
    job_idx: usize,
}

/// The super-batch the consumer is currently draining, batch by batch.
struct CurrentWindow {
    filled: FilledWindow,
    /// Next row (0-based, within `filled`) to slice from.
    next_row: usize,
}

/// Mutable engine state behind the pyclass's `Mutex` (the pyclass methods are `&self`).
/// Identical shape/discipline to `stream_engine.rs`'s `EngineState` — `FilledWindow`'s type
/// is the only difference.
struct EngineState {
    started: bool,
    done: bool,
    /// Recycle drained slots back to the producer. `None` until the producer is spawned.
    tx_free: Option<Sender<FilledWindow>>,
    /// Receive filled super-batches from the producer. `None` until spawned.
    rx_filled: Option<Receiver<FilledWindow>>,
    /// Producer handle — joined (and classified) exactly once, on channel close.
    producer: Option<JoinHandle<anyhow::Result<()>>>,
    current: Option<CurrentWindow>,
}

impl EngineState {
    fn new() -> Self {
        Self {
            started: false,
            done: false,
            tx_free: None,
            rx_filled: None,
            producer: None,
            current: None,
        }
    }
}

impl Drop for EngineState {
    /// Deterministic teardown: if the engine is dropped mid-stream (producer still live),
    /// join the producer instead of leaking a detached thread. Dropping the free `Sender`
    /// and the filled `Receiver` FIRST closes both channels, so a producer blocked on
    /// `rx_free.recv()` (no free slot) or `tx_filled.send()` (consumer gone) unblocks and
    /// returns `Ok(())` — the join then completes promptly. Cannot double-join: the normal
    /// exhaustion/error/panic paths already `take()` the handle in `next_batch_core`,
    /// leaving `producer == None` here. Copied verbatim from `stream_engine.rs`.
    fn drop(&mut self) {
        self.tx_free = None;
        self.rx_filled = None;
        if let Some(h) = self.producer.take() {
            let _ = h.join();
        }
    }
}

/// Producer/consumer SVAR2 streamer (issue #278, PR-3 Task 2). See the module docs.
#[pyclass]
pub struct Svar2StreamEngine {
    /// The engine's OWN store (opened via `Svar2Store::open`, Arc-shared with the producer
    /// thread; mmaps the same finished sidecars a Python-owned `Svar2Store` would, so no
    /// data duplication beyond the shared OS page cache).
    store: Arc<Svar2Store>,
    contig_refs: Arc<Vec<ContigRef>>,
    /// Super-batch-granular, plan order.
    jobs: Arc<Vec<SbJob>>,
    /// Public→physical sample map (`O(n_samples)`, ONE copy); each job's `[s_lo, s_hi)`
    /// slices into this — same cohort-independent job residency contract as SVAR1's engine
    /// (issue #284).
    phys_sample_idx: Arc<Vec<usize>>,
    ploidy: usize,
    pad_char: u8,
    batch_size: usize,
    state: Mutex<EngineState>,
}

impl Svar2StreamEngine {
    /// Shared constructor for `#[new]` and Rust tests. `jobs` are already expanded to
    /// super-batch grain (Rust tests build these directly; `#[new]` expands them from the
    /// flat per-window Python arrays).
    fn build(
        store: Svar2Store,
        contig_refs: Vec<ContigRef>,
        jobs: Vec<SbJob>,
        phys_sample_idx: Vec<usize>,
        ploidy: usize,
        pad_char: u8,
        batch_size: usize,
    ) -> Self {
        Self {
            store: Arc::new(store),
            contig_refs: Arc::new(contig_refs),
            jobs: Arc::new(jobs),
            phys_sample_idx: Arc::new(phys_sample_idx),
            ploidy,
            pad_char,
            batch_size: batch_size.max(1),
            state: Mutex::new(EngineState::new()),
        }
    }

    /// Spawn the detached producer thread (once). Prefills `free` with 2 default slots,
    /// then the producer fills super-batch after super-batch, blocked only by the
    /// free-slot pool.
    fn ensure_started(&self, state: &mut EngineState) -> anyhow::Result<()> {
        if state.started {
            return Ok(());
        }
        state.started = true;

        // 2 slots (ping-pong). Only 2 buffers ever exist, recycled between the two
        // channels, so no `send` can block; memory is capped regardless of plan length.
        let n_slots = 2usize;
        let (tx_filled, rx_filled) = bounded::<FilledWindow>(n_slots);
        let (tx_free, rx_free) = bounded::<FilledWindow>(n_slots);
        for _ in 0..n_slots {
            tx_free
                .send(FilledWindow::default())
                .expect("prefill free slots (receiver just created, cannot be closed)");
        }

        // `tx_filled` and `rx_free` are MOVED into the producer — it is their sole owner.
        // When the producer returns (all jobs done, an error via `?`, or a `recv`/`send`
        // seeing the consumer gone), `tx_filled` drops and the consumer's
        // `rx_filled.recv()` observes close as soon as the channel drains.
        let store = Arc::clone(&self.store);
        let jobs = Arc::clone(&self.jobs);
        let contig_refs = Arc::clone(&self.contig_refs);
        let phys_sample_idx = Arc::clone(&self.phys_sample_idx);
        let ploidy = self.ploidy;
        let pad_char = self.pad_char;

        let handle = std::thread::Builder::new()
            .name("gvl-svar2-stream-producer".into())
            .spawn(move || -> anyhow::Result<()> {
                // Per-window `find_ranges` cache: recomputed only when the window
                // (contig_idx, sample span, and full region start/end match) changes from
                // the previous job. Consecutive super-batch jobs of one window share the
                // same `regions` Vec (cloned once when the window was expanded), so this
                // is a cheap comparison, not a re-derivation.
                let mut cached: WindowRangesCache = None;

                for (job_idx, job) in jobs.iter().enumerate() {
                    // Recycle a drained slot. Err => consumer gone (engine dropped); stop
                    // quietly and let the consumer's own outcome stand.
                    let Ok(mut slot) = rx_free.recv() else {
                        return Ok(());
                    };

                    // Panics on OOB `contig_idx` — caught by the consumer's
                    // join-then-classify as a producer panic (tested).
                    let cref = &contig_refs[job.contig_idx];
                    let reader = store
                        .reader(&cref.name)
                        .ok_or_else(|| anyhow::anyhow!("contig {} not in store", cref.name))?;
                    let phys: Vec<usize> = phys_sample_idx[job.s_lo..job.s_hi].to_vec();

                    let key = (job.contig_idx, job.s_lo, job.s_hi);
                    let need = cached
                        .as_ref()
                        .map(|(k, _, rb)| {
                            *k != key
                                || rb.len() != job.regions.len()
                                || rb.iter().zip(&job.regions).any(|(&(rs, re), &(js, je))| {
                                    rs as u32 != js || re as u32 != je
                                })
                        })
                        .unwrap_or(true);
                    if need {
                        let region_bnd: Vec<(i32, i32)> = job
                            .regions
                            .iter()
                            .map(|&(s, e)| (s as i32, e as i32))
                            .collect();
                        let ranges =
                            Svar2WindowRanges::compute(reader, &job.regions, &phys, ploidy);
                        cached = Some((key, ranges, region_bnd));
                    }
                    let (_, ranges, region_bnd) = cached.as_ref().unwrap();

                    let n_rows = job.sb_hi - job.sb_lo;
                    let parallel = n_rows * ploidy >= PARALLEL_THRESHOLD_ROWS;
                    fill_super_batch_rs(
                        reader,
                        ranges,
                        region_bnd,
                        &cref.ref_bytes,
                        pad_char,
                        job.sb_lo,
                        job.sb_hi,
                        parallel,
                        &mut slot.data,
                        &mut slot.offsets,
                    );
                    slot.n_rows = n_rows;
                    slot.job_idx = job_idx;

                    if tx_filled.send(slot).is_err() {
                        return Ok(()); // consumer gone
                    }
                }
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn svar2 streaming producer: {e}"))?;

        state.tx_free = Some(tx_free);
        state.rx_filled = Some(rx_filled);
        state.producer = Some(handle);
        Ok(())
    }

    /// Slice `batch_size` rows (bounded by the current super-batch's remaining rows) out of
    /// the pre-reconstructed buffer. NO reconstruct here (unlike SVAR1's
    /// `generate_from_current`) — always a copy, offsets rebased to 0, matching
    /// `Svar2ReconBuf::batch`'s contract.
    fn slice_current(&self, cur: &mut CurrentWindow) -> (Array1<u8>, Array1<i64>) {
        let p = self.ploidy;
        let row_lo = cur.next_row;
        let row_hi = (row_lo + self.batch_size).min(cur.filled.n_rows);
        cur.next_row = row_hi;
        let o_lo = row_lo * p;
        let o_hi = row_hi * p;
        let byte_lo = cur.filled.offsets[o_lo] as usize;
        let byte_hi = cur.filled.offsets[o_hi] as usize;
        let data = Array1::from(cur.filled.data[byte_lo..byte_hi].to_vec());
        let offsets: Vec<i64> = cur.filled.offsets[o_lo..=o_hi]
            .iter()
            .map(|&o| o - byte_lo as i64)
            .collect();
        (data, Array1::from(offsets))
    }

    /// The consumer, GIL-free. Returns:
    ///   * `Some(Ok((data, offsets)))` — the next batch;
    ///   * `Some(Err(_))` — a producer error/panic (join-then-classified);
    ///   * `None` — the plan is exhausted (clean, idempotent, no hang).
    ///
    /// Discipline identical to SVAR1's `next_batch_core`: drains the current super-batch
    /// slice-by-slice; when spent it is recycled to the producer and the next filled
    /// super-batch is `recv`'d. On channel close the producer is joined and classified
    /// before returning.
    fn next_batch_core(&self) -> Option<anyhow::Result<(Array1<u8>, Array1<i64>)>> {
        // Recover from a poisoned lock rather than propagating panic-on-panic.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());

        if state.done {
            return None;
        }
        if let Err(e) = self.ensure_started(&mut state) {
            state.done = true;
            return Some(Err(e));
        }

        loop {
            // 1. If the current super-batch has rows left, slice the next batch.
            let has_rows = match state.current.as_ref() {
                Some(cur) => cur.next_row < cur.filled.n_rows,
                None => false,
            };
            if has_rows {
                let cur = state
                    .current
                    .as_mut()
                    .expect("has_rows implies current is Some");
                return Some(Ok(self.slice_current(cur)));
            }

            // 2. Current super-batch is spent (or absent): recycle it, then fetch the next.
            if let Some(spent) = state.current.take() {
                if let Some(tx) = state.tx_free.as_ref() {
                    // Always recycle (Err only if the producer already exited) so the
                    // producer can finish rather than block on rx_free.recv().
                    let _ = tx.send(spent.filled);
                }
            }

            let recv = state
                .rx_filled
                .as_ref()
                .expect("rx_filled set by ensure_started")
                .recv();
            match recv {
                Ok(fw) => {
                    state.current = Some(CurrentWindow {
                        filled: fw,
                        next_row: 0,
                    });
                    // Loop back to slice from the newly received super-batch.
                }
                Err(_) => {
                    // Channel closed => producer finished. JOIN FIRST, classify AFTER —
                    // never return with the producer live and unjoined.
                    state.done = true;
                    if let Some(h) = state.producer.take() {
                        return match h.join() {
                            Err(_) => Some(Err(anyhow::anyhow!(
                                "svar2 streaming producer thread panicked"
                            ))),
                            Ok(Err(e)) => Some(Err(e)),
                            Ok(Ok(())) => None,
                        };
                    }
                    return None;
                }
            }
        }
    }
}

#[pymethods]
impl Svar2StreamEngine {
    /// Construct the engine: open the store, register per-contig reference bytes, and
    /// expand the flat per-window plan into super-batch-granular jobs.
    ///
    /// Per-contig records are parallel arrays indexed by `contig_names[i]` /
    /// `contig_ref_bytes[i]`, referenced from job records by `job_contig_idx[j]`. The full
    /// public→physical sample map `phys_sample_idx` (this backend's `n_samples`) crosses
    /// ONCE. Per-window job records are parallel arrays indexed by window:
    /// `job_contig_idx[j]`, `job_region_starts[j]`, `job_region_ends[j]`, `job_s_lo[j]`,
    /// `job_s_hi[j]` — each expands into `ceil(n_rows / super_batch_rows)` `SbJob`s.
    #[new]
    #[pyo3(signature = (
        store_path, store_contigs, n_samples, ploidy,
        contig_names, contig_ref_bytes, phys_sample_idx,
        job_contig_idx, job_region_starts, job_region_ends, job_s_lo, job_s_hi,
        pad_char, super_batch_rows, batch_size,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        store_path: &str,
        store_contigs: Vec<String>,
        n_samples: usize,
        ploidy: usize,
        contig_names: Vec<String>,
        contig_ref_bytes: Vec<Vec<u8>>,
        phys_sample_idx: Vec<usize>,
        job_contig_idx: Vec<usize>,
        job_region_starts: Vec<Vec<u32>>,
        job_region_ends: Vec<Vec<u32>>,
        job_s_lo: Vec<usize>,
        job_s_hi: Vec<usize>,
        pad_char: u8,
        super_batch_rows: usize,
        batch_size: usize,
    ) -> PyResult<Self> {
        let store = Svar2Store::open(store_path, store_contigs, n_samples, ploidy)?;

        if contig_ref_bytes.len() != contig_names.len() {
            return Err(PyValueError::new_err(
                "Svar2StreamEngine: contig_names and contig_ref_bytes must have the same length",
            ));
        }
        let contig_refs: Vec<ContigRef> = contig_names
            .into_iter()
            .zip(contig_ref_bytes)
            .map(|(name, ref_bytes)| ContigRef { name, ref_bytes })
            .collect();

        let n_windows = job_contig_idx.len();
        if [
            job_region_starts.len(),
            job_region_ends.len(),
            job_s_lo.len(),
            job_s_hi.len(),
        ]
        .iter()
        .any(|&l| l != n_windows)
        {
            return Err(PyValueError::new_err(
                "Svar2StreamEngine: per-window job arrays must all have the same length",
            ));
        }

        let sb_rows = super_batch_rows.max(1);
        let mut jobs = Vec::new();
        for w in 0..n_windows {
            let starts = &job_region_starts[w];
            let ends = &job_region_ends[w];
            if starts.len() != ends.len() {
                return Err(PyValueError::new_err(
                    "Svar2StreamEngine: job_region_starts and job_region_ends must match",
                ));
            }
            let (s_lo, s_hi) = (job_s_lo[w], job_s_hi[w]);
            if s_lo > s_hi || s_hi > phys_sample_idx.len() {
                return Err(PyValueError::new_err(format!(
                    "Svar2StreamEngine: job sample range [{s_lo}, {s_hi}) is invalid for \
                     phys_sample_idx.len()={}",
                    phys_sample_idx.len()
                )));
            }
            let regions: Vec<(u32, u32)> = starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect();
            let n_rows = regions.len() * (s_hi - s_lo);
            let contig_idx = job_contig_idx[w];

            let mut lo = 0usize;
            while lo < n_rows {
                let hi = (lo + sb_rows).min(n_rows);
                jobs.push(SbJob {
                    contig_idx,
                    regions: regions.clone(),
                    s_lo,
                    s_hi,
                    sb_lo: lo,
                    sb_hi: hi,
                });
                lo = hi;
            }
        }

        Ok(Self::build(
            store,
            contig_refs,
            jobs,
            phys_sample_idx,
            ploidy,
            pad_char,
            batch_size,
        ))
    }

    /// Return the next batch's `(data, offsets)`, or `None` when the plan is exhausted.
    /// The GIL is released for the whole blocking body (recv/slice/join); it is reacquired
    /// only to marshal the owned arrays into numpy. A producer error/panic surfaces here as
    /// a `RuntimeError`, join-then-classified — never a hang.
    fn next_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>> {
        let out = py.detach(|| self.next_batch_core());
        match out {
            None => Ok(None),
            Some(Ok((data, offsets))) => {
                Ok(Some((data.into_pyarray(py), offsets.into_pyarray(py))))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use genoray_core::normalize::CheckRef;
    use genoray_core::orchestrator::{process_chromosome, SourceSpec};
    use genoray_core::svar2_view::OverlapMode;
    use rust_htslib::bcf::record::GenotypeAllele;
    use rust_htslib::bcf::{Format, Header, Writer};

    /// One synthetic single-base-SNP VCF record. Mirrors genoray_core's own test harness
    /// (`tests/common/mod.rs::SynthRecord`), which is not reachable from this downstream
    /// crate (it lives in a separate `tests/` integration-test binary, not the library).
    /// `gt` is flat `[s0_p0, s0_p1, s1_p0, s1_p1, ...]` allele indices (0 = ref, 1 = alt).
    struct SynthRecord {
        pos: i64,
        ref_base: u8,
        alt_base: u8,
        gt: Vec<i32>,
    }

    fn build_bcf_with_index(
        bcf_path: &std::path::Path,
        chrom: &str,
        chrom_len: u64,
        samples: &[&str],
        records: &[SynthRecord],
    ) {
        let mut header = Header::new();
        header.push_record(format!("##contig=<ID={chrom},length={chrom_len}>").as_bytes());
        header.push_record(b"##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">");
        for s in samples {
            header.push_sample(s.as_bytes());
        }
        {
            let mut writer =
                Writer::from_path(bcf_path, &header, false, Format::Bcf).expect("open BCF writer");
            for rec in records {
                let mut record = writer.empty_record();
                record.set_rid(Some(0));
                record.set_pos(rec.pos);
                record
                    .set_alleles(&[&[rec.ref_base][..], &[rec.alt_base][..]])
                    .expect("set alleles");
                let gt: Vec<GenotypeAllele> =
                    rec.gt.iter().map(|&i| GenotypeAllele::Phased(i)).collect();
                record.push_genotypes(&gt).expect("push genotypes");
                writer.write(&record).expect("write record");
            }
        }
        rust_htslib::bcf::index::build(bcf_path, None, 0, rust_htslib::bcf::index::Type::Csi(14))
            .expect("build BCF index");
    }

    /// Reference FASTA (+ `.fai`): `chrom_len` bytes of `ref_byte`, with each record's REF
    /// base stamped at its 0-based `pos` (a no-op here since every fixture record's REF
    /// equals `ref_byte`, but kept for parity with genoray's convention that the FASTA must
    /// agree with the BCF's REF calls under `CheckRef::Error`).
    fn build_fasta_with_index(
        fasta_path: &std::path::Path,
        chrom: &str,
        chrom_len: usize,
        ref_byte: u8,
        records: &[SynthRecord],
    ) {
        use std::io::Write;
        let mut seq = vec![ref_byte; chrom_len];
        for rec in records {
            seq[rec.pos as usize] = rec.ref_base;
        }
        let mut f = std::fs::File::create(fasta_path).expect("create fasta");
        writeln!(f, ">{chrom}").expect("write header");
        f.write_all(&seq).expect("write seq");
        writeln!(f).expect("write newline");
        rust_htslib::faidx::build(fasta_path).expect("build .fai");
    }

    /// Build a real, finished SVAR2 `chrom` contig under `out` via the actual conversion
    /// pipeline (`process_chromosome`, already a hard dependency through the `conversion`
    /// feature) fed a synthetic indexed BCF + FASTA. There is no SVAR1-style "write raw npy
    /// bytes" shortcut for SVAR2's on-disk layout (codec-encoded var_key/dense CSR
    /// sidecars) — genoray_core's own Rust tests build fixtures exactly this way.
    fn build_contig(
        out: &std::path::Path,
        chrom: &str,
        samples: &[&str],
        ploidy: usize,
        ref_byte: u8,
        records: &[SynthRecord],
    ) {
        let bcf = out.join("in.bcf");
        let fasta = out.join("in.fa");
        build_bcf_with_index(&bcf, chrom, 1000, samples, records);
        build_fasta_with_index(&fasta, chrom, 40, ref_byte, records);
        process_chromosome(
            SourceSpec::Vcf {
                vcf_path: bcf.to_str().unwrap().to_string(),
                htslib_threads: 1,
                regions: Vec::new(),
                overlap: OverlapMode::Pos,
            },
            Some(fasta.to_str().unwrap()),
            chrom,
            out.to_str().unwrap(),
            samples,
            1000, // chunk_size
            ploidy,
            4096, // long_allele_capacity
            false,
            CheckRef::Error,
            1,     // processing_threads
            false, // signatures
            &[],   // fields
        )
        .expect("process_chromosome should succeed");
    }

    /// The 4-hap fixture shared across this module's tests: chr1, 40bp reference of 'T',
    /// 2 samples x ploidy 2, two SNPs (var0 @10 alt 'A', var1 @20 alt 'C'). Per-hap
    /// variants: hap0=[var0] hap1=[] hap2=[var0,var1] hap3=[var1] — the same per-hap shape
    /// as `stream_engine.rs`'s SVAR1 fixture, so the two engines are tested to an
    /// equivalent bar.
    struct Fixture {
        _tmp: tempfile::TempDir,
        path: String,
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("out");
        std::fs::create_dir_all(&out).unwrap();
        let samples = ["S0", "S1"];
        let records = vec![
            SynthRecord {
                pos: 10,
                ref_base: b'T',
                alt_base: b'A',
                gt: vec![1, 0, 1, 0], // hap0=1 hap1=0 hap2=1 hap3=0
            },
            SynthRecord {
                pos: 20,
                ref_base: b'T',
                alt_base: b'C',
                gt: vec![0, 0, 1, 1], // hap0=0 hap1=0 hap2=1 hap3=1
            },
        ];
        build_contig(&out, "chr1", &samples, 2, b'T', &records);
        Fixture {
            path: out.to_str().unwrap().to_string(),
            _tmp: tmp,
        }
    }

    fn ref_bytes() -> Vec<u8> {
        vec![b'T'; 40]
    }

    fn open_store(f: &Fixture) -> Svar2Store {
        Svar2Store::open(&f.path, vec!["chr1".to_string()], 2, 2).unwrap()
    }

    /// Ground truth for one window: a direct `Svar2WindowRanges::compute` +
    /// `fill_super_batch_rs` over the WHOLE window, on an independently opened store.
    fn expected_window(
        f: &Fixture,
        regions: &[(u32, u32)],
        phys_samples: &[usize],
    ) -> (Vec<u8>, Vec<i64>) {
        let store = open_store(f);
        let reader = store.reader("chr1").unwrap();
        let ranges = Svar2WindowRanges::compute(reader, regions, phys_samples, 2);
        let region_bnd: Vec<(i32, i32)> =
            regions.iter().map(|&(s, e)| (s as i32, e as i32)).collect();
        let n_rows = regions.len() * phys_samples.len();
        let mut data = Vec::new();
        let mut offsets = Vec::new();
        fill_super_batch_rs(
            reader,
            &ranges,
            &region_bnd,
            &ref_bytes(),
            b'N',
            0,
            n_rows,
            false,
            &mut data,
            &mut offsets,
        );
        (data, offsets)
    }

    fn build_engine(f: &Fixture, jobs: Vec<SbJob>, batch_size: usize) -> Svar2StreamEngine {
        let store = open_store(f);
        let contig_refs = vec![ContigRef {
            name: "chr1".to_string(),
            ref_bytes: ref_bytes(),
        }];
        // Identity public->physical map for the 2-sample fixture; jobs slice it by range.
        Svar2StreamEngine::build(
            store,
            contig_refs,
            jobs,
            vec![0usize, 1],
            2,
            b'N',
            batch_size,
        )
    }

    /// The Task-2 gate: a >=2-window plan flows through the producer/consumer path; every
    /// batch arrives in plan order and byte-equals a direct `fill_super_batch_rs` over that
    /// window's full row span; the plan exhausts and `next_batch_core` returns `None`
    /// cleanly (idempotently, no hang). Mirrors
    /// `stream_engine.rs::svar1_stream_engine_yields_windows_in_plan_order`.
    #[test]
    fn svar2_stream_engine_yields_windows_in_plan_order() {
        let f = fixture();
        // Two windows on chr1: full region, then a narrower region (variant 0 only). Each
        // window is a single super-batch (sb_lo=0, sb_hi=n_rows).
        let jobs = vec![
            SbJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
                sb_lo: 0,
                sb_hi: 2,
            },
            SbJob {
                contig_idx: 0,
                regions: vec![(0, 15)],
                s_lo: 0,
                s_hi: 2,
                sb_lo: 0,
                sb_hi: 2,
            },
        ];
        let engine = build_engine(&f, jobs, 1000);

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, o) = r.expect("no producer error");
            batches.push((d.to_vec(), o.to_vec()));
        }
        assert_eq!(batches.len(), 2, "one batch per window, in plan order");

        let exp0 = expected_window(&f, &[(0, 30)], &[0, 1]);
        let exp1 = expected_window(&f, &[(0, 15)], &[0, 1]);
        assert_eq!(
            batches[0], exp0,
            "window 0 data/offsets must match a direct fill_super_batch_rs"
        );
        assert_eq!(
            batches[1], exp1,
            "window 1 data/offsets must match a direct fill_super_batch_rs"
        );

        // Exhaustion is clean and idempotent (must not hang or re-join).
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }

    /// `batch_size=1` splits each window's super-batch into per-row batches (issue #284
    /// bounding). The concatenation of a window's batch data must equal the full-window
    /// fill, and the batch count must equal the total row count. Exercises the
    /// multi-batch drain + slot-recycle path under threading.
    #[test]
    fn svar2_stream_engine_splits_windows_into_bounded_batches() {
        let f = fixture();
        let jobs = vec![
            SbJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
                sb_lo: 0,
                sb_hi: 2,
            },
            SbJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
                sb_lo: 0,
                sb_hi: 2,
            },
        ];
        let engine = build_engine(&f, jobs, 1);

        let mut all: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, o) = r.expect("no producer error");
            all.push((d.to_vec(), o.to_vec()));
        }
        // Each window has 1 region * 2 samples = 2 rows -> 2 batches; 2 windows -> 4.
        assert_eq!(
            all.len(),
            4,
            "batch_size=1 splits each 2-row window into 2 batches"
        );

        let exp = expected_window(&f, &[(0, 30)], &[0, 1]);
        let cat0: Vec<u8> = all[0].0.iter().chain(all[1].0.iter()).copied().collect();
        assert_eq!(
            cat0, exp.0,
            "concatenated split batches must equal full-window data"
        );
        let cat1: Vec<u8> = all[2].0.iter().chain(all[3].0.iter()).copied().collect();
        assert_eq!(cat1, exp.0, "second window's split batches must also match");
    }

    /// An empty plan must not hang: the producer spawns, finds no jobs, drops its filled
    /// `Sender`; the consumer sees the channel close, joins cleanly, and returns `None`.
    #[test]
    fn svar2_stream_engine_empty_plan_is_none() {
        let f = fixture();
        let engine = build_engine(&f, Vec::new(), 8);
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }

    /// A producer error must surface through the consumer's join-then-classify path as
    /// `Some(Err(_))` — NOT a clean/empty EOF and NOT a hang. Here the job's `ContigRef`
    /// name ("chrMISSING") is never opened by the store, so `store.reader(name)` returns
    /// `None`, the producer returns `Err`, and the classify `Ok(Err(e))` branch fires.
    #[test]
    fn svar2_stream_engine_producer_error_surfaces_not_eof() {
        let f = fixture();
        let store = open_store(&f);
        let contig_refs = vec![ContigRef {
            name: "chrMISSING".to_string(),
            ref_bytes: ref_bytes(),
        }];
        let jobs = vec![SbJob {
            contig_idx: 0,
            regions: vec![(0, 30)],
            s_lo: 0,
            s_hi: 2,
            sb_lo: 0,
            sb_hi: 2,
        }];
        let engine =
            Svar2StreamEngine::build(store, contig_refs, jobs, vec![0usize, 1], 2, b'N', 8);

        match engine.next_batch_core() {
            Some(Err(e)) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("not in store"),
                    "expected the reader-lookup error to propagate, got: {msg}"
                );
                assert!(
                    !msg.contains("panicked"),
                    "an Err() must classify as Ok(Err), NOT the panic branch: {msg}"
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

    /// Serializes the two tests that swap the process-global panic hook (same guard
    /// pattern as `stream_engine.rs` and `crate::stream`'s tests: `take_hook`/`set_hook` is
    /// not atomic, so without serialization one test's temporary silent hook can be
    /// observed by another as "the previous hook" and restored permanently).
    static PANIC_HOOK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// A producer PANIC must surface through join-then-classify as the `Err(_)` ("producer
    /// thread panicked") branch — distinct from the `Ok(Err)` error case above — and must
    /// not hang. The panic is induced with a job whose `contig_idx` is out of range of the
    /// engine's `contig_refs` vec, a clean, fully in-our-own-code `Vec` bounds-check panic
    /// (no dependence on genoray internals / no corrupt store).
    #[test]
    fn svar2_stream_engine_producer_panic_surfaces_not_hang() {
        let _guard = PANIC_HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let f = fixture();
        // Only contig index 0 exists; this job points at 5 -> producer panics on index.
        let jobs = vec![SbJob {
            contig_idx: 5,
            regions: vec![(0, 30)],
            s_lo: 0,
            s_hi: 2,
            sb_lo: 0,
            sb_hi: 2,
        }];
        let engine = build_engine(&f, jobs, 8);

        // Silence the deliberate producer-thread panic's backtrace on stderr.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = engine.next_batch_core();
        std::panic::set_hook(prev_hook);

        match outcome {
            Some(Err(e)) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("panicked"),
                    "a producer panic must hit the join-classify panic branch, got: {msg}"
                );
            }
            other => panic!(
                "producer panic must surface as Some(Err(panicked)), not {:?}",
                other.map(|r| r.is_ok())
            ),
        }
        // `done` after the panic: further pulls are clean None, never a hang or re-join.
        assert!(engine.next_batch_core().is_none());
    }

    /// Dropping the engine mid-stream (producer still live/blocked) must not hang and must
    /// not leak the detached thread — `Drop for EngineState` closes the channels and joins.
    /// A 4-window plan with `batch_size=1`: after pulling ONE row the producer has filled
    /// the 2 prefill slots and is blocked on `rx_free.recv()` for a third; dropping the
    /// engine here closes `free`, unblocking it. If the join wedged, this test (and the
    /// 20x race loop) would hang.
    #[test]
    fn svar2_stream_engine_drop_midstream_joins_cleanly() {
        let f = fixture();
        let jobs = (0..4)
            .map(|_| SbJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
                sb_lo: 0,
                sb_hi: 2,
            })
            .collect();
        let engine = build_engine(&f, jobs, 1);
        let first = engine.next_batch_core();
        assert!(
            matches!(first, Some(Ok(_))),
            "first pull must yield a batch"
        );
        // `engine` drops here with the producer still live -> Drop must join, not hang.
        drop(engine);
    }
}
