//! `Svar1StreamEngine` — the SVAR1 producer/consumer overlap engine (issue #283).
//!
//! A background *producer* thread reads window N+1's CSR offsets
//! (`Svar1Store::read_window`) and reads-through the runs it will hand off
//! (`geno_v_idxs()[o_lo..o_hi]`), warming the shared OS page cache, while the
//! *consumer* (`next_batch`, called from Python) generates batches from window N. What
//! is overlapped is I/O latency, not decode — SVAR1's on-disk layout is already the
//! target representation, so the producer's job is to fault the exact pages ahead of the
//! consumer (the OS page cache does not prefetch on an application's access pattern; a
//! thread walking the fixed, `__getitem__`-free traversal does).
//!
//! **The generic detached-thread producer/consumer machinery (channels, shutdown,
//! panic-classification, slot recycling) lives in [`crate::ffi::stream_core`]** —
//! [`Svar1StreamEngine`] is a thin pyclass wrapper around a
//! `StreamEngineCore<Svar1Backend>`; [`Svar1Backend`] below implements the two
//! SVAR1-specific divergence points (`fill`/`generate`) via the [`EngineBackend`] trait.
//! Read `stream_core`'s module doc for the threading/shutdown rationale — it is
//! load-bearing and shared verbatim with any future backend (e.g. a record-stream
//! VCF/PGEN engine, issue #276 task 3b).
//!
//! **The engine opens its OWN `Svar1Store`** from the store path (it does not share the
//! Python-owned `Svar1Store` pyclass instance). Because it mmaps the same file, the
//! producer's reads and the consumer's reads hit the same OS page-cache pages — no data
//! duplication. `Svar1Store` is `Send + Sync`, and the whole `Svar1Backend` (which owns
//! it) is `Arc`'d by the core and cloned into the producer thread, so `read_window`/
//! `geno_v_idxs` run GIL-free.
//!
//! **Memory (cohort-independent job residency — issue #284 / final-review Finding 1).**
//! The engine owns exactly ONE cohort-scale array: `phys_sample_idx` (the public→physical
//! sample map, `O(n_samples)`, the same single copy PR 1 already keeps). Everything else is
//! region-/contig-/variant-scale: the `jobs` vec carries per window only its `contig_idx`,
//! its `regions` (`window_regions`-scale, ≤ `REGION_TARGET` ≈ 64) and a two-`usize` sample
//! sub-range `(s_lo, s_hi)` — NOT a per-window copy of that window's physical samples. The
//! producer builds the window's physical sample slice on the fly as a BORROW,
//! `&phys_sample_idx[s_lo..s_hi]` (`_plan` always yields a contiguous `arange(s_lo, s_hi)`
//! sample chunk, so the sub-range reconstructs it losslessly with zero new allocation). Thus
//! total job metadata is `O(n_windows × window_regions)` (region-scale), never
//! `O(n_windows × window_samples)` ≈ cohort × regions. The only sample-scale residency
//! during iteration is the recycled per-window offsets in the ≤2 live `FilledWindow`s — the
//! intended, budgeted (max_mem-bounded, `window_samples`-scale) allocation, never the full
//! cohort.
//!
//! **Note — per-contig reference (deviation from the 8a design note).** The design note
//! listed a single engine-level `ref_`/`ref_offsets`. That is wrong for a multi-contig
//! plan: `generate_batch_core` builds per-row `regions` with `contig_idx = 0`, so it
//! always reconstructs against `ref_offsets[0..2]` — i.e. whatever contig slice it is
//! handed. The production read path passes `self._ref._contig_slice(contig_idx)` (that
//! contig's slice, offsets `[0, contig_len]`) per generate call, so byte-parity REQUIRES
//! the active contig's reference. This engine therefore carries the reference per contig
//! (`ContigData::ref_bytes`) and hands the current job's contig slice to
//! `generate_batch_core`. Single-contig behavior is identical; multi-contig is now
//! correct.

use std::sync::Arc;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::ffi::stream_core::{EngineBackend, StreamEngineCore};
use crate::svar1::store::Svar1Store;

/// Per-contig data the engine needs to serve `read_window` and generation for jobs on
/// that contig. `contig_start`/`n_local`/`max_v_len` are registered on the engine's own
/// store via `set_contig_meta_rs` at construction; the rest are held for the loop.
struct ContigData {
    name: String,
    contig_start: u32,
    n_local: usize,
    max_v_len: u32,
    /// This contig's LOCAL 0-based variant starts (ascending) and exclusive ends —
    /// passed straight to `Svar1Store::read_window`.
    v_starts_c: Vec<u32>,
    v_ends_c: Vec<u32>,
    /// This contig's reference bytes (offsets are implicitly `[0, ref_bytes.len()]`).
    /// See the module note on per-contig reference.
    ref_bytes: Vec<u8>,
}

/// One pre-expanded window of the fixed traversal: `regions` (0-based half-open, on
/// `contig_idx`) crossed with the contiguous physical-sample sub-range `[s_lo, s_hi)`.
/// The job carries only the two `usize` bounds, NOT a per-window copy of the physical
/// samples — the producer borrows `&phys_sample_idx[s_lo..s_hi]` from the engine's single
/// cohort-scale map when it fills the window (`_plan` always yields a contiguous
/// `arange(s_lo, s_hi)` sample chunk, so the bounds reconstruct it losslessly). This keeps
/// job residency region-scale, never `O(n_windows × window_samples)`. See the module docs.
struct WindowJob {
    contig_idx: usize,
    regions: Vec<(u32, u32)>,
    s_lo: usize,
    s_hi: usize,
}

/// One recycled slot: the whole window's CSR offsets. Its owning job (and thus the
/// contig/regions/sample-range needed to generate from it) is recovered by
/// `StreamEngineCore` from the plan-order job counter — see `stream_core`'s job-ordering
/// note — so the slot itself carries no `job_idx`.
#[derive(Default)]
struct FilledWindow {
    o_starts: Vec<i64>,
    o_stops: Vec<i64>,
}

/// The SVAR1 [`EngineBackend`]: owns the engine's own store, the plan (jobs/contigs),
/// the cohort-scale sample map, and the GLOBAL variant-scale tables. Wrapped in an
/// `Arc` by `StreamEngineCore` and shared between the producer and consumer threads.
struct Svar1Backend {
    store: Svar1Store,
    contigs: Vec<ContigData>,
    jobs: Vec<WindowJob>,
    /// The full public→physical sample map (`O(n_samples)`, ONE copy). Each job's
    /// `[s_lo, s_hi)` slices into this; `fill` borrows `&phys_sample_idx[s_lo..s_hi]` per
    /// window — no per-window owned copy.
    phys_sample_idx: Vec<usize>,
    /// GLOBAL variant-scale tables (indexed by global variant id from `geno_v_idxs`).
    v_starts: Array1<i32>,
    ilens: Array1<i32>,
    alt_alleles: Array1<u8>,
    alt_offsets: Array1<i64>,
    pad_char: u8,
    parallel: bool,
    /// -1 = ragged (per-hap actual length, pre-#277 behavior), >=0 = fixed length
    /// (issue #277 Wave A `with_len`). Forwarded verbatim to `generate_batch_core`.
    output_length: i64,
    /// Issue #277 Wave A Task 4: when `true`, `generate` requests the two annotation
    /// outputs (`annot_v_idxs`/`annot_ref_pos`) from `generate_batch_core`, with
    /// `global_v_idxs = None` — SVAR1's `store.geno_v_idxs()` is already dataset-GLOBAL
    /// (see the module doc's "GLOBAL variant-scale tables" note), so no remap is
    /// needed, even across contigs. `false` (default) preserves pre-Task-4 behavior
    /// exactly (no extra allocation/work).
    annotated: bool,
    /// Wave B PR-B1 (variants-output streaming): when `true`, callers may invoke
    /// [`EngineBackend::generate_variants`]'s `Svar1Backend` override to get flat
    /// variant buffers (`VariantsBatch`) for a batch row slice instead of reconstructed
    /// haplotype bytes. Mirrors `annotated`'s shape; guarded at the top of that override
    /// (bails if `false`) and surfaced to Python via `next_batch_variants`. `false`
    /// (default) preserves pre-Task-4 behavior exactly.
    variants: bool,
}

impl EngineBackend for Svar1Backend {
    type Slot = FilledWindow;

    fn n_jobs(&self) -> usize {
        self.jobs.len()
    }

    /// PRODUCER: read this window's CSR offsets and read-through prefetch the runs the
    /// consumer will read (fault the EXACT pages into the shared page cache — shared
    /// with the standalone `svar1_prefetch_runs` FFI entry, see
    /// `crate::ffi::prefetch_runs_core`'s doc comment).
    fn fill(&self, job_idx: usize, slot: &mut FilledWindow) -> anyhow::Result<()> {
        let job = &self.jobs[job_idx];
        let c = &self.contigs[job.contig_idx];
        // Borrow this window's physical samples from the single cohort-scale map — no
        // per-window owned copy (see the module memory note).
        let phys = &self.phys_sample_idx[job.s_lo..job.s_hi];
        let w = self
            .store
            .read_window(&c.name, &c.v_starts_c, &c.v_ends_c, &job.regions, phys)?;
        slot.o_starts = w.o_starts;
        slot.o_stops = w.o_stops;

        let vidx = self.store.geno_v_idxs();
        crate::ffi::prefetch_runs_core(vidx, &slot.o_starts, &slot.o_stops);
        Ok(())
    }

    fn n_batch_rows(&self, job_idx: usize, _slot: &FilledWindow) -> usize {
        let job = &self.jobs[job_idx];
        job.regions.len() * (job.s_hi - job.s_lo)
    }

    /// CONSUMER: generate the `[row_lo, row_hi)` slice of this window's rows. Output is
    /// `(row_hi-row_lo)`-bounded — never whole-window (issue #284). Delegates to the
    /// shared [`crate::ffi::generate_batch_core`].
    fn generate(
        &self,
        job_idx: usize,
        slot: &FilledWindow,
        row_lo: usize,
        row_hi: usize,
    ) -> anyhow::Result<(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)> {
        let ploidy = self.store.ploidy();
        let job = &self.jobs[job_idx];
        let c = &self.contigs[job.contig_idx];
        let n_samples = job.s_hi - job.s_lo;
        let n_rows = row_hi - row_lo;

        // Per (region, sample) row bounds for rows [row_lo, row_hi), C-order
        // (region, sample): window row bi = ri*n_samples + si -> region regions[bi/n_s].
        let mut rb = Array2::<i32>::zeros((n_rows, 2));
        for (i, bi) in (row_lo..row_hi).enumerate() {
            let ri = bi / n_samples;
            let (s, e) = job.regions[ri];
            rb[[i, 0]] = s as i32;
            rb[[i, 1]] = e as i32;
        }

        // CSR rows for this batch slice: [row_lo*ploidy, row_hi*ploidy).
        let o_lo = row_lo * ploidy;
        let o_hi = row_hi * ploidy;
        let o_starts_b = &slot.o_starts[o_lo..o_hi];
        let o_stops_b = &slot.o_stops[o_lo..o_hi];

        // This contig's reference slice (offsets [0, len]); see the module note.
        let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);
        let ref_view = ndarray::ArrayView1::from(c.ref_bytes.as_slice());

        Ok(crate::ffi::generate_batch_core(
            self.store.geno_v_idxs(),
            ploidy,
            o_starts_b,
            o_stops_b,
            rb.view(),
            self.v_starts.view(),
            self.ilens.view(),
            self.alt_alleles.view(),
            self.alt_offsets.view(),
            ref_view,
            ref_offsets.view(),
            self.pad_char,
            self.output_length,
            None, // shifts -- not yet wired for the engine path (jitter is Task 4+)
            self.annotated,
            None, // global_v_idxs -- SVAR1's geno_v_idxs is already dataset-global
            self.parallel,
        ))
    }

    /// Wave B PR-B1: generate the `[row_lo, row_hi)` slice of this window's rows as flat
    /// VARIANT buffers rather than reconstructed haplotype bytes — the SVAR1 counterpart
    /// of `RecordBackend::generate_variants` (`crate::record_stream::engine`), which this
    /// mirrors closely EXCEPT for two structural differences from SVAR1's on-disk layout:
    ///
    /// 1. **CSR is already region-expanded.** Unlike `DecodedWindow`'s single per-hap CSR
    ///    (`RecordBackend` must replicate a sample's run across regions, see that method's
    ///    doc comment), `slot.o_starts`/`slot.o_stops` (`FilledWindow`, from
    ///    `Svar1Store::read_window`) are ALREADY one absolute index-pair per
    ///    `(region, sample, ploid)` — `find_ranges` did the region expansion on read. So
    ///    this indexes them with the SAME flat `o_lo = row_lo*ploidy` slice `generate`
    ///    above uses, no `bi % n_samples` replication.
    /// 2. **GLOBAL tables, no remap.** `self.store.geno_v_idxs()` yields dataset-GLOBAL
    ///    variant ids directly (see the struct's "GLOBAL variant-scale tables" note), so
    ///    fields are gathered from the backend's GLOBAL `self.v_starts`/`self.ilens`/
    ///    `self.alt_alleles`/`self.alt_offsets` — not a per-window-local table like
    ///    `DecodedWindow`'s. (Still emits no id field in the output — variants output is
    ///    self-contained, issue #313.)
    ///
    /// Same region-overlap clip as `RecordBackend::generate_variants`:
    /// `v_end = v_start - v_ilen.min(0) + 1`, keep iff `v_start < r_e && v_end > r_s`
    /// (`src/genotypes/mod.rs:68-74`'s overlap-keep predicate, NOT `reconstruct/mod.rs`'s
    /// exonic containment check). Assumes phased ploidy: one output row per `(row, ploid)`,
    /// so `row_offsets` has length `(row_hi-row_lo)*ploidy + 1`.
    fn generate_variants(
        &self,
        job_idx: usize,
        slot: &FilledWindow,
        row_lo: usize,
        row_hi: usize,
    ) -> anyhow::Result<crate::variants::VariantsBatch> {
        if !self.variants {
            anyhow::bail!(
                "Svar1StreamEngine.next_batch_variants() called on an engine constructed \
                 with variants=false"
            );
        }
        let ploidy = self.store.ploidy();
        let job = &self.jobs[job_idx];
        let n_samples = job.s_hi - job.s_lo;
        let n_rows = row_hi - row_lo;
        let geno = self.store.geno_v_idxs();

        // CSR rows for this batch slice: [row_lo*ploidy, row_hi*ploidy) — already
        // region-expanded (see point 1 above), same flat slice `generate` uses.
        let o_lo = row_lo * ploidy;

        let mut kept: Vec<i32> = Vec::new();
        let mut row_offsets: Vec<i64> = Vec::with_capacity(n_rows * ploidy + 1);
        row_offsets.push(0);
        for r in 0..(n_rows * ploidy) {
            let bi = row_lo + r / ploidy;
            let ri = bi / n_samples;
            let (r_s, r_e) = job.regions[ri];
            let os = slot.o_starts[o_lo + r] as usize;
            let oe = slot.o_stops[o_lo + r] as usize;
            for o in os..oe {
                let gvi = geno[o];
                let v_start = self.v_starts[gvi as usize] as i64;
                let v_ilen = self.ilens[gvi as usize] as i64;
                let v_end = v_start - v_ilen.min(0) + 1;
                if v_start < r_e as i64 && v_end > r_s as i64 {
                    kept.push(gvi);
                }
            }
            row_offsets.push(kept.len() as i64);
        }

        let v_idxs = Array1::from_vec(kept);
        let (alt_data, alt_seq_offsets, start, ilen) = crate::variants::assemble_variants_window(
            v_idxs.view(),
            self.v_starts.view(),
            self.ilens.view(),
            self.alt_alleles.view(),
            self.alt_offsets.view(),
        );

        Ok(crate::variants::VariantsBatch {
            alt_data,
            alt_seq_offsets,
            start,
            ilen,
            row_offsets: Array1::from_vec(row_offsets),
        })
    }
}

/// Producer/consumer SVAR1 streamer (issue #283). See the module docs for the design.
#[pyclass]
pub struct Svar1StreamEngine {
    core: StreamEngineCore<Svar1Backend>,
}

impl Svar1StreamEngine {
    /// Shared constructor for `#[new]` and Rust tests. Registers each contig's meta on
    /// the store, then wraps everything for the producer/consumer loop. `store` is opened
    /// but not yet contig-registered.
    #[allow(clippy::too_many_arguments)]
    fn build(
        mut store: Svar1Store,
        contigs: Vec<ContigData>,
        jobs: Vec<WindowJob>,
        phys_sample_idx: Vec<usize>,
        v_starts: Array1<i32>,
        ilens: Array1<i32>,
        alt_alleles: Array1<u8>,
        alt_offsets: Array1<i64>,
        pad_char: u8,
        parallel: bool,
        batch_size: usize,
        output_length: i64,
        annotated: bool,
        variants: bool,
    ) -> Self {
        for c in &contigs {
            store.set_contig_meta_rs(&c.name, c.contig_start, c.n_local, c.max_v_len);
        }
        let backend = Svar1Backend {
            store,
            contigs,
            jobs,
            phys_sample_idx,
            v_starts,
            ilens,
            alt_alleles,
            alt_offsets,
            pad_char,
            parallel,
            output_length,
            annotated,
            variants,
        };
        Self {
            core: StreamEngineCore::new(Arc::new(backend), batch_size),
        }
    }

    /// Test-only accessor mirroring the pyclass's `next_batch`, minus the numpy
    /// marshaling — used directly by the `#[cfg(test)]` module below.
    #[cfg(test)]
    fn next_batch_core(
        &self,
    ) -> Option<anyhow::Result<(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)>>
    {
        self.core.next_batch_core()
    }
}

#[pymethods]
impl Svar1StreamEngine {
    /// Construct the engine: open the store, register per-contig meta, and hold the plan
    /// (pre-expanded window jobs) plus the global variant tables. All per-contig / per-job
    /// arrays cross as owned `Vec`s so the producer thread can hold them `'static`.
    ///
    /// Per-contig records are parallel arrays indexed by contig: `contig_names[i]`,
    /// `contig_starts[i]`, `n_locals[i]`, `max_v_lens[i]`, `v_starts_c[i]`, `v_ends_c[i]`,
    /// `contig_ref_bytes[i]`. The full public→physical sample map `phys_sample_idx`
    /// (length `n_samples`) crosses ONCE. Per-job records are parallel arrays indexed by
    /// job: `job_contig_idx[j]`, `job_region_starts[j]`, `job_region_ends[j]`,
    /// `job_s_lo[j]`, `job_s_hi[j]` — each job carries only its contiguous physical-sample
    /// sub-range `[s_lo, s_hi)` into `phys_sample_idx`, NOT a per-window sample copy.
    #[new]
    #[pyo3(signature = (
        store_path, n_samples, ploidy,
        contig_names, contig_starts, n_locals, max_v_lens, v_starts_c, v_ends_c,
        contig_ref_bytes, phys_sample_idx,
        job_contig_idx, job_region_starts, job_region_ends, job_s_lo, job_s_hi,
        v_starts, ilens, alt_alleles, alt_offsets,
        pad_char, parallel, batch_size, output_length, annotated=false,
        variants=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        store_path: &str,
        n_samples: usize,
        ploidy: usize,
        contig_names: Vec<String>,
        contig_starts: Vec<u32>,
        n_locals: Vec<usize>,
        max_v_lens: Vec<u32>,
        v_starts_c: Vec<Vec<u32>>,
        v_ends_c: Vec<Vec<u32>>,
        contig_ref_bytes: Vec<Vec<u8>>,
        phys_sample_idx: Vec<usize>,
        job_contig_idx: Vec<usize>,
        job_region_starts: Vec<Vec<u32>>,
        job_region_ends: Vec<Vec<u32>>,
        job_s_lo: Vec<usize>,
        job_s_hi: Vec<usize>,
        v_starts: PyReadonlyArray1<i32>,
        ilens: PyReadonlyArray1<i32>,
        alt_alleles: PyReadonlyArray1<u8>,
        alt_offsets: PyReadonlyArray1<i64>,
        pad_char: u8,
        parallel: bool,
        batch_size: usize,
        output_length: i64,
        annotated: bool,
        variants: bool,
    ) -> PyResult<Self> {
        let store = Svar1Store::open_meta(store_path, n_samples, ploidy)?;

        let n_contigs = contig_names.len();
        if [
            contig_starts.len(),
            n_locals.len(),
            max_v_lens.len(),
            v_starts_c.len(),
            v_ends_c.len(),
            contig_ref_bytes.len(),
        ]
        .iter()
        .any(|&l| l != n_contigs)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Svar1StreamEngine: per-contig arrays must all have the same length",
            ));
        }
        let mut contigs = Vec::with_capacity(n_contigs);
        for i in 0..n_contigs {
            contigs.push(ContigData {
                name: contig_names[i].clone(),
                contig_start: contig_starts[i],
                n_local: n_locals[i],
                max_v_len: max_v_lens[i],
                v_starts_c: v_starts_c[i].clone(),
                v_ends_c: v_ends_c[i].clone(),
                ref_bytes: contig_ref_bytes[i].clone(),
            });
        }

        if phys_sample_idx.len() != n_samples {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Svar1StreamEngine: phys_sample_idx has length {} but n_samples={n_samples}",
                phys_sample_idx.len()
            )));
        }

        let n_jobs = job_contig_idx.len();
        if [
            job_region_starts.len(),
            job_region_ends.len(),
            job_s_lo.len(),
            job_s_hi.len(),
        ]
        .iter()
        .any(|&l| l != n_jobs)
        {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Svar1StreamEngine: per-job arrays must all have the same length",
            ));
        }
        let mut jobs = Vec::with_capacity(n_jobs);
        for j in 0..n_jobs {
            let starts = &job_region_starts[j];
            let ends = &job_region_ends[j];
            if starts.len() != ends.len() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Svar1StreamEngine: job_region_starts and job_region_ends must match",
                ));
            }
            let (s_lo, s_hi) = (job_s_lo[j], job_s_hi[j]);
            if s_lo > s_hi || s_hi > n_samples {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Svar1StreamEngine: job sample range [{s_lo}, {s_hi}) is invalid \
                     for n_samples={n_samples}",
                )));
            }
            let regions: Vec<(u32, u32)> = starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect();
            jobs.push(WindowJob {
                contig_idx: job_contig_idx[j],
                regions,
                s_lo,
                s_hi,
            });
        }

        Ok(Self::build(
            store,
            contigs,
            jobs,
            phys_sample_idx,
            v_starts.as_array().to_owned(),
            ilens.as_array().to_owned(),
            alt_alleles.as_array().to_owned(),
            alt_offsets.as_array().to_owned(),
            pad_char,
            parallel,
            batch_size,
            output_length,
            annotated,
            variants,
        ))
    }

    /// Return the next batch's `(data, offsets)`, or `None` when the plan is exhausted.
    /// The GIL is released for the whole blocking body (recv/generate/join); it is
    /// reacquired only to marshal the owned arrays into numpy. A producer error/panic
    /// surfaces here as a `RuntimeError`, join-then-classified — never a hang.
    fn next_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>> {
        let out = py.detach(|| self.core.next_batch_core());
        match out {
            None => Ok(None),
            Some(Ok((data, _annot_v, _annot_pos, offsets))) => {
                Ok(Some((data.into_pyarray(py), offsets.into_pyarray(py))))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Annotated counterpart of `next_batch` (issue #277 Wave A Task 4): returns
    /// `(data, annot_v_idxs, annot_ref_pos, offsets)`, or `None` when the plan is
    /// exhausted. Only valid when the engine was constructed with `annotated=true` —
    /// otherwise this raises `RuntimeError` (a caller bug, not a data condition;
    /// `StreamingDataset._iter_batches` never calls this unless `annotated` was
    /// passed to `build_engine`).
    fn next_batch_annotated<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<
        Option<(
            Bound<'py, PyArray1<u8>>,
            Bound<'py, PyArray1<i32>>,
            Bound<'py, PyArray1<i32>>,
            Bound<'py, PyArray1<i64>>,
        )>,
    > {
        let out = py.detach(|| self.core.next_batch_core());
        match out {
            None => Ok(None),
            Some(Ok((data, Some(annot_v), Some(annot_pos), offsets))) => Ok(Some((
                data.into_pyarray(py),
                annot_v.into_pyarray(py),
                annot_pos.into_pyarray(py),
                offsets.into_pyarray(py),
            ))),
            Some(Ok((_, None, _, _))) | Some(Ok((_, _, None, _))) => {
                Err(PyRuntimeError::new_err(
                    "Svar1StreamEngine.next_batch_annotated() called on an engine \
                     constructed with annotated=false",
                ))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// Variants-output counterpart of `next_batch` (Wave B PR-B1): returns a `dict` with
    /// keys `alt` (u8), `alt_offsets` (i64, ragged ALT-byte boundaries), `start` (i32),
    /// `ilen` (i32), `offsets` (i64, per-`(row, ploid)` variant-count boundaries) — or
    /// `None` when the plan is exhausted. Only valid when the engine was constructed with
    /// `variants=true` — otherwise this raises `RuntimeError` (a caller bug, not a data
    /// condition; `StreamingDataset._iter_batches` never calls this unless `variants` was
    /// passed to `build_engine`). Mirrors `RecordStreamEngine::next_batch_variants`
    /// (`crate::record_stream::engine`) exactly.
    fn next_batch_variants<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let out = py.detach(|| self.core.next_batch_variants_core());
        match out {
            None => Ok(None),
            Some(Ok(batch)) => {
                let dict = PyDict::new(py);
                dict.set_item("alt", batch.alt_data.into_pyarray(py))?;
                dict.set_item("alt_offsets", batch.alt_seq_offsets.into_pyarray(py))?;
                dict.set_item("start", batch.start.into_pyarray(py))?;
                dict.set_item("ilen", batch.ilen.into_pyarray(py))?;
                dict.set_item("offsets", batch.row_offsets.into_pyarray(py))?;
                Ok(Some(dict))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_raw<T: bytemuck::NoUninit>(dir: &std::path::Path, name: &str, data: &[T]) {
        let mut f = std::fs::File::create(dir.join(name)).unwrap();
        f.write_all(bytemuck::cast_slice(data)).unwrap();
    }

    /// The 4-hap fixture shared with `store.rs`: variant_idxs=[0,0,1,1],
    /// offsets=[0,1,1,3,4], 2 samples x ploidy 2. Per-hap sorted global ids:
    ///   hap0:[0]  hap1:[]  hap2:[0,1]  hap3:[1]
    /// Two SNPs: var0 @10 alt 'A', var1 @20 alt 'C'. chr1 reference: 30 bp of 'T'.
    struct Fixture {
        _tmp: tempfile::TempDir,
        path: String,
        v_starts: Array1<i32>,
        ilens: Array1<i32>,
        alt_alleles: Array1<u8>,
        alt_offsets: Array1<i64>,
        ref_bytes: Vec<u8>,
    }

    fn fixture() -> Fixture {
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 0, 1, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 1, 3, 4]);
        let path = tmp.path().to_str().unwrap().to_string();
        Fixture {
            _tmp: tmp,
            path,
            v_starts: Array1::from(vec![10i32, 20]),
            ilens: Array1::from(vec![0i32, 0]),
            alt_alleles: Array1::from(vec![b'A', b'C']),
            alt_offsets: Array1::from(vec![0i64, 1, 2]),
            ref_bytes: vec![b'T'; 30],
        }
    }

    fn chr1_contig(f: &Fixture) -> ContigData {
        ContigData {
            name: "chr1".into(),
            contig_start: 0,
            n_local: 2,
            max_v_len: 1,
            v_starts_c: vec![10, 20],
            v_ends_c: vec![11, 21],
            ref_bytes: f.ref_bytes.clone(),
        }
    }

    /// Ground truth for one window: a direct `read_window` + a single full-window
    /// `generate_batch_core`, on an independent store instance.
    fn expected_window(
        f: &Fixture,
        regions: &[(u32, u32)],
        phys_samples: &[usize],
    ) -> (Vec<u8>, Vec<i64>) {
        let mut store = Svar1Store::open_meta(&f.path, 2, 2).unwrap();
        store.set_contig_meta_rs("chr1", 0, 2, 1);
        let w = store
            .read_window("chr1", &[10, 20], &[11, 21], regions, phys_samples)
            .unwrap();

        let n_regions = regions.len();
        let n_samples = phys_samples.len();
        let n_rows = n_regions * n_samples;
        let mut rb = Array2::<i32>::zeros((n_rows, 2));
        for bi in 0..n_rows {
            let ri = bi / n_samples;
            rb[[bi, 0]] = regions[ri].0 as i32;
            rb[[bi, 1]] = regions[ri].1 as i32;
        }
        let ref_offsets = Array1::from(vec![0i64, f.ref_bytes.len() as i64]);
        let (data, _annot_v, _annot_pos, offs) = crate::ffi::generate_batch_core(
            store.geno_v_idxs(),
            store.ploidy(),
            &w.o_starts,
            &w.o_stops,
            rb.view(),
            f.v_starts.view(),
            f.ilens.view(),
            f.alt_alleles.view(),
            f.alt_offsets.view(),
            ndarray::ArrayView1::from(f.ref_bytes.as_slice()),
            ref_offsets.view(),
            b'N',
            -1, // ragged
            None,
            false, // annotated
            None,  // global_v_idxs
            false,
        );
        (data.to_vec(), offs.to_vec())
    }

    fn build_engine(f: &Fixture, jobs: Vec<WindowJob>, batch_size: usize) -> Svar1StreamEngine {
        let store = Svar1Store::open_meta(&f.path, 2, 2).unwrap();
        // Identity public→physical map for the 2-sample fixture; jobs slice it by range.
        Svar1StreamEngine::build(
            store,
            vec![chr1_contig(f)],
            jobs,
            vec![0usize, 1],
            f.v_starts.clone(),
            f.ilens.clone(),
            f.alt_alleles.clone(),
            f.alt_offsets.clone(),
            b'N',
            false,
            batch_size,
            -1,    // ragged
            false, // annotated
            false, // variants
        )
    }

    /// The 8a gate: a >=2-window plan flows through the producer/consumer path; every
    /// batch arrives in plan order and byte-equals a direct `read_window` + generate; the
    /// plan exhausts and `next_batch` returns `None` cleanly (idempotently, no hang).
    #[test]
    fn svar1_stream_engine_yields_windows_in_plan_order() {
        let f = fixture();
        // Two windows on chr1: full region, then a narrower region (variant 0 only).
        let jobs = vec![
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
            },
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 15)],
                s_lo: 0,
                s_hi: 2,
            },
        ];
        // batch_size larger than any window -> one batch per window.
        let engine = build_engine(&f, jobs, 1000);

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, _av, _ap, o) = r.expect("no producer error");
            batches.push((d.to_vec(), o.to_vec()));
        }
        assert_eq!(batches.len(), 2, "one batch per window, in plan order");

        let exp0 = expected_window(&f, &[(0, 30)], &[0, 1]);
        let exp1 = expected_window(&f, &[(0, 15)], &[0, 1]);
        assert_eq!(
            batches[0], exp0,
            "window 0 data/offsets must match direct read"
        );
        assert_eq!(
            batches[1], exp1,
            "window 1 data/offsets must match direct read"
        );

        // Exhaustion is clean and idempotent (must not hang or re-join).
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }

    /// batch_size=1 splits each window into per-row batches (issue #284 bounding). The
    /// concatenation of a window's batch data must equal the full-window generate, and the
    /// batch count must equal the total window-row count. Exercises the multi-batch drain
    /// + slot-recycle path under threading.
    #[test]
    fn svar1_stream_engine_splits_windows_into_bounded_batches() {
        let f = fixture();
        let jobs = vec![
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
            },
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
            },
        ];
        let engine = build_engine(&f, jobs, 1);

        let mut all: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, _av, _ap, o) = r.expect("no producer error");
            all.push((d.to_vec(), o.to_vec()));
        }
        // Each window has 1 region * 2 samples = 2 rows -> 2 batches; 2 windows -> 4.
        assert_eq!(
            all.len(),
            4,
            "batch_size=1 splits each 2-row window into 2 batches"
        );

        let exp = expected_window(&f, &[(0, 30)], &[0, 1]);
        // Window 0's two batches concatenate to the full-window data.
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
    fn svar1_stream_engine_empty_plan_is_none() {
        let f = fixture();
        let engine = build_engine(&f, Vec::new(), 8);
        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }

    /// Serializes the two tests that swap the process-global panic hook. See the identical
    /// guard in `crate::stream`'s tests: `take_hook`/`set_hook` is not atomic, so without
    /// serialization one test's temporary silent hook can be observed by another as "the
    /// previous hook" and restored permanently. Diagnostics-only, but no reason to leave it
    /// racy. A panicking test poisons the mutex; recover via `into_inner()`.
    static PANIC_HOOK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// I1 (error branch): a producer `read_window` error must surface through the
    /// consumer's join-then-classify path as `Some(Err(_))` — NOT a clean/empty EOF and
    /// NOT a hang. Mirrors `run_windows`' `FailingBackend` regression, adapted to the
    /// engine. Here the contig is registered with `n_local=5` but its `v_starts_c` has
    /// length 2, so `Svar1Store::read_window` bails (`... has n_local=5 but got v_starts=2`),
    /// the producer returns `Err`, and the classify `Ok(Err(e))` branch fires.
    #[test]
    fn svar1_stream_engine_producer_error_surfaces_not_eof() {
        let f = fixture();
        let bad_contig = ContigData {
            name: "chr1".into(),
            contig_start: 0,
            n_local: 5, // deliberately mismatched vs v_starts_c.len() == 2
            max_v_len: 1,
            v_starts_c: vec![10, 20],
            v_ends_c: vec![11, 21],
            ref_bytes: f.ref_bytes.clone(),
        };
        let jobs = vec![WindowJob {
            contig_idx: 0,
            regions: vec![(0, 30)],
            s_lo: 0,
            s_hi: 2,
        }];
        let store = Svar1Store::open_meta(&f.path, 2, 2).unwrap();
        let engine = Svar1StreamEngine::build(
            store,
            vec![bad_contig],
            jobs,
            vec![0usize, 1],
            f.v_starts.clone(),
            f.ilens.clone(),
            f.alt_alleles.clone(),
            f.alt_offsets.clone(),
            b'N',
            false,
            8,
            -1,    // ragged
            false, // annotated
            false, // variants
        );

        match engine.next_batch_core() {
            Some(Err(e)) => {
                let msg = format!("{e:?}");
                assert!(
                    msg.contains("n_local"),
                    "expected the read_window error to propagate, got: {msg}"
                );
                assert!(
                    !msg.contains("panicked"),
                    "an fill() Err must classify as Ok(Err), NOT the panic branch: {msg}"
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

    /// I1 (panic branch): a producer PANIC must surface through join-then-classify as the
    /// `Err(_)` ("producer thread panicked") branch — distinct from the `Ok(Err)` error
    /// case above — and must not hang. Mirrors `run_windows`' `PanickingBackend`. The
    /// panic is induced with a job whose `contig_idx` is out of range of the engine's
    /// `contigs` vec, so the producer's `&contigs[job.contig_idx]` is a clean, fully
    /// in-our-own-code `Vec` bounds-check panic (no dependence on genoray internals / no
    /// corrupt store) that unwinds the producer thread and is caught by `.join() == Err`.
    #[test]
    fn svar1_stream_engine_producer_panic_surfaces_not_hang() {
        // Serialize the global panic-hook swap; recover from a poisoned lock.
        let _guard = PANIC_HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let f = fixture();
        // Only contig index 0 exists; this job points at 5 -> producer panics on index.
        let jobs = vec![WindowJob {
            contig_idx: 5,
            regions: vec![(0, 30)],
            s_lo: 0,
            s_hi: 2,
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

    /// I2: multi-contig per-contig reference. Two contigs with DISTINGUISHABLE references
    /// (chr1 = all 'G', chr2 = all 'T'); a job on each. The chr2 window's output must be
    /// reconstructed against chr2's ref slice, NOT chr1's (the bug a single engine-level
    /// `ref_` would cause). An empty hap on chr2 yields pure chr2 ref bytes, making the ref
    /// choice directly observable. Asserts the engine's chr2 batch equals a direct generate
    /// with chr2's ref AND differs from the same generate fed chr1's ref.
    #[test]
    fn svar1_stream_engine_uses_per_contig_reference() {
        // Store: 1 sample x ploidy 2 -> 2 haps. Global var 0 on chr1, global var 1 on chr2.
        //   hap0 -> variant_idxs[0..1] = [0] (chr1 var);  hap1 -> [1..2] = [1] (chr2 var)
        let tmp = tempfile::tempdir().unwrap();
        write_raw::<i32>(tmp.path(), "variant_idxs.npy", &[0, 1]);
        write_raw::<i64>(tmp.path(), "offsets.npy", &[0, 1, 2]);
        let path = tmp.path().to_str().unwrap().to_string();

        // Global variant tables: var0 @ chr1 pos10 alt 'A'; var1 @ chr2 pos5 alt 'C'.
        let v_starts = Array1::from(vec![10i32, 5]);
        let ilens = Array1::from(vec![0i32, 0]);
        let alt_alleles = Array1::from(vec![b'A', b'C']);
        let alt_offsets = Array1::from(vec![0i64, 1, 2]);
        let chr1_ref = vec![b'G'; 20];
        let chr2_ref = vec![b'T'; 20];

        let chr1 = ContigData {
            name: "chr1".into(),
            contig_start: 0,
            n_local: 1,
            max_v_len: 1,
            v_starts_c: vec![10],
            v_ends_c: vec![11],
            ref_bytes: chr1_ref.clone(),
        };
        let chr2 = ContigData {
            name: "chr2".into(),
            contig_start: 1, // chr2's first global variant id is 1
            n_local: 1,
            max_v_len: 1,
            v_starts_c: vec![5],
            v_ends_c: vec![6],
            ref_bytes: chr2_ref.clone(),
        };

        // Jobs: window on chr1 (region [0,20)), then on chr2 (region [0,10)).
        let jobs = vec![
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 20)],
                s_lo: 0,
                s_hi: 1,
            },
            WindowJob {
                contig_idx: 1,
                regions: vec![(0, 10)],
                s_lo: 0,
                s_hi: 1,
            },
        ];

        let store = Svar1Store::open_meta(&path, 1, 2).unwrap();
        let engine = Svar1StreamEngine::build(
            store,
            vec![chr1, chr2],
            jobs,
            vec![0usize], // 1-sample store: identity map
            v_starts.clone(),
            ilens.clone(),
            alt_alleles.clone(),
            alt_offsets.clone(),
            b'N',
            false,
            1000,
            -1,    // ragged
            false, // annotated
            false, // variants
        );

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, _av, _ap, o) = r.expect("no producer error");
            batches.push((d.to_vec(), o.to_vec()));
        }
        assert_eq!(batches.len(), 2, "one batch per window");

        // Direct chr2-window generate with the CORRECT (chr2) ref, and with the WRONG
        // (chr1) ref, on an independent store with both contigs registered.
        let direct = |ref_bytes: &[u8]| -> (Vec<u8>, Vec<i64>) {
            let mut s = Svar1Store::open_meta(&path, 1, 2).unwrap();
            s.set_contig_meta_rs("chr1", 0, 1, 1);
            s.set_contig_meta_rs("chr2", 1, 1, 1);
            let w = s.read_window("chr2", &[5], &[6], &[(0, 10)], &[0]).unwrap();
            // 1 region * 1 sample = 1 row; region bounds [0, 10).
            let mut rb = Array2::<i32>::zeros((1, 2));
            rb[[0, 0]] = 0;
            rb[[0, 1]] = 10;
            let ref_offsets = Array1::from(vec![0i64, ref_bytes.len() as i64]);
            let (data, _annot_v, _annot_pos, offs) = crate::ffi::generate_batch_core(
                s.geno_v_idxs(),
                s.ploidy(),
                &w.o_starts,
                &w.o_stops,
                rb.view(),
                v_starts.view(),
                ilens.view(),
                alt_alleles.view(),
                alt_offsets.view(),
                ndarray::ArrayView1::from(ref_bytes),
                ref_offsets.view(),
                b'N',
                -1, // ragged
                None,
                false, // annotated
                None,  // global_v_idxs
                false,
            );
            (data.to_vec(), offs.to_vec())
        };

        let exp_correct = direct(&chr2_ref);
        let exp_wrong = direct(&chr1_ref);
        assert_ne!(
            exp_correct, exp_wrong,
            "sanity: chr1 vs chr2 ref must produce different output (else the test proves nothing)"
        );
        assert_eq!(
            batches[1], exp_correct,
            "chr2 window must reconstruct against chr2's ref slice"
        );
        assert_ne!(
            batches[1], exp_wrong,
            "chr2 window must NOT use chr1's ref (the single-engine-ref bug)"
        );
        // The chr2 empty hap (hap0, no chr2 variant) is pure chr2 ref ('T'), never 'G'.
        assert!(
            batches[1].0.contains(&b'T') && !batches[1].0.contains(&b'G'),
            "chr2 output must contain chr2 ref bytes ('T') and none of chr1's ('G')"
        );
    }

    /// M1: dropping the engine mid-stream (producer still live/blocked) must not hang and
    /// must not leak the detached thread — `Drop for EngineState` closes the channels and
    /// joins. A 4-window plan with `batch_size=1`: after pulling ONE row the producer has
    /// filled the 2 prefill slots and is blocked on `rx_free.recv()` for a third; dropping
    /// the engine here closes `free`, unblocking it. If the join wedged, this test (and the
    /// 20x race loop) would hang.
    #[test]
    fn svar1_stream_engine_drop_midstream_joins_cleanly() {
        let f = fixture();
        let jobs = (0..4)
            .map(|_| WindowJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                s_lo: 0,
                s_hi: 2,
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

    /// Finding 1 guard: a job's `[s_lo, s_hi)` must slice the ENGINE'S single
    /// `phys_sample_idx` map (borrowed per window), NOT be a per-window copy. With a
    /// NON-identity map (samples swapped) a full-cohort window must reconstruct the
    /// PHYSICAL samples `phys_sample_idx[0..2] = [1, 0]` in that order — i.e. it must
    /// byte-equal a direct `read_window` fed `&[1, 0]` and DIFFER from one fed `&[0, 1]`.
    /// If the engine ignored the map (or sliced the wrong thing) this would fail.
    #[test]
    fn svar1_stream_engine_slices_phys_samples_by_range() {
        let f = fixture();
        // The two fixture samples produce distinguishable haps (sample 0: hap0=[0],
        // hap1=[]; sample 1: hap2=[0,1], hap3=[1]), so a swap changes the output.
        let jobs = vec![WindowJob {
            contig_idx: 0,
            regions: vec![(0, 30)],
            s_lo: 0,
            s_hi: 2,
        }];
        let store = Svar1Store::open_meta(&f.path, 2, 2).unwrap();
        let engine = Svar1StreamEngine::build(
            store,
            vec![chr1_contig(&f)],
            jobs,
            vec![1usize, 0], // swapped map: public 0 -> physical 1, public 1 -> physical 0
            f.v_starts.clone(),
            f.ilens.clone(),
            f.alt_alleles.clone(),
            f.alt_offsets.clone(),
            b'N',
            false,
            1000,
            -1,    // ragged
            false, // annotated
            false, // variants
        );

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, _av, _ap, o) = r.expect("no producer error");
            batches.push((d.to_vec(), o.to_vec()));
        }
        assert_eq!(
            batches.len(),
            1,
            "one batch for the single full-cohort window"
        );

        let exp_swapped = expected_window(&f, &[(0, 30)], &[1, 0]);
        let exp_identity = expected_window(&f, &[(0, 30)], &[0, 1]);
        assert_ne!(
            exp_swapped, exp_identity,
            "sanity: swapping physical samples must change the output"
        );
        assert_eq!(
            batches[0], exp_swapped,
            "window must reconstruct phys_sample_idx[s_lo..s_hi] = [1, 0], not [0, 1]"
        );
    }
}
