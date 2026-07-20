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
use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::ffi::stream_core::{EngineBackend, StreamEngineCore};
use crate::record_stream::pgen::PgenWindowFiller;
use crate::record_stream::vcf::VcfWindowFiller;
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
    /// -1 = ragged (per-hap actual length, pre-#277 behavior), >=0 = fixed length
    /// (issue #277 Wave A `with_len`). Forwarded verbatim to `generate_batch_core`.
    output_length: i64,
    /// Issue #277 Wave A Task 4: when `true`, `generate` requests the two annotation
    /// outputs (`annot_v_idxs`/`annot_ref_pos`) from `generate_batch_core`, mapped to
    /// dataset-GLOBAL ids via a per-variant gather through the filled slot's
    /// `global_v_idxs` (issue #305; see `remap_annot_local_to_global`). `false`
    /// (default) preserves pre-Task-4 behavior exactly (no extra allocation/work).
    annotated: bool,
}

impl RecordBackend {
    /// Test-only accessor (issue #276 task 7): decode `job`'s window into a fresh scratch
    /// `DecodedWindow` via the filler, bypassing the producer/consumer channel machinery
    /// entirely — no genotype CSR generation, just the local static table. Backs
    /// `RecordStreamEngine::debug_decode_window`, the parity gate that pins the streamed
    /// window's local variant table against the written dataset's stored variant table for
    /// the same VCF (see that pymethod's doc comment for why this exists).
    fn debug_fill(&self, job: &RecordJob) -> anyhow::Result<DecodedWindow> {
        let c = &self.contigs[job.contig_idx];
        let mut slot = DecodedWindow::default();
        self.filler.fill(job, c, &mut slot)?;
        Ok(slot)
    }
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
    /// a whole-store mmap AND are already region-expanded (`find_ranges` emits one pair per
    /// (region, sample, ploid)). `DecodedWindow`'s per-hap CSR is instead a single
    /// window-local `geno_offsets` (length `n_samples*ploidy + 1`, NO region dimension): hap
    /// `h`'s run is `geno_v_idxs[geno_offsets[h]..geno_offsets[h+1]]`. That asymmetry is the
    /// crux — a batch row `bi` maps to sample `si = bi % n_samples`, so for `regions.len() >
    /// 1` we must REPLICATE sample `si`'s run across regions (the kernel clips it per-row via
    /// `rb`), not flat-slice `geno_offsets[bi*ploidy..]` (which runs off the per-sample CSR).
    /// `o_starts_b`/`o_stops_b` are built here as owned `Vec<i64>` since `generate_batch_core`
    /// wants matching start/stop slices, not one shared array. See the assembly below.
    fn generate(
        &self,
        job_idx: usize,
        slot: &DecodedWindow,
        row_lo: usize,
        row_hi: usize,
    ) -> anyhow::Result<(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)> {
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

        // CSR rows for this batch slice, expanded per (region, sample). `slot.geno_offsets`
        // is a per-hap CSR over the window's `n_samples * ploidy` haps ONLY — it has NO
        // region dimension (length `n_samples*ploidy + 1`), unlike SVAR1's `read_window`
        // output which is ALREADY region-expanded (`find_ranges` emits one CSR-index pair
        // per (region, sample, ploid)). So the naive `geno_offsets[row_lo*ploidy..]` slice
        // is wrong for `regions.len() > 1`: row `bi` maps to sample `si = bi % n_samples`
        // (region `ri = bi / n_samples`), and once `bi >= n_samples` (2nd+ region) the flat
        // offset runs off the per-sample CSR — out of bounds, or the wrong sample's run.
        //
        // Every region of a given sample reuses that sample's SAME per-hap run; the kernel
        // clips it to the per-row region bounds `rb` (upstream SNPs via `v_pos < ref_idx`,
        // spanning DELs via the DEL-span branch, downstream variants via the out-buffer
        // break). This is byte-identical to SVAR1: `find_ranges`' per-region narrowing is an
        // overshoot-safe overlap window (`max_v_len`-padded), which the same kernel then
        // clips. So expand the per-sample CSR across regions here, keeping the transpose
        // region-agnostic: for each batch row's `ploidy` haps push sample `si`'s offset pair.
        let mut o_starts_b: Vec<i64> = Vec::with_capacity(n_rows * self.ploidy);
        let mut o_stops_b: Vec<i64> = Vec::with_capacity(n_rows * self.ploidy);
        for bi in row_lo..row_hi {
            let si = bi % n_samples;
            for p in 0..self.ploidy {
                let h = si * self.ploidy + p;
                o_starts_b.push(slot.geno_offsets[h]);
                o_stops_b.push(slot.geno_offsets[h + 1]);
            }
        }

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
            self.output_length,
            None, // shifts -- not yet wired for the engine path (jitter is Task 4+)
            self.annotated,
            Some(ndarray::ArrayView1::from(slot.global_v_idxs.as_slice())),
            self.parallel,
        ))
    }
}

/// Producer/consumer record-stream engine (issue #276 task 3b). Python surface (issue
/// #276 tasks 5/11): `#[new]` builds a `VcfWindowFiller` for `source_kind="vcf"` or a
/// `PgenWindowFiller` for `source_kind="pgen"` and calls [`RecordStreamEngine::new_rs`];
/// `next_batch` mirrors `Svar1StreamEngine`'s (`crate::ffi::stream_engine`) shape exactly.
#[pyclass]
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
        output_length: i64,
        annotated: bool,
    ) -> Self {
        let backend = RecordBackend {
            filler,
            contigs,
            jobs,
            n_samples,
            ploidy,
            pad_char,
            parallel,
            output_length,
            annotated,
        };
        Self {
            core: StreamEngineCore::new(Arc::new(backend), batch_size),
        }
    }

    /// Return the next batch's `(data, annot_v_idxs, annot_ref_pos, offsets)`, or
    /// `None` when the plan is exhausted. Delegates to the shared core; see
    /// `stream_core`'s doc comment for the exhaustion/error/panic classification
    /// contract. The two annotation arrays are `Some` iff the engine was constructed
    /// with `annotated: true`.
    pub fn next_batch_core(
        &self,
    ) -> Option<anyhow::Result<(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)>>
    {
        self.core.next_batch_core()
    }
}

#[pymethods]
impl RecordStreamEngine {
    /// Construct the engine from Python. Per-contig records are parallel arrays indexed
    /// by contig: `contig_names[i]`/`contig_ref_bytes[i]`. Per-job records are parallel
    /// arrays indexed by job: `job_contig_idx[j]`, `job_region_starts[j]`,
    /// `job_region_ends[j]`, `job_s_lo[j]`, `job_s_hi[j]` — mirrors `Svar1StreamEngine`'s
    /// `#[new]` shape (`crate::ffi::stream_engine`) minus the SVAR1-only store/physical-
    /// sample-map arguments (a record-stream job's `[s_lo, s_hi)` indexes straight into
    /// `sample_names`, no public->physical indirection).
    ///
    /// `source_kind` selects the filler: `"vcf"` builds a [`VcfWindowFiller`] (`vcf_path`,
    /// `fasta_path` — see that module's doc for the `fasta_path: None` parity default);
    /// `"pgen"` builds a [`PgenWindowFiller`] over the SAME `vcf_path` param (it carries
    /// the `.pgen` path for this source kind — the filler derives the sibling `.pvar` and
    /// `.psam` itself, see `pgen.rs`'s doc), passing `sample_names` (the sorted-name
    /// `sample_idx` order) so the filler can map it onto the physical `.psam` column order
    /// and read the full on-disk cohort size from the `.psam`. Requires `ploidy == 2`
    /// (PGEN is diploid-only; the filler hardwires `PGEN_PLOIDY` internally and has no
    /// ploidy parameter of its own).
    #[new]
    #[pyo3(signature = (
        source_kind, vcf_path, sample_names, ploidy,
        contig_names, contig_ref_bytes,
        job_contig_idx, job_region_starts, job_region_ends, job_s_lo, job_s_hi,
        fasta_path, pad_char, parallel, batch_size, output_length, annotated=false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        source_kind: &str,
        vcf_path: String,
        sample_names: Vec<String>,
        ploidy: usize,
        contig_names: Vec<String>,
        contig_ref_bytes: Vec<Vec<u8>>,
        job_contig_idx: Vec<usize>,
        job_region_starts: Vec<Vec<u32>>,
        job_region_ends: Vec<Vec<u32>>,
        job_s_lo: Vec<usize>,
        job_s_hi: Vec<usize>,
        fasta_path: Option<String>,
        pad_char: u8,
        parallel: bool,
        batch_size: usize,
        output_length: i64,
        annotated: bool,
    ) -> PyResult<Self> {
        let n_contigs = contig_names.len();
        if contig_ref_bytes.len() != n_contigs {
            return Err(PyValueError::new_err(
                "RecordStreamEngine: contig_names and contig_ref_bytes must have the same length",
            ));
        }
        let contigs: Vec<ContigRef> = contig_names
            .into_iter()
            .zip(contig_ref_bytes)
            .map(|(name, ref_bytes)| ContigRef { name, ref_bytes })
            .collect();

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
            return Err(PyValueError::new_err(
                "RecordStreamEngine: per-job arrays must all have the same length",
            ));
        }
        let n_samples = sample_names.len();
        let mut jobs = Vec::with_capacity(n_jobs);
        for j in 0..n_jobs {
            let starts = &job_region_starts[j];
            let ends = &job_region_ends[j];
            if starts.len() != ends.len() {
                return Err(PyValueError::new_err(
                    "RecordStreamEngine: job_region_starts and job_region_ends must match",
                ));
            }
            let (s_lo, s_hi) = (job_s_lo[j], job_s_hi[j]);
            if s_lo > s_hi || s_hi > n_samples {
                return Err(PyValueError::new_err(format!(
                    "RecordStreamEngine: job sample range [{s_lo}, {s_hi}) is invalid \
                     for n_samples={n_samples}",
                )));
            }
            let regions: Vec<(u32, u32)> = starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect();
            jobs.push(RecordJob {
                contig_idx: job_contig_idx[j],
                regions,
                s_lo,
                s_hi,
            });
        }

        match source_kind {
            "vcf" => {
                let sample_refs: Vec<&str> = sample_names.iter().map(String::as_str).collect();
                let filler =
                    VcfWindowFiller::new(&vcf_path, &sample_refs, ploidy, fasta_path.as_deref())
                        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(Self::new_rs(
                    Box::new(filler),
                    contigs,
                    jobs,
                    n_samples,
                    ploidy,
                    pad_char,
                    parallel,
                    batch_size,
                    output_length,
                    annotated,
                ))
            }
            "pgen" => {
                // PGEN is diploid-only by format; `PgenWindowFiller` hardwires
                // `PGEN_PLOIDY = 2` internally (it takes no ploidy parameter at all) and
                // `RecordBackend::generate` above indexes the window's per-hap CSR as
                // `si * self.ploidy + p` -- a caller-supplied `ploidy != 2` would silently
                // desync that indexing from the filler's actual 2-hap-per-sample layout,
                // so this is validated here rather than passed through unchecked.
                if ploidy != 2 {
                    return Err(PyValueError::new_err(format!(
                        "RecordStreamEngine: source_kind=\"pgen\" requires ploidy=2 \
                         (PGEN is diploid-only), got ploidy={ploidy}",
                    )));
                }
                // `vcf_path` carries the `.pgen` path for this source kind (see the
                // `#[new]` doc comment); `sample_names` is the caller's sorted-name
                // `sample_idx` order, which `PgenWindowFiller::new` maps onto the physical
                // `.psam` column order (it reads the `.psam` for the full cohort size and
                // the public->physical map -- see `pgen.rs`'s "Sample subsetting" section).
                let sample_refs: Vec<&str> = sample_names.iter().map(String::as_str).collect();
                let filler = PgenWindowFiller::new(&vcf_path, &sample_refs)
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
                Ok(Self::new_rs(
                    Box::new(filler),
                    contigs,
                    jobs,
                    n_samples,
                    ploidy,
                    pad_char,
                    parallel,
                    batch_size,
                    output_length,
                    annotated,
                ))
            }
            other => Err(PyValueError::new_err(format!(
                "RecordStreamEngine: unknown source_kind {other:?} (expected \"vcf\" or \"pgen\")",
            ))),
        }
    }

    /// Return the next batch's `(data, offsets)`, or `None` when the plan is exhausted.
    /// Identical shape to `Svar1StreamEngine::next_batch` (`crate::ffi::stream_engine`):
    /// the GIL is released for the whole blocking body, reacquired only to marshal the
    /// owned arrays into numpy.
    fn next_batch<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyArray1<u8>>, Bound<'py, PyArray1<i64>>)>> {
        let out = py.detach(|| self.next_batch_core());
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
    /// otherwise the two annotation arrays are absent and this raises `RuntimeError`
    /// (a caller bug, not a data condition; `StreamingDataset._iter_batches` never
    /// calls this unless `annotated` was passed to `build_engine`).
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
        let out = py.detach(|| self.next_batch_core());
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
                    "RecordStreamEngine.next_batch_annotated() called on an engine \
                     constructed with annotated=false",
                ))
            }
            Some(Err(e)) => Err(PyRuntimeError::new_err(e.to_string())),
        }
    }

    /// **Test-only accessor** (issue #276 task 7 — the streaming-vs-write-path parity
    /// gate). Runs exactly ONE `WindowFiller::fill` for the window described by
    /// `(contig_idx, region_starts/region_ends, s_lo, s_hi)` into a scratch
    /// `DecodedWindow` and returns its local static table `(v_starts, ilens,
    /// alt_alleles, alt_offsets)` — NO genotype CSR generation/haplotype reconstruction,
    /// unlike `next_batch`. This bypasses the producer/consumer plan entirely (the job
    /// need not be part of the engine's constructed `jobs` list at all), so it can be
    /// called ad hoc against any window for a quick table-only decode.
    ///
    /// Exists to pin the #1 streaming-vs-write-path risk at the cheapest layer, BEFORE
    /// reconstruction: genoray's Rust `ChunkAssembler` (this engine's decoder) and
    /// gvl.write's Python cyvcf2 + `dense2sparse` decoder are independent
    /// implementations, and a divergence between them (e.g. in atomization/left-align/
    /// check-ref handling) is far cheaper to catch here — a plain array comparison of
    /// the variant table — than after it has propagated into an opaque haplotype
    /// byte-diff. See `tests/dataset/test_streaming_vcf_parity.py`.
    ///
    /// Ships in the production build (not `#[cfg(test)]`) because it must be callable
    /// from Python test code, but it is documented as test-only: no production code path
    /// calls it — they only ever need `next_batch`'s CSR-woven output.
    #[pyo3(signature = (contig_idx, region_starts, region_ends, s_lo, s_hi))]
    #[allow(clippy::too_many_arguments)]
    fn debug_decode_window(
        &self,
        contig_idx: usize,
        region_starts: Vec<u32>,
        region_ends: Vec<u32>,
        s_lo: usize,
        s_hi: usize,
    ) -> PyResult<(Vec<i32>, Vec<i32>, Vec<u8>, Vec<i64>)> {
        if region_starts.len() != region_ends.len() {
            return Err(PyValueError::new_err(
                "debug_decode_window: region_starts and region_ends must have the same length",
            ));
        }
        let regions: Vec<(u32, u32)> = region_starts.into_iter().zip(region_ends).collect();
        let job = RecordJob {
            contig_idx,
            regions,
            s_lo,
            s_hi,
        };
        let backend = Arc::clone(self.core.backend());
        let slot = backend
            .debug_fill(&job)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            slot.v_starts,
            slot.ilens,
            slot.alt_alleles,
            slot.alt_offsets,
        ))
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

    /// Filler with DISTINGUISHABLE per-sample runs, for the multi-region regression test.
    /// 2 SNPs (v0 @pos5, v1 @pos15), 2 samples x ploidy 2 = 4 haps. Per-hap window-local
    /// CSR (`geno_offsets=[0,1,1,2,4]`, `geno_v_idxs=[0,1,0,1]`):
    ///   hap0(s0,p0)=[v0]  hap1(s0,p1)=[]  hap2(s1,p0)=[v1]  hap3(s1,p1)=[v0,v1]
    /// The two samples carry different variants and the two SNPs sit in different regions,
    /// so a wrong-sample read OR a missing region-clip both change the bytes.
    struct MultiRegionFiller;
    impl WindowFiller for MultiRegionFiller {
        fn fill(
            &self,
            _job: &RecordJob,
            _c: &ContigRef,
            slot: &mut DecodedWindow,
        ) -> anyhow::Result<()> {
            slot.v_starts = vec![5, 15];
            slot.ilens = vec![0, 0];
            slot.alt_alleles = vec![b'A', b'C'];
            slot.alt_offsets = vec![0, 1, 2];
            slot.geno_v_idxs = vec![0, 1, 0, 1];
            slot.geno_offsets = vec![0, 1, 1, 2, 4];
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

        let (data, _annot_v, _annot_pos, offs) = crate::ffi::generate_batch_core(
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
            -1, // ragged
            None,
            false, // annotated
            None,  // global_v_idxs
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
            -1,   // ragged
            false, // annotated
        );

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, _av, _ap, o) = r.expect("no producer error");
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
            -1,    // ragged
            false, // annotated
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
            -1,    // ragged
            false, // annotated
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

    /// REGRESSION (critical): a job with `regions.len() >= 2` must expand the per-sample
    /// window CSR across regions. The pre-fix `generate` flat-sliced
    /// `geno_offsets[row_lo*ploidy..row_hi*ploidy]`, treating the per-hap CSR as if it were
    /// region-expanded like SVAR1's `read_window` output; for the 2nd+ region's rows the
    /// index ran off the per-sample CSR (length `n_samples*ploidy+1`) — an out-of-bounds
    /// panic (or, where it fit, the WRONG sample's run). Only single-region jobs (where
    /// `bi == si`) were accidentally correct, and both other tests here use those.
    ///
    /// Oracle is INDEPENDENT of `RecordBackend::generate`: it hand-expands the per-sample
    /// CSR across regions (the expected `o_starts_b`/`o_stops_b` are written out as literals
    /// below, NOT recomputed by the code under test) and calls `generate_batch_core`
    /// directly. Row bi -> (region ri = bi/n_samples, sample si = bi%n_samples), C-order.
    #[test]
    fn record_stream_engine_expands_csr_per_region_and_sample() {
        // 2 regions x 2 samples = 4 rows; regions chosen so the SNPs clip differently:
        // v0@5 is in [0,10) only; v1@15 is in [10,20) only.
        let regions = vec![(0u32, 10u32), (10u32, 20u32)];
        let job = RecordJob {
            contig_idx: 0,
            regions: regions.clone(),
            s_lo: 0,
            s_hi: 2,
        };
        let c = chr1(); // 30 bp of 'T'
        let ploidy = 2usize;
        let n_samples = 2usize;
        let n_rows = regions.len() * n_samples; // 4

        // ---- Independent oracle -------------------------------------------------------
        // Fill a scratch window with the SAME filler (its table is region-agnostic).
        let mut slot = DecodedWindow::default();
        MultiRegionFiller.fill(&job, &c, &mut slot).unwrap();

        // Per-row region bounds (C-order region, sample): bi -> regions[bi/n_samples].
        let mut rb = Array2::<i32>::zeros((n_rows, 2));
        for bi in 0..n_rows {
            let ri = bi / n_samples;
            rb[[bi, 0]] = regions[ri].0 as i32;
            rb[[bi, 1]] = regions[ri].1 as i32;
        }

        // HAND-EXPANDED per (region, sample) CSR, as literals — the expected mapping the
        // fix must produce, computed by hand from geno_offsets=[0,1,1,2,4]:
        //   bi0 (r0,s0): hap0->[0,1) hap1->[1,1)   bi1 (r0,s1): hap2->[1,2) hap3->[2,4)
        //   bi2 (r1,s0): hap0->[0,1) hap1->[1,1)   bi3 (r1,s1): hap2->[1,2) hap3->[2,4)
        let o_starts_b: Vec<i64> = vec![
            0, 1, /*bi0*/ 1, 2, /*bi1*/ 0, 1, /*bi2*/ 1, 2, /*bi3*/
        ];
        let o_stops_b: Vec<i64> = vec![
            1, 1, /*bi0*/ 2, 4, /*bi1*/ 1, 1, /*bi2*/ 2, 4, /*bi3*/
        ];
        let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);
        let (exp_data, _exp_annot_v, _exp_annot_pos, exp_offs) = crate::ffi::generate_batch_core(
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
            -1, // ragged
            None,
            false, // annotated
            None,  // global_v_idxs
            false,
        );
        let expected = (exp_data.to_vec(), exp_offs.to_vec());

        // Sanity: the oracle actually exercises region-dependent clipping — region [0,10)'s
        // sample-0 hap0 applies v0 ('A'), region [10,20)'s does not (pure ref 'T'), so the
        // window contains a variant byte and is not all-reference.
        assert!(
            expected.0.contains(&b'A') && expected.0.contains(&b'C'),
            "oracle must apply both SNPs somewhere (else it proves nothing)"
        );

        // ---- Engine under test --------------------------------------------------------
        let engine = RecordStreamEngine::new_rs(
            Box::new(MultiRegionFiller),
            vec![chr1()],
            vec![job],
            n_samples,
            ploidy,
            b'N',
            false,
            1000, // one batch for the whole 4-row window
            -1,    // ragged
            false, // annotated
        );

        let mut batches: Vec<(Vec<u8>, Vec<i64>)> = Vec::new();
        while let Some(r) = engine.next_batch_core() {
            let (d, _av, _ap, o) = r.expect("no producer error / no OOB panic");
            batches.push((d.to_vec(), o.to_vec()));
        }
        assert_eq!(
            batches.len(),
            1,
            "one batch for the single multi-region window"
        );
        assert_eq!(
            batches[0], expected,
            "multi-region window must expand the per-sample CSR across regions \
             (region-replicated + kernel-clipped), byte-identical to the oracle"
        );

        assert!(engine.next_batch_core().is_none());
        assert!(engine.next_batch_core().is_none());
    }
}
