//! `Svar1StreamEngine` — the SVAR1 producer/consumer overlap engine (issue #283).
//!
//! **This is gvl's first production threading.** A background *producer* thread reads
//! window N+1's CSR offsets (`Svar1Store::read_window`) and reads-through the runs it
//! will hand off (`geno_v_idxs()[o_lo..o_hi]`), warming the shared OS page cache, while
//! the *consumer* (`next_batch`, called from Python) generates batches from window N.
//! What is overlapped is I/O latency, not decode — SVAR1's on-disk layout is already the
//! target representation, so the producer's job is to fault the exact pages ahead of the
//! consumer (the OS page cache does not prefetch on an application's access pattern; a
//! thread walking the fixed, `__getitem__`-free traversal does).
//!
//! **Why bespoke and not `crate::stream::run_windows`.** `run_windows` uses
//! `std::thread::scope` + `spawn_scoped`, whose scope must complete within one function
//! call. The engine's producer must instead outlive many separate `next_batch()` FFI
//! calls, so it is a *detached* `std::thread::spawn` owning an `Arc<Svar1Store>`, the
//! jobs, and the channel `Sender`s. The shutdown/panic discipline is copied EXACTLY from
//! `run_windows` (read its doc comments — they explain WHY each ownership move matters):
//!
//!   * two `crossbeam_channel::bounded(2)` channels — `filled` (producer→consumer) and
//!     `free` (consumer→producer, slot recycling) — with `free` prefilled with 2 default
//!     slots. Only 2 buffers exist in total, ping-ponging between the channels, so no
//!     `send` can ever block and memory is capped at `2 * window_offsets` regardless of
//!     plan length;
//!   * **shutdown by dropping the producer's `Sender<FilledWindow>`** (the filled tx) when
//!     the job loop ends — the consumer's `recv()` then observes channel close. The
//!     consumer holds NO clone of the filled tx, so there is nothing to forget to drop;
//!   * **join-then-classify**: on channel close the consumer JOINS the producer and only
//!     then classifies its `anyhow::Result` (`Err(_)` join ⇒ producer panicked; `Ok(Err)`
//!     ⇒ propagate; `Ok(Ok)` ⇒ clean end). The consumer NEVER early-returns with the
//!     producer live and unjoined.
//!
//! **The engine opens its OWN `Arc<Svar1Store>`** from the store path (it does not
//! Arc-share the Python-owned `Svar1Store` pyclass instance). Because it mmaps the same
//! file, the producer's reads and the consumer's reads hit the same OS page-cache pages —
//! no data duplication. `Svar1Store` is `Send + Sync`, so the `Arc` crosses to the
//! producer thread and `read_window`/`geno_v_idxs` run GIL-free.
//!
//! **Memory:** the engine owns only variant-/contig-scale arrays plus, per contig, its
//! `v_starts_c`/`v_ends_c` and reference slice. The only sample-scale residency is the
//! recycled per-window offsets in the ≤2 live `FilledWindow`s — the intended, budgeted
//! (max_mem-bounded, `window_samples`-scale) allocation, never the full cohort.
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

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, Receiver, Sender};
use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

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
/// `contig_idx`) crossed with `phys_samples` (physical/native genotype-column indices).
/// `phys_samples` is `window_samples`-scale (max_mem-bounded), NOT the full cohort.
struct WindowJob {
    contig_idx: usize,
    regions: Vec<(u32, u32)>,
    phys_samples: Vec<usize>,
}

/// One recycled slot: the whole window's CSR offsets plus the `job_idx` that produced it
/// (so the consumer can recover the window's regions/phys_samples/contig for generation).
#[derive(Default)]
struct FilledWindow {
    o_starts: Vec<i64>,
    o_stops: Vec<i64>,
    job_idx: usize,
}

/// The window the consumer is currently draining, batch by batch.
struct CurrentWindow {
    filled: FilledWindow,
    /// Next window row (region×sample, C-order) to generate from.
    next_row: usize,
    /// Total window rows = `regions.len() * phys_samples.len()`.
    n_batch_rows: usize,
}

/// Mutable engine state behind the pyclass's `Mutex` (the pyclass methods are `&self`).
struct EngineState {
    started: bool,
    done: bool,
    /// Recycle drained slots back to the producer. `None` until the producer is spawned.
    tx_free: Option<Sender<FilledWindow>>,
    /// Receive prefetched windows from the producer. `None` until spawned.
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

/// Producer/consumer SVAR1 streamer (issue #283). See the module docs for the design.
#[pyclass]
pub struct Svar1StreamEngine {
    /// The engine's OWN store (Arc-shared with the producer thread; same mmap file, same
    /// page cache — NOT the Python-owned pyclass instance).
    store: Arc<Svar1Store>,
    contigs: Arc<Vec<ContigData>>,
    jobs: Arc<Vec<WindowJob>>,
    /// GLOBAL variant-scale tables (indexed by global variant id from `geno_v_idxs`).
    v_starts: Array1<i32>,
    ilens: Array1<i32>,
    alt_alleles: Array1<u8>,
    alt_offsets: Array1<i64>,
    pad_char: u8,
    parallel: bool,
    batch_size: usize,
    state: Mutex<EngineState>,
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
        v_starts: Array1<i32>,
        ilens: Array1<i32>,
        alt_alleles: Array1<u8>,
        alt_offsets: Array1<i64>,
        pad_char: u8,
        parallel: bool,
        batch_size: usize,
    ) -> Self {
        for c in &contigs {
            store.set_contig_meta_rs(&c.name, c.contig_start, c.n_local, c.max_v_len);
        }
        Self {
            store: Arc::new(store),
            contigs: Arc::new(contigs),
            jobs: Arc::new(jobs),
            v_starts,
            ilens,
            alt_alleles,
            alt_offsets,
            pad_char,
            parallel,
            batch_size: batch_size.max(1),
            state: Mutex::new(EngineState::new()),
        }
    }

    /// Spawn the detached producer thread (once). Prefills `free` with 2 default slots,
    /// then the producer fills window after window, blocked only by the free-slot pool.
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
        // When the producer returns (all jobs done, a `read_window` error via `?`, or a
        // `recv`/`send` seeing the consumer gone), `tx_filled` drops and the consumer's
        // `rx_filled.recv()` observes close as soon as the channel drains. No clone of
        // `tx_filled` is held anywhere else, so shutdown-by-drop is by construction.
        let store = Arc::clone(&self.store);
        let jobs = Arc::clone(&self.jobs);
        let contigs = Arc::clone(&self.contigs);

        let handle = std::thread::Builder::new()
            .name("gvl-svar1-stream-producer".into())
            .spawn(move || -> anyhow::Result<()> {
                for (job_idx, job) in jobs.iter().enumerate() {
                    // Recycle a drained slot. Err => consumer is gone (engine dropped);
                    // stop quietly and let the consumer's own outcome stand.
                    let Ok(mut slot) = rx_free.recv() else {
                        return Ok(());
                    };

                    let c = &contigs[job.contig_idx];
                    let w = store.read_window(
                        &c.name,
                        &c.v_starts_c,
                        &c.v_ends_c,
                        &job.regions,
                        &job.phys_samples,
                    )?;
                    slot.o_starts = w.o_starts;
                    slot.o_stops = w.o_stops;
                    slot.job_idx = job_idx;

                    // Read-through prefetch: fault the EXACT pages the consumer will read
                    // into the shared page cache. `black_box` stops the compiler eliding
                    // the fold. `o_starts`/`o_stops` are absolute indices into the mmap.
                    let vidx = store.geno_v_idxs();
                    for (lo, hi) in slot.o_starts.iter().zip(slot.o_stops.iter()) {
                        let (lo, hi) = (*lo as usize, *hi as usize);
                        let _ = std::hint::black_box(
                            vidx[lo..hi].iter().fold(0i64, |a, &x| a ^ x as i64),
                        );
                    }

                    if tx_filled.send(slot).is_err() {
                        return Ok(()); // consumer gone
                    }
                }
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn svar1 streaming producer: {e}"))?;

        state.tx_free = Some(tx_free);
        state.rx_filled = Some(rx_filled);
        state.producer = Some(handle);
        Ok(())
    }

    /// The consumer, GIL-free. Returns:
    ///   * `Some(Ok((data, offsets)))` — the next batch;
    ///   * `Some(Err(_))` — a producer error/panic (join-then-classified);
    ///   * `None` — the plan is exhausted (clean, no hang).
    ///
    /// Drains the current window batch-by-batch; when a window is spent it is recycled to
    /// the producer and the next prefetched window is `recv`'d. On channel close the
    /// producer is joined and classified before returning.
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
            // 1. If the current window has rows left, generate the next batch.
            let has_rows = match state.current.as_ref() {
                Some(cur) => cur.next_row < cur.n_batch_rows,
                None => false,
            };
            if has_rows {
                return Some(self.generate_from_current(&mut state));
            }

            // 2. Current window is spent (or absent): recycle it, then fetch the next.
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
                    let job = &self.jobs[fw.job_idx];
                    let n_batch_rows = job.regions.len() * job.phys_samples.len();
                    state.current = Some(CurrentWindow {
                        filled: fw,
                        next_row: 0,
                        n_batch_rows,
                    });
                    // Loop back to generate from the newly received window.
                }
                Err(_) => {
                    // Channel closed => producer finished. JOIN FIRST, classify AFTER —
                    // never return with the producer live and unjoined.
                    state.done = true;
                    if let Some(h) = state.producer.take() {
                        return match h.join() {
                            Err(_) => Some(Err(anyhow::anyhow!(
                                "svar1 streaming producer thread panicked"
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

    /// Generate the next `batch_size`-bounded slice of the current window's rows. Output
    /// is `(hi-lo)`-bounded — never whole-window (issue #284). Delegates to the shared
    /// [`crate::ffi::generate_batch_core`].
    fn generate_from_current(
        &self,
        state: &mut EngineState,
    ) -> anyhow::Result<(Array1<u8>, Array1<i64>)> {
        let ploidy = self.store.ploidy();

        // Advance the cursor first (mutable borrow), then reborrow immutably to read.
        let (row_lo, row_hi, job_idx) = {
            let cur = state
                .current
                .as_mut()
                .expect("generate_from_current called with a live current window");
            let row_lo = cur.next_row;
            let row_hi = (row_lo + self.batch_size).min(cur.n_batch_rows);
            cur.next_row = row_hi;
            (row_lo, row_hi, cur.filled.job_idx)
        };
        let filled = &state.current.as_ref().unwrap().filled;
        let job = &self.jobs[job_idx];
        let c = &self.contigs[job.contig_idx];
        let n_samples = job.phys_samples.len();
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
        let o_starts_b = &filled.o_starts[o_lo..o_hi];
        let o_stops_b = &filled.o_stops[o_lo..o_hi];

        // This contig's reference slice (offsets [0, len]); see the module note.
        let ref_offsets = Array1::from(vec![0i64, c.ref_bytes.len() as i64]);
        let ref_view = ndarray::ArrayView1::from(c.ref_bytes.as_slice());

        Ok(crate::ffi::generate_batch_core(
            &self.store,
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
            self.parallel,
        ))
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
    /// `contig_ref_bytes[i]`. Per-job records are parallel arrays indexed by job:
    /// `job_contig_idx[j]`, `job_region_starts[j]`, `job_region_ends[j]`,
    /// `job_phys_samples[j]`.
    #[new]
    #[pyo3(signature = (
        store_path, n_samples, ploidy,
        contig_names, contig_starts, n_locals, max_v_lens, v_starts_c, v_ends_c,
        contig_ref_bytes,
        job_contig_idx, job_region_starts, job_region_ends, job_phys_samples,
        v_starts, ilens, alt_alleles, alt_offsets,
        pad_char, parallel, batch_size,
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
        job_contig_idx: Vec<usize>,
        job_region_starts: Vec<Vec<u32>>,
        job_region_ends: Vec<Vec<u32>>,
        job_phys_samples: Vec<Vec<usize>>,
        v_starts: PyReadonlyArray1<i32>,
        ilens: PyReadonlyArray1<i32>,
        alt_alleles: PyReadonlyArray1<u8>,
        alt_offsets: PyReadonlyArray1<i64>,
        pad_char: u8,
        parallel: bool,
        batch_size: usize,
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

        let n_jobs = job_contig_idx.len();
        if [
            job_region_starts.len(),
            job_region_ends.len(),
            job_phys_samples.len(),
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
            let regions: Vec<(u32, u32)> = starts.iter().zip(ends).map(|(&s, &e)| (s, e)).collect();
            jobs.push(WindowJob {
                contig_idx: job_contig_idx[j],
                regions,
                phys_samples: job_phys_samples[j].clone(),
            });
        }

        Ok(Self::build(
            store,
            contigs,
            jobs,
            v_starts.as_array().to_owned(),
            ilens.as_array().to_owned(),
            alt_alleles.as_array().to_owned(),
            alt_offsets.as_array().to_owned(),
            pad_char,
            parallel,
            batch_size,
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
        let (data, offs) = crate::ffi::generate_batch_core(
            &store,
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
            false,
        );
        (data.to_vec(), offs.to_vec())
    }

    fn build_engine(f: &Fixture, jobs: Vec<WindowJob>, batch_size: usize) -> Svar1StreamEngine {
        let store = Svar1Store::open_meta(&f.path, 2, 2).unwrap();
        Svar1StreamEngine::build(
            store,
            vec![chr1_contig(f)],
            jobs,
            f.v_starts.clone(),
            f.ilens.clone(),
            f.alt_alleles.clone(),
            f.alt_offsets.clone(),
            b'N',
            false,
            batch_size,
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
                phys_samples: vec![0, 1],
            },
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 15)],
                phys_samples: vec![0, 1],
            },
        ];
        // batch_size larger than any window -> one batch per window.
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
                phys_samples: vec![0, 1],
            },
            WindowJob {
                contig_idx: 0,
                regions: vec![(0, 30)],
                phys_samples: vec![0, 1],
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
}
