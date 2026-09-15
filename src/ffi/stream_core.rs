//! Generic detached-thread producer/consumer engine core (issue #283 / #276).
//!
//! This is the reusable machinery originally written inline in `Svar1StreamEngine`
//! (`stream_engine.rs`) — extracted so a second backend (the record-stream VCF/PGEN
//! engine, issue #276 task 3b) can reuse the SAME threading/shutdown/panic discipline
//! without duplicating it. Only the two per-engine divergence points — how a job gets
//! decoded into a slot (the producer's work) and how batch rows get generated from a
//! filled slot (the consumer's work) — are abstracted behind [`EngineBackend`].
//!
//! **Why bespoke and not `crate::stream::run_windows`.** `run_windows` uses
//! `std::thread::scope` + `spawn_scoped`, whose scope must complete within one function
//! call. This engine's producer must instead outlive many separate `next_batch()` FFI
//! calls (each one is a distinct call from Python's iterator protocol), so it is a
//! *detached* `std::thread::spawn` owning an `Arc<B>` backend and the channel `Sender`s.
//! The shutdown/panic discipline is copied EXACTLY from `run_windows` (read its doc
//! comments — they explain WHY each ownership move matters):
//!
//!   * two `crossbeam_channel::bounded(2)` channels — `filled` (producer→consumer) and
//!     `free` (consumer→producer, slot recycling) — with `free` prefilled with 2 default
//!     slots. Only 2 buffers exist in total, ping-ponging between the channels, so no
//!     `send` can ever block and memory is capped at `2 * one_window` regardless of plan
//!     length;
//!   * **shutdown by dropping the producer's `Sender<B::Slot>`** (the filled tx) when the
//!     job loop ends — the consumer's `recv()` then observes channel close. The consumer
//!     holds NO clone of the filled tx, so there is nothing to forget to drop;
//!   * **join-then-classify**: on channel close the consumer JOINS the producer and only
//!     then classifies its `anyhow::Result` (`Err(_)` join ⇒ producer panicked; `Ok(Err)`
//!     ⇒ propagate; `Ok(Ok)` ⇒ clean end). The consumer NEVER early-returns with the
//!     producer live and unjoined.
//!
//! **Job ordering.** The plan is processed strictly in index order by a SINGLE producer
//! feeding one FIFO channel, so a filled window's `job_idx` is recovered from a counter
//! (`EngineState::next_job_idx`) rather than the slot needing to carry it itself — one
//! less thing for a backend's `Slot` to duplicate.
//!
//! **`Send + Sync` on `EngineBackend`.** The core holds `Arc<B>` and clones it into the
//! detached producer thread (which calls `fill`) while the consumer thread calls
//! `generate` — both need `&B` concurrently from different threads, hence `Sync`; the
//! `Arc` itself crossing to the producer thread requires `Send`.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crossbeam_channel::bounded;
use ndarray::Array1;
use pyo3::Python;

use crate::variants::{VariantWindowsBatch, VariantsBatch};

/// The two per-backend divergence points of the producer/consumer engine, plus the
/// plan shape. A backend bundles whatever state it needs (store handles, global
/// tables, the job list) — the core only ever sees it through this trait.
pub(crate) trait EngineBackend: Send + Sync + 'static {
    /// Reusable per-window buffer. Recycled between the `filled` and `free` channels so
    /// only 2 ever exist at once regardless of plan length.
    type Slot: Default + Send + 'static;

    /// Number of jobs (windows) in the plan.
    fn n_jobs(&self) -> usize;

    /// PRODUCER thread: decode/fill job `job_idx` into `slot` (reusing its allocations).
    fn fill(&self, job_idx: usize, slot: &mut Self::Slot) -> anyhow::Result<()>;

    /// Rows this filled window will emit (the batch-iteration bound).
    fn n_batch_rows(&self, job_idx: usize, slot: &Self::Slot) -> usize;

    /// CONSUMER thread: generate batch rows `[row_lo, row_hi)` from the filled slot.
    /// The two `Option<Array1<i32>>` are the annotation outputs (issue #277 Wave A
    /// Task 4, `annot_v_idxs`/`annot_ref_pos`) — `Some` iff the backend was
    /// constructed with `annotated: true`, `None` otherwise (pre-Task-4 behavior,
    /// zero extra allocation).
    fn generate(
        &self,
        job_idx: usize,
        slot: &Self::Slot,
        row_lo: usize,
        row_hi: usize,
    ) -> anyhow::Result<(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)>;

    /// Variants-output counterpart of `generate` (Wave B PR-B1). Default: unsupported.
    /// RecordBackend overrides it; Svar1Backend overrides it in Task 4. The returned
    /// `VariantsBatch.info_out` (Wave B PR-B3a) carries ride-along per-variant INFO
    /// columns gathered by the same kept `v_idxs` as `start`/`ilen` — `RecordBackend`
    /// populates it from `DecodedWindow.info_cols`; `Svar1Backend` leaves it empty
    /// until a later task.
    fn generate_variants(
        &self,
        _job_idx: usize,
        _slot: &Self::Slot,
        _row_lo: usize,
        _row_hi: usize,
    ) -> anyhow::Result<VariantsBatch> {
        anyhow::bail!("variants output is not supported by this backend")
    }

    /// `variant-windows`-output counterpart of `generate_variants` (Wave B PR-B4, #304):
    /// same `[row_lo, row_hi)` slice of the same filled window, but returns tokenized
    /// ref/alt window (or bare-allele) buffers via `crate::variants::windows::assemble_windows_mode`
    /// instead of (or alongside) `generate_variants`'s ride-along scalar fields. Default:
    /// unsupported, exactly like `generate_variants`'s default — `RecordBackend` and
    /// `Svar1Backend` both override it.
    fn generate_variant_windows(
        &self,
        _job_idx: usize,
        _slot: &Self::Slot,
        _row_lo: usize,
        _row_hi: usize,
    ) -> anyhow::Result<VariantWindowsBatch> {
        anyhow::bail!("variant-windows output is not supported by this backend")
    }
}

/// The window the consumer is currently draining, batch by batch.
struct CurrentWindow<Slot> {
    filled: Slot,
    /// The job that produced `filled` — recovered from `EngineState::next_job_idx` at
    /// receive time (see the module job-ordering note), not carried by `Slot` itself.
    job_idx: usize,
    /// Next window row (region×sample, C-order) to generate from.
    next_row: usize,
    /// Total window rows for this job.
    n_batch_rows: usize,
}

/// Mutable engine state behind the core's `Mutex` (the pyclass methods are `&self`).
struct EngineState<Slot> {
    started: bool,
    done: bool,
    /// Recycle drained slots back to the producer. `None` until the producer is spawned.
    tx_free: Option<crossbeam_channel::Sender<Slot>>,
    /// Receive prefetched windows from the producer. `None` until spawned.
    rx_filled: Option<crossbeam_channel::Receiver<Slot>>,
    /// Producer handle — joined (and classified) exactly once, on channel close.
    producer: Option<JoinHandle<anyhow::Result<()>>>,
    current: Option<CurrentWindow<Slot>>,
    /// `job_idx` of the NEXT window expected off `rx_filled` (see the module job-ordering
    /// note). Incremented once per successful `recv`.
    next_job_idx: usize,
}

impl<Slot> EngineState<Slot> {
    fn new() -> Self {
        Self {
            started: false,
            done: false,
            tx_free: None,
            rx_filled: None,
            producer: None,
            current: None,
            next_job_idx: 0,
        }
    }
}

impl<Slot> Drop for EngineState<Slot> {
    /// Deterministic teardown: if the engine is dropped mid-stream (producer still live),
    /// join the producer instead of leaking a detached thread. Dropping the free `Sender`
    /// and the filled `Receiver` FIRST closes both channels, so a producer blocked on
    /// `rx_free.recv()` (no free slot) or `tx_filled.send()` (consumer gone) unblocks and
    /// returns `Ok(())` — the join then completes promptly (bounded by at most one
    /// in-flight `fill`). Cannot double-join: the normal exhaustion / error / panic paths
    /// already `take()` the handle in `next_batch_core`, leaving `producer == None` here.
    ///
    /// **The join MUST NOT hold the GIL** (issue #399). Closing the channels unblocks a
    /// producer parked in `rx_free.recv()`/`tx_filled.send()`, but it does nothing for one
    /// parked *inside* `fill` waiting on the GIL — and `PgenWindowFiller::fill` drives
    /// pgenlib through `Python::attach`, so that is a real state. This `drop` normally runs
    /// from Python deallocation, i.e. with the GIL held, so joining directly deadlocks the
    /// pair: we wait on the producer, the producer waits on the GIL we are holding. The
    /// bound in the paragraph above ("at most one in-flight `fill`") only holds once the
    /// GIL is released across the join, which is what `detach` does here. This is the same
    /// discipline `next_batch`/`window_realign_inputs` already follow via `py.detach`.
    fn drop(&mut self) {
        self.tx_free = None;
        self.rx_filled = None;
        if let Some(h) = self.producer.take() {
            // `Python::attach` is re-entrant when this thread already holds the GIL and
            // acquires it otherwise; `detach` is what actually releases it across the
            // join. Skip it entirely once the interpreter is gone (a drop racing
            // `Py_Finalize`, or a pure-Rust unit test with no interpreter): there is no
            // GIL to release, and attaching then would be undefined.
            if unsafe { pyo3::ffi::Py_IsInitialized() } != 0 {
                Python::attach(|py| {
                    py.detach(|| {
                        let _ = h.join();
                    })
                });
            } else {
                let _ = h.join();
            }
        }
    }
}

/// Output-agnostic result of one iteration step (Wave B PR-B1): either a generatable row
/// slice of the current window, clean exhaustion, or a producer error/panic. Shared by
/// `StreamEngineCore::next_batch_core` and `next_batch_variants_core` so the window
/// recycle/fetch/join-then-classify plumbing in `advance` is written exactly once — only
/// what happens to a `Ready` slice (`generate` vs `generate_variants`) differs per caller.
enum NextSlice {
    Ready {
        job_idx: usize,
        row_lo: usize,
        row_hi: usize,
    },
    Done,
    Failed(anyhow::Error),
}

/// Generic producer/consumer engine core over an [`EngineBackend`]. A concrete engine
/// pyclass (e.g. `Svar1StreamEngine`) holds one of these as a plain field and delegates
/// `next_batch`/`next_batch_core` to it.
pub(crate) struct StreamEngineCore<B: EngineBackend> {
    backend: Arc<B>,
    batch_size: usize,
    state: Mutex<EngineState<B::Slot>>,
}

impl<B: EngineBackend> StreamEngineCore<B> {
    pub(crate) fn new(backend: Arc<B>, batch_size: usize) -> Self {
        Self {
            backend,
            batch_size: batch_size.max(1),
            state: Mutex::new(EngineState::new()),
        }
    }

    /// Escape hatch: expose the shared backend handle directly, bypassing the
    /// producer/consumer channel machinery entirely, so a caller can run a single
    /// `WindowFiller::fill` against a scratch slot on its own thread while the producer
    /// thread (if started) is concurrently doing the same against the same `Arc`-shared
    /// backend. Two callers:
    ///
    /// - `RecordStreamEngine::debug_decode_window` (issue #276 task 7) — test/debug-only,
    ///   parity testing.
    /// - `RecordStreamEngine::window_realign_inputs` (issue #375 Track B) — a genuine
    ///   production path for mixed VCF/PGEN variants+tracks streams, called per window.
    ///   As shipped, Python's `_mixed_engine()` calls it against a separate,
    ///   plan-less engine (`jobs=[]`) built just for this, so in production
    ///   there is no producer thread on that engine's `Arc`-shared backend to
    ///   contend with -- the concurrent-with-a-live-producer case this doc
    ///   describes is currently exercised only by
    ///   `test_window_realign_inputs_matches_before_and_during_producer`,
    ///   which deliberately drives one engine both ways. It stays a genuine
    ///   safety property of this accessor regardless, since folding the mixed
    ///   path onto the drive engine (removing the double decode) is the
    ///   obvious next step and would make this the live case (final review,
    ///   M1).
    ///
    /// Both release the GIL (`py.detach`) around their call in here, so both are safe to
    /// call against a live producer — safety does not rest on a call-site convention like
    /// "only call this before iteration starts".
    ///
    /// The synchronous-decode-on-the-caller's-thread pattern this enables is legitimate
    /// production use ONLY under two conditions, both the caller's responsibility: the
    /// caller must release the GIL before calling in (`py.detach`), since the producer
    /// needs the GIL and a caller holding it while blocked on a filler-internal lock would
    /// deadlock the two; and any filler with shared mutable state (e.g.
    /// `PgenWindowFiller`, whose `fill` mutates a single shared pgenlib reader) must
    /// serialize its own `fill` internally (see `PgenWindowFiller::reader_lock`) rather
    /// than relying on this accessor for exclusion — it provides none.
    pub(crate) fn backend(&self) -> &Arc<B> {
        &self.backend
    }

    /// Spawn the detached producer thread (once). Prefills `free` with 2 default slots,
    /// then the producer fills window after window, blocked only by the free-slot pool.
    fn ensure_started(&self, state: &mut EngineState<B::Slot>) -> anyhow::Result<()> {
        if state.started {
            return Ok(());
        }
        state.started = true;

        // 2 slots (ping-pong). Only 2 buffers ever exist, recycled between the two
        // channels, so no `send` can block; memory is capped regardless of plan length.
        let n_slots = 2usize;
        let (tx_filled, rx_filled) = bounded::<B::Slot>(n_slots);
        let (tx_free, rx_free) = bounded::<B::Slot>(n_slots);
        for _ in 0..n_slots {
            tx_free
                .send(B::Slot::default())
                .expect("prefill free slots (receiver just created, cannot be closed)");
        }

        // `tx_filled` and `rx_free` are MOVED into the producer — it is their sole owner.
        // When the producer returns (all jobs done, a `fill` error via `?`, or a
        // `recv`/`send` seeing the consumer gone), `tx_filled` drops and the consumer's
        // `rx_filled.recv()` observes close as soon as the channel drains. No clone of
        // `tx_filled` is held anywhere else, so shutdown-by-drop is by construction.
        let backend = Arc::clone(&self.backend);
        let n_jobs = backend.n_jobs();

        let handle = std::thread::Builder::new()
            .name("gvl-stream-producer".into())
            .spawn(move || -> anyhow::Result<()> {
                for job_idx in 0..n_jobs {
                    // Recycle a drained slot. Err => consumer is gone (engine dropped);
                    // stop quietly and let the consumer's own outcome stand.
                    let Ok(mut slot) = rx_free.recv() else {
                        return Ok(());
                    };

                    backend.fill(job_idx, &mut slot)?;

                    if tx_filled.send(slot).is_err() {
                        return Ok(()); // consumer gone
                    }
                }
                Ok(())
            })
            .map_err(|e| anyhow::anyhow!("failed to spawn streaming producer: {e}"))?;

        state.tx_free = Some(tx_free);
        state.rx_filled = Some(rx_filled);
        state.producer = Some(handle);
        Ok(())
    }

    /// Advance the shared iteration state to the next generatable row slice, or to
    /// exhaustion/error. Byte-identical control flow to the pre-refactor `next_batch_core`
    /// loop (issue #276/#283): the only change is that the terminal action — generating
    /// bytes from `[row_lo, row_hi)` — is left to the caller instead of being inlined here,
    /// so both the haplotype and variants-output paths can reuse this same cursor walk.
    fn advance(&self, state: &mut EngineState<B::Slot>) -> NextSlice {
        if state.done {
            return NextSlice::Done;
        }
        if let Err(e) = self.ensure_started(state) {
            state.done = true;
            return NextSlice::Failed(e);
        }

        loop {
            // 1. If the current window has rows left, advance the cursor and yield the
            //    next slice.
            let has_rows = match state.current.as_ref() {
                Some(cur) => cur.next_row < cur.n_batch_rows,
                None => false,
            };
            if has_rows {
                let cur = state
                    .current
                    .as_mut()
                    .expect("has_rows implies a live current window");
                let row_lo = cur.next_row;
                let row_hi = (row_lo + self.batch_size).min(cur.n_batch_rows);
                cur.next_row = row_hi;
                return NextSlice::Ready {
                    job_idx: cur.job_idx,
                    row_lo,
                    row_hi,
                };
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
                Ok(slot) => {
                    let job_idx = state.next_job_idx;
                    state.next_job_idx += 1;
                    let n_batch_rows = self.backend.n_batch_rows(job_idx, &slot);
                    state.current = Some(CurrentWindow {
                        filled: slot,
                        job_idx,
                        next_row: 0,
                        n_batch_rows,
                    });
                    // Loop back to yield from the newly received window.
                }
                Err(_) => {
                    // Channel closed => producer finished. JOIN FIRST, classify AFTER —
                    // never return with the producer live and unjoined.
                    state.done = true;
                    if let Some(h) = state.producer.take() {
                        return match h.join() {
                            Err(_) => NextSlice::Failed(anyhow::anyhow!(
                                "streaming producer thread panicked"
                            )),
                            Ok(Err(e)) => NextSlice::Failed(e),
                            Ok(Ok(())) => NextSlice::Done,
                        };
                    }
                    return NextSlice::Done;
                }
            }
        }
    }

    /// The consumer, GIL-free. Returns:
    ///   * `Some(Ok((data, offsets)))` — the next batch;
    ///   * `Some(Err(_))` — a producer error/panic (join-then-classified);
    ///   * `None` — the plan is exhausted (clean, no hang).
    ///
    /// Drains the current window batch-by-batch; when a window is spent it is recycled to
    /// the producer and the next prefetched window is `recv`'d. On channel close the
    /// producer is joined and classified before returning.
    pub(crate) fn next_batch_core(
        &self,
    ) -> Option<anyhow::Result<(Array1<u8>, Option<Array1<i32>>, Option<Array1<i32>>, Array1<i64>)>>
    {
        // Recover from a poisoned lock rather than propagating panic-on-panic.
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match self.advance(&mut state) {
            NextSlice::Ready {
                job_idx,
                row_lo,
                row_hi,
            } => {
                let filled = &state.current.as_ref().unwrap().filled;
                Some(self.backend.generate(job_idx, filled, row_lo, row_hi))
            }
            NextSlice::Done => None,
            NextSlice::Failed(e) => Some(Err(e)),
        }
    }

    /// Variants-output counterpart of `next_batch_core` (Wave B PR-B1): same iteration
    /// state, same `advance` cursor walk, but yields flat `VariantsBatch` buffers via
    /// `EngineBackend::generate_variants` instead of reconstructed haplotype bytes. Does
    /// NOT fit `next_batch_core`'s 4-tuple shape, hence a sibling method rather than an
    /// overload.
    pub(crate) fn next_batch_variants_core(&self) -> Option<anyhow::Result<VariantsBatch>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match self.advance(&mut state) {
            NextSlice::Ready {
                job_idx,
                row_lo,
                row_hi,
            } => {
                let filled = &state.current.as_ref().unwrap().filled;
                Some(
                    self.backend
                        .generate_variants(job_idx, filled, row_lo, row_hi),
                )
            }
            NextSlice::Done => None,
            NextSlice::Failed(e) => Some(Err(e)),
        }
    }

    /// `variant-windows`-output counterpart of `next_batch_variants_core` (Wave B
    /// PR-B4, #304): same iteration state, same `advance` cursor walk, but yields
    /// `VariantWindowsBatch` via `EngineBackend::generate_variant_windows` instead of
    /// `VariantsBatch`. Sibling method, not an overload, for the same reason
    /// `next_batch_variants_core` is a sibling of `next_batch_core`.
    pub(crate) fn next_batch_variant_windows_core(
        &self,
    ) -> Option<anyhow::Result<VariantWindowsBatch>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match self.advance(&mut state) {
            NextSlice::Ready {
                job_idx,
                row_lo,
                row_hi,
            } => {
                let filled = &state.current.as_ref().unwrap().filled;
                Some(
                    self.backend
                        .generate_variant_windows(job_idx, filled, row_lo, row_hi),
                )
            }
            NextSlice::Done => None,
            NextSlice::Failed(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    /// A backend whose `fill` needs the GIL — the shape of `PgenWindowFiller`, which
    /// drives pgenlib through `Python::attach` (issue #399). `VcfWindowFiller` touches
    /// Python not at all, which is exactly why only PGEN streams ever wedged.
    struct GilHungryBackend {
        /// Fires once, immediately before the producer parks in `Python::attach`, so the
        /// test can drop the engine at the one instant that actually reproduces #399.
        entered_fill: Mutex<Option<mpsc::Sender<()>>>,
    }

    impl EngineBackend for GilHungryBackend {
        type Slot = ();

        fn n_jobs(&self) -> usize {
            1
        }

        fn fill(&self, _job_idx: usize, _slot: &mut ()) -> anyhow::Result<()> {
            if let Some(tx) = self.entered_fill.lock().unwrap().take() {
                let _ = tx.send(());
            }
            Python::attach(|_py| ());
            Ok(())
        }

        fn n_batch_rows(&self, _job_idx: usize, _slot: &()) -> usize {
            1
        }

        fn generate(
            &self,
            _job_idx: usize,
            _slot: &(),
            _row_lo: usize,
            _row_hi: usize,
        ) -> anyhow::Result<(
            Array1<u8>,
            Option<Array1<i32>>,
            Option<Array1<i32>>,
            Array1<i64>,
        )> {
            unreachable!("this backend is never consumed from — the test only drops it")
        }
    }

    /// Issue #399: dropping a mid-stream engine while holding the GIL must not deadlock.
    ///
    /// Closing the channels (which `drop` does first) unblocks a producer parked on
    /// `rx_free.recv()`/`tx_filled.send()`, but does nothing for one parked *inside*
    /// `fill` waiting on the GIL. Only releasing the GIL across the join does. The
    /// engine is therefore dropped from inside `Python::attach` here, reproducing the
    /// real caller: `EngineState::drop` runs during Python deallocation.
    ///
    /// Structured so a regression FAILS instead of hanging forever: the whole
    /// GIL-holding sequence runs on a worker thread and the test thread waits on a
    /// bounded `recv_timeout`. (The test thread must not hold the GIL itself, or it
    /// would be the thing blocking the producer.)
    #[test]
    fn dropping_mid_stream_engine_does_not_deadlock_a_gil_needing_filler() {
        let (entered_tx, entered_rx) = mpsc::channel::<()>();
        let (dropped_tx, dropped_rx) = mpsc::channel::<()>();

        let worker = std::thread::spawn(move || {
            let backend = Arc::new(GilHungryBackend {
                entered_fill: Mutex::new(Some(entered_tx)),
            });
            let core = StreamEngineCore::new(backend, 1);

            Python::attach(|_py| {
                {
                    let mut state = core.state.lock().unwrap();
                    core.ensure_started(&mut state)
                        .expect("producer thread should spawn");
                }
                // Wait until the producer is at the doorstep of `Python::attach`, then
                // give it a moment to actually park there — this thread holds the GIL,
                // so it cannot get in. Without the sleep the drop can win the race and
                // the pre-fix code would pass by luck.
                entered_rx
                    .recv_timeout(Duration::from_secs(30))
                    .expect("producer never reached fill");
                std::thread::sleep(Duration::from_millis(100));

                drop(core); // -> EngineState::drop -> join, with the GIL held
            });

            let _ = dropped_tx.send(());
        });

        dropped_rx
            .recv_timeout(Duration::from_secs(30))
            .expect("EngineState::drop deadlocked: it joined a GIL-needing producer without releasing the GIL (issue #399)");
        worker.join().expect("worker thread should not panic");
    }
}
