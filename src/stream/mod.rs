//! Generic streaming engine: a producer thread fills window N+1 while the consumer
//! reconstructs from window N.
//!
//! **What is overlapped is I/O latency, not decode.** For SVAR1 there is nothing to
//! decode — the producer faults in `variant_idxs` mmap pages and runs binary searches
//! ahead of the consumer. The OS page cache does not prefetch on an application's
//! access pattern; a producer thread walking a known traversal does. Because
//! `StreamingDataset` has no `__getitem__`, the traversal is fixed and fully known, so
//! the prefetch is speculation-free. (VCF/PGEN backends, #276, DO have decode to
//! amortize — that is the other half of the premise.)
//!
//! Pattern cribbed from genoray's `orchestrator.rs`: named per-stage threads, shutdown
//! by `Sender` drop, and join-everything-then-classify-panics (early-returning on a
//! producer error would leave the consumer blocked on `recv()` forever). Both channels
//! use `crossbeam_channel::bounded(n_slots)`, but only `n_slots` buffers exist in
//! total (recycled between the two channels, never duplicated) — so no single
//! channel's queue can ever exceed its own `n_slots` capacity, meaning **no `send` on
//! either channel can ever block**; `bounded` and `unbounded` are behaviorally
//! identical here. The real backpressure is the free-slot pool: the producer can't
//! get more than `n_slots` windows ahead of the consumer because it has to receive a
//! recycled slot back from `rx_free` before it can fill another.
//!
//! **Slot recycling is NET-NEW here** — genoray does *not* recycle (it allocates each
//! chunk fresh and drops it). We return drained slots to the producer so memory is
//! capped at `n_slots * slot_bytes` regardless of plan length. Start with 2 slots
//! (ping-pong); promote to an N-slot ring only on profiling evidence.
//!
//! **Status: groundwork, no production caller yet.** `run_windows`/`StreamBackend` are
//! exercised only by the unit tests in this module. Wiring `Svar1Store` as a
//! `StreamBackend` and routing the SVAR1 read path through this engine is Task 5's
//! Step 5, deliberately split into a follow-up (see
//! `docs/roadmaps/streaming-dataset.md`); VCF/PGEN backends (#276) are also expected
//! to land on this engine later.

use crossbeam_channel::bounded;

/// One window of the fixed cartesian traversal: regions `[r_lo, r_hi)` on `contig_idx`,
/// crossed with samples `[s_lo, s_hi)`.
#[derive(Clone, Debug)]
pub struct WindowSpec {
    pub contig_idx: usize,
    pub r_lo: usize,
    pub r_hi: usize,
    pub s_lo: usize,
    pub s_hi: usize,
}

/// A source that can fill a window buffer. `Sync` because the producer thread borrows
/// it across the spawn.
///
/// `Buffer` is per-backend by design: SVAR1's is degenerate (offsets only — its
/// on-disk layout is already the target representation), while VCF/PGEN's is an owned
/// decoded table. Do not expect one concrete buffer type across backends.
pub trait StreamBackend: Sync {
    type Buffer: Send;
    /// Fill `slot` with `window`'s data. Called on the producer thread. Implementations
    /// should reuse `slot`'s allocation rather than replacing it (slot recycling).
    fn fill(&self, window: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()>;
}

/// Drive `windows` through a producer/consumer pair, calling `consume` on each filled
/// slot **in plan order**. `n_slots` bounds live buffers (2 = ping-pong).
///
/// Both stages' errors surface as `Err`; neither can hang the other. Slots are recycled
/// via a return channel, so memory is `n_slots * slot_bytes` regardless of plan length.
pub fn run_windows<B, F>(
    backend: &B,
    windows: &[WindowSpec],
    n_slots: usize,
    mut consume: F,
) -> anyhow::Result<()>
where
    B: StreamBackend,
    B::Buffer: Default,
    F: FnMut(&B::Buffer) -> anyhow::Result<()>,
{
    if windows.is_empty() {
        return Ok(());
    }
    let n_slots = n_slots.max(2);

    // filled: producer -> consumer. free: consumer -> producer (slot recycling).
    let (tx_filled, rx_filled) = bounded::<B::Buffer>(n_slots);
    let (tx_free, rx_free) = bounded::<B::Buffer>(n_slots);
    for _ in 0..n_slots {
        tx_free.send(B::Buffer::default()).expect("prefill free slots");
    }

    std::thread::scope(|scope| -> anyhow::Result<()> {
        // Force this closure to OWN `tx_free`/`rx_filled` (not merely borrow them) --
        // do this FIRST, before anything that could panic. `std::thread::scope` catches
        // a panicking closure's unwind internally via `catch_unwind`, and it does so
        // *before* it waits for spawned threads to finish. If `tx_free`/`rx_filled`
        // were left as bare locals of `run_windows` and only borrowed here, a panic in
        // `consume` below would unwind this closure's frame without dropping them --
        // they live one frame further up, in `run_windows` itself, which only unwinds
        // *after* `thread::scope` finishes its wait loop. `tx_free` would stay alive
        // through that wait, and the producer's `rx_free.recv()` would block forever
        // waiting for a slot nothing will ever send: `run_windows` deadlocks and never
        // returns. Moving them here means the panic's unwind drops them as part of
        // unwinding *this* closure, so the producer's `recv()` sees the channel close,
        // returns `Ok(())`, and the wait loop completes. Regression test:
        // `consumer_panic_does_not_hang_producer`.
        let tx_free = tx_free;
        let rx_filled = rx_filled;

        // `tx_filled` and `rx_free` are MOVED into the producer -- it is their sole
        // owner, full stop. No clone of either is ever held back in this scope, so
        // when the producer thread returns (success, `fill` error, or `?` early-exit)
        // its `tx_filled` is dropped and the consumer's `rx_filled.recv()` observes
        // close as soon as the channel drains. This sidesteps orchestrator.rs:184-186's
        // hazard (a stray `Sender` clone kept around for introspection that outlives
        // the producer and blocks shutdown forever) by construction: there is no extra
        // clone to forget to drop.
        let producer = std::thread::Builder::new()
            .name("gvl-stream-producer".into())
            .spawn_scoped(scope, move || -> anyhow::Result<()> {
                for w in windows {
                    // Recycle a drained slot. Err => consumer is gone; stop quietly and
                    // let the consumer's own error be the one reported.
                    let Ok(mut slot) = rx_free.recv() else { return Ok(()) };
                    backend.fill(w, &mut slot)?;
                    if tx_filled.send(slot).is_err() {
                        return Ok(()); // consumer gone
                    }
                }
                Ok(())
            })?;

        let mut consumer_err: Option<anyhow::Error> = None;
        while let Ok(slot) = rx_filled.recv() {
            if consumer_err.is_none() {
                if let Err(e) = consume(&slot) {
                    consumer_err = Some(e);
                }
            }
            // Always recycle, even after an error, so the producer can finish rather
            // than block on rx_free.recv() -- otherwise the join below deadlocks.
            let _ = tx_free.send(slot);
        }

        // Join FIRST, classify AFTER -- never early-return with the producer live.
        match producer.join() {
            Err(_) => anyhow::bail!("streaming producer thread panicked"),
            Ok(Err(e)) => return Err(e),
            Ok(Ok(())) => {}
        }
        if let Some(e) = consumer_err {
            return Err(e);
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Guards `std::panic::take_hook`/`set_hook` for the two tests below that
    /// swap the global panic hook. That swap is process-global and NOT atomic
    /// (take, then set is two separate calls), and cargo runs tests in parallel
    /// threads of one process -- without serializing, one test's `take_hook()`
    /// can observe the other's temporary silent hook as "the previous hook" and
    /// restore *that* instead of the real original, permanently silencing panics
    /// for the rest of the test binary. Diagnostics-only impact (a failing test
    /// still reports FAILED, just without its panic message), but no reason to
    /// leave it racy. A panicking test poisons the mutex; recover with
    /// `into_inner()` rather than propagating the poison to the next test.
    static PANIC_HOOK_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct CountingBackend {
        fills: AtomicUsize,
    }
    impl StreamBackend for CountingBackend {
        type Buffer = Vec<usize>;
        fn fill(&self, w: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()> {
            slot.clear();
            slot.extend(w.r_lo..w.r_hi);
            self.fills.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct FailingBackend;
    impl StreamBackend for FailingBackend {
        type Buffer = Vec<usize>;
        fn fill(&self, _w: &WindowSpec, _slot: &mut Self::Buffer) -> anyhow::Result<()> {
            anyhow::bail!("boom")
        }
    }

    fn windows(n: usize) -> Vec<WindowSpec> {
        (0..n)
            .map(|i| WindowSpec {
                contig_idx: 0,
                r_lo: i * 10,
                r_hi: i * 10 + 10,
                s_lo: 0,
                s_hi: 1,
            })
            .collect()
    }

    #[test]
    fn window_spec_carries_sample_span() {
        let w = WindowSpec { contig_idx: 0, r_lo: 0, r_hi: 10, s_lo: 5, s_hi: 12 };
        assert_eq!(w.s_hi - w.s_lo, 7);
    }

    #[test]
    fn engine_yields_every_window_in_plan_order() {
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        let mut seen = Vec::new();
        run_windows(&be, &windows(5), 2, |slot| {
            seen.push(slot[0]);
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, vec![0, 10, 20, 30, 40], "windows must arrive in plan order");
        assert_eq!(be.fills.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn engine_recycles_slots_and_caps_live_buffers() {
        // Slot recycling is NET-NEW here (genoray's orchestrator does not recycle --
        // it allocates fresh and drops). With 2 slots, at most 2 buffers should ever be
        // *constructed* for the whole run, regardless of window count -- that's the
        // actual claim in this test's name. Counting `consume` calls (as the previous
        // version of this test did) doesn't prove that: an implementation that
        // allocated a fresh buffer per window and never recycled anything would produce
        // an identical `n == 50`. So instead count `Buffer::default()` constructions via
        // a counting buffer type, and assert that count is capped at `n_slots`.
        //
        // The counter is a `static` scoped to this function (not module-level), so no
        // other test can read or perturb it even under parallel `cargo test` execution;
        // still reset it at the top in case this test is ever run more than once (e.g.
        // under a retry harness).
        static CONSTRUCTIONS: AtomicUsize = AtomicUsize::new(0);
        CONSTRUCTIONS.store(0, Ordering::SeqCst);

        #[derive(Debug)]
        struct CountedBuffer(Vec<usize>);
        impl Default for CountedBuffer {
            fn default() -> Self {
                CONSTRUCTIONS.fetch_add(1, Ordering::SeqCst);
                CountedBuffer(Vec::new())
            }
        }

        struct RecyclingBackend;
        impl StreamBackend for RecyclingBackend {
            type Buffer = CountedBuffer;
            fn fill(&self, w: &WindowSpec, slot: &mut Self::Buffer) -> anyhow::Result<()> {
                slot.0.clear();
                slot.0.extend(w.r_lo..w.r_hi);
                Ok(())
            }
        }

        let be = RecyclingBackend;
        let mut n = 0;
        run_windows(&be, &windows(50), 2, |_slot| {
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 50, "every window must still be delivered to consume");
        assert_eq!(
            CONSTRUCTIONS.load(Ordering::SeqCst),
            2,
            "buffer construction count must stay capped at n_slots regardless of plan length"
        );
    }

    /// Regression test for the deadlock fixed alongside this test: an unwinding panic
    /// in `consume` used to leave `tx_free`/`rx_filled` alive as bare locals of
    /// `run_windows`, only *borrowed* (not owned) by the `thread::scope` closure.
    /// `std::thread::scope` catches that closure's panic internally, *before* it waits
    /// for spawned threads to finish -- so those locals never dropped in time, `tx_free`
    /// stayed alive, and the producer's `rx_free.recv()` blocked forever waiting for a
    /// slot nothing would ever send. Empirically reproduced before the fix: the run
    /// below hung with exactly 2 fills completed (the 2 prefilled slots) and never
    /// returned.
    ///
    /// This must show up as a clean, bounded *failure* if the defect regresses, not as
    /// a hang that stalls the whole test binary -- so the risky call runs on its own
    /// detached thread, and this test thread only waits on it with a generous timeout.
    /// If the defect is back, `recv_timeout` times out, the detached thread is leaked
    /// harmlessly (process exit doesn't wait on it), and the assertion below fails.
    #[test]
    fn consumer_panic_does_not_hang_producer() {
        use std::sync::mpsc;
        use std::time::Duration;

        // Serialize with `producer_panic_surfaces_as_err_not_hang` -- see
        // `PANIC_HOOK_LOCK`'s doc comment. `unwrap_or_else` recovers from
        // poisoning (a previous panicking test) instead of propagating it.
        let _guard = PANIC_HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        // Suppress the panic's default backtrace/message noise on stderr -- we are
        // deliberately triggering this panic and asserting on the outcome, not
        // debugging it.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));

        let (done_tx, done_rx) = mpsc::channel();
        std::thread::spawn(move || {
            let be = CountingBackend { fills: AtomicUsize::new(0) };
            // 200 windows, matching the size of plan the defect was reproduced against.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_windows(&be, &windows(200), 2, |_slot| {
                    panic!("deliberate consumer panic");
                })
            }));
            let _ = done_tx.send(result.is_err());
        });

        let finished = done_rx.recv_timeout(Duration::from_secs(10)).unwrap_or(false);
        std::panic::set_hook(prev_hook);
        assert!(
            finished,
            "run_windows must not hang when `consume` panics -- it must unwind (with the \
             panic surfacing to the caller) within the timeout instead of deadlocking"
        );
    }

    /// The `bail!("streaming producer thread panicked")` branch (producer.join() ==
    /// Err, i.e. the producer thread itself panicked rather than `fill()` returning an
    /// `Err`) was reachable but untested. Unlike the consumer-panic case above, this
    /// path does not depend on the ownership fix: a panicking `spawn_scoped` thread
    /// unwinds and drops its own owned locals (including `tx_filled`, moved into it)
    /// regardless, so the consumer's `rx_filled.recv()` sees the channel close and the
    /// join always completes -- no timeout guard needed here.
    #[test]
    fn producer_panic_surfaces_as_err_not_hang() {
        struct PanickingBackend;
        impl StreamBackend for PanickingBackend {
            type Buffer = Vec<usize>;
            fn fill(&self, _w: &WindowSpec, _slot: &mut Self::Buffer) -> anyhow::Result<()> {
                panic!("deliberate producer panic");
            }
        }

        // Serialize with `consumer_panic_does_not_hang_producer` -- see
        // `PANIC_HOOK_LOCK`'s doc comment.
        let _guard = PANIC_HOOK_LOCK.lock().unwrap_or_else(|e| e.into_inner());

        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_windows(&PanickingBackend, &windows(3), 2, |_slot| Ok(()))
        }));
        std::panic::set_hook(prev_hook);

        let r = outcome.expect(
            "run_windows itself must not panic -- the producer's panic must be caught by \
             `.join()` and reported as an Err",
        );
        assert!(r.is_err(), "producer panic must surface as an Err");
        assert!(format!("{:?}", r.unwrap_err()).contains("streaming producer thread panicked"));
    }

    #[test]
    fn producer_error_propagates_and_does_not_hang() {
        // orchestrator.rs's hard-won lesson: early-returning on a producer error
        // leaves the consumer blocked on recv() forever. This must return an Err,
        // not deadlock.
        let be = FailingBackend;
        let r = run_windows(&be, &windows(3), 2, |_slot| Ok(()));
        assert!(r.is_err(), "producer error must surface");
        assert!(format!("{:?}", r.unwrap_err()).contains("boom"));
    }

    #[test]
    fn consumer_error_propagates_and_does_not_hang() {
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        let r = run_windows(&be, &windows(10), 2, |_slot| anyhow::bail!("consumer boom"));
        assert!(r.is_err());
        assert!(format!("{:?}", r.unwrap_err()).contains("consumer boom"));
    }

    #[test]
    fn empty_plan_is_ok() {
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        run_windows(&be, &[], 2, |_slot| Ok(())).unwrap();
        assert_eq!(be.fills.load(Ordering::Relaxed), 0);
    }
}
