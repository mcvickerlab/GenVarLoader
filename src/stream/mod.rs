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
//! Pattern cribbed from genoray's `orchestrator.rs`: named per-stage threads, a
//! `crossbeam_channel::bounded` for backpressure, shutdown by `Sender` drop, and
//! join-everything-then-classify-panics (early-returning on a producer error would
//! leave the consumer blocked on `recv()` forever).
//!
//! **Slot recycling is NET-NEW here** — genoray does *not* recycle (it allocates each
//! chunk fresh and drops it). We return drained slots to the producer so memory is
//! capped at `n_slots * slot_bytes` regardless of plan length. Start with 2 slots
//! (ping-pong); promote to an N-slot ring only on profiling evidence.

use crossbeam_channel::bounded;

/// One window of the fixed cartesian traversal: regions `[r_lo, r_hi)` on `contig_idx`,
/// crossed with every sample.
#[derive(Clone, Debug)]
pub struct WindowSpec {
    pub contig_idx: usize,
    pub r_lo: usize,
    pub r_hi: usize,
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
            .map(|i| WindowSpec { contig_idx: 0, r_lo: i * 10, r_hi: i * 10 + 10 })
            .collect()
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
        // it allocates fresh and drops). With 2 slots, at most 2 buffers exist for
        // the whole run regardless of window count.
        let be = CountingBackend { fills: AtomicUsize::new(0) };
        let mut n = 0;
        run_windows(&be, &windows(50), 2, |_slot| {
            n += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(n, 50);
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
