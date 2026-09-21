# Streaming Mixed Record Decode Fold (#400) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fold the record-backend (VCF/PGEN) mixed variants+tracks window decode into the producer engine, so a mixed record stream decodes each window exactly once instead of twice.

**Architecture:** A new core primitive `StreamEngineCore::ensure_current_window` (split out of `advance`) plus `with_current_window`, exposed as the pymethod `RecordStreamEngine.current_window_realign_inputs`, lets the Python drive read the window the producer has **already** filled. A new inherent `RecordBackend::realign_inputs` clones that slot's four arrays, and the pymethod validates the requested job against the engine's current job so drive/plan drift fails loudly instead of silently pairing one window's tracks with another's variants. Python's record `mixed_realign_window` then takes the drive's `engine` and drops its private zero-job engine.

**Tech Stack:** Rust (PyO3 0.26-style `py.detach`, `numpy` crate, anyhow), Python 3.10+, pytest, pixi + maturin.

**Spec:** `docs/superpowers/specs/2026-09-21-streaming-mixed-record-decode-fold-design.md`

## Global Constraints

- **The gate is deterministic decode counters, never wall-clock.** For the same mixed fixture sweep, `transpose_word_reads` with `realign_tracks=True` must be **exactly equal** to the `realign_tracks=False` run (and `> 0`); for PGEN, `pgen_variants_decoded` must be equal too. Before the fold the realign run is ~2x.
- **Byte-identical parity is the safety net.** Every other test in `tests/dataset/test_streaming_tracks_record.py`, plus `tests/dataset/test_streaming_tracks.py`, `test_streaming_tracks_svar2.py`, `test_streaming_scale.py`, must stay green **unmodified** — except the two tests this plan explicitly ports.
- **No public API change.** Nothing new in `python/genvarloader/__init__.py::__all__`; do **not** touch `docs/source/api.md` or `skills/genvarloader/SKILL.md`. `RecordStreamEngine` is not exported (no `api.md` entry), so the new pymethod is library-internal. `python/genvarloader/_dataset/_streaming.py` is the only Python module that changes.
- **SVAR1 is NOT folded** — `_Svar1Backend.mixed_realign_window` reads offsets with no variant decode. It gains only the `engine` parameter it ignores. SVAR2 likewise.
- **The `jitter > 0` + `realign_tracks` guard stays** (`_streaming.py:2054-2066`). Do not lift it; issue #383 owns that.
- **Rebuild the extension before every Python test run**: `pixi run -e dev maturin develop --release`. Otherwise pytest imports the stale `.so` and every result is about the old binary.
- **`pyproject.toml` relaxes `missing-attribute`/`bad-argument-type`/`unexpected-keyword`/`bad-return` to WARN** (`[tool.pyrefly.errors]`), so the `object`-typed engine boundary cannot fail the typecheck gate. Do not add `# type: ignore` comments (an unused ignore is itself an error in some envs; the repo prefers WARN).
- **Commits trigger the installed prek hooks.** Never `--no-verify`. Commit type prefixes: `refactor(streaming)` / `feat(streaming)` / `perf(streaming)` / `docs(streaming)`. Use `git commit -F <file>` for any message containing backticks (zsh `-m` chokes on them).
- Ruff config ignores E501; docstrings are Google style; `python scripts/docstring_style.py --check python/genvarloader` is a gate. **Do not add code comments other than the ones this plan specifies.**
- All commands run from the worktree root (`git rev-parse --show-toplevel`) via `pixi run -e dev <cmd>`.

## File Structure

- `src/ffi/stream_core.rs` — generic engine state machine. Add `ensure_current_window` (private) and `with_current_window` (`pub(crate)`); refactor `advance` onto them; add the `MarkerBackend` stub + cursor test to `mod tests`. Also gains the updated `backend()` doc (Task 4).
- `src/record_stream/engine.rs` — `RecordBackend` owns decode + CSR; `RecordStreamEngine` is the PyO3 surface. Add the inherent `RecordBackend::realign_inputs`; add the `current_window_realign_inputs` pymethod next to `window_realign_inputs`; `window_realign_inputs`'s doc becomes test-only (Task 4).
- `python/genvarloader/_dataset/_streaming.py` — the drive. `engine` parameter on the `_MixedTracksBackend` protocol + all 4 implementations + `_record_mixed_realign_window`; forward it from `_mixed_track_window`; delete `_VcfBackend._mixed_engine`/`_mixed_engine_obj` and `_PgenBackend._mixed_engine`/`_mixed_engine_obj`; `build_engine` doc fixups; the guard comment gains one sentence.
- `tests/dataset/test_streaming_tracks_record.py` — the gate test, the new pymethod test, the ported CSR test, the module docstring, the `window_realign_inputs` test's docstring.
- `docs/roadmaps/streaming-dataset.md` — the two `#400` follow-up mentions become `fixed` entries with the PR number and the measured counter numbers.

---

## Task 1: `ensure_current_window` + `with_current_window` in the engine core

**Files:**
- Modify: `src/ffi/stream_core.rs` (refactor `advance`, add two methods, extend `#[cfg(test)] mod tests`)
- Test: `src/ffi/stream_core.rs` (`mod tests`)

**Interfaces:**
- Consumes: existing `EngineState<Slot>` (`current: Option<CurrentWindow>`, `next_job_idx`, `done`), `ensure_started`, `B::n_batch_rows`, `B::generate`.
- Produces:
  - `fn ensure_current_window(&self, state: &mut EngineState<B::Slot>) -> anyhow::Result<bool>` — private. `Ok(true)` = `state.current` is set with rows left and `next_row` **untouched**; `Ok(false)` = plan exhausted, `state.done` set; `Err(_)` = producer failure/panic, `state.done` set. Never yields a row slice.
  - `pub(crate) fn with_current_window<R>(&self, f: impl FnOnce(usize, &B::Slot) -> anyhow::Result<R>) -> Option<anyhow::Result<R>>` — `None` = exhausted; `Some(Ok(_))` = closure result for the current `(job_idx, slot)`; `Some(Err(_))` = producer failure/panic. Holds the state lock while `f` runs; consumes **no** rows.

- [ ] **Step 1: Write the failing test**

Append to the `mod tests` block in `src/ffi/stream_core.rs`, immediately **before** its closing `}` (after `dropping_mid_stream_engine_does_not_deadlock_a_gil_needing_filler`):

```rust
    /// Backend with no filler internals, no GIL and no channels of its own: each filled
    /// slot records the job index that produced it (so a test can tell WHICH window the
    /// core is holding) and `generate` returns one byte per requested row (so a test can
    /// tell which rows the cursor asked for).
    struct MarkerBackend {
        n_jobs: usize,
        rows_per_job: usize,
    }

    impl EngineBackend for MarkerBackend {
        type Slot = Vec<usize>;

        fn n_jobs(&self) -> usize {
            self.n_jobs
        }

        fn fill(&self, job_idx: usize, slot: &mut Vec<usize>) -> anyhow::Result<()> {
            slot.clear();
            slot.push(job_idx);
            Ok(())
        }

        fn n_batch_rows(&self, _job_idx: usize, _slot: &Vec<usize>) -> usize {
            self.rows_per_job
        }

        fn generate(
            &self,
            _job_idx: usize,
            _slot: &Vec<usize>,
            row_lo: usize,
            row_hi: usize,
        ) -> anyhow::Result<(
            Array1<u8>,
            Option<Array1<i32>>,
            Option<Array1<i32>>,
            Array1<i64>,
        )> {
            Ok((
                Array1::from_vec(vec![0u8; row_hi - row_lo]),
                None,
                None,
                Array1::from_vec(vec![0i64, (row_hi - row_lo) as i64]),
            ))
        }
    }

    /// Issue #400: `with_current_window` must hand back the window the producer has
    /// already filled WITHOUT consuming any of its rows, and must only move on once that
    /// window is spent. This is what lets the Python mixed record path size its track
    /// query from the producer's decode instead of decoding the window a second time.
    #[test]
    fn with_current_window_exposes_the_current_window_without_consuming_rows() {
        let core = StreamEngineCore::new(
            Arc::new(MarkerBackend {
                n_jobs: 3,
                rows_per_job: 2,
            }),
            1,
        );

        // Starts the producer and receives window 0, consuming nothing.
        let first = core
            .with_current_window(|job_idx, slot| {
                assert_eq!(
                    slot.as_slice(),
                    [job_idx],
                    "slot must carry its own job marker"
                );
                Ok(job_idx)
            })
            .expect("a current window must be available")
            .expect("no producer error");
        assert_eq!(first, 0);

        // Idempotent: window 0 still has all of its rows, so the same window comes back.
        let again = core
            .with_current_window(|job_idx, _slot| Ok(job_idx))
            .expect("window 0 still has rows")
            .expect("no producer error");
        assert_eq!(again, 0, "the peek consumed rows instead of just reading");

        // The row cursor was untouched: window 0 still yields BOTH of its rows.
        let widths: Vec<usize> = (0..2)
            .map(|_| {
                core.next_batch_core()
                    .expect("row slice")
                    .expect("no error")
                    .0
                    .len()
            })
            .collect();
        assert_eq!(widths, [1, 1], "peeked window lost rows");

        // Only a SPENT window advances the core's idea of "current".
        let second = core
            .with_current_window(|job_idx, _slot| Ok(job_idx))
            .expect("window 1 is now current")
            .expect("no producer error");
        assert_eq!(second, 1, "the core advanced before the window was spent");

        // 3 jobs x 2 rows at batch_size 1: 6 batches total, 2 already taken.
        let mut remaining = 0usize;
        while let Some(batch) = core.next_batch_core() {
            batch.expect("no producer error");
            remaining += 1;
        }
        assert_eq!(
            remaining, 4,
            "peeking must not change how many rows the plan yields"
        );

        // Fully drained: the peek reports exhaustion rather than hanging or re-yielding.
        assert!(core
            .with_current_window(|job_idx, _slot| Ok(job_idx))
            .is_none());
    }
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev cargo test --release with_current_window`

Expected: **compile failure** — `no method named 'with_current_window' found for struct 'StreamEngineCore'`. (First `cargo test` in a session compiles the crate; give it a few minutes.)

- [ ] **Step 3: Implement `ensure_current_window`, `with_current_window`, and the `advance` refactor**

In `src/ffi/stream_core.rs`, replace the whole `fn advance(...)` (its current body is the `if state.done { ... }` -> `ensure_started` -> `loop { ... }` block, ending with the `match recv` that returns `NextSlice::Ready/ Failed/ Done`) with the three functions below. Keep `advance`'s existing doc-comment text (the "Byte-identical control flow to the pre-refactor `next_batch_core` loop (issue #276/#283)" paragraph) on `advance` itself, extended with the sentence below:

```rust
    /// Ensure `state.current` is a window with rows left, starting the producer,
    /// recycling a spent window, and receiving the next one as needed.
    ///
    /// `Ok(true)` leaves `state.current` set and `state.current.next_row` UNTOUCHED --
    /// the caller decides whether to consume rows (`advance`) or merely read the window
    /// (`with_current_window`). `Ok(false)` is `NextSlice::Done` (`state.done` set), and
    /// `Err(_)` is `NextSlice::Failed(_)` (`state.done` set); this never yields a row
    /// slice.
    fn ensure_current_window(&self, state: &mut EngineState<B::Slot>) -> anyhow::Result<bool> {
        if state.done {
            return Ok(false);
        }
        if let Err(e) = self.ensure_started(state) {
            state.done = true;
            return Err(e);
        }

        loop {
            if let Some(cur) = state.current.as_ref() {
                if cur.next_row < cur.n_batch_rows {
                    return Ok(true);
                }
            }

            // Current window is spent (or absent): recycle it, then fetch the next.
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
                    // Loop back to check the newly received window.
                }
                Err(_) => {
                    // Channel closed => producer finished. JOIN FIRST, classify AFTER --
                    // never return with the producer live and unjoined.
                    state.done = true;
                    if let Some(h) = state.producer.take() {
                        return match h.join() {
                            Err(_) => Err(anyhow::anyhow!(
                                "streaming producer thread panicked"
                            )),
                            Ok(Err(e)) => Err(e),
                            Ok(Ok(())) => Ok(false),
                        };
                    }
                    return Ok(false);
                }
            }
        }
    }

    /// Run `f` against the job index and filled slot of the CURRENT window, ensuring one
    /// exists (issue #400). Consumes nothing: a following `next_batch_core` still starts
    /// at `state.current.next_row`.
    ///
    /// `None` = plan exhausted; `Some(Err(_))` = producer error/panic, joined and
    /// classified exactly as `next_batch_core` does. `f` runs while the state lock is
    /// held -- the slot it reads must not be recycled under it -- so it must not re-enter
    /// the engine.
    pub(crate) fn with_current_window<R>(
        &self,
        f: impl FnOnce(usize, &B::Slot) -> anyhow::Result<R>,
    ) -> Option<anyhow::Result<R>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        match self.ensure_current_window(&mut state) {
            Ok(true) => {
                let cur = state
                    .current
                    .as_ref()
                    .expect("Ok(true) implies a current window");
                Some(f(cur.job_idx, &cur.filled))
            }
            Ok(false) => None,
            Err(e) => Some(Err(e)),
        }
    }

    /// Advance the shared iteration state to the next generatable row slice, or to
    /// exhaustion/error. Byte-identical control flow to the pre-refactor `next_batch_core`
    /// loop (issue #276/#283): the only change is that the terminal action -- generating
    /// bytes from `[row_lo, row_hi)` -- is left to the caller instead of being inlined here,
    /// so both the haplotype and variants-output paths can reuse this same cursor walk.
    ///
    /// The cursor walk itself now lives in `ensure_current_window`, split out so
    /// `with_current_window` can read the current window without consuming its rows
    /// (issue #400).
    fn advance(&self, state: &mut EngineState<B::Slot>) -> NextSlice {
        match self.ensure_current_window(state) {
            Ok(true) => {}
            Ok(false) => return NextSlice::Done,
            Err(e) => return NextSlice::Failed(e),
        }
        let cur = state
            .current
            .as_mut()
            .expect("ensure_current_window guarantees a current window");
        let row_lo = cur.next_row;
        let row_hi = (row_lo + self.batch_size).min(cur.n_batch_rows);
        cur.next_row = row_hi;
        NextSlice::Ready {
            job_idx: cur.job_idx,
            row_lo,
            row_hi,
        }
    }
```

Do **not** change `NextSlice`, `EngineState`, `next_batch_core`, `next_batch_variants_core`, or any other `advance` caller: they all keep working because `advance`'s signature and observable behaviour are unchanged.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev cargo test --release with_current_window`
Expected: `test tests::with_current_window_exposes_the_current_window_without_consuming_rows ... ok`, `1 passed`.

Then run the whole Rust suite for the crate to prove the `advance` refactor is behaviour-preserving:

Run: `pixi run -e dev cargo test --release`
Expected: all tests pass (`0 failed`) — in particular the two `next_batch`/`advance`-driven tests in `stream_core.rs` and every `#[cfg(test)]` test in `src/record_stream/*`.

- [ ] **Step 5: Lint and commit**

Run: `pixi run -e dev cargo clippy --all-targets 2>&1 | tail -n 30`
Expected: no new warning mentioning `stream_core.rs` (the tree may already have unrelated warnings).

```bash
cat > /tmp/400-task1-commit.txt <<'EOF'
refactor(streaming): split ensure_current_window out of advance (#400)

with_current_window needs the current window's slot WITHOUT consuming a row,
which advance's fused check-and-advance cannot express. Extract the ensure
half and reimplement advance on top of it; control flow is unchanged, so every
existing advance caller (next_batch_core, next_batch_variants_core, ...) keeps
its exact behaviour. Rust unit test pins the peek semantics on a stub backend.
EOF
git add src/ffi/stream_core.rs
git commit -F /tmp/400-task1-commit.txt
```

---

## Task 2: `RecordBackend::realign_inputs` + the `current_window_realign_inputs` pymethod

**Files:**
- Modify: `src/record_stream/engine.rs` (add an inherent method to `impl RecordBackend`, add a pymethod to `#[pymethods] impl RecordStreamEngine`)
- Test: `tests/dataset/test_streaming_tracks_record.py` (new test)

**Interfaces:**
- Consumes: `StreamEngineCore::with_current_window` (Task 1).
- Produces:
  - `RecordBackend::realign_inputs(&self, slot: &DecodedWindow) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i64>)` — clones `slot.v_starts`, `slot.ilens`, `slot.geno_v_idxs`, `slot.geno_offsets`.
  - `RecordStreamEngine.current_window_realign_inputs(contig_idx: int, region_starts: Sequence[int], region_ends: Sequence[int], s_lo: int, s_hi: int) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.int32], NDArray[np.int64]] | None` — the current window's arrays, or `None` when exhausted; `ValueError` if the requested job is not the engine's current job; `RuntimeError` if the producer failed/panicked.

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_streaming_tracks_record.py`, after `test_record_window_csr_replicates_across_regions` (before the `# --- Issue #375 Track B, Task 8` comment):

```python
@pytest.mark.parametrize("backend", BACKENDS)
def test_current_window_realign_inputs_reads_the_producers_window(
    streaming_record_tracks_fixture, backend
):
    """Issue #400: the window the producer has ALREADY filled must be readable from
    the drive's own engine -- that is what removes the second decode the track side
    used to do. Checked against the independent `window_realign_inputs` decode of the
    same window, which runs in the caller's thread (`debug_fill`) and is now
    test-only (its `_mixed_engine()` production caller was deleted).

    Also pins the identity guard: a request that does not describe the engine's
    current window must raise rather than silently pair one window's tracks with
    another's variants.
    """
    f = streaming_record_tracks_fixture(backend)
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    b = sds._backend
    n_s = sds.n_samples
    r_idx = np.arange(len(sds._regions), dtype=np.intp)
    contig_idx = int(b._regions[r_idx[0], 0])
    t_starts = np.ascontiguousarray(b._regions[r_idx, 1], np.uint32)
    t_ends = np.ascontiguousarray(b._regions[r_idx, 2], np.uint32)

    engine = b.build_engine([(contig_idx, t_starts, t_ends, 0, n_s)], 4, -1)

    # A window that is NOT the engine's current one must fail loudly.
    with pytest.raises(ValueError, match="does not match"):
        engine.current_window_realign_inputs(
            contig_idx, t_starts.tolist(), (t_ends + 1).tolist(), 0, n_s
        )

    got = engine.current_window_realign_inputs(
        contig_idx, t_starts.tolist(), t_ends.tolist(), 0, n_s
    )
    assert got is not None, "the first window must be available after the peek"

    # Independent source of truth: a plan-less engine decodes this exact window in the
    # caller's thread, with no producer and no slot involved.
    oracle = b.build_engine([], 1, 1)
    expected = oracle.window_realign_inputs(
        contig_idx, t_starts.tolist(), t_ends.tolist(), 0, n_s
    )
    for name, a, e in zip(
        ("v_starts", "ilens", "geno_v_idxs", "geno_offsets"), got, expected
    ):
        e = np.ascontiguousarray(e)
        assert a.dtype == e.dtype, f"{name} dtype drifted"
        np.testing.assert_array_equal(
            a, e, err_msg=f"{name} drifted from the window's own decode"
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pixi run -e dev maturin develop --release && \
pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py::test_current_window_realign_inputs_reads_the_producers_window -q
```

Expected: FAIL — `AttributeError: 'RecordStreamEngine' object has no attribute 'current_window_realign_inputs'` (maturin here only proves the build is current; nothing new is implemented yet).

- [ ] **Step 3: Implement `RecordBackend::realign_inputs`**

In `src/record_stream/engine.rs`, insert immediately after the `debug_fill` method (which ends with `self.filler.fill(job, c, &mut slot)?;` ... `Ok(slot)`) inside `impl RecordBackend`:

```rust
    /// Clone the filled window's realign inputs out of the producer's slot: the
    /// window-local static table (`v_starts`/`ilens`), the per-hap CSR values
    /// (`geno_v_idxs`) and its offsets (`geno_offsets`). Cloned, not moved: the slot stays
    /// owned by the engine state so `generate` can still reconstruct this window's
    /// batches (issue #400). Returns owned `Vec`s so the caller can convert to numpy
    /// after releasing the engine lock.
    fn realign_inputs(&self, slot: &DecodedWindow) -> (Vec<i32>, Vec<i32>, Vec<i32>, Vec<i64>) {
        (
            slot.v_starts.clone(),
            slot.ilens.clone(),
            slot.geno_v_idxs.clone(),
            slot.geno_offsets.clone(),
        )
    }
```

- [ ] **Step 4: Implement the `current_window_realign_inputs` pymethod**

In `src/record_stream/engine.rs`, insert immediately after the `window_realign_inputs` method (i.e. just before the closing `}` of the `#[pymethods] impl RecordStreamEngine` block):

```rust
    /// Read the CURRENT window's realign inputs from the producer's already-filled slot
    /// (issue #400).
    ///
    /// Returns the same 4-tuple `window_realign_inputs` returns -- `(v_starts, ilens,
    /// geno_v_idxs, geno_offsets)`, all window-local -- but decodes NOTHING: the window was
    /// decoded once by the producer (`fill`) and this reads that slot. `None` means the
    /// engine's plan is exhausted.
    ///
    /// The requested job (`contig_idx`, `region_starts`/`region_ends`, `s_lo`/`s_hi`) must
    /// EQUAL the engine's current job: the Python drive derives the request from its own
    /// plan, and if the two plans have drifted apart, pairing this window's track query
    /// with another window's variants would be silently wrong. `ValueError` on mismatch --
    /// loud, because the alternative is wrong bytes.
    ///
    /// Consumes no rows: the caller can size its track query and then pull the window's
    /// batches as usual.
    #[pyo3(signature = (contig_idx, region_starts, region_ends, s_lo, s_hi))]
    #[allow(clippy::too_many_arguments)]
    fn current_window_realign_inputs<'py>(
        &self,
        py: Python<'py>,
        contig_idx: usize,
        region_starts: Vec<u32>,
        region_ends: Vec<u32>,
        s_lo: usize,
        s_hi: usize,
    ) -> PyResult<Option<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i64>>,
    )>> {
        if region_starts.len() != region_ends.len() {
            return Err(PyValueError::new_err(
                "current_window_realign_inputs: region_starts and region_ends must have the same length",
            ));
        }
        let regions: Vec<(u32, u32)> = region_starts.into_iter().zip(region_ends).collect();
        let out = py.detach(|| {
            self.core.with_current_window(|job_idx, slot| {
                Ok((job_idx, self.core.backend().realign_inputs(slot)))
            })
        });
        let Some(out) = out else {
            return Ok(None);
        };
        let (job_idx, (v_starts, ilens, geno_v_idxs, geno_offsets)) =
            out.map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

        let backend = self.core.backend();
        let job = &backend.jobs[job_idx];
        if job.contig_idx != contig_idx
            || job.regions != regions
            || job.s_lo != s_lo
            || job.s_hi != s_hi
        {
            return Err(PyValueError::new_err(format!(
                "current_window_realign_inputs: the engine's current window is job {job_idx} \
                 (contig {}, regions {:?}, samples {}..{}), which does not match the requested \
                 contig {contig_idx}, regions {regions:?}, samples {s_lo}..{s_hi}; the drive and \
                 the engine's plan have drifted apart",
                job.contig_idx, job.regions, job.s_lo, job.s_hi
            )));
        }
        Ok(Some((
            Array1::from_vec(v_starts).into_pyarray(py),
            Array1::from_vec(ilens).into_pyarray(py),
            Array1::from_vec(geno_v_idxs).into_pyarray(py),
            Array1::from_vec(geno_offsets).into_pyarray(py),
        )))
    }
```

`PyValueError`, `PyRuntimeError`, `Array1`, `IntoPyArray`, `PyArray1` and `RecordJob`'s fields are already imported/visible in this file (`use pyo3::exceptions::{PyRuntimeError, PyValueError};`, line 33).

- [ ] **Step 5: Rebuild and run the test**

Run:

```bash
pixi run -e dev maturin develop --release && \
pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py::test_current_window_realign_inputs_reads_the_producers_window -q
```

Expected: PASS, both `vcf` and `pgen` parametrizations (`2 passed`).

- [ ] **Step 6: Commit**

```bash
cat > /tmp/400-task2-commit.txt <<'EOF'
feat(streaming): expose the current window's realign inputs (#400)

RecordStreamEngine::current_window_realign_inputs reads the producer's
already-decoded slot via the new with_current_window core primitive, so the
Python drive can size a window's track query without a second decode. The
requested job is validated against the engine's current job: a drifted plan
raises ValueError instead of pairing one window's tracks with another's
variants.
EOF
git add src/record_stream/engine.rs tests/dataset/test_streaming_tracks_record.py
git commit -F /tmp/400-task2-commit.txt
```

---

## Task 3: Fold the Python record mixed path onto the drive's engine

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py`
- Test: `tests/dataset/test_streaming_tracks_record.py`

**Interfaces:**
- Consumes: `RecordStreamEngine.current_window_realign_inputs` (Task 2).
- Produces: `_MixedTracksBackend.mixed_realign_window(..., engine: object | None = None)` — the drive's engine for the record backends, ignored by SVAR1/SVAR2. `_mixed_track_window(..., engine: object | None = None)` forwards it. `_record_mixed_realign_window(..., engine: object | None = None)` requires it.

- [ ] **Step 1: Write the failing gate test**

Append to `tests/dataset/test_streaming_tracks_record.py` (after the test from Task 2):

```python
@pytest.mark.parametrize("backend", BACKENDS)
def test_mixed_record_realign_decodes_each_window_once(
    streaming_record_tracks_fixture, backend
):
    """Issue #400 gate: with `realign_tracks=True` the mixed record path must NOT
    decode the window a second time. The producer already decodes every window
    (`fill`); before this change the track side ALSO decoded it synchronously via a
    private plan-less engine (`_mixed_engine()`), roughly doubling every window's
    decode.

    Measured with the deterministic decode counters, never wall-clock: this tree
    shares a noisy CI node. `transpose_word_reads` counts the genotype transpose
    both `VcfWindowFiller` and `PgenWindowFiller` run (`fill_decoded_window`), and
    `pgen_variants_decoded` additionally counts PGEN's decoded variants. Both are
    process-wide atomics, reset immediately before each sweep and read immediately
    after it is fully drained (so no decode is still in flight).
    """
    from genvarloader.genvarloader import (
        pgen_variants_decoded,
        pgen_variants_decoded_reset,
        transpose_word_reads,
        transpose_word_reads_reset,
    )

    f = streaming_record_tracks_fixture(backend)

    def sweep(realign: bool) -> tuple[int, int]:
        sds = (
            gvl.StreamingDataset(
                f.bed,
                reference=f.reference_path,
                variants=f.variants_path,
                tracks=[f.table, f.bigwigs],
            )
            .with_seqs("haplotypes")
            .with_settings(realign_tracks=realign)
        )
        transpose_word_reads_reset()
        pgen_variants_decoded_reset()
        for _ in sds.to_iter(batch_size=4):
            pass
        return transpose_word_reads(), pgen_variants_decoded()

    words_without, decoded_without = sweep(False)
    words_with, decoded_with = sweep(True)

    assert words_without > 0, "counter is not wired (no window was decoded)"
    assert words_with == words_without, (
        f"realign_tracks=True made {words_with} transposed word reads vs "
        f"{words_without} with realign_tracks=False; the window is being decoded "
        "twice again (issue #400 regression)"
    )
    if backend == "pgen":
        assert decoded_with == decoded_without, (
            f"realign_tracks=True decoded {decoded_with} PGEN variants vs "
            f"{decoded_without} with realign_tracks=False (issue #400 regression)"
        )
```

- [ ] **Step 2: Run it to verify it fails, and record the before numbers**

Run:

```bash
pixi run -e dev pytest "tests/dataset/test_streaming_tracks_record.py::test_mixed_record_realign_decodes_each_window_once" -q
```

Expected: FAIL for BOTH backends with `words_with == 2 * words_without` (the second decode from `_mixed_engine`). **Copy both backends' `words_without`/`words_with` and `decoded_without`/`decoded_with` numbers into your task notes** — Task 4 writes them into the roadmap.

- [ ] **Step 3: Add `engine` to the protocol and all four implementations**

In `python/genvarloader/_dataset/_streaming.py`, add the parameter to each of these five signatures, as the last parameter, written **unquoted** exactly like this (the module has `from __future__ import annotations`, and `tests/dataset/test_streaming_tracks.py:809-831` compares parameter names AND annotations across backends with `eval_str=True`, so the text must match everywhere; `object | None` resolves without needing a module-level import):

```python
        engine: object | None = None,
```

1. `_MixedTracksBackend.mixed_realign_window` (protocol, ~line 488) — add it after `row_ends`, before the return annotation, and add to its docstring: `The drive's engine, for backends whose window decode the engine owns (the record backends); SVAR1/SVAR2 ignore it.`
2. `_Svar1Backend.mixed_realign_window` (~line 4020) — add the same line plus a docstring sentence: `engine` is accepted for protocol uniformity and ignored: SVAR1's state comes from its on-disk CSR, not from a stream engine.
3. `_Svar2Backend.mixed_realign_window` (~line 4573) — same as SVAR1's note.
4. `_VcfBackend.mixed_realign_window` (~line 4834) — add the line, and change its delegation to:

```python
        return _record_mixed_realign_window(
            self, r_idx, s_idx, t_starts, t_ends, row_starts, row_ends, engine
        )
```

5. `_PgenBackend.mixed_realign_window` (~line 5090) — exactly the same delegation change.

- [ ] **Step 4: Make `_record_mixed_realign_window` read the engine's current window**

In `_record_mixed_realign_window` (~line 735):

1. Add the parameter as the last one: `engine: object | None = None,`.
2. Add to the docstring, right after the `Shared by \`_VcfBackend\` and \`_PgenBackend\` ...` paragraph:

```
    The variant tables and CSR come from the DRIVE's engine, which decoded
    this window in its producer thread: `current_window_realign_inputs` reads
    that already-filled slot and decodes nothing (issue #400 -- this used to
    decode the window a second time through a private plan-less engine). It
    also validates that the engine's current job is exactly this window, so a
    drive/plan drift raises `ValueError` instead of silently pairing one
    window's tracks with another's variants.
```

3. Replace the `engine = backend._mixed_engine()` / `engine.window_realign_inputs(...)` block with:

```python
    if engine is None:
        raise ValueError(
            "_record_mixed_realign_window requires the drive's RecordStreamEngine: "
            "the window's variant decode is read from the engine's current slot "
            "(issue #400). Pass engine=<RecordStreamEngine>."
        )
    window_inputs = engine.current_window_realign_inputs(
        contig_idx,
        np.ascontiguousarray(t_starts, np.uint32).tolist(),
        np.ascontiguousarray(t_ends, np.uint32).tolist(),
        int(s_idx[0]),
        int(s_idx[-1]) + 1,
    )
    if window_inputs is None:
        raise RuntimeError("streaming engine exhausted before the plan did")
    v_starts, ilens, geno_v_idxs, csr = window_inputs
```

4. In the jitter paragraph of the same docstring, change "The engine is queried with `t_starts`/`t_ends`" to "The engine call is validated against `t_starts`/`t_ends`" (the rest of that paragraph stays true).

- [ ] **Step 5: Forward the engine from the drive**

1. `_mixed_track_window` (~line 1677): add `engine: object | None = None,` as the last parameter (after `region_offsets`), document it in the Args block (`engine: The drive's engine, forwarded to ``backend.mixed_realign_window``; required by the record backends.`), and append `engine` as the final POSITIONAL argument of its existing `backend.mixed_realign_window(...)` call. Do not reorder or rename the arguments already there (it passes `r_idx`, the sample/track bounds and the row bounds); the composed call must end up shaped like:

```python
        window = backend.mixed_realign_window(
            r_idx,
            s_idx_w,
            t_starts,
            t_ends,
            row_starts,
            row_ends,
            engine,
        )
```

Keep whatever local variable names that call site already uses; only the trailing `engine` is new.
2. The engine drive's call site (~line 2277, inside `_iter_batches`) becomes:

```python
                    if tb is not None:
                        track_w = self._mixed_track_window(
                            tb,
                            backend,
                            r_idx,
                            np.arange(s_lo, s_hi, dtype=np.intp),
                            region_offsets,
                            engine,
                        )
```

(`engine` is the `build_engine` result from ~line 2221. The SVAR2 sync call site ~line 2657 stays as-is: it passes no engine, so SVAR2 gets `None`, which it ignores.)

- [ ] **Step 6: Delete the dead private engines**

1. Remove `self._mixed_engine_obj = None` from `_VcfBackend.__init__` (~line 4747) and from `_PgenBackend.__init__` (~line 5062).
2. Remove the entire `_mixed_engine` method from `_VcfBackend` (~4816-4832) and from `_PgenBackend` (~5072-5088).
3. In both `build_engine` docstrings (~4920, ~5162), the comment that reads

```
        # -- including `_mixed_engine()`'s `build_engine([], 1, 1)`, where
        # `touched_contigs` is empty and this now allocates nothing at all.
```

becomes

```
        # -- including the plan-less engine the CSR test builds as its
        # independent oracle (`build_engine([], 1, 1)`), where
        # `touched_contigs` is empty and this allocates nothing at all.
```

- [ ] **Step 7: Add the one guard-comment sentence**

At the `jitter > 0` + `realign_tracks` guard in `_iter_batches` (~line 2054-2066), add one sentence to the existing explanation: `Record backends no longer decode in the consumer thread at all (issue #400): their decode comes from the drive's engine, whose job bounds are the jitter-translated ones. This guard is kept for SVAR1's raw-bounds re-read (#383).` Do not change the guard's behaviour.

- [ ] **Step 8: Port `test_record_window_csr_replicates_across_regions`**

The mixed call now needs an engine, and `b._mixed_engine()` no longer exists. In `tests/dataset/test_streaming_tracks_record.py`:

1. Move `contig_idx = int(b._regions[r_idx[0], 0])` up so it is computed right after `row_ends`.
2. Build the plan engine and pass it:

```python
    engine = b.build_engine(
        [(contig_idx, t_starts.astype(np.uint32), t_ends.astype(np.uint32), 0, n_s)],
        4,
        -1,
    )
    state, t_ends_ext = b.mixed_realign_window(
        r_idx, s_idx, t_starts, t_ends, row_starts, row_ends, engine
    )
```

3. Replace `engine = b._mixed_engine()` in the independent-oracle block with a plan-less engine, and rename the local so the two engines cannot be confused:

```python
    # Independent source of truth: a plan-less engine decodes this exact window in the
    # caller's thread (`debug_fill`), with no producer and no slot involved.
    oracle = b.build_engine([], 1, 1)
    _, _, _, csr = oracle.window_realign_inputs(
        contig_idx,
        np.ascontiguousarray(t_starts, np.uint32).tolist(),
        np.ascontiguousarray(t_ends, np.uint32).tolist(),
        int(s_idx[0]),
        int(s_idx[-1]) + 1,
    )
```

- [ ] **Step 9: Update the two stale docstrings in that test file**

1. Module docstring (lines 1-6) becomes:

```python
"""Rust seam for VCF/PGEN mixed variants+tracks (issue #375, Track B).

`current_window_realign_inputs` is the production seam: the drive reads the window
the producer has already filled, so a mixed record window is decoded once (issue
#400). `window_realign_inputs` decodes a window it is handed, in the caller's thread;
it is now TEST-ONLY and stays as the independent oracle these tests check the folded
path against.
"""
```

2. `test_window_realign_inputs_matches_before_and_during_producer` (~line 112): append to its docstring: `Its production caller (`_mixed_engine()`) was deleted in issue #400, so this test is now the only reason the pymethod stays: it still pins the concurrent-call safety of the shared filler.`

- [ ] **Step 10: Rebuild and run the file, then the gate**

Run:

```bash
pixi run -e dev maturin develop --release && \
pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q
```

Expected: all tests PASS, including `test_mixed_record_realign_decodes_each_window_once` (both backends) and the ported CSR test.

Then re-run the gate alone to confirm and to read the after numbers:

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py::test_mixed_record_realign_decodes_each_window_once -q
```

Expected: PASS. Record the after numbers for Task 4: `words_with == words_without` for VCF and PGEN, `decoded_with == decoded_without` for PGEN.

- [ ] **Step 11: Commit**

```bash
cat > /tmp/400-task3-commit.txt <<'EOF'
perf(streaming): fold the mixed record decode into the producer (#400)

The VCF/PGEN mixed variants+tracks path decoded every window twice: the
producer filled it for the haplotypes, then the track side re-decoded it
synchronously through a private plan-less engine. The drive's engine is now
the only decoder: _mixed_track_window forwards it and
_record_mixed_realign_window reads the producer's current slot via
current_window_realign_inputs, so realign_tracks=True costs the same decode
as realign_tracks=False (gated on transpose_word_reads and
pgen_variants_decoded, not wall-clock). The private engines are deleted.
EOF
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks_record.py
git commit -F /tmp/400-task3-commit.txt
```

---

## Task 4: Retire the stale docs, tick the roadmap, verify the whole tree

**Files:**
- Modify: `src/ffi/stream_core.rs` (`backend()` doc), `src/record_stream/engine.rs` (`window_realign_inputs` doc), `src/record_stream/pgen.rs` (`reader_lock` doc), `docs/roadmaps/streaming-dataset.md`
- Test: whole tree

**Interfaces:**
- Consumes: everything above; the measured counter numbers from Task 3; the PR number you create in Step 5.
- Produces: docs that describe the post-#400 arrangement truthfully.

- [ ] **Step 1: Fix the three Rust doc comments**

1. `src/ffi/stream_core.rs`, `backend()`'s doc (~lines 228-261): delete the M1 paragraph ("both callers ... `RecordStreamEngine::window_realign_inputs` (issue #375 Track B) -- a genuine production path ... folding the mixed path onto the drive engine (removing the double decode) is the obvious next step and would make this the live case (final review, M1)"). Replace it with: `- `RecordStreamEngine::window_realign_inputs` and `debug_decode_window` (issues #375/#391) -- now TEST-ONLY: it decodes a window in the caller's thread for parity oracles. Issue #400 removed the production consumer (a private plan-less engine), so the concurrent-with-a-live-producer case is exercised only by `test_window_realign_inputs_matches_before_and_during_producer`. The safety property stands: both callers release the GIL, so both remain safe against a live producer.` Keep the remaining paragraphs about `py.detach` and the two caller conditions.
2. `src/record_stream/engine.rs`, `window_realign_inputs`'s doc (~1049-1083): replace the "Python needs these BEFORE the window's first batch ... Net cost: roughly 2x decode on the mixed path (one per engine); folding this into a single shared engine is a tracked follow-up, not a v1 requirement." paragraphs with a short TEST-ONLY note: the production consumer was deleted in #400 (the drive now reads the producer's slot via `current_window_realign_inputs`), and this stays as the independent CSR oracle used by `test_record_window_csr_replicates_across_regions` and `test_window_realign_inputs_matches_before_and_during_producer`. Keep the bullet list describing the returned arrays' layout.
3. `src/record_stream/pgen.rs`, `PgenWindowFiller::reader_lock`'s doc (~430-434): `fill` is still called concurrently (test-only decodes vs a live producer), so the lock stays, but drop any claim that a production consumer shares the reader: add `Since issue #400 the mixed path decodes only in the producer, so this lock exists for the test-only decode accessors and for any future shared-engine caller.` Keep the LOCK ORDERING line verbatim.

- [ ] **Step 2: Full Python + Rust verification**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset tests/unit -q
pixi run -e dev cargo test --release
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
pixi run -e dev python scripts/docstring_style.py --check python/genvarloader
```

Expected: all green. `typecheck` MAY report `missing-attribute`/`bad-argument-type` WARNINGS (relaxed to warn in `pyproject.toml`); the command must exit 0. If any test outside this plan's scope fails, **stop and report** — do not "fix" parity tests.

- [ ] **Step 3: Tick the roadmap**

In `docs/roadmaps/streaming-dataset.md`, two follow-up mentions of #400 must become fixed entries (both are inside the long table rows for the #279 intervals plan at ~line 91 and the `2026-09-11-streaming-mixed-tracks-non-svar1.md` plan at ~line 92).

In line 91, replace

```markdown
[#400](https://github.com/mcvickerlab/GenVarLoader/issues/400) (fold the mixed window decode into the producer to remove the 2x decode).
```

with

```markdown
[#400](https://github.com/mcvickerlab/GenVarLoader/issues/400) (fold the mixed window decode into the producer to remove the 2x decode) ✅ **fixed**, PR [#<PR>](https://github.com/mcvickerlab/GenVarLoader/pull/<PR>) — the drive's `RecordStreamEngine` is now the only decoder: `RecordStreamEngine.current_window_realign_inputs` reads the producer's already-filled slot (validating the requested job against the engine's current job), `_mixed_track_window` forwards `engine=`, and the private plan-less `_mixed_engine()` is deleted. Deterministic-counter gate (not wall-clock): `transpose_word_reads` on the record mixed fixture is now identical with `realign_tracks` on and off (<WORDS> word reads, both) and PGEN's `pgen_variants_decoded` likewise (<DECODED> variants, both) — the realign run was ~2x before.
```

In line 92, replace

```markdown
[#400](https://github.com/mcvickerlab/GenVarLoader/issues/400) (fold the mixed window decode into the producer).
```

with the same ✅ sentence from line 91 (identical replacement text, `<PR>`, `<WORDS>` and `<DECODED>` filled in).

`<PR>` comes from Step 5; `<WORDS>` from Task 3's `words_without`/`words_with` (equal after the fold); `<DECODED>` from Task 3's PGEN `decoded_without`/`decoded_with` (equal after the fold). Do not write placeholders into the file.

- [ ] **Step 4: Commit the docs**

```bash
cat > /tmp/400-task4-commit.txt <<'EOF'
docs(streaming): record the #400 decode fold (#400)

window_realign_inputs and debug_decode_window are now test-only seams: the
record mixed path reads the producer's own slot. Update the three Rust doc
comments that still describe the double decode, and tick the roadmap's #400
follow-ups with the measured decode-counter numbers.
EOF
git add src/ffi/stream_core.rs src/record_stream/engine.rs src/record_stream/pgen.rs docs/roadmaps/streaming-dataset.md
git commit -F /tmp/400-task4-commit.txt
```

- [ ] **Step 5: Push and open the PR**

Write the PR body first:

```bash
cat > /tmp/400-pr-body.md <<'EOF'
Closes #400.

The VCF/PGEN mixed variants+tracks path decoded every window twice: the Rust
producer filled it for the haplotypes, then the track side re-decoded it
synchronously through a private plan-less `_mixed_engine()`. The drive's
`RecordStreamEngine` is now the only decoder.

- `StreamEngineCore::with_current_window` + `ensure_current_window` (split out of
  `advance`): read the current window's slot without consuming a row.
- `RecordStreamEngine.current_window_realign_inputs`: returns that window's
  `(v_starts, ilens, geno_v_idxs, geno_offsets)`, validating the requested job
  against the engine's current job so a drifted plan raises `ValueError` instead
  of pairing one window's tracks with another's variants.
- Python: `engine` threaded through `_mixed_track_window` and
  `_RecordStreamEngine`-backed `mixed_realign_window`; the private plan-less
  engines are deleted. SVAR1/SVAR2 accept and ignore the parameter.

Gate is deterministic, not wall-clock: `transpose_word_reads` (and PGEN's
`pgen_variants_decoded`) are now identical with `realign_tracks` on and off,
where the realign run was ~2x before. Every existing mixed parity suite is
unchanged and green. No public API change; `window_realign_inputs` stays as the
test-only oracle.
EOF
git push -u origin feat/400-mixed-record-decode-fold
gh pr create --base streaming --title 'perf(streaming): fold the mixed record decode into the producer (#400)' --body-file /tmp/400-pr-body.md
```

Then **re-run Step 3 with the real PR number**, commit that as `docs(streaming): link the #400 PR` (same `git commit -F` discipline), and push again. Add the PR to the **StreamingDataset** GitHub Project (`gh pr edit <PR> --add-project StreamingDataset`; if the token lacks `read:project`, ask the user to run `gh auth refresh -s read:project`).

---

## Final Verification Checklist

- [ ] `pixi run -e dev cargo test --release` — all Rust tests pass, including `with_current_window_exposes_the_current_window_without_consuming_rows`.
- [ ] `pixi run -e dev pytest tests/dataset tests/unit -q` — all pass, and `git diff --stat tests/dataset/test_streaming_tracks.py tests/dataset/test_streaming_tracks_svar2.py tests/dataset/test_streaming_scale.py` shows **no changes** to those files.
- [ ] The gate: `transpose_word_reads` equal with `realign_tracks` on/off for both backends; `pgen_variants_decoded` equal for PGEN; both non-zero without the fold (numbers recorded in the roadmap).
- [ ] `git grep -n '_mixed_engine'` returns nothing.
- [ ] `pixi run -e dev ruff check python/ tests/`, `pixi run -e dev typecheck`, `pixi run -e dev python scripts/docstring_style.py --check python/genvarloader` all exit 0.
- [ ] No `__all__`/`docs/source/api.md`/`skills/` changes: `git diff --stat` lists only the five files in File Structure plus the roadmap.
