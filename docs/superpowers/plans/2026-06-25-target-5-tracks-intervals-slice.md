# Target 5 — tracks-only intervals slice optimization — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drop per-interval `SliceInfo` construction from `intervals_to_tracks` so the tracks-only read path runs ≥ 1.0× numba, byte-identically.

**Architecture:** Address the contiguous `out` buffer as a raw `&mut [f32]` via one hoisted `as_slice_mut()`, replacing `out.slice_mut(s![a..b]).fill(value)` with `out_slice[a..b].fill(value)`. Pure-Rust refactor under the existing cargo tests; same arithmetic, same write order, same values. Unsafe `get_unchecked_mut` is a measured contingency only if the safe form misses the perf gate.

**Tech Stack:** Rust (`ndarray`, PyO3/maturin), Python (pytest, pytest-benchmark, numba oracle), pixi (`-e dev`).

**Spec:** `docs/superpowers/specs/2026-06-25-target-5-tracks-intervals-slice-design.md`

## Global Constraints

- Branch: `opt/target-5-intervals-slice` off `rust-migration` (already created and checked out).
- **Byte-identical** to the numba oracle — non-negotiable landing gate.
- **Only** `src/intervals.rs` changes (the kernel body; one added test only if the unsafe fallback lands). No Python, no FFI-signature, no oracle changes.
- **Keep the `out.fill(0.0)` zero prelude** — tracks-only relies on inter-interval gaps reading 0.
- The 8 existing cargo tests in `src/intervals.rs` must stay green **untouched**.
- Measure with `NUMBA_NUM_THREADS=1`; compare the **min** of `pedantic(iterations=10, rounds=50)`.
- Release build before any perf measurement: `pixi run -e dev maturin develop --release`.
- HPC: dataset tests need `--basetemp=$(pwd)/.pytest_tmp` (cross-device `os.link` fails with Errno 18 otherwise).
- Per CLAUDE.md, prefix shell commands with `rtk`.

---

### Task 1: Establish green baseline + record starting ratio

**Files:**
- Read only: `src/intervals.rs`

**Interfaces:**
- Consumes: nothing.
- Produces: a recorded baseline tracks-only `min rust ÷ min numba` ratio (expected ≈ 0.63×) used to confirm improvement in Task 4.

- [ ] **Step 1: Confirm clean tree on the right branch**

Run: `rtk git status && rtk git branch --show-current`
Expected: branch `opt/target-5-intervals-slice`, only the untracked handoff + the committed spec/plan present.

- [ ] **Step 2: Release build**

Run: `pixi run -e dev maturin develop --release`
Expected: builds `genvarloader.abi3.so` with no errors.

- [ ] **Step 3: Run the cargo unit tests (baseline green)**

Run: `pixi run -e dev cargo-test`
Expected: PASS, including the 8 `intervals_to_tracks` tests (`test_basic_paint`, `test_empty_intervals`, `test_end_clamp`, `test_break_on_start_ge_length`, `test_interval_starts_before_query_full_cover`, `test_interval_starts_before_query_partial`, `test_interval_fully_left_of_query`, `test_multi_query_disjoint`).

- [ ] **Step 4: Capture the baseline tracks-only ratio**

Run: `NUMBA_NUM_THREADS=1 pixi run -e dev pytest tests/benchmarks/test_e2e.py -k tracks --basetemp=$(pwd)/.pytest_tmp -q`
Expected: completes; note the tracks-only min rust and min numba times. Record the ratio (≈ 0.63×) in scratch — this is the before-number for the roadmap.

No commit (measurement only).

---

### Task 2: Refactor `intervals_to_tracks` to a raw contiguous slice

**Files:**
- Modify: `src/intervals.rs:23-69` (the function body)

**Interfaces:**
- Consumes: the existing `intervals_to_tracks` signature — unchanged.
- Produces: identical output buffer; no signature change. Later tasks rely on the public signature staying exactly as-is.

- [ ] **Step 1: Confirm the tests already pin the contract (no new test needed)**

The 8 cargo tests in `src/intervals.rs:72-219` exhaust the behavior (paint, empty, end-clamp, break, the three #242 jitter cases, multi-query). This is a byte-identical refactor, so they ARE the failing/passing gate — do not add or edit them.

- [ ] **Step 2: Apply the refactor**

Replace the body from the zero-prelude through the inner write. Change `out.fill(0.0)` and the per-interval `out.slice_mut(...)` to operate on a hoisted raw slice:

```rust
    // Step 1: zero the whole output buffer, exactly like `out[:] = 0.0`.
    // The out buffer is freshly allocated and contiguous; address it as a raw
    // &mut [f32] so per-interval writes avoid ndarray SliceInfo construction.
    let out_slice = out.as_slice_mut().unwrap();
    out_slice.fill(0.0);

    let n_queries = starts.len();

    for query in 0..n_queries {
        let idx = offset_idxs[query] as usize;
        let itv_s = itv_offsets[idx] as usize;
        let itv_e = itv_offsets[idx + 1] as usize;

        if itv_s == itv_e {
            // No intervals for this query — out slice stays 0.
            continue;
        }

        let out_s = out_offsets[query] as usize;
        let out_e = out_offsets[query + 1] as usize;
        // length as i64 to do signed arithmetic below.
        let length = (out_e - out_s) as i64;
        let query_start = starts[query] as i64;

        for interval in itv_s..itv_e {
            // start/end computed in i64 (avoids i32 overflow for large coords).
            let start = itv_starts[interval] as i64 - query_start;
            let end = itv_ends[interval] as i64 - query_start;
            let value = itv_values[interval];

            if start >= length {
                // start >= length: intervals are sorted, all remaining are
                // also out of range — break.
                break;
            }
            // Clip to the query window. Intervals may start before query_start
            // (jitter-expanded interval storage vs. the per-read query origin;
            // see issue #242) or end past it. No negative-index wrap.
            let s = start.max(0);
            let e = end.min(length);
            if e > s {
                let a = out_s + s as usize;
                let b = out_s + e as usize;
                out_slice[a..b].fill(value);
            }
        }
    }
```

Note: `out` is now bound only to produce `out_slice`; the `mut out: ArrayViewMut1<f32>` parameter stays as-is. The doc comment at `src/intervals.rs:3-15` remains accurate (semantics unchanged) — leave it.

- [ ] **Step 3: Run the cargo tests (must stay green, untouched)**

Run: `pixi run -e dev cargo-test`
Expected: PASS — all 8 `intervals_to_tracks` tests green, identical to Task 1 Step 3.

- [ ] **Step 4: Commit**

```bash
rtk git add src/intervals.rs
rtk git commit -m "perf(intervals): paint tracks via raw contiguous slice

Hoist out.as_slice_mut() once and write out_slice[a..b].fill(value)
per interval, dropping per-interval ndarray SliceInfo construction
(~20.5% self-time on the tracks-only read path). Byte-identical:
same arithmetic, same write order, zero prelude retained.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Parity gate on both backends

**Files:**
- Read only: `tests/parity/`

**Interfaces:**
- Consumes: the refactored kernel from Task 2.
- Produces: proof of byte-identical output vs the numba oracle on the live `__getitem__` path.

- [ ] **Step 1: Rebuild release (Task 2 changed Rust)**

Run: `pixi run -e dev maturin develop --release`
Expected: builds cleanly.

- [ ] **Step 2: Parity — rust default backend**

Run: `pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS, including the `intervals_to_tracks` hypothesis parity gate and the tracks dataset backstop (`tests/parity/test_dataset_parity.py`) that spies on the kernel to prove it runs.

- [ ] **Step 3: Parity — numba oracle backend**

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (byte-identical to Step 2).

No commit (verification only). If either fails, the refactor diverged — return to Task 2; do not proceed.

---

### Task 4: Perf gate — re-measure, escalate to unsafe only if short

**Files:**
- Modify (conditional): `src/intervals.rs` inner write + one added test, **only if** the safe form misses ≥ 1.0×.

**Interfaces:**
- Consumes: the refactored kernel.
- Produces: the recorded post-change tracks-only ratio for the roadmap.

- [ ] **Step 1: Re-measure tracks-only**

Run: `NUMBA_NUM_THREADS=1 pixi run -e dev pytest tests/benchmarks/test_e2e.py -k tracks --basetemp=$(pwd)/.pytest_tmp -q`
Expected: completes. Compute `min rust ÷ min numba`.

- [ ] **Step 2: Branch on the result**

- **If ≥ 1.0×** → gate cleared. Skip Steps 3–5; record the ratio for Task 5.
- **If < 1.0×** → proceed to Step 3 (unsafe fallback).

- [ ] **Step 3 (conditional): Escalate the inner write to `get_unchecked_mut`**

In `src/intervals.rs`, replace the safe inner write with:

```rust
            if e > s {
                let a = out_s + s as usize;
                let b = out_s + e as usize;
                // SAFETY: 0 <= s <= e <= length, and out_s + length == out_e,
                // where out_offsets is a valid CSR layout over out_slice
                // (out_e <= out_slice.len()). Hence out_s <= a <= b <= out_e
                // <= out_slice.len(), so a..b is in bounds.
                unsafe { out_slice.get_unchecked_mut(a..b).fill(value); }
            }
```

- [ ] **Step 4 (conditional): Add a test pinning the SAFETY invariant**

Append to the `tests` module in `src/intervals.rs`:

```rust
    /// SAFETY invariant: a painted interval never writes past its query's
    /// out slice end (b <= out_e), even when the interval end far exceeds it.
    #[test]
    fn test_paint_never_exceeds_query_slice() {
        // Two adjacent queries; query 0's interval ends at 1000 but its slice
        // is out[0..5]; query 1's slice (out[5..10]) must remain untouched
        // except by its own interval.
        let result = run(
            &[0, 1],
            &[0, 0],
            &[2, 0],
            &[1000, 1],
            &[7.0, 9.0],
            &[0, 1, 2],
            10,
            &[0, 5, 10],
        );
        // query 0: out[2..5]=7.0 (clamped at 5, no spill into query 1)
        // query 1: out[5..6]=9.0
        assert_eq!(
            result,
            vec![0.0, 0.0, 7.0, 7.0, 7.0, 9.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
```

- [ ] **Step 5 (conditional): Rebuild, retest, re-measure**

Run: `pixi run -e dev maturin develop --release && pixi run -e dev cargo-test`
Expected: PASS (9 tests now).
Then re-run Step 1's benchmark; confirm ≥ 1.0×.

- [ ] **Step 6 (conditional): Commit the fallback**

```bash
rtk git add src/intervals.rs
rtk git commit -m "perf(intervals): elide bounds-check on per-interval paint

Safe slice indexing fell short of numba on tracks-only; use
get_unchecked_mut with a proven SAFETY invariant (a..b within the
query's CSR out slice) plus a test pinning no cross-query spill.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: Full-tree gate, lint, roadmap update, PR

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (round-2 block: tick Target 5, record ratio, set PR link)

**Interfaces:**
- Consumes: the green kernel + recorded ratio.
- Produces: the landed, documented workstream + PR.

- [ ] **Step 1: Full tree — rust default**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (covers `tests/unit/` which scoped runs skip).

- [ ] **Step 2: Full tree — numba oracle**

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

- [ ] **Step 3: Lint / format / typecheck**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck`
Expected: clean (no Python changed, but the project gates on it).

- [ ] **Step 4: Update the roadmap**

In `docs/roadmaps/rust-migration.md`, in the round-2 optimization block: tick Target 5, set its phase marker, and record the re-measured tracks-only ratio (before ≈ 0.63× → after, from Task 4 Step 1) plus whether the safe or unsafe form landed. Add the PR link once opened (Step 6).

- [ ] **Step 5: Commit the roadmap**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): tick Target 5, record tracks-only ratio

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 6: Push and open the parity-gated PR**

```bash
rtk git push -u origin opt/target-5-intervals-slice
rtk gh pr create --base rust-migration --title "perf(intervals): tracks-only raw-slice paint (Target 5)" --body "$(cat <<'EOF'
Closes Target 5 of the Phase 5 read-path optimization (handoff
docs/handoffs/2026-06-25-phase5-getitem-optimization.md).

Byte-identical refactor of intervals_to_tracks to drop per-interval
ndarray SliceInfo construction. tracks-only min rust ÷ min numba:
<BEFORE 0.63x> → <AFTER>.

Parity: green on both backends (rust default + GVL_BACKEND=numba),
incl. the intervals_to_tracks hypothesis gate and tracks dataset
backstop. Full tree green both backends.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

Then edit the roadmap PR-link placeholder (Step 4) to the real URL and amend Step 5's commit, or push a follow-up.

---

## Self-Review

**Spec coverage:**
- Problem / SliceInfo cost → Task 2 (the refactor). ✓
- Keep zero prelude → Task 2 Step 2 comment + Global Constraints. ✓
- Byte-identical parity, both backends, hypothesis gate + dataset backstop → Task 3. ✓
- Existing 8 cargo tests stay green untouched → Task 1 Step 3, Task 2 Step 3. ✓
- Perf gate ≥ 1.0×, min-of-pedantic, NUMBA_NUM_THREADS=1 → Task 1 Step 4, Task 4. ✓
- Unsafe fallback with SAFETY proof + added test → Task 4 Steps 3–6. ✓
- Full tree both backends + lint/format/typecheck → Task 5 Steps 1–3. ✓
- Roadmap update (tick, ratio, PR link) → Task 5 Steps 4–5. ✓
- Branch off rust-migration, parity-gated PR → Global Constraints, Task 5 Step 6. ✓

**Placeholder scan:** `<BEFORE 0.63x>` / `<AFTER>` in the PR body and roadmap are intentional runtime-measured values, filled from Task 4's measurement — not unspecified work. No "TBD"/"add error handling"/"write tests for the above" left.

**Type consistency:** `intervals_to_tracks` signature untouched throughout; the test helper `run(...)` argument order in Task 4's added test matches the existing helper at `src/intervals.rs:77-100` (offset_idxs, starts, itv_starts, itv_ends, itv_values, itv_offsets, out_len, out_offsets). `out_slice` / `a` / `b` names consistent across Task 2 and Task 4.
