# Fix #242 — intervals_to_tracks jitter clip — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `intervals_to_tracks` (numba + rust) correctly paint intervals whose start falls before the per-read jittered query window, fixing silent wrong track output (numba) and panics (rust) on datasets with tracks + `max_jitter>0`.

**Architecture:** Both kernels currently assume `itv.start >= query_start`. That contract is incompatible with `max_jitter>0` (intervals are clipped once to a fixed expanded window at write; the read window slides per-read). Fix: clip each interval to the query window — `s = max(start, 0)`, `e = min(end, length)`, paint only if `e > s` — in both backends. No write-path or read-path coordinate changes. This is the complete, jitter-preserving fix; the right edge is already correct (`output right edge <= chromEnd + max_jitter`).

**Tech Stack:** Python + numba (`python/genvarloader/_dataset/_intervals.py`), Rust + ndarray/PyO3 (`src/intervals.rs`), pixi for env/build, pytest + hypothesis parity harness, cargo for rust unit tests.

**Spec:** `docs/superpowers/specs/2026-06-24-issue-242-intervals-track-jitter-clip-design.md`

## Global Constraints

- Both backends MUST produce byte-identical output (validated by `tests/parity` + hand-computed oracle).
- The kernels are sequential-equivalent: keep the sorted-by-start `break` (interval `start >= length` ⇒ all remaining out of range).
- Do NOT change the write path (`_prep_bed`), the read path coordinate derivation (`_full_regions`, `_query.py`), or the public API. No `skills/genvarloader/SKILL.md` change.
- Editable maturin install: after editing `src/*.rs`, rebuild the extension with `pixi run -e dev maturin develop --release` before any Python test that exercises the rust backend. Rust-only logic is also covered by `pixi run -e dev cargo-test`.
- Per CLAUDE.md: any rust-touching task must read and update `docs/roadmaps/rust-migration.md`.
- Branch is `fix/issue-242-intervals-track-jitter-clip` (already created; spec committed there).

---

### Task 1: Clip negative interval starts in both kernels

**Files:**
- Modify: `python/genvarloader/_dataset/_intervals.py:11-81` (`_intervals_to_tracks_numba`)
- Modify: `src/intervals.rs:46-66` (paint loop) and `src/intervals.rs:71-195` (`mod tests`)
- Test: `tests/unit/dataset/test_intervals_kernel.py` (add cross-backend oracle cases)

**Interfaces:**
- Consumes: the existing registered kernel `intervals_to_tracks` (dispatch wrapper at `python/genvarloader/_dataset/_intervals.py:92`), forced per-backend via `GVL_BACKEND` env (`monkeypatch.setenv`).
- Produces: kernels that, for each interval, compute `start = itv_start - query_start`, `end = itv_end - query_start`, then paint `out[max(start,0) : min(end,length)] = value` (no-op when `min(end,length) <= max(start,0)`), and `break` when `start >= length`. Output dtype/layout unchanged (`out` written in place, float32).

- [ ] **Step 1: Write the failing cross-backend oracle test**

First add `import pytest` to the top of `tests/unit/dataset/test_intervals_kernel.py` (it currently imports only `numpy` and `intervals_to_tracks`); keep it with the other top-level imports to avoid ruff E402. Then append the helper and tests below:

```python
def _run(backend, monkeypatch, starts, itv_starts, itv_ends, itv_values, out_len):
    """Single query, one interval-slice; force `backend` and return the out buffer."""
    monkeypatch.setenv("GVL_BACKEND", backend)
    offset_idxs = np.array([0], dtype=np.intp)
    starts = np.array(starts, dtype=np.int32)
    itv_starts = np.array(itv_starts, dtype=np.int32)
    itv_ends = np.array(itv_ends, dtype=np.int32)
    itv_values = np.array(itv_values, dtype=np.float32)
    itv_offsets = np.array([0, len(itv_starts)], dtype=np.int64)
    out = np.empty(out_len, dtype=np.float32)
    out_offsets = np.array([0, out_len], dtype=np.int64)
    intervals_to_tracks(
        offset_idxs,
        starts,
        itv_starts,
        itv_ends,
        itv_values,
        itv_offsets,
        out,
        out_offsets,
    )
    return out


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_starts_before_query_full_cover(backend, monkeypatch):
    # issue #242: interval [96,114) value 5, query_start=100, length=10 -> all 5s
    out = _run(backend, monkeypatch, [100], [96], [114], [5.0], 10)
    np.testing.assert_array_equal(out, np.full(10, 5.0, dtype=np.float32))


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_starts_before_query_partial(backend, monkeypatch):
    # interval [8,13) value 5, query_start=10, length=5 -> [5,5,5,0,0]
    out = _run(backend, monkeypatch, [10], [8], [13], [5.0], 5)
    np.testing.assert_array_equal(
        out, np.array([5.0, 5.0, 5.0, 0.0, 0.0], dtype=np.float32)
    )


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_left_overlap_end_in_window(backend, monkeypatch):
    # interval [4,8) value 5, query_start=10, length=5 -> all zeros (no overlap)
    out = _run(backend, monkeypatch, [10], [4], [8], [5.0], 5)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))


@pytest.mark.parametrize("backend", ["numba", "rust"])
def test_interval_fully_left_of_query(backend, monkeypatch):
    # interval [2,6) ends at/below query_start=10 -> all zeros
    out = _run(backend, monkeypatch, [10], [2], [6], [5.0], 5)
    np.testing.assert_array_equal(out, np.zeros(5, dtype=np.float32))
```

- [ ] **Step 2: Run the oracle test, verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_intervals_kernel.py -v`
Expected: the four new `*before_query*` / `*left*` tests FAIL — numba paints wrong positions (negative-index wrap), and the `rust` params error/panic (debug_assert / bounds). Existing four tests still PASS.

- [ ] **Step 3: Fix the numba kernel**

In `python/genvarloader/_dataset/_intervals.py`, edit the docstring (remove the contract line) and the paint loop.

Remove this line from the `Assumptions:` block (lines ~23-26):
```python
    - no intervals start before query start
```

Replace the interval loop (lines ~71-81):
```python
        # if parallelized, a data race will occur if there are any overlapping intervals
        for interval in range(itv_s, itv_e):
            #! assumes itv.start >= query_start
            start = itv_starts[interval] - query_start
            end = itv_ends[interval] - query_start
            value = itv_values[interval]
            if start < length:
                _out[start:end] = value
            else:
                #! assumes intervals are sorted by start
                # cannot break if parallelized
                break
```
with:
```python
        # if parallelized, a data race will occur if there are any overlapping intervals
        for interval in range(itv_s, itv_e):
            start = itv_starts[interval] - query_start
            end = itv_ends[interval] - query_start
            value = itv_values[interval]
            if start >= length:
                #! assumes intervals are sorted by start
                # cannot break if parallelized
                break
            # Clip to the query window. Intervals may start before query_start
            # (jitter-expanded storage vs. the per-read query origin; see #242)
            # or end past it.
            s = max(start, 0)
            e = min(end, length)
            if e > s:
                _out[s:e] = value
```

- [ ] **Step 4: Run only the numba params, verify they pass**

Run: `pixi run -e dev pytest "tests/unit/dataset/test_intervals_kernel.py" -v -k numba`
Expected: all `[numba]` params PASS (the new ones now correct). `[rust]` params still fail — fixed next.

- [ ] **Step 5: Fix the rust kernel**

In `src/intervals.rs`, replace the paint loop (lines ~46-66):
```rust
        for interval in itv_s..itv_e {
            // start/end computed in i64 (avoids i32 overflow for large coords).
            let start = itv_starts[interval] as i64 - query_start;
            let end = itv_ends[interval] as i64 - query_start;
            let value = itv_values[interval];

            if start < length {
                // Replicate numpy slice semantics: clamp end, no-op if empty.
                // Contract guarantees start >= 0, so no negative-index wrap.
                debug_assert!(start >= 0, "itv.start must be >= query_start per contract");
                let clamped_end = end.min(length);
                if clamped_end > start {
                    let a = out_s + start as usize;
                    let b = out_s + clamped_end as usize;
                    out.slice_mut(ndarray::s![a..b]).fill(value);
                }
            } else {
                // start >= length: intervals are sorted, all remaining are
                // also out of range — break.
                break;
            }
        }
```
with:
```rust
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
                out.slice_mut(ndarray::s![a..b]).fill(value);
            }
        }
```

- [ ] **Step 6: Add rust unit tests mirroring the oracle**

In `src/intervals.rs`, inside `mod tests` (before the closing `}` at line ~195), add:
```rust
    /// #242: interval starts before query_start, fully covers the window.
    #[test]
    fn test_interval_starts_before_query_full_cover() {
        // query_start=100, interval [96,114) on length-10 out -> all 5.0
        let result = run(&[0], &[100], &[96], &[114], &[5.0], 10, &[0, 10]);
        assert_eq!(result, vec![5.0; 10]);
    }

    /// #242: partial left overlap -> clipped at 0.
    #[test]
    fn test_interval_starts_before_query_partial() {
        // query_start=10, interval [8,13) on length-5 out -> [5,5,5,0,0]
        let result = run(&[0], &[10], &[8], &[13], &[5.0], 5, &[0, 5]);
        assert_eq!(result, vec![5.0, 5.0, 5.0, 0.0, 0.0]);
    }

    /// #242: interval ends at/below query_start -> no paint.
    #[test]
    fn test_interval_fully_left_of_query() {
        let result = run(&[0], &[10], &[2], &[6], &[5.0], 5, &[0, 5]);
        assert_eq!(result, vec![0.0; 5]);
    }
```

- [ ] **Step 7: Run cargo unit tests, verify they pass**

Run: `pixi run -e dev cargo-test`
Expected: all `intervals` tests PASS (existing 5 + 3 new), no `debug_assert` panic.

- [ ] **Step 8: Rebuild the extension and run the full oracle test on both backends**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/unit/dataset/test_intervals_kernel.py tests/unit/dataset/test_intervals_dispatch.py -v
```
Expected: all tests PASS for both `[numba]` and `[rust]` params.

- [ ] **Step 9: Commit**

```bash
rtk git add python/genvarloader/_dataset/_intervals.py src/intervals.rs tests/unit/dataset/test_intervals_kernel.py
rtk git commit -m "$(cat <<'EOF'
fix(intervals): clip sub-query interval starts in both kernels (#242)

Tracks + max_jitter>0 store intervals against a jitter-expanded window,
but the read path queries the original chromStart, so left-edge intervals
have start < query_start. numba wrapped the negative index (silent wrong
output); rust hit debug_assert / bounds panic. Both kernels now clip to
the query window (s=max(start,0), e=min(end,length)), which is correct and
jitter-preserving. Adds cross-backend oracle tests + rust unit tests.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Widen the parity strategy to cover negative interval starts

**Files:**
- Modify: `tests/parity/strategies.py:9-65` (`intervals_to_tracks_inputs`)
- Test: `tests/parity/test_intervals_to_tracks_parity.py` (unchanged; runs the widened strategy)

**Interfaces:**
- Consumes: the fixed kernels from Task 1.
- Produces: a hypothesis strategy that can generate a first interval start `< qstart` (negative offset relative to the query), still sorted and non-overlapping, so numba↔rust parity exercises the previously-impossible contract-violating inputs.

- [ ] **Step 1: Widen the first-interval start**

In `tests/parity/strategies.py`, update the docstring and the first-start draw.

Change the docstring line (line ~14):
```python
    intervals are sorted, non-overlapping, and start at >= the query's start.
```
to:
```python
    intervals are sorted and non-overlapping; the first interval may start
    before the query's start (negative relative start) to cover #242.
```

Change the first-start initialization (lines ~34-36):
```python
        cur = qstart + draw(
            st.integers(min_value=0, max_value=10)
        )  # first start >= qstart
```
to:
```python
        # first start may be below qstart (negative relative start; #242) or above
        cur = qstart + draw(st.integers(min_value=-10, max_value=10))
```

- [ ] **Step 2: Run the parity suite, verify it passes**

Run: `pixi run -e dev pytest tests/parity/test_intervals_to_tracks_parity.py -v`
Expected: PASS (numba and rust byte-identical, now including negative-start cases). If a falsifying example surfaces, it indicates a residual kernel discrepancy — return to Task 1.

- [ ] **Step 3: Run the dataset-level parity backstop**

Run: `pixi run -e dev pytest tests/parity -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/parity/strategies.py
rtk git commit -m "$(cat <<'EOF'
test(parity): cover sub-query interval starts in intervals_to_tracks (#242)

Widen the hypothesis strategy so the first interval may start before the
query (negative relative start), exercising the case that previously
violated the kernel contract. numba↔rust parity holds.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Verify the affected end-to-end tests and full suite gates

**Files:**
- No code changes. Verification only. (The four failing tests use runtime `pytest.skip` guards, not #242 gates, so no gate removal is needed — they should now pass.)

**Interfaces:**
- Consumes: fixed kernels (Task 1) and widened parity (Task 2).
- Produces: green status on the four issue-named tests and the full tree, confirming the read path is correct for tracks + `max_jitter>0`.

- [ ] **Step 1: Run the four tests named in the issue**

Run:
```bash
pixi run -e dev pytest \
  tests/unit/dataset/test_output_bytes_per_instance.py::test_haplotypes_plus_tracks_exact \
  tests/unit/dataset/test_output_bytes_per_instance.py::test_reference_plus_tracks_exact \
  tests/integration/dataset/test_dummy_dataset_insertion_fill.py::test_end_to_end_set_insertion_fill \
  tests/integration/dataset/test_dummy_dataset_insertion_fill.py::test_dummy_dataset_with_default_insertion_fill_does_not_crash \
  -v
```
Expected: all four PASS (previously failing per the issue).

- [ ] **Step 2: Run the full pytest tree + cargo (shared kernel changed)**

Run: `pixi run -e dev test`
Expected: full `pytest tests` + `cargo test --release` PASS. (Per CLAUDE.md, run the whole tree because shared kernel code changed.)

- [ ] **Step 3: Lint and typecheck**

Run:
```bash
pixi run -e dev ruff format python/ tests/
pixi run -e dev ruff check python/ tests/
pixi run -e dev typecheck
```
Expected: format clean (no diffs to commit, or commit them), check passes, pyrefly passes.

- [ ] **Step 4: Commit any formatting changes (only if `ruff format` modified files)**

```bash
rtk git add -A
rtk git commit -m "style: ruff format after #242 fix"
```
(Skip if there is nothing to commit.)

---

### Task 4: Update the Rust migration roadmap

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (Phase 0 section, ~lines 158-183)

**Interfaces:**
- Consumes: the completed fix.
- Produces: roadmap reflecting that `intervals_to_tracks` contract handling for sub-query starts is resolved and `max_jitter>0` track parity is no longer deferred.

- [ ] **Step 1: Update the parity gate description**

In `docs/roadmaps/rust-migration.md`, update the harness bullet (lines ~160-161). Change:
```
      property generator. Per-kernel gate (`test_intervals_to_tracks_parity`, 100
      contract-valid examples) + a MEANINGFUL dataset-level read-path backstop
```
to:
```
      property generator. Per-kernel gate (`test_intervals_to_tracks_parity`, 100
      examples incl. sub-query interval starts — #242 fixed both backends to clip
      to the query window) + a MEANINGFUL dataset-level read-path backstop
```

- [ ] **Step 2: Add a note to the Phase 0 checkpoint**

In the Phase 0 **Checkpoint** paragraph (lines ~181-183), append one sentence:
```
The `intervals_to_tracks` sub-query-start contract gap (max_jitter>0 tracks,
#242) is resolved: both kernels clip to the query window and parity covers it.
```

- [ ] **Step 3: Commit**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): record #242 intervals_to_tracks contract fix"
```

---

## Notes for the implementer

- `GVL_BACKEND` env forces a single backend across all kernels (`python/genvarloader/_dispatch.py:39`). Tests set it via `monkeypatch.setenv` so it's reverted automatically.
- The dispatch wrapper coerces input dtypes (`_intervals.py:109-115`); pass arrays in the canonical dtypes shown in Task 1's `_run` helper to avoid surprises.
- The rust `run` helper in `mod tests` (`src/intervals.rs:76`) signature is `run(offset_idxs, starts, itv_starts, itv_ends, itv_values, out_len, out_offsets)` — note `out_len` precedes `out_offsets`.
- If `maturin develop` is slow/unavailable, the rust logic is still fully covered by `cargo-test`; the Python `[rust]` params require a rebuilt extension to be meaningful.
