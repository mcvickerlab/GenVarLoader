# Fix #242 — kernel-clip both backends for sub-query interval starts

**Issue:** [#242](https://github.com/mcvickerlab/GenVarLoader/issues/242) — Tracks + `max_jitter>0`: silent wrong track output (numba) / panic (rust) from interval/query coordinate mismatch.

**Date:** 2026-06-24

## Problem

Reading a dataset with **tracks + non-zero `max_jitter`** produces wrong track output
(numba, silently) or crashes (rust). Root cause is a coordinate mismatch between the
write path and the read path:

- **Write** (`_dataset/_write.py::_prep_bed`, line ~560): with `max_jitter>0`, stored
  regions and their clipped BigWig intervals live in the **jitter-expanded** window
  `[chromStart - max_jitter, chromEnd + max_jitter]` (saved to `regions.npy`). A stored
  interval start can be as low as `chromStart - max_jitter`.
- **Read** (`_dataset/_open.py::_build_indexer` → `_full_regions`): `_full_regions` is built
  from `input_regions.arrow` = the **original** bed. At getitem
  (`_dataset/_query.py:166-172`) jitter slides the window: `query_start = chromStart + jitter_off`,
  length stays original. That `query_start` becomes `starts[:]` for `intervals_to_tracks`.

A left-edge stored interval therefore has `start = itv.start - query_start < 0`, violating the
contract `itv.start >= query_start` that both kernels assume:

- **numba** (`_dataset/_intervals.py:71-81`): `_out[start:end] = value` with negative `start`
  is a numpy from-the-end index → left-edge values silently painted into the wrong positions
  (or dropped). A non-overlapping left interval can even paint a value into a window it does
  not touch.
- **rust** (`src/intervals.rs:55`): `debug_assert!(start >= 0, ...)` panics in debug builds;
  release builds wrap `start` to a huge `usize` and bounds-panic in ndarray.

This predates the Rust migration (reproduces on the numba backend and on `master`).

## Key decision

The kernel contract (`itv.start >= query_start`) **can only hold when `max_jitter == 0`**.
Under jitter the read window slides *per-read*, but intervals are clipped *once* at write to a
*fixed* expanded window — no single write-time clip satisfies a moving query origin. Therefore:

- **Rejected — suggested fix #2** (clip intervals to original `chromStart` at write): breaks
  left-jitter. Sliding left (`jitter_off < 0`) queries `[chromStart - jitter, ...)` where the
  signal would now be clipped away → missing left-edge track data.
- **Rejected — suggested fix #1** (query from expanded `regions.npy`): not a localized change.
  `query_start` sets both the coordinate origin *and*, via `out_offsets`, the output window.
  Setting `query_start = chromStart - max_jitter` with the original output length paints the
  *wrong (left-shifted)* window; making it correct requires reconstructing over the full
  expanded window then slicing — a redesign of track length/jitter semantics.
- **Chosen — kernel clip on both backends.** Clipping `s = max(start, 0)`, `e = min(end, length)`,
  painting only if `e > s`, is the mathematically correct painting of any interval onto any
  query window, **and** it preserves jitter: the expanded stored window guarantees data for any
  `|jitter_off| <= max_jitter`. The right edge is already correct (output right edge
  `<= chromEnd + max_jitter`). This is not defense-in-depth — it is the complete, correct,
  jitter-preserving fix.

## Components

### 1. numba kernel — `python/genvarloader/_dataset/_intervals.py` (`_intervals_to_tracks_numba`, ~line 71)

Replace the per-interval paint body:

```python
start = itv_starts[interval] - query_start
end = itv_ends[interval] - query_start
value = itv_values[interval]
if start >= length:
    # intervals sorted by start; all remaining are out of range
    break
s = max(start, 0)
e = min(end, length)
if e > s:
    _out[s:e] = value
```

Update the docstring: drop the "no intervals start before query start" assumption; document
that negative starts are clipped to the query window. Keep the sorted / non-overlapping
assumptions and the `break` (cannot break only matters under `prange` overlap, unchanged here).

### 2. rust kernel — `src/intervals.rs` (`intervals_to_tracks`, lines 46-66)

Delete `debug_assert!(start >= 0, "itv.start must be >= query_start per contract")`. Replace
the paint block:

```rust
let start = itv_starts[interval] as i64 - query_start;
let end = itv_ends[interval] as i64 - query_start;
let value = itv_values[interval];
if start >= length {
    break;
}
let s = start.max(0);
let e = end.min(length);
if e > s {
    let a = out_s + s as usize;
    let b = out_s + e as usize;
    out.slice_mut(ndarray::s![a..b]).fill(value);
}
```

Update the contract comment to describe clipping instead of the `start >= 0` guarantee. The
existing rust unit tests in `#[cfg(test)] mod tests` stay; add cases mirroring the oracle below.

### 3. Oracle test — `tests/unit/dataset/test_intervals_kernel.py`

Hand-computed cases run through the dispatch on **both** backends (parametrize over
`GVL_BACKEND` / the dispatch registry). At minimum:

- left-overlap: interval `[96, 114)` value `5.0`, `query_start=100`, `length=10` → `[5.0]*10`
- partial left-overlap: interval `[8, 13)` value `5.0`, `query_start=10`, `length=5` → `[5,5,5,0,0]`
- non-overlapping left (end past but start left): interval `[4, 8)` value `5.0`, `query_start=10`,
  `length=5` → all zeros
- fully left (`end <= query_start`): interval `[2, 6)`, `query_start=10` → all zeros

### 4. Parity strategy — `tests/parity/strategies.py` (`intervals_to_tracks_inputs`)

Widen so the first interval start may be `< qstart` (a negative offset relative to the query),
keeping intervals sorted and non-overlapping. This makes numba↔rust parity cover the
contract-violating inputs that previously could not be generated (current code uses
`cur = qstart + draw(0..10)`, i.e. first start always `>= qstart`).

### 5. Re-enable / verify the failing tests

The four tests named in the issue should pass unchanged after the fix:

- `tests/unit/dataset/test_output_bytes_per_instance.py::test_haplotypes_plus_tracks_exact`
- `tests/unit/dataset/test_output_bytes_per_instance.py::test_reference_plus_tracks_exact`
- `tests/integration/dataset/test_dummy_dataset_insertion_fill.py::test_end_to_end_set_insertion_fill`
- `tests/integration/dataset/test_dummy_dataset_insertion_fill.py::test_dummy_dataset_with_default_insertion_fill_does_not_crash`

If any of these are currently skipped/xfail-gated, remove the gate.

### 6. Roadmap — `docs/roadmaps/rust-migration.md`

Per CLAUDE.md, update the migration roadmap: record that the deferred `intervals_to_tracks`
contract handling is resolved and that `max_jitter>0` track parity is no longer deferred.

## Out of scope

- No write-path or read-path coordinate changes (fixes #1/#2 rejected above).
- Public API unchanged → no `skills/genvarloader/SKILL.md` update needed.
- Right-edge / deletion-driven track length behavior (covered by the existing chromEnd-floor
  fix); unchanged here.

## Validation (TDD order)

1. Oracle test (§3) → RED on at least one backend.
2. numba fix (§1).
3. rust fix (§2) + rust unit cases.
4. Widen parity strategy (§4).
5. Run: oracle + `tests/parity` + `pixi run -e dev cargo test` + the four affected suites,
   then the full tree (`pixi run -e dev pytest tests -q`) since shared kernel code changed.
6. Roadmap update (§6).

Both backends must produce byte-identical output, validated by the widened parity harness and
the hand-computed oracle.
