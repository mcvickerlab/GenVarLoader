# Target 5 — tracks-only ndarray slicing optimization

**Date:** 2026-06-25
**Workstream:** Phase 5, optimization round 2, Target 5 (rust-only, byte-identical).
**Branch:** `opt/target-5-intervals-slice` off `rust-migration`.
**Roadmap:** `docs/roadmaps/rust-migration.md` — Phase 5 ⬜, "Optimization targets — round 2".
**Handoff:** `docs/handoffs/2026-06-25-phase5-getitem-optimization.md` (Target 5 section).

## Problem

`intervals_to_tracks` (`src/intervals.rs`) is the kernel behind the cheapest read
path (tracks-only, ~1.1–1.7 ms/batch). On that path Rust runs at **0.63× numba**
— the single read path where Rust is clearly slower. `perf` flat self-time
attributes ~20.5% of the kernel to ndarray slice machinery:
`ndarray::slice_mut` (11%) + `ndarray::do_slice` (9.5%), all from constructing a
`SliceInfo` per painted interval in:

```rust
out.slice_mut(ndarray::s![a..b]).fill(value);
```

numba compiles the equivalent `out[a:b] = value` to a direct memset and pays none
of this. Because tracks-only does no sequence work, this fixed per-interval cost
dominates with nothing to amortize it against.

## Goal

Close the deficit so Rust is **≥ 1.0× numba** on tracks-only, while keeping the
output **byte-identical** to the numba oracle. The kernel is shared by the
combined **tracks** (seqs + read-depth) path, which improves with it.

## Scope

- **In:** `src/intervals.rs` — the `intervals_to_tracks` body, and (only if the
  perf fallback lands) one added cargo test.
- **Out:** No Python changes. No FFI-signature changes. No oracle change. No
  changes to `out.fill(0.0)` semantics. No overlap with Targets 6/7 (they touch
  `intervals.rs` too, but Target 5 merges first and they rebase onto it).

## Design

The `out` buffer is freshly allocated and contiguous, so we can address it as a
raw `&mut [f32]` and drop the per-interval `SliceInfo`.

1. **Hoist the slice once**, at the top of the function, after the zero prelude:
   ```rust
   let out_slice = out.as_slice_mut().unwrap();
   ```
   `.unwrap()` is intentional: a non-contiguous `out` is an invariant violation,
   not a recoverable case, and should fail loud.

2. **Zero prelude on the raw slice:**
   ```rust
   out_slice.fill(0.0);
   ```
   **Keep the zero prelude.** tracks-only depends on it — gaps between intervals
   must read 0. This is unlike the fully-overwritten sequence buffers whose
   zero-init was skipped in commit `1b3e355`; that optimization does not apply
   here.

3. **Per-interval write on the raw slice** (default, safe form):
   ```rust
   let a = out_s + s as usize;
   let b = out_s + e as usize;
   out_slice[a..b].fill(value);
   ```
   This keeps a single range bounds-check but removes `SliceInfo` construction —
   the proven cost.

All surrounding arithmetic and control flow is **unchanged**:
- `start = itv_starts[i] - query_start`, `end = itv_ends[i] - query_start` in i64.
- `break` when `start >= length` (intervals sorted by start).
- `s = start.max(0)`, `e = end.min(length)`; write only when `e > s`.
- Per-query `itv_s == itv_e` → skip (out slice stays 0).

## Parity

Byte-identical by construction — same arithmetic, same write order, same values,
only a different way to address the contiguous buffer.

Gates (all must stay green):
- `pixi run -e dev cargo-test` — the 8 existing unit tests in `src/intervals.rs`
  pin the full contract (basic paint, empty intervals, end-clamp, break-on-
  start≥length, the three #242 jitter cases, multi-query disjoint). Refactor
  **under** them, untouched.
- `pixi run -e dev pytest tests/parity -q` (rust default) **and**
  `GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q` (oracle) — including
  the `intervals_to_tracks` hypothesis parity gate and the tracks dataset
  backstop that proves the kernel runs on the live `__getitem__` path.

No new test is required for the safe form (no new behavior). A SAFETY-proof test
is added **only if** the unsafe fallback (below) is needed.

## Perf gate and fallback

Build release first: `pixi run -e dev maturin develop --release`. Re-measure
tracks-only via `tests/benchmarks/test_e2e.py` — `_bench_indexing` uses
`benchmark.pedantic(iterations=10, rounds=50)`; compare the **min** rust ÷ min
numba (cleanest CPU-bound estimate), with `NUMBA_NUM_THREADS=1`.

- **≥ 1.0×** → done. Record the ratio in the roadmap round-2 re-measurement block.
- **< 1.0×** → escalate the inner write to elide the bounds-check:
  ```rust
  // SAFETY: a = out_s + s, b = out_s + e with 0 <= s <= e <= length and
  // out_s + length == out_e <= out_slice.len() (out_offsets is a valid CSR
  // layout over out_slice), so a..b is in bounds.
  unsafe { out_slice.get_unchecked_mut(a..b).fill(value); }
  ```
  Add one cargo test asserting the bounds invariant the SAFETY comment relies on,
  re-measure, then record.

The expected outcome is that the safe form clears the gate (the `SliceInfo`
construction, not the bounds-check, was the dominant cost); the unsafe form is a
contingency, not the plan.

## Definition of done

1. Refactored `intervals_to_tracks`, all existing cargo tests green untouched.
2. `cargo-test` + `pytest tests/parity` on **both** backends green.
3. Full tree on both backends (`pixi run -e dev pytest tests -q`, then
   `GVL_BACKEND=numba …`) — scoped runs skip `tests/unit/`.
4. `ruff check python/ tests/` + `ruff format python/ tests/` + `typecheck`
   clean (no Python changes expected, but run them).
5. tracks-only re-measured ≥ 1.0×; ratio recorded in
   `docs/roadmaps/rust-migration.md` with Target 5 ticked and the PR link set.
6. Parity-gated PR opened from `opt/target-5-intervals-slice`.
