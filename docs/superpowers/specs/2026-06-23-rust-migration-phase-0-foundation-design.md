# Design: Rust migration Phase 0 — Foundation & differential-test harness

**Date:** 2026-06-23
**Status:** approved (brainstorming)
**Roadmap:** `docs/roadmaps/rust-migration.md` (Phase 0)
**Related:** [[project_seqpro_core_shared_crate]] (Phase 1, shipped), [[feedback_yagni_test_builders]],
[[feedback_macos_profiling_handoff]], [[project_dataloader_bench]], [[project_vcfixture_migration]]

---

## Goal

Stand up the reusable parity machinery and the perf baselines that gate every later phase of the
Rust migration, and **prove the whole strangler loop works end-to-end on one real kernel**. This is
foundation only — no mass kernel migration (that is Phase 2). When Phase 0 lands, every subsequent
phase can migrate a kernel by: implement in Rust → wrap in `ffi/` → register → add a per-kernel
parity test → flip the kernel's default → delete the numba impl, with byte-identical parity proven
automatically.

### Why these decisions (brainstorming outcomes)

- **Phase 0 before more kernels.** Phase 1 shipped with *ad-hoc* parity (a hand-written `to_padded`
  test). Paying down the harness now makes every later phase's parity + perf gate systematic.
- **Harness at both layers.** Per-kernel tight gate (localizes failures, shrinks fast) *plus* a thin
  dataset-level integration backstop (catches inter-kernel wiring bugs). Matches the contract's
  "a unit lands only when parity holds" while still covering composition.
- **Central backend registry** for the Rust-vs-numba switch (not 34 ad-hoc env vars): one surface,
  one Phase-5 removal point, scales to all kernels.
- **The dispatch unit is a Python-entry kernel, not every njit function.** Some numba kernels are
  *leaves* called only from inside other `@njit` functions (e.g. `padded_slice`, called from
  `_reference._fetch_row`/`_get_reference_row`). You cannot route a Python-level `dispatch.get()` out
  from inside `@njit` without objmode destroying performance, so leaf kernels **cannot be dispatched
  individually** — they migrate as part of their Python-entry caller's subtree (the Rust caller calls
  the Rust leaf directly). Only kernels invoked from Python get a registry entry. This is why the
  proof-point is `splits_sum_le_value` (Python call site) and not `padded_slice` (njit-internal leaf).
- **ffi/ seam only; grow domain modules lazily.** Empty `genotypes/`, `variants/`, … modules were
  never load-bearing; creating them now is speculative scaffolding ([[feedback_yagni_test_builders]]).
- **Capture all four baselines now** so every later gate has its number ready.

---

## Components

### 1. Backend dispatch registry — `python/genvarloader/_dispatch.py`

The single switch surface for the strangler window.

- A module-level registry mapping `name: str → Entry`, where
  `Entry = {numba_fn: Callable, rust_fn: Callable, default: "numba" | "rust"}`.
- `register(name, *, numba, rust, default="numba")` — called at import time by each migrated kernel's
  module. New entries default to `numba` until their parity test is green.
- `get(name) -> Callable` — returns the active callable. Resolution order:
  1. env `GVL_BACKEND` (`numba` | `rust`) — **global override**, used by CI parity sweeps to force
     all kernels onto one backend.
  2. otherwise the entry's **per-kernel** `default`.
  This gives independent per-kernel flips (change one entry's `default`) plus a force-all sweep.
- `backends(name) -> tuple[Callable, Callable]` — returns `(numba_fn, rust_fn)`; consumed by the
  per-kernel harness so it can run both regardless of the active default.
- Production call sites call `dispatch.get("splits_sum_le_value")(...)` instead of importing the
  kernel directly.
- **Phase 5** deletes this module and inlines the Rust calls — one place, one diff.

Invalid `GVL_BACKEND` values raise a clear error. Unknown `name` in `get`/`backends` raises `KeyError`
with the registered names listed.

### 2. ffi/ seam — `src/ffi/`

- New `src/ffi/mod.rs` becomes the **only** place `#[pyfunction]` wrappers live. Core algorithm code
  lives in lazily-created domain modules and is plain Rust (`ndarray`/`rayon`, no PyO3).
- Phase 0 grows **exactly one** domain module: `src/utils.rs`, home of the `splits_sum_le_value`
  proof-point kernel. It is created because it now holds real code — not as a speculative stub.
- `src/lib.rs` registers the new pyfunction(s) in the pymodule alongside the existing bigwig/tables
  exports. `bigwig.rs` and `tables.rs` are left untouched (no churn).
- Future phases add their domain module (`genotypes/`, `variants/`, …) + their `ffi/` wrappers the
  same way, on demand.

### 3. Differential-test harness — `tests/parity/`

Two layers, both built in Phase 0:

**Per-kernel (tight gate).**
- `tests/parity/strategies.py` — hypothesis strategies producing each kernel's exact numpy inputs,
  respecting structural invariants (sorted/monotonic offsets, valid dtypes, ploidy/shape constraints).
  Phase 0 ships the `splits_sum_le_value` strategy as the template others copy.
- `tests/parity/_harness.py` — core helper `assert_kernel_parity(name, *inputs)` that pulls both
  callables via `dispatch.backends(name)`, runs both on the same inputs, and asserts **exact**
  equality: same dtype, shape, and bytes (`np.testing.assert_array_equal` plus an explicit
  `.dtype` check; for float kernels, exact bit-equality is the contract unless a kernel is explicitly
  documented otherwise).
- One parametrized `test_<kernel>_parity` per migrated kernel. Phase 0 ships
  `test_splits_sum_le_value_parity`.

**Dataset-level (integration backstop).**
- `tests/parity/test_dataset_parity.py` — builds a small dataset via `vcfixture`, exercises a
  representative `Dataset.__getitem__` (and a `write()` round-trip where relevant) once under
  `GVL_BACKEND=numba` and once under `GVL_BACKEND=rust`, and asserts equal outputs.
- Purpose: catch wiring/composition bugs a per-kernel test cannot see. Kept thin in Phase 0 (the only
  Rust-backed kernel is `splits_sum_le_value`); each later phase extends coverage as it migrates
  kernels.

**Markers.** Parity tests carry a `parity` pytest marker so they can be selected/excluded; the
dataset-level sweep runs in CI's existing matrix.

### 4. Proof-point kernel — `splits_sum_le_value`

Migrate `_utils.splits_sum_le_value` end-to-end to validate the full machinery on a real kernel:

1. Rust impl in `src/utils.rs` (plain ndarray, no PyO3).
2. `#[pyfunction]` wrapper in `src/ffi/mod.rs`, registered in `lib.rs`.
3. Python `register("splits_sum_le_value", numba=<existing njit>, rust=<ffi fn>, default=...)` in
   `_utils.py`; the call site routed through `dispatch.get("splits_sum_le_value")`.
4. `splits_sum_le_value` hypothesis strategy + `test_splits_sum_le_value_parity`.
5. Appears in at least one dataset-level parity test path.

`splits_sum_le_value` is chosen because it is small, pure (`(array, scalar) → intp array`), and — the
deciding factor — **called directly from Python** (`_write.py:1280`, computing memory-bounded write
splits), so it exercises the `dispatch.get(...)` Python-call-site path the registry is built around.
`padded_slice`, the other `_utils` kernel, was rejected: it is an njit-internal leaf (see "dispatch
unit" note above) and cannot validate the dispatch pattern.

Because its only call site is in the write path, the dataset-level backstop exercises it via a
`gvl.write()` round-trip (run once under each `GVL_BACKEND`), not `__getitem__`. Whether its `default`
flips to `rust` in this PR or stays `numba` (Rust present but not yet default) is a plan-level call;
parity must be green either way. This is **foundation validation, not Phase 4 scope** — a test harness
with zero real subjects is itself untested scaffolding.

### 5. cargo test + pixi wiring

- Add a `cargo-test` task to the `dev` environment in `pixi.toml`.
- Fold it into the existing `test` task so `pixi run -e dev test` runs pytest **and** cargo (matching
  CLAUDE.md's description of `test` as the whole-tree runner).

### 6. abi3 wheel invariant

- **Confirm, do not rebuild.** Verify the abi3 wheel CI job still builds py310–313 × linux/macOS green
  after the `ffi/` seam + pymodule registration changes. Standing invariant; the only action is
  checking the workflow stays green (and fixing it if the new seam breaks the build).

### 7. Baselines — all four

Corpus: 1kg chr21/chr22 (vcfixture tier), matching the roadmap baseline table.

| Metric | Tool | Notes |
|---|---|---|
| `gvl.write()` wall-clock | timing in committed bench script | runnable directly |
| `gvl.write()` peak RSS | memray | runnable directly |
| `gvl.update()` wall-clock | timing in committed bench script | runnable directly |
| `Dataset.__getitem__` throughput | dataloader bench + py-spy A/B | bench pulled from `prefetching-dataloader`; py-spy via hand-off bash script (sudo on macOS) |

- Bench scripts committed (under `repro/` or `benchmarks/`, following existing layout) so numbers are
  reproducible.
- py-spy is **not** invoked directly — a bash script with the exact profile commands is handed to
  David to run with sudo ([[feedback_macos_profiling_handoff]]). memray is fine to run directly.
- Results written into the `docs/roadmaps/rust-migration.md` baseline table with the "Captured" marks
  flipped, and the bench environment (machine, OS) noted alongside, mirroring the existing bigWig
  write-slice rows.

---

## Data flow (a kernel's lifecycle after Phase 0)

```
implement in src/<domain>.rs (plain ndarray)
        │
        ▼
#[pyfunction] wrapper in src/ffi/mod.rs ──► registered in src/lib.rs pymodule
        │
        ▼
_<module>.py: register(name, numba=njit_fn, rust=ffi_fn, default="numba")
        │                                   call sites use dispatch.get(name)(...)
        ▼
tests/parity/: strategy + assert_kernel_parity(name, ...)  ── byte-identical gate
        │
        ▼   (parity green)
flip entry default → "rust"; delete numba impl in same bundled PR
        │
        ▼   (phase closes)
Phase 5: delete _dispatch.py, inline rust calls
```

---

## Testing strategy

- **The harness is the migration's test strategy.** Phase 0 validates it on `splits_sum_le_value`.
- **TDD for the foundation code itself:** write tests for the registry resolution logic
  (`GVL_BACKEND` override vs per-kernel default, unknown-name errors, invalid-backend errors) and for
  `assert_kernel_parity` (passes on equal, fails on differing dtype/shape/bytes) *before*
  implementing them.
- Run the full tree before pushing (`pixi run -e dev test`) per CLAUDE.md — the new `cargo-test`
  fold means this now also runs cargo.

---

## Out of scope (YAGNI)

- Empty domain modules (`genotypes/`, `variants/`, `reconstruct/`, `tracks/`, `reference/`, `write/`)
  — grown per-phase when they hold real code.
- Migrating any kernel beyond the `splits_sum_le_value` proof-point.
- Removing the dispatch registry (Phase 5).
- Reworking `bigwig.rs` / `tables.rs`.

---

## Roadmap updates (part of this PR)

Per the roadmap's self-maintenance contract:

- Tick Phase 0 tasks as completed; set the Phase 0 status marker ⬜ → ✅ (or 🚧 if split) + PR link.
- Fill the baseline table rows + "Captured" marks.
- Add a dated entry to the "Notes & decisions log" recording the harness/registry/ffi-seam design,
  the Python-entry-vs-leaf dispatch rule, and the `splits_sum_le_value` proof-point.
