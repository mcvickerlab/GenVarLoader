# Design: Wrap up Phase 5 of the Rust migration (sans genoray)

**Date:** 2026-06-27
**Branch:** `phase-5-w6-wrapup` (off `rust-migration`)
**Roadmap:** `docs/roadmaps/rust-migration.md` (Phase 5, 🚧 — W1–W5 done, W6–W9 remain)
**Status going in:** Phases 0–4 ✅. W5 (PR #260) golden-snapshotted the numba-oracle parity
suites, deleted all gvl-own numba kernels (count = 0), and added rayon batch parallelism
gated byte-identical to the serial golden result.

## Goal

Finish Phase 5's open finalization threads so the Rust migration is shippable, **excluding
Phase 6 (absorb genoray)** which stays out of scope. Land everything as **one PR into
`rust-migration`** (NOT master). The `rust-migration → master` merge is left to the
maintainer to trigger (no-squash, per [[no-squash-merges]]).

**Explicitly NOT in scope:** the "single big `__getitem__` kernel" architectural collapse.
Instead of building it, Unit A *audits* whether it is still warranted and records the verdict
in the roadmap.

## Context discovered during brainstorming

- **No dispatch layer remains.** `python/genvarloader/_dispatch.py` is deleted (only a stale
  `.pyc` lingers); zero `GVL_BACKEND` / `import numba` / `nb.njit` references in source. W5
  already collapsed the rust/numba switch — Python calls Rust directly via
  `from ..genvarloader import (...)` (the compiled `genvarloader.genvarloader` pymodule).
- **~28 FFI entries** registered in `src/lib.rs`, including the fused one-FFI-crossing
  `__getitem__` kernels from Phase 3/W3 (`reconstruct_haplotypes_fused`,
  `reconstruct_annotated_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused`,
  `reconstruct_annotated_haplotypes_spliced_fused`, `intervals_and_realign_track_fused`).
- **seqpro-core is already a released dep.** `Cargo.toml` has `seqpro-core = "0.1"` and
  `Cargo.lock` resolves `seqpro-core 0.1.0` from the crates.io registry with a checksum — no
  path dep, no `[patch]`. The Phase 1 "editable path-dep, flip before shipping" note is stale.

The upshot: "collapse the PyO3 surface to a thin shim" is **largely already realized** at the
indirection level. What is left to determine is how much Python *orchestration glue* still
sits between `__getitem__` and the fused calls — that is what Unit A measures.

## Units of work

The units are mostly independent. Unit D (perf) is the long pole. Units B/C are quick
verifications. Unit A is investigation + roadmap text with no code change.

### Unit A — PyO3 surface / thin-shim audit (reframed Phase 5 item)

Inventory the live **read path** (`Dataset.__getitem__` → reconstructor in
`_dataset/_reconstruct.py` / `_haps.py` / `_query.py` → fused FFI kernel) and the **write
path**, and classify every remaining piece of Python between the public API and the FFI call
into one of three buckets:

1. **Intentional shim** — indexing sugar, torch integration, validation / error messages.
   Stays in Python by design (this is the migration's end state).
2. **Genuinely-remaining collapsible glue** — per-batch coercions, allocations, or Python
   object churn on the hot path that a future "bigger kernel" would absorb.
3. **Already-collapsed** — confirmed to be one FFI crossing with no material Python work.

**Output:** a precise "what's left for the thin shim" list written into the roadmap (Phase 5
section + notes log). Given W5 removed dispatch and Phase 3/W3 fused each path to one
crossing, the expectation is the bucket-2 list is short or empty. **No code changes in this
unit.**

### Unit B — `cargo test` standalone verification

Confirm the crate builds and tests purely via `cargo test` (rlib path, no pixi / maturin /
Python-extension layer). The lib is `crate-type = ["cdylib", "rlib"]`; the
`extension-module` pyo3 feature is non-default, so `cargo test` links a real libpython. If it
is broken, record the minimal fix or the documented invocation. Record the result under the
Phase 5 checkpoint ("crate is fully cargo-testable standalone").

### Unit C — seqpro-core released-dep verification

Already resolves `seqpro-core 0.1.0` from crates.io (verified in `Cargo.lock`). Confirm a
clean build against the published crate with no lingering path / `[patch]` override, and
**correct the stale Phase 1 roadmap note** ("editable path-dep, flip to git/crates.io before
shipping") to reflect that it is already released.

### Unit D — W6 perf re-baseline (long pole)

On Carter (AMD EPYC 7543, linux-64), corpus `chr22_geuv.gvl` (format 2.0, 165 regions × 5
samples, chr22), using the established de-noised harness (`tests/benchmarks/test_e2e.py`
pedantic-min, iterations=10/rounds=50/warmup=5, + `tests/benchmarks/profiling/profile.py`
wall-clock for the variants paths). Release build (`maturin develop --release`).

- **Primary new signal:** rust **serial vs rayon multi-thread** — a clean *same-session* A/B
  via the `parallel` toggle W5 added to the read kernels. Measure **serial + a thread sweep
  (2 / 4 / 8 / default-all-cores)** across the read paths (tracks-only, tracks-seqs,
  haplotypes, annotated, variants, variant-windows) to capture the rayon speedup **curve** and
  the gvl-attributable **peak-RSS** deltas.
- **Constraint — no live numba A/B.** numba was deleted in W5, so we compare against the
  **W4-recorded** same-session numba numbers (`docs/roadmaps/phase-5-w4-final-ab.md`) and the
  Phase 0 / Phase 4 baselines. We do **not** re-checkout a numba commit: W4 already locked the
  single-thread numba A/B, and [[gvl-rust-perf-gate-shared-node-noise]] makes cross-session
  absolute wall-clock unreliable. The durable signals are byte-identical parity (already
  gated) + same-session serial-vs-rayon improve-or-hold + deterministic counts.
- **Output:** record the rayon speedup curve + RSS deltas under the Phase 5 checkpoint
  ("full perf re-baseline recorded here").

### Phase 5 status disposition

Set by Unit A's verdict:

- If the audit shows the shim is already thin (likely) **and** the checkpoint criteria are met
  (numba count = 0 ✓; perf re-baseline ✓; cargo-testable standalone ✓), mark **Phase 5 ✅** and
  re-file any residual collapse as a separate, clearly-labelled optimization track (it was
  never part of the Phase 5 checkpoint gate).
- If real bucket-2 glue remains, keep **Phase 5 🚧** with the audited list as the explicit
  remainder, and note that this branch advanced W6 + the verifications.

## Gate (per CLAUDE.md)

1. `pixi run -e dev maturin develop --release` **first** (pytest does not rebuild Rust).
2. Full tree: `pixi run -e dev pytest tests -q` green (numba backend is gone, so a single
   rust-only run — no A/B matrix).
3. `cargo test --release` green.
4. `pixi run -e dev ruff check python/ tests/` + `ruff format` + `typecheck` + `cargo clippy`
   clean.
5. abi3 wheel builds.
6. Roadmap updated: tick completed items, set Phase 5 marker, add a notes-log entry, record
   the Unit D measurements under the checkpoint, correct the stale seqpro-core note.

## Deliverable

One PR into `rust-migration` covering Units A–D + the roadmap finalization. The maintainer
performs the `rust-migration → master` merge separately.

## Open questions

None blocking. Thread-sweep granularity for Unit D (2/4/8/all) confirmed during brainstorming;
adjustable if the corpus is too small for higher thread counts to show signal.
