# Design: shared `seqpro-core` Rust ragged crate (GVL Phase 1, realized)

**Date:** 2026-06-23
**Status:** approved, pending implementation plan
**Repos touched:** `seqpro` (substrate) + `GenVarLoader` (first consumer)
**Roadmap:** `docs/roadmaps/rust-migration.md` — this realizes Phase 1 and reframes Phases 1 & 6.

---

## Context

The Rust-migration roadmap (2026-06-18) assumed GVL would build its **own** native ragged
layout in Rust (Phase 1) and **absorb** seqpro/genoray last (Phase 6, "bring ragged
primitives in-house — drop the seqpro hot-path dependency").

Reality has moved past that framing:

- `seqpro/src/ragged.rs` (~22 KB Rust) already implements ragged kernels — `validate`,
  `select`, `nested_gather`, `nested_pack`, `pack`, `concat` — all rayon-parallel and
  `#[pyfunction]`-registered.
- seqpro's `Ragged` class (Python, `rag/_core.py`) is now subclass-preserving (0.18.0), and
  **GVL's `RaggedVariants` subclasses it** (PR #239, merged to `main`).
- Only **two** ragged ops remain on numba in seqpro: `_to_padded_copy` and
  `_reverse_complement_ragged` (both in `rag/_ops.py`).
- seqpro's crate is already `crate-type = ["cdylib", "rlib"]` — consumable as a Rust library.

**Decision (owner of both repos):** seqpro owns the Rust ragged layout. GVL's crate consumes
it rather than reimplementing. This inverts Phase 6 (seqpro becomes the shared substrate, not
something to absorb away) and changes Phase 1 from "reimplement" to "consume."

### Why a pyo3-free core (the key architectural move)

Two pyo3 versions cannot link into a **single** cdylib (pyo3 owns one global interpreter
binding). seqpro is on pyo3 0.20 / numpy 0.20; GVL is on pyo3 0.28 / numpy 0.28. So "GVL
depends on seqpro's crate as-is" is a non-starter.

The resolution: the shared substrate is a **pyo3-free `rlib`**. GVL links only pure Rust from
it — zero pyo3 symbols cross over. Each pymodule does its own numpy↔core bridging. (seqpro's
`.so` and GVL's `.so` already coexist as separate extension modules in one interpreter today;
the single-version rule is per-cdylib, not per-process.)

Independently, we **bump seqpro's pymodule to pyo3 0.28 / numpy 0.28** in this phase — it is
not required for the shared-core link, but it is low-cost hygiene and keeps the two repos on
the same pyo3 generation. `abi3-py39` is retained (do not narrow seqpro's Python support).

---

## Architecture

```
seqpro repo
  crates/seqpro-core/          NEW — pure Rust, NO pyo3. Owns the Ragged layout + ops.
    src/lib.rs                 Ragged layout type + ops + unit/proptest tests
    Cargo.toml                 deps: ndarray, rayon (+ num-traits/thiserror as needed)
  src/ (pymodule)              pyo3 0.28 / numpy 0.28; #[pyfunction]s now wrap seqpro-core
  Cargo.toml                   [workspace]; seqpro-core = { path = "crates/seqpro-core" }

GenVarLoader repo
  src/ragged/                  NEW — pyo3 0.28 bridge: numpy → seqpro_core::Ragged → op → numpy
  Cargo.toml                   seqpro-core = { path = "../seqpro/crates/seqpro-core" }  (local editable)
```

The shared substrate is a normal `rlib`. This *is* the roadmap's "native ragged layout in
Rust" — it just lives in the repo that owns ragged. GVL's `src/ragged/` is a **bridge**, not a
reimplementation.

**Working mode:** develop against a local editable `seqpro-core` (path dep) until the seqpro
changes are merged/released; then repoint GVL's dep to a git/crates.io release. (Mirrors the
seqpro↔genoray editable→released coupling handled in prior PRs.)

---

## Components

### 1. `seqpro-core` crate (pure Rust)

- **`Ragged` layout type.** Borrowed form over `(offsets: ArrayView1<i64>, data: &[u8],
  elem_size, ndim)` for zero-copy numpy input; ops produce owned output buffers. Ops are
  byte-oriented with an element-size parameter (matching today's `ragged.rs` kernels); dtype
  semantics stay at the Python edge.
- **Ops** (methods / associated fns):
  - Moved verbatim from `src/ragged.rs`, minus pyo3: `validate`, `select`, `nested_gather`,
    `nested_pack`, `pack`, `concat`.
  - **Newly ported from numba:** `to_padded` (takes a fill value; pad direction matches the
    existing `_to_padded_copy` semantics) and `reverse_complement` (S1 / single-byte only).
- **Tests:** the existing Rust unit tests in `ragged.rs` move with the code; add `proptest`
  cases for the two ported ops.

### 2. seqpro pymodule (pyo3 0.28)

- **Bump** `pyo3` 0.20→0.28 and `numpy` 0.20→0.28 across `src/{lib,ragged,kshuffle,translate,
  kmer_encode}.rs` (Bound API, GIL/text-signature changes). Retain `abi3-py39`.
- **Rewire** `src/ragged.rs` `#[pyfunction]`s to thin wrappers over `seqpro_core` — identical
  behavior, identical Python-visible signatures.
- **Add** two `#[pyfunction]`s exposing the ported `to_padded` / `reverse_complement`.
- **`rag/_ops.py`** swaps its two numba calls for the new Rust functions. seqpro's `rag` layer
  becomes **numba-free**.

### 3. GVL consumer beachhead (pyo3 0.28)

- New `src/ragged/` module: bridges numpy ↔ `seqpro_core::Ragged` and exposes the op(s) GVL
  needs via pyo3.
- **Proof-point milestone (exactly one op):** reroute GVL's most-used ragged op,
  `RaggedSeqs.to_padded()`, to call `seqpro_core` directly through the rlib (Rust→Rust, no
  Python-seqpro round-trip). Everything else stays on the existing path.

---

## Data flow (proof-point op)

```
Dataset.__getitem__ → RaggedSeqs.to_padded()
  Python: hand offsets+data numpy buffers to GVL's pyo3 bridge
  GVL src/ragged/: construct seqpro_core::Ragged (borrowed, zero-copy) → .to_padded(fill)
  return owned buffer → numpy → dense array
```

vs. legacy path (Python `seqpro` Ragged → numba `_to_padded_copy`). The two must be
byte-identical.

---

## Testing & parity

Scoped to this phase — **not** the full reusable Phase 0 harness.

- **`seqpro-core`:** Rust unit tests (migrated) + proptest for `to_padded`/`reverse_complement`.
- **seqpro Python differential:** new Rust `to_padded`/`reverse_complement` vs the **old numba**
  as oracle, across the py310–313 × linux/macOS matrix. Delete the numba impls only once parity
  holds.
- **GVL differential:** rerouted `RaggedSeqs.to_padded()` (Rust-via-core) vs the existing
  Python-seqpro path, byte-identical.
- **Standing CI invariant:** abi3 wheels keep building for both repos across the py-matrix.
- **No new perf baselines.** Phase 1 is foundational (no perf gate); record incidental wins
  only. The full differential-test harness and write/getitem baselines remain deferred Phase 0
  work.

---

## Roadmap edits (`docs/roadmaps/rust-migration.md`)

- **Goal & end state / Phase 6:** drop "bring ragged primitives in-house — drop the seqpro
  hot-path dependency." Invert: `seqpro-core` is the shared Rust substrate. Narrow Phase 6 to
  genoray (variant IO) only; note seqpro stays as the ragged owner.
- **Target crate layout:** GVL's `ragged/` is a **bridge** to `seqpro-core`, not a
  reimplementation.
- **Migration contract:** clarify that the ragged layer is consumed, not reimplemented.
- **Phase 1:** reframe to "extract `seqpro-core`, port its last 2 numba ops, GVL consumes via
  rlib." Mark awkward-removal done (already true).
- **Notes & decisions log (2026-06-23):** record the pyo3-free-core decision; the pyo3/numpy
  0.20→0.28 bump (in scope, hygiene, not required for the link and why); seqpro `rag` becomes
  numba-free; GVL beachhead = `to_padded` proof-point.

---

## Scope boundaries (YAGNI)

**In scope:** `seqpro-core` crate split; the 2 numba→Rust ports; seqpro pyo3/numpy 0.20→0.28
bump; GVL `src/ragged/` bridge + exactly one consumer op (`to_padded`) with parity; roadmap
refresh.

**Out of scope:** the full reusable differential-test harness; write/getitem perf baselines;
any GVL Phase 2–5 kernel; genoray; narrowing seqpro's abi3 target; additional GVL consumer ops
beyond the single proof-point.

---

## Risks / open items for the plan

- **pyo3 0.20→0.28 migration surface** in seqpro touches all `#[pyfunction]`s (Bound API). Land
  the bump as its own reviewable step before/independent of the core extraction so a parity
  failure is easy to localize.
- **`Ragged` generics vs byte+elem-size.** Keep the existing byte-oriented kernel convention to
  minimize the port surface; revisit generic typing only if a later phase needs it.
- **Editable→released seqpro coupling** (path dep → git/crates.io) must be flipped before GVL
  ships; mirror the prior seqpro↔genoray rebase/lock handling.
- **`reverse_complement` is S1-only** — guard/assert dtype at the bridge.
