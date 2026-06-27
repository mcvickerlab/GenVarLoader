# Phase 3 — Reconstruction + track realignment (design)

**Date:** 2026-06-24
**Branch:** `phase-3-reconstruction` (off the persistent `rust-migration` integration branch)
**Roadmap:** `docs/roadmaps/rust-migration.md` → Phase 3
**Status:** design approved 2026-06-24; spec under review

This spec covers the largest migration phase — the numba bulk of the read path. It
follows the established strangler-fig + byte-identical-parity contract from Phases 0–2,
and additionally **begins the read-path consolidation** (single large `__getitem__`
kernel) that Phase 2 profiling identified as the real throughput win.

---

## Goal

1. Port the 8 numba-only kernel groups across the Phase 3 read-path files to Rust as
   **1:1 parity twins** behind per-kernel dispatch (numba retained as registered parity
   reference, deleted wholesale in Phase 5).
2. **Begin consolidation**: fuse the two hot read paths — **haplotypes** and **tracks** —
   into single Rust `__getitem__` kernels that cross the Python/Rust boundary once,
   eliminating the redundant `np.ascontiguousarray` glue Phase 2 profiling pinned at
   62% of the variants loop.

## Decisions captured during brainstorming (2026-06-24)

- **Port strategy:** 1:1 parity twins **+** begin consolidation (not strict 1:1-only,
  not fused-from-scratch).
- **Gate:** **parity is the hard gate** (byte-identical, blocks landing) for every ported
  kernel; **throughput is recorded only** — no throughput gate in Phase 3. The final
  throughput gate remains in the Phase 5 consolidation pass. (This supersedes the stale
  `Gate: parity + Dataset.__getitem__ throughput` line in the current roadmap Phase 3
  section, which predates the Phase 2 branch/gate-strategy change; that line will be
  corrected as part of this work.)
- **Consolidation beachhead:** fuse **both** the haplotypes and tracks read paths this
  phase (not haplotypes-only, not deferred to end-of-phase profiling).
- **Sequencing:** easiest→hairiest so parity tooling matures before the risky kernels:
  reference → haplotype reconstruction → track realignment → fusion.
- **Out of scope this phase:** `_insertion_fill.py:lower` and `_splice.py:build_splice_plan`
  stay plain Python (array-packing / plan-building, not hot; they feed the kernels).

---

## Architecture

Identical shape to Phase 2:

- Pure-`ndarray` / `rayon` cores in new `src/` domain modules — no PyO3.
- PyO3 wrappers confined to `src/ffi/`.
- Per-kernel dispatch via `genvarloader._dispatch` (default `rust`; `GVL_BACKEND`
  override; numba impl kept as the registered parity reference).
- `main`/`rust-migration` stays shippable; every step reversible until parity is proven.

### New Rust modules

```
src/
├── reconstruct/   # reconstruct_haplotypes_from_sparse (+ singular inner),
│                  # annotated variant (per-bp v_idx + ref-coord) variant
├── tracks/        # shift_and_realign_track[s]_sparse, _apply_insertion_fill (4 strategies),
│                  # _xorshift64 / _hash4 PRNG, tracks_to_intervals RLE
│                  # (+ _scanned_mask / _compact_mask)
└── reference/     # get_reference (par/ser), padded_slice, spliced-ref fetch
```

`padded_slice` moves out of `_utils.py`'s numba surface into the `reference` core (it is
a reference-assembly leaf). `_insertion_fill.py:lower` and `_splice.py:build_splice_plan`
remain plain Python and continue to produce the packed strategy arrays / splice
permutation+offsets the kernels consume.

### Fused `__getitem__` kernels (consolidation)

Two new Rust entry points that compose what are today multiple per-kernel boundary
crossings into one:

- **Fused haplotypes**: `get_diffs_sparse` (already Rust) + `reconstruct_*_from_sparse`
  in a single crossing, returning the reconstructed haplotype bytes (and, for the
  annotated mode, the per-bp variant-index and ref-coordinate arrays) without
  intermediate Python-side `np.ascontiguousarray` coercions.
- **Fused tracks**: `get_diffs_sparse` → `shift_and_realign_tracks_sparse` →
  `intervals_to_tracks` (already Rust) in a single crossing.

These are **new** entry points, not 1:1 twins; they are parity-verified at the dataset
level (see Testing) against the composed numba pipeline.

---

## Work breakdown (incremental landings on the branch; one bundled PR at phase close)

Each sub-unit lands incrementally on `phase-3-reconstruction` with its own parity suite,
mirroring Phase 2's task-by-task cadence. The whole phase merges into `rust-migration` as
one bundled PR.

### 3a — Reference path (warm-up; low parity risk)
- Port `get_reference` (parallel + serial selection), `_get_reference_row`, and
  `padded_slice` into `src/reference/`.
- Port the spliced-reference fetch (`_fetch_spliced_ref` consumes `build_splice_plan`'s
  permutation; the plan builder stays Python).
- Parity: byte-identical reference assembly (incl. boundary padding) over hypothesis
  inputs; spy-guarded reference-mode dataset backstop.

### 3b — Haplotype reconstruction (core)
- Port `reconstruct_haplotypes_from_sparse` (batch/parallel) + `reconstruct_haplotype_from_sparse`
  (singular: shifting, variant overlaps, padding) into `src/reconstruct/`.
- Port the annotated variant used by `_haps.py:_reconstruct_annotated_haplotypes`
  (returns per-bp variant indices + ref coordinates alongside the S1 bytes).
- Parity: byte-identical haplotype bytes **and** annotation arrays (variant idx + ref pos).

### 3c — Track realignment + RLE (hairiest; the parity risks live here)
- Port `shift_and_realign_tracks_sparse` (batch) + `shift_and_realign_track_sparse`
  (singular) into `src/tracks/`, including `_apply_insertion_fill` with all four
  strategies (Repeat5p, Constant, FlankSample, Interpolate) and the `_xorshift64`/`_hash4`
  PRNG.
- Port `tracks_to_intervals` (RLE) + `_scanned_mask` + `_compact_mask`.
- Parity: byte-identical tracks across **all four** fill strategies (incl. the RNG-driven
  FlankSample), plus byte-identical RLE round-trip.

### 3d — Consolidation (fused kernels; throughput recorded, not gated)
- Build the fused haplotype `__getitem__` Rust kernel and the fused tracks `__getitem__`
  Rust kernel (single boundary crossing each; drop redundant `np.ascontiguousarray`).
- Re-profile `chr22_geuv` (haplotypes + tracks modes, `NUMBA_NUM_THREADS=1`, Carter) and
  **record** throughput + peak RSS in the roadmap. Confirm via cProfile that the
  `np.ascontiguousarray` glue tax is gone from the fused paths.

---

## Parity strategy

- Per-kernel `@pytest.mark.parity` hypothesis suites asserting **byte-identical** output;
  for tuple-returning kernels, assert every returned array.
- Spy-guarded **dataset backstops** for haplotypes and tracks modes proving the fused
  kernels are actually invoked on the live `Dataset.__getitem__` path (the Phase 0
  lesson: a backstop must spy + assert non-trivial output so a vacuous pass is impossible).
- Parity is verified across the standing py310–313 × linux/macOS matrix per the contract;
  a kernel only lands when parity holds.

### Two identified parity risks (both in 3c)

1. **FlankSample PRNG.** `_xorshift64`/`_hash4` are seeded and deterministic, so
   byte-identical parity is achievable **only if** the Rust port reproduces the exact
   `u64` wrapping arithmetic and hash-mixing order. Mitigation: port bit-for-bit and add a
   direct PRNG-sequence unit test (Rust output == numba output for a fixed seed grid)
   *before* wiring it into the kernel.
2. **Interpolate fill (float32).** Byte-identical float parity requires identical
   operation order. Both numba and Rust lower through LLVM, so this is achievable but is
   the most likely 1-ULP break. Mitigation: attempt strict byte-identical first; if
   intractable, fall back to the Phase 2 pattern (dtype/strategy-dispatched Rust core with
   a numba fallback for the offending strategy), documented in the roadmap if used.

---

## Testing & close-out

- Full tree green on **both** backends (`GVL_BACKEND=rust` and `GVL_BACKEND=numba`):
  `pixi run -e dev pytest tests -q` (dataset + unit).
- `cargo test` green; `ruff check`/`ruff format` clean on `python/ tests/`; `typecheck`
  clean; abi3 wheel builds.
- Env note (from Phase 2): dataset tests need pytest's tmp on the same filesystem as
  `tests/data` (`--basetemp=<repo>/.pytest_tmp`) or the write-path `os.link` hardlink
  fails cross-device (Errno 18).

## Roadmap maintenance (part of the work)

- Correct the stale `Gate: parity + Dataset.__getitem__ throughput` line in the Phase 3
  section to **parity hard-gate; throughput recorded only** (matches the 2026-06-24
  decision and the Phase 2 branch/gate strategy).
- Tick Phase 3 tasks and record measurements under the relevant checkpoint as each
  sub-unit lands; set the phase status marker (⬜→🚧→✅) + PR link.
- Add a Notes & decisions log entry for Phase 3 mirroring the Phase 2 entry.

## Out of scope

- `_insertion_fill.py:lower`, `_splice.py:build_splice_plan` (stay plain Python).
- Variant-flat / flank kernels already handled in Phase 2.
- The final crate consolidation and wholesale numba deletion (Phase 5).
- genoray variant IO (Phase 6).

## Success criteria

- All 8 Phase 3 kernel groups have byte-identical Rust twins behind dispatch (parity
  hard-gate met).
- Fused haplotypes + tracks `__getitem__` kernels land and are parity-verified at the
  dataset level; their throughput + peak RSS are recorded in the roadmap.
- Full tree green on both backends; cargo/lint/typecheck/abi3 clean.
- Roadmap updated (gate line corrected, tasks ticked, measurements + decisions logged,
  status marker + PR link set).
