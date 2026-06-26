# Design: Target 7 — variant-windows/variants assembly in one Rust call

**Date:** 2026-06-25
**Branch:** `opt/target-7-windows-rust-assembly` off `zero-copy-scale-safe-readpath`
**Roadmap:** `docs/roadmaps/rust-migration.md` — Phase 5 round-2 target 7 (⬜)
**Handoff:** `docs/handoffs/2026-06-25-phase5-getitem-optimization.md`

## Problem

The `variant-windows` (and `variants`) flat-output read path is **Python-overhead / GC-bound,
not kernel-bound**. `perf` flat self-time on `profile.py --mode variant-windows` shows no dominant
Rust kernel; the cost is the interpreter + allocator: `_PyEval_EvalFrameDefault` ~8.5%, GC
(`gc_collect_main` + `deduce_unreachable` + `visit_reachable` + `dict_traverse`) **~14% combined**,
dict/attr lookups, and ctypes/cffi dynamic-symbol lookup ~2.3%.

The source is the per-batch object graph the assembly tail allocates: a `Ragged` from
`reference.fetch`, numpy LUT-gather temporaries (`lut[bytes]`), `np.concatenate`/`reshape`
temporaries, and wrapper dataclasses (`_FlatWindow` / `_FlatAlleles` / `_FlatVariants` /
`_FlatVariantWindows` / scalar `_Flat`). The fix is to collapse the **ragged byte/token assembly**
into **one Rust call** that returns the final flat `(data, offsets)` buffers, so Python builds the
wrapper objects once and the numpy temporaries disappear.

This is the windows half of the deferred Phase-5 single-big-kernel rewrite.

## Decisions (locked during brainstorming)

1. **Scope:** cover **all** of `variants` + `variant-windows` (alleles, windows, bare alleles, the
   `flank_tokens` ride-along) — the full collapse, not windows-only.
2. **Fetch boundary:** the Rust call **owns the reference fetch** internally (the reference is a
   contiguous `u8` buffer + `i64` contig offsets — the same inputs `get_reference` already takes),
   removing the per-batch `Ragged` allocation and a Python round-trip.
3. **Granularity:** **one mega-call** (flag-driven) returning a bundle of all requested flat
   buffers in a single FFI crossing — fewest objects/crossings.
4. **Front edge:** **assembly tail only.** The mega-call takes already-gathered `v_idxs` /
   `row_offsets` + dataset-static per-variant arrays and returns all ragged byte/token buffers. The
   `v_idxs` gather + AF filter + compaction front-end and the cheap, dtype-polymorphic scalar-field
   gathers stay in Python — this keeps the issue-#231 custom-FORMAT-field numba fallback intact.
5. **Empty-group fill:** **not** folded into the mega-call. `fill_empty_groups` runs afterward on
   the wrapped buffers via the existing `fill_empty_seq/scalar/fixed` Rust cores, keeping the
   offset-consistency logic in one place.

## Architecture

Three layers; only the middle changes.

| Layer | Status | What |
|---|---|---|
| **Front-end** | unchanged (Python) | `geno_offset_idx` → `gather_rows` → `v_idxs`/`row_offsets`, AF filter, `compact_keep`, dosage gather, unphased-union fold → compacted `v_idxs`, `row_offsets`, `eff_ploidy` |
| **Scalar fields** | unchanged (Python) | `arr[v_idxs]` + `_Flat` wrap for start/ilen/dosage/info/custom-FORMAT — cheap fancy-indexing, dtype-polymorphic, #231 fallback preserved |
| **Ragged byte/token assembly** | **NEW (Rust mega-call)** | one FFI call owning `gather_alleles`, reference fetch, LUT tokenize, flank slice, alt-window assemble, flank-tokens — returns all requested flat `(data, seq_offsets)` buffers in one crossing |
| **Empty-group fill** | unchanged (Python + existing Rust cores) | `fill_empty_groups` on wrapped buffers, only when `dummy_variant` is set |

Python wraps the returned buffers into `_FlatAlleles` / `_FlatWindow` / `_Flat` **once** and
assembles `_FlatVariants` / `_FlatVariantWindows`. **No consumer change:** `reshape` / `squeeze` /
`to_ragged` / `fill_empty_groups` still operate on the same wrapper types; flat output mode returns
`_FlatVariantWindows` directly as before.

## The mega-call

`assemble_variant_buffers(...)` — Rust pyfunction in `src/variants/windows.rs`, registered in the
dispatch registry (`python/genvarloader/_dispatch.py`) with `rust` default and `numba` = today's
Python/numba assembly composed into the same bundle shape (the parity oracle).

### Inputs

- `v_idxs (i32)` — compacted variant indices, length `n_var`.
- `row_offsets (i64)` — per-`(b*p_eff)`-row variant boundaries, length `b*p_eff + 1`.
- Dataset-static globals (reuse `Haps.ffi_static` where already cached):
  - `v_starts (i32)`, `ilens (i32)` — global per-variant arrays (gathered by `v_idxs` inside Rust).
  - `alt_bytes (u8)` + `alt_off (i64)` — global allele byte buffer + offsets.
  - `ref_bytes (u8)` + `ref_off (i64)` — global, when ref is requested.
- `reference (u8)` + `contig_offsets (i64)` + `pad_char` — reference genome (owns the fetch).
- `v_contigs (i32)` — per-variant contig id (computed in Python via
  `np.repeat(regions[:,0], eff_ploidy)` then repeat by row counts; precomputed, cheap).
- `flank_length (i32)`.
- `token_lut ((256,) u8 | i32)` — `unknown_token` already baked in.
- **Flag set** describing which outputs to emit and the `ref` / `alt` ∈ {`window`, `allele`, `byte`}
  modes.

### Internals (small, individually unit-tested Rust cores)

Mirror today's Python/numba helpers:
- `gather_alleles` — variable-length allele bytestrings for `v_idxs`.
- `fetch_window` — reuse `get_reference`'s core; `[start-L, end+L)` read with absolute-coordinate
  OOB padding.
- `slice_flanks` — `f5` = first `L` bytes, `f3` = last `L` bytes of each window read.
- `assemble_alt_window` — `flank5 · alt · flank3` per variant.
- `tokenize` — apply the 256-entry LUT (output dtype = `lut.dtype`).

Preserve the **single fused fetch** for the `ref=window & alt=window` hot path (derive alt-window
flanks by slicing the one ref read), exactly as `compute_windows` does today. Fetch only when a
window output is actually requested.

### Returns

A dict keyed by field name → flat buffers:
- `alt` / `ref` (plain variants): `(byte_data u8, seq_offsets i64)`.
- `ref_window` / `alt_window` / bare `ref` / bare `alt` (windows): `(token_data lut.dtype, seq_offsets i64)`.
- `flank_tokens`: `(token_data,)` with fixed inner `2L`, offsets = `row_offsets`.

`var_offsets` equals `row_offsets` unchanged (no fill applied yet), so Python reuses it rather than
returning a copy. Token dtype follows `lut.dtype` (two monomorphizations: `u8` / `i32`).

## Parity strategy

Byte-identical gate, both backends. The assembly is **not** currently dispatched, so:

1. Register `assemble_variant_buffers` in the dispatch registry with:
   - `numba` = today's exact Python/numba helpers (`compute_windows`, `compute_ref_window`,
     `compute_alt_window`, `tokenize_alleles`, `compute_flank_tokens`, `gather_alleles`) composed to
     return the same bundle shape.
   - `rust` = the new mega-call.
2. TDD: pin the current flat `(data, offsets)` bundle as the oracle, build Rust under it.
3. The dataset backstop (`tests/parity/test_dataset_parity.py`) spies on the kernel to prove it runs
   on the live `__getitem__` path (no vacuous pass).

Reproduce exactly:
- `ends = starts - min(ilens, 0) + 1`.
- absolute-coordinate OOB padding with `pad_char`.
- `flank5 · alt · flank3` byte order.
- `[flank5 | flank3]` variant-major `2L` layout for `flank_tokens`.
- LUT mapping incl. `unknown_token` and `N` / out-of-alphabet bytes.

**Pre-existing xfail:** `test_e2e_variants` xfails today (`_FlatVariants.to_fixed` missing). Confirm
it xfails identically at base before starting; it is **not** a regression introduced here.

## Testing & perf gate

- Rust unit tests on each core (`gather_alleles`, `slice_flanks`, `assemble_alt_window`, `tokenize`,
  fused windows) + the orchestrator.
- `pixi run -e dev pytest tests/parity tests/unit -q` on both backends
  (`GVL_BACKEND=numba` too). Add fixtures covering the full `ref`/`alt` ∈ {window, allele} mode
  matrix, empty groups (dummy-variant fill), and the `flank_tokens` ride-along.
- `pixi run -e dev cargo-test`.
- Full tree before push (`pixi run -e dev pytest tests -q`, then `GVL_BACKEND=numba …`) per
  CLAUDE.md (scoped runs skip `tests/unit/`).
- Lint/format/typecheck: `ruff check python/ tests/ && ruff format … && typecheck`.
- Perf: re-measure `variant-windows` and `variants` via `tests/benchmarks/test_e2e.py` (min of
  `benchmark.pedantic`); expect GC/eval self-time to drop. Record the re-measured ratios in the
  roadmap, set the Phase-5 target-7 marker + PR link.
- HPC gotcha: `--basetemp=$(pwd)/.pytest_tmp` so the write path's `os.link` hardlink doesn't fail
  cross-device (Errno 18).

## Files

- **New:** `src/variants/windows.rs` — the cores + `assemble_variant_buffers` pyfunction. Wire into
  `src/ffi/mod.rs` (re-export) and `src/lib.rs` (`add_function`).
- **Rewrite:** `python/genvarloader/_dataset/_flat_variants.py` (`get_variants_flat` assembly tail
  calls the dispatched mega-call and wraps once) and `python/genvarloader/_dataset/_flat_flanks.py`
  (helpers retained as the numba oracle behind the registry).
- **Tests:** `tests/parity/` fixtures (mode matrix + empty + flank), Rust unit tests in
  `src/variants/windows.rs`.
- **Roadmap:** tick target 7, record ratios, set PR link.

## Out of scope

- Folding `fill_empty_groups` into the mega-call (kept as a separate post-pass).
- Folding the `v_idxs` gather / AF filter / compaction / scalar-field gather into Rust (front edge =
  assembly tail only; preserves #231 dtype-polymorphic fallback).
- Strand reverse-complement (target 6) and rayon batch parallelism (blocked until 5/6/7 land).
- Deleting the numba assembly helpers — they remain as registered parity oracles (wholesale numba
  deletion is a later Phase-5 step, not this workstream).
