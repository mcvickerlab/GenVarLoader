# Spec: Rust variant-allele reverse-complement (churn-free)

**Date:** 2026-06-25
**Branch base:** `rust-migration`
**Roadmap:** completes the deferred variant-RC half of optimization Target 6
(`docs/roadmaps/rust-migration.md`, §"Optimization targets" #6); the Target-6 note
said `RaggedVariants` + `_FlatVariants` RC were "targeted in Target 7", but Target 7
(PR #250) collapsed object churn for *windows* and never folded their RC. This closes
that loose end.

## Background / corrected premise

- "RC variants" **is** a supported feature: on the read path, negative-strand regions
  reverse-complement the variant **alleles** (`alt`/`ref` byte strings) whenever
  `view.rc_neg` is set. `_FlatVariants.reverse_masked` / `RaggedVariants.rc_` /
  `_FlatAlleles.reverse_masked` implement it.
- It is **already numba-free**: those methods call seqpro-core's Rust
  `reverse_complement_masked`. The `_rag_variants.rc_helper-*.nbc` files in `__pycache__`
  are **stale** numba caches from an older version — no live `rc_helper` exists.
- `_FlatVariantWindows` (the Target-7 `assemble_variant_buffers` output) is **never**
  reverse-complemented — `reverse_complement_ragged` returns it unchanged
  ("reference-oriented"). So the windows path needs nothing here.

## Problem

The RC runs as a Python **post-pass** (`_query.py` → `reverse_complement_ragged` →
`reverse_masked`/`rc_`) whose inner implementation rebuilds layered ragged objects per
batch — `to_chars().to_packed()`, `Ragged.from_offsets(...)` view + rebuild, `np.repeat`
mask expansion — purely to hand contiguous byte buffers to seqpro. The byte buffers in
`_FlatAlleles` are **already** plain `uint8` data + `int64` offset arrays; the object
churn is pure overhead.

## Goal

Replace the seqpro call + per-batch object churn with a thin gvl-owned Rust kernel that
reverse-complements the masked alleles **in place on the raw `_FlatAlleles` buffers**,
reusing the Target-6 primitives. Keep the existing seqpro path as the dispatch
**reference** backend (retained for byte-identical parity + perf gating; deleted in
Phase 5, **not now** — `rust-migration` is not ready to merge and numba/reference
backends must stay for performance comparison).

Non-goals: no on-disk format change; no change to `_FlatVariantWindows` (still not RC'd);
no change to flank-token handling (the current post-pass RCs only `alt`/`ref`, never
`flank_tokens` — preserve exactly).

## Placement decision (settled)

RC is a **dedicated Rust call applied after dummy-fill**, at the same point in the
pipeline as today's seqpro pass — *not* folded inside `assemble_variant_buffers`.

```
assemble_variant_buffers (unchanged, no to_rc)
  -> _FlatVariants
  -> fill_empty_groups (dummy)             # unchanged
  -> rc_alleles_inplace(byte_data, seq_offsets, var_offsets, to_rc_row)   # NEW, rust
```

Rationale: preserves the exact `assemble → fill → RC` ordering, so dummy-filled alleles
(including a **custom** non-palindromic `DummyVariant.alt`, e.g. `b"AC"`) are RC'd
identically to today. The default `DummyVariant.alt`/`.ref` is `b"N"` (RC-invariant), but
custom dummies are reachable, so ordering parity matters. The one extra FFI crossing is on
already-contiguous buffers (negligible vs. the deleted Python allocation churn). Folding
into `assemble_variant_buffers` would put RC *before* fill and require a mask-aware
`fill_empty_groups` to RC the dummy allele — more moving parts for no measurable gain.

## Design

### 1. Rust kernel (`src/variants/` + `src/ffi/`)

Core (pure, in e.g. `src/variants/mod.rs` or `windows.rs` neighborhood), reusing
`crate::reverse::{rc_flat_rows_inplace, COMP}`:

```rust
/// Reverse-complement the alleles of mask-selected (b*p) rows, in place.
/// `byte_data`        contiguous allele bytes (uint8)
/// `seq_offsets`      per-allele byte boundaries (len n_alleles + 1)
/// `var_offsets`      per-(b*p)-row allele boundaries (len n_rows + 1)
/// `to_rc_row`        per-(b*p)-row bool mask (len n_rows)
pub fn rc_alleles_inplace(
    byte_data: &mut [u8],
    seq_offsets: ArrayView1<i64>,
    var_offsets: ArrayView1<i64>,
    to_rc_row: ArrayView1<bool>,
)
```

Implementation: for each row `g` with `to_rc_row[g]`, the alleles `a` in
`var_offsets[g]..var_offsets[g+1]` are RC'd — i.e. build the per-allele mask from the row
mask + `var_offsets` and delegate to `rc_flat_rows_inplace(byte_data, seq_offsets,
per_allele_mask)`. (Equivalent to today's `np.repeat(per_bp, np.diff(var_offsets))`
expansion, done in Rust.)

FFI wrapper `rc_alleles` in `src/ffi/mod.rs`: takes a `PyReadwriteArray1<u8>` (mutated in
place) + the three views; registered in `lib.rs`. Mirrors the in-place convention of the
other read-path kernels.

### 2. Dispatch registration

Register `rc_alleles` in `_dispatch`:
- **rust**: the new FFI kernel above.
- **numba** (reference): the existing seqpro-`reverse_complement_masked` implementation,
  extracted into a small function so it can be the registered reference.

`GVL_BACKEND=numba` therefore keeps variant RC on the seqpro reference (clean perf gating:
a numba-backend read does not smuggle in the new rust RC). `GVL_BACKEND` unset ⇒ rust.

### 3. Python call sites

- `_FlatAlleles.reverse_masked` (`_flat_variants.py`): replace the
  `Ragged.from_offsets(...) + reverse_complement_masked(...)` body with
  `get("rc_alleles")(self.byte_data, self.seq_offsets, self.var_offsets, per_bp_mask)`,
  where `per_bp_mask = np.repeat(mask, self.ploidy)` (same broadcast as today). Operates in
  place on `byte_data`; returns `self`.
- `RaggedVariants.rc_` (`_rag_variants.py`): keep the existing buffer extraction
  (`to_chars().to_packed()` is needed to *reach* the contiguous char buffer + offsets) but
  replace the inner `_sp_reverse_complement(view, _COMP, mask=allele_mask)` call with
  `get("rc_alleles")(data, char_off, var_off, to_rc_row)`. (This path is the cold
  non-flat route; the hot flat read path goes through `_FlatAlleles.reverse_masked`.)
- Both keep the early-out when the mask is all-False.

### 4. `_query.py`

- **Unspliced post-pass: unchanged in structure.** It already routes variant kinds through
  `reverse_complement_ragged` on both backends; backend choice now happens *inside*
  `reverse_masked`/`rc_` via the `rc_alleles` dispatch. No backend-split edits needed here.
- **Remove the dead spliced variant guard** in `_getitem_spliced`: spliced variants are
  rejected upstream (`__call__` raises `NotImplementedError` for spliced variant/
  variant-windows kinds), so the `_VARIANT_TYPES_S` branch is unreachable. Delete it.

## Parity & testing

Byte-identical differential testing is the standing migration contract; the reference here
is the existing seqpro implementation.

1. **Rust unit tests** (`#[cfg(test)]`): `rc_alleles_inplace` on multi-row, multi-allele
   buffers — masked vs unmasked rows, empty rows, odd-length + `N` alleles, all-False mask
   no-op. (Mirrors the `reverse.rs` test style.)
2. **Kernel parity** (`tests/parity/`, hypothesis): `rc_alleles` rust vs reference,
   byte-identical, over property-generated `(byte_data, seq_offsets, var_offsets, mask)`
   for both the `_FlatAlleles` layout and the `RaggedVariants.rc_` char-buffer layout.
3. **Dummy-fill + custom-allele edge cases** (locks the ordering risk): a neg-strand query
   with empty `(region, sample, ploid)` groups, run with **(a)** the default `b"N"` dummy
   and **(b)** a custom non-palindromic dummy (`alt=b"AC"`, `ref=...`), asserting rust ==
   reference end-to-end. This is the case that would diverge under an in-kernel
   (pre-fill) fold.
4. **Live-path spy** (`tests/parity/test_dataset_parity.py` precedent): open a variants
   dataset with negative-strand regions, index it, assert the `rc_alleles` kernel is
   actually invoked and the result is byte-identical to the numba/reference backend.

Full-tree gate before close: `pixi run -e dev pytest tests -q` on **both** backends,
`cargo test`, lint/format/typecheck, abi3 wheel build. Update
`docs/roadmaps/rust-migration.md` (tick the Target-6 variant-RC follow-up; record that the
deferred `RaggedVariants`/`_FlatVariants` RC now runs on a gvl rust kernel, reference
retained).

## Files touched

- `src/variants/...` — `rc_alleles_inplace` core + tests
- `src/ffi/mod.rs`, `src/lib.rs` — `rc_alleles` pyfunction + registration
- `python/genvarloader/_dataset/_flat_variants.py` — `_FlatAlleles.reverse_masked`
- `python/genvarloader/_dataset/_rag_variants.py` — `RaggedVariants.rc_`
- `python/genvarloader/_dataset/_query.py` — remove dead spliced variant guard
- `python/genvarloader/_dispatch.py` (or the per-module registration site) — register
  `rc_alleles`
- `tests/parity/...`, `tests/dataset/...` — parity + edge-case + spy tests
- `docs/roadmaps/rust-migration.md` — status update

## Out of scope

- Assembly / instruction-count micro-optimization (owned separately, in parallel).
- Deleting the seqpro reference path (Phase 5).
- Any change to `_FlatVariantWindows` RC behavior (remains a no-op).
