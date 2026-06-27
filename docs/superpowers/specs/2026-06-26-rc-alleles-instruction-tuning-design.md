# rc_alleles_inplace Instruction-Level Tuning — Design

**Date:** 2026-06-26
**Branch target:** `opt/rc-alleles-instruction-tuning` → `rust-migration`
**Roadmap:** lands under Phase 3, Target 6 / round-3 area of `docs/roadmaps/rust-migration.md`

## Context

PR #251 (`rust-variant-rc-fold`) folded variant-allele reverse-complement into a
gvl-owned Rust kernel, `variants::rc_alleles_inplace` (`src/variants/mod.rs`). PR #252
(round-3 instruction-level tuning) applied `cargo asm`-driven instruction-count /
autovectorization passes to seven hot kernels — but `rc_alleles_inplace` was **not** in
its target list. This is a follow-up pass closing that gap, using the same round-3
methodology, scoped to the full #251 Rust surface.

### Audit of the full #251 Rust surface

| File | #251 addition | Optimizable? |
|---|---|---|
| `src/variants/mod.rs` | `rc_alleles_inplace` core (67 lines) | **Yes** — the only compute kernel |
| `src/ffi/mod.rs` | `rc_alleles` PyO3 wrapper (17 lines) | No — `as_slice_mut().unwrap()` + 3 `as_array()` borrows, zero-cost boundary glue, no hot loop |
| `src/lib.rs` | registration (1 line) | No |

The wrapper and registration carry no hot loop; the entire optimizable surface is
`rc_alleles_inplace`.

## The inefficiency

Current `rc_alleles_inplace`:

```rust
let mut per_allele = vec![false; n_alleles];           // ① heap alloc + memset every call
for g in 0..to_rc_row.len() { ... per_allele[a]=true }  // ② expand row→allele mask (pass 1)
let per_allele = ndarray::Array1::from_vec(per_allele); // ③ Array1 wrap
crate::reverse::rc_flat_rows_inplace(byte_data, seq_offsets, per_allele.view()); // ④ rescans ALL alleles checking the mask (pass 2)
```

It materializes an intermediate per-allele bool mask only to hand it to a generic helper
that re-scans every allele. Two passes (build mask → scan mask) plus a per-call heap
allocation and memset.

## The change

**One logical change in `src/variants/mod.rs`, with a small extract in `src/reverse.rs`.**

### 1. Shared `#[inline]` reverse+complement helper

Factor the per-row body inside `rc_flat_rows_inplace`'s masked branch — `row.reverse()`
followed by the round-3 branchless-vectorized complement — into:

```rust
#[inline]
pub(crate) fn rc_row(row: &mut [u8]) { /* row.reverse() + vectorized COMP arithmetic */ }
```

`rc_flat_rows_inplace` calls `rc_row` per masked row. Same vectorized complement, DRY.

### 2. Fuse `rc_alleles_inplace` into a single pass

```rust
pub fn rc_alleles_inplace(byte_data, seq_offsets, var_offsets, to_rc_row) {
    for g in 0..to_rc_row.len() {
        if !to_rc_row[g] { continue; }
        for a in var_offsets[g] as usize..var_offsets[g + 1] as usize {
            let s = seq_offsets[a] as usize;
            let e = seq_offsets[a + 1] as usize;
            crate::reverse::rc_row(&mut byte_data[s..e]);
        }
    }
}
```

Deletes the `vec![false; n_alleles]` alloc+memset (①), the `Array1::from_vec` wrap (③),
and the redundant full-allele rescan (④); collapses the two passes into one. `n_alleles`
is no longer computed.

### Byte-identity argument

`var_offsets` partition the alleles by row (contiguous, disjoint), so each allele belongs
to exactly one row. The old code RC'd allele `a` iff its owning row was masked; the fused
loop RCs exactly that set, in the same order (rows ascending, alleles ascending within a
row). Empty allele (`s == e`) → `rc_row` on an empty slice is a no-op; empty row
(`a0 == a1`) → inner loop skips. Behavior is identical to today on every input.

### Risk control on the shared kernel

`rc_flat_rows_inplace` sits on the round-3-tuned haplotype hot path. The `#[inline]`
extract must leave its codegen equivalent. **Gate:** confirm `rc_flat_rows_inplace`'s asm
is unchanged/equivalent after the extract. If extraction perturbs it, fall back to
duplicating the ~6-line complement locally in `rc_alleles_inplace` and leave
`rc_flat_rows_inplace` byte-for-byte untouched. DRY is preferred but never at the cost of
regressing the tuned kernel.

## Gate (parity + instruction-count drop + no regression)

This path (`rc_alleles` fires only on negative-strand variants / `RaggedVariants` reads)
is noise-dominated in wall-clock per the roadmap, so the gate is **not** round-3's strict
"improve throughput or revert." Keep the change iff:

1. **Parity byte-identical, both backends:** `tests/parity/test_rc_alleles_parity.py` +
   cargo unit tests (`rc_alleles_*` in `variants`, `reverse` module tests).
2. **Instruction count drops:** `cargo asm --rust genvarloader::variants::rc_alleles_inplace`
   before/after — record the delta as evidence (the deterministic win).
3. **No throughput regression:** `profile.py --mode variants` rust÷numba **holds**
   (same session, both backends); not required to improve.
4. **`rc_flat_rows_inplace` asm equivalent** after the extract (risk control above).

Plus the standard full gate: full pytest tree on both backends, `cargo test`,
`ruff check`/`format`, `typecheck`, abi3 wheel build.

## Process

Round-3 precedent: worktree off `rust-migration` with its **own** fresh pixi env (never
symlink `.pixi` — `maturin develop` repoints the shared env), one commit for the kernel +
roadmap update, PR into `rust-migration` (**no squash merge**). Update the roadmap under
the Target-6 / round-3 area noting `rc_alleles_inplace` was tuned (instr before→after,
rust÷numba held).

## Out of scope

No on-disk format change, no public API change, no new kernels, no rayon/batch
parallelism (Phase 5), no numba/seqpro-reference deletion (Phase 5). No change to
`flank_tokens` or `_FlatVariantWindows` (never RC'd).
