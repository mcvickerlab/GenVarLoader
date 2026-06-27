# Design: Rust migration Phase 2 — Genotype assembly + variant gather

**Date:** 2026-06-24
**Roadmap:** `docs/roadmaps/rust-migration.md` (Phase 2)
**Status:** approved design, pre-implementation

## Context

Phases 0 (foundation + `intervals_to_tracks` proof-point) and 1 (ragged primitives
via `seqpro-core`) have landed. Phase 2 is the next bottom-up step: migrate the
genotype assembly/selection kernels and the flat variant-gather kernels from
numba to the Rust crate, following the strangler-fig + byte-identical-parity
contract established in Phase 0.

## Scope

### Port (live kernels)

From `python/genvarloader/_dataset/_genotypes.py`:
- `get_diffs_sparse` — per-`(query, hap)` reference-length diffs; called from
  `_haps.py:474` for haplotype-length sizing.
- `choose_exonic_variants` (+ inner `_choose_exonic_variants`) — keep-mask for
  variants fully contained in a query interval; called from `_haps.py`
  (spliced/exonic path).

From `python/genvarloader/_dataset/_flat_variants.py` (7 kernels, variants output
mode only — driven by `get_variants_flat`, not the default tracks/haps getitem):
- `_gather_v_idxs`, `_gather_v_idxs_ss` — gather variant indices for contiguous
  `(n+1,)` and non-contiguous `(2, n)` offset forms.
- `_gather_alleles` — two-level allele-byte gather.
- `_compact_keep` — compact a flat buffer + offsets under a keep mask.
- `_fill_empty_scalar`, `_fill_empty_seq`, `_fill_empty_fixed` — dummy-variant
  fill for empty `(region, sample, ploid)` groups (scalar / bytestring /
  fixed-inner-stride).

### Delete (dead kernel)

- `filter_af` (`_genotypes.py`) — superseded by inline numpy AF filtering in
  `_haps.py:734-737` and `_flat_variants.py:698-701`; **zero callers**. This is the
  same dead-code situation as the Phase 0 `splits_sum_le_value` pivot. Removed in
  this PR rather than ported.

### Phase boundary fix

The roadmap text "`_genotypes.py` kernels (6 numba)" double-counts the two
reconstruction kernels (`reconstruct_haplotypes_from_sparse`,
`reconstruct_haplotype_from_sparse`) that live in `_genotypes.py` but belong to
**Phase 3** (next to `_reconstruct.py`/`_haps.py`, where the big read-path win is
measured as one unit). Phase 2 covers assembly/selection only. The roadmap is
updated to remove the double-count.

## Architecture

Follows the Phase 0 seam (`src/ffi/` is the only place touching PyO3; core logic
in lazily-grown pure-`ndarray` domain modules).

- New domain modules: `src/genotypes/mod.rs` (assembly/selection) and
  `src/variants/mod.rs` (flat gather/fill). Pure `ndarray`, no PyO3.
- All PyO3 wrappers in `src/ffi/`, mirroring the `intervals_to_tracks` pattern.
- **FFI signatures mirror the numba signatures exactly** — same inputs, same
  `(data, offsets)`-tuple returns. Python keeps wrapping results into
  `seqpro.rag.Ragged` / `keep_offsets` exactly as today, so dispatch is a drop-in
  swap and parity is byte-identical.
- **Both offset forms**: handle 1-D `(n+1,)` and 2-D `(2, n_slices)` `geno_offsets`
  (windowed/sliced queries) — both branches exist in the numba kernels.
- **Parallelism**: sequential first. Per-`(query, hap)` writes are disjoint
  (`diffs[q,h]`, `keep[k_s:k_e]`), so sequential output is byte-identical to
  numba's `prange` — same argument as the Phase 0 proof-point. Add `rayon` only if
  the no-regression gate requires it.

## Dispatch & strangler-fig contract

- Register each ported kernel in `python/genvarloader/_dispatch.py` (per-kernel
  default `rust`, `GVL_BACKEND` global override), routing the call sites in
  `_haps.py` / `_flat_variants.py`.
- Keep the numba impls as the parity reference until the phase closes, then delete
  them + the switch in the same bundled PR (per the migration contract).
- `filter_af` is deleted immediately (dead, nothing to keep as a reference).

## Testing

Extends the Phase 0 harness (`tests/parity/`).

- **Per-kernel hypothesis parity gates** — run-both-assert-byte-identical,
  covering the branch matrix:
  - `get_diffs_sparse`: 1-D vs 2-D offsets; `keep`/`keep_offsets` present/absent;
    the `q_starts`/`q_ends`/`v_starts` query-clipping path; empty groups.
  - `choose_exonic_variants`: 1-D vs 2-D offsets; empty groups; variants partially
    vs fully contained in the interval.
  - flat kernels: contiguous vs non-contiguous gather; keep-mask compaction;
    empty-group fill for scalar / seq / fixed fields.
- **New variants-mode dataset-level backstop** with a kernel spy (mirrors the
  tracks-mode backstop). Variants mode (`with_seqs("variants")`) has no
  differential coverage today; this is genuinely new and asserts the Rust kernels
  are actually invoked (no vacuous pass — the lesson baked in after the splits
  backstop).
- `cargo test` units per kernel.

## Gate & measurement

Gate = **parity + no regression** (per decision; the dramatic read-path speedup is
Phase 3's, not Phase 2's — these kernels are cheap index-math and buffer gathers).

- Parity green across py310–313 × linux/macOS.
- No `__getitem__` throughput regression on `chr22_geuv`:
  - `profile.py --mode haps` vs baseline **123.9 batch/s** (exercises
    `get_diffs_sparse` + `choose_exonic_variants`).
  - `profile.py --mode variants` vs baseline **145.3 batch/s** (exercises the flat
    gather/fill kernels).
- abi3 wheel still builds (standing CI invariant).
- Record any incidental wins (kernel count down by 3 incl. the dead `filter_af`;
  reduced JIT warmup / RSS).

## Sequencing (one bundled PR)

Internal beachhead order: genotypes-first, then variants.

1. `get_diffs_sparse` → Rust + ffi + dispatch + parity gate.
2. `choose_exonic_variants` (+ inner) → same loop.
3. Delete dead `filter_af`.
4. The 7 `_flat_variants.py` kernels → Rust + ffi + dispatch + parity gates +
   variants-mode backstop.
5. Flip defaults, delete numba impls + switch, measure, update roadmap.

## Roadmap update (part of the PR)

- Fix the Phase 2 double-count (reconstruction kernels → Phase 3).
- Mark `filter_af` deleted-as-dead.
- Note the variants-mode gate uses the variants baseline (145.3 batch/s).
- Record decisions in the notes log; set the Phase 2 status marker + PR link;
  record measurements.

## Non-goals

- Reconstruction kernels (`reconstruct_*`) — Phase 3.
- Track realignment, reference, insertion-fill, splice — Phase 3.
- Write/update pipeline — Phase 4.
- Any rayon parallelism unless the no-regression gate forces it.
