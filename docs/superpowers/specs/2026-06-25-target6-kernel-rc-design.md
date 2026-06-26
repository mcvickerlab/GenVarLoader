# Design — Target 6: fold strand reverse-complement into the Rust read-path kernels

**Date:** 2026-06-25
**Workstream:** Phase 5, Target 6 (rust-migration roadmap, round-2 optimization block)
**Branch:** `opt/target-6-kernel-rc` off `zero-copy-scale-safe-readpath`
**Handoff:** `docs/handoffs/2026-06-25-phase5-getitem-optimization.md` (Target 6 section)

## Goal

Delete the per-batch reverse-complement (RC) post-pass on the read path by emitting
negative-strand regions already reverse-complemented from the Rust kernels. This is the
largest single-thread throughput lever left before rayon, and it is **backend-agnostic**
(numba pays the same cost), so it must land before rayon batch parallelism.

## Corrected cost model (why this design, not the handoff's literal framing)

The handoff calls the RC cost a "numpy post-pass." The code shows otherwise: RC today runs
through seqpro's **compiled** flat kernels (`_reverse_rows_masked` /
`reverse_complement_masked` via `_query.py::reverse_complement_ragged` and
`_flat.py::_Flat.reverse_masked`), not a Python loop. Both backends call the *same* RC code
*after* reconstruction, which is exactly why numba shows the same ~19% self-time on
haplotypes.

Therefore the cost is **the second full-batch traversal of the output buffer** (re-read +
complement + numpy re-wrap), **not** an FFI crossing unique to rust. This rules out a
"rewrite the post-pass in Rust but keep it batch-wide" approach — it would re-read the same
cold buffer and barely move the number.

The chosen approach removes the **cold, batch-wide** traversal: RC each negative-strand
query's slice **in-place, immediately after that query is written, inside the existing
per-query kernel loop**, while the slice is still hot in L1/L2. A second hot pass over a
~16 KB slice is near-noise next to reconstruction; today's cost is high precisely because
the pass is cold, whole-batch, and materialized through numpy.

### Approach considered and rejected

- **A — fold the reversed write into the reconstruct core** (emit bytes already RC'd, no
  second pass at all). Rejected: maximum single-thread perf, but RC logic entangles with
  indel + insertion-fill + trailing-fill in the hottest kernels, is bespoke per output kind,
  and the annotated/splice cases make a subtle parity break likely. Its only gain over the
  chosen approach is eliminating one *hot* pass — not worth the risk. Revisit only if the
  chosen approach's measured ratio still lags numba.
- **C — Rust post-pass called from Python** (replace `reverse_complement_ragged` with one
  Rust pyfunction over the returned flat buffers). Rejected: keeps the exact cold,
  batch-wide traversal; captures neither the cache-locality win nor a meaningful dispatch
  win, since RC is not an extra rust FFI crossing today.

## Scope

In scope — five flat-buffer output kinds, all sharing the in-place primitives:

| Kind | Buffers | RC behavior |
|---|---|---|
| haplotypes (S1) | `out_data: u8` | reverse + complement |
| reference (S1) | `out_data: u8` | reverse + complement |
| tracks (f32) | `out_data: f32` | reverse only (no complement) |
| annotated | `haps: u8`, `var_idxs: i32`, `ref_coords: i32/i64` | haps reverse+complement; both index arrays reverse-only; all three in lockstep per query |
| splice (haps / ref / tracks) | permuted element buffer | same primitive per spliced **element**, using permuted offsets + permuted per-element mask |

Out of scope:

- **`RaggedVariants` (`variants` mode) RC — deferred to Target 7.** Its RC is structurally
  different (reverse allele order within each row **and** complement allele bytes over the
  nested ragged allele structure, `RaggedVariants.rc_`) and lives in the `src/variants/`
  gather path that Target 7 is concurrently rewriting. Target 6 leaves a slimmed
  `reverse_complement_ragged` husk handling only this case; Target 7 absorbs it and deletes
  the husk.
- **`variant-windows` and `intervals`** — reference-oriented, RC is a no-op today and stays a
  no-op.

## Components — Rust primitives

A new small module (`src/reverse.rs`) with two generic in-place primitives, each over a flat
`(data, offsets)` buffer + a per-row `to_rc` mask:

1. `reverse_flat_rows_inplace<T: Copy>(data: &mut [T], offsets, to_rc)` — reverses element
   order within each masked row. Order only, no complement. Generic over element width
   (`u8`, `f32`, `i32`, `i64`).
2. `rc_flat_rows_inplace(data: &mut [u8], offsets, to_rc)` — reverses **and** complements
   bytes via a 256-entry `_COMP` LUT.

**`_COMP` LUT contract:** reproduce `bytes.maketrans(b"ACGT", b"TGCA")`
(`python/genvarloader/_ragged.py:330`) exactly — a `[u8; 256]` that is **identity for
everything** except `A↔T` and `C↔G` (uppercase only). `N`, IUPAC codes, and lowercase
`a/c/g/t` are pass-through (identity), matching today's behavior byte-for-byte.

Output-kind → primitive mapping:

- haplotypes, reference → `rc_flat_rows_inplace`
- tracks → `reverse_flat_rows_inplace::<f32>`
- annotated → `rc_flat_rows_inplace` on `haps`; `reverse_flat_rows_inplace` on `var_idxs`
  and `ref_coords`; applied in lockstep per query.
- splice → the relevant primitive per spliced element.

## Mask threading & per-kernel integration

The `to_rc` mask is **computed in Python and passed into each kernel** as a new
`Option<PyReadonlyArray1<bool>>` argument. Rationale: the strand→mask logic and (critically)
the splice permutation logic already exist and are tested; reproducing the permutation in
Rust would be gratuitous risk.

- **Unspliced kernels** (`reconstruct_haplotypes_fused` `src/ffi/mod.rs:393`,
  `reconstruct_annotated_haplotypes_fused` `:604`, `intervals_and_realign_track_fused`
  `:848`, `get_reference` `:728`): Python passes `to_rc = full_regions[r_idx, 3] == -1`
  (one bool per query). The kernel applies the primitive to query `k`'s just-written slice
  when `to_rc[k]`.
- **Spliced kernels** (`reconstruct_haplotypes_spliced_fused` `:521`, the spliced-reference
  fetch `_fetch_spliced_ref` / reference core): Python passes the **already-permuted
  per-element** mask — the existing `to_rc_per_elem` (`_query.py:259-280`) / `to_rc_perm`
  (`_reference.py:438-444`) computation moves from post-pass input to kernel input,
  unchanged. The spliced kernel's loop is already per-element over permuted `out_offsets`,
  so the primitive applies per element with no new boundary math. **Assert** the element
  boundaries being RC'd match `plan.group_offsets` (handoff warning).

**`Option` keeps the fast path trivially byte-identical:** when `rc_neg` is off or no
negative-strand region is selected (`to_rc.any() == false`), Python passes `None` and the
kernel does zero extra work. All-positive datasets are provably unchanged; existing fixtures
and the scale guard cannot regress.

**Insertion-fill / trailing-fill ordering preserved for free:** RC runs *after* a query's
full forward write (fills already placed), so it sees the exact final post-fill bytes the
current post-pass sees. No interleaving with fill logic.

**Rust files touched:** `src/ffi/mod.rs` (6 kernel signatures + call sites), the
reconstruct/track/reference cores under `src/{reconstruct,tracks,intervals,reference}/`, and
the new `src/reverse.rs` (with cargo unit tests).

## Python-side changes & deletion plan

- **`_query.py::_getitem_unspliced`** (`:188-190`): delete the
  `reverse_complement_ragged` post-pass; compute `to_rc` and thread it through
  `view.recon(...)` into the kernels. Only the deferred `RaggedVariants` case still routes
  through the husk.
- **`_query.py::_getitem_spliced`** (`:259-280`): keep the permuted `to_rc_per_elem`
  computation, but hand its result to the kernel via the splice plan / recon call instead of
  to `reverse_complement_ragged`.
- **`_query.py::reverse_complement_ragged`** (`:374-410`): shrink to the **husk** — only the
  `RaggedVariants` branch survives (`return rag.rc_(to_rc)`); delete the `_Flat`,
  `_FlatAnnotatedHaps`, and no-op branches. Add `# TODO(target-7)` noting Target 7 absorbs
  and deletes it.
- **`_reference.py`** (`:438-444`): delete the spliced-reference
  `per_elem.reverse_masked(to_rc_perm, comp=_COMP)` post-pass; thread `to_rc_perm` into
  `_fetch_spliced_ref` / the reference kernel. (Third RC site, missed by the handoff, now
  in-scope.)
- **Reconstructors** (`Haps`, `Ref`, `Tracks`, `HapsTracks`, `SeqsTracks`, annotated) gain a
  `to_rc` parameter on their recon entry that they forward to the FFI kernel. Exact signature
  confirmed when reading `_reconstruct.py`; principle: mask flows region-compute → recon →
  kernel, and the only Python RC left anywhere is the variants husk.
- **No stray callers:** `grep -rn reverse_complement_ragged python/` and
  `grep -rn reverse_masked python/` confirm nothing else depends on the deleted paths.

## Parity, tests & perf gate

**Primary risk: vacuous parity pass.** Default fixtures use `max_jitter=0` and may be
all-positive-strand, so RC code could never fire and parity would pass trivially. Guards:

- **New strand=−1 fixtures** in `tests/parity/test_dataset_parity.py`: datasets mixing `+`
  and `−` regions, covering every in-scope kind (haplotypes, reference, tracks, annotated)
  and the spliced variant of each. Reuse the kernel-spy backstop to prove RC executes on the
  live `__getitem__` path.
- **Non-vacuity assertion:** for a `−`-strand region, assert output bytes ≠ the `+`-strand
  orientation (RC genuinely fired), and assert exact RC'd bytes for a known fixture.
- **Rust unit tests** (`src/reverse.rs`): empty rows, single byte, odd/even lengths,
  `to_rc` all-false (no-op) / all-true / mixed; LUT identity on `N`/lowercase/IUPAC; `f32`
  reverse-only; lockstep reversal of the three annotated buffers.

**Parity gate (byte-identical vs current post-pass), both backends:**

```bash
pixi run -e dev cargo-test
pixi run -e dev pytest tests/parity -q                       # rust default
GVL_BACKEND=numba pixi run -e dev pytest tests/parity -q      # oracle
```

**TDD order:** reference (simplest, no fill) → haplotypes → tracks (reverse-only) →
annotated → **splice last**. Land each kind behind parity before deleting its Python
post-pass branch. Variants deferred.

**Before push:** full tree both backends (`pixi run -e dev pytest tests -q`, then
`GVL_BACKEND=numba …`) to catch `tests/unit/` references to deleted code; lint/format/
typecheck on `python/ tests/`.

**Perf gate:** re-measure `haplotypes`, `tracks-only`, `tracks-seqs`, `annotated` via the
de-noised `tests/benchmarks/test_e2e.py` harness (min over `pedantic(iterations=10,
rounds=50)`, release build). Expect the RC self-time gone from `perf` flat profiles and the
rust÷numba ratios up (haplotypes was 0.94× with RC its biggest sink at ~19% self). Record
re-measured ratios in `docs/roadmaps/rust-migration.md` under the Phase 5 round-2 block,
tick Target 6, set the PR link, and set the marker that Target 6 must merge before rayon.

**HPC gotcha:** run pytest with `--basetemp=$(pwd)/.pytest_tmp` so the write path's `os.link`
hardlink does not fail cross-device (Errno 18). Work in a dedicated git worktree.

## Coordination with parallel workstreams

- **Target 7** (variants/windows assembly): owns the deferred `RaggedVariants.rc_` port and
  the `reverse_complement_ragged` husk deletion. Overlaps Target 6 in `src/ffi/mod.rs`
  (additive — new pyfunction args vs new pyfunctions, low conflict).
- **Target 5** (intervals slicing): overlaps `src/intervals.rs`; merge order is 5 first, then
  6/7. Rebase Target 6 onto 5 if 5 lands first.
- **Rayon** is blocked until 5 + 6 + 7 are on the base branch. The in-loop, per-query RC of
  this design parallelizes cleanly (disjoint per-query slices).
