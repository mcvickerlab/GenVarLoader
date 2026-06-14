# Design: ploidy-1 unphased *union* `variant-windows` view (issue #222)

**Repo:** `mcvickerlab/GenVarLoader` · **Date:** 2026-06-14 · **Issue:** #222
**Companion to:** #214 (flat mode, landed), #221 (fetch overhead)

## Problem

The flat `variant-windows` output mode is ploidy-aware: it emits per-`(variant,
haplotype)` windows over `P = genotypes.shape[-2]` slots. For **somatic** data
stored as diploid genotypes (`ploidy=2`), this breaks haploid modeling. Somatic
mutations are not a diploid genotype — they are "present on the tumor genome" —
and the consumer (`gvf-germ-som`) models them as haploid (`PLOIDY=1`). The
diploid windows split the called ALTs across 2 haplotypes, and genvarformer's
`IntraGenicVarGeneEncoder` requires `model_ploidy == data_ploidy`, producing a
shape mismatch under `PLOIDY=1`.

The two haplotypes are **not** redundant: on the reference dataset, somatic ALTs
sit overwhelmingly on hap-1 (5,226 occurrences) vs hap-0 (29) across chr21, so
"just take haplotype 0" loses ~all variants. A correct haploid view must
**union** the called ALTs across both stored haplotypes.

Today `ploidy` is read straight from the svar genotypes (`genotypes.shape[-2]`)
and is not configurable on `Dataset.open` / `with_settings`.

## Goal

Add an opt-in **unphased ploidy-1 view** that folds variant occurrences across
the stored haplotypes onto a single haploid sequence (union of called ALTs per
`(region, sample)`), so `n_variants(...).shape[-1] == 1` and the
windows / coding-annotation decode at ploidy 1. The stored genotypes remain
diploid on disk; this is a read-time view.

## Decisions (from brainstorming)

- **Scope:** applies to **all variant outputs** — `variant-windows`, flat
  `variants`, and `n_variants`. Phased haplotype-sequence reconstruction stays
  ploidy-2 and **errors** under the flag (a union of phased sequences is
  ill-defined).
- **API:** opt in via `with_settings(unphased_union=True)`; also accepted on
  `Dataset.open`. The stored on-disk `ploidy` is unchanged.
- **Ploidy reporting:** under `unphased_union`, `ds.ploidy` reports the
  **effective** ploidy `1` (chosen for coherence so downstream needs no special
  case). The stored ploidy remains in metadata.
- **Union semantics:** order and dedup **do not matter**. Use a **naive
  combine** — concatenate hap-0's calls then hap-1's, **no sort, no dedup** —
  whatever is fastest. The downstream consumer (genvarformer transformer) is
  permutation-invariant. A hom call therefore appears twice; this is
  intentional and documented.
- **Drop the retired CCF path:** `RaggedVariants.infer_germline_ccfs_` and the
  `_infer_germline_ccfs` numba kernel assume start-ordering. That research
  direction (simultaneous germline + somatic) is retired and unused for ~1 year.
  Remove it; it is the only order-sensitive consumer and would otherwise
  conflict with the no-sort union.

## Mechanism

### Union fold (variant-windows + flat variants)

The gather in `python/genvarloader/_dataset/_flat_variants.py` already produces
variant rows in **C-order `(b, ploidy)`**: for each region/sample, hap-0's
variants then hap-1's, contiguously, with `row_offsets` of length
`b*ploidy + 1`.

Under `unphased_union`, collapse to `b` rows by taking `row_offsets[::ploidy]`
plus the final offset. This concatenates hap-0-then-hap-1 with no sort and no
dedup; the underlying `v_idxs` buffer is untouched — only the offset grouping
changes. Output shape goes from `(b, ploidy, None, None)` → `(b, 1, None, None)`.

AF filtering (`keep` / `_compact_keep`) runs **before** the collapse, unchanged —
it is per-variant and order-independent.

The same collapse applies to the flat `"variants"` (non-window) path.

**Documented order:** within the single haploid window set, variants appear in
stored-haplotype order (hap-0's calls, then hap-1's), unsorted, undeduplicated.

### `n_variants`

Currently `(regions, samples, ploidy)` from `ak.num(genotypes, -1)`. Under
`unphased_union`, sum over the ploidy axis → `(regions, samples, 1)`, surfaced at
the `Dataset` level (`_impl.py`). The on-disk `Haps.n_variants` is untouched.

### Phased haplotype paths

`with_seqs("haplotypes")` and `with_seqs("annotated")` raise a clear
`ValueError` under `unphased_union` (e.g. "haplotype reconstruction is
incompatible with unphased_union; use variant-windows or variants"). Reference-
only and track output are unaffected.

### Drop CCF machinery

Remove `RaggedVariants.infer_germline_ccfs_` and `_infer_germline_ccfs` from
`python/genvarloader/_dataset/_rag_variants.py`, and the corresponding unit
tests in `tests/unit/ragged/test_rag_variants.py`. Grep confirms these are never
auto-invoked in the read path.

## Affected code

| Area | File | Change |
|------|------|--------|
| Setting plumbing | `_dataset/_impl.py` (`Dataset`, `open`, `with_settings`) | new `unphased_union` field/param; forward to `Haps` |
| Ploidy property | `_dataset/_impl.py` (`Dataset.ploidy`) | return 1 when flag on |
| `n_variants` | `_dataset/_impl.py` (`n_variants`) | sum over ploidy axis when flag on |
| Union fold | `_dataset/_flat_variants.py` (`get_variants_flat`) | collapse `(b, ploidy)` → `(b, 1)` offsets |
| Phased guard | `_dataset/_impl.py` (`with_seqs`) | raise on `haplotypes`/`annotated` + flag |
| CCF removal | `_dataset/_rag_variants.py`, `tests/unit/ragged/test_rag_variants.py` | delete `infer_germline_ccfs_` / `_infer_germline_ccfs` + tests |
| Skill | `skills/genvarloader/SKILL.md` | document `unphased_union` (public API change) |

## Acceptance criteria

1. With `unphased_union=True`, `n_variants(...).shape[-1] == 1` and
   `variant-windows` decode emits one window set per `(region, sample)` = the
   combined called ALTs across stored haplotypes.
2. genvarformer's `IntraGenicAnn` detects `P=1`, so `IntraGenicVarGeneEncoder`
   runs with model `ploidy=1` (no `ctx.ploidy` vs `P_data` mismatch).
3. The combined count equals `n_hap0 + n_hap1` (naive combine, no dedup).
   Byte-equivalence to the OLD `with_seqs("variants")` unphased set is validated
   by `gvf-germ-som` against their guardrail (out of scope here; we expose the
   view).
4. `with_seqs("haplotypes")` / `"annotated"` raise `ValueError` under the flag.
5. `ds.ploidy == 1` under the flag.

## Testing

- Synthetic diploid dataset with disjoint ALTs on hap-0 / hap-1 (mirrors chr21
  census). Assert `n_variants(...).shape[-1] == 1`, combined count
  `== n_hap0 + n_hap1`, `ds.ploidy == 1`.
- Variant-windows decode emits one window set per `(region, sample)` carrying all
  ALTs from both haplotypes.
- `ValueError` on `with_seqs("haplotypes")` + `unphased_union`.
- AF filter (`min_af`/`max_af`) + `unphased_union` compose correctly.

## Out of scope / non-goals

- Arbitrary configurable ploidy (only ploidy-1 union is meaningful).
- Re-writing stored genotypes; this is a read-time view.
- The byte-equivalence guardrail itself (owned by `gvf-germ-som`).
- Sorting / dedup within the unioned set (consumer is permutation-invariant).

## Notes

- The old "ploidy=1 corrupted the heap" comment in genvarformer was an unrelated
  `nb.prange` write-race in a since-deleted cost-model kernel — not an obstacle.
- Stopgap on the consumer side (model `PLOIDY=2`, average the 2 haplotype
  embeddings) works but dilutes the somatic-haplotype signal ~50%; this view
  removes the need for it.
