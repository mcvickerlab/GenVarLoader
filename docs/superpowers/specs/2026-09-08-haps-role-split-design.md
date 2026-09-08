# Splitting the `Haps` role from the SVAR1 implementation

Status: approved 2026-09-08. Supersedes the placeholder-shrinking workaround in
PR #356 (which stays as a hotfix; see "Relationship to PR #356").

## Problem

`Haps` fuses three unrelated jobs into one class. Only two of them are shared
across variant backends.

| # | Job | Consumers | Where it lives today |
|---|-----|-----------|----------------------|
| 1 | **Role marker** — "these seqs are variant-aware" | ~30 `isinstance(self._seqs, Haps)` / `case Haps()` sites across `_impl.py`, `_open.py`, `_reconstruct.py`, `_double_buffered_loader.py` | the class identity |
| 2 | **View settings** — `kind`, `filter`, `min_af`/`max_af`, `var_fields`, `flank_length`, `token_lut`/`token_dtype`/`token_alphabet`/`unknown_token`, `window_opt`, `unphased_union`, `dummy_variant` | `Dataset.with_*`, via `dataclasses.replace` | base-class fields |
| 3 | **SVAR1 storage** — `variants: _Variants`, `genotypes: Ragged`, `dosages`, `var_field_data`, `_ffi_static` | the SVAR1 kernels only | base-class fields |

`Svar2Haps` needs jobs 1 and 2. It inherits job 3, so `Svar2Haps.from_path` must
fabricate an empty `_Variants` and a fake `genotypes` `Ragged` purely to satisfy
a data model it never reads. Its own module docstring says so outright: it
"subclasses `Haps` only so the many `isinstance(_, Haps)` / `case Haps()` checks
throughout the dataset machinery keep working; every read method is overridden."

The fabrication is not inert. It leaks into callers, which must then be taught
that the base class's own fields are lies:

1. `_info_field_dtype` (`_impl.py:53-76`) special-cases `Svar2Haps` because
   `variants.info` is permanently empty.
2. The two memory-estimate branches in `Dataset._output_bytes_per_instance`
   (`_impl.py:1437-1830`, "variants" and "variant-windows") each carry an
   `isinstance(haps_obj, Svar2Haps)` fork plus ~10 comment blocks explaining
   which base fields must not be trusted.
3. `Dataset.ploidy`, `Dataset.haplotype_lengths`, and
   `_output_bytes_per_instance` read `genotypes.shape[-2]` — a *storage* detail
   standing in for a *dataset* property.
4. `HapsTracks.__call__` routes on `isinstance(self.haps, Svar2Haps)`
   (`_reconstruct.py:147-162`) before its SVAR1 body.
5. `Dataset.with_var_fields` forks on `Svar2Haps` (`_impl.py:361-403`).
6. `get_variants_flat` (`_flat_variants.py:883`) reads `haps.genotypes`
   directly; it is only ever reached from the SVAR1 `__call__`, but its
   signature claims any `Haps`.

The failure mode this shape invites is silent, not loud: a base method that
`Svar2Haps` forgets to override does not raise — it reads zeros out of the
placeholder and returns a plausible wrong answer. See "Bug found while
designing this" below for a live instance.

PR #356 makes the fake `genotypes` cheap (`_ShapeOnlyGenotypes`, a zero-stride
`broadcast_to` view) rather than removing the need for a fake. That is the right
hotfix and the wrong end state.

## Design

`Haps` becomes an abstract base holding only jobs 1 and 2. Today's `Haps` body
becomes `Svar1Haps`. `Svar2Haps` becomes its sibling rather than its subclass.

```
Haps[_H]  (ABC, Reconstructor[_H])      <- the ~30 isinstance sites keep working verbatim
|-- settings fields (job 2, unchanged, so replace() keeps working)
|-- abstract: __call__, to_kind
|-- abstract query surface (job 1):
|     ploidy                                     -> int
|     n_variants                                 -> NDArray[int32], shape (R, S, P)
|     _haplotype_ilens(idx, regions, deterministic, keep=None, keep_offsets=None)
|     haplotype_lengths_for_plan(idx, regions)
|     measure_variant_payload(idx, regions)      -> (n_vars, ref_span, alt_bytes)
|     var_field_dtype(field)                     -> np.dtype
|     has_ref_alleles                            -> bool
|     available_var_fields                       -> list[str]
|     realign_track_block(...)
|
|-- Svar1Haps(Haps[_H])   owns variants / genotypes / dosages / var_field_data / _ffi_static
`-- Svar2Haps(Haps[_H])   owns store / cache; no placeholders at all
```

`Haps` is **not** in `genvarloader.__all__`, so the rename is internal: no public
API change, and no `skills/genvarloader/SKILL.md` or `docs/source/api.md` update
is required by the CLAUDE.md public-API gate.

### The abstract query surface, derived from actual callers

Each member exists because a caller in `_impl.py`/`_reconstruct.py` asks for it
today by reaching into SVAR1 storage. Nothing is added speculatively.

| Member | Replaces | SVAR1 impl | SVAR2 impl |
|---|---|---|---|
| `ploidy` | `genotypes.shape[-2]` (`_impl.py:1061`, `:1318`, `:1485`, `:1648`) | `genotypes.shape[-2]` | stored `P` from the svar2 cache meta |
| `n_variants` | `genotypes.lengths` set in `__post_init__` | unchanged | real per-(r,s,p) counts, or an explicit documented zero contract |
| `_haplotype_ilens` | already abstract in spirit | unchanged | delegates to existing `_haplotype_diffs` |
| `measure_variant_payload` | the `_allele_bytes_sum` + `variants.start.dtype` open-coding in both estimate branches | new wrapper over existing `_allele_bytes_sum` | **already exists** (`_svar2_haps.py:851`) |
| `var_field_dtype` | `variants.info[f].dtype` + the `Svar2Haps` fork in `_info_field_dtype` | `variants.info[f].dtype` | `store_fields[f].dtype` |
| `has_ref_alleles` | `variants.ref is not None` (`_impl.py:759`) | `variants.ref is not None` | `False` |
| `realign_track_block` | `isinstance(self.haps, Svar2Haps)` in `HapsTracks.__call__` | the current SVAR1 fused body | the current `_call_svar2` body |

`measure_variant_payload` is the keystone: `Svar2Haps` already implements it
precisely because the estimate branches could not trust the placeholders. Making
it part of the role and giving `Svar1Haps` the matching implementation collapses
both `isinstance` forks into one unconditional call.

### What deletes

- `_ShapeOnlyGenotypes` and the whole PR #356 workaround.
- Both `isinstance(haps_obj, Svar2Haps)` forks in `_output_bytes_per_instance`.
- The `Svar2Haps` fork in `_info_field_dtype` and in `with_var_fields`.
- The `isinstance(self.haps, Svar2Haps)` fork in `HapsTracks.__call__`.
- The ~10 comment blocks across `_impl.py` documenting which inherited fields
  are not to be trusted.

### What stays

- `HapsTracks._call_svar2`'s body — a genuine kernel-dispatch difference. It
  moves from an `isinstance` in `HapsTracks` to a `realign_track_block`
  override on each subclass.
- Every `NotImplementedError` guard in `Svar2Haps` (`min_af`/`max_af`,
  annotated haps, splicing, jitter). Those are real capability gaps, not
  artifacts of the class shape.
- `get_variants_flat`, narrowed from `Haps` to `Svar1Haps`.

## Bug found while designing this

`Svar2Haps` implements `_haplotype_diffs` but does **not** override
`_haplotype_ilens`. `Dataset.haplotype_lengths` calls `_haplotype_ilens`
(`_impl.py:1309`), so on an SVAR2 dataset it falls through to the SVAR1 base
implementation, which reads the empty `genotypes` placeholder, finds every
group empty, and returns all-zero length deltas. `haplotype_lengths()` therefore
reports the unadjusted reference span, ignoring indels.

`Dataset._output_bytes_per_instance` consumes `haplotype_lengths()` for the
`"haplotypes"` and `"annotated"` branches, so the same zeros propagate into the
double-buffered slot-size estimate — the sibling of the `"variants"` /
`"variant-windows"` defect already tracked as #315.

This is precisely the silent-wrong-answer failure mode the current class shape
invites, and it is the strongest argument for the split: under an ABC, a missing
override is an instantiation-time `TypeError`, not a plausible zero.

It gets its own issue and its own bottom-of-stack PR so it can land ahead of the
refactor.

## Relationship to PR #356

PR #356 (external contributor) fixes a real 64 GB open-time allocation at All of
Us scale and unblocks their chr19 run today. Recommendation: **merge #356
as-is**, then delete `_ShapeOnlyGenotypes` in the final layer of this stack.
Asking the contributor to absorb this refactor would be unfair scope creep on a
correct bug report.

If #356 has not merged when the final layer is written, that layer deletes the
`Ragged` placeholder instead; the end state is identical either way.

## Implementation: a 4-PR stack

Each layer is independently green and independently reviewable.

1. `fix/svar2-haplotype-lengths-indels` — override `_haplotype_ilens` on
   `Svar2Haps` (delegating to `_haplotype_diffs`) plus a regression test
   asserting `haplotype_lengths()` matches reconstructed lengths on an
   indel-bearing SVAR2 fixture. Small, valuable on its own, merges first.
2. `refactor/haps-query-surface` — add the query surface listed above to `Haps`,
   implement on both classes, switch every caller off SVAR1 internals. No
   rename yet, so the diff is readable. The `isinstance(_, Svar2Haps)` forks
   delete here.
3. `refactor/haps-role-abc` — rename `Haps` -> `Svar1Haps`, hoist the ABC into
   `Haps`, reparent `Svar2Haps` to it. Mostly mechanical.
4. `refactor/haps-drop-placeholders` — delete the fabricated `genotypes` /
   `_Variants` from `Svar2Haps.from_path` and `_ShapeOnlyGenotypes`.

## Testing

This is a behavior-preserving refactor except for layer 1, which fixes a bug.

- Gate for every layer: the full tree, `pixi run -e dev pytest tests -q`. Scoped
  runs skip `tests/unit/` (per CLAUDE.md), which is exactly where the
  reconstructor-construction tests live (`tests/unit/dataset/test_build_reconstructor.py`,
  `tests/unit/test_slot_fit_property.py`).
- The existing SVAR1/SVAR2 parity suite in `tests/dataset/` already pins the
  behavior that must not move.
- New in layer 1: the `haplotype_lengths` regression test described above.
- New in layer 4: assert `Svar2Haps` has no `genotypes`/`variants` attribute at
  all — replacing PR #356's two allocation-size tests, which become moot once
  there is nothing to allocate.
- New in layer 3: round-trip `replace(haps, min_af=...)` on both subclasses, to
  pin that the settings block stays `dataclasses.replace`-able across the
  hierarchy change.
- `tests/unit/test_slot_fit_property.py:85` and several docstrings in
  `tests/dataset/test_svar2_*.py` describe the placeholders; they need updating
  in the layer that removes what they describe.

## Non-goals

- No change to the `.svar2` on-disk format or to any Rust kernel.
- No change to public API, `__all__`, docs, or the `genvarloader` skill.
- Not lifting any `Svar2Haps` capability guard (`min_af`/`max_af`, annotated
  haps, splicing, jitter). Those are separate features.
- Not touching the `StreamingDataset` `StreamBackend` path, which is coordinated
  through the StreamingDataset project board and targets the `streaming` branch.
  This work targets `main`.
