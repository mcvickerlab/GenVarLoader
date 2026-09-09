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
| `check_track_realign_support` + `track_realigner` | `isinstance(self.haps, Svar2Haps)` in `HapsTracks.__call__` | the guards + fused kernel call from the current SVAR1 body | the guards from `_call_svar2` + its existing `realign_track_block` |

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

- The kernel-dispatch difference `HapsTracks._call_svar2` existed for. It moves
  from an `isinstance` in `HapsTracks` onto per-class `track_realigner`
  overrides; the duplicated orchestration around it does not survive.
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

#356 had not merged when the final layer was written, so layer 5 deletes the
plain `Ragged` placeholder instead of `_ShapeOnlyGenotypes`. The end state is
identical either way, and #356 can be closed unmerged once this stack lands:
there is no longer an allocation for it to shrink.

## Implementation: a 5-PR stack

Each layer is independently green and independently reviewable. (Planned as
four, then six, and landed as five. The query surface split cleanly into a cheap
"read scalars off the reconstructor" layer and a heavier "measure the variant
payload" layer, and splitting them kept each diff reviewable. Then the track
dispatch had to move *below* the ABC hoist rather than into it -- see layer 4.
Finally the placeholder deletion turned out to be inseparable from the hoist --
see layer 5.)

1. `fix/svar2-haplotype-lengths-indels` — override `_haplotype_ilens` on
   `Svar2Haps` (delegating to `_haplotype_diffs`) plus a regression test
   asserting `haplotype_lengths()` matches reconstructed lengths on an
   indel-bearing SVAR2 fixture. Small, valuable on its own, merges first.
2. `refactor/haps-query-surface` — add the cheap scalar members
   (`stored_ploidy`, `has_ref_alleles`, `var_field_dtype`) to `Haps`,
   implement on both classes, switch every caller off SVAR1 internals.
   `_info_field_dtype` deletes here. No rename yet, so the diff is readable.
3. `refactor/haps-payload-measure` — add `measure_variant_payload`,
   `ref_allele_bytes` and `prepare_var_fields`; implement on `Haps`. The two
   `isinstance(haps_obj, Svar2Haps)` estimate forks and the `with_var_fields`
   fork all delete here, and `_impl.py` stops importing `Svar2Haps`.
4. `refactor/haps-track-dispatch` — collapse `HapsTracks.__call__`'s two arms
   into one body that reaches the backend only through
   `Haps.check_track_realign_support` and `Haps.track_realigner` (returning a
   per-batch `TrackRealigner`). `_call_svar2` and the last
   `isinstance(self.haps, Svar2Haps)` delete here, and `_reconstruct.py` stops
   importing `Svar2Haps`. `has_dosages` joins the query surface, retiring the
   last `_impl.py` reach into SVAR1 storage.
5. `refactor/haps-role-abc` — rename `Haps` -> `Svar1Haps`, hoist the ABC into
   `Haps`, reparent `Svar2Haps` to it, and delete the fabricated `genotypes` /
   `_Variants` from `Svar2Haps.from_path` in the same commit. Mechanical outside
   `_haps.py`, because after layer 4 nothing else touches SVAR1 storage except
   `get_variants_flat`.

## Testing

This is a behavior-preserving refactor except for layer 1, which fixes a bug.

- Gate for every layer: the full tree, `pixi run -e dev pytest tests -q`. Scoped
  runs skip `tests/unit/` (per CLAUDE.md), which is exactly where the
  reconstructor-construction tests live (`tests/unit/dataset/test_build_reconstructor.py`,
  `tests/unit/test_slot_fit_property.py`).
- The existing SVAR1/SVAR2 parity suite in `tests/dataset/` already pins the
  behavior that must not move.
- New in layer 1: the `haplotype_lengths` regression test described above.
- New in layer 5 (`tests/unit/dataset/test_haps_role_split.py`): `Haps` is
  abstract and uninstantiable; neither implementation is left abstract; the role
  declares no SVAR1 storage field and `Svar2Haps` has no such *attribute* at all
  (`slots=True` makes that stronger than a field check); and the settings block
  stays `dataclasses.replace`-able on both sides of the hierarchy. The first two
  replace PR #356's allocation-size tests, which become moot once there is
  nothing to allocate.
- `tests/unit/test_slot_fit_property.py:85` and several docstrings in
  `tests/dataset/test_svar2_*.py` describe the placeholders; they need updating
  in the layer that removes what they describe.

## Found while implementing layer 3

`Haps.measure_variant_payload` returns the *raw* on-disk variant count while
returning post-AF-filter spans and byte sums beside it. That asymmetry is not a
design choice — it reproduces the accounting `_output_bytes_per_instance` has
always used, because the raw count's over-charge under AF filtering is
currently the only thing covering a constant per-offsets-array deficit
elsewhere in the estimate.

Every offsets array `write_chunk` serializes has `n_groups + 1` entries; the
estimate charges `n_groups`, losing `OFF` bytes per array. Filed as **#362**,
with the arithmetic and a reproduction. Tightening the count before that lands
turns three `tests/unit/dataset/test_output_bytes_dummy_variant.py` cases red
(`estimated=7696 < actual=7728`, deficit 32 = 4 x 8). Fixing #362 first, then
tightening, is the correct order; both are out of scope for this stack.

## Found while implementing layer 4

Two things in the layer-4 plan above were wrong, and reordering the stack was
the fix.

**`realign_track_block` was already taken.** The design named the new
track-dispatch override after a method that already exists on `Svar2Haps`
(`_svar2_haps.py`) as a *lower-level kernel helper* — the thing `_call_svar2`
called inside its per-track loop. The role-level member is now
`Haps.track_realigner`, returning a per-batch `TrackRealigner`; the SVAR2
kernel helper keeps its name.

**The dispatch had to move below the ABC hoist, not into it.** After layer 3,
`_reconstruct.py` still read `haps.genotypes` and `haps.ffi_static` inside
`HapsTracks.__call__`'s SVAR1 arm, and `_impl.py` still read `haps_obj.dosages`.
Hoisting the ABC first would have left those as type errors against the role,
to be papered over with `cast`/`assert isinstance` and then unpicked again one
layer later. Doing the dispatch first means the hoist touches only class
headers and field placement.

**Why a `TrackRealigner` object rather than a per-track hook.** The obvious
shape — one abstract "fill this track's block" method — would have re-run
`_as_starts_stops(self.genotypes.offsets)` once per track. That array is
`(2, regions*samples*ploidy)`, and the old code deliberately hoisted it out of
the loop. Caching it on the reconstructor instead would pin a per-sample-scale
allocation for the dataset's lifetime. Splitting the hook into "prepare once
per batch" + "fill once per track" preserves the hoist by construction, and
gives the per-batch state a name and a type.

## Found while implementing layer 5

**Layers 5 and 6 are one layer.** The plan had the ABC hoist and the placeholder
deletion as separate PRs, on the theory that the placeholders could keep
existing for a commit after `Svar2Haps` was reparented. They cannot. Both
classes are `slots=True` dataclasses, so reparenting `Svar2Haps` onto a role
that does not declare `genotypes` / `variants` / `dosages` / `var_field_data`
removes those slots outright — there is nowhere left for a fabricated value to
be stored. Deleting them is not a follow-up to the hoist; it *is* the hoist. The
stack is five PRs, not six.

**`kw_only=True` is what makes the split possible at all.** `Svar1Haps` adds
four required fields, and they follow the role's defaulted ones (`var_fields`,
`dummy_variant`, the token block, ...). Without `kw_only` that is a
`TypeError` at class creation, and the workaround — giving every storage field a
meaningless default — is exactly the shape that let the placeholders exist. It
costs nothing here because every construction site was already all-keyword.

It also pays for itself immediately on the other side: `Svar2Haps.store` and
`.cache` were `X | None = None` purely to satisfy the same ordering rule, with
three `assert ... is not None` lines apologising for it downstream. Keyword-only
fields let both become required, and the asserts delete.

**`Svar2Haps` needs `n_regions` / `n_samples` of its own.** They were the last
thing the placeholder `genotypes` was really carrying: `_gathered_groups`
unravels a flat dataset index against `genotypes.shape[:2]`. `from_path` already
computed both from the range cache and threw them away. They are now fields.

**`n_variants` stays, and stays zero.** It is on the role rather than the SVAR1
side because `_impl.py` reads `.n_variants.shape[-1]` for ploidy on any `Haps`.
`Svar2Haps.__post_init__` now sets it explicitly to a documented
`zeros((R, S, P))` rather than inheriting zeros as a side effect of an empty
placeholder — same value, but no longer an accident. #363 tracks making it
correct.

## Non-goals

- No change to the `.svar2` on-disk format or to any Rust kernel.
- No change to public API, `__all__`, docs, or the `genvarloader` skill.
- Not lifting any `Svar2Haps` capability guard (`min_af`/`max_af`, annotated
  haps, splicing, jitter). Those are separate features.
- Not touching the `StreamingDataset` `StreamBackend` path, which is coordinated
  through the StreamingDataset project board and targets the `streaming` branch.
  This work targets `main`.
