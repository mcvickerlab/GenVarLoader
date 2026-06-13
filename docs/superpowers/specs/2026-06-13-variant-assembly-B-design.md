# Variant assembly (B) — design: always-flat decode + empty-region dummy padding

**Date:** 2026-06-13
**Repo:** `mcvickerlab/GenVarLoader` (gvl)
**Driven by:** gvl issue #214 (super-batch / flat-buffer output mode), sub-project **B**.
Consumer: genvarformer `Variants` source (`tokens.py`).
**Status:** approved design; ready for `writing-plans`.

Builds directly on sub-project **A** (flat variant decode, PR #215):
`docs/superpowers/specs/2026-06-13-flat-output-mode-design.md`.
Sibling sub-project **C** (flank fetch + tokenization) is independent and specced separately.

---

## 0. Context

Issue #214 decomposes a flat-buffer / super-batch output mode into sub-projects A0, A, B,
C, D, E. **A** (merged) made `get_variants_flat` produce pure-numpy `_FlatVariants` with zero
awkward on the hot path, byte-identical (via `.to_ragged()`) to the legacy awkward
`_get_variants`. The byte-identity gate is the committed snapshot
`tests/dataset/_snapshots/variants_ragged.npz`.

The design doc's original **B** was "absorb variant assembly into gvl: phased+unphased ploidy
merge, within-group sort, dummy-variant / empty-region padding." During brainstorming the scope
was deliberately trimmed:

- **Phased+unphased sorted-merge — deferred.** No project in the last year has needed phased
  and unphased variants in the same input, and none is emerging. Adding a multi-source genotype
  concept to gvl is unjustified now.
- **Within-group sort — deferred (falls out of the above).** Verified in genvarformer
  (`src/genvarformer/data/sources/tokens.py:526–533`): `_merge_phased_unphased_numba`
  (the `expand_groups_*` + `concat_*` kernels) runs **only** when both phased and unphased
  datasets are present (`len(rag_vars) == 2`), and `_sort_by_start_numba`
  (`segmented_argsort` + `take_*`) runs **only** inside that branch. The single-source path
  calls `infer_germline_ccfs_()` with no preceding sort, relying on gvl's native
  per-group sorted order. So a within-group sort is needed *only* to repair the order that
  `concat` breaks during the dual-source merge — i.e. only the deferred case needs it.

What remains of B is therefore: **(B1)** finish what A started by making the flat decode the
*single* variant decode path, and **(B2)** add optional **empty-region dummy padding**.

---

## 1. Goal & scope

Two coupled changes to gvl's variant-record output (`with_seqs("variants")`):

- **B1 — always-flat variant decode (refactor).** Make `get_variants_flat` the single decode.
  Retire the awkward `_get_variants` / `_get_alleles` / `_get_info`. Ragged-mode variant output
  becomes flat decode + boundary `to_ragged()`.
- **B2 — empty-group dummy padding (feature).** Optionally, for any
  `(region, sample, ploid)` variant group with **zero** variants, insert exactly one
  caller-specified dummy variant. Non-empty groups are left untouched. Guarantees ≥1 variant per
  group. Works identically in flat (`_FlatVariants`) and ragged (`RaggedVariants`) modes.

**Out of scope (own future specs):** phased+unphased sorted-merge and within-group sort (§9);
sub-project C (flank + tokenize); D (windows); E (`double_buffered` / `__getitems__`);
the tracks / intervals flat path.

---

## 2. Decision: padding semantics

**Fill only empty groups** (not gvf's "prepend a dummy to every group").

A group here is one `(region, sample, ploid)` row — i.e. one row of the `b*p` offset structure
of the `(b, p, ~v)` variant output. A group with N ≥ 1 real variants is unchanged; a group with
0 variants receives exactly one dummy, becoming length 1. This guarantees no zero-length groups
(which break downstream tensorization / per-group reductions) without forcing a sentinel slot
into every group.

This **diverges** from genvarformer's current `_add_dummy_variant`, which prepends a dummy to
every group. The genvarformer consumer (and its golden snapshot) adapt to fill-only-empty when
gvf adopts B — a deliberate behavior change on the consumer side, not a byte-match (§8, §10).

---

## 3. Public API

Configured via the existing `Dataset.with_settings(...)` — **not** a new `with_*` method, since
padding does not change the return type (gvl convention: `with_*` is reserved for type-state
transitions like `with_seqs` / `with_output_format`; type-preserving knobs live in
`with_settings`).

```python
from genvarloader import DummyVariant

ds.with_settings(dummy_variant=DummyVariant(start=-1, ref=b"N", alt=b"N", ilen=0, dosage=0.0))
ds.with_settings(dummy_variant=False)   # disable; mirrors the min_af/max_af `False` convention
```

- New parameter `dummy_variant: DummyVariant | Literal[False] | None = None` on `with_settings`
  (`None` = leave unchanged, `False` = disable, a `DummyVariant` = enable). Backed by a frozen
  `Dataset` field `dummy_variant: DummyVariant | None = None`, threaded through `QueryView` into
  the decode using the same plumbing pattern as A's `flat_output`.

- **`DummyVariant`** — new public dataclass (exported in `genvarloader.__all__`, alongside the
  existing config types `Constant` / `FlankSample` / `InsertionFill`). Per-`var_field` values,
  all with sensible defaults and all caller-overridable:

  | field | default | notes |
  |-------|---------|-------|
  | `start` | `-1` | sentinel position |
  | `ref` | `b"N"` | length-1 placeholder allele (avoid `b""` zero-length leaf) |
  | `alt` | `b"N"` | length-1 placeholder allele |
  | `ilen` | `0` | no indel |
  | `dosage` | `0.0` | |
  | `info` | `{}` | `dict[str, Any]`; per-info-field overrides. Unspecified info fields default to `0` (integer columns) / `NaN` (float columns) |

  The dummy must supply a value for every **active** `var_field`; unspecified builtin fields take
  the defaults above, unspecified info fields take the per-dtype default. An `info` key that is
  not an active var_field raises `ValueError`.

- **Validation:** if `dummy_variant` is set but the output kind is **not** variants
  (i.e. `with_seqs(...)` is haplotypes / reference / annotated), `Dataset.__getitem__` (or
  `with_settings`) raises `ValueError`. Padding applies only to the variant-record output.

---

## 4. B1 — always-flat variant decode

- `Haps.__call__` and the `get_haps_and_shifts` variants branch (`_haps.py:523`, `:574`) always
  call `get_variants_flat(...) -> _FlatVariants`, regardless of the `flat` flag.
- The `_query.py` boundary already converts `_Flat` / `_FlatAnnotatedHaps -> to_ragged()` in
  ragged mode (`getitem`, the `if not view.flat_output:` block). **Add `_FlatVariants` to that
  conversion.** In flat mode `_FlatVariants` passes through unconverted (A's behavior).
  `_reshape_outer` and `squeeze` already handle `_FlatVariants` (added in A).
- **Delete** `_get_variants`, `_get_alleles`, `_get_info` (`_haps.py`). Verified they are used
  only by the variant decode — haplotype/annotated reconstruction uses
  `reconstruct_haplotypes_from_sparse`, not these helpers.
- The `flat` parameter on `Haps.__call__` / reconstructors stays (it still selects the *return
  boundary* behavior via `flat_output`); it simply no longer selects between two decode
  implementations, because there is now only one.

**Regression oracle.** The committed snapshot `tests/dataset/_snapshots/variants_ragged.npz`
(generated from the legacy awkward path) is the gate proving the refactor preserves ragged output.
It **must not be regenerated** during B. Note that A's equivalence test
(`flat.to_ragged() == ds[idx]` in ragged mode) becomes tautological once ragged output *is*
flat-derived; the snapshot becomes the real regression gate. To retain an independent awkward
reference, keep one small inline awkward-built expectation in the unit tests (the
`test_flat_variants.py` `to_packed`/`rc` tests already build awkward expectations inline and do
not depend on `_get_variants`, so they survive the deletion).

---

## 5. B2 — empty-group dummy fill

Numba kernels on flat `(data, offsets)` buffers — the same family as genvarformer's `prepend_*`,
but **conditional**: act on a row `i` only when `offsets[i+1] == offsets[i]` (empty), inserting
one dummy element; non-empty rows are copied through unchanged.

- `fill_empty_scalar(data, offsets, fill) -> (new_data, new_offsets)` — scalar fields
  (`start` / `ilen` / `dosage` / info).
- `fill_empty_seq(data, var_offsets, seq_offsets, dummy_bytes) -> (new_data, new_var_offsets, new_seq_offsets)`
  — allele fields (`alt` / `ref`), inserting one `dummy_bytes` allele into empty variant rows.

Exposed as **`_FlatVariants.fill_empty_groups(dummy: DummyVariant) -> _FlatVariants`**, which
dispatches `_Flat` fields to the scalar kernel and `_FlatAlleles` fields to the seq kernel,
producing a new `_FlatVariants` with consistent offsets across all fields. (`_Flat` /
`_FlatAlleles` gain thin per-field `fill_empty(fill)` helpers that the dispatcher calls, mirroring
their existing `reshape` / `squeeze` / `reverse_masked` surface.)

Applied as the **final step inside `get_variants_flat`**, *after* AF/exonic compaction, so a
group emptied by filtering is also padded. Because the decode is now always flat (B1), ragged mode
inherits the fill for free via the boundary `to_ragged()`.

---

## 6. Data flow

```
get_variants_flat(haps, idx, dummy_variant):
  gather v_idxs + row_offsets           (numba; from sparse genotypes)
  AF / exonic compaction                (if filters active)
  build flat fields                     (alt/ref -> _FlatAlleles; start/ilen/dosage/info -> _Flat)
  if dummy_variant is not None:
      fill_empty_groups(dummy_variant)  (numba; only zero-length rows)
  -> _FlatVariants

_query.py getitem:
  reshape / squeeze   (flat methods, _reshape_outer)
  flat mode  -> return _FlatVariants as-is
  ragged mode-> _FlatVariants.to_ragged() -> RaggedVariants
```

`rc_neg` runs in `_getitem_unspliced` before the boundary squeeze (A). The dummy is inserted
during decode, i.e. before `rc`; with the default `b"N"` alleles, reverse-complement is a no-op
(`N -> N`). A caller-supplied non-`N` dummy allele *would* be reverse-complemented like any other
allele on a negative-strand region — documented so it is not surprising.

---

## 7. Error handling & edge cases

- `dummy_variant` set with a non-variant output kind → `ValueError` (§3).
- `DummyVariant.info` key not among active var_fields → `ValueError`.
- Default allele `b"N"` (length 1) avoids reintroducing zero-length allele leaves.
- Composes with `subset_to`, `output_length` (variable / int), `rc_neg`, multi-ploidy, and both
  output formats. In flat mode the consumer densifies via `.to_fixed` / `.to_padded` as usual; the
  fill changes only which rows are non-empty, not the flat container contract.
- A region that is empty for *every* `(sample, ploid)` is handled uniformly — each empty group
  gets its own dummy.

---

## 8. Testing & acceptance

1. **B1 snapshot regression (primary gate):** `tests/dataset/_snapshots/variants_ragged.npz`
   stays green via `test_flat_getitem_snapshot.py`, with the file **not** regenerated. This proves
   retiring `_get_variants` preserves ragged output byte-for-byte.
2. **Flat↔ragged parity for the fill:** `ds.with_output_format("flat").with_settings(dummy_variant=D)[idx].to_ragged()`
   element-identical to `ds.with_settings(dummy_variant=D)[idx]` (ragged), across scalar /
   scalar-scalar (squeeze) / 1-D / 2-D indices, multi-ploidy, naturally-empty regions, and
   filter-emptied groups. Extends `tests/dataset/test_flat_mode_equivalence.py`.
3. **Fill golden unit tests** (`tests/unit/dataset/`): empty group → exactly one dummy with the
   specified field values; non-empty groups untouched; defaults vs per-field overrides; all
   var_fields including dynamically-loaded info; the `fill_empty_scalar` / `fill_empty_seq` kernels
   in isolation (hand-built offsets, including the all-empty and no-empty extremes).
4. **No-awkward guard:** the fill path runs zero awkward kernels (extend
   `test_no_awkward_in_hotpath.py`), since `fill_empty_groups` is pure numba on flat buffers.
5. **Validation tests:** `dummy_variant` + non-variant kind raises; bad `info` key raises.
6. **No ragged-path regression:** full suite green (`pixi run -e dev test`).

---

## 9. Deferred (explicitly out of scope)

- **Phased+unphased sorted-merge** and the **within-group sort** it requires. Rationale in §0:
  no current/near-term need, and both are reachable in genvarformer only in the simultaneous
  dual-dataset branch. When revived, the merge should exploit that **both sources are already
  position-sorted** — a sorted-merge (merge-step of merge-sort, O(n+m)) rather than gvf's current
  concat-then-`segmented_argsort`. `sortednp`
  (https://gitlab.sauerburger.com/frank/sortednp) is a reference for the algorithm, but it is a
  compiled C++/numpy extension and **not** `@nb.njit`-callable, so it cannot live in the numba hot
  path — at most a non-njit fallback or an implementation reference. This becomes its own
  sub-project (provisionally "B2/merge") with its own source-provision design (read-time merge of
  two `Dataset`s vs a write-time combined dataset).

---

## 10. genvarformer consumer thinning (validation, separate repo)

Lands alongside B to prove the win and exercise the API (its own gvf PR, like A's Task 8):

- Switch the `Variants` source to gvl's `with_settings(dummy_variant=...)` for empty-region
  padding, deleting gvf's `_add_dummy_variant` for the variant-record fields and the conditional
  `prepend_*` kernels for those fields. (The flank-row dummy stays in gvf until C/D.)
- Because B2 is **fill-only-empty**, gvf's behavior changes from prepend-to-every-group:
  update `tests/test_ragged_numba.py::test_add_dummy_variant_golden_snapshot` (and the model's
  expectation of a per-group sentinel slot, if any) to the new semantics. This is a deliberate
  behavior change validated by gvf's batch-equality guardrail under the new contract — not a
  byte-identity match against the old gvf output.
- The retired awkward decode (`_get_variants`) means gvf's `Variants._fetch` consumes
  `FlatVariants` directly in flat mode (already enabled by A); B removes the last awkward decode
  fallback on gvl's side.
