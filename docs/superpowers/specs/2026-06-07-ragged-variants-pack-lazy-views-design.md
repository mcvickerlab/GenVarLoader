# RaggedVariants.to_packed() / rc_() crash on sliced or reordered views

**Date:** 2026-06-07
**Status:** Design — approved scope, pending spec review
**Area:** `python/genvarloader/_dataset/_rag_variants.py`, `python/genvarloader/_dataset/_haps.py`

## Problem

Users report `RaggedVariants.to_packed()` crashing with "ListArray contents":

```
AttributeError: 'ListArray' object has no attribute 'offsets'
```

Reproduced and confirmed. The crash is one symptom of a broader bug: **`to_packed()` and
`rc_()` are broken on any `RaggedVariants` that has been sliced, reversed, or fancy-indexed.**

A freshly reconstructed `RaggedVariants` (straight out of `dataset[region, sample]`) has the
canonical awkward layout that `_build_allele_layout` produces:

```
alt/ref : RegularArray(ploidy) → ListOffsetArray(variants) → ListOffsetArray(alleles) → NumpyArray(bytes)
numeric : ListOffsetArray(variants) → NumpyArray
```

But any ordinary awkward operation on the result rebuilds the layout as a *lazy view*:

| User op                         | Resulting outer node            |
|---------------------------------|---------------------------------|
| `rv[perm]` (fancy index)        | `IndexedArray`                  |
| `rv[::-1]` (reverse)            | `IndexedArray`                  |
| batch/buffer round-trips        | `ListArray` (`.starts`/`.stops`)|

These are *equivalent* representations of the same data, but the current code assumes the
canonical chain and reads attributes that only the canonical nodes expose:

- `_alt_layout_parts()` (`_haps.py:220`) reads `lay.size`, `lay.content.offsets`,
  `lay.content.content.offsets`. `IndexedArray` has no `.size`; `ListArray` has `.starts`/`.stops`,
  not `.offsets`. → `AttributeError`.
- Numeric fields route through seqpro `Ragged.to_packed()`, which rejects the `IndexedArray`-wrapped
  layout: `ValueError: Expected 1 ragged dimension, got 0`.

So a user who does `ds[...]`, shuffles/reorders the batch, then calls `.to_packed()` (e.g. before
`to_nested_tensor_batch`, which documents that it assumes packed data) hits the crash. `rc_()` shares
the same fragile `_alt_layout_parts` and fails identically.

### Confirmed reproductions

```python
rv = <freshly built RaggedVariants>          # canonical: to_packed() OK
rv[::-1].to_packed()                          # AttributeError: 'IndexedArray' has no 'size'
rv[perm].to_packed()                          # same
<explicit ListArray variant level>.to_packed()# AttributeError: 'ListArray' has no 'offsets'
rv[perm]["start"].to_packed()                 # ValueError: Expected 1 ragged dimension, got 0
rv[::-1].rc_()                                # ValueError: Expected 1 ragged dimension, got 0
```

## Constraints

- **No `ak.to_packed`.** It was central to recent serious performance regressions and is banned —
  not just on hot paths.
- **Pack with numba.** The reorder/gather must be done in numba kernels (or seqpro's numba-backed
  ops), not via awkward gather primitives (`ak.to_packed`, `project()`, `to_ListOffsetArray64`).
- **Do not regress the canonical path.** `rc_()` is on the eager-indexing hot path
  (`_getitem_unspliced`/`_getitem_spliced` → `reverse_complement_ragged`), always called on
  freshly-built canonical arrays. That path must stay byte-identical and zero-overhead — guarded by
  the existing tests in `tests/dataset/test_flat_variants.py`.

## Design

Resolve a lazy/reordered awkward view into contiguous, canonical, zero-based buffers using
numba-based packing. Gate on a cheap layout type-check so the canonical path is untouched.

### 1. Gate: canonical vs. non-canonical

A cheap `isinstance`-chain check classifies the field layout:

- **alt/ref canonical:** `RegularArray → ListOffsetArray → ListOffsetArray → NumpyArray`.
- **numeric canonical:** `ListOffsetArray → NumpyArray` (i.e. a clean seqpro `Ragged`).

Canonical → keep the **existing fast path** exactly (seqpro `Ragged.to_packed()` for numeric; the
current allele-level seqpro pack + `_build_allele_layout` for alt/ref; in-place `rc_`). Non-canonical
→ resolve via the steps below.

### 2. Extract the row permutation

The only reordering ordinary user ops introduce is at the outer (batch) level via
`IndexedArray`/`IndexedOptionArray`. Extract `row_src = np.asarray(layout.index)` and unwrap to the
clean inner layout. Absent an index, `row_src` is identity (plain slices like `rv[1:]` already yield
a clean `RegularArray`/`ListOffsetArray` and need no reorder).

For ploidy `p`, the per-`(b, p)`-row source into the variant-list level is `index[b] * p + h`
(verified). For numeric fields it is `index` directly (verified).

### 3. Pack numeric fields (start, dosage, ilen, …) — seqpro numba

Build a clean `Ragged` from the inner `ListOffsetArray` (offsets + data, in original order), then
fancy-index + pack with seqpro:

```python
clean = Ragged.from_offsets(inner_data, (n_orig_rows, None), inner_offsets)
packed = clean[row_src].to_packed()           # seqpro 0.14 numba, 0 awkward calls
```

`Ragged[idx].to_packed()` is the awkward-free gather already used elsewhere in this codebase
(the flat-variants path). Verified to reproduce the reordered field byte-for-byte.

### 4. Pack alt/ref (doubly-nested) — new numba kernel

seqpro `Ragged` is single-level, so the doubly-nested allele arrays need a dedicated kernel.

Decompose the (possibly `ListArray`) inner layout by reading `.starts`/`.stops`, which both
`ListArray` *and* `ListOffsetArray` expose uniformly:

- `var_starts` / `var_stops` — variant ranges (index into the allele-record level)
- `allele_starts` / `allele_stops` — byte ranges in the leaf (index into the leaf buffer)
- `leaf` — `NumpyArray.data` as `uint8`
- `ploidy` — `RegularArray.size`

New numba kernel:

```
_pack_alleles(row_src, ploidy, var_starts, var_stops, allele_starts, allele_stops, leaf)
    -> (packed_bytes: uint8[], allele_off: int64[], group_off: int64[])
```

For each output `(b, p)` row in order, for each variant in
`[var_starts[src], var_stops[src])`, copy `leaf[allele_starts[a]:allele_stops[a]]` into the output;
accumulate `allele_off` (per-allele byte boundaries) and `group_off` (per-row variant counts). Output
is contiguous, zero-based, in canonical `(b, p, ~v, ~l)` row-major order. Then
`_build_allele_layout(packed_bytes, allele_off, group_off, ploidy)` rebuilds the canonical
`ak.Array`. For canonical input this is byte-identical to the existing path.

### 5. `to_packed()`

Per field: gate (§1). Canonical → existing fast path. Non-canonical → §2 once, then §3 (numeric) or
§4 (alt/ref). `to_packed()` always returns a fresh object, so materialization is free of side-effect
concerns.

### 6. `rc_()`

`rc_` is private with a single call site (`reverse_complement_ragged`) that uses the **return value**.

- Canonical (hot path) → unchanged: in-place reverse-complement of the shared leaf, `return self`.
- Non-canonical → materialize a contiguous canonical copy (reuse §2–§4 / the `to_packed` machinery),
  reverse-complement the copy's leaf in place, and **return the new object**. No write-back into the
  original is required (the caller uses the return value), and in-place mutation of a reordered view
  is unavoidable-to-copy anyway.

### Decomposition helper

Generalize the layout extraction (in `_alt_layout_parts` or a sibling) to return, for a
non-canonical alt/ref field, `(row_src, leaf, allele_starts, allele_stops, var_starts, var_stops,
ploidy)` — reading `.starts`/`.stops` uniformly and the optional outer index. The canonical fast path
keeps the existing simpler extraction.

## Testing (TDD)

Append to `tests/dataset/test_flat_variants.py`:

1. **`to_packed` on lazy views** — reversed (`rv[::-1]`), fancy-indexed (`rv[perm]`), and an
   explicitly-constructed `ListArray` variant level. Each: no crash, and byte-identical to the
   `ak.to_packed(ak.Array(rv))` oracle (allowed *in tests* as the reference) for alt, ref, **and**
   numeric `start`/`dosage` values + offsets.
2. **`rc_` on lazy views** — reversed / fancy-indexed input matches an independent awkward
   reverse-complement reference; returned object is correct.
3. **ploidy=2 reordered** — exercises `row_src` expansion by ploidy and numeric-vs-alt nesting
   consistency.
4. **Canonical regression** — existing `test_to_packed_matches_awkward_*` and
   `test_rc_matches_awkward` pass unchanged (proves the fast path is byte-identical and unaffected).

## Edge cases & non-goals

- Empty rows (zero variants in a `(b, p)` row), zero-length alleles — kernel must handle.
- `IndexedOptionArray` with `None` — gvl variants should never contain `None`; raise a clear error
  (or treat defensively) rather than silently mis-pack.
- Inner-axis slicing (`rv[:, :, 1:]`) crashes *inside awkward itself* before reaching gvl — out of
  scope (upstream awkward issue).
- **Non-goal:** any change to canonical-path behavior or performance, and any reintroduction of
  `ak.to_packed` / awkward gather primitives.

## Files

- `python/genvarloader/_dataset/_rag_variants.py` — `to_packed`, `rc_`.
- `python/genvarloader/_dataset/_haps.py` — generalize the layout-decomposition helper; add the
  numba `_pack_alleles` kernel (final location — `_haps.py` vs `_rag_variants.py` vs a kernels
  module — decided in the plan).
- `tests/dataset/test_flat_variants.py` — regression tests.
