# RaggedVariants.to_packed() / rc_() crash on sliced or reordered views

**Date:** 2026-06-07
**Status:** Design — approved scope, pending spec review
**Area (gvl):** `python/genvarloader/_dataset/_rag_variants.py`, `python/genvarloader/_dataset/_haps.py`
**Area (seqpro, upstream):** `python/seqpro/rag/_array.py` (`/Users/david/projects/SeqPro`)

> **Two-repo change.** The numeric-field half of this bug is an upstream defect in seqpro's
> `unbox()`; we fix it in seqpro (§3a), release, and bump gvl's pin. The doubly-nested alt/ref half
> is gvl's own (§4). Ordering: land seqpro first, then the gvl PR depends on the new seqpro.

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

- **alt/ref (gvl):** `_alt_layout_parts()` (`_haps.py:220`) reads `lay.size`, `lay.content.offsets`,
  `lay.content.content.offsets`. `IndexedArray` has no `.size`; `ListArray` has `.starts`/`.stops`,
  not `.offsets`. → `AttributeError`.
- **numeric (seqpro, upstream):** Numeric fields route through seqpro `Ragged.to_packed()`, which
  rejects the `IndexedArray`-wrapped layout: `ValueError: Expected 1 ragged dimension, got 0`. This
  is a seqpro bug — its layout walkers `unbox()` (`_array.py:759`) and `_extract_list_offsets()`
  (`_array.py:85`) loop `while isinstance(node, (ListArray, ListOffsetArray, RegularArray,
  RecordArray))`, which omits `IndexedArray`/`IndexedOptionArray`. An indexed layout stops the loop
  immediately → `n_ragged == 0` → raise. Yet seqpro *constructs* a `Ragged` over the indexed layout
  via behavior dispatch, so the failure is deferred and surprising. Confirmed with a seqpro-only
  repro: `Ragged(ak.zip({"x": r}, depth_limit=1)[perm]["x"])`. `ListArray` is handled; only the
  `Indexed*` case (which arises from indexing a record then extracting a field) breaks.

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
- **Pack with numba (gvl paths).** In gvl, the reorder/gather is done in numba kernels (or seqpro's
  numba-backed ops), not via awkward gather primitives (`ak.to_packed`, `project()`,
  `to_ListOffsetArray64`). Exception: the one-time `project()` inside seqpro's `unbox` (§3a) runs
  only when an index is actually present — off every hot path — and is acceptable there.
- **Do not regress the canonical path.** `rc_()` is on the eager-indexing hot path
  (`_getitem_unspliced`/`_getitem_spliced` → `reverse_complement_ragged`), always called on
  freshly-built canonical arrays. That path must stay byte-identical and zero-overhead — guarded by
  the existing tests in `tests/dataset/test_flat_variants.py`.

## Design

Resolve a lazy/reordered awkward view into contiguous, canonical, zero-based buffers using
numba-based packing. Gate on a cheap layout type-check so the canonical path is untouched.

### 1. Gate: canonical vs. non-canonical (alt/ref only)

After the seqpro fix (§3a), **numeric fields need no gate** — seqpro's `Ragged.to_packed()` handles
canonical, `ListArray`, and `Indexed*` layouts uniformly. The gate applies only to the doubly-nested
alt/ref fields, where gvl owns the packing.

A cheap `isinstance`-chain check classifies the alt/ref field layout against the canonical
`RegularArray → ListOffsetArray → ListOffsetArray → NumpyArray`. Canonical → keep the **existing fast
path** exactly (current allele-level seqpro pack + `_build_allele_layout`; in-place `rc_`).
Non-canonical → resolve via §2 + §4.

### 2. Extract the row permutation (for the alt/ref kernel)

The only reordering ordinary user ops introduce is at the outer (batch) level via
`IndexedArray`/`IndexedOptionArray`. Extract `row_src = np.asarray(layout.index)` and unwrap to the
clean inner layout. Absent an index, `row_src` is identity (plain slices like `rv[1:]` already yield
a clean `RegularArray`/`ListOffsetArray` and need no reorder).

For ploidy `p`, the per-`(b, p)`-row source into the variant-list level is `index[b] * p + h`
(verified). (Numeric fields are now handled entirely by seqpro and do not use `row_src`.)

### 3a. seqpro upstream fix (numeric fields)

In seqpro `python/seqpro/rag/_array.py`, make both layout walkers traverse `Indexed*`:

- `unbox()` (`_array.py:759`) and `_extract_list_offsets()` (`_array.py:85`): when the current node
  is `IndexedArray`/`IndexedOptionArray`, project it (`node = node.project()`) before/within the
  walk, then continue. Projection materializes the gather **only when an index is actually present**;
  canonical layouts never enter this branch, so there is no regression on seqpro's (or gvl's) hot
  paths. Verified: `to_packed` succeeds after `field.layout.project()`.

Add seqpro regression tests (`tests/test_rag_to_packed.py` and/or `tests/test_ragged.py`):
construct a record-layout Ragged, index it, extract a field, and assert `.offsets`, `.data`, and
`.to_packed()` all succeed and match the reordered expectation. Bump the seqpro version and release.

### 3b. gvl numeric fields (start, dosage, ilen, …)

Once seqpro handles `Indexed*`, numeric fields need **no special handling** in gvl — the existing
field-wise `Ragged.to_packed()` (and `Ragged(arr).to_packed()`) path just works on the
indexed/`ListArray` layout. Pin gvl to the fixed seqpro release (update `pyproject.toml` +
`pixi.lock`; verify genoray remains compatible — see seqpro↔genoray version-coupling gotcha).

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

- **Numeric fields:** `Ragged.to_packed()` (resp. `Ragged(arr).to_packed()`) unchanged — now correct
  for all layouts thanks to §3a.
- **alt/ref:** gate (§1). Canonical → existing fast path. Non-canonical → §2 + §4 kernel.

`to_packed()` always returns a fresh object, so materialization is free of side-effect concerns.

### 6. `rc_()`

`rc_` is private with a single call site (`reverse_complement_ragged`) that uses the **return value**.

- Canonical (hot path) → unchanged: in-place reverse-complement of the shared leaf, `return self`.
- Non-canonical → materialize a contiguous canonical copy of the alt/ref fields (reuse §2 + §4),
  reverse-complement the copy's leaf in place, and **return the new object**. No write-back into the
  original is required (the caller uses the return value), and in-place mutation of a reordered view
  is unavoidable-to-copy anyway.

### Decomposition helper

Generalize the layout extraction (in `_alt_layout_parts` or a sibling) to return, for a
non-canonical alt/ref field, `(row_src, leaf, allele_starts, allele_stops, var_starts, var_stops,
ploidy)` — reading `.starts`/`.stops` uniformly and the optional outer index. The canonical fast path
keeps the existing simpler extraction.

## Testing (TDD)

**seqpro** (`tests/test_rag_to_packed.py` / `tests/test_ragged.py`): record-layout Ragged → index →
extract field → assert `.offsets`/`.data`/`.to_packed()` succeed and match the reordered expectation;
keep existing canonical tests green.

**gvl** — append to `tests/dataset/test_flat_variants.py`:

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

**seqpro** (`/Users/david/projects/SeqPro`, land + release first):
- `python/seqpro/rag/_array.py` — `unbox()` (`:759`), `_extract_list_offsets()` (`:85`): traverse
  `Indexed*`.
- `tests/test_rag_to_packed.py` / `tests/test_ragged.py` — regression tests.
- version bump + release.

**gvl** (depends on the new seqpro):
- `pyproject.toml` / `pixi.lock` — bump seqpro pin (verify genoray compat).
- `python/genvarloader/_dataset/_rag_variants.py` — `to_packed` (alt/ref branch), `rc_`.
- `python/genvarloader/_dataset/_haps.py` — generalize the layout-decomposition helper; add the
  numba `_pack_alleles` kernel (final location — `_haps.py` vs `_rag_variants.py` vs a kernels
  module — decided in the plan).
- `tests/dataset/test_flat_variants.py` — regression tests.
