# SVAR2 INFO/FORMAT field routing → `RaggedVariants` / `FlatVariants`

**Date:** 2026-07-12
**Worktree/branch:** `svar2-m6b-kernel` (draft PR #266)
**Status:** design approved; ready for implementation plan

## Goal

Route arbitrary scalar-numeric INFO/FORMAT fields stored in an SVAR2 (`.svar`)
store all the way through gvl's read-bound SVAR2 path into the `variants`
(`RaggedVariants`) and variant-windows (`_FlatVariantWindows`, the "flat"
variant output) outputs. Today gvl's SVAR2 path surfaces only `alt`/`start`/
`ilen`; the field values genoray now persists (`SparseVar2.from_vcf(info_fields=,
format_fields=)` / `from_pgen`) are dropped.

**Scope (approved):** both INFO and FORMAT fields; both `RaggedVariants` and
variant-windows outputs. Scalar-numeric only (`Integer`/`Float`, `Flag` for
INFO; `Number` `1` or `A`) — exactly what genoray's SVAR2 store can hold.

## Background — where the seam is

gvl's SVAR2 path does **not** call genoray's Python `SparseVar2.decode()`. It
uses gvl's own read-bound Rust kernel `decode_variants_from_svar2_readbound`
(`src/ffi/mod.rs:1330`), which links `genoray_core` as a crate, calls
`genoray_core::query::gather_haps_readbound(reader, &HapRanges) ->
BatchResultSplit`, and runs its own `var_key ⋈ dense_snp ⋈ dense_indel`
position-merge (`src/svar2/mod.rs:decode_variants_from_split`) over plain
`KeyRef { position, key }`. To attach a field value to a decoded variant we need
that variant's **source index** in the store, which plain `KeyRef` does not
carry.

genoray (current main `acc59cb`) already exports everything needed to recover it
— **no genoray-side code change is required**:

- `gather_haps_readbound_src` — same as `gather_haps_readbound` but populates
  `BatchResultSplit.vk_src` (packed var_key provenance). The plain variant
  leaves `vk_src` empty (it does not pay for provenance).
- `pack_vk_src` / `unpack_vk_src` / `VK_SRC_INDEL_BIT` — `vk_src[i]` →
  `(is_indel, call_idx)`, where `call_idx` is the absolute call index into
  `var_key/{snp,indel}`.
- `dense_abs_row(on_disk_range, out_range, i)` — recovers a dense variant's
  absolute source row from the `HapRanges` on-disk dense range + the
  `BatchResultSplit` output dense range. (No `dense_src` array in the split
  path; it's pure arithmetic.)
- `FieldView` (+ `FieldValue`) with `value_at(i)` / `format_at(dense_row,
  orig_sample)` / `bytes_at(i)`; `crate::layout::{ContigPaths, FieldSub}`,
  `crate::field::StorageDtype`. `FieldSub::all()` order is `[VkSnp, VkIndel,
  DenseSnp, DenseIndel]`.

Reference implementation to mirror: genoray's own `gather_batch_fields`
(`src/py_query_decode.rs:156`), which does this gather for the whole-cohort
batch path. **The one difference in the read-bound path:** the split has
`n_samples == 1`, so a dense FORMAT lookup must stride by the real cohort sample
`HapRanges::orig_samples[q]`, i.e. `format_at(dense_row, orig_samples[q])` —
*not* the split's sample slot (always 0).

## Prerequisite — genoray dependency (build a local wheel)

The field-read API is **unreleased**: it is 87 commits past the `3.0.0` tag and
in no tag. The existing `genoray-3.0.0` wheel lacks it. Plan (approved):

1. **Drop gvl's genoray version pin.** `pyproject.toml:14` `"genoray>=3,<4"` →
   `"genoray"`. (Re-pin to a real release later, when genoray ships svar-2.)
2. **Build a genoray manylinux wheel from current main** (`acc59cb`, has the
   field-read API):
   ```
   cd /carter/users/dlaub/projects/genoray
   pixi run --manifest-path ci/wheel/pixi.toml build     # -> wheelhouse/*.whl
   pixi run --manifest-path ci/wheel/pixi.toml repair     # auditwheel -> dist/*.whl
   ```
   The wheel reports version `2.15.0` (genoray's `pyproject` version); that's
   fine once gvl's pin is dropped. Mind the NFS build gotchas
   (`CARGO_TARGET_DIR=/tmp/...`).
3. **Point gvl's pixi pin at the new wheel** (`pixi.toml:110`), then
   `pixi install` / re-solve. This also fixes the currently-unsolvable env
   (`pyproject` `>=3,<4` vs the pinned `2.15.0` wheel).

The Rust side already links the live genoray repo via `Cargo.toml`
`genoray_core = { path = ".../genoray", ... }`, so it sees the exports at build
time regardless of the Python wheel.

## Rust design (`genvarloader` crate)

Files: `src/ffi/mod.rs`, `src/svar2/mod.rs`, `src/svar2/store.rs`, `src/lib.rs`
(registration unchanged unless signature changes require it).

1. **`decode_variants_from_svar2_readbound` (`src/ffi/mod.rs:1330`)** gains a
   `fields: Vec<(String /*category*/, String /*name*/, String /*dtype*/)>`
   param (empty ⇒ current behavior, zero overhead). Returns the existing 5-tuple
   plus:
   - `field_bufs: Vec<Py<PyArray1<u8>>>` — one flat little-endian byte buffer
     per requested field, length `n_var * itemsize`, in the same variant order
     as `pos` (so it shares `var_off`).
   - `field_itemsizes: Vec<usize>` — parallel; Python asserts
     `itemsize == dtype.itemsize` (mirrors genoray `_svar2_decode.py:70`).
2. When `fields` is non-empty, call **`gather_haps_readbound_src`** instead of
   `gather_haps_readbound`, and keep the `HapRanges` alive to feed
   `dense_abs_row`.
3. Open, per field, the four `FieldView`s (`FieldSub::all()`) from the store's
   `ContigReader` paths. `Svar2Store`/`ContigReader` gain the accessor(s) needed
   to reach `ContigPaths` + the store's cohort `n_samples` and each field's
   `StorageDtype` (resolved from the store `meta.json` manifest — genoray's
   `StorageDtype::from_meta_str` is public). If the store lacks a requested
   field, error clearly in Python before calling the kernel (see Python §1).
4. **Thread provenance through the merge** in `decode_variants_from_split`
   (`src/svar2/mod.rs`). For each emitted variant, in lockstep with `pos`:
   - var_key entry `i` → `unpack_vk_src(split.vk_src[i]) = (is_indel, call_idx)`;
     `is_dense=false`, `idx=call_idx`.
   - dense entry `i` in the snp/indel window for query `q` →
     `idx = dense_abs_row(hapranges.dense_{snp,indel}_range[q],
     split.dense_{snp,indel}_range[q], i)`; `is_dense=true`, `is_indel` by
     channel.
   Then for each field pick the sub-view by `(is_dense, is_indel)` and append:
   - INFO → `view.bytes_at(idx)`
   - FORMAT → var_key: `view.bytes_at(idx)`; dense:
     `view.bytes_at(dense_row * cohort_n_samples + orig_samples[q])`
     (== `format_at(dense_row, orig_samples[q])` as bytes).
   Copy bytes verbatim (no dtype dispatch in Rust; Python `.view()`s). Missing
   values already carry genoray's stored default/sentinel — passed through.

**Provenance-vs-decode ordering invariant (must-verify):** the merge already
emits variants in a defined tie order (`var_key < dense-snp < dense-indel` on
equal positions). The field append must use the provenance of *the same* entry
chosen at each merge step — i.e. provenance is carried on the merge cursor, not
recomputed. This is the single highest-risk part; the plan must add an identity
test that a field carrying "the variant's own source row index" decodes back to
a strictly-consistent mapping.

## Python design (`Svar2Haps`, `python/genvarloader/_dataset/_svar2_haps.py`)

1. **Field discovery** (`from_path`): after `sv = SparseVar2(str(svar2_path))`,
   read the store manifest via `sv.available_fields` (genoray 3.x). Set
   `self.available_var_fields = ["alt","ilen","start"] + list(store_field_keys)`
   (replaces the hard-coded `:174`). Keep a `dict[key -> (category, name,
   np.dtype)]` on the instance for the kernel call and dtype `.view()`. Thread
   the user's `var_fields` into `from_path` and onto the instance (today it
   defaults to `["alt","ilen","start"]` and is never set from the Dataset for
   svar2 — wire it like the SVAR1 path).
2. **Request set:** `requested = [f for f in self.var_fields if f not in
   {"alt","start","ref","ilen","dosage"}]`. Validate each is in
   `available_var_fields`; raise a clear error otherwise. Map to `(category,
   name, dtype)` triples for the kernel.
3. **`_reconstruct_variants`:** pass the triples to the kernel; receive
   `field_bufs`/`itemsizes`. Each field is per-variant (one element per
   variant), so it is **parallel to `pos`** and flows through the exact existing
   machinery with `field_g = field_c[src]`:
   - single-contig fast path: `Ragged.from_offsets(buf.view(dtype), shape,
     var_off_g)`;
   - multi-contig: index by the same `src` from `_ragged_arange_src`;
   - `unphased_union`: the `var_off[::P]` fold needs **no special handling**
     (field data stays whole, offsets reindex).
   Add each as `RaggedVariants(**{key: field_ragged})`, sharing `alt`'s offsets.
4. **`_reconstruct_variant_windows`:** same kernel call; add each field to the
   `_FlatVariantWindows.fields` dict as a `_Flat.from_offsets(buf.view(dtype),
   shape, row_off)`, exactly like `start`/`ilen` (respecting the single- vs
   multi-contig branches and the `include_*` gating by `var_fields`).
5. **`DummyVariant` / empty-group fill:** variant-windows fills empty groups via
   `dummy_variant`. Extend the dummy to carry a per-field fill (genoray's stored
   default/sentinel, or NaN/sentinel) so `fill_empty_groups` has a value for
   each field. (`DummyVariant` already has an `info: dict` slot.)

## Data-flow / alignment invariants

- Field buffers are 1-level ragged, per-variant, sharing the variant-axis
  offsets (`var_off`) with `pos`/`ilen`. Every reorder/fold applied to `pos`
  applies identically to fields.
- Dtypes are preserved end-to-end (stored → raw bytes → `.view(dtype)`); no
  widening. Ints keep genoray's lossless auto-narrowed width; floats stay
  `f16`/`f32`; `Flag`→`bool`.
- Missing entries carry genoray's `default`/sentinel verbatim — gvl does not
  reinterpret them.

## Field key naming & dtypes

Use the keys genoray's manifest exposes (`sv.available_fields` keys): bare field
name when unambiguous, else category-prefixed (genoray already disambiguates
INFO vs FORMAT collisions). These become the `RaggedVariants(**fields)` kwargs
and `_FlatVariantWindows.fields` keys. Never collide with builtin keys
(`alt`/`start`/`ref`/`ilen`/`dosage`) — guard in discovery.

## Testing

Round-trip oracle test mirroring genoray's `test_svar2_fields_read.py`:

1. Build a small VCF with ≥2 INFO fields (e.g. an `Integer` and a `Float=AF`)
   and ≥1 FORMAT field (e.g. `DP` Integer). Convert with
   `SparseVar2.from_vcf(info_fields=..., format_fields=...)`.
2. `gvl.write` a dataset over it; read `variants` and variant-windows with
   `var_fields` including the field names.
3. Assert field **values** (not just shape/dtype) against the VCF ground truth,
   across:
   - var_key-routed vs dense-routed variants (both present in the fixture);
   - multi-contig (exercise the `_ragged_arange_src` reorder);
   - `unphased_union=True`;
   - VCF-missing entries (assert the configured default / sentinel).
4. Rust unit test on `decode_variants_from_split` asserting provenance identity
   (the ordering invariant above).

## Docs / skill updates

- Update `skills/genvarloader/SKILL.md` if the public `var_fields` surface for
  svar2 datasets changes (it will: svar2 `variants` now honors arbitrary store
  fields).
- Add a `CHANGELOG.md` (Unreleased) entry.

## Out of scope (unchanged guards)

`min_af`/`max_af`, `filter=="exonic"`, spliced output, annotated haps, in-kernel
RC — all still `NotImplementedError` for svar2. Non-scalar / `Number`-other
fields are rejected at the genoray write boundary, so they never reach here.
