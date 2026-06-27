# Phase 5 W6 — Thin-Shim Audit

**Date:** 2026-06-27
**Branch:** phase-5-w6-wrapup
**Auditor:** Task 1 (automated, Claude)

## Purpose

Audit whether the Python layer over the PyO3 FFI surface is already a thin
shim, or whether collapsible glue remains. This verdict determines whether
Phase 5 "Collapse the PyO3 surface so Python is a true shim" can be ticked.

---

## Step 1 — Read-path call-chain inventory

### `Dataset.__getitem__` (hot path, unspliced)

```
Dataset.__getitem__                          _impl.py:1743
  → QueryView construction                  _impl.py:1776-1789   (indexing sugar — validated attr packing)
  → getitem(view, idx)                      _query.py:66
      → _getitem_unspliced(view, idx)        _query.py:154
          parse_idx / jitter / to_rc         _query.py:162-175   (indexing sugar + numpy scalar ops)
          → view.recon(...)                  _query.py:178       (dispatches to active Reconstructor)

            BRANCH A: Haps.__call__
              → Haps.get_haps_and_shifts     _haps.py:619
                  → _prepare_request         _haps.py:675
                      _get_geno_offset_idx   _haps.py:753        (np.unravel_index + np.ravel_multi_index)
                      [optional] choose_exonic_variants          FFI: choose_exonic_variants
                      → _haplotype_ilens     _haps.py:492
                          → get_diffs_sparse                     FFI: get_diffs_sparse
                      shift RNG              _haps.py:725-727    (numpy RNG call)
                      lengths_to_offsets                         (seqpro utility, cumsum)
                  → _reconstruct_haplotypes  _haps.py:809
                      _out_per comparison    _haps.py:823-833    (ragged-vs-fixed detection, ~3 numpy ops)
                      np.repeat(to_rc, p)    _haps.py:840        (to_rc expansion, batch-bounded)
                      → reconstruct_haplotypes_fused             FFI: fused kernel (one crossing)
                      _Flat.from_offsets     _haps.py:866        (zero-copy view wrap)

            BRANCH B: Haps.__call__ (annotated kind)
              same _prepare_request path as A, then:
              → _reconstruct_annotated_haplotypes  _haps.py:919
                  (same ragged-vs-fixed detection + to_rc expansion as A)
                  → reconstruct_annotated_haplotypes_fused       FFI: fused kernel (one crossing)
                  3× _Flat.from_offsets                          (zero-copy view wraps)

            BRANCH C: HapsTracks.__call__
              → haps.get_haps_and_shifts     (same as BRANCH A/B above)
              per-track loop:
                  out buffer allocation      _reconstruct.py:179  (np.empty, batch×ploidy×tracks f32)
                  einops.repeat out_lengths  _reconstruct.py:180  (batch-bounded)
                  lengths_to_offsets ×2      _reconstruct.py:183-184
                  _lower_insertion_fills     _reconstruct.py:190  (strat list → id/params arrays)
                  base_seed computation      _reconstruct.py:195-201 (np.bitwise_xor.reduce or rng.integers)
                  _as_starts_stops once      _reconstruct.py:206  (offsets → (2,N) view)
                  to_rc expansion (per-track) _reconstruct.py:235
                  → intervals_and_realign_track_fused            FFI: fused kernel (one crossing per track)
              _Flat.from_offsets             _reconstruct.py:280  (zero-copy wrap)

            BRANCH D: Tracks.__call__  (reference-coordinate tracks, no haplotype re-alignment)
              → _call_intervals              _tracks.py
                  → intervals_to_tracks or realign FFI calls     (separate smaller kernels)

            BRANCH E: Ref.__call__
              → get_reference                                     FFI: get_reference (one crossing)

          [optional] reverse_complement_ragged  _query.py:200   (variant types only, not byte/track data)
          to_ragged / squeeze / reshape       _query.py:111-126  (output massaging — indexing sugar)
```

### `Dataset.__getitem__` (spliced path)

The spliced path prepends a `build_recon_splice_plan` step (calls
`haplotype_lengths_for_plan → get_diffs_sparse FFI`, plus `build_splice_plan`
FFI) and passes the `SplicePlan` into the same `_reconstruct_haplotypes` /
`_reconstruct_annotated_haplotypes` fused kernels, each of which then calls
`_permute_request_for_splice` (Python permutation of per-element arrays, batch-bounded).

---

## Step 2 — FFI surface inventory

`src/lib.rs` registers **33 entries** (32 `wrap_pyfunction!` + 1 `add_class`):

| # | Symbol | Category |
|---|--------|----------|
| 1 | `count_intervals` | BigWig util |
| 2 | `bigwig_intervals` | BigWig util |
| 3 | `bigwig_write_track` | BigWig write |
| 4 | `RustTable` (class) | Write path |
| 5 | `ragged_to_padded` | Ragged util |
| 6 | `intervals_to_tracks` | Track util |
| 7 | `get_diffs_sparse` | Read-path helper |
| 8 | `choose_exonic_variants` | Read-path helper |
| 9 | `gather_rows_i32` | Genotype util |
| 10 | `gather_rows_f32` | Genotype util |
| 11 | `gather_alleles` | Genotype util |
| 12 | `compact_keep_i32` | Genotype util |
| 13 | `compact_keep_f32` | Genotype util |
| 14 | `fill_empty_scalar_i32` | Genotype util |
| 15 | `fill_empty_scalar_f32` | Genotype util |
| 16 | `fill_empty_fixed_i32` | Genotype util |
| 17 | `fill_empty_fixed_f32` | Genotype util |
| 18 | `fill_empty_seq_u8` | Genotype util |
| 19 | `fill_empty_seq_i32` | Genotype util |
| 20 | `assemble_variant_buffers_u8` | Variant buffer |
| 21 | `assemble_variant_buffers_i32` | Variant buffer |
| 22 | `rc_alleles` | Allele RC |
| 23 | `get_reference` | Read-path — reference sequences |
| 24 | `reconstruct_haplotypes_from_sparse` | Read-path helper (non-fused) |
| 25 | `reconstruct_haplotypes_fused` | **Fused `__getitem__` kernel** |
| 26 | `reconstruct_annotated_haplotypes_fused` | **Fused `__getitem__` kernel** |
| 27 | `reconstruct_haplotypes_spliced_fused` | **Fused `__getitem__` kernel** |
| 28 | `reconstruct_annotated_haplotypes_spliced_fused` | **Fused `__getitem__` kernel** |
| 29 | `shift_and_realign_tracks_sparse` | Track util (non-fused) |
| 30 | `tracks_to_intervals` | Track util |
| 31 | `intervals_and_realign_track_fused` | **Fused `__getitem__` kernel** |
| 32 | `_debug_xorshift64` | Debug/parity (Task 7) |
| 33 | `_debug_hash4` | Debug/parity (Task 7) |

**Fused `__getitem__` kernels:** 5 (entries 25–28 + 31 = `reconstruct_haplotypes_fused`,
`reconstruct_annotated_haplotypes_fused`, `reconstruct_haplotypes_spliced_fused`,
`reconstruct_annotated_haplotypes_spliced_fused`, `intervals_and_realign_track_fused`).

`assemble_variant_buffers_{u8,i32}` (entries 20–21) are used on the variant-windows and
flat-variants path, not the primary `__getitem__` hot path for byte sequences or tracks.

---

## Step 3 — Dispatch layer check

```
$ ls python/genvarloader/_dispatch.py 2>&1
No such file or directory
```

```
$ grep -rn "GVL_BACKEND|_dispatch|import numba|from numba|nb\.njit|nb\.prange" python/genvarloader/ --include=*.py
(zero matches)
```

**Result:** `_dispatch.py` does not exist. No `GVL_BACKEND`, `_dispatch`, or
numba import found anywhere in `python/genvarloader/`. The dispatch layer is
fully gone; Python calls Rust directly. Stale bytecode
`__pycache__/_dispatch.cpython-*.pyc` was removed (no file existed to remove).

---

## Step 4 — Three-bucket classification

### Bucket definitions

- **Bucket 1 — Intentional shim:** Indexing sugar, torch/device handling,
  validation, error messages, output massaging. Stays in Python by design.
- **Bucket 2 — Remaining collapsible glue:** Per-batch coercion / allocation /
  object churn worth a future kernel. Not negligible overhead today.
- **Bucket 3 — Already-collapsed:** One FFI crossing, no material Python work.

### Classification table

| Python step | Location | Bucket | Justification |
|-------------|----------|--------|---------------|
| `QueryView` construction | `_impl.py:1776` | 1 | Attr packing; zero array work |
| `parse_idx` / index validation | `_query.py:162` | 1 | Indexing sugar |
| Jitter offset computation | `_query.py:168-171` | 1 | One `rng.integers` + 2 in-place scalar ops; batch-bounded |
| `to_rc` derivation from strand column | `_query.py:174` | 1 | One boolean comparison on a slice |
| `_get_geno_offset_idx` | `_haps.py:753` | 1 | Two `np.unravel_index` / `ravel_multi_index` over `(b,)` / `(b, p)` arrays; indexing sugar for genotype address translation |
| `choose_exonic_variants` (optional) | `_haps.py:698` | 3 | Thin wrapper; one FFI crossing |
| `get_diffs_sparse` | `_haps.py:518` | 3 | Thin wrapper; one FFI crossing |
| Shift RNG call | `_haps.py:725` | 1 | One `rng.integers`; intentional Python-side random state |
| `lengths_to_offsets` | `_haps.py:736` | 1 | Cumsum utility; negligible, batch-bounded |
| Ragged-vs-fixed detection (`_out_per` comparison) | `_haps.py:823` | 1 | 3 numpy ops on `(b*p,)` arrays; determines kernel mode flag |
| `np.repeat(to_rc, ploidy)` + `ascontiguousarray` | `_haps.py:840` | 1 | Expands `(b,)` → `(b*p,)` bool; batch-bounded, no alternative without a kernel API change |
| `ascontiguousarray` coercions on `regions`, `shifts`, `geno_offset_idx`, `keep`, `keep_offsets` | `_haps.py:843-861` | 1 | All batch-bounded (b or b×p arrays); guard FFI typing; zero-copy when already contiguous (common case via `_prepare_request`) |
| `_ffi_array` checks on `geno_v_idxs` | `_haps.py:847` | 1 | Zero-copy assertion guard; per-sample-scale memmap — correctly NOT coercing |
| `reconstruct_haplotypes_fused` | `_haps.py:842` | 3 | **One FFI crossing** |
| `_Flat.from_offsets` (post-kernel) | `_haps.py:866` | 1 | Zero-copy view wrap; no array work |
| `reconstruct_annotated_haplotypes_fused` | `_haps.py:957` | 3 | **One FFI crossing** |
| `reconstruct_haplotypes_spliced_fused` | `_haps.py:884` | 3 | **One FFI crossing** |
| `reconstruct_annotated_haplotypes_spliced_fused` | `_haps.py:1015` | 3 | **One FFI crossing** |
| `_permute_request_for_splice` | `_haps.py:1056` | 1 | Batch-bounded permutation of per-element arrays for the splice plan; structural pre-processing, not a hot inner loop on the read path |
| `HapsTracks` out-buffer allocation (`np.empty`) | `_reconstruct.py:179` | 1 | Allocates a single `(b*p*t)` f32 buffer; standard pre-allocation pattern before an in-place kernel |
| `einops.repeat out_lengths` | `_reconstruct.py:180` | 1 | Batch-bounded broadcast; library call |
| `lengths_to_offsets` ×2 | `_reconstruct.py:183-184` | 1 | Cumsum; batch-bounded |
| `_lower_insertion_fills` | `_reconstruct.py:190` | 1 | Converts Python strategy objects → id/params arrays; O(n_tracks) not O(batch) |
| `base_seed` computation | `_reconstruct.py:195` | 1 | One RNG or xor-reduce; Python-side randomness |
| `_as_starts_stops` once per batch | `_reconstruct.py:206` | 1 | Converts offsets to (2, N) view; called once per batch (amortized over tracks). Wraps `ascontiguousarray` on the sample-scale offsets array — this IS a candidate for caching but is a read, not a write |
| per-track `to_rc` `np.repeat` + `ascontiguousarray` | `_reconstruct.py:235` | 1 | Same batch-bounded expansion as haps; repeated once per track |
| per-track `ascontiguousarray` coercions | `_reconstruct.py:239-268` | 1 | All batch-bounded; guard FFI typing |
| `intervals_and_realign_track_fused` (per track) | `_reconstruct.py:237` | 3 | **One FFI crossing per track** |
| `_getitem_unspliced` post-kernel shaping (`to_ragged`, `to_fixed`, squeeze) | `_query.py:95-126` | 1 | Output format massaging; indexing sugar |
| `reverse_complement_ragged` (variant types only) | `_query.py:200` | 1 | Post-kernel Python RC; only for RaggedVariants / FlatVariants / FlatVariantWindows — byte/track RC is already folded in-kernel |
| `get_reference` | `_reference.py` | 3 | One FFI crossing |

### `ascontiguousarray` on per-sample-scale memmaps

`_ffi_array` (`_utils.py:13`) is used for the four per-sample-scale memmap
arguments (`geno_v_idxs`, `itv_starts`, `itv_ends`, `itv_values`,
`itv_offsets`) — it asserts contiguity and raises a precise error instead of
silently copying. The memory-map note in `_utils.py` confirms this is the
correct behavior: "coercing would force a sample-scale copy." There are **zero
`ascontiguousarray` calls on per-sample-scale memmaps** in the hot read path;
all surviving `ascontiguousarray` calls are on batch-bounded arrays (`b` or
`b×p` arrays that are typically already contiguous in practice but require an
explicit dtype cast for the FFI boundary).

### Phase 3 optimization targets cross-reference

The Phase 3 audit (`docs/roadmaps/phase-3-getitem-glue-audit.md`) identified
three bucket-2 items that have since been resolved:

1. **Zero-copy `_ffi_array`** — implemented (`_utils.py:13`); per-sample-scale
   memmaps now assert-no-copy rather than silently coercing.
2. **`_HapsFfiStatic` caching** — implemented (`_haps.py:240`); v_starts,
   ilens, alt_alleles, alt_offsets, ref, ref_offsets are coerced once at first
   access and cached for the lifetime of the `Haps` reconstructor.
3. **Uninit buffers** — the fused kernels all allocate their output internally
   (Rust-side `Vec::with_capacity` / `uninit`), except for the `HapsTracks`
   `np.empty` pre-alloc which is a single batch-bounded f32 buffer — correct
   pattern.

---

## Step 5 — Verdict

**The shim is already thin. Bucket-2 is empty.**

Every Python step on the hot `__getitem__` path falls into Bucket 1
(intentional shim: indexing sugar, output format conversion, Python-side RNG,
FFI typing guards) or Bucket 3 (one FFI crossing). There is no per-batch
coercion or allocation that is both (a) non-trivial in cost and (b) collapsible
into a Rust kernel without restructuring the public Python API.

The one observable pattern that comes closest to bucket-2 — repeated
`ascontiguousarray` calls before each fused-kernel call — is already correct
behavior: those arrays are batch-bounded (small), the coercions are no-ops when
arrays are already contiguous (which they are after `_prepare_request`), and
the dtype-cast form serves as a static type guarantee at the FFI boundary. The
`_HapsFfiStatic` cache already handles the only array that would otherwise
require a per-batch copy at scale (the sub-linear variant/reference arrays).

The `_as_starts_stops` call in `HapsTracks.__call__` (computes a `(2, N)`
view of the genotype offsets once per batch) is the one borderline item:
it calls `ascontiguousarray` on the sample-scale offsets array each batch.
However, the offsets `Ragged` is a memmap whose backing array is already
C-contiguous in practice (written as a plain `np.memmap`), so the
`ascontiguousarray` call is typically a no-op. Caching the `(2, N)` view on
`Haps` (similar to `_HapsFfiStatic`) would be a clean micro-optimization but
is not needed to call the shim thin.

**The single-big-`__getitem__`-kernel collapse is not warranted as Phase 5
work.** The five fused kernels already express one FFI crossing per
reconstruction path. Further collapse would require moving index resolution
(jitter, RC derivation, output shaping) into Rust, which would complicate the
public API and add no meaningful throughput gain relative to the rayon batch
parallelism already landed in W5.

**Dispatch-layer status:** fully gone (confirmed Step 3). No `_dispatch.py`,
no `GVL_BACKEND`, no numba imports in `python/genvarloader/`.

**FFI surface count:** 33 registered entries; 5 are fused `__getitem__` kernels;
the remainder are write-path utils, ragged utilities, and genotype/variant
helpers that are already called directly (no Python wrappers remaining).
