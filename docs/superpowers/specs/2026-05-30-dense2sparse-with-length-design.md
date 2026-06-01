# Use `_dense2sparse_with_length` for VCF/PGEN writes

**Date:** 2026-05-30
**Branch:** `worktree-feat-dense2sparse-with-length`
**Status:** Approved design

## Problem

genoray 2.7.0 added `genoray._svar._dense2sparse_with_length`, which converts a
dense, over-extended region window into **per-haplotype-minimal** sparse output —
identical to what `SparseVar.read_ranges_with_length` produces. GVL's VCF/PGEN
write path does not yet use it, and as a result has two issues:

1. **Over-extension.** The current VCF/PGEN path calls plain `dense2sparse` on the
   over-extended window, so *every* haplotype keeps all ALT variants in the
   extension tail — including haplotypes with few/no deletions, which then extend
   past the region length. The SVAR path already trims per-haplotype to the
   minimal set via `read_ranges_with_length`. VCF/PGEN and SVAR therefore produce
   different sparse output for the same input.

2. **`extend_to_length=False` is a silent no-op for VCF/PGEN.** `_write_from_vcf`
   and `_write_from_pgen` accept `extend_to_length` but never pass it to their
   chunk generators (`_vcf_region_chunks` / `_pgen_region_chunks`), which always
   call `_chunk_ranges_with_length` (which always extends). So `False` still
   extends today.

## Goal

Make VCF/PGEN writes produce per-haplotype-minimal sparse output identical to
SVAR when `extend_to_length=True`, and make `extend_to_length=False` actually
take effect (no extension). Fold in a `max_ends` consistency fix so all three
input types (VCF, PGEN, SVAR) agree on the per-region `chromEnd`.

## The new genoray function

```python
# genoray._svar
def _dense2sparse_with_length(
    genos: NDArray[np.integer],      # (samples, ploidy, variants) — full window
    var_idxs: NDArray[V_IDX_TYPE],   # (variants,) — global, window-aligned
    q_start: int,                    # 0-based, original (unextended) query start
    q_end: int,                      # 0-based exclusive, original query end
    v_starts: NDArray[np.int32],     # POS - 1, window-aligned with var_idxs
    ilens: NDArray[np.int32],        # ILEN (ALT - REF length), window-aligned
    dosages: NDArray | None = None,
) -> Ragged[V_IDX_TYPE] | tuple[Ragged, Ragged]:
    """Convert a dense with_length window into per-haplotype-minimal sparse output,
    identical to SparseVar.read_ranges_with_length for the same query."""
```

**Key constraint:** the function does a per-haplotype length walk over the variant
axis, so it requires the **entire region window** (all memory-chunks concatenated)
in one dense `(samples, ploidy, variants)` array. It cannot be applied
chunk-by-chunk. GVL does not use the `dosages` parameter (genotypes only).

The VCF/PGEN variant index (genoray `_index`) exposes the `POS` (i32) and `ILEN`
(list[i32], first element for bi-allelic variants) columns needed to build
`v_starts` and `ilens`.

## Architecture (Approach 2: shared window helper)

`_vcf_region_chunks` and `_pgen_region_chunks` are ~95% duplicates. Refactor both
to: (a) accept `extend_to_length`, (b) assemble each region's full dense window,
and (c) delegate conversion to one shared helper. `_write_phased_chunked` keeps
its current contract — it consumes `(list[Ragged], region_end, desc)` per region.

### New shared helper

```python
def _window_to_sparse(
    genos: NDArray[np.integer],      # (samples, ploidy, variants) — full region window
    var_idxs: NDArray[V_IDX_TYPE],   # (variants,) — global, window-aligned
    q_start: int,
    q_end: int,
    v_starts: NDArray[np.int32],     # POS - 1, window-aligned
    ilens: NDArray[np.int32],        # ILEN first-elem, window-aligned
    extend_to_length: bool,
) -> Ragged:
    if extend_to_length:
        return _dense2sparse_with_length(
            genos, var_idxs, q_start, q_end, v_starts, ilens
        )
    return dense2sparse(genos, var_idxs)
```

`v_starts` / `ilens` are built by indexing the loaded variant index
(`POS` / `ILEN`) with the window's global `var_idxs`, cast to contiguous int32.

### Data flow per generator

**`extend_to_length=True`:**
1. Read the region's chunks via `_chunk_ranges_with_length` (existing API).
2. Concatenate dense genos along the variant axis into one full window.
3. Build the full-window global `var_idxs`:
   - VCF: unextended idxs (from `_var_idxs`) + `arange` extension tail derived
     from the last chunk's `n_ext`.
   - PGEN: concatenate the per-chunk `chunk_idxs` genoray already returns.
4. Call `_window_to_sparse(...)` once → yield `([single_ragged], region_end, desc)`.

**`extend_to_length=False`:**
1. Read via the non-length API: `VCF.chunk` (per region) / `PGEN.chunk_ranges`.
   These return exactly the in-range variants, aligned with `_var_idxs`.
2. `dense2sparse` per chunk (current structure minus the extension logic).
3. Yield `([chunk_ragged, ...], region_end, desc)`.

Keeping the yielded type as `list[Ragged]` for both paths leaves
`_write_phased_chunked`'s aggregation (concatenate + flatten + lengths) unchanged.

### max_ends fix (folded in)

Each generator computes, once from its index:

```python
v_ends = POS - ILEN_first.clip(upper_bound=0)   # = POS + deletion length
```

For each region, `region_end = v_ends[max retained global v_idx]`, falling back to
the input `chromEnd` when the region has no retained variants. This matches
`_write_from_svar` exactly and is applied to **both** the True and False paths,
replacing the current `region_end = chunk_end` logic. `_write_phased_chunked`
continues to write `region_end` into the output bed's `chromEnd` column.

## Edge cases

- **Empty region** (no variants for any sample): existing "no variant" warning in
  `_write_phased_chunked` still fires; `region_end` falls back to input `chromEnd`.
- **Missing / unnormalized contig:** genoray yields empty chunks (with a warning),
  handled by the empty-region fallback.
- **Memory:** full-window assembly holds one region's dense window in memory at
  once; `max_mem` still bounds the per-chunk *read*. Acceptable — genoray's own
  `read_ranges_with_length` materializes similarly. A pathologically large single
  region is the only risk; noted, not guarded.

## Testing

- **Dependency bump** (applied): `pixi.toml` `genoray = "==2.7.0"`,
  `pyproject.toml` `genoray>=2.7.0,<3`.
- **Regenerate test ground-truth** (`pixi run -e dev gen`): VCF/PGEN stored
  genotypes change (now per-haplotype-minimal), so committed ground-truth must be
  regenerated.
- **Parity test (acceptance criterion):** VCF and PGEN datasets must produce
  **identical** sparse output to the SVAR dataset for the same regions/samples.
  This is the strongest correctness check and the explicit acceptance criterion.
- Existing reconstruction/dataset tests pass against regenerated ground-truth.

## Out of scope

- SVAR writing (`_write_from_svar`) — already correct, untouched.
- Public API signatures — `gvl.write`'s `extend_to_length` keeps its meaning; it
  simply now actually takes effect for VCF/PGEN when `False`.
- Dosage support in the new conversion path (genotypes only).
- Full unification of the SVAR write path with VCF/PGEN (different storage
  mechanism; out of scope).
