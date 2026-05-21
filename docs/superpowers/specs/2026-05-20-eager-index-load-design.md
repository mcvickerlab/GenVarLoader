# Eager variant-index load in `gvl.write` + budget enforcement

## Problem

`gvl.write` accounts for the genoray variant index's resident memory by
subtracting `variants.nbytes` from `max_mem` (see
`python/genvarloader/_dataset/_write.py:205`). Both `genoray.VCF` and
`genoray.PGEN` can be constructed without the index loaded
(`VCF(..., with_gvi_index=False)` or `PGEN(..., load_index=False)`), and in
that case `nbytes` returns `0` (or undercounts — see
`genoray/_vcf.py:330-337` and `genoray/_pgen.py:278-293`).

`_write_from_vcf` then calls `vcf._load_index()` at
`_write.py:366` *inside* the writing loop, after the budget has already been
computed. The resulting `effective_max_mem` overstates available memory by
the size of the index, which can be hundreds of MB on large cohorts —
silently blowing the soft budget.

The user's instinct was "can we skip the index?" — no. The output dataset
hardlinks `variants.arrow` to the genoray index file itself
(`_write.py:375, :486`), and downstream `Dataset` reads variants from this
file. The index is structural to the on-disk format.

## Goal

Make `gvl.write`'s memory accounting honest, and refuse to start when the
post-index budget is too small to make progress.

## Design

### 1. Load the index eagerly, before measuring `nbytes`

In `_write.py`, after the reader has been resolved from a path
(around line 169, before `available_samples` is computed), ensure the
index is resident:

- `VCF`: if `vcf._index is None`, run the existing
  `_valid_index()` / `_write_gvi_index()` / `_load_index()` sequence that
  currently lives in `_write_from_vcf:361-366`. Move it up to the top-level
  `write` function.
- `PGEN`: call `pgen._init_index()`. This is idempotent (no-op when the
  index is already loaded — see `_pgen.py:242-244`).
- `SparseVar`: unchanged; it has no lazy-index concept relevant here.

After this, `idx_bytes = variants.nbytes` at line 205 reflects reality for
both reader types.

### 2. Hard-error when the post-index budget can't fit a single variant chunk

Replace the existing warn-at-50% block (`_write.py:212-218`) with an
enforcement check:

```python
if isinstance(variants, VCF):
    bytes_per_var = variants.n_samples * variants.ploidy  # Genos8 = 1 byte
elif isinstance(variants, PGEN):
    bytes_per_var = variants.n_samples * variants.ploidy * 4  # int32
else:
    bytes_per_var = 0  # SparseVar: no chunking path

if bytes_per_var and effective_max_mem < bytes_per_var:
    raise ValueError(
        f"max_mem ({format_memory(max_mem)}) is too small: the variant "
        f"index alone consumes {format_memory(idx_bytes)}, leaving "
        f"{format_memory(max(effective_max_mem, 0))} for chunking, but "
        f"at least {format_memory(bytes_per_var)} is needed per variant. "
        f"Increase max_mem."
    )
```

The existing informational log line (`_write.py:207-211`) stays.

### 3. Drop the now-redundant block in `_write_from_vcf`

Lines 361-368 of `_write_from_vcf` collapse to a single
`assert vcf._index is not None` — the caller now guarantees the index is
loaded. The multi-allelic check at line 370 is unchanged but is now
guaranteed to have a loaded index.

`_write_from_pgen` is structurally fine but should also be passed a
`pgen` whose index is loaded; the current code path already assumes this.

### 4. Docstring

Update `_write.py:95-101` (the `max_mem` parameter doc):

> Approximate maximum total memory to use, including the genoray variant
> index. The reader's index is loaded eagerly at the start of `write`, and
> its `nbytes` is subtracted from `max_mem` to determine the budget for
> genotype chunking. A `ValueError` is raised if the budget remaining
> after the index is too small to fit even a single variant chunk. This is
> a soft limit and may be exceeded by a small amount.

## Out of scope

- Skipping the index entirely. The on-disk dataset format requires it.
- Adding a `bytes_per_genotype` property to genoray. The branch in
  `_write.py` is small and local; not worth API surface.
- Estimating PGEN's `_sei` (StartsEndsIlens) cache before `_init_index` —
  `_init_index` populates `_sei`, and `nbytes` covers it afterward.

## Test plan

- Existing `gvl.write` tests must pass with the index now eagerly loaded
  (behavior is otherwise unchanged when `max_mem` is generous).
- New test: construct a `VCF` with `with_gvi_index=False`, call `gvl.write`
  with a tight `max_mem`, assert `ValueError` is raised with a message
  pointing at the index size.
- New test: construct a `PGEN` with `load_index=False`, same assertion.
- New test: with a comfortable `max_mem`, both lazy-index constructions
  produce identical output bytes to their eager-index counterparts.
