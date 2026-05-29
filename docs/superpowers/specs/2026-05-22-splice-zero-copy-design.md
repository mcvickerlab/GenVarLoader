# Zero-copy splicing via query-flattened reconstruction

**Status:** Implemented
**Date:** 2026-05-22

## Problem

Spliced output is built in two phases: the reconstructor writes per-region
haplotypes/tracks into a Ragged buffer in `(query, ploidy, bytes)` C-order,
and `_cat_length` (in `_dataset/_splice.py`) afterwards concatenates the
per-region bytes that belong to the same splice row.

The concatenation step exists only because the bytes for a single
`(splice_row, sample, ploidy)` are scattered across the buffer. For
`ploidy == 1` (or any inner-fixed dim of size 1) it collapses to
`np.add.reduceat` on the lengths. For `ploidy > 1` (or any non-trivial inner
fixed dim) it falls into `_cat_length_inner`, which goes through
`ak.flatten`/`ak.concatenate` to interleave bytes correctly. That path
allocates new awkward arrays and copies all output bytes, on every batch.
Hap-tracks (`(b, t, p, ~l)`) currently raise `NotImplementedError` outright.

If we instead arrange queries so the kernel writes bytes directly in the
final spliced layout, the concatenation step disappears — splicing collapses
to a coarser reinterpretation of the same byte buffer.

## Core idea

The numba kernel `reconstruct_haplotypes_from_sparse` writes each
`(query, hap)` sub-write at `out[out_offsets[k_idx]:out_offsets[k_idx+1]]`
where `k_idx = query * ploidy + hap`. The kernel locks ploidy as the
innermost buffer axis, which is why pure query reordering can't produce
`(splice_row, sample, ploidy, None)` layout.

We sidestep this by **flattening ploidy (and any other inner-fixed axis,
e.g. tracks) into the query dimension**: call the kernel with `ploidy = 1`
and `B·E` queries, where `E` is the product of inner-fixed extents. We then
control the global write order completely.

For spliced output we permute queries to
`(splice_row, sample, e_inner, splice_element)` C-order. The kernel writes
bytes in that order. Splice-row offsets become
`np.add.reduceat(permuted_lengths, group_starts)`. The same buffer is then
exposed as a Ragged of shape `(n_splice_rows, n_samples, *inner_fixed, None)`
with no byte copies.

`_cat_length` and `_cat_length_inner` are deleted.

## Scope

- **In scope:** spliced reconstruction for `Ref`, `Haps` (both `RaggedSeqs`
  and `RaggedAnnotatedHaps`), and `Tracks` (sample + annotation tracks),
  shape combinations that work today.
- **Out of scope:** hap-tracks (`(b, t, p, ~l)`). The reconstructor raises
  `NotImplementedError` before building the splice plan. Same user-visible
  behavior as today, just raised from a different layer.
- **Unchanged:** the non-spliced reconstruction path. Only the spliced
  branch in `_recon` / `_call_float32` / `Ref.__call__` is touched, and only
  when invoked from `_getitem_spliced` / `RefDataset._getitem_spliced`.
- **Unchanged:** the numba kernel `reconstruct_haplotypes_from_sparse` and
  the inner-loop helper. We only change how it is called.

## Constraints kept from today

- `output_length` must be `"ragged"` or `"variable"` (not `int`).
- `jitter == 0`.
- `deterministic == True`.

These are enforced in `_getitem_spliced` already and are not relaxed here.

## Design

### `SplicePlan` and `build_splice_plan`

A single, shape-agnostic helper in `_dataset/_splice.py` builds everything
the reconstructors need:

```python
@define
class SplicePlan:
    perm: NDArray[np.intp]  # length K·E
    permuted_lengths: NDArray[np.int32]  # length K·E
    permuted_out_offsets: NDArray[OFFSET_TYPE]  # length K·E + 1
    group_offsets: NDArray[OFFSET_TYPE]  # length n_rows·S·E + 1
    out_shape: tuple[int | None, ...]  # final ragged outer shape


def build_splice_plan(
    lengths: NDArray[np.int32],  # shape (B, *inner_fixed)
    splice_row_offsets: NDArray[np.int64],  # from SpliceIndexer.parse_idx
    n_samples: int,
    n_rows: int,
) -> SplicePlan: ...
```

`B` is the total number of `(splice_row, sample, splice_element)` queries
already produced by `SpliceIndexer.parse_idx`. `inner_fixed` is the product
of inner ragged-buffer axes the caller wants flattened into queries
(`(ploidy,)` for Haps, `(n_tracks,)` for sample-track Tracks, `()` for
Ref). `E = prod(inner_fixed)`.

The helper owns:

- Constructing `perm` so that the global k-index order becomes
  `(splice_row, sample, *inner_fixed, splice_element)` C-order.
- Permuting `lengths` and reducing to `permuted_lengths`.
- `permuted_out_offsets = lengths_to_offsets(permuted_lengths)`.
- `group_offsets`: built by `np.add.reduceat(permuted_lengths, starts)`
  where `starts` are the indices in `permuted_lengths` corresponding to
  each `(splice_row, sample, e_inner)` boundary. Length
  `n_rows · n_samples · E + 1`.
- `out_shape = (n_rows, n_samples, *inner_fixed, None)`.

The helper depends on `n_samples` and `n_rows` so the call sites don't have
to do the boundary math.

### Reconstructor wiring

Each reconstructor's `__call__` (or `_recon` / `_call_float32`) accepts an
optional `splice_plan: SplicePlan | None`. Single mode — when present, the
function:

1. Computes per-query lengths in the standard `(B, *inner_fixed)` shape it
   already computes today.
2. Asserts that `plan.permuted_lengths`'s implied lengths match (cheap
   sanity check — the caller built the plan from the same lengths).
3. Reshapes/permutes its kernel inputs (`regions`, `shifts`,
   `geno_offset_idx`, etc.) to `(B·E, 1)` in `perm` order.
4. Allocates one flat output buffer sized `plan.permuted_out_offsets[-1]`
   and any parallel buffers (`annot_v_idxs`, `annot_ref_pos`).
5. Calls `reconstruct_haplotypes_from_sparse` (or the track equivalent)
   with `ploidy=1`, `out_offsets=plan.permuted_out_offsets`, and the
   permuted inputs.
6. Wraps the buffer(s) as Ragged with the per-element offsets (not the
   group offsets) for downstream `_rc`.

When `splice_plan` is `None`, the function takes its existing code path
unchanged.

The caller (`Dataset._getitem_spliced`, `RefDataset._getitem_spliced`)
builds the plan once after computing `lengths` and passes it to `_recon`.

### Length computation ordering

`Haps._recon` currently computes `hap_lengths` *before* allocating
`out_offsets` (see `_haplotype_ilens`, `_reconstruct.py:443`). The spliced
caller computes lengths, builds the plan, then calls into `_recon` with the
plan. Mechanically:

- `_getitem_spliced` calls a new lightweight method
  `Haps.haplotype_lengths_for_recon(idx, regions, keep, keep_offsets)`
  that returns `hap_lengths` of shape `(B, ploidy)`.
- It also calls the analogous methods on `Tracks` / `Ref`.
- It then builds the plan and dispatches to `_recon(..., splice_plan=plan)`.

For `Ref` and `Tracks` the lengths are deterministic from regions alone, so
this is cheap. For `Haps` the length computation already runs in the
current `_recon`; we just lift it out so the plan can be built first.

### Reverse complement timing

`_rc` continues to run before the splice-grouping is exposed. After the
kernel writes the permuted buffer, the data is wrapped as Ragged with
`permuted_out_offsets` (per-element granularity) and a permuted `to_rc`
mask of length `B·E`. The existing `_rc` body (which calls
`reverse_complement` and `ak.where(to_rc, rev, rag)` then `ak.to_packed`)
runs as today. The single `to_packed` byte rewrite already produces the
canonical contiguous layout we want; reassembling under `group_offsets`
afterwards is free.

When `rc_neg` is `False`, no copies happen at all.

For `RaggedAnnotatedHaps`, the three parallel buffers (`haps`, `var_idxs`,
`ref_coords`) share the plan; `_rc` reverses each, gated on the same mask.

### Final assembly

After `_rc`, the spliced wrapper builds the final Ragged from the same
data buffer and `plan.group_offsets`. For `RaggedAnnotatedHaps` it builds
three such views. The returned tuple shape matches what
`Dataset._getitem_spliced` returns today, with `_cat_length` removed from
the call.

### Files touched

| File | Change |
|------|--------|
| `python/genvarloader/_dataset/_splice.py` | Delete `_cat_length`, `_cat_length_inner`. Add `SplicePlan`, `build_splice_plan`. |
| `python/genvarloader/_dataset/_reconstruct.py` | Add optional `splice_plan` to `Ref.__call__`, `Haps._recon` / `_get_haplotypes`, `Tracks.__call__` / `_call_float32`. Lift `Haps` length computation into a small public method. Raise `NotImplementedError` for hap-tracks before plan build. |
| `python/genvarloader/_dataset/_impl.py` | `_getitem_spliced` builds the plan, passes it down, drops `_cat_length` call and import. |
| `python/genvarloader/_dataset/_reference.py` | `_getitem_spliced` analogous to `_impl.py`. Drop `_cat_length` import. |
| `python/genvarloader/_dataset/_genotypes.py` | Unchanged. Same kernel, called with `ploidy=1`. |

## Testing

- Existing splice parity tests in `tests/` (the suite that compares spliced
  output against per-region reconstruction + concatenation) is the primary
  correctness gate. It must pass unchanged.
- Unit tests for `build_splice_plan`:
  - `inner_fixed = ()`, `(1,)`, `(2,)`, `(3,)` permutation correctness.
  - `n_samples = 1` and `n_samples > 1`.
  - splice rows of size 1 and mixed sizes.
  - `permuted_out_offsets[-1] == group_offsets[-1] == sum(lengths)`.
- Regression test: spliced output with `ploidy = 2` and `rc_neg = True`
  matches the current behavior byte-for-byte.
- Annotated haps spliced output: `var_idxs` and `ref_coords` align with
  `haps` after splicing for `ploidy > 1`.
- Hap-tracks spliced query raises `NotImplementedError` (same observable
  behavior as today, source line changes).

## Non-goals

- Performance work on the non-spliced path.
- Restructuring the kernel signature (e.g. accepting non-monotone offsets
  or moving ploidy to outer). The flattened-query call is sufficient.
- Splice support for hap-tracks `(b, t, p, ~l)`.

## Open questions

None outstanding at spec time. Hap-tracks support, if pursued later, is a
follow-up that extends `inner_fixed` to `(n_tracks, ploidy)` and adds
test coverage; the helper is already generic enough.
