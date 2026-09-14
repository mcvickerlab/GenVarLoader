# Sparse `svar2_ranges` cache

Design for #357 (on-disk dense range cache) and the remainder of #355 (dense
in-memory `n_variants`). Both are the same defect: an array indexed by
`(region, sample, ploid)` that is over 99% empty at cohort scale.

## Problem

A `.svar2`-backed dataset stores two `(R, S, P, 2)` int64 memmaps under
`genotypes/svar2_ranges/` — 64 bytes for every `(region, sample)` at ploidy 2 —
plus, at open time, a dense `(R, S, P)` int32 `n_variants` of zeros.

Measured on All of Us v9 (535,662 samples, MANE Select exons, GVL 0.42.1):

| chromosome | regions | cache on disk | `n_variants` at open |
|---|---|---|---|
| chr21 | 1,901 | 61 GB | 8 GB |
| chr22 | 3,734 | 128 GB | 16 GB |
| chr19 | 11,834 | ~406 GB (projected) | 51 GB |
| genome | 202,053 | ~6.9 TB (projected) | 866 GB |

Across the AoU chr22 cache only **0.45%** of `(region, sample, ploid)` windows
are non-empty. The other 99.55% are 16 bytes of `(x, x)`. A per-chromosome
build therefore needs a work disk sized for the cache rather than for the data,
and every sequential pass streams the whole thing through the page cache to
touch about 0.6 GB of useful windows.

Both terms scale with `regions x samples`. Nothing else in the dataset does.

## Key fact: empty cells carry no information

`genoray_core::query::gather_vk` (`src/query/gather.rs:145`, and the batched
form at `:762`) consumes a range only as a slice bound and an index base:

```rust
let (vs, ve) = (vk_snp_range[row].start, vk_snp_range[row].end);
for (k, &pos) in snp_positions[vs..ve].iter().enumerate() {
    let j = vs + k;
    ...
}
```

When `vs == ve` the loop body never executes, so the value of `vs` is
unobservable. Storing `(0, 0)` for an empty cell is therefore **byte-identical**
to storing its true insertion point. This is what makes a sparse encoding exact
rather than approximate, and it is the assumption the whole design rests on.

## Key fact: there is exactly one read seam

`Svar2Haps._gather_inputs` (`python/genvarloader/_dataset/_svar2_haps.py:1546`)
is the only code that touches the `vk_*` arrays. It fancy-indexes
`cache.vk_snp_range[r_q, si_q]`, reshapes to `(n*P, 2)` and hands the result to
the Rust FFI. Any layout that can answer that one question is a drop-in: no
kernel, no FFI signature and no other Python module changes.

## Layout

Replace the two dense memmaps with one key-sorted entry table holding only the
cells where the SNP **or** the indel range is non-empty.

| File | Shape | dtype | Notes |
|---|---|---|---|
| `cell_key.npy` | `(N,)` | int64 | `(r * S + slot) * P + ploid`, strictly increasing |
| `cell_vk.npy` | `(N, 4)` | int64 | `snp_start, snp_end, indel_start, indel_end` |

`dense_snp_range.npy`, `dense_indel_range.npy` and `sample_cols.npy` are
unchanged — they are per-region or per-sample and already small.

`svar2_meta.json` gains `"layout": "sparse"`. A file without the key is the
existing dense layout and is read through the dense reader (see
"Backward compatibility").

The key is a C-order flat slot index into the same `(R, S, P)` space the dense
array used, so sorting by key is region-major, then sample slot, then ploid.
The maximum value for the AoU genome is `202,053 x 535,662 x 2 = 2.2e11`,
comfortably inside int64.

One entry costs 40 bytes against 32 bytes for one dense cell (two int64 pairs),
so the sparse layout is larger only above **80% fill**. #357 option 2 proposed a
fill threshold with a dense fallback to guard that case; this design does not,
because the crossover cannot bite where it would cost anything. Fill is under
1% at every scale where the cache is large, and a dataset dense enough to cross
80% is small enough that both layouts are negligible — the test fixture's dense
cache is about 6 KB. A threshold would buy kilobytes at the price of two live
write paths forever.

The same reasoning retires the `svar2_ranges="lazy"` flag of option 1: once
chr22's cache is 720 MB rather than 128 GB, skipping the cache buys nothing.

Projected sizes at the measured 0.45% fill:

| dataset | dense | sparse | entries |
|---|---|---|---|
| AoU chr22 | 128 GB | ~720 MB | 1.8e7 |
| AoU chr19 | ~406 GB | ~2.3 GB | 5.7e7 |
| AoU genome | ~6.9 TB | ~39 GB | 9.7e8 |

Storage now scales with the number of variants observed inside regions, not
with `regions x samples`.

## Lookup

`_Svar2Cache` keeps `dense_snp_range`, `dense_indel_range` and `sample_cols`,
and delegates the per-sample ranges to a small protocol:

```python
class _RangeLookup(Protocol):
    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(vk_snp, vk_indel), each (len(r_q) * P, 2) C-contiguous int64."""

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """(key, vk) blocks in ascending key order. Used by concat."""
```

The sparse implementation is a handful of vectorized numpy steps, with no
Python loop and no per-region offsets array:

```python
tgt = ((r_q[:, None] * self.S + si_q[:, None]) * P + np.arange(P)).ravel()
order = np.argsort(tgt)            # probe locality, not correctness
probe = tgt[order]
pos = np.minimum(np.searchsorted(self.key, probe), len(self.key) - 1)
hit = self.key[pos] == probe
found = np.where(hit[:, None], self.vk[pos], 0)

vk = np.empty_like(found)
vk[order] = found                  # undo the probe sort
return (
    np.ascontiguousarray(vk[:, 0:2]),
    np.ascontiguousarray(vk[:, 2:4]),
)
```

Sorting the probe keys first turns `searchsorted`'s random walk into a
near-sequential one, which matters because the table is memmapped. An empty
table (`N == 0`) is handled by returning zeros without probing.

### Why this is not slower in practice

A dense lookup is one page touch per query row; a sparse lookup is `log2(N)`
touches. At AoU chr22, `N ~ 18M` entries is 720 MB, which fits in page cache;
the 128 GB dense array does not. The dense layout's O(1) index is only cheap
when the array is small enough to be resident, and at the scale where that
holds the sparse table is small too. This is benchmarked on the fixture dataset
during implementation; a measurable regression on small datasets is a blocker,
not an accepted cost.

## Write path

`_write_from_svar2` (`_write.py:1139`) currently scatters hap-major chunks into
region-major memmaps. It instead accumulates entries per contig and appends
them, sorted, to the two output files.

Per contig `c` covering region rows `[lo, hi)`:

1. `stream = svar2._find_ranges_chunked(c, starts, ends, samples=sel, max_mem=max_mem)`
   (unchanged).
2. For each chunk covering sample slots `[s0, s1)`, transpose the hap-major
   `(ns, P, rc, 2)` arrays to region-major `(rc, ns, P, 2)` as today, then:
   ```python
   nonempty = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
   ri, sj, pj = np.nonzero(nonempty)
   keys = ((lo + ri) * S + (s0 + sj)) * P + pj
   vals = np.stack(
       [
           snp[..., 0][ri, sj, pj], snp[..., 1][ri, sj, pj],
           indel[..., 0][ri, sj, pj], indel[..., 1][ri, sj, pj],
       ],
       -1,
   )
   ```
3. After the contig's chunks, concatenate, `np.argsort(keys, kind="stable")`,
   and append to `cell_key.npy` / `cell_vk.npy`.

`max_end_keys` handling and the returned `chromEnd` extension are unchanged.

**No global sort is needed.** `_prep_bed` runs `sp.bed.sort`, so `bed` is
contig-grouped and `contig_offset` advances monotonically (`_write.py:1493`
already relies on this). Region indices therefore increase across contig
iterations, and appending per-contig sorted blocks yields a globally sorted
file. The implementation asserts this rather than assuming it.

**Memory bound.** The per-contig buffer is 40 bytes times that contig's
non-empty cells — about 720 MB at AoU chr22. This is proportional to variants
observed, not to `R x S`, which is the point of the change. If it ever binds,
each chunk's own entries are already key-sorted (region outer, contiguous slot
block, ploid innermost), so the per-contig combine can become a k-way merge of
sorted runs without changing the format. Not built now (YAGNI).

### Preflight

`_svar2_ranges_cache_bytes` can no longer be computed before the ranges are
known. `_svar2_preflight` changes to:

- log `40 * R * S * P` explicitly labelled as a **worst case that is not
  reached in practice**, alongside the note that actual size scales with
  variants inside regions;
- log the running actual byte count after each contig, so a user sees the real
  trajectory early instead of only after committing to the build.

The free-disk check keeps its current warn-do-not-raise behaviour, and fires
only when free space is below the worst case.

## `n_variants` (the #355 remainder)

`_svar2_haps.py:277` allocates `np.zeros((R, S, P), np.int32)` — 51 GB at AoU
chr19 — that is only ever read for its shape and for the documented zeros
`Dataset.n_variants()` returns on SVAR2 (#363). Replace it with a read-only,
zero-stride view:

```python
self.n_variants = np.broadcast_to(
    np.zeros((), np.int32), (self.n_regions, self.n_samples, self.ploidy)
)
```

Every consumer either reads `.shape` (`_impl.py:1363`) or fancy-indexes it
(`_impl.py:1290`, `_haps.py:1212`), both of which materialise only the indexed
subset. Nothing writes into it; the read-only flag turns a future write into a
loud error rather than a silent per-cell allocation.

## Concat

`_concat_svar2_ranges` (`_concat.py:235`) merges the caches with
`gather_fixed(..., record_bytes=16)`, which assumes fixed per-cell records and
cannot work on a sparse table. It is replaced by a vectorized key remap that is
layout-agnostic on the input side.

For each input dataset `d`, with `r_map_d` (input region -> merged region) and
`s_map_d` (input slot -> merged slot) derived from the existing `order`:

```python
for key, vk in input_d.iter_entries():
    r, rem = np.divmod(key, S_d * P)
    slot, ploid = np.divmod(rem, P)
    new_key = (r_map_d[r] * S + s_map_d[slot]) * P + ploid
```

Concatenate every input's `(new_key, vk)` and sort once by `new_key`.

Because inputs are read through `iter_entries`, a dense input merges into a
sparse output with no special case: the dense reader's `iter_entries` scans its
memmap in region blocks and yields the non-empty cells. Mixed dense/sparse
concat therefore works without extra code, and the output is always sparse.

`dense_*_range` and `sample_cols` merging is unchanged. The output
`svar2_meta.json` gains `"layout": "sparse"` and replaces the `vk_*` shape
entries with `{"cell_key": {"shape": [N]}, "cell_vk": {"shape": [N, 4]}}`.

## Backward compatibility

`svar2_meta.json` without `"layout"` means the dense layout.
`Svar2Haps.from_path` picks the reader from that key:

- `_DenseRanges` — memmaps the two `(R, S, P, 2)` arrays and implements
  `lookup` with today's fancy-index code, verbatim. About 15 lines, and the
  only thing keeping existing AoU datasets openable.
- `_SparseRanges` — the searchsorted path above.

`gvl.write` only ever emits sparse. There is no migration tool and no public
API change; re-running `gvl.write` is how an existing dataset is shrunk.

The format changelog gains a row: `0.43.0` — `genotypes/svar2_ranges/` switches
to a key-sorted sparse layout (`cell_key.npy` + `cell_vk.npy`;
`svar2_meta.json` gains `layout`); the dense layout still opens.

## Testing

The contract is byte-identical output. The existing SVAR2 suite
(`tests/dataset/test_svar2_*.py`, `tests/test_svar2_*.py`) is the primary
regression gate and must pass unchanged.

New tests:

1. **Lookup parity.** On the fixture dataset, `_SparseRanges.lookup` equals
   `_DenseRanges.lookup` for every `(r_q, si_q)` combination, including
   duplicated and out-of-order query rows.
2. **All-empty cells.** A region with no variants for any sample, and a sample
   with no variants in any region, both yield `(0, 0)` ranges and produce the
   same haplotypes and variants as the dense path.
3. **Write round-trip.** `gvl.write` to sparse, open, and compare haplotypes,
   variants and realigned tracks byte-for-byte against a dataset written with
   the dense layout.
4. **Dense dataset still opens.** A fixture whose `svar2_meta.json` has no
   `layout` key opens and reads correctly.
5. **Concat.** Both axes, over sparse x sparse and dense x sparse inputs,
   against a dense-path reference.
6. **`n_variants` is free.** Zero strides, read-only, `int32`;
   `Dataset.n_variants()` shape and values unchanged. A `tracemalloc` test at
   `11,834 x 535,662 x 2` asserts under 1 MiB.
7. **Size.** On the fixture, `cell_key.npy + cell_vk.npy` is no larger than the
   dense layout would have been.

## Out of scope

- The `svar2_ranges="lazy"` / no-cache mode of #357 option 1. A 720 MB cache
  removes its motivation, and it would add a second read path answering the
  same question.
- Narrowing an entry below 40 bytes. Storing `(start int64, len int32)` per
  channel would make it exactly 32 bytes — equal to a dense cell at 100% fill
  and 20% smaller on disk — at the price of reconstructing `end = start + len`
  on every lookup. Worth measuring later; it is a pure format tweak behind the
  same `layout` key.
- genoray's O(S^2) `_sample_idxs` name resolution (d-laub/genoray#168), dodged
  here as it is today by passing `samples=None` when the selection matches the
  store's order.

## Risks

1. **Concat key remap** is the fiddliest part and the least covered by existing
   tests. Mitigated by routing every input through `iter_entries` so there is
   one merge implementation, and by testing against a dense-path reference.
2. **`searchsorted` on small datasets** could regress per-batch latency where
   the dense array was resident. Benchmarked during implementation; a
   measurable regression on the fixture blocks the change.
3. **Per-contig write buffer** is unbounded in principle. Bounded in practice
   by variants-in-regions; the k-way merge escape hatch is designed for but not
   built.
