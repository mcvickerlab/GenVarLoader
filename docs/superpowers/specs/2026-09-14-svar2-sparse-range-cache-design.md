# Sparse `svar2_ranges` cache

Design for #357 (on-disk dense range cache) and the remainder of #355 (dense
in-memory `n_variants`). Both are the same defect: an array indexed by
`(region, sample, ploid)` that is over 99% empty at the scale that motivated
the issue.

Revised after adversarial review. Findings that changed the design, rather than
just its prose, are called out in "What review changed" at the end.

## Problem

A `.svar2`-backed dataset stores two `(R, S, P, 2)` int64 memmaps under
`genotypes/svar2_ranges/` — 32 bytes for every `(region, sample, ploid)` cell —
plus, at open time, a dense `(R, S, P)` int32 `n_variants` of zeros.

Measured on All of Us v9 (535,662 samples, MANE Select exons, GVL 0.42.1):

| chromosome | regions | cache on disk | `n_variants` at open |
|---|---|---|---|
| chr21 | 1,901 | 65.2 GB | 8.1 GB |
| chr22 | 3,734 | 128.0 GB | 16.0 GB |
| chr19 | 11,834 | 405.7 GB (projected) | 50.7 GB |
| genome | 202,053 | 6.93 TB (projected) | 865.9 GB |

Across the AoU chr22 cache only **0.45%** of `(region, sample, ploid)` windows
are non-empty. The other 99.55% are 16 bytes of `(x, x)`. A per-chromosome
build needs a work disk sized for the cache rather than for the data, and every
sequential pass streams the whole thing through the page cache to touch about
0.58 GB of useful windows.

Both terms scale with `regions x samples`. Nothing else in the dataset does.

## Key fact: empty cells carry no information

This is what makes a sparse encoding exact rather than approximate, and the
whole design rests on it. It is an **upstream** contract in genoray, not in this
repo: `genoray_core`, pinned at `d-laub/genoray` rev `d66ec0e` (`Cargo.toml:31-32`).

GenVarLoader's read path is `gather_haps_readbound_impl`
(`genoray:src/query/gather.rs:679`). Its var-key walk (`:772`, `:791`) is:

```rust
let (vs, ve) = (vk_snp_range[row].start, vk_snp_range[row].end);
let mut snp_run: Vec<T> = Vec::with_capacity(ve.saturating_sub(vs));
for (k, &pos) in snp_positions[vs..ve].iter().enumerate() {
    let j = vs + k;
    ...
}
```

`j = vs + k` is *inside* the loop body, so it never evaluates when `vs == ve`.
Nothing else observes an empty range: no total-span or capacity computation
across rows, no min/max reduction over inputs, no monotonicity or contiguity
assertion (`HapRanges::new` at `:621-660` asserts slice *lengths* only), no use
of one row's `end` as the next row's `start`, no sort or dedup keyed on start,
and no `debug_assert!` on ranges anywhere in `src/query/`.

Storing `(0, 0)` for an empty cell is therefore not merely equivalent — it is
**strictly safer**. Rust slicing panics if `vs > ve` or `ve > len`, and
`Vec::with_capacity(ve.saturating_sub(vs))` masks only the first of those. The
dense layout stores `(x, x)` with `x` up to `positions.len()`: in bounds, but at
the edge. `(0, 0)` is unconditionally in bounds even for an empty `positions`.

Two consequences to record so they are not rediscovered:

1. **`gather_vk` (`gather.rs:145`) is NOT this path.** It is `pub(crate)`,
   reached only from `gather_ranges_impl` (`:548`) and `oracle.rs:149`, neither
   of which GVL calls. `gather_haps_readbound_impl` deliberately keeps the walk
   hand-inlined (see its comment at `:800-820`). Cite `:772`, not `:145`.
2. **`dense_snp_range` / `dense_indel_range` cannot be sparsified.**
   `dense_abs_row` (`gather.rs:920-928`) computes `on_disk.start + (i - out.start)`,
   and `src/ffi/mod.rs:1990-1992` passes the on-disk dense range in a second time
   for exactly this. Those two channels use `.start` as an index base
   unconditionally. They stay dense here because they are per-region and small,
   **and** because they could not be sparsified even if they were not.

This invariant lives in a pinned upstream crate and nothing in GVL's suite would
catch genoray changing it. The parity tests below are the only guard.

## Read seam

`Svar2Haps._gather_inputs` (`python/genvarloader/_dataset/_svar2_haps.py:1546`)
is the only place the `vk_*` arrays are *read for a query*. It fancy-indexes
`cache.vk_snp_range[r_q, si_q]`, reshapes to `(n*P, 2)` and hands the result to
the Rust FFI. No kernel and no FFI signature changes: all seven entry points
(`src/ffi/mod.rs:1029, 1266, 1340, 1405, 1598, 1695, 1879`) funnel through
`arr2_to_ranges` (`:38-42`) into `HapRanges::new`, and the `_into` / `hap_diffs`
variants skip the ranges entirely when handed a cached `Svar2ReadboundGather`.

That is one *read* seam. The change still touches construction
(`_svar2_haps.py:418-441`), the writer (`_write.py`), concat (`_concat.py`) and
the meta schema. The earlier draft's "no other Python module changes" was wrong.

## Layout

Replace the two dense memmaps with a region-CSR table holding only the cells
where the SNP **or** the indel range is non-empty.

| File | Shape | dtype | Notes |
|---|---|---|---|
| `region_ptr` | `(R+1,)` | int64 | `region_ptr[r]:region_ptr[r+1]` is region `r`'s entry block |
| `cell_id` | `(N,)` | int32 | `slot * P + ploid`, ascending within each region block |
| `cell_vk` | `(N, 3)` | mixed | see below |

`cell_vk` is a structured/record array of `snp_start` int64, `snp_len` int32,
`indel_start` int64, `indel_len` int32 = 24 bytes. Storing lengths rather than
ends costs one add on gather and saves 8 bytes per entry.

Total **28 bytes per entry** against **32 bytes per dense cell**, so the sparse
layout is **strictly smaller than dense at every fill level**. That is what
removes #357 option 2's fill threshold and dense-fallback writer, and #357
option 1's `svar2_ranges="lazy"` flag: there is no fill regime where the old
layout wins, and at 0.45% fill a 128 GB cache becomes 504 MB.

`dense_snp_range.npy`, `dense_indel_range.npy` and `sample_cols.npy` are
unchanged.

### Sizes

| dataset | dense | sparse | entries | `region_ptr` |
|---|---|---|---|---|
| AoU chr22 | 128.0 GB | 504 MB | 1.8e7 | 30 KB |
| AoU chr19 | 405.7 GB | 1.60 GB | 5.7e7 | 95 KB |
| AoU genome | 6.93 TB | 27.3 GB | 9.74e8 | 1.6 MB |

Storage now scales with variants observed inside regions, not with
`regions x samples`.

### Bounds

`cell_id = slot * P + ploid` must fit int32: the writer asserts
`S * P < 2**31` (1.07e6 at AoU, six orders of margin) and raises a clear error
otherwise. `snp_start` / `indel_start` stay int64 — they are absolute indices
into a contig's var-key table across the whole cohort and can exceed 2**31.

### Fill is not always low

The 0.45% figure is MANE Select exons (~150-200 bp). Inverting it to a per-bp
hazard and extrapolating: a 2 kb window is ~5-6% fill, an 8 kb window ~20%, an
Enformer 196 kb window ~99.5%, Borzoi ~100%. GVL's headline use case is large
sequence-model windows, so **high fill is a normal operating point, not an edge
case**. This is precisely why the 28-byte entry matters: at the 40-byte entry of
the first draft, a Borzoi-scale cohort dataset would have been 25% *larger* than
dense. At 28 bytes the worst case is 0.875x dense and the crossover does not
exist.

The writer records realized fill (`"fill": N / (R*S*P)`) in `svar2_meta.json` so
the operating point is visible without re-deriving it.

## Lookup

`_Svar2Cache` keeps `dense_snp_range`, `dense_indel_range` and `sample_cols`,
and delegates the per-sample ranges to a protocol:

```python
class _RangeLookup(Protocol):
    def lookup(
        self, r_q: NDArray[np.integer], si_q: NDArray[np.integer], P: int
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """(vk_snp, vk_indel), each (len(r_q) * P, 2) C-contiguous int64."""

    def iter_entries(self) -> Iterator[tuple[NDArray[np.int64], NDArray[np.int64]]]:
        """(global_key, vk) blocks in ascending key order, for concat."""
```

Both layouts are built by a path-level factory `_ranges_reader(ranges_dir)`,
used by `Svar2Haps.from_path` **and** by concat. Concat must not go through
`from_path`, which also resolves and fingerprints the external `.svar2` store
(`_svar2_haps.py:444-445`).

### The probe

Two levels. `lo = region_ptr[r_q]`, `hi = region_ptr[r_q+1]` is a vectorized
gather; the search for `target = si_q * P + ploid` then runs inside that block.

`np.searchsorted` cannot express per-element `lo`/`hi`, so the general path is a
**manually vectorized binary search** — about 12 iterations of

```python
mid = (lo + hi) >> 1
go = cell_id[mid] < target
lo = np.where(go, mid + 1, lo)
hi = np.where(go, mid, hi)
```

No Python loop over queries; the loop is over bit-depth, which is
`log2(N/R) ~ 12.2` at chr22 against `log2(N) ~ 24.1` for a flat global table.

Three properties this buys:

1. **Half the search depth**, on the term measured at 91% of total lookup cost.
2. **A zero-search fast path.** When a region's queried slots cover the whole
   block (the `ds[:, :]` whole-dataset extraction that motivated #357, and
   `sweep_svar2_splice_variants.py:112`), the block is a contiguous slice and
   the lookup degenerates to a scatter with no binary search at all.
3. `region_ptr` costs 1.6 MB genome-wide against a 27 GB table.

**Defined fallback.** If the manual search loses to a flat global table on the
benchmark below, store an int64 global key (`r * S * P + cell_id`) instead of
`cell_id`, use one `np.searchsorted`, and keep `region_ptr` for the contiguous
fast path. That is 32 bytes per entry — still never larger than dense, just not
smaller. The implementer decides on measurement, not on taste.

### Required guards

These are corollaries of "empty cells carry no information": once an empty cell
and a missing cell are indistinguishable, several errors stop being loud.

1. **Bounds check `r_q` and `si_q`.** Today `vk_snp_range[r_q, si_q]` raises
   `IndexError` on an out-of-range index. A sparse miss silently returns
   `(0, 0)` — a reference-only haplotype with no error. `lookup` validates
   against `R` and `S` explicitly.
2. **Cross-check the grid at open.** A wrong `S` makes *every* probe miss and
   yields a silently variant-free dataset. `from_path` asserts
   `len(sample_cols) == S` and `dense_snp_range.shape[0] == R`.
3. **int64 key arithmetic.** `r_q[:, None] * S` keeps int32 under both
   value-based casting and NEP 50, wraps silently, and the later `+ arange(P)`
   promotes the *already-wrapped* value to int64 — so the dtype gives no tell.
   Use `np.multiply(r_q, S, dtype=np.int64)`. The protocol annotates
   `NDArray[np.integer]`, which invites int32; production is currently safe only
   because `np.unravel_index` (`_svar2_haps.py:1494`) returns `intp`.
4. **`N == 0` never memmaps.** `np.memmap` on a 0-byte file raises
   `ValueError: cannot mmap an empty file`. A dataset where no window contains a
   variant is plausible for small fixtures and per-contig shards. The reader
   builds `np.empty((0,), ...)` and `lookup` returns zeros without probing.

### Probe sorting

Sort the probe targets before searching. The win is real — 1.44x at 16,384
probes over an 18M-entry table, 1.96x at 2,048 probes over 2M — but it comes
from numpy's galloping fast path for sorted needles, **not** from page-cache
residency; it does not go away when the table is resident. Crossover is around
500-1,000 probes and the loss below it is under 5 microseconds, so apply it
unconditionally rather than branching on batch size.

Do **not** micro-optimize the payload gather: `cell_vk[pos]` is 0.11 ms of an
18.3 ms lookup (0.6%), and masking it to hits only measured 1.03x. The search is
the cost.

### What the performance claim actually is

Measured, 8192-cell batch (16,384 probes), table resident:

| configuration | sparse lookup | dense alternative |
|---|---|---|
| dense viable (1000x2000x2, 128 MB, N=1.8e4) | 1.9 ms | 0.17 ms, both ~1% of a 171 ms batch |
| AoU chr22 (N=1.8e7), clustered probes | 2.45 ms | a 128 GB file |
| AoU chr22, shuffled probes | 18.4 ms | a 128 GB file |

The honest statement is **not** "sparse is not slower". It is: *there is no
configuration where the dense layout is both viable and meaningfully faster in
absolute terms.* Where dense fits in RAM, sparse costs about 1% of batch wall.
Where sparse costs 18 ms, dense is a 128 GB file that cannot be resident.
`to_dataloader(shuffle=True)` (`_impl.py:1825`) shuffles the flat `(R*S)` index,
so the shuffled row is the training case, not a pathological one; `region_ptr`
and the contiguous fast path both target it.

### Benchmark gate

The fixture dataset is useless as a gate: its dense cache is ~6 KB (N ~ 192,
a key array that lives in L1), where the measured delta is 9.5 microseconds —
0.0095% of a 100 ms batch. Read as a ratio it blocks the change over nothing;
read as wall-clock it can never fire.

The gate is a synthetic harness, no AoU access needed, ~30 s:

- Synthetic table at `(R=3734, S=535662, P=2)`, `N=1.8e7` (about 500 MB).
  Sweep `N` over `1e5, 1e6, 1e7, 1e8` to expose the depth slope.
- Both probe distributions, both reported: **clustered** (few regions x many
  sorted slots, what `ds[:, :]` produces) and **globally shuffled** (what
  `to_dataloader(shuffle=True)` produces). The spread is 7.5x.
- `n_q` over `512, 2048, 8192` at `P=2`.
- Baseline is a **resident dense array at a size that fits** (1000x2000x2,
  128 MB), not the fixture and not a 128 GB array nobody can run.
- **Threshold: sparse lookup <= 2% of batch wall** against the 171 ms
  single-threaded / 88 ms 8-thread spliced batch recorded in
  `_readbound_gather`'s docstring, on the **shuffled** distribution.

A flat 40-byte global table measured 11% / 21% against that threshold — it would
have failed its own gate. The two-level probe exists to pass it; if it does not,
take the fallback and re-measure.

## Write path

`_write_from_svar2` (`_write.py:1140`) currently scatters hap-major chunks into
region-major memmaps. It instead accumulates entries per contig and appends.

Per contig `c` covering region rows `[lo, hi)`:

1. `stream = svar2._find_ranges_chunked(c, starts, ends, samples=sel, max_mem=max_mem)`
   (unchanged).
2. Each chunk covers a contiguous, ascending block of sample slots `[s0, s1)`
   and **all** of the contig's regions, shaped `(ns, P, rc, 2)`
   (genoray `_svar2_batch.py:286-301`). `transpose(2, 0, 1, 3)` gives
   `(rc, ns, P, 2)`. Then:
   ```python
   nonempty = (snp[..., 1] > snp[..., 0]) | (indel[..., 1] > indel[..., 0])
   ri, sj, pj = np.nonzero(nonempty)
   region = lo + ri
   cell = ((s0 + sj) * P + pj).astype(np.int32)
   ```
   with the four `vk` columns fancy-indexed by `(ri, sj, pj)` and `*_len`
   derived as `end - start`.
3. At contig end, sort once by the composite int64 key
   `region * (S * P) + cell` with `kind="stable"` (radix for integers, O(N)),
   then append `cell_id` and `cell_vk`, and extend `region_ptr` by that contig's
   per-region counts (`np.bincount(region - lo, minlength=rc).cumsum()`).

Keys are unique by construction: contigs own disjoint region ranges, chunks own
disjoint slot blocks, and `(ri, sj, pj)` is distinct within a chunk.
`max_end_keys` handling and the returned `chromEnd` extension are unchanged.

**Why a sort and not an interleave.** Each chunk's entries are already key-sorted
(region outer, contiguous slot block, ploid innermost), so the cross-chunk
combine is in principle a pure interleave: per-`(chunk, region)` counts give
exact destination offsets via `cumsum`, with no comparisons. That is the better
algorithm *when the chunk count is small*. It is not safe as the only algorithm:
`per` can be 1 (genoray `_svar2_batch.py:270-283`, and
`tests/dataset/test_write_svar2.py:334+` deliberately drives `max_mem=256` to
force one sample per chunk), which at AoU scale is 535,662 chunks per contig and
a counts matrix of `n_chunks x rc`. A single radix sort is O(N) regardless and
is the one obvious way. The interleave is recorded here as a profile-driven
optimization with that caveat attached.

**No global sort across contigs is needed.** `_prep_bed` calls `sp.bed.sort`
(`_write.py:564`, `:586`), which sorts by a natsorted `chrom` enum then
`chromStart`/`chromEnd`, so each contig's rows are one contiguous block and
`partition_by("chrom", maintain_order=True)` (`_write.py:1215`) yields blocks in
first-appearance order. `_write_from_svar2` has exactly one caller
(`_write.py:342`, always `gvl_bed` straight from `_prep_bed`); the GTF path
funnels through the same `write()`, and concat never calls it.

The load-bearing invariant is narrower than "contigs are natsorted": it is that
`partition_by` blocks partition `[0, R)` into contiguous ranges and that the
running `contig_offset` (`_write.py:1206`, `:1259`) advances in the same
iteration order. Rather than cite a precedent, **assert it**: on each append,
`region[0] > last_region_written` and `keys` strictly increasing, skipped when a
contig produced zero entries. That single assert covers every failure mode,
including a future bed that is not contig-grouped.

### File convention and append

`genotypes/svar2_ranges/` already mixes two conventions: `vk_*_range.npy` and
`dense_*_range.npy` are **raw headerless memmaps** (`np.memmap(..., "w+")` at
`_write.py:1164-1170`, read by `_mm` at `_svar2_haps.py:418-421`), while
`sample_cols.npy` is a real npy (`np.save` / `np.load`).

The new files follow the **raw** convention, shape recorded in the meta. Append
is `with open(p, "ab") as f: arr.tofile(f)`; the reader memmaps with the shape
from `svar2_meta.json`. Real `.npy` would require pre-writing a placeholder
header and seeking back to patch the shape — rejected as fragile. Because the
files are raw, they keep the `.npy` suffix only if the dense ones do; name them
`region_ptr.npy`, `cell_id.npy`, `cell_vk.npy` for consistency with the existing
(equally misnamed) neighbours, and say so in `format.md`.

Two consequences:

- **The meta write moves after the contig loop.** It is currently written before
  it (`_write.py:1199-1211`) because all shapes are known up front; `N` is only
  known at the end. An aborted write then leaves data files with no meta, which
  is acceptable only because `atomic_dir` discards `tmp` — state that explicitly.
- **The per-chunk `flush()` is replaced, not dropped.** `vk_snp.flush()` /
  `vk_indel.flush()` (`_write.py:1245-1249`) exist to bound dirty pages; the
  equivalent for buffered appends is `f.flush()` per contig.

### Memory

The per-contig accumulator peaks at roughly **80 bytes per entry**: 32 B
retained columns, 8 B composite key, 8 B sort permutation, 32 B gathered output.
That is **~1.4 GB at AoU chr22** and **~4.6 GB at chr19** — not the 720 MB the
first draft claimed. Per-chunk transients add three `rc*ns*P`-byte bool arrays
for the mask and 24 B per found entry from `np.nonzero`; compute the mask with
`np.greater(..., out=)` into one preallocated buffer and fill a preallocated
`(k, ...)` record array by column rather than `np.stack`.

**This is not bounded by `max_mem`, and `docs/source/write.md:110` currently
promises that it is.** The spec's position: keep the accumulator (it is bounded
by variants-in-regions, which is the entire point of the change) and **correct
`write.md`** to state that `max_mem` bounds the genoray chunk stream but not the
range-cache accumulator, giving the real formula
`~80 bytes x entries on the largest contig`. Spilling per contig and merging is
the escape hatch if that proves unacceptable; it is not built now.

### Preflight

`_svar2_ranges_cache_bytes` can no longer be computed before the ranges are
known, and a worst-case bound is actively misleading: `28 * R * S * P`
genome-wide is 6.06 TB, within noise of the 6.93 TB dense cache this change
exists to eliminate, against a real answer of 27 GB — an overstatement of 220x
that would fire on essentially every cohort build.

`_svar2_preflight` therefore:

- logs the dense-equivalent figure as **"what the old layout would have cost"**,
  for context, not as a projection;
- after the first contig, logs a **fill-based projection** extrapolated from
  realized fill, and runs the free-space check against *that*;
- keeps warn-do-not-raise.

`_svar2_ranges_cache_bytes` keeps its current dense formula and its current
meaning (the old layout's size) so `test_svar2_ranges_cache_bytes` stays
meaningful; only its call site and log wording change.

## `n_variants` (the #355 remainder)

`_svar2_haps.py:277` allocates `np.zeros((R, S, P), np.int32)` — 50.7 GB at AoU
chr19 — that is only ever read for its shape and for the documented zeros
`Dataset.n_variants()` returns on SVAR2 (#363). Replace with a read-only,
zero-stride view:

```python
self.n_variants = np.broadcast_to(
    np.zeros((), np.int32), (self.n_regions, self.n_samples, self.ploidy)
)
```

Verified at `(11834, 535662, 2)`: `strides == (0, 0, 0)`,
`flags.writeable is False`, `dtype == int32`, tracemalloc peak **0.80 KiB**.

Consumers: `_impl.py:1363` reads `.shape`; `_impl.py:1290` fancy-indexes, which
always copies, so the returned array stays writable and correct. Nothing writes
to it. (The first draft also cited `_haps.py:1212` — that path reads
`self.genotypes.shape[:2]`, which `Svar2Haps` does not have; it is SVAR1-only
and unreachable here. Do not write a test against it.)

Three caveats to record:

1. `.nbytes` still reports 50,712,192,864. Anything sizing a buffer or logging
   from `.nbytes` is off by 50 GB.
2. `np.broadcast_to` **pickles by materializing**, so `to_dataloader(num_workers>0)`
   under spawn (`_torch.py:133-146`) still serializes the full array. Identical
   to today's `np.zeros`, so not a regression — but it undercuts "free" and
   should not be claimed otherwise.
3. `_haps.py:294` annotates `n_variants` as a public-ish `NDArray[np.int32]`.
   Making the SVAR2 one read-only changes that contract asymmetrically (SVAR1
   keeps a writable real array at `_haps.py:577`); `+=` on it now raises.

## Concat

`_concat_svar2_ranges` (`_concat.py:235`) merges the caches with
`gather_fixed(..., record_bytes=16)`, which assumes fixed per-cell records.

### Direction of the index maps

`provenance`'s `order` (`_concat_plan.py:64-68`, built by `_region_order` at
`_concat.py:67-81` and `_sample_order` at `:83-95`) is an
`(n_merged_along_axis, 2)` array of `(dataset_idx, within_dataset_idx)`
**indexed by merged position** — that is merged to source. The remap needs
source to merged, i.e. the scatter-inverse. Spelling it out, because a naive
reading of "derived from `order`" yields `r_map_d = order[:, 1]`, which is the
inverse permutation and produces a **silently region-scrambled dataset**:

```python
# axis="regions": s_map is the IDENTITY (_concat_validate.py:243-249 requires
# identical samples in identical order); r_map is the scatter-inverse.
r_map = [np.empty(R_d, np.int64) for R_d, _ in shapes]
for i, (d, w) in enumerate(order):
    r_map[d][w] = i

# axis="samples": r_map is the IDENTITY (_concat_validate.py:221-226 requires
# inp.bed.equals(ref.bed)); s_map is the scatter-inverse of `order`.
```

Both identities are load-bearing and neither appeared in the first draft.
`S_d` comes from the `shapes` argument `_concat_svar2_ranges` already receives —
not from a meta key that may not exist.

### Merge, not sort

Each input's remapped keys stay **sorted**, which is the precondition that makes
a merge possible:

- `axis="samples"`: `samples` is `sorted(union)` (`_concat.py:365`), each input's
  `meta.samples` is itself sorted, and `_concat_validate.py:227-239` forbids
  overlap — so restricting the sorted union to one input yields that input's own
  order, and `s_map_d` is strictly increasing.
- `axis="regions"`: `_region_order`'s running-count construction gives a
  strictly increasing `r_map_d` under the same consistent-sort assumption its
  docstring states.

So: stream a k-way merge over the per-input `iter_entries()` generators
(`heapq.merge` on block heads, or a smallest-head loop), flushing in
`CONCAT_CHUNK_BYTES` blocks (`_concat_plan.py:21`). Peak is
`n_inputs x block` rather than the `~88 bytes x N` a concatenate-and-sort would
cost (85 GB of peak RSS at the genome projection). The merge shares the append
helper with `_write_from_svar2`.

**Framing correction.** The first draft was reviewed as "turning an O(1)
streaming merge into an in-core sort". That is wrong in both directions:
`gather_fixed` does stream, but `provenance` itself allocates
`np.empty((n_merged * cell, 2), np.int64)` (`_concat_plan.py:98`) — 16 bytes per
`(region, sample, ploid)` slot, i.e. **64 GB at AoU chr22 before any I/O**.
Today's concat is already unbounded; the sparse merge is a large improvement
below ~40% fill. Note also that `concat`'s `max_mem` is **advisory and bounds
nothing today** (`_concat.py:320`, `:341`); actual bounding is
`CONCAT_CHUNK_BYTES` inside `_stream_range`.

**Scope honesty:** this does not make concat scale. Per-sample tracks still call
`provenance(axis, shapes, 1, order)` (`_concat.py:443`) = 16 B x R x S (32 GB at
chr22), and `copy_runs` materializes two more slot-sized int64 arrays
(`_concat_io.py:60-69`). Say so, so nobody reads this as "concat now scales".

### Output meta

`_concat.py:302-309` builds the output meta by **mutating input #0's**. That
breaks both ways under a layout split: if input #0 is sparse the `vk_*` keys do
not exist (`KeyError`); if input #0 is dense and the output is sparse, stale
`vk_*` keys survive and the reader's `layout` dispatch sees a file claiming both
layouts — and (see below) would read `S` from a stale dense shape.

The output meta is **constructed fresh** with an explicit carried-key list:
`layout`, `ploidy`, `n_regions`, `n_samples`, `fill`, `dense_snp_range`,
`dense_indel_range`, `sample_cols`, `region_ptr`, `cell_id`, `cell_vk`.

`gather_fixed` itself stays — it is still used for `dense_*` on the region axis
(`_concat.py:285-292`). Only the `vk_*` call goes away, and the function's
docstring (`_concat.py:245-250`), which documents the vk arrays as fixed 16-byte
records, needs rewriting.

## Meta schema and backward compatibility

`svar2_meta.json` gains `"layout": "sparse"`, `"n_regions"`, `"n_samples"` and
`"fill"`. A file without `"layout"` is the dense layout.

**`n_regions` / `n_samples` are not optional.** `Svar2Haps.from_path:426` today
derives the dataset's sample count as
`S = int(meta["vk_snp_range"]["shape"][1])` — the key this change deletes — and
`n_samples`'s own docstring (`:226-231`) documents that as the source, noting
there is no genotypes array to read it from. `S` feeds
`np.unravel_index(idx, (R_all, S_all))` (`:1494`) and is baked into every probe.
Without an explicit field a sparse dataset cannot compute it; with a wrong one
every probe misses silently. `from_path` reads both from the meta for the sparse
layout, falls back to the `vk_*` shape for the dense layout, and cross-checks
against `len(sample_cols)` and `dense_snp_range.shape[0]`. The docstrings at
`:225-231` change with it.

Readers:

- `_DenseRanges` — the existing fancy-index `lookup`, plus a **new**
  `iter_entries` that block-scans the `(R, S, P, 2)` memmaps emitting non-empty
  cells in key order. This is not "15 lines verbatim": `iter_entries` has no
  existing analogue and is the sole input path for dense-to-sparse concat, where
  it reads the entire dense array once (a real cost at AoU scale).
- `_SparseRanges` — the two-level probe above.

`gvl.write` emits only sparse. There is no migration tool; re-running
`gvl.write` is how an existing dataset is shrunk. Say this in `format.md`'s
upgrade note so users do not reach for `gvl.migrate` (public in `__all__`, but
it only handles the 1.x to 2.0 track AoS-to-SoA migration, `_migrate.py:65`).

**`DATASET_FORMAT_VERSION` stays `2.0.0`.** `_validate._check_format_version`
(`_validate.py:31-45`) requires `fmt.major == DATASET_FORMAT_VERSION.major`
exactly, so bumping to `3.0.0` would make new GVL refuse every existing dense
dataset — the opposite of the goal. The repo has no minor-version discrimination
on open. Consequence, stated rather than discovered: **GVL 0.42.1 opening a
0.43.0 dataset dies at `_svar2_haps.py:426` with a bare
`KeyError: 'vk_snp_range'`** — loud, not silently wrong, but unhelpful.
Optionally add a `layout`-aware guard to `_validate.py` so new readers give a
clean message on a truncated or half-written cache.

Package version: `0.43.0`. `pyproject.toml` has `major_version_zero = true`, so
commitizen bumps MINOR for a `feat:` or a `BREAKING CHANGE:` alike. `Cargo.toml`
(`0.2.1`) is independent and needs no bump — `src/` is untouched.

## Testing

### Tests that must be rewritten

The first draft claimed the existing suite passes unchanged. It does not. All in
`tests/dataset/test_write_svar2.py`:

| test | line | why |
|---|---|---|
| meta key assertion | 100-107 | asserts `set(meta) >= {"vk_snp_range", ...}` |
| the layout oracle | 128-168 | memmaps `vk_*_range.npy` and diffs elementwise against `_find_ranges`, **including every empty cell's true insertion point** — exactly the semantic being discarded. Rewrite to compare widths and non-empty entries; do not delete, it is the only test locking region-major layout. |
| `max_mem` invariance | 382-392 | byte-compares `vk_*_range.npy` between chunked and unchunked writes. **Retarget at `cell_id`/`cell_vk`** — it is the test that would catch a chunk-merge ordering bug. |
| `sample_cols` permutation oracle | 519-531 | memmaps via `meta["vk_snp_range"]["shape"]` |
| `test_svar2_ranges_cache_bytes` | 427-434 | survives unchanged given the preflight decision above |
| `test_svar2_preflight_warns_when_disk_is_short` | 437-457 | asserts preflight's return equals the dense formula; update to the new contract |

`tests/unit/dataset/test_svar2_gather_memo.py` is **unaffected** — it stubs
`_gather_inputs` entirely (`:121`) and never touches the cache. Stated so the
implementer does not touch it.

### The fixture cannot test this

`tests/dataset/test_write_svar2.py:22-33` is 3 variants x 2 samples x ploidy 2
over beds `[0,20)` and `[5,15)`; **all 8 cells are non-empty — 100% fill.** So
the `hit == False` branch never executes under any existing test, and a
sparse/dense divergence on empty cells would not show.

Required fixture work, before anything else: add a region outside `[2,14)` (the
reference is 40 bp, `_REF` at `:25`) and a third all-`0|0` sample to `_VCF`, so
empty rows, empty columns and partial fill all exist.

### Characterization test, before the refactor

`_concat_svar2_ranges` has **zero coverage today** — `tests/dataset/test_concat.py`
covers only the `pgen_vcf` and `svar` backends. So "test the new merge against a
dense-path reference" has no baseline: the reference would be new code written in
the same PR. Land a characterization test of *today's* dense
`_concat_svar2_ranges` first (both axes, interleaved sample sets, reads equal to
a single-shot `gvl.write`). This is the highest-value item in the test plan.

### Tests to add

1. **Lookup parity**, `_SparseRanges` vs `_DenseRanges`, over duplicated,
   out-of-order and corner `(r_q, si_q)`. Best expressed as a **Hypothesis**
   property over `(R, S, P, occupancy mask)` — the repo already runs Hypothesis
   (`tests/parity/strategies.py`, `tests/integration/dataset/test_haps_property.py`)
   and this needs no genoray store, so it also covers `N == 0` and all-hits.
2. **Empty cells** produce `(0, 0)` and the same haplotypes/variants as dense.
3. **Write round-trip** byte-identical against a dense-layout dataset.
4. **Dense dataset still opens** and reads correctly.
5. **Concat** on both axes, sparse x sparse and dense x sparse, against the
   characterization baseline.
6. **`n_variants`**: assert `strides == (0, 0, 0)`, `writeable is False`,
   `dtype == int32` **on a small shape**. Do *not* assert via a cohort-scale
   allocation: it is O(1) while the change holds, but the moment someone reverts
   to `np.zeros` the test attempts 50 GB and the CI runner is killed rather than
   the test failing. (Also: numpy registers buffers under
   `np.lib.tracemalloc_domain`, not the default domain, so a naive
   `get_traced_memory()` assertion is fragile.)
7. **Size**: assert sparse <= dense-equivalent at the *realized* fill. Do not
   assert it on the 100%-fill fixture — at 28 B/entry it holds, but state the
   assertion in terms of fill so it stays meaningful.
8. **Benchmark harness** per the gate above, as a non-collected script
   alongside `tests/benchmarks/`.

Items 3 and 5 need a dense-layout dataset that `gvl.write` can no longer produce.
**Decision: a test-only helper in `tests/` that writes the dense layout
directly** (raw memmaps + dense meta) from a `SparseVar2._find_ranges` call —
not a `layout=` kwarg on the public `write()` (which would make a deprecated
format reachable from the public API) and not a frozen binary fixture (which
would rot silently). Without this the entire backward-compat path ships untested.

## Docs

The first draft offered only a changelog row. `CLAUDE.md`'s docs gate applies
(on-disk format change), and the changelog explicitly does not count. Stale
sites:

- `docs/source/format.md:19, 27-28` — "caches per-`(region, sample, ploidy)`
  range arrays".
- `docs/source/format.md:80-93` — the layout table (`vk_*_range.npy` rows) and
  the `svar2_meta.json` description (must gain `layout`, `n_regions`,
  `n_samples`, `fill`).
- `docs/source/format.md:95-107` — the whole size block: the
  `2 x regions x samples x ploidy x 2 x 8` formula, "grows linearly in **both**",
  "approximately **98 GiB**", and "logs the projected size before allocating".
  All four claims die.
- `docs/source/format.md:109-111` — "slices these memmaps (numpy fancy-indexing;
  no interval search)" becomes a two-level probe.
- `docs/source/format.md:188` — the `0.37.0` row's description; add the `0.43.0`
  row and the "re-run `gvl.write` to shrink" upgrade note.
- `docs/source/write.md:110` — "not small at cohort scale" **and** the `max_mem`
  RAM-bounding promise (see "Memory" above).
- `docs/source/faq.md:99` — "not small at cohort scale, see the size formula".
- `python/genvarloader/_dataset/_write.py:162-166` — **`gvl.write`'s own
  docstring**, which states the cache "is two `(n_regions, n_samples, ploidy, 2)`
  int64 memmaps ... 60.7 GiB for 1,901 regions x 535,662 diploid samples". Every
  clause becomes false. `CLAUDE.md`'s skill rule names this docstring explicitly.
- `skills/genvarloader/SKILL.md:86, 135, 440, 503-508` — the size formula, the
  "scales with `regions x samples x ploidy`" claim, the layout tree, and the
  gotcha bullet.

`docs/source/api.md` and `README.md` are clean — no `__all__` change, and the
README does not mention the cache.

## Process

- **Target `main`, not the `streaming` branch.** `_streaming.py` and `src/stream/`
  do not exist on `main`; the SVAR2 stream backend builds ranges in memory per
  window and never opens `genotypes/svar2_ranges/`. It **mirrors**
  `_gather_inputs` rather than sharing it (`_streaming.py:4326`). Per `CLAUDE.md`'s
  explicit carve-out this is not StreamingDataset-board work: no `streaming:`
  prefix, no board issue.
- **Merge-drift warning.** `origin/streaming` already modifies `_svar2_haps.py`,
  and `_streaming.py` hard-codes line references into files this change edits
  (`_svar2_haps.py:271`, `:702`, and `_write.py:_write_from_svar2` at
  `_streaming.py:4107`, `:4152`). Expect conflict work at the next
  `main -> streaming` merge.
- **No Rust gate.** `src/` is untouched (`grep svar2_ranges src/` is empty; the
  FFI receives the ranges as arrays from Python), so
  `docs/roadmaps/rust-migration.md` does not apply and `maturin develop --release`
  is not needed before pytest. This changes if the implementer moves the probe
  into Rust — see the fallback in "Lookup".

## Out of scope

- The `svar2_ranges="lazy"` / no-cache mode of #357 option 1. A 504 MB cache
  removes its motivation, and it would add a second read path answering the same
  question.
- Moving the probe into Rust. The two-level numpy search is designed to pass the
  gate; if it does not, the flat-table fallback comes first because it does not
  touch `src/`.
- Making concat scale. It remains `R x S`-bound via `provenance` for per-sample
  tracks.
- genoray's O(S^2) `_sample_idxs` name resolution (d-laub/genoray#168), dodged as
  today by passing `samples=None` when the selection matches store order.

## Risks

1. **Concat key remap.** Highest risk: the maps are inverses of what `order`
   provides, a wrong direction silently scrambles regions, and there is no
   existing test. Mitigated by the characterization test landing first, by the
   explicit scatter-inverse code above, and by one merge implementation behind
   `iter_entries`.
2. **The two-level probe may not pass its own gate.** Mitigated by a defined,
   pre-agreed fallback (flat int64 key, 32 B/entry) decided on measurement.
3. **The upstream invariant is unguarded.** "Empty cells carry no information"
   is a contract in a pinned genoray rev that GVL cannot enforce. A genoray bump
   that changed it would surface only as wrong output. The parity tests are the
   only guard; note it at the genoray pin in `Cargo.toml`.
4. **Write accumulator escapes `max_mem`** (~1.4 GB at chr22, ~4.6 GB at chr19).
   Accepted and documented rather than fixed; spill-and-merge is the hatch.

## What review changed

Recorded so the reasoning is not re-litigated:

- **Added `region_ptr`.** The first draft advertised "no per-region offsets
  array" as a simplification. It was the single worst decision in the spec: the
  search term is 91% of lookup cost, and the offsets array halves its depth and
  enables a zero-search path for the whole-dataset extraction that motivated
  #357, for 1.6 MB genome-wide.
- **Narrowed the entry from 40 to 28 bytes.** The first draft's "fill is under
  1% wherever the cache is large" generalized from MANE exons; at
  sequence-model window sizes fill approaches 100%, where a 40-byte entry is 25%
  *larger* than dense. 28 bytes makes sparse strictly smaller at every fill and
  removes the crossover instead of managing it.
- **Corrected the performance claim** from "not slower in practice" to the
  measured, configuration-specific statement, and replaced an unfalsifiable
  fixture benchmark with a real gate the first design would have failed.
- **Added the fail-fast and silent-miss guards**, which are corollaries of the
  core claim that the first draft did not follow through.
- **Found `S` has no source** after the change — flagged independently by all
  four reviewers.
- **Reversed the concat memory framing**: today's `provenance` is already dense
  at 64 GB, so the merge is an improvement, not a regression.
- Corrected: chr21 65 GB (not 61), the `gather.rs:772` citation (not `:145`,
  which GVL cannot reach), `_write.py`'s no-global-sort evidence, the
  `_haps.py:1212` consumer, the `argsort` justification, and the docs audit from
  one changelog row to twelve sites.
