# SVAR2 write path: bound memory, fix `find_ranges` complexity, report progress

Design for [gvl#333](https://github.com/mcvickerlab/GenVarLoader/issues/333).
Cross-cutting: genoray and GenVarLoader each get a PR.

## Problem

`gvl.write(..., variants=SparseVar2(...), max_mem=...)` was SIGKILLed (exit 137)
after hours at 0% progress on an All of Us chr22 build: 414,830 samples, ~3,964
MANE Select CDS regions. `max_mem` never reaches the SVAR2 genotype branch.

Investigation found three independent defects, only one of which the issue names.

### 1. `find_ranges` is `O(regions x total_variants)` (genoray, dominant)

`query::find_ranges` (`src/query/gather.rs`) loops region-outer, hap-inner and
calls `reader.vk_snp_overlap(col, qs, qe)` / `vk_indel_overlap(...)` per
`(region, column)`. Each call rebuilds region-independent state from scratch
(`src/query/reader.rs:244`, `:260`):

```rust
let v_ends: Vec<u32> = positions.iter().map(|&p| p + 1).collect();
let tree = SearchTree::new(positions);
```

`SearchTree::new` is `O(n)` and allocates two `Vec`s sized to the column. At
3,964 regions x 414,830 samples x 2 ploidy x 2 channels that is **6.6 billion
full tree builds**, and the vk store is swept 3,964 times instead of once.

This is why the job sat at 0% for hours. Bounding memory alone would convert an
OOM into a job that never finishes.

### 2. `_svar2_region_max_ends` decodes every sample (GenVarLoader)

`python/genvarloader/_dataset/_write.py:1067` calls
`svar2.decode(contig, all_regions)`, which materializes shape
`(R, S_all, P, None)` — **all 414,830 samples**, ignoring the caller's `samples`
selection, filtering only afterward. That is the full genotype content of chr22
in RAM plus a `R*S_all*P+1` int64 offsets array (~26 GB). An independent, and
arguably larger, OOM than the one reported.

### 3. `find_ranges` triple-copies its payload (genoray)

`bundle_to_dict` (`src/py_query_ranges.rs:78`) flattens `Vec<Range<usize>>` into
a `Vec<i64>` and then `ToPyArray`s it, so each 49 GiB channel exists roughly 3x
at peak (Rust ranges + flattened vec + numpy destination). The binding also
never releases the GIL.

Compounding all three: `max_mem` is parsed in `write()` but the SVAR2 branch is
called without it, and the progress bar advances only once per contig, so a
single-contig BED shows 0% for the entire run.

## Non-goals

- Shrinking the permanent on-disk range cache below 16 bytes/entry. It is
  ~98 GiB for the reported input (`2 channels x R x S x P x 2 x 8`). This design
  preflights and reports it; narrowing the format is a separate issue.
- The SVAR1 `_write_from_svar` branch, which also ignores `max_mem` but uses a
  different mechanism (`_find_starts_ends_with_length(..., out=)`) with no
  transient amplification. Separate follow-up issue.
- Making the write-time range cache optional or lazily computed at read time.

## Scope and sequencing

Two PRs, both targeting `main`. This is the file-backed `gvl.write` path, not
StreamingDataset-board work.

1. **genoray** (`main`, 3.3.0 -> 3.4.0) — loop inversion, chunked private API,
   `max_ends`. Land and release first.
2. **GenVarLoader** (`main`) — plumb `max_mem`, consume the chunked API,
   preflight, delete `_svar2_region_max_ends`, bump the pin to
   `genoray>=3.4,<4`.

## genoray: Rust core

### Hoist the per-column index

`src/query/reader.rs` gains a struct holding the region-independent state:

```rust
pub(crate) struct VkColumnIndex {
    o0: usize,          // absolute base offset of this column
    tree: SearchTree,   // over positions[o0..o1]
    v_ends: Vec<u32>,
    max_del: u32,       // indel channel only; 0 for snp
}

impl VkColumnIndex {
    fn overlap(&self, qs: u32, qe: u32) -> Range<usize>;
}
```

`vk_snp_overlap` / `vk_indel_overlap` are **replaced** by
`vk_snp_index(col)` / `vk_indel_index(sample, p)` returning a `VkColumnIndex`,
plus the cheap `.overlap()`. The old per-call methods are deleted rather than
kept alongside, so there is one way to do this.

### Column-outer core

`src/query/gather.rs` gains:

```rust
pub fn find_ranges_haps(
    reader: &ContigReader,
    regions: &[(u32, u32)],
    sample_cols: &[usize],
    hap_lo: usize, hap_hi: usize,   // half-open, into the selected hap axis
    out_snp: &mut [i64],            // (hap_hi - hap_lo, R, 2), hap-major
    out_indel: &mut [i64],
    // returns Vec<u64>: (R,) partial max of the packed (pos << 21) | ext
    // composite key over this hap slice. 0 = no variant.
)
```

The loop is column-outer / region-inner, so each column's tree is built exactly
once and the store is swept once. Parallelized with
`out_snp.par_chunks_mut(R * 2)` zipped over columns — disjoint mutable slices,
no `unsafe`.

Tree builds drop from `R x H x 2` to `H x 2`: 3,964x fewer for the reported
input, with the store read once instead of `R` times.

### Output order

`find_ranges_haps` produces **hap-major** `(H, R, 2)` — what the column-outer
loop naturally produces and what `par_chunks_mut` can write safely.

The existing `find_ranges` binding keeps its region-major `(R*H, 2)` contract by
transposing the hap-major fill back into its `Vec<Range<usize>>`. That is one
extra copy of the payload, paid only on the un-chunked path: `find_ranges` is the
small-batch read-path entry point, while the population-scale writer uses the
chunked API and never builds a bundle at all. `_gather_ranges` and the read path
(`_svar2_haps.py`) are untouched.

`find_ranges` becomes a thin wrapper over `find_ranges_haps` covering all haps.

## genoray: chunked Python API

New private method on `_BatchQueryMixin` (`python/genoray/_svar2_batch.py`),
alongside the existing `_find_ranges`:

```python
@dataclass(frozen=True)
class RangesChunk:
    sample_start: int                    # into the SELECTED sample axis
    n_samples: int
    vk_snp_range: NDArray[np.int64]      # (n_samples, ploidy, R, 2), hap-major
    vk_indel_range: NDArray[np.int64]
    max_end_keys: NDArray[np.int64]      # (R,), packed key; 0 = no variant


@dataclass(frozen=True)
class RangesStream:
    n_regions: int
    n_samples: int                       # progress denominator
    ploidy: int
    samples_per_chunk: int               # derived; exposed for observability
    region_starts: NDArray[np.int32]     # eager, R-sized
    dense_range: NDArray[np.int32]       # (R, 2)
    dense_snp_range: NDArray[np.int32]   # (R, 2)
    dense_indel_range: NDArray[np.int32] # (R, 2)
    sample_cols: NDArray[np.int64]       # (S,)
    dense_max_end_keys: NDArray[np.int64] # (R,), dense-channel contribution
    chunks: Iterator[RangesChunk]


def _find_ranges_chunked(
    self, contig, starts, ends, samples=None, *, max_mem: int | None = None
) -> RangesStream: ...
```

The R-sized region-level arrays are cheap, so they are computed eagerly and
returned in the header; the `O(R x S x P)` payload arrives in chunks. This is
the "(progress denominator, generator)" contract, typed. `max_mem=None` yields a
single chunk.

Each chunk is one Rust call — `find_ranges_chunk(regions, sample_idxs, hap_lo,
hap_hi)` — with the generator driving the loop in Python. No callbacks cross the
FFI boundary. The call releases the GIL (`py.detach`), which the current
`find_ranges` does not; without it the new rayon parallelism would serialize
against the consumer's progress bar.

### Chunk sizing

The binding allocates each destination numpy array first and fills it in place
(`PyArray2::zeros` -> `&mut [i64]`), eliminating the
`Vec<Range<usize>>` -> `Vec<i64>` -> `ToPyArray` triple. Peak per chunk is then
exactly one copy of the payload:

```
bytes_per_sample  = R * ploidy * 2 endpoints * 8 bytes * 2 channels  # R * P * 32
samples_per_chunk = max(1, max_mem // (2 * bytes_per_sample))        # 2x slop
```

At `R=3,964`, `P=2` that is ~248 KiB/sample; a 2 GiB budget gives ~4,200
samples/chunk, ~99 chunks for 414,830 samples. Chunks are whole samples so both
ploids stay together and the consumer's destination slice is clean.

If `max_mem` cannot fit a single sample, raise with the required minimum,
mirroring the existing gvl error at `_write.py:300`.

### Rejected: `find_ranges_into(memmap_slice)`

Once chunks are bounded, the remaining copy is one chunk-sized memcpy (~0.2 s
per 2 GiB) against seconds of chunk compute, and it would couple genoray to
gvl's `(R, S, P, 2)` destination layout. The in-place numpy fill above already
removes the two copies that mattered.

## genoray: `max_ends`

Computed inside the same column sweep. No second pass, no decode.

**vk channels.** Positions are sorted within a column, so the max-position
variant for a `(region, hap)` pair is the last element of the range
`find_ranges_haps` just computed. Read its `pos`/`key`, apply
`end = pos + 1 - min(ilen, 0)`, reduce per region. Free.

**dense channel.** `DenseView::carried(hap, col)` indexes a memmapped hap-major
bit matrix at `hap * n_dense_variants + col` (`src/query/sidecar.rs:87`), and
there is no per-variant carrier-count sidecar. So "is dense variant `j` carried
by any selected hap" is a strided probe across haps. The algorithm is a backward
walk from `de - 1` over the region's dense range, stopping at the first variant
carried by any selected hap:

- Dense variants are common by construction, so the walk almost always
  terminates on the first variant after a handful of probes.
- When `samples is None` (gvl's `write` default), every dense variant in the
  store has at least one carrier among all samples, so the last dense variant in
  the region is exact with **zero** bitmap access. This is the fast path.
- Worst case — a selected subset carrying none of a region's dense variants —
  degrades to `dense_in_region x H_selected` bit probes for that region. Bounded
  and documented, not guarded against.

The dense contribution is eager (`RangesStream.dense_max_end_keys`); the
consumer reduces `np.maximum` over it and each chunk's `max_end_keys`, then
unpacks once.

**The reduction unit is the packed key, not an unpacked end.** The SVAR1 rule
orders by position first and end second, so reducing unpacked ends across hap
chunks would let a lower-position variant with a longer deletion win. Packing
`ext` (bounded) rather than the absolute end is what makes an integer `max` over
the key reproduce that ordering.

This replaces gvl's `_svar2_region_max_ends`.

### Parity risk

Two details must be confirmed against the current gvl implementation before this
is considered equivalent:

1. Whether `SparseVar2.decode` (and therefore today's `max_ends`) includes
   dense-channel variants.
2. Exact reproduction of the `(pos << 21) | ext` composite-key tie-break and the
   0-based-to-1-based `pos` conversion at `_write.py:1100-1120`.

If genoray's implementation diverges from gvl's, determine which is correct
rather than treating gvl as the oracle. If today's gvl behavior is the buggy
one, that becomes its own issue and PR, and the divergent case is excluded from
the parity test.

## GenVarLoader: writer

`_write_from_svar2` gains a `max_mem: int` parameter, plumbed from `write()`'s
`effective_max_mem` — the same value the VCF/PGEN branch already receives.

### Preflight

Before creating any memmap:

```python
cache_bytes = 2 * R * S * P * 2 * 8   # both vk channels
```

Log it through the writer's logger and compare against
`shutil.disk_usage(path).free`; warn when it exceeds free space. Warn, do not
hard-error: free-space reporting is unreliable on some filesystems and a false
refusal would block valid large builds.

### Per-contig loop

```python
stream = svar2._find_ranges_chunked(c, starts, ends, samples=samples, max_mem=max_mem)
dense_snp[lo:hi] = stream.dense_snp_range
dense_indel[lo:hi] = stream.dense_indel_range
keys = stream.dense_max_end_keys.copy()
for ch in stream.chunks:
    s0, s1 = ch.sample_start, ch.sample_start + ch.n_samples
    vk_snp[lo:hi, s0:s1] = ch.vk_snp_range.transpose(2, 0, 1, 3)
    vk_indel[lo:hi, s0:s1] = ch.vk_indel_range.transpose(2, 0, 1, 3)
    np.maximum(keys, ch.max_end_keys, out=keys)
    vk_snp.flush()
    vk_indel.flush()
    pbar.update(rc * ch.n_samples / S)
mask = (1 << 21) - 1
region_ends = np.asarray(ends, np.int64).copy()
has = keys > 0                       # 0 = no variant; keep the original chromEnd
region_ends[has] = (keys[has] >> 21) + (keys[has] & mask)
max_ends[lo:hi] = region_ends.astype(np.int32)
```

`transpose(2, 0, 1, 3)` turns `(n_samples, ploidy, R, 2)` into
`(R, n_samples, ploidy, 2)` as a view; numpy performs the strided copy directly
into the memmap slice, with no intermediate array.

The per-chunk `flush()` matters at this scale. Without it ~98 GiB of dirty pages
accumulate and the kernel reclaims them at unpredictable times.

### Progress

The bar stays region-denominated (`total=R, unit=" region"`) but takes
fractional updates, so a single-contig BED advances smoothly instead of sitting
at 0%. This is the issue's second reported symptom.

### Deletions

`_svar2_region_max_ends` is removed.

## Testing

### genoray

- **Loop-inversion guard.** `find_ranges` returns a byte-identical bundle before
  and after the refactor, on existing fixtures. This covers the whole
  `VkColumnIndex` extraction.
- **Complexity regression guard.** A Rust unit test asserting the existing
  `TREE_BUILDS` counter (`src/search.rs:48`) scales as `O(H)`, not `O(R x H)`,
  across two region counts. A wall-clock test would be too noisy on a shared
  node; the counter is deterministic.
- **Chunk equivalence.** Property test over `max_mem` values (including one
  sample per chunk) asserting the reassembled chunked result equals the
  unchunked bundle, so chunk boundaries land in different places.
- **`max_ends` parity.** Against a Python oracle reproducing the current gvl
  semantics, including the tie-break packing.
- **`samples` subset.** `_find_ranges_chunked(samples=subset)` agrees with
  `_find_ranges(samples=subset)`, exercising the non-fast-path dense walk.

### GenVarLoader

Covering issue #333 section 4:

- Monkeypatch `_find_ranges_chunked` to record chunk sizes: assert more than one
  chunk under a small `max_mem`, and that no chunk exceeds the budget.
- Written cache byte-identical to a dataset produced by the current code path,
  on an existing fixture (`phased_svar_gvl` / `build_case` session fixtures).
- Progress advances per chunk, and each chunk is released before the next.
- Preflight logs the expected byte count.

Run `pixi run -e dev pytest tests -q` (full tree) before pushing: this touches
shared write-path code and renames a private symbol.

## Documentation

- `docs/source/format.md` — the `genotypes/svar2_ranges` section calls these
  arrays "small". Replace with the `R x S x P` scaling formula and a worked
  population-scale example.
- `docs/source/write.md` and the `gvl.write` docstring — `max_mem` now governs
  the SVAR2 branch.
- `skills/genvarloader/SKILL.md` — `max_mem` behavior note under `gvl.write`;
  add to "Common gotchas" that SVAR2 range caches scale with
  `regions x samples x ploidy`.
- genoray `CHANGELOG.md`. The `genoray-api` skill documents public surface only;
  `_find_ranges_chunked` is private, so no change expected there.

## Follow-up issues to file

1. genoray/gvl: shrink the on-disk SVAR2 range cache below 16 bytes/entry.
2. gvl: `_write_from_svar` (SVAR1) also ignores `max_mem`.
