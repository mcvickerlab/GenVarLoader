# Dataset concatenation (`gvl.concat`)

Design for [#334](https://github.com/mcvickerlab/GenVarLoader/issues/334).

## 1. Scope

Issue #334 asks two things. They get different answers.

**Q1 — "can we reuse variant indexing across cohorts with an identical variant
space?" — needs no new feature.** It is already supported:

```python
gvl.write(path, bed, "parent.pgen", samples=cohort_A_samples)
```

`write()` calls `variants.set_samples(samples)` (`_write.py:307`). genoray builds the
PGEN index once from `parent.pvar`, caches it, and gvl hardlinks it into each dataset as
`genotypes/variants.arrow` (`_link_or_copy`, `_write.py:608`). Pre-splitting the PGEN
with `plink2 --keep` is precisely what forces the per-cohort index rebuild the issue
reports. This is a documentation fix, not a code change. Building a separate "shared
variant index" mechanism would duplicate what `write()` already does.

The per-cohort genotype extraction cost is irreducible — it is proportional to
samples x regions and is genuinely per-cohort work.

**Q2 — merging datasets — is the real feature.** It serves the issue's other two stated
use cases: splitting genomes by BED region for parallel preprocessing, and incrementally
adding samples.

Not in scope: merging datasets built from *different* variant sources (e.g. per-chromosome
PGENs). All inputs must share one variant source. See §5.

## 2. Public API

```python
gvl.concat(
    path: str | Path,
    datasets: Sequence[str | Path | Dataset],
    axis: Literal["regions", "samples"],
    *,
    overwrite: bool = False,
    max_mem: int | str = "4g",
) -> None
```

Naming and signature follow `np.concatenate` / `pl.concat`; the return-`None`,
`overwrite`, `max_mem` and `atomic_dir` conventions follow `gvl.write` / `gvl.update`.

One function rather than two: `axis` selects only a provenance map (§3), and both axes
run the identical gather.

## 3. Merge core

A gvl dataset is a set of parallel ragged stores over an `(R, S[, P])` grid in C order —
the flat slot is `((r * S) + s) * P + p` (`_haps.py:765`).

Merging builds a **provenance map**: merged flat slot `j` -> `(dataset d, source flat slot i)`.

- `axis="regions"`: `S' = S`, `R' = sum(R_d)`; cell `(r', s') -> (d(r'), r_d(r'), s')`
- `axis="samples"`: `R' = R`, `S' = sum(S_d)`; cell `(r', s') -> (d(s'), r, s_d(s'))`

Every store is then one streaming gather. Stores differ only in what happens to their
offsets:

**(a) Cumulative offsets into an owned payload** — PGEN/VCF `genotypes/`, all
`intervals/<t>/`, all `annot_intervals/<n>/`. Offsets are re-cumsum'd and the payload
moves with them.

**(b) Absolute ranges into an external store** — `.svar` `offsets.npy` `(2,R,S,P)`,
`.svar2` `svar2_ranges/*`. Values copy verbatim as a fixed-stride row gather. No payload
moves; it stays in the `.svar`/`.svar2`.

Both are bulk streaming moves. Kind (b) is cheaper only because there is no second array
to move — **it is not small**. `.svar` offsets are `16 * R * S * P` bytes; at R=1000,
S=500k, P=2 that is 16 GB, and `.svar2`'s `vk_snp_range` + `vk_indel_range` are 32 GB
combined. Nothing here fits in memory at biobank scale.

### 3.1 Execution

Single-threaded. No rayon, no native-extension threading. The work is memory- and
disk-bandwidth-bound, not compute-bound, so parallelism buys nothing and costs NFS
process-hygiene risk.

**Read sources via `np.memmap(mode="r")`; write destinations via buffered `file.write()`
in 16 MiB chunks.** This is measured, not assumed — see §3.2. Writing into a
`np.memmap(mode="w+")` destination is ~21x slower on NFSv3.

**Iterate in destination order.** A buffered write stream must be filled sequentially, so
destination-ordered iteration is required, not merely preferred. It is also
bandwidth-optimal: each source's runs appear in the destination in monotonically
increasing source order, so both sides stay sequential.

**Coalesce runs.** A run is a maximal span of merged slots where `d` is constant and `i`
increments by 1; each run maps to one contiguous source byte range. `axis="regions"`
degenerates to a single run per dataset per store — the "just increment the offsets" fast
path. `axis="samples"` yields `S_d`-sized runs per region.

**Two streaming passes, neither materializing a full array:**

1. Offsets -> per-block lengths -> merged offsets + running total, chunked, written
   incrementally. This reuses the shape `_write_phased_variants_chunk` already uses in
   `write()`. At 16 GB of offsets, `np.diff` over the whole array is not an option.
2. Payload move, using the run list and total size from pass 1.

Flush periodically; `/carter`-style NFS mounts are `hard` with no `intr`.

### 3.2 Measured basis

512 MiB, each source read exactly once so all reads are cold. Mount:
`carter-storage:/carter` -> `nfs vers=3,rsize=1048576,wsize=1048576,hard,proto=tcp`.

| pattern | throughput |
|---|---|
| memmap read -> buffered write | **316 MB/s** |
| buffered read -> buffered write | 142 MB/s |
| buffered read -> memmap write | 15.2 MB/s |
| memmap read -> memmap write | 13.6 MB/s |

Chunk size on the buffered path: 1 MiB = 178 MB/s, 16 MiB = 241 MB/s, 64 MiB = 240 MB/s.
16 MiB is the knee.

Caveats: single runs on a shared node, so the exact figures are soft; the 21x gap is far
outside that noise. Only NFSv3 was measured. `os.copy_file_range` is unavailable in this
environment (Python 3.10 conda build) and NFSv3 has no server-side copy, so reflink /
server-side copy is off the table regardless.

The same finding applies to `gvl.write`'s existing bulk output and is filed separately as
[#338](https://github.com/mcvickerlab/GenVarLoader/issues/338).

### 3.3 Cost model

Bytes moved is approximately the size of the merged dataset, at sequential-IO speed.
This is not free — a 1 TB merge is roughly an hour at the measured 316 MB/s. It pays off
only against re-extracting genotypes, which is substantially more expensive. The design
should not pretend otherwise, and the docs should state it.

## 4. Per-store rules

| Store | `axis="regions"` | `axis="samples"` |
|---|---|---|
| `input_regions.arrow` | row-concat inputs, re-run `_prep_bed` for sorted order + fresh `r_idx_map` | must be identical across inputs; copy input[0] |
| `regions.npy` | gather rows | elementwise `max` of the end column |
| `genotypes/` (PGEN/VCF) | kind (a) | kind (a) |
| `genotypes/variants.arrow` | hardlink from input[0] + fingerprint | same |
| `genotypes/` (`.svar`/`.svar2`) | kind (b) | kind (b), plus concat/reorder `sample_cols` |
| `intervals/<t>/` | kind (a) | kind (a) |
| `annot_intervals/<n>/` | kind (a) over R | sample-independent; fingerprint-compare across inputs, then copy input[0] |
| `metadata.json` | `n_regions` summed | `samples` = sorted union |

**Region order.** The merged on-disk order must be `sp.bed.sort` of the concatenated
input beds, because `_build_indexer` (`_open.py:127`) re-sorts and assumes the on-disk
grid matches. Since each input is already in its own sorted order, the merged order is a
mergesort-style interleave. The run coalescer handles this with no special case. Feeding
the concatenated input beds back through `_prep_bed` reproduces both the sorted order and
the `r_idx_map`, reusing existing code rather than recomputing the permutation by hand.

**`regions.npy` on the sample axis takes the elementwise max.** It is read only by the
track-truncation warning (`_warn_truncated_tracks`, `_open.py`) and by `gvl.update`. Each input already clears
the `chromEnd` floor, so max preserves the warning's correctness, and it gives `update` a
superset window when writing new tracks. Min would also clear the floor but would starve
`update`.

**`variants.arrow` stays a hardlink**, with an added fingerprint. Rationale in §5.

## 5. Validation

All checks run before any bytes move.

Must match across inputs: `format_version`, `ploidy`, `max_jitter`, `contigs`, track name
sets, dosages presence, and variant-source identity. These raise:

- mixed variant backends (e.g. one PGEN-backed and one `.svar2`-backed input)
- overlapping sample sets, on `axis="samples"`
- duplicate region names on `axis="regions"` — specifically, colliding keys in the merged
  `r2i_map`, since `DatasetIndexer` resolves string region lookups through it and a
  duplicate would make `subset_to(regions=...)` ambiguous. Identical *coordinates* across
  shards are fine; only name collisions raise.

**Variant-source identity** uses the codebase's existing bounded-fingerprint idiom rather
than hashing a multi-GB Arrow file: `_fasta_cache.Fingerprint` — blake2b over a bounded
`FINGERPRINT_WINDOW` (1 MiB) plus `size_bytes`. For `.svar`/`.svar2`-backed inputs the
existing link fingerprint is compared directly.

### 5.1 Why `variants.arrow` is hardlinked, not copied

The concern that a hardlink risks SIGBUS does not apply here:

- **gvl never mmaps `variants.arrow`.** `_haps.py:121` and `_haps.py:185` both pass
  `memory_map=False`. With no mapping there is no SIGBUS exposure; the realistic aliasing
  failure is a torn read, which is detectable.
- **Every writer in our stack rewrites atomically.** genoray's PGEN index
  (`_pgen.py:1125`) and VCF index (`_vcf.py:1095`) both go through `atomic_write_path` —
  temp file + `os.replace`. A rename does not touch the old inode, so a hardlinked
  `variants.arrow` keeps pointing at the intact old index even when genoray regenerates it.

Three rejected alternatives:

- **Fallible hardlink.** Buys nothing: on NFS `link()` *succeeds*. It fails only on
  `EXDEV` or `EPERM`/`EOPNOTSUPP`, which `_link_or_copy` already handles. The hazard is
  what happens to the inode later, so there is no link-time error to route off.
- **Catching SIGBUS.** Not practical from Python. Handlers run at bytecode boundaries,
  while a page-fault SIGBUS is delivered synchronously to the faulting thread inside a
  numpy/Arrow C loop; returning from the handler re-executes the faulting instruction,
  giving an infinite fault loop or death. Recovery needs `sigsetjmp`/`siglongjmp` from C,
  and even then numpy/Arrow internals are undefined.
- **Detecting NFS and routing off it.** NFS-ness is the wrong predicate and the check
  would be actively harmful: NFS supports hardlinks fine, and clusters where this matters
  are NFS end to end, so the check would disable zero-copy exactly where it is most
  valuable.

The residual risk — a third-party tool that truncate-rewrites the index in place — is real
but narrow, and is precisely what a fingerprint catches at open. This mirrors
`svar_link`/`svar2_link`, which already point at a large external artifact gvl does not
own and guard it with a fingerprint. Copying would be the inconsistent choice.

`concat` records the fingerprint from the outset. Adding the same guard to `write()` is
filed as [#337](https://github.com/mcvickerlab/GenVarLoader/issues/337).

## 6. Testing

Oracle: shard, concat, and compare against a single-shot `gvl.write` of the whole thing.

**`axis="regions"` asserts byte-identical payloads.** Regions are independent and the
sample set is unchanged, so exact identity should hold.

**`axis="samples"` asserts per-cell read equality across output modes, not byte-identity.**
`extend_to_length=True` sizes each region's read window to the max over the cohort present
at write time (`_region_end`, `_write.py:674`), so a shard's stored variant set can
legitimately differ from a full write's. Byte-identity is not expected and will not be
claimed. Any *read-level* divergence is a real finding and should be reported as such
rather than absorbed into the expected values.

Coverage: PGEN, VCF, `.svar`, `.svar2`, tracks-only, annot-tracks, and an indel-heavy
fixture. Plus validation-failure tests for each rejected precondition in §5.

## 7. Documentation

Required by the repo's docs gates:

- `concat` in `__all__` -> autodoc entry in `docs/source/api.md`
- `skills/genvarloader/SKILL.md` — new public symbol
- `docs/source/write.md` / `faq.md` — the concat workflow, its cost model (§3.3), **and**
  the `parent.pgen` + `samples=` guidance that answers #334's Q1
- `docs/source/format.md` — unchanged; no on-disk format change beyond the
  `variants.arrow` fingerprint field, which lands with #337

Also worth a docs pointer: genoray already ships `SparseVar2.concat(out, sources, mode=...)`
(`_svar2.py:289`, plus a `genoray concat` CLI) for merging `.svar2` stores at contig
granularity. For contig-sharded `.svar2` workflows the cheaper path is to concat the
*stores* and then `gvl.write` once, since `.svar2` writes are only range caching.
`gvl.concat` remains necessary for within-contig region shards, for tracks, and for
PGEN/VCF entirely.

## 8. Related issues

- [#334](https://github.com/mcvickerlab/GenVarLoader/issues/334) — this work
- [#337](https://github.com/mcvickerlab/GenVarLoader/issues/337) — fingerprint guard for `write()`'s hardlink
- [#338](https://github.com/mcvickerlab/GenVarLoader/issues/338) — buffered writes for bulk output on NFS
