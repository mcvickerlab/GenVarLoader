# Dataset format

A GVL dataset is a directory written by [`gvl.write`](api.md#genvarloader.write) and read
by [`gvl.Dataset.open`](api.md#genvarloader.Dataset.open). This page is the authoritative
description of its on-disk layout.

## Directory layout

```
dataset_dir/
├── metadata.json          # the Metadata schema (below)
├── input_regions.arrow    # original BED regions + region-index map
├── genotypes/             # present iff variants were provided to gvl.write
│   ├── offsets.npy        # per (region, sample, ploidy) offsets into variant_idxs.npy; absent when sourced from .svar2
│   ├── svar_meta.json     # shape + dtype of offsets.npy — present iff source was .svar
│   ├── variant_idxs.npy   # variant indices; absent when sourced from .svar or .svar2
│   ├── dosages.npy        # optional, absent when sourced from .svar or .svar2
│   ├── variants.arrow     # variant table; absent when sourced from .svar or .svar2
│   └── svar2_ranges/      # present iff source was .svar2 — see "svar2_ranges layout" below
└── intervals/             # or annot_intervals/ when annotated; present iff tracks given
```

When the dataset was built from an `.svar`, the heavy per-variant arrays (`variant_idxs.npy`,
`dosages.npy`, `index.arrow`) are **not duplicated** into the dataset. Instead the dataset
records a back-reference to the source `.svar` in `metadata.json` (see `svar_link` below).
Likewise, a dataset built from an `.svar2` records a back-reference (`svar2_link`, below)
and caches the var-key window for each `(region, sample, ploidy)` that holds a variant,
under `genotypes/svar2_ranges/` — the bulk variant data stays in the `.svar2` store. See
"`genotypes/svar2_ranges/` layout" below for the on-disk size of this cache; it scales with
variants observed inside regions rather than with `regions x samples`, so it is small at
cohort scale.

## `metadata.json` schema

`metadata.json` is the serialization of `genvarloader._dataset._write.Metadata`:

| Field | Type | Notes |
|-------|------|-------|
| `samples` | `list[str]` | Sample identifiers, sorted. |
| `contigs` | `list[str]` | Contig names used to interpret BED coords. |
| `n_regions` | `int` | Number of regions (after jitter padding). |
| `ploidy` | `int \| None` | Ploidy when the dataset has genotypes. |
| `max_jitter` | `int` | Maximum coordinate jitter (defaults to 0). |
| `version` | `SemanticVersion \| None` | Package version that wrote this dataset. Drives format dispatch. |
| `svar_link` | `SvarLink \| None` | Back-reference to a source `.svar`, when present. |
| `svar2_link` | `Svar2Link \| None` | Back-reference to a source `.svar2`, when present. |
| `variants_fingerprint` | `Fingerprint \| None` | Bounded content fingerprint of `genotypes/variants.arrow` (blake2b over the first 1 MiB + total size). Set only when [`gvl.concat`](api.md#genvarloader.concat) hardlinks `variants.arrow` from a PGEN/VCF-backed input; `None` for datasets written directly by `gvl.write`. `Dataset.open` verifies it against the on-disk file and raises `ValueError` on mismatch (e.g. the source variant index was rewritten out-of-band); absent (`None`) skips the check. |

`SvarLink`:

| Field | Type | Notes |
|-------|------|-------|
| `relative_path` | `str` | POSIX path from `dataset_dir` to the `.svar`. |
| `absolute_path` | `str` | Original absolute path; used as a fallback. |
| `fingerprint` | `SvarFingerprint` | Integrity check (see below). |

`SvarFingerprint`:

| Field | Type | Notes |
|-------|------|-------|
| `n_variants` | `int` | Row count of the svar's `index.arrow`. |
| `variant_idxs_bytes` | `int` | Byte size of the svar's `variant_idxs.npy`. |

`Svar2Link` (mirrors `SvarLink` for a `.svar2` source):

| Field | Type | Notes |
|-------|------|-------|
| `relative_path` | `str` | POSIX path from `dataset_dir` to the `.svar2`. |
| `absolute_path` | `str` | Original absolute path; used as a fallback. |
| `fingerprint` | `Svar2Fingerprint` | Integrity check (see below). |

`Svar2Fingerprint`:

| Field | Type | Notes |
|-------|------|-------|
| `n_files` | `int` | Count of the `.svar2` store's `.bin`/`.npy` data files. |
| `store_bytes` | `int` | Summed byte size of those data files. |

`.svar2` has no `variant_idxs.npy`/`index.arrow` analogue exposed cheaply, so its fingerprint
keys on file count + total byte size of the store's data files rather than a variant count.

## `genotypes/svar2_ranges/` layout

Written only when the dataset's variant source is a `.svar2` store. `R` = number of regions,
`S` = number of the dataset's **selected** samples (not necessarily the full `.svar2` cohort),
`P` = ploidy. A region-CSR (compressed sparse row) table stores only the
`(region, sample, ploid)` windows that hold a variant:

| File | Shape | Notes |
|------|-------|-------|
| `region_ptr.npy` | `(R + 1,)` int64 | CSR row pointer: region `r`'s non-empty cells are `cell_id[region_ptr[r]:region_ptr[r + 1]]` (and the parallel slice of `cell_vk`). |
| `cell_id.npy` | `(N,)` int32 | `slot * ploidy + ploid` for each non-empty cell, ascending within a region's block. `N` is the total non-empty cell count (`svar2_meta.json`'s `n_entries`). |
| `cell_vk.npy` | `(N,)` | A 24-byte record per non-empty cell: `snp_start` int64, `indel_start` int64, `snp_len` int32, `indel_len` int32 — the half-open SNP/indel variant-key ranges into the `.svar2` store, as a start + length rather than start/end. |
| `dense_snp_range.npy` | `(R, 2)` int64 | Per-region (sample-independent) range into the dense SNP store. |
| `dense_indel_range.npy` | `(R, 2)` int64 | Per-region (sample-independent) range into the dense indel store. |
| `sample_cols.npy` | `(S,)` int64 | Maps the dataset's selected-sample slot to the `.svar2` store's original sample index. |
| `svar2_meta.json` | — | Records `layout` (`"sparse"`), `n_regions`, `n_samples`, `n_entries`, `fill` (`n_entries / (n_regions * n_samples * ploidy)`), each array's `shape`/`dtype`, and `ploidy`. |

`region_ptr.npy`, `cell_id.npy`, and `cell_vk.npy` are raw, headerless `tofile` dumps despite
the `.npy` extension, like `dense_snp_range.npy` and `dense_indel_range.npy`. Only
`sample_cols.npy` is a real `.npy` file (written with `np.save`).

Each non-empty cell costs 28 bytes (24 for `cell_vk` + 4 for `cell_id`), so size scales with
the number of variants observed inside regions rather than with `regions x samples`: a dense
`(R, S, P, 2)` int64 cache — 32 bytes for every cell, including empty ones — would be 128 GB
for the All of Us chr22 grid (`R` = 3,734, `S` = 535,662, `P` = 2); the sparse layout is 504 MB
there, at a realized fill of 0.45%. Genome-wide, the same comparison is 6.93 TB dense against
27.3 GB sparse. `gvl.write` logs realized fill after the first contig and projects the final
on-disk size from it, warning when the filesystem reports too little free space.

At read time, `Dataset.__getitem__` looks up each queried `(region, sample, ploid)` with a
bounded binary search within that region's `region_ptr` block (no search at all when the block
is fully occupied, i.e. every sample/ploid combination in the region holds a variant) to build
the flat per-query inputs for the read-bound Rust kernels — no interval-search tree and no
dense-union rebuild happen per read, unlike the `.svar` path.

A producer must emit `cell_id` in strictly ascending order within each region's block, since the
binary search assumes it. This is not verified on read — doing so would be an O(N) scan over the
whole table, 27 GB genome-wide — so a producer that violates it degrades to silent misses (the
search lands on a wrong-but-plausible position, and the confirmation step then reports "not
found") rather than an error.

## SVAR resolution at open time

When opening a dataset whose `metadata.svar_link` is non-null,
[`Dataset.open`](api.md#genvarloader.Dataset.open) resolves the svar in this order:

1. Caller-provided `svar=...` argument.
2. `svar_link.relative_path` resolved against the dataset directory.
3. `svar_link.absolute_path`.
4. A unique `*.svar` directory next to the dataset.

If none match, a `FileNotFoundError` is raised naming the expected `.svar` basename. After
resolution, the fingerprint is verified; a mismatch raises `ValueError` and lists both
expected and observed values.

## `.svar2` resolution at open time

When opening a dataset whose `metadata.svar2_link` is non-null,
[`Dataset.open`](api.md#genvarloader.Dataset.open) resolves the `.svar2` store in the same order
as `.svar`:

1. Caller-provided `svar2=...` argument.
2. `svar2_link.relative_path` resolved against the dataset directory.
3. `svar2_link.absolute_path`.
4. A unique `*.svar2` directory next to the dataset.

If none match, a `FileNotFoundError` is raised naming the expected `.svar2` basename and
suggesting `svar2=`. After resolution, the fingerprint (`Svar2Fingerprint`, above) is verified;
a mismatch raises `ValueError` and lists both expected and observed values.

## `.svar2` pure-deletion ALT bytes

genoray's `.svar2` stores an atomized pure deletion (e.g. VCF `GTA>G`) at the VCF anchor
position with an empty ALT (`b""`); `.svar` keeps the VCF anchor base (`b"G"`). The variant's
position, `ilen` and reference span (anchor plus deleted bases) are the same in both stores.

- Bare ALT bytes differ: `RaggedVariants.alt` (and the flat `FlatVariants.fields["alt"]`) from
  `with_seqs("variants")`, and the `alt` field of `with_seqs("variant-windows")` with
  `VarWindowOpt(alt="allele")`, are `b""` on `.svar2` and `b"G"` on `.svar`, for pure-deletion
  records only. On a negative-strand region the `.svar` anchor is reverse-complemented
  (`b"C"`); the `.svar2` empty ALT stays `b""`.
- `start` and `ilen` are the same, so the ride-along `FlatVariants.flank_tokens`
  (`ref[start-L:start] . ref[end:end+L]`, `end = start - min(ilen, 0) + 1`) are byte-identical:
  they read only the reference around that span and never look at the ALT bytes.
- Flanked windows do not differ. `ref_window` reads `[start-L, end+L)` from the reference, and
  `alt_window` is `flank5 . ALT . flank3` with the anchor base restored for an empty-ALT
  deletion, so both are byte-identical between the backends and `alt_window` equals the
  reference with that one variant applied.
- Haplotypes (`with_seqs("haplotypes")` / `"annotated"`) are byte-identical: reconstruction
  also keeps the anchor (`ref[pos]`) for an empty-ALT deletion.

## `.svar2` Phase-1 unsupported combinations

A `.svar2`-backed dataset supports all four output modes (`haplotypes`, `variants`,
`variant-windows`, and haplotype-realigned `tracks`), `unphased_union`, and
`var_fields`-selected store INFO/FORMAT fields (on both `"variants"` and `"variant-windows"`).
`"variants"` output supports both output formats: ragged `RaggedVariants` and flat
`FlatVariants` (`with_output_format("flat")`) with the ride-along `flank_tokens`, and both honor
`dummy_variant`. Flat `FlatVariants` match `.svar` field for field (`alt`, `start`, `ilen` when
requested, store fields, `flank_tokens`) except the pure-deletion ALT bytes below.
Haplotype and `variants` output also support splicing, `var_filter="exonic"`, and
automatic reverse-complementation for negative-strand regions. Spliced
`RaggedVariants` are returned as complete `(transcript, sample, phase, ~variants)`
cells.
The following combinations are Phase-1 scope and raise `NotImplementedError` (or, for
`extend_to_length`, at write time) instead of silently mis-computing:

- Splicing with `variant-windows` or haplotype-realigned tracks.
- `var_filter="exonic"` with `variant-windows` or haplotype-realigned tracks.
- `min_af` / `max_af` filtering (`.svar` only; see "Should I use `.svar` or `.svar2`" in the FAQ).
- `annotated` haplotypes (`with_seqs("annotated")`).
- `VarWindowOpt(ref="allele")` (bare-allele REF mode; REF alleles aren't stored in `.svar2`).
- Reverse-complement with haplotype-realigned tracks.
- Fixed-length (integer `output_length`) haplotype-realigned **track** output.
- `variants` / `variant-windows` output on a dataset written with `max_jitter>0` or read with
  `jitter>0` (the read-bound decode does not right-clip to the post-jitter window).
- `gvl.write(..., extend_to_length=False)` for a `.svar2` variant source.

(Multi-contig `FlankSample` track fills are now supported and byte-identical to the `.svar`
backend — issue #267.)

See the `genvarloader` skill's `.svar2` section for the full narrative and `var_fields` semantics.

## Format changelog

| Version | Change |
|---------|--------|
| `< 0.18.0` | Variant coordinates stored 0-based. |
| `0.18.0` | Variant coordinates switched to 1-based. |
| `0.25.0` | `metadata.json` gains `svar_link`; old `genotypes/link.svar` symlink layout deprecated. `Metadata.version` typed as `SemanticVersion` (on-disk JSON unchanged). |
| `0.37.0` | `metadata.json` gains `svar2_link`; `.svar2` accepted as a `gvl.write` variant source, cached under `genotypes/svar2_ranges/` and read via a read-bound, all-Rust path. |
| `0.42.0` | `metadata.json` gains `variants_fingerprint`, set when [`gvl.concat`](api.md#genvarloader.concat) hardlinks `variants.arrow` from an input dataset; `Dataset.open` verifies it and raises on mismatch. |
| `0.43.0` | `genotypes/svar2_ranges/` switches from the dense `(R, S, P, 2)` `vk_snp_range.npy`/`vk_indel_range.npy` layout to the sparse region-CSR `region_ptr.npy`/`cell_id.npy`/`cell_vk.npy` layout described above. |

> **Upgrading legacy datasets.** A dataset written before `0.25.0` that was built from an
> `.svar` will still open (with a `DeprecationWarning`). Run
> `genvarloader.migrate_svar_link(path)` to convert the symlink layout to the new metadata
> layout in place.
>
> **Shrinking a `.svar2` dataset's range cache.** `gvl.migrate` does **not** convert the
> `0.42.x` dense range-cache layout to `0.43.0`'s sparse one — it only handles the 1.x -> 2.0
> track array-of-structs to struct-of-arrays migration. To shrink an existing `.svar2`-backed
> dataset, re-run `gvl.write` against the same `bed`/`variants`/`samples`. GVL `<= 0.42.1`
> cannot open a `0.43.0`-or-later `.svar2` dataset: it fails with a bare `KeyError:
> 'vk_snp_range'` — loud, not silently wrong, but unhelpful; upgrade GVL to open it instead.
