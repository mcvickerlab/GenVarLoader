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
and caches only small per-`(region, sample, ploidy)` range arrays under `genotypes/svar2_ranges/`
— the bulk variant data stays in the `.svar2` store.

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
`P` = ploidy. All arrays are `int64`:

| File | Shape | Notes |
|------|-------|-------|
| `vk_snp_range.npy` | `(R, S, P, 2)` | Per-`(region, sample, ploid)` half-open range into the `.svar2` store's SNP variant-key column. |
| `vk_indel_range.npy` | `(R, S, P, 2)` | Same, for the indel variant-key column. |
| `dense_snp_range.npy` | `(R, 2)` | Per-region (sample-independent) range into the dense SNP store. |
| `dense_indel_range.npy` | `(R, 2)` | Per-region (sample-independent) range into the dense indel store. |
| `region_starts.npy` | `(R,)` | Per-region write-time start coordinate. Retained for parity/debugging; the read path derives per-query starts from the (post-jitter) query regions and does **not** read this array's values. |
| `sample_cols.npy` | `(S,)` | Maps the dataset's selected-sample slot to the `.svar2` store's original sample index. |
| `svar2_meta.json` | — | Records each array's `shape`/`dtype` plus `ploidy`. |

At read time, `Dataset.__getitem__` slices these memmaps (numpy fancy-indexing; no interval
search) to build the flat per-query inputs for the read-bound Rust kernels — no interval-search
tree and no dense-union rebuild happen per read, unlike the `.svar` path.

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

## `.svar2` variants ALT convention

For a pure deletion (e.g. VCF `GTA>G`), decoding `with_seqs("variants")` yields different raw
ALT bytes depending on the backing store: `.svar` reports the VCF anchor base (`b"G"`), while
`.svar2` reports the atomized empty ALT (`b""`) — a genoray `.svar2` format convention, not a
bug. Both stores consume the ALT identically when reconstructing haplotype sequence, so
`with_seqs("haplotypes")` / `with_seqs("annotated")` output is byte-identical between the two
backends; only `RaggedVariants.alt` differs, and only for pure-deletion records. The same holds
for `with_seqs("variant-windows")`: `ref_window` is byte-identical between the backends, while the
`alt`/`alt_window` fields differ only for pure-deletion records (the same empty-vs-anchor ALT).

## `.svar2` Phase-1 unsupported combinations

A `.svar2`-backed dataset supports all four output modes (`haplotypes`, `variants`,
`variant-windows`, and haplotype-realigned `tracks`), `unphased_union`, and
`var_fields`-selected store INFO/FORMAT fields (on both `"variants"` and `"variant-windows"`).
The following combinations are Phase-1 scope and raise `NotImplementedError` (or, for
`extend_to_length`, at write time) instead of silently mis-computing:

- Spliced output.
- The `var_filter="exonic"` (keep-mask) variant filter.
- `min_af` / `max_af` filtering (`.svar` only; see "Should I use `.svar` or `.svar2`" in the FAQ).
- `annotated` haplotypes (`with_seqs("annotated")`).
- `VarWindowOpt(ref="allele")` (bare-allele REF mode; REF alleles aren't stored in `.svar2`).
- In-kernel reverse-complement (`to_rc`).
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

> **Upgrading legacy datasets.** A dataset written before `0.25.0` that was built from an
> `.svar` will still open (with a `DeprecationWarning`). Run
> `genvarloader.migrate_svar_link(path)` to convert the symlink layout to the new metadata
> layout in place.
