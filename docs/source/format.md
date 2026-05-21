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
│   ├── offsets.npy        # per (region, sample, ploidy) offsets into variant_idxs.npy
│   ├── svar_meta.json     # shape + dtype of offsets.npy — present iff source was .svar
│   ├── variant_idxs.npy   # variant indices; absent when sourced from .svar
│   ├── dosages.npy        # optional, absent when sourced from .svar
│   └── variants.arrow     # variant table; absent when sourced from .svar
└── intervals/             # or annot_intervals/ when annotated; present iff tracks given
```

When the dataset was built from an `.svar`, the heavy per-variant arrays (`variant_idxs.npy`,
`dosages.npy`, `index.arrow`) are **not duplicated** into the dataset. Instead the dataset
records a back-reference to the source `.svar` in `metadata.json` (see `svar_link` below).

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

## Format changelog

| Version | Change |
|---------|--------|
| `< 0.18.0` | Variant coordinates stored 0-based. |
| `0.18.0` | Variant coordinates switched to 1-based. |
| `0.25.0` | `metadata.json` gains `svar_link`; old `genotypes/link.svar` symlink layout deprecated. `Metadata.version` typed as `SemanticVersion` (on-disk JSON unchanged). |

> **Upgrading legacy datasets.** A dataset written before `0.25.0` that was built from an
> `.svar` will still open (with a `DeprecationWarning`). Run
> `genvarloader.migrate_svar_link(path)` to convert the symlink layout to the new metadata
> layout in place.
