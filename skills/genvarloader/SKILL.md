---
name: genvarloader
description: Use when writing or reading GenVarLoader (gvl) datasets — preparing VCF/PGEN/SVAR variant sources with bcftools/plink2, calling gvl.write, configuring gvl.Dataset for haplotype/reference/annotated/variants output modes, attaching BigWig or Table tracks, setting up spliced haplotypes from a GTF, choosing track insertion-fill strategies for indels, or filtering variants by allele frequency.
---

# GenVarLoader Public API

GenVarLoader (`gvl`) reconstructs personalized haplotypes and re-aligns functional genomic tracks on the fly from a reference + variants + BigWig/Table tracks, without writing personalized genomes to disk. Variable-length output is the norm (indels make lengths region- and sample-dependent).

This skill is a pointer-dense overview. Symbol names link to where to find the authoritative docstring or source.

## End-to-end shape

```python
import genvarloader as gvl

# 1. Preprocess variants outside Python (see "Variant preprocessing")
# 2. Write the dataset
gvl.write(
    path="ds.gvl",
    bed="rois.bed",
    variants="normed.bcf",  # or .pgen, or .svar directory
    tracks=[gvl.BigWigs.from_table("signal", "bw_table.tsv")],
    max_jitter=128,
)

# 3. Open and configure (chainable fluent API)
ds = (
    gvl.Dataset
    .open("ds.gvl", reference="ref.fa")
    .with_seqs("haplotypes")
    .with_tracks(["signal"])
    .with_insertion_fill(gvl.Repeat5pNormalized())
    .with_len(2048)  # or "ragged" / "variable"
    .with_settings(jitter=32, deterministic=False)
)

# 4. Eager indexing: dataset[region_idx, sample_idx]
batch = ds[0:8, :]  # shape depends on with_* state — see "Output shapes"
```

## Variant preprocessing requirements

Variants passed to `gvl.write` **must be** left-aligned, bi-allelic, and atomized (no MNPs or compound MNP-indels). VCFs must be indexed.

```bash
# VCF/BCF
bcftools norm -f ref.fa \
    -a --atom-overlaps . \
    -m -any --multi-overlaps . \
    -O b -o normed.bcf in.vcf.gz
bcftools index normed.bcf

# PGEN
plink2 --make-bpgen --pfile in --out tmp
plink2 --make-pgen --normalize --ref-from-fa --fa ref.fa --bpfile tmp --out normed
```

See `docs/source/write.md` for the canonical recipe and BED/BigWig table layouts.

## When to use SVAR vs BCF/PGEN

`.svar` is a sparse columnar variant archive (from `genoray`). Pass it to `gvl.write(variants="x.svar")` exactly like a BCF or PGEN — the resulting dataset stores a back-reference instead of duplicating per-variant arrays.

Use SVAR when:
- You need **allele-frequency filtering at read time** (`Dataset.open(min_af=..., max_af=...)` requires SVAR-backed genotypes — will raise otherwise).
- Many datasets share the same variant source — SVAR avoids duplicating `variant_idxs.npy`/`dosages.npy`/`variants.arrow` into each `.gvl` directory.
- You're working at population scale and want compact on-disk variant storage.

Use BCF/PGEN directly when you have a one-off dataset and don't need AF filtering.

Create an SVAR from a normalized VCF/PGEN with `genoray`:

```python
from genoray._svar import dense2sparse
from genoray import VCF

dense2sparse(VCF("normed.bcf"), "normed.svar")  # writes a .svar/ directory
```

SVARs are resolved at `Dataset.open` time via `metadata.json` → caller `svar=` arg → recorded relative path → recorded absolute path → sibling `*.svar`. See `docs/source/format.md` ("SVAR resolution at open time") and `_dataset/_svar_link.py`. Legacy symlink-based SVAR layouts: run `gvl.migrate_svar_link(path)` once to upgrade.

## `gvl.write` — key arguments

```python
gvl.write(
    path,
    bed,
    variants=None,
    tracks=None,
    samples=None,
    max_jitter=None,
    overwrite=False,
    max_mem="4g",
    extend_to_length=True,
)
```

Notable:
- `bed`: path or polars DataFrame with `chrom, chromStart, chromEnd` (0-based). Optional `strand` (`+`/`-`/`.`) controls reverse-complement on read. Extra columns are preserved on `Dataset.regions`.
- `tracks`: a `gvl.BigWigs`, `gvl.Table`, or a list of them. Each must have a unique `.name`. BigWigs need a sample→path mapping (dict or table with `sample`, `path` columns; see `BigWigs.from_table`).
- `max_jitter`: max read-time jitter; pads stored data on both sides of every region by this many bases so `Dataset.with_settings(jitter=j)` works for any `j <= max_jitter`.
- `extend_to_length=True` keeps reading past the BED end until every haplotype is ≥ the region length (matters when deletions would shorten output); set `False` for faster writes if shorter haps are acceptable.
- Inner-joins samples across `variants` and all `tracks`.

Source: `python/genvarloader/_dataset/_write.py`.

## `Dataset.open` — key arguments

```python
gvl.Dataset.open(
    path, reference=None, jitter=0, rng=None,
    deterministic=True, rc_neg=True,
    min_af=None, max_af=None,           # SVAR only
    region_names=None,
    splice_info=None,                    # see "Spliced haplotypes"
    var_filter=None,                     # None | "exonic"
    var_fields=None,                     # list[str] | None — see below
    *, svar=None,
)
```

Without `reference=`, only variants/haplotypes are available (you can't produce reference-overlaid sequences). `svar=` overrides the recorded SVAR location.

- **`var_fields: list[str] | None`** — Variant fields to include on `RaggedVariants` output. Defaults to the minimum useful set `["alt", "ilen", "start"]`. Pass additional names (e.g. `"ref"`, `"dosage"`, or any numeric info column in the source variants table) to load them eagerly at open time. Must be a subset of `Dataset.available_var_fields`. Can be reconfigured later via `Dataset.with_settings(var_fields=...)`, which lazily loads any newly-requested columns. `"dosage"` must be requested explicitly — it is *not* added automatically even when `dosages.npy` exists on disk.

## Output modes — `with_seqs` × `with_tracks`

`with_seqs(kind)` selects the sequence output channel:

| `kind`         | Returns                                    | Use when                                          |
|----------------|--------------------------------------------|---------------------------------------------------|
| `"reference"`  | Reference sequence (`S1`)                  | Baseline / no personalization                     |
| `"haplotypes"` | Personalized haplotypes with indels (`S1`) | Standard variant-aware modeling                   |
| `"annotated"`  | `AnnotatedHaps` (haps + var_idxs + ref_coords) | Need to map back to variants/ref coords      |
| `"variants"`   | `RaggedVariants` (variants only, no seq)   | Variant-centric tasks                             |
| `None`         | No sequences                               | Tracks-only datasets                              |

`with_tracks(tracks=..., kind=...)` selects tracks:
- `tracks`: `None` (default), `False` (disable), a single name, or a list of names.
- `kind`: `"tracks"` (re-aligned numeric values) or `"intervals"` (raw interval representation).

`with_len(L)` controls output shape:
- `"ragged"` (default): returns `gvl.Ragged` (variable length per item).
- `"variable"`: NumPy array right-padded to the batch's longest item (`N` for seqs, `0` for tracks).
- integer `L`: fixed length; jitter/random shift/truncate/pad-with-more-personalized-data combine to meet `L`. Must satisfy `L + 2·jitter ≤ min(region_length) + 2·max_jitter`.

Returns either a `RaggedDataset` or `ArrayDataset` (frozen dataclass views) based on `with_len`. See `docs/source/dataset.md` for diagrams.

## Track insertion fill (only when haps + tracks together)

Indels make track length differ from reference length. `Dataset.with_insertion_fill(fill)` controls what gets written into inserted positions. Only valid when the dataset returns **both** haplotypes and tracks — pure-ref and pure-hap datasets ignore it (raises if attempted).

| Strategy                       | Behavior                                                              |
|--------------------------------|-----------------------------------------------------------------------|
| `gvl.Repeat5p()` *(default)*   | Repeat the value at variant POS across the insertion.                 |
| `gvl.Repeat5pNormalized()`     | Repeat `track[POS] / (insertion_len + 1)`. Preserves sum.             |
| `gvl.Constant(value=nan)`      | Constant value (default NaN) across the insertion.                    |
| `gvl.FlankSample(flank_width=5)` | Resample with replacement from a 2·flank_width+1 window around POS. |
| `gvl.Interpolate(order=1)`     | Polynomial interp (order 1/2/3) between flanking reference values.    |

Pass a single strategy (applies to every track) or a `dict[track_name, strategy]` (missing tracks fall back to `Repeat5p`). Source: `python/genvarloader/_dataset/_insertion_fill.py`.

## Spliced haplotypes

Splicing is opt-in at `Dataset.open` (or via `with_settings`). It groups the BED rows for one transcript and concatenates exon-level sequences/tracks per sample.

```python
splice_bed = gvl.get_splice_bed("annotation.gtf", transcript_support_level="1")
gvl.write(path="splice.gvl", bed=splice_bed, variants="normed.svar")

sds = gvl.Dataset.open(
    "splice.gvl",
    reference="ref.fa",
    splice_info=("transcript_id", "exon_number"),  # tuple = (group_col, order_col)
    var_filter="exonic",  # optional: drop intronic variants
)
```

`splice_info` accepts:
- a column name string (single grouping column, order inferred from BED row order), or
- a `(group_col, order_col)` tuple (explicit ordering, e.g. exon number).

`get_splice_bed` does GTF→BED with TSL filtering and an optional "CDS length multiple of 3" filter. To roll your own splice BED, just include `transcript_id` (or any grouping column) and `exon_number` columns on the BED. See `docs/source/splicing.ipynb`.

### RefDataset splicing

`gvl.RefDataset` accepts the same `splice_info` argument as `Dataset.open`. Pass either a transcript-ID column name (rows already in splice order) or a `(group_col, sort_col)` tuple to reorder exons. `with_settings(splice_info=False)` disables splicing on an existing `RefDataset`; pass a new value to re-enable. Splicing requires `output_length` in `{"ragged", "variable"}`, `jitter=0`, and `deterministic=True`. `subset_to(transcript_ids)` works the same as for `Dataset`.

```python
ref = gvl.Reference.from_path("hg38.fa.bgz")
bed = gvl.get_splice_bed("annotations.gtf")
ref_ds = gvl.RefDataset(ref, bed, splice_info="transcript_id")
seqs = ref_ds[:]  # Ragged[S1], one row per transcript
```

## Site-only variants (e.g. ClinVar)

Use `gvl.sites_vcf_to_table(vcf)` → `pl.DataFrame` (bi-allelic SNPs only), then wrap an `ArrayDataset[AnnotatedHaps, ...]` with `gvl.DatasetWithSites(ds, sites, max_variants_per_region=1)`. Returns `(wt_haps, mut_haps, flags[, tracks])`; flags encode applied / deleted-overlap / already-existing. See `_variants/_sitesonly.py`.

## Prefetching dataloader (`mode=...` on `to_dataloader`)

`Dataset.to_dataloader()` accepts an optional `mode` to coarsen fetching: gvl's throughput scales with fetch size (internal multithreading amortizes overhead), so one big `dataset[r_idx, s_idx]` call sliced into mini-batches outperforms many small per-batch calls.

```python
loader = ds.to_dataloader(
    batch_size=32,
    mode="double_buffered",   # or "buffered", or None
    buffer_bytes=2 * 1024**3, # total RAM budget; split across slots in double mode
    copy=True,                # zero-copy opt-out (default True = safe)
    heartbeat_seconds=60.0,   # double_buffered: max wait per chunk before liveness check
)
```

Modes:

- `None` (default) — plain `torch.utils.data.DataLoader`; existing behavior.
- `"buffered"` — main process fetches one chunk per call (sized to `buffer_bytes`), slices into mini-batches. Refill latency visible but amortized over many batches.
- `"double_buffered"` — subprocess producer fills one shm slot while consumer drains the other; refill latency hidden. Requires a file-backed `Dataset.open(path)`.

Preconditions (all raise `ValueError` at construction):

- `with_seqs in {"haplotypes", "annotated"}` requires `deterministic=True` (set via `with_settings(deterministic=True)`). `reference` and `variants` modes have no determinism requirement.
- Spliced datasets are not supported.
- `num_workers > 0` is rejected — the new loader IS the concurrency strategy.
- A single mini-batch whose exact footprint exceeds the per-slot capacity raises with the offending size and remediation knobs (`batch_size↓`, `buffer_bytes↑`).

Footprint is computed exactly via `Dataset._output_bytes_per_instance(...)` (uses `haplotype_lengths`, `n_variants`, and allele offset tables) — no Zipf-style worst-case slack.

## Other public surface (one-liners)

- `gvl.Reference.from_path(fasta, contigs=None)` — wrap a FASTA. Cached.
- `gvl.read_bedlike(path)` / `gvl.with_length(bed, L)` — BED helpers (re-exported from `seqpro`).
- `gvl.Ragged`, `gvl.RaggedAnnotatedHaps`, `gvl.RaggedVariants`, `gvl.RaggedIntervals` — ragged return containers.
- `gvl.to_nested_tensor(ragged)` — convert to a PyTorch nested tensor (requires `torch`).
- `gvl.get_dummy_dataset()` — small in-memory dataset for examples/tests.
- `gvl.RefDataset` — reference-only dataset (no genotypes).
- `gvl.Table` — generic interval track from a DataFrame.
- `gvl.data_registry.fetch(name)` — download public test/demo datasets.

Full list lives in `python/genvarloader/__init__.py` `__all__`.

## On-disk layout (quick reference)

```
ds.gvl/
├── metadata.json          # version, samples, contigs, ploidy, max_jitter, svar_link?
├── input_regions.arrow    # BED + region index map
├── genotypes/             # variant_idxs.npy, dosages.npy, variants.arrow
│                          # (absent when sourced from .svar; see svar_link)
└── intervals/<track>/     # per-track interval data
```

See `docs/source/format.md` for the full schema, versioning, and SVAR-link details.

## Where to look next

| For…                                  | Read…                                                  |
|---------------------------------------|--------------------------------------------------------|
| End-to-end RNA-seq example            | `docs/source/geuvadis.ipynb`                           |
| Splicing tutorial                     | `docs/source/splicing.ipynb`                           |
| Deep-learning eval pipeline           | `docs/source/basenji2_eval.ipynb`                      |
| BED / BigWig / bcftools recipes       | `docs/source/write.md`                                 |
| `Dataset` shapes, ragged/var/fixed    | `docs/source/dataset.md`                               |
| On-disk format + SVAR resolution      | `docs/source/format.md`                                |
| FAQ (`with_*` design, typing)         | `docs/source/faq.md`                                   |
| Auto-generated reference              | `docs/source/api.md` → https://genvarloader.readthedocs.io |
| Track re-alignment internals          | `python/genvarloader/_dataset/_tracks.py`, `_reconstruct.py` |
| Insertion fill internals              | `python/genvarloader/_dataset/_insertion_fill.py`      |
| SVAR back-reference / migration       | `python/genvarloader/_dataset/_svar_link.py`           |

## Common gotchas

- `with_insertion_fill` raises unless the dataset has both haplotypes AND tracks active.
- `min_af` / `max_af` raise unless the dataset is SVAR-backed.
- `with_len(L)` requires `L + 2·jitter ≤ min(region_length) + 2·max_jitter` — set `max_jitter` accordingly at `write` time.
- Tracks must have unique `.name`; the on-disk layout is `intervals/<name>/`.
- BED `strand` of `.` is treated as `+`. Reverse-complement happens automatically when `rc_neg=True` (default) and `strand == "-"`.
- Splicing is a read-time setting on a *flat* BED of exons — do not pre-concatenate exons before `gvl.write`.
- `extend_to_length=False` at write time will produce haplotypes shorter than the BED region when deletions are present; downstream code must tolerate `<` region length.
- Missing a `dosage` field on a `RaggedVariants` output you expected? Check `var_fields` — `dosage` must be requested explicitly even if `dosages.npy` exists on disk.

## Maintaining this skill

Whenever a PR changes the **public API** (anything in `python/genvarloader/__init__.py` `__all__`, or the docstring/signature of `gvl.write`, `Dataset.open`, or any `Dataset.with_*` method), the author must also update this `SKILL.md`. New public symbols, removed symbols, renamed args, changed defaults, and new output modes are all in scope. CLAUDE.md enforces this as part of the contribution checklist.
