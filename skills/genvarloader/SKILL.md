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

Variants passed to `gvl.write` **must be** left-aligned, bi-allelic, atomized (no MNPs or compound MNP-indels), and **free of symbolic (`<DEL>`, `<INS>`, …) and breakend ALT alleles**. gvl expands every ALT into literal haplotype sequence and cannot reconstruct symbolic or breakend records — `gvl.write` raises `ValueError` (with per-class counts of multi-allelic, symbolic, and breakend variants) if any are present, for VCF, PGEN, and SVAR inputs alike. VCFs must be indexed.

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

## `.svar2` — the read-bound sparse variant format

`.svar2` is genoray's newer sparse columnar variant store. Pass it to `gvl.write` exactly like a `.svar`, BCF, or PGEN — `gvl.write(path, bed, variants="cohort.svar2")` or `variants=SparseVar2("cohort.svar2")`. Like `.svar`, the dataset stores a back-reference (`metadata.json` → `svar2_link`) instead of duplicating per-variant arrays, so the `.svar2` store must remain accessible at read time.

Unlike `.svar` (whose read path builds an interval search tree + a per-read dense-union over the queried window), a `.svar2`-backed dataset reconstructs via a **read-bound** path: `gvl.write` caches small per-`(region, sample, ploid)` variant-key ranges under `<dataset>/genotypes/svar2_ranges/` (sized to the dataset's *selected* samples, not the full `.svar2` cohort), and at read time gvl gathers directly off that cache and calls all-Rust kernels — **no interval-search-tree build and no dense-union rebuild per read**. `.svar2` stores are also typically smaller on disk than `.svar`, especially for large cohorts. See `docs/source/faq.md`.

`.svar2` is resolved at `Dataset.open` time in the same order as `.svar`: caller `svar2=` arg → recorded relative path → recorded absolute path → sibling `*.svar2`. `Dataset.open(path, svar2=<override>)` mirrors `svar=`. See `docs/source/format.md` ("`.svar2` resolution at open time").

**Phase-1 scope — unsupported combinations raise `NotImplementedError`.** `.svar2`-backed datasets support all four output modes (`haplotypes`, `variants`, `variant-windows`, and haplotype-realigned `tracks`) byte-identical to the `.svar`/union-oracle backend, and `with_seqs("variant-windows")` (`ref="window"`, `alt ∈ {"window", "allele"}`) and `unphased_union` (for both `"variants"` and `"variant-windows"` output) are both fully wired for `.svar2`. The following are still not yet wired and raise a clear error instead of silently mis-computing:
- Spliced output.
- The `var_filter="exonic"` (keep-mask) variant filter.
- `min_af` / `max_af` filtering.
- `annotated` haplotypes (`with_seqs("annotated")`).
- `VarWindowOpt(ref="allele")` (bare-allele REF mode; blocked upstream of `.svar2` too — REF alleles aren't stored).
- In-kernel reverse-complement (`to_rc`).
- Fixed-length (integer `output_length`) haplotype-realigned **track** output (plain haplotype output at a fixed length is fine — only the track kernel is guarded).
- `variants` / `variant-windows` output on a dataset written with `max_jitter>0` or read with `jitter>0` (the read-bound decode does not right-clip to the post-jitter window; haplotypes and tracks are unaffected and support jitter fully).
- `FlankSample` insertion-fill for tracks spanning **multiple contigs** in one query (single-contig queries and non-seeded fills like the default `Repeat5p` are exact).

**`variants`/`variant-windows` ALT bytes differ from `.svar` for pure deletions (format convention, not a bug).** For a pure deletion (e.g. VCF `GTA>G`), `with_seqs("variants")` on a `.svar` dataset yields the VCF anchor base as ALT (`b"G"`), while a `.svar2` dataset yields the atomized empty ALT (`b""`) — this is how genoray's `.svar2` format represents pure deletions. The same convention carries into `with_seqs("variant-windows")`: `ref_window` is byte-identical between `.svar`/`.svar2`, but `alt`/`alt_window` differ for pure-deletion records for the same reason. Reconstructed **haplotypes are byte-identical** between the two backends (both consume the ALT identically when building sequence); only the raw allele/window bytes differ for pure-deletion records. See `docs/source/faq.md`.

Symbolic/breakend variants are rejected the same as `.svar`, but for `.svar2` the rejection happens **upstream, at `.svar2` conversion time** (the store format cannot represent them) — a `.svar2` must be built from an already-filtered source; gvl cannot re-filter a materialized `.svar2` any more than it can a materialized `.svar`.

## `gvl.write` — key arguments

```python
gvl.write(
    path,
    bed,
    variants=None,
    tracks=None,
    annot_tracks=None,
    samples=None,
    max_jitter=None,
    overwrite=False,
    max_mem="4g",
    extend_to_length=True,
)
```

Notable:
- `bed`: path or polars DataFrame with `chrom, chromStart, chromEnd` (0-based). Optional `strand` (`+`/`-`/`.`) controls reverse-complement on read. Extra columns are preserved on `Dataset.regions`.
- `tracks`: a `gvl.BigWigs` (or a list of them), or a `gvl.Table`. Each must have a unique `.name`. BigWigs need a sample→path mapping (dict or table with `sample`, `path` columns; see `BigWigs.from_table`). `gvl.Table` is a core interval-track source backed by a Rust COITrees overlap engine (zero-based half-open coordinates); pass it directly as a `tracks=` or `annot_tracks=` source in `gvl.write`.
- `annot_tracks`: `dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None` — sample-independent annotation tracks, written to `<path>/annot_intervals/<name>/`. Each value is either a path to an interval table/bigWig file, or a polars DataFrame/LazyFrame with BED-like columns (`chrom`, `chromStart`, `chromEnd`, `score`). Annotation tracks are sample-independent and can be read without a per-sample variant source.
- `max_jitter`: max read-time jitter; pads stored data on both sides of every region by this many bases so `Dataset.with_settings(jitter=j)` works for any `j <= max_jitter`.
- `extend_to_length=True` keeps reading past the BED end until every haplotype is ≥ the region length (matters when deletions would shorten output); set `False` for faster writes if shorter haps are acceptable.
- Inner-joins samples across `variants` and all `tracks`.

**Parallelism:** `gvl.write` now parallelizes over write categories. Variants are processed first (serially). Then per-sample `tracks` and `annot_tracks` run concurrently (joblib loky backend). The `max_mem` budget is divided across the concurrently-running categories.

Source: `python/genvarloader/_dataset/_write.py`.

**Atomic creation:** `gvl.write` builds into a private sibling temp directory and publishes via an atomic `os.replace`. A best-effort `filelock` avoids redundant rebuilds when parallel jobs share the same destination, but correctness relies on the rename — the lock is advisory only. **Datasets do not auto-rebuild**; if the on-disk artifact is missing or corrupt, re-run `gvl.write`.

**Out-of-scope:** `genoray` `.gvi` index files and `pysam` `.fai`/`.gzi` index files are created by those libraries and are **not** covered by gvl's atomic/locked creation. Concurrent jobs that trigger index creation for those files depend on the upstream libraries' behavior.

## `gvl.update` — add tracks to an existing dataset

```python
gvl.update(
    dataset,          # str | Path | Dataset
    tracks=None,      # BigWigs | Table | Sequence[BigWigs | Table] | None
    annot_tracks=None,  # dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None
    *,
    overwrite=False,
    max_mem="4g",
) -> None
```

Adds tracks to an **existing** on-disk GVL dataset without rewriting it from scratch.

- `dataset`: path to a dataset directory, or an opened `Dataset` (its `.path` is used). A live dataset can be read during `update`; it will not observe the new track until reopened.
- `tracks`: per-sample `BigWigs` or `Table` sources. The track's sample set must match the dataset's **exactly** (no missing, no extra); samples are reordered to dataset order automatically. Written to `<path>/intervals/<track>/`.
- `annot_tracks`: sample-independent sources, identical to `gvl.write`'s `annot_tracks` (path to interval table, path to bigWig, or polars DataFrame/LazyFrame with BED-like columns). Written to `<path>/annot_intervals/<name>/`.
- `overwrite=True`: replace a same-named existing track; `False` (default) raises `FileExistsError` if the name already exists.
- `max_mem`: approximate memory budget, divided across concurrently-running categories.

Each track subdirectory is published **atomically** (built into a temp sibling, then `os.replace`d into place), so a reader can never see a half-written track.

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
    *, svar=None, svar2=None,
)
```

Without `reference=`, a genotypes-only dataset opens with the **`"variants"`** view by default (yielding `RaggedVariants`) — `Dataset.open(path)` just works, no `with_seqs` needed. The `"haplotypes"`, `"annotated"`, and `"reference"` views all require a reference; requesting one via `with_seqs` on a reference-less dataset raises a clear `ValueError`. `svar=` overrides the recorded SVAR location; `svar2=` mirrors it for a `.svar2`-backed dataset.

**`with_settings(dummy_variant=...)`** — inserts a `gvl.DummyVariant` into every **empty** `(region, sample, ploid)` variant group so that every group has at least one variant. Only fills groups that are empty; non-empty groups are unchanged. Valid for both `"variants"` and `"variant-windows"` output kinds; indexing raises `ValueError` if `dummy_variant` is set and the output kind is any other kind (`"haplotypes"`, `"annotated"`, `"reference"`, or no seqs) — the check is order-independent with `with_seqs`. `False` disables dummy padding; `None` (default) leaves the current setting unchanged. Setting `dummy_variant=False` when the output is an unsupported kind is a harmless no-op.

For **token outputs** (ride-along `FlatVariants.flank_tokens` and `"variant-windows"` fields `ref_window`/`alt_window`/bare `ref`/`alt`), each empty group's dummy entry is filled entirely with the configured `unknown_token` (the user-supplied integer that out-of-alphabet bytes map to — alphabet-agnostic, so `N` for DNA/RNA, `X` for amino acids tokenize to it). The `DummyVariant.ref`/`.alt` bytes only determine the dummy allele's byte-length, not the token value. Fill lengths per empty `(region, sample, ploid)` group: ride-along `flank_tokens` → `2·flank_length` unknown tokens; `ref_window`/`alt_window` → `2·flank_length + len(dummy allele)` unknown tokens; bare `ref`/`alt` → `len(dummy allele)` unknown tokens. (Default `b"N"` allele → length 1.)

Scalar fields (`start`/`ilen`/`dosage`/`info[...]`) are still filled from `DummyVariant` values as before. The dummy fill applies before reverse-complementing on the `"variants"`-output alleles only, so a non-`b"N"` dummy allele is reverse-complemented on negative-strand regions like any real allele — the default `b"N"` is rc-invariant. Flank tokens and the variant-window token buffers are reference-oriented and are NOT RC'd.

**`with_settings(unphased_union=...)`** — fold the stored diploid haplotypes onto a single haploid sequence: the union of called ALTs per `(region, sample)`. When `True`, `ds.ploidy` reports `1` (instead of the stored `2`); `n_variants(...)` reports a single ploidy slot (shape `(..., 1)`), with counts equal to the naive per-haplotype sum (a hom call appears twice — once per haplotype — with no dedup). `"variants"` and `"variant-windows"` output decode at ploidy `1`; ALT occurrences are concatenated across haplotypes with no sort and no dedup. Phase is discarded — intended for haploid somatic modeling of unphased somatic calls. Requires a dataset with genotypes (raises `ValueError` on reference-only datasets). Incompatible with `"haplotypes"` / `"annotated"` output — `with_seqs("haplotypes")` or `with_seqs("annotated")` (or setting this flag while one of those is the active output kind) raises `ValueError`. See issue #222.

**Format validation:** `Dataset.open` validates the dataset's `format_version` and structural integrity (file presence + sizes). A corrupt dataset raises a `ValueError` instructing regeneration with `gvl.write`. Datasets do **not** auto-rebuild.

**Format version gate (2.0):** the current on-disk format is **2.0.0**. Opening a dataset written by genvarloader **< 2.0** (or any unversioned dataset) raises a `ValueError` whose message points at `gvl.migrate(path)`; a dataset written by a *newer* major raises a `ValueError` telling you to upgrade genvarloader. Run `gvl.migrate(path)` **once** to upgrade a pre-2.0 dataset in place — it is streaming (peak extra disk is one track's interval store), idempotent, and crash-safe (metadata is bumped only after every track's struct-of-arrays files are durable, then the old array-of-structs files are deleted). It converts the track-interval storage only; genotypes, regions, and reference are untouched.

- **`var_fields: list[str] | None`** — Variant fields to include on `RaggedVariants` output. Defaults to the minimum useful set `["alt", "ilen", "start"]`. Pass additional names (e.g. `"ref"`, `"dosage"`, or any numeric info column in the source variants table) to load them eagerly at open time. Must be a subset of `Dataset.available_var_fields`. Can be reconfigured later via `Dataset.with_settings(var_fields=...)`, which lazily loads any newly-requested columns. `"dosage"` must be requested explicitly — it is *not* added automatically even when `dosages.npy` exists on disk. Beyond the built-ins (`alt`, `start`, `ref`, `ilen`, `dosage`) and per-variant INFO columns, a genoray `.svar` may register arbitrary per-call (`Number=G`) FORMAT fields in `<svar>/metadata.json["fields"]`; these appear in `Dataset.available_var_fields` and can be requested via `Dataset.open(..., var_fields=[...])` or `with_settings(var_fields=[...])`. Each surfaces in `variants`, `variant-windows`, and `flat` outputs as a per-call ragged field aligned with the genotypes. A FORMAT field shadows a same-named INFO column.

## Output modes — `with_seqs` × `with_tracks`

`with_seqs(kind)` selects the sequence output channel:

| `kind`              | Returns                                    | Use when                                          |
|---------------------|--------------------------------------------|---------------------------------------------------|
| `"reference"`       | Reference sequence (`S1`)                  | Baseline / no personalization                     |
| `"haplotypes"`      | Personalized haplotypes with indels (`S1`) | Standard variant-aware modeling                   |
| `"annotated"`       | `AnnotatedHaps` (haps + var_idxs + ref_coords) | Need to map back to variants/ref coords      |
| `"variants"`        | `RaggedVariants` (variants only, no seq)   | Variant-centric tasks                             |
| `"variant-windows"` | `FlatVariantWindows` (per-allele window/allele token buffers; flat mode only) | Tokenized model input around each variant |
| `None`              | No sequences                               | Tracks-only datasets                              |

`"variant-windows"` requires a `VarWindowOpt` second argument (`with_seqs("variant-windows", gvl.VarWindowOpt(...))`), `with_output_format("flat")`, and a reference genome. It does **not** inherit flank settings from `with_settings`.

`variant-windows` may be combined with tracks when `with_settings(realign_tracks=False)` is set; the returned tracks/intervals are reference-coordinate (as-is). Float tracks come back as `FlatRagged`, interval tracks as `FlatIntervals`.

`with_tracks(tracks=..., kind=...)` selects tracks:
- `tracks`: `None` (default), `False` (disable), a single name, or a list of names.
- `kind`: `"tracks"` (re-aligned numeric values) or `"intervals"` (raw interval representation).

Track **re-alignment** to haplotype coordinates is controlled by `with_settings(realign_tracks=True)` (default). Set `realign_tracks=False` for reference-coordinate ("as-is") tracks. `realign_tracks=False` is **required** for `kind="intervals"` with any variant-aware seq mode, and for `variant-windows` + tracks. `with_insertion_fill` requires `realign_tracks=True`.

`with_len(L)` controls output shape:
- `"ragged"` (default): returns `gvl.Ragged` (variable length per item).
- `"variable"`: NumPy array right-padded to the batch's longest item (`N` for seqs, `0` for tracks).
- integer `L`: fixed length; jitter/random shift/truncate/pad-with-more-personalized-data combine to meet `L`. Must satisfy `L + 2·jitter ≤ min(region_length) + 2·max_jitter`.

Returns either a `RaggedDataset` or `ArrayDataset` (frozen dataclass views) based on `with_len`. See `docs/source/dataset.md` for diagrams.

`with_output_format(fmt)` selects the container type returned by eager indexing:

| `fmt`      | Container types returned                                                   | Default? |
|------------|----------------------------------------------------------------------------|----------|
| `"ragged"` | `Ragged` / `RaggedVariants` / `RaggedAnnotatedHaps` (all `seqpro._core.Ragged`-backed) | Yes |
| `"flat"`   | Pure-numpy `FlatRagged` / `FlatVariants` / `FlatAnnotatedHaps` / `FlatIntervals` | No       |

In `"flat"` mode the hot path is zero-awkward; the returned containers carry `.data` (flat numpy array) and `.offsets` (int64). Every flat type has `.to_ragged()` back to its `_core.Ragged`-backed form. Densification escape hatches vary by type: `FlatRagged` has `.to_fixed(length)` and `.to_padded(pad_value)`; `FlatAnnotatedHaps` has `.to_fixed(length)` and `.to_padded()` (no arg — uses per-field pad defaults); `FlatVariants`/`FlatAlleles` expose only `.to_ragged()` (plus `.reshape`/`.squeeze`).

```python
ds_flat = ds.with_output_format("flat")
result = ds_flat[0:8, :]   # FlatRagged or FlatAnnotatedHaps or FlatVariants
# direct tensorization — no awkward round-trip
import torch
t = torch.from_numpy(result.data)
# or convert back
ragged = result.to_ragged()
```

`with_output_format` is orthogonal to and composes with `with_len` and `subset_to`.

**Scope note:** In `"flat"` mode, `kind="intervals"` tracks return `FlatIntervals` (`.to_ragged()` → `RaggedIntervals`; fields `.starts`/`.ends`/`.values` are each a `FlatRagged`). Float (numeric) tracks return `FlatRagged` in flat mode. Seqs/haplotypes/annotated-haps/reference and variants outputs are also flattened as before.

**Flat variants extras — ride-along flank tokens and variant windows:**

```python
import seqpro as sp
import genvarloader as gvl

# Both paths need tracks disabled — the flat variants/windows channel is not
# produced when tracks are active (see gotchas).

# (a) ride-along flank tokens on the "variants" output
fv = (ds.with_tracks(False).with_seqs("variants").with_output_format("flat")
        .with_settings(flank_length=128, token_alphabet=sp.DNA.alphabet,
                       unknown_token=len(sp.DNA)))[0:8]
fv.flank_tokens          # FlatRagged, shape (b, p, ~v, 2*128), or None if not configured

# (b) per-allele windows: ref as a flanked window, alt as a bare tokenized allele
fw = (ds.with_tracks(False).with_output_format("flat")
        .with_seqs("variant-windows",
                   gvl.VarWindowOpt(flank_length=128, token_alphabet=sp.DNA.alphabet,
                                    unknown_token=len(sp.DNA), ref="window", alt="allele")))[0:8]
fw.ref_window            # flanked ref window tokens (two-level token buffer)
fw.alt                   # bare alt allele tokens (no flanks); fw.alt_window is None
fw.ref_window.shape      # the window buffer's own shape: (b, p, ~v, ~len)
```

**Ride-along `FlatVariants.flank_tokens`** (`with_seqs("variants")` + `with_settings(flank_length=L, token_alphabet=..., unknown_token=...)`): appends a `FlatRagged` of shape `(b, p, ~v, 2L)` to the returned `FlatVariants`. Per variant the buffer holds `[flank5 | flank3]` reference-context tokens (each `L` long). Coordinate rule: `flank5 = [start-L, start)`, `flank3 = [end, end+L)` where `end = start - min(ilen, 0) + 1`. `token_alphabet` (bytes) and `unknown_token` (int) together build a 256-entry byte→token LUT (seqpro-style): each alphabet byte → its 0-based index; every other byte (including `N` and out-of-bounds padding) → `unknown_token`. `flank_length=0`/`None` disables; both `token_alphabet` and `unknown_token` must be set together. Token dtype is `uint8` when max token id ≤ 255, else `int32`; offsets are `int64`. When `with_settings(dummy_variant=...)` is set, each empty `(region, sample, ploid)` group's `flank_tokens` row is a `2L`-long run of `unknown_token`.

**`VarWindowOpt` / `FlatVariantWindows`** (`with_seqs("variant-windows", opt)`): each variant gets a fixed-length token buffer in two modes selected independently for ref and alt via `VarWindowOpt.ref` / `VarWindowOpt.alt` ∈ `{"window", "allele"}`:
- `"window"`: flanked + tokenized — ref-window = tokenized `[start-L, end+L)` reference read; alt-window = tokenized `flank5 · alt-allele · flank3` assembly.
- `"allele"`: the bare tokenized allele (ref or alt bases) with no flanks.

`FlatVariantWindows` sets exactly one of `.ref_window` / `.ref` (the other is `None`) and one of `.alt_window` / `.alt` (the other is `None`). `.fields` is a dict of scalar `FlatRagged` (`start`/`ilen`/`dosage`/info; raw byte alleles are dropped). Flanks and windows are **reference-oriented** — NOT reverse-complemented even when `rc_neg=True`. Splicing is not supported with `"variant-windows"`. When `with_settings(dummy_variant=...)` is set, each empty `(region, sample, ploid)` group is padded with one all-`unknown_token` entry: length `2·flank_length + len(dummy allele)` tokens for `ref_window`/`alt_window` (`"window"` mode), or `len(dummy allele)` tokens for bare `ref`/`alt` (`"allele"` mode).

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

**Flat output composes with buffered modes.** `ds.with_output_format("flat").to_dataloader(mode="buffered" | "double_buffered")` yields `Flat*` mini-batches (`FlatRagged` / `FlatVariants` / `FlatAnnotatedHaps`) instead of `Ragged` / `RaggedVariants`, with zero awkward on the transport path — the `double_buffered` producer writes flat buffers and the consumer reads them back without re-wrapping. Densify each batch with `.to_fixed(length)` / `.to_padded(pad)`, or wrap `batch.data` / `batch.offsets` with `torch.from_numpy` (offsets are int64; cast to int32 for a torch `Nested` tensor). Re-wrap to the `_core.Ragged`-backed types with `.to_ragged()` (element-identical to ragged mode). The mode's preconditions are unchanged: `double_buffered` still requires a file-backed `Dataset.open(path)`, still rejects spliced datasets and non-default `insertion_fill`, and haplotype/annotated output still needs `deterministic=True`. Variants output that carries ride-along flank tokens (`with_settings(flank_length=...)`) and `"variant-windows"` output are **not** supported over the buffered transport — both are rejected up front at `to_dataloader` construction (raising `ValueError`) rather than crashing mid-iteration. For ride-along flank tokens, drop `flank_length` or use the default ragged output (`with_output_format("ragged")`). For `"variant-windows"` (which is flat-only and has no ragged form), use `mode=None` — the plain `DataLoader` indexes per-item and supports it.

Preconditions (all raise `ValueError` at construction):

- `with_seqs in {"haplotypes", "annotated"}` requires `deterministic=True` (set via `with_settings(deterministic=True)`). `reference` and `variants` modes have no determinism requirement.
- Spliced datasets are not supported.
- `num_workers > 0` is rejected — the new loader IS the concurrency strategy.
- A single mini-batch whose exact footprint exceeds the per-slot capacity raises with the offending size and remediation knobs (`batch_size↓`, `buffer_bytes↑`).

Footprint is computed exactly via `Dataset._output_bytes_per_instance(...)` (uses `haplotype_lengths`, `n_variants`, and allele offset tables) — no Zipf-style worst-case slack.

## Other public surface (one-liners)

- `gvl.Reference.from_path(fasta, contigs=None)` — wrap a FASTA (path to a `.fa`/`.fa.bgz`, or a `.gvlfa` cache dir). Builds/reuses a sibling `.gvlfa` cache directory (self-describing, fingerprint-validated; legacy `.fa.gvl` caches auto-migrate). The cache is built atomically (temp + `os.replace`) under a best-effort lock, so concurrent builders sharing one reference are safe; the cache **auto-rebuilds** from its source when stale or missing.
- `gvl.read_bedlike(path)` / `gvl.with_length(bed, L)` — BED helpers (re-exported from `seqpro`).
- `gvl.Ragged`, `gvl.RaggedAnnotatedHaps`, `gvl.RaggedVariants`, `gvl.RaggedIntervals` — ragged return containers. All are backed by `seqpro.rag.Ragged` (`_core.Ragged` Rust backend); **not** `awkward`. `RaggedVariants` is a **subclass** of `seqpro.rag.Ragged` (`class RaggedVariants(seqpro.rag.Ragged)`), so `isinstance(rv, Ragged) is True`. Structural methods — indexing, `reshape`, `squeeze`, `to_packed` — are inherited from the base and **preserve the `RaggedVariants` type** (positional/structural operations return `RaggedVariants`). A **string key** (`rv["start"]`) returns a bare `Ragged` field, not a `RaggedVariants`. `reshape` takes the new shape either as unpacked ints — e.g. `rv.reshape(1, 2, None)` — or as a single tuple `rv.reshape((1, 2, None))`; the base `Ragged` signature accepts both. `squeeze(axis=None)` is a real axis-squeeze (base semantics) — it squeezes any size-1 axis, **not** a fixed "drop axis 0". An int index collapses the leading axis (numpy-consistent); slice/array indexing preserves it. Named properties (`.alt`, `.ref`, `.start`, `.ilen`, `.end`) are the primary access point; extra fields (e.g. `AF`, custom FORMAT fields) are also accessible via `rv["field"]` or `rv.field` (via `__getattr__`). `RaggedVariants` itself does not define `__eq__` (wrapper-level `==` is Python object-identity, not element-wise); to compare contents, compare individual fields — e.g. `rv["alt"] == other_alt` or `rv.start == other_start` — which use `seqpro.rag.Ragged`'s ufunc-based (element-wise) comparison. Domain methods retained on `RaggedVariants`: `.rc_()`, `.pad()`, `.to_nested_tensor_batch()`; derived read-only properties: `.ilen`, `.end`; fields: `.alt`, `.ref`, `.start`, `.dosage`.
- `gvl.FlatRagged` — flat analog of `Ragged`: `.data` (flat numpy array), `.offsets` (int64), `.shape`. Methods: `.to_ragged()`, `.to_fixed(length)`, `.to_padded(pad_value)`, `.reshape(shape)`, `.squeeze(axis)`. Source: `python/genvarloader/_flat.py`.
- `gvl.FlatIntervals` — flat-buffer interval container returned by `with_tracks(kind="intervals")` + `with_output_format("flat")`. Fields `.starts`/`.ends`/`.values` are `FlatRagged`; `.to_ragged()` → `RaggedIntervals`; `.reshape(...)`, `.squeeze(...)`, `.shape`. Source: `python/genvarloader/_ragged.py`.
- `gvl.FlatAnnotatedHaps` — flat analog of `RaggedAnnotatedHaps`: fields `.haps`, `.var_idxs`, `.ref_coords` (each a `FlatRagged`). Methods: `.to_ragged()`, `.to_fixed(length)`, `.to_padded()`, `.reshape(shape)`, `.squeeze(axis)`. Source: `python/genvarloader/_flat.py`.
- `gvl.FlatVariants` — flat analog of `RaggedVariants`: `.fields` dict mapping field names to `FlatRagged` (scalar fields: `start`/`ilen`/`dosage`/info) or `FlatAlleles` (`alt`/`ref`). `.shape` delegates to `fields["start"].shape`. `.flank_tokens`: optional ride-along `FlatRagged` of shape `(b, p, ~v, 2L)` (set when `with_settings(flank_length=L, ...)` is configured; `None` otherwise). Methods: `.to_ragged()`, `.reshape(shape)`, `.squeeze(axis)`. Source: `python/genvarloader/_dataset/_flat_variants.py`.
- `gvl.FlatAlleles` — two-level flat bytestring for allele fields: `.byte_data` (uint8), `.seq_offsets` (per-variant byte offsets, int64), `.var_offsets` (per-(batch×ploidy)-row variant offsets, int64), `.shape`. Methods: `.to_ragged()`, `.reshape(shape)`, `.squeeze(axis)`. Source: `python/genvarloader/_dataset/_flat_variants.py`.
- `gvl.FlatVariantWindows` — returned by `with_seqs("variant-windows", VarWindowOpt(...))` in flat mode. `.fields`: dict of scalar `FlatRagged` (`start`/`ilen`/`dosage`/info; raw byte alleles are dropped). Per-allele token buffers — exactly one of `.ref_window` (flanked ref window, `"window"` mode) or `.ref` (bare ref allele tokens, `"allele"` mode) is set; same for `.alt_window` / `.alt`. Each non-None buffer is a two-level token buffer (internal `_FlatWindow`, not the public `FlatRagged`) of shape `(b, p, ~v, ~len)` with its own `.to_ragged()`. The container's `.shape` delegates to `fields["start"].shape`. Methods: `.to_ragged()` (returns dict of ragged parts), `.reshape(shape)`, `.squeeze(axis)`. Source: `python/genvarloader/_dataset/_flat_variants.py`.
- `gvl.VarWindowOpt` — frozen config dataclass for `with_seqs("variant-windows", ...)`. Fields: `flank_length` (int), `token_alphabet` (bytes), `unknown_token` (int), `ref` ∈ `{"window","allele"}`, `alt` ∈ `{"window","allele"}`. `ref` and `alt` are chosen independently. `"window"` = flanked + tokenized reference read (ref) or flank·alt·flank assembly (alt); `"allele"` = bare tokenized allele with no flanks. Source: `python/genvarloader/_dataset/_flat_variants.py`.
- `gvl.DummyVariant` — frozen dataclass used with `with_settings(dummy_variant=...)`. Fields and defaults: `start: int = -1`, `ilen: int = 0`, `dosage: float = 0.0`, `ref: bytes = b"N"`, `alt: bytes = b"N"`, `info: dict = {}`. Unspecified `info` keys default to `0` for integer columns and `NaN` for float columns. Source: `python/genvarloader/_dataset/_flat_variants.py`.
- `gvl.migrate(path)` — upgrade a pre-2.0 (array-of-structs) dataset to format 2.0 (struct-of-arrays) **in place**. Streaming, idempotent, crash-safe; converts `intervals/<track>/` and `annot_intervals/<track>/` interval storage and bumps `metadata.json`. A no-op (with leftover-AoS cleanup) on an already-2.0 dataset. Source: `python/genvarloader/_dataset/_migrate.py`. (Distinct from `gvl.migrate_svar_link`, which upgrades legacy SVAR symlink layouts.)
- `gvl.to_nested_tensor(ragged)` — convert to a PyTorch nested tensor (requires `torch`).
- `gvl.get_dummy_dataset()` — small in-memory dataset for examples/tests.
- `gvl.RefDataset` — reference-only dataset (no genotypes).
- `gvl.Table` — core interval-track source backed by a Rust COITrees overlap engine. Zero-based half-open coordinates; positive-width intervals assumed. Usable directly as a `tracks=` or `annot_tracks=` source in `gvl.write` and `gvl.update`. No extra install required — ships as part of the core package. CI-covered via a brute-force numpy oracle + property tests.
- `gvl.data_registry.fetch(name)` — download public test/demo datasets.

Full list lives in `python/genvarloader/__init__.py` `__all__`.

## On-disk layout (quick reference)

```
ds.gvl/
├── metadata.json              # version, samples, contigs, ploidy, max_jitter, svar_link?
├── input_regions.arrow        # BED + region index map
├── genotypes/                 # variant_idxs.npy, dosages.npy, variants.arrow
│                              # (absent when sourced from .svar; see svar_link)
│                              # svar2_ranges/ present iff sourced from .svar2 (see svar2_link)
├── intervals/<track>/         # per-sample track data (BigWigs / Table)
└── annot_intervals/<track>/   # sample-independent annotation track data
```

In **format 2.0**, each `intervals/<track>/` (and `annot_intervals/<track>/`) directory stores its intervals as **struct-of-arrays** — three contiguous files `starts.npy` (int32), `ends.npy` (int32), `values.npy` (float32), sharing one `offsets.npy` (int64) — replacing the format 1.x single `intervals.npy` record array. This lets the contiguous memmaps cross the Python→Rust boundary zero-copy. Upgrade a 1.x dataset with `gvl.migrate(path)` (see the format version gate above).

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
| `gvl.write` / `gvl.update` internals  | `python/genvarloader/_dataset/_write.py`               |
| Track re-alignment internals          | `python/genvarloader/_dataset/_tracks.py`, `_reconstruct.py` |
| Insertion fill internals              | `python/genvarloader/_dataset/_insertion_fill.py`      |
| SVAR back-reference / migration       | `python/genvarloader/_dataset/_svar_link.py`           |
| `.svar2` back-reference / read-bound wiring | `python/genvarloader/_dataset/_svar2_link.py`, `_svar2_haps.py` |
| Format 1.x → 2.0 migration internals  | `python/genvarloader/_dataset/_migrate.py`             |
| Flat-buffer ragged containers         | `python/genvarloader/_flat.py`                         |
| Flat variants + alleles types         | `python/genvarloader/_dataset/_flat_variants.py`       |
| Flank fetch + tokenization + windows  | `python/genvarloader/_dataset/_flat_flanks.py`         |

## Common gotchas

- **Pre-2.0 datasets must be migrated once before opening.** `Dataset.open` rejects any dataset written by genvarloader < 2.0 (or unversioned) with a `ValueError` pointing at `gvl.migrate(path)`. Run it once (in place, idempotent, crash-safe). A dataset written by a *newer* major raises a different `ValueError` asking you to upgrade genvarloader. Note `gvl.migrate` (format upgrade) is **not** the same as `gvl.migrate_svar_link` (SVAR symlink-layout upgrade).
- **`gvl.update` does not hot-reload open datasets.** A `Dataset` instance that was opened before `gvl.update` ran will not see the new track; reopen the dataset to pick it up. The update itself is safe to run while readers are active — each track is published atomically so a reader never sees a half-written track.
- **`Dataset.write_annot_tracks` has been removed.** Use `gvl.update(dataset, annot_tracks={"name": source})` instead, or pass `annot_tracks=` to `gvl.write` at creation time.
- **`gvl.Table` is a core public API.** No extra install required. It uses a Rust COITrees overlap engine and is CI-covered. Import it as `gvl.Table` (re-exported from the top-level package).
- **Symbolic / breakend variants are rejected, not skipped.** Remove them before `gvl.write` — e.g. `bcftools view -e 'ALT~"<" || ALT~"\["'` (drop SVs and breakends), or construct the genoray reader with `filter=genoray.exprs.is_biallelic & ~genoray.exprs.is_symbolic & ~genoray.exprs.is_breakend`. SVAR inputs must be built from an already-filtered source, since gvl validates but cannot re-filter a materialized `.svar`. For `.svar2` the same variants are rejected **upstream at `.svar2` conversion time** (genoray), not at `gvl.write` time — the store format cannot represent them at all.
- **`.svar2` has a Phase-1 unsupported-combination matrix.** Spliced output, `var_filter="exonic"`, `min_af`/`max_af`, `annotated` haplotypes, `VarWindowOpt(ref="allele")`, in-kernel `to_rc`, fixed-length haplotype-realigned tracks, `variants`/`variant-windows` output with jitter (`max_jitter>0` at write or `jitter>0` at read), and multi-contig `FlankSample` track fills all raise `NotImplementedError` on a `.svar2`-backed dataset instead of silently mis-computing. `with_seqs("variant-windows")` and `unphased_union` are now supported for `.svar2`. See "`.svar2` — the read-bound sparse variant format" above.
- **`.svar2` `variants`/`variant-windows` ALT bytes differ from `.svar` for pure deletions.** `.svar` keeps the VCF anchor base (`b"G"` for `GTA>G`); `.svar2` decodes the atomized empty ALT (`b""`). Reconstructed haplotypes are byte-identical either way; `ref_window` is also byte-identical — only raw ALT/`alt_window` bytes differ for pure-deletion records.
- Opening a genotypes-only dataset without a `reference=` defaults to the `"variants"` view (`RaggedVariants`), not `"haplotypes"`. Only `"variants"` is available without a reference; `with_seqs("haplotypes" | "annotated" | "reference")` raises `ValueError` if no reference was provided.
- `with_insertion_fill` raises unless the dataset has both haplotypes AND tracks active.
- `min_af` / `max_af` raise unless the dataset is SVAR-backed.
- `with_len(L)` requires `L + 2·jitter ≤ min(region_length) + 2·max_jitter` — set `max_jitter` accordingly at `write` time.
- Tracks must have unique `.name`; the on-disk layout is `intervals/<name>/`.
- BED `strand` of `.` is treated as `+`. Reverse-complement happens automatically when `rc_neg=True` (default) and `strand == "-"`.
- Splicing is a read-time setting on a *flat* BED of exons — do not pre-concatenate exons before `gvl.write`.
- `extend_to_length=False` at write time will produce haplotypes shorter than the BED region when deletions are present; downstream code must tolerate `<` region length.
- Missing a `dosage` field on a `RaggedVariants` output you expected? Check `var_fields` — `dosage` must be requested explicitly even if `dosages.npy` exists on disk.
- `FlatRagged` / `FlatVariants` offsets are **int64**. PyTorch nested tensors require int32 offsets — cast with `.astype(np.int32)` or `tensor.to(torch.int32)` before passing to `torch.nested.narrow`.
- `kind="intervals"` cannot be re-aligned: combining it with a variant-aware seq mode (`haplotypes`/`annotated`/`variants`/`variant-windows`) raises unless `with_settings(realign_tracks=False)`. (Breaking change: `haplotypes`+`intervals` previously returned un-realigned intervals silently under the default.)
- `with_insertion_fill` raises when `realign_tracks=False`.
- `with_seqs("variant-windows")` requires a `VarWindowOpt` second argument, `with_output_format("flat")`, and a reference genome. Querying in `"ragged"` mode raises. It does **not** inherit flank settings from `with_settings`.
- **Flat ride-along flank tokens need tracks disabled.** `FlatVariants.flank_tokens` (ride-along on `"variants"` output) is only produced on the pure flat variants channel — with active tracks the variants output falls back to ragged `RaggedVariants` (no `flank_tokens`). Call `with_tracks(False)` for the ride-along. `"variant-windows"` may be combined with tracks via `with_settings(realign_tracks=False)` (tracks are returned reference-coordinate alongside the windows output).
- Flank tokens (`FlatVariants.flank_tokens`) and variant windows (`FlatVariantWindows`) are **reference-oriented** — they are NOT reverse-complemented even when `rc_neg=True`. Only the alt/ref allele fields of `FlatVariants` / `RaggedVariants` are RC'd (not their scalar fields, not flank tokens, not windows).
- Token dtype is `uint8` when max token id ≤ 255, else `int32`; offsets are `int64`.
- `VarWindowOpt(ref="allele")` (bare allele mode) requires REF alleles on disk. `flank_length` ride-along on `FlatVariants` requires `token_alphabet` and `unknown_token` to be set together in the same or a prior `with_settings` call.
- `dummy_variant` padding applies to **both `"variants"` and `"variant-windows"`** outputs. Setting `dummy_variant=<DummyVariant>` and then indexing with any other kind (`"haplotypes"`, `"annotated"`, `"reference"`, or no seqs) raises `ValueError`. For token fields (`flank_tokens`, `ref_window`/`alt_window`, bare `ref`/`alt`), the dummy fill is all-`unknown_token` — the `DummyVariant.ref`/`.alt` bytes only set the dummy allele's byte-length, not the token value. `dummy_variant=False` with an unsupported output kind is silently ignored.
- A non-`b"N"` `DummyVariant.alt` (or `.ref`) **is reverse-complemented** on negative-strand regions, exactly like a real variant allele. The default `b"N"` is rc-invariant; use it if you want a strand-neutral sentinel.
- `unphased_union=True` + `with_seqs("haplotypes")` / `with_seqs("annotated")` raises — `unphased_union` only applies to `"variants"` / `"variant-windows"` output.

## Maintaining this skill

Whenever a PR changes the **public API** (anything in `python/genvarloader/__init__.py` `__all__`, or the docstring/signature of `gvl.write`, `Dataset.open`, or any `Dataset.with_*` method), the author must also update this `SKILL.md`. New public symbols, removed symbols, renamed args, changed defaults, and new output modes are all in scope. CLAUDE.md enforces this as part of the contribution checklist.
