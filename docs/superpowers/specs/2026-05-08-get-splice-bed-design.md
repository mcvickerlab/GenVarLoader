# `get_splice_bed`: GTF → splicing-ready BED helper

**Date:** 2026-05-08
**Branch:** `worktree-feat-gtf-opts`
**Status:** Design approved

## Motivation

Users who want to train splicing models with GenVarLoader currently hand-roll a polars pipeline to convert an Ensembl-style GTF into a BED suitable for `gvl.write()` and downstream `Dataset.with_settings(splice_info=("transcript_id", "exon_number"))`. The pipeline is mechanical (CDS rows, 1→0-based start, attribute extraction, sort) and easy to get subtly wrong. We want a single function that produces the BED, with the user free to apply any further transforms before passing it to `gvl.write()`.

Reference pipeline: <https://gist.github.com/d-laub/06cce214eb378cefec9468ddba47d1ab> (cell 5 in particular).

## API

```python
def get_splice_bed(
    gtf: str | Path,
    contigs: list[str] | None = None,
    transcript_support_level: str | None = "1",
    require_multiple_of_3: bool = True,
) -> pl.DataFrame:
    """Process a GTF into a BED-compatible DataFrame for splicing datasets."""
```

**Parameters**

- `gtf`: path to a GTF file (gzipped or plain), as accepted by `seqpro.gtf.scan`.
- `contigs`: if provided, keep only rows whose `seqname` is in this list. `None` (default) applies no contig filter.
- `transcript_support_level`: if a string (default `"1"`), require the GTF `transcript_support_level` attribute to equal it. `None` disables the filter.
- `require_multiple_of_3`: if `True` (default), keep only transcripts whose summed CDS length is a multiple of 3.

**Returns** — a `pl.DataFrame` with columns:

- `chrom` (Utf8)
- `chromStart` (Int64, 0-based)
- `chromEnd` (Int64, half-open, unchanged from GTF)
- `strand` (Utf8, `+` or `-`)
- `gene_name` (Utf8, may contain nulls)
- `transcript_id` (Utf8)
- `exon_number` (Int32)

Sorted by `sp.bed.sort` (chrom natural order, then chromStart).

The result is directly compatible with `gvl.write(path, bed=...)` and with `Dataset.with_settings(splice_info=("transcript_id", "exon_number"))`.

## Pipeline

Implementation lives in `python/genvarloader/_dataset/_write.py`, alongside `write()`:

1. `lf = sp.gtf.scan(gtf)`
2. If `contigs is not None`: `lf = lf.filter(pl.col("seqname").is_in(contigs))`
3. `lf = lf.filter(pl.col("feature") == "CDS")`
4. `lf = lf.rename({"seqname": "chrom", "start": "chromStart", "end": "chromEnd"})`
5. `lf = lf.with_columns(pl.col("chromStart") - 1, sp.gtf.attr("gene_name"), sp.gtf.attr("transcript_id"), sp.gtf.attr("exon_number").cast(pl.Int32))`
6. Build filter predicates conditionally:
   - If `require_multiple_of_3`: add `transcript_len = (chromEnd - chromStart).sum().over("transcript_id")` then filter `transcript_len % 3 == 0`.
   - If `transcript_support_level is not None`: filter `sp.gtf.attr("transcript_support_level") == transcript_support_level`.
7. Drop `source`, `score`, `frame`, `feature`, `attribute` (and `transcript_len` if it was added).
8. `df = lf.collect()`
9. `df = sp.bed.sort(df)` and return.

No `gene_name` non-null filter is applied; `gene_name` may contain nulls in the output.

## Export

Add `get_splice_bed` to `python/genvarloader/__init__.py` so it is reachable as `gvl.get_splice_bed`.

## Tests

New file `tests/dataset/test_get_splice_bed.py`. Build a small synthetic GTF (string written to `tmp_path`) covering:

- Two transcripts on different contigs, each with 2–3 CDS rows.
- One transcript whose summed CDS length is **not** a multiple of 3.
- One CDS row with `transcript_support_level "2"` and one with `"1"`.
- One CDS row with no `gene_name` attribute (to verify nulls survive).
- A non-CDS row (e.g. `exon`, `five_prime_utr`) to verify it is dropped.
- An out-of-order row to verify sorting.

Assertions:

- Output columns equal the documented schema, with the documented dtypes.
- All `chromStart` values are 0-based (one less than the corresponding GTF `start`).
- No non-CDS rows survive.
- `require_multiple_of_3=True` removes the bad transcript; `require_multiple_of_3=False` keeps it.
- `transcript_support_level="1"` keeps only TSL-1 rows; `transcript_support_level=None` keeps all.
- `contigs=["1"]` filters to that contig.
- Output is sorted (chrom natural order, then chromStart ascending).
- A row with no `gene_name` has `gene_name == None` and is retained.

## Out of scope

- No GTF feature other than `CDS` (no `exon`, `five_prime_utr`, `three_prime_utr`).
- No interval merging (the splicing path keeps individual CDS rows; merging is for non-splicing exon BEDs).
- No writing the intermediate BED to disk; the function returns a DataFrame.
- No integration into `gvl.write()` itself — users compose `gvl.write(path, bed=get_splice_bed(...), ...)`.
