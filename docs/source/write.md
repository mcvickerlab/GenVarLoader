# How to write a `Dataset`

Depending on your needs, you'll potentially need to normalize variants and prepare a mapping from sample names to BigWig files before you can write a [`Dataset`](api.md#genvarloader.Dataset).

## Preparing input data

### Normalizing variants

Before passing variants to GenVarLoader, they must be:
- left-aligned
- bi-allelic
- atomic (no MNPs or compound MNP-indels)
- for VCFs, indexed with `bcftools index` or `tabix`

In general, any VCF can be preprocessed to meet these requirements using [`bcftools norm`](https://samtools.github.io/bcftools/bcftools.html#norm), for example:

```bash
bcftools norm -f $reference \
    -a --atom-overlaps . \
    -m -any --multi-overlaps . \
    -O b -o $out $in
```

Alternatively, if your data is already in the PGEN format you can use PLINK 2.0:

```bash
plink2 --make-bpgen --pfile $input --out $intermediate
plink2 --make-pgen --normalize --ref-from-fa --fa reference.fa --bpfile $intermediate --out $normalized
```

See the PLINK 2.0 documentation for details.

### BigWig table

To process BigWig data, GenVarLoader needs a mapping from sample names to BigWig file paths. This can be provided either as a dictionary using [`BigWig()`](api.md#genvarloader.BigWigs.__init__) or a table using [`BigWig.from_table()`](api.md#genvarloader.BigWigs.from_table). If using a table, it must at least contain the columns `sample` and `path`, for example:

| sample |                    path |
|-------:|------------------------:|
|   Aang |   /data/bigwigs/aang.bw |
| Katara | /data/bigwigs/katara.bw |
|  Sokka |  /data/bigwigs/sokka.bw |

### Regions of interest (ROIs)

Whether you're working with variants, BigWigs, or both, you will need regions of interest specified in a BED-like table. This means your table must have the columns `chrom`, `chromStart`, and `chromEnd` as 0-based coordinates. You can also include a `strand` column consisting of `+`, `-`, and `.` (treated as `+`) which will let you [choose to reverse and/or reverse complement](api.md#genvarloader.Dataset.with_settings) data on negative strands. Any extra columns will be kept in the table and exist in [`Dataset.regions`](api.md#genvarloader.Dataset.regions) after writing. An example of a BED-like table:

| split | chrom | chromStart | chromEnd | strand |
|------:|------:|-----------:|---------:|-------:|
| train |  chr1 |          0 |        5 |      - |
|   val |  chr2 |          3 |        8 |      + |
|  test |  chr3 |         20 |       21 |      . |

## Writing data

Once your data is prepared, you can use [`gvl.write()`](api.md#genvarloader.write) to convert the data for regions of interest to a format that `gvl.Dataset` can open. You can include a single source of variants and/or any number of BigWig files. Some examples:

```python
import genvarloader as gvl

gvl.write(
    path="1000_genomes_haplotypes.gvl",
    bed="tiling_windows.bed",
    variants="all_chroms.bcf",
    # OR variants='all_chroms.pgen',
)
```

This dataset would have haplotypes available for all samples in `all_chroms.bcf`.

**Allele-frequency caching for VCF/BCF sources.** When `variants=` is a VCF/BCF whose header
declares an `INFO/AF` field, `gvl.write()` now caches an `AF` column into the written index at
index-build time. This is what lets a written `Dataset` open the VCF-sourced dataset and
AF-filter it (`Dataset.with_settings(min_af=, max_af=)`) the same way a `.svar` store does after
`SparseVar.cache_afs()` — see [dataset.md](dataset.md) (the `StreamingDataset` section) and the
[FAQ](faq.md) for the guard and per-source rules. VCFs with no `INFO/AF` field are unaffected
(no `AF` column is written, and AF filtering raises the same guard it always did). The cache is
only attached when `gvl.write()` builds a fresh `.gvi` index — if a *valid* `.gvi` index already
exists on disk without an `AF` column (e.g. built by an older gvl or by genoray directly), it is
not rewritten, so AF filtering keeps raising until the index is rebuilt. AF caching also needs a
single AF value per record: if a record's `INFO/AF` carries more than one value — an ambiguous
ALT→AF mapping, e.g. a `bcftools norm -m` split that left a `Number=.` `AF` un-subset so a
bi-allelic `G>A` record still lists `AF=0.333,0.667` — `gvl.write()` cannot tell which value the
kept ALT maps to, so it logs a warning and does **not** cache `AF` (the write still succeeds; AF
filtering then raises the usual guard). Normalize so each record carries a single per-ALT `AF`
(`bcftools norm -m -any`) to enable AF filtering.

```python
gvl.write(
    path="1000_genomes_lncRNA.gvl",
    bed="lncRNA.bed",  # can be varying length regions
    variants="all_chroms.bcf",
    tracks=[
        gvl.BigWigs.from_table("pos", "pos_strands.tsv"),
        gvl.BigWigs.from_table("neg", "pos_strands.tsv"),
    ],
)
```

This dataset would have both haplotypes and two tracks (`pos` and `neg`) available for samples that exist in both `all_chroms.bcf` and the BigWig tables (i.e. `gvl.write()` performs an inner join on samples).

## Variants from a genoray sparse store (`.svar` / `.svar2`)

Besides BCF/VCF and PGEN, `variants=` also accepts a genoray sparse columnar variant store — either the original `.svar` format or the newer `.svar2` format. Build one from a normalized VCF/BCF with `genoray`:

```python
from genoray import VCF, SparseVar, SparseVar2

# .svar (SVAR1): a VCF reader + a memory budget
SparseVar.from_vcf("all_chroms.svar", VCF("normed.bcf"), max_mem="4g")

# .svar2 (SVAR2): the VCF/BCF path + a reference FASTA (or no_reference=True)
SparseVar2.from_vcf("all_chroms.svar2", "normed.bcf", reference="ref.fa")
```

Then pass the resulting store to `gvl.write`:

```python
gvl.write(
    path="1000_genomes_haplotypes.gvl",
    bed="tiling_windows.bed",
    variants="all_chroms.svar2",  # or "all_chroms.svar", or a SparseVar/SparseVar2 instance
)
```

Both formats store a back-reference in the dataset's `metadata.json` instead of duplicating per-variant arrays, so the source store must remain accessible when the dataset is later opened with [`gvl.Dataset.open()`](api.md#genvarloader.Dataset.open) (override its location with `svar=`/`svar2=` if it has moved).

`.svar2` additionally produces a small write-time cache under `<path>/genotypes/svar2_ranges/` and reads back through an all-Rust, read-bound path with no interval-search-tree build and no dense-union rebuild per read — see [the FAQ](faq.md) for the read-path and on-disk-size tradeoffs, and [the format reference](format.md) for the on-disk layout. `.svar2` currently has a Phase-1 scope: a handful of output combinations (`annotated` haplotypes, `min_af`/`max_af`, splicing with variant/track outputs, etc.) aren't wired yet and raise `NotImplementedError` — see the `genvarloader` skill or `format.md` for the full list. Plain haplotype output supports splicing and `var_filter="exonic"`.
