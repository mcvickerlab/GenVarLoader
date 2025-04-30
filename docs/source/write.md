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
    path='1000_genomes_haplotypes.gvl',
    bed='tiling_windows.bed',
    variants='all_chroms.bcf',
    # OR variants='all_chroms.pgen',
)
```

This dataset would have haplotypes available for all samples in `all_chroms.bcf`.

```python
gvl.write(
    path='1000_genomes_lncRNA.gvl',
    bed='lncRNA.bed',  # can be varying length regions
    variants='all_chroms.bcf',
    bigwigs=[
        gvl.BigWigs.from_table('pos', 'pos_strands.tsv'),
        gvl.BigWigs.from_table('neg', 'pos_strands.tsv'),
    ]
)
```

This dataset would have both haplotypes and two tracks (`pos` and `neg`) available for samples that exist in both `all_chroms.bcf` and the BigWig tables (i.e. `gvl.write()` performs an inner join on samples).