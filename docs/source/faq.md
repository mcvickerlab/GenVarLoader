# FAQ

## Why does a Dataset return "Ragged" objects and what are they?

For why, see ["What's a `gvl.Dataset`?"](dataset.md). [`Ragged`](api.md#genvarloader.Ragged) arrays are similar to NumPy arrays except that the final axis is a variable size. For example, a 2D ragged array might look like:

:::{image} _static/ragged.svg
:alt: A 2D ragged array with 3 rows.
:align: center
:width: 150
:::

To store this, a [`Ragged`](api.md#genvarloader.Ragged) array minimally consists of two NumPy arrays: a 1D array `data` with shape `(size)` containing the values, and another 1- or 2-D array `offsets` with shape `(n_rows+1)` or `(2, n_rows)`, respectively, specifying the start and end position (exclusive) of every row's data in the `data` array. We could thus create the above example:

```python
data = np.array([1, 2, 3, 4, 5, 6])
offsets = np.array([0, 2, 3, 6])
shape = (3,)
ragged = gvl.Ragged.from_offsets(data, shape, offsets)
# [
#     [1, 2],
#     [3],
#     [4, 5, 6]
# ]
```

Ragged arrays are subclasses of [Awkward Arrays](https://github.com/scikit-hep/awkward), so anything you can do with Awkward Arrays you can do with Ragged arrays. Within GVL, we use numba JIT'd functions to compute on the ragged objects' buffers directly since it's relatively straightforward (i.e. iterating over the rows of `data` via the `offsets` array).

.. note::

    GVL Datasets can also return several other kinds of objects, see the [API reference](api.md#containers) for more details.

## I have multiple tracks per sample, how can I add them?

If you provide multiple BigWigs to [`gvl.write()`](api.md#genvarloader.write), all of them can be returned simultaneously from the resulting [`Dataset`](api.md#genvarloader.Dataset) and placed along the track axis, sorted by name. By default, a Dataset sets all tracks to active when opened. i.e. tracks have shape `(batch, tracks, [ploidy], length)`. For example:

```python
import genvarloader as gvl

pos_strand = gvl.BigWigs.from_table("pos", "pos_strand.tsv")
neg_strand = gvl.BigWigs.from_table("neg", "neg_strand.tsv")
gvl.write("path/to/dataset.gvl", bed="path/to/regions.bed", bigwigs=[pos_strand, neg_strand])
```

## How does GVL handle negative stranded regions provided to [`gvl.write()`](api.md#genvarloader.write)?

By default, GVL will automatically reverse (and complement) negative stranded regions. You can modify this behavior using
[`gvl.Dataset.with_settings()`](api.md#genvarloader.Dataset.with_settings) and setting `rc_neg` to False.

## How does GVL handle unphased genotypes?

GVL assumes all genotypes are phased and will not warn you if any genotypes are unphased. Generally, unphased
genotypes cannot be resolved into haplotypes so we make this simplifying assumption. If you aren't sure whether your genotypes are phased, it is relatively easy to inspect from the CLI using [bcftools view](https://samtools.github.io/bcftools/bcftools.html#view) or [plink2](https://www.cog-genomics.org/plink/2.0/basic_stats#pgen_info):

```bash
# for VCF, -p filters for records where all samples are phased
bcftools view -Hp $vcf | wc -l
# returns number of phased records

# for PLINK
plink2 --pgen-info $prefix
```

## How can I get personalized protein/spliced RNA sequences?

This is not yet supported but on GVL's roadmap for the near future. Keep an eye out in future releases!

<!-- Example of variable length regions

Example of spliced gvl.write() and enabling splicing

Example of SeqPro translate for RNA and AA -->

## Why aren't the methods `with_len()`, `with_seqs()`, etc. combined into `with_settings()`?

These methods modify the type of output returned by a `gvl.Dataset`. In order to allow type checkers like mypy and pyright to know how these settings modify state, they are given their own methods. As a result, if you use a type checker, you will have access to an improved developer workflow with compile-time errors for many common issues. For example, using an incompatible transform or unpacking return values into the wrong number of arguments.