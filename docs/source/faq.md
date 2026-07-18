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

Ragged arrays are backed by [`seqpro`](https://github.com/ML4GLand/SeqPro)'s `Ragged` type (a Rust-backed `_core.Ragged`). GVL computes on the `data` and `offsets` buffers directly in Rust, which is relatively straightforward (i.e. iterating over the rows of `data` via the `offsets` array). (Earlier releases subclassed [Awkward Arrays](https://github.com/scikit-hep/awkward); GVL no longer depends on `awkward`.)

.. note::

    GVL Datasets can also return several other kinds of objects, see the [API reference](api.md#containers) for more details.

## I have multiple tracks per sample, how can I add them?

If you provide multiple BigWigs to [`gvl.write()`](api.md#genvarloader.write), all of them can be returned simultaneously from the resulting [`Dataset`](api.md#genvarloader.Dataset) and placed along the track axis, sorted by name. By default, a Dataset sets all tracks to active when opened. i.e. tracks have shape `(batch, tracks, [ploidy], length)`. For example:

```python
import genvarloader as gvl

pos_strand = gvl.BigWigs.from_table("pos", "pos_strand.tsv")
neg_strand = gvl.BigWigs.from_table("neg", "neg_strand.tsv")
gvl.write(
    "path/to/dataset.gvl", bed="path/to/regions.bed", tracks=[pos_strand, neg_strand]
)
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

## How do I control how many threads GVL uses?

GVL's read path (haplotype reconstruction and track re-alignment) is parallelized in Rust with [rayon](https://github.com/rayon-rs/rayon). By default it uses one worker per available CPU, detected from the Linux cgroup cpuset (`sched_getaffinity`) so it respects container limits, and falling back to `os.cpu_count()` elsewhere. Three environment variables tune this:

- **`GVL_NUM_THREADS`** — set the worker count explicitly (e.g. `GVL_NUM_THREADS=4`). Overrides cgroup detection. Resolved once, on first use, so set it before your first GVL call.
- **`GVL_FORCE_PARALLEL`** — set to a truthy value (`1`, `true`, `yes`, `on`) to force the multithreaded paths even on small inputs. By default GVL runs small inputs serially because thread overhead would dominate; this bypasses that size gate. Mainly useful for benchmarking.
- **`RAYON_NUM_THREADS`** — GVL **overwrites** this with its own resolved count so an inherited value (e.g. baked into a base image) can't defeat the cgroup-aware cap. To size the pool yourself, use `GVL_NUM_THREADS` instead.

## Should I use `.svar` or `.svar2` as my variant source?

Both are sparse columnar variant archives from [`genoray`](https://github.com/mcvickerlab/genoray) that `gvl.write(variants=...)` accepts alongside BCF/PGEN; see [write.md](write.md) for how to build one. The two differ in their read-time behavior:

- **`.svar`** reconstructs by building an interval search tree over the queried window and a per-read dense union of the overlapping variants.
- **`.svar2`** reconstructs via a **read-bound** path: `gvl.write` caches small per-`(region, sample, ploid)` variant-key ranges at write time, and `Dataset.__getitem__` gathers directly off that cache and calls all-Rust kernels — it builds **no interval search tree and no dense union per read**. `.svar2` stores are also typically smaller on disk than `.svar`, especially for large cohorts.

`.svar2` is Phase-1 scope: a handful of combinations (`annotated` haplotypes, `min_af`/`max_af`, `VarWindowOpt(ref="allele")`, fixed-length haplotype-realigned tracks, splicing or exonic filtering with non-haplotype outputs, and `variants`/`variant-windows` output with jitter) aren't wired yet and raise `NotImplementedError` rather than silently mis-computing. Plain haplotype output supports splicing, `var_filter="exonic"`, and negative-strand reverse-complementation. `"variant-windows"` output, `unphased_union` (for both `"variants"` and `"variant-windows"`), and `var_fields`-selected store INFO/FORMAT fields (also for both, when the `.svar2` was written with them) are also supported. See the `genvarloader` skill's `.svar2` section or `docs/source/format.md` for the full list. Everything else — haplotypes, tracks, and variants/variant-windows at any supported jitter/output-length combination — is byte-identical between the two backends.

One documented difference in raw output: for a pure deletion, `with_seqs("variants")` on a `.svar` dataset reports the VCF anchor base as ALT (e.g. `b"G"` for `GTA>G`), while a `.svar2` dataset reports the atomized empty ALT (`b""`) — a genoray `.svar2` format convention, not a bug. Reconstructed haplotypes are unaffected; only `RaggedVariants.alt` differs (and `FlatVariantWindows.alt`/`.alt_window` for `"variant-windows"`), and only for pure-deletion records. `ref_window` is byte-identical between the two backends.

## How can I get personalized protein/spliced RNA sequences?

Write a dataset from an exon-level BED containing transcript and exon-order columns,
then open it with `splice_info` and haplotype output. Use `var_filter="exonic"` to
drop variants whose reference span crosses an exon boundary:

```python
ds = gvl.Dataset.open(
    "transcripts.gvl",
    reference="ref.fa",
    splice_info=("transcript_id", "exon_number"),
    var_filter="exonic",
).with_seqs("haplotypes")
```

This works with `.svar` and `.svar2` variant sources. Negative-strand transcripts
are reverse-complemented automatically when the BED includes `strand="-"`. See
the [splicing guide](splicing.html) for BED construction and output shapes.

<!-- Example of variable length regions

Example of spliced gvl.write() and enabling splicing

Example of SeqPro translate for RNA and AA -->

## Can I use gvl without writing a dataset first?

Yes, for haplotype output: [`gvl.StreamingDataset`](api.md#genvarloader.StreamingDataset) reconstructs haplotype batches directly from a live variant source, with no [`gvl.write()`](api.md#genvarloader.write) call and no `.gvl` directory written to disk. `variants=` accepts either a `.svar` (SparseVar/SVAR1) store or a VCF/BCF path (`.vcf`, `.vcf.gz`/`.vcf.bgz`, `.bcf`; indexed, same preprocessing requirements as `gvl.write` — see [write.md](write.md)) — both go through a shared Rust engine (`RecordStreamEngine`); VCF/BCF is decoded window-by-window via genoray's Rust `VcfRecordSource → ChunkAssembler → DenseChunk` pipeline. For the VCF/BCF path specifically, atomization (biallelic split) is applied automatically on read, but left-alignment and REF/check-ref are **not** performed and **not** validated — un-normalized input is silently accepted (no error) and will break byte-identical parity, so pre-normalizing with `bcftools norm` as `gvl.write` expects is on you. The `.svar` path, by contrast, is built from a store that was already validated at build time.

```python
sds = gvl.StreamingDataset(
    "rois.bed", reference="ref.fa", variants="normed.svar", max_mem="512MB"
).with_seqs("haplotypes")
# or: variants="normed.vcf.gz" / "normed.bcf" -- reads VCF/BCF directly, no .svar needed

for data, region_idxs, sample_idxs in sds.to_iter(batch_size=32):
    ...
```

Haplotype output is byte-identical to writing a dataset and indexing it (`gvl.write(...)` + `Dataset.open(...)[r, s]`, at `jitter=0`) — you're trading the write step for a slower per-epoch read, since `StreamingDataset` re-reads the live source on every window instead of hitting a pre-indexed on-disk dataset. It's a good fit for one-shot inference or when you can't afford (or don't want) the `gvl.write()` step; for repeated-epoch training, a written [`Dataset`](api.md#genvarloader.Dataset) is still faster.

**Streaming VCF/BCF requires htslib**, statically linked into gvl's wheel (genoray's `conversion` feature); no separate install step, but it means htslib is now a hard runtime requirement of the package, not just an internal detail of `gvl.write`'s own VCF ingestion.

`StreamingDataset` is currently narrower than `Dataset`: `.svar` and VCF/BCF variant sources only (PGEN and `.svar2` raise `NotImplementedError`; PGEN is landing on the same branch/PR shortly), `with_seqs("haplotypes")` only, `jitter=0` only, ragged output only, and it's **iterable-only** — `sds[r, s]` raises `TypeError` because iteration order is fixed by the data layout. `sds.to_iter(...)` is the one entry point; `to_torch_dataset()` and `to_dataloader()` are thin wrappers over it. Iteration is region-major, read one window at a time so each read stays within one contig; `to_iter(..., return_indices=True)` (the default) rides along `(region_idxs, sample_idxs)` in your original BED-row order. `sample_idxs` index into `sds.samples` (lexicographically-sorted sample names, matching `gvl.write()`), not the variant source's native column order. See the `genvarloader` skill for the full scope.

**Peak memory does not scale with cohort size.** The `max_mem` constructor argument (default `"512MB"`; an int byte count or a size string like `"1g"`/`"2GiB"`) bounds the read window's offsets buffer, which `StreamingDataset` chunks over regions and samples so it stays within budget regardless of how many samples the dataset has. Generation is separately bounded by `to_iter`'s `batch_size`: each window is read once, then reconstructed one `batch_size` slice at a time, so haplotype output is never materialized for a whole window at once. Peak memory is therefore `max_mem` (offsets) + `batch_size` (output) — neither term grows with cohort size.

## Why aren't the methods `with_len()`, `with_seqs()`, etc. combined into `with_settings()`?

These methods modify the type of output returned by a `gvl.Dataset`. In order to allow type checkers like mypy and pyright to know how these settings modify state, they are given their own methods. As a result, if you use a type checker, you will have access to an improved developer workflow with compile-time errors for many common issues. For example, using an incompatible transform or unpacking return values into the wrong number of arguments.
