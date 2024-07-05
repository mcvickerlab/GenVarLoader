```{toctree}
:hidden: true

api
```

# GenVarLoader

```{image} _static/gvl_logo.png
:alt: GenVarLoader logo
:align: center
:width: 200
```

```{image} https://badge.fury.io/py/genvarloader.svg
:alt: PyPI version
:target: https://badge.fury.io/py/genvarloader
:class: inline-link
```

```{image} https://readthedocs.org/projects/genvarloader/badge/?version=latest
:alt: Documentation Status
:target: https://genvarloader.readthedocs.io/en/latest/index.html
:class: inline-link
```

```{image} https://img.shields.io/pypi/dm/genvarloader
:alt: PyPI - Downloads
:class: inline-link
```


GenVarLoader provides a fast, memory efficient data loader for training sequence models on genetic variation. For example, this can be used to train a DNA language model on human genetic variation (e.g. [Nucleotide Transformer](https://www.biorxiv.org/content/10.1101/2023.01.11.523679)) or train sequence to function models with genetic variation (e.g. [BigRNA](https://www.biorxiv.org/content/10.1101/2023.09.20.558508v1)).

## Features
- Avoids writing any sequences to disk
- Generates haplotypes up to 1,000 times faster than reading a FASTA file
- Generates tracks up to 450 times faster than reading a BigWig
- Supports indels and re-aligns tracks to haplotypes that have them
- Extensible to new file formats: drop a feature request! Currently supports VCF, PGEN, and BigWig

## Tutorial

### Installation

```bash
pip install genvarloader
```

A PyTorch dependency is not included since it may require [special instructions](https://pytorch.org/get-started/locally/).

### Write a [`gvl.Dataset`](#genvarloader.Dataset)

GenVarLoader has both a CLI and Python API for writing datasets. The Python API provides some extra flexibility, for example for a multi-task objective.

```bash
genvarloader cool_dataset.gvl interesting_regions.bed --variants cool_variants.vcf --bigwig-table samples_to_bigwigs.csv --length 2048 --max-jitter 128
```

Where `samples_to_bigwigs.csv` has columns `sample` and `path` mapping each sample to its BigWig.

This could equivalently be done in Python as:

```python
import genvarloader as gvl

gvl.write(
    path="cool_dataset.gvl",
    bed="interesting_regions.bed",
    variants="cool_variants.vcf",
    bigwigs=gvl.BigWigs.from_table("bigwig", "samples_to_bigwigs.csv"),
    length=2048,
    max_jitter=128,
)
```

### Open a [`gvl.Dataset`](#genvarloader.Dataset) and get a PyTorch DataLoader

```python
import genvarloader as gvl

dataset = gvl.Dataset.open(path="cool_dataset.gvl", reference="hg38.fa")
train_samples = ["David", "Aaron"]
train_dataset = dataset.subset_to(regions="train_regions.bed", samples=train_samples)
train_dataloader = train_dataset.to_dataloader(batch_size=32, shuffle=True, num_workers=1)

# use it in your training loop
for haplotypes, tracks in train_dataloader:
    ...
```

### Inspect specific instances

```python
dataset[99]  # 100-th instance of the raveled dataset
dataset[0, 9]  # first region, 10th sample
dataset.isel(regions=0, samples=9)
dataset.sel(regions=dataset.get_bed()[0], samples=dataset.samples[9])
dataset[:10]  # first 10 instances
dataset[:10, :5]  # first 10 regions and 5 samples
```

### Transform the data on-the-fly

```python
import seqpro as sp
from einops import rearrange

def transform(haplotypes, tracks):
    ohe = sp.DNA.ohe(haplotypes)
    ohe = rearrange(ohe, "batch length alphabet -> batch alphabet length")
    return ohe, tracks

transformed_dataset = dataset.with_settings(transform=transform)
```

### Pre-computing transformed tracks

Suppose we want to return tracks that are the z-scored, log(CPM + 1) version of the original. Sometimes it is better to write this to disk to avoid having to recompute it during training or inference.

```python
import numpy as np

# We'll assume we already have an array of total counts for each sample.
# This usually can't be derived from a gvl.Dataset since it only has data for specific regions.
total_counts = np.load('total_counts.npy')  # shape: (samples) float32

# We'll compute the mean and std log(CPM + 1) using the training split
means = np.empty((train_dataset.n_regions, train_dataset.region_length), np.float32)
stds = np.empty_like(means)
just_tracks = train_dataset.with_settings(return_sequences=False, jitter=0)
for region in range(len(means)):
    cpm = np.log1p(just_tracks[region, :] / total_counts[:, None] * 1e6)
    means[region] = cpm.mean(0)
    stds[region] = cpm.std(0)

# Define our transformation
def z_log_cpm(dataset_indices, region_indices, sample_indices, tracks: gvl.Ragged[np.float32]):
    # In the event that the dataset only has SNPs, the full length tracks will all be the same length.
    # So, we can reshape the ragged data into a regular array.
    _tracks = tracks.data.reshape(-1, dataset.region_length)
    
    # Otherwise, we would have to leave `tracks`as a gvl.Ragged array to accommodate different lengths.
    # In that case, we could do the transformation with a Numba compiled function instead.

    # original tracks -> log(CPM + 1) -> z-score
    _tracks = np.log1p(_tracks / total_counts[sample_indices, None] * 1e6)
    _tracks = (_tracks - means[region_indices]) / stds[region_indices]

    return gvl.Ragged.from_offsets(_tracks.ravel(), tracks.shape, tracks.offsets)

# This can take about as long as writing the original tracks or longer, depending on the transformation.
dataset_with_zlogcpm = dataset.write_transformed_track("z-log-cpm", "bigwig", transform=z_log_cpm)

# The dataset now has both tracks available, "bigwig" and "z-log-cpm", and we can choose to return either one or both.
haps_and_zlogcpm = dataset_with_zlogcpm.with_settings(return_tracks="z-log-cpm")

# If we re-opened the dataset after running this then we could write...
dataset = gvl.Dataset.open("cool_dataset.gvl", "hg38.fa", return_tracks="z-log-cpm")
```

## Performance tips
- GenVarLoader uses multithreading extensively, so it's best to use 0 or 1 workers with your PyTorch `DataLoader`.
- A GenVarLoader [`Dataset`](#genvarloader.Dataset) is most efficient when given batches of indices, rather than one at a time. PyTorch `DataLoader` by default uses one index at a time, so if you want to use a ***custom*** PyTorch `Sampler` you should wrap it with a PyTorch `BatchSampler` before passing it to `Dataset.to_dataloader()`.

# Indices and tables

[Index](#genindex)

[Module Index](#modindex)

[Search Page](#search)
