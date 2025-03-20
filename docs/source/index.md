```{toctree}
:hidden: true

dataset
write
geuvadis
faq
api
```

# GenVarLoader

```{image} https://badge.fury.io/py/genvarloader.svg
:alt: PyPI version
:target: https://badge.fury.io/py/genvarloader
```

```{image} https://readthedocs.org/projects/genvarloader/badge/?version=latest
:alt: Documentation Status
:target: https://genvarloader.readthedocs.io/en/latest/index.html
```

```{image} https://static.pepy.tech/badge/genvarloader
:alt: Downloads
```

```{image} https://img.shields.io/pypi/dm/genvarloader
:alt: PyPI - Downloads
```

```{image} https://badgen.net/github/stars/mcvickerlab/GenVarLoader
:alt: GitHub stars
```

```{image} https://img.shields.io/badge/bioRxiv-2025.01.15.633240-b31b1b.svg
:alt: bioRxiv link
:target: https://www.biorxiv.org/content/10.1101/2025.01.15.633240
```

## Features

GenVarLoader provides a fast, memory efficient data structure for training sequence models on genetic variation. For example, this can be used to train a DNA language model on human genetic variation (e.g. [Dalla-Torre et al.](https://www.biorxiv.org/content/10.1101/2023.01.11.523679)) or train sequence to function models with genetic variation (e.g. [Celaj et al.](https://www.biorxiv.org/content/10.1101/2023.09.20.558508v1), [Drusinsky et al.](https://www.biorxiv.org/content/10.1101/2024.07.27.605449v1), [He et al.](https://www.biorxiv.org/content/10.1101/2024.10.15.618510v1), and [Rastogi et al.](https://www.biorxiv.org/content/10.1101/2024.09.23.614632v1)).

- Avoid writing any sequences to disk (can save >2,000x storage vs. writing personalized genomes with bcftools consensus)
- Generate haplotypes up to 1,000 times faster than reading a FASTA file
- Generate tracks up to 450 times faster than reading a BigWig
- **Supports indels** and re-aligns tracks to haplotypes that have them
- Extensible to new file formats: drop a feature request! Currently supports VCF, PGEN, and BigWig

See our [preprint](https://www.biorxiv.org/content/10.1101/2025.01.15.633240) for benchmarking and implementation details.

## Installation

```bash
pip install genvarloader
```

A PyTorch dependency is **not** included since it may require [special instructions](https://pytorch.org/get-started/locally/).

## Quick Start

### Write a [`gvl.Dataset`](api.md#genvarloader.Dataset)

```python
import genvarloader as gvl

gvl.write(
    path="cool_dataset.gvl",
    bed="interesting_regions.bed",
    variants="cool_variants.vcf",
    bigwigs=gvl.BigWigs.from_table("bigwig", "samples_to_bigwigs.csv"),
    max_jitter=128,
)
```

Where `samples_to_bigwigs.csv` has columns `sample` and `path` mapping each sample to its BigWig.

### Open a [`gvl.Dataset`](api.md#genvarloader.Dataset) and get a PyTorch DataLoader

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
dataset[0, 9]  # first region, 10th sample
dataset[:10, 4]  # first 10 regions, 5th sample
dataset[:10, :5]  # first 10 regions and first 5 samples
```

### Transform the data on-the-fly

```python
import seqpro as sp
from einops import rearrange

def transform(haplotypes, tracks):
    ohe = sp.DNA.ohe(haplotypes)
    ohe = rearrange(ohe, "... length alphabet -> ... alphabet length")
    return ohe, tracks

transformed_dataset = dataset.with_settings(transform=transform)
```

## Performance tips
- GenVarLoader uses multithreading extensively, so it's best to use `0` or `1` workers with your [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader).
- A GenVarLoader [`Dataset`](api.md#genvarloader.Dataset) is most efficient when given batches of indices, rather than one at a time. By default, [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)s use one index at a time, so if you want to use a ***custom*** [`Sampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Sampler) you should wrap it with a [`BatchSampler`](https://pytorch.org/docs/stable/data.html#torch.utils.data.BatchSampler) before passing it to [`Dataset.to_dataloader()`](api.md#genvarloader.Dataset.to_dataloader).