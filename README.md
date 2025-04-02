<img src=docs/source/_static/gvl_logo.png width="200">

[![PyPI version](https://badge.fury.io/py/genvarloader.svg)](https://pypi.org/project/genvarloader/)
[![Documentation Status](https://readthedocs.org/projects/genvarloader/badge/?version=latest)](https://genvarloader.readthedocs.io)
[![Downloads](https://static.pepy.tech/badge/genvarloader)](https://pepy.tech/project/genvarloader)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/genvarloader)](https://img.shields.io/pypi/dm/genvarloader)
[![GitHub stars](https://badgen.net/github/stars/mcvickerlab/GenVarLoader)](https://GitHub.com/Naereen/mcvickerlab/GenVarLoader)
[![bioRxiv](https://img.shields.io/badge/bioRxiv-2025.01.15.633240-b31b1b.svg)](https://www.biorxiv.org/content/10.1101/2025.01.15.633240)

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

## Contributing

1. Clone the repo.
2. Assuming you have [Pixi](https://pixi.sh/latest/), install pre-commit hooks `pixi run -e dev pre-commit`
3. Use the appropriate Pixi environment for your needs. A decent catch-all is `dev` but you might need a different environment if using a GPU.