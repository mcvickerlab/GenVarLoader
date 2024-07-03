"""Rapidly generate haplotypes and functional tracks for sequence models."""

import importlib.metadata

from ._bigwig import BigWigs
from ._dataset import Dataset
from ._dataset._write import write
from ._fasta import Fasta
from ._types import Ragged
from ._utils import read_bedlike, with_length
from ._variants import DenseGenotypes, Variants

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "write",
    "read_bedlike",
    "with_length",
    "Dataset",
    "Ragged",
    "Fasta",
    "BigWigs",
    "Variants",
    "DenseGenotypes",
]
