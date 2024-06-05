"""GenVarLoader

GenVarLoader is a library for rapidly loading haplotypes and next generation sequencing read depth into sequence models.
"""

import importlib.metadata

from .bigwig import BigWigs
from .dataset import Dataset
from .dataset.write import write
from .fasta import Fasta
from .types import Ragged
from .utils import read_bedlike, with_length
from .variants import Variants

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "Fasta",
    "BigWigs",
    "Variants",
    "write",
    "Dataset",
    "read_bedlike",
    "with_length",
    "Ragged",
]
