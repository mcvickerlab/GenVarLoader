"""GenVarLoader

GenVarLoader is a library for rapidly loading haplotypes and next generation sequencing read depth into sequence models.
"""

import importlib.metadata

from .bigwig import BigWigs
from .dataset import Dataset
from .dataset.write import write
from .fasta import Fasta
from .genvarloader import intervals as bw_intervals
from .haplotypes import Haplotypes
from .intervals import Intervals
from .loader import GVL, construct_virtual_data
from .torch import get_dataloader, get_sampler
from .types import Ragged
from .utils import read_bedlike, with_length
from .variants import Variants
from .zarr import ZarrTracks

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "Haplotypes",
    "Fasta",
    "Intervals",
    "BigWigs",
    "GVL",
    "construct_virtual_data",
    "Variants",
    "ZarrTracks",
    "write",
    "Dataset",
    "read_bedlike",
    "with_length",
    "get_dataloader",
    "get_sampler",
    "Ragged",
    "bw_intervals",
]
