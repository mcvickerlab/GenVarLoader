"""GenVarLoader

GenVarLoader is a library for rapidly loading haplotypes and next generation sequencing read depth into sequence models.
"""

from .bigwig import BigWigs
from .dataset import Dataset
from .fasta import Fasta
from .haplotypes import Haplotypes
from .intervals import Intervals
from .loader import GVL, construct_virtual_data
from .torch import get_dataloader, get_sampler
from .util import read_bedlike, with_length
from .variants import Variants
from .write import write
from .zarr import ZarrTracks

__version__ = "0.0.0"  # managed by poetry-dynamic-versioning

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
]
