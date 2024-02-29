"""GenVarLoader

Idea behind this implementation is to efficiently materialize sequences from long,
overlapping ROIs. The algorithm is roughly:
1. Partition the ROIs to maximize the size of the union of ROIs while respecting
memory limits. Note any union of ROIs must be on the same contig. Buffer this
union of ROIs in memory.
2. Materialize batches of subsequences, i.e. the ROIs, by slicing the buffer. This
keeps memory usage to a minimum since we only need enough for the buffer + a single
batch. This should be fast because the buffer is the only part that uses file I/O
whereas the batches are materialized from the buffer.
"""

from .fasta import Fasta
from .haplotypes import Haplotypes
from .intervals import Intervals
from .loader import GVL, construct_virtual_data
from .types import Reader
from .variants import MemmapGenos, PgenGenos, Records, Variants, VCFGenos, ZarrGenos
from .zarr import ZarrTracks

__version__ = "0.0.0"  # managed by poetry-dynamic-versioning

__all__ = [
    "Haplotypes",
    "Fasta",
    "Intervals",
    "GVL",
    "construct_virtual_data",
    "Reader",
    "Variants",
    "PgenGenos",
    "ZarrGenos",
    "MemmapGenos",
    "VCFGenos",
    "Records",
    "ZarrTracks",
]
