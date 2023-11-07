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
from .fasta_variants import FastaVariants
from .loader import GVL, view_virtual_data
from .pgen import Pgen
from .rle_table import RLE_Table
from .types import Reader, Variants

__version__ = "0.0.0"  # managed by poetry-dynamic-versioning

__all__ = [
    "Fasta",
    "FastaVariants",
    "RLE_Table",
    "Pgen",
    "GVL",
    "view_virtual_data",
    "Reader",
    "Variants",
]
