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

import xarray as xr

from .fasta import Fasta
from .fasta_variants import FastaVariants
from .loader import GVL
from .pgen import Pgen
from .rle_table import RLE_Table
from .tiledb_vcf import TileDB_VCF
from .types import Reader, Variants

__all__ = [
    "Fasta",
    "TileDB_VCF",
    "FastaVariants",
    "RLE_Table",
    "Pgen",
    "GVL",
    "view_virtual_data",
    "Reader",
    "Variants",
]


def view_virtual_data(*readers: Reader):
    """View the virtual data corresponding from multiple readers. This is useful to
    inspect what non-length dimensions will be exist when constructing a GVL loader
    from them.

    Parameters
    ----------
    *readers : Reader
        Readers to inspect.
    """
    return xr.merge([r.virtual_data for r in readers], join="exact")
