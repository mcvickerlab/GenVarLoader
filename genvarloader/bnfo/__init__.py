from typing import Iterable, Union

import xarray as xr

from .bigwig import BigWig
from .fasta import Fasta
from .fasta_variants import FastaVariants
from .loader import GVL
from .pgen import Pgen
from .rle_table import RLE_Table
from .tiledb_vcf import TileDB_VCF
from .types import Reader

__all__ = [
    "BigWig",
    "Fasta",
    "TileDB_VCF",
    "FastaVariants",
    "RLE_Table",
    "Pgen",
    "GVL",
    "view_virtual_data",
]


def view_virtual_data(readers: Union[Reader, Iterable[Reader]]):
    """View the virtual data corresponding from multiple readers. This is useful to
    inspect what non-length dimensions will be exist when constructing a GVL loader
    from them.

    Parameters
    ----------
    readers : Reader, Iterable[Reader]
        Readers to inspect.
    """
    if not isinstance(readers, Iterable):
        readers = [readers]
    return xr.merge([r.virtual_data for r in readers], join="exact")
