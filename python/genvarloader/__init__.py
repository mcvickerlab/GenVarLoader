import importlib.metadata

from loguru import logger

from ._bigwig import BigWigs
from ._dataset._impl import ArrayDataset, Dataset, RaggedDataset
from ._dataset._write import write
from ._dummy import get_dummy_dataset
from ._ragged import Ragged
from ._utils import read_bedlike, with_length
from ._variants import Variants
from ._variants._sitesonly import SitesOnlyTable, SitesOnlyVCF

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "write",
    "Dataset",
    "Variants",
    "BigWigs",
    "read_bedlike",
    "with_length",
    "Ragged",
    "get_dummy_dataset",
    "SitesOnlyTable",
    "SitesOnlyVCF",
    "ArrayDataset",
    "RaggedDataset",
]


logger.disable("genvarloader")
