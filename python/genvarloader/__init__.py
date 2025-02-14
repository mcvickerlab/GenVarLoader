import importlib.metadata

from loguru import logger

from ._bigwig import BigWigs
from ._dataset import Dataset
from ._dataset._write import write
from ._ragged import Ragged
from ._utils import read_bedlike
from ._variants import Variants

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "write",
    "Dataset",
    "Variants",
    "BigWigs",
    "read_bedlike",
    "Ragged",
]


logger.disable("genvarloader")
