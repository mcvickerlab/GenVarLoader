import importlib.metadata

from seqpro._ragged import Ragged
from seqpro.bed import read_bedlike, with_length

from ._bigwig import BigWigs
from ._dataset._impl import ArrayDataset, Dataset, RaggedDataset
from ._dataset._reconstruct import Reference
from ._dataset._write import write
from ._dummy import get_dummy_dataset
from ._torch import to_nested_tensor

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "write",
    "Dataset",
    "BigWigs",
    "read_bedlike",
    "with_length",
    "Ragged",
    "get_dummy_dataset",
    "ArrayDataset",
    "RaggedDataset",
    "Reference",
    "to_nested_tensor",
]
