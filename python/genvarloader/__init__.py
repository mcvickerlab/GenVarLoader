import importlib.metadata

from seqpro.bed import read as read_bedlike
from seqpro.bed import with_len as with_length
from seqpro.rag import Ragged

from . import data_registry
from ._bigwig import BigWigs
from ._dataset._impl import ArrayDataset, Dataset, RaggedDataset
from ._dataset._rag_variants import RaggedVariants
from ._dataset._reference import RefDataset, Reference
from ._dataset._write import write
from ._dummy import get_dummy_dataset
from ._ragged import RaggedAnnotatedHaps
from ._torch import to_nested_tensor
from ._types import AnnotatedHaps
from ._variants._sitesonly import DatasetWithSites, SitesSchema, sites_vcf_to_table

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
    "sites_vcf_to_table",
    "SitesSchema",
    "DatasetWithSites",
    "RefDataset",
    "data_registry",
    "AnnotatedHaps",
    "RaggedAnnotatedHaps",
    "RaggedVariants",
]
