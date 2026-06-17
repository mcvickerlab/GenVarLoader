# ruff: noqa: E402  cap_numba_threads() must run before any numba kernel imports
import importlib.metadata

from ._threads import cap_numba_threads

cap_numba_threads()

from seqpro.bed import read as read_bedlike
from seqpro.bed import with_len as with_length
from seqpro.rag import Ragged

from . import data_registry
from ._bigwig import BigWigs
from ._dataset._flat_variants import DummyVariant, VarWindowOpt
from ._dataset._flat_variants import _FlatAlleles as FlatAlleles
from ._dataset._flat_variants import _FlatVariants as FlatVariants
from ._dataset._flat_variants import _FlatVariantWindows as FlatVariantWindows
from ._dataset._impl import ArrayDataset, Dataset, RaggedDataset
from ._dataset._insertion_fill import (
    Constant,
    FlankSample,
    InsertionFill,
    Interpolate,
    Repeat5p,
    Repeat5pNormalized,
)
from ._dataset._rag_variants import RaggedVariants
from ._dataset._reference import RefDataset, Reference
from ._dataset._svar_link import migrate_svar_link
from ._dataset._write import get_splice_bed, update, write
from ._dummy import get_dummy_dataset
from ._flat import _Flat as FlatRagged
from ._flat import _FlatAnnotatedHaps as FlatAnnotatedHaps
from ._ragged import RaggedAnnotatedHaps, RaggedIntervals
from ._torch import to_nested_tensor
from ._types import AnnotatedHaps
from ._variants._sitesonly import DatasetWithSites, SitesSchema, sites_vcf_to_table

__version__ = importlib.metadata.version("genvarloader")

__all__ = [
    "AnnotatedHaps",
    "ArrayDataset",
    "BigWigs",
    "Constant",
    "Dataset",
    "DatasetWithSites",
    "DummyVariant",
    "FlankSample",
    "FlatAlleles",
    "FlatAnnotatedHaps",
    "FlatRagged",
    "FlatVariantWindows",
    "FlatVariants",
    "InsertionFill",
    "Interpolate",
    "Ragged",
    "RaggedAnnotatedHaps",
    "RaggedDataset",
    "RaggedIntervals",
    "RaggedVariants",
    "RefDataset",
    "Reference",
    "Repeat5p",
    "Repeat5pNormalized",
    "SitesSchema",
    "VarWindowOpt",
    "data_registry",
    "get_dummy_dataset",
    "get_splice_bed",
    "migrate_svar_link",
    "read_bedlike",
    "sites_vcf_to_table",
    "to_nested_tensor",
    "update",
    "with_length",
    "write",
]
