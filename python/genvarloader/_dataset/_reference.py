from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    Union,
    cast,
)

import numpy as np
from attrs import define
from loguru import logger
from numpy.typing import NDArray

from .._fasta import Fasta
from .._utils import _normalize_contig_name

__all__ = []


@define
class Reference:
    reference: NDArray[np.uint8]
    contigs: List[str]
    offsets: NDArray[np.uint64]
    pad_char: int

    @classmethod
    def from_path_and_contigs(cls, fasta: Union[str, Path], contigs: List[str]):
        _fasta = Fasta("ref", fasta, "N")

        if not _fasta.cache_path.exists():
            logger.info("Memory-mapping FASTA file for faster access.")
            _fasta._write_to_cache()

        contigs = cast(
            List[str],
            [_normalize_contig_name(c, _fasta.contigs) for c in contigs],
        )
        _fasta.sequences = _fasta._get_sequences(contigs)
        if TYPE_CHECKING:
            assert _fasta.sequences is not None
            assert _fasta.pad is not None
        refs: List[NDArray[np.bytes_]] = []
        next_offset = 0
        _ref_offsets: Dict[str, int] = {}
        for contig in contigs:
            arr = _fasta.sequences[contig]
            refs.append(arr)
            _ref_offsets[contig] = next_offset
            next_offset += len(arr)
        reference = np.concatenate(refs).view(np.uint8)
        pad_char = ord(_fasta.pad)
        if any(c is None for c in contigs):
            raise ValueError("Contig names in metadata do not match reference.")
        ref_offsets = np.empty(len(contigs) + 1, np.uint64)
        ref_offsets[:-1] = np.array([_ref_offsets[c] for c in contigs], dtype=np.uint64)
        ref_offsets[-1] = len(reference)
        return cls(reference, contigs, ref_offsets, pad_char)
