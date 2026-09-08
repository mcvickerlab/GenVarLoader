from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pysam
import seqpro as sp
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm
from typing_extensions import assert_never

from . import _fasta_cache
from ._types import Reader
from ._utils import get_rel_starts, normalize_contig_name

__all__ = ["Fasta", "NoPadError"]


class NoPadError(Exception):
    pass


class Fasta(Reader):
    dtype = np.dtype("S1")
    """Data type of the sequences."""
    sizes = {}
    """Sizes of non-sequence length dimensions."""
    coords = {}
    """Coordinates of non-sequence length dimensions."""
    chunked = False
    """Whether the reader represents a chunked file format."""

    def __init__(
        self,
        name: str,
        path: str | Path,
        pad: str | None = None,
        alphabet: str | sp.NucleotideAlphabet | None = None,
        in_memory: bool = False,
        cache: bool = False,
    ) -> None:
        """Read sequences from a FASTA file.

        Args:
            name (str): Name of the reader, for example `'seq'`.
            path (Union[str, Path]): Path to the FASTA file or a `.gvlfa` cache directory.
            pad (Optional[str], optional): A single character which, if passed, will pad
                out-of-bound ranges with this value. By default no padding is done and
                out-of-bound ranges raise an error.
            alphabet (str, sp.NucleotideAlphabet, sp.AminoAlphabet, optional): Alphabet to use
                for the sequences. If not passed, defaults to DNA.
            in_memory (bool, optional): Whether to load the sequences into memory. If `True`,
                the sequences will be loaded into memory and the FASTA file will be closed. If
                `False`, the sequences will be read from the FASTA file on demand. Defaults to
                `False`.
            cache (bool, optional): Whether to cache the sequences to disk. If `True`, the
                sequences are written to a sibling `.gvlfa` directory (a self-describing,
                fingerprint-validated cache). Defaults to `False`. Only used if `in_memory` is
                `True`. A legacy `.fa.gvl` cache, if present, is migrated automatically. A
                `.gvlfa` directory may also be passed directly as `path`.

        Raises:
            ValueError: If pad value is not a single character.
        """
        self.name = name
        path = Path(path)
        self.pad: bytes | None
        if pad is None:
            self.pad = pad
        else:
            if len(pad) > 1:
                raise ValueError("Pad value must be a single character.")
            self.pad = pad.encode("ascii")

        self._is_cache_input = _fasta_cache.is_gvlfa(path)
        if self._is_cache_input:
            meta, data_path = _fasta_cache.ensure_cache(path)
            self.cache_path = path
            self._data_path = data_path
            self.contigs = dict(meta.contigs)
            source = _fasta_cache.resolve_source(path, meta)
            self._source_available = source is not None
            self.path = source if source is not None else path
        else:
            self.path = path
            self.cache_path = _fasta_cache._cache_dir_for(path)
            self._data_path = self.cache_path / _fasta_cache.DATA_FILENAME
            self._source_available = True
            with self._open() as f:
                self.contigs = {c: f.get_reference_length(c) for c in f.references}

        if alphabet is None:
            self.alphabet: sp.NucleotideAlphabet = sp.alphabets.DNA
        elif isinstance(alphabet, str):
            alphabet = alphabet.upper()
            try:
                self.alphabet = cast(
                    sp.NucleotideAlphabet, getattr(sp.alphabets, alphabet)
                )
            except AttributeError:
                raise ValueError(f"Alphabet {alphabet} not found.")
        else:
            self.alphabet = alphabet

        if isinstance(self.alphabet, sp.NucleotideAlphabet):

            def rev_strand_fn(data: NDArray[np.bytes_]) -> NDArray[np.bytes_]:
                return self.alphabet.reverse_complement(data, length_axis=-1)

        elif isinstance(self.alphabet, sp.AminoAlphabet):

            def rev_strand_fn(data: NDArray[np.bytes_]):
                return data[::-1]

        else:
            assert_never(self.alphabet)

        self.rev_strand_fn = rev_strand_fn
        self.handle: pysam.FastaFile | None = None

        if not in_memory:
            self.sequences = None
        elif self._is_cache_input:
            self.sequences = self._memmap_sequences()
        elif cache:
            _, self._data_path = _fasta_cache.ensure_cache(self.path)
            self.sequences = self._memmap_sequences()
        else:
            self.sequences = self._read_all_from_fasta()

    def _memmap_sequences(self) -> dict[str, NDArray[np.bytes_]]:
        """Load contigs as views into the cached sequence.bin memmap."""
        seqs = np.memmap(self._data_path, dtype="S1", mode="r")
        out: dict[str, NDArray[np.bytes_]] = {}
        offset = 0
        for contig, length in self.contigs.items():
            out[contig] = seqs[offset : offset + length]
            offset += length
        return out

    def _read_all_from_fasta(self) -> dict[str, NDArray[np.bytes_]]:
        """Load all contigs into memory directly from the FASTA (no disk cache)."""
        out: dict[str, NDArray[np.bytes_]] = {}
        with self._open() as f:
            for c in tqdm(self.contigs, total=len(self.contigs)):
                out[c] = np.frombuffer(f.fetch(c).encode("ascii").upper(), "S1")
        return out

    def _open(self):
        if not self._source_available:
            raise FileNotFoundError(
                f"Source FASTA for cache {self.cache_path} could not be located; "
                "on-demand reads require the source file. Re-open with in_memory=True "
                "to use cached data, or restore the source FASTA."
            )
        return pysam.FastaFile(str(self.path))

    def close(self):
        if self.handle is not None:
            self.handle.close()
        self.handle = None

    def read(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        **kwargs,
    ) -> NDArray[np.bytes_]:
        """Read a sequence from a FASTA file.

        Args:
            contig (str): Name of the contig/chromosome.
            starts (ArrayLike): Start coordinates, 0-based.
            ends (ArrayLike): End coordinates, 0-based exclusive.
            **kwargs: Not used.

        Returns:
            NDArray[np.bytes_]: Shape: (length). Sequence corresponding to the given genomic
                coordinates.

        Raises:
            ValueError: Coordinates are out-of-bounds and pad value is not set.
        """
        _contig = normalize_contig_name(contig, self.contigs)
        if _contig is None:
            raise ValueError(f"Contig {contig} not found.")
        else:
            contig = _contig
        contig_len = self.contigs[contig]

        starts = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends = np.atleast_1d(np.asarray(ends, dtype=np.int64))

        if starts.ndim > 1 or ends.ndim > 1:
            raise ValueError("Starts and ends must be coercible to 1D arrays.")

        if len(starts) != len(ends):
            raise ValueError("Starts and ends must be the same length.")

        lengths = ends - starts

        q_starts = np.maximum(0, starts)
        q_ends = np.minimum(self.contigs[contig], ends)

        left_pads = -np.minimum(0, starts)
        right_pads = np.maximum(0, ends - contig_len)

        if self.pad is None:
            if (left_pads > 0).any():
                raise NoPadError("Padding is disabled and a start is < 0.")
            if (right_pads > 0).any():
                raise NoPadError("Padding is disabled and an end is > contig length.")
            out = np.empty(lengths.sum(), dtype="S1")
        else:
            out = np.full(lengths.sum(), self.pad, dtype="S1")

        rel_starts = get_rel_starts(starts, ends)
        rel_ends = rel_starts + lengths - right_pads
        rel_starts += left_pads

        if self.sequences is None:
            if self.handle is None:
                self.handle = self._open()

            for q_start, q_end, rel_start, rel_end in zip(
                q_starts, q_ends, rel_starts, rel_ends
            ):
                seq = self.handle.fetch(contig, q_start, q_end)
                seq = np.frombuffer(seq.encode("ascii").upper(), "S1")
                out[rel_start:rel_end] = seq
        else:
            for q_start, q_end, rel_start, rel_end in zip(
                q_starts, q_ends, rel_starts, rel_ends
            ):
                seq = self.sequences[contig][q_start:q_end]
                out[rel_start:rel_end] = seq

        return out
