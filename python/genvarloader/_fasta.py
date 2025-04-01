from __future__ import annotations

import re
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Optional, Union, cast

import numpy as np
import pysam
import seqpro as sp
from numpy.typing import ArrayLike, NDArray
from tqdm.auto import tqdm
from typing_extensions import assert_never

from ._types import Reader
from ._utils import _get_rel_starts, _normalize_contig_name

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
        path: Union[str, Path],
        pad: Optional[str] = None,
        alphabet: Optional[Union[str, sp.NucleotideAlphabet, sp.AminoAlphabet]] = None,
        in_memory: bool = False,
        cache: bool = False,
    ) -> None:
        """Read sequences from a FASTA file.

        Parameters
        ----------
        name : str
            Name of the reader, for example `'seq'`.
        path : Union[str, Path]
            Path to the FASTA file.
        pad : Optional[str], optional
            A single character which, if passed, will pad out-of-bound ranges with this
            value. By default no padding is done and out-of-bound ranges raise an error.
        alphabet : str, sp.NucleotideAlphabet, sp.AminoAlphabet, optional
            Alphabet to use for the sequences. If not passed, defaults to DNA.
        in_memory : bool, optional
            Whether to load the sequences into memory. If `True`, the sequences will be
            loaded into memory and the FASTA file will be closed. If `False`, the sequences
            will be read from the FASTA file on demand. Defaults to `False`.
        cache : bool, optional
            Whether to cache the sequences to disk. If `True`, the sequences will be written
            to disk in a `.fa.gvl` file. Defaults to `False`. Only used if `in_memory` is `True`.

        Raises
        ------
        ValueError
            If pad value is not a single character.
        """
        self.name = name
        self.path = Path(path)
        if pad is None:
            self.pad = pad
        else:
            if len(pad) > 1:
                raise ValueError("Pad value must be a single character.")
            self.pad = pad.encode("ascii")

        with self._open() as f:
            self.contigs = {c: f.get_reference_length(c) for c in f.references}

        if alphabet is None:
            self.alphabet: Union[sp.NucleotideAlphabet, sp.AminoAlphabet] = (
                sp.alphabets.DNA
            )
        elif isinstance(alphabet, str):
            alphabet = alphabet.upper()
            try:
                self.alphabet = cast(
                    Union[sp.NucleotideAlphabet, sp.AminoAlphabet],
                    getattr(sp.alphabets, alphabet),
                )
            except AttributeError:
                raise ValueError(f"Alphabet {alphabet} not found.")
        else:
            self.alphabet = alphabet

        if isinstance(self.alphabet, sp.NucleotideAlphabet):
            self.rev_strand_fn = partial(
                self.alphabet.reverse_complement, length_axis=-1
            )
        elif isinstance(self.alphabet, sp.AminoAlphabet):

            def rev_strand_fn(a: NDArray[np.bytes_]):
                return a[::-1]

            self.rev_strand_fn = rev_strand_fn  # type: ignore[assignment]
        else:
            assert_never(self.alphabet)

        self.contigs = self._get_contig_lengths()

        self.handle: Optional[pysam.FastaFile] = None
        fa_extension = re.compile(r"\.(fa|fna|fasta)(\.b?gz)?$")
        self.cache_path = Path(fa_extension.sub(".fa.gvl", str(self.path)))

        if not in_memory:
            self.sequences = None
        else:
            if cache:
                if not self.cache_path.exists():
                    self._write_to_cache()
                self.sequences = self._get_sequences(self.contigs)
            else:
                self.sequences = self._get_sequences(self.contigs)

    def _get_contig_lengths(self) -> Dict[str, int]:
        with self._open() as f:
            return {c: f.get_reference_length(c) for c in f.references}

    def _get_sequences(
        self, contigs: Iterable[str], from_fasta=False
    ) -> Dict[str, NDArray[np.bytes_]]:
        """Load contigs into memory."""
        sequences: Dict[str, NDArray[np.bytes_]] = {}
        if from_fasta or not self.cache_path.exists():
            with self._open() as f:
                pbar = tqdm(total=len(self.contigs))
                for c in contigs:
                    pbar.set_description(f"Reading contig {c}")
                    sequences[c] = np.frombuffer(
                        f.fetch(c).encode("ascii").upper(), "S1"
                    )
                    pbar.update()
                pbar.close()
        elif self.cache_path.exists():
            for contig in contigs:
                sequences[contig] = np.load(self.cache_path / f"{contig}.npy")
        return sequences

    def _write_to_cache(self):
        """Write contigs to cache."""
        self.cache_path.mkdir(parents=True, exist_ok=True)

        with self._open() as f:
            pbar = tqdm(total=len(self.contigs))
            for c in self.contigs:
                if self.sequences is None:
                    pbar.set_description(f"Reading contig {c}")
                    seq = np.frombuffer(f.fetch(c).encode("ascii").upper(), "S1")
                else:
                    seq = self.sequences[c]
                pbar.set_description(f"Writing contig {c}")
                np.save(self.cache_path / f"{c}.npy", seq)
                pbar.update()
            pbar.close()

    def _open(self):
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

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : ArrayLike
            Start coordinates, 0-based.
        ends : ArrayLike
            End coordinates, 0-based exclusive.
        **kwargs
            Not used.

        Returns
        -------
        NDArray[np.bytes_]
            Shape: (length). Sequence corresponding to the given genomic coordinates.

        Raises
        ------
        ValueError
            Coordinates are out-of-bounds and pad value is not set.
        """
        _contig = _normalize_contig_name(contig, self.contigs)
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

        rel_starts = _get_rel_starts(starts, ends)
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
