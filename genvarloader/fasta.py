from pathlib import Path
from typing import Optional, Union, cast

import dask.array as da
import numpy as np
import pysam
import seqpro as sp
import xarray as xr
from numpy.typing import NDArray

from .types import Reader
from .util import splice_subarrays


class Fasta(Reader):
    def __init__(
        self,
        name: str,
        path: Union[str, Path],
        pad: Optional[str] = None,
        alphabet: Optional[Union[str, sp.NucleotideAlphabet, sp.AminoAlphabet]] = None,
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

        Raises
        ------
        ValueError
            If pad value is not a single character.
        """
        self.virtual_data = xr.DataArray(
            da.empty(0, dtype="S1"),  # pyright: ignore[reportPrivateImportUsage]
            name=name,
            dims="",
        )
        self.path = path
        if pad is not None:
            if len(pad) > 1:
                raise ValueError("Pad value must be a single character.")
            self.pad = pad.encode("ascii")
        else:
            self.pad = pad

        with self._open() as f:
            self.contigs = {c: f.get_reference_length(c) for c in f.references}

        if alphabet is None:
            self.alphabet = sp.alphabets.DNA
        elif isinstance(alphabet, str):
            alphabet = alphabet.upper()
            try:
                self.alphabet = getattr(sp.alphabets, alphabet)
            except AttributeError:
                raise ValueError(f"Alphabet {alphabet} not found.")
        else:
            self.alphabet = alphabet

        if isinstance(self.alphabet, sp.NucleotideAlphabet):
            self.rev_strand_fn = self.alphabet.reverse_complement
        elif isinstance(self.alphabet, sp.AminoAlphabet):

            def rev_strand_fn(a: NDArray[np.bytes_]):
                return a[::-1]

            self.rev_strand_fn = rev_strand_fn

        self.contig_starts_with_chr = self.infer_contig_prefix(self.contigs)

    def _open(self):
        return pysam.FastaFile(str(self.path))

    def read(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        out: Optional[NDArray] = None,
        **kwargs,
    ) -> xr.DataArray:
        """Read a sequence from a FASTA file.

        Parameters
        ----------
        contig : str
            Name of the contig/chromosome.
        starts : NDArray[np.int64]
            Start coordinates, 0-based.
        ends : NDArray[np.int64]
            End coordinates, 0-based exclusive.
        out : NDArray, optional
            Array to put the result into. Otherwise allocates one.
        **kwargs
            Not used.

        Returns
        -------
        xr.DataArray
            Sequence from FASTA file.

        Raises
        ------
        ValueError
            Coordinates are out-of-bounds and pad value is not set.
        """
        contig = self.normalize_contig_name(contig)

        start = starts.min()
        end = ends.max()

        pad_left = -min(0, start)
        if pad_left > 0 and self.pad is None:
            raise ValueError("Padding is disabled and a start is < 0.")

        with self._open() as f:
            pad_right = max(0, end - f.get_reference_length(contig))
            if pad_right > 0 and self.pad is None:
                raise ValueError("Padding is disabled and end is > contig length.")

            # pysam behavior
            # start < 0 => error
            # end > contig length => truncate
            seq = f.fetch(contig, max(0, start), end)

        seq = np.frombuffer(seq.encode("ascii"), "S1")
        padded_seq = []
        if pad_left > 0:
            pad_left = np.full(pad_left, self.pad)
            padded_seq.append(pad_left)
        padded_seq.append(seq)
        if pad_right > 0:
            pad_right = np.full(pad_right, self.pad)
            padded_seq.append(pad_right)
        seq = cast(NDArray[np.bytes_], np.concatenate(padded_seq))

        out = splice_subarrays(seq, starts, ends)

        return xr.DataArray(out, dims="length")
