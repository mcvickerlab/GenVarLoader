import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Union, cast

import numpy as np
import zarr
from numpy.typing import NDArray
from pysam import FastaFile

from genome_loader.loaders import Queries
from genome_loader.loaders.utils import ts_open_zarr
from genome_loader.types import SequenceEncoding
from genome_loader.utils import (
    ALPHABETS,
    DNA_COMPLEMENT,
    PathType,
    bytes_to_ohe,
    rev_comp_byte,
    rev_comp_ohe,
)


class Sequence(ABC):

    path: Path

    @abstractmethod
    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Load sequences matching query intervals.

        Query intervals can go beyond the range of contigs and will be padded with 'N' where
        there is no underlying sequence to fetch. For example, if an interval has a
        negative start, there will be 'N' added to the start of the returned sequence.

        Parameters
        ----------
        queries: Queries
            Query intervals.
        length : int
            Length of all intervals.
        **kwargs : dict, optional
            encoding : 'bytes' or 'onehot', how to encode the sequences

        Returns
        -------
        seqs : ndarray[bytes or uint8]
            Sequences for each interval. Has shape (intervals length [alphabet])
            where the final dimension is only present if one hot encoding.
        """
        raise NotImplementedError


class FastaSequence(Sequence):
    def __init__(self, fasta_path: PathType) -> None:
        """Load sequences from a fasta as NumPy arrays.

        Parameters
        ----------
        fasta_path : str, Path
        """
        self.path = Path(fasta_path)
        self.fasta = FastaFile(str(fasta_path))

    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        sorted = kwargs.get("sorted", False)
        encoding = SequenceEncoding(kwargs.get("encoding"))
        dtype = np.uint8 if encoding == "onehot" else "|S1"
        seqs = np.empty((len(queries), length), dtype=dtype)  # type: ignore

        # go in sorted order to minimize file seeking
        if not sorted:
            _queries = queries.sort_values(["contig", "start"])
        else:
            _queries = queries
        for tup in _queries.itertuples():
            i = tup.Index
            contig = tup.contig
            start = tup.start
            end = start + length
            prepend = min(start, 0)
            if prepend > 0:
                start = 0
            append = max(end - self.fasta.get_reference_length(contig), 0)
            if append > 0:
                end = self.fasta.get_reference_length(contig)
            seq = np.full_like(seqs[0], b"N")
            seq[prepend : length - append] = np.frombuffer(
                self.fasta.fetch(contig, start, end).encode(), "S1"
            )
            seqs[i] = seq

        rev_comp_idx = np.flatnonzero(queries.strand == "-")
        if len(rev_comp_idx) > 0:
            seqs[rev_comp_idx] = rev_comp_byte(
                seqs[rev_comp_idx], complement_map=DNA_COMPLEMENT
            )
        if encoding is SequenceEncoding.ONEHOT:
            seqs = bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"])

        return seqs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fasta.close()


# TODO: deprecate
""" class H5Sequence(Sequence):
    
    embedding: str
    contigs: NDArray[np.str_]
    spec: Optional[NDArray[np.bytes_]]
    
    def __init__(self, h5_path: PathType) -> None:
        self.path = Path(h5_path)
        
        # get info from h5
        with self._open() as f:
            self.embedding = f.attrs['id']
            self.spec = cast(type(self.spec), f.attrs.get('encode_spec', None))
            if self.spec is not None:
                self.spec = self.spec.astype('S')
                self._assert_reverse_is_complement(self.spec, DNA_COMPLEMENT)
            self.contigs: NDArray[np.str_] = np.array(list(f.keys()), dtype="U")
            self._dtype = f[self.contigs[0]][self.embedding].dtype
        
        # get encoding
        self.encoding = self.Encoding.ONEHOT if self.embedding == 'onehot' else self.Encoding.BYTES
        
        # get padding value for out of bounds queries
        pad_char = b'N'
        if self.encoding is self.Encoding.BYTES:
            self.pad_arr = np.array([pad_char], dtype=self._dtype)
        elif self.embedding is self.Encoding.ONEHOT:
            pad_arr = np.zeros_like(self.spec, dtype=self._dtype)
            if pad_char not in self.spec:
                warnings.warn("'N' is not in spec, will pad with 0 vector.")
            else:
                pad_arr[self.spec == pad_char] = 1
            self.pad_arr = pad_arr[None, :]
    
    def _open(self):
        return h5py.File(self.path)
    
    def _assert_reverse_is_complement(
        self, alphabet: NDArray[np.bytes_], complement_map: Dict[bytes, bytes]
    ):
        _alphabet = alphabet[:-1] if b"N" in alphabet else alphabet
        rev_alpha = _alphabet[::-1]
        for a, r in zip(_alphabet, rev_alpha):
            if complement_map[a] != r:
                raise ValueError("Reverse of alphabet does not yield the complement.")

    def sel(self, queries: Queries, length: int, **kwargs) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        sorted = kwargs.get("sorted", False)
        queries = cast(Queries, queries.reset_index(drop=True))
        queries['end'] = queries.start + length
        # queries['ref_start'] = 
        # queries['ref_end'] = 
        
        shape = [len(queries), length]
        out = np.tile(self.pad_arr, (len(queries), length, 1))
        with self._open() as f:
            for contig, group in queries.groupby('contig'):
                idx = group.index.values
                coords = (
                    np.tile(np.arange(length, dtype=np.uint32), (len(group), 1))
                    + rel_starts[:, None]
                )
                out[idx] = f[contig][coords]
                
        raise NotImplementedError """


class ZarrSequence(Sequence):
    def __init__(self, zarr_path: PathType):
        """Load sequences from a Zarr file.

        Note: this class assumes any one hot encoded sequences use an alphabet that
        1. Has N at the end.
        2. Is complemented by being reversed (without N).
            For example, `reverse(ACGT) = complement(ACGT) = TGCA`

        Parameters
        ----------
        zarr_path : PathType
        """
        self.path = Path(zarr_path)
        root = zarr.open_group(self.path, mode="r")
        self.encodings = {
            SequenceEncoding(enc)
            for enc in root.group_keys()
            if enc in {e.value for e in SequenceEncoding}
        }
        self.contig_lengths: Dict[str, int] = root.attrs["contig_lengths"]
        if SequenceEncoding.ONEHOT in self.encodings:
            self.alphabet = np.frombuffer(
                root.attrs["alphabet"].encode("ascii"), dtype="|S1"
            )

        self.tstores: Dict[str, Any] = {}

        def add_array_to_tstores(p: str, val: Union[zarr.Group, zarr.Array]):
            if isinstance(val, zarr.Array):
                self.tstores[p] = ts_open_zarr(self.path / p)

        root.visit(add_array_to_tstores)

    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        # get encoding
        encoding = SequenceEncoding(kwargs.get("encoding"))
        if encoding not in self.encodings:
            raise ValueError(f"Encoding '{encoding}' not found in Zarr.")

        out_shape = [len(queries), length]
        if encoding is SequenceEncoding.BYTES:
            dtype = "|S1"
            pad_arr = np.array(b"N")
        elif encoding is SequenceEncoding.ONEHOT:
            out_shape.append(len(self.alphabet))
            dtype = "u1"
            pad_arr = np.zeros(len(self.alphabet), dtype=dtype)
            pad_arr[self.alphabet == b"N"] = 1

        queries["end"] = queries.start + length
        # map negative starts to 0
        queries["in_start"] = queries.start.clip(lower=0)
        # map ends > contig length to contig length
        queries["contig_length"] = queries.contig.replace(self.contig_lengths)
        queries["in_end"] = np.minimum(queries.end, queries.contig_length)
        # get start, end index in output array
        queries["out_start"] = queries.in_start - queries.start
        queries["out_end"] = queries.in_end - queries.end

        def get_read(query):
            contig = query.contig
            path = f"{encoding}/{contig}"
            return self.tstores[path][query.in_start : query.in_end].read()

        # (q l [a])
        reads = await asyncio.gather(
            *[get_read(query) for query in queries.itertuples()]
        )

        # init array that will pad out-of-bound sequences
        out = np.full(out_shape, pad_arr, dtype=dtype)  # type: ignore
        for i, (read, query) in enumerate(zip(reads, queries.itertuples())):
            if encoding is SequenceEncoding.BYTES:
                # TensorStore doesn't yet support bytes, so we store the ascii code
                read = read.view(dtype)  # type: ignore
            # (1 l [a]) = (l [a])
            out[i, query.out_start : query.out_end] = read

        # reverse complement negative stranded queries
        to_rev_comp = cast(NDArray[np.bool_], (queries["strand"] == "-").values)
        if encoding is SequenceEncoding.BYTES:
            out[to_rev_comp] = rev_comp_byte(out[to_rev_comp], DNA_COMPLEMENT)
        elif encoding is SequenceEncoding.ONEHOT:
            out[to_rev_comp] = rev_comp_ohe(out[to_rev_comp], has_N=True)

        return out


def ref_loader_factory(ref_path: PathType) -> Sequence:
    _ref_path = Path(ref_path)
    if ".fa" in _ref_path.name:
        return FastaSequence(_ref_path)
    elif ".zarr" in _ref_path.name:
        return ZarrSequence(_ref_path)
    else:
        raise ValueError("File extension for reference is neither FASTA nor HDF5.")
