import asyncio
from pathlib import Path
from typing import Any, Dict, Optional, Set, Union, cast

import numpy as np
import zarr
from numpy.typing import NDArray
from pysam import FastaFile

from genvarloader.loaders.types import Queries, _TStore
from genvarloader.loaders.utils import ts_readonly_zarr
from genvarloader.types import ALPHABETS, PathType, SequenceAlphabet, SequenceEncoding
from genvarloader.utils import bytes_to_ohe, rev_comp_byte, rev_comp_ohe


class FastaSequence:
    def __init__(self, fasta_path: PathType) -> None:
        """Load sequences from a fasta as NumPy arrays.

        Parameters
        ----------
        fasta_path : str, Path
        """
        self.path = Path(fasta_path)
        self.fasta = FastaFile(str(fasta_path))
        self.contig_lengths = {
            c: self.fasta.get_reference_length(c) for c in self.fasta.references
        }

    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        sorted = kwargs.get("sorted", False)
        encoding = SequenceEncoding(kwargs.get("encoding"))

        queries["end"] = queries.start + length
        # map negative starts to 0
        queries["in_start"] = queries.start.clip(lower=0)
        # map ends > contig length to contig length
        queries["contig_length"] = queries.contig.replace(self.contig_lengths).astype(
            int
        )
        queries["in_end"] = np.minimum(queries.end, queries.contig_length)
        # get start, end index in output array
        queries["out_start"] = queries.in_start - queries.start
        queries["out_end"] = queries.in_end - queries.in_start

        # go in sorted order to minimize file seeking
        if not sorted:
            _queries = queries.sort_values(["contig", "start"])
        else:
            _queries = queries

        seqs = cast(NDArray[np.bytes_], np.full_like((len(queries), length), b"N"))
        for q in _queries.itertuples():
            i = q.Index
            seqs[i, q.out_start : q.out_end] = np.frombuffer(
                self.fasta.fetch(q.contig, q.in_start, q.in_end).encode(), "S1"
            )

        to_rev_comp = cast(NDArray[np.bool_], (queries.strand == "-").values)
        if to_rev_comp.any():
            seqs[to_rev_comp] = rev_comp_byte(
                seqs[to_rev_comp], alphabet=ALPHABETS["DNA"]
            )

        if encoding is SequenceEncoding.ONEHOT:
            seqs = bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"])

        return seqs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.fasta.close()


class Sequence:
    path: Path
    encodings: Set[SequenceEncoding]
    contig_lengths: Dict[str, int]
    alphabet: SequenceAlphabet
    tstores: Dict[str, _TStore]

    def __init__(self, zarr_path: PathType, ts_kwargs: Optional[Dict[str, Any]] = None):
        """Load sequences from a Zarr file.

        Note: this class assumes any one hot encoded sequences use an alphabet that
        1. Has N at the end.
        2. Is complemented by being reversed (without N).
            For example, `reverse(ACGT) = complement(ACGT) = TGCA`

        Parameters
        ----------
        zarr_path : PathType
        ts_kwargs : dict[str, any]
            Keyword arguments to pass to tensorstore.open(). Useful e.g. to specify a shared cache pool across
            loaders.
        """
        self.path = Path(zarr_path)
        root = cast(zarr.Group, zarr.open_consolidated(str(self.path), mode="r"))
        self.encodings = {
            SequenceEncoding(enc)
            for enc in root.group_keys()
            if enc in set(SequenceEncoding)
        }
        self.contig_lengths: Dict[str, int] = root.attrs["lengths"]
        self.alphabet = SequenceAlphabet(
            root.attrs["alphabet"], root.attrs["alphabet"][:-1][::-1] + "N"
        )

        self.tstores = {}
        if ts_kwargs is None:
            ts_kwargs = {}

        def add_array_to_tstores(p: str, val: Union[zarr.Group, zarr.Array]):
            if isinstance(val, zarr.Array):
                self.tstores[p] = ts_readonly_zarr(self.path / p, **ts_kwargs).result()

        root.visititems(add_array_to_tstores)

    def sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select query sequences.

        NOTE: Must provide the keyword argument `encoding`.

        Parameters
        ----------
        queries : Queries
        length : int
        **kwargs
            encoding : str
                Either "bytes" or "onehot"

        Returns
        -------
        sequence : ndarray
        """
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Select query sequences.

        NOTE: Must provide the keyword argument `encoding`.

        Parameters
        ----------
        queries : Queries
        length : int
        **kwargs
            encoding : str
                Required. Either "bytes" or "onehot".

        Returns
        -------
        sequence : ndarray
        """
        queries = cast(Queries, queries.reset_index(drop=True))
        if "strand" not in queries:
            queries["strand"] = "+"
            queries["strand"] = queries.strand.astype("category")

        query_contigs_not_in_sequence = np.setdiff1d(
            queries.contig.unique(), list(self.contig_lengths.keys())
        )
        if len(query_contigs_not_in_sequence) > 0:
            raise ValueError(
                "Got contigs that aren't available in the sequence file:",
                query_contigs_not_in_sequence,
            )

        # get encoding
        _enc = kwargs.get("encoding", None)
        if _enc is None or _enc not in list(SequenceEncoding):
            raise ValueError(
                f'The keyword argument "encoding" must be either "bytes" or "onehot", not {_enc}.'
            )
        encoding = SequenceEncoding(_enc)
        if encoding not in self.encodings:
            raise ValueError(f"Encoding '{encoding}' not found in Zarr.")

        out_shape = [len(queries), length]
        if encoding is SequenceEncoding.BYTES:
            dtype = "|S1"
            pad_arr = np.array(b"N")
        elif encoding is SequenceEncoding.ONEHOT:
            out_shape.append(len(self.alphabet.array))
            dtype = "u1"
            pad_arr = np.zeros(len(self.alphabet.array), dtype=dtype)
            pad_arr[self.alphabet == b"N"] = 1

        queries["end"] = queries.start + length
        # map negative starts to 0
        queries["in_start"] = queries.start.clip(lower=0)
        # map ends > contig length to contig length
        queries["contig_length"] = queries.contig.replace(self.contig_lengths).to_numpy(
            np.int64
        )
        queries["in_end"] = np.minimum(queries.end, queries.contig_length)
        # get start, end index in output array
        queries["out_start"] = queries.in_start - queries.start
        queries["out_end"] = queries.in_end - queries.in_start

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
        to_rev_comp = cast(NDArray[np.bool_], (queries.strand == "-").values)
        if encoding is SequenceEncoding.BYTES and to_rev_comp.any():
            out[to_rev_comp] = rev_comp_byte(out[to_rev_comp], self.alphabet)
        elif encoding is SequenceEncoding.ONEHOT:
            out[to_rev_comp] = rev_comp_ohe(out[to_rev_comp], has_N=True)

        return out
