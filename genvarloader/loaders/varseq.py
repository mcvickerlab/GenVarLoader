import asyncio
from typing import Dict, Set, Union, cast

import numba
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pyfaidx import Fasta, FastaVariant

from genvarloader.loaders.sequence import Sequence
from genvarloader.loaders.variants import Variants
from genvarloader.types import ALPHABETS, PathType, SequenceAlphabet, SequenceEncoding
from genvarloader.utils import bytes_to_ohe, rev_comp_byte


@numba.njit(nogil=True, parallel=True)
def apply_variants(
    seqs: NDArray[np.bytes_],
    starts: NDArray[np.integer],
    variants: NDArray[np.bytes_],
    positions: NDArray[np.integer],
    offsets: NDArray[np.unsignedinteger],
):
    # shapes:
    # seqs (i l)
    # starts (i)
    # variants (v)
    # positions (v)
    # offsets (i+1)

    for i in numba.prange(len(seqs)):
        i_vars = variants[offsets[i] : offsets[i + 1]]
        i_pos = positions[offsets[i] : offsets[i + 1]] - starts[i]
        seq = seqs[i]
        seq[i_pos] = i_vars


class VarSequence:
    sequence: Sequence
    variants: Variants
    encodings: Set[SequenceEncoding]
    alphabet: SequenceAlphabet

    def __init__(
        self,
        sequence: Sequence,
        variants: Variants,
    ) -> None:
        self.sequence = sequence
        self.variants = variants
        self.encodings = self.sequence.encodings
        self.alphabet = self.sequence.alphabet

    def sel(
        self,
        queries: pd.DataFrame,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences with sample's variants applied to them.

        Parameters
        ----------
        queries : pd.DataFrame
        length : int
        **kwargs : dict
            encoding : 'bytes' or 'onehot', required
                How to encode the sequences.

        Returns
        -------
        seqs : ndarray[bytes | uint8]
            Sequences with variants applied to them.
        """
        seqs = asyncio.run(self.async_sel(queries, length, **kwargs))
        return seqs

    async def async_sel(
        self,
        queries: pd.DataFrame,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences with sample's variants applied to them.

        Parameters
        ----------
        queries : pd.DataFrame
        length : int
        **kwargs : dict
            sorted : bool, default False
            encoding : 'bytes' or 'onehot', required
                How to encode the sequences.

        Returns
        -------
        seqs : ndarray[bytes | uint8]
            Sequences with variants applied to them.
        """
        encoding = SequenceEncoding(kwargs.get("encoding"))
        positive_stranded_queries = queries.assign(strand="+")
        positive_stranded_queries["strand"] = positive_stranded_queries.strand.astype(
            "category"
        )

        # apply variants as bytes to reduce how much data is moving around
        # S1 is the same size as uint8
        res: Dict[str, NDArray]
        seqs, res = await asyncio.gather(
            *[
                self.sequence.async_sel(
                    positive_stranded_queries, length, encoding="bytes"
                ),
                self.variants.async_sel(queries, length),
            ]
        )
        seqs = cast(NDArray[np.bytes_], seqs)

        if res["alleles"].size > 0:
            apply_variants(
                seqs,
                queries.start.to_numpy(),
                res["alleles"],
                res["positions"],
                res["offsets"],
            )

        if "strand" in queries:
            to_rev_comp = cast(NDArray[np.bool_], (queries["strand"] == "-").to_numpy())
            if to_rev_comp.any():
                seqs[to_rev_comp] = rev_comp_byte(
                    seqs[to_rev_comp], alphabet=ALPHABETS["DNA"]
                )

        if encoding is SequenceEncoding.ONEHOT:
            seqs = cast(NDArray[np.uint8], bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"]))  # type: ignore

        return seqs


class _PyfaidxVarSequence:
    """Always returns byte arrays."""

    def __init__(self, reference_path: PathType, vcfs: Dict[str, PathType]) -> None:
        self.reference = str(reference_path)
        self.vcfs = {k: str(v) for k, v in vcfs.items()}
        with Fasta(self.reference) as ref:
            self.contig_lengths: Dict[str, int] = {k: len(v) for k, v in ref.items()}

    def sel(self, queries: pd.DataFrame, length: int, **kwargs) -> NDArray[np.bytes_]:
        if "strand" not in queries:
            queries["strand"] = "+"
            queries["strand"] = queries.strand.astype("category")

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

        groups = queries.groupby("sample", sort=False)
        out = np.full((len(queries), length), np.array(b"N"), dtype="|S1")  # type: ignore
        for sample, group in groups:
            with FastaVariant(self.reference, self.vcfs[sample], het=True, hom=True) as f:  # type: ignore
                for query in group.itertuples(index=True):
                    string_out = f[query.contig][query.in_start : query.in_end].seq
                    out[query[0], query.out_start : query.out_end] = np.frombuffer(
                        string_out, dtype="|S1"
                    )

        # reverse complement negative stranded queries
        to_rev_comp = cast(NDArray[np.bool_], (queries.strand == "-").values)
        out[to_rev_comp] = rev_comp_byte(out[to_rev_comp], ALPHABETS["DNA"])

        return out
