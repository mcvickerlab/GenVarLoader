import asyncio
from typing import Dict, Optional, Tuple, Union, cast

import numba
import numpy as np
from numpy.typing import NDArray

from genvarloader.loaders.sequence import Sequence
from genvarloader.loaders.types import Queries
from genvarloader.loaders.variants import Variants
from genvarloader.types import ALPHABETS, SequenceEncoding
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
    def __init__(
        self,
        sequence: Sequence,
        variants: Variants,
    ) -> None:
        self.sequence = sequence
        self.variants = variants

    def sel(
        self,
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences with sample's variants applied to them.

        Parameters
        ----------
        queries : Queries
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
        queries: Queries,
        length: int,
        **kwargs,
    ) -> Union[NDArray[np.bytes_], NDArray[np.uint8]]:
        """Get sequences with sample's variants applied to them.

        Parameters
        ----------
        queries : Queries
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
        if "strand" not in queries:
            queries["strand"] = "+"
            queries["strand"] = queries.strand.astype("category")

        encoding = SequenceEncoding(kwargs.get("encoding"))
        positive_stranded_queries = cast(Queries, queries.assign(strand="+"))
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

        to_rev_comp = cast(NDArray[np.bool_], (queries["strand"] == "-").values)
        if to_rev_comp.any():
            seqs[to_rev_comp] = rev_comp_byte(
                seqs[to_rev_comp], alphabet=ALPHABETS["DNA"]
            )

        if encoding is SequenceEncoding.ONEHOT:
            seqs = cast(NDArray[np.uint8], bytes_to_ohe(seqs, alphabet=ALPHABETS["DNA"]))  # type: ignore

        return seqs


# TODO: implement for benchmarking
class _PyfaidxVarSequence:
    raise NotImplementedError
