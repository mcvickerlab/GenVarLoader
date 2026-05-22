from __future__ import annotations

from typing import overload

import awkward as ak
import numpy as np
from awkward.contents import ListOffsetArray, NumpyArray
from awkward.index import Index64
from numpy.typing import NDArray
from seqpro.rag import DTYPE_co as DTYPE, Ragged, is_rag_dtype
from typing_extensions import assert_never

from .._ragged import RaggedAnnotatedHaps


@overload
def _cat_length(rag: Ragged[DTYPE], offsets: NDArray[np.integer]) -> Ragged[DTYPE]: ...
@overload
def _cat_length(
    rag: RaggedAnnotatedHaps, offsets: NDArray[np.integer]
) -> RaggedAnnotatedHaps: ...
def _cat_length(
    rag: Ragged | RaggedAnnotatedHaps, offsets: NDArray[np.integer]
) -> Ragged | RaggedAnnotatedHaps:
    """Concatenate the lengths of the ragged data.

    ``offsets`` groups contiguous ranges along the outer (batch) axis. Every
    other fixed dimension (e.g. ploidy, tracks) is preserved; we merge
    bytes/elements across the grouped batch slots within each inner index.

    Branch on ``shape[:-1]`` (the fixed axes; ``shape[-1]`` is always ``None``
    for the ragged axis) instead of ``Ragged.ndim``, which counts bytestring
    dtypes differently from numeric dtypes.

    The fast ``np.add.reduceat`` path only works when the data buffer is
    already laid out in batch order — i.e. a single fixed dim, or every inner
    fixed dim == 1. Otherwise the buffer is interleaved by the inner axes and
    walking it sequentially produces wrong bytes (pre-fix ploid-1 corruption).
    """
    if isinstance(rag, Ragged):
        fixed = rag.shape[:-1]  # the non-ragged axes
        inner_is_trivial = all(s == 1 for s in fixed[1:])
        if len(fixed) == 1 or inner_is_trivial:
            new_lengths = np.add.reduceat(rag.lengths, offsets[:-1], 0)
            cat = Ragged.from_lengths(rag.data, new_lengths)
        elif len(fixed) == 2:
            # (b, p, ~l) or (b, t, ~l) — concatenate bytes across the grouped
            # batch slots, per inner index, using awkward.
            cat = _cat_length_inner(
                rag, offsets, is_bytestring=is_rag_dtype(rag, np.bytes_)
            )
        else:  # hap tracks: (b, t, p, ~l) or deeper
            raise NotImplementedError(
                f"Splicing with shape {rag.shape} (≥3 fixed axes) is not implemented."
            )

        if is_rag_dtype(rag, np.bytes_):
            cat = cat.view("S1")  # type: ignore
        return cat
    elif isinstance(rag, RaggedAnnotatedHaps):
        haps = _cat_length(rag.haps, offsets)
        var_idxs = _cat_length(rag.var_idxs, offsets)
        ref_coords = _cat_length(rag.ref_coords, offsets)
        return RaggedAnnotatedHaps(haps, var_idxs, ref_coords)
    else:
        assert_never(rag)


def _cat_length_inner(
    rag: Ragged, offsets: NDArray[np.integer], is_bytestring: bool
) -> Ragged:
    """Per-inner-axis bytes concatenation for (b, inner, ~l) Ragged arrays.

    Groups the batch axis via ``offsets`` and, for each inner index, flattens
    the grouped batch slots' bytes into a single run. Preserves the inner
    axis. Required because the data buffer is laid out in (batch, inner)
    interleaved order, so a naive reduceat walks the wrong bytes.
    """
    inner = rag.shape[1]
    grouped = ak.Array(ListOffsetArray(Index64(offsets), rag.to_ak().layout))
    parts = []
    for i in range(inner):  # type: ignore
        sel = grouped[:, :, i]
        if is_bytestring:
            # Bytestrings are atomic w.r.t. ak.flatten; strip the parameter to
            # expose them as a list of uint8, flatten batch + bytes together,
            # then re-wrap the inner axis as a bytestring.
            sel_raw = ak.without_parameters(sel)
            flat = ak.flatten(sel_raw, -1)
            inner_np = flat.layout.content  # type: ignore
            inner_np = NumpyArray(
                inner_np.data.view(np.uint8),  # type: ignore
                parameters={"__array__": "byte"},
            )
            wrapped = ListOffsetArray(
                flat.layout.offsets,  # type: ignore
                inner_np,
                parameters={"__array__": "bytestring"},
            )
            part = ak.Array(wrapped)
        else:
            part = ak.flatten(sel, -1)
        parts.append(part[:, None])
    return Ragged(ak.concatenate(parts, 1))  # type: ignore
