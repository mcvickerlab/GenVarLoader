from __future__ import annotations

from typing import cast, overload

import awkward as ak
import numpy as np
import polars as pl
from attrs import define, evolve
from awkward.contents import ListOffsetArray, NumpyArray
from awkward.index import Index64
from hirola import HashTable
from numpy.typing import NDArray
from seqpro.rag import DTYPE_co as DTYPE, Ragged, is_rag_dtype
from typing_extensions import assert_never

from typing_extensions import Self

from .._ragged import RaggedAnnotatedHaps
from .._types import Idx, StrIdx
from .._utils import lengths_to_offsets
from ._indexing import s2i


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


@define
class SpliceMap:
    """Sample-agnostic mapping from splice row → ordered region indices.

    Owns the parsed splice BED, the name → row hash table, the awkward
    splice-map (rows → list[region_idx]), and any active row subset. Used by
    both `Dataset` (via `SpliceIndexer`) and `RefDataset`.
    """

    names: HashTable
    splice_map: ak.Array
    full_splice_map: ak.Array
    row_idxs: NDArray[np.intp]
    row_subset_idxs: NDArray[np.intp] | None = None

    @classmethod
    def from_bed(
        cls,
        splice_info: str | tuple[str, str],
        full_bed: pl.DataFrame,
    ) -> tuple["SpliceMap", pl.DataFrame]:
        """Parse splice_info into a (SpliceMap, spliced_bed) pair. Pure — no sampler."""
        if isinstance(splice_info, str):
            sp_bed = (
                full_bed.rename({splice_info: "splice_id"})
                .with_row_index()
                .group_by("splice_id", maintain_order=True)
                .agg(pl.all())
            )
        elif isinstance(splice_info, tuple):
            if len(splice_info) != 2:
                raise ValueError(
                    "Splice info tuple must be of length 2, corresponding to columns "
                    "names for splice IDs and element ordering."
                )
            sp_bed = (
                full_bed.rename({splice_info[0]: "splice_id"})
                .with_row_index()
                .group_by("splice_id", maintain_order=True)
                .agg(pl.all().sort_by(splice_info[1]))
            )
        else:
            assert_never(splice_info)

        names = sp_bed["splice_id"].to_numpy().astype(np.str_)
        lengths = sp_bed["index"].list.len().to_numpy()
        splice_map = Ragged.from_lengths(
            sp_bed["index"].explode().to_numpy(), lengths
        ).to_ak()
        splice_map = cast(ak.Array, splice_map)

        rows = HashTable(max=len(names) * 2, dtype=names.dtype)  # type: ignore
        rows.add(names)

        return (
            cls(
                names=rows,
                splice_map=splice_map,
                full_splice_map=splice_map,
                row_idxs=np.arange(len(splice_map), dtype=np.intp),
                row_subset_idxs=None,
            ),
            sp_bed,
        )

    @property
    def n_rows(self) -> int:
        return len(self.splice_map)

    @property
    def _r_idx(self) -> NDArray[np.intp]:
        if self.row_subset_idxs is None:
            return self.row_idxs
        return self.row_subset_idxs

    def row2idx(self, rows: StrIdx) -> Idx:
        """Convert row names (or already-int indices) to int indices."""
        return s2i(rows, self.names)

    def subset_to(self, rows: StrIdx | None) -> Self:
        """Return a new SpliceMap restricted to the given rows."""
        if rows is None:
            return self
        row_idxs = self._r_idx[self.row2idx(rows)]
        splice_map = cast(ak.Array, self.full_splice_map[row_idxs])
        return evolve(self, splice_map=splice_map, row_subset_idxs=row_idxs)

    def to_full(self) -> Self:
        """Reset to the un-subsetted splice map."""
        return evolve(
            self, splice_map=self.full_splice_map, row_subset_idxs=None
        )

    def parse_rows(
        self, rows: Idx | StrIdx
    ) -> tuple[NDArray[np.intp], NDArray[np.int64], tuple[int, ...] | None, bool]:
        """Parse a row index into the inputs needed for a per-region fetch.

        Returns
        -------
        flat_region_idxs
            1-D region indices to feed the unspliced reader.
        offsets
            For ``np.add.reduceat``-style concat (len == n_selected_rows + 1).
        out_reshape
            Target shape for fancy/combo indexing, or ``None``.
        squeeze
            Whether to squeeze the row dim out (scalar index).
        """
        out_reshape = None
        squeeze = False

        r_idx_raw = self.row2idx(rows)
        if isinstance(r_idx_raw, (int, np.integer)):
            squeeze = True
            local = np.atleast_1d(np.asarray(r_idx_raw, dtype=np.intp))
        elif isinstance(r_idx_raw, slice):
            local = np.arange(self.n_rows, dtype=np.intp)[r_idx_raw]
        else:
            local = np.asarray(r_idx_raw)
            if local.ndim > 1:
                out_reshape = local.shape
            local = local.ravel().astype(np.intp)

        abs_idx = self._r_idx[local]
        sel = cast(ak.Array, self.full_splice_map[abs_idx])
        lengths = ak.count(sel, -1)
        if not isinstance(lengths, np.integer):
            lengths = lengths.to_numpy()
        lengths = cast(NDArray[np.int64], lengths)
        offsets = lengths_to_offsets(lengths)
        flat_region_idxs = ak.flatten(sel, -1).to_numpy().astype(np.intp)

        return flat_region_idxs, offsets, out_reshape, squeeze
