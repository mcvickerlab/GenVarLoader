from __future__ import annotations

from typing import cast

import awkward as ak
import numpy as np
import polars as pl
from dataclasses import dataclass, replace
from hirola import HashTable
from numpy.typing import NDArray
from seqpro.rag import Ragged
from typing_extensions import Self, assert_never

from .._types import Idx, StrIdx
from .._utils import lengths_to_offsets
from ._indexing import s2i


@dataclass(slots=True)
class SplicePlan:
    """Permutation + offsets that re-target the kernel write into spliced layout.

    The kernel is called with ``ploidy=1`` and one query per element of the
    flattened ``(B, *inner_fixed)`` length array. ``perm`` reorders those
    flattened k-indices so the global write order becomes
    ``(splice_row, sample, *inner_fixed, splice_element)`` C-order. After the
    kernel writes, the data buffer can be exposed as a Ragged with either
    ``permuted_out_offsets`` (per-element) or ``group_offsets`` (per
    ``(splice_row, sample, inner)`` cell).
    """

    perm: NDArray[np.intp]
    permuted_lengths: NDArray[np.int32]
    permuted_out_offsets: NDArray[np.int64]
    group_offsets: NDArray[np.int64]
    out_shape: tuple[int | None, ...]

    @property
    def flat_out_shape(self) -> tuple[int | None, ...]:
        """Shape with the leading ``(n_rows, n_samples)`` axes collapsed to ``n_pairs``.

        ``out_shape`` is ``(n_rows, n_samples, *inner_fixed, None)``.
        ``flat_out_shape`` is ``(n_rows * n_samples, *inner_fixed, None)``.

        This is the shape passed to ``_regroup`` after the kernel writes
        per-element data into the flattened ``(n_pairs, ...)`` layout.
        Future callers (e.g. a Tracks splice path) should use this property
        rather than recomputing the flattening inline.
        """
        n_rows, n_samples = self.out_shape[0], self.out_shape[1]
        assert isinstance(n_rows, int) and isinstance(n_samples, int)
        return (n_rows * n_samples, *self.out_shape[2:])


def build_splice_plan(
    lengths: NDArray[np.int32],
    splice_row_offsets: NDArray[np.int64],
    n_samples: int,
    n_rows: int,
) -> SplicePlan:
    """Build a splice plan from per-query lengths and splice-row boundaries.

    Parameters
    ----------
    lengths
        Shape ``(B, *inner_fixed)``. Per-query lengths in current ``(splice_row,
        sample, splice_element)`` C-order, with any inner fixed axes (ploidy,
        tracks) intact. ``E = prod(inner_fixed)`` is the inner flatten factor.
    splice_row_offsets
        Shape ``(n_rows * n_samples + 1,)``. Cumulative count of elements per
        ``(splice_row, sample)`` pair — i.e. the ``offsets`` returned by
        ``SpliceIndexer.parse_idx``.
    n_samples
        Number of samples in the outer ``(splice_row, sample)`` grid.
    n_rows
        Number of splice rows in the outer ``(splice_row, sample)`` grid.
    """
    if lengths.ndim == 1:
        inner_fixed: tuple[int, ...] = ()
        flat_lengths = lengths.astype(np.int32, copy=False)
    else:
        inner_fixed = tuple(lengths.shape[1:])
        # (B, *inner) -> (B, E) -> (B*E,) in (query, inner) C-order.
        flat_lengths = lengths.reshape(lengths.shape[0], -1).astype(
            np.int32, copy=False
        )
    E = int(np.prod(inner_fixed)) if inner_fixed else 1
    B = int(lengths.shape[0])
    # k-index in the current layout: k = query * E + e.
    # We want to permute into (row, sample, e, element) C-order, which means:
    #   for each (row, sample) pair p (in C-order):
    #     for each e in 0..E:
    #       for each element q in the pair's element range:
    #         emit k = q * E + e
    # The element range for pair p is splice_row_offsets[p]:splice_row_offsets[p+1].
    n_pairs = n_rows * n_samples
    pair_lengths = np.diff(splice_row_offsets)  # length n_pairs
    if E == 1:
        # Identity permutation; flat_lengths shape is (B,) already permuted.
        perm = np.arange(B, dtype=np.intp)
        permuted_lengths_flat = flat_lengths.reshape(-1).astype(np.int32, copy=False)
    else:
        # Build perm by iterating (pair, e, element).
        # For a pair p with element range [s, s+L):
        #   for e in 0..E:
        #     k-indices = [(s+0)*E + e, (s+1)*E + e, ..., (s+L-1)*E + e]
        # Vectorized: outer product of "queries within pair" and a per-e offset.
        # Build with broadcasting.
        flat_2d = flat_lengths  # (B, E)
        perm_parts = []
        for p_idx in range(n_pairs):
            s = int(splice_row_offsets[p_idx])
            L = int(pair_lengths[p_idx])
            if L == 0:
                continue
            q_range = np.arange(s, s + L, dtype=np.intp)  # (L,)
            # (E, L): each row e is q_range*E + e.
            ke = q_range[None, :] * E + np.arange(E, dtype=np.intp)[:, None]
            perm_parts.append(ke.reshape(-1))
        perm = np.concatenate(perm_parts) if perm_parts else np.empty(0, dtype=np.intp)
        permuted_lengths_flat = flat_2d.reshape(-1)[perm].astype(np.int32, copy=False)

    permuted_out_offsets = lengths_to_offsets(permuted_lengths_flat, dtype=np.int64)

    # group_offsets at (row, sample, *inner_fixed) granularity:
    # each cell aggregates L elements (or 0 for empty pairs).
    # Within the permuted layout, cells are laid out as: for each pair p, E
    # cells of L lengths back-to-back. So the cell-boundary indices in the
    # flat permuted_lengths array are:
    #   pair_offsets[p]*E + e*L_p     for e in 0..E
    # Equivalently: take pair_lengths repeated E times then cumsum.
    if E == 1:
        cell_lengths = pair_lengths.astype(np.int64, copy=False)
    else:
        cell_lengths = np.repeat(pair_lengths.astype(np.int64), E)
    # cell_lengths length = n_pairs * E. group_offsets indexes the
    # *permuted_lengths* array at cell boundaries.
    cell_starts = np.concatenate(
        ([0], np.cumsum(cell_lengths, dtype=np.int64))
    )  # length n_pairs*E + 1
    # group_offsets[i] = permuted_out_offsets[cell_starts[i]]
    group_offsets = permuted_out_offsets[cell_starts]

    if inner_fixed:
        out_shape: tuple[int | None, ...] = (n_rows, n_samples, *inner_fixed, None)
    else:
        out_shape = (n_rows, n_samples, None)

    return SplicePlan(
        perm=perm,
        permuted_lengths=permuted_lengths_flat,
        permuted_out_offsets=permuted_out_offsets,
        group_offsets=group_offsets,
        out_shape=out_shape,
    )


@dataclass(slots=True)
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
        return replace(self, splice_map=splice_map, row_subset_idxs=row_idxs)

    def to_full(self) -> Self:
        """Reset to the un-subsetted splice map."""
        return replace(self, splice_map=self.full_splice_map, row_subset_idxs=None)

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
