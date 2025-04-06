from __future__ import annotations

from typing import List, Literal, Optional, Sequence, cast

import awkward as ak
import numba as nb
import numpy as np
from attrs import define, evolve
from einops import repeat
from hirola import HashTable
from more_itertools import collapse
from numpy.typing import NDArray
from typing_extensions import Self

from .._ragged import Ragged
from .._types import Idx, StrIdx
from .._utils import _lengths_to_offsets, idx_like_to_array, is_dtype
from ._utils import oidx_to_raveled_idx


@define
class DatasetIndexer:
    full_region_idxs: NDArray[np.integer]
    """Full map from input region indices to on-disk region indices."""
    full_sample_idxs: NDArray[np.integer]
    """Full map from input sample indices to on-disk sample indices."""
    i2d_map: NDArray[np.integer]
    """Map from input indices to on-disk dataset indices."""
    d2i_map: NDArray[np.integer]
    """Map from on-disk dataset indices to input indices."""
    s2i_map: HashTable
    """Map from input sample names to on-disk sample indices."""
    region_subset_idxs: Optional[NDArray[np.integer]] = None
    """Which input regions are included in the subset."""
    sample_subset_idxs: Optional[NDArray[np.integer]] = None
    """Which input samples are included in the subset."""

    @classmethod
    def from_region_and_sample_idxs(
        cls,
        r_idxs: NDArray[np.integer],
        s_idxs: NDArray[np.integer],
        samples: List[str],
    ):
        shape = len(r_idxs), len(s_idxs)
        i2d_map = oidx_to_raveled_idx(r_idxs, s_idxs, shape)
        d2i_map = oidx_to_raveled_idx(np.argsort(r_idxs), s_idxs, shape)
        _samples = np.array(samples)
        s2i_map = HashTable(
            max=len(_samples) * 2,  # type: ignore | 2x size for perf > mem
            dtype=_samples.dtype,
        )
        s2i_map.add(_samples)
        return cls(
            full_region_idxs=r_idxs,
            full_sample_idxs=s_idxs,
            i2d_map=i2d_map,
            d2i_map=d2i_map,
            s2i_map=s2i_map,
        )

    @property
    def full_samples(self) -> NDArray[np.str_]:
        return self.s2i_map.keys

    @property
    def is_subset(self) -> bool:
        return (
            self.region_subset_idxs is not None or self.sample_subset_idxs is not None
        )

    @property
    def n_regions(self) -> int:
        if self.region_subset_idxs is None:
            return len(self.full_region_idxs)
        return len(self.region_subset_idxs)

    @property
    def n_samples(self) -> int:
        if self.sample_subset_idxs is None:
            return len(self.full_sample_idxs)
        return len(self.sample_subset_idxs)

    def region_idxs(self, mode: Literal["i2d", "d2i"]) -> NDArray[np.integer]:
        """Map from input region indices to on-disk region indices or vice versa."""
        if mode == "i2d":
            return np.unravel_index(self.i2d_map[:: self.n_samples], self.full_shape)[0]
        else:
            return np.unravel_index(self.d2i_map[:: self.n_samples], self.full_shape)[0]

    def sample_idxs(self, mode: Literal["i2d", "d2i"]) -> NDArray[np.integer]:
        """Map from input sample indices to on-disk sample indices or vice versa."""
        if mode == "i2d":
            return np.unravel_index(self.i2d_map[: self.n_samples], self.full_shape)[1]
        else:
            return np.unravel_index(self.d2i_map[: self.n_samples], self.full_shape)[1]

    @property
    def samples(self) -> List[str]:
        if self.sample_subset_idxs is None:
            return self.full_samples.tolist()
        return self.full_samples[self.sample_subset_idxs].tolist()

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_regions, self.n_samples

    @property
    def full_shape(self) -> tuple[int, int]:
        return len(self.full_region_idxs), len(self.full_sample_idxs)

    def __len__(self):
        return len(self.i2d_map)

    def subset_to(
        self,
        regions: Optional[Idx] = None,
        samples: Optional[Idx] = None,
    ) -> Self:
        """Subset the dataset to specific regions and/or samples."""
        if regions is None and samples is None:
            return self

        if samples is not None:
            sample_idxs = idx_like_to_array(samples, self.n_samples)
        else:
            sample_idxs = np.arange(self.n_samples, dtype=np.intp)

        if regions is not None:
            region_idxs = idx_like_to_array(regions, self.n_regions)
        else:
            region_idxs = np.arange(self.n_regions, dtype=np.intp)

        idx = oidx_to_raveled_idx(
            row_idx=region_idxs,
            col_idx=sample_idxs,
            shape=self.shape,
        )

        i2d_map = self.i2d_map[idx]

        return evolve(
            self,
            i2d_map=i2d_map,
            region_subset_idxs=region_idxs,
            sample_subset_idxs=sample_idxs,
        )

    def to_full_dataset(self) -> Self:
        """Return a full sized dataset, undoing any subsettting."""
        i2d_map = oidx_to_raveled_idx(
            self.full_region_idxs, self.full_sample_idxs, self.full_shape
        )
        return evolve(
            self,
            i2d_map=i2d_map,
            region_subset_idxs=None,
            sample_subset_idxs=None,
        )

    def parse_idx(
        self, idx: Idx | tuple[Idx] | tuple[Idx, StrIdx]
    ) -> tuple[NDArray[np.integer], bool, tuple[int, ...] | None]:
        if not isinstance(idx, tuple):
            regions = idx
            samples = slice(None)
        elif len(idx) == 1:
            regions = idx[0]
            samples = slice(None)
        else:
            regions, samples = idx

        s_idx = s2i(samples, self.s2i_map)
        idx = self.i2d_map.reshape(self.shape)[regions, s_idx]

        out_reshape = None
        squeeze = False
        if idx.ndim > 1:
            out_reshape = idx.shape
        elif idx.ndim == 0:
            squeeze = True

        idx = idx.ravel()

        return idx, squeeze, out_reshape

    def s2i(self, samples: StrIdx) -> Idx:
        """Convert sample names to sample indices."""
        return s2i(samples, self.s2i_map)


@define
class SpliceIndexer:
    rows: HashTable
    """Map from splice element names to row indices."""
    splice_map: ak.Array
    """Map from splice indices to region indices in splicing order."""
    full_splice_map: ak.Array
    """Non-subset map from splice indices to region indices."""
    dsi: DatasetIndexer
    i2d_map: ak.Array
    """Shape: (rows, samples, ~regions). Map from spliced row/sample indices to on-disk dataset indices."""
    row_subset_idxs: Optional[NDArray[np.intp]] = None
    """Subset of row indices."""

    @classmethod
    def _init(
        cls,
        names: Sequence[str] | NDArray[np.str_],
        splice_map: ak.Array,
        dsi: DatasetIndexer,
    ) -> "SpliceIndexer":
        _names = np.array(names)
        rows = HashTable(
            max=len(names) * 2,  # type: ignore | 2x size for perf > mem
            dtype=_names.dtype,
        )
        rows.add(_names)

        if (
            ak.max(splice_map, None) >= dsi.n_regions
            or ak.min(splice_map, None) < -dsi.n_regions
        ):
            raise ValueError(
                "Found indices in the splice map that are out of bounds for the dataset."
            )

        return cls(
            rows=rows,
            splice_map=splice_map,
            full_splice_map=splice_map,
            dsi=dsi,
            i2d_map=cls.get_i2d_map(splice_map, dsi),
            row_subset_idxs=None,
        )

    @staticmethod
    def get_i2d_map(splice_map: ak.Array, dsi: DatasetIndexer):
        regs_per_row = ak.count(splice_map, -1).to_numpy()
        row_offsets = _lengths_to_offsets(regs_per_row)
        s_idxs = (
            dsi.full_sample_idxs
            if dsi.sample_subset_idxs is None
            else dsi.sample_subset_idxs
        )
        i2d_map = _spliced_i2d_map_helper(
            dsi.i2d_map.reshape(dsi.shape), splice_map, row_offsets, s_idxs
        )
        i2d_map = Ragged.from_lengths(
            i2d_map, repeat(regs_per_row, "r -> r s", s=dsi.n_samples)
        ).to_awkward()
        return i2d_map

    @property
    def n_rows(self) -> int:
        return len(self.splice_map)

    @property
    def n_samples(self) -> int:
        return self.dsi.n_samples

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_rows, self.n_samples

    @property
    def full_shape(self) -> tuple[int, int]:
        return len(self.full_splice_map), len(self.dsi.full_samples)

    def __len__(self):
        return self.n_rows * self.n_samples

    def subset_to(
        self,
        rows: Optional[Idx] = None,
        samples: Optional[Idx] = None,
    ) -> tuple[Self, DatasetIndexer]:
        """Subset to specific regions and/or samples."""
        if rows is None and samples is None:
            return self, self.dsi

        if rows is not None:
            row_idxs = idx_like_to_array(rows, self.n_rows)
        else:
            row_idxs = np.arange(self.n_rows, dtype=np.intp)

        splice_map = cast(ak.Array, self.splice_map[row_idxs])
        # splice_map is to absolute indices so don't subset dsi regions
        sub_dsi = self.dsi.subset_to(samples=samples)
        i2d_map = self.get_i2d_map(splice_map, sub_dsi)
        region_idxs = ak.flatten(splice_map, None).to_numpy()
        eff_dsi = self.dsi.subset_to(regions=region_idxs, samples=samples)

        return evolve(
            self,
            splice_map=splice_map,
            dsi=sub_dsi,
            i2d_map=i2d_map,
            row_subset_idxs=row_idxs,
        ), eff_dsi

    def to_full_dataset(self) -> Self:
        """Return a full sized dataset, undoing any subsettting."""
        return evolve(
            self,
            splice_map=self.full_splice_map,
            dsi=self.dsi.to_full_dataset(),
            row_subset_idxs=None,
        )

    def parse_idx(
        self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]
    ) -> tuple[NDArray[np.integer], bool, tuple[int, ...] | None, NDArray[np.integer]]:
        """Parse the index into a format suitable for indexing.

        Parameters
        ----------
        idx
            The index to parse. This can be a single index, a tuple of indices,
            or a tuple of indices and a list of sample names.

        Returns
        -------
        idx
            1-D raveled dataset indices.
        squeeze
            Whether to squeeze the output.
        out_reshape
            The intended shape of the output, ready to be passed to reshape().
        reducer
            Indices for np.add.reduceat() to get the correct lengths for each splice element. Example:
            spliced_lengths = np.add.reduceat(ragged.lengths, reduce_indices, axis=0)
        rows
            Indices of the splice elements.
        s_idx
            Indices of the samples.
        """
        if not isinstance(idx, tuple):
            rows = idx
            samples = slice(None)
        elif len(idx) == 1:
            rows = idx[0]
            samples = slice(None)
        else:
            rows, samples = idx

        rows = s2i(rows, self.rows)
        samples = s2i(samples, self.dsi.s2i_map)

        ds_idx = cast(ak.Array, self.i2d_map[rows, samples])
        out_reshape = tuple(map(int, ds_idx.typestr.split(" * ")[:-2]))
        squeeze = False
        if len(out_reshape) == 1:
            out_reshape = None
        elif out_reshape == ():
            out_reshape = None
            squeeze = True

        lengths = ak.count(ds_idx, -1)
        if not isinstance(lengths, np.integer):
            lengths = lengths.to_numpy()
        lengths = cast(NDArray[np.int64], lengths)
        reducer = _lengths_to_offsets(lengths)[:-1]
        ds_idx = ak.flatten(ds_idx, None).to_numpy()

        return ds_idx, squeeze, out_reshape, reducer

    def r2i(self, regions: StrIdx) -> Idx:
        """Convert region names to region indices."""
        return s2i(regions, self.rows)

    def s2i(self, samples: StrIdx) -> Idx:
        """Convert sample names to sample indices."""
        return s2i(samples, self.dsi.s2i_map)


def s2i(str_idx: StrIdx, map: HashTable) -> Idx:
    """Convert a string index to an integer index using a hirola.HashTable."""
    if (
        isinstance(str_idx, str)
        or (isinstance(str_idx, np.ndarray) and is_dtype(str_idx, np.str_))
        or (isinstance(str_idx, Sequence) and isinstance(next(collapse(str_idx)), str))
    ):
        idx = map.get(str_idx)
        if (np.atleast_1d(idx) == -1).any():
            raise KeyError(
                f"Some keys not found in mapping: {np.unique(np.array(str_idx)[idx == -1])}"
            )
    else:
        idx = str_idx

    idx = cast(Idx, idx)  # above clause does this, but can't narrow type

    return idx


@nb.njit(nogil=True, cache=True)
def _spliced_i2d_map_helper(
    i2d_map: NDArray[np.integer],
    sp_map: ak.Array,
    row_offsets: NDArray[np.int64],
    s_idxs: NDArray[np.integer],
):
    n_samples = len(s_idxs)
    # (rows samples ~regions)
    out = np.empty(row_offsets[-1] * n_samples, dtype=np.int32)
    for row, r_idxs in enumerate(sp_map):
        for r_idx in r_idxs:
            for s_idx in s_idxs:
                out[
                    row_offsets[row] * (n_samples - 1) + s_idx * (len(r_idxs)) + r_idx
                ] = i2d_map[r_idx, s_idx]
    return out
