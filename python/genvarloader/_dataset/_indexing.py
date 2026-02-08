from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, cast

import awkward as ak
import numpy as np
from attrs import define, evolve
from hirola import HashTable
from numpy.typing import NDArray
from typing_extensions import Self, TypeGuard, assert_never

from .._types import Idx, StrIdx
from .._utils import idx_like_to_array, is_dtype, lengths_to_offsets


@define
class DatasetIndexer:
    full_region_idxs: NDArray[np.integer]
    """Full map from input region indices to on-disk region indices."""
    full_sample_idxs: NDArray[np.integer]
    """Full map from input sample indices to on-disk sample indices."""
    s2i_map: HashTable
    """Map from input sample names to on-disk sample indices."""
    r2i_map: HashTable | None = None
    """Map from input region names to on-disk region indices."""
    region_subset_idxs: NDArray[np.integer] | None = None
    """Which input regions are included in the subset."""
    sample_subset_idxs: NDArray[np.integer] | None = None
    """Which input samples are included in the subset."""

    @classmethod
    def from_region_and_sample_idxs(
        cls,
        r_idxs: NDArray[np.integer],
        s_idxs: NDArray[np.integer],
        samples: list[str],
        regions: list[str] | None = None,
    ):
        if regions is not None:
            _regions = np.array(regions)
            r2i_map = HashTable(
                max=len(_regions) * 2,  # type: ignore | 2x size for perf > mem
                dtype=_regions.dtype,
            )
            r2i_map.add(_regions)
        else:
            r2i_map = None

        _samples = np.array(samples)
        s2i_map = HashTable(
            max=len(_samples) * 2,  # type: ignore | 2x size for perf > mem
            dtype=_samples.dtype,
        )
        s2i_map.add(_samples)

        return cls(
            full_region_idxs=r_idxs,
            full_sample_idxs=s_idxs,
            r2i_map=r2i_map,
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

    @property
    def samples(self) -> list[str]:
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
        return np.prod(self.shape)

    def subset_to(
        self,
        regions: StrIdx | None = None,
        samples: StrIdx | None = None,
    ) -> "DatasetIndexer":
        """Subset the dataset to specific regions and/or samples."""
        if regions is None and samples is None:
            return self

        if samples is not None:
            samples = self.sample2idx(samples)
            sample_idxs = idx_like_to_array(samples, self.n_samples)
        else:
            sample_idxs = np.arange(self.n_samples, dtype=np.intp)

        if regions is not None:
            regions = self.region2idx(regions)
            region_idxs = idx_like_to_array(regions, self.n_regions)
        else:
            region_idxs = np.arange(self.n_regions, dtype=np.intp)

        return evolve(
            self, region_subset_idxs=region_idxs, sample_subset_idxs=sample_idxs
        )

    def to_full_dataset(self) -> Self:
        """Return a full sized dataset, undoing any subsettting."""
        return evolve(self, region_subset_idxs=None, sample_subset_idxs=None)

    def parse_idx(
        self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]
    ) -> tuple[NDArray[np.integer], bool, tuple[int, ...] | None]:
        out_reshape = None
        squeeze = False

        if not isinstance(idx, tuple):
            regions = idx
            samples = slice(None)
        elif len(idx) == 1:
            regions = idx[0]
            samples = slice(None)
        else:
            regions, samples = idx

        r_idx = self.region2idx(regions)
        s_idx = self.sample2idx(samples)
        idx = (r_idx, s_idx)
        idx_t = idx_type(idx)
        if idx_t == "basic":
            if all(isinstance(i, (int, np.integer)) for i in idx):
                squeeze = True
            r_idx = np.atleast_1d(self._r_idx[r_idx])
            s_idx = np.atleast_1d(self._s_idx[s_idx])
            idx = np.ravel_multi_index(np.ix_(r_idx, s_idx), self.full_shape).squeeze()
            if isinstance(regions, slice) and isinstance(samples, slice):
                out_reshape = (len(r_idx), len(s_idx))
        elif idx_t == "adv":
            r_idx = self._r_idx[r_idx]
            s_idx = self._s_idx[s_idx]
            idx = np.ravel_multi_index((r_idx, s_idx), self.full_shape)
        elif idx_t == "combo":
            r_idx = self._r_idx[r_idx]
            s_idx = self._s_idx[s_idx]
            idx = np.ravel_multi_index(
                np.ix_(r_idx.ravel(), s_idx.ravel()), self.full_shape
            )
            if r_idx.ndim > 1 or s_idx.ndim > 1:
                out_reshape = (*r_idx.shape, *s_idx.shape)
            elif idx.ndim > 1:
                out_reshape = idx.shape
        else:
            assert_never(idx_t)

        if idx_t != "combo" and idx.ndim > 1:
            out_reshape = idx.shape
        idx = idx.ravel()

        return idx, squeeze, out_reshape

    @property
    def _r_idx(self):
        if self.region_subset_idxs is None:
            return self.full_region_idxs
        return self.full_region_idxs[self.region_subset_idxs]

    @property
    def _s_idx(self):
        if self.sample_subset_idxs is None:
            return self.full_sample_idxs
        return self.full_sample_idxs[self.sample_subset_idxs]

    def sample2idx(self, samples: StrIdx) -> Idx:
        """Convert sample names to sample indices."""
        return s2i(samples, self.s2i_map)

    def region2idx(self, regions: StrIdx) -> Idx:
        """Convert region names to region indices."""
        return s2i(regions, self.r2i_map)


@define
class SpliceIndexer:
    rows: HashTable
    """Map from splice element names to row indices."""
    splice_map: ak.Array
    """Map from splice indices to region indices in splicing order."""
    full_splice_map: ak.Array
    """Non-subset map from splice indices to region indices."""
    dsi: DatasetIndexer
    row_subset_idxs: NDArray[np.intp] | None = None
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
            row_subset_idxs=None,
        )

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
        rows: StrIdx | None = None,
        samples: StrIdx | None = None,
    ) -> tuple[Self, DatasetIndexer]:
        """Subset to specific regions and/or samples."""
        if rows is None and samples is None:
            return self, self.dsi

        if rows is not None:
            row_idxs = self.r2i(rows)
        else:
            row_idxs = np.arange(self.n_rows, dtype=np.intp)

        splice_map = cast(ak.Array, self.splice_map[row_idxs])
        # splice_map is to absolute indices so don't subset dsi regions
        sub_dsi = self.dsi.subset_to(samples=samples)
        region_idxs = ak.flatten(splice_map, None).to_numpy()  # type: ignore
        eff_dsi = self.dsi.subset_to(regions=region_idxs, samples=samples)

        return evolve(
            self,
            splice_map=splice_map,
            dsi=sub_dsi,
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

    def parse_idx(self, idx: StrIdx | tuple[StrIdx] | tuple[StrIdx, StrIdx]):
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
        out_reshape = None
        squeeze = False

        if not isinstance(idx, tuple):
            rows = idx
            samples = slice(None)
        elif len(idx) == 1:
            rows = idx[0]
            samples = slice(None)
        else:
            rows, samples = idx

        rows = self.r2i(rows)
        samples = self.s2i(samples)

        if isinstance(rows, (int, np.integer)) and isinstance(
            samples, (int, np.integer)
        ):
            squeeze = True

        r_idx = idx_like_to_array(rows, self.n_rows)
        s_idx = idx_like_to_array(samples, self.n_samples)

        idx_t = idx_type((r_idx, s_idx))
        if idx_t == "basic":
            # * FYI this will never execute because idx type is guaranteed to be adv or combo by casting
            # basic indices to arrays above
            idx = np.ravel_multi_index(np.ix_(r_idx, s_idx), self.full_shape)
        elif idx_t == "adv":
            idx = np.ravel_multi_index((r_idx, s_idx), self.full_shape)
        elif idx_t == "combo":
            idx = np.ravel_multi_index(
                np.ix_(r_idx.ravel(), s_idx.ravel()), self.full_shape
            )
            if squeeze:
                pass
            elif r_idx.ndim > 1 or s_idx.ndim > 1:
                out_reshape = (*r_idx.shape, *s_idx.shape)
            elif idx.ndim > 1:
                out_reshape = idx.shape
        else:
            assert_never(idx_t)

        if idx_t != "combo" and idx.ndim > 1:
            out_reshape = idx.shape
        idx = idx.ravel()
        (
            r_idx,
            s_idx,
        ) = np.unravel_index(idx, self.full_shape)

        r_idx = self.splice_map[r_idx]
        lengths = ak.count(r_idx, -1)
        if not isinstance(lengths, np.integer):
            lengths = lengths.to_numpy()
        lengths = cast(NDArray[np.int64], lengths)
        offsets = lengths_to_offsets(lengths)
        r_idx = ak.flatten(r_idx, -1).to_numpy()
        s_idx = s_idx.repeat(lengths)

        ds_idx, *_ = self.dsi.parse_idx((r_idx, s_idx))

        return ds_idx, squeeze, out_reshape, offsets

    def r2i(self, regions: StrIdx) -> Idx:
        """Convert region names to region indices."""
        return s2i(regions, self.rows)

    def s2i(self, samples: StrIdx) -> Idx:
        """Convert sample names to sample indices."""
        return s2i(samples, self.dsi.s2i_map)


def s2i(str_idx: StrIdx, map: HashTable | None) -> Idx:
    """Convert a string index to an integer index using a hirola.HashTable."""
    if not isinstance(str_idx, (np.ndarray, slice)):
        str_idx = np.asarray(str_idx)

    if is_str_arr(str_idx):
        if map is None:
            raise ValueError(
                "Queries are names/strings, but no string-to-integer mapping is available."
            )

        idx = map.get(str_idx)
        if (np.atleast_1d(idx) == -1).any():
            raise KeyError(
                f"Some keys not found in mapping: {np.unique(str_idx[idx == -1])}"
            )
    else:
        idx = str_idx

    if isinstance(idx, np.ndarray) and idx.ndim == 0:
        idx = idx.item()

    idx = cast(Idx, idx)

    return idx


def idx_type(
    idx: Idx | tuple[Idx] | tuple[Idx, Idx],
) -> Literal["basic", "adv", "combo"]:
    """Check if the index is a fancy index."""
    if not isinstance(idx, tuple):
        idx = (idx,)
    n_adv = sum(map(lambda idx: isinstance(idx, (Sequence, np.ndarray)), idx))
    if n_adv == 0:
        return "basic"
    elif n_adv == 1:
        return "combo"
    elif n_adv == 2:
        return "adv"
    else:
        raise ValueError(f"Invalid index type: {idx}")


def is_str_arr(obj: Any) -> TypeGuard[NDArray[np.str_] | NDArray[np.object_]]:
    """Check if the object is a string array."""
    return is_dtype(obj, np.str_) or is_dtype(obj, np.object_)
