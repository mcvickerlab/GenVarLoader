from collections.abc import Sequence
from typing import Literal, cast

import numpy as np
from attrs import define, evolve
from hirola import HashTable
from numpy.typing import NDArray
from typing_extensions import assert_never

from .._types import Idx, StrIdx
from .._utils import idx_like_to_array, is_dtype


@define
class DatasetIndexer:
    full_region_idxs: NDArray[np.integer]
    """Full map from input region indices to on-disk region indices."""
    full_sample_idxs: NDArray[np.integer]
    """Full map from input sample indices to on-disk sample indices."""
    s2i_map: HashTable
    """Map from input sample names to on-disk sample indices."""
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
    ):
        _samples = np.array(samples)
        s2i_map = HashTable(
            max=len(_samples) * 2,  # type: ignore | 2x size for perf > mem
            dtype=_samples.dtype,
        )
        s2i_map.add(_samples)
        return cls(
            full_region_idxs=r_idxs,
            full_sample_idxs=s_idxs,
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
            return self.full_samples.tolist()  # type: ignore
        return self.full_samples[self.sample_subset_idxs].tolist()  # type: ignore

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
        regions: Idx | None = None,
        samples: StrIdx | None = None,
    ) -> "DatasetIndexer":
        """Subset the dataset to specific regions and/or samples."""
        if regions is None and samples is None:
            return self

        if samples is not None:
            samples = self.s2i(samples)
            sample_idxs = idx_like_to_array(samples, self.n_samples)
        else:
            sample_idxs = np.arange(self.n_samples, dtype=np.intp)

        if regions is not None:
            region_idxs = idx_like_to_array(regions, self.n_regions)
        else:
            region_idxs = np.arange(self.n_regions, dtype=np.intp)

        return evolve(
            self, region_subset_idxs=region_idxs, sample_subset_idxs=sample_idxs
        )

    def to_full_dataset(self) -> "DatasetIndexer":
        """Return a full sized dataset, undoing any subsettting."""
        return evolve(self, region_subset_idxs=None, sample_subset_idxs=None)

    def parse_idx(
        self, idx: Idx | tuple[Idx] | tuple[Idx, StrIdx]
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

        s_idx = self.s2i(samples)
        idx = (regions, s_idx)
        idx_t = idx_type(idx)
        if idx_t == "basic":
            if all(isinstance(i, (int, np.integer)) for i in idx):
                squeeze = True
            r_idx = np.atleast_1d(self._r_idx[regions])
            s_idx = np.atleast_1d(self._s_idx[s_idx])
            idx = np.ravel_multi_index(np.ix_(r_idx, s_idx), self.full_shape).squeeze()
            if isinstance(regions, slice) and isinstance(samples, slice):
                out_reshape = (len(r_idx), len(s_idx))
        elif idx_t == "adv":
            r_idx = self._r_idx[regions]
            s_idx = self._s_idx[s_idx]
            idx = np.ravel_multi_index((r_idx, s_idx), self.full_shape)
        elif idx_t == "combo":
            r_idx = self._r_idx[regions]
            s_idx = self._s_idx[s_idx]
            idx = np.ravel_multi_index(
                np.ix_(r_idx.ravel(), s_idx.ravel()), self.full_shape
            )
            if (
                isinstance(r_idx, np.ndarray)
                and r_idx.ndim > 1
                or isinstance(s_idx, np.ndarray)
                and s_idx.ndim > 1
            ):
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

    def s2i(self, samples: StrIdx) -> Idx:
        """Convert sample names to sample indices."""
        return s2i(samples, self.s2i_map)


def s2i(str_idx: StrIdx, map: HashTable) -> Idx:
    """Convert a string index to an integer index using a hirola.HashTable."""
    if not isinstance(str_idx, (np.ndarray, slice)):
        str_idx = np.asarray(str_idx)

    if is_dtype(str_idx, np.str_) or is_dtype(str_idx, np.object_):
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
    n_adv = sum(map(is_adv_idx, idx))
    if n_adv == 0:
        return "basic"
    elif n_adv == 1:
        return "combo"
    elif n_adv == 2:
        return "adv"
    else:
        raise ValueError(f"Invalid index type: {idx}")


def is_adv_idx(idx: Idx) -> bool:
    """Check if the index is a fancy index."""
    return isinstance(idx, (Sequence, np.ndarray))
