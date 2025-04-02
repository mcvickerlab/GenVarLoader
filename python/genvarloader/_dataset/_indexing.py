from itertools import chain
from typing import List, Literal, Optional, Sequence, cast

import numpy as np
from attrs import define, evolve
from hirola import HashTable
from numpy.typing import NDArray

from genvarloader._dataset._utils import oidx_to_raveled_idx
from genvarloader._types import Idx
from genvarloader._utils import idx_like_to_array, is_dtype


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
    ) -> "DatasetIndexer":
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

    def to_full_dataset(self) -> "DatasetIndexer":
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
        self, idx: Idx | tuple[Idx] | tuple[Idx, Idx | str | Sequence[str]]
    ) -> tuple[NDArray[np.integer], bool, tuple[int, ...] | None]:
        if not isinstance(idx, tuple):
            regions = idx
            samples = slice(None)
        elif len(idx) == 1:
            regions = idx[0]
            samples = slice(None)
        else:
            regions, samples = idx

        if (
            isinstance(samples, str)
            or (isinstance(samples, np.ndarray) and is_dtype(samples, np.str_))
            or (
                isinstance(samples, Sequence)
                and isinstance(next(chain.from_iterable(samples)), str)  # type: ignore
            )
        ):
            s_idx = self.s2i_map.get(samples)
            if (np.atleast_1d(s_idx) == -1).any():
                raise KeyError(
                    f"Some samples not found in dataset: {np.unique(np.array(samples)[s_idx == -1])}"
                )
        else:
            s_idx = samples

        s_idx = cast(Idx, s_idx)  # above clause does this, but can't narrow type

        idx = self.i2d_map.reshape(self.shape)[regions, s_idx]

        out_reshape = None
        squeeze = False
        if idx.ndim > 1:
            out_reshape = idx.shape
        elif idx.ndim == 0:
            squeeze = True

        idx = idx.ravel()

        return idx, squeeze, out_reshape
