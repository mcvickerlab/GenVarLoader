from typing import Optional

import numpy as np
from attrs import define, evolve
from numpy.typing import NDArray
from typing_extensions import Self

from .._types import Idx
from ._utils import idx_like_to_array, oidx_to_raveled_idx


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
    region_subset_idxs: Optional[NDArray[np.integer]] = None
    """Which input regions are included in the subset."""
    sample_subset_idxs: Optional[NDArray[np.integer]] = None
    """Which input samples are included in the subset."""

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
    def region_idxs(self) -> NDArray[np.integer]:
        """Map from input region indices to on-disk region indices."""
        return np.unravel_index(self.i2d_map[:: self.n_samples], self.full_shape)[0]

    @property
    def sample_idxs(self) -> NDArray[np.integer]:
        """Map from input sample indices to on-disk sample indices."""
        return np.unravel_index(self.i2d_map[: self.n_samples], self.full_shape)[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_regions, self.n_samples

    @property
    def full_shape(self) -> tuple[int, int]:
        return len(self.full_region_idxs), len(self.full_sample_idxs)

    def __getitem__(self, idx: Idx) -> NDArray[np.integer]:
        return self.i2d_map[idx]

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

        idx_map = self.i2d_map[idx]

        return evolve(
            self,
            idx_map=idx_map,
            region_subset_idxs=region_idxs,
            sample_subset_idxs=sample_idxs,
        )

    def to_full_dataset(self) -> Self:
        """Return a full sized dataset, undoing any subsettting."""
        idx_map = oidx_to_raveled_idx(
            self.full_region_idxs, self.full_sample_idxs, self.full_shape
        )
        return evolve(
            self, idx_map=idx_map, region_subset_idxs=None, sample_subset_idxs=None
        )
