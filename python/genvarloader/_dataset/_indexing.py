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
    full_sample_idxs: NDArray[np.integer]
    idx_map: NDArray[np.integer]
    n_regions: int
    n_samples: int

    @property
    def is_subset(self) -> bool:
        return self.n_regions < len(self.full_region_idxs) or self.n_samples < len(
            self.full_sample_idxs
        )

    @property
    def region_idxs(self) -> NDArray[np.integer]:
        return np.unravel_index(self.idx_map[:: self.n_samples], self.full_shape)[0]

    @property
    def sample_idxs(self) -> NDArray[np.integer]:
        return np.unravel_index(self.idx_map[: self.n_samples], self.full_shape)[1]

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_regions, self.n_samples

    @property
    def full_shape(self) -> tuple[int, int]:
        return len(self.full_region_idxs), len(self.full_sample_idxs)

    def __getitem__(self, idx: Idx) -> NDArray[np.integer]:
        return self.idx_map[idx]

    def __len__(self):
        return len(self.idx_map)

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

        idx_map = self.idx_map[idx]

        return evolve(
            self,
            idx_map=idx_map,
            n_regions=len(region_idxs),
            n_samples=len(sample_idxs),
        )

    def to_full_dataset(self) -> Self:
        """Return a full sized dataset, undoing any subsettting."""
        idx_map = oidx_to_raveled_idx(
            self.full_region_idxs, self.full_sample_idxs, self.full_shape
        )
        n_regions, n_samples = self.full_shape
        return evolve(self, idx_map=idx_map, n_regions=n_regions, n_samples=n_samples)
