from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import polars as pl
from torch.utils.data import Dataset

from .types import Reader
from .util import construct_virtual_data, process_bed


class GVLDataset(Dataset):
    def __init__(
        self,
        *readers: Reader,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_dims: Optional[List[str]] = None,
    ):
        self.readers = readers
        self.fixed_length = fixed_length
        self.bed = process_bed(bed, fixed_length)
        self.virtual_data = construct_virtual_data(
            *readers, n_regions=self.bed.height, fixed_length=fixed_length
        )
        self.batch_dims = ["region"]
        if batch_dims is not None:
            self.batch_dims += batch_dims
        self._batch_shape = tuple(self.virtual_data.sizes[d] for d in self.batch_dims)

    def __len__(self):
        return np.prod([self.virtual_data.sizes[d] for d in self.batch_dims])

    def __getitem__(self, idx):
        _indices = np.unravel_index(idx, self._batch_shape)
        indices: Dict[str, int] = {
            d: _indices[i] for i, d in enumerate(self.batch_dims)
        }
        contig, start, end = self.bed.select("chrom", "chromStart", "chromEnd").row(
            indices["region"]
        )
        read_kwargs = {
            d: self.virtual_data.coords[d][[indices[d]]].to_numpy()
            for d in self.batch_dims[1:]
        }
        batch = {
            r.name: r.read(contig, start, end, **read_kwargs).to_numpy()
            for r in self.readers
        }
        return batch
