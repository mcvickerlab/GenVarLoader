from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray

from .types import Reader
from .util import construct_virtual_data, process_bed

try:
    from torch.utils.data import Dataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GVLDataset(Dataset):  # pyright: ignore[reportPossiblyUnboundVariable]
    def __init__(
        self,
        *readers: Reader,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "Could not import PyTorch. Please install PyTorch to use torch features."
            )
        self.readers = readers
        self.fixed_length = fixed_length
        self.transform = transform
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
            r.name: r.read(contig, start, end, **read_kwargs) for r in self.readers
        }
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


# class ZarrSampler(Sampler):
#     def __init__(self, bed: pl.DataFrame, dim_sizes: Dict[str, int], chunk_shape: Tuple[int, ...], batch_size: int):

#         contigs = np.asarray(contigs)
#         contig_offsets = np.unique(contigs, return_counts=True)
#         self.data_by_contig = {

#         }
#         self.chunk_shape = chunk_shape

#     def __iter__(self):
#         return iter(range(len(self.dataset)))

#     def __len__(self):
#         return len(self.dataset)
