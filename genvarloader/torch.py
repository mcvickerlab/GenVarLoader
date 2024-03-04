from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import polars as pl
from numpy.typing import NDArray
from torch.utils.data import Dataset, Sampler

from .util import construct_virtual_data, process_bed

if TYPE_CHECKING:
    from .haplotypes import Haplotypes
    from .types import Reader


class GVLDataset(Dataset):
    def __init__(
        self,
        *readers: Reader,
        bed: Union[pl.DataFrame, str, Path],
        fixed_length: int,
        batch_dims: Optional[List[str]] = None,
        transform: Optional[Callable[[Dict[str, NDArray]], Dict[str, NDArray]]] = None,
    ):
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


class ZarrBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        bed: pl.DataFrame,
        readers: List[Union[Reader, Haplotypes]],
        length: int,
        batch_size: int,
        cache_pool: int,
    ):
        """Batch sampler for Zarr-backed datasets that will respect chunk boundaries.

        Parameters
        ----------
        bed : pl.DataFrame
            _description_
        dim_sizes : Dict[str, int]
            _description_
        chunk_shape : Tuple[int, ...]
            _description_
        batch_size : int
            _description_
        cache_pool : size in bytes of in-memory LRU cache for chunks
        """

        """
        Implementation
        --------------
        1. Infer grid of chunks
        2. Sample chunks that fit into cache_pool
        3. Sample indices from chunks
        4. If chunks exhausted, repeat 2-3 until batch_size is reached
        """
        self.bed = bed
        self.batch_size = batch_size
        self.cache_pool = cache_pool

        self.readers: List[Reader] = []
        for r in readers:
            if isinstance(r, Haplotypes):
                self.readers.extend(r.readers)
            else:
                self.readers.append(r)
        self.vdata = construct_virtual_data(
            *self.readers, n_regions=bed.height, fixed_length=length
        )

    def __iter__(self) -> Iterator[List[int]]:
        self.chunk_ptrs(self.dataset_shape, self.chunk_shape)

    def chunk_ptrs(self, dataset_shape: Tuple[int, ...], chunk_shape: Tuple[int, ...]):
        chunk_grid = np.ceil(np.array(dataset_shape) / np.array(chunk_shape)).astype(
            np.int32
        )
        chunk_multi_ptrs = (
            np.mgrid[tuple(slice(d) for d in chunk_grid)].T.reshape(-1, len(chunk_grid))
            * chunk_shape
        )
        chunk_ptrs = np.ravel_multi_index(chunk_multi_ptrs.T, dataset_shape)
        return chunk_ptrs

    def __len__(self):
        return len(self.dataset)
