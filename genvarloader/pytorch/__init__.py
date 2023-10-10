"""
Usage
-----

Out of core I/O, constructing features on-the-fly
```
from torch.utils.data import DataLoader
import genvarloader.loaders as gvl
import genvarloader.torch as gvl_torch

gvloader = gvl.GenVarLoader(...)

length = 600
sel_kwargs = {'encoding': 'bytes'}
collate_fn = gvloader.get_torch_collator(length, sel_kwargs)

ds = gvl_torch.QueriesDataset(queries)
sampler = gvloader.get_sorted_query_sampler(queries, batch_size=8)
dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate_fn)
next(iter(dl))
{
    'varseq': # tensor with shape (8 600)
    'coverage': # tensor with shape (8 600)
}
```

Writing queries to disk and reading those instead, faster for variant sequences and transformed data.
```
from torch.utils.data import DataLoader
import genvarloader.loaders as gvl
import genvarloader.torch as gvl_torch

length = 600
sel_kwargs = {'encoding': 'bytes'}
gvloader = gvl.GenVarLoader({'varseq': ..., 'coverage': ...})

queries = gvl.read_queries('queries.csv')
gvloader.write_queries('out.zarr', queries, length, sel_kwargs)

collate_fn = gvl_torch.ZarrCollator('out.zarr')
# remember: batched sampling must be enabled so the collate function is used
dl = DataLoader(collate_fn.index, batch_size=8, collate_fn=collate_fn)
next(iter(dl))
{
    'varseq': # tensor with shape (8 600)
    'coverage': # tensor with shape (8 600)
}
```
"""

try:
    import pytorch
except ImportError:
    raise ImportError("The `torch` submodule requires PyTorch.")

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Optional, cast

import numpy as np
import pandas as pd
import zarr
from torch.utils.data import Dataset, Sampler

from genvarloader.loaders.types import Queries
from genvarloader.loaders.utils import _ts_readonly_zarr
from genvarloader.types import PathType

if TYPE_CHECKING:
    from genvarloader.loaders import GenVarLoader

__all__ = ["TorchCollator", "ZarrCollator", "QueriesDataset", "SortedQuerySampler"]


class QueriesDataset(Dataset):
    def __init__(self, queries: pd.DataFrame) -> None:
        self.queries = Queries(queries)

    def __getitem__(self, index) -> pd.DataFrame:
        return self.queries.iloc[index]

    def __len__(self):
        return len(self.queries)


class TorchCollator:
    def __init__(
        self,
        genvarloader: "GenVarLoader",
        length: int,
        sel_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.gvl = genvarloader
        self.length = length
        self.sel_kwargs = {} if sel_kwargs is None else sel_kwargs

        if "index" in self.gvl.loaders:
            raise RuntimeError(
                """
                GenVarLoader has a loader named 'index' which causes a naming
                conflict since the collator needs to use a key called 'index'
                to provide batch indices. Create a new GenVarLoader that doesn't
                have any loaders named 'index'.
                """
            )

    def __call__(self, queries: pd.DataFrame) -> Dict[str, pytorch.Tensor]:
        _out = self.gvl.sel(queries, self.length, **self.sel_kwargs)
        out = {k: pytorch.as_tensor(v) for k, v in _out.items()}
        out["index"] = pytorch.as_tensor(queries.index.values)
        return out


class ZarrCollator:
    def __init__(
        self,
        zarr_path: PathType,
        groups: Optional[List[str]] = None,
        ts_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        ts_kwargs = {} if ts_kwargs is None else ts_kwargs
        if groups is None:
            groups = list(zarr.open_group(zarr_path).array_keys())  # type: ignore
            groups = cast(List[str], groups)
        self.groups = {
            g: _ts_readonly_zarr(Path(zarr_path) / g, **ts_kwargs).result()
            for g in groups
        }

    def __call__(self, indices=List[int]) -> Dict[str, pytorch.Tensor]:
        out = asyncio.run(self.async_call(indices))
        return out

    async def async_call(self, indices=List[int]) -> Dict[str, pytorch.Tensor]:
        group_arrays = await asyncio.gather(
            *[ts[indices].read() for ts in self.groups.values()]
        )
        group_arrays = [pytorch.as_tensor(a) for a in group_arrays]
        out = dict(zip(self.groups.keys(), group_arrays))
        return out

    @property
    def index(self):
        tstore = next(iter(self.groups.values()))
        return np.arange(tstore.shape[0])


class SortedQuerySampler(Sampler):
    def __init__(
        self,
        queries: pd.DataFrame,
        batch_size: int = 1,
        sort_order: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ) -> None:
        if batch_size > len(queries):
            raise ValueError("Batch size is greater than the # of queries.")

        # sort by sample (if present) then contig by default
        if sort_order is None:
            sort_order = []
            if "sample" in queries:
                sort_order.append("sample")
            sort_order.append("contig")

        # randomly reorder contig and sample
        rng = np.random.default_rng(seed)

        contigs = queries.contig.cat.categories
        contigs = rng.choice(contigs, len(contigs), replace=False)
        queries["contig"] = queries.contig.cat.reorder_categories(contigs, ordered=True)

        if "sample" in queries:
            samples = queries["sample"].unique()
            samples = rng.choice(samples, len(samples), replace=False)
            queries["sample"] = pd.Series(
                pd.Categorical(queries["sample"], categories=samples, ordered=True)
            )

        # shuffle queries
        queries = queries.groupby(sort_order, sort=True).sample(
            frac=1, replace=False, random_state=seed
        )

        # split batches
        n_equal_sized_batches, mod = divmod(len(queries), batch_size)
        end_equal_sized_batches = len(queries) - mod
        batches = np.split(
            queries.iloc[:end_equal_sized_batches], n_equal_sized_batches
        )
        if end_equal_sized_batches != len(queries):
            batches.append(queries.iloc[end_equal_sized_batches:])  # type: ignore

        self.batches = cast(List[pd.DataFrame], batches)

    def __iter__(self) -> Generator[pd.DataFrame, None, None]:
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)
