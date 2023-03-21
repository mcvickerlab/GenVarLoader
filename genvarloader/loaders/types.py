import asyncio
import enum
from asyncio import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union, cast

import numpy as np
import pandas as pd
import pandera as pa
import zarr
from natsort import natsorted
from numpy.typing import NDArray
from pandera.engines import pandas_engine
from pandera.typing import DataFrame, Series
from typing_extensions import Self

from genvarloader.loaders.utils import ts_readonly_zarr


# Register pandera dtype that is guaranteed to have naturally ordered categories
# Important for contig columns
@pandas_engine.Engine.register_dtype  # type: ignore[arg-type]
@pa.dtypes.immutable  # type: ignore
class NatOrderCategory(pandas_engine.Category):
    """Naturally ordered categorical data. This means, for example,
    that '1' < '2' < '15' < 'X' rather than '1' < '15' < '2' < 'X'.
    """

    def coerce(self, series: pd.Series):
        data = series.values
        return pd.Series(
            pd.Categorical(data, categories=natsorted(np.unique(data)), ordered=True)  # type: ignore
        )


class QueriesSchema(pa.SchemaModel):
    contig: Series[NatOrderCategory] = pa.Field(coerce=True)  # type: ignore
    start: Series[pa.Int]
    strand: Optional[Series[pa.Category]] = pa.Field(coerce=True, isin=["+", "-"])
    sample: Optional[Series[pa.Category]] = pa.Field(coerce=True)
    ploid_idx: Optional[Series[pa.Int]] = pa.Field(ge=0)


Queries = DataFrame[QueriesSchema]

LoaderOutput = Union[NDArray, Dict[str, NDArray]]


class Loader(Protocol):
    def sel(self, queries: Queries, length: int, **kwargs) -> LoaderOutput:
        ...


class AsyncLoader(Protocol):
    def sel(self, queries: Queries, length: int, **kwargs) -> LoaderOutput:
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(self, queries: Queries, length: int, **kwargs) -> LoaderOutput:
        ...


# non-user facing, just for type checking TensorStore
_DTYPE = TypeVar("_DTYPE", bound=np.generic)


class _TStore(Protocol[_DTYPE]):
    def __getitem__(self, idx) -> Self:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def dtype(self) -> _DTYPE:
        ...

    def astype(self, dtype) -> Self:
        ...

    def read(self) -> Future[NDArray[_DTYPE]]:
        ...


class _VCFTSDataset:
    """An sgkit Zarr dataset for a single sample using TensorStore for I/O."""

    sample_id: str
    call_genotype: _TStore[np.int8]  # (v s p)
    variant_allele: _TStore[np.uint8]  # (v a)
    variant_contig: _TStore[np.int16]  # (v)
    variant_position: zarr.Group  # (v)
    contig_offsets: _TStore[np.integer]  # (c)
    contig_idx: Dict[str, int]
    contig_offset_idx: Dict[str, int]

    def __init__(self) -> None:
        self._initalized = False

    @classmethod
    async def create(cls, path: Path, ts_kwargs: Dict):
        self = cls()

        z = cast(zarr.Group, zarr.open_consolidated(str(path), mode="r"))

        self.sample_id = cast(str, z["sample_id"][0])

        # We have to eagerly read all the positions for a contig downstream
        # so there's no need to make tensorstores here.
        self.variant_position = cast(zarr.Group, z["variant_position"])

        # open tensorstores
        gvl_array_names = {
            "call_genotype",
            "contig_offsets",
            "variant_allele",
            "variant_contig",
        }
        arrays = [
            ts_readonly_zarr(path.resolve() / n, **ts_kwargs) for n in gvl_array_names
        ]
        gvl_arrays = await asyncio.gather(*arrays)
        self.call_genotype = gvl_arrays[0]
        self.contig_offsets = gvl_arrays[1]
        self.variant_allele = gvl_arrays[2]
        self.variant_contig = gvl_arrays[3]

        self.contig_idx = z.attrs["contig_idx"]
        self.contig_offset_idx = z.attrs["contig_offset_idx"]

        self._initalized = True

        return self
