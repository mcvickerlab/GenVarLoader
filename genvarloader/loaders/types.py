from asyncio import Future
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import pandera as pa
import zarr
from natsort import natsorted
from numpy.typing import NDArray
from pandera.engines import pandas_engine
from pandera.typing import DataFrame, Series
from typing_extensions import Self


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
        ...

    async def async_sel(self, queries: Queries, length: int, **kwargs) -> LoaderOutput:
        ...


# non-user facing, just for type checking TensorStore
_T = TypeVar("_T", bound=np.generic)


class _TStore(Protocol[_T]):
    def __getitem__(self, idx) -> Self:
        ...

    @property
    def shape(self) -> Tuple[int, ...]:
        ...

    @property
    def size(self) -> int:
        ...

    @property
    def dtype(self) -> _T:
        ...

    def astype(self, dtype) -> Self:
        ...

    def read(self) -> Future[NDArray[_T]]:
        ...


@dataclass
class _VCFTSDataset:
    """An sgkit Zarr dataset for a single sample using TensorStore for I/O."""

    call_genotype: _TStore[np.int8]  # (v s p)
    variant_allele: _TStore[np.uint8]  # (v a)
    variant_contig: _TStore[np.int16]  # (v)
    variant_position: zarr.Group  # (v)
    contigs: NDArray[np.object_]  # (c)
    contig_offsets: _TStore[np.integer]  # (c)
