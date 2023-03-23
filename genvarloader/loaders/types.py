import asyncio
from asyncio import Future
from typing import Dict, Optional, Protocol, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import pandera as pa
from natsort import natsorted
from numpy.typing import NDArray
from pandera.engines import pandas_engine
from pandera.typing import DataFrame, Series
from typing_extensions import Self


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


@pandas_engine.Engine.register_dtype  # type: ignore[arg-type]
@pa.dtypes.immutable  # type: ignore
class StrandCategory(pandas_engine.Category):
    """Allow ["+", "-", "."] as input but map "." to "+"."""

    def coerce(self, series: pd.Series):
        return (
            series.replace({".": "+"}).astype("category").cat.set_categories(["+", "-"])
        )


class QueriesSchema(pa.DataFrameModel):
    contig: Series[NatOrderCategory] = pa.Field(coerce=True)  # type: ignore
    start: Series[pa.Int]
    strand: Optional[Series[StrandCategory]] = pa.Field(coerce=True, isin=["+", "-"])  # type: ignore
    sample: Optional[Series[pa.Category]] = pa.Field(coerce=True)
    ploid_idx: Optional[Series[pa.Int]] = pa.Field(ge=0)


Queries = DataFrame[QueriesSchema]

LoaderOutput = Union[NDArray, Dict[str, NDArray]]


class Loader(Protocol):
    def sel(self, queries: pd.DataFrame, length: int, **kwargs) -> LoaderOutput:
        ...


class AsyncLoader(Protocol):
    def sel(self, queries: pd.DataFrame, length: int, **kwargs) -> LoaderOutput:
        out = asyncio.run(self.async_sel(queries, length, **kwargs))
        return out

    async def async_sel(
        self, queries: pd.DataFrame, length: int, **kwargs
    ) -> LoaderOutput:
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
