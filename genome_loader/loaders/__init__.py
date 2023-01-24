from inspect import signature
from typing import Optional, Protocol, cast

import numpy as np
import pandas as pd
import pandera as pa
from natsort import natsorted
from numpy.typing import NDArray
from pandera.engines import pandas_engine
from pandera.typing import DataFrame, Series

from genome_loader.utils import PathType

try:
    import torch

    from genome_loader.torch import parse_queries
except ImportError as e:
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True


# Add a new data type that is guaranteed to have naturally ordered categories
@pandas_engine.Engine.register_dtype  # type: ignore[arg-type]
@pa.dtypes.immutable
class NatOrderCategory(pandas_engine.Category):
    """Naturally ordered categorical data. This means, for example,
    that '1' < '2' < '15' < 'X' rather than '1' < '15' < '2' < 'X'.
    """

    def coerce(self, series: pd.Series):
        data = series.values
        return pd.Series(
            pd.Categorical(data, categories=natsorted(np.unique(data)), ordered=True)
        )


class QueriesSchema(pa.SchemaModel):
    contig: Series[NatOrderCategory] = pa.Field(coerce=True)
    start: Series[pa.Int]
    strand: Optional[Series[pa.Category]] = pa.Field(coerce=True, isin=["+", "-"])
    sample: Optional[Series[pa.Category]] = pa.Field(coerce=True)
    ploid_idx: Optional[Series[pa.Int]] = pa.Field(ge=0)


Queries = DataFrame[QueriesSchema]


class Loader(Protocol):
    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        ...


class GenVarLoader:
    def __init__(self, loaders: dict[str, Loader]) -> None:
        self.loaders = loaders

    def sel(self, queries: Queries, length: int, **kwargs) -> dict[str, NDArray]:
        out = {}
        for name, loader in self.loaders.items():
            out[name] = loader.sel(queries, length, **kwargs)
        return out

    def get_torch_collator(self, queries_path: PathType, length: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("Using torch collators requires PyTorch.")
        return self.TorchCollator(self, queries_path, length)

    class TorchCollator:
        def __init__(
            self, genvarloader: "GenVarLoader", queries_path: PathType, length: int
        ) -> None:
            self.gvl = genvarloader
            self.queries = parse_queries(queries_path)
            self.length = length

            if "index" in self.gvl.loaders:
                raise RuntimeError(
                    """
                    GenVarLoader has as loader named 'index' which causes a naming
                    conflict since the collator needs to use a key called 'index'
                    to store batch indices. Create a new GenVarLoader that doesn't
                    have any loaders named 'index'.
                    """
                )

        def __call__(self, batch_indices: list[int]) -> dict[str, torch.Tensor]:
            batch = cast(Queries, self.queries[batch_indices])
            out = {
                k: torch.as_tensor(v)
                for k, v in self.gvl.sel(batch, self.length).items()
            }
            out["index"] = torch.as_tensor(batch_indices)
            return out
