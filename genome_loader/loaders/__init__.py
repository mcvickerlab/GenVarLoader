import asyncio
from typing import Dict, Protocol, Union, cast

from numpy.typing import NDArray

from genome_loader.loaders.types import Queries
from genome_loader.types import PathType

try:
    from genome_loader.torch import TorchCollator
except ImportError as e:
    _TORCH_AVAILABLE = False
else:
    _TORCH_AVAILABLE = True


class Loader(Protocol):
    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        ...


class AsyncLoader(Protocol):
    def sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        ...

    async def async_sel(self, queries: Queries, length: int, **kwargs) -> NDArray:
        ...


class GenVarLoader:
    def __init__(self, loaders: Dict[str, Union[Loader, AsyncLoader]]) -> None:
        self.loaders = loaders
        self.async_loaders = {
            k: cast(AsyncLoader, v)
            for k, v in loaders.items()
            if hasattr(v, "async_sel")
        }
        self.sync_loaders = {
            k: cast(Loader, v)
            for k, v in loaders.items()
            if k not in self.async_loaders
        }

    def sel(self, queries: Queries, length: int, **kwargs) -> Dict[str, NDArray]:
        out = {}
        for name, loader in self.sync_loaders.items():
            out[name] = loader.sel(queries, length, **kwargs)
        out.update(asyncio.run(self.async_sel(queries, length, **kwargs)))
        return out

    async def async_sel(
        self, queries: Queries, length: int, **kwargs
    ) -> Dict[str, NDArray]:
        out_ls = await asyncio.gather(
            *[
                l.async_sel(queries, length, **kwargs)
                for l in self.async_loaders.values()
            ]
        )
        out = dict(zip(self.async_loaders.keys(), out_ls))
        return out

    def get_torch_collator(self, queries_path: PathType, length: int):
        if not _TORCH_AVAILABLE:
            raise ImportError("Getting a torch collator requires PyTorch.")
        return TorchCollator(self, queries_path, length)
