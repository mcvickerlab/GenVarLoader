from pathlib import Path

import xarray as xr
import zarr

from .types import Reader


class Zarr(Reader):
    def __init__(self, name: str, path: Path) -> None:
        self.name = name
        self.path = path
        self.root = zarr.open_group(path)
        # self._data = ...

    def read(self, contig: str, start: int, end: int) -> xr.DataArray:
        return xr.DataArray(self.root[contig][start:end], dims=..., coords=...)
