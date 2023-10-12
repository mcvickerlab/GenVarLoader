from pathlib import Path
from typing import Union

import dask.array as da
import xarray as xr

from .types import Reader


class Zarr(Reader):
    def __init__(self, name: str, path: Union[str, Path]) -> None:
        """Read data from an XArray-compliant Zarr store. Should have contigs as
        separate arrays, a list of contigs as an attribute, and `length` as a dimension.
        For example:

        ```plain
        ./
        ├── chr1
        ├── chr2
        ├── ...
        └── attrs (dict)
            └── contigs (list[str])
        ```

        Parameters
        ----------
        name : str
            Name of the reader.
        path : Path
            Path to the store.
        """
        self.name = name
        self.path = path
        self.ds = xr.open_zarr(path, mask_and_scale=False, concat_characters=False)
        contigs = self.ds.attrs["contigs"]
        shape = tuple(
            s for d, s in self.ds.sizes.items() if not str(d).endswith("length")
        )
        dims = tuple(d for d in self.ds.sizes.keys() if not str(d).endswith("length"))
        coords = {
            d: c for d, c in self.ds.coords.items() if not str(d).endswith("length")
        }
        self.virtual_data = xr.DataArray(
            da.empty(shape=shape, dtype=self.ds.dtypes[contigs[0]]),
            dims=dims,
            coords=coords,
        )

    def read(self, contig: str, start: int, end: int) -> xr.DataArray:
        return self.ds[contig].isel(length=slice(start, end)).compute()
