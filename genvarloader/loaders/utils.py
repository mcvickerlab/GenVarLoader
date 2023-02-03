import tensorstore as ts

from genvarloader.loaders.types import _TStore
from genvarloader.types import PathType


def ts_readonly_zarr(path: PathType) -> _TStore:
    return ts.open(  # type: ignore
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}},
        read=True,
        write=False,
        open=True,
        create=False,
        delete_existing=False,
    ).result()
