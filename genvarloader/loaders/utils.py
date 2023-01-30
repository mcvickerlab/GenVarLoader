import tensorstore as ts

from genvarloader.types import PathType


def ts_open_zarr(path: PathType):
    return ts.open(  # type: ignore
        {"driver": "zarr", "kvstore": {"driver": "file", "path": str(path)}},
        read=True,
        write=False,
        open=True,
        create=False,
        delete_existing=False,
    ).result()
