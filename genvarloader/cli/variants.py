from pathlib import Path
from typing import List

import typer


def vcfs_merge_filter_to_zarr(
    out_merged: Path,
    out_zarr: Path,
    reference: Path,
    renamer: Path,
    n_threads: int,
    vcfs: List[Path],
):
    from genvarloader.writers.variants import vcfs_merge_filter_to_zarr

    vcfs_merge_filter_to_zarr(out_merged, out_zarr, reference, renamer, n_threads, vcfs)


if __name__ == "__main__":
    typer.run(vcfs_merge_filter_to_zarr)
