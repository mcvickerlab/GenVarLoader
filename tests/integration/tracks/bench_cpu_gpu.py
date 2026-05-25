import json
from enum import Enum
from pathlib import Path

import typer


class Arch(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


def main(
    n_queries: int,
    min_intervals: int,
    max_intervals: int,
    min_length: int,
    max_length: int,
    min_width: int,
    max_width: int,
    max_value: float,
    ti_arch: Arch,
    out: Path,
):
    from time import perf_counter

    import numpy as np
    import polars as pl
    import taichi as ti
    from genvarloader.dataset.intervals import (
        intervals_to_tracks,
        ti_intervals_to_tracks,
    )
    from genvarloader.types import INTERVAL_DTYPE
    from genvarloader.utils import lengths_to_offsets
    from loguru import logger
    from tqdm.auto import tqdm
    from utils import nonoverlapping_intervals

    ti.init(ti.gpu if ti_arch is Arch.GPU else ti.cpu)

    rng = np.random.default_rng(0)

    regions = np.empty((n_queries, 4), dtype=np.int32)
    n_itvs_per_query = rng.integers(min_intervals, max_intervals, size=n_queries)
    intervals = np.empty(n_itvs_per_query.sum(), dtype=INTERVAL_DTYPE)
    offsets = lengths_to_offsets(n_itvs_per_query)
    offset_idxs = np.arange(len(offsets) - 1, dtype=np.int32)

    logger.info("Generating intervals")
    for query in range(n_queries):
        n_intervals = n_itvs_per_query[query]
        offset = offsets[query]
        length = rng.integers(min_length, max_length)
        seed = rng.integers(0, 2**31)
        regions[query] = np.array([0, 0, length, 1], dtype=np.int32)
        coordinates = nonoverlapping_intervals(
            n_intervals, 0, length, min_width, max_width, seed
        )
        values = np.random.rand(n_intervals).astype(np.float32) * max_value
        intervals[offset : offset + n_intervals]["start"] = coordinates[:, 0]
        intervals[offset : offset + n_intervals]["end"] = coordinates[:, 1]
        intervals[offset : offset + n_intervals]["value"] = values

    logger.info("Calling once for compilation")
    _ = intervals_to_tracks(
        offset_idxs=offset_idxs,
        regions=regions,
        intervals=intervals,
        offsets=offsets,
    )

    _ = ti_intervals_to_tracks(
        offset_idxs=offset_idxs,
        regions=regions,
        intervals=intervals,
        offsets=offsets,
    )

    n_loops = 50
    nb_times = np.empty(n_loops, dtype=np.float64)
    ti_times = np.empty(n_loops, dtype=np.float64)
    for i in tqdm(range(n_loops), desc="Benchmarking"):
        t0 = perf_counter()
        nb_tracks = intervals_to_tracks(
            offset_idxs=offset_idxs,
            regions=regions,
            intervals=intervals,
            offsets=offsets,
        )
        nb_times[i] = perf_counter() - t0

        t0 = perf_counter()
        ti_tracks = ti_intervals_to_tracks(
            offset_idxs=offset_idxs,
            regions=regions,
            intervals=intervals,
            offsets=offsets,
        )
        ti_times[i] = perf_counter() - t0

        np.testing.assert_equal(nb_tracks, ti_tracks)
    pl.DataFrame(
        {
            "numba": nb_times,
            f"taichi_{ti_arch}": ti_times,
        }
    ).write_csv(out.with_suffix(".csv"))
    with open(out.with_suffix(".json"), "w") as f:
        metadata = {
            "n_queries": n_queries,
            "min_intervals": min_intervals,
            "max_intervals": max_intervals,
            "min_length": min_length,
            "max_length": max_length,
            "min_width": min_width,
            "max_width": max_width,
            "max_value": max_value,
            "ti_arch": ti_arch,
        }
        json.dump(metadata, f, indent=4)


if __name__ == "__main__":
    typer.run(main)
