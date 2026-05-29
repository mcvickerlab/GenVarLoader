"""Shared synthetic data generator for Table.count_intervals backend benches."""

from __future__ import annotations

import numpy as np
import polars as pl

# Restricted bench space: just small + medium to keep runtime manageable
CASES: list[dict] = [
    {"name": "small", "n_regions": 100, "n_samples": 10, "ipp": 100},
    {"name": "medium", "n_regions": 500, "n_samples": 50, "ipp": 500},
    {"name": "large", "n_regions": 2000, "n_samples": 200, "ipp": 500},
]

N_TRIALS = 20


def gen_table(
    n_samples: int, ipp: int, contig_len: int = 1_000_000, seed: int = 0
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    n_total = n_samples * ipp
    starts = rng.integers(0, contig_len - 1000, size=n_total, dtype=np.int64)
    ends = starts + rng.integers(50, 500, size=n_total, dtype=np.int64)
    sample_ids = np.repeat(
        np.array([f"s{i}" for i in range(n_samples)], dtype=object), ipp
    )
    return pl.DataFrame({
        "sample_id": sample_ids.astype(str),
        "chrom": np.full(n_total, "chr1", dtype=object).astype(str),
        "start": starts,
        "end": ends,
        "value": rng.standard_normal(n_total).astype(np.float32),
    })


def gen_queries(
    n_regions: int, contig_len: int = 1_000_000, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, contig_len - 5000, size=n_regions, dtype=np.int64)
    ends = starts + rng.integers(500, 5000, size=n_regions, dtype=np.int64)
    return starts, ends
