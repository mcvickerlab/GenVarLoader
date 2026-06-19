"""Reproducible synthetic bigWig corpus for parity tests and benchmarks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig

DEFAULT_CONTIGS = {"chr21": 200_000, "chr22": 150_000}


def make_synthetic_bigwigs(
    out_dir: Path,
    n_samples: int,
    *,
    contigs: dict[str, int] = DEFAULT_CONTIGS,
    density: float = 0.01,
    seed: int = 0,
) -> list[Path]:
    """Write `sample_{i}.bw` for i in range(n_samples). Deterministic given `seed`.

    Each contig gets contiguous, non-overlapping intervals: starts are a sorted
    random subset of positions (~`density` fraction), each running to the next start.
    """
    out_dir = Path(out_dir)
    paths: list[Path] = []
    header = [(c, int(length)) for c, length in contigs.items()]
    for i in range(n_samples):
        rng = np.random.default_rng(seed + i)
        path = out_dir / f"sample_{i}.bw"
        with pyBigWig.open(str(path), "w") as bw:
            bw.addHeader(header, maxZooms=0)
            for contig, length in contigs.items():
                n = max(1, int(length * density))
                starts = np.unique(rng.integers(0, length - 1, size=n).astype(np.int64))
                starts.sort()
                ends = np.empty_like(starts)
                ends[:-1] = starts[1:]
                ends[-1] = min(int(starts[-1]) + 1, length)
                # drop any zero-width tail
                keep = ends > starts
                starts, ends = starts[keep], ends[keep]
                values = rng.standard_normal(len(starts)).astype(np.float32)
                bw.addEntries(
                    [contig] * len(starts),
                    [int(s) for s in starts],
                    ends=[int(e) for e in ends],
                    values=[float(v) for v in values],
                )
        paths.append(path)
    return paths


def make_regions(
    contigs: dict[str, int],
    n_per_contig: int,
    width: int,
    *,
    seed: int = 0,
) -> pl.DataFrame:
    """Contig-grouped regions DataFrame (chrom, chromStart, chromEnd)."""
    rng = np.random.default_rng(seed)
    chrom, start, end = [], [], []
    for contig, length in contigs.items():
        starts = rng.integers(0, max(1, length - width), size=n_per_contig)
        for s in starts:
            chrom.append(contig)
            start.append(int(s))
            end.append(int(s) + width)
    return pl.DataFrame(
        {"chrom": chrom, "chromStart": start, "chromEnd": end},
        schema={"chrom": pl.Utf8, "chromStart": pl.Int64, "chromEnd": pl.Int64},
    )
