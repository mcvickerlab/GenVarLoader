"""On-the-fly fixture builders for dataset-level parity tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyBigWig

import genvarloader as gvl
from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions, make_synthetic_bigwigs

# Contigs used by the session-level synthetic case (build_case / conftest).
# These match _SESSION_CONTIGS in tests/_builders/case.py.
_SESSION_CONTIGS = {"chr1": 1_300_000, "chr2": 1_300_000}
_SESSION_SAMPLES = ["s0", "s1", "s2"]


def build_track_dataset(work_dir: Path) -> Path:
    """Write a small track-only GVL dataset and return its path.

    No variants, no reference — just three synthetic BigWig samples over two
    contigs.  Regions are chosen to overlap the synthetic intervals so that
    reads produce non-zero signal.
    """
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    bw_dir = work_dir / "bw"
    bw_dir.mkdir(exist_ok=True)

    paths = make_synthetic_bigwigs(bw_dir, n_samples=3, seed=0)
    samples = [p.stem for p in paths]
    track = gvl.BigWigs("signal", {s: str(p) for s, p in zip(samples, paths)})

    bed = make_regions(DEFAULT_CONTIGS, n_per_contig=8, width=2000, seed=0)

    out = work_dir / "ds.gvl"
    gvl.write(path=out, bed=bed, tracks=track, overwrite=True)
    return out


def _make_session_bigwigs(bw_dir: Path, seed: int = 42) -> dict[str, str]:
    """Write one BigWig per session sample over the session contigs.

    Uses dense, non-overlapping intervals with density=0.05 (one interval
    every ~20 bp on average) so that synthetic regions of width ~200–2000 bp
    reliably contain multiple non-zero values.  The function is deterministic
    given `seed` so repeated calls produce identical files.

    Returns a mapping {sample_name: str(bw_path)}.
    """
    bw_dir.mkdir(parents=True, exist_ok=True)
    header = [(c, length) for c, length in _SESSION_CONTIGS.items()]
    paths: dict[str, str] = {}
    for i, sample in enumerate(_SESSION_SAMPLES):
        rng = np.random.default_rng(seed + i)
        path = bw_dir / f"{sample}.bw"
        with pyBigWig.open(str(path), "w") as bw:
            bw.addHeader(header, maxZooms=0)
            for contig, length in _SESSION_CONTIGS.items():
                # ~5 % density → one interval per ~20 bp
                n = max(2, int(length * 0.05))
                starts = np.unique(
                    rng.integers(0, length - 1, size=n).astype(np.int64)
                )
                starts.sort()
                ends = np.empty_like(starts)
                ends[:-1] = starts[1:]
                ends[-1] = min(int(starts[-1]) + 1, length)
                keep = ends > starts
                starts, ends = starts[keep], ends[keep]
                values = rng.standard_normal(len(starts)).astype(np.float32)
                bw.addEntries(
                    [contig] * len(starts),
                    [int(s) for s in starts],
                    ends=[int(e) for e in ends],
                    values=[float(v) for v in values],
                )
        paths[sample] = str(path)
    return paths


def build_haps_tracks_dataset(work_dir: Path, svar_path: Path) -> Path:
    """Write a variants+tracks GVL dataset and return its path.

    Uses the caller-supplied SparseVar file (which must cover chr1/chr2
    with samples s0/s1/s2, as produced by the session-level build_case
    fixture).  Synthetic BigWig tracks are written with matching samples
    and contigs.  The dataset is written with **max_jitter=0** to ensure
    that stored interval starts always equal the region query starts,
    satisfying the ``intervals_to_tracks`` Rust contract
    (``itv_start >= query_start``).

    Background on the landmine
    --------------------------
    When ``max_jitter > 0``, ``gvl.write`` / ``gvl.update`` clip BigWig
    intervals to the jitter-**expanded** boundaries stored in
    ``regions.npy`` (``chromStart - max_jitter``).  But
    ``Dataset.open`` derives ``_full_regions`` from the **original**
    ``input_regions.arrow`` boundaries (``chromStart``).  The gap of
    ``max_jitter`` bp means stored interval starts are
    ``chromStart - max_jitter < chromStart = query_start``, which
    violates the contract and triggers a ``PanicException`` in the Rust
    ``intervals_to_tracks`` kernel.  Setting ``max_jitter=0`` eliminates
    the gap.  The variants (including indels) still trigger
    ``shift_and_realign_tracks_sparse``, which is what this fixture exists
    to test.

    Returns the path to the written dataset directory.
    """
    from genoray import SparseVar
    import polars as pl

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Build BigWigs for the three session samples over chr1/chr2.
    bw_dir = work_dir / "bw"
    sample_to_bw = _make_session_bigwigs(bw_dir, seed=42)
    track = gvl.BigWigs("signal", sample_to_bw)

    # Derive regions from the SparseVar file: one short region per indel
    # so that we are guaranteed to have indel-bearing regions (which are
    # needed to exercise the realignment kernel).  Width=200 is wide enough
    # to overlap several BigWig intervals at density=0.05.
    sv = SparseVar(svar_path)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
            "chromStart": [
                1010685,  # overlaps GAGA→G deletion on chr1
                1110686,  # overlaps A→TTT insertion on chr1
                1210686,  # overlaps C→G SNP on chr1 (mixed indels)
                14360,    # overlaps chr2 SNP region
                1110686,  # chr2 G→A/T multiallelic (indel neighbours)
            ],
            "chromEnd": [
                1010705,
                1110706,
                1210706,
                14380,
                1110706,
            ],
        }
    )

    out = work_dir / "ds.gvl"
    # max_jitter=0: no jitter expansion → interval starts == query starts
    # → the intervals_to_tracks Rust contract is satisfied.
    gvl.write(
        path=out,
        bed=bed,
        variants=sv,
        tracks=track,
        max_jitter=0,
        overwrite=True,
    )
    return out
