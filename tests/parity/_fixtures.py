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


# Contigs and samples for the jittered-track fixture (§242 regression coverage).
_JITTER_CONTIGS = {"chr21": 200_000, "chr22": 150_000}
_JITTER_SAMPLES = ["s0", "s1", "s2"]
# Constant BigWig signal value per sample: s0→1.0, s1→2.0, s2→3.0.
# Hand-computable: for any region [start, end), sample j yields [j+1.0] * (end-start).
_JITTER_SIGNAL_PER_SAMPLE: dict[str, float] = {
    s: float(i + 1) for i, s in enumerate(_JITTER_SAMPLES)
}


def build_track_dataset_jittered(work_dir: Path, max_jitter: int) -> Path:
    """Write a track-only GVL dataset with ``max_jitter > 0`` for #242 parity coverage.

    Signal design
    -------------
    Each sample has a SINGLE constant BigWig interval covering the ENTIRE contig
    (s0=1.0, s1=2.0, s2=3.0).  Any read window is fully covered, so the expected
    track over any region [start, end) with jitter=0 is just the per-sample constant
    repeated for ``(end - start)`` positions — trivially hand-computable.

    #242 condition
    --------------
    ``gvl.write`` clips BigWig intervals to the jitter-EXPANDED window
    ``[chromStart - max_jitter, chromEnd + max_jitter]``, so the stored interval
    start is ``chromStart - max_jitter < chromStart``.  ``Dataset.open`` queries
    at the ORIGINAL ``chromStart``.  This means ``itv.start < query_start`` — the
    exact boundary condition that PR #244 fixed in both kernels.

    Regions are placed well inside contig bounds so the expanded write window
    ``[chromStart - max_jitter, chromEnd + max_jitter]`` never underflows (all
    chromStarts ≥ 1000, so expanded start ≥ 996 ≥ 0 for max_jitter ≤ 1000).
    """
    import polars as pl

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    bw_dir = work_dir / "bw"
    bw_dir.mkdir(exist_ok=True)

    header = [(c, length) for c, length in _JITTER_CONTIGS.items()]
    sample_to_bw: dict[str, str] = {}
    for sample, value in _JITTER_SIGNAL_PER_SAMPLE.items():
        bw_path = bw_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(header, maxZooms=0)
            for contig, length in _JITTER_CONTIGS.items():
                # Single interval covering the entire contig → constant signal everywhere.
                bw.addEntries([contig], [0], ends=[int(length)], values=[float(value)])
        sample_to_bw[sample] = str(bw_path)

    track = gvl.BigWigs("signal", sample_to_bw)

    # Three regions spanning two contigs, already in natural sort order
    # (chr21 before chr22, ascending chromStart within contig).  This keeps
    # regions.npy and input_regions.arrow in the same row order so the
    # r_idx_map alignment in the test is trivially [0, 1, 2].
    bed = pl.DataFrame(
        {
            "chrom": ["chr21", "chr21", "chr22"],
            "chromStart": [1000, 5000, 1000],
            "chromEnd": [1020, 5020, 1020],
        }
    )

    out = work_dir / "jittered_ds.gvl"
    gvl.write(path=out, bed=bed, tracks=track, max_jitter=max_jitter, overwrite=True)
    return out


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
                starts = np.unique(rng.integers(0, length - 1, size=n).astype(np.int64))
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


def build_strand_mixed_dataset(work_dir: Path, svar_path: Path) -> Path:
    """Write a variants+tracks GVL dataset with mixed + and − strand regions.

    Strand layout (index → region → strand):
      0: chr1:1010685-1010705  strand=+1  (overlaps GAGA→G deletion on chr1)
      1: chr1:1110686-1110706  strand=−1  (non-vacuity anchor: GAATGTAAGACGCAGCGTGC)
      2: chr1:1210686-1210706  strand=+1
      3: chr2:14360-14380      strand=−1
      4: chr2:1110686-1110706  strand=+1

    Region 1 (the first -strand region) carries a non-palindromic reference
    sequence so the non-vacuity assertion in
    ``test_negative_strand_actually_reverse_complements`` reliably fires.

    ``max_jitter=0`` is used here for the simplest deterministic geometry (no
    jitter expansion, so stored interval starts equal query starts).  The #242
    boundary condition (stored interval starts preceding the query start) was
    fixed in both ``intervals_to_tracks`` kernels via the left-clip
    ``s = max(itv.start - query_start, 0)`` (PR #244; #242 CLOSED).
    End-to-end max_jitter>0 parity is covered by
    ``test_tracks_max_jitter_intervals_parity_and_oracle``.
    """
    from genoray import SparseVar
    import polars as pl

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    bw_dir = work_dir / "bw"
    sample_to_bw = _make_session_bigwigs(bw_dir, seed=42)
    track = gvl.BigWigs("signal", sample_to_bw)
    sv = SparseVar(svar_path)

    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2", "chr2"],
            "chromStart": [1010685, 1110686, 1210686, 14360, 1110686],
            "chromEnd": [1010705, 1110706, 1210706, 14380, 1110706],
            "strand": ["+", "-", "+", "-", "+"],
        }
    )

    out = work_dir / "strand_ds.gvl"
    gvl.write(
        path=out,
        bed=bed,
        variants=sv,
        tracks=track,
        max_jitter=0,
        overwrite=True,
    )
    return out


def build_haps_tracks_dataset(work_dir: Path, svar_path: Path) -> Path:
    """Write a variants+tracks GVL dataset and return its path.

    Uses the caller-supplied SparseVar file (which must cover chr1/chr2
    with samples s0/s1/s2, as produced by the session-level build_case
    fixture).  Synthetic BigWig tracks are written with matching samples
    and contigs.  The dataset is written with **max_jitter=0** for the
    simplest deterministic geometry: no jitter expansion, so stored
    interval starts equal the query starts.  This keeps the fixture
    focused on what it exists to test — variants (including indels) that
    trigger ``shift_and_realign_tracks_sparse``.

    #242 / PR #244
    --------------
    The boundary condition where stored interval starts precede the query
    start (``itv.start < query_start``) was root-caused and fixed in both
    ``intervals_to_tracks`` kernels via the left-clip
    ``s = max(itv.start - query_start, 0)`` (PR #244; #242 CLOSED).
    ``max_jitter=0`` here is retained only for the simplest deterministic
    geometry, not because of any live panic or contract violation.
    End-to-end max_jitter>0 parity is covered by
    ``test_tracks_max_jitter_intervals_parity_and_oracle``.

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
                14360,  # overlaps chr2 SNP region
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
    # max_jitter=0: simplest deterministic geometry (no jitter expansion).
    # #242 is fixed via the intervals_to_tracks left-clip (PR #244, #242 CLOSED);
    # max_jitter=0 here keeps interval starts == query starts for straightforward
    # indel-realignment testing. See test_tracks_max_jitter_intervals_parity_and_oracle
    # for max_jitter>0 end-to-end parity coverage.
    gvl.write(
        path=out,
        bed=bed,
        variants=sv,
        tracks=track,
        max_jitter=0,
        overwrite=True,
    )
    return out
