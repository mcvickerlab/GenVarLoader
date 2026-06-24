"""On-the-fly fixture builders for dataset-level parity tests."""

from __future__ import annotations

from pathlib import Path

import genvarloader as gvl
from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions, make_synthetic_bigwigs


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
