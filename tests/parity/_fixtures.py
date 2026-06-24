"""Synthetic fixtures for the dataset-level parity backstop.

Builds a small BigWigs track + BED regions on the fly using the corpus
helpers already in tests/_bigwig_corpus.py. No variants, no vcfixture.
"""

from __future__ import annotations

from pathlib import Path

import genvarloader as gvl

from tests._bigwig_corpus import DEFAULT_CONTIGS, make_regions, make_synthetic_bigwigs

# Use 4 samples and 30 regions per contig (60 total across chr21/chr22).
# density=0.01 over 5000-bp windows → ~50 intervals/region/sample →
# mem_per_r ≈ 4 * 50 * 24 = 4800 bytes.
# max_mem=50_000 forces ≥2 splits without tripping the per-region guard.
N_SAMPLES = 4
N_PER_CONTIG = 30
REGION_WIDTH = 5000
# Bytes: total ≈ 60*4800 = 288 KB; max_mem=50 KB → ~6 splits.
# Each region costs ≤ ~4800 B << 50 KB, so the per-region guard won't fire.
MAX_MEM = 50_000


def build_write_inputs(work_dir: Path):
    """Return (bed, bigwigs) suitable for _write_track_legacy / gvl.write.

    Creates synthetic bigWig files under ``work_dir/bigwigs/`` if they do
    not already exist (deterministic — safe to call twice from different
    subprocesses if the caller arranges sequencing).
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    bw_dir = work_dir / "bigwigs"
    bw_dir.mkdir(exist_ok=True)

    paths = make_synthetic_bigwigs(
        bw_dir,
        N_SAMPLES,
        contigs=DEFAULT_CONTIGS,
        density=0.01,
        seed=42,
    )
    samples = [p.stem for p in paths]
    track = gvl.BigWigs("signal", {s: str(p) for s, p in zip(samples, paths)})

    bed = make_regions(
        DEFAULT_CONTIGS, n_per_contig=N_PER_CONTIG, width=REGION_WIDTH, seed=0
    )

    return bed, track
