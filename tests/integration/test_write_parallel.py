"""Integration test: parallel gvl.write (tracks ∥ annot_tracks) matches sequential.

Proof of correctness: write the same data via the parallel path (2 job categories →
loky) and independently via two single-category inline writes, then compare the raw
interval bytes on disk.
"""

from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig
import pytest
from genoray import VCF

import genvarloader as gvl


# ---------------------------------------------------------------------------
# Fixture: per-sample BigWigs whose sample set is {s0, s1, s2} (matching the
# filtered_source.vcf.gz fixture).  Written into tmp_path so they are
# function-scoped and do not interfere across tests.
# ---------------------------------------------------------------------------

VCF_SAMPLES = ["s0", "s1", "s2"]
CONTIG_SIZES = [("chr1", 2_000_000), ("chr2", 2_000_000)]

# Use chr1:100-200 — overlaps the first cluster of VCF variants (~POS 111).
# The bigwig entries at chr1:[1-5) and chr1:[100-105) sit inside this window.
BED = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})


@pytest.fixture()
def bigwigs(tmp_path: Path) -> gvl.BigWigs:
    """BigWigs with one bw per VCF sample, covering chr1:100-200."""
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(VCF_SAMPLES):
        bw_path = tmp_path / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(CONTIG_SIZES, maxZooms=0)
            # one entry inside the query window
            value = float(i + 1)
            bw.addEntries(["chr1"], [100], ends=[110], values=[value])
        bw_paths[sample] = str(bw_path)
    return gvl.BigWigs("signal", bw_paths)


@pytest.fixture()
def annot_bw(tmp_path: Path) -> Path:
    """A single sample-less bigwig for annotation (also covers chr1:100-200)."""
    bw_path = tmp_path / "annot.bw"
    with pyBigWig.open(str(bw_path), "w") as bw:
        bw.addHeader(CONTIG_SIZES, maxZooms=0)
        bw.addEntries(["chr1"], [105], ends=[115], values=[7.0])
    return bw_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_intervals(ds_path: Path, subdir: str, name: str) -> np.ndarray:
    """Load intervals.npy from ``ds_path/<subdir>/<name>/intervals.npy``."""
    return np.array(np.memmap(ds_path / subdir / name / "intervals.npy", mode="r"))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_parallel_write_matches_sequential(
    bigwigs: gvl.BigWigs,
    annot_bw: Path,
    vcf_dir: Path,
    ref_fasta: Path,
    tmp_path: Path,
) -> None:
    """Parallel write (tracks ∥ annot_tracks, loky) must produce bytes equal to
    two independent single-category inline writes."""
    vcf = VCF(vcf_dir / "filtered_source.vcf.gz")

    a_dir = tmp_path / "a"  # parallel: tracks + annot_tracks → 2 jobs → loky
    b_dir = tmp_path / "b"  # tracks only → 1 job → inline
    c_dir = tmp_path / "c"  # annot_tracks only → 1 job → inline

    gvl.write(
        a_dir,
        BED,
        variants=vcf,
        tracks=bigwigs,
        annot_tracks={"ann": annot_bw},
    )

    vcf2 = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(b_dir, BED, variants=vcf2, tracks=bigwigs)

    vcf3 = VCF(vcf_dir / "filtered_source.vcf.gz")
    gvl.write(c_dir, BED, variants=vcf3, annot_tracks={"ann": annot_bw})

    # --- compare track bytes ---
    a_track = _load_intervals(a_dir, "intervals", "signal")
    b_track = _load_intervals(b_dir, "intervals", "signal")
    assert np.array_equal(a_track, b_track), (
        f"Track intervals differ between parallel (a) and sequential (b):\n"
        f"a={a_track}\nb={b_track}"
    )

    # --- compare annot bytes ---
    a_annot = _load_intervals(a_dir, "annot_intervals", "ann")
    c_annot = _load_intervals(c_dir, "annot_intervals", "ann")
    assert np.array_equal(a_annot, c_annot), (
        f"Annot intervals differ between parallel (a) and sequential (c):\n"
        f"a={a_annot}\nc={c_annot}"
    )
