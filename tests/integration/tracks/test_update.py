"""Integration tests for gvl.update — post-hoc track addition to an existing dataset."""

from __future__ import annotations

import pytest
import pyBigWig
import polars as pl
from genoray import VCF

import genvarloader as gvl


@pytest.fixture
def phased_vcf(vcf_dir):
    """Opened VCF fixture (samples s0, s1, s2)."""
    return VCF(vcf_dir / "filtered_source.vcf.gz")


@pytest.fixture
def bigwigs(tmp_path):
    """A BigWigs track whose sample set exactly matches the VCF (s0, s1, s2).

    BigWig intervals are written under tmp_path/bw/; tests should use a
    distinct subdir (tmp_path/ds/) for their dataset output to avoid
    path collisions.
    """
    bw_dir = tmp_path / "bw"
    bw_dir.mkdir()
    contig_sizes = [("chr1", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(["s0", "s1", "s2"]):
        bw_path = bw_dir / f"{sample}.bw"
        with pyBigWig.open(str(bw_path), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            bw.addEntries(
                ["chr1"],
                [110],
                ends=[150],
                values=[float(i + 1)],
            )
        bw_paths[sample] = str(bw_path)
    return gvl.BigWigs("signal", bw_paths)


# BED region that overlaps the first VCF variant on chr1 (POS=111)
BED = pl.DataFrame({"chrom": ["chr1"], "chromStart": [100], "chromEnd": [200]})


def test_update_adds_sample_track(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    gvl.write(out, BED, variants=phased_vcf)
    gvl.update(out, tracks=bigwigs)
    ds = gvl.Dataset.open(out, ref_fasta)
    assert bigwigs.name in ds.available_tracks


def test_update_accepts_dataset_object(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    gvl.write(out, BED, variants=phased_vcf)
    ds = gvl.Dataset.open(out, ref_fasta)
    gvl.update(ds, tracks=bigwigs)  # extracts .path
    assert bigwigs.name in gvl.Dataset.open(out, ref_fasta).available_tracks


def test_update_rejects_extra_or_missing_samples(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    # Write dataset with only 2 samples; bigwigs has 3 → extra sample
    gvl.write(out, BED, variants=phased_vcf, samples=list(bigwigs.samples[:-1]))
    with pytest.raises(ValueError, match="sample"):
        gvl.update(out, tracks=bigwigs)


def test_update_overwrite(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    gvl.write(out, BED, variants=phased_vcf)
    gvl.update(out, tracks=bigwigs)
    with pytest.raises(FileExistsError):
        gvl.update(out, tracks=bigwigs)
    gvl.update(out, tracks=bigwigs, overwrite=True)  # ok
    assert not list((out / "intervals").glob("*.tmp.*"))
