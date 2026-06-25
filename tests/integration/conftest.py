"""Shared fixtures for tests/integration/."""

from __future__ import annotations

from pathlib import Path

import pyBigWig
import pytest

import genvarloader as gvl


@pytest.fixture
def track_dataset_path(source_bed, vcf_dir, tmp_path) -> Path:
    """A freshly-written 2.0 dataset (phased VCF + one BigWig 'cov' track),
    yielded as a writable path so tests may downgrade/migrate it in place.

    Mirrors tests/dataset/conftest.py::snap_dataset but yields a path (not an
    opened Dataset) and is function-scoped so each test gets a mutable copy.
    """
    from genoray import VCF

    samples = ["s0", "s1", "s2"]
    contig_sizes = [("chr1", 2_000_000), ("chr2", 2_000_000)]
    bw_paths: dict[str, str] = {}
    for i, s in enumerate(samples):
        p = tmp_path / f"{s}.bw"
        with pyBigWig.open(str(p), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            v = float(i + 1)
            bw.addEntries(
                ["chr1", "chr1", "chr2", "chr2"],
                [499_990, 1_010_686, 17_320, 1_234_560],
                ends=[500_030, 1_010_706, 17_340, 1_234_580],
                values=[v, v, v, v],
            )
        bw_paths[s] = str(p)
    out = tmp_path / "ds.gvl"
    gvl.write(
        path=out,
        bed=source_bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        tracks=gvl.BigWigs("cov", bw_paths),
        max_jitter=2,
    )
    return out
