"""Smoke test for the bulk SVAR2 streaming fixture builder. Skips without vcfixture."""

from __future__ import annotations

import shutil
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _has_vcfixture() -> bool:
    return bool(os.environ.get("VCFIXTURE_BIN") or shutil.which("vcfixture"))


@pytest.mark.skipif(not _has_vcfixture(), reason="vcfixture-rs bulk CLI not available")
def test_build_bulk_svar2_stream_fixture(tmp_path: Path) -> None:
    from tests.benchmarks.data.build_svar2_stream_bulk import build, BulkStreamFixture

    fx = build(tmp_path, n_samples=8, records=200, n_regions=4, region_len=100)
    assert isinstance(fx, BulkStreamFixture)
    assert fx.svar2_path.exists() and (fx.svar2_path / "meta.json").exists()
    assert (fx.gvl_path).exists()  # the gvl.write oracle dataset
    assert fx.reference.exists()
    assert fx.bed.height == 4
    assert set(fx.bed.columns) >= {"chrom", "chromStart", "chromEnd"}
