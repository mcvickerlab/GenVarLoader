"""Smoke tests for BigWigs reader."""

from pathlib import Path

import numpy as np
import pytest

from genvarloader import BigWigs


def test_bigwigs_read_basic(bigwig_dir: Path):
    """Smoke: read a range on chr1, verify shape and dtype."""
    paths = {
        "sample_0": str(bigwig_dir / "sample_0.bw"),
        "sample_1": str(bigwig_dir / "sample_1.bw"),
    }
    bws = BigWigs("signal", paths)
    # chr1 is 2000 bp; read 200 bp around the stored intervals
    out = bws.read("chr1", np.array([0]), np.array([200]))
    assert out.dtype == np.float32
    # shape is (n_samples, total_length) — 2 samples, 200 bp
    assert out.shape == (2, 200)


def test_bigwigs_reader_protocol_attrs(bigwig_dir: Path):
    """BigWigs exposes the Reader protocol attributes correctly."""
    paths = {
        "sample_0": str(bigwig_dir / "sample_0.bw"),
        "sample_1": str(bigwig_dir / "sample_1.bw"),
    }
    bws = BigWigs("signal", paths)
    assert bws.name == "signal"
    assert bws.dtype == np.float32
    # contigs dict should contain chr1 and chr2
    assert "chr1" in bws.contigs
    assert "chr2" in bws.contigs


def test_bigwigs_missing_path_raises():
    """Constructor raises when a path does not exist."""
    paths = {"sample_x": "/nonexistent/path/sample.bw"}
    with pytest.raises(RuntimeError):
        BigWigs("signal", paths)
