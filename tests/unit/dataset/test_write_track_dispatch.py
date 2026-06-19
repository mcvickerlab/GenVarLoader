# tests/unit/dataset/test_write_track_dispatch.py
from pathlib import Path

import numpy as np

from genvarloader import BigWigs
from genvarloader._dataset import _write
from genvarloader._ragged import INTERVAL_DTYPE


def _track(data_dir: Path) -> BigWigs:
    return BigWigs(
        "signal",
        {
            "sample_0": str(data_dir / "sample_0.bw"),
            "sample_1": str(data_dir / "sample_1.bw"),
        },
    )


def test_write_track_rust_writes_files(tmp_path):
    import polars as pl

    data_dir = Path(__file__).parents[2] / "data" / "bigwig"
    track = _track(data_dir)
    bed = pl.DataFrame(
        {"chrom": ["chr1", "chr1"], "chromStart": [0, 50], "chromEnd": [200, 110]}
    )
    out = tmp_path / "signal"
    out.mkdir()
    _write._write_track_rust(out, bed, track, ["sample_0", "sample_1"], 1 << 30)
    itvs = np.memmap(out / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r")
    offsets = np.memmap(out / "offsets.npy", dtype=np.int64, mode="r")
    assert len(offsets) == 2 * 2 + 1
    assert offsets[-1] == len(itvs)


def test_dispatch_env_off_uses_legacy(monkeypatch):
    monkeypatch.delenv("GVL_RUST_BIGWIG_WRITE", raising=False)
    assert _write._rust_bigwig_write_enabled() is False
    monkeypatch.setenv("GVL_RUST_BIGWIG_WRITE", "1")
    assert _write._rust_bigwig_write_enabled() is True
