"""Error-path coverage for Dataset.open / OpenRequest.resolve.

Each test creates the minimum on-disk state needed to trigger a specific
error condition in ``_open.py``, then asserts the actual exception type
raised (not a substring of the message).
"""

from __future__ import annotations

import json
from pathlib import Path

import genvarloader as gvl
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_minimal_metadata(path: Path, *, ploidy: int | None = None) -> None:
    """Write the minimum valid ``metadata.json`` into *path* (must already exist)."""
    meta = {
        "samples": ["s1"],
        "contigs": ["chr1"],
        "n_regions": 1,
        "max_jitter": 0,
        "ploidy": ploidy,
        "version": None,
        "svar_link": None,
    }
    (path / "metadata.json").write_text(json.dumps(meta))


def _write_minimal_regions(path: Path) -> None:
    """Write a minimal ``input_regions.arrow`` into *path* (must already exist)."""
    table = pa.table(
        {
            "chrom": pa.array(["chr1"], type=pa.large_utf8()),
            "chromStart": pa.array([0], type=pa.int32()),
            "chromEnd": pa.array([100], type=pa.int32()),
            "r_idx_map": pa.array([0], type=pa.int64()),
        }
    )
    with pa_ipc.new_file(path / "input_regions.arrow", table.schema) as writer:
        writer.write_table(table)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_open_missing_dir_raises(tmp_path: Path) -> None:
    """Opening a path that does not exist raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        gvl.Dataset.open(tmp_path / "does_not_exist.gvl")


def test_open_dir_without_metadata_raises(tmp_path: Path) -> None:
    """Directory exists but lacks metadata.json — raises FileNotFoundError."""
    bad = tmp_path / "empty.gvl"
    bad.mkdir()
    with pytest.raises(FileNotFoundError):
        gvl.Dataset.open(bad)


def test_open_dir_without_regions_raises(tmp_path: Path) -> None:
    """metadata.json present but input_regions.arrow missing — raises FileNotFoundError."""
    ds = tmp_path / "no_regions.gvl"
    ds.mkdir()
    _write_minimal_metadata(ds)
    # no input_regions.arrow written
    with pytest.raises(FileNotFoundError):
        gvl.Dataset.open(ds)


def test_open_genotypes_without_ploidy_raises(tmp_path: Path) -> None:
    """genotypes/ dir present but metadata has ploidy=None — raises ValueError."""
    ds = tmp_path / "no_ploidy.gvl"
    ds.mkdir()
    _write_minimal_metadata(ds, ploidy=None)
    _write_minimal_regions(ds)
    (ds / "genotypes").mkdir()  # presence triggers the ploidy check

    with pytest.raises(ValueError, match="ploidy"):
        gvl.Dataset.open(ds)


def test_open_empty_dataset_raises(tmp_path: Path) -> None:
    """No genotypes/ and no intervals/ — raises RuntimeError (malformed dataset)."""
    ds = tmp_path / "no_seqs_no_tracks.gvl"
    ds.mkdir()
    _write_minimal_metadata(ds)
    _write_minimal_regions(ds)
    # neither genotypes/ nor intervals/ written

    with pytest.raises(RuntimeError, match="neither genotypes nor intervals"):
        gvl.Dataset.open(ds)
