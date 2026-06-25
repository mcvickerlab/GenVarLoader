"""Format 2.0 stores track intervals as struct-of-arrays (Task 1)."""

from __future__ import annotations

import json

import numpy as np

import genvarloader as gvl
from genvarloader._dataset._write import DATASET_FORMAT_VERSION


def test_dataset_version_is_2(track_dataset_path):
    assert str(DATASET_FORMAT_VERSION) == "2.0.0"
    meta = json.loads((track_dataset_path / "metadata.json").read_text())
    assert meta["format_version"] == "2.0.0"


def test_soa_files_present_and_aos_absent(track_dataset_path):
    track_dir = track_dataset_path / "intervals" / "cov"
    assert (track_dir / "starts.npy").exists()
    assert (track_dir / "ends.npy").exists()
    assert (track_dir / "values.npy").exists()
    assert (track_dir / "offsets.npy").exists()
    assert not (track_dir / "intervals.npy").exists()


def test_soa_files_contiguous_and_typed(track_dataset_path):
    track_dir = track_dataset_path / "intervals" / "cov"
    starts = np.memmap(track_dir / "starts.npy", dtype=np.int32, mode="r")
    ends = np.memmap(track_dir / "ends.npy", dtype=np.int32, mode="r")
    values = np.memmap(track_dir / "values.npy", dtype=np.float32, mode="r")
    assert starts.flags["C_CONTIGUOUS"]
    assert ends.flags["C_CONTIGUOUS"]
    assert values.flags["C_CONTIGUOUS"]
    assert len(starts) == len(ends) == len(values)


def test_reads_back(track_dataset_path, reference):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference).with_tracks("cov")
    out = ds[0, 0]
    assert out is not None
