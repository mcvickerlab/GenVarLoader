"""Open-time format_version gate (Task 2)."""

from __future__ import annotations

import json
import shutil

import pytest

import genvarloader as gvl


def _set_version(path, version):
    meta_path = path / "metadata.json"
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = version
    meta_path.write_text(json.dumps(raw))


def test_old_major_raises_migrate_hint(track_dataset_path, reference):
    _set_version(track_dataset_path, "1.0.0")
    with pytest.raises(ValueError, match="migrate"):
        gvl.Dataset.open(track_dataset_path, reference=reference)


def test_none_version_raises_migrate_hint(track_dataset_path, reference, tmp_path):
    dst = tmp_path / "noneversion.gvl"
    shutil.copytree(track_dataset_path, dst)
    meta_path = dst / "metadata.json"
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = None
    meta_path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="migrate"):
        gvl.Dataset.open(dst, reference=reference)


def test_future_major_raises_upgrade_hint(track_dataset_path, reference):
    _set_version(track_dataset_path, "3.0.0")
    with pytest.raises(ValueError, match="[Uu]pgrade"):
        gvl.Dataset.open(track_dataset_path, reference=reference)


def test_current_major_opens(track_dataset_path, reference):
    # written fresh at 2.0.0 by the fixture
    ds = gvl.Dataset.open(track_dataset_path, reference=reference)
    assert ds is not None
