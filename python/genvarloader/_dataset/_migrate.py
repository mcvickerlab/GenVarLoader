"""In-place, streaming, idempotent migration of a 1.x AoS dataset to 2.0 SoA.

Per track under ``intervals/<track>/`` and ``annot_intervals/<track>/``:
stream ``intervals.npy`` (INTERVAL_DTYPE) in record chunks into three contiguous
``starts/ends/values.npy`` files. Only after every track's SoA is durable do we
bump ``metadata.json`` (last durable write); then delete the AoS files.

Crash-safety by ordering: an interruption before the metadata bump leaves the
dataset still-1.x (old AoS intact, re-runnable); an interruption after the bump
but before deletion leaves both layouts, and a re-run completes the cleanup.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from loguru import logger
from pydantic_extra_types.semantic_version import SemanticVersion

from .._ragged import INTERVAL_DTYPE
from ._write import DATASET_FORMAT_VERSION

_CHUNK = 1_000_000  # records per streamed block


def _track_dirs(path: Path) -> Iterator[Path]:
    for base in ("intervals", "annot_intervals"):
        d = path / base
        if d.is_dir():
            for child in sorted(d.iterdir()):
                if child.is_dir():
                    yield child


def _migrate_track(track_dir: Path) -> None:
    """Stream one track's AoS intervals.npy into SoA starts/ends/values.npy.

    No-op if intervals.npy is absent (already migrated or never AoS). Leaves the
    AoS file in place; the caller deletes it only after metadata is bumped.
    """
    aos = track_dir / "intervals.npy"
    if not aos.exists():
        return
    src = np.memmap(aos, dtype=INTERVAL_DTYPE, mode="r")
    n = int(src.shape[0])
    starts = np.memmap(track_dir / "starts.npy", dtype=np.int32, mode="w+", shape=n)
    ends = np.memmap(track_dir / "ends.npy", dtype=np.int32, mode="w+", shape=n)
    values = np.memmap(track_dir / "values.npy", dtype=np.float32, mode="w+", shape=n)
    for i in range(0, n, _CHUNK):
        j = min(i + _CHUNK, n)
        block = src[i:j]
        starts[i:j] = block["start"]
        ends[i:j] = block["end"]
        values[i:j] = block["value"]
    for m in (starts, ends, values):
        m.flush()
    logger.info(f"Migrated {n} intervals in {track_dir} to SoA.")
    del src, starts, ends, values


def migrate(path: str | Path) -> None:
    """Migrate a GVL dataset's track intervals from format 1.x (array-of-structs) to format 2.0 (struct-of-arrays), in place.

    Streaming and crash-safe: peak extra disk is one track's interval store.
    Genotypes, regions, and reference are untouched. Idempotent — a no-op (with
    leftover-AoS cleanup) on a dataset that is already 2.0.

    Args:
        path: Path to the GVL dataset directory.
    """
    path = Path(path)
    meta_path = path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json at {meta_path}")
    raw = json.loads(meta_path.read_text())
    fv = raw.get("format_version")
    already_v2 = (
        fv is not None
        and SemanticVersion.parse(fv).major >= DATASET_FORMAT_VERSION.major
    )
    track_dirs = list(_track_dirs(path))

    if already_v2:
        # Idempotent cleanup: remove leftover AoS from an interrupted delete.
        for d in track_dirs:
            aos = d / "intervals.npy"
            if aos.exists() and (d / "starts.npy").exists():
                aos.unlink()
        return

    # 1. Convert every track to SoA (AoS left in place).
    for d in track_dirs:
        _migrate_track(d)

    # 2. Durably bump metadata LAST (atomic replace).
    raw["format_version"] = str(DATASET_FORMAT_VERSION)
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw))
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, meta_path)

    # 3. Delete AoS files.
    for d in track_dirs:
        aos = d / "intervals.npy"
        if aos.exists():
            aos.unlink()
    logger.info(f"Migrated dataset {path} to format {DATASET_FORMAT_VERSION}.")
