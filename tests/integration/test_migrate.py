"""gvl.migrate: 1.x AoS -> 2.0 SoA round-trip, idempotency, crash-safety (Task 3)."""

from __future__ import annotations

import json

import numpy as np

import genvarloader as gvl
from genvarloader._ragged import INTERVAL_DTYPE


def _track_dirs(path):
    for base in ("intervals", "annot_intervals"):
        d = path / base
        if d.is_dir():
            for child in sorted(d.iterdir()):
                if child.is_dir():
                    yield child


def _downgrade_to_aos(path):
    """Rewrite a fresh 2.0 SoA dataset back to a 1.x AoS dataset in place."""
    for d in _track_dirs(path):
        starts = np.memmap(d / "starts.npy", dtype=np.int32, mode="r")
        ends = np.memmap(d / "ends.npy", dtype=np.int32, mode="r")
        values = np.memmap(d / "values.npy", dtype=np.float32, mode="r")
        rec = np.empty(len(starts), dtype=INTERVAL_DTYPE)
        rec["start"] = starts
        rec["end"] = ends
        rec["value"] = values
        out = np.memmap(
            d / "intervals.npy", dtype=INTERVAL_DTYPE, mode="w+", shape=rec.shape
        )
        out[:] = rec
        out.flush()
        del starts, ends, values, out
        (d / "starts.npy").unlink()
        (d / "ends.npy").unlink()
        (d / "values.npy").unlink()
    meta_path = path / "metadata.json"
    raw = json.loads(meta_path.read_text())
    raw["format_version"] = "1.0.0"
    meta_path.write_text(json.dumps(raw))


def _read_track_values(ds):
    """Return the raw realigned track float values for region 0, sample 0.

    With both seqs and tracks active, [0, 0] returns a 2-tuple (seq, tracks).
    We take the last element (tracks), which is a Ragged[float32] / RaggedTracks,
    and return its flat data buffer for byte-identical comparison.
    """
    result = ds.with_tracks("cov")[0, 0]
    # When both seqs and tracks are active the result is a 2-tuple; take tracks.
    trk = result[-1] if isinstance(result, tuple) else result
    return trk.data.copy()


def test_round_trip_byte_identical(track_dataset_path, reference):
    ds = gvl.Dataset.open(track_dataset_path, reference=reference)
    before = _read_track_values(ds)

    _downgrade_to_aos(track_dataset_path)
    gvl.migrate(track_dataset_path)

    track_dir = track_dataset_path / "intervals" / "cov"
    assert (track_dir / "starts.npy").exists()
    assert (track_dir / "ends.npy").exists()
    assert (track_dir / "values.npy").exists()
    assert not (track_dir / "intervals.npy").exists()
    assert (
        json.loads((track_dataset_path / "metadata.json").read_text())["format_version"]
        == "2.0.0"
    )

    after = gvl.Dataset.open(track_dataset_path, reference=reference)
    np.testing.assert_array_equal(_read_track_values(after), before)


def test_idempotent(track_dataset_path):
    _downgrade_to_aos(track_dataset_path)
    gvl.migrate(track_dataset_path)
    gvl.migrate(track_dataset_path)  # second run is a no-op, must not raise
    track_dir = track_dataset_path / "intervals" / "cov"
    assert not (track_dir / "intervals.npy").exists()


def test_resumable_after_interrupt_before_metadata_bump(track_dataset_path):
    """Crash after SoA written but before metadata bump: still 1.x, re-runnable."""
    _downgrade_to_aos(track_dataset_path)
    # Simulate partial migration: write SoA, leave AoS + 1.x metadata.
    from genvarloader._dataset._migrate import _migrate_track

    for d in _track_dirs(track_dataset_path):
        _migrate_track(d)
    meta = json.loads((track_dataset_path / "metadata.json").read_text())
    assert meta["format_version"] == "1.0.0"  # not bumped yet
    track_dir = track_dataset_path / "intervals" / "cov"
    assert (track_dir / "intervals.npy").exists()  # AoS still present

    gvl.migrate(track_dataset_path)  # completes the migration
    assert (
        json.loads((track_dataset_path / "metadata.json").read_text())["format_version"]
        == "2.0.0"
    )
    assert not (track_dir / "intervals.npy").exists()


def test_cleans_leftover_aos_after_interrupt_before_delete(track_dataset_path):
    """Crash after metadata bump but before AoS delete: re-run removes AoS."""
    _downgrade_to_aos(track_dataset_path)
    gvl.migrate(track_dataset_path)  # full migration -> SoA + 2.0 metadata
    track_dir = track_dataset_path / "intervals" / "cov"
    # Re-introduce a leftover AoS file (as if delete was interrupted).
    starts = np.memmap(track_dir / "starts.npy", dtype=np.int32, mode="r")
    rec = np.zeros(len(starts), dtype=INTERVAL_DTYPE)
    out = np.memmap(
        track_dir / "intervals.npy", dtype=INTERVAL_DTYPE, mode="w+", shape=rec.shape
    )
    out[:] = rec
    out.flush()
    del starts, out

    gvl.migrate(track_dataset_path)  # idempotent cleanup
    assert not (track_dir / "intervals.npy").exists()
