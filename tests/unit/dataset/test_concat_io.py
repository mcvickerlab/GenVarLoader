"""Unit tests for gvl.concat's buffered streaming IO primitives."""

import errno
from pathlib import Path

import numpy as np
import pytest

from genvarloader._dataset._concat_io import (
    copy_runs,
    gather_fixed,
    link_or_copy_buffered,
)
from genvarloader._dataset._concat_plan import Run


def _write_raw(p, arr):
    with open(p, "wb") as f:
        f.write(np.ascontiguousarray(arr).tobytes())


def test_copy_runs_concatenates_two_ragged_sources(tmp_path):
    # A: 2 slots holding [1,2] and [3]; B: 1 slot holding [4,5,6]
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    _write_raw(a, np.array([1, 2, 3], np.int32))
    _write_raw(b, np.array([4, 5, 6], np.int32))
    off_a = np.array([0, 2, 3], np.int64)
    off_b = np.array([0, 3], np.int64)

    dst = tmp_path / "out.npy"
    runs = [Run(0, 0, 2, 0), Run(1, 0, 1, 2)]
    merged = copy_runs([a, b], dst, runs, [off_a, off_b], itemsize=4)

    assert merged.tolist() == [0, 2, 3, 6]
    got = np.frombuffer(dst.read_bytes(), np.int32)
    assert got.tolist() == [1, 2, 3, 4, 5, 6]


def test_copy_runs_interleaves_out_of_order_runs(tmp_path):
    # sample-axis shape: A slot0, B slot0, A slot1, B slot1
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    _write_raw(a, np.array([10, 11, 12], np.int32))  # slots [10,11], [12]
    _write_raw(b, np.array([20, 21], np.int32))  # slots [20], [21]
    off_a = np.array([0, 2, 3], np.int64)
    off_b = np.array([0, 1, 2], np.int64)

    dst = tmp_path / "out.npy"
    runs = [Run(0, 0, 1, 0), Run(1, 0, 1, 1), Run(0, 1, 2, 2), Run(1, 1, 2, 3)]
    merged = copy_runs([a, b], dst, runs, [off_a, off_b], itemsize=4)

    assert merged.tolist() == [0, 2, 3, 4, 5]
    got = np.frombuffer(dst.read_bytes(), np.int32)
    assert got.tolist() == [10, 11, 20, 12, 21]


def test_copy_runs_handles_empty_slots(tmp_path):
    a = tmp_path / "a.npy"
    _write_raw(a, np.array([7], np.int32))
    off_a = np.array([0, 0, 1, 1], np.int64)  # slot0 empty, slot1 has [7], slot2 empty

    dst = tmp_path / "out.npy"
    merged = copy_runs([a], dst, [Run(0, 0, 3, 0)], [off_a], itemsize=4)
    assert merged.tolist() == [0, 0, 1, 1]
    assert np.frombuffer(dst.read_bytes(), np.int32).tolist() == [7]


def test_copy_runs_spans_multiple_chunks(tmp_path, monkeypatch):
    """Force >1 chunk to exercise the streaming loop."""
    monkeypatch.setattr("genvarloader._dataset._concat_io.CONCAT_CHUNK_BYTES", 64)
    a = tmp_path / "a.npy"
    data = np.arange(1000, dtype=np.int32)
    _write_raw(a, data)
    off_a = np.array([0, 1000], np.int64)

    dst = tmp_path / "out.npy"
    merged = copy_runs([a], dst, [Run(0, 0, 1, 0)], [off_a], itemsize=4)
    assert merged.tolist() == [0, 1000]
    assert np.frombuffer(dst.read_bytes(), np.int32).tolist() == data.tolist()


def test_gather_fixed_reorders_records(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    _write_raw(a, np.array([[0, 1], [2, 3]], np.int64))
    _write_raw(b, np.array([[4, 5]], np.int64))

    dst = tmp_path / "out.npy"
    runs = [Run(0, 0, 1, 0), Run(1, 0, 1, 1), Run(0, 1, 2, 2)]
    gather_fixed([a, b], dst, runs, record_bytes=16)

    got = np.frombuffer(dst.read_bytes(), np.int64).reshape(-1, 2)
    assert got.tolist() == [[0, 1], [4, 5], [2, 3]]


def test_link_or_copy_produces_identical_bytes(tmp_path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"variant table bytes")
    dst = tmp_path / "dst.bin"
    link_or_copy_buffered(src, dst)
    assert dst.read_bytes() == src.read_bytes()
    # Must be an actual hardlink (same inode), not a copy that merely happens
    # to have matching bytes.
    src_stat, dst_stat = src.stat(), dst.stat()
    assert (dst_stat.st_dev, dst_stat.st_ino) == (src_stat.st_dev, src_stat.st_ino)


def test_link_or_copy_falls_back_on_exdev(tmp_path, monkeypatch):
    def _raise_exdev(self, target):
        raise OSError(errno.EXDEV, "Invalid cross-device link")

    monkeypatch.setattr(Path, "hardlink_to", _raise_exdev)

    src = tmp_path / "src.bin"
    src.write_bytes(b"variant table bytes")
    dst = tmp_path / "dst.bin"
    link_or_copy_buffered(src, dst)

    assert dst.read_bytes() == src.read_bytes()
    # A real copy landed, not a link: distinct inode proves the fallback ran.
    src_stat, dst_stat = src.stat(), dst.stat()
    assert (dst_stat.st_dev, dst_stat.st_ino) != (src_stat.st_dev, src_stat.st_ino)


def test_link_or_copy_reraises_non_exdev_errors(tmp_path, monkeypatch):
    def _raise_eperm(self, target):
        raise OSError(errno.EPERM, "Operation not permitted")

    monkeypatch.setattr(Path, "hardlink_to", _raise_eperm)

    src = tmp_path / "src.bin"
    src.write_bytes(b"variant table bytes")
    dst = tmp_path / "dst.bin"
    with pytest.raises(OSError) as exc_info:
        link_or_copy_buffered(src, dst)
    assert exc_info.value.errno == errno.EPERM
    assert not dst.exists()


def test_copy_runs_writes_nothing_for_no_runs(tmp_path):
    dst = tmp_path / "out.npy"
    merged = copy_runs([], dst, [], [], itemsize=4)
    assert merged.tolist() == [0]
    assert dst.read_bytes() == b""
