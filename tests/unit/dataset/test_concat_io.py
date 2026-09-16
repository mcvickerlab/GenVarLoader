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


def test_copy_runs_accepts_a_run_plan_and_matches_an_explicit_list(tmp_path):
    """Driving copy_runs from RunPlan must be byte-identical to the old path."""
    from genvarloader._dataset._concat_plan import RunPlan, coalesce, provenance

    shapes = [(2, 2), (2, 1)]
    axis, ploidy = "samples", 2
    n_src = [r * s * ploidy for r, s in shapes]

    srcs, offsets = [], []
    for d, n in enumerate(n_src):
        lens = np.arange(1, n + 1, dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        payload = np.arange(off[-1], dtype=np.int32) + d * 1000
        p = tmp_path / f"src{d}.bin"
        _write_raw(p, payload)
        srcs.append(p)
        offsets.append(off)

    out_plan = tmp_path / "plan.npy"
    out_list = tmp_path / "list.npy"
    plan = RunPlan(axis, shapes, ploidy)
    merged_plan = copy_runs(srcs, out_plan, plan, offsets, itemsize=4)
    merged_list = copy_runs(
        srcs, out_list, coalesce(provenance(axis, shapes, ploidy)), offsets, itemsize=4
    )

    np.testing.assert_array_equal(merged_plan, merged_list)
    assert out_plan.read_bytes() == out_list.read_bytes()


def test_copy_runs_in_place_cumsum_matches_out_of_place(tmp_path):
    """merged[1:] is cumsummed into itself; pin that against the naive form."""
    lens = np.array([3, 0, 5, 2, 7], np.int64)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    payload = np.arange(off[-1], dtype=np.int32)
    p = tmp_path / "src.bin"
    _write_raw(p, payload)

    runs = [Run(0, 0, 5, 0)]
    merged = copy_runs([p], tmp_path / "out.npy", runs, [off], itemsize=4)

    want = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.testing.assert_array_equal(merged, want)


def test_gather_fixed_from_a_run_plan_is_byte_identical_to_the_oracle(tmp_path):
    """gather_fixed must produce the same bytes from a plan as from a run list."""
    from genvarloader._dataset._concat_plan import RunPlan, coalesce, provenance

    shapes, axis, ploidy = [(2, 2), (3, 2)], "regions", 1
    srcs = []
    for d, (r, sm) in enumerate(shapes):
        p = tmp_path / f"g{d}.bin"
        _write_raw(p, np.arange(r * sm * ploidy, dtype=np.int32) + d * 100)
        srcs.append(p)

    plan = RunPlan(axis, shapes, ploidy)
    out_plan, out_list = tmp_path / "plan.bin", tmp_path / "list.bin"
    gather_fixed(srcs, out_plan, plan, record_bytes=4)
    gather_fixed(
        srcs, out_list, coalesce(provenance(axis, shapes, ploidy)), record_bytes=4
    )

    assert out_plan.read_bytes() == out_list.read_bytes()
    assert out_plan.stat().st_size == plan.n_slots * 4


@pytest.mark.parametrize(
    ("axis", "shapes", "order"),
    [
        # axis="samples" requires equal n_regions (index 0) across datasets.
        ("samples", [(2, 2), (2, 1)], None),
        # Interleaved merged-sample order: ds0 sample0, ds1 sample0, ds0
        # sample1 -- built the same way test_concat_plan.py's interleaved-order
        # tests do, with an explicit (dataset_idx, within_idx) literal.
        ("samples", [(2, 2), (2, 1)], np.array([[0, 0], [1, 0], [0, 1]])),
        # axis="regions" requires equal n_samples (index 1) across datasets.
        ("regions", [(2, 2), (3, 2)], None),
        # Interleaved merged-region order over ds0's 2 regions and ds1's 3.
        (
            "regions",
            [(2, 2), (3, 2)],
            np.array([[1, 0], [0, 0], [1, 1], [0, 1], [1, 2]]),
        ),
    ],
    ids=[
        "samples-block",
        "samples-interleaved",
        "regions-block",
        "regions-interleaved",
    ],
)
def test_gather_svar_offsets_from_a_run_plan_is_byte_identical(
    tmp_path, axis, shapes, order
):
    """The svar offsets gather must agree with the retained oracle.

    This one cannot go through `gather_fixed`: its slot axis is nested inside
    two leading planes, so a slot's start and stop live `n_slots` elements
    apart. It therefore has its own run-consuming loop, and needs its own pin.
    Parametrized over both axes and both an `order=None` block merge and an
    interleaved `order`, since every production call site always passes an
    explicit `order` and interleaving is the case the design calls load-bearing.
    """
    from genvarloader._dataset._concat import _gather_svar_offsets
    from genvarloader._dataset._concat_plan import RunPlan, coalesce, provenance

    ploidy = 2

    paths = []
    for d, (r, sm) in enumerate(shapes):
        n = r * sm * ploidy
        p = tmp_path / f"ds{d}"
        (p / "genotypes").mkdir(parents=True)
        starts = np.arange(n, dtype=np.int64) + d * 100
        planes = np.stack([starts, starts + 1])
        _write_raw(p / "genotypes" / "offsets.npy", planes)
        paths.append(p)

    out_plan, out_list = tmp_path / "plan", tmp_path / "list"
    out_plan.mkdir()
    out_list.mkdir()

    _gather_svar_offsets(
        paths, out_plan, RunPlan(axis, shapes, ploidy, order=order), shapes, ploidy
    )
    _gather_svar_offsets(
        paths,
        out_list,
        coalesce(provenance(axis, shapes, ploidy, order=order)),
        shapes,
        ploidy,
    )

    plan_bytes = (out_plan / "offsets.npy").read_bytes()
    assert plan_bytes == (out_list / "offsets.npy").read_bytes()
    # A shared bug that writes one plane instead of two would still pass the
    # byte-identity check above (both paths would be wrong the same way), so
    # also pin the absolute size against a freshly built plan's `n_slots`.
    expected_n_slots = RunPlan(axis, shapes, ploidy, order=order).n_slots
    assert len(plan_bytes) == 2 * expected_n_slots * 8
