"""Buffered streaming IO for :func:`genvarloader.concat`.

Everything here uses buffered ``read``/``write`` rather than ``np.memmap``. On an
NFSv3 mount a memmap in the copy path — read side, write side, or both — runs about
5-6x slower than buffered IO, because page faults go out as 4 KiB RPCs instead of
using the mount's 1 MiB ``rsize``/``wsize``. On local XFS the two are equivalent, so
buffered IO is the safe unconditional choice. See the design doc's "Measured basis".
"""

from __future__ import annotations

import errno
import shutil
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ._concat_plan import CONCAT_CHUNK_BYTES, Run

__all__ = ["copy_runs", "gather_fixed", "link_or_copy_buffered"]


def _stream_range(fi, fo, start: int, stop: int) -> None:
    """Copy bytes ``[start, stop)`` from ``fi`` to ``fo`` in bounded chunks."""
    fi.seek(start)
    remaining = stop - start
    while remaining > 0:
        n = min(CONCAT_CHUNK_BYTES, remaining)
        buf = fi.read(n)
        if not buf:
            raise OSError(
                f"unexpected EOF: wanted {remaining} more bytes from {fi.name}"
            )
        fo.write(buf)
        remaining -= len(buf)


def copy_runs(
    srcs: list[Path],
    dst: Path,
    runs: list[Run],
    src_offsets: list[NDArray[np.int64]],
    itemsize: int,
) -> NDArray[np.int64]:
    """Stream a ragged payload through a run plan and return merged offsets.

    Args:
        srcs: Payload file per input dataset (raw, headerless arrays).
        dst: Destination payload file, created or truncated.
        runs: Destination-ordered runs from :func:`._concat_plan.coalesce`.
        src_offsets: Cumulative offsets per input dataset, each of length
            ``n_source_slots + 1``, in elements (not bytes).
        itemsize: Bytes per element of the payload dtype.

    Returns:
        Merged cumulative offsets, length ``total_merged_slots + 1``, in elements.
    """
    n_slots = sum(r.src_stop - r.src_start for r in runs)
    lengths = np.empty(n_slots, np.int64)
    for r in runs:
        off = src_offsets[r.src]
        n = r.src_stop - r.src_start
        lengths[r.dst_start : r.dst_start + n] = (
            off[r.src_start + 1 : r.src_stop + 1] - off[r.src_start : r.src_stop]
        )

    merged = np.empty(n_slots + 1, np.int64)
    merged[0] = 0
    np.cumsum(lengths, out=merged[1:])

    handles: dict[int, object] = {}
    try:
        with open(dst, "wb") as fo:
            for r in runs:
                if r.src not in handles:
                    handles[r.src] = open(srcs[r.src], "rb")
                fi = handles[r.src]
                off = src_offsets[r.src]
                start = int(off[r.src_start]) * itemsize
                stop = int(off[r.src_stop]) * itemsize
                if stop > start:
                    _stream_range(fi, fo, start, stop)
            fo.flush()
    finally:
        for fh in handles.values():
            fh.close()

    return merged


def gather_fixed(
    srcs: list[Path],
    dst: Path,
    runs: list[Run],
    record_bytes: int,
) -> None:
    """Gather fixed-size records through a run plan.

    Used for stores whose values are absolute ranges into an external ``.svar`` /
    ``.svar2`` store: the values copy verbatim, so only their order changes.

    Args:
        srcs: Source file per input dataset.
        dst: Destination file, created or truncated.
        runs: Destination-ordered runs.
        record_bytes: Bytes per slot.
    """
    handles: dict[int, object] = {}
    try:
        with open(dst, "wb") as fo:
            for r in runs:
                if r.src not in handles:
                    handles[r.src] = open(srcs[r.src], "rb")
                _stream_range(
                    handles[r.src],
                    fo,
                    r.src_start * record_bytes,
                    r.src_stop * record_bytes,
                )
            fo.flush()
    finally:
        for fh in handles.values():
            fh.close()


def link_or_copy_buffered(src: Path, dst: Path) -> None:
    """Hardlink ``src`` to ``dst``, falling back to a buffered copy across devices.

    The hardlink is deliberate: gvl reads ``variants.arrow`` with
    ``memory_map=False`` and genoray rewrites its indexes atomically (temp file +
    ``os.replace``, which never touches the old inode), so aliasing the source is
    safe. Integrity is guarded by a recorded fingerprint rather than by copying.

    Args:
        src: Existing source file.
        dst: Destination path, must not already exist.
    """
    try:
        dst.hardlink_to(src)
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
        shutil.copyfile(src, dst)
