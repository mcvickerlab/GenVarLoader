"""Pure-logic tests for the dataloader bench. Run explicitly:

    pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v

These are NOT collected by the default `pixi run -e dev test` task.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import _common as C


def test_axis_constants_match_spec():
    assert C.OUTPUTS == ["haplotypes", "annotated", "variants"]
    assert C.MODES_NEW == ["buffered", "double_buffered"]
    assert C.ALL_MODES == [None, "buffered", "double_buffered"]

    assert C.THREADS_FACT == [1, 8]
    assert C.REGION_FACT == [1_000, 10_000]
    assert C.BATCH_FACT == [16, 128]
    assert C.BUFFER_FACT == [256 * C.MiB, 2 * C.GiB]

    assert C.THREADS_FAN == [2, 4, 16]
    assert C.REGION_FAN == [2_500, 5_000, 25_000]
    assert C.BATCH_FAN == [32, 64, 256]
    assert C.BUFFER_FAN == [512 * C.MiB, 1 * C.GiB, 4 * C.GiB]

    assert C.THREADS_MID == 4
    assert C.REGION_MID == 5_000
    assert C.BATCH_MID == 64
    assert C.BUFFER_MID == 1 * C.GiB


def test_dispatch_unions():
    assert C.ALL_THREADS == [1, 2, 4, 8, 16]
    assert C.REGION_LENGTHS == [1_000, 2_500, 5_000, 10_000, 25_000]
    # midpoints are members of their own fans
    assert C.THREADS_MID in C.THREADS_FAN
    assert C.REGION_MID in C.REGION_FAN
    assert C.BATCH_MID in C.BATCH_FAN
    assert C.BUFFER_MID in C.BUFFER_FAN
