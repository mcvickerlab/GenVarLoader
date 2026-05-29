"""Shared config + helpers for the dataloader throughput bench.

Pure-logic functions (axis constants, cell enumeration) are unit-tested in
``tests/test_common.py``. The dataset-prep and measurement helpers consume the
prefetching-dataloader public API and are exercised by ``bench.py``.
"""

from __future__ import annotations

MiB = 1024**2
GiB = 1024**3

# ── modes & outputs ──────────────────────────────────────────────────────────
OUTPUTS = ["haplotypes", "annotated", "variants"]
MODES_NEW = ["buffered", "double_buffered"]
ALL_MODES = [None, "buffered", "double_buffered"]

# ── factorial corner values (2 per axis) ─────────────────────────────────────
THREADS_FACT = [1, 8]
REGION_FACT = [1_000, 10_000]
BATCH_FACT = [16, 128]
BUFFER_FACT = [256 * MiB, 2 * GiB]

# ── fan values (each includes its axis midpoint) ─────────────────────────────
THREADS_FAN = [2, 4, 16]
REGION_FAN = [2_500, 5_000, 25_000]
BATCH_FAN = [32, 64, 256]
BUFFER_FAN = [512 * MiB, 1 * GiB, 4 * GiB]

# ── midpoints (other axes pinned here during a 1-axis fan) ────────────────────
THREADS_MID = 4
REGION_MID = 5_000
BATCH_MID = 64
BUFFER_MID = 1 * GiB

# ── unions used for process dispatch / dataset prep ──────────────────────────
ALL_THREADS = sorted(set(THREADS_FACT) | set(THREADS_FAN))
REGION_LENGTHS = sorted(set(REGION_FACT) | set(REGION_FAN))
