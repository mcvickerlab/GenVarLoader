"""Shared config + helpers for the dataloader throughput bench.

Pure-logic functions (axis constants, cell enumeration) are unit-tested in
``tests/test_common.py``. The dataset-prep and measurement helpers consume the
prefetching-dataloader public API and are exercised by ``bench.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class Cell:
    """One benchmark configuration. ``buffer_bytes`` is ``None`` for the
    ``mode=None`` baseline (which ignores the buffer budget)."""

    mode: str | None
    with_seqs: str
    threads: int
    region_length: int
    batch_size: int
    buffer_bytes: int | None


def _add(cells: list[Cell], seen: set, cell: Cell) -> None:
    """Append ``cell`` unless an identical configuration was already added."""
    key = (
        cell.mode,
        cell.with_seqs,
        cell.threads,
        cell.region_length,
        cell.batch_size,
        cell.buffer_bytes,
    )
    if key in seen:
        return
    seen.add(key)
    cells.append(cell)


def _new_mode_cells(mode: str, output: str) -> list[Cell]:
    cells: list[Cell] = []
    seen: set = set()
    # 16 factorial corners
    for t in THREADS_FACT:
        for r in REGION_FACT:
            for b in BATCH_FACT:
                for buf in BUFFER_FACT:
                    _add(cells, seen, Cell(mode, output, t, r, b, buf))
    # 4 fans, each pinning the other three axes at midpoint; the shared
    # all-midpoint cell is deduped by _add.
    for v in THREADS_FAN:
        _add(cells, seen, Cell(mode, output, v, REGION_MID, BATCH_MID, BUFFER_MID))
    for v in REGION_FAN:
        _add(cells, seen, Cell(mode, output, THREADS_MID, v, BATCH_MID, BUFFER_MID))
    for v in BATCH_FAN:
        _add(cells, seen, Cell(mode, output, THREADS_MID, REGION_MID, v, BUFFER_MID))
    for v in BUFFER_FAN:
        _add(cells, seen, Cell(mode, output, THREADS_MID, REGION_MID, BATCH_MID, v))
    return cells


def _baseline_cells(output: str) -> list[Cell]:
    cells: list[Cell] = []
    seen: set = set()
    # 8 factorial corners (no buffer budget for mode=None)
    for t in THREADS_FACT:
        for r in REGION_FACT:
            for b in BATCH_FACT:
                _add(cells, seen, Cell(None, output, t, r, b, None))
    # midpoint-anchored fans on the three axes baseline supports
    for v in THREADS_FAN:
        _add(cells, seen, Cell(None, output, v, REGION_MID, BATCH_MID, None))
    for v in REGION_FAN:
        _add(cells, seen, Cell(None, output, THREADS_MID, v, BATCH_MID, None))
    for v in BATCH_FAN:
        _add(cells, seen, Cell(None, output, THREADS_MID, REGION_MID, v, None))
    return cells


def enumerate_cells() -> list[Cell]:
    """All benchmark cells: 150 new-mode + 45 baseline = 195 unique configs."""
    cells: list[Cell] = []
    for output in OUTPUTS:
        for mode in MODES_NEW:
            cells.extend(_new_mode_cells(mode, output))
        cells.extend(_baseline_cells(output))
    return cells


def cells_for_threads(n_threads: int) -> list[Cell]:
    """Subset of cells whose ``threads`` equals ``n_threads`` (a child runs
    only these, since rayon thread count is pinned per child process)."""
    return [c for c in enumerate_cells() if c.threads == n_threads]
