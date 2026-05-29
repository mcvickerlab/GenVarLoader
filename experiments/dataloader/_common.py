"""Shared config + helpers for the dataloader throughput bench.

Pure-logic functions (axis constants, cell enumeration) are unit-tested in
``tests/test_common.py``. The dataset-prep and measurement helpers consume the
prefetching-dataloader public API and are exercised by ``bench.py``.
"""

from __future__ import annotations

import csv as _csv
import gc
import resource
import time
from dataclasses import dataclass
from pathlib import Path

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
# buffer_bytes is the total RAM budget; double_buffered splits it across two
# slots and the producer+consumer each hold a chunk copy, so keep the ceiling
# modest for workstation RAM. Max 512 MiB.
BUFFER_FACT = [64 * MiB, 512 * MiB]

# ── fan values (each includes its axis midpoint) ─────────────────────────────
THREADS_FAN = [2, 4, 16]
REGION_FAN = [2_500, 5_000, 25_000]
BATCH_FAN = [32, 64, 256]
BUFFER_FAN = [128 * MiB, 256 * MiB, 512 * MiB]

# ── midpoints (other axes pinned here during a 1-axis fan) ────────────────────
THREADS_MID = 4
REGION_MID = 5_000
BATCH_MID = 64
BUFFER_MID = 256 * MiB

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


def generate_bed(regions_bed: str | Path, region_length: int):
    """Read the canonical regions BED and recenter+resize every region to
    ``region_length`` (``sp.bed.with_len`` resizes around the midpoint)."""
    import seqpro as sp

    bed = sp.bed.read(regions_bed)
    return sp.bed.with_len(bed, region_length)


def output_bytes_table(dataset):
    """Compute the exact per-instance byte table once for a configured dataset.

    Returns ``(instances, total_bytes, table)`` where ``table`` is the
    ``(n_regions, n_samples)`` int64 array from
    ``Dataset._output_bytes_per_instance``. For a full epoch with shuffle off,
    ``instances == table.size`` and ``total_bytes == table.sum()``.
    """
    table = dataset._output_bytes_per_instance()
    return int(table.size), int(table.sum()), table


def prepare_datasets(
    region_lengths: list[int],
    svar_path: str | Path,
    regions_bed: str | Path,
    tmp_dir: str | Path,
) -> dict[int, Path]:
    """Write one fresh ``.gvl`` dataset per region length, keyed by length.

    Amortized once at bench startup. Returns ``{length: dataset_path}``.
    """
    import genvarloader as gvl

    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    out: dict[int, Path] = {}
    for length in region_lengths:
        bed = generate_bed(regions_bed, length)
        ds_path = tmp_dir / f"dataset_rL{length}.gvl"
        gvl.write(
            path=ds_path,
            bed=bed,
            variants=Path(svar_path),
            overwrite=True,
        )
        out[length] = ds_path
    return out


CSV_COLUMNS = [
    "mode",
    "with_seqs",
    "threads",
    "region_length",
    "batch_size",
    "buffer_bytes",
    "n_epochs",
    "instances",
    "bytes",
    "wall_s",
    "instances_per_s",
    "MiB_per_s",
    "peak_rss_MiB",
    "timed_out",
    "git_sha",
    "host",
    "started_at",
]


def init_csv(path) -> None:
    """Write the header row, truncating any existing file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_row(path, row: dict) -> None:
    """Append one result row. ``extrasaction='raise'`` rejects unknown keys."""
    with Path(path).open("a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="raise")
        writer.writerow(row)


def _build_dataset(cell: Cell, dataset_path, reference):
    import genvarloader as gvl

    # Open the reference as a memmap (in_memory=False) — the full hg38 is ~3 GB
    # and Reference.from_path defaults to in_memory=True, which would load it
    # into RAM per cell AND per double_buffered producer subprocess. A memmap is
    # file-backed/reclaimable; the producer mirrors it (reference_in_memory=False).
    if isinstance(reference, (str, Path)):
        reference = gvl.Reference.from_path(reference, in_memory=False)

    ds = gvl.Dataset.open(dataset_path, reference=reference).with_seqs(cell.with_seqs)
    if cell.with_seqs in ("haplotypes", "annotated"):
        ds = ds.with_settings(deterministic=True)
    return ds


def _build_loader(cell: Cell, dataset):
    kwargs = dict(batch_size=cell.batch_size, shuffle=False, num_workers=0)
    if cell.mode is not None:
        kwargs["mode"] = cell.mode
        kwargs["buffer_bytes"] = cell.buffer_bytes
    return dataset.to_dataloader(**kwargs)


def _drain(loader) -> None:
    for _ in loader:
        pass


def measure_cell(
    cell: Cell,
    dataset_path,
    reference,
    *,
    min_epochs: int = 3,
    min_seconds: float = 1.5,
    hard_cap_s: float = 10.0,
    git_sha: str = "",
    host: str = "",
    started_at: str = "",
) -> dict:
    """Run the spec measurement protocol for one cell; return a CSV row dict."""
    dataset = _build_dataset(cell, dataset_path, reference)
    instances_per_epoch, bytes_per_epoch, _ = output_bytes_table(dataset)
    loader = _build_loader(cell, dataset)

    # warmup (discarded)
    _drain(loader)

    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    epochs = 0
    total_wall = 0.0
    timed_out = False
    while total_wall < min_seconds or epochs < min_epochs:
        if total_wall >= hard_cap_s:
            timed_out = True
            break
        t0 = time.perf_counter()
        _drain(loader)
        total_wall += time.perf_counter() - t0
        epochs += 1
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Tear down before the next cell. Explicitly close the buffered/
    # double_buffered iterable so its producer subprocess + shm slots are
    # released NOW, rather than relying on GC timing — otherwise per-cell
    # loaders accumulate and exhaust RAM.
    inner = getattr(loader, "dataset", None)
    impl = getattr(inner, "_impl", None)
    if impl is not None and hasattr(impl, "close"):
        impl.close()
    del loader, dataset, inner, impl
    gc.collect()

    instances = instances_per_epoch * epochs
    total_bytes = bytes_per_epoch * epochs
    wall_s = total_wall if total_wall > 0 else float("nan")
    # ru_maxrss is KiB on Linux; delta is the high-water-mark growth
    peak_rss_MiB = max(0, rss_after - rss_before) / 1024

    return {
        "mode": "" if cell.mode is None else cell.mode,
        "with_seqs": cell.with_seqs,
        "threads": cell.threads,
        "region_length": cell.region_length,
        "batch_size": cell.batch_size,
        "buffer_bytes": "" if cell.buffer_bytes is None else cell.buffer_bytes,
        "n_epochs": epochs,
        "instances": instances,
        "bytes": total_bytes,
        "wall_s": wall_s,
        "instances_per_s": instances / wall_s,
        "MiB_per_s": (total_bytes / wall_s) / MiB,
        "peak_rss_MiB": peak_rss_MiB,
        "timed_out": timed_out,
        "git_sha": git_sha,
        "host": host,
        "started_at": started_at,
    }
