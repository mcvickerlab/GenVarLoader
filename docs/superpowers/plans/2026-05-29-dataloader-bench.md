# DataLoader Bench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained benchmark under `experiments/dataloader/` that compares the three `to_dataloader` modes (`None`, `"buffered"`, `"double_buffered"`) head-to-head and maps throughput scaling across four knobs (threads, region length, batch size, buffer bytes), emitting a CSV and a summary plot within a ~20-minute wall budget.

**Architecture:** Three new files mirroring `experiments/bench_table_overlap/`. `_common.py` holds pure config logic (axis constants, the deduped cell enumeration, BED/dataset prep, the output-bytes table, the per-cell measurement protocol, CSV I/O). `bench.py` is a thin entrypoint that re-execs itself once per `threads` value (to pin `RAYON_NUM_THREADS` before the rayon pool initializes) and dispatches the matching cells. `plot_results.py` reads the CSV into a 3×4 small-multiples grid. The bench is a **pure consumer** of the prefetching-dataloader public API — it adds no code under `python/genvarloader/`.

**Tech Stack:** Python, `genvarloader` (`gvl.write`, `Dataset.open`, `.with_seqs`, `.with_settings`, `.to_dataloader`, `._output_bytes_per_instance`), `seqpro` (`sp.bed.read`, `sp.bed.with_len`), `polars`, `matplotlib`, `numpy`, stdlib `subprocess`/`resource`/`csv`/`gc`.

---

## Prerequisite

This bench consumes the API delivered by **`docs/superpowers/plans/2026-05-28-prefetching-dataloader-implementation.md`**. Before executing, confirm these exist on the working branch:

- `Dataset.to_dataloader(..., mode=..., buffer_bytes=...)` accepts `mode in {None, "buffered", "double_buffered"}` and `buffer_bytes: int`.
- `Dataset._output_bytes_per_instance(regions=None, samples=None) -> NDArray[int64]` of shape `(n_regions, n_samples)`.

Verify with:

```bash
pixi run -e dev python -c "import genvarloader as gvl, inspect; sig=inspect.signature(gvl.Dataset.to_dataloader); assert 'mode' in sig.parameters and 'buffer_bytes' in sig.parameters, sig; assert hasattr(gvl.Dataset, '_output_bytes_per_instance'); print('prereq OK')"
```

Expected: `prereq OK`. If it errors, stop and land the prefetching-dataloader plan first.

## Spec reconciliation (read before coding)

> **CORRECTION (mid-execution, buffer sizes).** The `buffer_bytes` axis was
> reduced for workstation RAM (`double_buffered` allocates two slots totaling
> `buffer_bytes`, plus producer+consumer chunk copies). Live values in
> `_common.py`: `BUFFER_FACT=[64 MiB, 512 MiB]`, `BUFFER_FAN=[128, 256, 512] MiB`,
> `BUFFER_MID=256 MiB` (ceiling 512 MiB). The code blocks below show the
> original `{256 MiB … 4 GiB}` values; `_common.py` is authoritative. Cell
> counts are unchanged (the all-midpoint cell is now `(4, 5000, 64, 256 MiB)`).

The spec (`docs/superpowers/specs/2026-05-29-dataloader-bench-design.md`) quotes **192 cells** and per-cell counts of 28 (new modes) / 8 (baseline). Two refinements, both decided during planning, change the realized count to **195**:

1. **Shared-midpoint dedup (new modes).** Every axis fan includes that axis's *midpoint* value (threads `{2,4,16}`∋4, region `{2500,5000,25000}`∋5000, batch `{32,64,256}`∋64, buffer `{512MiB,1GiB,4GiB}`∋1GiB). So the all-midpoint cell `(4, 5000, 64, 1GiB)` is produced by all four fans. It is the **same physical configuration** and must be measured **once**, not four times. Deduped new-mode count is `16 (factorial) + 9 (fans, 12 raw − 3 duplicate midpoints) = 25` per `(mode, output)`. New-mode total: `25 × 2 × 3 = 150`.
2. **Baseline midpoint fans (user decision).** Baseline (`mode=None`) gets midpoint-anchored fans on the three axes it supports (threads, region, batch; it ignores `buffer_bytes`) so all three modes plot as comparable 3-point curves. Baseline = `8 (factorial corners) + 7 (fans, 9 raw − 2 duplicate midpoints) = 15` per output → `45` total.

**Grand total: 150 + 45 = 195 cells.** Budget impact is negligible (baseline cells are the cheapest path); ~4 s/cell × 195 ≈ 13 min measurement + ~1–3 min dataset writes, inside the 20-minute budget.

The fan values include each axis's midpoint **by design** — this is what makes each fan a clean 3-point sweep (below / mid / above) for plotting, with the other axes pinned at midpoint.

## File structure

```
experiments/dataloader/
├── _common.py        # axis constants, Cell, enumerate_cells, prep, measure_cell, CSV I/O
├── bench.py          # entrypoint: subprocess re-exec per threads value → cells → CSV
├── plot_results.py   # results.csv → results_plot.png (3×4 small multiples)
├── tests/
│   └── test_common.py   # pure-logic tests, run on demand (NOT part of `pixi run test`)
├── results.csv       # generated; committed only on intentional refresh
└── results_plot.png  # generated; committed only on intentional refresh
```

The bench's own tests live under `experiments/dataloader/tests/` and are run explicitly (`pixi run -e dev pytest experiments/dataloader/tests/`). The repo's default `test` task targets `tests/`, so these do not run in CI.

---

## Task 1: Scaffold + axis constants

**Files:**
- Create: `experiments/dataloader/_common.py`
- Create: `experiments/dataloader/tests/test_common.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing test**

Create `experiments/dataloader/tests/test_common.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '_common'` (file not created yet).

- [ ] **Step 3: Write minimal implementation**

Create `experiments/dataloader/_common.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Add the tmp/ ignore**

`tests/data/` temp datasets are written under a bench-managed `tmp/` dir. The repo `.gitignore` already ignores `data/` but not `tmp/` generically. Add an explicit entry. In `.gitignore`, find the `# Extra` block (top of file) and add a line after `repro/`:

```
# Extra
sbox*
archive/
tests/data/fasta
examples/
results/
.uuid
notebooks/
scripts/
.ruff_cache/
.benchmarks/
.custom_benchmarks/
benchmarking/
genvarloader*.tar.gz
.nfs*
commit_msg.txt
data/
.claude/worktrees/*
.worktrees/
scratch/
repro/
experiments/dataloader/tmp/
uv.lock
```

- [ ] **Step 6: Commit**

```bash
rtk git add experiments/dataloader/_common.py experiments/dataloader/tests/test_common.py .gitignore
rtk git commit -m "feat(bench): scaffold dataloader bench axis constants"
```

---

## Task 2: Cell dataclass + enumerate_cells (deduped)

**Files:**
- Modify: `experiments/dataloader/_common.py`
- Test: `experiments/dataloader/tests/test_common.py`

- [ ] **Step 1: Write the failing test**

Append to `experiments/dataloader/tests/test_common.py`:

```python
def test_new_mode_cell_count_is_25_per_mode_output():
    for mode in C.MODES_NEW:
        for output in C.OUTPUTS:
            cells = [
                c for c in C.enumerate_cells()
                if c.mode == mode and c.with_seqs == output
            ]
            # 16 factorial + 9 fan (12 raw − 3 shared midpoints) = 25
            assert len(cells) == 25, (mode, output, len(cells))


def test_baseline_cell_count_is_15_per_output_and_has_no_buffer():
    for output in C.OUTPUTS:
        cells = [
            c for c in C.enumerate_cells()
            if c.mode is None and c.with_seqs == output
        ]
        # 8 factorial corners + 7 fan (9 raw − 2 shared midpoints) = 15
        assert len(cells) == 15, (output, len(cells))
        assert all(c.buffer_bytes is None for c in cells)


def test_total_cell_count_is_195_and_all_unique():
    cells = C.enumerate_cells()
    assert len(cells) == 195
    keys = {
        (c.mode, c.with_seqs, c.threads, c.region_length, c.batch_size, c.buffer_bytes)
        for c in cells
    }
    assert len(keys) == 195  # no duplicate configurations


def test_baseline_fan_cells_sit_at_midpoints():
    # the threads fan for baseline pins region=MID, batch=MID
    base = [c for c in C.enumerate_cells() if c.mode is None and c.with_seqs == "variants"]
    threads_fan = [
        c for c in base
        if c.region_length == C.REGION_MID and c.batch_size == C.BATCH_MID
    ]
    assert sorted(c.threads for c in threads_fan) == [2, 4, 16]


def test_new_mode_buffer_fan_pins_other_axes_at_midpoint():
    buf_fan = [
        c for c in C.enumerate_cells()
        if c.mode == "buffered" and c.with_seqs == "haplotypes"
        and c.threads == C.THREADS_MID
        and c.region_length == C.REGION_MID
        and c.batch_size == C.BATCH_MID
    ]
    assert sorted(c.buffer_bytes for c in buf_fan) == sorted(C.BUFFER_FAN)


def test_cells_for_threads_partitions_by_thread_count():
    all_cells = C.enumerate_cells()
    union = []
    for n in C.ALL_THREADS:
        sub = C.cells_for_threads(n)
        assert all(c.threads == n for c in sub)
        union.extend(sub)
    assert len(union) == len(all_cells)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k "cell or threads or fan or midpoint"`
Expected: FAIL — `AttributeError: module '_common' has no attribute 'enumerate_cells'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/dataloader/_common.py`:

```python
from dataclasses import dataclass


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v`
Expected: PASS (all tests).

- [ ] **Step 5: Commit**

```bash
rtk git add experiments/dataloader/_common.py experiments/dataloader/tests/test_common.py
rtk git commit -m "feat(bench): enumerate deduped dataloader bench cells"
```

---

## Task 3: BED generation + dataset preparation

**Files:**
- Modify: `experiments/dataloader/_common.py`
- Test: `experiments/dataloader/tests/test_common.py`

- [ ] **Step 1: Write the failing test**

Append to `experiments/dataloader/tests/test_common.py`:

```python
import pytest


@pytest.mark.slow
def test_prepare_datasets_writes_one_gvl_per_region_length(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    svar = repo / "tests" / "data" / "1kg" / "filtered.svar"
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    if not svar.is_dir():
        pytest.skip("missing tests/data/1kg/filtered.svar; run pixi run -e dev gen")

    paths = C.prepare_datasets([1_000, 2_500], svar, regions, tmp_path)

    assert set(paths) == {1_000, 2_500}
    for length, p in paths.items():
        assert p.is_dir(), p
        assert (p / "metadata.json").exists()


@pytest.mark.slow
def test_generate_bed_resizes_to_target_length():
    import seqpro as sp

    repo = Path(__file__).resolve().parents[3]
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    if not regions.exists():
        pytest.skip("missing regions.bed")

    bed = C.generate_bed(regions, 2_500)
    lengths = (bed["chromEnd"] - bed["chromStart"]).unique().to_list()
    assert lengths == [2_500]
    assert bed.height == 100  # regions.bed has 100 regions
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -m slow`
Expected: FAIL — `AttributeError: module '_common' has no attribute 'prepare_datasets'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/dataloader/_common.py`:

```python
from pathlib import Path


def generate_bed(regions_bed: str | Path, region_length: int):
    """Read the canonical regions BED and recenter+resize every region to
    ``region_length`` (``sp.bed.with_len`` resizes around the midpoint)."""
    import seqpro as sp

    bed = sp.bed.read(regions_bed)
    return sp.bed.with_len(bed, region_length)


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -m slow`
Expected: PASS (writes 2 small datasets; takes a few seconds). If `filtered.svar` is missing, run `pixi run -e dev gen` first.

- [ ] **Step 5: Commit**

```bash
rtk git add experiments/dataloader/_common.py experiments/dataloader/tests/test_common.py
rtk git commit -m "feat(bench): add BED resize + per-region-length dataset prep"
```

---

## Task 4: Output-bytes table helper

**Files:**
- Modify: `experiments/dataloader/_common.py`
- Test: `experiments/dataloader/tests/test_common.py`

> **CORRECTION (mid-execution, reference genome).** The spec/plan originally
> opened bench datasets with no reference. But `haplotypes` and `annotated`
> output reconstruct sequences by applying variants onto a reference genome and
> therefore **require** one; only `variants` works reference-free. The bench
> now opens every dataset **with** the hg38 reference fixture
> `tests/data/fasta/hg38.fa.bgz` (symlinked into the worktree from the main
> repo; ~941 MB bgz + 3 GB `.gvl` index, loaded via memmap, `in_memory=False`).
> Consequences threaded through Tasks 4, 5, 7:
> - All `Dataset.open(...)` calls pass `reference=<hg38 path>`.
> - The out-of-scope `_open.py` patch (defaulting reference-less opens to
>   `"variants"`) is reverted — the bench is again a pure consumer.
> - `measure_cell` / `_build_dataset` take a `reference` argument; `bench.py`
>   defines `REF = REPO / "tests/data/fasta/hg38.fa.bgz"` and passes it down.

The measurement protocol needs total instances and total bytes per epoch. With `shuffle=False, drop_last=False`, one full epoch touches every `(region, sample)` pair exactly once, so `instances == table.size` and `bytes == int(table.sum())`. `_output_bytes_per_instance` is computed once per cell (cheap relative to epoch time; see spec "Pre-pass cost").

- [ ] **Step 1: Write the failing test**

Append to `experiments/dataloader/tests/test_common.py`:

```python
@pytest.mark.slow
def test_output_bytes_table_matches_actual_nbytes(tmp_path):
    import genvarloader as gvl

    repo = Path(__file__).resolve().parents[3]
    svar = repo / "tests" / "data" / "1kg" / "filtered.svar"
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    ref = repo / "tests" / "data" / "fasta" / "hg38.fa.bgz"
    if not svar.is_dir() or not ref.exists():
        pytest.skip("missing filtered.svar or hg38 reference; run pixi run -e dev gen")

    paths = C.prepare_datasets([1_000], svar, regions, tmp_path)
    ds = gvl.Dataset.open(paths[1_000], reference=ref).with_seqs("variants")

    instances, total_bytes, table = C.output_bytes_table(ds)
    assert instances == table.size
    assert total_bytes == int(table.sum())
    assert instances == 100 * 5  # 100 regions × 5 samples
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k output_bytes`
Expected: FAIL — `AttributeError: module '_common' has no attribute 'output_bytes_table'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/dataloader/_common.py`:

```python
def output_bytes_table(dataset):
    """Compute the exact per-instance byte table once for a configured dataset.

    Returns ``(instances, total_bytes, table)`` where ``table`` is the
    ``(n_regions, n_samples)`` int64 array from
    ``Dataset._output_bytes_per_instance``. For a full epoch with shuffle off,
    ``instances == table.size`` and ``total_bytes == table.sum()``.
    """
    table = dataset._output_bytes_per_instance()
    return int(table.size), int(table.sum()), table
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k output_bytes`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add experiments/dataloader/_common.py experiments/dataloader/tests/test_common.py
rtk git commit -m "feat(bench): add exact output-bytes table helper"
```

---

## Task 5: Per-cell measurement protocol

**Files:**
- Modify: `experiments/dataloader/_common.py`
- Test: `experiments/dataloader/tests/test_common.py`

Implements the spec "Measurement protocol": build the configured dataset + loader, one discarded warmup epoch, then iterate until `elapsed ≥ min_seconds AND epochs ≥ min_epochs`, with a `hard_cap_s` guard (mark `timed_out=True` if it trips before the stop condition). Peak RSS is the `ru_maxrss` delta over the measurement window.

- [ ] **Step 1: Write the failing test**

Append to `experiments/dataloader/tests/test_common.py`:

```python
@pytest.mark.slow
def test_measure_cell_returns_a_complete_row(tmp_path):
    repo = Path(__file__).resolve().parents[3]
    svar = repo / "tests" / "data" / "1kg" / "filtered.svar"
    regions = repo / "tests" / "data" / "1kg" / "regions.bed"
    ref = repo / "tests" / "data" / "fasta" / "hg38.fa.bgz"
    if not svar.is_dir() or not ref.exists():
        pytest.skip("missing filtered.svar or hg38 reference; run pixi run -e dev gen")

    paths = C.prepare_datasets([1_000], svar, regions, tmp_path)
    cell = C.Cell(
        mode=None, with_seqs="variants",
        threads=1, region_length=1_000, batch_size=16, buffer_bytes=None,
    )
    # tiny stop conditions so the test is fast
    row = C.measure_cell(
        cell, paths[1_000], ref, min_epochs=1, min_seconds=0.0, hard_cap_s=10.0,
    )

    for col in C.CSV_COLUMNS:
        assert col in row, col
    assert row["mode"] == ""          # None serialized as empty
    assert row["with_seqs"] == "variants"
    assert row["n_epochs"] >= 1
    assert row["instances"] == 100 * 5 * row["n_epochs"]
    assert row["instances_per_s"] > 0
    assert row["timed_out"] in (True, False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k measure_cell`
Expected: FAIL — `AttributeError: module '_common' has no attribute 'measure_cell'` (and `CSV_COLUMNS`).

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/dataloader/_common.py`:

```python
import gc
import resource
import time

CSV_COLUMNS = [
    "mode", "with_seqs", "threads", "region_length", "batch_size", "buffer_bytes",
    "n_epochs", "instances", "bytes", "wall_s", "instances_per_s", "MiB_per_s",
    "peak_rss_MiB", "timed_out", "git_sha", "host", "started_at",
]


def _build_dataset(cell: Cell, dataset_path, reference):
    import genvarloader as gvl

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

    # tear down before measuring nothing else; clip RSS between cells
    del loader, dataset
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k measure_cell`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add experiments/dataloader/_common.py experiments/dataloader/tests/test_common.py
rtk git commit -m "feat(bench): add per-cell measurement protocol"
```

---

## Task 6: CSV writer

**Files:**
- Modify: `experiments/dataloader/_common.py`
- Test: `experiments/dataloader/tests/test_common.py`

The parent writes the header once; each child appends its rows. Sequential `subprocess.run` calls mean appends never overlap.

- [ ] **Step 1: Write the failing test**

Append to `experiments/dataloader/tests/test_common.py`:

```python
def test_csv_init_then_append_roundtrips(tmp_path):
    import csv

    csv_path = tmp_path / "results.csv"
    C.init_csv(csv_path)

    with csv_path.open() as f:
        header = next(csv.reader(f))
    assert header == C.CSV_COLUMNS

    row = {col: 0 for col in C.CSV_COLUMNS}
    row["mode"] = "buffered"
    row["with_seqs"] = "variants"
    C.append_row(csv_path, row)
    C.append_row(csv_path, row)

    with csv_path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["mode"] == "buffered"
    assert rows[0]["with_seqs"] == "variants"


def test_append_row_rejects_unknown_columns(tmp_path):
    csv_path = tmp_path / "results.csv"
    C.init_csv(csv_path)
    with pytest.raises((ValueError, KeyError)):
        C.append_row(csv_path, {"bogus": 1})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k csv`
Expected: FAIL — `AttributeError: module '_common' has no attribute 'init_csv'`.

- [ ] **Step 3: Write minimal implementation**

Append to `experiments/dataloader/_common.py`:

```python
import csv as _csv


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pixi run -e dev pytest experiments/dataloader/tests/test_common.py -v -k csv`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add experiments/dataloader/_common.py experiments/dataloader/tests/test_common.py
rtk git commit -m "feat(bench): add CSV header/append helpers"
```

---

## Task 7: bench.py entrypoint (subprocess re-exec orchestration)

**Files:**
- Create: `experiments/dataloader/bench.py`

`RAYON_NUM_THREADS` (and BLAS pools) must be pinned **before** the process imports genvarloader / initializes the rayon pool. The parent therefore writes datasets + the CSV header, then re-execs itself once per `threads` value with the env pinned; each child runs only its matching cells and appends rows. The parent passes the shared tmp dir, git sha, and start timestamp to children via env so all rows agree.

- [ ] **Step 1: Write bench.py**

Create `experiments/dataloader/bench.py`:

```python
"""DataLoader throughput bench entrypoint.

Compares to_dataloader modes (None / buffered / double_buffered) across
threads × region_length × batch_size × buffer_bytes for three output modes.

Run (writes experiments/dataloader/results.csv):
    pixi run -e dev python experiments/dataloader/bench.py

Thread counts are pinned per child process via re-exec, because rayon's pool
size is fixed once initialized.
"""

from __future__ import annotations

import datetime as _dt
import os
import socket
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results.csv"
TMP_DIR = HERE / "tmp"

REPO = HERE.parents[1]
SVAR = REPO / "tests" / "data" / "1kg" / "filtered.svar"
REGIONS_BED = REPO / "tests" / "data" / "1kg" / "regions.bed"
# haplotypes/annotated reconstruct against a reference; variants ignores it.
REF = REPO / "tests" / "data" / "fasta" / "hg38.fa.bgz"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def run_child(n_threads: int) -> None:
    """Child process: run only the cells pinned to this thread count."""
    git_sha = os.environ.get("BENCH_GIT_SHA", "")
    started_at = os.environ.get("BENCH_STARTED_AT", "")
    host = socket.gethostname()
    tmp_dir = Path(os.environ["BENCH_TMP"])

    # dataset paths were written by the parent, keyed by region length
    ds_paths = {
        length: tmp_dir / f"dataset_rL{length}.gvl"
        for length in C.REGION_LENGTHS
    }

    cells = C.cells_for_threads(n_threads)
    for i, cell in enumerate(cells, 1):
        ds_path = ds_paths[cell.region_length]
        try:
            row = C.measure_cell(
                cell, ds_path, REF,
                git_sha=git_sha, host=host, started_at=started_at,
            )
        except Exception as e:  # noqa: BLE001 - one bad cell must not kill the run
            print(f"[threads={n_threads}] cell {i}/{len(cells)} FAILED: {cell} -> {e}")
            continue
        C.append_row(RESULTS_CSV, row)
        print(
            f"[threads={n_threads}] {i}/{len(cells)} "
            f"{row['mode'] or 'None'}/{row['with_seqs']} "
            f"r{cell.region_length} b{cell.batch_size} "
            f"-> {row['instances_per_s']:.0f} inst/s"
            + (" TIMEOUT" if row["timed_out"] else "")
        )


def run_parent() -> None:
    print(f"Preparing datasets in {TMP_DIR} ...")
    C.prepare_datasets(C.REGION_LENGTHS, SVAR, REGIONS_BED, TMP_DIR)
    C.init_csv(RESULTS_CSV)

    started_at = _dt.datetime.now().isoformat(timespec="seconds")
    git_sha = _git_sha()

    for n_threads in C.ALL_THREADS:
        env = {
            **os.environ,
            "BENCH_THREADS": str(n_threads),
            "BENCH_TMP": str(TMP_DIR),
            "BENCH_GIT_SHA": git_sha,
            "BENCH_STARTED_AT": started_at,
            "RAYON_NUM_THREADS": str(n_threads),
            "POLARS_MAX_THREADS": str(n_threads),
            "OMP_NUM_THREADS": str(n_threads),
            "MKL_NUM_THREADS": str(n_threads),
            "OPENBLAS_NUM_THREADS": str(n_threads),
        }
        print(f"\n=== child: threads={n_threads} ===")
        subprocess.run([sys.executable, __file__, "--child"], env=env, check=True)

    print(f"\nDone. Results: {RESULTS_CSV}")


def main() -> None:
    if "--child" in sys.argv:
        run_child(int(os.environ["BENCH_THREADS"]))
    else:
        try:
            run_parent()
        finally:
            import shutil

            shutil.rmtree(TMP_DIR, ignore_errors=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Sanity-check the child wiring without a full run**

Run a single child against a single thread value, but first confirm the prereq API and that datasets can be written. This is a smoke check, not the full bench:

```bash
pixi run -e dev python -c "
import sys; sys.path.insert(0, 'experiments/dataloader')
import _common as C
print('threads union:', C.ALL_THREADS)
print('cells @ threads=1:', len(C.cells_for_threads(1)))
print('total cells:', len(C.enumerate_cells()))
"
```

Expected output (counts):
```
threads union: [1, 2, 4, 8, 16]
cells @ threads=1: ...
total cells: 195
```

- [ ] **Step 3: Commit**

```bash
rtk git add experiments/dataloader/bench.py
rtk git commit -m "feat(bench): add bench.py thread-pinned orchestration"
```

---

## Task 8: plot_results.py (3×4 small multiples)

**Files:**
- Create: `experiments/dataloader/plot_results.py`

Rows = output mode; cols = axis. Each panel draws up to 3 lines (one per mode), each line being that mode's fan along the column's axis with the other three axes pinned at midpoint. `mode=None` is absent from the `buffer_bytes` column (baseline ignores the buffer budget). Y = `instances_per_s` (linear). A `MiB_per_s` variant is included but commented out.

- [ ] **Step 1: Write plot_results.py**

Create `experiments/dataloader/plot_results.py`:

```python
"""Read results.csv → results_plot.png (3 outputs × 4 axes small multiples).

Each panel: x = the column's axis values, y = instances_per_s, one line per
mode, holding the other three axes at their midpoint. mode=None is absent from
the buffer_bytes column (baseline ignores the buffer budget).

Run:
    pixi run -e dev python experiments/dataloader/plot_results.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import _common as C

HERE = Path(__file__).resolve().parent
RESULTS_CSV = HERE / "results.csv"
OUT_PNG = HERE / "results_plot.png"

# column axis -> (csv column, fan values, {pinned other axis: midpoint})
AXES = {
    "threads": ("threads", C.THREADS_FAN,
                {"region_length": C.REGION_MID, "batch_size": C.BATCH_MID}),
    "region_length": ("region_length", C.REGION_FAN,
                      {"threads": C.THREADS_MID, "batch_size": C.BATCH_MID}),
    "batch_size": ("batch_size", C.BATCH_FAN,
                   {"threads": C.THREADS_MID, "region_length": C.REGION_MID}),
    "buffer_bytes": ("buffer_bytes", C.BUFFER_FAN,
                     {"threads": C.THREADS_MID, "region_length": C.REGION_MID,
                      "batch_size": C.BATCH_MID}),
}

MODE_STYLE = {
    "": ("None", "tab:gray", "o"),
    "buffered": ("buffered", "tab:blue", "s"),
    "double_buffered": ("double_buffered", "tab:green", "^"),
}


def _panel_series(df: pl.DataFrame, axis_col: str, pins: dict, mode_key: str):
    """Return (x, y) sorted by x for one mode on one panel."""
    sub = df.filter(pl.col("mode") == mode_key)
    for col, mid in pins.items():
        sub = sub.filter(pl.col(col) == mid)
    # baseline has no buffer_bytes; the buffer panel pins it for new modes only
    if axis_col == "buffer_bytes" and mode_key == "":
        return [], []
    if axis_col != "buffer_bytes":
        # new modes pin buffer at midpoint; baseline rows have null buffer
        if mode_key == "":
            sub = sub.filter(pl.col("buffer_bytes").is_null())
        else:
            sub = sub.filter(pl.col("buffer_bytes") == C.BUFFER_MID)
    sub = sub.sort(axis_col)
    return sub[axis_col].to_list(), sub["instances_per_s"].to_list()


def main() -> None:
    if not RESULTS_CSV.exists():
        print(f"Missing {RESULTS_CSV} — run bench.py first.")
        return
    # buffer_bytes is empty for baseline rows -> read as nullable int
    df = pl.read_csv(RESULTS_CSV, null_values=[""])

    axis_names = list(AXES)
    fig, axs = plt.subplots(
        len(C.OUTPUTS), len(axis_names),
        figsize=(4 * len(axis_names), 3 * len(C.OUTPUTS)),
        constrained_layout=True, squeeze=False,
    )

    for r, output in enumerate(C.OUTPUTS):
        out_df = df.filter(pl.col("with_seqs") == output)
        for c, axis_name in enumerate(axis_names):
            ax = axs[r][c]
            axis_col, _vals, pins = AXES[axis_name]
            for mode_key, (label, color, marker) in MODE_STYLE.items():
                x, y = _panel_series(out_df, axis_col, pins, mode_key)
                if x:
                    ax.plot(x, y, marker=marker, color=color, label=label)
            if axis_name == "buffer_bytes":
                ax.set_xscale("log", base=2)
            if r == 0:
                ax.set_title(axis_name)
            if c == 0:
                ax.set_ylabel(f"{output}\ninstances/s")
            ax.grid(True, alpha=0.3)
    axs[0][-1].legend(fontsize=8)

    fig.suptitle(
        "DataLoader throughput: mode comparison across knobs "
        "(other axes pinned at midpoint)",
        fontsize=13, fontweight="bold",
    )
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")
    plt.close(fig)

    # --- optional MiB_per_s variant (uncomment to render bandwidth view) ---
    # Re-run with "MiB_per_s" substituted for "instances_per_s" in
    # _panel_series and OUT_PNG -> results_plot_bandwidth.png.


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-check the plot logic with a tiny synthetic CSV**

Run:

```bash
pixi run -e dev python -c "
import sys; sys.path.insert(0, 'experiments/dataloader')
import _common as C
from pathlib import Path
p = Path('experiments/dataloader/results.csv')
C.init_csv(p)
# minimal rows: buffered + None along the threads fan at midpoints
import socket
for mode in ('buffered',''):
    for t in C.THREADS_FAN:
        row = {col: 0 for col in C.CSV_COLUMNS}
        row.update(mode=mode, with_seqs='variants', threads=t,
                   region_length=C.REGION_MID, batch_size=C.BATCH_MID,
                   buffer_bytes=(C.BUFFER_MID if mode else ''),
                   instances_per_s=100*t, MiB_per_s=t)
        C.append_row(p, row)
import plot_results; plot_results.main()
Path('experiments/dataloader/results.csv').unlink()
"
ls -la experiments/dataloader/results_plot.png
```

Expected: prints `Saved .../results_plot.png` and `ls` shows a non-empty PNG. (Delete the synthetic PNG afterward; it is regenerated by the real run in Task 9.)

- [ ] **Step 3: Commit**

```bash
rtk git add experiments/dataloader/plot_results.py
rtk git commit -m "feat(bench): add 3x4 small-multiples results plot"
```

---

## Task 9: End-to-end run + results refresh

**Files:**
- Generated: `experiments/dataloader/results.csv`, `experiments/dataloader/results_plot.png`

- [ ] **Step 1: Ensure test data exists**

Run: `ls tests/data/1kg/filtered.svar/`
Expected: a directory listing (contains `variant_idxs.npy`, `index.arrow`, etc.). If missing:

```bash
pixi run -e dev gen
```

- [ ] **Step 2: Run the full bench (timed)**

Run: `time pixi run -e dev python experiments/dataloader/bench.py`
Expected: per-child progress lines (`=== child: threads=N ===` followed by `i/total ... inst/s`), then `Done. Results: .../results.csv`. Total wall time should be **under ~20 minutes**. If a child errors out (`subprocess.run(..., check=True)` raises), inspect the failing cell line printed just before.

- [ ] **Step 3: Validate the CSV row count**

Run:

```bash
pixi run -e dev python -c "
import polars as pl, sys
sys.path.insert(0, 'experiments/dataloader'); import _common as C
df = pl.read_csv('experiments/dataloader/results.csv', null_values=[''])
print('rows:', df.height, 'expected <=', len(C.enumerate_cells()))
print('timed_out:', df.filter(pl.col('timed_out')).height)
print(df.group_by('mode','with_seqs').len().sort('mode','with_seqs'))
"
```

Expected: `rows` equals 195 minus any failed/timed-out-skipped cells (failures are skipped, timeouts are still recorded with `timed_out=True`). Note any non-zero `timed_out` count — if large, the spec's deferred question about `region_length=25_000` setup/cost may apply (consider trimming the region fan to 4 values).

- [ ] **Step 4: Render the plot**

Run: `pixi run -e dev python experiments/dataloader/plot_results.py`
Expected: `Saved .../results_plot.png`. Open it and confirm a 3×4 grid where the `buffer_bytes` column shows only the two new-mode lines and the other three columns show all three modes.

- [ ] **Step 5: Lint**

Run: `pixi run -e dev ruff check experiments/dataloader/`
Expected: no errors (or only acknowledged ones). Fix any reported issues.

- [ ] **Step 6: Commit the refreshed results**

```bash
rtk git add experiments/dataloader/results.csv experiments/dataloader/results_plot.png
rtk git commit -m "chore(bench): refresh dataloader bench results + plot"
```

---

## Self-review notes

- **Spec coverage:** motivation (3-mode head-to-head + scaling) → Tasks 2 (cells) + 8 (plot); data prep from `filtered.svar` keyed by region length → Task 3; per-cell epoch sizing & measurement protocol with hard cap → Task 5; CSV columns verbatim → Task 6; thread isolation via subprocess re-exec → Task 7; plotting 3×4 grid with `mode=None` absent from buffer column → Task 8; gitignored generated artifacts → Tasks 1 & 9. Out-of-scope items (tracks, splice, VCF/PGEN, multi-GPU, `copy=False`, `pin_memory`, real GPU step) are simply never constructed.
- **Reconciliation surfaced:** cell count is **195** (deduped shared midpoints + user-chosen baseline midpoint fans), documented in "Spec reconciliation" — supersedes the spec's nominal 192/168/28/8.
- **Deferred open questions** (from spec §"Open questions") are flagged in Task 9 Step 3 (25k setup cost → trim region fan if it dominates) and addressed by design in Task 4 (output-bytes table computed once per cell).
- **Type consistency:** `Cell` fields, `CSV_COLUMNS`, and the dicts returned by `measure_cell` use identical keys; `enumerate_cells`/`cells_for_threads`/`prepare_datasets`/`output_bytes_table`/`measure_cell`/`init_csv`/`append_row` names are referenced consistently across `_common.py`, `bench.py`, and `plot_results.py`.
```

