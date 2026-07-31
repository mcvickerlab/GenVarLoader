# `gvl.concat` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Tasks 2, 3, and 4 are independent of each other (all depend only on Task 1). Dispatch them together using superpowers:dispatching-parallel-agents. Tasks 5-8 are sequential after that. Use **Sonnet or weaker** for implementation subagents; reserve stronger models for second-pass fixes where an implementer critically failed.

**Goal:** Add `gvl.concat(path, datasets, axis="regions"|"samples")`, which merges on-disk GVL datasets along the region or sample axis without re-extracting genotypes.

**Architecture:** A GVL dataset is a set of parallel ragged stores over an `(R, S[, P])` C-order grid. Merging builds a *provenance map* (merged flat slot → source dataset + source flat slot), coalesces it into maximal contiguous runs, and streams each run with buffered IO. Both axes use the same gather; `axis` only selects the provenance map.

**Tech Stack:** Python 3.10+, numpy, polars, pydantic, pytest. No Rust changes. No new dependencies.

## Global Constraints

Copied verbatim from `docs/superpowers/specs/2026-07-31-dataset-concat-design.md`:

- **Buffered IO on both sides.** `seek()`+`read()` for sources, `write()` for destinations, **16 MiB chunks**. **No `np.memmap` anywhere in the bulk data path.** On NFSv3 any memmap in the path costs ~5-6x; on local XFS all patterns are equivalent, so buffered IO is never the wrong choice.
- **Single-threaded.** No rayon, no native-extension threading, no `joblib`. The work is bandwidth-bound; parallelism buys nothing and costs NFS process-hygiene risk.
- **Destination-ordered iteration.** A buffered write stream must be filled sequentially. This is required, not merely preferred.
- **Neither pass may materialize a full array.** Offsets alone reach 16-32 GB at biobank scale. Stream them in chunks.
- **All inputs must share one variant source.** Merging distinct variant tables is out of scope and must raise.
- `variants.arrow` is **hardlinked** from `datasets[0]` (chunked-copy fallback on `EXDEV`), plus a recorded fingerprint. Do not replace the hardlink with a copy.
- Chunk constant: `CONCAT_CHUNK_BYTES = 16 << 20`.
- All validation runs **before any bytes move**.
- Build into `atomic_dir(dest, overwrite=overwrite)`, matching `gvl.write`.

**Repo conventions (from CLAUDE.md):**
- Google-style docstrings on `python/genvarloader/` (`Args:`/`Returns:`/`Raises:`), enforced by ruff pydocstyle.
- Conventional commits (commitizen). Prefix with `feat(concat):` / `test(concat):` / `docs(concat):`.
- Lint covers **both** `python/` and `tests/`: `pixi run -e dev ruff check python/ tests/`.
- `pixi run -e dev pytest` does **not** rebuild Rust. Irrelevant here (no Rust changes), but do not add any.

**Verification commands:**
```bash
pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py -q      # per-task
pixi run -e dev pytest tests/dataset/test_concat.py -q                # per-task
pixi run -e dev pytest tests/dataset tests/unit -q                    # before final commit
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/
```

**Note on commits in this worktree:** the `pyrefly-check` pre-commit hook runs `pixi run -e dev pyrefly ...` and needs a built dev env. If a commit hangs past ~2 min, that is why. Prefix with `SKIP=pyrefly-check` only for commits that touch **no** Python; for Python commits let pyrefly run.

---

## File Structure

| File | Responsibility |
|---|---|
| `python/genvarloader/_dataset/_concat_plan.py` (new) | Pure planning: provenance maps, run coalescing, merged-offset computation. No IO. Independently unit-testable. |
| `python/genvarloader/_dataset/_concat_io.py` (new) | Buffered streaming primitives: run copier, chunked offset reader/writer, hardlink-or-copy. |
| `python/genvarloader/_dataset/_concat.py` (new) | `concat()` public entry point: validation, metadata/bed merge, orchestration of the per-store merges. |
| `python/genvarloader/_dataset/_concat_validate.py` (new) | Precondition checks + `variants.arrow` fingerprint. |
| `python/genvarloader/__init__.py` (modify) | Export `concat` in `__all__`. |
| `docs/source/api.md` (modify) | Autodoc entry (required — `api.md` must stay in sync with `__all__`). |
| `docs/source/write.md`, `docs/source/faq.md` (modify) | Concat workflow, cost model, and the `parent.pgen` + `samples=` guidance answering #334 Q1. |
| `skills/genvarloader/SKILL.md` (modify) | New public symbol (required by CLAUDE.md). |
| `tests/unit/dataset/test_concat_plan.py` (new) | Unit tests for planning + IO primitives. |
| `tests/dataset/test_concat.py` (new) | Integration: shard → concat → compare vs single-shot `gvl.write`. |

Splitting plan/IO/validate from `_concat.py` keeps each file focused and lets Tasks 2-4 run in parallel against a frozen interface.

---

### Task 1: Provenance map and run coalescing

Pure functions, no IO. This is the interface every later task builds on, so it lands first and alone.

**Files:**
- Create: `python/genvarloader/_dataset/_concat_plan.py`
- Test: `tests/unit/dataset/test_concat_plan.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `CONCAT_CHUNK_BYTES: int = 16 << 20`
  - `Run` — `NamedTuple(src: int, src_start: int, src_stop: int, dst_start: int)`; half-open slot ranges.
  - `provenance(axis: str, shape_per_ds: list[tuple[int, int]], ploidy: int) -> NDArray[np.int64]` → `(n_slots, 2)` array of `(dataset_idx, source_flat_slot)`.
  - `coalesce(prov: NDArray[np.int64]) -> list[Run]`
  - `sample_provenance(axis, shape_per_ds, ploidy=1)` — same as `provenance` but for `(R, S)` stores (tracks); pass `ploidy=1`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dataset/test_concat_plan.py`:

```python
"""Unit tests for gvl.concat's pure planning layer (no IO)."""

import numpy as np
import pytest

from genvarloader._dataset._concat_plan import (
    CONCAT_CHUNK_BYTES,
    Run,
    coalesce,
    provenance,
)


def test_chunk_bytes_is_16mib():
    assert CONCAT_CHUNK_BYTES == 16 << 20


def test_provenance_regions_appends_blocks():
    # two datasets, 2 samples each, ploidy 1; A has 2 regions, B has 1.
    prov = provenance("regions", [(2, 2), (1, 2)], ploidy=1)
    # merged order is (r, s): A(r0s0) A(r0s1) A(r1s0) A(r1s1) B(r0s0) B(r0s1)
    assert prov.tolist() == [
        [0, 0], [0, 1], [0, 2], [0, 3],
        [1, 0], [1, 1],
    ]


def test_provenance_samples_interleaves_per_region():
    # two datasets, 2 regions each; A has 1 sample, B has 2. ploidy 1.
    prov = provenance("samples", [(2, 1), (2, 2)], ploidy=1)
    # merged S' = 3. Per region: A's sample, then B's two.
    assert prov.tolist() == [
        [0, 0], [1, 0], [1, 1],   # region 0
        [0, 1], [1, 2], [1, 3],   # region 1
    ]


def test_provenance_accounts_for_ploidy():
    prov = provenance("regions", [(1, 1), (1, 1)], ploidy=2)
    # each (r, s) contributes P consecutive slots
    assert prov.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]


def test_coalesce_regions_gives_one_run_per_dataset():
    prov = provenance("regions", [(2, 2), (1, 2)], ploidy=1)
    runs = coalesce(prov)
    assert runs == [
        Run(src=0, src_start=0, src_stop=4, dst_start=0),
        Run(src=1, src_start=0, src_stop=2, dst_start=4),
    ]


def test_coalesce_samples_gives_run_per_dataset_per_region():
    prov = provenance("samples", [(2, 1), (2, 2)], ploidy=1)
    runs = coalesce(prov)
    assert runs == [
        Run(src=0, src_start=0, src_stop=1, dst_start=0),
        Run(src=1, src_start=0, src_stop=2, dst_start=1),
        Run(src=0, src_start=1, src_stop=2, dst_start=3),
        Run(src=1, src_start=2, src_stop=4, dst_start=4),
    ]


def test_coalesce_covers_every_slot_exactly_once():
    prov = provenance("samples", [(3, 2), (3, 1), (3, 3)], ploidy=2)
    runs = coalesce(prov)
    covered = np.zeros(len(prov), dtype=np.int32)
    for r in runs:
        n = r.src_stop - r.src_start
        covered[r.dst_start : r.dst_start + n] += 1
    assert (covered == 1).all()


def test_coalesce_runs_are_monotonic_within_each_source():
    """Destination-ordered iteration must keep each source's reads sequential."""
    prov = provenance("samples", [(4, 2), (4, 3)], ploidy=1)
    runs = coalesce(prov)
    for src in (0, 1):
        starts = [r.src_start for r in runs if r.src == src]
        assert starts == sorted(starts)


def test_provenance_rejects_unknown_axis():
    with pytest.raises(ValueError, match="axis must be"):
        provenance("chromosomes", [(1, 1)], ploidy=1)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'genvarloader._dataset._concat_plan'`

- [ ] **Step 3: Write the implementation**

Create `python/genvarloader/_dataset/_concat_plan.py`:

```python
"""Pure planning for :func:`genvarloader.concat` — no IO.

A GVL dataset stores parallel ragged arrays over an ``(R, S[, P])`` C-order grid;
the flat slot for ``(r, s, p)`` is ``((r * S) + s) * P + p``. Merging two or more
datasets means deciding, for each *merged* flat slot, which input dataset and which
*source* flat slot it comes from. That mapping is the provenance map, and
:func:`coalesce` compresses it into maximal contiguous runs so the IO layer can move
large byte ranges instead of individual slots.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

__all__ = ["CONCAT_CHUNK_BYTES", "Run", "coalesce", "provenance"]

CONCAT_CHUNK_BYTES = 16 << 20
"""Buffered-IO chunk size. 16 MiB is the measured knee on NFSv3; 64 MiB is no better."""


class Run(NamedTuple):
    """A maximal contiguous span of merged slots drawn from one source dataset.

    Attributes:
        src: Index into the input dataset list.
        src_start: First source flat slot, inclusive.
        src_stop: Last source flat slot, exclusive.
        dst_start: First merged flat slot, inclusive.
    """

    src: int
    src_start: int
    src_stop: int
    dst_start: int


def provenance(
    axis: str,
    shape_per_ds: list[tuple[int, int]],
    ploidy: int,
) -> NDArray[np.int64]:
    """Map each merged flat slot to its ``(dataset, source flat slot)`` origin.

    Args:
        axis: Either ``"regions"`` or ``"samples"``.
        shape_per_ds: ``(n_regions, n_samples)`` per input dataset, in input order.
        ploidy: Slots per ``(region, sample)`` cell. Pass ``1`` for interval stores,
            which have no ploidy axis.

    Returns:
        An ``(n_slots, 2)`` int64 array; column 0 is the dataset index and column 1
        is the flat slot within that dataset.

    Raises:
        ValueError: If ``axis`` is not ``"regions"`` or ``"samples"``.
    """
    if axis not in ("regions", "samples"):
        raise ValueError(f'axis must be "regions" or "samples", got {axis!r}')

    n_ds = len(shape_per_ds)

    if axis == "regions":
        n_samples = shape_per_ds[0][1]
        blocks = []
        for d in range(n_ds):
            n_slots = shape_per_ds[d][0] * n_samples * ploidy
            block = np.empty((n_slots, 2), np.int64)
            block[:, 0] = d
            block[:, 1] = np.arange(n_slots, dtype=np.int64)
            blocks.append(block)
        return np.concatenate(blocks, axis=0)

    # axis == "samples": regions are shared; per region, lay out each dataset's
    # samples in input-dataset order.
    n_regions = shape_per_ds[0][0]
    per_ds_samples = [s for _, s in shape_per_ds]
    cell_blocks = []
    for d, n_s in enumerate(per_ds_samples):
        block = np.empty((n_s * ploidy, 2), np.int64)
        block[:, 0] = d
        cell_blocks.append(block)

    out = []
    for r in range(n_regions):
        for d, n_s in enumerate(per_ds_samples):
            block = cell_blocks[d].copy()
            base = r * n_s * ploidy
            block[:, 1] = np.arange(base, base + n_s * ploidy, dtype=np.int64)
            out.append(block)
    return np.concatenate(out, axis=0)


def coalesce(prov: NDArray[np.int64]) -> list[Run]:
    """Compress a provenance map into maximal contiguous runs.

    A run is a maximal span of consecutive merged slots over which the source
    dataset is constant and the source slot increases by exactly 1. Iterating the
    returned runs in order walks the destination sequentially *and* each source
    monotonically, so both sides of the copy stay sequential.

    Args:
        prov: An ``(n_slots, 2)`` provenance map from :func:`provenance`.

    Returns:
        Runs in destination order. Empty if ``prov`` has no rows.
    """
    if len(prov) == 0:
        return []

    src = prov[:, 0]
    slot = prov[:, 1]
    # A new run starts wherever the dataset changes or the source slot jumps.
    breaks = (src[1:] != src[:-1]) | (slot[1:] != slot[:-1] + 1)
    starts = np.concatenate([[0], np.flatnonzero(breaks) + 1])
    stops = np.concatenate([starts[1:], [len(prov)]])

    return [
        Run(
            src=int(src[a]),
            src_start=int(slot[a]),
            src_stop=int(slot[a]) + int(b - a),
            dst_start=int(a),
        )
        for a, b in zip(starts, stops)
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Lint**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_concat_plan.py tests/unit/dataset/test_concat_plan.py
git commit -m "feat(concat): provenance map and run coalescing for dataset merge

Pure planning layer for gvl.concat. Both axes reduce to one provenance
map over the (R, S[, P]) C-order grid; coalesce() compresses it into
maximal contiguous runs so the IO layer moves large byte ranges.

Relates to #334."
```

---

### Tasks 2, 3, 4 run in PARALLEL

All three depend only on Task 1 and touch disjoint files. Dispatch them together via superpowers:dispatching-parallel-agents.

---

### Task 2: Buffered streaming IO primitives

**Files:**
- Create: `python/genvarloader/_dataset/_concat_io.py`
- Test: `tests/unit/dataset/test_concat_io.py`

**Interfaces:**
- Consumes: `CONCAT_CHUNK_BYTES`, `Run` from `._concat_plan`.
- Produces:
  - `copy_runs(srcs: list[Path], dst: Path, runs: list[Run], src_offsets: list[NDArray[np.int64]], itemsize: int) -> NDArray[np.int64]` — streams payload; returns merged offsets, length `n_slots + 1`.
  - `gather_fixed(srcs: list[Path], dst: Path, runs: list[Run], record_bytes: int) -> None` — fixed-stride row gather for absolute-range stores.
  - `link_or_copy_buffered(src: Path, dst: Path) -> None` — hardlink, chunked-copy fallback on `EXDEV`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dataset/test_concat_io.py`:

```python
"""Unit tests for gvl.concat's buffered streaming IO primitives."""

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
    _write_raw(a, np.array([10, 11, 12], np.int32))   # slots [10,11], [12]
    _write_raw(b, np.array([20, 21], np.int32))       # slots [20], [21]
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
    monkeypatch.setattr(
        "genvarloader._dataset._concat_io.CONCAT_CHUNK_BYTES", 64
    )
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


def test_copy_runs_writes_nothing_for_no_runs(tmp_path):
    dst = tmp_path / "out.npy"
    merged = copy_runs([], dst, [], [], itemsize=4)
    assert merged.tolist() == [0]
    assert dst.read_bytes() == b""
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_io.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'genvarloader._dataset._concat_io'`

- [ ] **Step 3: Write the implementation**

Create `python/genvarloader/_dataset/_concat_io.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_io.py -q`
Expected: PASS, 7 tests.

- [ ] **Step 5: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat_io.py tests/unit/dataset/test_concat_io.py
git commit -m "feat(concat): buffered streaming IO primitives

Buffered read/write on both sides in 16 MiB chunks, no np.memmap in the
bulk path: on NFSv3 any memmap in the path costs ~5-6x, while on local
XFS the patterns are equivalent.

Relates to #334."
```

---

### Task 3: Validation and fingerprinting

**Files:**
- Create: `python/genvarloader/_dataset/_concat_validate.py`
- Test: `tests/unit/dataset/test_concat_validate.py`

**Interfaces:**
- Consumes: `Metadata` from `._write`; `Fingerprint`, `fingerprint` from `.._fasta_cache`.
- Produces:
  - `ConcatInput` — frozen dataclass: `path: Path`, `meta: Metadata`, `bed: pl.DataFrame`, `n_regions: int`, `n_samples: int`, `backend: str`, `tracks: list[str]`, `annot_tracks: list[str]`, `has_dosages: bool`.
  - `load_inputs(paths: list[Path]) -> list[ConcatInput]`
  - `validate_concat(inputs: list[ConcatInput], axis: str) -> None`

Backend is one of `"pgen_vcf"`, `"svar"`, `"svar2"`, `"tracks_only"`.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dataset/test_concat_validate.py`:

```python
"""Unit tests for gvl.concat preconditions."""

from dataclasses import replace

import polars as pl
import pytest

from genvarloader._dataset._concat_validate import ConcatInput, validate_concat
from genvarloader._dataset._write import Metadata


def _mk(samples, n_regions, *, backend="pgen_vcf", ploidy=2, tracks=(), chroms=None):
    n = n_regions
    bed = pl.DataFrame(
        {
            "chrom": chroms or ["chr1"] * n,
            "chromStart": list(range(0, 100 * n, 100)),
            "chromEnd": list(range(50, 100 * n + 50, 100)),
        }
    )
    meta = Metadata(
        samples=list(samples),
        contigs=["chr1", "chr2"],
        n_regions=n,
        ploidy=ploidy,
    )
    return ConcatInput(
        path=None, meta=meta, bed=bed, n_regions=n, n_samples=len(samples),
        backend=backend, tracks=list(tracks), annot_tracks=[], has_dosages=False,
    )


def test_accepts_disjoint_samples_on_sample_axis():
    validate_concat([_mk(["a"], 2), _mk(["b"], 2)], "samples")


def test_rejects_overlapping_samples_on_sample_axis():
    with pytest.raises(ValueError, match="overlapping samples"):
        validate_concat([_mk(["a", "b"], 2), _mk(["b"], 2)], "samples")


def test_rejects_mismatched_regions_on_sample_axis():
    with pytest.raises(ValueError, match="identical regions"):
        validate_concat([_mk(["a"], 2), _mk(["b"], 3)], "samples")


def test_rejects_mismatched_samples_on_region_axis():
    with pytest.raises(ValueError, match="identical samples"):
        validate_concat([_mk(["a"], 2), _mk(["b"], 2)], "regions")


def test_rejects_mixed_backends():
    with pytest.raises(ValueError, match="same variant source"):
        validate_concat(
            [_mk(["a"], 2, backend="pgen_vcf"), _mk(["a"], 2, backend="svar2")],
            "regions",
        )


def test_rejects_mismatched_ploidy():
    with pytest.raises(ValueError, match="ploidy"):
        validate_concat([_mk(["a"], 2, ploidy=2), _mk(["a"], 2, ploidy=1)], "regions")


def test_rejects_mismatched_track_sets():
    with pytest.raises(ValueError, match="track"):
        validate_concat(
            [_mk(["a"], 2, tracks=("atac",)), _mk(["a"], 2, tracks=("dnase",))],
            "regions",
        )


def test_rejects_single_input():
    with pytest.raises(ValueError, match="at least two"):
        validate_concat([_mk(["a"], 2)], "regions")


def test_rejects_duplicate_region_names_on_region_axis():
    a = _mk(["a"], 2)
    b = _mk(["a"], 2)  # identical coordinates -> identical derived names
    a = replace(a, bed=a.bed.with_columns(name=pl.Series(["r0", "r1"])))
    b = replace(b, bed=b.bed.with_columns(name=pl.Series(["r1", "r2"])))
    with pytest.raises(ValueError, match="duplicate region name"):
        validate_concat([a, b], "regions")


def test_identical_coordinates_without_names_are_allowed():
    """Coordinate collisions are fine; only *name* collisions raise."""
    validate_concat([_mk(["a"], 2), _mk(["a"], 2)], "regions")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_validate.py -q`
Expected: FAIL — `ModuleNotFoundError`.

- [ ] **Step 3: Write the implementation**

Create `python/genvarloader/_dataset/_concat_validate.py`:

```python
"""Preconditions for :func:`genvarloader.concat`.

All checks run before any bytes move, so a rejected merge costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .._fasta_cache import Fingerprint, fingerprint
from ._write import Metadata

__all__ = ["ConcatInput", "load_inputs", "validate_concat", "variants_fingerprint"]


@dataclass(frozen=True)
class ConcatInput:
    """One resolved input dataset plus the facts validation needs."""

    path: Path | None
    meta: Metadata
    bed: pl.DataFrame
    n_regions: int
    n_samples: int
    backend: str
    tracks: list[str]
    annot_tracks: list[str]
    has_dosages: bool


def _backend_of(path: Path, meta: Metadata) -> str:
    if meta.svar2_link is not None:
        return "svar2"
    if meta.svar_link is not None:
        return "svar"
    if (path / "genotypes").is_dir():
        return "pgen_vcf"
    return "tracks_only"


def load_inputs(paths: list[Path]) -> list[ConcatInput]:
    """Read each dataset's metadata, bed, and store inventory.

    Args:
        paths: Dataset directories.

    Returns:
        One :class:`ConcatInput` per path, in the given order.
    """
    out = []
    for p in paths:
        meta = Metadata.model_validate_json((p / "metadata.json").read_text())
        bed = pl.read_ipc(p / "input_regions.arrow")
        tracks = sorted(d.name for d in (p / "intervals").glob("*") if d.is_dir())
        annot = sorted(d.name for d in (p / "annot_intervals").glob("*") if d.is_dir())
        out.append(
            ConcatInput(
                path=p,
                meta=meta,
                bed=bed,
                n_regions=meta.n_regions,
                n_samples=len(meta.samples),
                backend=_backend_of(p, meta),
                tracks=tracks,
                annot_tracks=annot,
                has_dosages=(p / "genotypes" / "dosages.npy").exists(),
            )
        )
    return out


def variants_fingerprint(path: Path) -> Fingerprint:
    """Bounded content fingerprint of a dataset's ``genotypes/variants.arrow``.

    Args:
        path: Dataset directory.

    Returns:
        A blake2b-over-1-MiB-plus-size fingerprint, cheap on a multi-GB index.
    """
    return fingerprint(path / "genotypes" / "variants.arrow")


def _region_names(inp: ConcatInput) -> list[str] | None:
    if "name" not in inp.bed.columns:
        return None
    return inp.bed["name"].to_list()


def validate_concat(inputs: list[ConcatInput], axis: str) -> None:
    """Check every precondition for a merge.

    Args:
        inputs: Resolved input datasets, in merge order.
        axis: Either ``"regions"`` or ``"samples"``.

    Raises:
        ValueError: If any precondition fails. The message names the offending
            input and the expected value.
    """
    if axis not in ("regions", "samples"):
        raise ValueError(f'axis must be "regions" or "samples", got {axis!r}')
    if len(inputs) < 2:
        raise ValueError(f"concat needs at least two datasets, got {len(inputs)}")

    ref = inputs[0]
    for i, inp in enumerate(inputs[1:], start=1):
        if inp.backend != ref.backend:
            raise ValueError(
                f"input #{i} uses variant source {inp.backend!r} but input #0 uses "
                f"{ref.backend!r}; all inputs must share the same variant source"
            )
        if inp.meta.ploidy != ref.meta.ploidy:
            raise ValueError(
                f"input #{i} has ploidy {inp.meta.ploidy}, expected {ref.meta.ploidy}"
            )
        if inp.meta.max_jitter != ref.meta.max_jitter:
            raise ValueError(
                f"input #{i} has max_jitter {inp.meta.max_jitter}, "
                f"expected {ref.meta.max_jitter}"
            )
        if inp.meta.contigs != ref.meta.contigs:
            raise ValueError(f"input #{i} has different contigs than input #0")
        if inp.tracks != ref.tracks:
            raise ValueError(
                f"input #{i} has tracks {inp.tracks}, expected {ref.tracks}"
            )
        if inp.annot_tracks != ref.annot_tracks:
            raise ValueError(
                f"input #{i} has annot tracks {inp.annot_tracks}, "
                f"expected {ref.annot_tracks}"
            )
        if inp.has_dosages != ref.has_dosages:
            raise ValueError(
                f"input #{i} {'has' if inp.has_dosages else 'lacks'} dosages, "
                "which does not match input #0"
            )

    if axis == "samples":
        for i, inp in enumerate(inputs[1:], start=1):
            if inp.n_regions != ref.n_regions or not inp.bed.equals(ref.bed):
                raise ValueError(
                    f"axis='samples' requires identical regions across inputs; "
                    f"input #{i} differs from input #0"
                )
        seen: dict[str, int] = {}
        for i, inp in enumerate(inputs):
            for s in inp.meta.samples:
                if s in seen:
                    raise ValueError(
                        f"axis='samples' requires disjoint sample sets, but sample "
                        f"{s!r} appears in inputs #{seen[s]} and #{i}"
                    )
                seen[s] = i
    else:
        for i, inp in enumerate(inputs[1:], start=1):
            if inp.meta.samples != ref.meta.samples:
                raise ValueError(
                    f"axis='regions' requires identical samples in identical order; "
                    f"input #{i} differs from input #0"
                )
        names_seen: dict[str, int] = {}
        for i, inp in enumerate(inputs):
            names = _region_names(inp)
            if names is None:
                continue
            for nm in names:
                if nm in names_seen:
                    raise ValueError(
                        f"duplicate region name {nm!r} in inputs #{names_seen[nm]} "
                        f"and #{i}; region names must be unique after merging"
                    )
                names_seen[nm] = i
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_validate.py -q`
Expected: PASS, 10 tests.

- [ ] **Step 5: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat_validate.py tests/unit/dataset/test_concat_validate.py
git commit -m "feat(concat): preconditions and variants.arrow fingerprint

All checks run before any bytes move. Variant-source identity reuses the
bounded blake2b fingerprint idiom from _fasta_cache rather than hashing a
multi-GB Arrow file.

Relates to #334."
```

---

### Task 4: Integration test fixtures (shard builders)

Builds the fixtures Tasks 6-7 assert against. Independent of Tasks 2 and 3 — it only writes datasets with the existing `gvl.write`.

**Files:**
- Create: `tests/dataset/test_concat.py` (fixtures + one skipped placeholder test)

**Interfaces:**
- Produces pytest fixtures:
  - `concat_case` — session-scoped; returns an object with `.bed_path`, `.samples`, `.pgen_path`, `.vcf_path`, `.svar_path`, `.ref_path`.
  - `region_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]` — `(shard_paths, whole_path)` split by region.
  - `sample_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]` — `(shard_paths, whole_path)` split by sample.

- [ ] **Step 1: Write the fixture module**

Create `tests/dataset/test_concat.py`:

```python
"""Integration tests for gvl.concat.

Oracle: shard a dataset, concat the shards, and compare against a single-shot
gvl.write of the whole thing.
"""

from pathlib import Path

import polars as pl
import pytest

import genvarloader as gvl


@pytest.fixture(scope="session")
def concat_case(synthetic_case):
    """Reuse the shared synthetic case (VCF + PGEN + SVAR + BED + samples)."""
    return synthetic_case


@pytest.fixture(scope="session")
def region_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    """Split the BED in half by row; write one dataset per half plus the whole."""
    d = tmp_path_factory.mktemp("concat_region")
    bed = gvl.read_bedlike(concat_case.bed_path)
    half = bed.height // 2
    assert half >= 1, "need >=2 regions to shard"
    parts = [bed[:half], bed[half:]]

    shards = []
    for i, part in enumerate(parts):
        p = d / f"shard{i}.gvl"
        gvl.write(p, part, concat_case.pgen_path, samples=concat_case.samples)
        shards.append(p)

    whole = d / "whole.gvl"
    gvl.write(whole, bed, concat_case.pgen_path, samples=concat_case.samples)
    return shards, whole


@pytest.fixture(scope="session")
def sample_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    """Split samples in half; write one dataset per half plus the whole."""
    d = tmp_path_factory.mktemp("concat_sample")
    bed = gvl.read_bedlike(concat_case.bed_path)
    samples = sorted(concat_case.samples)
    half = len(samples) // 2
    assert half >= 1, "need >=2 samples to shard"
    groups = [samples[:half], samples[half:]]

    shards = []
    for i, grp in enumerate(groups):
        p = d / f"shard{i}.gvl"
        gvl.write(p, bed, concat_case.pgen_path, samples=grp)
        shards.append(p)

    whole = d / "whole.gvl"
    gvl.write(whole, bed, concat_case.pgen_path, samples=samples)
    return shards, whole


def test_region_shards_fixture_builds(region_shards):
    shards, whole = region_shards
    assert len(shards) == 2
    assert all((p / "metadata.json").exists() for p in shards)
    assert (whole / "metadata.json").exists()


def test_sample_shards_fixture_builds(sample_shards):
    shards, whole = sample_shards
    assert len(shards) == 2
    assert all((p / "metadata.json").exists() for p in shards)
    assert (whole / "metadata.json").exists()
```

- [ ] **Step 2: Run to verify the fixtures build**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q`
Expected: PASS, 2 tests. If `synthetic_case` lacks >=2 regions or >=2 samples, the asserts fire — report that rather than weakening the assert.

- [ ] **Step 3: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add tests/dataset/test_concat.py
git commit -m "test(concat): shard-builder fixtures for the concat oracle

Region- and sample-sharded datasets plus a single-shot whole dataset,
which later tasks compare against.

Relates to #334."
```

---

### Task 5: `concat()` entry point — genotypes and metadata

Depends on Tasks 1, 2, 3. Handles the PGEN/VCF backend end to end; tracks come in Task 6.

**Files:**
- Create: `python/genvarloader/_dataset/_concat.py`
- Modify: `python/genvarloader/__init__.py`
- Modify: `tests/dataset/test_concat.py`

**Interfaces:**
- Consumes: `provenance`, `coalesce`, `Run` (`._concat_plan`); `copy_runs`, `gather_fixed`, `link_or_copy_buffered` (`._concat_io`); `load_inputs`, `validate_concat`, `variants_fingerprint` (`._concat_validate`); `_prep_bed`, `_write_regions`, `Metadata`, `DATASET_FORMAT_VERSION` (`._write`); `atomic_dir` (`.._atomic`).
- Produces: `concat(path, datasets, axis, *, overwrite=False, max_mem="4g") -> None`, exported as `gvl.concat`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_concat.py`:

```python
def _read_offsets(p: Path):
    import numpy as np
    return np.fromfile(p / "genotypes" / "offsets.npy", dtype=np.int64)


def _read_v_idxs(p: Path):
    import numpy as np
    from genoray._types import V_IDX_TYPE
    return np.fromfile(p / "genotypes" / "variant_idxs.npy", dtype=V_IDX_TYPE)


def test_concat_regions_matches_single_shot_bytes(tmp_path, region_shards):
    """Region concat must be byte-identical: regions are independent and the
    sample set is unchanged."""
    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    assert (_read_offsets(out) == _read_offsets(whole)).all()
    assert (_read_v_idxs(out) == _read_v_idxs(whole)).all()


def test_concat_regions_metadata_matches(tmp_path, region_shards):
    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    import json
    got = json.loads((out / "metadata.json").read_text())
    exp = json.loads((whole / "metadata.json").read_text())
    assert got["samples"] == exp["samples"]
    assert got["n_regions"] == exp["n_regions"]
    assert got["ploidy"] == exp["ploidy"]
    assert got["contigs"] == exp["contigs"]


def test_concat_samples_merges_sample_list(tmp_path, sample_shards):
    shards, whole = sample_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")

    import json
    got = json.loads((out / "metadata.json").read_text())
    exp = json.loads((whole / "metadata.json").read_text())
    assert got["samples"] == exp["samples"]
    assert got["n_regions"] == exp["n_regions"]


def test_concat_rejects_wrong_axis(tmp_path, region_shards):
    shards, _ = region_shards
    with pytest.raises(ValueError, match="axis must be"):
        gvl.concat(tmp_path / "x.gvl", shards, axis="contigs")


def test_concat_refuses_overwrite_by_default(tmp_path, region_shards):
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")
    with pytest.raises(FileExistsError):
        gvl.concat(out, shards, axis="regions")


def test_concat_records_variants_fingerprint(tmp_path, region_shards):
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    import json
    meta = json.loads((out / "metadata.json").read_text())
    fp = meta["variants_fingerprint"]
    assert fp["algorithm"] == "blake2b"
    assert fp["size_bytes"] > 0
    assert len(fp["digest"]) > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q -k "concat_"`
Expected: FAIL — `AttributeError: module 'genvarloader' has no attribute 'concat'`

- [ ] **Step 3: Add the `variants_fingerprint` field to `Metadata`**

In `python/genvarloader/_dataset/_write.py`, add to `class Metadata` (after `svar2_link`):

```python
    variants_fingerprint: "Fingerprint | None" = None
```

and add near the other imports:

```python
from .._fasta_cache import Fingerprint
```

- [ ] **Step 4: Write `_concat.py`**

Create `python/genvarloader/_dataset/_concat.py`:

```python
"""Merge on-disk GVL datasets along the region or sample axis."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Sequence

import numpy as np
import polars as pl

from .._atomic import atomic_dir
from ._concat_io import copy_runs, gather_fixed, link_or_copy_buffered
from ._concat_plan import coalesce, provenance
from ._concat_validate import load_inputs, validate_concat, variants_fingerprint
from ._write import DATASET_FORMAT_VERSION, Metadata, _prep_bed, _write_regions

if TYPE_CHECKING:
    from ._impl import Dataset

__all__ = ["concat"]


def _merged_bed(inputs, axis: str) -> tuple[pl.DataFrame, pl.DataFrame, np.ndarray]:
    """Return (input_bed_with_map, sorted_gvl_bed, dataset_of_each_sorted_row)."""
    if axis == "samples":
        bed = inputs[0].bed.drop("r_idx_map", strict=False)
        gvl_bed, _contigs, r_map = _prep_bed(bed, None)
        return bed.with_columns(r_idx_map=pl.Series(r_map)), gvl_bed, None

    parts = []
    for d, inp in enumerate(inputs):
        b = inp.bed.drop("r_idx_map", strict=False)
        parts.append(b.with_columns(_ds=pl.lit(d, pl.Int64)))
    merged_in = pl.concat(parts, how="vertical_relaxed")
    gvl_bed, _contigs, r_map = _prep_bed(merged_in.drop("_ds"), None)
    # r_map maps input row order -> sorted order; invert to get sorted -> input row.
    sorted_to_input = np.argsort(r_map)
    ds_of_sorted = merged_in["_ds"].to_numpy()[sorted_to_input]
    out_bed = merged_in.drop("_ds").with_columns(r_idx_map=pl.Series(r_map))
    return out_bed, gvl_bed, ds_of_sorted


def concat(
    path: str | Path,
    datasets: "Sequence[str | Path | Dataset]",
    axis: Literal["regions", "samples"],
    *,
    overwrite: bool = False,
    max_mem: int | str = "4g",
) -> None:
    """Merge GVL datasets along the region or sample axis.

    All inputs must share one variant source: the same PGEN/VCF variant table, or
    the same ``.svar``/``.svar2`` store. Merging datasets built from *different*
    variant sources is not supported and raises.

    On ``axis="regions"`` the inputs must have identical samples in identical order
    and their regions are concatenated. On ``axis="samples"`` the inputs must have
    identical regions and disjoint sample sets, and their samples are merged into
    sorted order.

    This moves roughly the full size of the merged dataset at sequential-IO speed.
    It is worth doing only against the alternative of re-extracting genotypes.

    Args:
        path: Destination dataset directory.
        datasets: Two or more dataset directories or opened :class:`Dataset` objects.
        axis: ``"regions"`` to concatenate regions, ``"samples"`` to concatenate samples.
        overwrite: Replace ``path`` if it already exists.
        max_mem: Advisory memory budget. Accepted for symmetry with :func:`write`.

    Raises:
        ValueError: If any precondition fails; the message names the offending input.
        FileExistsError: If ``path`` exists and ``overwrite`` is False.
    """
    from ._impl import Dataset as _Dataset

    paths = [
        Path(d.path if isinstance(d, _Dataset) else d) for d in datasets
    ]
    dest = Path(path)
    if dest.exists() and not overwrite:
        raise FileExistsError(f"{dest} exists; pass overwrite=True to replace it")

    inputs = load_inputs(paths)
    validate_concat(inputs, axis)

    ref = inputs[0]
    ploidy = ref.meta.ploidy or 1
    shapes = [(i.n_regions, i.n_samples) for i in inputs]

    if axis == "regions":
        n_regions = sum(i.n_regions for i in inputs)
        samples = list(ref.meta.samples)
    else:
        n_regions = ref.n_regions
        samples = sorted(s for i in inputs for s in i.meta.samples)

    input_bed, gvl_bed, _ds_of_sorted = _merged_bed(inputs, axis)

    with atomic_dir(dest, overwrite=overwrite) as tmp:
        input_bed.write_ipc(tmp / "input_regions.arrow")
        _write_regions(tmp, gvl_bed, ref.meta.contigs)

        meta: dict = {
            "samples": samples,
            "contigs": ref.meta.contigs,
            "n_regions": n_regions,
            "ploidy": ref.meta.ploidy,
            "max_jitter": ref.meta.max_jitter,
            "version": ref.meta.version,
            "format_version": DATASET_FORMAT_VERSION,
        }

        if ref.backend == "pgen_vcf":
            geno = tmp / "genotypes"
            geno.mkdir(parents=True, exist_ok=True)
            link_or_copy_buffered(
                paths[0] / "genotypes" / "variants.arrow",
                geno / "variants.arrow",
            )
            meta["variants_fingerprint"] = variants_fingerprint(paths[0])

            prov = provenance(axis, shapes, ploidy)
            runs = coalesce(prov)
            src_offsets = [
                np.fromfile(p / "genotypes" / "offsets.npy", dtype=np.int64)
                for p in paths
            ]
            from genoray._types import V_IDX_TYPE

            merged = copy_runs(
                [p / "genotypes" / "variant_idxs.npy" for p in paths],
                geno / "variant_idxs.npy",
                runs,
                src_offsets,
                itemsize=np.dtype(V_IDX_TYPE).itemsize,
            )
            with open(geno / "offsets.npy", "wb") as f:
                f.write(merged.tobytes())

            if ref.has_dosages:
                from ._write import DOSAGE_TYPE  # noqa: PLC0415

                copy_runs(
                    [p / "genotypes" / "dosages.npy" for p in paths],
                    geno / "dosages.npy",
                    runs,
                    src_offsets,
                    itemsize=np.dtype(DOSAGE_TYPE).itemsize,
                )
        elif ref.backend in ("svar", "svar2"):
            raise NotImplementedError(
                f"concat for {ref.backend!r}-backed datasets lands in a follow-up task"
            )

        with open(tmp / "metadata.json", "w") as f:
            f.write(Metadata(**meta).model_dump_json())
```

- [ ] **Step 5: Export `concat`**

In `python/genvarloader/__init__.py`, add `from ._dataset._concat import concat` alongside the other `_dataset` imports, and add `"concat"` to `__all__` in alphabetical position (between `"Table"`-group entries and `"data_registry"` — the list is alphabetical with capitalized names first, so `"concat"` goes immediately before `"data_registry"`).

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q`
Expected: PASS. If `DOSAGE_TYPE` is not importable from `._write`, find its real home with `grep -rn "DOSAGE_TYPE" python/genvarloader/` and import from there.

- [ ] **Step 7: Verify api.md sync**

Run:
```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: ['concat']` — that is fixed in Task 8. Note it and continue.

- [ ] **Step 8: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat.py python/genvarloader/_dataset/_write.py python/genvarloader/__init__.py tests/dataset/test_concat.py
git commit -m "feat(concat): gvl.concat entry point for PGEN/VCF-backed datasets

Merges genotypes, regions, and metadata along either axis. variants.arrow
is hardlinked from the first input and guarded by a recorded fingerprint.

Relates to #334."
```

---

### Task 6: Tracks, annot tracks, and the svar/svar2 backends

Completes the per-store table from the spec. Depends on Task 5.

**Files:**
- Modify: `python/genvarloader/_dataset/_concat.py`
- Modify: `tests/dataset/test_concat.py`

**Interfaces:**
- Consumes: everything from Task 5.
- Produces: no new public API; `concat` gains track and `.svar`/`.svar2` support.

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_concat.py`:

```python
@pytest.fixture(scope="session")
def svar_region_shards(tmp_path_factory, concat_case) -> tuple[list[Path], Path]:
    d = tmp_path_factory.mktemp("concat_svar")
    bed = gvl.read_bedlike(concat_case.bed_path)
    half = bed.height // 2
    shards = []
    for i, part in enumerate([bed[:half], bed[half:]]):
        p = d / f"shard{i}.gvl"
        gvl.write(p, part, concat_case.svar_path, samples=concat_case.samples)
        shards.append(p)
    whole = d / "whole.gvl"
    gvl.write(whole, bed, concat_case.svar_path, samples=concat_case.samples)
    return shards, whole


def test_concat_svar_regions_matches_single_shot(tmp_path, svar_region_shards):
    import numpy as np
    shards, whole = svar_region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    got = np.fromfile(out / "genotypes" / "offsets.npy", dtype=np.int64)
    exp = np.fromfile(whole / "genotypes" / "offsets.npy", dtype=np.int64)
    assert (got == exp).all()


def test_concat_svar_preserves_link(tmp_path, svar_region_shards):
    import json
    shards, whole = svar_region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    got = json.loads((out / "metadata.json").read_text())
    exp = json.loads((whole / "metadata.json").read_text())
    assert got["svar_link"]["fingerprint"] == exp["svar_link"]["fingerprint"]


def test_concat_regions_reads_equal_to_single_shot(tmp_path, region_shards, reference):
    """The real acceptance check: every merged cell reads identically."""
    import numpy as np
    shards, whole = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    a = gvl.Dataset.open(out, reference).with_seqs("haplotypes")
    b = gvl.Dataset.open(whole, reference).with_seqs("haplotypes")
    assert a.shape == b.shape
    for r in range(a.shape[0]):
        for s in range(a.shape[1]):
            ha, hb = a[r, s], b[r, s]
            assert np.array_equal(ha.to_padded(b"N"), hb.to_padded(b"N")), (r, s)


def test_concat_samples_reads_equal_to_single_shot(tmp_path, sample_shards, reference):
    """Sample axis asserts read equality, NOT byte identity.

    extend_to_length sizes each region's window to the max over the cohort present
    at write time, so a shard's stored variant set can legitimately differ from the
    full write's. Byte identity is not expected here.
    """
    import numpy as np
    shards, whole = sample_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="samples")

    a = gvl.Dataset.open(out, reference).with_seqs("haplotypes")
    b = gvl.Dataset.open(whole, reference).with_seqs("haplotypes")
    assert a.shape == b.shape
    for r in range(a.shape[0]):
        for s in range(a.shape[1]):
            ha, hb = a[r, s], b[r, s]
            assert np.array_equal(ha.to_padded(b"N"), hb.to_padded(b"N")), (r, s)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q -k "svar or reads_equal"`
Expected: FAIL — `NotImplementedError` for svar, and the read-equality tests fail because tracks/opening paths are incomplete.

- [ ] **Step 3: Implement the remaining stores**

In `_concat.py`, replace the `elif ref.backend in ("svar", "svar2")` branch and add track handling before the metadata write:

```python
        elif ref.backend == "svar":
            geno = tmp / "genotypes"
            geno.mkdir(parents=True, exist_ok=True)
            import json as _json

            svar_meta = _json.loads(
                (paths[0] / "genotypes" / "svar_meta.json").read_text()
            )
            # offsets are (2, R, S, P) absolute start/stop pairs into the svar's
            # global array. Gather the R/S axes; values copy verbatim.
            prov = provenance(axis, shapes, ploidy)
            runs = coalesce(prov)
            _gather_svar_offsets(paths, geno, runs, shapes, ploidy, axis)
            shape = [2, n_regions, len(samples), ploidy]
            (geno / "svar_meta.json").write_text(
                _json.dumps({"shape": shape, "dtype": svar_meta["dtype"]})
            )
            meta["svar_link"] = ref.meta.svar_link.model_dump()

        elif ref.backend == "svar2":
            out_dir = tmp / "genotypes" / "svar2_ranges"
            out_dir.mkdir(parents=True, exist_ok=True)
            _concat_svar2_ranges(paths, out_dir, axis, shapes, ploidy,
                                 n_regions, len(samples), inputs)
            meta["svar2_link"] = ref.meta.svar2_link.model_dump()

        # per-sample tracks: offsets over (R, S), no ploidy axis
        if ref.tracks:
            t_prov = provenance(axis, shapes, 1)
            t_runs = coalesce(t_prov)
            for name in ref.tracks:
                src_dirs = [p / "intervals" / name for p in paths]
                out_t = tmp / "intervals" / name
                out_t.mkdir(parents=True, exist_ok=True)
                t_offsets = [
                    np.fromfile(d / "offsets.npy", dtype=np.int64) for d in src_dirs
                ]
                merged_t = None
                for fname, dt in (
                    ("starts", np.int32), ("ends", np.int32), ("values", np.float32)
                ):
                    merged_t = copy_runs(
                        [d / f"{fname}.npy" for d in src_dirs],
                        out_t / f"{fname}.npy",
                        t_runs,
                        t_offsets,
                        itemsize=np.dtype(dt).itemsize,
                    )
                with open(out_t / "offsets.npy", "wb") as f:
                    f.write(merged_t.tobytes())

        # annot tracks: sample-independent, offsets over R only
        if ref.annot_tracks:
            for name in ref.annot_tracks:
                out_a = tmp / "annot_intervals" / name
                out_a.mkdir(parents=True, exist_ok=True)
                src_dirs = [p / "annot_intervals" / name for p in paths]
                if axis == "samples":
                    for fname in ("starts", "ends", "values", "offsets"):
                        link_or_copy_buffered(
                            src_dirs[0] / f"{fname}.npy", out_a / f"{fname}.npy"
                        )
                else:
                    a_prov = provenance("regions", [(r, 1) for r, _ in shapes], 1)
                    a_runs = coalesce(a_prov)
                    a_offsets = [
                        np.fromfile(d / "offsets.npy", dtype=np.int64) for d in src_dirs
                    ]
                    merged_a = None
                    for fname, dt in (
                        ("starts", np.int32), ("ends", np.int32), ("values", np.float32)
                    ):
                        merged_a = copy_runs(
                            [d / f"{fname}.npy" for d in src_dirs],
                            out_a / f"{fname}.npy",
                            a_runs,
                            a_offsets,
                            itemsize=np.dtype(dt).itemsize,
                        )
                    with open(out_a / "offsets.npy", "wb") as f:
                        f.write(merged_a.tobytes())
```

Add these two module-level helpers to `_concat.py`:

```python
def _gather_svar_offsets(paths, out_dir, runs, shapes, ploidy, axis) -> None:
    """Gather a .svar dataset's (2, R, S, P) absolute start/stop offsets.

    The two leading planes (starts, stops) are gathered independently because the
    slot axis is not the outermost one; each plane is a contiguous fixed-stride
    array of R*S*P int64 values.
    """
    import numpy as np

    n_src_slots = [r * s * ploidy for r, s in shapes]
    planes = []
    for plane in (0, 1):
        srcs = []
        for p, n in zip(paths, n_src_slots):
            arr = np.fromfile(p / "genotypes" / "offsets.npy", dtype=np.int64)
            srcs.append(arr.reshape(2, -1)[plane])
        out = np.empty(sum(r.src_stop - r.src_start for r in runs), np.int64)
        for r in runs:
            n = r.src_stop - r.src_start
            out[r.dst_start : r.dst_start + n] = srcs[r.src][r.src_start : r.src_stop]
        planes.append(out)
    with open(out_dir / "offsets.npy", "wb") as f:
        f.write(np.stack(planes).tobytes())


def _concat_svar2_ranges(
    paths, out_dir, axis, shapes, ploidy, n_regions, n_samples, inputs
) -> None:
    """Merge a .svar2 dataset's cached range arrays.

    ``vk_*_range`` are per-(region, sample, ploid); ``dense_*_range`` are
    per-region only; ``sample_cols`` maps merged sample slot -> store sample index.
    """
    import json as _json

    import numpy as np

    from ._concat_plan import coalesce as _coalesce
    from ._concat_plan import provenance as _provenance

    prov = _provenance(axis, shapes, ploidy)
    runs = _coalesce(prov)

    for name in ("vk_snp_range", "vk_indel_range"):
        srcs = [
            np.fromfile(p / "genotypes" / "svar2_ranges" / f"{name}.npy", np.int64)
            .reshape(-1, 2)
            for p in paths
        ]
        out = np.empty((n_regions * n_samples * ploidy, 2), np.int64)
        for r in runs:
            n = r.src_stop - r.src_start
            out[r.dst_start : r.dst_start + n] = srcs[r.src][r.src_start : r.src_stop]
        out.tofile(out_dir / f"{name}.npy")

    for name in ("dense_snp_range", "dense_indel_range"):
        srcs = [
            np.fromfile(p / "genotypes" / "svar2_ranges" / f"{name}.npy", np.int64)
            .reshape(-1, 2)
            for p in paths
        ]
        if axis == "samples":
            out = srcs[0]
        else:
            out = np.concatenate(srcs, axis=0)
        out.tofile(out_dir / f"{name}.npy")

    cols = [
        np.load(p / "genotypes" / "svar2_ranges" / "sample_cols.npy") for p in paths
    ]
    if axis == "samples":
        all_samples = [(s, c) for i, inp in zip(cols, inputs)
                       for s, c in zip(inp.meta.samples, i)]
        merged_cols = np.array(
            [c for _s, c in sorted(all_samples, key=lambda t: t[0])], np.int64
        )
    else:
        merged_cols = cols[0]
    np.save(out_dir / "sample_cols.npy", merged_cols)

    src_meta = _json.loads(
        (paths[0] / "genotypes" / "svar2_ranges" / "svar2_meta.json").read_text()
    )
    R, S, P = n_regions, n_samples, ploidy
    src_meta["vk_snp_range"]["shape"] = [R, S, P, 2]
    src_meta["vk_indel_range"]["shape"] = [R, S, P, 2]
    src_meta["dense_snp_range"]["shape"] = [R, 2]
    src_meta["dense_indel_range"]["shape"] = [R, 2]
    src_meta["sample_cols"]["shape"] = [S]
    (out_dir / "svar2_meta.json").write_text(_json.dumps(src_meta))
```

Also add `regions.npy` handling for the sample axis before `_write_regions`: when `axis == "samples"`, take the elementwise max of the end column across inputs rather than trusting input 0.

```python
        if axis == "samples":
            stored = [np.load(p / "regions.npy") for p in paths]
            merged_regions = stored[0].copy()
            for s in stored[1:]:
                merged_regions[:, 2] = np.maximum(merged_regions[:, 2], s[:, 2])
            np.save(tmp / "regions.npy", merged_regions)
        else:
            _write_regions(tmp, gvl_bed, ref.meta.contigs)
```

(replacing the unconditional `_write_regions` call from Task 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q`
Expected: PASS. If `test_concat_samples_reads_equal_to_single_shot` fails, **do not weaken the assertion.** Per the spec, a read-level divergence on the sample axis is a real finding — capture the failing `(r, s)` and the two haplotypes, and report it.

- [ ] **Step 5: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat.py tests/dataset/test_concat.py
git commit -m "feat(concat): tracks, annot tracks, and svar/svar2 backends

Completes the per-store merge table: per-sample tracks gather over (R, S),
annot tracks are sample-independent, and the svar/svar2 range arrays copy
verbatim as fixed-stride gathers since they hold absolute ranges into an
external store.

Relates to #334."
```

---

### Task 7: Fingerprint verification at open

Depends on Task 5. Closes the loop on the hardlink decision.

**Files:**
- Modify: `python/genvarloader/_dataset/_open.py`
- Test: `tests/dataset/test_concat.py`

**Interfaces:**
- Consumes: `Metadata.variants_fingerprint` (Task 5).
- Produces: `Dataset.open` raises `ValueError` on a mismatched `variants.arrow`.

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_concat.py`:

```python
def test_open_rejects_mutated_variants_arrow(tmp_path, region_shards, reference):
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    # Break the hardlink, then corrupt the copy so the fingerprint mismatches.
    va = out / "genotypes" / "variants.arrow"
    data = bytearray(va.read_bytes())
    va.unlink()
    data[:16] = b"\x00" * 16
    va.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="variants.arrow"):
        gvl.Dataset.open(out, reference)


def test_open_accepts_absent_fingerprint(tmp_path, region_shards, reference):
    """Datasets written before the field exists must still open."""
    import json
    shards, _ = region_shards
    out = tmp_path / "merged.gvl"
    gvl.concat(out, shards, axis="regions")

    meta = json.loads((out / "metadata.json").read_text())
    meta["variants_fingerprint"] = None
    (out / "metadata.json").write_text(json.dumps(meta))

    gvl.Dataset.open(out, reference)  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q -k "fingerprint or mutated"`
Expected: the mismatch test FAILS (no error raised).

- [ ] **Step 3: Implement verification**

In `_open.py`, add a stage method to `OpenRequest` and call it from `resolve()` right after `self._load_metadata()`:

```python
    def _verify_variants_fingerprint(self, metadata: Metadata) -> None:
        """Verify the hardlinked variants.arrow still matches what was recorded.

        Absent fingerprints (datasets written before the field existed) skip the
        check, mirroring the pre-0.25.0 svar_link migration path.

        Args:
            metadata: The dataset's parsed metadata.

        Raises:
            ValueError: If the recorded fingerprint does not match the file.
        """
        fp = metadata.variants_fingerprint
        if fp is None:
            return
        va = self.path / "genotypes" / "variants.arrow"
        if not va.exists():
            return
        from .._fasta_cache import fingerprint as _fingerprint

        if va.stat().st_size != fp.size_bytes or _fingerprint(va).digest != fp.digest:
            raise ValueError(
                f"Dataset at {self.path}: genotypes/variants.arrow does not match the "
                f"fingerprint recorded when the dataset was created. The variant index "
                f"was modified out of band; variant indices would resolve against the "
                f"wrong table. Rebuild the dataset."
            )
```

Wire it in `resolve()`:

```python
        metadata = self._load_metadata()
        self._verify_variants_fingerprint(metadata)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_concat.py -q -k "fingerprint or mutated"`
Expected: PASS, 2 tests.

- [ ] **Step 5: Run the dataset + unit trees**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: PASS. The new `_open.py` stage runs for every dataset, so a regression here would surface broadly.

- [ ] **Step 6: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_open.py tests/dataset/test_concat.py
git commit -m "feat(concat): verify variants.arrow fingerprint at open

Converts an out-of-band variant-index rewrite from silent wrong data into
a loud error. Absent fingerprints skip the check so pre-existing datasets
still open.

Relates to #334, #337."
```

---

### Task 8: Documentation

Depends on Tasks 5-7. Required by the repo's docs gates in CLAUDE.md.

**Files:**
- Modify: `docs/source/api.md`, `docs/source/write.md`, `docs/source/faq.md`, `skills/genvarloader/SKILL.md`

- [ ] **Step 1: Add the autodoc entry**

In `docs/source/api.md`, add alongside the other function entries (match the surrounding directive style exactly — check how `write` and `update` are declared and copy that form):

```markdown
```{eval-rst}
.. autofunction:: genvarloader.concat
```
```

- [ ] **Step 2: Verify api.md is in sync**

Run:
```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`

- [ ] **Step 3: Document the workflow and the #334 Q1 answer**

Add to `docs/source/write.md` a section titled "Reusing a variant index across cohorts":

```markdown
## Reusing a variant index across cohorts

If several cohorts are sample subsets of one parent PGEN, do **not** pre-split the
PGEN with `plink2 --keep`. Point `gvl.write` at the parent and pass `samples=`:

```python
gvl.write("cohort_A.gvl", bed, "parent.pgen", samples=cohort_A_samples)
gvl.write("cohort_B.gvl", bed, "parent.pgen", samples=cohort_B_samples)
```

genoray builds and caches the parent's variant index once, and each dataset
hardlinks it rather than rebuilding it. Splitting the PGEN first is what forces the
expensive per-cohort index rebuild.

## Merging datasets

`gvl.concat` merges datasets that share one variant source, along either axis:

```python
gvl.concat("merged.gvl", ["chr1.gvl", "chr2.gvl"], axis="regions")
gvl.concat("merged.gvl", ["cohortA.gvl", "cohortB.gvl"], axis="samples")
```

`axis="regions"` requires identical samples in identical order; `axis="samples"`
requires identical regions and disjoint sample sets.

This moves roughly the full size of the merged dataset at sequential-IO speed — a
1 TB merge takes on the order of hours. It is worth doing against the alternative of
re-extracting genotypes, not as a routine step.

For contig-sharded `.svar2` workflows there is a cheaper path: merge the *stores*
with genoray's `SparseVar2.concat`, then run `gvl.write` once, since `.svar2` writes
only cache ranges.
```

- [ ] **Step 4: Add an FAQ entry**

Add to `docs/source/faq.md`:

```markdown
### Can I build a dataset in parallel shards and merge them?

Yes. Split your BED across jobs, `gvl.write` one dataset per shard, then merge with
`gvl.concat(out, shards, axis="regions")`. All shards must use the same variant
source. See "Merging datasets" in the write guide for the cost model.
```

- [ ] **Step 5: Update the skill**

In `skills/genvarloader/SKILL.md`, add `concat` to the public-API section with its signature, both axes' preconditions, and a one-line cost warning. Then re-check the "Common gotchas" and "Where to look next" pointer tables for anything the new symbol makes stale.

- [ ] **Step 6: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS. CLAUDE.md requires the full tree before pushing a change that adds a public symbol — a scoped run would miss stale references in `tests/unit/`.

- [ ] **Step 7: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/
git add docs/source/api.md docs/source/write.md docs/source/faq.md skills/genvarloader/SKILL.md
git commit -m "docs(concat): document gvl.concat and parent-PGEN cohort workflow

Adds the api.md autodoc entry required to keep docs in sync with __all__,
the concat workflow and cost model, and the parent.pgen + samples=
guidance that answers the first half of #334.

Closes #334."
```

---

## Self-Review

**1. Spec coverage.** Every spec section maps to a task: §2 API → Task 5; §3 merge core → Tasks 1, 2; §3.1 execution constraints → Global Constraints + Task 2; §4 per-store rules → Tasks 5, 6 (including `regions.npy` elementwise max and the region re-sort via `_prep_bed`); §5 validation → Task 3; §5.1 hardlink+fingerprint → Tasks 2, 5, 7; §6 testing → Tasks 4, 6 (byte-identity on regions, read-equality on samples); §7 docs → Task 8.

**2. Placeholder scan.** No TBD/TODO. Every code step carries real code. Two deliberate lookups are flagged with the exact command to resolve them (`DOSAGE_TYPE`'s module in Task 5 Step 6; the `api.md` directive style in Task 8 Step 1) rather than left vague.

**3. Type consistency.** `Run`'s four fields are used identically in Tasks 1, 2, 5, 6. `provenance(axis, shape_per_ds, ploidy)` keeps its three-arg form throughout, with `ploidy=1` for interval stores. `copy_runs` returns merged offsets in *elements*, and every caller writes them with `.tobytes()`. `ConcatInput` field names match between Task 3's definition and Tasks 5-6's use.

**Known risk, deliberately not designed around:** Task 6's `test_concat_samples_reads_equal_to_single_shot` may fail because of the `extend_to_length` window-sizing difference the spec calls out. The plan instructs the implementer to report it rather than weaken the assertion, because that outcome is information, not an obstacle.

## Related

- Spec: `docs/superpowers/specs/2026-07-31-dataset-concat-design.md`
- Issue [#334](https://github.com/mcvickerlab/GenVarLoader/issues/334), PR [#339](https://github.com/mcvickerlab/GenVarLoader/pull/339)
- Out of scope: [#337](https://github.com/mcvickerlab/GenVarLoader/issues/337), [#338](https://github.com/mcvickerlab/GenVarLoader/issues/338)
