# Generalize `gvl.write()` to accept Table tracks — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `gvl.write(..., bigwigs=...)` with a generalized `tracks=` parameter that accepts `BigWigs`, a new `Table` class, or a heterogeneous sequence of either, unified behind an `IntervalTrack` Protocol.

**Architecture:** Define a structural `IntervalTrack` Protocol (`name`, `samples`, `contigs`, `count_intervals`, `_intervals_from_offsets`) that `BigWigs` already satisfies. Add a new `Table` class backed by an eager polars DataFrame and powered by `polars-bio`'s `count_overlaps`/`overlap` operations, so the existing `_write_bigwigs` routine generalizes (renamed `_write_track`) with no behavioral change. The on-disk layout is unchanged — readers don't care which track type produced `intervals/<name>/`.

**Tech Stack:** Python 3.10+, polars, polars-bio (new dep), numpy, pytest, pixi (`pixi run -e dev ...`), maturin build system.

**Spec:** `docs/superpowers/specs/2026-05-08-tracks-generalize-design.md`

---

## File Structure

**Created:**
- `python/genvarloader/_table.py` — new `Table` class.
- `tests/test_table.py` — unit tests for `Table` (init forms, `column_map`, `from_path`, `count_intervals`, `_intervals_from_offsets`).
- `tests/dataset/test_write_tracks.py` — end-to-end tests for the new `tracks=` write path (Table-only and mixed BigWigs+Table).

**Modified:**
- `python/genvarloader/_types.py` — add `IntervalTrack` Protocol.
- `python/genvarloader/_dataset/_write.py` — rename param `bigwigs` → `tracks`, normalize to list, validate unique names, rename helper `_write_bigwigs` → `_write_track`.
- `python/genvarloader/__init__.py` — export `Table`.
- `pyproject.toml` — add `polars-bio` to `dependencies`.

**Unchanged:** `_bigwig.py`, `_dataset/_impl.py`, `_dataset/_reconstruct.py`, all reader-side code.

---

## Task 1: Add `IntervalTrack` Protocol

**Files:**
- Modify: `python/genvarloader/_types.py`

- [ ] **Step 1: Add the Protocol at the bottom of `_types.py`**

After the existing `Reader` class, append:

```python
class IntervalTrack(Protocol):
    """Structural protocol implemented by interval-valued track readers
    (e.g. :class:`BigWigs`, :class:`Table`). Used by :func:`gvl.write()` to
    accept either source via the ``tracks=`` parameter.
    """

    name: str
    samples: list[str]
    contigs: Mapping[str, int]

    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        """Return shape ``(regions, samples)`` count of intervals overlapping each
        ``(region, sample)`` cell."""
        ...

    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> "RaggedIntervals":
        """Read intervals using offsets pre-computed from :meth:`count_intervals`."""
        ...
```

Add `from ._ragged import RaggedIntervals` inside a `TYPE_CHECKING` block at the top so the runtime is unaffected:

```python
from typing import Protocol, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from ._ragged import RaggedIntervals
```

Add `"IntervalTrack"` to `__all__`.

- [ ] **Step 2: Run typecheck to verify Protocol is well-formed**

```bash
pixi run -e dev basedpyright python/genvarloader/_types.py
```

Expected: no new errors.

- [ ] **Step 3: Commit**

```bash
git add python/genvarloader/_types.py
git commit -m "feat: add IntervalTrack Protocol for unified track sources"
```

---

## Task 2: Verify `BigWigs` conforms to `IntervalTrack`

**Files:**
- Test: `tests/test_interval_track.py` (new)

- [ ] **Step 1: Write the failing test**

Create `tests/test_interval_track.py`:

```python
from pathlib import Path

from genvarloader._bigwig import BigWigs
from genvarloader._types import IntervalTrack


def test_bigwigs_satisfies_interval_track_protocol():
    data_dir = Path(__file__).parent / "data" / "bigwig"
    bw = BigWigs(
        "signal",
        {"sample_0": str(data_dir / "sample_0.bw"),
         "sample_1": str(data_dir / "sample_1.bw")},
    )
    # runtime structural check: required attributes/methods are present
    assert hasattr(bw, "name")
    assert hasattr(bw, "samples")
    assert hasattr(bw, "contigs")
    assert callable(getattr(bw, "count_intervals", None))
    assert callable(getattr(bw, "_intervals_from_offsets", None))
    # static checker affirms it via type alias usage
    track: IntervalTrack = bw  # noqa: F841
```

- [ ] **Step 2: Run test to verify it passes**

```bash
pixi run -e dev pytest tests/test_interval_track.py -v
```

Expected: PASS (BigWigs already has all required members; this test is a regression guard).

- [ ] **Step 3: Commit**

```bash
git add tests/test_interval_track.py
git commit -m "test: BigWigs conforms to IntervalTrack protocol"
```

---

## Task 3: Add `polars-bio` dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add `polars-bio` to `dependencies`**

In `pyproject.toml`, in the `[project] dependencies = [...]` list (currently lines 12-35), add a new line:

```
    "polars-bio",
```

- [ ] **Step 2: Refresh the pixi/lock environment**

```bash
pixi install -e dev
```

Expected: `polars-bio` installs cleanly. If pixi's lockfile resolution fails, surface the error rather than working around it.

- [ ] **Step 3: Smoke-test the import**

```bash
pixi run -e dev python -c "import polars_bio as pb; print(pb.__version__); print(hasattr(pb, 'overlap'), hasattr(pb, 'count_overlaps'))"
```

Expected output: a version string and `True True`.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add polars-bio dependency for Table interval queries"
```

---

## Task 4: API spike — characterize `polars_bio.overlap`/`count_overlaps`

This is a one-off exploratory script to lock down exact column names and zero-fill behavior. It is NOT a permanent test; it produces notes used in subsequent tasks. The spec called this out as a risk to verify before wiring.

**Files:**
- Create (temporary): `scratch/polars_bio_spike.py`

- [ ] **Step 1: Write a small probe script**

Create `scratch/polars_bio_spike.py`:

```python
"""Throwaway: probe polars_bio.overlap / count_overlaps behavior.

Captures the exact output schema (column names) and zero-fill behavior of
count_overlaps for query rows with no matches. Run once; record findings in
the implementation; delete afterward.
"""
import polars as pl
import polars_bio as pb

queries = pl.DataFrame({
    "chrom": ["chr1", "chr1", "chr2"],
    "start": [0, 50, 0],
    "end":   [10, 60, 10],
    "_q":    [0, 1, 2],
    "sample_id": ["s0", "s0", "s0"],
})
table = pl.DataFrame({
    "chrom": ["chr1", "chr1"],
    "start": [2, 100],
    "end":   [5, 105],
    "value": [1.0, 2.0],
    "sample_id": ["s0", "s0"],
})

print("=== overlap ===")
ov = pb.overlap(
    queries, table,
    cols1=["chrom", "start", "end"],
    cols2=["chrom", "start", "end"],
    on_cols=["sample_id"],
    output_type="polars.DataFrame",
)
print(ov.schema)
print(ov)

print("=== count_overlaps ===")
co = pb.count_overlaps(
    queries, table,
    cols1=["chrom", "start", "end"],
    cols2=["chrom", "start", "end"],
    on_cols=["sample_id"],
    output_type="polars.DataFrame",
)
print(co.schema)
print(co)
```

- [ ] **Step 2: Run it**

```bash
pixi run -e dev python scratch/polars_bio_spike.py
```

- [ ] **Step 3: Record findings as a comment block at the top of `python/genvarloader/_table.py` (created in Task 5)**

Note these specifics, which will guide column references in Tasks 8/9:

- Output column names from `overlap` (default suffixes — likely `_1`/`_2`, e.g. `start_1`, `end_2`).
- Whether `count_overlaps` includes a row for every query (zero-fill) or only matched queries.
- Whether `_q` (and other passthrough columns) survive in the join output and on which side.

If `count_overlaps` does NOT zero-fill, Task 8 must left-join the result back onto the queries frame.

- [ ] **Step 4: Delete the spike script and commit findings**

```bash
rm scratch/polars_bio_spike.py
rmdir scratch 2>/dev/null || true
```

(No commit yet — the findings will be committed inline with Task 5.)

---

## Task 5: Create `Table` skeleton — long-form DataFrame init

**Files:**
- Create: `python/genvarloader/_table.py`
- Test: `tests/test_table.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_table.py`:

```python
import numpy as np
import polars as pl
import pytest

from genvarloader._table import Table


def make_long_df():
    return pl.DataFrame({
        "sample_id": ["s0", "s0", "s1", "s1"],
        "chrom": ["chr1", "chr1", "chr1", "chr2"],
        "start": [10, 100, 20, 0],
        "end":   [20, 110, 30, 5],
        "value": [1.0, 2.0, 3.0, 4.0],
    })


def test_table_init_from_long_df():
    t = Table("signal", make_long_df())
    assert t.name == "signal"
    assert t.samples == ["s0", "s1"]
    assert set(t.contigs) == {"chr1", "chr2"}
    assert t.contigs["chr1"] >= 110
    assert t.contigs["chr2"] >= 5


def test_table_init_missing_canonical_column_raises():
    bad = make_long_df().drop("value")
    with pytest.raises(ValueError, match="value"):
        Table("signal", bad)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_table.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'genvarloader._table'`.

- [ ] **Step 3: Implement minimal `Table`**

Create `python/genvarloader/_table.py`:

```python
"""Tabular interval track source for :func:`gvl.write()`.

Mirrors the :class:`BigWigs` reader API surface so that
:func:`genvarloader._dataset._write._write_track` can dispatch to either.

polars-bio output schema notes (from API spike, see plan Task 4):
- TODO: paste actual column names + zero-fill behavior here from spike.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from ._ragged import RaggedIntervals


CANONICAL_COLS = ("sample_id", "chrom", "start", "end", "value")


class Table:
    """Long-form interval track keyed by ``(sample_id, chrom, start, end, value)``."""

    name: str
    samples: list[str]
    contigs: dict[str, int]

    def __init__(
        self,
        name: str,
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None = None,
    ) -> None:
        self.name = name
        df = self._normalize_input(data, column_map)
        df = df.cast({
            "sample_id": pl.Utf8,
            "chrom": pl.Utf8,
            "start": pl.Int64,
            "end": pl.Int64,
            "value": pl.Float32,
        }).sort("chrom", "sample_id", "start")
        self._df = df
        self.samples = sorted(df["sample_id"].unique().to_list())
        self.contigs = {
            row["chrom"]: int(row["max_end"])
            for row in df.group_by("chrom").agg(pl.col("end").max().alias("max_end")).iter_rows(named=True)
        }

    @staticmethod
    def _normalize_input(
        data: pl.DataFrame | Mapping[str, pl.DataFrame],
        column_map: Mapping[str, str] | None,
    ) -> pl.DataFrame:
        if isinstance(data, pl.DataFrame):
            df = Table._apply_column_map(data, column_map, expect_sample_id=True)
        else:
            # dict[sample_id, df] without sample_id col
            frames: list[pl.DataFrame] = []
            for sid, sub in data.items():
                renamed = Table._apply_column_map(sub, column_map, expect_sample_id=False)
                frames.append(renamed.with_columns(sample_id=pl.lit(sid)))
            if not frames:
                raise ValueError("Empty mapping passed to Table.")
            df = pl.concat(frames, how="vertical_relaxed")
        missing = [c for c in CANONICAL_COLS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required column(s) {missing}. "
                f"Use `column_map` to rename if your columns differ from {CANONICAL_COLS}."
            )
        return df.select(*CANONICAL_COLS)

    @staticmethod
    def _apply_column_map(
        df: pl.DataFrame,
        column_map: Mapping[str, str] | None,
        expect_sample_id: bool,
    ) -> pl.DataFrame:
        if not column_map:
            return df
        # column_map is canonical -> actual; invert to actual -> canonical for rename
        rename = {actual: canonical for canonical, actual in column_map.items() if actual in df.columns}
        if not expect_sample_id:
            rename.pop("sample_id", None)
        return df.rename(rename)
```

Update `python/genvarloader/__init__.py` to export `Table`:

```python
from ._table import Table
```

and add `"Table"` to `__all__`.

- [ ] **Step 4: Run tests**

```bash
pixi run -e dev pytest tests/test_table.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_table.py tests/test_table.py python/genvarloader/__init__.py
git commit -m "feat: add Table skeleton with long-form DataFrame init"
```

---

## Task 6: Add `dict[str, pl.DataFrame]` and `column_map` support

**Files:**
- Modify: `tests/test_table.py`

(`Table` already supports both — this task locks it in with tests.)

- [ ] **Step 1: Append failing tests to `tests/test_table.py`**

```python
def test_table_init_from_dict_of_dfs():
    per_sample = {
        "s0": pl.DataFrame({"chrom": ["chr1"], "start": [10], "end": [20], "value": [1.0]}),
        "s1": pl.DataFrame({"chrom": ["chr2"], "start": [0],  "end": [5],  "value": [2.0]}),
    }
    t = Table("signal", per_sample)
    assert t.samples == ["s0", "s1"]
    assert set(t.contigs) == {"chr1", "chr2"}


def test_table_column_map_renames_long_form():
    df = pl.DataFrame({
        "donor":      ["s0"],
        "chrom":      ["chr1"],
        "chromStart": [10],
        "chromEnd":   [20],
        "signal":     [1.5],
    })
    t = Table(
        "signal",
        df,
        column_map={"sample_id": "donor", "start": "chromStart",
                    "end": "chromEnd", "value": "signal"},
    )
    assert t.samples == ["s0"]
    assert t.contigs["chr1"] == 20


def test_table_column_map_per_sample_dict():
    per_sample = {
        "s0": pl.DataFrame({
            "chrom": ["chr1"], "chromStart": [10], "chromEnd": [20], "signal": [1.5],
        }),
    }
    t = Table(
        "signal",
        per_sample,
        column_map={"start": "chromStart", "end": "chromEnd", "value": "signal"},
    )
    assert t.samples == ["s0"]
```

- [ ] **Step 2: Run tests**

```bash
pixi run -e dev pytest tests/test_table.py -v
```

Expected: all PASS (logic was already implemented in Task 5).

- [ ] **Step 3: Commit**

```bash
git add tests/test_table.py
git commit -m "test: Table accepts dict-of-DFs and column_map"
```

---

## Task 7: Add `Table.from_path` classmethod

**Files:**
- Modify: `python/genvarloader/_table.py`
- Modify: `tests/test_table.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_table.py`:

```python
@pytest.fixture
def long_df():
    return make_long_df()


@pytest.mark.parametrize("ext,reader_attr", [
    (".csv", "csv"),
    (".tsv", "tsv"),
    (".parquet", "parquet"),
    (".arrow", "arrow"),
])
def test_table_from_path_long_form(long_df, tmp_path, ext, reader_attr):
    p = tmp_path / f"data{ext}"
    if ext == ".csv":
        long_df.write_csv(p)
    elif ext == ".tsv":
        long_df.write_csv(p, separator="\t")
    elif ext == ".parquet":
        long_df.write_parquet(p)
    elif ext == ".arrow":
        long_df.write_ipc(p)
    t = Table.from_path("signal", p)
    assert t.samples == ["s0", "s1"]


def test_table_from_path_per_sample_dict(long_df, tmp_path):
    s0 = long_df.filter(pl.col("sample_id") == "s0").drop("sample_id")
    s1 = long_df.filter(pl.col("sample_id") == "s1").drop("sample_id")
    p0 = tmp_path / "s0.parquet"
    p1 = tmp_path / "s1.parquet"
    s0.write_parquet(p0)
    s1.write_parquet(p1)
    t = Table.from_path("signal", {"s0": p0, "s1": p1})
    assert t.samples == ["s0", "s1"]


def test_table_from_path_unknown_extension(tmp_path):
    p = tmp_path / "data.bogus"
    p.write_text("nope")
    with pytest.raises(ValueError, match="extension"):
        Table.from_path("signal", p)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pixi run -e dev pytest tests/test_table.py -k from_path -v
```

Expected: FAIL with `AttributeError: type object 'Table' has no attribute 'from_path'`.

- [ ] **Step 3: Implement `from_path`**

In `python/genvarloader/_table.py`, add as a classmethod:

```python
    @classmethod
    def from_path(
        cls,
        name: str,
        path: str | Path | Mapping[str, str | Path],
        column_map: Mapping[str, str] | None = None,
    ) -> "Table":
        if isinstance(path, Mapping):
            data: dict[str, pl.DataFrame] = {
                sid: cls._read_path(Path(p)) for sid, p in path.items()
            }
            return cls(name, data, column_map)
        return cls(name, cls._read_path(Path(path)), column_map)

    @staticmethod
    def _read_path(p: Path) -> pl.DataFrame:
        suf = p.suffix.lower()
        if suf == ".csv":
            return pl.read_csv(p)
        if suf in (".tsv", ".txt"):
            return pl.read_csv(p, separator="\t")
        if suf == ".parquet":
            return pl.read_parquet(p)
        if suf in (".arrow", ".ipc"):
            return pl.read_ipc(p)
        raise ValueError(
            f"Unsupported file extension {suf!r}. "
            "Expected one of .csv, .tsv, .txt, .parquet, .arrow, .ipc."
        )
```

- [ ] **Step 4: Run tests**

```bash
pixi run -e dev pytest tests/test_table.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_table.py tests/test_table.py
git commit -m "feat: add Table.from_path for csv/tsv/parquet/arrow files"
```

---

## Task 8: Implement `Table.count_intervals`

**Files:**
- Modify: `python/genvarloader/_table.py`
- Modify: `tests/test_table.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_table.py`:

```python
def _brute_count(df: pl.DataFrame, contig: str, starts, ends, samples):
    """Reference implementation: O(n*m) overlap count."""
    out = np.zeros((len(starts), len(samples)), dtype=np.int32)
    for si, s in enumerate(samples):
        sub = df.filter((pl.col("sample_id") == s) & (pl.col("chrom") == contig))
        ts = sub["start"].to_numpy()
        te = sub["end"].to_numpy()
        for ri, (rs, re_) in enumerate(zip(starts, ends)):
            out[ri, si] = int(((ts < re_) & (te > rs)).sum())
    return out


def test_table_count_intervals_matches_brute_force():
    df = pl.DataFrame({
        "sample_id": ["s0", "s0", "s0", "s1", "s1"],
        "chrom":     ["chr1", "chr1", "chr1", "chr1", "chr1"],
        "start":     [0, 50, 200, 10, 60],
        "end":       [10, 60, 210, 20, 70],
        "value":     [1.0, 2.0, 3.0, 4.0, 5.0],
    })
    t = Table("signal", df)
    starts = np.array([0, 55, 100, 200], dtype=np.int32)
    ends   = np.array([15, 65, 150, 205], dtype=np.int32)
    counts = t.count_intervals("chr1", starts, ends, sample=["s0", "s1"])
    expected = _brute_count(df, "chr1", starts, ends, ["s0", "s1"])
    assert counts.dtype == np.int32
    assert counts.shape == (4, 2)
    np.testing.assert_array_equal(counts, expected)


def test_table_count_intervals_unknown_contig_returns_zeros():
    t = Table("signal", make_long_df())
    counts = t.count_intervals("chrX", np.array([0]), np.array([10]), sample=["s0"])
    np.testing.assert_array_equal(counts, np.zeros((1, 1), dtype=np.int32))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_table.py -k count_intervals -v
```

Expected: FAIL with `AttributeError: 'Table' object has no attribute 'count_intervals'`.

- [ ] **Step 3: Implement `count_intervals`**

Append to `python/genvarloader/_table.py` (inside the class):

```python
    def count_intervals(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> NDArray[np.int32]:
        import polars_bio as pb

        samples = self._resolve_samples(sample)
        starts_arr = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends_arr = np.atleast_1d(np.asarray(ends, dtype=np.int64))
        n_regions = len(starts_arr)

        if contig not in self.contigs:
            return np.zeros((n_regions, len(samples)), dtype=np.int32)

        subset = self._df.filter(
            (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
        )
        queries = self._build_queries(contig, starts_arr, ends_arr, samples)

        if subset.height == 0:
            return np.zeros((n_regions, len(samples)), dtype=np.int32)

        counts_df = pb.count_overlaps(
            queries,
            subset,
            cols1=["chrom", "start", "end"],
            cols2=["chrom", "start", "end"],
            on_cols=["sample_id"],
            output_type="polars.DataFrame",
        )
        # Left-join back onto queries to ensure zero-fill for unmatched cells.
        # The exact `count` column name in polars_bio output is captured in the
        # spike notes; if it differs, adjust the alias below.
        count_col = "count" if "count" in counts_df.columns else counts_df.columns[-1]
        merged = queries.join(
            counts_df.select("_q", "sample_id", pl.col(count_col).alias("count")),
            on=["_q", "sample_id"],
            how="left",
        ).with_columns(pl.col("count").fill_null(0))

        sample_to_idx = {s: i for i, s in enumerate(samples)}
        out = np.zeros((n_regions, len(samples)), dtype=np.int32)
        for q, sid, c in merged.select("_q", "sample_id", "count").iter_rows():
            out[int(q), sample_to_idx[sid]] = int(c)
        return out

    def _resolve_samples(self, sample) -> list[str]:
        if sample is None:
            return list(self.samples)
        if isinstance(sample, str):
            samples = [sample]
        else:
            samples = list(sample)
        if missing := set(samples) - set(self.samples):
            raise ValueError(f"Sample(s) {missing} not found in Table.")
        return samples

    def _build_queries(
        self,
        contig: str,
        starts: NDArray[np.int64],
        ends: NDArray[np.int64],
        samples: list[str],
    ) -> pl.DataFrame:
        n = len(starts)
        return (
            pl.DataFrame({
                "_q": np.arange(n, dtype=np.int64),
                "chrom": np.repeat(np.array([contig], dtype=object), n),
                "start": starts,
                "end": ends,
            })
            .join(pl.DataFrame({"sample_id": samples}), how="cross")
        )
```

- [ ] **Step 4: Run tests**

```bash
pixi run -e dev pytest tests/test_table.py -k count_intervals -v
```

Expected: PASS. If `count_overlaps`'s column name differs from `"count"`, the spike notes from Task 4 indicate the correct name — replace the literal in the implementation.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_table.py tests/test_table.py
git commit -m "feat: Table.count_intervals via polars_bio.count_overlaps"
```

---

## Task 9: Implement `Table._intervals_from_offsets`

**Files:**
- Modify: `python/genvarloader/_table.py`
- Modify: `tests/test_table.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_table.py`:

```python
from genvarloader._utils import lengths_to_offsets


def test_table_intervals_from_offsets_roundtrip():
    df = pl.DataFrame({
        "sample_id": ["s0", "s0", "s1"],
        "chrom":     ["chr1", "chr1", "chr1"],
        "start":     [0, 50, 10],
        "end":       [10, 60, 20],
        "value":     [1.5, 2.5, 3.5],
    })
    t = Table("signal", df)
    starts = np.array([0, 40], dtype=np.int32)
    ends = np.array([15, 70], dtype=np.int32)
    samples = ["s0", "s1"]

    counts = t.count_intervals("chr1", starts, ends, sample=samples)
    offsets = lengths_to_offsets(counts.ravel())
    intervals = t._intervals_from_offsets("chr1", starts, ends, offsets, sample=samples)

    # shape: (regions=2, samples=2, ragged)
    assert intervals.starts.data.dtype == np.int32
    assert intervals.values.data.dtype == np.float32
    # cell (region=0, sample=s0): one interval [0, 10) value 1.5
    flat_start = intervals.starts.data
    flat_end   = intervals.ends.data
    flat_val   = intervals.values.data
    assert flat_start[0] == 0 and flat_end[0] == 10 and flat_val[0] == np.float32(1.5)
    # total interval count == sum of counts
    assert len(flat_start) == int(counts.sum())
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pixi run -e dev pytest tests/test_table.py -k intervals_from_offsets -v
```

Expected: FAIL with `AttributeError: 'Table' object has no attribute '_intervals_from_offsets'`.

- [ ] **Step 3: Implement `_intervals_from_offsets`**

Append to `python/genvarloader/_table.py` (inside the class):

```python
    def _intervals_from_offsets(
        self,
        contig: str,
        starts: ArrayLike,
        ends: ArrayLike,
        offsets: NDArray[np.int64],
        sample: str | list[str] | None = None,
        **kwargs,
    ) -> "RaggedIntervals":
        import polars_bio as pb
        from seqpro.rag import Ragged

        from ._ragged import RaggedIntervals

        samples = self._resolve_samples(sample)
        starts_arr = np.atleast_1d(np.asarray(starts, dtype=np.int64))
        ends_arr = np.atleast_1d(np.asarray(ends, dtype=np.int64))
        n_regions = len(starts_arr)
        shape = (n_regions, len(samples), None)

        total = int(offsets[-1])
        flat_starts = np.empty(total, dtype=np.int32)
        flat_ends = np.empty(total, dtype=np.int32)
        flat_values = np.empty(total, dtype=np.float32)

        if contig in self.contigs and total > 0:
            subset = self._df.filter(
                (pl.col("chrom") == contig) & pl.col("sample_id").is_in(samples)
            )
            if subset.height > 0:
                queries = self._build_queries(contig, starts_arr, ends_arr, samples)
                joined = pb.overlap(
                    queries,
                    subset,
                    cols1=["chrom", "start", "end"],
                    cols2=["chrom", "start", "end"],
                    on_cols=["sample_id"],
                    output_type="polars.DataFrame",
                )
                # Right-side columns get a suffix; spike notes from Task 4
                # confirm the exact suffix. Below assumes the table's start/end/value
                # appear as start_2/end_2/value (or value_2). Adjust per spike findings.
                start_col = "start_2" if "start_2" in joined.columns else "start_right"
                end_col   = "end_2"   if "end_2"   in joined.columns else "end_right"
                value_col = "value"   if "value"   in joined.columns else (
                    "value_2" if "value_2" in joined.columns else "value_right"
                )
                joined = joined.sort("_q", "sample_id", start_col)
                # Group rows into per-(region, sample) cells matching offsets layout.
                sample_to_idx = {s: i for i, s in enumerate(samples)}
                cell_idx_col = (
                    pl.col("_q") * len(samples)
                    + pl.col("sample_id").replace_strict(sample_to_idx, return_dtype=pl.Int64)
                )
                joined = joined.with_columns(_cell=cell_idx_col).sort("_cell", start_col)

                cells = joined["_cell"].to_numpy()
                # Compute write positions from offsets in the flat layout.
                # offsets is over flattened (region, sample) cells; row i of the
                # joined frame goes to position offsets[cells[i]] + intra_cell_index.
                # Compute intra-cell running index:
                _, first_idx = np.unique(cells, return_index=True)
                pos_in_cell = np.arange(len(cells)) - np.repeat(first_idx, np.diff(np.r_[first_idx, len(cells)]))
                write_pos = offsets[cells] + pos_in_cell

                flat_starts[write_pos] = joined[start_col].to_numpy().astype(np.int32, copy=False)
                flat_ends[write_pos]   = joined[end_col].to_numpy().astype(np.int32, copy=False)
                flat_values[write_pos] = joined[value_col].to_numpy().astype(np.float32, copy=False)

        return RaggedIntervals(
            Ragged.from_offsets(flat_starts, shape, offsets),
            Ragged.from_offsets(flat_ends, shape, offsets),
            Ragged.from_offsets(flat_values, shape, offsets),
        )
```

- [ ] **Step 4: Run test**

```bash
pixi run -e dev pytest tests/test_table.py -k intervals_from_offsets -v
```

Expected: PASS. If column-name guesses are wrong, replace the candidates per Task 4 spike notes.

- [ ] **Step 5: Run the full Table test module to confirm no regressions**

```bash
pixi run -e dev pytest tests/test_table.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_table.py tests/test_table.py
git commit -m "feat: Table._intervals_from_offsets via polars_bio.overlap"
```

---

## Task 10: Rename `_write_bigwigs` → `_write_track`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py`

The body is unchanged — it already only calls `track.name`, `track.contigs`, `track.samples`, `track.count_intervals`, `track._intervals_from_offsets`.

- [ ] **Step 1: Rename the function and parameter**

In `python/genvarloader/_dataset/_write.py` line 580, change:

```python
def _write_bigwigs(
    path: Path,
    bed: pl.DataFrame,
    bigwigs: BigWigs,
    samples: list[str] | None,
    max_mem: int,
):
```

to:

```python
def _write_track(
    path: Path,
    bed: pl.DataFrame,
    track: "IntervalTrack",
    samples: list[str] | None,
    max_mem: int,
):
```

Replace every reference to `bigwigs` inside this function body with `track`. The function calls `bigwigs.samples`, `bigwigs.contigs`, `bigwigs.count_intervals`, `bigwigs._intervals_from_offsets`, `bigwigs.name` — rename all to `track.*`.

Add the import at the top of `_write.py`:

```python
from .._types import IntervalTrack
```

- [ ] **Step 2: Update the single call site**

`_write.py:209` currently reads:

```python
_write_bigwigs(path, gvl_bed, bw, samples, max_mem)
```

Change to:

```python
_write_track(path, gvl_bed, bw, samples, max_mem)
```

(The variable `bw` will be renamed in Task 11 — don't touch it yet.)

- [ ] **Step 3: Verify the existing test suite still passes**

```bash
pixi run -e dev pytest tests/ -x -q -m "not slow"
```

Expected: same tests pass as before. (No behavioral change — pure rename.)

- [ ] **Step 4: Commit**

```bash
git add python/genvarloader/_dataset/_write.py
git commit -m "refactor: rename _write_bigwigs -> _write_track"
```

---

## Task 11: Rename `bigwigs=` parameter → `tracks=` in `write()`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py`

- [ ] **Step 1: Update the `write()` signature and body**

In `python/genvarloader/_dataset/_write.py`:

1. Line 58 — change parameter:
   ```python
   tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
   ```
   Add the `Sequence` import if not already present:
   ```python
   from collections.abc import Sequence
   ```

2. Update the docstring (lines 82-83) to describe `tracks` instead of `bigwigs`:
   ```
   tracks
       An :class:`IntervalTrack` (e.g. :class:`BigWigs`, :class:`Table`) or a
       sequence of them. Each track must have a unique ``name``; the on-disk
       layout writes to ``<path>/intervals/<track.name>/``.
   ```

3. Lines 101-105 — update precondition and normalization:
   ```python
   if variants is None and tracks is None:
       raise ValueError("At least one of `variants` or `tracks` must be provided.")

   if tracks is not None and not isinstance(tracks, Sequence):
       tracks = [tracks]
   elif tracks is not None:
       tracks = list(tracks)

   if tracks is not None:
       names = [t.name for t in tracks]
       if len(set(names)) != len(names):
           raise ValueError(f"Duplicate track names: {names}. Each track must have a unique `name`.")
   ```

   Note: `BigWigs` is not a `Sequence`, but neither is `Table` — the `isinstance(..., Sequence)` check correctly distinguishes a single track from `[track, track]`. Verify by writing a quick interactive check if uncertain.

4. Lines 151-165 — replace the `for bw in bigwigs` loop:
   ```python
   if tracks is not None:
       unavail = []
       for tr in tracks:
           if unavailable_contigs := set(contigs) - {
               normalize_contig_name(c, contigs) for c in tr.contigs
           }:
               unavail.append(unavailable_contigs)
           if available_samples is None:
               available_samples = set(tr.samples)
           else:
               available_samples.intersection_update(tr.samples)
       if unavail:
           logger.warning(
               f"Contigs in queries {set(unavail)} are not found in one or more tracks."
           )
   ```

5. Line 168-170 — update error message:
   ```python
   raise ValueError("No samples available across all variant file(s) and/or tracks.")
   ```

6. Line 174 — update error:
   ```python
   raise ValueError(f"Samples {missing} not found in variants or tracks.")
   ```

7. Lines 206-209 — update write loop:
   ```python
   if tracks is not None:
       logger.info("Writing track intervals.")
       for tr in tracks:
           _write_track(path, gvl_bed, tr, samples, max_mem)
   ```

- [ ] **Step 2: Update the existing in-tree call sites**

```bash
rtk grep -rn "bigwigs=" python/ tests/ docs/source/ 2>/dev/null
```

Expected: zero callers in `python/`. If any call sites remain in `tests/` or `docs/source/`, update them to `tracks=`.

- [ ] **Step 3: Run the full suite to confirm no regressions**

```bash
pixi run -e dev pytest tests/ -x -q -m "not slow"
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add python/genvarloader/_dataset/_write.py
git commit -m "feat: rename write() param bigwigs= -> tracks=, support mixed sequences"
```

---

## Task 12: End-to-end test — `tracks=Table(...)` round-trip

**Files:**
- Create: `tests/dataset/test_write_tracks.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_write_tracks.py`:

```python
from pathlib import Path

import genvarloader as gvl
import numpy as np
import polars as pl
import seqpro as sp
from genvarloader._table import Table

ddir = Path(__file__).parents[1] / "data"


def _make_bed(tmp_path: Path) -> pl.DataFrame:
    bed = pl.DataFrame({
        "chrom":      ["chr1", "chr1"],
        "chromStart": [0, 100],
        "chromEnd":   [50, 200],
    })
    return bed


def _make_table_df() -> pl.DataFrame:
    return pl.DataFrame({
        "sample_id": ["s0", "s0", "s1", "s1"],
        "chrom":     ["chr1", "chr1", "chr1", "chr1"],
        "start":     [10, 110, 5, 150],
        "end":       [20, 130, 15, 160],
        "value":     [1.0, 2.0, 3.0, 4.0],
    })


def test_write_with_table_only_roundtrip(tmp_path):
    bed = _make_bed(tmp_path)
    table = Table("signal", _make_table_df())

    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, tracks=table)

    # Sanity: the dataset directory has the expected per-track folder.
    assert (out / "intervals" / "signal" / "intervals.npy").exists()
    assert (out / "intervals" / "signal" / "offsets.npy").exists()

    # Read intervals back and confirm values round-trip.
    INTERVAL_DTYPE = np.dtype(
        [("start", np.int32), ("end", np.int32), ("value", np.float32)],
        align=True,
    )
    arr = np.memmap(out / "intervals" / "signal" / "intervals.npy", dtype=INTERVAL_DTYPE, mode="r")
    # Both samples + both regions should produce 4 intervals total.
    assert arr.shape[0] == 4
    values = sorted(float(v) for v in arr["value"])
    assert values == [1.0, 2.0, 3.0, 4.0]
```

- [ ] **Step 2: Run the test**

```bash
pixi run -e dev pytest tests/dataset/test_write_tracks.py::test_write_with_table_only_roundtrip -v
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_write_tracks.py
git commit -m "test: gvl.write(tracks=Table(...)) round-trip"
```

---

## Task 13: End-to-end test — mixed `tracks=[BigWigs, Table]`

**Files:**
- Modify: `tests/dataset/test_write_tracks.py`

- [ ] **Step 1: Append the failing test**

Append to `tests/dataset/test_write_tracks.py`:

```python
def test_write_with_mixed_bigwigs_and_table(tmp_path):
    bed = pl.DataFrame({
        "chrom":      ["chr1"],
        "chromStart": [0],
        "chromEnd":   [200],
    })
    bw_dir = ddir / "bigwig"
    bw = gvl.BigWigs("bw_signal", {
        "sample_0": str(bw_dir / "sample_0.bw"),
        "sample_1": str(bw_dir / "sample_1.bw"),
    })
    # Table sample IDs match the BigWigs sample IDs so the intersection is non-empty.
    table = Table("tab_signal", pl.DataFrame({
        "sample_id": ["sample_0", "sample_1"],
        "chrom":     ["chr1", "chr1"],
        "start":     [0, 50],
        "end":       [10, 60],
        "value":     [9.0, 8.0],
    }))

    out = tmp_path / "mixed.gvl"
    gvl.write(path=out, bed=bed, tracks=[bw, table])

    assert (out / "intervals" / "bw_signal" / "intervals.npy").exists()
    assert (out / "intervals" / "tab_signal" / "intervals.npy").exists()


def test_write_duplicate_track_names_rejected(tmp_path):
    import pytest
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [100]})
    t1 = Table("dup", pl.DataFrame({
        "sample_id": ["s0"], "chrom": ["chr1"], "start": [0], "end": [10], "value": [1.0],
    }))
    t2 = Table("dup", pl.DataFrame({
        "sample_id": ["s0"], "chrom": ["chr1"], "start": [50], "end": [60], "value": [2.0],
    }))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        gvl.write(path=tmp_path / "x.gvl", bed=bed, tracks=[t1, t2])
```

- [ ] **Step 2: Run tests**

```bash
pixi run -e dev pytest tests/dataset/test_write_tracks.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_write_tracks.py
git commit -m "test: mixed BigWigs+Table tracks and duplicate-name rejection"
```

---

## Task 14: Final verification

- [ ] **Step 1: Run full test suite (excluding slow)**

```bash
pixi run -e dev pytest tests/ -x -q -m "not slow"
```

Expected: all PASS.

- [ ] **Step 2: Run typecheck**

```bash
pixi run -e dev basedpyright python/
```

Expected: no new errors introduced by this change. Pre-existing errors are out of scope.

- [ ] **Step 3: Run lint**

```bash
pixi run -e dev ruff check python/
```

Expected: clean.

- [ ] **Step 4: Confirm public re-exports**

```bash
pixi run -e dev python -c "import genvarloader as gvl; print(gvl.Table); print(gvl.BigWigs); print(gvl.write)"
```

Expected: three classes/functions print without error.

- [ ] **Step 5: No commit needed if all green.** If any fixups were required, commit them under a single `chore:` commit.

---

## Self-Review

**Spec coverage check:**
- ✅ `Table` public API — Tasks 5, 6, 7 (init forms + column_map + from_path).
- ✅ `IntervalTrack` Protocol — Task 1, regression-tested in Task 2.
- ✅ `Table.count_intervals` via polars-bio — Task 8.
- ✅ `Table._intervals_from_offsets` via polars-bio — Task 9.
- ✅ `polars-bio` dependency — Task 3.
- ✅ Spike on polars-bio output schema — Task 4.
- ✅ `_write_bigwigs` → `_write_track` rename — Task 10.
- ✅ `bigwigs=` → `tracks=` rename + mixed sequence support + duplicate-name validation — Task 11.
- ✅ End-to-end Table-only test — Task 12.
- ✅ End-to-end mixed test + duplicate-name test — Task 13.
- ✅ Public export of `Table` — Task 5 step 3.
- ✅ Final regression sweep + lint/typecheck — Task 14.

**Out-of-scope items (consistent with spec):** reader-side awareness of Table (none needed); streaming Table input; `bigwigs=` back-compat shim.

**Type/name consistency:** `_write_track(track: IntervalTrack)` defined in Task 10 matches the call site in Task 11. `Table.name`, `Table.samples`, `Table.contigs` types match the Protocol in Task 1.
