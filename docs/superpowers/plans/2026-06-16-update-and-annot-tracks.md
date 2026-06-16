# `gvl.update`, `annot_tracks` in `write`, and parallel writing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a top-level `gvl.update()` for adding tracks to an existing dataset on disk, give `gvl.write()` an `annot_tracks` parameter, parallelize `write` over variants/tracks/annot_tracks, and re-implement annotation extraction on polars-bio (DRY with `gvl.Table`).

**Architecture:** Both `write` and `update` live in `_dataset/_write.py` and share leaf interval writers (`_write_track` for per-sample, `_write_annot_track` for sample-less). `write` builds the whole dataset in one `atomic_dir` temp dir and publishes once; `update` publishes each track subdir atomically into a live dataset via `atomic_dir`. Parallelism is process-based (joblib loky); because variants run first (serially) in `write`, the only parallel jobs are track/annot writers whose closures are picklable (no genoray readers).

**Tech Stack:** Python, polars, polars-bio (optional `[table]` extra), joblib (loky backend), numpy memmaps, seqpro `Ragged`, pyBigWig via the Rust backend.

## Global Constraints

- **Conventional commits** (commitizen). Use `feat:`/`refactor:`/`test:`/`docs:` prefixes.
- All dev commands run under pixi: `pixi run -e dev <cmd>`. Run a single test with
  `pixi run -e dev pytest <path>::<name> -v`.
- **E501 (line length) is ignored** by ruff; do not reflow for line length.
- **polars-bio is gated**: any code path that imports polars-bio must go through
  `genvarloader._table._import_polars_bio` (raises the `pip install genvarloader[table]`
  error when missing) and emit `genvarloader._table.ExperimentalWarning`. Tests touching
  polars-bio are skipped unless `GVL_TEST_EXPERIMENTAL=1` (see `tests/unit/test_table.py`).
- **Public API change**: removing `Dataset.write_annot_tracks` and adding `gvl.update`
  requires updating `skills/genvarloader/SKILL.md` and `docs/source/changelog.md`
  (per `CLAUDE.md`).
- **Atomicity**: never write track data directly into a live dataset dir; build into an
  `atomic_dir` temp and let it `os.replace` into place. The temp/aside siblings are named
  `<name>.tmp.<pid>-<hex>`, `<name>.old.<hex>`, and a `<name>.lock` file lives in the same
  parent.
- annot source column convention is **BED-like**: `chrom`, `chromStart`, `chromEnd`,
  `score` (matches the current `write_annot_tracks` / `_annot_to_intervals` contract).
  Paths are read with `seqpro.bed.read`.

## File Structure

- `python/genvarloader/_dataset/_write.py` — `write` (+ `annot_tracks` + parallel
  orchestration), new `update`, new leaf writers `_write_annot_track` /
  `_write_ragged_intervals`, refactored `_write_track(out_dir, ...)`, new `_run_jobs`
  helper, new `_annot_intervals` dispatcher.
- `python/genvarloader/_table.py` — new module-level `annot_overlap(regions, annot)`
  helper (polars-bio overlap → sample-less `RaggedIntervals`), reusing
  `_import_polars_bio` + `ExperimentalWarning`.
- `python/genvarloader/_dataset/_impl.py` — remove `Dataset.write_annot_tracks`; delete
  `_annot_to_intervals`.
- `python/genvarloader/_dataset/_tracks.py` — make `Tracks.from_path` ignore
  non-directories and `.tmp.`/`.old.`/`.lock` siblings.
- `python/genvarloader/__init__.py` — export `update`.
- `tests/unit/test_write_annot.py` (new) — annot extraction unit tests.
- `tests/integration/tracks/test_update.py` (new) — `gvl.update` integration tests.
- `tests/integration/tracks/test_annot_tracks.py` — migrate to `gvl.update`.
- `tests/integration/test_write_parallel.py` (new) — parallel-write equivalence.
- `skills/genvarloader/SKILL.md`, `docs/source/changelog.md` — docs.

---

### Task 1: Refactor `_write_track` to take an explicit output dir + extract `_write_ragged_intervals`

Decouples the leaf writers from "compute `path/intervals/<name>` internally" so both
`write` (into a temp dataset) and `update` (into an `atomic_dir` temp) can drive them.

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`_write_track` at ~902; its caller in
  `write` at ~286-289)
- Test: existing `pixi run -e dev pytest tests/integration -k track` must still pass.

**Interfaces:**
- Produces: `_write_track(out_dir: Path, bed: pl.DataFrame, track: IntervalTrack, samples: list[str] | None, max_mem: int) -> None` — writes `intervals.npy` + `offsets.npy` directly into `out_dir`.
- Produces: `_write_ragged_intervals(out_dir: Path, itvs: RaggedIntervals) -> None`.

- [ ] **Step 1: Add `_write_ragged_intervals` helper**

Add to `_write.py` (near `_write_track`):

```python
def _write_ragged_intervals(out_dir: Path, itvs: "RaggedIntervals") -> None:
    """Write a RaggedIntervals (values/starts/ends share offsets) to out_dir as
    intervals.npy + offsets.npy. Single-chunk writer used for annotation tracks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out = np.memmap(
        out_dir / "intervals.npy",
        dtype=INTERVAL_DTYPE,
        mode="w+",
        shape=itvs.values.data.shape,
    )
    out["start"] = itvs.starts.data
    out["end"] = itvs.ends.data
    out["value"] = itvs.values.data
    out.flush()

    offsets = itvs.values.offsets
    out = np.memmap(
        out_dir / "offsets.npy",
        dtype=offsets.dtype,
        mode="w+",
        shape=len(offsets),
    )
    out[:] = offsets
    out.flush()
```

(`RaggedIntervals` is imported lazily inside functions today; add
`from .._ragged import RaggedIntervals` under the existing `TYPE_CHECKING` block for the
annotation only, and import the concrete class inside functions that build one.)

- [ ] **Step 2: Change `_write_track` signature to accept `out_dir`**

In `_write_track`, delete the internal `out_dir = path / "intervals" / track.name` line
(~970-971) and rename the parameter `path` → `out_dir`. Add `out_dir.mkdir(parents=True,
exist_ok=True)` where the old `mkdir` was. Leave the rest of the body unchanged.

```python
def _write_track(
    out_dir: Path,
    bed: pl.DataFrame,
    track: "IntervalTrack",
    samples: list[str] | None,
    max_mem: int,
):
    ...
    # (was: out_dir = path / "intervals" / track.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    ...
```

- [ ] **Step 3: Update the caller in `write`**

In `write` (~286-289) change:

```python
            if tracks is not None:
                logger.info("Writing track intervals.")
                for tr in tracks:
                    _write_track(path / "intervals" / tr.name, gvl_bed, tr, samples, max_mem)
```

- [ ] **Step 4: Run the existing track tests**

Run: `pixi run -e dev pytest tests/integration -k "track" -v`
Expected: PASS (pure refactor; on-disk output unchanged).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "refactor: _write_track takes explicit out_dir; add _write_ragged_intervals"
```

---

### Task 2: polars-bio annotation extraction (`annot_overlap`) + `_annot_intervals` dispatcher

Re-implements sample-less annotation extraction on polars-bio (table/DataFrame sources)
and the Rust BigWigs backend (bigwig sources), replacing the pyranges `_annot_to_intervals`.

**Files:**
- Modify: `python/genvarloader/_table.py` (add `annot_overlap`)
- Modify: `python/genvarloader/_dataset/_write.py` (add `_annot_intervals`,
  `_write_annot_track`)
- Test: `tests/unit/test_write_annot.py` (new)

**Interfaces:**
- Consumes: `genvarloader._dataset._impl._annot_to_intervals(regions, annot)` (the existing pyranges impl — still present until Task 6; used here only as the regression oracle in the test).
- Produces: `genvarloader._table.annot_overlap(regions: pl.DataFrame, annot: pl.DataFrame) -> RaggedIntervals` — polars-bio overlap, sample-less `(n_regions, None)`; emits `ExperimentalWarning`, requires `[table]`.
- Produces: `_annot_intervals(regions: pl.DataFrame, source, max_mem: int) -> RaggedIntervals` in `_write.py` — dispatches on source type.
- Produces: `_write_annot_track(out_dir: Path, regions: pl.DataFrame, source, max_mem: int) -> None`.

- [ ] **Step 1: Write the failing test (polars-bio path, env-gated)**

Create `tests/unit/test_write_annot.py`:

```python
import os

import numpy as np
import polars as pl
import pytest

# polars-bio is gated exactly like gvl.Table (see tests/unit/test_table.py).
if not os.environ.get("GVL_TEST_EXPERIMENTAL"):
    pytest.skip(
        "annot polars-bio path is experimental; set GVL_TEST_EXPERIMENTAL=1 to run.",
        allow_module_level=True,
    )

pytestmark = pytest.mark.filterwarnings(
    "ignore::genvarloader._table.ExperimentalWarning"
)


def _regions():
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr2"],
            "chromStart": [0, 50, 0],
            "chromEnd": [100, 150, 100],
        }
    )


def _annot():
    return pl.DataFrame(
        {
            "chrom": ["chr1", "chr1", "chr1", "chr2"],
            "chromStart": [10, 60, 90, 5],
            "chromEnd": [20, 70, 95, 15],
            "score": [1.0, 2.0, 3.0, 4.0],
        }
    )


def test_annot_overlap_matches_pyranges_oracle():
    from genvarloader._dataset._impl import _annot_to_intervals  # pyranges oracle
    from genvarloader._table import annot_overlap

    regions, annot = _regions(), _annot()
    got = annot_overlap(regions, annot)
    want = _annot_to_intervals(regions, annot)

    # same per-region counts and the same multiset of intervals per region
    np.testing.assert_array_equal(got.values.offsets, want.values.offsets)
    for r in range(regions.height):
        gs, ge, gv = got.starts[r], got.ends[r], got.values[r]
        ws, we, wv = want.starts[r], want.ends[r], want.values[r]
        go = np.lexsort((gv, ge, gs))
        wo = np.lexsort((wv, we, ws))
        np.testing.assert_array_equal(np.asarray(gs)[go], np.asarray(ws)[wo])
        np.testing.assert_array_equal(np.asarray(ge)[go], np.asarray(we)[wo])
        np.testing.assert_allclose(np.asarray(gv)[go], np.asarray(wv)[wo])
```

- [ ] **Step 2: Run to verify it fails**

Run: `GVL_TEST_EXPERIMENTAL=1 pixi run -e dev pytest tests/unit/test_write_annot.py -v`
Expected: FAIL with `ImportError: cannot import name 'annot_overlap'`.

- [ ] **Step 3: Implement `annot_overlap` in `_table.py`**

Add to `python/genvarloader/_table.py` (module level, after `Table`):

```python
def annot_overlap(regions: pl.DataFrame, annot: pl.DataFrame) -> "RaggedIntervals":
    """Sample-less interval overlap of `regions` (chrom/chromStart/chromEnd) against a
    BED-like `annot` (chrom/chromStart/chromEnd/score), via polars-bio. Returns a
    RaggedIntervals of shape (n_regions, None) ordered by (region, start). Experimental:
    requires the `table` extra and emits ExperimentalWarning."""
    import numpy as np
    from seqpro.rag import Ragged

    from ._ragged import RaggedIntervals
    from ._utils import lengths_to_offsets, normalize_contig_name

    warnings.warn(_TABLE_EXPERIMENTAL_MSG, ExperimentalWarning, stacklevel=2)
    pb = _import_polars_bio()
    pb.set_option("datafusion.bio.coordinate_system_check", "false")
    pb.set_option("datafusion.bio.coordinate_system_zero_based", True)

    # normalize annot contig names to the region naming (e.g. "1" -> "chr1")
    reg_c = regions["chrom"].unique().to_list()
    renamer = {
        c: nc
        for c in annot["chrom"].unique().to_list()
        if (nc := normalize_contig_name(c, reg_c)) is not None
    }
    annot = annot.with_columns(chrom=pl.col("chrom").replace(renamer))

    n_regions = regions.height
    q = regions.with_row_index("_q").select(
        "chrom",
        pl.col("chromStart").alias("start"),
        pl.col("chromEnd").alias("end"),
        "_q",
    )
    db = annot.select(
        "chrom",
        pl.col("chromStart").alias("start"),
        pl.col("chromEnd").alias("end"),
        "score",
    )
    joined = pb.overlap(
        q,
        db,
        cols1=["chrom", "start", "end"],
        cols2=["chrom", "start", "end"],
        output_type="polars.DataFrame",
    )

    shape = (n_regions, None)
    if joined.height == 0:
        offsets = lengths_to_offsets(np.zeros(n_regions, np.int32))
        empty_i = np.empty(0, np.int32)
        empty_f = np.empty(0, np.float32)
        return RaggedIntervals(
            Ragged.from_offsets(empty_i, shape, offsets),
            Ragged.from_offsets(empty_i, shape, offsets),
            Ragged.from_offsets(empty_f, shape, offsets),
        )

    # polars-bio suffixes query cols with _1 and database cols with _2 (see Table).
    q_idx = joined["_q_1"].to_numpy()
    j_starts = joined["start_2"].to_numpy()
    order = np.lexsort((j_starts, q_idx))  # primary key = q_idx, then start
    q_idx = q_idx[order]

    counts = np.zeros(n_regions, np.int32)
    uniq, cnt = np.unique(q_idx, return_counts=True)
    counts[uniq] = cnt
    offsets = lengths_to_offsets(counts)

    starts = j_starts[order].astype(np.int32, copy=False)
    ends = joined["end_2"].to_numpy()[order].astype(np.int32, copy=False)
    values = joined["score_2"].to_numpy()[order].astype(np.float32, copy=False)
    return RaggedIntervals(
        Ragged.from_offsets(starts, shape, offsets),
        Ragged.from_offsets(ends, shape, offsets),
        Ragged.from_offsets(values, shape, offsets),
    )
```

Add `from ._ragged import RaggedIntervals` to the `TYPE_CHECKING` block (already present)
so the annotation resolves.

- [ ] **Step 4: Run the polars-bio test**

Run: `GVL_TEST_EXPERIMENTAL=1 pixi run -e dev pytest tests/unit/test_write_annot.py::test_annot_overlap_matches_pyranges_oracle -v`
Expected: PASS. (If polars-bio is not installed, it skips — install with the `table`
extra to actually exercise it.)

- [ ] **Step 5: Add the bigwig-annot test (runs in CI — no polars-bio)**

The module-level skip above gates the whole file. Move the bigwig test into a separate,
ungated file `tests/unit/test_write_annot_bigwig.py`:

```python
from pathlib import Path

import numpy as np

from genvarloader._dataset._write import _annot_intervals


def test_annot_intervals_from_bigwig(tmp_path):
    import polars as pl

    data_dir = Path(__file__).parent.parent / "data" / "bigwig"
    bw = data_dir / "sample_0.bw"
    # a region known to overlap intervals in the fixture bigwig
    regions = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [1000]}
    )
    itvs = _annot_intervals(regions, bw, max_mem=2**30)
    # shape (regions, None), one region
    assert itvs.values.offsets.shape == (2,)
    assert itvs.starts.data.dtype == np.int32
```

(Confirm `tests/data/bigwig/sample_0.bw` exists — it is used by
`tests/test_interval_track.py`. Adjust the relative path / contig / region if the fixture
differs.)

- [ ] **Step 6: Implement `_annot_intervals` + `_write_annot_track` in `_write.py`**

```python
def _annot_intervals(
    regions: pl.DataFrame,
    source: "str | Path | pl.DataFrame | pl.LazyFrame",
    max_mem: int,
) -> "RaggedIntervals":
    """Build a sample-less RaggedIntervals (n_regions, None) from an annotation source.

    - bigwig path -> Rust per-region extraction (BigWigs), squeezed sample-less.
    - table path / DataFrame / LazyFrame (BED-like: chrom, chromStart, chromEnd, score)
      -> polars-bio overlap (experimental, requires the `table` extra).
    """
    from .._ragged import RaggedIntervals

    if isinstance(source, (str, Path)) and Path(source).suffix.lower() in (
        ".bw",
        ".bigwig",
    ):
        return _annot_intervals_from_bigwig(regions, Path(source), max_mem)

    if isinstance(source, pl.LazyFrame):
        annot = source.collect()
    elif isinstance(source, pl.DataFrame):
        annot = source
    else:
        annot = sp.bed.read(str(source))

    from .._table import annot_overlap

    return annot_overlap(regions, annot)


def _annot_intervals_from_bigwig(
    regions: pl.DataFrame, path: Path, max_mem: int
) -> "RaggedIntervals":
    from seqpro.rag import Ragged

    from .._bigwig import BigWigs
    from .._ragged import RaggedIntervals

    # single pseudo-sample; collapse its sample axis to produce a sample-less track
    bw = BigWigs(name="__annot__", paths={"__annot__": str(path)})
    out_starts, out_ends, out_values, lengths = [], [], [], []
    for (contig,), part in regions.partition_by(
        "chrom", as_dict=True, maintain_order=True
    ).items():
        contig = cast(str, contig)
        starts = part["chromStart"].to_numpy()
        ends = part["chromEnd"].to_numpy()
        # (regions, 1)
        itvs = bw.intervals(contig, starts, ends, sample="__annot__")
        for r in range(part.height):
            s = itvs.starts[r, 0]
            out_starts.append(np.asarray(s, dtype=np.int32))
            out_ends.append(np.asarray(itvs.ends[r, 0], dtype=np.int32))
            out_values.append(np.asarray(itvs.values[r, 0], dtype=np.float32))
            lengths.append(len(s))
    flat_starts = (
        np.concatenate(out_starts) if out_starts else np.empty(0, np.int32)
    )
    flat_ends = np.concatenate(out_ends) if out_ends else np.empty(0, np.int32)
    flat_values = (
        np.concatenate(out_values) if out_values else np.empty(0, np.float32)
    )
    offsets = lengths_to_offsets(np.asarray(lengths, np.int32))
    shape = (regions.height, None)
    return RaggedIntervals(
        Ragged.from_offsets(flat_starts, shape, offsets),
        Ragged.from_offsets(flat_ends, shape, offsets),
        Ragged.from_offsets(flat_values, shape, offsets),
    )


def _write_annot_track(
    out_dir: Path,
    regions: pl.DataFrame,
    source: "str | Path | pl.DataFrame | pl.LazyFrame",
    max_mem: int,
) -> None:
    itvs = _annot_intervals(regions, source, max_mem)
    _write_ragged_intervals(out_dir, itvs)
```

> Note: `regions` here is the BED-3 frame in the dataset's on-disk row order, produced by
> `regions_to_bed(np.load(path / "regions.npy"), contigs)` (matches today's
> `write_annot_tracks`, which uses `regions_to_bed(self._full_regions, self.contigs)`).
> The bigwig partition_by preserves on-disk order via `maintain_order=True`, so per-region
> rows align with `regions.npy`.

- [ ] **Step 7: Run both annot test files**

Run: `pixi run -e dev pytest tests/unit/test_write_annot_bigwig.py -v`
Expected: PASS.
Run: `GVL_TEST_EXPERIMENTAL=1 pixi run -e dev pytest tests/unit/test_write_annot.py -v`
Expected: PASS (or skip without the `table` extra).

- [ ] **Step 8: Commit**

```bash
rtk git add python/genvarloader/_table.py python/genvarloader/_dataset/_write.py tests/unit/test_write_annot.py tests/unit/test_write_annot_bigwig.py
rtk git commit -m "feat: polars-bio annotation extraction + bigwig annot source"
```

---

### Task 3: `gvl.write(annot_tracks=...)` (sequential, no parallelism yet)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`write` signature + body)
- Test: `tests/integration/tracks/test_annot_tracks.py` (add a write-based test;
  the existing test still uses the soon-removed `Dataset.write_annot_tracks` — leave it
  for now, it is migrated in Task 6)

**Interfaces:**
- Produces: `write(..., annot_tracks: dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None = None)` writing each into `<path>/annot_intervals/<name>/`.

- [ ] **Step 1: Write the failing test**

Add to `tests/integration/tracks/test_annot_tracks.py`:

```python
def test_write_with_annot_tracks(phased_vcf, ref_fasta, tmp_path):
    import genvarloader as gvl
    import polars as pl

    out = tmp_path / "ds"
    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]}
    )
    annot = bed.with_columns(chromEnd=pl.col("chromStart") + 1, score=pl.lit(1.0))
    gvl.write(out, bed, variants=phased_vcf, annot_tracks={"5ss": annot})
    ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated").with_tracks(
        "5ss", "tracks"
    )
    assert "5ss" in ds.available_tracks
```

(Use whatever phased-VCF / ref fixtures the file already imports; mirror the existing
`dataset` fixture's sources. The `score`/`chromEnd` annot mirrors the existing
`test_annot_tracks` body. If the table source needs the `[table]` extra, gate this test
behind `GVL_TEST_EXPERIMENTAL` like Task 2 — a DataFrame source uses polars-bio.)

> If you want this test to run in CI without the `table` extra, use a **bigwig** annot
> source instead of a DataFrame: `annot_tracks={"sig": tests/data/bigwig/sample_0.bw}`.

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/integration/tracks/test_annot_tracks.py::test_write_with_annot_tracks -v`
Expected: FAIL with `TypeError: write() got an unexpected keyword argument 'annot_tracks'`.

- [ ] **Step 3: Add the parameter and the write step**

In `write` add the parameter after `tracks`:

```python
def write(
    path: str | Path,
    bed: str | Path | pl.DataFrame,
    variants: str | Path | Reader | None = None,
    tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
    annot_tracks: "dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None" = None,
    samples: list[str] | None = None,
    max_jitter: int | None = None,
    overwrite: bool = False,
    max_mem: int | str = "4g",
    extend_to_length: bool = True,
):
```

Update the "at least one input" guard:

```python
        if variants is None and tracks is None and annot_tracks is None:
            raise ValueError(
                "At least one of `variants`, `tracks`, or `annot_tracks` must be provided."
            )
```

After the existing track-writing block (~286-289), add:

```python
            if annot_tracks is not None:
                logger.info("Writing annotation tracks.")
                annot_bed = regions_to_bed(
                    np.load(path / "regions.npy"), contigs
                ).select("chrom", "chromStart", "chromEnd")
                for name, source in annot_tracks.items():
                    _write_annot_track(
                        path / "annot_intervals" / name, annot_bed, source, max_mem
                    )
```

Add the import at the top of `_write.py`:

```python
from ._utils import bed_to_regions, regions_to_bed, splits_sum_le_value
```

(`regions_to_bed` lives in `python/genvarloader/_dataset/_utils.py`.) `_write_regions`
must have already written `regions.npy` — it runs at ~284, before this block, so the
read is valid. The annot extraction needs the **finalized** region ends (post-variant),
which `regions.npy` reflects.

- [ ] **Step 4: Document the parameter**

Add to the `write` docstring, after the `tracks` entry:

```
    annot_tracks
        Sample-independent annotation tracks, as a mapping of track name to source.
        Each source is a path to an interval table, a path to a bigWig, or a polars
        DataFrame/LazyFrame interpreted as a BED-like interval table (columns ``chrom``,
        ``chromStart``, ``chromEnd``, ``score``). Table/DataFrame sources use the
        polars-bio overlap backend and require the ``table`` extra
        (``pip install genvarloader[table]``); they emit an ``ExperimentalWarning``.
        bigWig sources do not. Written to ``<path>/annot_intervals/<name>/``.
```

- [ ] **Step 5: Run the test**

Run: `pixi run -e dev pytest tests/integration/tracks/test_annot_tracks.py::test_write_with_annot_tracks -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/integration/tracks/test_annot_tracks.py
rtk git commit -m "feat: gvl.write accepts annot_tracks"
```

---

### Task 4: Parallelize `write` (variants first, then tracks ∥ annot_tracks)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`write` body; new `_run_jobs` helper)
- Test: `tests/integration/test_write_parallel.py` (new)

**Interfaces:**
- Produces: `_run_jobs(jobs: list[Callable[[int], None]], max_mem: int) -> None` — runs each `job(per_job_max_mem)`; ≤1 job runs inline, else joblib loky with `n_jobs=len(jobs)` and `max_mem // len(jobs)` apportioned per job.

- [ ] **Step 1: Write the equivalence test**

Create `tests/integration/test_write_parallel.py`:

```python
import numpy as np
import polars as pl

import genvarloader as gvl


def _open_track(path, name):
    return np.memmap(path / "intervals" / name / "intervals.npy", mode="r")


def test_parallel_write_matches_sequential(phased_vcf, ref_fasta, bigwigs, tmp_path):
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]})

    a = tmp_path / "a"
    b = tmp_path / "b"
    gvl.write(a, bed, variants=phased_vcf, tracks=bigwigs)  # parallel path
    # force the inline path for the oracle by writing categories one at a time
    gvl.write(b, bed, variants=phased_vcf)
    gvl.update(b, tracks=bigwigs)  # update is single-category -> inline

    da = gvl.Dataset.open(a, ref_fasta).with_tracks(bigwigs.name)
    db = gvl.Dataset.open(b, ref_fasta).with_tracks(bigwigs.name)
    ha, ta = da[:]
    hb, tb = db[:]
    assert (np.asarray(ta.data) == np.asarray(tb.data)).all()
```

(Use the file's existing fixtures for `phased_vcf`, `ref_fasta`, and a `BigWigs`
fixture named `bigwigs`. If no `bigwigs` fixture exists, build one from
`tests/data/bigwig/sample_*.bw` whose samples match the VCF samples.)

> This test depends on Task 5 (`gvl.update`). If executing strictly in order, write the
> oracle inline instead: call the internal sequential writers directly, or compare the
> parallel run against a known-good fixture. Prefer running Task 5 first if convenient;
> otherwise stub the oracle with a single-track sequential `gvl.write` and compare track
> bytes only.

- [ ] **Step 2: Run to verify it fails / is red**

Run: `pixi run -e dev pytest tests/integration/test_write_parallel.py -v`
Expected: FAIL (parallel path not yet implemented or `update` missing).

- [ ] **Step 3: Add `_run_jobs` and rewire `write`**

Add the helper to `_write.py`:

```python
from collections.abc import Callable

from joblib import Parallel, delayed


def _run_jobs(jobs: "list[Callable[[int], None]]", max_mem: int) -> None:
    """Run track/annot writer jobs. Each job is called with a per-job max_mem budget.
    0/1 jobs run inline; otherwise jobs run concurrently on the loky backend with the
    budget divided evenly so total peak stays under max_mem."""
    jobs = [j for j in jobs if j is not None]
    if len(jobs) <= 1:
        for j in jobs:
            j(max_mem)
        return
    per = max(max_mem // len(jobs), 1)
    Parallel(n_jobs=len(jobs), backend="loky")(delayed(j)(per) for j in jobs)
```

Replace the sequential track / annot blocks in `write` with job construction. Variants
still run first, serially (they finalize `gvl_bed` and write `regions.npy`):

```python
            # variants already written above; regions.npy is finalized here.
            _write_regions(path, gvl_bed, contigs)

            jobs: list[Callable[[int], None]] = []
            if tracks is not None:
                _tracks = list(tracks)
                _bed = gvl_bed

                def _tracks_job(mm: int, _tracks=_tracks, _bed=_bed):
                    for tr in _tracks:
                        _write_track(path / "intervals" / tr.name, _bed, tr, samples, mm)

                jobs.append(_tracks_job)

            if annot_tracks is not None:
                annot_bed = regions_to_bed(
                    np.load(path / "regions.npy"), contigs
                ).select("chrom", "chromStart", "chromEnd")
                _annots = dict(annot_tracks)

                def _annot_job(mm: int, _annots=_annots, _bed=annot_bed):
                    for name, source in _annots.items():
                        _write_annot_track(
                            path / "annot_intervals" / name, _bed, source, mm
                        )

                jobs.append(_annot_job)

            if jobs:
                logger.info(f"Writing {len(jobs)} track categor(ies).")
                _run_jobs(jobs, max_mem)
```

Remove the now-superseded sequential `if tracks` / `if annot_tracks` blocks added in
earlier tasks. Keep `_write_regions` exactly once.

> Picklability: the job closures capture `path` (Path), `gvl_bed` (polars DataFrame),
> `samples` (list[str]), `BigWigs`/`Table` (both picklable — `BigWigs.readers` is lazy
> `None`; `Table` holds a polars DataFrame), and plain sources. They do **not** capture
> genoray readers (variants run before fan-out), so loky pickling is safe.

- [ ] **Step 4: Run the equivalence test**

Run: `pixi run -e dev pytest tests/integration/test_write_parallel.py -v`
Expected: PASS.

- [ ] **Step 5: Run the broader write/track suite for regressions**

Run: `pixi run -e dev pytest tests/integration -k "write or track" -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/integration/test_write_parallel.py
rtk git commit -m "feat: parallelize gvl.write over track categories (variants first)"
```

---

### Task 5: `gvl.update()` + robust `Tracks.from_path`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (new `update`)
- Modify: `python/genvarloader/_dataset/_tracks.py` (`from_path` robustness)
- Modify: `python/genvarloader/__init__.py` (export)
- Test: `tests/integration/tracks/test_update.py` (new)

**Interfaces:**
- Consumes: `_write_track`, `_write_annot_track`, `_run_jobs` (Tasks 1, 2, 4), `Metadata` (in `_write.py`), `regions_to_bed`, `atomic_dir`.
- Produces: `update(dataset: str | Path | Dataset, tracks=None, annot_tracks=None, *, overwrite=False, max_mem="4g") -> None`, exported as `gvl.update`.

- [ ] **Step 1: Write the failing tests**

Create `tests/integration/tracks/test_update.py`:

```python
import numpy as np
import polars as pl
import pytest

import genvarloader as gvl


def test_update_adds_sample_track(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]})
    gvl.write(out, bed, variants=phased_vcf)
    gvl.update(out, tracks=bigwigs)
    ds = gvl.Dataset.open(out, ref_fasta)
    assert bigwigs.name in ds.available_tracks


def test_update_accepts_dataset_object(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]})
    gvl.write(out, bed, variants=phased_vcf)
    ds = gvl.Dataset.open(out, ref_fasta)
    gvl.update(ds, tracks=bigwigs)  # extracts .path
    assert bigwigs.name in gvl.Dataset.open(out, ref_fasta).available_tracks


def test_update_rejects_extra_or_missing_samples(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]})
    gvl.write(out, bed, variants=phased_vcf, samples=bigwigs.samples[:-1])
    with pytest.raises(ValueError, match="sample"):
        gvl.update(out, tracks=bigwigs)  # bigwigs has an extra sample


def test_update_overwrite(phased_vcf, ref_fasta, bigwigs, tmp_path):
    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]})
    gvl.write(out, bed, variants=phased_vcf)
    gvl.update(out, tracks=bigwigs)
    with pytest.raises(FileExistsError):
        gvl.update(out, tracks=bigwigs)
    gvl.update(out, tracks=bigwigs, overwrite=True)  # ok
    assert not list((out / "intervals").glob("*.tmp.*"))
```

(Use the suite's existing fixtures; `bigwigs` is a `BigWigs` whose sample set equals the
VCF's. For `test_update_rejects_extra_or_missing_samples`, the dataset is written with a
strict subset of samples so the track has an extra one.)

- [ ] **Step 2: Run to verify they fail**

Run: `pixi run -e dev pytest tests/integration/tracks/test_update.py -v`
Expected: FAIL with `AttributeError: module 'genvarloader' has no attribute 'update'`.

- [ ] **Step 3: Implement `update` in `_write.py`**

```python
def update(
    dataset: "str | Path | Dataset",
    tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
    annot_tracks: "dict[str, str | Path | pl.DataFrame | pl.LazyFrame] | None" = None,
    *,
    overwrite: bool = False,
    max_mem: int | str = "4g",
) -> None:
    """Add tracks to an existing on-disk GVL dataset, analogous to :func:`write`.

    Parameters
    ----------
    dataset
        Path to a dataset directory, or an opened :class:`Dataset` (its ``.path`` is used).
        A live dataset can be read while it is being updated; it will not observe the new
        track until reopened.
    tracks
        Per-sample :class:`IntervalTrack` source(s) (:class:`BigWigs`, :class:`Table`),
        written to ``<path>/intervals/<name>/``. The track's sample set must match the
        dataset's exactly (no missing, no extra); samples are reordered to the dataset
        order automatically.
    annot_tracks
        Sample-independent sources, identical to :func:`write`'s ``annot_tracks``, written
        to ``<path>/annot_intervals/<name>/``.
    overwrite
        Replace a track of the same name if present; otherwise adding a duplicate name
        raises ``FileExistsError``.
    max_mem
        Approximate memory budget, divided across concurrently-running categories.
    """
    warnings.simplefilter("ignore", RuntimeWarning)
    try:
        from ._impl import Dataset

        path = Path(dataset.path if isinstance(dataset, Dataset) else dataset)
        if not (path / "metadata.json").exists():
            raise FileNotFoundError(f"{path} is not a GVL dataset (no metadata.json).")

        meta = Metadata.model_validate_json((path / "metadata.json").read_text())
        contigs = meta.contigs
        ds_samples = meta.samples
        max_mem_b = parse_memory(max_mem)

        if tracks is not None and not isinstance(tracks, (list, tuple)):
            tracks = [tracks]
        _tracks = list(tracks) if tracks is not None else []

        if tracks is None and annot_tracks is None:
            raise ValueError("At least one of `tracks` or `annot_tracks` must be provided.")

        # validate strict sample-set agreement for per-sample tracks
        for tr in _tracks:
            if set(tr.samples) != set(ds_samples):
                missing = set(ds_samples) - set(tr.samples)
                extra = set(tr.samples) - set(ds_samples)
                raise ValueError(
                    f"Track {tr.name!r} samples must exactly match the dataset's. "
                    f"missing={missing or '{}'} extra={extra or '{}'}"
                )

        bed = regions_to_bed(np.load(path / "regions.npy"), contigs)
        sample_bed = bed.select("chrom", "chromStart", "chromEnd")
        annot_bed = sample_bed

        jobs: list[Callable[[int], None]] = []

        if _tracks:
            (path / "intervals").mkdir(exist_ok=True)

            def _tracks_job(mm: int, _tracks=_tracks, _bed=sample_bed):
                for tr in _tracks:
                    with atomic_dir(
                        path / "intervals" / tr.name, overwrite=overwrite
                    ) as tmp:
                        _write_track(tmp, _bed, tr, ds_samples, mm)

            jobs.append(_tracks_job)

        if annot_tracks is not None:
            (path / "annot_intervals").mkdir(exist_ok=True)
            _annots = dict(annot_tracks)

            def _annot_job(mm: int, _annots=_annots, _bed=annot_bed):
                for name, source in _annots.items():
                    with atomic_dir(
                        path / "annot_intervals" / name, overwrite=overwrite
                    ) as tmp:
                        _write_annot_track(tmp, _bed, source, mm)

            jobs.append(_annot_job)

        _run_jobs(jobs, max_mem_b)
    finally:
        warnings.simplefilter("default")
```

> `_write_track` is given `ds_samples` (the dataset's stored sample order). It already
> raises on samples missing from the track; the exact-set check above additionally
> rejects extras. Passing `ds_samples` performs the reorder-to-dataset-order.

- [ ] **Step 4: Make `Tracks.from_path` robust to in-flight `atomic_dir` siblings**

In `python/genvarloader/_dataset/_tracks.py` `from_path`, both scan loops currently do
`for p in strack_dir.iterdir(): if len(list(p.iterdir())) == 0: ...`. A concurrent
`update` leaves `<name>.tmp.<pid>-<hex>`, `<name>.old.<hex>` dirs and a `<name>.lock`
file in the same parent; the latter is a file and `p.iterdir()` on it raises. Guard both
loops:

```python
        def _is_track_dir(p: Path) -> bool:
            return (
                p.is_dir()
                and ".tmp." not in p.name
                and ".old." not in p.name
                and not p.name.endswith(".lock")
            )

        available_tracks: list[str] = []
        if strack_dir.exists():
            for p in strack_dir.iterdir():
                if not _is_track_dir(p):
                    continue
                if len(list(p.iterdir())) == 0:
                    p.rmdir()
                else:
                    available_tracks.append(p.name)
            available_tracks.sort()
```

Apply the same `_is_track_dir` guard to the `available_annots` loop. Define
`_is_track_dir` once at the top of `from_path`.

- [ ] **Step 5: Export `update`**

In `python/genvarloader/__init__.py`:

```python
from ._dataset._write import get_splice_bed, update, write
```

and add `"update",` to `__all__` (keep it alphabetized).

- [ ] **Step 6: Run the update tests**

Run: `pixi run -e dev pytest tests/integration/tracks/test_update.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py python/genvarloader/_dataset/_tracks.py python/genvarloader/__init__.py tests/integration/tracks/test_update.py
rtk git commit -m "feat: add gvl.update for atomic post-hoc track addition"
```

---

### Task 6: Remove `Dataset.write_annot_tracks`, migrate its test, delete `_annot_to_intervals`

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` (delete method + `_annot_to_intervals`)
- Modify: `tests/integration/tracks/test_annot_tracks.py` (migrate to `gvl.update`)

**Interfaces:**
- Removes: `Dataset.write_annot_tracks`, `genvarloader._dataset._impl._annot_to_intervals`.
- The Task 2 regression test imported `_annot_to_intervals` as an oracle; that test has
  served its purpose. Update it to assert against a hand-computed expectation instead (see
  Step 3) so deleting the oracle does not break the suite.

- [ ] **Step 1: Migrate the existing `test_annot_tracks` test**

Replace the body of `test_annot_tracks` in `tests/integration/tracks/test_annot_tracks.py`
so it writes the annot via `gvl.update` rather than `dataset.write_annot_tracks`:

```python
def test_annot_tracks(phased_vcf, ref_fasta, tmp_path):
    import genvarloader as gvl
    import polars as pl

    out = tmp_path / "ds"
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]})
    gvl.write(out, bed, variants=phased_vcf)
    ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated")
    annots = ds.regions.with_columns(
        chromEnd=pl.col("chromStart") + 1, score=pl.lit(1.0)
    )
    gvl.update(out, annot_tracks={"5ss": annots})
    annot_ds = gvl.Dataset.open(out, ref_fasta).with_seqs("annotated").with_tracks(
        "5ss", "tracks"
    )
    haps, tracks = annot_ds[:]
    mask = haps.ref_coords == ak.Array(
        annot_ds.regions["chromStart"].to_numpy()[:, None, None]
    )
    assert ak.all(tracks[:, :, 0][mask] == 1)
```

> A DataFrame annot source uses polars-bio, so gate this test with the
> `GVL_TEST_EXPERIMENTAL` module skip (mirror `tests/unit/test_table.py`), since
> `test_annot_tracks.py` would otherwise pull polars-bio into CI. If keeping it in CI is
> required, switch the source to a bigWig fixture.

- [ ] **Step 2: Delete `Dataset.write_annot_tracks`**

Remove the whole method `Dataset.write_annot_tracks` (`_impl.py` ~1548-1613).

- [ ] **Step 3: Delete `_annot_to_intervals` and update the Task-2 oracle test**

Remove `_annot_to_intervals` (`_impl.py` ~1871-1898). Then in
`tests/unit/test_write_annot.py` replace the oracle comparison with a hand-computed
expectation so the test no longer imports the deleted function:

```python
def test_annot_overlap_explicit():
    from genvarloader._table import annot_overlap

    regions, annot = _regions(), _annot()
    got = annot_overlap(regions, annot)
    # region 0 [chr1:0-100] overlaps the 3 chr1 annots; region 1 [chr1:50-150] overlaps
    # the chr1 annots at 60-70 and 90-95; region 2 [chr2:0-100] overlaps the chr2 annot.
    np.testing.assert_array_equal(
        np.diff(got.values.offsets), np.array([3, 2, 1])
    )
    # region 2's single interval is the chr2 annot 5-15 with score 4.0
    np.testing.assert_array_equal(np.asarray(got.starts[2]), [5])
    np.testing.assert_array_equal(np.asarray(got.ends[2]), [15])
    np.testing.assert_allclose(np.asarray(got.values[2]), [4.0])
```

Delete the now-stale `test_annot_overlap_matches_pyranges_oracle`.

- [ ] **Step 4: Grep for stragglers**

Run: `pixi run -e dev python -c "import genvarloader"` (import sanity)
Run: `rtk grep -rn "write_annot_tracks\|_annot_to_intervals" python/ tests/`
Expected: no references except possibly in `docs/` (handled in Task 7). Fix any code hits.

- [ ] **Step 5: Run the affected tests**

Run: `GVL_TEST_EXPERIMENTAL=1 pixi run -e dev pytest tests/integration/tracks/test_annot_tracks.py tests/unit/test_write_annot.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_impl.py tests/integration/tracks/test_annot_tracks.py tests/unit/test_write_annot.py
rtk git commit -m "refactor: remove Dataset.write_annot_tracks and _annot_to_intervals"
```

---

### Task 7: Docs — SKILL.md, changelog

**Files:**
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `docs/source/changelog.md`

- [ ] **Step 1: Update the skill**

In `skills/genvarloader/SKILL.md`:
- Add `gvl.update(dataset, tracks=, annot_tracks=, *, overwrite=, max_mem=)` to the public
  API section, describing: accepts a path or `Dataset`; per-sample `tracks` need exact
  sample-set agreement; `annot_tracks` are sample-less (path to table / path to bigWig /
  DataFrame/LazyFrame, BED-like columns); table/DataFrame annot sources need the `table`
  extra and are experimental; atomic per-track publish; live datasets don't auto-discover
  updates.
- Add `annot_tracks=` to the `gvl.write` signature and notes.
- Note `gvl.write` now parallelizes over variants/tracks/annot_tracks (variants first,
  then tracks ∥ annot_tracks; `max_mem` divided across concurrent categories).
- Remove `Dataset.write_annot_tracks` from the API surface and the gotchas table; point
  readers to `gvl.update(..., annot_tracks=)`.

- [ ] **Step 2: Update the changelog**

Add an entry to `docs/source/changelog.md` under the current unreleased/next version:

```
- **write/update**: `gvl.write` gains `annot_tracks=` and parallelizes over
  variants/tracks/annot_tracks. New `gvl.update(dataset, tracks=, annot_tracks=)` adds
  tracks to an existing dataset with atomic per-track publish. `Dataset.write_annot_tracks`
  removed (use `gvl.update`). Annotation extraction now uses the polars-bio backend
  (table/DataFrame sources require the `table` extra; bigWig sources do not).
```

- [ ] **Step 3: Commit**

```bash
rtk git add skills/genvarloader/SKILL.md docs/source/changelog.md
rtk git commit -m "docs: document gvl.update, write annot_tracks, parallel writing"
```

---

## Final verification

- [ ] Run the full track/write/update suite:
  `GVL_TEST_EXPERIMENTAL=1 pixi run -e dev pytest tests/integration -k "track or write or update or annot" -v`
- [ ] Run ruff + typecheck:
  `pixi run -e dev ruff check python/` and `pixi run -e dev typecheck`
- [ ] Import sanity: `pixi run -e dev python -c "import genvarloader as gvl; gvl.update; gvl.write"`

## Self-review notes (spec coverage)

- annot DRY-with-Table via polars-bio → Tasks 2, 6 (retire pyranges).
- annot accepts path/DataFrame/LazyFrame/bigwig (not Table/BigWigs objects) → Task 2/3.
- `write` gains `annot_tracks` → Task 3.
- `write` parallel, per-category, variants-first, `max_mem` divided → Task 4.
- `gvl.update` top-level, path-or-Dataset, atomic per-track, sample-set agreement,
  parallel → Task 5.
- per-source polars-bio gating + `ExperimentalWarning` → Tasks 2, 3, 5 (warning emitted in
  `annot_overlap`; `Table` warns on its own construction).
- remove `Dataset.write_annot_tracks`; leave `write_transformed_track` stub untouched →
  Task 6 (stub never referenced).
- docs/skill/changelog → Task 7.
- reader-during-update robustness (`.tmp.`/`.old.`/`.lock` siblings) → Task 5 Step 4.
