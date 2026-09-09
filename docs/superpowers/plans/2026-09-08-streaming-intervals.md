# Interval (BigWigs/Table) Streaming Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend `StreamingDataset` to read interval tracks (`BigWigs` / `Table`) directly from source files and reconstruct them on the fly — including re-alignment to haplotype coordinates when indels are present — at byte-identical parity with `gvl.write()` + `Dataset.open()[r, s]`.

**Architecture:** One new `_TrackBackend` (a separate module) keyed on the existing `IntervalTrack` Protocol, calling only `count_intervals` + `_intervals_from_offsets`. It is driven from the per-window Python loop that every `_iter_batches` prefetch drive already has. Mixed variants+tracks reuses the existing `intervals_and_realign_track_fused` kernel and is **SVAR1-only** in v1. A `RustTable` per-contig COITree cache is a blocking prerequisite.

**Tech Stack:** Python 3.10-3.13, Rust (PyO3 0.29 / maturin), polars, numpy, seqpro Ragged, coitrees, bigtools, pixi, pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-streaming-intervals-design.md`

**Issue:** [#279](https://github.com/mcvickerlab/GenVarLoader/issues/279) · **Target branch:** `streaming` (NOT `main`)

## Global Constraints

- **Target branch is `streaming`.** Streaming PRs merge into `streaming`, never `main`. Add the PR to the StreamingDataset project board and keep "Closes/relates to #279" accurate.
- **Byte-identical parity is the oracle.** Streamed output must equal `gvl.Dataset.open(...)[r, s]` exactly at `jitter=0`. Compare as a **set** of `(region, sample)` cells — iteration order differs by design.
- **Parity is gated on `gvl.write(..., extend_to_length=False, max_jitter=None)`** in v1. See spec §3.2.
- **Rebuild Rust before running Python tests that import the extension:** `pixi run -e dev maturin develop --release`. `pixi run -e dev pytest` does NOT rebuild; a stale `.so` silently passes or fails against the old binary.
- **Slow tests are NOT deselected.** `pyproject.toml` has no `addopts`. Run `pixi run -e dev gen` before the first test run.
- **Run test suites via `sbatch`, not in the interactive session.** Spawn-child tests time out on a loaded node. Inside a batch script, `unset CLAUDE_JOB_DIR` first and use `/local/$USER` for scratch — `$CLAUDE_JOB_DIR/tmp` is a symlink to the *submitting* node's local disk and dangles elsewhere.
- **Never use bare `git stash` / `git stash pop`** — the stash stack is shared across worktrees.
- **Google-style docstrings** in `python/genvarloader/` (ruff pydocstyle, `Args:`/`Returns:`/`Raises:`, no NumPy underlines). Verify with `python scripts/docstring_style.py --check python/genvarloader`.
- **Public-API changes must update `skills/genvarloader/SKILL.md`**, and `docs/source/api.md` must stay in sync with `__all__`.
- **Mixed variants+tracks is SVAR1-only.** VCF / PGEN / SVAR2 + `tracks=` must raise `NotImplementedError`. Tracks-only must work on every backend.
- **Track axis order is the lexicographic sort of `track.name`**, never `tracks=` argument order.
- **The track axis is never squeezed.** One track is `(batch, 1, …)`.

---

## File Structure

| File | Responsibility | Tasks |
|---|---|---|
| `src/tables.rs` | Add a per-contig COITree cache behind a `Mutex` | 1 |
| `tests/dataset/conftest.py` | Track parity fixtures (BigWigs + Table, non-alphabetical names, indel case) | 2 |
| `python/genvarloader/_dataset/_track_stream.py` | **New.** `_TrackBackend`: window reads over the `IntervalTrack` Protocol | 3 |
| `python/genvarloader/_dataset/_streaming.py` | `iteration_order`, `_plan()` swap, `tracks=` construction, per-window track read, guards, `max_mem` | 4, 5, 6, 7, 8, 9 |
| `tests/dataset/test_streaming_tracks.py` | **New.** Track parity + shape/order/guard tests | 2, 5, 6, 7, 8 |
| `docs/source/*.md`, `skills/genvarloader/SKILL.md` | User-facing docs for `tracks=` / `iteration_order=` | 10 |

## Parallelization

Dispatch with **superpowers:dispatching-parallel-agents** using **superpowers:subagent-driven-development**. Use **Sonnet or weaker** for implementation; reserve stronger models for second-pass fixes where the implementer critically failed.

- **Wave A (fully parallel — disjoint files):** Task 1 (`src/tables.rs`), Task 2 (`tests/dataset/conftest.py`), Task 3 (new `_track_stream.py`).
- **Wave B (sequential spine — all touch `_streaming.py`, run in order):** Task 4 → 5 → 6 → 7 → 8 → 9.
- **Wave C:** Task 10 (docs), after Wave B.

Wave B tasks must NOT be parallelized: they edit overlapping regions of `_streaming.py` and will conflict.

---

## Task 1: `RustTable` per-contig COITree cache

`RustTable::build_trees` rebuilds a COITree for **every sample on the contig** on every `count` / `intervals_from_offsets` call. `write_track_impl` already avoids this with a `cur_chrom` guard; the query path does not. Streaming is the first consumer of the query path, so this is a prerequisite.

**Files:**
- Modify: `src/tables.rs:24-27` (struct), `:57-70` (`build_trees`), `:80-107` (`count`), `:107-147` (`intervals_from_offsets`)
- Test: `src/tables.rs` (`#[cfg(test)]` module at `:328+`)

**Interfaces:**
- Consumes: nothing.
- Produces: no signature changes. `RustTable::count` and `RustTable::intervals_from_offsets` keep their exact signatures and outputs; only their internal tree acquisition changes.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)]` module in `src/tables.rs`:

```rust
#[test]
fn test_tree_cache_reuses_same_contig() {
    let t = toy();
    // Two queries on the same contig must reuse one build.
    let _ = t.count(0, &[0], &[100], &[0]);
    let _ = t.count(0, &[0], &[100], &[0]);
    assert_eq!(t.builds_for_test(), 1, "same contig must build trees once");

    // Switching contigs rebuilds; switching back rebuilds again (one slot).
    let _ = t.count(1, &[0], &[100], &[0]);
    assert_eq!(t.builds_for_test(), 2);
    let _ = t.count(0, &[0], &[100], &[0]);
    assert_eq!(t.builds_for_test(), 3);
}

#[test]
fn test_cache_does_not_change_results() {
    let t = toy();
    let a = t.count(0, &[0, 10], &[100, 50], &[0, 1]);
    let b = t.count(0, &[0, 10], &[100, 50], &[0, 1]);
    assert_eq!(a, b);
}
```

Note: `toy()` already exists at `src/tables.rs:329`. If it builds only one contig, extend it to two contigs so the contig-switch assertions are meaningful.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/spec-279-interval-streaming
export LD_LIBRARY_PATH="$PWD/.pixi/envs/dev/lib:${LD_LIBRARY_PATH:-}"
pixi run -e dev cargo test --lib tables
```

Expected: FAIL — `no method named 'builds_for_test'`.

- [ ] **Step 3: Add the cache**

Replace the struct and add the cached accessor:

```rust
use std::sync::Mutex;

#[pyclass]
pub struct RustTable {
    store: Vec<ContigStore>, // indexed by chrom_code (0..n_contigs)
    /// Most recently queried contig's trees. One slot: `_plan()` is
    /// contig-run-major in both iteration orders, so a sweep rebuilds each
    /// contig exactly once. `Mutex` (not `RefCell`) because `#[pyclass]`
    /// requires `Sync`; this mirrors the codebase's pyclass-mutable-state
    /// pattern in `src/ffi/stream_core.rs:124`.
    tree_cache: Mutex<Option<(usize, Arc<Vec<BasicCOITree<u32, u32>>>)>>,
    #[cfg(test)]
    build_count: std::sync::atomic::AtomicUsize,
}
```

Add `use std::sync::Arc;` at the top. In `build`, initialise the new fields:

```rust
RustTable {
    store,
    tree_cache: Mutex::new(None),
    #[cfg(test)]
    build_count: std::sync::atomic::AtomicUsize::new(0),
}
```

Add the cached accessor next to `build_trees` (keep `build_trees` as-is; it becomes the cache-miss path):

```rust
/// Trees for `chrom`, reusing the cached set when it is the same contig.
///
/// Returns an `Arc` so the lock is released before querying — a query must
/// not hold the cache mutex, or a future track producer thread would
/// serialize on it.
fn trees_for(&self, chrom: usize) -> Arc<Vec<BasicCOITree<u32, u32>>> {
    let mut slot = self.tree_cache.lock().unwrap();
    if let Some((cached_chrom, trees)) = slot.as_ref() {
        if *cached_chrom == chrom {
            return Arc::clone(trees);
        }
    }
    #[cfg(test)]
    self.build_count
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let trees = Arc::new(self.build_trees(chrom));
    *slot = Some((chrom, Arc::clone(&trees)));
    trees
}

#[cfg(test)]
fn builds_for_test(&self) -> usize {
    self.build_count.load(std::sync::atomic::Ordering::Relaxed)
}
```

In `count` (`src/tables.rs:92`) replace `let trees = self.build_trees(chrom_code as usize);` with:

```rust
let trees = self.trees_for(chrom_code as usize);
```

In `intervals_from_offsets` (`src/tables.rs:127`) replace `let trees = self.build_trees(chrom);` with:

```rust
let trees = self.trees_for(chrom);
```

Leave `write_track_impl` alone — it has its own `cur_chrom` guard and is already correct.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pixi run -e dev cargo test --lib tables
pixi run -e dev cargo clippy --all-targets -- -D warnings
```

Expected: PASS, no clippy warnings.

- [ ] **Step 5: Verify the Python side still matches**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -k table -q
```

Expected: PASS — the cache must not change any result.

- [ ] **Step 6: Commit**

```bash
git add src/tables.rs
git commit -m "perf(tables): cache per-contig COITrees in RustTable query path

build_trees rebuilds a tree for every sample on the contig on every count/
intervals_from_offsets call. write_track_impl already avoids this with its
cur_chrom guard; the query path did not. Streaming is its first consumer.

One cache slot, behind a Mutex (pyclass requires Sync; RefCell will not do),
returning an Arc so the lock is released before querying. _plan() is
contig-run-major in both iteration orders, so a sweep rebuilds each contig
exactly once.

Relates to #279"
```

---

## Task 2: Track parity fixtures

**Files:**
- Modify: `tests/dataset/conftest.py`
- Test: exercised by Tasks 5-8

**Interfaces:**
- Consumes: nothing.
- Produces: pytest fixture `streaming_tracks_fixture` returning a `StreamingTracksFixture` dataclass with fields `bed: Path`, `reference_path: Path`, `svar_path: Path`, `dataset_path: Path`, `bigwigs: gvl.BigWigs`, `table: gvl.Table`, `samples: list[str]`. Consumed by every test in Tasks 5-8.

- [ ] **Step 1: Write the fixture**

Append to `tests/dataset/conftest.py`. Model it on the existing bigwig fixture at `tests/dataset/conftest.py:370-401` (which uses `pyBigWig` directly) and on `svar1_multicontig_fixture` (`:122`) for the variant side.

```python
@dataclass(slots=True)
class StreamingTracksFixture:
    """Fixture for interval-streaming parity (issue #279).

    Deliberately hostile to the two bugs the spec calls out (§8):

    - Track names are passed in NON-alphabetical order (``zeta`` before
      ``alpha``) so a test that assumes ``tracks=`` argument order fails
      loudly. The written path sorts (`_tracks.py:283`), so the expected
      track axis is ``[alpha, zeta]``.
    - The bed includes a region overlapping a deletion, so the indel
      re-alignment path is exercised rather than the trivial one.
    """

    bed: Path
    reference_path: Path
    svar_path: Path
    dataset_path: Path
    bigwigs: "gvl.BigWigs"
    table: "gvl.Table"
    #: Same track as `bigwigs` plus one extra sample the dataset does not have.
    #: Exercises the §8 "strict superset" row and the query-by-NAME rule.
    bigwigs_superset: "gvl.BigWigs"
    samples: list[str]
    #: Contig order the dataset's `regions[:, 0]` indexes into.
    contigs_list: list[str]


@pytest.fixture(scope="module")
def streaming_tracks_fixture(
    tmp_path_factory, svar1_multicontig_fixture
) -> StreamingTracksFixture:
    """SVAR1 variants + two interval tracks, written with parity-safe flags.

    Scope is ``module``, not ``session``: it depends on the module-scoped
    ``svar1_multicontig_fixture`` and pytest forbids the wider scope.
    """
    import pyBigWig

    import genvarloader as gvl

    base = svar1_multicontig_fixture
    tmp_dir = tmp_path_factory.mktemp("streaming_tracks")

    # `Svar1MultiContigFixture` has fields svar_path / reference_path /
    # contigs / bed / dataset_path. `bed` is ALREADY a pl.DataFrame (do not
    # call read_bedlike on it), and there is NO `samples` field -- take the
    # sample names from the written dataset so they match its public order.
    bed = base.bed
    samples = list(gvl.Dataset.open(base.dataset_path).samples)

    # Contig sizes come from the reference .fai, NOT from the bed. Both
    # contigs are only 40 bp; padding past the reference length would make
    # the bigwig header disagree with the reference.
    fai = pl.read_csv(
        str(base.reference_path) + ".fai",
        separator="\t",
        has_header=False,
        new_columns=["chrom", "length", "offset", "linebases", "linewidth"],
    )
    contig_sizes = [
        (r["chrom"], int(r["length"]))
        for r in fai.iter_rows(named=True)
        if r["chrom"] in set(bed["chrom"].to_list())
    ]

    # --- track "alpha": one bigwig per sample -------------------------------
    # bigwig entries must be SORTED and NON-OVERLAPPING. The bed's regions are
    # 20 bp sliding windows at starts 0,4,...,20, so they overlap heavily --
    # emitting intervals per region would produce an invalid file that
    # `addEntries` rejects. Instead tile each contig with disjoint 10 bp bins,
    # which still covers every region.
    BIN = 10
    bw_paths: dict[str, str] = {}
    for i, sample in enumerate(samples):
        p = tmp_dir / f"{sample}.alpha.bw"
        with pyBigWig.open(str(p), "w") as bw:
            bw.addHeader(contig_sizes, maxZooms=0)
            chroms, starts, ends, values = [], [], [], []
            for contig, size in contig_sizes:
                for b, lo in enumerate(range(0, size, BIN)):
                    hi = min(lo + BIN, size)
                    chroms.append(contig)
                    starts.append(lo)
                    ends.append(hi)
                    # Distinct per (sample, contig, bin) so a wrong sample, a
                    # wrong contig, or an off-by-one bin is visible in values.
                    values.append(float(10 * (i + 1) + b) + (0.5 if contig != contig_sizes[0][0] else 0.0))
            bw.addEntries(chroms, starts, ends=ends, values=values)
        bw_paths[sample] = str(p)
    alpha = gvl.BigWigs("alpha", bw_paths)

    # A superset variant: the same per-sample files plus one extra sample the
    # dataset never sees. Keyed FIRST in the dict so a positional index would
    # shift every sample by one and fail loudly.
    superset_paths = {"zz_extra_sample": bw_paths[samples[0]], **bw_paths}
    alpha_superset = gvl.BigWigs("alpha", superset_paths)

    # --- track "zeta": a long-form Table ------------------------------------
    # Same disjoint binning, different values, so the two tracks are never
    # confusable with each other.
    rows = []
    for i, sample in enumerate(samples):
        for contig, size in contig_sizes:
            for b, lo in enumerate(range(0, size, BIN)):
                rows.append(
                    {
                        "sample_id": sample,
                        "chrom": contig,
                        "start": lo,
                        "end": min(lo + BIN, size),
                        "value": float(100 * (i + 1) + b),
                    }
                )
    zeta = gvl.Table("zeta", pl.DataFrame(rows))

    out = tmp_dir / "tracks.gvl"
    gvl.write(
        path=out,
        bed=bed,
        variants=base.svar_path,
        # NON-alphabetical on purpose: the written track axis must still be
        # [alpha, zeta] because Tracks.from_path sorts (_tracks.py:283).
        tracks=[zeta, alpha],
        # Parity is gated on these in v1 -- see spec §3.2.
        extend_to_length=False,
        max_jitter=None,
    )

    return StreamingTracksFixture(
        bed=bed,
        reference_path=base.reference_path,
        svar_path=base.svar_path,
        dataset_path=out,
        bigwigs=alpha,
        table=zeta,
        bigwigs_superset=alpha_superset,
        samples=samples,
        contigs_list=list(base.contigs),
    )
```

`bed` is a `pl.DataFrame` on the fixture, so tests pass it straight to `gvl.StreamingDataset(f.bed, ...)` — never `gvl.read_bedlike(f.bed)`. Where a test needs region bounds, use `f.bed` directly.

Indel coverage is already satisfied: `_SVAR1_MC_VCF` carries insertions (`chr1:7 C→CAT`, `chr2:9 T→TGG`) **and** a deletion (`chr1:12 GTA→G`) across 3 samples, and the bed's 20 bp windows at starts 0-20 overlap all of them. Task 7's re-alignment path is therefore genuinely exercised. Assert it: at least one cell's re-aligned track length must differ from its region length, or the test is silently checking the trivial case.

If `svar1_multicontig_fixture` does not expose `samples` / `reference_path` / `svar_path` / `bed` under those exact names, read its dataclass at `tests/dataset/conftest.py:122` and use its real field names — do not guess.

- [ ] **Step 2: Verify the fixture builds**

Add a throwaway smoke test in `tests/dataset/test_streaming_tracks.py`:

```python
def test_fixture_builds(streaming_tracks_fixture):
    import genvarloader as gvl

    f = streaming_tracks_fixture
    ds = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
    assert list(ds.available_tracks) == ["alpha", "zeta"], (
        "written track axis must be name-sorted, not tracks= argument order"
    )
```

Run:

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
```

Expected: PASS. This also *proves* the sorted-order claim in spec §2.4 against the real code, which is the assumption the whole track axis rests on.

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/conftest.py tests/dataset/test_streaming_tracks.py
git commit -m "test(streaming): add interval-streaming parity fixtures

Two tracks passed in NON-alphabetical order so any test assuming tracks=
argument order fails loudly; the written axis is name-sorted. Written with
extend_to_length=False, max_jitter=None per the v1 parity gate.

Relates to #279"
```

---

## Task 3: `_TrackBackend` window reads

**Files:**
- Create: `python/genvarloader/_dataset/_track_stream.py`
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: the `IntervalTrack` Protocol (`python/genvarloader/_types.py:119`).
- Produces:
  - `class _TrackBackend` with `__init__(self, tracks: Sequence[IntervalTrack], regions: NDArray[np.int32], contigs: list[str], samples: list[str])`
  - `.names -> list[str]` (name-sorted)
  - `.read_window(r_idx: NDArray[np.intp], s_idx: NDArray[np.intp], starts: NDArray[np.int32] | None = None, ends: NDArray[np.int32] | None = None) -> list[RaggedIntervals]` (one per track, in `.names` order). `starts`/`ends` override the region bounds — Task 7 passes jitter-translated bounds through them; `None` means "use `self._regions`".
  - Raises `ValueError` at construction for duplicate names, uncovered contigs, and missing samples.

Tasks 5-8 consume all of these.

- [ ] **Step 1: Write the failing tests**

In `tests/dataset/test_streaming_tracks.py`:

```python
import numpy as np
import pytest

import genvarloader as gvl
from genvarloader._dataset._track_stream import _TrackBackend


def _backend(f, tracks):
    """Build a `_TrackBackend` over the fixture's regions.

    Take `_regions` off a constructed `StreamingDataset` rather than calling
    `bed_to_regions` directly: that helper is
    `bed_to_regions(bed: pl.DataFrame, contig_norm: ContigNormalizer)` and
    returns ONE `(n_regions, 4)` array, not a `(regions, sort_order)` pair,
    so hand-rolling the call means also hand-rolling a `ContigNormalizer`.
    Reading the attribute keeps the test honest about what production builds.

    Uses the VARIANTS-only constructor, which already works today. Do not use
    `tracks=` here: that argument does not exist until Task 5, and this task
    must be testable on its own.
    """
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    )
    return _TrackBackend(tracks, sds._regions, list(sds.contigs), list(sds.samples))


def test_names_are_sorted_not_argument_order(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    b = _backend(f, [f.table, f.bigwigs])  # zeta, alpha
    assert b.names == ["alpha", "zeta"]


def test_duplicate_track_names_raise(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        _backend(f, [f.bigwigs, f.bigwigs])


def test_missing_sample_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    )
    with pytest.raises(ValueError, match="not present"):
        _TrackBackend(
            [f.bigwigs],
            sds._regions,
            list(sds.contigs),
            [*f.samples, "no_such_sample"],
        )


def test_read_window_shape(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    b = _backend(f, [f.bigwigs])
    r_idx = np.array([0, 1], dtype=np.intp)
    s_idx = np.arange(len(f.samples), dtype=np.intp)
    (itvs,) = b.read_window(r_idx, s_idx)
    assert itvs.shape[:2] == (len(r_idx), len(s_idx))
```

`f.contigs_list` is provided by Task 2's fixture (`list(base.contigs)`), and `f.bed` is a `pl.DataFrame` — pass it to `StreamingDataset` directly, never through `read_bedlike`.

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
```

Expected: FAIL — `No module named 'genvarloader._dataset._track_stream'`.

- [ ] **Step 3: Implement `_TrackBackend`**

Create `python/genvarloader/_dataset/_track_stream.py`:

```python
"""Streaming read backend for interval-valued tracks (issue #279).

One backend serves every :class:`~genvarloader._types.IntervalTrack` — both
:class:`~genvarloader.BigWigs` and :class:`~genvarloader.Table` — because the
Protocol is the only surface this module touches. Nothing here branches on the
concrete class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np
from numpy.typing import NDArray

from .._ragged import RaggedIntervals
from .._utils import lengths_to_offsets, normalize_contig_name

if TYPE_CHECKING:
    from .._types import IntervalTrack


class _TrackBackend:
    """Window-granular interval reads over the ``IntervalTrack`` Protocol.

    The window (regions x sample-chunk) is the READ granularity, mirroring the
    variant backends' ``read_window``/``generate_batch`` split. Every call is
    single-contig, which ``_plan()``'s contig-run outer loop guarantees.

    Args:
        tracks: One or more interval tracks. Stored sorted by ``name`` — the
            written path's track axis comes from a sorted directory listing
            (``_tracks.py:283``), NOT from argument order, so sorting here is
            what makes byte-parity hold.
        regions: ``(n_regions, 4)`` sorted regions, as held by
            ``StreamingDataset._regions``. Only columns 0-2 are read.
        contigs: Contig names indexed by ``regions[:, 0]``.
        samples: Dataset sample names in public (sorted) order.

    Raises:
        ValueError: If two tracks share a name, if a track does not cover a
            contig the BED references, or if a track is missing a dataset
            sample.
    """

    def __init__(
        self,
        tracks: "Sequence[IntervalTrack]",
        regions: NDArray[np.int32],
        contigs: list[str],
        samples: list[str],
    ) -> None:
        _tracks = list(tracks)
        names = [t.name for t in _tracks]
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(
                f"Duplicate track name(s) {dupes}. Track names must be unique:"
                " the written layout stores each track under intervals/<name>/,"
                " so duplicates are indistinguishable."
            )
        # Name-sorted, matching the written path's `available_tracks.sort()`.
        order = sorted(range(len(_tracks)), key=lambda i: names[i])
        self._tracks = [_tracks[i] for i in order]
        self.names = [names[i] for i in order]

        self._regions = regions
        self._contigs = contigs
        self._samples = samples

        # Contig coverage, validated ONCE. The two implementations fail
        # differently at read time -- BigWigs raises, Table silently returns
        # zeros (src/tables.rs:93-95) -- so neither is an acceptable guard.
        used = {contigs[i] for i in np.unique(regions[:, 0])}
        for track in self._tracks:
            missing = sorted(
                c for c in used if normalize_contig_name(c, track.contigs) is None
            )
            if missing:
                raise ValueError(
                    f"Track {track.name!r} does not cover contig(s) {missing},"
                    " which the BED references."
                )
            absent = sorted(set(samples) - set(track.samples))
            if absent:
                raise ValueError(
                    f"Sample(s) {absent} are not present in track"
                    f" {track.name!r}. Tracks may have extra samples, but must"
                    " cover every sample in the dataset."
                )

    def read_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        starts: NDArray[np.int32] | None = None,
        ends: NDArray[np.int32] | None = None,
    ) -> list[RaggedIntervals]:
        """Read one window's intervals for every track, single-contig.

        Args:
            r_idx: Region indices, all on one contig.
            s_idx: Public sample indices (into the dataset's sorted samples).
            starts: Query starts overriding the stored region starts. Under
                `jitter > 0` the caller MUST pass the same translated bounds
                the variant engine got, or tracks and haplotypes silently
                disagree by up to `jitter` bases. `None` uses the regions.
            ends: Query ends, same contract as `starts`.

        Returns:
            One ``RaggedIntervals`` per track in ``self.names`` order, each of
            shape ``(len(r_idx), len(s_idx), None)``.

        Raises:
            ValueError: If the window spans more than one contig.
        """
        r_idx = np.asarray(r_idx, dtype=np.intp)
        s_idx = np.asarray(s_idx, dtype=np.intp)

        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError(
                "_TrackBackend.read_window: window spans multiple contigs;"
                " every track query must be single-contig."
            )
        contig = self._contigs[contig_idx]
        if starts is None:
            starts = self._regions[r_idx, 1]
        if ends is None:
            ends = self._regions[r_idx, 2]
        starts = np.ascontiguousarray(starts, np.int32)
        ends = np.ascontiguousarray(ends, np.int32)

        # Query by NAME, never by index: BigWigs.samples is dict-insertion
        # order (_bigwig.py:41) while Table.samples is sorted (_table.py:58),
        # so a positional index means different samples in the two classes.
        # This is the track-side analogue of _Svar1Backend._phys_sample_idx.
        names = [self._samples[int(i)] for i in s_idx]

        out: list[RaggedIntervals] = []
        for track in self._tracks:
            counts = track.count_intervals(contig, starts, ends, sample=names)
            offsets = lengths_to_offsets(np.asarray(counts).ravel())
            out.append(
                track._intervals_from_offsets(
                    contig, starts, ends, offsets, sample=names
                )
            )
        return out
```

Verify the import paths for `lengths_to_offsets` and `normalize_contig_name` against `python/genvarloader/_bigwig.py`'s imports and fix if they differ.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
pixi run -e dev ruff check python/ tests/ && pixi run -e dev typecheck
python scripts/docstring_style.py --check python/genvarloader
```

Expected: PASS, clean.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_track_stream.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): add _TrackBackend interval window reads

One backend for every IntervalTrack -- the Protocol is the only surface it
touches. Tracks are name-sorted (the written axis comes from a sorted
directory listing, not tracks= order) and queried by sample NAME, since
BigWigs.samples is insertion-ordered and Table.samples is sorted.

Contig coverage and sample coverage are validated once at construction:
BigWigs raises at read time while Table silently returns zeros, so neither
read-time behaviour is an acceptable guard.

Relates to #279"
```

---

## Task 4: `iteration_order` and the `_plan()` loop swap

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:566-590` (`_plan`), `:309+` (`__init__`), field block `:263+`
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `StreamingDataset.__init__(..., iteration_order: Literal["auto", "regions", "samples"] = "auto")`; the resolved value on `self._iteration_order` (never `"auto"` after construction).

- [ ] **Step 1: Write the failing tests**

```python
def test_iteration_order_rejects_bad_value(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    with pytest.raises(ValueError, match="iteration_order"):
        gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order="sideways",
        )


def test_both_orders_visit_the_same_windows(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    made = {}
    for order in ("regions", "samples"):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order=order,
        )
        # Force the sample axis to chunk; otherwise both orders are trivially
        # identical (spec §4.4).
        object.__setattr__(sds, "_window_samples", 1)
        made[order] = sorted(
            (tuple(r.tolist()), tuple(s.tolist())) for r, s in sds._plan()
        )
    assert made["regions"] == made["samples"]


def test_orders_differ_in_sequence_when_samples_chunk(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    seqs = {}
    for order in ("regions", "samples"):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order=order,
        )
        object.__setattr__(sds, "_window_samples", 1)
        object.__setattr__(sds, "_window_regions", 1)
        seqs[order] = [(tuple(r.tolist()), tuple(s.tolist())) for r, s in sds._plan()]
    assert seqs["regions"] != seqs["samples"], (
        "with both axes chunked the visit ORDER must differ"
    )


def test_auto_resolves_to_regions_for_variants_only(svar1_multicontig_fixture):
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    )
    assert sds._iteration_order == "regions"


def test_n_batches_is_order_invariant(svar1_multicontig_fixture):
    """Spec §7.3: the batch-span multiset is identical between orders."""
    f = svar1_multicontig_fixture
    counts = {}
    for order in ("regions", "samples"):
        sds = gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.svar_path,
            iteration_order=order,
        )
        object.__setattr__(sds, "_window_samples", 1)
        object.__setattr__(sds, "_window_regions", 1)
        counts[order] = sds.n_batches(3)
    assert counts["regions"] == counts["samples"]


def test_return_indices_are_original_bed_rows(svar1_multicontig_fixture):
    """Spec §7.2: returned region indices are BED-row order, not sweep order."""
    f = svar1_multicontig_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path
    )
    n_regions, n_samples = sds.shape
    seen = set()
    for _data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        seen.update(zip(map(int, r_idx), map(int, s_idx)))
    assert seen == {(r, s) for r in range(n_regions) for s in range(n_samples)}
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q -k iteration_order or orders or auto_resolves
```

Expected: FAIL — `__init__` got an unexpected keyword argument `iteration_order`.

- [ ] **Step 3: Implement**

Add the field next to `_window_samples` (`_streaming.py:264`):

```python
    # Resolved iteration order; NEVER "auto" after __init__. See `_plan`.
    _iteration_order: str = "regions"
```

Add the keyword-only argument to `__init__`'s signature (after `contigs`):

```python
        iteration_order: Literal["auto", "regions", "samples"] = "auto",
```

Resolve it near the end of `__init__`, after `_window_samples`/`_window_regions` are set:

```python
        if iteration_order not in ("auto", "regions", "samples"):
            raise ValueError(
                "iteration_order must be one of 'auto', 'regions', 'samples';"
                f" got {iteration_order!r}."
            )
        if iteration_order == "auto":
            # Keyed off the SOURCE MIX, never per-class micro-properties.
            # Tracks-only -> samples, because BigWigs holds one file per sample.
            # Mixed -> regions, the roadmap's decision (non-optimal for the
            # track axis; `iteration_order="samples"` is the escape hatch).
            has_tracks = _track_backend_obj is not None
            has_variants = _backend_obj is not None
            resolved = "samples" if (has_tracks and not has_variants) else "regions"
        else:
            resolved = iteration_order
        object.__setattr__(self, "_iteration_order", resolved)
```

`_track_backend_obj` does not exist until Task 5. For THIS task, use `has_tracks = False` and leave a comment `# Task 5 wires tracks in here.`; Task 5 replaces it.

Rewrite `_plan` (`_streaming.py:566`):

```python
    def _plan(self) -> Iterator[tuple[NDArray[np.intp], NDArray[np.intp]]]:
        """Yield one WINDOW per step: `(region_idxs, sample_chunk)`.

        Cartesian and single-contig. The contig-run loop is outermost in BOTH
        iteration orders: it is required by the single-contig Rust invariant,
        and it is what lets `RustTable`'s per-contig tree cache hold.

        `iteration_order` only changes the VISIT ORDER, never the window set.
        It is also a no-op whenever `_window_samples == n_samples` (the default
        for any cohort under ~8.4M at `max_mem="512MB"`), because the sample
        loop then runs exactly once.
        """
        n_regions, n_samples = self.shape
        if n_regions == 0:
            return
        contig_idxs = self._regions[:, 0]
        run_bounds = np.flatnonzero(np.diff(contig_idxs)) + 1
        run_starts = np.concatenate(([0], run_bounds))
        run_ends = np.concatenate((run_bounds, [n_regions]))
        sample_major = self._iteration_order == "samples"
        for r_lo, r_hi in zip(run_starts, run_ends):
            # Materialize each region window's index array ONCE and re-yield the
            # same object per sample chunk. `_iter_batches` stores these in
            # `plan_jobs`, so allocating inside the inner loop would multiply
            # job residency by n_sample_chunks.
            windows = [
                np.arange(w_lo, min(w_lo + self._window_regions, int(r_hi)), dtype=np.intp)
                for w_lo in range(int(r_lo), int(r_hi), self._window_regions)
            ]
            chunks = [
                np.arange(s_lo, min(s_lo + self._window_samples, n_samples), dtype=np.intp)
                for s_lo in range(0, n_samples, self._window_samples)
            ]
            if sample_major:
                for s in chunks:
                    for w in windows:
                        yield w, s
            else:
                for w in windows:
                    for s in chunks:
                        yield w, s
```

Also add `iteration_order` to whatever `replace`/`evolve` path `with_settings` uses so it survives `with_*` calls, if that class copies fields explicitly.

- [ ] **Step 4: Run the tests to verify they pass**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
pixi run -e dev pytest tests/dataset tests/unit -q
```

Expected: PASS, and no regression in the existing streaming suite (the default resolves to `"regions"`, which is exactly today's behaviour).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): add iteration_order and the _plan sample-major swap

The contig-run loop stays outermost in both orders. Region-window index arrays
are materialized once per contig run and re-yielded per sample chunk, so job
residency stays O(n_region_windows x window_regions) in both orders rather
than gaining an n_sample_chunks factor.

iteration_order is a no-op whenever _window_samples == n_samples, which is the
default for any cohort under ~8.4M at max_mem=512MB; documented in _plan.

Relates to #279"
```

---

## Task 5: `tracks=` construction and tracks-only datasets

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:309-500` (`__init__`), `:400-405` (the variants-required guard), `:476` (`cell_bytes`)
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: `_TrackBackend` (Task 3), `_iteration_order` (Task 4).
- Produces: `StreamingDataset(..., tracks: IntervalTrack | Sequence[IntervalTrack] | None = None)`; `self._track_backend: _TrackBackend | None`; a constructible tracks-only dataset whose `shape` is `(n_regions, n_samples)` and whose `samples` come from the tracks' intersection.

- [ ] **Step 1: Write the failing tests**

```python
def test_tracks_only_constructs(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    assert sds.shape == (len(f.bed), len(f.samples))
    assert sds.samples == sorted(f.samples)


def test_tracks_only_auto_is_sample_major(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    assert sds._iteration_order == "samples"


def test_mixed_auto_is_region_major(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    )
    assert sds._iteration_order == "regions"


def test_no_sources_still_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    with pytest.raises(ValueError, match="variants|tracks"):
        gvl.StreamingDataset(f.bed)


def test_with_seqs_on_tracks_only_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    with pytest.raises(ValueError, match="no variant source"):
        sds.with_seqs("haplotypes")
```

- [ ] **Step 2: Run to verify they fail**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q -k tracks_only or mixed_auto or no_sources or with_seqs_on
```

Expected: FAIL — unexpected keyword argument `tracks`.

- [ ] **Step 3: Implement**

Add to `__init__`'s signature after `variants`:

```python
        tracks: "IntervalTrack | Sequence[IntervalTrack] | None" = None,
```

Replace the variants-required guard (`_streaming.py:400-405`) so tracks alone suffice:

```python
        if variants is None and tracks is None and _reconstruct_window is None:
            raise ValueError(
                "StreamingDataset requires at least one source: `variants=`,"
                " `tracks=`, or the internal `_reconstruct_window=`."
            )
```

Normalize `tracks` early:

```python
        from .._types import IntervalTrack  # noqa: F401  (runtime duck-typing)

        if tracks is None:
            _track_list = []
        elif isinstance(tracks, Sequence) and not isinstance(tracks, (str, bytes)):
            _track_list = list(tracks)
        else:
            _track_list = [tracks]
```

When there is no variant source, derive the dataset's identity from the tracks instead:

```python
        if _backend_obj is None and _track_list:
            # Tracks-only. `contigs`/`samples` come from the tracks, mirroring
            # gvl.write's intersection rule (_write.py:346). There is no
            # ploidy: output carries no ploidy axis.
            if contigs is None:
                contigs = sorted(
                    set.intersection(*(set(t.contigs) for t in _track_list))
                )
            if samples is None:
                samples = sorted(
                    set.intersection(*(set(t.samples) for t in _track_list))
                )
                if not samples:
                    raise ValueError(
                        "Tracks share no samples; a tracks-only StreamingDataset"
                        " needs at least one sample common to every track."
                    )
            n_samples = len(samples)
            ploidy = 1  # placeholder; never used for output shaping
```

Build the backend after `_regions`/`contigs`/`samples` are final:

```python
        _track_backend_obj = None
        if _track_list:
            from ._track_stream import _TrackBackend

            _track_backend_obj = _TrackBackend(
                _track_list, self._regions, list(self.contigs), list(self._samples)
            )
        object.__setattr__(self, "_track_backend", _track_backend_obj)
```

Replace Task 4's `has_tracks = False` placeholder with `has_tracks = _track_backend_obj is not None`.

Fold a track-side per-cell cost into the `max_mem` derivation (`_streaming.py:476`) so `_window_samples` can actually shrink — without this, `iteration_order` is permanently a no-op (spec §2.5, §4.4):

```python
        cell_bytes = getattr(_backend_obj, "_cell_bytes", int(ploidy) * 16)
        if _track_backend_obj is not None:
            # Each (region, sample) cell also materializes intervals:
            # start+end (i32) + value (f32) = 12 B per interval. Budget a
            # conservative constant per cell per track until a measured
            # estimate is available (follow-up).
            TRACK_BYTES_PER_CELL = 12 * 64
            cell_bytes += TRACK_BYTES_PER_CELL * len(_track_backend_obj.names)
```

Guard `with_seqs` (`_streaming.py:1368`) at the top:

```python
        if self._backend is None:
            raise ValueError(
                "with_seqs() requires a variant source; this StreamingDataset"
                " has no variant source (tracks only), so it yields tracks"
                " alone from to_iter()."
            )
```

Add `IntervalTrack` to the module's `TYPE_CHECKING` imports and `Sequence` to the typing imports if absent.

- [ ] **Step 4: Run to verify they pass**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
pixi run -e dev pytest tests/dataset tests/unit -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): accept tracks= and allow tracks-only datasets

A tracks-only StreamingDataset derives contigs/samples from the tracks'
intersection (mirroring gvl.write, _write.py:346), carries no ploidy axis, and
rejects with_seqs(). Track memory now enters the max_mem budget, which is what
lets _window_samples shrink -- without it iteration_order can never bite.

Relates to #279"
```

---

## Task 6: Tracks-only reconstruction and parity

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:656+` (`_iter_batches`)
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: `_TrackBackend.read_window` (Task 3), `self._track_backend` (Task 5).
- Produces: `to_iter()` yields `(batch, n_tracks, None)` `Ragged` float tracks for a tracks-only dataset.

- [ ] **Step 1: Write the failing parity test**

```python
def test_tracks_only_parity(streaming_tracks_fixture):
    """Tracks WITHOUT variants: intervals_to_tracks, no realignment."""
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=[f.table, f.bigwigs])
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs(None)
        .with_settings(realign_tracks=False)
    )

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            streamed = data[i]
            expected = written[r, s]
            assert streamed.shape[0] == 2, "track axis is never squeezed"
            np.testing.assert_array_equal(
                np.asarray(streamed), np.asarray(expected)
            )
            seen.add((r, s))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_single_track_keeps_its_axis(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    data, _r, _s = next(iter(sds.to_iter(batch_size=1, return_indices=True)))
    assert data[0].shape[0] == 1, "one track must still be (1, ...), not squeezed"
```

If `with_seqs(None)` is not the written path's way to get tracks-only, read `_reconstruct.py:522` (`case None, Tracks()`) and use whatever `Dataset` API reaches that branch.

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q -k tracks_only_parity or single_track
```

Expected: FAIL — `to_iter` yields nothing for a tracks-only dataset.

- [ ] **Step 3: Implement the tracks-only drive**

In `_iter_batches`, add a branch BEFORE the `self._backend is not None` branch:

```python
        if self._backend is None and self._track_backend is not None:
            # Tracks-only: no variant engine, so drive the plan directly. The
            # window is the read granularity; batches slice it.
            from .._utils import lengths_to_offsets
            from ..genvarloader import intervals_to_tracks

            tb = self._track_backend
            n_tracks = len(tb.names)
            for r_idx, s_idx in self._plan():
                per_track = tb.read_window(r_idx, s_idx)
                n_s = len(s_idx)
                flat_r = np.repeat(self._sort_order[r_idx], n_s)
                flat_s = np.tile(np.asarray(s_idx, np.intp), len(r_idx))
                n_rows = len(flat_r)
                starts = self._regions[r_idx, 1]
                ends = self._regions[r_idx, 2]
                flat_starts = np.repeat(starts, n_s).astype(np.int32)
                lengths = np.repeat(ends - starts, n_s).astype(np.int64)
                # Flatten (region, sample) -> row ONCE per window, so each
                # batch is a contiguous row slice and offset_idxs is a plain
                # arange. `output_length` is applied here when it is an int.
                if isinstance(self._output_length, int):
                    lengths = np.full(len(lengths), self._output_length, np.int64)
                flat_itvs = [itvs.reshape(n_rows) for itvs in per_track]
                for lo in range(0, n_rows, batch_size):
                    hi = min(lo + batch_size, n_rows)
                    out = _tracks_from_intervals(
                        [itvs[lo:hi] for itvs in flat_itvs],
                        flat_starts[lo:hi],
                        lengths[lo:hi],
                    )
                    yield out, flat_r[lo:hi], flat_s[lo:hi]
            return
```

`itvs.reshape(n_rows)` and `itvs[lo:hi]` are the intended operations, not verified API — read `RaggedIntervals` in `python/genvarloader/_ragged.py` and use its real reshape/slice methods. If slicing a `RaggedIntervals` by row range is not directly supported, slice its `offsets` and `data` instead; do not fall back to a Python row loop.

Read `_call_float32:381-386` for how `output_length` becomes `out_lengths` and mirror it — `with_len(L)` must produce length-`L` rows, `with_len("ragged")` region-length rows.

Add the helper at module scope. **Mirror `Tracks._call_float32`'s buffer discipline exactly** (`_tracks.py:388-419`) — one vectorized `intervals_to_tracks` call per track writing into a track-major contiguous block, with the final `Ragged` built from interleaved `(b, t)` offsets. Byte-parity is the spec, so copying the written layout verbatim is the requirement, not a starting point to improve on:

```python
def _tracks_from_intervals(per_track, starts, lengths):
    """Rasterize one batch's intervals into a `(batch, n_tracks, None)` Ragged.

    Mirrors `Tracks._call_float32` (`_tracks.py:388-419`) exactly: each track
    fills its own contiguous `n_per_track` block with a single vectorized
    `intervals_to_tracks` call, and the returned `Ragged` uses the interleaved
    `repeat(lengths, "b -> b t")` offsets. Deviating from that layout breaks
    byte-parity, so do not "fix" it here.

    Args:
        per_track: One `RaggedIntervals` per track, name-sorted, each already
            sliced to this batch's `(row,)` cells.
        starts: `(batch,)` int32 query starts, one per row.
        lengths: `(batch,)` int64 output length, one per row.

    Returns:
        A `RaggedTracks` of shape `(batch, n_tracks, None)`.
    """
    from einops import repeat

    from .._ragged import RaggedTracks
    from .._utils import lengths_to_offsets
    from ._intervals import intervals_to_tracks

    n_tracks = len(per_track)
    batch = len(lengths)
    ofsts_per_t = lengths_to_offsets(lengths)
    n_per_track = int(ofsts_per_t[-1])
    out = np.empty(n_tracks * n_per_track, np.float32)
    out_lens = repeat(np.asarray(lengths), "b -> b t", t=n_tracks)
    out_offsets = lengths_to_offsets(out_lens)

    row_idx = np.arange(batch, dtype=np.int64)
    for t, itvs in enumerate(per_track):
        intervals_to_tracks(
            offset_idxs=row_idx,
            starts=starts,
            itv_starts=itvs.starts.data,
            itv_ends=itvs.ends.data,
            itv_values=itvs.values.data,
            itv_offsets=itvs.starts.offsets,
            out=out[t * n_per_track : (t + 1) * n_per_track],
            out_offsets=ofsts_per_t,
        )
    return RaggedTracks.from_offsets(out, (batch, n_tracks, None), out_offsets)
```

Two things to get right when wiring this in:

1. `read_window` returns `(n_regions, n_samples, None)` intervals, but a batch is a slice of the flattened `(region, sample)` rows. Flatten each `RaggedIntervals` to `(n_regions * n_samples, None)` once per window and slice `[lo:hi]`, so `offset_idxs=arange(batch)` indexes the batch's own rows. Do not loop in Python over rows.
2. `RaggedTracks.from_offsets` may not be the exact constructor `_call_float32` uses (it builds a `_Flat` then converts). Read `_tracks.py:419-421` and use whatever it uses, so the returned type matches what `to_iter` already yields.

- [ ] **Step 4: Run to verify it passes**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
```

Expected: PASS, byte-identical.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): tracks-only reconstruction at byte parity

Drives the plan directly (no variant engine), rasterizing each window's
intervals with intervals_to_tracks and interleaving tracks the way the written
float-track path does. The track axis is never squeezed.

Relates to #279"
```

---

## Task 7: Mixed SVAR1 variants + tracks with re-alignment

The hardest task. Three non-obvious pieces the spec's §3 table pins down: a Python-side `variant_idxs` memmap, a **window-local** `geno_offset_idx` re-base, and a `get_diffs_sparse` call for `track_lengths`.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:875+` (the `"engine"` drive's per-window loop), `_streaming.py:1680+` (`_Svar1Backend`)
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: `_TrackBackend.read_window` (Task 3), `self._track_backend` (Task 5).
- Produces: `_Svar1Backend.geno_v_idxs -> NDArray` (a lazily-opened `np.memmap` of `variant_idxs.npy`); mixed output of shape `(batch, n_tracks, ploidy, None)`.

- [ ] **Step 1: Write the failing parity test**

```python
def test_mixed_parity_with_indels(streaming_tracks_fixture):
    """Tracks re-aligned to haplotype coordinates -- the case #279 calls out."""
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar_path,
        tracks=[f.table, f.bigwigs],
    )
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            streamed, expected = data[i], written[r, s]
            # Rank first: a missing ploidy axis must fail as a shape error,
            # not silently broadcast.
            assert np.asarray(streamed).ndim == np.asarray(expected).ndim
            for t in range(2):
                for h in range(sds.ploidy):
                    got = np.asarray(streamed[t][h])
                    want = np.asarray(expected[t][h])
                    # Length first: a mismatch is the signature of the
                    # extend_to_length span bug (spec §3.2).
                    assert len(got) == len(want), f"track {t} hap {h} length"
                    np.testing.assert_array_equal(got, want)
            seen.add((r, s))
    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


def test_fixed_output_length_parity(streaming_tracks_fixture):
    """Spec §8 matrix: `with_len(L)` as well as `with_len("ragged")`."""
    f = streaming_tracks_fixture
    L = 64
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_len(L)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_len(L)
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2, return_indices=True)))
    for i in range(len(r_idx)):
        np.testing.assert_array_equal(
            np.asarray(data[i]), np.asarray(written[int(r_idx[i]), int(s_idx[i])])
        )


def test_non_default_insertion_fill_parity(streaming_tracks_fixture):
    """Spec §8 matrix: a non-default insertion fill, not just Repeat5p()."""
    f = streaming_tracks_fixture
    # The three exported strategies are InsertionFill, Repeat5p (the default)
    # and Repeat5pNormalized; pick a non-default one.
    fill = gvl.Repeat5pNormalized()
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_insertion_fill(fill)
    written = gvl.Dataset.open(
        f.dataset_path, reference=f.reference_path
    ).with_insertion_fill(fill)
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2, return_indices=True)))
    for i in range(len(r_idx)):
        np.testing.assert_array_equal(
            np.asarray(data[i]), np.asarray(written[int(r_idx[i]), int(s_idx[i])])
        )


def test_track_with_superset_samples_parity(streaming_tracks_fixture):
    """Spec §8 matrix: a track whose samples strictly contain the dataset's.

    Extra track samples must be ignored, and the ones that remain must still be
    matched by NAME -- a positional index would silently shift every sample.
    """
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar_path,
        tracks=f.bigwigs_superset,
    )
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=2, return_indices=True)))
    for i in range(len(r_idx)):
        np.testing.assert_array_equal(
            np.asarray(data[i])[0],
            np.asarray(written[int(r_idx[i]), int(s_idx[i])])[0],
        )


def test_realign_false_drops_ploidy_axis(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=False)
    written = gvl.Dataset.open(
        f.dataset_path, reference=f.reference_path
    ).with_settings(realign_tracks=False)
    data, r_idx, s_idx = next(iter(sds.to_iter(batch_size=1, return_indices=True)))
    expected = written[int(r_idx[0]), int(s_idx[0])]
    assert np.asarray(data[0]).ndim == np.asarray(expected).ndim
```

Adjust the indexing (`streamed[t][h]`) to whatever the written `Dataset[r, s]` actually returns for tracks — read `_impl.py:1845` first and mirror it exactly.

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q -k mixed_parity or realign_false
```

Expected: FAIL — tracks are ignored on the mixed path.

- [ ] **Step 3: Expose `geno_v_idxs` on `_Svar1Backend`**

`variant_idxs.npy` is a zero-copy mmap *inside* the Rust `Svar1Store`; Python only `stat()`s it (`_streaming.py:2083-86`). Add a lazily-opened memmap property to `_Svar1Backend`:

```python
    @property
    def geno_v_idxs(self) -> NDArray:
        """The store's `variant_idxs` as a read-only memmap.

        The Rust `Svar1Store` holds this as a zero-copy mmap for its own reads;
        the fused track kernel needs it on the Python side, so open a second
        read-only view rather than copying. Opened lazily: a haplotype-only
        stream never touches it.

        Returns:
            A 1-D read-only `np.memmap` of the store's variant indices.
        """
        if self._geno_v_idxs is None:
            from genoray._types import V_IDX_TYPE

            self._geno_v_idxs = np.memmap(
                Path(self._svar_path) / "variant_idxs.npy",
                dtype=V_IDX_TYPE,
                mode="r",
            )
        return self._geno_v_idxs
```

Initialise `self._geno_v_idxs = None` in `_Svar1Backend.__init__`.

- [ ] **Step 4: Wire the track read into the per-window loop**

In `_iter_batches`'s `"engine"` branch, inside `for _contig_idx, r_idx, s_lo, s_hi in plan_jobs:` (`_streaming.py:875`), read the track window once per window — this is the composition point, and no cursor is needed because this loop already exists:

```python
                    tb = self._track_backend
                    if tb is not None:
                        s_idx_w = np.arange(s_lo, s_hi, dtype=np.intp)
                        if self._jitter > 0:
                            # Tracks MUST use the same translated bounds the
                            # engine got, or tracks and haplotypes silently
                            # disagree by up to `jitter` bases. The offsets are
                            # keyed by absolute region index, so this is a
                            # lookup, not a redraw.
                            t_starts, t_ends = self._jitter_region_bounds(
                                region_offsets, r_idx
                            )
                        else:
                            t_starts = self._regions[r_idx, 1]
                            t_ends = self._regions[r_idx, 2]
                        per_track = tb.read_window(r_idx, s_idx_w, t_starts, t_ends)
```

`read_window` already accepts the `starts`/`ends` overrides (Task 3); this is their only caller.

Read the real jitter-offset accessor before writing `_jitter_region_bounds` — the offsets are drawn once per `to_iter` and indexed by absolute region index (`_streaming.py:594-600`). If the engine drive computes translated bounds inline rather than through a helper, extract that expression into a small method and call it from both places rather than duplicating the arithmetic.

- [ ] **Step 5: Build the fused-kernel inputs per batch**

Inside the `for lo in range(0, n_rows, batch_size):` loop, after the variant batch is pulled, compute the re-based genotype views and call the fused kernel. The three pieces the spec's §3 table pins down:

```python
                        P = backend.ploidy
                        # geno_offsets: window-local, already a (starts, stops)
                        # pair from read_window -- no _as_starts_stops needed.
                        geno_offsets = np.stack([o_starts, o_stops])
                        # geno_offset_idx: the WRITTEN path uses a
                        # dataset-global ravel_multi_index (_haps.py:776). Here
                        # geno_offsets is window-local, so index it window-locally.
                        geno_offset_idx = np.arange(
                            lo * P, hi * P, dtype=np.intp
                        ).reshape(hi - lo, P)
                        diffs = get_diffs_sparse(
                            geno_offset_idx,
                            backend.geno_v_idxs,
                            geno_offsets,
                            backend._ilens,
                        )
                        track_lengths = lengths[lo:hi] - diffs.clip(max=0).min(1)
```

Then call `intervals_and_realign_track_fused` with exactly the arguments the written call site uses (`_reconstruct.py:257`), substituting:

- `geno_v_idxs=backend.geno_v_idxs`
- `geno_offsets=geno_offsets`
- `geno_offset_idx=geno_offset_idx`
- `v_starts=backend._v_starts`, `ilens=backend._ilens`
- `offset_idxs=np.arange(lo, hi)` (always per-cell; the `TrackType.ANNOT` per-region branch is unreachable from `tracks=`)
- `shifts=np.zeros((hi - lo, P), np.int32)` (zero for every v1 combination — `_haps.py:740-742`)
- `params` / `strategy_id` from the per-track `insertion_fill`

Read `_reconstruct.py:257-280` and mirror every remaining argument verbatim. Do not invent argument names.

- [ ] **Step 6: Run to verify it passes**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
```

Expected: PASS, byte-identical including the indel case.

- [ ] **Step 7: Run the full tree via sbatch**

Write `/carter/users/dlaub/gvl-279-tests.sbatch`:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=gvl-279 --time=02:00:00 --cpus-per-task=16 --mem=32G
#SBATCH --output=/carter/users/dlaub/gvl-279-%j.out
unset CLAUDE_JOB_DIR
set -euo pipefail
export TMPDIR="/local/$USER/gvl-279-$SLURM_JOB_ID"; mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/spec-279-interval-streaming
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests -q
```

Submit with `sbatch`, then check `sacct -j <id>` and the `.out` file. Do NOT run the suite in the interactive session.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): mixed SVAR1 variants + re-aligned tracks

Reuses intervals_and_realign_track_fused with no new kernel, but needs three
things the design's first draft missed: a Python-side variant_idxs memmap (the
array lives inside the Rust Svar1Store and never reached Python), a
window-local geno_offset_idx re-base (the written index is dataset-global), and
a get_diffs_sparse call for track_lengths.

The track read goes in the per-window loop the engine drive already has -- no
cursor. Under jitter>0 the track query is translated with the same per-region
offsets the engine got, or tracks and haplotypes disagree silently.

Relates to #279"
```

---

## Task 8: Guards

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:656+` (`_iter_batches` guards), `:1482+` (`with_settings`), and a new `with_insertion_fill`
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `StreamingDataset.with_insertion_fill(fill)`; `with_settings(realign_tracks=...)`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("src", ["vcf", "pgen", "svar2"])
def test_tracks_with_non_svar1_variants_raise(streaming_tracks_fixture, request, src):
    """Mixed is SVAR1-only in v1 (spec §3)."""
    f = streaming_tracks_fixture
    path = request.getfixturevalue(f"{src}_source_path")
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=path, tracks=f.bigwigs
    )
    with pytest.raises(NotImplementedError, match="SVAR1|\\.svar"):
        next(iter(sds.to_iter(batch_size=1)))


def test_variant_windows_with_realigned_tracks_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    )
    with pytest.raises(ValueError, match="realign_tracks"):
        next(iter(sds.with_seqs("variant-windows").to_iter(batch_size=1)))


def test_insertion_fill_without_realign_raises(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.svar_path, tracks=f.bigwigs
    ).with_settings(realign_tracks=False)
    with pytest.raises(ValueError, match="no effect when realign_tracks=False"):
        sds.with_insertion_fill(gvl.Repeat5p())
```

Substitute the real fixture names for the VCF/PGEN/SVAR2 sources — check `tests/dataset/conftest.py` for what exists rather than inventing `{src}_source_path`.

- [ ] **Step 2: Run to verify they fail**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q -k raise
```

- [ ] **Step 3: Implement the guards**

In `_iter_batches`, near the existing SVAR2/VCF output-mode guards (`_streaming.py:735-762`):

```python
            if self._track_backend is not None and not isinstance(
                self._backend, _Svar1Backend
            ):
                raise NotImplementedError(
                    "Streaming tracks alongside variants is currently supported"
                    " only for SVAR1 (.svar) sources. VCF/PGEN backends have no"
                    " read_window/CSR seam and SVAR2 uses a different track"
                    " kernel, so neither can supply the genotype arrays the"
                    " fused re-alignment kernel needs. Use a .svar source, or"
                    " stream tracks without variants."
                )
```

Mirror the three `_build_reconstructor` guards (`_reconstruct.py:522-546`) with the written path's exact messages: `with_seqs("variant-windows")` + tracks + `realign_tracks=True`; `with_seqs("variants")` + tracks + `realign_tracks=True`.

Add `realign_tracks` to `with_settings` (default `True`, matching `Dataset`), and add:

```python
    def with_insertion_fill(self, fill) -> "StreamingDataset":
        """Set how re-aligned tracks fill inserted bases.

        Args:
            fill: An insertion-fill strategy, or a mapping of track name to
                strategy. Matches `Dataset.with_insertion_fill`.

        Returns:
            A new `StreamingDataset` with the fill applied.

        Raises:
            ValueError: If there are no tracks, or if `realign_tracks=False`.
        """
        if self._track_backend is None:
            raise ValueError(
                "with_insertion_fill requires tracks; pass tracks= first."
            )
        if not self._realign_tracks:
            raise ValueError(
                "with_insertion_fill has no effect when realign_tracks=False."
                " Use with_settings(realign_tracks=True) first, or drop the call."
            )
        ...
```

- [ ] **Step 4: Run to verify they pass**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
pixi run -e dev pytest tests/dataset tests/unit -q
```

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): guard unsupported track combinations

Mixed variants+tracks raises NotImplementedError for VCF/PGEN/SVAR2, and the
three _build_reconstructor guards are mirrored with the written path's exact
messages. Adds with_insertion_fill and realign_tracks to with_settings.

Relates to #279"
```

---

## Task 9: `iteration_order` observability

`iteration_order` is a no-op whenever `_window_samples == n_samples`. Users will set it, measure nothing, and file a bug. Make that discoverable.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (`to_iter` docstring, `iteration_order` docstring)
- Test: `tests/dataset/test_streaming_tracks.py`

**Interfaces:**
- Consumes: Task 4's `_iteration_order`.
- Produces: `StreamingDataset.iteration_order_is_active -> bool`.

- [ ] **Step 1: Write the failing test**

```python
def test_iteration_order_reports_whether_it_is_active(streaming_tracks_fixture):
    f = streaming_tracks_fixture
    sds = gvl.StreamingDataset(f.bed, tracks=f.bigwigs)
    # Default max_mem keeps the whole sample axis in one chunk.
    assert sds._window_samples == len(f.samples)
    assert sds.iteration_order_is_active is False

    small = gvl.StreamingDataset(f.bed, tracks=f.bigwigs, max_mem="1KB")
    assert small._window_samples < len(f.samples)
    assert small.iteration_order_is_active is True
```

- [ ] **Step 2: Run to verify it fails**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q -k reports_whether
```

- [ ] **Step 3: Implement**

```python
    @property
    def iteration_order_is_active(self) -> bool:
        """Whether `iteration_order` actually changes the visit order.

        `iteration_order` only matters when the sample axis is chunked. The
        chunk size is derived from `max_mem`, and at the default
        `max_mem="512MB"` it holds the entire sample axis for any cohort below
        roughly 8.4 million — so the sample loop runs once and both orders emit
        the identical plan.

        Returns:
            `True` when `_window_samples < n_samples`, so the two orders differ.
        """
        return self._window_samples < self.shape[1]
```

Add the same caveat to the `iteration_order` argument docstring in `__init__` and to `to_iter`'s "Iteration is a fixed cartesian sweep…" paragraph.

- [ ] **Step 4: Run to verify it passes, then run the full tree via sbatch**

```bash
pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q
sbatch /carter/users/dlaub/gvl-279-tests.sbatch
```

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks.py
git commit -m "feat(streaming): expose iteration_order_is_active

iteration_order is a no-op whenever the sample axis fits in one chunk, which
is the default for any realistic cohort. Make that observable rather than
leaving users to measure no difference and file a bug.

Relates to #279"
```

---

## Task 10: Documentation

**Files:**
- Modify: `skills/genvarloader/SKILL.md`, `docs/source/dataset.md`, `docs/source/api.md`, `docs/roadmaps/streaming-dataset.md`
- Test: the `api.md` sync check below

- [ ] **Step 1: Update the skill**

In `skills/genvarloader/SKILL.md`, document on `StreamingDataset`:

- `tracks=` — accepts `IntervalTrack | Sequence[IntervalTrack]` (`BigWigs`, `Table`), same as `gvl.write`.
- Track axis is **name-sorted**, never argument order; one track is `(batch, 1, …)`, never squeezed.
- Output rank: `(b, t, ~l)` without variants or with `realign_tracks=False`; `(b, t, p, ~l)` with re-alignment.
- Mixed variants+tracks is **SVAR1-only**; VCF/PGEN/SVAR2 raise `NotImplementedError`.
- `iteration_order=` — `"auto" | "regions" | "samples"`, and the caveat that it is a no-op unless `max_mem` forces the sample axis to chunk (point at `iteration_order_is_active`).
- New methods: `with_insertion_fill`, `with_settings(realign_tracks=)`.

- [ ] **Step 2: Update prose docs**

`docs/source/dataset.md`: add a streaming-tracks section, including that v1 parity is gated on `extend_to_length=False`.

- [ ] **Step 3: Verify `api.md` is in sync**

```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```

Expected: `MISSING: none`.

- [ ] **Step 4: Update the roadmap and board**

Add a Plans-table row pointing at this file; set the status marker; mirror on the StreamingDataset project board.

- [ ] **Step 5: Commit and open the PR**

```bash
git add skills/ docs/
git commit -m "docs(streaming): document tracks= and iteration_order=

Relates to #279"
git push origin HEAD
gh pr create --base streaming --title "streaming: interval (BigWigs/Table) streaming + mixed scheduler" --draft
```

The PR must target **`streaming`**, not `main`.

---

## Follow-up issues to file

File these on the StreamingDataset project board with a `streaming:` title prefix, cross-linked to #279:

1. **Mixed variants+tracks for VCF / PGEN / SVAR2** — each needs a genotype seam that does not exist today.
2. **Parity against `extend_to_length=True` / `max_jitter>0`** — needs write-time `max_ends`, a whole-cohort scan.
3. **`annot_tracks=` streaming** — needs per-region `offset_idxs` and a non-`IntervalTrack` source type.
4. **A GIL-free track producer thread** — `RustTable::py_count` / `py_intervals` never call `py.detach`, so the track read serializes against the variant consumer. Gate on measurement, mirroring SVAR2 PR-3.
5. **`n_batches` is wrong for the SVAR2 super-batched drives** (`type: bug`) — `_iter_batch_spans` (`_streaming.py:1352-66`) assumes a flat `range(0, n_rows, batch_size)`, but `"sync"` / `"svar2_engine"` nest batches inside super-batches (`:1174-88`), so `len(dl)` under-reports whenever `sb_rows` does not divide `n_rows`. Found during spec review; not caused by this work.
