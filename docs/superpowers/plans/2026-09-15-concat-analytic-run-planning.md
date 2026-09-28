# Analytic Run Planning for `concat` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove `concat`'s materialized `(R*S*P, 2)` provenance map and its `list[Run]` — 32–768 GB at the All of Us chr22 grid — by deriving runs analytically from `order`, and strengthen the `svar2_store` test fixture that #357 found nearly degenerate.

**Architecture:** `provenance()` + `coalesce()` stay in `_concat_plan.py` as the reference implementation, no longer called on the hot path. A new `RunPlan` derives the identical run sequence directly from `order` in `O(R + S)` memory, exploiting the fact that the run-break pattern depends only on `order` and never on the region index. It is re-iterable (not a generator) because `copy_runs` and `_gather_svar_offsets` each iterate runs twice. A `slot_batches()` method lets `copy_runs` build merged offsets with `R` vectorized gathers instead of one tiny numpy call per run.

**Tech Stack:** Python 3.10–3.13, numpy, pytest, `pixi run -e dev`. No Rust changes. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-15-concat-analytic-run-planning-design.md`

## Global Constraints

- **Branch:** `feat/concat-analytic-run-planning`, already created off `main` at `d0e2f498`. One PR targeting `main`.
- **`provenance` and `coalesce` are never deleted.** They are the oracle the new path is verified against. Removing them removes the proof.
- **Behavior-preserving.** Every existing test in `tests/unit/dataset/test_concat_plan.py`, `test_concat_io.py`, and the concat dataset tests must pass **without modification**. If an existing concat test needs editing to go green, that is a signal the change is wrong — stop and report it, do not edit the test.
- **No on-disk format change.** `DATASET_FORMAT_VERSION` stays `2.0.0`. Nothing in `metadata.json` or any payload file changes bytes.
- **No public API change.** `python/genvarloader/_dataset/_concat_plan.py` is private; `gvl.concat`'s signature and semantics are unchanged.
- **No `feat:` commits and no `BREAKING CHANGE` trailers.** This releases as a PATCH (`0.43.0` → `0.43.1`) under commitizen (`version_provider = "pep621"`, `major_version_zero = true`). Use `perf(concat):`, `refactor(concat):`, `test(svar2):`, `docs(concat):`.
- **Commit message trailers**, on every commit:
  ```
  Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_015VxRqNngU7Eg1wdb1aEgD6
  ```
- **prek hooks must be installed** before committing (`prek install`). They run ruff check, ruff format, pyrefly, and commitizen.
- **Lint covers both trees:** `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev ruff format python/ tests/`. `python/` alone misses test-only issues.
- **Docstrings are Google style** (`Args:` / `Returns:` / `Raises:`), enforced by ruff pydocstyle on `python/genvarloader/` only.
- **Test data must be generated once per worktree** before any test run: `pixi run -e dev gen`. Without it a fresh worktree reports ~387 errors and ~52 failures from missing fixture files, which look like real breakage and are not. (Already done in this worktree.)
- **The chr22 reference grid**, used for every size figure in this plan: `R` = 3,734 regions, `S` = 535,662 samples, `P` = 2. `R*S` = 2,000,161,908; `R*S*P` = 4,000,323,816. A `Run` measures 184 bytes on this codebase.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `python/genvarloader/_dataset/_concat_plan.py` | Pure planning, no IO | Add `RunPlan`, `ExplicitRunPlan`, `as_plan`, `_default_order`, `_SLOT_BATCH_SLOTS`. Keep `provenance`/`coalesce` as the oracle. |
| `python/genvarloader/_dataset/_concat_io.py` | Buffered streaming IO | `copy_runs` and `gather_fixed` take a run source; `copy_runs` builds offsets from `slot_batches()` and cumsums in place. |
| `python/genvarloader/_dataset/_concat.py` | Merge orchestration | Five `provenance`+`coalesce` sites become `RunPlan`; `_gather_svar_offsets` takes a run source. |
| `tests/unit/dataset/test_concat_plan.py` | Planning unit tests | Add the differential sweep, `slot_batches` tests, the memory bound. |
| `tests/unit/dataset/test_concat_io.py` | IO unit tests | Add `RunPlan`-driven equivalence and the cumsum-aliasing pin. |
| `tests/conftest.py` | Tree-wide fixtures | **Create.** Shared `_SVAR2_REF`, `_SVAR2_VCF_2S`, `svar2_store_2s`. |
| `tests/dataset/conftest.py` | Dataset-suite fixtures | Enriched `_VCF`; `_REF` re-exported from `tests/conftest.py`. |
| Six test modules | svar2 read-path tests | Delete local `svar2_store` shadows; consume `svar2_store_2s`. |
| `tests/dataset/test_write_svar2.py` | svar2 write oracle | Rewrite the fixture-occupancy guard against the 7-cell grid. |
| `docs/source/write.md`, `skills/genvarloader/SKILL.md` | User docs | State concat's planning memory, which the change makes true. |

---

### Task 1: `RunPlan.__iter__` — the analytic planner

**Files:**
- Modify: `python/genvarloader/_dataset/_concat_plan.py`
- Test: `tests/unit/dataset/test_concat_plan.py`

**Interfaces:**
- Consumes: the existing `Run` NamedTuple, `provenance`, `coalesce` (all already in this file).
- Produces: `class RunPlan` with `__init__(self, axis: str, shape_per_ds: list[tuple[int, int]], ploidy: int, *, order: NDArray[np.int64] | None = None)`, `__iter__(self) -> Iterator[Run]`, and public attributes `axis`, `ploidy`, `shape_per_ds`, `order`. Tasks 2–4 extend and consume it.

**Background the implementer needs.** A GVL dataset stores ragged arrays over an `(R, S[, P])` C-order grid; the flat slot for `(r, s, p)` is `((r * S) + s) * P + p`. Merging decides, per *merged* flat slot, which input dataset and which *source* flat slot it came from. `order` is an `(n_merged_along_axis, 2)` int64 array giving `(dataset_idx, within_dataset_idx)` per merged position along the axis, in destination order. A **run** is a maximal span of consecutive merged slots over which the source dataset is constant *and* the source slot increases by exactly 1.

**The property this task exploits.** The break condition between adjacent merged slots is `(source dataset changes) OR (source slot does not increment)`. On the samples axis, between merged samples `j` and `j+1` the source slots are `(r*S_{d_j} + w_j)*P + P-1` and `(r*S_{d_{j+1}} + w_{j+1})*P`. A dataset change is a break unconditionally; if the dataset does not change, the `r*S_d` terms cancel and the condition reduces to `w_{j+1} == w_j + 1`, with no `r` in it. The same cancellation happens across a region boundary. So the entire break pattern is a function of `order` alone.

**Do not add a special case for the region boundary.** The pending-run carry in `_iter_samples` below already merges runs that span one, and this was verified at 2,880 configurations. Adding an explicit `cross` branch is dead code.

- [ ] **Step 1: Write the failing differential test**

Add to `tests/unit/dataset/test_concat_plan.py`. Import `RunPlan` alongside the existing imports.

```python
def _rand_order(rng, counts):
    """A random valid merged order: every (ds, within) slot exactly once."""
    rows = [(d, i) for d, c in enumerate(counts) for i in range(c)]
    rng.shuffle(rows)
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def _sorted_interleave_order(rng, counts):
    """Each input's keys sorted, merged by global key -- the real two-cohort case."""
    keys = []
    for d, c in enumerate(counts):
        ks = sorted(rng.choice(10_000, size=c, replace=False))
        keys.extend((k, d, i) for i, k in enumerate(ks))
    keys.sort()
    return np.array([(d, i) for _, d, i in keys], dtype=np.int64).reshape(-1, 2)


def test_run_plan_matches_coalesce_provenance_exhaustively():
    """RunPlan must reproduce coalesce(provenance(...)) exactly, everywhere.

    This is the correctness proof for the whole change: `provenance` + `coalesce`
    are retained purely as this oracle. A plan that is wrong in a way this sweep
    misses produces a merged dataset whose slots point at the wrong samples --
    readable, and not obviously corrupt -- so the sweep deliberately includes the
    sorted key-interleave order that models the real two-cohort merge, not only
    shuffled and block-default orders.
    """
    import itertools

    rng = np.random.default_rng(0)
    n_checked = 0

    for axis, ploidy, n_ds in itertools.product(
        ("regions", "samples"), (1, 2, 3), (1, 2, 3)
    ):
        for _ in range(40):
            counts = [int(rng.integers(0, 5)) for _ in range(n_ds)]
            if axis == "regions":
                n_samples = int(rng.integers(0, 5))
                shapes = [(c, n_samples) for c in counts]
            else:
                n_regions = int(rng.integers(0, 5))
                shapes = [(n_regions, c) for c in counts]

            for mode in ("none", "default", "shuffled", "interleaved"):
                if mode == "none":
                    order = None
                elif mode == "default":
                    order = _default_order(len(counts), counts)
                elif mode == "shuffled":
                    order = _rand_order(rng, counts)
                else:
                    order = _sorted_interleave_order(rng, counts)

                want = coalesce(provenance(axis, shapes, ploidy, order=order))
                got = list(RunPlan(axis, shapes, ploidy, order=order))
                n_checked += 1
                assert got == want, (
                    f"axis={axis} ploidy={ploidy} shapes={shapes} mode={mode}\n"
                    f"want={want[:6]}\ngot={got[:6]}"
                )

    assert n_checked == 2880, "sweep shrank; the oracle coverage is the whole point"
```

Import `_default_order` from `genvarloader._dataset._concat_plan` in this test module's import block.

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py::test_run_plan_matches_coalesce_provenance_exhaustively -v`
Expected: FAIL with `ImportError: cannot import name 'RunPlan'`.

- [ ] **Step 3: Hoist `_default_order` to module scope**

In `_concat_plan.py`, `provenance` currently defines `_default_order` as a closure that captures `n_ds`. Lift it out so `RunPlan` shares it, and take `n_ds` as a parameter:

```python
def _default_order(n_ds: int, counts: list[int]) -> NDArray[np.int64]:
    """Block-concatenation order: dataset 0's whole block, then dataset 1's, etc.

    Args:
        n_ds: Number of input datasets.
        counts: Positions along the merged axis contributed by each dataset.

    Returns:
        An ``(sum(counts), 2)`` int64 array of ``(dataset_idx, within_idx)``.
    """
    if n_ds == 0:
        return np.zeros((0, 2), np.int64)
    ds_col = np.repeat(np.arange(n_ds, dtype=np.int64), counts)
    w_col = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
    return np.stack([ds_col, w_col], axis=1)
```

Then in `provenance`, delete the nested `def _default_order(counts)` and change both call sites from `_default_order([...])` to `_default_order(n_ds, [...])`. The two sites are the `axis == "regions"` branch (`order = _default_order([shape_per_ds[d][0] for d in range(n_ds)])`) and the samples branch (`order = _default_order(per_ds_samples)`).

- [ ] **Step 4: Add `RunPlan` with `__iter__`**

Append to `_concat_plan.py`:

```python
class RunPlan:
    """Destination-ordered runs, derived from ``order`` without materializing slots.

    Equivalent to ``coalesce(provenance(axis, shape_per_ds, ploidy, order=order))``
    and pinned against it by
    ``test_run_plan_matches_coalesce_provenance_exhaustively``, but it never
    builds the ``(n_slots, 2)`` map: at the All of Us chr22 grid that map is
    32-64 GB and the run list it compresses to is 384-768 GB, because on an
    interleaved sample merge every run is one slot long.

    The whole thing rests on one property: the run-break condition is "source
    dataset changes, or source slot does not increment", and on both axes that
    reduces to a predicate over ``order`` alone -- the region index ``r`` cancels
    out. So the break pattern is computed once, in ``O(R + S)``, and the runs
    stream in ``O(1)``.

    **Re-iterable on purpose, not a generator.** ``copy_runs`` iterates runs
    twice (once for offsets, once to stream bytes) and so does
    ``_gather_svar_offsets``; a one-shot iterator would yield an empty second
    pass and silently truncate the output rather than raise.

    Args:
        axis: Either ``"regions"`` or ``"samples"``.
        shape_per_ds: ``(n_regions, n_samples)`` per input dataset, in input order.
        ploidy: Slots per ``(region, sample)`` cell. Pass ``1`` for interval
            stores, which have no ploidy axis.
        order: ``(n_merged_along_axis, 2)`` int array of ``(dataset_idx,
            within_dataset_idx)`` per merged position along ``axis``, in
            destination order. ``None`` reproduces block-concatenation.

    Raises:
        ValueError: If ``axis`` is not ``"regions"`` or ``"samples"``.
    """

    def __init__(
        self,
        axis: str,
        shape_per_ds: list[tuple[int, int]],
        ploidy: int,
        *,
        order: "NDArray[np.int64] | None" = None,
    ) -> None:
        if axis not in ("regions", "samples"):
            raise ValueError(f'axis must be "regions" or "samples", got {axis!r}')
        self.axis = axis
        self.ploidy = int(ploidy)
        self.shape_per_ds = [(int(r), int(s)) for r, s in shape_per_ds]
        n_ds = len(self.shape_per_ds)
        if order is None:
            counts = [r if axis == "regions" else s for r, s in self.shape_per_ds]
            self.order = _default_order(n_ds, counts)
        else:
            self.order = np.asarray(order, dtype=np.int64).reshape(-1, 2)

    def _segments(self) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Half-open ``[start, stop)`` spans of ``order`` with no break inside."""
        ds, w = self.order[:, 0], self.order[:, 1]
        brk = (ds[1:] != ds[:-1]) | (w[1:] != w[:-1] + 1)
        starts = np.concatenate([np.zeros(1, np.int64), np.flatnonzero(brk) + 1])
        stops = np.concatenate([starts[1:], np.array([len(self.order)], np.int64)])
        return starts, stops

    def __iter__(self) -> "Iterator[Run]":
        if self.axis == "regions":
            yield from self._iter_regions()
        else:
            yield from self._iter_samples()

    def _iter_regions(self) -> "Iterator[Run]":
        # Merged region i owns one contiguous S*P block in BOTH source and
        # destination, so a segment of `order` maps to exactly one run and no
        # carry is needed: adjacent segments are non-contiguous by construction.
        cell = self.shape_per_ds[0][1] * self.ploidy if self.shape_per_ds else 0
        if len(self.order) == 0 or cell == 0:
            return
        ds, w = self.order[:, 0], self.order[:, 1]
        for a, b in zip(*self._segments()):
            yield Run(
                src=int(ds[a]),
                src_start=int(w[a]) * cell,
                src_stop=(int(w[a]) + int(b - a)) * cell,
                dst_start=int(a) * cell,
            )

    def _iter_samples(self) -> "Iterator[Run]":
        n_regions = self.shape_per_ds[0][0] if self.shape_per_ds else 0
        n_merged = len(self.order)
        if n_regions == 0 or n_merged == 0 or self.ploidy == 0:
            return
        per_ds_samples = np.asarray([s for _, s in self.shape_per_ds], np.int64)
        ds, w = self.order[:, 0], self.order[:, 1]
        seg_starts, seg_stops = self._segments()

        # One pending run, extended whenever the next segment continues it in
        # both source and destination. This is what makes a run that spans a
        # region boundary come out merged, with no special case for that
        # boundary -- verified against the oracle at 2,880 configurations.
        pending: Run | None = None
        for r in range(n_regions):
            for a, b in zip(seg_starts, seg_stops):
                d = int(ds[a])
                s_d = int(per_ds_samples[d])
                src_start = (r * s_d + int(w[a])) * self.ploidy
                src_stop = (r * s_d + int(w[b - 1]) + 1) * self.ploidy
                dst_start = (r * n_merged + int(a)) * self.ploidy
                if (
                    pending is not None
                    and pending.src == d
                    and pending.src_stop == src_start
                    and pending.dst_start + (pending.src_stop - pending.src_start)
                    == dst_start
                ):
                    pending = Run(d, pending.src_start, src_stop, pending.dst_start)
                else:
                    if pending is not None:
                        yield pending
                    pending = Run(d, src_start, src_stop, dst_start)
        if pending is not None:
            yield pending
```

Add `RunPlan` to `__all__`, keeping it sorted: `__all__ = ["CONCAT_CHUNK_BYTES", "Run", "RunPlan", "coalesce", "provenance"]`. Add `from typing import Iterator, NamedTuple` to the imports (the file already has `from typing import NamedTuple`).

- [ ] **Step 5: Run the differential test**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py -v`
Expected: PASS, including every pre-existing test in the file unchanged.

- [ ] **Step 6: Verify the test is not vacuous**

Temporarily change `_iter_samples`'s `src_stop` from `int(w[b - 1]) + 1` to `int(w[b - 1])`, re-run, and confirm the differential test FAILS. Revert the mutation. This proves the sweep actually compares run contents rather than only lengths.

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py::test_run_plan_matches_coalesce_provenance_exhaustively -q`
Expected before revert: FAIL. Expected after revert: PASS.

- [ ] **Step 7: Lint and commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat_plan.py tests/unit/dataset/test_concat_plan.py
git commit -m "perf(concat): derive merge runs from order without materializing slots"
```

---

### Task 2: `n_slots` and `slot_batches`

**Files:**
- Modify: `python/genvarloader/_dataset/_concat_plan.py`
- Test: `tests/unit/dataset/test_concat_plan.py`

**Interfaces:**
- Consumes: `RunPlan` from Task 1.
- Produces: `RunPlan.n_slots -> int` (a property), `RunPlan.slot_batches() -> Iterator[tuple[int, NDArray[np.int64], NDArray[np.int64]]]`, `class ExplicitRunPlan`, `def as_plan(runs) -> RunPlan | ExplicitRunPlan`, and the module constant `_SLOT_BATCH_SLOTS = 1 << 20`. Task 3 and Task 4 consume all of these.

**Why `slot_batches` exists.** `copy_runs` currently builds its `lengths` array with one numpy slice-assignment *per run*. On an interleaved sample merge every run is one slot long, so that is 2.0e9 tiny numpy calls at chr22 — slower than the byte streaming it feeds. `slot_batches` yields destination-contiguous batches instead: one per region on the samples axis (`n_merged * ploidy` slots, 8.6 MB of int64 at chr22), and fixed-size chunks of a run on the regions axis. A regions-axis run can cover the whole merged grid, so it **must** be chunked — a naive `np.arange(run.src_start, run.src_stop)` there would be 32 GB and defeat the entire change.

**This deliberately diverges from the spec.** The spec's `slot_batches` paragraph says the regions axis yields "one per run"; that is wrong, for the reason just given, and chunking is the correction. The spec has been amended to match. Do not "restore" the one-batch-per-run form.

**Why `ExplicitRunPlan` exists.** `tests/unit/dataset/test_concat_io.py` builds small `Run` lists by hand to test IO behavior independently of planning, and that separation is correct. `as_plan` normalizes both inputs so the IO layer has exactly one consuming path.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/dataset/test_concat_plan.py`:

```python
def test_slot_batches_reproduce_the_provenance_map():
    """Concatenating the batches must rebuild `provenance` row for row.

    `copy_runs` gathers source lengths through these batches, so a batch whose
    slots are right but whose dataset column is wrong would read the correct
    offsets out of the wrong file -- a silent wrong merge, not a crash.
    """
    for axis, shapes, ploidy in (
        ("regions", [(3, 4), (2, 4)], 2),
        ("samples", [(3, 2), (3, 3)], 2),
        ("samples", [(2, 1), (2, 2)], 1),
    ):
        counts = [s for _, s in shapes] if axis == "samples" else [r for r, _ in shapes]
        interleaved = _sorted_interleave_order(np.random.default_rng(1), counts)
        for order in (None, interleaved):
            plan = RunPlan(axis, shapes, ploidy, order=order)
            want = provenance(axis, shapes, ploidy, order=order)
            got = np.zeros_like(want)
            seen = np.zeros(len(want), bool)
            for dst_start, ds_vec, slots in plan.slot_batches():
                n = len(slots)
                assert len(ds_vec) == n
                got[dst_start : dst_start + n, 0] = ds_vec
                got[dst_start : dst_start + n, 1] = slots
                seen[dst_start : dst_start + n] = True
            assert seen.all(), f"{axis} {shapes} left slots uncovered"
            np.testing.assert_array_equal(got, want)


def test_n_slots_matches_provenance_length():
    for axis, shapes, ploidy in (
        ("regions", [(3, 4), (2, 4)], 2),
        ("samples", [(3, 2), (3, 3)], 2),
        ("regions", [(0, 4), (2, 4)], 1),
        ("samples", [(0, 2), (0, 3)], 2),
    ):
        plan = RunPlan(axis, shapes, ploidy)
        assert plan.n_slots == len(provenance(axis, shapes, ploidy))


def test_slot_batches_chunk_large_region_runs(monkeypatch):
    """A regions-axis run can span the whole grid, so batches must be chunked.

    The production cap is 1Mi slots, far above anything a unit test can build,
    so this shrinks it and checks the chunking actually happens. Asserting only
    `max(sizes) <= cap` would pass vacuously on a single batch -- including
    against the unchunked implementation this test exists to rule out.
    """
    from genvarloader._dataset import _concat_plan

    monkeypatch.setattr(_concat_plan, "_SLOT_BATCH_SLOTS", 5)

    # One input, identity order -> exactly one run over 4 * 3 * 2 = 24 slots.
    plan = RunPlan("regions", [(4, 3)], 2)
    assert len(list(plan)) == 1

    batches = list(plan.slot_batches())
    assert [len(slots) for _, _, slots in batches] == [5, 5, 5, 5, 4]
    assert plan.n_slots == 24
    # Destination stays contiguous across every chunk boundary.
    assert [dst for dst, _, _ in batches] == [0, 5, 10, 15, 20]
    np.testing.assert_array_equal(
        np.concatenate([slots for _, _, slots in batches]), np.arange(24)
    )


def test_explicit_run_plan_round_trips_a_hand_built_list():
    runs = [Run(0, 0, 2, 0), Run(1, 5, 7, 2)]
    plan = as_plan(runs)
    assert list(plan) == runs
    assert plan.n_slots == 4
    assert as_plan(plan) is plan
    batches = list(plan.slot_batches())
    assert [b[0] for b in batches] == [0, 2]
    np.testing.assert_array_equal(batches[0][2], [0, 1])
    np.testing.assert_array_equal(batches[1][1], [1, 1])


def test_run_plan_is_re_iterable():
    """copy_runs iterates twice; a generator here would silently truncate output."""
    plan = RunPlan("samples", [(2, 2), (2, 1)], 2)
    assert list(plan) == list(plan)
    assert len(list(plan)) > 0
```

Add `RunPlan`, `as_plan` and `Run` to the module's import block if not already present.

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py -k "slot_batches or n_slots or explicit or re_iterable" -v`
Expected: FAIL with `AttributeError: 'RunPlan' object has no attribute 'slot_batches'` / `ImportError` for `as_plan`.

- [ ] **Step 3: Implement**

Add the constant next to `CONCAT_CHUNK_BYTES` in `_concat_plan.py`:

```python
_SLOT_BATCH_SLOTS = 1 << 20
"""Slots per `slot_batches` chunk on the regions axis: 8 MiB of int64 indices.

A regions-axis run can cover the whole merged grid (4.0e9 slots at chr22), so
emitting one batch per run would rebuild exactly the array this module exists to
avoid. The samples axis is naturally bounded at `n_merged * ploidy` instead.
"""
```

Add to `RunPlan`:

```python
@property
def n_slots(self) -> int:
    """Total merged flat slots this plan covers, computed arithmetically."""
    if not self.shape_per_ds:
        return 0
    if self.axis == "regions":
        return len(self.order) * self.shape_per_ds[0][1] * self.ploidy
    return self.shape_per_ds[0][0] * len(self.order) * self.ploidy


def slot_batches(
    self,
) -> "Iterator[tuple[int, NDArray[np.int64], NDArray[np.int64]]]":
    """Yield ``(dst_start, src_ds, src_slots)`` batches in destination order.

    Each batch describes a destination-contiguous span: ``src_ds[i]`` and
    ``src_slots[i]`` are the origin of merged slot ``dst_start + i``.
    Concatenating every batch in order rebuilds :func:`provenance`'s output
    exactly, which is what pins this method.

    Yields:
        ``(dst_start, src_ds, src_slots)``, where the two arrays are int64
        and equal in length.
    """
    if self.axis == "regions":
        for run in self:
            pos, dst = run.src_start, run.dst_start
            while pos < run.src_stop:
                n = min(_SLOT_BATCH_SLOTS, run.src_stop - pos)
                yield (
                    dst,
                    np.full(n, run.src, np.int64),
                    np.arange(pos, pos + n, dtype=np.int64),
                )
                pos += n
                dst += n
        return

    n_regions = self.shape_per_ds[0][0] if self.shape_per_ds else 0
    n_merged = len(self.order)
    if n_regions == 0 or n_merged == 0 or self.ploidy == 0:
        return
    per_ds_samples = np.asarray([s for _, s in self.shape_per_ds], np.int64)
    ds, w = self.order[:, 0], self.order[:, 1]
    s_d = per_ds_samples[ds]
    p = np.arange(self.ploidy, dtype=np.int64)
    # `order` is per merged SAMPLE; each contributes `ploidy` adjacent slots.
    ds_vec = np.repeat(ds, self.ploidy)
    for r in range(n_regions):
        base = (r * s_d + w) * self.ploidy
        slots = (base[:, None] + p[None, :]).reshape(-1)
        yield (r * n_merged * self.ploidy, ds_vec, slots)
```

Then append the adapter and the normalizer:

```python
class ExplicitRunPlan:
    """A hand-built run list presented through :class:`RunPlan`'s interface.

    The IO layer takes either this or a :class:`RunPlan`, so its unit tests can
    exercise streaming with a two-run list without constructing a merge.

    Args:
        runs: Destination-ordered runs. Materialized, so it may be any iterable.
    """

    def __init__(self, runs: "Iterable[Run]") -> None:
        self._runs = list(runs)

    def __iter__(self) -> "Iterator[Run]":
        return iter(self._runs)

    @property
    def n_slots(self) -> int:
        """Total merged flat slots covered by the run list."""
        return sum(r.src_stop - r.src_start for r in self._runs)

    def slot_batches(
        self,
    ) -> "Iterator[tuple[int, NDArray[np.int64], NDArray[np.int64]]]":
        """Yield one ``(dst_start, src_ds, src_slots)`` batch per run."""
        for r in self._runs:
            n = r.src_stop - r.src_start
            yield (
                r.dst_start,
                np.full(n, r.src, np.int64),
                np.arange(r.src_start, r.src_stop, dtype=np.int64),
            )


def as_plan(
    runs: "RunPlan | ExplicitRunPlan | Sequence[Run]",
) -> "RunPlan | ExplicitRunPlan":
    """Normalize a run source so the IO layer has one consuming path.

    Args:
        runs: A plan, or a re-iterable sequence of runs. A one-shot generator is
            deliberately not accepted: ``copy_runs`` iterates its runs twice.

    Returns:
        ``runs`` itself when it is already a plan, else an
        :class:`ExplicitRunPlan` wrapping it.
    """
    if isinstance(runs, (RunPlan, ExplicitRunPlan)):
        return runs
    return ExplicitRunPlan(runs)
```

Extend imports to `from typing import Iterable, Iterator, NamedTuple, Sequence` and `__all__` to `["CONCAT_CHUNK_BYTES", "ExplicitRunPlan", "Run", "RunPlan", "as_plan", "coalesce", "provenance"]`.

- [ ] **Step 4: Run the tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py -v`
Expected: PASS, all tests including Task 1's sweep.

- [ ] **Step 5: Commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat_plan.py tests/unit/dataset/test_concat_plan.py
git commit -m "perf(concat): add vectorized slot batches and analytic slot count to RunPlan"
```

---

### Task 3: `copy_runs` builds offsets from batches

**Files:**
- Modify: `python/genvarloader/_dataset/_concat_io.py:39-89` (`copy_runs`), `:92-124` (`gather_fixed`)
- Test: `tests/unit/dataset/test_concat_io.py`

**Interfaces:**
- Consumes: `RunPlan`, `ExplicitRunPlan`, `as_plan` from Task 2; `Run` from `_concat_plan`.
- Produces: `copy_runs(srcs: list[Path], dst: Path, runs: RunPlan | ExplicitRunPlan | Sequence[Run], src_offsets: list[NDArray[np.int64]], itemsize: int) -> NDArray[np.int64]` and `gather_fixed(srcs: list[Path], dst: Path, runs: RunPlan | ExplicitRunPlan | Sequence[Run], record_bytes: int) -> None`. Return values and written bytes are unchanged; only the accepted `runs` type widens.

**What changes and why.** Today `copy_runs` allocates a separate `lengths` array of `n_slots` int64 (16.0 GB at chr22) and fills it one run at a time. It will instead write lengths directly into `merged[1:]` and cumsum in place, filling from `slot_batches()`. The `n_slots = sum(...)` pre-pass is replaced by `plan.n_slots`. The byte-streaming loop is untouched.

**The aliasing detail.** `np.cumsum(merged[1:], out=merged[1:])` reads and writes the same memory. numpy supports exact aliasing here, but it is load-bearing and gets its own pin rather than an assumption.

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/dataset/test_concat_io.py`:

```python
def test_copy_runs_accepts_a_run_plan_and_matches_an_explicit_list(tmp_path):
    """Driving copy_runs from RunPlan must be byte-identical to the old path."""
    from genvarloader._dataset._concat_plan import RunPlan, coalesce, provenance

    shapes = [(2, 2), (2, 1)]
    axis, ploidy = "samples", 2
    n_src = [r * s * ploidy for r, s in shapes]

    srcs, offsets = [], []
    for d, n in enumerate(n_src):
        lens = np.arange(1, n + 1, dtype=np.int64)
        off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        payload = np.arange(off[-1], dtype=np.int32) + d * 1000
        p = tmp_path / f"src{d}.bin"
        _write_raw(p, payload)
        srcs.append(p)
        offsets.append(off)

    out_plan = tmp_path / "plan.npy"
    out_list = tmp_path / "list.npy"
    plan = RunPlan(axis, shapes, ploidy)
    merged_plan = copy_runs(srcs, out_plan, plan, offsets, itemsize=4)
    merged_list = copy_runs(
        srcs, out_list, coalesce(provenance(axis, shapes, ploidy)), offsets, itemsize=4
    )

    np.testing.assert_array_equal(merged_plan, merged_list)
    assert out_plan.read_bytes() == out_list.read_bytes()


def test_copy_runs_in_place_cumsum_matches_out_of_place(tmp_path):
    """merged[1:] is cumsummed into itself; pin that against the naive form."""
    lens = np.array([3, 0, 5, 2, 7], np.int64)
    off = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    payload = np.arange(off[-1], dtype=np.int32)
    p = tmp_path / "src.bin"
    _write_raw(p, payload)

    runs = [Run(0, 0, 5, 0)]
    merged = copy_runs([p], tmp_path / "out.npy", runs, [off], itemsize=4)

    want = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
    np.testing.assert_array_equal(merged, want)


def test_gather_fixed_from_a_run_plan_is_byte_identical_to_the_oracle(tmp_path):
    """gather_fixed must produce the same bytes from a plan as from a run list."""
    from genvarloader._dataset._concat_plan import RunPlan, coalesce, provenance

    shapes, axis, ploidy = [(2, 2), (3, 2)], "regions", 1
    srcs = []
    for d, (r, sm) in enumerate(shapes):
        p = tmp_path / f"g{d}.bin"
        _write_raw(p, np.arange(r * sm * ploidy, dtype=np.int32) + d * 100)
        srcs.append(p)

    plan = RunPlan(axis, shapes, ploidy)
    out_plan, out_list = tmp_path / "plan.bin", tmp_path / "list.bin"
    gather_fixed(srcs, out_plan, plan, record_bytes=4)
    gather_fixed(
        srcs, out_list, coalesce(provenance(axis, shapes, ploidy)), record_bytes=4
    )

    assert out_plan.read_bytes() == out_list.read_bytes()
    assert out_plan.stat().st_size == plan.n_slots * 4
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_io.py -k "run_plan or in_place_cumsum" -v`
Expected: FAIL — `copy_runs` raises `AttributeError` because a `RunPlan` has no `__len__`/is consumed as a list, or `TypeError` on `sum(...)`.

- [ ] **Step 3: Rewrite `copy_runs`**

Replace the body of `copy_runs` (`_concat_io.py:39-89`) with:

```python
def copy_runs(
    srcs: list[Path],
    dst: Path,
    runs: "RunPlan | ExplicitRunPlan | Sequence[Run]",
    src_offsets: list[NDArray[np.int64]],
    itemsize: int,
) -> NDArray[np.int64]:
    """Stream a ragged payload through a run plan and return merged offsets.

    Args:
        srcs: Payload file per input dataset (raw, headerless arrays).
        dst: Destination payload file, created or truncated.
        runs: Destination-ordered runs — a :class:`._concat_plan.RunPlan`, or any
            re-iterable sequence of :class:`._concat_plan.Run`. Iterated twice,
            so a one-shot generator is not accepted.
        src_offsets: Cumulative offsets per input dataset, each of length
            ``n_source_slots + 1``, in elements (not bytes).
        itemsize: Bytes per element of the payload dtype.

    Returns:
        Merged cumulative offsets, length ``total_merged_slots + 1``, in elements.
    """
    plan = as_plan(runs)
    n_slots = plan.n_slots

    # Lengths are written straight into the output buffer and cumsummed in
    # place: a separate `lengths` array would be another n_slots int64s, 16.0 GB
    # at the All of Us chr22 grid. Filling happens per destination-contiguous
    # BATCH rather than per run, because on an interleaved sample merge every
    # run is one slot long and a per-run numpy slice-assignment would be 2.0e9
    # scalar calls -- slower than the byte streaming it feeds.
    merged = np.empty(n_slots + 1, np.int64)
    merged[0] = 0
    lengths = merged[1:]
    for dst_start, ds_vec, slots in plan.slot_batches():
        n = len(slots)
        out = lengths[dst_start : dst_start + n]
        for d in np.unique(ds_vec):
            m = ds_vec == d
            off = src_offsets[int(d)]
            sl = slots[m]
            out[m] = off[sl + 1] - off[sl]
    np.cumsum(lengths, out=lengths)

    handles: dict[int, object] = {}
    try:
        with open(dst, "wb") as fo:
            for r in plan:
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
```

- [ ] **Step 4: Widen `gather_fixed`**

In `gather_fixed` (`_concat_io.py:92`), change the `runs` annotation to `"RunPlan | ExplicitRunPlan | Sequence[Run]"`, add `plan = as_plan(runs)` as the first statement, change `for r in runs:` to `for r in plan:`, and update the `runs:` docstring line to match `copy_runs`'. Nothing else changes — it is a single-pass consumer.

Update the import at `_concat_io.py:19` to:

```python
from ._concat_plan import (
    CONCAT_CHUNK_BYTES,
    ExplicitRunPlan,
    Run,
    RunPlan,
    as_plan,
)
```

and add `Sequence` to the `typing` imports (`from typing import Sequence`).

- [ ] **Step 5: Run the IO tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_io.py -v`
Expected: PASS — the nine pre-existing tests must pass **unmodified**, since they pass plain `list[Run]` and `as_plan` accepts that.

- [ ] **Step 6: Commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add python/genvarloader/_dataset/_concat_io.py tests/unit/dataset/test_concat_io.py
git commit -m "perf(concat): build merged offsets from slot batches, cumsum in place"
```

---

### Task 4: Route `_concat.py`'s five sites through `RunPlan`

**Files:**
- Modify: `python/genvarloader/_dataset/_concat.py:203-238` (`_gather_svar_offsets`), `:340-344`, `:529-530`, `:557-558`, `:585-586`, `:631-634`
- Test: existing concat dataset tests (no new test file)

**Interfaces:**
- Consumes: `RunPlan`, `as_plan` from Task 2; `copy_runs`, `gather_fixed` from Task 3.
- Produces: nothing new. This task removes every `provenance` + `coalesce` call from the merge path.

**All five sites, exactly.** Three are the unbounded ones (#403); two are already bounded at `(R, 2)` and move for consistency, so there is one obvious way to plan a merge.

- [ ] **Step 1: Update the import**

`_concat.py:16` currently reads `from ._concat_plan import Run, coalesce, provenance`. Change to:

```python
from ._concat_plan import ExplicitRunPlan, Run, RunPlan, as_plan
```

`coalesce` and `provenance` stay in `_concat_plan.py` — they are just no longer imported here.

- [ ] **Step 2: Convert `_gather_svar_offsets`**

In `_concat.py:203-238`, change the signature's `runs: list[Run]` to `runs: "RunPlan | ExplicitRunPlan | Sequence[Run]"`, and replace the two lines

```python
        out = np.empty(sum(r.src_stop - r.src_start for r in runs), np.int64)
        for r in runs:
```

with

```python
        out = np.empty(plan.n_slots, np.int64)
        for r in plan:
```

adding `plan = as_plan(runs)` as the first statement of the function body, above `n_src_slots = ...`. The docstring's reference to the space "``runs`` was coalesced over" should read "the space ``runs`` plans over (see ``_concat_plan``)", since nothing is coalesced any more.

- [ ] **Step 3: Pin `_gather_svar_offsets` equivalence**

The spec requires byte-identical output from a plan versus a run list for all three consumers; this is the third. It goes in `tests/unit/dataset/test_concat_io.py` alongside the other two rather than in a new file — it is a gather consumer, even though the function itself lives in `_concat.py`.

```python
def test_gather_svar_offsets_from_a_run_plan_is_byte_identical(tmp_path):
    """The svar offsets gather must agree with the retained oracle.

    This one cannot go through `gather_fixed`: its slot axis is nested inside
    two leading planes, so a slot's start and stop live `n_slots` elements
    apart. It therefore has its own run-consuming loop, and needs its own pin.
    """
    from genvarloader._dataset._concat import _gather_svar_offsets
    from genvarloader._dataset._concat_plan import RunPlan, coalesce, provenance

    shapes, axis, ploidy = [(2, 2), (2, 1)], "samples", 2

    paths = []
    for d, (r, sm) in enumerate(shapes):
        n = r * sm * ploidy
        p = tmp_path / f"ds{d}"
        (p / "genotypes").mkdir(parents=True)
        starts = np.arange(n, dtype=np.int64) + d * 100
        planes = np.stack([starts, starts + 1])
        _write_raw(p / "genotypes" / "offsets.npy", planes)
        paths.append(p)

    out_plan, out_list = tmp_path / "plan", tmp_path / "list"
    out_plan.mkdir()
    out_list.mkdir()

    _gather_svar_offsets(paths, out_plan, RunPlan(axis, shapes, ploidy), shapes, ploidy)
    _gather_svar_offsets(
        paths,
        out_list,
        coalesce(provenance(axis, shapes, ploidy)),
        shapes,
        ploidy,
    )

    assert (out_plan / "offsets.npy").read_bytes() == (
        out_list / "offsets.npy"
    ).read_bytes()
```

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_io.py -k gather_svar_offsets -v`
Expected: PASS once Step 2's conversion is in place.

- [ ] **Step 4: Convert the region_runs site (`:340-344`)**

Replace:

```python
    region_runs = (
        coalesce(provenance("regions", [(r, 1) for r, _ in shapes], 1, order=order))
        if axis == "regions"
        else None
    )
```

with:

```python
    region_runs = (
        RunPlan("regions", [(r, 1) for r, _ in shapes], 1, order=order)
        if axis == "regions"
        else None
    )
```

Leave the explanatory comment above it intact, and leave both `assert region_runs is not None` guards in place.

- [ ] **Step 5: Convert the three unbounded sites**

At `:529-530` (the `pgen_vcf` backend) and `:557-558` (the `svar` backend), replace each two-line pair

```python
            prov = provenance(axis, shapes, ploidy, order=order)
            runs = coalesce(prov)
```

with the single line

```python
            runs = RunPlan(axis, shapes, ploidy, order=order)
```

At `:585-586` (per-sample tracks), replace

```python
            t_prov = provenance(axis, shapes, 1, order=order)
            t_runs = coalesce(t_prov)
```

with

```python
            t_runs = RunPlan(axis, shapes, 1, order=order)
```

Keep `t_runs` hoisted above `for name in ref.tracks:` — it is built once regardless of track count, and that must not regress.

- [ ] **Step 6: Convert the annot-track site (`:631-634`)**

Replace

```python
a_prov = provenance("regions", [(r, 1) for r, _ in shapes], 1, order=order)
a_runs = coalesce(a_prov)
```

with

```python
a_runs = RunPlan("regions", [(r, 1) for r, _ in shapes], 1, order=order)
```

- [ ] **Step 7: Correct the two comments this change makes false**

`_concat.py` documents the old behavior in prose in two places, and both become wrong here. Neither is caught by any test.

First, the `region_runs` comment at `:331-339` warns about "a future edit that changes one `provenance` call and not the other". There is no `provenance` call left after Step 3. Change that clause to "a future edit that changes one `RunPlan` construction and not the other".

Second, and more important, `_merge_svar2_ranges`'s docstring at `:259-265` currently reads:

```
    After this change an svar2 ``concat`` with no per-sample tracks holds nothing
    ``R x S``-sized: the ``(R*S*P, 2)`` ``provenance`` array (64 GB at All of Us
    chr22) and the ``list[Run]`` ``coalesce`` builds from it (~204 bytes per run,
    which on an interleaved sample merge degenerates to one run per slot) are both
    gone from this path. Per-sample tracks still plan in core at
    ``_concat.py:465`` -- 32 GB plus a ~424 GB run list at the same projection --
    so ``concat`` is bounded only for tracks-free datasets. Tracked separately.
```

Every claim in its second half is now false: per-sample tracks no longer plan in core, `concat` is no longer bounded only for tracks-free datasets, and the thing "tracked separately" is the issue this branch closes. (The `:465` line reference was already stale before this change — the site is at `:585` — and `~204 bytes per run` disagrees with the 184 bytes measured for this codebase's `Run`. Do not preserve either figure.) Replace the whole paragraph with:

```
    No stage of this merge holds anything ``R x S``-sized. The
    ``(R*S*P, 2)`` ``provenance`` array (64 GB at the All of Us chr22 grid) and
    the ``list[Run]`` ``coalesce`` built from it (184 bytes per run, degenerating
    to one run per slot on an interleaved sample merge) are gone from every path
    in this module, per-sample tracks included: ``RunPlan`` derives the same runs
    from ``order`` alone. See ``_concat_plan.RunPlan``.
```

- [ ] **Step 8: Confirm no planning calls remain on the merge path**

Run: `pixi run -e dev python -c "import re; s=open('python/genvarloader/_dataset/_concat.py').read(); print('provenance:', s.count('provenance(')); print('coalesce:', s.count('coalesce('))"`
Expected: both counts `0`.

- [ ] **Step 9: Run the concat suite**

Run: `pixi run -e dev pytest tests/unit/dataset tests/dataset -k concat -v`
Expected: PASS, with **zero test files modified** in this task. If a concat test fails, the change is not behavior-preserving — report it rather than editing the test.

- [ ] **Step 10: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS. The measured baseline on this branch is **1250 passed, 61 skipped, 4 xfailed** (macOS/darwin, `pixi run -e dev pytest tests -q`, after `pixi run -e dev gen`). Counts are platform-dependent — CI runs slow/torch tiers this does not — so treat a differing skip count on another machine as environment, and a differing *failure* count as a real regression.

- [ ] **Step 11: Commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
git add python/genvarloader/_dataset/_concat.py
git commit -m "perf(concat): plan every merge through RunPlan, dropping the provenance map"
```

---

### Task 5: Pin the memory bound

**Files:**
- Test: `tests/unit/dataset/test_concat_plan.py`

**Interfaces:**
- Consumes: `RunPlan` from Tasks 1–2.
- Produces: nothing consumed by later tasks.

**Why this is its own test rather than a comment.** The whole change is invisible to every behavioral test: `RunPlan` and `coalesce(provenance(...))` produce identical output, so a future edit that quietly reintroduces a materialized slot-space array would keep the suite green while restoring the 32 GB allocation. This test is the only thing that fails in that case.

- [ ] **Step 1: Write the failing test**

```python
def test_run_plan_never_materializes_the_slot_space():
    """Peak allocation must stay O(R + S), not O(R * S * P).

    The old path on this grid allocates a (800_000, 2) int64 provenance map
    (12.8 MB) and coalesces it into 800_000 one-slot Runs (~147 MB at 184 bytes
    each). Both are invisible to every behavioural test, because RunPlan yields
    exactly the same runs -- so this bound is what stops a future edit from
    silently restoring them.
    """
    import tracemalloc

    shapes = [(200, 1000), (200, 1000)]
    # Interleaved samples: the worst case, one run per slot.
    order = np.array(
        [(d, i) for i in range(1000) for d in (0, 1)], dtype=np.int64
    ).reshape(-1, 2)

    tracemalloc.start()
    try:
        plan = RunPlan("samples", shapes, 2, order=order)
        n_runs = sum(1 for _ in plan)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert plan.n_slots == 200 * 2000 * 2
    assert n_runs == plan.n_slots // 2, "interleaved merge should give one run per cell"
    assert peak < (1 << 20), f"peak {peak} bytes: the slot space is being materialized"
```

The `n_runs` expectation: with `ploidy=2` and fully interleaved samples, each merged sample's two ploidy slots are contiguous and every sample boundary breaks, so there is one run per `(region, sample)` cell — `200 * 2000 = 400_000` runs covering `800_000` slots.

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/unit/dataset/test_concat_plan.py::test_run_plan_never_materializes_the_slot_space -v`
Expected: PASS (Tasks 1–2 already satisfy it).

- [ ] **Step 3: Verify it is not vacuous**

Temporarily add `_ = provenance("samples", shapes, 2, order=order)` immediately after the `RunPlan(...)` construction inside the `tracemalloc` block and re-run.
Expected: FAIL on the `peak < (1 << 20)` assertion. Remove the line and confirm PASS.

- [ ] **Step 4: Commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add tests/unit/dataset/test_concat_plan.py
git commit -m "test(concat): pin RunPlan's peak allocation below the materialized slot space"
```

---

### Task 6: Consolidate the seven `svar2_store` fixtures

**Files:**
- Create: `tests/conftest.py`
- Modify: `tests/test_svar2_reconstruct.py`, `tests/unit/dataset/test_svar2_store.py`, `tests/unit/dataset/test_svar2_link.py`, `tests/dataset/test_svar2_readbound_variants.py`, `tests/dataset/test_svar2_readbound_haps.py`, `tests/dataset/test_svar2_readbound_diffs.py`, `tests/dataset/conftest.py`
- **Do not modify:** `tests/dataset/test_svar2_readbound_tracks.py` (see "The situation" below)

**Interfaces:**
- Consumes: nothing from earlier tasks. Independent of Tasks 1–5.
- Produces: `tests/conftest.py` exporting `_SVAR2_REF` and the `svar2_store_2s` fixture. Task 7 adds the enriched 3-sample VCF on top.

**This task changes no test content.** It is pure deduplication, and its gate is that the full suite stays green with **zero assertion edits**. Do not enrich anything here — that is Task 7.

**The situation, as measured.** `svar2_store` is defined **eight** times. `tests/dataset/conftest.py:52` builds the **3-sample** store (`S0, S1, S2`) from `vcf_and_ref` — that is Task 7's fixture, untouched here. The other seven are **2-sample** (`S0, S1`), each with a private copy of the same 40 bp `_REF` and a `_VCF`, each re-running `samtools faidx` + `bcftools view` + `bcftools index`.

**Only six of those seven are duplicates.** Their `_VCF` bodies are byte-identical (the same three variants at chr1:3, :7, :12):

| module | line | fold in? |
|---|---|---|
| `tests/test_svar2_reconstruct.py` | 34 | yes |
| `tests/unit/dataset/test_svar2_store.py` | 28 | yes |
| `tests/unit/dataset/test_svar2_link.py` | 59 | yes |
| `tests/dataset/test_svar2_readbound_variants.py` | 36 | yes |
| `tests/dataset/test_svar2_readbound_haps.py` | 34 | yes |
| `tests/dataset/test_svar2_readbound_diffs.py` | 38 | yes |
| `tests/dataset/test_svar2_readbound_tracks.py` | 39 | **NO** |

`test_svar2_readbound_tracks.py` carries a **four**-variant `_VCF` — it adds `chr1 10 . G C ... 1|1 1|0` to the other three. Folding it onto the shared fixture would silently change the data its assertions were written against, which is exactly what this task's zero-assertion-edit gate forbids. Leave its `_VCF` and its `svar2_store` fixture exactly where they are. It may take `_SVAR2_REF` from the shared module (its `_REF` *is* identical), but nothing else.

**Also leave alone**, all purpose-built rather than duplicated: `svar2_store_dense_snp` in **three** modules (`test_svar2_readbound_variants.py:140`, `test_svar2_readbound_haps.py:280`, `test_svar2_readbound_diffs.py:83`) and `svar2_store_unsorted` (`test_write_svar2.py:526`).

Four of the six folded shadows live in `tests/dataset/`, where they silently shadow `tests/dataset/conftest.py`'s 3-sample fixture of the same name. Renaming the shared 2-sample fixture to `svar2_store_2s` makes that distinction explicit instead of accidental — that is the point of the rename, not cosmetics.

- [ ] **Step 1: Create `tests/conftest.py`**

```python
"""Fixtures shared across the whole test tree."""

import subprocess
from pathlib import Path

import pytest

_SVAR2_REF = "ACAGTACATGGGTACTAGCTAGGCTAACCGGTTAACCGGT"
"""40 bp synthetic chr1 backing every .svar2 fixture.

Positions used by the VCFs below: 3=A, 7=C, 9=T, 12=GTA, 17=A, 19=C, 30=G.
"""

_SVAR2_VCF_2S = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t1|1
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t1|1\t0|1
"""


def _build_svar2(vcf_text: str, samples: list[str], d: Path, name: str) -> Path:
    """Write a VCF + FASTA under ``d`` and convert them to a .svar2 store.

    Args:
        vcf_text: Full VCF text, including header.
        samples: Sample names to convert, in the order genoray should store them.
        d: Directory to build in.
        name: Basename of the resulting store directory.

    Returns:
        Path to the finished ``.svar2`` store.
    """
    from genoray import _core

    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR2_REF}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(vcf_text)
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    out = d / name
    _core.run_conversion_pipeline(
        str(bcf),
        str(ref),
        ["chr1"],
        str(out),
        samples,
        25_000,
        2,
        1,
        8 * 1024 * 1024,
    )
    assert (out / "meta.json").exists(), "conversion did not finish"
    return out


@pytest.fixture(scope="module")
def svar2_store_2s(tmp_path_factory) -> Path:
    """A two-sample (S0, S1) .svar2 store.

    Named apart from the dataset suite's three-sample ``svar2_store`` on purpose:
    three of this fixture's former copies lived in ``tests/dataset/`` and silently
    shadowed ``tests/dataset/conftest.py``'s same-named three-sample fixture.
    """
    d = tmp_path_factory.mktemp("svar2_2s")
    return _build_svar2(_SVAR2_VCF_2S, ["S0", "S1"], d, "store")
```

- [ ] **Step 2: Delete the six shadows and repoint their consumers**

In each of the **six** modules marked "fold in" above — `tests/test_svar2_reconstruct.py`, `tests/unit/dataset/test_svar2_store.py`, `tests/unit/dataset/test_svar2_link.py`, `tests/dataset/test_svar2_readbound_variants.py`, `tests/dataset/test_svar2_readbound_haps.py`, `tests/dataset/test_svar2_readbound_diffs.py` (**not** `test_svar2_readbound_tracks.py`):

1. Delete the module-level `_REF` and `_VCF` constants.
2. Delete the whole `svar2_store` fixture function and its `@pytest.fixture` decorator.
3. Rename every `svar2_store` **parameter** in that module's test functions to `svar2_store_2s`, and every in-body use of the name.
4. Leave `svar2_store_dense_snp` (and its own `_VCF_*` constant) untouched in the two modules that define it. If it references `_REF`, change that reference to `_SVAR2_REF` imported from `tests.conftest` — do **not** delete the constant it needs.
5. Drop any now-unused `import subprocess` / `from pathlib import Path` / `from genoray import _core` left behind. `ruff check` will name them.

- [ ] **Step 3: Repoint `tests/dataset/conftest.py`**

Delete its local `_REF` and import the shared one instead, leaving its `_VCF` (the 3-sample one) and its `vcf_and_ref` / `svar2_store` / `svar1_store` fixtures in place:

```python
from tests.conftest import _SVAR2_REF as _REF
```

If that import path does not resolve under this repo's pytest configuration, define `tests/dataset/conftest.py`'s `_REF` as `_REF = _SVAR2_REF` by importing from the `conftest` module pytest already loaded, or simply keep `_REF` inline in `tests/dataset/conftest.py` with a comment pointing at `tests/conftest.py::_SVAR2_REF` as the canonical copy. Pick whichever the test run actually accepts and say which in the task report — do not leave a broken import.

- [ ] **Step 4: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS with the same counts as before this task (**1250 passed, 61 skipped, 4 xfailed** on darwin). **No assertion may have been edited to achieve this.** If a test fails, the consolidation changed behavior — most likely a module silently picked up the 3-sample fixture — and the fix is to repoint it, never to weaken the assertion.

- [ ] **Step 5: Confirm the duplication is actually gone**

Run: `pixi run -e dev python -c "import subprocess; print(subprocess.run(['grep','-rn','def svar2_store','tests/'],capture_output=True,text=True).stdout)"`

Expected: exactly **seven** definitions remain, and no more —

| definition | where |
|---|---|
| `svar2_store_2s` | `tests/conftest.py` (new, shared) |
| `svar2_store` | `tests/dataset/conftest.py` (3-sample) |
| `svar2_store` | `tests/dataset/test_svar2_readbound_tracks.py` (4-variant, deliberately kept) |
| `svar2_store_dense_snp` | `test_svar2_readbound_variants.py`, `test_svar2_readbound_haps.py`, `test_svar2_readbound_diffs.py` |
| `svar2_store_unsorted` | `test_write_svar2.py` |

Six definitions were removed. If any other `def svar2_store(` survives, the dedup is incomplete; if one of the seven above is gone, something was folded in that should not have been.

- [ ] **Step 6: Commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add tests/
git commit -m "test(svar2): hoist the duplicated svar2_store fixture into tests/conftest.py"
```

---

### Task 7: Enrich the three-sample fixture and rewrite its guards

**Files:**
- Modify: `tests/dataset/conftest.py` (the `_VCF` constant), `tests/dataset/test_write_svar2.py:624-690` (the occupancy guard), `tests/dataset/test_write_svar2.py:737-758` (the dense-layout test's docstring)

**Interfaces:**
- Consumes: the consolidated fixture layout from Task 6.
- Produces: nothing consumed by later tasks.

**What is already done, so do not redo it.** #406 lists three follow-ons; two landed during #357 in commit `c3da6bda` and must be left alone:
- The per-channel oracle masks at `test_write_svar2.py:~117-133` already read `present = (widths_snp > 0) | (widths_indel > 0)` — cell-level, with a comment explaining why per-channel masking is wrong.
- `test_dense_layout_dataset_still_opens_and_reads` already says in its docstring that it is a smoke test, not a parity pin.

**The problem that remains.** The 3-sample fixture's vk (sparse) channel has **one** non-empty cell in an 18-cell grid (3 regions × 3 samples × ploidy 2). genoray's `cost_model.rs::choose_representation` routes a variant whose carrier-call count crosses a bit-cost threshold to the per-region **dense** channel, which has no sample axis; the fixture's SNP has one carrier and stays sparse, both indels have three carrier calls and route dense. So S1 has zero cells in the vk view anywhere, structurally indistinguishable from S2's deliberately empty column. Its `dense_snp_range` is also `[[0,0],[0,0],[0,0]]` — the dense **SNP** channel is never exercised at all.

**The replacement was built and measured, not reasoned about.** Every sparse variant below carries exactly **one** call, which is what keeps it out of the dense channel; the two 3-carrier variants exist to populate both dense channels.

- [ ] **Step 1: Replace `_VCF` in `tests/dataset/conftest.py`**

```python
_VCF = """\
##fileformat=VCFv4.2
##contig=<ID=chr1,length=40>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS0\tS1\tS2
chr1\t3\t.\tA\tG\t.\t.\t.\tGT\t1|0\t0|0\t0|0
chr1\t7\t.\tC\tCAT\t.\t.\t.\tGT\t0|1\t0|0\t0|0
chr1\t9\t.\tT\tC\t.\t.\t.\tGT\t0|0\t0|1\t0|0
chr1\t12\t.\tGTA\tG\t.\t.\t.\tGT\t0|0\t1|0\t0|0
chr1\t17\t.\tA\tC\t.\t.\t.\tGT\t0|1\t0|0\t0|0
chr1\t19\t.\tC\tCGG\t.\t.\t.\tGT\t1|1\t1|1\t1|1
chr1\t30\t.\tG\tA\t.\t.\t.\tGT\t1|1\t1|1\t1|1
"""
```

Regions are unchanged: `[0, 20)`, `[5, 15)`, `[25, 40)`. `_REF` is unchanged.

- [ ] **Step 2: Run the suite to see what the enrichment breaks**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: FAIL — at minimum `test_fixture_has_empty_cells`, which pins the old one-cell grid. Record every failure; each one is a test that was passing because the fixture was weak, and each is fixed in the following steps rather than weakened.

- [ ] **Step 3: Rewrite the occupancy guard**

Replace the body of the fixture guard at `tests/dataset/test_write_svar2.py:624-690` (the test asserting `sorted_samples == ["S0", "S1", "S2"]`, reshaping `nonempty` to `(3, S, P)`, and pinning `grid[0, S0, 0]`) with:

```python
svar2 = SparseVar2(svar2_store)
sorted_samples = sorted(svar2.available_samples)
assert sorted_samples == ["S0", "S1", "S2"], "fixture lost its third sample"

d = svar2._find_ranges(
    "chr1",
    np.array([0, 5, 25]),
    np.array([20, 15, 40]),
    samples=sorted_samples,
)
snp = np.asarray(d["vk_snp_range"], np.int64)  # (R*S*P, 2)
indel = np.asarray(d["vk_indel_range"], np.int64)
w_snp = (snp[:, 1] - snp[:, 0]).reshape(3, len(sorted_samples), svar2.ploidy)
w_indel = (indel[:, 1] - indel[:, 0]).reshape(w_snp.shape)
occ = (w_snp > 0) | (w_indel > 0)

s0, s1, s2 = (sorted_samples.index(s) for s in ("S0", "S1", "S2"))

# Row-major (R, S, P) -- pinned by the layout oracle in
# test_write_svar2_emits_cache, which asserts this same reshape against the
# cache memmaps. Assert each structure SEPARATELY: a single `not occ.all()`
# is a disjunction that stays green when any one of them regresses alone.
assert not occ[:, s2].any(), "S2 is no longer all-reference; the empty COLUMN is gone"
assert not occ[2].any(), (
    "region [25, 40) now holds sparse variants; the empty ROW is gone"
)

# The measured grid. Every sparse variant carries exactly one call, which is
# what keeps genoray's cost model from routing it to the per-region dense
# channel; if a cost-model change pushes any of them dense this fails loudly
# rather than letting the sparse/dense parity tests pass against a thinner
# table while appearing green.
expected = np.zeros_like(occ)
expected[0, s0] = [True, True]
expected[0, s1] = [True, True]
expected[1, s0] = [False, True]
expected[1, s1] = [True, True]
np.testing.assert_array_equal(occ, expected)
assert int(occ.sum()) == 7, "fixture occupancy changed; update this pin deliberately"

# Non-vacuity, kept from the original guard: an all-empty vk grid would make
# every sparse/dense parity test built on this fixture pass trivially.
assert occ.any(), (
    "vk channel is entirely empty: genoray routed every variant to the "
    "dense channel, so all sparse-cache parity tests on this fixture are "
    "now vacuous"
)

# Mixed per-channel emptiness inside a PRESENT cell -- the property the old
# one-cell grid could not express, and the reason two shipped oracles were
# able to mask per channel without failing.
assert (w_snp[0, s0, 0] > 0) and (w_indel[0, s0, 0] == 0)
assert (w_snp[0, s0, 1] > 0) and (w_indel[0, s0, 1] > 0)
assert (w_snp[0, s1, 0] == 0) and (w_indel[0, s1, 0] > 0)

# An ABSENT cell inside a non-empty region: region 1 holds variants, but
# (S0, ploid 0) has none, so lookup must return (0, 0) there rather than a
# neighbour's range.
assert not occ[1, s0, 0]

# The sample axis is now ORDERED, not just occupied: S0 and S1 differ at
# (region 1, ploid 0), so a transposed sample axis is detectable. During
# #357 this guard had to be withdrawn as unsatisfiable.
assert not np.array_equal(occ[:, s0], occ[:, s1])

# Both dense channels are exercised. dense_snp_range was all zeros before
# this fixture was enriched, so the dense SNP path had no coverage here.
dense_snp = np.asarray(d["dense_snp_range"], np.int64)
dense_indel = np.asarray(d["dense_indel_range"], np.int64)
assert (dense_snp[:, 1] > dense_snp[:, 0]).any(), "dense SNP channel is empty again"
assert (dense_indel[:, 1] > dense_indel[:, 0]).any(), "dense indel channel is empty"
```

Rename the test to `test_fixture_grid_is_non_degenerate` and update its docstring to describe the 7-of-18 grid. Keep the module's existing import style (`from genoray import SparseVar2` inside the test, as the current version does).

- [ ] **Step 4: Fix the remaining fallout from Step 2**

For each other failure recorded in Step 2, fix it by making the assertion express the **new, stronger** reality — never by loosening it. Expect `test_write_svar2_emits_cache`'s `n_empty_seen > 0` and its `n * 28 < bed.height * S * P * 32` size assertion to still hold (fill rises from 1/18 to 7/18, well under the 28/32 break-even). If any test now needs a *weaker* assertion to pass, stop and report it — that means the enrichment broke a real invariant rather than a weak test.

- [ ] **Step 5: Promote the dense-layout test**

At `tests/dataset/test_write_svar2.py:737-758`, the docstring justifies the test being a smoke test by "this fixture's grid has only one non-empty cell (see `svar2_store`), so it can't distinguish a correct dense reader from one that always returns the same wrong answer." That reason is now false. Update the docstring to state the grid is 7-of-18 and that the test *does* pin dense-vs-sparse parity, and add an assertion comparing the dense-layout dataset's reads against the sparse-layout dataset's reads over the full `(region, sample)` grid rather than only asserting it opens.

- [ ] **Step 6: Mutation-check the promotion**

Temporarily inject `r_q = np.zeros_like(r_q)` at the top of `_DenseRanges.lookup` in `python/genvarloader/_dataset/_svar2_ranges.py` and run:

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py::test_dense_layout_dataset_still_opens_and_reads -q`
Expected: FAIL. Revert the injection and confirm PASS. If it still passes with the mutation, the promotion did not happen — the assertion is not reaching the dense reader.

- [ ] **Step 7: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS. Note the new counts in the task report; they will differ from the 1250-passed baseline if assertions were added.

- [ ] **Step 8: Commit**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
git add tests/
git commit -m "test(svar2): enrich the fixture grid from 1 to 7 occupied cells"
```

---

### Task 8: Documentation

**Files:**
- Modify: `docs/source/write.md:158-163`, `skills/genvarloader/SKILL.md:196-199`

**Interfaces:**
- Consumes: the finished behavior from Tasks 1–5.
- Produces: nothing.

**Why docs change at all.** `gvl.concat` is public API (`python/genvarloader/__init__.py:73`), and both docs currently claim its cost "is dominated by I/O rather than computation" without mentioning memory — which was false before this change, since planning allocated 32–768 GB at cohort scale. The claim becomes true here, and the repo's docs-audit rule requires user-facing docs to stay true for any user-facing change. `docs/source/changelog.md` is auto-generated from commit messages and does **not** count as documentation; do not hand-edit it.

- [ ] **Step 1: Update `docs/source/write.md`**

After the existing paragraph ending "not as a routine step." (`write.md:158-163`), add:

```markdown
`gvl.concat`'s planning is streaming: it derives the merge plan from the sorted
merge order rather than building a map over every `(region, sample, ploid)` slot,
so its memory stays proportional to regions plus samples rather than to their
product. The merged `offsets.npy` it writes is still one int64 per merged slot —
about 16 GB for a 3,734-region by 535,662-sample grid — and that is a property of
the ragged on-disk format, not of the merge.
```

- [ ] **Step 2: Update `skills/genvarloader/SKILL.md`**

In the `**Cost:**` paragraph at `SKILL.md:196-199`, after "it moves roughly the full size of the merged dataset.", insert:

```markdown
Planning is streaming — memory scales with regions plus samples, not their
product — so the resident cost is the merged offsets array (one int64 per merged
`(region, sample[, ploid])` slot), not the merge plan.
```

- [ ] **Step 3: Verify the docs build**

Run: `pixi run -e docs doc`
Expected: builds without new warnings. If the `docs` environment is stale, run `pixi install -e docs` first.

- [ ] **Step 4: Check `api.md` is still in sync with `__all__`**

Run: `pixi run -e dev python -c "import genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"`
Expected: `MISSING: none`. This change adds no public symbol, so this is a regression check, not new work.

- [ ] **Step 5: Commit**

```bash
git add docs/source/write.md skills/genvarloader/SKILL.md
git commit -m "docs(concat): state that merge planning is streaming, not slot-sized"
```

---

## Final Verification

After Task 8, before opening the PR:

- [ ] `pixi run -e dev test` — the full pytest + cargo gate, whole tree.
- [ ] `pixi run -e dev ruff check python/ tests/` — clean.
- [ ] `pixi run -e dev ruff format python/ tests/` — reports no reformatting.
- [ ] `pixi run -e dev typecheck` — pyrefly reports 0 errors. **Confirm it is not vacuous:** this worktree lives under the git-ignored `.claude/worktrees/`, which has previously made pyrefly check zero files and report a meaningless "0 errors". A real run reports a non-zero count of suppressed errors and warnings alongside the zero.
- [ ] `git log --oneline origin/main..HEAD` — every commit is `perf:`, `refactor:`, `test:`, or `docs:`. No `feat:`, no `BREAKING CHANGE`.
- [ ] `pixi run -e dev python -c "import re; s=open('python/genvarloader/_dataset/_concat_plan.py').read(); assert 'def provenance' in s and 'def coalesce' in s, 'the oracle was deleted'"` — the oracle survives.
