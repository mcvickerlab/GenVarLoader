# Test Suite Overhaul — Phase 5 Reconstruct Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the reconstruct component of Phase 5 — promote a `make_tracks` builder from the existing `_make_tracks` helper in `test_insertion_fill.py`, then extract 19 unit-portable tests from `test_insertion_fill.py` into `tests/unit/tracks/test_insertion_fill.py`, leaving 3 truly-integration tests behind.

**Architecture:** Most reconstruction-kernel tests already moved in the prelude (`test_realign.py`, `test_reconstruct.py`, `test_build_reconstructor.py`). The only sizeable kernel-level file remaining is `test_insertion_fill.py`, which mixes 19 pure-unit tests with 3 dataset-dependent tests. Split it cleanly. The 19 extracted tests use a new `make_tracks` builder (promoted from the file's existing `_make_tracks` helper). The 3 dataset-dependent tests stay because they use `gvl.get_dummy_dataset()` and exercise the full `with_insertion_fill` → reconstruction call path.

**Tech Stack:** pytest, pytest-cases, numpy, awkward; no production code changes.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`
- Audit: `docs/superpowers/specs/2026-05-24-test-audit.md` (especially the `test_insertion_fill.py` per-file section, lines 177-186)

---

## Pre-flight baseline (after Phase 5 prelude + ragged)

- Non-slow tier: **351 passed, 3 skipped, 3 deselected, 2 xfailed**
- Coverage: **63%**
- Unit tier: **80 passed, 1 xfailed**

Tests stay at the same totals throughout (relocations, not additions).

---

## Test classification (from audit)

`tests/integration/dataset/test_insertion_fill.py` has 22 test functions; the audit's per-file table groups them into 4 buckets:

| Group | Count | Bucket | Destination |
|---|---|---|---|
| `InsertionFill` subclass unit tests (`test_lower_*`, `test_constant_default_is_nan`, `test_flank_sample_negative_width_rejected`, `test_interpolate_order_capped`, `test_insertion_fill_base_not_instantiable`) | 7 | Port | `tests/unit/tracks/test_insertion_fill.py` |
| `shift_and_realign_track_sparse` kernel tests (`test_kernel_*`, 9 functions) | 9 | Port | same file |
| `with_insertion_fill` plumbing on synthetic `Tracks` (`test_with_insertion_fill_single_applies_to_all`, `test_with_insertion_fill_dict_partial_falls_back`, `test_with_tracks_prunes_insertion_fill`) | 3 | Port | same file |
| Dataset-dependent (`test_end_to_end_set_insertion_fill`, `test_dummy_dataset_with_default_insertion_fill_does_not_crash`, `test_with_insertion_fill_rejects_when_no_tracks_active`) | 3 | Keep / needs `make_dataset` | stay in integration |

The third "Keep" row blends the audit's 2 Keep-as-integration tests with `test_with_insertion_fill_rejects_when_no_tracks_active`. The audit classifies the latter as Port-with-builder-needed (`make_dataset`); since that builder doesn't exist yet (dataset polymorphism component is later in the spec sequence), keep it in integration for now.

Result: **19 tests move to unit, 3 stay in integration.**

---

## Builder design — `tests/_builders/reconstruct.py`

Promotes the file-local `_make_tracks` helper into a reusable test builder. Minimal scope — only what `test_insertion_fill.py` actually needs; richer reconstruct builders (e.g., `make_haps`, `make_ref_reconstructor`) come in later component plans when consumers exist.

The current `_make_tracks` in the integration file (lines 305-326) takes a list of track names and builds a 1-region, 1-sample `Tracks` with `Repeat5p()` insertion-fill by default. We promote it with one change: accept an optional `insertion_fill` arg (mapping per-track) to make the builder more flexible.

---

## Task 1: Add `make_tracks` builder to `tests/_builders/reconstruct.py`

**Files:**
- Create: `tests/_builders/reconstruct.py`

- [ ] **Step 1: Create the builder file**

Write `tests/_builders/reconstruct.py` with this exact content:

```python
"""In-memory builders for reconstruction-component types.

Currently exports ``make_tracks`` — a minimal-shape ``Tracks`` reconstructor
suitable for testing track-method plumbing (``with_insertion_fill``,
``with_tracks``, etc.) without opening a Dataset.

The builder produces a 1-region, 1-sample, 1-interval ``Tracks`` per
named track, with default ``Repeat5p`` insertion-fill unless overridden.
Callers that need more elaborate shapes should add a new builder rather
than parametrize this one.
"""

from __future__ import annotations

import numpy as np
from seqpro.rag import Ragged

from genvarloader._dataset._insertion_fill import InsertionFill, Repeat5p
from genvarloader._dataset._reconstruct import Tracks, TrackType
from genvarloader._ragged import RaggedIntervals, RaggedTracks


def make_tracks(
    names: list[str],
    insertion_fill: dict[str, InsertionFill] | None = None,
) -> Tracks:
    """Build a minimal ``Tracks`` instance for plumbing tests.

    Each name produces a 1-region, 1-sample track with a single dummy
    interval at position 0. The default ``insertion_fill`` is
    ``{name: Repeat5p()}`` for every name; pass an explicit mapping to
    override.

    Example::

        tracks = make_tracks(["a", "b"])
        tracks.insertion_fill["a"]  # Repeat5p()

        tracks = make_tracks(["a"], insertion_fill={"a": Constant(0.0)})
    """
    starts = ends = values = np.array([0], dtype=np.int32)
    offsets = np.array([0, 1], dtype=np.int64)
    intervals = {
        n: RaggedIntervals(
            Ragged.from_offsets(starts, (1, None), offsets),
            Ragged.from_offsets(ends, (1, None), offsets),
            Ragged.from_offsets(values, (1, None), offsets),
        )
        for n in names
    }
    active = {n: TrackType.SAMPLE for n in names}
    if insertion_fill is None:
        insertion_fill = {n: Repeat5p() for n in names}
    return Tracks(
        intervals=intervals,
        active_tracks=active,
        available_tracks=active,
        kind=RaggedTracks,
        n_regions=1,
        n_samples=1,
        insertion_fill=insertion_fill,
    )
```

- [ ] **Step 2: Smoke-test the builder**

```
pixi run -e dev python -c "
import sys
sys.path.insert(0, 'tests')
from _builders.reconstruct import make_tracks
from genvarloader._dataset._insertion_fill import Constant, Repeat5p

t = make_tracks(['a', 'b'])
assert isinstance(t.insertion_fill['a'], Repeat5p)
assert isinstance(t.insertion_fill['b'], Repeat5p)
assert set(t.active_tracks) == {'a', 'b'}

t2 = make_tracks(['a'], insertion_fill={'a': Constant(0.0)})
assert isinstance(t2.insertion_fill['a'], Constant)
print('make_tracks OK')
"
```

Expected output: `make_tracks OK`.

- [ ] **Step 3: Full non-slow suite (no consumers yet — sanity check only)**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed`.

- [ ] **Step 4: Commit**

```bash
git add tests/_builders/reconstruct.py
git commit -m "test(builders): add make_tracks reconstruct builder"
```

Verify `git status` clean.

---

## Task 2: Extract 19 unit tests + trim integration file (one atomic commit)

This is the load-bearing task. Both halves must land together — partial application would either lose tests or double-count them.

**Files:**
- Create: `tests/unit/tracks/test_insertion_fill.py` (19 extracted tests)
- Modify: `tests/integration/dataset/test_insertion_fill.py` (keep only 3 dataset-dependent tests + their imports)

### Step 1: Create `tests/unit/tracks/test_insertion_fill.py`

Write the new unit file with this exact content. It contains:
- The 7 `InsertionFill` subclass tests (lines 27-91 of the integration source).
- The 9 `shift_and_realign_track_sparse` kernel tests (lines 123-298 of the integration source), plus the `test_kernel_flank_sample_query_hap_affects_hash` test from line 260.
- The 3 `with_insertion_fill` plumbing tests (lines 329-351 of the integration source), rewritten to use `make_tracks` from the new builder instead of the file-local `_make_tracks` helper.

- [ ] **Write the file**

Create `tests/unit/tracks/test_insertion_fill.py`:

```python
"""Unit tests for InsertionFill subclasses, the shift_and_realign_track_sparse
kernel, and Tracks.with_insertion_fill plumbing.

Originally lived in tests/integration/dataset/test_insertion_fill.py;
extracted to the unit tier because every test here constructs its inputs
in-memory and exercises only the in-memory reconstruction path — no
Dataset, no disk I/O. Three dataset-dependent tests (end-to-end + dummy
dataset + rejects-when-no-tracks) remain in the original file.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
from genoray._svar import dense2sparse

from genvarloader._dataset._insertion_fill import (
    CONSTANT,
    FLANK_SAMPLE,
    INTERPOLATE,
    REPEAT_5P,
    REPEAT_5P_NORM,
    Constant,
    FlankSample,
    Interpolate,
    Repeat5p,
    Repeat5pNormalized,
    lower,
)
from genvarloader._dataset._tracks import shift_and_realign_track_sparse

# Builder import: pytest puts the tests/ directory on sys.path via its
# import-mode handling, so `_builders` is importable from any test file.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _builders.reconstruct import make_tracks  # noqa: E402


# ---------------------------------------------------------------------------
# InsertionFill subclasses — pure unit tests of the dataclass / serializer
# ---------------------------------------------------------------------------


def test_lower_all_strategies():
    strategies = [
        Repeat5p(),
        Repeat5pNormalized(),
        FlankSample(),
        Constant(7.0),
        Interpolate(order=3),
    ]
    out = lower(strategies)
    assert out == [
        (REPEAT_5P, ()),
        (REPEAT_5P_NORM, ()),
        (FLANK_SAMPLE, (5.0, 0)),
        (CONSTANT, (7.0,)),
        (INTERPOLATE, (3.0,)),
    ]


def test_lower_empty():
    assert lower([]) == []


def test_constant_default_is_nan():
    assert math.isnan(Constant().value)


def test_flank_sample_negative_width_rejected():
    with pytest.raises(ValueError):
        FlankSample(flank_width=-1.0)


def test_interpolate_order_capped():
    with pytest.raises(ValueError):
        Interpolate(order=100)


def test_lower_unknown_class_raises():
    class _Unknown:  # not an InsertionFill subclass
        pass

    with pytest.raises(TypeError):
        lower([_Unknown()])  # type: ignore[list-item]


def test_insertion_fill_base_not_instantiable():
    from genvarloader._dataset._insertion_fill import InsertionFill

    with pytest.raises(TypeError):
        InsertionFill()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Kernel-level tests — shift_and_realign_track_sparse with each strategy
# ---------------------------------------------------------------------------


def _kernel_call(
    track,
    v_starts,
    ilens,
    sparse_genos,
    *,
    strategy_id,
    params=None,
    base_seed=0,
    out_size=10,
    query=0,
    hap=0,
):
    """Thin wrapper around shift_and_realign_track_sparse for kernel tests."""
    out = np.zeros(out_size, dtype=np.float32)
    if params is None:
        params = np.zeros(1, dtype=np.float64)
    shift_and_realign_track_sparse(
        offset_idx=0,
        geno_v_idxs=sparse_genos.data,
        geno_offsets=sparse_genos.offsets,
        v_starts=v_starts,
        ilens=ilens,
        shift=0,
        track=track,
        query_start=0,
        out=out,
        params=params,
        strategy_id=strategy_id,
        base_seed=np.uint64(base_seed),
        query=query,
        hap=hap,
    )
    return out


def _single_insertion(*, ilen=2):
    """One variant at pos 1 with given ilen; geno = present (genotype 1)."""
    v_starts = np.array([1], dtype=np.int32)
    ilens = np.array([ilen], dtype=np.int32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)
    return v_starts, ilens, sparse_genos


def test_kernel_repeat_5p_default():
    track = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=2)
    out = _kernel_call(track, v_starts, ilens, genos, strategy_id=REPEAT_5P)
    # First 4 elements: track[0], then 2 fill bytes repeating 5' (track[0]), then track[1].
    np.testing.assert_array_equal(out[:4], [10.0, 10.0, 10.0, 20.0])


def test_kernel_repeat_5p_normalized():
    track = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=2)
    out = _kernel_call(track, v_starts, ilens, genos, strategy_id=REPEAT_5P_NORM)
    # Normalized: divide repeat values by (ilen + 1) = 3 so total integral matches.
    expected_repeat = 10.0 / 3.0
    np.testing.assert_allclose(out[:4], [10.0, expected_repeat, expected_repeat, 20.0])


def test_kernel_constant_nan():
    track = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=2)
    params = np.array([float("nan")], dtype=np.float64)
    out = _kernel_call(
        track, v_starts, ilens, genos, strategy_id=CONSTANT, params=params
    )
    assert math.isnan(out[1])
    assert math.isnan(out[2])
    assert out[0] == 10.0
    assert out[3] == 20.0


def test_kernel_flank_sample_pool_membership():
    track = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=3)
    params = np.array([2.0], dtype=np.float64)  # flank_width=2
    out = _kernel_call(
        track,
        v_starts,
        ilens,
        genos,
        strategy_id=FLANK_SAMPLE,
        params=params,
        base_seed=42,
    )
    # Sampled values must come from the flank pool {0.0, 10.0, 20.0, 30.0}.
    pool = {0.0, 10.0, 20.0, 30.0}
    for v in out[1:4]:
        assert float(v) in pool


def test_kernel_flank_sample_deterministic():
    track = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=3)
    params = np.array([2.0], dtype=np.float64)
    out_a = _kernel_call(
        track,
        v_starts,
        ilens,
        genos,
        strategy_id=FLANK_SAMPLE,
        params=params,
        base_seed=7,
    )
    out_b = _kernel_call(
        track,
        v_starts,
        ilens,
        genos,
        strategy_id=FLANK_SAMPLE,
        params=params,
        base_seed=7,
    )
    np.testing.assert_array_equal(out_a, out_b)


def test_kernel_interpolate_linear():
    track = np.array([0.0, 10.0, 30.0, 40.0, 50.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=1)
    params = np.array([1.0], dtype=np.float64)  # order=1 (linear)
    out = _kernel_call(
        track, v_starts, ilens, genos, strategy_id=INTERPOLATE, params=params
    )
    # Linear interpolate between anchors 10.0 and 30.0 for 1 fill cell → 20.0
    np.testing.assert_allclose(out[1], 20.0)


def test_kernel_interpolate_cubic_passes_through_anchors():
    track = np.array([0.0, 10.0, 30.0, 40.0, 50.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=1)
    params = np.array([3.0], dtype=np.float64)
    out = _kernel_call(
        track, v_starts, ilens, genos, strategy_id=INTERPOLATE, params=params
    )
    # Anchors at indices 0 and 1 of original track must survive.
    np.testing.assert_allclose(out[0], 10.0)
    np.testing.assert_allclose(out[2], 30.0)


def test_kernel_flank_sample_edge_clamp():
    track = np.array([10.0, 20.0], dtype=np.float32)
    v_starts, ilens, genos = _single_insertion(ilen=3)
    params = np.array([10.0], dtype=np.float64)  # flank_width clamped to track size
    out = _kernel_call(
        track,
        v_starts,
        ilens,
        genos,
        strategy_id=FLANK_SAMPLE,
        params=params,
        base_seed=99,
        out_size=5,
    )
    # Pool clamped to {10.0, 20.0}; all fill values must be one of those.
    pool = {10.0, 20.0}
    for v in out[1:4]:
        assert float(v) in pool


def test_kernel_flank_sample_query_hap_affects_hash():
    """Different (query, hap) seeds must drive different samples for the same base_seed."""
    v_starts = np.array([1], dtype=np.int32)
    ilens = np.array([3], dtype=np.int32)
    track = np.array([0.0, 10.0, 20.0, 30.0, 40.0], dtype=np.float32)
    genos = np.array([[[1]]], dtype=np.int8)
    var_idxs = np.array([0], dtype=np.int32)
    sparse_genos = dense2sparse(genos, var_idxs)

    params = np.zeros(1, dtype=np.float64)
    params[0] = 2.0

    def run(query, hap):
        out = np.zeros(8, dtype=np.float32)
        shift_and_realign_track_sparse(
            offset_idx=0,
            geno_v_idxs=sparse_genos.data,
            geno_offsets=sparse_genos.offsets,
            v_starts=v_starts,
            ilens=ilens,
            shift=0,
            track=track,
            query_start=0,
            out=out,
            params=params,
            strategy_id=FLANK_SAMPLE,
            base_seed=np.uint64(99),
            query=query,
            hap=hap,
        )
        return out[1:5].copy()

    a = run(0, 0)
    b = run(1, 0)
    c = run(0, 1)
    # Each pair should differ at least one position.
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


# ---------------------------------------------------------------------------
# Tracks reconstructor — insertion_fill plumbing (uses make_tracks builder)
# ---------------------------------------------------------------------------


def test_with_insertion_fill_single_applies_to_all():
    tracks = make_tracks(["a", "b"])
    new = tracks.with_insertion_fill(Constant(0.0))
    assert isinstance(new.insertion_fill["a"], Constant)
    assert isinstance(new.insertion_fill["b"], Constant)
    # original unchanged (evolve returns new instance)
    assert isinstance(tracks.insertion_fill["a"], Repeat5p)


def test_with_insertion_fill_dict_partial_falls_back():
    tracks = make_tracks(["a", "b"])
    new = tracks.with_insertion_fill({"a": FlankSample(flank_width=2)})
    assert isinstance(new.insertion_fill["a"], FlankSample)
    assert isinstance(new.insertion_fill["b"], Repeat5p)


def test_with_tracks_prunes_insertion_fill():
    tracks = make_tracks(
        ["a", "b"],
        insertion_fill={"a": Constant(0.0), "b": FlankSample()},
    )
    new = tracks.with_tracks("a")
    assert "a" in new.insertion_fill
    assert "b" not in new.insertion_fill
```

**CRITICAL** — for the kernel tests, the bodies above are reasonable interpretations of what the integration tests do, but the EXACT bodies must come from the existing integration file (`tests/integration/dataset/test_insertion_fill.py` lines 123-220, plus 260-298 for `test_kernel_flank_sample_query_hap_affects_hash`). If any test body in the existing file differs from what's shown here, copy the EXISTING body verbatim into the new unit file — do NOT use my reconstructions if they diverge. The goal is to preserve test semantics exactly, not to rewrite.

After writing, compare each test function body line-by-line against the source file. Fix divergences before proceeding.

For `test_with_tracks_prunes_insertion_fill`: the original (line 345-351) ends with `new = tracks.with_tracks("a")` followed by trailing assertion lines that may have been cut off in my read. Read lines 345-360 of the source file and confirm the full test body.

### Step 2: Verify the new unit file runs in isolation

```
pixi run -e dev pytest tests/unit/tracks/test_insertion_fill.py -q 2>&1 | tail -5
```

Expected: **19 passed** (7 + 9 + 3). If any test fails, the most likely cause is a body that diverged from the integration file — re-check against the source.

### Step 3: Trim the integration file

Edit `tests/integration/dataset/test_insertion_fill.py` to keep ONLY the 3 dataset-dependent tests (`test_end_to_end_set_insertion_fill`, `test_dummy_dataset_with_default_insertion_fill_does_not_crash`, `test_with_insertion_fill_rejects_when_no_tracks_active`) plus the imports those tests still need.

After trimming, the file should look approximately like:

```python
"""End-to-end insertion-fill tests requiring a real Dataset.

The unit-tier insertion-fill tests live in
``tests/unit/tracks/test_insertion_fill.py``. The tests here use
``gvl.get_dummy_dataset()`` and exercise the full
``with_insertion_fill`` → reconstruction call path.
"""

import math

import pytest

import genvarloader as gvl
from genvarloader._dataset._insertion_fill import Constant, Repeat5p


def test_end_to_end_set_insertion_fill():
    """Use the dummy dataset to confirm with_insertion_fill plumbing works end-to-end."""
    ds = gvl.get_dummy_dataset()
    # Only haps+tracks datasets support insertion fill.
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    first_track = next(iter(ds._tracks.active_tracks))
    ds_nan = ds.with_insertion_fill({first_track: Constant(float("nan"))})
    assert isinstance(ds_nan._tracks.insertion_fill[first_track], Constant)
    assert math.isnan(ds_nan._tracks.insertion_fill[first_track].value)
    # Immutability: original dataset's insertion_fill is not mutated by the new dataset.
    assert first_track not in ds._tracks.insertion_fill
    # Trigger actual reconstruction to verify the full call path executes without error.
    _ = ds_nan[0, 0]


def test_dummy_dataset_with_default_insertion_fill_does_not_crash():
    """Tracks created outside from_path may have empty insertion_fill — must not KeyError."""
    ds = gvl.get_dummy_dataset()
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Just trigger reconstruction; the call must not raise KeyError.
    _ = ds[0, 0]


def test_with_insertion_fill_rejects_when_no_tracks_active():
    """A dataset with tracks disabled should reject with_insertion_fill."""
    ds = gvl.get_dummy_dataset()
    if ds._tracks is None or ds._seqs is None:
        pytest.skip("dummy dataset shape does not include both seqs and tracks")
    # Disable tracks: view-state no longer has active tracks.
    ds_no_tracks = ds.with_tracks(False)
    with pytest.raises(ValueError, match="with_tracks"):
        ds_no_tracks.with_insertion_fill(Repeat5p())
```

**Verify against source:** the bodies of these 3 tests must match the integration source EXACTLY. Read lines 224-258 of the original file and confirm. If anything diverges, prefer the source over what's shown here.

### Step 4: Run the trimmed integration file

```
pixi run -e dev pytest tests/integration/dataset/test_insertion_fill.py -q 2>&1 | tail -3
```

Expected: 3 tests collected; pass count depends on whether the dummy dataset includes both seqs and tracks (the `pytest.skip` guards) — so it may show "1 passed, 2 skipped" or "3 passed" depending on environment. Either is acceptable. Do NOT see import errors.

### Step 5: Full non-slow suite

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed` — same totals as the baseline (19 tests moved from one tier to another, totals unchanged).

If pass count drops or skip count drifts, something went wrong. Likely culprit: a test body diverged from the source. Bisect.

### Step 6: Commit atomically

```bash
git add tests/integration/dataset/test_insertion_fill.py tests/unit/tracks/test_insertion_fill.py
git commit -m "test: extract insertion-fill unit tests to unit/tracks/"
```

Verify `git status` clean. The single commit must touch both files together — never separately.

---

## Task 3: End-of-plan verification

- [ ] **Step 1: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: **`351 passed, 3 skipped, 3 deselected, 2 xfailed`** (same as Phase 5 prelude + ragged baseline).

- [ ] **Step 2: Unit tier collection**

```
pixi run -e dev pytest tests/unit -q 2>&1 | tail -2
```

Expected: roughly **99 passed, 1 xfailed** (80 from prelude + 19 from insertion-fill).

- [ ] **Step 3: Coverage parity**

```
pixi run -e dev pytest tests -m "not slow" --cov --cov-report=term -q 2>&1 | grep "^TOTAL"
```

Expected: `TOTAL ... 63%` (±1pp).

- [ ] **Step 4: Commit graph**

```
git log --oneline -5
```

Expected: 2 new commits since end of Phase 5 prelude+ragged:

```
<sha> test: extract insertion-fill unit tests to unit/tracks/
<sha> test(builders): add make_tracks reconstruct builder
<previous Phase 5 head>
```

- [ ] **Step 5: Confirm file structure**

```
/bin/ls tests/unit/tracks/
```

Expected: `test_i2t_t2i.py`, `test_insertion_fill.py`, `test_realign.py`, `test_tracks_splice.py` (plus the original `.gitkeep` may be present or absent).

```
/bin/ls tests/_builders/
```

Expected: `__init__.py`, `ragged.py`, `reconstruct.py`.

```
/bin/ls tests/integration/dataset/test_insertion_fill.py
```

Expected: file still exists, ~40 lines.

---

## Out of scope (deferred to subsequent component plans)

- **variants** — `test_issue_191_var_fields.py` ports (5 tests), `test_choose_exonic_variants.py` (2 tests). Need a `_Variants.from_table` builder or similar.
- **haps** — Currently no port-bucket tests in audit specifically for `Haps.from_path`; the existing `test_issue_191_var_fields.py` has Haps-related ports under the "variants" header.
- **tracks (broader)** — `test_random_nonoverlapping.py` (1 port), `test_write_tracks.py:test_write_duplicate_track_names_rejected` (1 port), `test_table.py` (12 ports).
- **splice (broader)** — `test_get_splice_bed.py` (11 ports), `test_ref_ds_splicing.py` (5 ports).
- **ref/fasta** — `test_fasta.py` (3 ports), `test_ref_ds.py` (2 remaining ports).
- **dataset polymorphism** — `test_svar_link.py` Pydantic ports (8), and the lone `test_with_insertion_fill_rejects_when_no_tracks_active` once a `make_dataset` builder exists.
- **utility** — `test_utils.py` (5 ports).
