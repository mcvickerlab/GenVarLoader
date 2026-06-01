# Flat-buffer `__getitem__` pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Dataset.__getitem__` operate on flat numpy `(data, offsets, shape)` buffers end-to-end, constructing awkward-backed seqpro `Ragged` only at the return boundary, so the awkward `_carry`/`_kernels`/`concat` churn leaves the per-batch hot path while outputs stay byte-identical.

**Architecture:** Introduce a gvl-internal `_Flat` container (pure numpy + offsets, never runs awkward kernels) plus composite analogs. Reconstructors return `_Flat` instead of calling `Ragged.from_offsets`; the post-reconstruction transforms (reverse-complement, densify) run on `_Flat`; `getitem` converts to `Ragged`/`RaggedAnnotatedHaps`/`RaggedVariants`/`RaggedIntervals` only when the requested output is ragged. Legacy `Ragged`-returning paths coexist during migration (the boundary dispatches on type) and are deleted in the final guard task.

**Tech Stack:** Python 3.10, numpy 1.26, numba 0.59.1, seqpro 0.13.0 (`seqpro.rag.Ragged`/`to_padded`/`reverse_complement`), pytest, pixi (`pixi run -e dev …`), py-spy + memray for profiling.

---

## Spec

`docs/superpowers/specs/2026-05-31-flat-buffer-getitem-pipeline-design.md`

## Orientation (read before starting)

- `python/genvarloader/_ragged.py` — `Ragged` re-export, `RaggedIntervals`, `RaggedTracks`, `RaggedAnnotatedHaps`, `to_padded` (seqpro pass-through), `reverse_complement_masked` (flat DNA RC), `_COMP` LUT. **`_Flat` lands here.**
- `python/genvarloader/_dataset/_query.py` — `getitem` (boundary), `_getitem_unspliced`, `_getitem_spliced`, `reverse_complement_ragged`, `pad`, `_regroup`. **The transform + boundary rewiring happens here.**
- `python/genvarloader/_dataset/_haps.py` — `Haps._reconstruct_haplotypes` (`:754`), `_reconstruct_annotated_haplotypes` (`:826`), `get_haps_and_shifts` (`:485`), `__call__` (`:455`).
- `python/genvarloader/_dataset/_ref.py` — `Ref.__call__` (`:27`).
- `python/genvarloader/_dataset/_tracks.py` — `Tracks._call_float32` (`:571`), `_call_intervals` (`:682`), `__call__` (`:548`).
- `python/genvarloader/_dataset/_reconstruct.py` — `RefTracks` (`:52`), `HapsTracks` (`:93`).
- `python/genvarloader/_dataset/_rag_variants.py` — `RaggedVariants` (an `ak.Array` subclass — awkward-native).
- `python/genvarloader/_utils.py` — `lengths_to_offsets`.
- `tests/benchmarks/profiling/profile.py` — profiling driver (`N_BATCHES`, `BATCH`).

**Key facts pinned from the current code:**
- `Ragged.from_offsets(data, shape, offsets)` — `shape` is a tuple with exactly one `None` (the ragged axis); `offsets` is `int64`, length `prod(non-None dims)+1`.
- Reconstructors already allocate a flat buffer, run a numba kernel that writes into `.data` using `.offsets`, then wrap with `Ragged.from_offsets`. The flat refactor replaces only that final wrap.
- `getitem` ordering today: densify (`pad`/`to_numpy`, lines 94–103) → `out_reshape` (105–106) → `squeeze` (108–110) → unwrap single tuple (112–113).
- For `output_length == int`, every row has exactly `output_length` elements, so densify is a pure reshape.
- The S1 `haps` RC already runs flat via `reverse_complement_masked`; only the int32 `var_idxs`/`ref_coords` reverse (`_query.py:340`) and the variants path still use awkward in the RC step.

## Regression gate (used by every task)

Two gates run after each implementation task:

1. **Characterization snapshot** (added in Task 1): `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q` — asserts `ds[...]` output is byte-identical to committed reference `.npz` files across every (mode × output_length) combination.
2. **Dataset suite** (final correctness net): `pixi run -e dev pytest tests/dataset -q -m "not slow"`.

If either regresses, the task is not done.

## File structure

- **Create** `python/genvarloader/_flat.py` — `_Flat` container, `_FlatAnnotatedHaps`, the flat int masked-reverse kernel. (Kept separate from `_ragged.py` so the flat transport has one clear home and `_ragged.py` stays the awkward-facing module.)
- **Create** `tests/dataset/test_flat.py` — unit tests for `_Flat` and the reverse kernel.
- **Create** `tests/dataset/test_flat_getitem_snapshot.py` — characterization gate.
- **Create** `tests/dataset/_snapshots/` — committed reference `.npz` files.
- **Modify** `python/genvarloader/_dataset/_query.py` — boundary + transforms.
- **Modify** `python/genvarloader/_dataset/{_ref,_tracks,_haps,_reconstruct}.py` — recon boundary.
- **Modify** `tests/benchmarks/profiling/profile.py` — `N_BATCHES`.
- **Create** `tests/benchmarks/profiling/run_pyspy.sh` (per re-profile task) — sudo profiling commands for David.

---

## Task 0: Profiling baseline (bump N_BATCHES, capture clean A/B reference)

**Files:**
- Modify: `tests/benchmarks/profiling/profile.py:28`
- Create: `tests/benchmarks/profiling/run_pyspy.sh`

- [ ] **Step 1: Bump iteration count**

In `tests/benchmarks/profiling/profile.py`, change line 28:

```python
N_BATCHES = 2000
```

(BATCH stays 32 — already ≥ 16 instances/batch.)

- [ ] **Step 2: Write the sudo profiling script**

Create `tests/benchmarks/profiling/run_pyspy.sh` (py-spy needs root on macOS; David runs this):

```bash
#!/usr/bin/env bash
# py-spy requires root on macOS. Run:  sudo bash tests/benchmarks/profiling/run_pyspy.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
PY="$(pixi run -e dev which python)"
PYSPY="$(pixi run -e dev which py-spy)"
OUT=tests/benchmarks/profiling
for mode in tracks haplotypes variants; do
  echo "=== py-spy $mode ==="
  "$PYSPY" record -o "$OUT/${mode}.speedscope.json" -f speedscope -- \
    "$PY" "$OUT/profile.py" --mode "$mode"
done
echo "Wrote $OUT/{tracks,haplotypes,variants}.speedscope.json"
```

- [ ] **Step 3: Capture the memray baseline (no root needed; agent runs this)**

Run:
```bash
pixi run -e dev memray run -fo tests/benchmarks/profiling/tracks.baseline.memray.bin tests/benchmarks/profiling/profile.py --mode tracks
pixi run -e dev memray run -fo tests/benchmarks/profiling/haps.baseline.memray.bin tests/benchmarks/profiling/profile.py --mode haplotypes
pixi run -e dev memray run -fo tests/benchmarks/profiling/variants.baseline.memray.bin tests/benchmarks/profiling/profile.py --mode variants
```
Expected: three `.baseline.memray.bin` files; each run prints `done`.

- [ ] **Step 4: Hand the py-spy script to David**

Tell David: "Run `sudo bash tests/benchmarks/profiling/run_pyspy.sh` and let me know when the three `*.speedscope.json` files are written." Record the baseline hot-path self-time per mode in a scratch note for the REGRESSIONS A/B (used in Task 11).

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/profiling/profile.py tests/benchmarks/profiling/run_pyspy.sh
git commit --no-verify -m "perf(bench): 10x profiling iterations (N_BATCHES=2000) + sudo py-spy script"
```

(`--no-verify`: the pyrefly pre-commit hook fails in worktrees without the built Rust ext — documented gotcha.)

---

## Task 1: Characterization snapshot gate

Freeze current `getitem` outputs to committed `.npz` files so every later task proves byte-identity independently of the full suite.

**Files:**
- Create: `tests/dataset/test_flat_getitem_snapshot.py`
- Create: `tests/dataset/_snapshots/` (generated `.npz`, committed)

- [ ] **Step 1: Write the snapshot test (generates-on-first-run, asserts-thereafter)**

Create `tests/dataset/test_flat_getitem_snapshot.py`:

```python
"""Byte-identical characterization gate for the flat-buffer getitem refactor.

First run (no snapshot present) writes reference .npz files from the CURRENT
code; commit them. Every later run asserts getitem output is byte-identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import genvarloader as gvl
from genvarloader._ragged import RaggedAnnotatedHaps
from seqpro.rag import Ragged

# Reuse the same test fixture the rest of tests/dataset uses.
from tests.dataset.conftest import dataset_path_with_ref  # type: ignore

SNAP = Path(__file__).parent / "_snapshots"
SEQLEN = 64
CASES = [
    ("haps_ragged", dict(seqs="haplotypes"), "ragged"),
    ("haps_fixed", dict(seqs="haplotypes"), SEQLEN),
    ("haps_variable", dict(seqs="haplotypes"), "variable"),
    ("annot_fixed", dict(seqs="annotated"), SEQLEN),
    ("tracks_ragged", dict(seqs=None, tracks="read-depth"), "ragged"),
    ("tracks_fixed", dict(seqs=None, tracks="read-depth"), SEQLEN),
    ("ref_fixed", dict(seqs="reference"), SEQLEN),
    ("haps_tracks_fixed", dict(seqs="haplotypes", tracks="read-depth"), SEQLEN),
]


def _build(ds, seqs, tracks=None, out_len="ragged"):
    ds = ds.with_seqs(seqs)
    if tracks is not None:
        ds = ds.with_tracks(tracks)
    if out_len == "ragged":
        return ds
    if out_len == "variable":
        return ds.with_len("variable")
    return ds.with_len(out_len)


def _flatten_output(obj) -> dict[str, np.ndarray]:
    """Normalize any getitem return into a dict of plain ndarrays."""
    out = {}
    if isinstance(obj, tuple):
        for i, o in enumerate(obj):
            out.update({f"{i}_{k}": v for k, v in _flatten_output(o).items()})
    elif isinstance(obj, RaggedAnnotatedHaps):
        out["haps_data"] = np.asarray(obj.haps.data)
        out["haps_off"] = np.asarray(obj.haps.offsets)
        out["vidx_data"] = np.asarray(obj.var_idxs.data)
        out["pos_data"] = np.asarray(obj.ref_coords.data)
    elif isinstance(obj, Ragged):
        out["data"] = np.asarray(obj.data)
        out["off"] = np.asarray(obj.offsets)
    elif hasattr(obj, "haps"):  # AnnotatedHaps (dense)
        out["haps"] = np.asarray(obj.haps)
        out["var_idxs"] = np.asarray(obj.var_idxs)
        out["ref_coords"] = np.asarray(obj.ref_coords)
    else:
        out["arr"] = np.asarray(obj)
    return out


@pytest.mark.parametrize("name,view,out_len", [(c[0], c[1], c[2]) for c in CASES])
def test_getitem_snapshot(dataset_path_with_ref, name, view, out_len):
    ds = gvl.Dataset.open(*dataset_path_with_ref)
    ds = _build(ds, view["seqs"], view.get("tracks"), out_len)
    regions = list(range(min(8, ds.shape[0])))
    samples = [i % ds.shape[1] for i in range(len(regions))]
    result = _flatten_output(ds[regions, samples])

    SNAP.mkdir(exist_ok=True)
    path = SNAP / f"{name}.npz"
    if not path.exists():
        np.savez(path, **result)
        pytest.skip(f"wrote snapshot {path.name}; commit it and re-run")
    ref = np.load(path)
    assert set(ref.files) == set(result), f"{name}: key drift"
    for k in ref.files:
        np.testing.assert_array_equal(result[k], ref[k], err_msg=f"{name}:{k}")
```

> **Implementer note:** `dataset_path_with_ref` is illustrative — inspect `tests/dataset/conftest.py` and use the existing fixture that yields a `(dataset_dir, reference_fasta)` pair with haplotypes + a `read-depth` track. If none exists, build one from `tests/data` (produced by `pixi run -e dev gen`) following the pattern in `tests/dataset/test_dataset.py`. Pick `seqs`/`tracks` names that exist in the fixture. Drop any CASES the fixture can't satisfy (e.g. `variants` if no multi-allelic fixture) and note it in the commit.

- [ ] **Step 2: Generate the snapshots (first run writes, skips)**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q`
Expected: all cases SKIP with "wrote snapshot …".

- [ ] **Step 3: Re-run to confirm the gate passes against committed snapshots**

Run: `pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q`
Expected: all cases PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/dataset/test_flat_getitem_snapshot.py tests/dataset/_snapshots
git commit --no-verify -m "test(dataset): byte-identical getitem characterization snapshots"
```

---

## Task 2: `_Flat` container

**Files:**
- Create: `python/genvarloader/_flat.py`
- Create: `tests/dataset/test_flat.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/dataset/test_flat.py`:

```python
import numpy as np
import pytest
from seqpro.rag import Ragged

from genvarloader._flat import _Flat


def _rag(data, shape, offsets):
    return Ragged.from_offsets(data, shape, np.asarray(offsets, np.int64))


def test_to_ragged_roundtrip():
    data = np.arange(6, dtype=np.int32)
    f = _Flat.from_offsets(data, (2, None), np.array([0, 3, 6], np.int64))
    r = f.to_ragged()
    np.testing.assert_array_equal(r.data, data)
    np.testing.assert_array_equal(r.offsets, [0, 3, 6])
    assert r.shape == (2, None)


def test_to_fixed_matches_ragged_to_numpy():
    data = np.arange(8, dtype=np.float32)
    off = np.array([0, 4, 8], np.int64)
    f = _Flat.from_offsets(data, (2, None), off)
    expected = _rag(data, (2, None), off).to_numpy()
    np.testing.assert_array_equal(f.to_fixed(4), expected)


def test_to_fixed_multi_outer():
    data = np.arange(12, dtype=np.float32)
    off = np.arange(0, 13, 2, dtype=np.int64)  # 6 rows of length 2
    f = _Flat.from_offsets(data, (3, 2, None), off)
    out = f.to_fixed(2)
    assert out.shape == (3, 2, 2)
    np.testing.assert_array_equal(out.reshape(-1), data)


def test_to_padded_matches_seqpro():
    data = np.array([1, 2, 3, 4, 5], np.int32)
    off = np.array([0, 2, 5], np.int64)  # rows len 2 and 3
    f = _Flat.from_offsets(data, (2, None), off)
    from genvarloader._ragged import to_padded
    expected = to_padded(_rag(data, (2, None), off), -1)
    np.testing.assert_array_equal(f.to_padded(-1), expected)


def test_view_changes_dtype_not_offsets():
    data = np.zeros(4, np.uint8)
    f = _Flat.from_offsets(data, (2, None), np.array([0, 2, 4], np.int64))
    fv = f.view("S1")
    assert fv.data.dtype == np.dtype("S1")
    np.testing.assert_array_equal(fv.offsets, f.offsets)


def test_squeeze_outer_one():
    data = np.arange(4, dtype=np.int32)
    f = _Flat.from_offsets(data, (1, 2, None), np.array([0, 2, 4], np.int64))
    s = f.squeeze(0)
    assert s.shape == (2, None)
    np.testing.assert_array_equal(s.offsets, f.offsets)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_flat.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'genvarloader._flat'`.

- [ ] **Step 3: Implement `_Flat`**

Create `python/genvarloader/_flat.py`:

```python
"""Flat-buffer ragged transport used inside the getitem hot path.

`_Flat` is a pure-numpy `(data, offsets, shape)` container. Unlike seqpro
`Ragged` it never wraps an awkward array, so operating on it runs no awkward
kernels. It converts to seqpro `Ragged` only via `to_ragged()`, called at the
getitem return boundary when the caller requested ragged output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic

import numpy as np
from numpy.typing import NDArray
from seqpro.rag import RDTYPE_co as RDTYPE
from seqpro.rag import Ragged
from seqpro.rag import to_padded as _sp_to_padded


@dataclass(slots=True, frozen=True)
class _Flat(Generic[RDTYPE]):
    data: NDArray
    offsets: NDArray[np.int64]
    shape: tuple[int | None, ...]  # outer fixed dims; exactly one None (ragged axis)

    @classmethod
    def from_offsets(
        cls, data: NDArray, shape: tuple[int | None, ...], offsets: NDArray
    ) -> "_Flat":
        return cls(data, np.asarray(offsets, np.int64), tuple(shape))

    @property
    def rag_dim(self) -> int:
        return self.shape.index(None)

    @property
    def n_rows(self) -> int:
        return int(np.prod([d for d in self.shape if d is not None], dtype=np.int64))

    def view(self, dtype: Any) -> "_Flat":
        return _Flat(self.data.view(dtype), self.offsets, self.shape)

    def to_ragged(self) -> Ragged:
        return Ragged.from_offsets(self.data, self.shape, self.offsets)

    def to_fixed(self, length: int) -> NDArray:
        """Densify when every row has exactly `length` elements: pure reshape."""
        outer = tuple(d for d in self.shape if d is not None)
        return self.data.reshape(*outer, length)

    def to_padded(self, pad_value: Any) -> NDArray:
        """Variable-length densify via the flat seqpro kernel."""
        return _sp_to_padded(self.to_ragged(), pad_value)

    def reshape(self, shape: int | tuple[int, ...]) -> "_Flat":
        if isinstance(shape, int):
            shape = (shape,)
        new = tuple(shape) + (None,)
        return _Flat(self.data, self.offsets, new)

    def squeeze(self, axis: int | None = None) -> "_Flat":
        outer = [d for d in self.shape if d is not None]
        if axis is None:
            outer = [d for d in outer if d != 1]
        else:
            if outer[axis] != 1:
                raise ValueError(f"cannot squeeze axis {axis} with size {outer[axis]}")
            del outer[axis]
        return _Flat(self.data, self.offsets, (*outer, None))
```

> **Implementer note:** `reshape` mirrors `Ragged.reshape` semantics — it re-labels the outer (fixed) dims and keeps the ragged axis last; offsets are unchanged because the number and contents of rows don't change. Confirm against `Ragged.reshape` behavior for the shapes `getitem`'s `out_reshape` produces (it prepends the user's index shape, e.g. `(n_regions, n_samples)`); add a test if a shape in `_query.py`'s `out_reshape` path isn't covered.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_flat.py tests/dataset/test_flat.py
git commit --no-verify -m "feat(flat): _Flat numpy ragged transport (no awkward kernels)"
```

---

## Task 3: Flat masked-reverse kernel + `_Flat.reverse_masked`

Replaces the awkward `ak.where(to_rc, rag[..., ::-1], rag)` reverse for int32 `var_idxs`/`ref_coords`, and routes S1 DNA through the existing flat seqpro RC.

**Files:**
- Modify: `python/genvarloader/_flat.py`
- Modify: `tests/dataset/test_flat.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/dataset/test_flat.py`:

```python
def test_reverse_masked_int_matches_awkward():
    import awkward as ak
    data = np.arange(10, dtype=np.int32)
    off = np.array([0, 3, 6, 10], np.int64)  # 3 rows
    mask = np.array([True, False, True])
    f = _Flat.from_offsets(data.copy(), (3, None), off)
    out = f.reverse_masked(mask)
    # awkward reference: reverse masked rows only
    rag = _rag(data.copy(), (3, None), off)
    expected = ak.to_packed(ak.where(mask, rag[..., ::-1], rag))
    np.testing.assert_array_equal(out.data, ak.flatten(expected, None).to_numpy())
    np.testing.assert_array_equal(out.offsets, off)


def test_reverse_masked_dna_matches_existing():
    from genvarloader._ragged import reverse_complement_masked, _COMP  # noqa
    seq = np.frombuffer(b"ACGTAACCGGTT", dtype="S1")
    off = np.array([0, 4, 12], np.int64)  # 2 rows
    mask = np.array([True, False])
    f = _Flat.from_offsets(seq.view(np.uint8).copy(), (2, None), off)
    out = f.reverse_masked(mask, comp=_COMP).view("S1")
    expected = reverse_complement_masked(_rag(seq.copy(), (2, None), off), mask)
    np.testing.assert_array_equal(out.data, np.asarray(expected.data))
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_flat.py -k reverse -q`
Expected: FAIL — `_Flat` has no attribute `reverse_masked`.

- [ ] **Step 3: Implement the kernel + method**

Add to `python/genvarloader/_flat.py` (imports `numba as nb`; for DNA, import inside the method to avoid a circular import with `_ragged`):

```python
import numba as nb


@nb.njit(parallel=True, cache=True)
def _reverse_rows_masked(data, offsets, mask):  # pragma: no cover - njit
    n = mask.shape[0]
    for i in nb.prange(n):
        if mask[i]:
            lo = offsets[i]
            hi = offsets[i + 1] - 1
            while lo < hi:
                tmp = data[lo]
                data[lo] = data[hi]
                data[hi] = tmp
                lo += 1
                hi -= 1
```

Add the method to `_Flat`:

```python
    def reverse_masked(self, mask: NDArray[np.bool_], comp: NDArray | None = None) -> "_Flat":
        """Reverse (DNA: reverse-complement) the `mask`-selected rows, in place.

        `mask` is one entry per outer query; replicate across any inner fixed
        axes in C order to get one entry per flattened ragged row, matching the
        awkward `ak.where` broadcast it replaces.
        """
        m = np.ascontiguousarray(mask, np.bool_).reshape(-1)
        if m.size != self.n_rows:
            factor, rem = divmod(self.n_rows, m.size)
            if rem != 0:
                raise ValueError(
                    f"mask has {m.size} entries but {self.n_rows} rows "
                    "(not an integer multiple)."
                )
            m = np.repeat(m, factor)
        if comp is not None:
            # DNA reverse-complement via the flat seqpro kernel (reuses gvl's LUT).
            from ._ragged import reverse_complement_masked

            rag = reverse_complement_masked(self.to_ragged(), m)
            return _Flat(np.asarray(rag.data), self.offsets, self.shape)
        _reverse_rows_masked(self.data, self.offsets, m)
        return self
```

> **Implementer note:** `reverse_complement_masked` already does the mask replication; passing the already-replicated `m` is harmless (replication is idempotent when `m.size == n_rows`). If pyrefly complains about the circular import, the local import inside the method is the fix.

- [ ] **Step 4: Run to verify pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat.py -q`
Expected: PASS (8 tests).

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_flat.py tests/dataset/test_flat.py
git commit --no-verify -m "feat(flat): masked reverse/RC on flat buffers"
```

---

## Task 4: Flat-aware boundary in `_query.py` (coexists with legacy Ragged)

Teach `reverse_complement_ragged`, `pad`, the int-densify branch, and a new final wrap step to handle `_Flat`/`_FlatAnnotatedHaps`, while leaving the legacy `Ragged` branches intact. No reconstructor returns `_Flat` yet, so this task changes no outputs — the snapshot gate must stay green.

**Files:**
- Modify: `python/genvarloader/_dataset/_query.py`
- Create: `python/genvarloader/_flat.py` (add `_FlatAnnotatedHaps`)
- Modify: `tests/dataset/test_flat.py`

- [ ] **Step 1: Add `_FlatAnnotatedHaps` + failing test**

Add to `python/genvarloader/_flat.py`:

```python
@dataclass(slots=True)
class _FlatAnnotatedHaps:
    haps: _Flat
    var_idxs: _Flat
    ref_coords: _Flat

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.haps.shape

    def reverse_masked(self, mask: NDArray[np.bool_], comp: NDArray) -> "_FlatAnnotatedHaps":
        self.haps = self.haps.reverse_masked(mask, comp=comp)
        self.var_idxs = self.var_idxs.reverse_masked(mask)
        self.ref_coords = self.ref_coords.reverse_masked(mask)
        return self

    def reshape(self, shape) -> "_FlatAnnotatedHaps":
        return _FlatAnnotatedHaps(
            self.haps.reshape(shape),
            self.var_idxs.reshape(shape),
            self.ref_coords.reshape(shape),
        )

    def squeeze(self, axis=None) -> "_FlatAnnotatedHaps":
        return _FlatAnnotatedHaps(
            self.haps.squeeze(axis), self.var_idxs.squeeze(axis), self.ref_coords.squeeze(axis)
        )

    def to_ragged(self):
        from ._ragged import RaggedAnnotatedHaps  # boundary import

        return RaggedAnnotatedHaps(
            self.haps.view("S1").to_ragged(),
            self.var_idxs.to_ragged(),
            self.ref_coords.to_ragged(),
        )

    def to_fixed(self, length: int):
        from ._types import AnnotatedHaps

        return AnnotatedHaps(
            self.haps.view("S1").to_fixed(length),
            self.var_idxs.to_fixed(length),
            self.ref_coords.to_fixed(length),
        )

    def to_padded(self):
        from ._types import AnnotatedHaps

        return AnnotatedHaps(
            self.haps.view("S1").to_padded(b"N"),
            self.var_idxs.to_padded(-1),
            self.ref_coords.to_padded(np.iinfo(self.ref_coords.data.dtype).max),
        )
```

Append to `tests/dataset/test_flat.py`:

```python
def test_flat_annotated_to_ragged():
    from genvarloader._flat import _Flat, _FlatAnnotatedHaps
    off = np.array([0, 2, 4], np.int64)
    h = _Flat.from_offsets(np.frombuffer(b"ACGT", "S1").view(np.uint8).copy(), (2, None), off)
    v = _Flat.from_offsets(np.array([0, 1, 2, 3], np.int32), (2, None), off)
    p = _Flat.from_offsets(np.array([10, 11, 12, 13], np.int32), (2, None), off)
    rah = _FlatAnnotatedHaps(h, v, p).to_ragged()
    assert rah.haps.data.dtype == np.dtype("S1")
    np.testing.assert_array_equal(np.asarray(rah.var_idxs.data), [0, 1, 2, 3])
```

> **Implementer note:** confirm `AnnotatedHaps` lives in `genvarloader._types` and its field order (`haps, var_idxs, ref_coords`) by reading `_types.py`. The `to_padded` pad values mirror `RaggedAnnotatedHaps.to_padded` (`_ragged.py:205-211`): `b"N"`, `-1`, `iinfo(dtype).max`.

- [ ] **Step 2: Run to verify failure, then make the unit test pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat.py -k annotated -q`
Expected: FAIL then (after implementing above) PASS.

- [ ] **Step 3: Make `reverse_complement_ragged` flat-aware**

In `python/genvarloader/_dataset/_query.py`, add to the imports:

```python
from .._flat import _Flat, _FlatAnnotatedHaps
from .._ragged import _COMP  # the A<->T,C<->G LUT
```

> If `_COMP` is not exported from `_ragged`, import it directly (it's a module global at `_ragged.py:280`).

At the top of `reverse_complement_ragged` (before the `isinstance(rag, Ragged)` chain at `:335`), add:

```python
    if isinstance(rag, _Flat):
        comp = _COMP if rag.data.dtype == np.uint8 else None
        return rag.reverse_masked(to_rc, comp=comp)
    if isinstance(rag, _FlatAnnotatedHaps):
        return rag.reverse_masked(to_rc, _COMP)
```

> **Implementer note:** the S1 haps buffer is stored as `uint8` inside `_Flat` until `.view("S1")` at the boundary, so the `dtype == np.uint8` check selects DNA RC for haps and plain reverse for int32 `var_idxs`/`ref_coords`. Verify the reconstructor (Task 6) stores haps as `uint8` in `_Flat`; if it stores `S1`, branch on `rag.data.dtype.kind == "S"` instead.

- [ ] **Step 4: Make `pad` and the int-densify + final-wrap steps flat-aware**

In `pad` (`:357`), add before the `isinstance(rag, Ragged)` chain:

```python
    if isinstance(rag, (_Flat, _FlatAnnotatedHaps)):
        if isinstance(rag, _Flat):
            pad_value = b"N" if rag.data.dtype.kind in "SU" else 0
            return rag.view("S1").to_padded(pad_value) if rag.data.dtype == np.uint8 else rag.to_padded(pad_value)
        return rag.to_padded()
```

In `getitem`, replace the densify block (`:94-103`) so it handles flat and routes ragged-output flats to `to_ragged`:

```python
    if view.output_length == "variable":
        recon = tuple(
            r if isinstance(r, (RaggedVariants, RaggedIntervals)) else pad(r)
            for r in recon
        )
    elif isinstance(view.output_length, int):
        recon = tuple(
            r if isinstance(r, (RaggedVariants, RaggedIntervals))
            else r.to_fixed(view.output_length) if isinstance(r, (_Flat, _FlatAnnotatedHaps))
            else r.to_numpy()
            for r in recon
        )
```

Then, immediately before the `out_reshape` step (`:105`), add a final wrap so any still-flat (ragged-output) element becomes its public `Ragged` type:

```python
    recon = tuple(
        o.to_ragged() if isinstance(o, (_Flat, _FlatAnnotatedHaps)) else o
        for o in recon
    )
```

> **Implementer note:** for ragged output, `reshape`/`squeeze` then run on the resulting `Ragged` (existing behavior). `_Flat.reshape`/`squeeze` exist too, so if you prefer to reshape-then-wrap, that also works — but wrapping first keeps the diff smallest and reuses the proven `Ragged` reshape. The `to_fixed`/`to_padded` for `_Flat[uint8]` must `.view("S1")` first so dtype matches the legacy `to_numpy()` output; the helper methods handle this when called via `_FlatAnnotatedHaps`, but a bare S1 `_Flat` (haplotypes mode) needs the `.view("S1")` — encode that in `_Flat.to_fixed`/`to_padded` by checking `data.dtype == np.uint8`? No — keep `_Flat` dtype-agnostic; instead the reconstructor returns the haps `_Flat` already `.view("S1")` (Task 6). Verify dtype at the boundary with the snapshot gate.

- [ ] **Step 5: Run both gates (no behavior change expected)**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_flat.py tests/dataset/test_flat_getitem_snapshot.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: all PASS (no reconstructor returns `_Flat` yet, so the new branches are dormant).

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_flat.py python/genvarloader/_dataset/_query.py tests/dataset/test_flat.py
git commit --no-verify -m "feat(query): flat-aware getitem boundary (legacy Ragged still supported)"
```

---

## Task 5: Tracks reconstructor → `_Flat`

Simplest reconstructor: `_call_float32`'s non-splice path already builds `(out, out_offsets, out_shape)`.

**Files:**
- Modify: `python/genvarloader/_dataset/_tracks.py:619-624`

- [ ] **Step 1: Return `_Flat` from `_call_float32` (non-splice path)**

In `python/genvarloader/_dataset/_tracks.py`, add import:

```python
from .._flat import _Flat
```

Replace lines `619-624` (the `out_shape` + `RaggedTracks.from_offsets` + `return cast(...)`):

```python
            out_shape = (len(idx), len(self.active_tracks), None)
            # flat (b t l)
            return cast(RaggedTracks, _Flat.from_offsets(out, out_shape, out_offsets))
```

> **Implementer note:** the `cast(RaggedTracks, …)` keeps the declared return type quiet under pyrefly (`bad-return` is warn-level). The object is a `_Flat`; the boundary in Task 4 handles it. Leave the splice-plan path (`:626-680`) returning `RaggedTracks.from_offsets` for now — Task 9 converts it.

- [ ] **Step 2: Run gates**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -k tracks -q
pixi run -e dev pytest tests/dataset -q -m "not slow" -k "track or Track"
```
Expected: PASS — `tracks_fixed`, `tracks_ragged` byte-identical.

- [ ] **Step 3: Full gate + commit**

Run: `pixi run -e dev pytest tests/dataset -q -m "not slow"`
Expected: PASS.

```bash
git add python/genvarloader/_dataset/_tracks.py
git commit --no-verify -m "perf(tracks): return _Flat from float32 reconstruction"
```

- [ ] **Step 4: Re-profile tracks (hand script to David)**

Run `pixi run -e dev memray run -fo tests/benchmarks/profiling/tracks.t5.memray.bin tests/benchmarks/profiling/profile.py --mode tracks` and ask David to re-run `sudo bash tests/benchmarks/profiling/run_pyspy.sh` (tracks mode). Record the tracks hot-path self-time delta vs Task 0 baseline in a scratch note for Task 11.

---

## Task 6: Haps reconstructor (`RaggedSeqs` + `RaggedAnnotatedHaps`) → flat

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py:754-784` (`_reconstruct_haplotypes`), `:826-879` (`_reconstruct_annotated_haplotypes`), `:511-516` (`get_haps_and_shifts` assembly)

- [ ] **Step 1: Return `_Flat` from `_reconstruct_haplotypes` (non-splice path)**

Add import to `_haps.py`:
```python
from .._flat import _Flat, _FlatAnnotatedHaps
```

Replace the non-splice body of `_reconstruct_haplotypes` (`:758-784`). Allocate the buffer/offsets directly instead of via `Ragged.from_offsets`, run the kernel, return a `_Flat` viewed as S1:

```python
        if req.splice_plan is None:
            data = np.empty(req.out_offsets[-1], np.uint8)
            shape = (*req.shifts.shape, None)
            reconstruct_haplotypes_from_sparse(
                geno_offset_idx=req.geno_offset_idx,
                out=data,
                out_offsets=req.out_offsets,
                regions=req.regions,
                shifts=req.shifts,
                geno_offsets=self.genotypes.offsets,
                geno_v_idxs=self.genotypes.data,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
                alt_alleles=self.variants.alt.data.view(np.uint8),
                alt_offsets=self.variants.alt.offsets,
                ref=self.reference.reference,
                ref_offsets=self.reference.offsets,
                pad_char=self.reference.pad_char,
                keep=req.keep,
                keep_offsets=req.keep_offsets,
                annot_v_idxs=None,
                annot_ref_pos=None,
            )
            return cast("Ragged[np.bytes_]", _Flat.from_offsets(data, shape, req.out_offsets).view("S1"))
```

> **Implementer note:** the kernel writes into `data` (uint8); `.view("S1")` makes the `_Flat` dtype `S1` so the boundary's S1/`to_fixed` path matches the old `to_numpy()` dtype, AND the RC branch must then test `data.dtype.kind == "S"` (see Task 4 Step 3 note) — pick one convention and make both consistent. Recommended: keep `_Flat` data as `S1` for haps and branch RC on `dtype.kind == "S"`; update the Task 4 RC check accordingly. Leave the splice path (`:786-824`) on `Ragged` for Task 9.

- [ ] **Step 2: Return `_FlatAnnotatedHaps` from the annotated reconstruction**

Replace the non-splice body of `_reconstruct_annotated_haplotypes` (`:837-879`) to allocate three flat buffers, run the kernel, and return a tuple of `_Flat`s (keeping the method's `tuple` return contract):

```python
        if req.splice_plan is None:
            shape = (*req.shifts.shape, None)
            haps = np.empty(req.out_offsets[-1], np.uint8)
            annot_v = np.empty(req.out_offsets[-1], V_IDX_TYPE)
            annot_pos = np.empty(req.out_offsets[-1], np.int32)
            reconstruct_haplotypes_from_sparse(
                geno_offset_idx=req.geno_offset_idx,
                out=haps,
                out_offsets=req.out_offsets,
                regions=req.regions,
                shifts=req.shifts,
                geno_offsets=self.genotypes.offsets,
                geno_v_idxs=self.genotypes.data,
                v_starts=self.variants.start,
                ilens=self.variants.ilen,
                alt_alleles=self.variants.alt.data.view(np.uint8),
                alt_offsets=self.variants.alt.offsets,
                ref=self.reference.reference,
                ref_offsets=self.reference.offsets,
                pad_char=self.reference.pad_char,
                keep=req.keep,
                keep_offsets=req.keep_offsets,
                annot_v_idxs=annot_v,
                annot_ref_pos=annot_pos,
            )
            return (
                cast("Ragged[np.bytes_]", _Flat.from_offsets(haps, shape, req.out_offsets).view("S1")),
                cast("Ragged", _Flat.from_offsets(annot_v, shape, req.out_offsets)),
                cast("Ragged", _Flat.from_offsets(annot_pos, shape, req.out_offsets)),
            )
```

- [ ] **Step 3: Assemble `_FlatAnnotatedHaps` in `get_haps_and_shifts`**

In `get_haps_and_shifts` (`:514-516`), the `RaggedAnnotatedHaps` branch must wrap the flat tuple in `_FlatAnnotatedHaps` instead of `RaggedAnnotatedHaps`:

```python
        elif issubclass(self.kind, RaggedAnnotatedHaps):
            haps, annot_v_idx, annot_pos = self._reconstruct_annotated_haplotypes(req)
            out = _FlatAnnotatedHaps(haps, annot_v_idx, annot_pos)
```

> **Implementer note:** `out` is later returned as `_H`; the `cast(_H, …)`/return annotations stay (pyrefly warn-level). The boundary handles `_FlatAnnotatedHaps`.

- [ ] **Step 4: Run gates**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -k "haps or annot" -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: PASS — `haps_*`, `annot_fixed` byte-identical, full suite green.

- [ ] **Step 5: Commit + re-profile haps**

```bash
git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_query.py python/genvarloader/_flat.py
git commit --no-verify -m "perf(haps): return _Flat / _FlatAnnotatedHaps from reconstruction"
```
Run `pixi run -e dev memray run -fo tests/benchmarks/profiling/haps.t6.memray.bin tests/benchmarks/profiling/profile.py --mode haplotypes`; ask David to re-run the py-spy script (haplotypes). Record delta.

---

## Task 7: Ref reconstructor → `_Flat`

**Files:**
- Modify: `python/genvarloader/_dataset/_ref.py:49-67`

- [ ] **Step 1: Return `_Flat` from the non-splice path**

Add `from .._flat import _Flat` to `_ref.py`. Replace `:49-67`:

```python
        if splice_plan is None:
            out_offsets = lengths_to_offsets(out_lengths)
            ref = get_reference(
                regions=regions,
                out_offsets=out_offsets,
                reference=self.reference.reference,
                ref_offsets=self.reference.offsets,
                pad_char=self.reference.pad_char,
            )  # uint8 flat buffer
            return cast(
                "Ragged[np.bytes_]",
                _Flat.from_offsets(ref, (batch_size, None), out_offsets).view("S1"),
            )
```

> **Implementer note:** `get_reference` returns the flat uint8 buffer; the old code did `.view("S1")` before `from_offsets`. Keep the `.view("S1")` on the `_Flat` so dtype matches. Leave `_fetch_spliced_ref` (`:70`) for Task 9.

- [ ] **Step 2: Run gates + commit**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -k ref -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: PASS.

```bash
git add python/genvarloader/_dataset/_ref.py
git commit --no-verify -m "perf(ref): return _Flat from reference reconstruction"
```

---

## Task 8: Compound reconstructors (`HapsTracks`, `RefTracks`)

These call the leaf reconstructors (now flat) and `HapsTracks` also builds tracks via its own kernel block (`_reconstruct.py:151-228`).

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py:220-235` (HapsTracks tracks wrap); `RefTracks` needs no change if leaves return flat.

- [ ] **Step 1: Confirm `RefTracks` needs no change**

`RefTracks.__call__` (`:57-90`) returns `(seqs, tracks)` straight from the leaf reconstructors. Both now return `_Flat`. Add `from .._flat import _Flat` only if a type reference is needed; otherwise no code change. Verify by running the `ref` + `tracks` snapshot cases.

- [ ] **Step 2: Make `HapsTracks` tracks block return `_Flat`**

In `_reconstruct.py`, add `from .._flat import _Flat`. Replace the tracks wrap (`:220-228`):

```python
            out_shape = (
                len(idx),
                len(self.tracks.active_tracks),
                self.haps.genotypes.shape[-2],
                None,
            )
            # flat (b t [p] l)
            tracks = _Flat.from_offsets(out, out_shape, out_offsets)
```

The `haps` returned by `get_haps_and_shifts` is already flat (Task 6). The `else: tracks = self.tracks._call_intervals(idx)` branch (`:231`) stays awkward (RaggedIntervals output, not in the fixed/ragged hot path); leave it.

- [ ] **Step 3: Run gates + commit**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -k "haps_tracks or ref" -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: PASS.

```bash
git add python/genvarloader/_dataset/_reconstruct.py
git commit --no-verify -m "perf(reconstruct): _Flat tracks in HapsTracks; RefTracks flat via leaves"
```

---

## Task 9: Spliced path parity

The spliced reconstruction paths (`_reconstruct_haplotypes` splice branch, `_reconstruct_annotated_haplotypes` splice branch, `Tracks._call_float32` splice branch, `_fetch_spliced_ref`) still return `Ragged`, and `_query._getitem_spliced` calls `_regroup` (`_query.py:372-392`) which rewraps offsets via `Ragged.from_offsets`. Convert these to `_Flat` and make `_regroup` flat.

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` (splice branches), `_tracks.py` (splice branch `:673-680`), `_ref.py`/`_reference.py` (`_fetch_spliced_ref`), `_query.py:372-392` (`_regroup`), `_query.py:242,249` (spliced RC + regroup call sites).

- [ ] **Step 1: Add spliced snapshot cases**

Append spliced cases to `tests/dataset/test_flat_getitem_snapshot.py` CASES (regenerate snapshots for the new cases only — delete nothing existing):

```python
    ("haps_spliced", dict(seqs="haplotypes"), "ragged"),  # with a SpliceIndexer
```

> **Implementer note:** read `tests/dataset/` for an existing spliced test to copy the `subset_to`/splice setup; the snapshot harness's `_build` needs a splice branch. If splicing needs a specific BED/region setup the fixture lacks, add a minimal spliced unit test comparing flat vs a pre-refactor `Ragged` reference computed in the same test (capture by temporarily forcing the legacy path) instead of a committed snapshot.

- [ ] **Step 2: Convert `_regroup` to flat**

Replace `_regroup` (`_query.py:372-392`) to operate on `_Flat`/`_FlatAnnotatedHaps`:

```python
def _regroup(rag, group_offsets, out_shape):
    """Rewrap a per-element flat ragged with grouped offsets (shared data buffer)."""
    if isinstance(rag, _FlatAnnotatedHaps):
        return _FlatAnnotatedHaps(
            _regroup(rag.haps, group_offsets, out_shape),
            _regroup(rag.var_idxs, group_offsets, out_shape),
            _regroup(rag.ref_coords, group_offsets, out_shape),
        )
    return _Flat.from_offsets(rag.data, out_shape, group_offsets)
```

- [ ] **Step 3: Convert the splice branches of the reconstructors to return `_Flat`**

Apply the same `Ragged.from_offsets(buf, shape, off)` → `_Flat.from_offsets(buf, shape, off)` (with `.view("S1")` for byte buffers) transformation to: `_reconstruct_haplotypes` splice branch (`_haps.py:816-824`), `_reconstruct_annotated_haplotypes` splice branch (`:913-928`), `Tracks._call_float32` splice branch (`_tracks.py:673-680`), and `_fetch_spliced_ref` (in `_reference.py` — read it and convert its final wrap).

- [ ] **Step 4: Verify the spliced RC call site**

`_getitem_spliced` (`:242`) calls `reverse_complement_ragged(r, to_rc_per_elem)` — now flat-aware from Task 4, so no change needed. Confirm `to_rc_per_elem` length equals per-element row count (the flat `reverse_masked` replication asserts this).

- [ ] **Step 5: Run gates + commit**

Run:
```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -q
pixi run -e dev pytest tests/dataset -q -m "not slow"
```
Expected: PASS (including spliced tests in `tests/dataset`).

```bash
git add python/genvarloader/_dataset/ tests/dataset/test_flat_getitem_snapshot.py tests/dataset/_snapshots
git commit --no-verify -m "perf(splice): flat-buffer spliced reconstruction + _regroup"
```

---

## Task 10: Variants path — spike + conditional conversion

`RaggedVariants` is an `ak.Array` subclass holding variable-length allele strings (`alt`/`ref`) plus numeric fields (`start`, `ilen`, optional `dosage`). It is genuinely awkward-native; a full flat rewrite may not pay off (variants mode was the lowest awkward share, 39%). This task measures, then converts only what's worthwhile.

**Files:**
- Modify (conditional): `python/genvarloader/_dataset/_rag_variants.py`, `_haps.py` `_get_variants`.

- [ ] **Step 1: Profile the variants path post-Task-9**

Run `pixi run -e dev memray run -fo tests/benchmarks/profiling/variants.t10.memray.bin tests/benchmarks/profiling/profile.py --mode variants`; ask David for a fresh py-spy of `variants`. Identify which awkward frames remain in the variants hot path (assembly via `ak.zip`/`ak.transform` at `_rag_variants.py:60-102`, `rc_`'s `ak.where`/`ak.str` at `:200-236`, fancy-indexing).

- [ ] **Step 2: Decision gate (document the call)**

Record in `REGRESSIONS.md` which applies:
  - **(a) Numeric fields dominate** → convert `start`/`ilen`/`dosage` `rc_` and assembly to flat `_Flat` ops, keep `alt`/`ref` allele strings awkward-native (they are variable-length-of-variable-length; flattening buys little). Implement the flat numeric `rc_` (reverse via the Task 3 kernel; `start`/`ilen` recompute uses `_rc_numba_helper` at `:517`).
  - **(b) Allele-string ops dominate** → leave `RaggedVariants` awkward-native; the realistic end state is "Ragged-as-container at return for variants" with minimal pre-return awkward. Note this as the documented limit of "as much as possible."

- [ ] **Step 3: Implement the chosen path (if (a))**

For (a), write flat versions of the numeric-field transforms with a unit test comparing to the current `rc_` output (byte-identical), following the Task 3/6 pattern. Keep `RaggedVariants` as the return container (it's the public type).

- [ ] **Step 4: Run gates + commit**

Run: `pixi run -e dev pytest tests/dataset -q -m "not slow"`
Expected: PASS.

```bash
git add python/genvarloader/_dataset/_rag_variants.py python/genvarloader/_dataset/_haps.py docs/superpowers/REGRESSIONS.md
git commit --no-verify -m "perf(variants): <flat numeric fields | documented awkward-native limit>"
```

---

## Task 11: Guard, final re-profile, REGRESSIONS write-up

**Files:**
- Modify: `python/genvarloader/_dataset/_query.py`, `_haps.py`, `_ragged.py` (remove dead legacy branches), `docs/superpowers/REGRESSIONS.md`.
- Create: `tests/dataset/test_no_awkward_in_hotpath.py`.

- [ ] **Step 1: Remove now-dead legacy branches**

Now that every reconstructor returns `_Flat`, delete the dormant legacy branches that can no longer be reached: the `isinstance(rag, Ragged)` byte/float branches in `reverse_complement_ragged` (`_query.py:335-340`) and the `Ragged`/`RaggedAnnotatedHaps` densify branches in `pad` and the int-densify step that are now unreachable. Keep `RaggedVariants`/`RaggedIntervals` branches (still real). Run the snapshot + full suite after deletion to confirm nothing was reachable.

> **Implementer note:** be conservative — delete a branch only after grepping that no reconstructor returns that type. `RaggedIntervals` (from `_call_intervals`) and `RaggedVariants` are still produced; their branches stay.

- [ ] **Step 2: Add a hot-path awkward guard test**

Create `tests/dataset/test_no_awkward_in_hotpath.py` — a regression guard that runs a representative `ds[...]` under a hook counting awkward kernel dispatches, asserting zero for the fixed/ragged haps + tracks + ref paths:

```python
"""Guard: the fixed/ragged getitem hot path must not dispatch awkward kernels."""
import numpy as np
import pytest
import genvarloader as gvl
from tests.dataset.conftest import dataset_path_with_ref  # type: ignore


def test_tracks_fixed_no_awkward(monkeypatch, dataset_path_with_ref):
    import awkward as ak

    calls = {"n": 0}
    orig = ak.to_numpy

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(ak, "to_numpy", counting)
    ds = gvl.Dataset.open(*dataset_path_with_ref).with_seqs(None).with_tracks("read-depth").with_len(64)
    _ = ds[[0, 1, 2, 3], [0, 0, 0, 0]]
    assert calls["n"] == 0
```

> **Implementer note:** `ak.to_numpy` is the cleanest single chokepoint to assert on. If the densify still routes through it for some path, widen the guard or fix the path. Add haps + ref variants of this test. This guard is the executable form of the spec's success criterion.

- [ ] **Step 3: Final re-profile (all modes) + A/B write-up**

Run the full memray set (`*.final.memray.bin`) and ask David for a final py-spy of all three modes. In `docs/superpowers/REGRESSIONS.md`, write the clean same-dataset A/B: Task 0 baseline (N_BATCHES=2000, pre-refactor) vs final, per mode, hot-path self-time + allocation. This is the confound-free isolation the spec called for. Note the variants outcome (Task 10 decision).

- [ ] **Step 4: Run full gates + commit**

Run:
```bash
pixi run -e dev pytest tests/dataset -q
pixi run -e dev ruff check python/
```
Expected: PASS / clean.

```bash
git add -A
git commit --no-verify -m "perf(getitem): guard awkward out of hot path + final A/B write-up"
```

- [ ] **Step 5: Update the skill if the public API changed**

The refactor must NOT change the public API (return types unchanged). Confirm `skills/genvarloader/SKILL.md` needs no change (per CLAUDE.md, public API changes require a skill update — this refactor has none). If any signature/default drifted, update the skill.

---

## Self-review notes (for the executor)

- **Snapshot fixture is the linchpin.** Task 1's `dataset_path_with_ref` is a placeholder name — the first action is to read `tests/dataset/conftest.py` and bind to the real fixture. If no single fixture has haplotypes + tracks + reference, build one in the snapshot module from `tests/data`. Do this before Task 2.
- **dtype convention for haps `_Flat`.** Decide once (recommended: store S1, branch RC on `dtype.kind == "S"`) and apply consistently in Tasks 4 and 6. The snapshot gate catches a mismatch immediately.
- **pyrefly is warn-level** for `bad-return`/`bad-argument-type`; the `cast(...)` wrappers keep declared types stable while the runtime object is `_Flat`. Commit with `--no-verify` (pre-commit pyrefly hook fails in worktrees without the built Rust ext).
- **Variants (Task 10) is a spike, not a forced rewrite** — the decision gate is real because `RaggedVariants` is awkward-native; "eliminate as much as possible" permits leaving allele-string ops awkward.
- **Profiling is David-run on macOS** (sudo). Every re-profile step emits/uses `run_pyspy.sh`; the agent runs only memray directly.
