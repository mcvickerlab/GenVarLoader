# Flat tracks/intervals + decoupled track re-alignment — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a flat (pure-numpy) interval output type, document/test the already-working flat tracks, and add a `with_settings(realign_tracks=...)` toggle that decouples track re-alignment from the seq mode — unlocking tracks alongside `variant-windows`.

**Architecture:** A new `FlatIntervals` container mirrors `RaggedIntervals` over three `_Flat` buffers. `Tracks._call_intervals` gains a pure-numpy `flat` branch. The non-realigning compound reconstructor `RefTracks` is generalized to `SeqsTracks` (any seq reconstructor + un-realigned tracks); `HapsTracks` remains only for the realigning float path. A new `realign_tracks: bool` Dataset field (default `True`) routes `_build_reconstructor` between `HapsTracks` (realign) and `SeqsTracks` (as-is).

**Tech Stack:** Python, numpy, numba, awkward (seqpro `Ragged`), maturin/pixi build, pytest.

## Global Constraints

- Run everything via pixi: tests `pixi run -e dev pytest ...`; lint `pixi run -e dev ruff check python/`; format `pixi run -e dev ruff format python/`; typecheck `pixi run -e dev typecheck`.
- Generate test data once before first run: `pixi run -e dev gen` (only needed for on-disk dataset tests; the dummy-dataset tests here do not need it).
- E501 (line length) is ignored by ruff; still keep lines reasonable.
- **A pre-push hook runs `ruff format` (≠ `ruff check`).** Run both before any push.
- Conventional commits (project uses commitizen).
- Public-API contract (CLAUDE.md): any change to `__all__`, `gvl.write`, `Dataset.open`, or any `Dataset.with_*` method/default MUST update `skills/genvarloader/SKILL.md`. This plan adds `FlatIntervals` and `realign_tracks` → SKILL update is mandatory (Task 6).
- `realign_tracks` default is `True` (preserves today's haplotypes+tracks re-alignment).
- `realign_tracks` lives on `with_settings` only — NOT on `Dataset.open` (scope limit).
- Breaking change (accepted): `haplotypes`/`annotated`/`variants` + `kind="intervals"` now raises unless `realign_tracks=False`.
- `with_insertion_fill` raises when `realign_tracks=False`.

---

### Task 1: `FlatIntervals` container type + public export

**Files:**
- Modify: `python/genvarloader/_ragged.py` (add `FlatIntervals`, extend `__all__`, import `_Flat`)
- Modify: `python/genvarloader/__init__.py` (export `FlatIntervals`)
- Test: `tests/dataset/test_flat_intervals.py` (create)

**Interfaces:**
- Consumes: `_Flat` from `python/genvarloader/_flat.py` (`_Flat.from_offsets(data, shape, offsets)`, `.to_ragged()`, `.reshape(outer)`, `.squeeze(axis)`).
- Produces: `genvarloader._ragged.FlatIntervals` (also `gvl.FlatIntervals`) — a `@dataclass(slots=True)` with fields `starts: _Flat`, `ends: _Flat`, `values: _Flat`; methods `.to_ragged() -> RaggedIntervals`, `.reshape(shape) -> FlatIntervals`, `.squeeze(axis=None) -> FlatIntervals`; property `.shape` (delegates to `values.shape`).

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_flat_intervals.py`:

```python
import awkward as ak
import numpy as np

from genvarloader._flat import _Flat
from genvarloader._ragged import FlatIntervals, RaggedIntervals


def _flat(data, offsets, dtype):
    return _Flat.from_offsets(
        np.asarray(data, dtype), (2, None), np.asarray(offsets, np.int64)
    )


def test_flat_intervals_to_ragged_roundtrip():
    # Two groups: group 0 has 1 interval, group 1 has 2 intervals.
    offsets = [0, 1, 3]
    fi = FlatIntervals(
        starts=_flat([10, 20, 30], offsets, np.int32),
        ends=_flat([15, 25, 35], offsets, np.int32),
        values=_flat([1.0, 2.0, 3.0], offsets, np.float32),
    )
    assert fi.shape == (2, None)
    ri = fi.to_ragged()
    assert isinstance(ri, RaggedIntervals)
    assert ak.to_list(ri.starts) == [[10], [20, 30]]
    assert ak.to_list(ri.ends) == [[15], [25, 35]]
    assert ak.to_list(ri.values) == [[1.0], [2.0, 3.0]]


def test_flat_intervals_public_export():
    import genvarloader as gvl

    assert gvl.FlatIntervals is FlatIntervals
    assert "FlatIntervals" in gvl.__all__
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_intervals.py -v`
Expected: FAIL — `ImportError: cannot import name 'FlatIntervals' from genvarloader._ragged`.

- [ ] **Step 3: Add `FlatIntervals` to `_ragged.py`**

In `python/genvarloader/_ragged.py`, add the `_Flat` import near the other intra-package imports (after line 19 `from ._torch import TORCH_AVAILABLE`):

```python
from ._flat import _Flat
```

Extend `__all__` (line 26) to:

```python
__all__ = ["FlatIntervals", "Ragged", "RaggedIntervals", "RaggedTracks"]
```

Then add this class immediately after the `RaggedIntervals` class definition (after line 106, the end of `to_packed`/before `to_nested_tensor_batch` is fine — place it after the whole `RaggedIntervals` class, i.e. after its last method):

```python
@dataclass(slots=True)
class FlatIntervals:
    """Flat-buffer analog of :class:`RaggedIntervals` over three :class:`_Flat` s.

    Pure-numpy ``(data, offsets, shape)`` per field; converts to the awkward-backed
    :class:`RaggedIntervals` only via :meth:`to_ragged`. Returned by eager indexing
    when ``with_tracks(kind="intervals")`` is combined with
    ``with_output_format("flat")``.
    """

    starts: _Flat
    ends: _Flat
    values: _Flat

    @property
    def shape(self) -> tuple[int | None, ...]:
        return self.values.shape

    def to_ragged(self) -> RaggedIntervals:
        return RaggedIntervals(
            self.starts.to_ragged(),
            self.ends.to_ragged(),
            self.values.to_ragged(),
        )

    def reshape(self, shape: int | tuple[int, ...]) -> "FlatIntervals":
        return FlatIntervals(
            self.starts.reshape(shape),
            self.ends.reshape(shape),
            self.values.reshape(shape),
        )

    def squeeze(self, axis: int | None = None) -> "FlatIntervals":
        return FlatIntervals(
            self.starts.squeeze(axis),
            self.ends.squeeze(axis),
            self.values.squeeze(axis),
        )
```

Note `to_ragged()` relies on `_Flat.to_ragged()` producing `Ragged[int32]`/`Ragged[float32]`; `RaggedIntervals` does not validate dtypes so the int/float split is preserved by the buffers themselves.

- [ ] **Step 4: Export from the package**

In `python/genvarloader/__init__.py`, change the `_ragged` import (line 34) to include `FlatIntervals`:

```python
from ._ragged import FlatIntervals, RaggedAnnotatedHaps, RaggedIntervals
```

Add `"FlatIntervals"` to `__all__` (keep alphabetical order — place after `"FlatAnnotatedHaps"`, line 51):

```python
    "FlatAnnotatedHaps",
    "FlatIntervals",
    "FlatRagged",
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat_intervals.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_ragged.py python/genvarloader/__init__.py tests/dataset/test_flat_intervals.py
rtk git commit -m "feat: add FlatIntervals flat-buffer interval container

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: flat interval reconstruction + getitem boundary

**Files:**
- Modify: `python/genvarloader/_dataset/_tracks.py` (add `build_flat_intervals`; thread `flat` through `Tracks.__call__` → `_call_intervals`)
- Modify: `python/genvarloader/_dataset/_query.py` (pass `FlatIntervals` through the flat boundary: reshape/squeeze/reverse-complement)
- Test: `tests/dataset/test_flat_intervals.py` (append)

**Interfaces:**
- Consumes: `FlatIntervals` (Task 1); `_Flat`; `lengths_to_offsets` from `python/genvarloader/_utils.py`; `TrackType` (in `_tracks.py`); the per-track `RaggedIntervals` memmaps in `Tracks.intervals` (each has `.starts.data`/`.starts.offsets`, `.ends.data`, `.values.data`).
- Produces: `build_flat_intervals(active_tracks, intervals, r_idx, s_idx, n_samples) -> FlatIntervals`; `Tracks.__call__(..., flat=False)` and `Tracks._call_intervals(idx, flat=False)` now honor `flat`.

- [ ] **Step 1: Write the failing test (append)**

Append to `tests/dataset/test_flat_intervals.py`:

```python
import genvarloader as gvl


def test_flat_intervals_end_to_end_matches_ragged():
    ds = gvl.get_dummy_dataset()
    idx = ([0, 1], [0, 1])

    rag = ds.with_seqs(None).with_tracks(["read-depth"], kind="intervals")
    flat = (
        ds.with_seqs(None)
        .with_tracks(["read-depth"], kind="intervals")
        .with_output_format("flat")
    )

    ri = rag[idx]
    fi = flat[idx]

    assert type(fi).__name__ == "FlatIntervals"
    assert isinstance(ri, gvl.RaggedIntervals)

    back = fi.to_ragged()
    assert ak.to_list(back.starts) == ak.to_list(ri.starts)
    assert ak.to_list(back.ends) == ak.to_list(ri.ends)
    assert ak.to_list(back.values) == ak.to_list(ri.values)


def test_flat_intervals_multi_track_matches_ragged():
    ds = gvl.get_dummy_dataset()
    idx = ([0, 1, 2], [0, 1, 2])
    names = ["read-depth", "annot"]

    ri = ds.with_seqs(None).with_tracks(names, kind="intervals")[idx]
    fi = (
        ds.with_seqs(None)
        .with_tracks(names, kind="intervals")
        .with_output_format("flat")[idx]
    )
    back = fi.to_ragged()
    # (batch, track, ~itv) C-order must match awkward concat order
    assert ak.to_list(back.starts) == ak.to_list(ri.starts)
    assert ak.to_list(back.values) == ak.to_list(ri.values)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_flat_intervals.py::test_flat_intervals_end_to_end_matches_ragged -v`
Expected: FAIL — flat mode still returns a `RaggedIntervals` (`type(fi).__name__ == "RaggedIntervals"`), so the type assertion fails.

- [ ] **Step 3: Add the flat gather helper to `_tracks.py`**

In `python/genvarloader/_dataset/_tracks.py`, update the `.._ragged` import (line 18) to include `FlatIntervals`:

```python
from .._ragged import INTERVAL_DTYPE, FlatIntervals, RaggedIntervals, RaggedTracks
```

Add this module-level function after the `Tracks` class (end of file, after `write_transformed_track`):

```python
def build_flat_intervals(
    active_tracks: dict[str, TrackType],
    intervals: dict[str, RaggedIntervals],
    r_idx: NDArray[np.integer],
    s_idx: NDArray[np.integer],
    n_samples: int,
) -> FlatIntervals:
    """Pure-numpy gather of per-(region, sample, track) intervals into a
    :class:`FlatIntervals` of shape ``(batch, n_tracks, ~itvs)`` in C-order
    (batch outer, track inner) — matching the awkward concat order of
    :meth:`Tracks._call_intervals`.
    """
    B = len(r_idx)
    T = len(active_tracks)

    # Pass 1: gather each track's B groups in batch order (t, b layout).
    tb_starts: list[NDArray] = []
    tb_ends: list[NDArray] = []
    tb_values: list[NDArray] = []
    lengths_tb = np.empty((T, B), np.int64)
    for t, (name, tracktype) in enumerate(active_tracks.items()):
        itv = intervals[name]
        if tracktype is TrackType.SAMPLE:
            g = r_idx * n_samples + s_idx
        else:
            g = r_idx
        off = np.asarray(itv.starts.offsets)
        lo = off[g]
        lens = (off[g + 1] - lo).astype(np.int64)
        lengths_tb[t] = lens
        pt_off = lengths_to_offsets(lens)
        total = int(pt_off[-1])
        src = np.repeat(lo - pt_off[:-1], lens) + np.arange(total, dtype=np.int64)
        tb_starts.append(np.asarray(itv.starts.data)[src])
        tb_ends.append(np.asarray(itv.ends.data)[src])
        tb_values.append(np.asarray(itv.values.data)[src])

    data_starts = (
        np.concatenate(tb_starts) if tb_starts else np.empty(0, np.int32)
    )
    data_ends = np.concatenate(tb_ends) if tb_ends else np.empty(0, np.int32)
    data_values = (
        np.concatenate(tb_values) if tb_values else np.empty(0, np.float32)
    )
    offsets_tb = lengths_to_offsets(lengths_tb.ravel())  # (T*B + 1)

    # Pass 2: reorder groups (t, b) -> (b, t). For output group (b, t) the
    # source group in (t, b) layout is t*B + b.
    perm = (np.arange(T)[None, :] * B + np.arange(B)[:, None]).ravel()  # (B*T,)
    final_lengths = lengths_tb.ravel()[perm]
    final_offsets = lengths_to_offsets(final_lengths)
    total = int(final_offsets[-1])
    src = (
        np.repeat(offsets_tb[perm] - final_offsets[:-1], final_lengths)
        + np.arange(total, dtype=np.int64)
    )

    shape = (B, T, None)
    return FlatIntervals(
        starts=_Flat.from_offsets(data_starts[src], shape, final_offsets),
        ends=_Flat.from_offsets(data_ends[src], shape, final_offsets),
        values=_Flat.from_offsets(data_values[src], shape, final_offsets),
    )
```

- [ ] **Step 4: Thread `flat` through `Tracks.__call__` and `_call_intervals`**

In `python/genvarloader/_dataset/_tracks.py`, in `Tracks.__call__` (line 561), replace the dispatch tail (lines 577-583) with:

```python
        if issubclass(self.kind, RaggedTracks):
            out = self._call_float32(
                idx, r_idx, regions, output_length, splice_plan=splice_plan
            )
        else:
            out = self._call_intervals(idx, flat=flat)
        return cast(_T, out)
```

Change the `_call_intervals` signature (line 691) and add the flat branch at its top:

```python
    def _call_intervals(
        self, idx: NDArray[np.integer], flat: bool = False
    ) -> RaggedIntervals | FlatIntervals:
        r_idx, s_idx = np.unravel_index(idx, (self.n_regions, self.n_samples))

        if flat:
            return build_flat_intervals(
                self.active_tracks, self.intervals, r_idx, s_idx, self.n_samples
            )

        # out = (batch tracks ~itvs)
        out_starts = []
```

(The rest of the awkward path below is unchanged. `build_flat_intervals` is defined later in the module but called at runtime, so the forward reference is fine.)

- [ ] **Step 5: Pass `FlatIntervals` through the getitem flat boundary**

In `python/genvarloader/_dataset/_query.py`:

Update the `.._ragged` import (lines 21-25) to include `FlatIntervals`:

```python
from .._ragged import (
    FlatIntervals,
    RaggedAnnotatedHaps,
    RaggedIntervals,
    _COMP,
)
```

In `_reshape_outer` (line 141), add `FlatIntervals` to the flat-type tuple so it takes the outer-dims-only reshape branch:

```python
    if isinstance(
        o, (_Flat, _FlatAnnotatedHaps, _FlatVariants, _FlatVariantWindows, FlatIntervals)
    ):
```

In `reverse_complement_ragged`, add a no-op branch for `FlatIntervals` (intervals are not reverse-complemented, mirroring `RaggedIntervals`). Insert before the `if isinstance(rag, _Flat):` check (line 390):

```python
    if isinstance(rag, FlatIntervals):
        return rag
```

(No need to touch the `@overload` signatures — they are type-check-only; the runtime branch is what matters. Optionally add an overload for symmetry, but it is not required for correctness.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_flat_intervals.py -v`
Expected: PASS (4 passed).

- [ ] **Step 7: Run the broader track/flat suites for regressions**

Run: `pixi run -e dev pytest tests/dataset/test_flat_flanks.py tests/dataset -k "track or interval or flat" -q`
Expected: PASS (no regressions).

- [ ] **Step 8: Lint, format, commit**

```bash
pixi run -e dev ruff check python/ && pixi run -e dev ruff format python/
rtk git add python/genvarloader/_dataset/_tracks.py python/genvarloader/_dataset/_query.py tests/dataset/test_flat_intervals.py
rtk git commit -m "feat: flat interval reconstruction via FlatIntervals

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: generalize `RefTracks` → `SeqsTracks` (no behavior change)

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (rename `RefTracks` → `SeqsTracks`; widen `seqs` to `Ref | Haps`; thread `flat`; update `__all__` and factory `Ref` case)
- Modify: `python/genvarloader/_dataset/_impl.py` (import + `_recon` type union)
- Modify: `python/genvarloader/_dataset/_query.py` (`build_recon_splice_plan` isinstance)
- Test: `tests/dataset/test_seqs_tracks.py` (create)

**Interfaces:**
- Consumes: `Ref`, `Haps`, `Tracks` reconstructors; `_build_reconstructor`.
- Produces: `SeqsTracks(seqs: Ref | Haps, tracks: Tracks)` reconstructor returning `(seqs_out, tracks_out)`, calling `seqs(...)` and `tracks(...)` independently and passing `flat` to both; raises `NotImplementedError` on splice. `RefTracks` no longer exists.

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_seqs_tracks.py`:

```python
import genvarloader as gvl
from genvarloader._dataset._reconstruct import SeqsTracks


def test_reference_plus_tracks_uses_seqstracks():
    ds = gvl.get_dummy_dataset()
    rt = ds.with_seqs("reference").with_tracks(["read-depth"])
    assert type(rt._recon) is SeqsTracks
    seqs, tracks = rt[[0, 1], [0, 1]]
    assert seqs.shape[0] == 2
    assert tracks.shape[0] == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_seqs_tracks.py -v`
Expected: FAIL — `ImportError: cannot import name 'SeqsTracks'`.

- [ ] **Step 3: Rename and widen in `_reconstruct.py`**

In `python/genvarloader/_dataset/_reconstruct.py`:

Replace the `RefTracks` class (lines 54-93) with:

```python
@dataclass(slots=True)
class SeqsTracks(Reconstructor[tuple[Any, _T]]):
    """Any seq reconstructor (`Ref` or `Haps` in any kind) paired with
    un-realigned `Tracks`. Seqs and tracks are computed independently; tracks
    stay in reference coordinates (no haplotype re-alignment)."""

    seqs: Ref | Haps
    tracks: Tracks[_T]

    def __call__(
        self,
        idx: NDArray[np.integer],
        r_idx: NDArray[np.integer],
        regions: NDArray[np.int32],
        output_length: Literal["ragged", "variable"] | int,
        jitter: int,
        rng: np.random.Generator,
        deterministic: bool,
        splice_plan: SplicePlan | None = None,
        flat: bool = False,
    ) -> tuple[Any, _T]:
        if splice_plan is not None:
            raise NotImplementedError(
                "Splicing of sequences + un-realigned tracks is not supported."
            )
        seqs = self.seqs(
            idx=idx,
            r_idx=r_idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
            flat=flat,
        )
        tracks = self.tracks(
            idx=idx,
            r_idx=r_idx,
            regions=regions,
            output_length=output_length,
            jitter=jitter,
            rng=rng,
            deterministic=deterministic,
            flat=flat,
        )
        return seqs, tracks
```

Add `Any` to the `typing` import at the top of the file (line 16):

```python
from typing import Any, Literal, cast
```

Update `__all__` (lines 40-51): replace `"RefTracks"` with `"SeqsTracks"`.

In `_build_reconstructor`, change the `Ref + Tracks` dispatch case (line 306-307) to:

```python
        case Ref() as s, Tracks() as t:
            return SeqsTracks(seqs=s, tracks=t)
```

(The `Haps + Tracks` case stays `HapsTracks` for now — Task 5 rewires it. Leave its `variant-windows` raise in place for this task.)

- [ ] **Step 4: Update `_impl.py` references**

In `python/genvarloader/_dataset/_impl.py`, find the import of `RefTracks` from `._reconstruct` and rename it to `SeqsTracks`. Then in the `_recon` type union (lines 856-869) replace the `RefTracks` member with `SeqsTracks`:

```python
        | Tracks
        | SeqsTracks
        | HapsTracks[RaggedSeqs, RaggedTracks]
```

Run to find the exact import line:

```bash
rtk grep -n "RefTracks" python/genvarloader/_dataset/_impl.py
```

Replace every `RefTracks` occurrence in that file with `SeqsTracks`.

- [ ] **Step 5: Update the splice dispatch in `_query.py`**

In `python/genvarloader/_dataset/_query.py`, `build_recon_splice_plan` (lines 304-313), change:

```python
    from ._reconstruct import HapsTracks, SeqsTracks

    if isinstance(recon, HapsTracks):
        raise NotImplementedError(
            "Splicing of haplotypes + tracks (shape (b, t, p, ~l)) is not supported."
        )
    if isinstance(recon, SeqsTracks):
        raise NotImplementedError(
            "Splicing of sequences + un-realigned tracks is not supported."
        )
```

- [ ] **Step 6: Run tests to verify they pass + regressions**

Run: `pixi run -e dev pytest tests/dataset/test_seqs_tracks.py -v`
Expected: PASS.

Run: `pixi run -e dev pytest tests/dataset -k "ref and track or reference" -q`
Expected: PASS (reference+tracks behavior unchanged).

- [ ] **Step 7: Lint, format, typecheck, commit**

```bash
pixi run -e dev ruff check python/ && pixi run -e dev ruff format python/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_query.py tests/dataset/test_seqs_tracks.py
rtk git commit -m "refactor: generalize RefTracks into SeqsTracks (seqs + un-realigned tracks)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `realign_tracks` setting + dispatch rules + guards

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`_build_reconstructor` gains `realign_tracks` param + new `Haps + Tracks` rules; import `RaggedIntervals`)
- Modify: `python/genvarloader/_dataset/_impl.py` (new `realign_tracks` field; `with_settings` param + rebuild trigger; pass `realign_tracks` at all 4 factory call sites; `with_insertion_fill` guard; forward `realign_tracks` + `output_format` in `with_len` constructors)
- Test: `tests/dataset/test_realign_tracks.py` (create)

**Interfaces:**
- Consumes: `SeqsTracks`, `HapsTracks` (Task 3); `_build_reconstructor`.
- Produces: `Dataset.realign_tracks: bool = True`; `with_settings(realign_tracks: bool | None = None)`; `_build_reconstructor(seqs, tracks, seqs_kind, realign_tracks)` (new 4th positional param). New raises: variant-windows + tracks with `realign_tracks=True`; any Haps-backed seq + `kind="intervals"` with `realign_tracks=True`; `with_insertion_fill` with `realign_tracks=False`.

- [ ] **Step 1: Write the failing tests**

Create `tests/dataset/test_realign_tracks.py`:

```python
import awkward as ak
import numpy as np
import pytest

import genvarloader as gvl
from genvarloader._dataset._reconstruct import HapsTracks, SeqsTracks


def test_default_haps_tracks_realigns():
    ds = gvl.get_dummy_dataset()  # default: haplotypes + tracks
    assert type(ds._recon) is HapsTracks
    assert ds.realign_tracks is True


def test_realign_false_haps_tracks_uses_seqstracks_and_is_reference_coord():
    ds = gvl.get_dummy_dataset()
    asis = ds.with_seqs("haplotypes").with_tracks(["read-depth"]).with_settings(
        realign_tracks=False
    )
    assert type(asis._recon) is SeqsTracks

    # As-is track must equal the solo (reference-coordinate) track values.
    solo = ds.with_seqs(None).with_tracks(["read-depth"])
    _, t = asis[[0], [0]]
    t_solo = solo[[0], [0]]
    assert ak.to_list(t) == ak.to_list(t_solo)


def test_intervals_plus_haplotypes_requires_realign_false():
    ds = gvl.get_dummy_dataset()  # default haplotypes + tracks, realign True
    with pytest.raises(ValueError, match="realign"):
        ds.with_tracks(["read-depth"], kind="intervals")


def test_intervals_plus_haplotypes_ok_when_realign_false():
    ds = gvl.get_dummy_dataset()
    out = (
        ds.with_settings(realign_tracks=False)
        .with_tracks(["read-depth"], kind="intervals")[[0], [0]]
    )
    seqs, itvs = out
    assert isinstance(itvs, gvl.RaggedIntervals)


def test_insertion_fill_rejected_when_realign_false():
    ds = gvl.get_dummy_dataset().with_settings(realign_tracks=False)
    with pytest.raises(ValueError, match="realign"):
        ds.with_insertion_fill(gvl.Constant(value=0.0))


def test_realign_tracks_survives_with_len():
    ds = gvl.get_dummy_dataset().with_settings(realign_tracks=False)
    ds2 = ds.with_len("variable")
    assert ds2.realign_tracks is False
    assert type(ds2._recon) is SeqsTracks
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_realign_tracks.py -v`
Expected: FAIL — `Dataset` has no `realign_tracks` attribute; `with_settings` has no such param.

- [ ] **Step 3: Add the `realign_tracks` field**

In `python/genvarloader/_dataset/_impl.py`, after the `output_format` field (line 871-874), add:

```python
    realign_tracks: bool = True
    """Whether to re-align track *values* to haplotype coordinates when both
    haplotypes and float tracks (``kind="tracks"``) are active. ``True`` (default)
    uses the indel-aware realignment kernel; ``False`` returns reference-coordinate
    (as-is) tracks. Only affects ``Haps`` + float tracks; a no-op otherwise.
    Required ``False`` for ``variant-windows`` + tracks and for ``kind="intervals"``
    with any variant-aware seq mode."""
```

- [ ] **Step 4: Extend `_build_reconstructor`**

In `python/genvarloader/_dataset/_reconstruct.py`:

Add the `RaggedIntervals` import — extend the `.._ragged` import (line 25):

```python
from .._ragged import RaggedAnnotatedHaps, RaggedIntervals, RaggedSeqs, RaggedTracks
```

Change the signature (lines 242-249) to add the `realign_tracks` parameter:

```python
def _build_reconstructor(
    seqs: Haps | Ref | None,
    tracks: Tracks | None,
    seqs_kind: Literal[
        "haplotypes", "reference", "annotated", "variants", "variant-windows"
    ]
    | None,
    realign_tracks: bool = True,
) -> Reconstructor:
```

Replace the `Haps + Tracks` dispatch case (lines 308-314) with:

```python
        case Haps() as s, Tracks() as t:
            is_intervals = issubclass(t.kind, RaggedIntervals)
            if is_intervals:
                if realign_tracks:
                    raise ValueError(
                        "Track intervals cannot be re-aligned to haplotype"
                        " coordinates. Set with_settings(realign_tracks=False) to"
                        " return reference-coordinate (as-is) intervals."
                    )
                return SeqsTracks(seqs=s, tracks=t)
            # Float tracks (RaggedTracks).
            if seqs_kind == "variant-windows":
                if realign_tracks:
                    raise ValueError(
                        "with_seqs('variant-windows') with tracks requires"
                        " with_settings(realign_tracks=False) (windows are"
                        " reference-oriented; re-alignment is not supported)."
                    )
                return SeqsTracks(seqs=s, tracks=t)
            if realign_tracks:
                return HapsTracks(haps=s, tracks=t)
            return SeqsTracks(seqs=s, tracks=t)
```

- [ ] **Step 5: Pass `realign_tracks` at all factory call sites in `_impl.py`**

Update each `_build_reconstructor(...)` call to pass `self.realign_tracks`:

- `with_seqs` (line 717):

```python
        new_recon = _build_reconstructor(
            new_seqs, self._tracks, kind, self.realign_tracks
        )
```

- `with_tracks` (line 769):

```python
        new_recon = _build_reconstructor(
            self._seqs, new_tracks, self._seqs_kind, self.realign_tracks
        )
```

- `with_insertion_fill` (line 802) — but first add the guard (Step 6); the call becomes:

```python
        new_recon = _build_reconstructor(
            self._seqs, new_tracks, self._seqs_kind, self.realign_tracks
        )
```

- [ ] **Step 6: Add the `with_insertion_fill` guard**

In `with_insertion_fill` (lines 789-800), after the existing `if self._tracks is None:` check (line 789-790), add:

```python
        if not self.realign_tracks:
            raise ValueError(
                "with_insertion_fill has no effect when realign_tracks=False"
                " (insertion fill only applies during track re-alignment). Set"
                " with_settings(realign_tracks=True) first, or drop the call."
            )
```

- [ ] **Step 7: Wire `realign_tracks` into `with_settings`**

In `with_settings` (signature around lines 201-217), add the parameter (place after `unphased_union`):

```python
        unphased_union: bool | None = None,
        realign_tracks: bool | None = None,
    ) -> Self:
```

Add a docstring entry for it near the others (after the `unphased_union` doc block, before the closing `"""`):

```python
        realign_tracks
            Whether to re-align track values to haplotype coordinates when both
            haplotypes and float tracks are active. Default ``True``. Set ``False``
            for reference-coordinate (as-is) tracks; required ``False`` for
            ``variant-windows`` + tracks and for ``kind="intervals"`` with any
            variant-aware seq mode.
```

In the body, after the `unphased_union` block (lines 423-429), add:

```python
        if realign_tracks is not None:
            to_evolve["realign_tracks"] = realign_tracks
```

Update the recon-rebuild guard (lines 432-437) to also rebuild when `realign_tracks` changes and pass it through:

```python
        # If any source state changed, rebuild _recon via the factory.
        if (
            "_seqs" in to_evolve
            or "_tracks" in to_evolve
            or "realign_tracks" in to_evolve
        ):
            new_seqs = to_evolve.get("_seqs", self._seqs)
            new_tracks = to_evolve.get("_tracks", self._tracks)
            new_realign = to_evolve.get("realign_tracks", self.realign_tracks)
            to_evolve["_recon"] = _build_reconstructor(
                new_seqs, new_tracks, self._seqs_kind, new_realign
            )
```

- [ ] **Step 8: Forward `realign_tracks` (and `output_format`) through `with_len`**

The `with_len` explicit constructors (the `ArrayDataset(...)` at lines 544-563 and `RaggedDataset(...)` at lines 565-584) currently drop both `output_format` and `realign_tracks`, resetting them to defaults. Add both kwargs to **each** constructor call (insert alongside the other fields, e.g. right after `_rng=self._rng,`):

```python
                _rng=self._rng,
                output_format=self.output_format,
                realign_tracks=self.realign_tracks,
            )
```

(Apply to both the `ArrayDataset(...)` and `RaggedDataset(...)` constructions.)

- [ ] **Step 9: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_realign_tracks.py -v`
Expected: PASS (6 passed).

- [ ] **Step 10: Regression sweep**

Run: `pixi run -e dev pytest tests/dataset -q`
Expected: PASS (no regressions; default `realign_tracks=True` preserves prior behavior).

- [ ] **Step 11: Lint, format, typecheck, commit**

```bash
pixi run -e dev ruff check python/ && pixi run -e dev ruff format python/ && pixi run -e dev typecheck
rtk git add python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_impl.py tests/dataset/test_realign_tracks.py
rtk git commit -m "feat: realign_tracks setting decouples track re-alignment from seq mode

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 5: tracks alongside `variant-windows` (end-to-end)

**Files:**
- Test: `tests/dataset/test_realign_tracks.py` (append)

**Interfaces:**
- Consumes: the `variant-windows` seq mode, `SeqsTracks`, `realign_tracks=False`, `FlatIntervals`, `_Flat` (FlatRagged), `VarWindowOpt`.
- Produces: no new code — this task verifies the dispatch wired in Task 4 produces `(FlatVariantWindows, FlatRagged | FlatIntervals)` and raises when `realign_tracks=True`.

- [ ] **Step 1: Write the tests (append)**

Append to `tests/dataset/test_realign_tracks.py`:

```python
def _vw_opt():
    return gvl.VarWindowOpt(flank_length=4, token_alphabet=b"ACGT", unknown_token=4)


def test_variant_windows_plus_float_tracks():
    ds = gvl.get_dummy_dataset()
    vw = (
        ds.with_settings(realign_tracks=False)
        .with_output_format("flat")
        .with_seqs("variant-windows", _vw_opt())
        .with_tracks(["read-depth"])  # default kind="tracks"
    )
    out = vw[[0, 1], [0, 1]]
    assert isinstance(out, tuple) and len(out) == 2
    windows, tracks = out
    assert type(windows).__name__ == "_FlatVariantWindows"
    assert type(tracks).__name__ == "_Flat"  # FlatRagged float track


def test_variant_windows_plus_intervals():
    ds = gvl.get_dummy_dataset()
    vw = (
        ds.with_settings(realign_tracks=False)
        .with_output_format("flat")
        .with_seqs("variant-windows", _vw_opt())
        .with_tracks(["read-depth"], kind="intervals")
    )
    windows, itvs = vw[[0, 1], [0, 1]]
    assert type(windows).__name__ == "_FlatVariantWindows"
    assert type(itvs).__name__ == "FlatIntervals"


def test_variant_windows_plus_tracks_requires_realign_false():
    ds = gvl.get_dummy_dataset()  # tracks active by default, realign True
    with pytest.raises(ValueError, match="realign"):
        ds.with_output_format("flat").with_seqs("variant-windows", _vw_opt())
```

- [ ] **Step 2: Run tests**

Run: `pixi run -e dev pytest tests/dataset/test_realign_tracks.py -k variant_windows -v`
Expected: PASS (3 passed). If `test_variant_windows_plus_*` fail because the dummy dataset produces no variants for some (region, sample), inspect with a quick `print(windows.shape)`; the dummy dataset has exactly one variant per (region, sample) so windows should be non-empty.

- [ ] **Step 3: Commit**

```bash
rtk git add tests/dataset/test_realign_tracks.py
rtk git commit -m "test: tracks (float + intervals) alongside variant-windows

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 6: flat-track coverage + docs (SKILL, dataset.md, changelog)

**Files:**
- Test: `tests/dataset/test_flat_intervals.py` (append flat-track coverage) — or a small new section
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `docs/source/dataset.md`
- Modify: `docs/source/changelog.md`

**Interfaces:**
- Consumes: everything above. No new runtime code.

- [ ] **Step 1: Write flat-track coverage tests (append)**

Append to `tests/dataset/test_flat_intervals.py`:

```python
def test_flat_float_tracks_only_returns_flatragged():
    ds = gvl.get_dummy_dataset()
    flat = ds.with_seqs(None).with_tracks(["read-depth"]).with_output_format("flat")
    out = flat[[0, 1], [0, 1]]
    assert type(out).__name__ == "_Flat"  # FlatRagged
    # round-trips to the ragged track values
    rag = ds.with_seqs(None).with_tracks(["read-depth"])[[0, 1], [0, 1]]
    assert ak.to_list(out.to_ragged()) == ak.to_list(rag)


def test_flat_haps_plus_tracks_returns_flat_pair():
    ds = gvl.get_dummy_dataset()
    flat = (
        ds.with_seqs("haplotypes").with_tracks(["read-depth"]).with_output_format("flat")
    )
    seqs, tracks = flat[[0, 1], [0, 1]]
    assert type(seqs).__name__ == "_Flat"
    assert type(tracks).__name__ == "_Flat"
```

- [ ] **Step 2: Run tests**

Run: `pixi run -e dev pytest tests/dataset/test_flat_intervals.py -v`
Expected: PASS.

- [ ] **Step 3: Update `skills/genvarloader/SKILL.md`**

In the `with_tracks` description (around line 184-186), append a sentence documenting re-alignment control:

```markdown
- `kind`: `"tracks"` (re-aligned numeric values) or `"intervals"` (raw interval representation).

Track **re-alignment** to haplotype coordinates is controlled by `with_settings(realign_tracks=True)` (default). Set `realign_tracks=False` for reference-coordinate ("as-is") tracks. `realign_tracks=False` is **required** for `kind="intervals"` with any variant-aware seq mode, and for `variant-windows` + tracks. `with_insertion_fill` requires `realign_tracks=True`.
```

In the `"variant-windows"` row of the `with_seqs` table (line 180), update the "Use when" / note to mention tracks are now allowed with `realign_tracks=False`. Add to the paragraph under the table (after the `variant-windows requires...` sentence, line ~182):

```markdown
`variant-windows` may be combined with tracks when `with_settings(realign_tracks=False)` is set; the returned tracks/intervals are reference-coordinate (as-is). Float tracks come back as `FlatRagged`, interval tracks as `FlatIntervals`.
```

In the flat-output format table (lines 198-201), update the `"flat"` row to list `FlatIntervals`:

```markdown
| `"flat"`   | Pure-numpy `FlatRagged` / `FlatVariants` / `FlatAnnotatedHaps` / `FlatIntervals` | No       |
```

In the flat-output prose (line 203), note that `kind="intervals"` returns `FlatIntervals` in flat mode (with `.to_ragged()` → `RaggedIntervals`), and that float tracks already return `FlatRagged` in flat mode.

In "Other public surface" (around line 343), add a one-liner:

```markdown
- `gvl.FlatIntervals` — flat-buffer interval container returned by `with_tracks(kind="intervals")` + `with_output_format("flat")`. Fields `.starts`/`.ends`/`.values` are `FlatRagged`; `.to_ragged()` → `RaggedIntervals`; `.reshape(...)`, `.squeeze(...)`, `.shape`. Source: `python/genvarloader/_ragged.py`.
```

In "Common gotchas", add:

```markdown
- `kind="intervals"` cannot be re-aligned: combining it with a variant-aware seq mode (`haplotypes`/`annotated`/`variants`/`variant-windows`) raises unless `with_settings(realign_tracks=False)`. (Breaking change: `haplotypes`+`intervals` previously returned un-realigned intervals silently under the default.)
- `with_insertion_fill` raises when `realign_tracks=False`.
```

- [ ] **Step 4: Update `docs/source/dataset.md`**

Add a short subsection documenting `realign_tracks` (toggle, default, the two required-`False` cases, and that float tracks/intervals can ride alongside `variants`/`variant-windows` as-is). Mirror the wording from the SKILL update. Place it near the existing tracks / `with_tracks` discussion.

- [ ] **Step 5: Update `docs/source/changelog.md`**

Add entries under the appropriate (unreleased) version heading:

```markdown
### Feat

- **tracks**: `with_settings(realign_tracks=...)` toggles haplotype-coordinate track re-alignment (default `True`); `realign_tracks=False` returns reference-coordinate tracks/intervals and enables tracks alongside `variants` / `variant-windows`.
- **flat**: `FlatIntervals` flat-buffer interval output for `with_tracks(kind="intervals")` + `with_output_format("flat")`; flat float tracks return `FlatRagged`.

### Breaking

- `kind="intervals"` with a variant-aware seq mode now requires `realign_tracks=False` (previously `haplotypes`+`intervals` silently returned un-realigned intervals). `with_insertion_fill` now raises when `realign_tracks=False`.
```

- [ ] **Step 6: Full test + lint + format + typecheck**

```bash
pixi run -e dev pytest tests/dataset -q
pixi run -e dev ruff check python/ && pixi run -e dev ruff format python/ && pixi run -e dev typecheck
```

Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
rtk git add tests/dataset/test_flat_intervals.py skills/genvarloader/SKILL.md docs/source/dataset.md docs/source/changelog.md
rtk git commit -m "docs: document FlatIntervals + realign_tracks; flat-track test coverage

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review Notes (for the executor)

- **Spec coverage:** `FlatIntervals` (Task 1-2), flat-track test/doc (Task 6), `realign_tracks` setting (Task 4), `SeqsTracks` generalization (Task 3), variant-windows+tracks (Task 4-5), intervals require-realign-false + breaking change (Task 4 + docs Task 6), insertion-fill guard (Task 4), `with_settings`-only / no `Dataset.open` (Task 4). All spec sections map to a task.
- **Type/name consistency:** `FlatIntervals` (fields `starts`/`ends`/`values`), `SeqsTracks` (field `seqs`/`tracks`), `build_flat_intervals(active_tracks, intervals, r_idx, s_idx, n_samples)`, `_build_reconstructor(seqs, tracks, seqs_kind, realign_tracks)`, `realign_tracks` field/param — names used identically across tasks.
- **Watch-outs:** `_Flat` (internal class) is the public `FlatRagged`; tests assert `type(...).__name__ == "_Flat"` deliberately. The `with_len` constructors historically also dropped `output_format` — Task 4 Step 8 fixes both for correctness of flat + fixed-length usage. Verify no caller invokes `recon.to_kind` on a compound reconstructor (the refactor routes everything through `_build_reconstructor`); if one is found, route it through the factory instead.
```
