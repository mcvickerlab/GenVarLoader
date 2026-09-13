# Streaming Mixed Variants + Tracks for SVAR2 / VCF / PGEN — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
>
> **Parallelism:** Task 1 is a SERIAL PRELUDE and must land first. Tasks 2–4 (Track A) and Tasks 5–8 (Track B) are then INDEPENDENT and MUST be run concurrently via superpowers:dispatching-parallel-agents. Track A and Track B touch disjoint files and disjoint regions of `_streaming.py`; running them serially wastes ~a day of wall-clock.

**Goal:** Lift `StreamingDataset`'s `NotImplementedError` on `tracks=` + a non-SVAR1 variant source, so mixed `(haplotypes, tracks)` streaming is byte-identical to `gvl.write()` + `Dataset[r, s]` for all four backends (SVAR1, SVAR2, VCF, PGEN).

**Architecture:** Today the mixed path is hard-wired to `_Svar1Backend`: an `isinstance` guard at `_streaming.py:1254`, a window-level re-align setup block at `_streaming.py:1596-1620` that reads `_Svar1Backend`-only attributes (`geno_v_idxs`/`_v_starts`/`_ilens`) and calls `_Svar1Backend.read_window`, and a per-batch call site at `_streaming.py:1853-1867` that reaches back into `backend.*` for those same three arrays. Task 1 replaces all three with one seam: a `_MixedRealign` protocol whose implementations bundle *one window's re-align state together with the kernel that consumes it*, produced by a new per-backend `mixed_realign_window()` method and gated by a new `supports_mixed_tracks` class attribute. SVAR1's implementation is the existing code moved verbatim. Track A then adds an SVAR2 implementation (pure Python — both SVAR2 re-align kernels already exist and are already exported to Python); Track B adds one implementation shared by VCF and PGEN, backed by a small new Rust pymethod that hands Python the decoded window's variant table + genotype CSR.

**Tech Stack:** Python 3.10+ / NumPy / `seqpro.rag.Ragged`; Rust + PyO3 (maturin); pixi (`-e dev`); pytest.

**Spec:** `docs/superpowers/specs/2026-09-08-streaming-intervals-design.md` (the issue #279 streaming-intervals design). Section 3.2 fixes the parity preconditions this plan inherits; that spec explicitly deferred non-SVAR1 backends, which is issue #375 — what this plan implements.

## Global Constraints

- **Target branch is `streaming`, NOT `main`.** Every branch here is cut from `origin/streaming` and every PR targets `streaming`. (`CLAUDE.md`, "Streaming dataset work".)
- **Every issue and PR gets a `streaming:` title prefix and is added to the StreamingDataset GitHub Project.**
- **The correctness oracle is byte-identical parity with the written path**: `gvl.write(...)` then `Dataset.open(...)[r, s]`, compared cell-by-cell against `StreamingDataset(...).to_iter(...)`. Nothing else counts as "it works".
- **Mixed output is a 2-tuple `(haplotypes, tracks)`** in that order, matching `HapsTracks.__call__` (`_reconstruct.py:121-144`). Established by issue #380 / PR #388; both halves must be compared in every new parity test.
- **Parity fixtures must be written with `extend_to_length=False, max_jitter=None`** (spec Section 3.2). Anything else is out of parity scope in v1.
- **Rebuild before testing Rust changes:** `pixi run -e dev maturin develop --release`. `pixi run -e dev pytest` does NOT rebuild; without this you test the stale `.so`.
- **No squash merges.** Land with a real merge commit.
- **Conventional commits** (commitizen). Ensure `prek` git hooks are installed before committing.
- Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg
  ```
- PR descriptions end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.

---

## File Structure

| File | Task(s) | Responsibility |
| --- | --- | --- |
| `python/genvarloader/_dataset/_streaming.py` | 1, 2, 5, 6 | The seam (Task 1), the SVAR2 implementation + SVAR2 drive wiring (Task 2), the record-backend implementation + VCF/PGEN wiring (Tasks 5–6). **Task 1's edits are the only ones the other tasks share; after Task 1 lands, Tasks 2 and 5/6 edit disjoint regions** (Task 2: the `_Svar2Backend` class body at ~3359–3670 and the `"sync"` drive at ~2008–2054; Tasks 5/6: the `_VcfBackend`/`_PgenBackend` class bodies at ~3674+ and a shared module-level class). |
| `src/record_stream/engine.rs` | 5 | New `RecordStreamEngine.window_realign_inputs()` pymethod returning the decoded window's `(v_starts, ilens, geno_v_idxs, geno_offsets)`. |
| `tests/dataset/conftest.py` | 3, 7 | New mixed fixtures: `streaming_svar2_tracks_fixture` (Task 3), `streaming_record_tracks_fixture` (Task 7). |
| `tests/dataset/test_streaming_tracks_svar2.py` | 4 | SVAR2 mixed parity suite (new file). |
| `tests/dataset/test_streaming_tracks_record.py` | 8 | VCF + PGEN mixed parity suite (new file), parametrized over both backends. |
| `docs/source/dataset.md`, `skills/genvarloader/SKILL.md`, `docs/roadmaps/streaming-dataset.md` | 4, 8 | Drop the "SVAR1 only" caveat as each backend lands. |

---

## Task 1 (SERIAL PRELUDE): Extract the mixed re-align seam

**Why this is its own task and must land alone:** Track A and Track B would otherwise both rewrite `_streaming.py:1254`, `:1596-1620`, and `:1853-1867`. After this task those three sites are backend-agnostic and each track only *adds* a class + a method, in disjoint regions.

**No behavior change.** SVAR1 output must be byte-identical before and after. The existing suite is the gate.

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py:361-373` (`_RealignWindow`), `:375-390` (`_TrackWindow`), `:1254-1262` (guard), `:1596-1625` (window setup), `:1853-1867` (per-batch call), `_Svar1Backend` class body (add two members)
- Test: `tests/dataset/test_streaming_tracks.py` (existing, unchanged — it is the regression gate)

**Interfaces:**
- Consumes: nothing (first task).
- Produces, for Tasks 2 and 5:
  ```python
  class _MixedRealign(Protocol):
      """One window's haplotype-realign state PLUS the kernel that consumes it."""

      def realign_batch(
          self,
          lo: int,
          hi: int,
          itvs: "list[_ItvArrays]",
          names: list[str],
          insertion_fill: "dict[str, InsertionFill] | None",
          regions_batch: NDArray[np.int32],
          row_lengths: NDArray[np.int64],
          out_lengths: NDArray[np.int64],
          base_seed: int,
      ) -> "RaggedTracks": ...
  ```
  Backend contract (a backend that sets `supports_mixed_tracks = True` MUST define `mixed_realign_window`):
  ```python
  supports_mixed_tracks: ClassVar[bool]

  def mixed_realign_window(
      self,
      r_idx: NDArray[np.intp],
      s_idx: NDArray[np.intp],
      t_starts: NDArray[np.int32],   # (n_regions,) per-region track query start
      t_ends: NDArray[np.int32],     # (n_regions,) per-region track query end
      row_starts: NDArray[np.int32], # (n_rows,) per-(region, sample) start
      row_ends: NDArray[np.int32],   # (n_rows,) per-(region, sample) end
  ) -> tuple["_MixedRealign", NDArray[np.int32]]:
      """Per-window realign state + deletion-extended per-region track-query ends."""
  ```
  `_TrackWindow.realign` changes type from `_RealignWindow | None` to `_MixedRealign | None`.

- [ ] **Step 1: Confirm the SVAR1 mixed suite is green before touching anything**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q`
Expected: PASS (52 tests as of PR #388). Record the count; it must not change.

- [ ] **Step 2: Widen `_RealignWindow` and give it the batch kernel**

In `python/genvarloader/_dataset/_streaming.py`, replace the `_RealignWindow` class (currently at ~line 361) with:

```python
class _MixedRealign(Protocol):
    """One window's haplotype-realign state PLUS the kernel that consumes it.

    Mixed variants+tracks needs two things per window: state computed once
    (per-row genotype views, deletion diffs) and a per-batch kernel that turns
    that state plus the window's intervals into `RaggedTracks`. Different
    variant sources need BOTH to differ -- SVAR1 and the record backends share
    the fused `intervals_and_realign_track_fused` kernel over a CSR
    (`_RealignWindow`), while SVAR2 has no fused kernel and must split into
    `intervals_to_tracks` + `shift_and_realign_tracks_from_svar2_readbound`
    (`_Svar2Realign`). Bundling state with its kernel is what keeps the drive
    loop free of `isinstance` dispatch: the drive holds one optional and calls
    one method on it.
    """

    def realign_batch(
        self,
        lo: int,
        hi: int,
        itvs: "list[_ItvArrays]",
        names: list[str],
        insertion_fill: "dict[str, InsertionFill] | None",
        regions_batch: NDArray[np.int32],
        row_lengths: NDArray[np.int64],
        out_lengths: NDArray[np.int64],
        base_seed: int,
    ) -> "RaggedTracks":
        """Re-align window rows ``[lo, hi)``'s tracks to haplotype coordinates.

        Args:
            lo: First window row of this batch.
            hi: One past the last window row of this batch.
            itvs: One `_ItvArrays` per track, covering the WHOLE window's rows
                (already FFI-coerced once per window by `_coerce_window_itvs`).
            names: Track names in `itvs` order.
            insertion_fill: Per-track insertion-fill overrides, or `None`.
            regions_batch: `(hi-lo, 3)` int32 `(0, start, end)` rows.
            row_lengths: `(hi-lo,)` int64 reference-coordinate region lengths.
            out_lengths: `(hi-lo, ploidy)` int64 per-hap output length, taken
                from the batch's actual haplotype offsets.
            base_seed: `FlankSample` base seed.

        Returns:
            `RaggedTracks` of shape `(hi-lo, n_tracks, ploidy, None)`.
        """
        ...


class _RealignWindow(NamedTuple):
    """CSR-shaped realign state: SVAR1 and the record backends (VCF/PGEN).

    Exists as its own optional rather than separately-nullable fields on
    :class:`_TrackWindow` so that "re-alignment is on" and "the arrays
    re-alignment needs are present" cannot disagree: there is one thing to
    check, and checking it hands you everything already narrowed.

    The last three fields are the per-variant tables the fused kernel reads.
    They are dataset-GLOBAL for `_Svar1Backend` (memmaps over the whole store)
    and window-LOCAL for the record backends (one decoded window's table), but
    the kernel only ever indexes them through `geno_v_idxs`, so the two cases
    are interchangeable here -- which is exactly why they live on the state
    rather than being read off the backend at the call site.
    """

    diffs: "NDArray[np.int32]"
    geno_offsets: "NDArray[np.int64]"
    geno_offset_idx: "NDArray[np.intp]"
    geno_v_idxs: "NDArray"
    v_starts: "NDArray[np.int32]"
    ilens: "NDArray[np.int32]"

    def realign_batch(
        self,
        lo: int,
        hi: int,
        itvs: "list[_ItvArrays]",
        names: list[str],
        insertion_fill: "dict[str, InsertionFill] | None",
        regions_batch: NDArray[np.int32],
        row_lengths: NDArray[np.int64],
        out_lengths: NDArray[np.int64],
        base_seed: int,
    ) -> "RaggedTracks":
        """See :meth:`_MixedRealign.realign_batch`. Fused-kernel implementation."""
        return _realigned_tracks_from_intervals(
            itvs,
            names,
            insertion_fill,
            np.arange(lo, hi, dtype=np.int64),
            regions_batch,
            self.geno_offset_idx[lo:hi],
            self.geno_offsets,
            self.geno_v_idxs,
            self.v_starts,
            self.ilens,
            row_lengths,
            self.diffs[lo:hi],
            out_lengths,
            base_seed,
        )
```

Add `Protocol` to the `typing` import at the top of the file if it is not already there.

- [ ] **Step 3: Retype `_TrackWindow.realign`**

In `_TrackWindow` (~line 375), change:

```python
    realign: "_RealignWindow | None"
```

to:

```python
    realign: "_MixedRealign | None"
```

- [ ] **Step 4: Give `_Svar1Backend` the capability flag and the window hook**

Add to the `_Svar1Backend` class body (after `read_window`, ~line 3292):

```python
    #: Mixed variants+tracks is wired for this backend (issue #279).
    supports_mixed_tracks: ClassVar[bool] = True

    def mixed_realign_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        t_starts: NDArray[np.int32],
        t_ends: NDArray[np.int32],
        row_starts: NDArray[np.int32],
        row_ends: NDArray[np.int32],
    ) -> tuple["_MixedRealign", NDArray[np.int32]]:
        """One window's fused-kernel realign state + deletion-extended track ends.

        Moved verbatim out of the `to_iter` drive (issue #375): the drive used
        to reach into `_Svar1Backend`-only attributes directly, which is what
        made the mixed path SVAR1-only.

        `t_ends` is extended per REGION by that region's max deletion length
        across the window's samples, because `_regions` is the raw BED while
        the written path's intervals were extracted over the post-widening
        `gvl_bed`. The formula (`row_lengths - diffs.clip(max=0).min(1)`) is
        the read path's own, verbatim, from `_realigned_tracks_from_intervals`
        and the WRITTEN reader (`_reconstruct.py:191`, `:382`); taking the
        per-region max over the window's samples makes the query a superset of
        every row's buffer, so it can never under-read.

        It deliberately does NOT match the written WRITER's `chromEnd`
        extension (`_write.py:1084`), which stores less than its own reader
        asks for. That asymmetry can only leave the written path with zeros in
        a tail streaming fills with real values, and is unobservable today
        because ragged re-alignment never reads past reference index
        `region_len - 1` and `with_len(L)` is bounded by the minimum region
        length on both sides. An `extend_to_length` follow-up must revisit both.
        """
        from ._genotypes import get_diffs_sparse

        n_rows = len(row_starts)
        P = self.ploidy
        n_s = len(s_idx)
        o_starts, o_stops = self.read_window(r_idx, s_idx)
        geno_offsets_w = np.stack([o_starts, o_stops])
        geno_offset_idx_w = np.arange(n_rows * P, dtype=np.intp).reshape(n_rows, P)
        diffs_w = get_diffs_sparse(
            geno_offset_idx_w,
            self.geno_v_idxs,
            geno_offsets_w,
            self._ilens,
            q_starts=row_starts,
            q_ends=row_ends,
            v_starts=self._v_starts,
        )
        max_del_row = -diffs_w.clip(max=0).min(1)
        region_max_del = max_del_row.reshape(len(r_idx), n_s).max(1)
        t_ends_ext = np.ascontiguousarray(
            t_ends.astype(np.int64) + region_max_del, np.int32
        )
        return (
            _RealignWindow(
                diffs=diffs_w,
                geno_offsets=geno_offsets_w,
                geno_offset_idx=geno_offset_idx_w,
                geno_v_idxs=self.geno_v_idxs,
                v_starts=self._v_starts,
                ilens=self._ilens,
            ),
            t_ends_ext,
        )
```

Add `ClassVar` to the `typing` import if absent.

- [ ] **Step 5: Declare the three other backends as not-yet-supported**

Add exactly this line to the class body of each of `_Svar2Backend` (~line 3359), `_VcfBackend` (~line 3674), and `_PgenBackend` (~line 3871), immediately after the class docstring:

```python
    #: Mixed variants+tracks not wired for this backend yet (issue #375).
    supports_mixed_tracks: ClassVar[bool] = False
```

- [ ] **Step 6: Replace the `isinstance` guard with the capability check**

At `_streaming.py:1254`, replace:

```python
            if self._track_backend is not None and not isinstance(
                self._backend, _Svar1Backend
            ):
                raise NotImplementedError(
                    "StreamingDataset tracks= combined with a variant source is "
                    "only supported for the SVAR1 (.svar) backend; got "
                    f"{type(self._backend).__name__}. VCF/PGEN/SVAR2 + track "
                    "re-alignment is a later follow-up (issue #279)."
                )
```

with:

```python
            # Issue #375: mixed variants + tracks is a per-backend CAPABILITY,
            # not an isinstance test. A backend opts in by setting
            # `supports_mixed_tracks = True` and defining
            # `mixed_realign_window`; the two must be added together. Checked
            # BEFORE any engine/plan work so a doomed combination fails fast,
            # matching the SVAR2 jitter/out_len guard just below.
            if self._track_backend is not None and not self._backend.supports_mixed_tracks:
                raise NotImplementedError(
                    "StreamingDataset tracks= combined with a variant source is "
                    f"not supported for {type(self._backend).__name__} yet "
                    "(issue #375)."
                )
```

- [ ] **Step 7: Replace the window setup block with the hook call**

At `_streaming.py`, replace the whole `if self._realign_tracks: ... else: ...` block that currently spans the long comment through `t_ends_ext = t_ends` (~lines 1559–1625) with:

```python
                        if self._realign_tracks:
                            realign_w, t_ends_ext = backend.mixed_realign_window(
                                r_idx,
                                s_idx_w,
                                np.ascontiguousarray(t_starts, np.int32),
                                np.ascontiguousarray(t_ends, np.int32),
                                row_starts_w,
                                row_ends_w,
                            )
                        else:
                            # Un-realigned tracks stay in reference coordinates
                            # -- no deletion-extension read-ahead needed.
                            realign_w = None
                            t_ends_ext = t_ends
```

Delete the now-unused `from ._genotypes import get_diffs_sparse` import inside the `tb is not None` block if nothing else in that block uses it. Also delete the now-unused `P = backend.ploidy` local if nothing else below uses it (check first — `_batch_bounds`/`out_lengths` use `backend.ploidy` directly).

- [ ] **Step 8: Replace the per-batch call with the state's own kernel**

At `_streaming.py:1853-1867`, replace:

```python
                                    tracks = _realigned_tracks_from_intervals(
                                        track_w.itvs,
                                        track_w.names,
                                        self._insertion_fill,
                                        np.arange(lo, hi, dtype=np.int64),
                                        regions_batch,
                                        track_w.realign.geno_offset_idx[lo:hi],
                                        track_w.realign.geno_offsets,
                                        backend.geno_v_idxs,
                                        backend._v_starts,
                                        backend._ilens,
                                        track_w.row_lengths[lo:hi],
                                        track_w.realign.diffs[lo:hi],
                                        out_lengths,
                                        base_seed,
                                    )
```

with:

```python
                                    tracks = track_w.realign.realign_batch(
                                        lo,
                                        hi,
                                        track_w.itvs,
                                        track_w.names,
                                        self._insertion_fill,
                                        regions_batch,
                                        track_w.row_lengths[lo:hi],
                                        out_lengths,
                                        base_seed,
                                    )
```

- [ ] **Step 9: Run the SVAR1 mixed suite — it must be byte-identical**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks.py -q`
Expected: PASS, same test count as Step 1. Any failure means the move was not verbatim.

- [ ] **Step 10: Run the full streaming + unit tree**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: PASS. (`_RealignWindow` gained fields and `_TrackWindow` changed a type annotation; `tests/unit/` can reference either.)

- [ ] **Step 11: Lint and typecheck**

Run:
```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
```
Expected: clean. (If `pyrefly check` reports "matched by project-excludes" because the worktree lives under `.claude/`, re-run as `pyrefly check --use-ignore-files=false`.)

- [ ] **Step 12: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py
git commit -m "refactor(streaming): make mixed variants+tracks a per-backend seam

Replace the _Svar1Backend isinstance guard, the inline window realign
setup, and the per-batch backend attribute reads with one seam: a
_MixedRealign protocol bundling a window's realign state with the kernel
that consumes it, produced by a new per-backend mixed_realign_window()
and gated by supports_mixed_tracks. SVAR1's implementation is the moved
code verbatim; output is byte-identical.

Unblocks SVAR2 and VCF/PGEN mixed support to land in parallel.

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
```

---

# TRACK A — SVAR2 mixed (Tasks 2–4)

**Runs in parallel with Track B. Branch from `streaming` AFTER Task 1 has merged.**

Track A is pure Python. Both kernels the written SVAR2 mixed path uses already exist in Rust and are already exported (`intervals_to_tracks`, `shift_and_realign_tracks_from_svar2_readbound`, `hap_diffs_from_svar2_readbound` — see `src/lib.rs:33,78,99`), and `_Svar2Backend.read_window`/`_gather_rows` already produce exactly the 7-tuple those kernels want. No `maturin develop` needed.

## Task 2: `_Svar2Realign` + `_Svar2Backend.mixed_realign_window`

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` — add `_Svar2Realign` next to `_RealignWindow` (~line 440); add three members to the `_Svar2Backend` class body (~line 3670, after `_est_out_bytes`)
- Test: `tests/dataset/test_streaming_tracks_svar2.py` (created in Task 4)

**Interfaces:**
- Consumes (from Task 1): the `_MixedRealign` protocol and the `mixed_realign_window` backend contract, both quoted verbatim in Task 1's Interfaces block.
- Consumes (pre-existing, unchanged): `_Svar2Backend.read_window(r_idx, s_idx) -> dict[str, object]` with keys `contig_idx`, `region_bounds`, `orig_samples`, `vk_snp`, `vk_indel`, `dense_snp`, `dense_indel`; and `_Svar2Backend._gather_rows(r_idx, s_idx, window, lo, hi) -> tuple[...]` whose **first seven elements are exactly** `(region_starts, orig_samples, vk_snp, vk_indel, dense_snp, dense_indel, region_bounds)` — the `_GatherInputs` tuple `shift_and_realign_tracks_from_svar2_readbound` and `hap_diffs_from_svar2_readbound` take as their `gi[0..6]` positional arguments (compare `_svar2_haps.py:_gather_inputs`).
- Produces (for Task 3's drive wiring): `_Svar2Backend.supports_mixed_tracks == True` and `_Svar2Backend.mixed_realign_window(...)` returning `(_Svar2Realign, t_ends_ext)`.

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_streaming_tracks_svar2.py` with just this test for now:

```python
"""SVAR2 mixed variants + tracks parity (issue #375, Track A).

The written oracle is `HapsTracks._call_svar2` (`_reconstruct.py:309`), which
splits the interval->realign step into `intervals_to_tracks` +
`shift_and_realign_tracks_from_svar2_readbound` because SVAR2 has no fused
kernel. This suite pins the streaming SVAR2 mixed drive against
`Dataset[r, s]` cell by cell, BOTH halves of the `(haps, tracks)` pair
(issue #380).
"""

from __future__ import annotations

import numpy as np

import genvarloader as gvl


def test_svar2_backend_declares_mixed_support():
    from genvarloader._dataset._streaming import _Svar2Backend

    assert _Svar2Backend.supports_mixed_tracks is True
    assert hasattr(_Svar2Backend, "mixed_realign_window")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_svar2.py -q`
Expected: FAIL — `assert False is True` (Task 1 set the flag to `False`).

- [ ] **Step 3: Add `_Svar2Realign`**

In `python/genvarloader/_dataset/_streaming.py`, immediately after `_RealignWindow` (before `_TrackWindow`), add:

```python
class _Svar2Realign(NamedTuple):
    """Split-kernel realign state: the SVAR2 backend (issue #375, Track A).

    SVAR2 has no fused interval->realign kernel, so unlike
    :class:`_RealignWindow` this holds the read-bound gather inputs
    (`_Svar2Backend._gather_rows`' first seven elements, which are exactly
    `_svar2_haps._GatherInputs`) and runs the same TWO kernels the written
    path's `HapsTracks._call_svar2` runs (`_reconstruct.py:309`):
    `intervals_to_tracks` to materialize the reference-space window, then
    `shift_and_realign_tracks_from_svar2_readbound` to move it to haplotype
    coordinates.

    Single-contig by construction: `_Svar2Backend.read_window` raises on a
    multi-contig window, so the per-contig grouping (and the inverse row
    permutation) `Svar2Haps.realign_track_block` needs is unreachable here.
    That is why this state carries a plain `contig` string rather than a
    group list.
    """

    store: object
    contig: str
    ploidy: int
    #: `_gather_rows`' first seven elements for the WHOLE window's rows.
    gi: tuple
    #: `(n_rows, ploidy)` int32 per-hap reference-length diffs.
    diffs: "NDArray[np.int32]"
    #: `(n_rows, ploidy)` int32 all-zero shifts (streaming never sets
    #: `deterministic=False`; see `_realigned_tracks_from_intervals`).
    shifts: "NDArray[np.int32]"

    def realign_batch(
        self,
        lo: int,
        hi: int,
        itvs: "list[_ItvArrays]",
        names: list[str],
        insertion_fill: "dict[str, InsertionFill] | None",
        regions_batch: NDArray[np.int32],
        row_lengths: NDArray[np.int64],
        out_lengths: NDArray[np.int64],
        base_seed: int,
    ) -> "RaggedTracks":
        """See :meth:`_MixedRealign.realign_batch`. Split-kernel implementation.

        Mirrors `Svar2Haps.realign_track_block` (`_svar2_haps.py:702`) per
        track, then reorders the track-major buffer into `(b, t, p)` order the
        same way `_realigned_tracks_from_intervals` does -- see that function's
        docstring for why streaming reorders and the written path does not
        (issue #371).
        """
        from .._ragged import RaggedTracks
        from .._threads import should_parallelize
        from .._utils import lengths_to_offsets
        from ..genvarloader import (
            intervals_to_tracks,
            shift_and_realign_tracks_from_svar2_readbound,
        )
        from ._insertion_fill import Repeat5p
        from ._insertion_fill import lower as _lower_insertion_fills
        from ._svar2_haps import _ragged_arange_gather

        n_tracks = len(itvs)
        batch = hi - lo
        P = self.ploidy
        fill_map = insertion_fill or {}

        diffs_b = self.diffs[lo:hi]
        track_lengths = row_lengths - diffs_b.clip(max=0).min(1)
        track_ofsts = np.ascontiguousarray(
            lengths_to_offsets(track_lengths, np.int64), np.int64
        )
        out_ofsts_per_t = np.ascontiguousarray(
            lengths_to_offsets(out_lengths), np.int64
        )
        n_per_track = int(out_ofsts_per_t[-1])
        out = np.empty(n_tracks * n_per_track, np.float32)

        strat_list = [fill_map.get(name, Repeat5p()) for name in names]
        strat_ids, strat_params = _lower_insertion_fills(strat_list)

        # Slice the window's gather inputs down to this batch's rows. The
        # first six entries are per-row (or per-row*P for the vk_* pairs);
        # `region_bounds` is per-row too. Row order is C-order (region,
        # sample), identical to the drive's.
        region_starts, orig_samples, vk_snp, vk_indel, dense_snp, dense_indel, region_bounds = (
            self.gi[0][lo:hi],
            self.gi[1][lo:hi],
            self.gi[2][lo * P : hi * P],
            self.gi[3][lo * P : hi * P],
            self.gi[4][lo:hi],
            self.gi[5][lo:hi],
            self.gi[6][lo:hi],
        )
        shifts_b = np.ascontiguousarray(self.shifts[lo:hi], np.int32)
        offset_idxs = np.arange(lo, hi, dtype=np.int64)
        g_total = int(track_ofsts[-1])

        for t, (name, itv) in enumerate(zip(names, itvs)):
            tracks_buf = np.empty(g_total, np.float32)
            intervals_to_tracks(
                offset_idxs=offset_idxs,
                starts=np.ascontiguousarray(regions_batch[:, 1], np.int32),
                itv_starts=itv.starts,
                itv_ends=itv.ends,
                itv_values=itv.values,
                itv_offsets=itv.offsets,
                out=tracks_buf,
                out_offsets=track_ofsts,
            )
            block_data, _block_off = shift_and_realign_tracks_from_svar2_readbound(
                self.store,
                self.contig,
                np.ascontiguousarray(region_starts, np.uint32),
                np.ascontiguousarray(orig_samples, np.int64),
                np.ascontiguousarray(vk_snp, np.int64),
                np.ascontiguousarray(vk_indel, np.int64),
                np.ascontiguousarray(dense_snp, np.int64),
                np.ascontiguousarray(dense_indel, np.int64),
                np.ascontiguousarray(region_bounds, np.int32),
                shifts_b,
                tracks_buf,
                track_ofsts,
                np.ascontiguousarray(strat_params[t], np.float64),
                np.int64(strat_ids[t]),
                np.uint64(base_seed),
                # GLOBAL batch row per query. Single contig group here, so the
                # group's rows ARE the batch's rows, in order.
                np.arange(batch, dtype=np.int64),
                should_parallelize(g_total * 4),
            )
            out[t * n_per_track : (t + 1) * n_per_track] = np.asarray(
                block_data, np.float32
            )

        # `out` is TRACK-major (flat order `(t, row, hap)`); a
        # `(batch, n_tracks, ploidy, None)` Ragged wants `(row, t, hap)`.
        # Identical reorder to `_realigned_tracks_from_intervals`.
        n_bp = batch * P
        tm_offsets = lengths_to_offsets(np.tile(out_lengths.reshape(-1), n_tracks))
        perm = (
            np.arange(n_tracks * n_bp)
            .reshape(n_tracks, batch, P)
            .transpose(1, 0, 2)
            .reshape(-1)
            .astype(np.intp)
        )
        data, out_offsets = _ragged_arange_gather(out, tm_offsets, perm)
        return cast(
            "RaggedTracks",
            Ragged.from_offsets(data, (batch, n_tracks, P, None), out_offsets),  # type: ignore[bad-argument-type, no-matching-overload]
        )
```

- [ ] **Step 4: Add the three `_Svar2Backend` members**

In the `_Svar2Backend` class body, after `_est_out_bytes` (~line 3670), add:

```python
    #: Mixed variants+tracks is wired for this backend (issue #375, Track A).
    supports_mixed_tracks: ClassVar[bool] = True

    def mixed_realign_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        t_starts: NDArray[np.int32],
        t_ends: NDArray[np.int32],
        row_starts: NDArray[np.int32],
        row_ends: NDArray[np.int32],
    ) -> tuple["_MixedRealign", NDArray[np.int32]]:
        """One window's split-kernel realign state + deletion-extended track ends.

        Same contract and the same deletion-extension formula as
        `_Svar1Backend.mixed_realign_window` -- see that docstring for why the
        extension exists and why it deliberately does not match the written
        WRITER's `chromEnd`. Only the source of `diffs` differs: SVAR1 walks a
        CSR in numpy (`get_diffs_sparse`), SVAR2 asks the read-bound kernel
        (`hap_diffs_from_svar2_readbound`), which is the same call the written
        SVAR2 path makes (`Svar2Haps._haplotype_diffs`).
        """
        from .._threads import should_parallelize
        from ..genvarloader import hap_diffs_from_svar2_readbound

        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        n_s = len(s_idx)
        n_rows = len(r_idx) * n_s
        P = self.ploidy

        window = self.read_window(r_idx, s_idx)
        gathered = self._gather_rows(r_idx, s_idx, window, 0, n_rows)
        gi = tuple(gathered[:7])
        shifts = np.asarray(gathered[7], np.int32)
        contig_idx = cast(int, window["contig_idx"])
        contig = self._contigs[contig_idx]

        diffs = np.asarray(
            hap_diffs_from_svar2_readbound(
                self._store,
                contig,
                gi[0],
                gi[1],
                gi[2],
                gi[3],
                gi[4],
                gi[5],
                gi[6],
                P,
                False,
                should_parallelize(n_rows * P),
                None,
            ),
            np.int32,
        ).reshape(n_rows, P)

        max_del_row = -diffs.clip(max=0).min(1)
        region_max_del = max_del_row.reshape(len(r_idx), n_s).max(1)
        t_ends_ext = np.ascontiguousarray(
            t_ends.astype(np.int64) + region_max_del, np.int32
        )
        return (
            _Svar2Realign(
                store=self._store,
                contig=contig,
                ploidy=P,
                gi=gi,
                diffs=diffs,
                shifts=shifts,
            ),
            t_ends_ext,
        )
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_svar2.py -q`
Expected: PASS (1 test).

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks_svar2.py
git commit -m "feat(streaming): add SVAR2 split-kernel mixed realign state

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
```

## Task 3: Wire tracks into the SVAR2 `"sync"` drive + the mixed fixture

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` — the `elif self._prefetch_strategy == "sync":` branch (~lines 2008–2054), and the `"svar2_engine"` branch's guard (~line 2072)
- Modify: `tests/dataset/conftest.py` — add `StreamingSvar2TracksFixture` + `streaming_svar2_tracks_fixture`
- Test: `tests/dataset/test_streaming_tracks_svar2.py`

**Interfaces:**
- Consumes: `_Svar2Backend.mixed_realign_window` (Task 2); `_TrackWindow`, `_coerce_window_itvs`, `_tracks_from_intervals` (pre-existing, `_streaming.py`).
- Produces (for Task 4): the fixture
  ```python
  @dataclass
  class StreamingSvar2TracksFixture:
      bed: pl.DataFrame          # NOT a path -- pass straight to StreamingDataset
      reference_path: Path
      svar2_path: Path
      dataset_path: Path         # gvl.write output, the oracle
      bigwigs: gvl.BigWigs       # track "alpha"
      table: gvl.Table           # track "zeta"
      samples: list[str]
      contigs_list: list[str]
  ```

- [ ] **Step 1: Write the failing parity test**

Append to `tests/dataset/test_streaming_tracks_svar2.py`:

```python
def _assert_haps_cell_equal(streamed, expected, ploidy: int, ctx="") -> None:
    """Assert the HAPLOTYPE half of one mixed cell matches ``Dataset[r, s][0]``."""
    for h in range(ploidy):
        got = np.asarray(streamed[h])
        exp = np.asarray(expected[h])
        assert got.shape == exp.shape, (
            f"{ctx}hap {h}: shape {got.shape} != oracle {exp.shape}"
        )
        np.testing.assert_array_equal(
            got, exp, err_msg=f"{ctx}hap {h}: haplotype bytes differ"
        )


def _assert_tracks_cell_equal(streamed, expected, ctx="") -> None:
    """Assert the TRACK half of one mixed cell matches ``Dataset[r, s][1]``."""
    got_t, exp_t = np.asarray(streamed), np.asarray(expected)
    assert got_t.shape == exp_t.shape, (
        f"{ctx}track shape {got_t.shape} != oracle {exp_t.shape}"
    )
    np.testing.assert_allclose(
        got_t, exp_t, rtol=0, atol=0, err_msg=f"{ctx}track values differ"
    )


def test_svar2_mixed_parity_with_indels(streaming_svar2_tracks_fixture):
    f = streaming_svar2_tracks_fixture
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }
```

- [ ] **Step 2: Add the fixture**

In `tests/dataset/conftest.py`, add (place it directly after `streaming_tracks_fixture` so the two mixed fixtures read together):

```python
@dataclass
class StreamingSvar2TracksFixture:
    """SVAR2 variants + two interval tracks (issue #375, Track A).

    Deliberately mirrors `StreamingTracksFixture` field-for-field so the SVAR1
    and SVAR2 mixed suites can be read side by side, and hostile in the same
    two ways: tracks are passed NON-alphabetically (`zeta` before `alpha`, so
    a test assuming argument order fails loudly -- the written path sorts,
    `_tracks.py:283`) and the bed includes a region overlapping a deletion so
    the indel re-alignment path is exercised.
    """

    #: A `pl.DataFrame` of BED3+ regions (NOT a path).
    bed: pl.DataFrame
    reference_path: Path
    svar2_path: Path
    dataset_path: Path
    bigwigs: gvl.BigWigs
    table: gvl.Table
    samples: list[str]
    contigs_list: list[str]


@pytest.fixture(scope="module")
def streaming_svar2_tracks_fixture(
    tmp_path_factory, svar2_multicontig_fixture
) -> StreamingSvar2TracksFixture:
    """SVAR2 variants + two interval tracks, written with parity-safe flags.

    Scope is ``module``, not ``session``: it depends on the module-scoped
    ``svar2_multicontig_fixture`` and pytest forbids the wider scope.
    """
    base = svar2_multicontig_fixture
    tmp_dir = tmp_path_factory.mktemp("streaming_svar2_tracks")

    bed = base.bed
    samples = list(gvl.Dataset.open(base.dataset_path).samples)

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

    # Disjoint 10 bp bins: the bed's sliding windows overlap heavily, and
    # bigwig entries must be sorted and non-overlapping.
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
                    # wrong contig, or an off-by-one bin shows up in values.
                    values.append(
                        float(10 * (i + 1) + b)
                        + (0.5 if contig != contig_sizes[0][0] else 0.0)
                    )
            bw.addEntries(chroms, starts, ends=ends, values=values)
        bw_paths[sample] = str(p)
    alpha = gvl.BigWigs("alpha", bw_paths)

    rows = []
    for i, sample in enumerate(samples):
        for c_idx, (contig, size) in enumerate(contig_sizes):
            for b, lo in enumerate(range(0, size, BIN)):
                rows.append(
                    {
                        "sample_id": sample,
                        "chrom": contig,
                        "start": lo,
                        "end": min(lo + BIN, size),
                        "value": float(1000 * c_idx + 100 * (i + 1) + b),
                    }
                )
    zeta = gvl.Table("zeta", pl.DataFrame(rows))

    out = tmp_dir / "svar2_tracks.gvl"
    gvl.write(
        path=out,
        bed=bed,
        variants=base.svar2_path,
        tracks=[zeta, alpha],
        # Parity is gated on these in v1 -- see spec Section 3.2.
        extend_to_length=False,
        max_jitter=None,
    )

    return StreamingSvar2TracksFixture(
        bed=bed,
        reference_path=base.reference_path,
        svar2_path=base.svar2_path,
        dataset_path=out,
        bigwigs=alpha,
        table=zeta,
        samples=samples,
        contigs_list=list(base.contigs),
    )
```

If `svar2_multicontig_fixture` has no `contigs` attribute, use `list(gvl.Dataset.open(base.dataset_path).contigs)` instead — check the `Svar2MultiContigFixture` dataclass at `conftest.py:191` first and match its real field names for `bed`/`reference_path`/`svar2_path`/`dataset_path`.

- [ ] **Step 3: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_svar2.py -q`
Expected: FAIL on `test_svar2_mixed_parity_with_indels` with `AttributeError: 'Ragged' object has no attribute ...` or a tuple-unpack error — the `"sync"` drive still yields haplotypes alone.

- [ ] **Step 4: Wire tracks into the `"sync"` drive**

In `_streaming.py`, in the `elif self._prefetch_strategy == "sync":` branch, replace the body of the `for r_idx, s_idx in self._plan():` loop with:

```python
                for r_idx, s_idx in self._plan():
                    window = backend.read_window(r_idx, s_idx)
                    n_s = len(s_idx)
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(s_idx, len(r_idx))
                    n_rows = len(flat_r)
                    # Issue #375 Track A: read this window's tracks once, here
                    # -- the same composition point the SVAR1 engine drive
                    # uses. Guarded on `tb is not None` so a haplotype-only
                    # SVAR2 stream pays nothing extra.
                    tb = self._track_backend
                    if tb is not None:
                        s_idx_w = np.asarray(s_idx, np.intp)
                        t_starts = np.ascontiguousarray(
                            self._regions[r_idx, 1], np.int32
                        )
                        t_ends = np.ascontiguousarray(self._regions[r_idx, 2], np.int32)
                        row_starts_w = np.repeat(t_starts, n_s).astype(np.int32)
                        row_ends_w = np.repeat(t_ends, n_s).astype(np.int32)
                        row_lengths_w = (row_ends_w - row_starts_w).astype(np.int64)
                        if self._realign_tracks:
                            realign_w, t_ends_ext = backend.mixed_realign_window(
                                r_idx, s_idx_w, t_starts, t_ends,
                                row_starts_w, row_ends_w,
                            )
                        else:
                            realign_w = None
                            t_ends_ext = t_ends
                        per_track = tb.read_window(
                            r_idx,
                            s_idx_w,
                            np.ascontiguousarray(t_starts, np.int32),
                            np.ascontiguousarray(t_ends_ext, np.int32),
                        )
                        track_w = _TrackWindow(
                            itvs=_coerce_window_itvs(per_track),
                            names=tb.names,
                            row_starts=row_starts_w,
                            row_ends=row_ends_w,
                            row_lengths=row_lengths_w,
                            realign=realign_w,
                        )
                    else:
                        track_w = None
                    for sb_lo, sb_hi in _super_batch_bounds(n_rows, sb_rows):
                        backend._fill_super_batch(
                            r_idx,
                            s_idx,
                            window,
                            sb_lo,
                            sb_hi,
                            buf,
                            parallel=should_parallelize(
                                backend._est_out_bytes(r_idx, sb_hi - sb_lo)
                            ),
                        )
                        for lo, hi in _batch_bounds(sb_lo, sb_hi, batch_size):
                            data = backend._drain(buf, lo - sb_lo, hi - sb_lo)
                            if track_w is None:
                                yield data, flat_r[lo:hi], flat_s[lo:hi]
                                continue
                            out_lengths = np.asarray(
                                data.lengths, np.int64
                            ).reshape(hi - lo, backend.ploidy)
                            if track_w.realign is not None:
                                regions_batch = np.stack(
                                    [
                                        np.zeros(hi - lo, np.int32),
                                        track_w.row_starts[lo:hi],
                                        track_w.row_ends[lo:hi],
                                    ],
                                    axis=1,
                                ).astype(np.int32)
                                # The written path's own formula, verbatim
                                # (`_reconstruct.py:216-218`). Only
                                # `FlankSample` reads it; see the engine
                                # drive's identical comment for why exact
                                # `FlankSample` parity is unattainable in ANY
                                # batched path.
                                _idx = flat_r[lo:hi].astype(np.uint64) * np.uint64(
                                    self.n_samples
                                ) + flat_s[lo:hi].astype(np.uint64)
                                base_seed = int(np.bitwise_xor.reduce(_idx))
                                tracks = track_w.realign.realign_batch(
                                    lo,
                                    hi,
                                    track_w.itvs,
                                    track_w.names,
                                    self._insertion_fill,
                                    regions_batch,
                                    track_w.row_lengths[lo:hi],
                                    out_lengths,
                                    base_seed,
                                )
                            else:
                                if isinstance(self._output_length, int):
                                    _lengths = np.full(
                                        hi - lo, self._output_length, np.int64
                                    )
                                else:
                                    _lengths = track_w.row_lengths[lo:hi]
                                tracks = _tracks_from_intervals(
                                    track_w.itvs,
                                    np.arange(lo, hi, dtype=np.int64),
                                    track_w.row_starts[lo:hi],
                                    _lengths,
                                )
                            yield (data, tracks), flat_r[lo:hi], flat_s[lo:hi]
```

Note the two SVAR2-specific simplifications vs the engine drive: there is no jitter branch (`_jitter_region_bounds`) because the SVAR2 backend already rejects jitter at construction (`_streaming.py:1364`), and `out_lengths` comes from `data.lengths` rather than raw engine offsets because `_drain` already wrapped the batch as a `Ragged`.

- [ ] **Step 5: Guard the test-only `"svar2_engine"` strategy**

At the top of the `elif self._prefetch_strategy == "svar2_engine":` branch, immediately after its `assert isinstance(...)`, add:

```python
                if self._track_backend is not None:
                    raise NotImplementedError(
                        'StreamingDataset tracks= is not wired for the test-only '
                        '"svar2_engine" prefetch strategy; the default "sync" '
                        "strategy supports it (issue #375)."
                    )
```

- [ ] **Step 6: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_svar2.py -q`
Expected: PASS (2 tests).

- [ ] **Step 7: Prove the new haplotype assertions actually bite**

Temporarily change `_assert_haps_cell_equal`'s comparison to `np.testing.assert_array_equal(got.view(np.uint8) ^ 1, exp, ...)`. Run the suite: `test_svar2_mixed_parity_with_indels` MUST fail. Revert the change and confirm PASS. Do not commit the temporary edit.

- [ ] **Step 8: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/conftest.py tests/dataset/test_streaming_tracks_svar2.py
git commit -m "feat(streaming): support mixed variants+tracks on the SVAR2 backend

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
```

## Task 4: SVAR2 mixed edge cases, guards, and docs

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (guards), `docs/source/dataset.md`, `skills/genvarloader/SKILL.md`, `docs/roadmaps/streaming-dataset.md`
- Test: `tests/dataset/test_streaming_tracks_svar2.py`

**Interfaces:**
- Consumes: everything from Tasks 2–3.
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing guard + edge-case tests**

Append to `tests/dataset/test_streaming_tracks_svar2.py`:

```python
import pytest


def test_svar2_mixed_with_len_rejected(streaming_svar2_tracks_fixture):
    """The written SVAR2 mixed path refuses fixed-length realigned tracks
    (`_reconstruct.py:_call_svar2`: the readbound kernel always sizes each hap
    to ref_len + diff, so an int output_length cannot be honored
    byte-identically). Streaming must refuse identically rather than silently
    mis-size."""
    f = streaming_svar2_tracks_fixture
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    with pytest.raises(NotImplementedError, match="fixed-length|with_len"):
        next(iter(sds.with_len(10).to_iter(batch_size=4)))


def test_svar2_realign_false_matches_written(streaming_svar2_tracks_fixture):
    """`realign_tracks=False` leaves tracks in reference coordinates -- the
    un-realigned path is backend-independent (`_tracks_from_intervals`), so it
    must work for SVAR2 too, and the track axis loses ploidy."""
    f = streaming_svar2_tracks_fixture
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs("haplotypes")
        .with_tracks(["alpha", "zeta"])
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
        realign_tracks=False,
    ).with_seqs("haplotypes")

    n = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        # (batch, n_tracks, ~length) -- no ploidy axis when un-realigned.
        assert tracks.shape[1] == 2
        for i in range(len(r_idx)):
            _assert_haps_cell_equal(
                haps[i], written[int(r_idx[i]), int(s_idx[i])][0], sds.ploidy
            )
            n += 1
    assert n == written.shape[0] * written.shape[1]


def test_svar2_mixed_single_track(streaming_svar2_tracks_fixture):
    """One track, not two: exercises the n_tracks == 1 reorder path (the
    track-major -> (b, t, p) permutation is a no-op there, so a bug in the
    permutation shows up only by contrast with the two-track case)."""
    f = streaming_svar2_tracks_fixture
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_seqs("haplotypes")
        .with_tracks("alpha")
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.bigwigs],
    ).with_seqs("haplotypes")

    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 1
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)


def test_svar2_mixed_batch_size_one(streaming_svar2_tracks_fixture):
    """batch_size=1 is the ONE case where the written path's own track-major
    layout bug (#371) is invisible, so it is also the case where a streaming
    reorder bug hides. Pin it explicitly."""
    f = streaming_svar2_tracks_fixture
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.svar2_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")

    for data, r_idx, s_idx in sds.to_iter(batch_size=1, return_indices=True):
        haps, tracks = data
        r, s = int(r_idx[0]), int(s_idx[0])
        exp_haps, exp_tracks = written[r, s]
        ctx = f"cell (r={r}, s={s}): "
        _assert_haps_cell_equal(haps[0], exp_haps, sds.ploidy, ctx=ctx)
        _assert_tracks_cell_equal(tracks[0], exp_tracks, ctx=ctx)
```

- [ ] **Step 2: Run to verify the guard test fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_svar2.py -q`
Expected: `test_svar2_mixed_with_len_rejected` FAILS (`DID NOT RAISE`). The other three may pass already; if any fails, that is a real bug in Tasks 2–3 — fix it before continuing.

- [ ] **Step 3: Add the `with_len` guard**

In `_streaming.py`, in the same construction-time guard block that Task 1 edited (~line 1254, after the `supports_mixed_tracks` check), add:

```python
            # Issue #375 Track A: the SVAR2 read-bound track kernel always
            # sizes each hap to `ref_len + diff` with no `output_length`
            # override, so a fixed-length request cannot be honored
            # byte-identically. The WRITTEN path refuses this too, in the same
            # words (`_reconstruct.py:_call_svar2`) -- this is a mirrored
            # semantic rejection, not a streaming gap.
            if (
                self._track_backend is not None
                and self._realign_tracks
                and isinstance(self._output_length, int)
                and isinstance(self._backend, _Svar2Backend)
            ):
                raise NotImplementedError(
                    "Fixed-length (with_len) haplotype-realigned tracks are not "
                    "supported for svar2 sources; use ragged output or "
                    "realign_tracks=False."
                )
```

- [ ] **Step 4: Run the whole SVAR2 mixed suite**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_svar2.py -q`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS. (Slow tests are NOT deselected by default in this repo — `pyproject.toml` has no `addopts`.)

- [ ] **Step 6: Update the docs**

In `docs/source/dataset.md`, in the mixed-output `````{important}````` block, change any wording that scopes mixed streaming to `.svar` so it reads:

> Mixed variants + tracks streaming is supported for `.svar` (SVAR1) and `.svar2` sources. VCF and PGEN sources still raise `NotImplementedError` (issue #375).

In `skills/genvarloader/SKILL.md`, make the same change everywhere the SVAR1-only mixed limitation is stated (there are two sites, near lines 488 and 606 — grep for `SVAR1` to find them).

In `docs/roadmaps/streaming-dataset.md`, append to the #279 intervals row's "Follow-ups landed since" list: `#375 Track A (SVAR2 mixed variants+tracks)`.

- [ ] **Step 7: Lint, format, typecheck**

Run:
```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
pixi run -e dev python scripts/docstring_style.py --check python/genvarloader
```
Expected: clean.

- [ ] **Step 8: Commit and open the PR**

```bash
git add -A
git commit -m "feat(streaming): SVAR2 mixed track edge cases, with_len guard, docs

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
gh pr create --base streaming --title "streaming: mixed variants+tracks for the SVAR2 backend" --body "..."
```

The PR body must reference the Track A issue and be added to the StreamingDataset project.

---

# TRACK B — Record backends (VCF + PGEN) mixed (Tasks 5–8)

**Runs in parallel with Track A. Branch from `streaming` AFTER Task 1 has merged.**

VCF and PGEN are ONE task, not two: they share `RecordStreamEngine` (`src/record_stream/engine.rs`) and the `DecodedWindow` transpose (`src/record_stream/transpose.rs`), and differ only in their already-written `WindowFiller::fill` impls (`pgen.rs:548`, `vcf.rs:197`). Neither exposes a Python-level window seam today; adding one serves both.

**This track edits Rust. Every test run must be preceded by `pixi run -e dev maturin develop --release`.**

## Task 5: Rust seam — expose the decoded window's variant table + CSR

**Files:**
- Modify: `src/record_stream/engine.rs` — add a `window_realign_inputs` pymethod alongside `debug_decode_window` (~line 1012)
- Test: `tests/dataset/test_streaming_tracks_record.py` (created here)

**Interfaces:**
- Consumes: `RecordBackend::debug_fill(&job) -> anyhow::Result<DecodedWindow>` (`engine.rs:138`, pre-existing); `DecodedWindow` fields `v_starts: Vec<i32>`, `ilens: Vec<i32>`, `geno_v_idxs: Vec<i32>`, `geno_offsets: Vec<i64>` (`transpose.rs:65-71`).
- Produces (for Task 6):
  ```python
  # RecordStreamEngine method
  def window_realign_inputs(
      self,
      contig_idx: int,
      region_starts: Sequence[int],   # u32
      region_ends: Sequence[int],     # u32
      s_lo: int,
      s_hi: int,
  ) -> tuple[
      NDArray[np.int32],  # v_starts,     (n_window_variants,) window-local
      NDArray[np.int32],  # ilens,        (n_window_variants,) window-local
      NDArray[np.int32],  # geno_v_idxs,  flat CSR values, window-local column ids
      NDArray[np.int64],  # geno_offsets, (n_samples * ploidy + 1,) CSR row offsets
  ]: ...
  ```
  Note the CSR is **per-hap of the window's sample sub-range**, NOT per (region, sample) row: regions in a window share one decoded genotype table (see `RecordBackend::kept_v_idxs`, `h = si * ploidy + p`). Task 6 replicates it across regions.

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_streaming_tracks_record.py`:

```python
"""VCF + PGEN mixed variants + tracks parity (issue #375, Track B).

Both record backends share `RecordStreamEngine`, so one seam
(`window_realign_inputs`) and one `_MixedRealign` implementation serve both;
every test here is parametrized over the two.
"""

from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl

BACKENDS = ["vcf", "pgen"]


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_backend_declares_mixed_support(backend):
    from genvarloader._dataset._streaming import _PgenBackend, _VcfBackend

    cls = _VcfBackend if backend == "vcf" else _PgenBackend
    assert cls.supports_mixed_tracks is True
    assert hasattr(cls, "mixed_realign_window")
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q`
Expected: FAIL — `assert False is True`.

- [ ] **Step 3: Add the Rust pymethod**

In `src/record_stream/engine.rs`, immediately after the `debug_decode_window` pymethod, add:

```rust
    /// Decode one window's genotype table + CSR for the Python mixed
    /// variants+tracks path (issue #375, Track B).
    ///
    /// Returns `(v_starts, ilens, geno_v_idxs, geno_offsets)`, all WINDOW-LOCAL:
    /// `geno_v_idxs` holds column indices into `v_starts`/`ilens`, and
    /// `geno_offsets` is the per-hap CSR of the window's sample sub-range
    /// (length `(s_hi - s_lo) * ploidy + 1`), NOT per (region, sample) row --
    /// every region in a window shares one decoded genotype table, exactly as
    /// [`RecordBackend::kept_v_idxs`] assumes (`h = si * ploidy + p`). The
    /// caller replicates rows across regions.
    ///
    /// Python needs these BEFORE the window's first batch, to size the
    /// deletion-extended track query, so this cannot ride along on
    /// `next_batch` (which is sub-window) and cannot read the producer's slot
    /// (which the consumer owns). It therefore does its own synchronous decode
    /// in the calling thread via `debug_fill`, the same path
    /// `debug_decode_window` uses. That means a mixed VCF/PGEN stream decodes
    /// each window TWICE: once here for the track sizing, once in the
    /// producer for the haplotypes. Correct, thread-safe, and roughly 2x the
    /// decode cost on the mixed path only -- folding it into the producer is a
    /// tracked follow-up, not a v1 requirement.
    fn window_realign_inputs<'py>(
        &self,
        py: Python<'py>,
        contig_idx: usize,
        region_starts: Vec<u32>,
        region_ends: Vec<u32>,
        s_lo: usize,
        s_hi: usize,
    ) -> PyResult<(
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i32>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        if region_starts.len() != region_ends.len() {
            return Err(PyValueError::new_err(
                "window_realign_inputs: region_starts and region_ends must have the same length",
            ));
        }
        let regions: Vec<(u32, u32)> = region_starts.into_iter().zip(region_ends).collect();
        let job = RecordJob {
            contig_idx,
            regions,
            s_lo,
            s_hi,
        };
        let backend = Arc::clone(self.core.backend());
        let slot = backend
            .debug_fill(&job)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        Ok((
            Array1::from_vec(slot.v_starts).into_pyarray(py),
            Array1::from_vec(slot.ilens).into_pyarray(py),
            Array1::from_vec(slot.geno_v_idxs).into_pyarray(py),
            Array1::from_vec(slot.geno_offsets).into_pyarray(py),
        ))
    }
```

- [ ] **Step 4: Build and check Rust**

Run:
```bash
pixi run -e dev cargo clippy --all-targets -- -D warnings
pixi run -e dev cargo fmt --check
pixi run -e dev maturin develop --release
```
Expected: clean build.

- [ ] **Step 5: Add the two class members (flag + stub) so the test can pass**

In `_streaming.py`, in BOTH `_VcfBackend` and `_PgenBackend`, flip the line Task 1 added:

```python
    #: Mixed variants+tracks is wired for this backend (issue #375, Track B).
    supports_mixed_tracks: ClassVar[bool] = True
```

and add to each:

```python
    def mixed_realign_window(
        self,
        r_idx: NDArray[np.intp],
        s_idx: NDArray[np.intp],
        t_starts: NDArray[np.int32],
        t_ends: NDArray[np.int32],
        row_starts: NDArray[np.int32],
        row_ends: NDArray[np.int32],
    ) -> tuple["_MixedRealign", NDArray[np.int32]]:
        """See `_Svar1Backend.mixed_realign_window`. Delegates to the shared
        record-backend implementation -- VCF and PGEN differ only in their
        Rust `WindowFiller`, never here."""
        return _record_mixed_realign_window(
            self, r_idx, s_idx, t_starts, t_ends, row_starts, row_ends
        )
```

`_record_mixed_realign_window` is written in Task 6; for this step, define it as a module-level stub that raises `NotImplementedError("Task 6")` so Step 6's test can pass on the flag/attribute check alone.

- [ ] **Step 6: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q`
Expected: PASS (2 tests).

- [ ] **Step 7: Commit**

```bash
git add src/record_stream/engine.rs python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks_record.py
git commit -m "feat(streaming): expose a record-stream window's variant table and CSR

Adds RecordStreamEngine.window_realign_inputs, the seam the Python mixed
variants+tracks path needs to size a VCF/PGEN window's deletion-extended
track query before pulling that window's first batch.

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
```

## Task 6: `_record_mixed_realign_window` — window-local CSR into the fused kernel

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` — replace the Task 5 stub with the real implementation (module level, near `_RealignWindow`)
- Test: `tests/dataset/test_streaming_tracks_record.py`

**Interfaces:**
- Consumes: `RecordStreamEngine.window_realign_inputs` (Task 5, signature quoted above); `_RealignWindow` and the `mixed_realign_window` contract (Task 1); `get_diffs_sparse(geno_offset_idx, geno_v_idxs, geno_offsets, ilens, q_starts, q_ends, v_starts)` from `._genotypes` (pre-existing — the SVAR1 window hook calls it identically).
- Produces (for Tasks 7–8): a working mixed path for both record backends.

**Key shape translation.** The fused kernel wants `geno_offsets` as `(2, n_rows * ploidy)` stacked `(starts, stops)` in C-order `(region, sample, ploid)`. `window_realign_inputs` returns a `(n_s * ploidy + 1,)` CSR indexed by hap `h = si * ploidy + p`. For window row `bi` (C-order `(region, sample)`, so `si = bi % n_s`):

```
start[bi * P + p] = csr[(bi % n_s) * P + p]
stop [bi * P + p] = csr[(bi % n_s) * P + p + 1]
```

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_streaming_tracks_record.py`:

```python
@pytest.mark.parametrize("backend", BACKENDS)
def test_record_window_csr_replicates_across_regions(
    streaming_record_tracks_fixture, backend
):
    """The engine's CSR is per (sample, ploid) for the whole window; the mixed
    path must replicate it across the window's regions in C-order (region,
    sample). A transposed or un-replicated CSR reads another sample's variants
    and shows up as a silent parity failure, so pin the shape directly."""
    f = streaming_record_tracks_fixture(backend)
    sds = gvl.StreamingDataset(
        f.bed, reference=f.reference_path, variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")
    b = sds._backend
    n_reg, n_s = 2, sds.n_samples
    r_idx = np.arange(n_reg, dtype=np.intp)
    s_idx = np.arange(n_s, dtype=np.intp)
    t_starts = np.ascontiguousarray(sds._regions[r_idx, 1], np.int32)
    t_ends = np.ascontiguousarray(sds._regions[r_idx, 2], np.int32)
    row_starts = np.repeat(t_starts, n_s).astype(np.int32)
    row_ends = np.repeat(t_ends, n_s).astype(np.int32)
    state, t_ends_ext = b.mixed_realign_window(
        r_idx, s_idx, t_starts, t_ends, row_starts, row_ends
    )
    assert state.geno_offsets.shape == (2, n_reg * n_s * sds.ploidy)
    assert state.diffs.shape == (n_reg * n_s, sds.ploidy)
    assert t_ends_ext.shape == (n_reg,)
    assert (t_ends_ext >= t_ends).all()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q`
Expected: FAIL — `NotImplementedError: Task 6` from the stub. (The fixture arrives in Task 7; until then this test errors on the missing fixture — that is also a failure, and Task 7 resolves it. If you prefer a strictly green intermediate, write Task 7's fixture first, then return here.)

- [ ] **Step 3: Replace the stub with the real implementation**

In `_streaming.py`, replace the `_record_mixed_realign_window` stub with:

```python
def _record_mixed_realign_window(
    backend: "_VcfBackend | _PgenBackend",
    r_idx: NDArray[np.intp],
    s_idx: NDArray[np.intp],
    t_starts: NDArray[np.int32],
    t_ends: NDArray[np.int32],
    row_starts: NDArray[np.int32],
    row_ends: NDArray[np.int32],
) -> tuple["_MixedRealign", NDArray[np.int32]]:
    """One record-stream window's fused-kernel realign state (issue #375, Track B).

    Shared by `_VcfBackend` and `_PgenBackend`: the two differ only in their
    Rust `WindowFiller`, so the Python side is one function, not two.

    Produces exactly the same `_RealignWindow` shape `_Svar1Backend` produces,
    with two differences that the kernel cannot observe:

    - The variant tables (`v_starts`/`ilens`) and the CSR values
      (`geno_v_idxs`) are WINDOW-LOCAL, not dataset-global. The kernel only
      ever indexes the tables through `geno_v_idxs`, so a consistent local
      pair behaves identically to a consistent global pair. This is precisely
      why Task 1 moved those three arrays onto `_RealignWindow` instead of
      reading them off the backend.
    - The engine's CSR is per HAP of the window's sample sub-range
      (`h = si * ploidy + p`), because every region in a window shares one
      decoded genotype table (`RecordBackend::kept_v_idxs`). The fused kernel
      wants per (region, sample, ploid) rows, so the CSR is replicated across
      the window's regions here.

    The deletion-extension formula for `t_ends_ext` is `_Svar1Backend`'s,
    verbatim -- see that method's docstring for why it exists and why it
    deliberately does not match the written WRITER's `chromEnd`.
    """
    from ._genotypes import get_diffs_sparse

    r_idx = np.asarray(r_idx, np.intp)
    s_idx = np.asarray(s_idx, np.intp)
    n_reg, n_s = len(r_idx), len(s_idx)
    n_rows = n_reg * n_s
    P = backend.ploidy

    contig_idx = int(backend._regions[r_idx[0], 0])
    if not np.all(backend._regions[r_idx, 0] == contig_idx):
        raise ValueError(
            "_record_mixed_realign_window: window spans multiple contigs; "
            "every engine call must be single-contig."
        )

    engine = backend._mixed_engine()
    v_starts, ilens, geno_v_idxs, csr = engine.window_realign_inputs(
        contig_idx,
        np.ascontiguousarray(backend._regions[r_idx, 1], np.uint32).tolist(),
        np.ascontiguousarray(backend._regions[r_idx, 2], np.uint32).tolist(),
        int(s_idx[0]),
        int(s_idx[-1]) + 1,
    )
    v_starts = np.ascontiguousarray(v_starts, np.int32)
    ilens = np.ascontiguousarray(ilens, np.int32)
    geno_v_idxs = np.ascontiguousarray(geno_v_idxs, np.int32)
    csr = np.ascontiguousarray(csr, np.int64)

    # Replicate the per-(sample, ploid) CSR across the window's regions,
    # C-order (region, sample, ploid): row bi has si = bi % n_s.
    hap_of_row = (
        np.tile(np.arange(n_s, dtype=np.int64), n_reg)[:, None] * P
        + np.arange(P, dtype=np.int64)[None, :]
    ).reshape(-1)
    o_starts = csr[hap_of_row]
    o_stops = csr[hap_of_row + 1]
    geno_offsets_w = np.stack([o_starts, o_stops])
    geno_offset_idx_w = np.arange(n_rows * P, dtype=np.intp).reshape(n_rows, P)

    diffs_w = get_diffs_sparse(
        geno_offset_idx_w,
        geno_v_idxs,
        geno_offsets_w,
        ilens,
        q_starts=row_starts,
        q_ends=row_ends,
        v_starts=v_starts,
    )
    max_del_row = -diffs_w.clip(max=0).min(1)
    region_max_del = max_del_row.reshape(n_reg, n_s).max(1)
    t_ends_ext = np.ascontiguousarray(
        t_ends.astype(np.int64) + region_max_del, np.int32
    )
    return (
        _RealignWindow(
            diffs=diffs_w,
            geno_offsets=geno_offsets_w,
            geno_offset_idx=geno_offset_idx_w,
            geno_v_idxs=geno_v_idxs,
            v_starts=v_starts,
            ilens=ilens,
        ),
        t_ends_ext,
    )
```

- [ ] **Step 4: Give the record backends a `_mixed_engine()` accessor**

The mixed path needs *an* engine to call `window_realign_inputs` on, and the drive's engine is owned by the `to_iter` generator. Add to BOTH `_VcfBackend` and `_PgenBackend`:

```python
    def _mixed_engine(self):
        """A zero-job engine kept solely for `window_realign_inputs` calls.

        `window_realign_inputs` decodes the window it is handed and never
        touches the engine's own plan, so a plan-less engine is enough -- and
        keeping it separate from the drive's engine means the mixed track read
        cannot perturb the producer/consumer lockstep. Built lazily and cached
        on the backend: constructing it is the same cost as opening the source,
        paid once per `StreamingDataset`, not once per window.
        """
        if self._mixed_engine_obj is None:
            self._mixed_engine_obj = self.build_engine([], 1)
        return self._mixed_engine_obj
```

and initialize `self._mixed_engine_obj = None` in each `__init__`.

Before writing this, read each backend's `build_engine` signature and confirm an empty job list is accepted; if `RecordStreamEngine`'s constructor rejects zero jobs, instead pass the drive's engine down. In that case, change `mixed_realign_window`'s signature to take an `engine` argument and thread `engine` from the drive at the `mixed_realign_window(...)` call site — a one-line change in the drive, and Track A's SVAR2 implementation simply ignores the extra argument.

- [ ] **Step 5: Rebuild and run**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q
```
Expected: PASS (4 tests) once Task 7's fixture exists.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/test_streaming_tracks_record.py
git commit -m "feat(streaming): wire VCF/PGEN window CSR into the fused realign kernel

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
```

## Task 7: The VCF + PGEN mixed fixture

**Files:**
- Modify: `tests/dataset/conftest.py`
- Test: `tests/dataset/test_streaming_tracks_record.py`

**Interfaces:**
- Consumes: `vcf_snp_ins_del_multi` and `pgen_snp_ins_del_multi` (pre-existing module-scoped fixtures at `conftest.py:587` and `:768`; each carries a source file plus `.regions` and `.fasta`, and NO pre-written `gvl.Dataset` — `streaming_case` at `:1242` writes one lazily and is the pattern to follow).
- Produces (for Tasks 6 and 8): a callable fixture
  ```python
  streaming_record_tracks_fixture(backend: Literal["vcf", "pgen"]) -> StreamingRecordTracksFixture
  ```
  with fields `bed: pl.DataFrame`, `reference_path: Path`, `variants_path: str`, `dataset_path: Path`, `bigwigs: gvl.BigWigs`, `table: gvl.Table`, `samples: list[str]`.

- [ ] **Step 1: Add the fixture**

In `tests/dataset/conftest.py`, after `streaming_svar2_tracks_fixture` (or after `streaming_tracks_fixture` if Track A has not merged into your branch), add:

```python
@dataclass
class StreamingRecordTracksFixture:
    """VCF or PGEN variants + two interval tracks (issue #375, Track B).

    Same hostile shape as `StreamingTracksFixture`: tracks passed
    NON-alphabetically so a test assuming argument order fails loudly, and a
    bed with a region overlapping a deletion so indel re-alignment is
    exercised rather than the trivial path.
    """

    bed: pl.DataFrame
    reference_path: Path
    variants_path: str
    dataset_path: Path
    bigwigs: gvl.BigWigs
    table: gvl.Table
    samples: list[str]


@pytest.fixture
def streaming_record_tracks_fixture(request, tmp_path_factory):
    """Factory fixture: backend name -> `StreamingRecordTracksFixture`.

    Mirrors `streaming_case`'s lazy-write pattern (`conftest.py:1242`): the
    `vcf_snp_ins_del_multi` / `pgen_snp_ins_del_multi` fixtures carry only the
    source file, so the written oracle is produced here.
    """

    def _case(backend: str) -> StreamingRecordTracksFixture:
        if backend == "vcf":
            f = request.getfixturevalue("vcf_snp_ins_del_multi")
            variants_path = str(f.vcf)
        elif backend == "pgen":
            f = request.getfixturevalue("pgen_snp_ins_del_multi")
            variants_path = str(f.pgen)
        else:
            raise ValueError(
                f"streaming_record_tracks_fixture: unknown backend {backend!r}"
            )

        tmp_dir = tmp_path_factory.mktemp(f"streaming_{backend}_tracks")
        bed = f.regions
        reference_path = Path(f.fasta)

        # Write once WITHOUT tracks purely to learn the dataset's public
        # sample order -- track files must be keyed by those names.
        probe = tmp_dir / "probe.gvl"
        gvl.write(probe, bed, variants=variants_path, overwrite=True)
        samples = list(gvl.Dataset.open(probe).samples)

        fai = pl.read_csv(
            str(reference_path) + ".fai",
            separator="\t",
            has_header=False,
            new_columns=["chrom", "length", "offset", "linebases", "linewidth"],
        )
        contig_sizes = [
            (r["chrom"], int(r["length"]))
            for r in fai.iter_rows(named=True)
            if r["chrom"] in set(bed["chrom"].to_list())
        ]

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
                        values.append(
                            float(10 * (i + 1) + b)
                            + (0.5 if contig != contig_sizes[0][0] else 0.0)
                        )
                bw.addEntries(chroms, starts, ends=ends, values=values)
            bw_paths[sample] = str(p)
        alpha = gvl.BigWigs("alpha", bw_paths)

        rows = []
        for i, sample in enumerate(samples):
            for c_idx, (contig, size) in enumerate(contig_sizes):
                for b, lo in enumerate(range(0, size, BIN)):
                    rows.append(
                        {
                            "sample_id": sample,
                            "chrom": contig,
                            "start": lo,
                            "end": min(lo + BIN, size),
                            "value": float(1000 * c_idx + 100 * (i + 1) + b),
                        }
                    )
        zeta = gvl.Table("zeta", pl.DataFrame(rows))

        out = tmp_dir / f"{backend}_tracks.gvl"
        gvl.write(
            path=out,
            bed=bed,
            variants=variants_path,
            # NON-alphabetical on purpose (see the dataclass docstring).
            tracks=[zeta, alpha],
            # Parity is gated on these in v1 -- see spec Section 3.2.
            extend_to_length=False,
            max_jitter=None,
            overwrite=True,
        )

        return StreamingRecordTracksFixture(
            bed=bed,
            reference_path=reference_path,
            variants_path=variants_path,
            dataset_path=out,
            bigwigs=alpha,
            table=zeta,
            samples=samples,
        )

    return _case
```

Before writing this, read `VcfSnpInsDelMultiFixture` (`conftest.py:587`) and `PgenSnpInsDelMultiFixture` (`:768`) and match their real field names for the source path, `regions`, and `fasta`.

- [ ] **Step 2: Run Task 6's shape test against the real fixture**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q`
Expected: PASS (4 tests).

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/conftest.py
git commit -m "test(streaming): add the VCF/PGEN mixed variants+tracks fixture

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
```

## Task 8: VCF + PGEN mixed parity, edge cases, and docs

**Files:**
- Modify: `docs/source/dataset.md`, `skills/genvarloader/SKILL.md`, `docs/roadmaps/streaming-dataset.md`
- Test: `tests/dataset/test_streaming_tracks_record.py`

**Interfaces:**
- Consumes: everything from Tasks 5–7.
- Produces: nothing downstream.

- [ ] **Step 1: Write the failing parity tests**

Append to `tests/dataset/test_streaming_tracks_record.py`:

```python
def _assert_haps_cell_equal(streamed, expected, ploidy: int, ctx="") -> None:
    for h in range(ploidy):
        got = np.asarray(streamed[h])
        exp = np.asarray(expected[h])
        assert got.shape == exp.shape, (
            f"{ctx}hap {h}: shape {got.shape} != oracle {exp.shape}"
        )
        np.testing.assert_array_equal(
            got, exp, err_msg=f"{ctx}hap {h}: haplotype bytes differ"
        )


def _assert_tracks_cell_equal(streamed, expected, ctx="") -> None:
    got_t, exp_t = np.asarray(streamed), np.asarray(expected)
    assert got_t.shape == exp_t.shape, (
        f"{ctx}track shape {got_t.shape} != oracle {exp_t.shape}"
    )
    np.testing.assert_allclose(
        got_t, exp_t, rtol=0, atol=0, err_msg=f"{ctx}track values differ"
    )


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("batch_size", [1, 4])
def test_record_mixed_parity_with_indels(
    streaming_record_tracks_fixture, backend, batch_size
):
    """Both halves of every mixed cell against `Dataset[r, s]`.

    `batch_size=1` is included because it is the ONE case where the written
    path's own track-major layout bug (#371) is invisible, so it is also where
    a streaming reorder bug would hide.
    """
    f = streaming_record_tracks_fixture(backend)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(
        batch_size=batch_size, return_indices=True
    ):
        haps, tracks = data
        assert tracks.shape[1] == 2, f"track axis {tracks.shape} lost a track"
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_mixed_parity_under_forced_windowing(
    streaming_record_tracks_fixture, backend
):
    """A tiny `max_mem` forces multi-window plans, so the per-window CSR
    replication and the per-window deletion extension are exercised on windows
    that hold a strict subset of the regions/samples -- the case a single-window
    plan can never reach."""
    f = streaming_record_tracks_fixture(backend)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
        max_mem="1m",
    ).with_seqs("haplotypes")

    seen = set()
    for data, r_idx, s_idx in sds.to_iter(batch_size=2, return_indices=True):
        haps, tracks = data
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)
            seen.add((r, s))

    assert seen == {
        (r, s) for r in range(written.shape[0]) for s in range(written.shape[1])
    }


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_mixed_fixed_length_parity(streaming_record_tracks_fixture, backend):
    """`with_len(L)` on the record mixed path. Unlike SVAR2, the fused kernel
    honors an explicit output length (the drive takes `out_lengths` from the
    batch's own offsets, which already bake in `L`), so this must match the
    written oracle rather than raise."""
    f = streaming_record_tracks_fixture(backend)
    L = 10
    written = (
        gvl.Dataset.open(f.dataset_path, reference=f.reference_path)
        .with_len(L)
        .with_seqs("haplotypes")
    )
    sds = (
        gvl.StreamingDataset(
            f.bed,
            reference=f.reference_path,
            variants=f.variants_path,
            tracks=[f.table, f.bigwigs],
        )
        .with_len(L)
        .with_seqs("haplotypes")
    )

    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp_haps, exp_tracks = written[r, s]
            ctx = f"{backend} cell (r={r}, s={s}): "
            _assert_haps_cell_equal(haps[i], exp_haps, sds.ploidy, ctx=ctx)
            _assert_tracks_cell_equal(tracks[i], exp_tracks, ctx=ctx)


@pytest.mark.parametrize("backend", BACKENDS)
def test_record_realign_false_matches_written(
    streaming_record_tracks_fixture, backend
):
    """`realign_tracks=False` keeps tracks in reference coordinates via the
    backend-independent `_tracks_from_intervals`, so the track axis loses
    ploidy and no CSR is read at all."""
    f = streaming_record_tracks_fixture(backend)
    written = gvl.Dataset.open(f.dataset_path, reference=f.reference_path).with_seqs(
        "haplotypes"
    )
    sds = gvl.StreamingDataset(
        f.bed,
        reference=f.reference_path,
        variants=f.variants_path,
        tracks=[f.table, f.bigwigs],
        realign_tracks=False,
    ).with_seqs("haplotypes")

    n = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        haps, tracks = data
        assert tracks.shape[1] == 2
        for i in range(len(r_idx)):
            _assert_haps_cell_equal(
                haps[i], written[int(r_idx[i]), int(s_idx[i])][0], sds.ploidy
            )
            n += 1
    assert n == written.shape[0] * written.shape[1]
```

- [ ] **Step 2: Run to verify they fail (or reveal real bugs)**

Run:
```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/dataset/test_streaming_tracks_record.py -q
```
Expected: the four new tests exercise the path Tasks 5–7 built. If any fail, the bug is real — fix it in `_record_mixed_realign_window` or the Rust seam, not in the test. The two most likely defects, in order: the CSR replication in `_record_mixed_realign_window` (a transposed `hap_of_row` reads another sample's variants), and a window-local-vs-global mixup if any code path still reaches for a dataset-global variant table.

- [ ] **Step 3: Prove the new haplotype assertions actually bite**

Temporarily change `_assert_haps_cell_equal` to compare `got.view(np.uint8) ^ 1` against `exp`. Every test that compares haplotypes MUST fail. Revert and confirm PASS. Do not commit the temporary edit.

- [ ] **Step 4: Run the full tree**

Run: `pixi run -e dev pytest tests -q`
Expected: PASS.

- [ ] **Step 5: Update the docs**

In `docs/source/dataset.md`, in the mixed-output `````{important}````` block, change the backend scope sentence to:

> Mixed variants + tracks streaming is supported for every variant source: `.svar` (SVAR1), `.svar2`, VCF/BCF, and PGEN.

(If Track A has not merged yet, write "for `.svar` (SVAR1), VCF/BCF, and PGEN" and leave the SVAR2 wording to Track A; whichever track merges second reconciles the sentence.)

Make the same change in `skills/genvarloader/SKILL.md` at both SVAR1-only sites (grep for `SVAR1`).

In `docs/roadmaps/streaming-dataset.md`, append to the #279 intervals row's "Follow-ups landed since" list: `#375 Track B (VCF/PGEN mixed variants+tracks)`.

- [ ] **Step 6: File the double-decode follow-up issue**

```bash
gh issue create \
  --title "streaming: fold the record-backend mixed track window decode into the producer" \
  --body "PR for #375 Track B added \`RecordStreamEngine.window_realign_inputs\`, which decodes a window synchronously in the calling thread so Python can size the deletion-extended track query before pulling that window's first batch. A mixed VCF/PGEN stream therefore decodes each window TWICE: once here, once in the producer.

Correct and thread-safe, but roughly 2x the decode cost on the mixed path. Folding the variant table + CSR into what the producer already hands the consumer would remove the second decode. Not required for v1 parity.

Relates to #375."
```

Add the issue to the StreamingDataset project.

- [ ] **Step 7: Lint, format, typecheck, Rust checks**

Run:
```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
pixi run -e dev cargo clippy --all-targets -- -D warnings
pixi run -e dev cargo fmt --check
pixi run -e dev python scripts/docstring_style.py --check python/genvarloader
```
Expected: clean.

- [ ] **Step 8: Commit and open the PR**

```bash
git add -A
git commit -m "feat(streaming): mixed variants+tracks parity for VCF and PGEN

Relates to #375

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_014LTUVE8PofP9FD2m7A6uBg"
gh pr create --base streaming --title "streaming: mixed variants+tracks for the VCF and PGEN backends" --body "..."
```

---

## Merge Order and Conflict Surface

1. **Task 1 merges first, alone.** Both tracks branch from `streaming` after it lands.
2. **Tracks A and B then merge in either order.** Their only overlap in `_streaming.py` is:
   - `docs/source/dataset.md` and `skills/genvarloader/SKILL.md` — the same sentence. The second track to merge reconciles it (both PRs say so explicitly in their doc steps).
   - `docs/roadmaps/streaming-dataset.md` — adjacent list items, a trivial conflict.
   - `tests/dataset/conftest.py` — adjacent fixture definitions, a trivial conflict.

   Every code change is in a disjoint class body or a disjoint module-level definition.

## Self-Review Notes

- **Spec coverage.** The #279 spec deferred exactly one thing to #375: non-SVAR1 backends on the mixed path. Task 1 removes the structural blocker, Tasks 2–4 cover SVAR2, Tasks 5–8 cover VCF and PGEN. The spec's Section 3.2 parity preconditions (`extend_to_length=False`, `max_jitter=None`) are pinned in both new fixtures and restated in Global Constraints.
- **Deliberate v1 limitations, each with a guard and a test:** SVAR2 + `with_len` (Task 4 Step 3, mirroring the written path's own refusal); SVAR2 + the test-only `"svar2_engine"` strategy (Task 3 Step 5); the record backends' double window decode (Task 8 Step 6 files the follow-up). None of these is a silent gap.
- **Known asymmetry carried forward, not introduced:** the deletion-extension formula matches the read path, not the written WRITER's `chromEnd` (`_write.py:1084`). Task 1's docstring records why this is unobservable today and that an `extend_to_length` follow-up must revisit it.
- **Interface consistency check:** `mixed_realign_window` has one signature, used identically by all four backends; `_MixedRealign.realign_batch` has one signature, implemented by `_RealignWindow` (SVAR1 + record) and `_Svar2Realign` (Track A) and called from exactly two drive sites (the engine drive, Task 1 Step 8; the SVAR2 `"sync"` drive, Task 3 Step 4). `_RealignWindow`'s three new fields are populated by both producers and read only inside its own `realign_batch`.
