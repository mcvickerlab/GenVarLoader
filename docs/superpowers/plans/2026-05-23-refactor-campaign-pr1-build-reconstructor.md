# Refactor Campaign PR1 — Centralize Reconstructor Construction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate scattered, `isinstance`-dispatched construction of the 5 reconstructor classes (`Haps`, `Ref`, `Tracks`, `RefTracks`, `HapsTracks`) by introducing a single `_build_reconstructor(seqs, tracks) -> Reconstructor` factory. All `with_*` methods and `Dataset.open` route through it; the propagation-isinstance ladder in `with_settings` collapses to a one-liner.

**Architecture:** The 5 reconstructor classes already form a valid ADT — invalid `(seqs=None, tracks=None)` state is not representable in their union. The pain isn't the class count; it's that *construction* is duplicated across many sites in `_impl.py`, each independently doing `isinstance` dispatch to pick the right combined class. We centralize that dispatch in one factory and treat `_seqs` (which already carries the output kind via `Haps`'s generic parameter) as the single authoritative source of "which output mode + which sources." `_recon` stays as a stored attrs field (avoids attrs-frozen property awkwardness) but is only ever assigned via the factory.

**Tech Stack:** attrs, pyrefly (strict), pytest, pixi.

**Campaign context:** This is PR1 of the refactor campaign described in `docs/superpowers/specs/2026-05-23-refactor-campaign-design.md`. PR0 (pyrefly baseline) landed in #181. This PR re-shapes the campaign: the original PR1 ("DatasetSettings value object") and PR5 ("Pipeline composition") were merged conceptually into this single PR; subsequent PRs renumber as documented in the spec update bundled in this PR.

---

## File Structure

**Modified files:**
- `docs/superpowers/specs/2026-05-23-refactor-campaign-design.md` — campaign spec re-sequence (already staged on this branch as Task 0)
- `python/genvarloader/_dataset/_reconstruct.py` — add `_build_reconstructor` factory
- `python/genvarloader/_dataset/_impl.py` — replace scattered construction sites with factory calls; simplify `sequence_type`
- `tests/dataset/test_build_reconstructor.py` — **new file** — unit tests for the factory

**Out of scope (this PR):**
- Per-setting decomposition of `with_settings` (now smaller, but the per-method extraction is left to a future small PR if needed — with the factory in place, `with_settings` shrinks naturally from ~150 lines without further work)
- Splitting `_reconstruct.py` or `_impl.py` (that's PR6)
- Touching `Haps._get_haplotypes` or extracting the haplotype kernel (PR5)

---

## Task 0: Commit the campaign spec re-sequence

The spec update was made before branching (it motivates this PR's design). Commit it first as a docs-only change so the rest of the branch is purely implementation.

**Files:**
- Modify (already in working tree): `docs/superpowers/specs/2026-05-23-refactor-campaign-design.md`

- [ ] **Step 1: Verify the spec change is staged on this branch**

Run:

```bash
git status
git diff docs/superpowers/specs/2026-05-23-refactor-campaign-design.md | head -40
```

Expected: the spec file is modified; the diff shows the new PR1 framing (factory) replacing the old PR1 framing (DatasetSettings), and renumbering of PR5–PR8.

If `git status` is clean, the spec change was already committed — skip to Task 1.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/2026-05-23-refactor-campaign-design.md
git commit -m "docs(refactor-campaign): re-sequence PR1 as reconstructor factory"
```

---

## Task 1: Add `_build_reconstructor` factory to `_reconstruct.py`

The factory is the single source of truth for "given (seqs, tracks) sources, which of the 5 reconstructor classes to construct." It enforces the invariant that at least one source must be present (raises `ValueError` otherwise — this state should never arise in practice).

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` — append the factory at the bottom (after `HapsTracks`)
- Create: `tests/dataset/test_build_reconstructor.py`

**TDD: write the test first.**

- [ ] **Step 1: Write the failing test file**

Create `tests/dataset/test_build_reconstructor.py`:

```python
"""Unit tests for `_build_reconstructor` factory.

These exercise the factory in isolation, without spinning up a full Dataset.
Synthetic sources are constructed via direct attrs instantiation. The goal is
parity with the construction logic that previously lived inline in
`Dataset.open`, `with_seqs`, etc.
"""

from __future__ import annotations

import pytest
from unittest.mock import Mock

from genvarloader._dataset._reconstruct import (
    Haps,
    HapsTracks,
    Ref,
    RefTracks,
    Tracks,
    _build_reconstructor,
)


def _haps_mock() -> Haps:
    """A Haps stand-in. Real Haps construction needs file paths; for factory
    routing tests we only need isinstance dispatch to land in the right branch.
    We use a Mock with spec=Haps so isinstance(m, Haps) is True."""
    return Mock(spec=Haps)


def _ref_mock() -> Ref:
    return Mock(spec=Ref)


def _tracks_mock() -> Tracks:
    return Mock(spec=Tracks)


def test_haps_only_returns_haps():
    seqs = _haps_mock()
    result = _build_reconstructor(seqs, None)
    assert result is seqs


def test_ref_only_returns_ref():
    seqs = _ref_mock()
    result = _build_reconstructor(seqs, None)
    assert result is seqs


def test_tracks_only_returns_tracks():
    tracks = _tracks_mock()
    result = _build_reconstructor(None, tracks)
    assert result is tracks


def test_haps_and_tracks_returns_haps_tracks():
    seqs = _haps_mock()
    tracks = _tracks_mock()
    result = _build_reconstructor(seqs, tracks)
    assert isinstance(result, HapsTracks)
    assert result.haps is seqs
    assert result.tracks is tracks


def test_ref_and_tracks_returns_ref_tracks():
    seqs = _ref_mock()
    tracks = _tracks_mock()
    result = _build_reconstructor(seqs, tracks)
    assert isinstance(result, RefTracks)
    assert result.seqs is seqs
    assert result.tracks is tracks


def test_neither_raises_value_error():
    with pytest.raises(ValueError, match="at least one"):
        _build_reconstructor(None, None)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pixi run -e dev pytest tests/dataset/test_build_reconstructor.py -v
```

Expected: import error or `AttributeError: module ... has no attribute '_build_reconstructor'`.

- [ ] **Step 3: Implement the factory**

In `python/genvarloader/_dataset/_reconstruct.py`, append at the very bottom of the file (after `HapsTracks` and any other code):

```python
def _build_reconstructor(
    seqs: Haps | Ref | None,
    tracks: Tracks | None,
) -> Reconstructor:
    """Construct the reconstructor for the given sources.

    This is the single source of truth for "given (seqs, tracks), which of the
    5 reconstructor classes do we construct?" Callers in `_impl.py` route all
    construction through this function so the dispatch lives in exactly one
    place.

    Invariant: at least one of `seqs` or `tracks` must be non-None.
    """
    match seqs, tracks:
        case None, None:
            raise ValueError(
                "_build_reconstructor requires at least one of seqs or tracks "
                "to be non-None."
            )
        case (Haps() | Ref()) as s, None:
            return s
        case None, Tracks() as t:
            return t
        case Ref() as s, Tracks() as t:
            return RefTracks(seqs=s, tracks=t)
        case Haps() as s, Tracks() as t:
            return HapsTracks(haps=s, tracks=t)
        case _:
            # Unreachable given the input types — exhaustive match above.
            raise AssertionError(
                f"unreachable: _build_reconstructor got {type(seqs).__name__=}, "
                f"{type(tracks).__name__=}"
            )
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pixi run -e dev pytest tests/dataset/test_build_reconstructor.py -v
```

Expected: all 6 tests pass.

- [ ] **Step 5: pyrefly clean**

Run:

```bash
pixi run -e dev typecheck
```

Expected: exit 0 (no new errors). Pyrefly may still emit warnings (existing baseline relaxations) but no new errors.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_reconstruct.py tests/dataset/test_build_reconstructor.py
git commit -m "feat(reconstruct): add _build_reconstructor factory"
```

---

## Task 2: Use the factory in `Dataset.open`

`Dataset.open` currently contains a ~15-line match (around lines 246–261) that picks the right reconstructor class. Replace with a factory call.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py:246-261` (the match block in `Dataset.open`)

- [ ] **Step 1: Read the current match block**

Run:

```bash
sed -n '240,265p' python/genvarloader/_dataset/_impl.py
```

Confirm the match looks like:

```python
match seqs, tracks:
    case None, None:
        raise RuntimeError("Malformed dataset: neither genotypes nor intervals found.")
    case Ref() | Haps(), None:
        recon = seqs
    case None, Tracks():
        recon = tracks
    case Ref(), Tracks():
        recon = RefTracks(seqs, tracks)
    case Haps(), Tracks():
        recon = HapsTracks(seqs, tracks)
    case seqs, tracks:
        assert_never(seqs)
        assert_never(tracks)
```

(If structure differs, abort and report — the line numbers may have shifted since this plan was drafted.)

- [ ] **Step 2: Replace the match with a factory call**

Use the Edit tool to replace the entire match block (currently around lines 246–261) with:

```python
if seqs is None and tracks is None:
    raise RuntimeError("Malformed dataset: neither genotypes nor intervals found.")
recon = _build_reconstructor(seqs, tracks)
```

This keeps the user-facing "Malformed dataset" error message (the factory's `ValueError` message is for internal callers; the user-facing one for opening a dataset directory should stay).

- [ ] **Step 3: Update the import**

Find the existing import line:

```python
from ._reconstruct import Haps, HapsTracks, Ref, RefTracks, Tracks, TrackType
```

Add `_build_reconstructor`:

```python
from ._reconstruct import (
    Haps,
    HapsTracks,
    Ref,
    RefTracks,
    Tracks,
    TrackType,
    _build_reconstructor,
)
```

- [ ] **Step 4: Also remove the now-unused `_recon=recon  # pyrefly: ignore[unbound-name]` comment**

After the match replacement, the `recon` variable is now guaranteed to be initialized (the factory either returns or raises before reaching the next line). The PR0 suppression on line 306 (`_recon=recon,  # pyrefly: ignore[unbound-name]  # exhaustive match above`) is no longer needed. Remove the comment so it reads:

```python
_recon = (recon,)
```

- [ ] **Step 5: Run the typechecker**

```bash
pixi run -e dev typecheck
```

Expected: exit 0, no new errors. The `unbound-name` warning that we suppressed in PR0 should now genuinely be absent (because the if-raise pattern makes initialization unambiguous to pyrefly).

- [ ] **Step 6: Run the focused test for `Dataset.open`**

```bash
pixi run -e dev pytest tests/dataset/test_dataset.py -v -k "open" 2>&1 | tail -15
```

Expected: all `open`-related tests pass. (If pytest's `-k open` matches nothing, that's fine — proceed; the full suite check in Task 8 will catch regressions.)

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): route Dataset.open construction through factory"
```

---

## Task 3: Simplify `with_seqs` to use the factory

`Dataset.with_seqs` currently contains a ~90-line `match` (around lines 660–751) that hand-builds each combined reconstructor variant. The new shape is: validate the requested transition, update `_seqs` (via `Haps.to_kind` for haps datasets) and `_tracks` if needed, call the factory.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — the `with_seqs` method body

- [ ] **Step 1: Read the current `with_seqs` body**

Run:

```bash
sed -n '630,755p' python/genvarloader/_dataset/_impl.py
```

You should see the full `with_seqs` body, ending around line 751–753. Read it carefully to understand the validation rules:

- `kind = None` + tracks-only → returns tracks-only reconstructor
- `kind = None` + sequences-present → raises (would yield nothing)
- `kind = "reference"` → requires reference genome; sets `_seqs = Ref(...)`
- `kind = "haplotypes"` / `"annotated"` / `"variants"` → requires `Haps`; calls `haps.to_kind(...)` with the matching Ragged type
- Errors for invalid combinations

- [ ] **Step 2: Identify the variable that holds the user's `kind` arg and the current `_seqs`**

The method signature should look like:

```python
def with_seqs(
    self,
    kind: Literal["haplotypes", "reference", "annotated", "variants"] | None = None,
):
```

The current state is `self._seqs` and `self._tracks`; the current `_recon` is `self._recon`.

- [ ] **Step 3: Replace the match body with a smaller validate-then-build flow**

Replace the entire match block (the `match k, seqs, tracks, recon:` block from approx. lines 658–751) with:

```python
seqs = self._seqs
tracks = self._tracks

# Validate the requested kind against the available sources.
if kind is None:
    if tracks is None:
        raise RuntimeError(
            "Dataset is set to only return sequences, so setting"
            " sequence_type to None would result in a Dataset that"
            " cannot return anything."
        )
    # Drop the seqs entirely — return tracks-only.
    new_seqs: Haps | Ref | None = None
elif kind == "reference":
    if not isinstance(seqs, (Haps, Ref)):
        raise ValueError("Dataset has no reference to yield sequences from.")
    ref = seqs.reference
    if ref is None:
        raise ValueError(
            "Dataset has no reference genome to reconstruct sequences from."
        )
    new_seqs = Ref(reference=ref)
else:
    # "haplotypes" | "annotated" | "variants"
    if not isinstance(seqs, Haps):
        raise ValueError("Dataset has no genotypes to yield haplotypes/variants from.")
    kind_type = {
        "haplotypes": RaggedSeqs,
        "annotated": RaggedAnnotatedHaps,
        "variants": RaggedVariants,
    }[kind]
    new_seqs = seqs.to_kind(kind_type)

new_recon = _build_reconstructor(new_seqs, tracks)
return evolve(self, _seqs=new_seqs, _recon=new_recon)
```

Three things to watch for in this replacement:

1. **The variable names for the current state.** If the existing code uses different local names (`s` instead of `seqs`, etc.) for the match's captured pattern variables, the replacement still works because we read directly from `self._seqs` / `self._tracks`. The match-captured variables can go.
2. **Output kind → Ragged-type mapping.** Verify `RaggedSeqs`, `RaggedAnnotatedHaps`, `RaggedVariants` are already imported at the top of `_impl.py`. If not, add them: `from ._reconstruct import ..., RaggedAnnotatedHaps, RaggedSeqs, RaggedVariants` (or wherever they're currently defined — grep first).
3. **`evolve` import.** `evolve` is from `attrs`; should already be imported.

- [ ] **Step 4: Verify imports**

```bash
grep -n "RaggedSeqs\|RaggedAnnotatedHaps\|RaggedVariants" python/genvarloader/_dataset/_impl.py | head -5
```

If any of those names aren't imported in `_impl.py`, add them. Locate the right module (likely `._ragged` or `._reconstruct`):

```bash
grep -rn "^class RaggedSeqs\|^class RaggedAnnotatedHaps\|^class RaggedVariants" python/genvarloader/
```

Adjust the import in `_impl.py` accordingly.

- [ ] **Step 5: Run the typechecker**

```bash
pixi run -e dev typecheck
```

Expected: exit 0, no new errors.

- [ ] **Step 6: Run the focused tests**

```bash
pixi run -e dev pytest tests/dataset/test_dataset.py -v -k "with_seqs or seqs" 2>&1 | tail -25
```

Expected: tests pass. If any fail, stop and investigate — the validation logic may differ subtly from the previous match cases.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): simplify with_seqs to use _build_reconstructor"
```

---

## Task 4: Simplify `with_tracks` to use the factory

`Dataset.with_tracks` currently contains its own combined-class construction (around line 844 — `new_recon = evolve(self._recon, tracks=new_tracks)`).

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — the `with_tracks` method body around lines 753–870

- [ ] **Step 1: Read the current `with_tracks` body**

```bash
sed -n '753,870p' python/genvarloader/_dataset/_impl.py
```

Identify the spot where `_recon` is updated. The pattern is likely:

```python
new_recon = evolve(self._recon, tracks=new_tracks)
return evolve(self, _tracks=new_tracks, _recon=new_recon)
```

Or it may dispatch on `_recon`'s class. Either way, the fix is the same.

- [ ] **Step 2: Replace `_recon` construction with factory call**

Replace the line(s) that build `new_recon` with:

```python
new_recon = _build_reconstructor(self._seqs, new_tracks)
```

And the return:

```python
return evolve(self, _tracks=new_tracks, _recon=new_recon)
```

The body that computes `new_tracks` (track subsetting, etc.) stays unchanged.

- [ ] **Step 3: Run the typechecker**

```bash
pixi run -e dev typecheck
```

Expected: exit 0.

- [ ] **Step 4: Run the focused tests**

```bash
pixi run -e dev pytest tests/dataset/test_dataset.py -v -k "with_tracks or tracks" 2>&1 | tail -25
```

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): simplify with_tracks to use _build_reconstructor"
```

---

## Task 5: Simplify `with_settings` propagation to use the factory

`Dataset.with_settings` lines 429–477 contain a ~50-line isinstance-dispatched propagation block that updates `_recon` whenever `_seqs` changes (because the combined classes contain copies of `_seqs`). Replace with a factory call.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — `with_settings` body, specifically the seqs-propagation block

- [ ] **Step 1: Read the current propagation block**

```bash
sed -n '420,480p' python/genvarloader/_dataset/_impl.py
```

You should see two propagation blocks:

```python
if "_seqs" in to_evolve:
    haps = to_evolve["_seqs"]
    if isinstance(self._recon, Haps):
        recon = haps
        to_evolve["_recon"] = recon
    elif isinstance(self._recon, HapsTracks):
        recon = evolve(self._recon, haps=haps)
        to_evolve["_recon"] = recon
```

And in the `var_filter` branch (around lines 471–477):

```python
if isinstance(self._recon, Haps):
    recon_haps = to_evolve.get("_recon", self._recon)
    to_evolve["_recon"] = evolve(recon_haps, filter=var_filter)
elif isinstance(self._recon, HapsTracks):
    recon = to_evolve.get("_recon", self._recon)
    new_haps = evolve(recon.haps, filter=var_filter)
    to_evolve["_recon"] = evolve(recon, haps=new_haps)
```

- [ ] **Step 2: Delete the first propagation block (lines ~429–436) entirely**

The seqs→recon propagation is no longer needed as a separate block because the factory call at the end of `with_settings` does it uniformly. Delete the block:

```python
if "_seqs" in to_evolve:
    haps = to_evolve["_seqs"]
    if isinstance(self._recon, Haps):
        recon = haps
        to_evolve["_recon"] = recon
    elif isinstance(self._recon, HapsTracks):
        recon = evolve(self._recon, haps=haps)
        to_evolve["_recon"] = recon
```

- [ ] **Step 3: Replace the var_filter propagation block with a factory call**

Replace the `if isinstance(self._recon, Haps): ... elif isinstance(self._recon, HapsTracks): ...` block (around lines 471–477) with:

```python
# `_recon` will be rebuilt below via _build_reconstructor; nothing to do here.
```

(Or just delete those 7 lines entirely.)

- [ ] **Step 4: Add a single factory call at the end of `with_settings`, before `self = evolve(self, **to_evolve)`**

Just before the existing line:

```python
self = evolve(self, **to_evolve)
```

(currently around line 479)

Insert:

```python
# If any source changed, rebuild _recon via the factory.
new_seqs = to_evolve.get("_seqs", self._seqs)
new_tracks = to_evolve.get("_tracks", self._tracks)
if "_seqs" in to_evolve or "_tracks" in to_evolve:
    to_evolve["_recon"] = _build_reconstructor(new_seqs, new_tracks)
```

- [ ] **Step 5: Run the typechecker**

```bash
pixi run -e dev typecheck
```

Expected: exit 0.

- [ ] **Step 6: Run the focused tests**

```bash
pixi run -e dev pytest tests/dataset/test_dataset.py -v -k "with_settings or settings or filter or splice_info" 2>&1 | tail -30
```

Expected: tests pass.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): collapse with_settings _recon propagation via factory"
```

---

## Task 6: Audit remaining `_recon` construction sites

A few other `_recon`-related lines in `_impl.py` may still construct combined classes inline. Find and convert them.

**Files:**
- Modify (potentially): `python/genvarloader/_dataset/_impl.py`

- [ ] **Step 1: Grep for remaining direct constructions**

```bash
grep -n "RefTracks(\|HapsTracks(" python/genvarloader/_dataset/_impl.py
```

For each match, decide:
- If it's a CONSTRUCTION (not a pattern match or isinstance), check if it can be replaced with `_build_reconstructor(...)`.
- If it's a pattern match (`case RefTracks(...)`) or isinstance check, leave it.

A common remaining site is around line 1452 in `_getitem_spliced`:

```python
self._recon, tracks = ds_tracks.with_tracks(r.tracks.active_tracks)
```

This may be `evolve(self._recon, tracks=...)` — i.e. mutating the recon's tracks. Replace with:

```python
new_tracks = ds_tracks.with_tracks(r.tracks.active_tracks)
new_recon = _build_reconstructor(self._seqs, new_tracks)
```

Then use `new_recon` where the old expression was used.

- [ ] **Step 2: Also grep for any remaining `evolve(self._recon, ...)` patterns**

```bash
grep -n "evolve(self._recon" python/genvarloader/_dataset/_impl.py
```

Each should be replaceable with a factory call (or kept as-is if it's only updating non-source attributes like `splice_plan` — those aren't sources, so the factory doesn't apply).

If any remain after this audit that you can't cleanly route through the factory, document them with a one-line comment explaining why (and flag to the reviewer).

- [ ] **Step 3: Run the typechecker**

```bash
pixi run -e dev typecheck
```

Expected: exit 0.

- [ ] **Step 4: Commit (if any changes)**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): route remaining _recon construction through factory"
```

If no changes were needed, skip this commit and move on.

---

## Task 7: Simplify `sequence_type` to derive from `_seqs.kind`

`Dataset.sequence_type` currently matches on `_recon`'s class (lines 1031–1051). With the factory in place, `_recon` is fully derived from `_seqs` and `_tracks`, so we can simplify by inspecting `_seqs` directly. This removes another isinstance-dispatch site.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py:1031-1051` — the `sequence_type` property

- [ ] **Step 1: Read the current property**

```bash
sed -n '1031,1052p' python/genvarloader/_dataset/_impl.py
```

Confirm it looks like:

```python
@property
def sequence_type(
    self,
) -> Literal["haplotypes", "reference", "annotated", "variants"] | None:
    """The type of sequences in the dataset."""
    match self._recon:
        case Tracks():
            return
        case (Haps() as haps) | HapsTracks(haps=haps):
            if issubclass(haps.kind, RaggedAnnotatedHaps):
                return "annotated"
            elif issubclass(haps.kind, RaggedVariants):
                return "variants"
            elif issubclass(haps.kind, RaggedSeqs):
                return "haplotypes"
            else:
                assert_never(haps.kind)
        case Ref() | RefTracks():
            return "reference"
        case r:
            assert_never(r)
```

- [ ] **Step 2: Replace with a `_seqs`-driven implementation**

```python
@property
def sequence_type(
    self,
) -> Literal["haplotypes", "reference", "annotated", "variants"] | None:
    """The type of sequences in the dataset."""
    match self._seqs:
        case None:
            return None
        case Ref():
            return "reference"
        case Haps() as haps:
            if issubclass(haps.kind, RaggedAnnotatedHaps):
                return "annotated"
            elif issubclass(haps.kind, RaggedVariants):
                return "variants"
            elif issubclass(haps.kind, RaggedSeqs):
                return "haplotypes"
            else:
                assert_never(haps.kind)
        case s:
            assert_never(s)
```

- [ ] **Step 3: Run the focused tests**

```bash
pixi run -e dev pytest tests/dataset/ -v -k "sequence_type" 2>&1 | tail -15
```

If no specific `sequence_type` tests exist, run a broader subset:

```bash
pixi run -e dev pytest tests/dataset/test_dataset.py -v -x 2>&1 | tail -30
```

- [ ] **Step 4: pyrefly**

```bash
pixi run -e dev typecheck
```

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): derive sequence_type from _seqs.kind directly"
```

---

## Task 8: Full test suite parity check

After all the routing changes, the canonical parity check is the full test suite.

- [ ] **Step 1: Run the full suite**

```bash
pixi run -e dev test 2>&1 | tail -20
```

Expected: all tests pass (the same 475 passed / 7 skipped / 2 xfailed baseline as after PR0, plus the 6 new factory tests). Total: 481+ passed.

If anything fails:
- If it's in `tests/dataset/`, the routing in Tasks 2-7 broke a semantic somewhere. Investigate via `git bisect` across the branch's commits.
- If it's elsewhere (e.g. tracks tests, splice tests), check whether the failure is reachable from `with_seqs` / `with_tracks` / `with_settings` paths.

DO NOT mark the task complete unless all tests pass.

- [ ] **Step 2: Run pyrefly final check**

```bash
pixi run -e dev prek run --all-files
```

Expected: all hooks pass.

---

## Task 9: Final verification & PR prep

- [ ] **Step 1: Confirm all gates green**

```bash
pixi run -e dev prek run --all-files
pixi run -e dev test
```

Both exit 0.

- [ ] **Step 2: Review the branch's diff against main**

```bash
git log --oneline main..HEAD
git diff --stat main..HEAD
```

Expected commits (in order):

1. `docs(refactor-campaign): re-sequence PR1 as reconstructor factory`
2. `feat(reconstruct): add _build_reconstructor factory`
3. `refactor(impl): route Dataset.open construction through factory`
4. `refactor(impl): simplify with_seqs to use _build_reconstructor`
5. `refactor(impl): simplify with_tracks to use _build_reconstructor`
6. `refactor(impl): collapse with_settings _recon propagation via factory`
7. (Optional) `refactor(impl): route remaining _recon construction through factory`
8. `refactor(impl): derive sequence_type from _seqs.kind directly`

Verify the diff shows: ~30 lines added in `_reconstruct.py` (the factory), ~80–120 lines removed and ~30–50 added in `_impl.py` (net negative — the simplifications outweigh the new code).

If the diff is much larger than expected, stop and report — something extra got dragged in.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin refactor/pr1-build-reconstructor
gh pr create --title "refactor(impl): centralize reconstructor construction via factory (PR1)" --base main --head refactor/pr1-build-reconstructor --body "$(cat <<'EOF'
## Summary

PR1 of the refactor campaign. Centralizes construction of the 5 reconstructor classes via a single `_build_reconstructor(seqs, tracks)` factory in `_reconstruct.py`. All scattered construction sites in `_impl.py` (Dataset.open, with_seqs, with_tracks, with_settings) route through it. The propagation isinstance-ladder in with_settings collapses to a one-liner; the 90-line match in with_seqs becomes ~20 lines of validation + factory call; `sequence_type` simplifies to a lookup on `_seqs.kind`.

The 5 reconstructor classes are **kept** — they already form a valid ADT (invalid (None, None) is not representable in their union), and `RefTracks` / `HapsTracks` encode genuinely different combination strategies (naive composition vs indel-aware joint reconstruction) that don't belong in a single combined class.

## Campaign re-sequence

The first commit on this branch updates the campaign spec at `docs/superpowers/specs/2026-05-23-refactor-campaign-design.md`:
- Original PR1 ("DatasetSettings value object") didn't fit the code — only 4 of 9 args are settings; the others mutate sources. Reframed as factory-centralization.
- Original PR5 ("Pipeline composition") is no longer needed — the existing classes are the ADT we wanted.
- Subsequent PRs renumber: PR5 = ReconstructionRequest + haplotype-kernel extraction, PR6 = file splits, PR7 = naming + types.

## Test plan

- [ ] CI: Lint workflow (ruff + pyrefly via prek-action) passes
- [ ] CI: Test workflow passes across py310/py311/py312/py313
- [ ] `pixi run -e dev test` exits 0 locally (475+ tests + 6 new factory tests)
- [ ] `pixi run -e dev typecheck` exits 0
- [ ] New unit tests in `tests/dataset/test_build_reconstructor.py` cover the 5 routing paths + the invariant violation

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Report PR URL**

Capture the URL printed by `gh pr create` and pass it back to the controller for monitoring.

---

## Self-Review Checklist (original — superseded; see revision below)

The checklist above applies to the original PR1 scope. See the revised
self-review at the end of the "PLAN REVISION" section.

---

# PLAN REVISION — View-State Factoring

> This section supersedes Tasks 3–9 above. Tasks 1 and 2 (factory + `Dataset.open`
> routing) remain in effect and are already committed on this branch.

## Why the revision

The original plan assumed `(seqs, tracks)` storage state was sufficient to
determine which reconstructor to build. Investigation of `with_seqs(None)` /
`with_tracks(False)` revealed that `_recon`'s *runtime class* is actually
doing double duty:

1. **Callable** that produces output for `__getitem__`.
2. **Implicit encoding of the user's current view choice** — which sources are
   "active" and (for haps) which output kind.

Specifically: `with_tracks(False)` keeps `_tracks` populated as a `Tracks` object
(only `_tracks.active_tracks` becomes `None`) and signals "user doesn't want
tracks" by setting `_recon` to a sequence-only class. The factory taking
`(seqs, tracks)` storage state alone cannot recover this user-choice
information and would silently undo the user's prior `with_tracks(False)` when
asked to rebuild.

The fix is to make view-state **explicit, typed, first-class state** on
`Dataset` instead of leaving it encoded in `_recon`'s class.

## Revised design

**New `Dataset` field:**

```python
_seqs_kind: Literal["haplotypes", "reference", "annotated", "variants"] | None
```

Initial value at `open()` time: `"haplotypes"` if Haps available, `"reference"`
if only Ref available, `None` if no sequence source. Updated by
`with_seqs(kind)`.

**`_tracks_active` is NOT a new field.** It is derived from existing state:

```python
_tracks_active: bool = (
    self._tracks is not None and self._tracks.active_tracks is not None
)
```

`with_tracks(False)` continues to set `_tracks.active_tracks = None` (the
existing convention). The factory checks both `_tracks is not None` and
`_tracks.active_tracks is not None` to decide whether tracks are user-active.

**Factory signature grows one parameter:**

```python
def _build_reconstructor(
    seqs: Haps | Ref | None,
    tracks: Tracks | None,
    seqs_kind: Literal["haplotypes", "reference", "annotated", "variants"] | None,
) -> Reconstructor: ...
```

**Factory body:**

1. Resolve the user-active seqs from storage + kind:
   - `seqs_kind is None` → `active_seqs = None`
   - `seqs_kind == "reference"` and Haps stored → `active_seqs = Ref(reference=seqs.reference)` (cast Haps to Ref view)
   - `seqs_kind == "reference"` and Ref stored → `active_seqs = seqs`
   - `seqs_kind in {"haplotypes", "annotated", "variants"}` and Haps stored → `active_seqs = seqs.to_kind(<matching Ragged type>)`
   - Otherwise → `ValueError` (invalid kind for storage)
2. Resolve user-active tracks from `tracks.active_tracks`:
   - `tracks is None or tracks.active_tracks is None` → `active_tracks = None`
   - else → `active_tracks = tracks`
3. Dispatch on `(active_seqs, active_tracks)`:
   - both None → `ValueError("at least one")`
   - only seqs → `active_seqs`
   - only tracks → `active_tracks`
   - Ref + Tracks → `RefTracks(seqs=active_seqs, tracks=active_tracks)`
   - Haps + Tracks → `HapsTracks(haps=active_seqs, tracks=active_tracks)`

**All `with_*` methods become "update view state, call factory":**

```python
# with_seqs:
def with_seqs(self, kind):
    self._validate_seqs_kind(kind)  # ~10 lines of validation
    return evolve(
        self,
        _seqs_kind=kind,
        _recon=_build_reconstructor(self._seqs, self._tracks, kind),
    )
```

```python
# with_tracks:
def with_tracks(self, tracks=None, kind=None):
    if self._tracks is None:
        logger.warning(...)
        return self
    new_tracks = self._tracks.with_tracks(...).to_kind(...)  # update _tracks
    self._validate_at_least_one_active(self._seqs_kind, new_tracks)
    return evolve(
        self,
        _tracks=new_tracks,
        _recon=_build_reconstructor(self._seqs, new_tracks, self._seqs_kind),
    )
```

```python
# with_settings (the propagation block at the end):
if "_seqs" in to_evolve or "_tracks" in to_evolve:
    new_seqs = to_evolve.get("_seqs", self._seqs)
    new_tracks = to_evolve.get("_tracks", self._tracks)
    to_evolve["_recon"] = _build_reconstructor(new_seqs, new_tracks, self._seqs_kind)
```

```python
# sequence_type:
@property
def sequence_type(self):
    return self._seqs_kind
```

**Tasks 1 and 2 are bridge work** that stays valid: the factory exists with the
simpler signature, and `Dataset.open` already routes through it. Task 3 below
extends the factory's signature and updates the `open` call site.

---

## Task 3 (revised): Extend factory with `seqs_kind` + add `_seqs_kind` field to `Dataset`

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` — extend `_build_reconstructor` signature and body
- Modify: `python/genvarloader/_dataset/_impl.py` — add `_seqs_kind` field to `Dataset`, initialize in `open()`, pass to factory in `open()`
- Modify: `tests/dataset/test_build_reconstructor.py` — update existing tests for the new signature; add tests for the kind-resolution behavior

- [ ] **Step 1: Update factory signature and body**

Replace the existing `_build_reconstructor` in `python/genvarloader/_dataset/_reconstruct.py` with:

```python
def _build_reconstructor(
    seqs: Haps | Ref | None,
    tracks: Tracks | None,
    seqs_kind: Literal["haplotypes", "reference", "annotated", "variants"] | None,
) -> Reconstructor:
    """Construct the reconstructor for the given (storage + view) state.

    The user's view choice is carried in `seqs_kind` (`None` means "user does
    not want sequences"). Track activation is read from `tracks.active_tracks`
    (`None` means "user does not want tracks"). This function maps that
    explicit state to one of the 5 reconstructor classes.

    Invariant: after resolving view state, at least one of (active_seqs,
    active_tracks) must be non-None.
    """
    # Resolve active seqs from storage + view kind.
    active_seqs: Haps | Ref | None
    if seqs_kind is None or seqs is None:
        active_seqs = None
    elif seqs_kind == "reference":
        if isinstance(seqs, Ref):
            active_seqs = seqs
        elif isinstance(seqs, Haps):
            if seqs.reference is None:
                raise ValueError(
                    "Cannot view as 'reference': storage has no reference genome."
                )
            active_seqs = Ref(reference=seqs.reference)
        else:
            assert_never(seqs)
    elif seqs_kind in ("haplotypes", "annotated", "variants"):
        if not isinstance(seqs, Haps):
            raise ValueError(
                f"Cannot view as {seqs_kind!r}: storage has no haplotypes."
            )
        kind_map = {
            "haplotypes": RaggedSeqs,
            "annotated": RaggedAnnotatedHaps,
            "variants": RaggedVariants,
        }
        active_seqs = seqs.to_kind(kind_map[seqs_kind])
    else:
        assert_never(seqs_kind)

    # Resolve active tracks from storage + active_tracks subset.
    active_tracks = (
        tracks if (tracks is not None and tracks.active_tracks is not None) else None
    )

    # Dispatch.
    match active_seqs, active_tracks:
        case None, None:
            raise ValueError(
                "_build_reconstructor requires at least one of (seqs, tracks) "
                "to be active. Got seqs_kind=None and tracks inactive."
            )
        case (Haps() | Ref()) as s, None:
            return s
        case None, Tracks() as t:
            return t
        case Ref() as s, Tracks() as t:
            return RefTracks(seqs=s, tracks=t)
        case Haps() as s, Tracks() as t:
            return HapsTracks(haps=s, tracks=t)
        case _:
            raise AssertionError(
                f"unreachable: active_seqs={type(active_seqs).__name__}, "
                f"active_tracks={type(active_tracks).__name__}"
            )
```

This requires `assert_never` from `typing_extensions` (or `typing` on 3.11+) and `RaggedSeqs`, `RaggedAnnotatedHaps`, `RaggedVariants` to be importable. Check the existing imports at the top of `_reconstruct.py` and add any missing ones.

- [ ] **Step 2: Update existing factory tests**

In `tests/dataset/test_build_reconstructor.py`, every existing call to `_build_reconstructor(seqs, tracks)` needs a third arg. The tests check storage-state routing; the new tests will exercise view-state behavior.

Update the existing 6 tests:

- `test_haps_only_returns_haps` → pass `seqs_kind="haplotypes"`
- `test_ref_only_returns_ref` → pass `seqs_kind="reference"`
- `test_tracks_only_returns_tracks` → pass `seqs_kind=None`
- `test_haps_and_tracks_returns_haps_tracks` → pass `seqs_kind="haplotypes"`
- `test_ref_and_tracks_returns_ref_tracks` → pass `seqs_kind="reference"`
- `test_neither_raises_value_error` → pass `seqs_kind=None` (both inactive)

The mocks need `active_tracks` to be set so that the tracks-active check works. Update `_tracks_mock()`:

```python
def _tracks_mock() -> Tracks:
    m = Mock(spec=Tracks)
    m.active_tracks = ["dummy_track"]  # non-None signals "active"
    return m
```

For tests where tracks should be inactive, set `m.active_tracks = None` explicitly.

Update `_haps_mock()` to expose `.reference` so the "reference" kind test below works:

```python
def _haps_mock(with_reference: bool = True) -> Haps:
    m = Mock(spec=Haps)
    m.reference = Mock() if with_reference else None
    return m
```

Add a `kind_map` accessor: when `to_kind` is called on the haps mock, return the mock itself (or any object — the tests check routing, not the actual kind change):

```python
def _haps_mock(with_reference: bool = True) -> Haps:
    m = Mock(spec=Haps)
    m.reference = Mock() if with_reference else None
    m.to_kind = Mock(return_value=m)  # to_kind returns same mock for routing tests
    return m
```

- [ ] **Step 3: Add new tests for kind-resolution behavior**

Append these tests to `test_build_reconstructor.py`:

```python
def test_haps_with_kind_reference_returns_ref():
    """When user wants reference view but storage is Haps, factory builds Ref."""
    seqs = _haps_mock(with_reference=True)
    result = _build_reconstructor(seqs, None, "reference")
    assert isinstance(result, Ref)


def test_haps_with_kind_reference_no_reference_raises():
    """When storage Haps has no reference genome, kind='reference' raises."""
    seqs = _haps_mock(with_reference=False)
    with pytest.raises(ValueError, match="no reference"):
        _build_reconstructor(seqs, None, "reference")


def test_ref_with_haps_kind_raises():
    """Cannot view Ref-only storage as haplotypes."""
    seqs = _ref_mock()
    with pytest.raises(ValueError, match="no haplotypes"):
        _build_reconstructor(seqs, None, "haplotypes")


def test_seqs_kind_none_with_haps_storage_returns_tracks_only():
    """User said with_seqs(None); tracks are active → tracks-only."""
    seqs = _haps_mock()
    tracks = _tracks_mock()  # active by default
    result = _build_reconstructor(seqs, tracks, None)
    assert result is tracks


def test_tracks_inactive_with_seqs_returns_seqs_only():
    """User said with_tracks(False) — active_tracks=None — and seqs are active."""
    seqs = _haps_mock()
    tracks = Mock(spec=Tracks)
    tracks.active_tracks = None  # user-deactivated
    result = _build_reconstructor(seqs, tracks, "haplotypes")
    # Result should be the haps view (kind-resolved), not RefTracks/HapsTracks.
    assert isinstance(result, Haps)


def test_both_inactive_raises():
    """seqs_kind=None and tracks inactive — meaningless state."""
    seqs = _haps_mock()
    tracks = Mock(spec=Tracks)
    tracks.active_tracks = None
    with pytest.raises(ValueError, match="at least one"):
        _build_reconstructor(seqs, tracks, None)
```

- [ ] **Step 4: Add `_seqs_kind` field to `Dataset`**

Find the `Dataset` class definition in `python/genvarloader/_dataset/_impl.py` (look for `class Dataset` or `@define ... class Dataset`). Identify the existing fields including `_seqs`, `_tracks`, `_recon`. Add a new field:

```python
_seqs_kind: Literal["haplotypes", "reference", "annotated", "variants"] | None
```

Place it after `_tracks` and before `_recon` (or wherever fits the existing ordering). Since `Dataset` uses `attrs.define`, this is a typed class attribute.

- [ ] **Step 5: Initialize `_seqs_kind` in `Dataset.open`**

In `Dataset.open`, after the `seqs` variable is built but before the existing factory call, compute the initial kind:

```python
        # Initial view kind: matches the default class produced for each storage shape.
        if isinstance(seqs, Haps):
            seqs_kind = "haplotypes"
        elif isinstance(seqs, Ref):
            seqs_kind = "reference"
        else:
            seqs_kind = None
```

Then update the factory call to pass it:

```python
if seqs is None and tracks is None:
    raise RuntimeError("Malformed dataset: neither genotypes nor intervals found.")
recon = _build_reconstructor(seqs, tracks, seqs_kind)
```

And update the `RaggedDataset(...)` constructor call further down to set `_seqs_kind=seqs_kind`:

```python
        dataset = RaggedDataset(
            ...,
            _seqs=seqs,
            _tracks=tracks,
            _seqs_kind=seqs_kind,
            _recon=recon,
            ...
        )
```

- [ ] **Step 6: Run tests**

```bash
pixi run -e dev pytest tests/dataset/test_build_reconstructor.py -v
```

Expected: all 12 tests pass (6 original updated + 6 new).

```bash
pixi run -e dev pytest tests/dataset/test_dataset.py -v --tb=short 2>&1 | tail -20
```

Expected: all pass. If any fail because `_seqs_kind` isn't being passed somewhere, fix it in this commit. The full test suite check is Task 8 below.

```bash
pixi run -e dev typecheck
```

Expected: exit 0. If new errors appear (likely a missing `assert_never` import or similar), fix.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_impl.py tests/dataset/test_build_reconstructor.py
git commit -m "feat(reconstruct): promote view-state to explicit _seqs_kind field"
```

---

## Task 4 (revised): Simplify `with_seqs` using view-state

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — the `with_seqs` method (around line 603)

- [ ] **Step 1: Replace the 90-line match with a small validate + factory call**

The current `with_seqs` (lines 603–748) has a `match kind, self._seqs, self._tracks, self._recon:` with ~15 cases. Replace the entire match body (everything from `match kind, ...` to the final `assert_never` block) with:

```python
# Validate the requested kind against storage state.
if kind is None:
    tracks_active = self._tracks is not None and self._tracks.active_tracks is not None
    if not tracks_active:
        raise RuntimeError(
            "Dataset is set to only return sequences, so setting"
            " sequence_type to None would result in a Dataset that"
            " cannot return anything."
        )
elif kind == "reference":
    if not isinstance(self._seqs, (Haps, Ref)):
        raise ValueError("Dataset has no reference to yield sequences from.")
    if self._seqs.reference is None:
        raise ValueError(
            "Dataset has no reference genome to reconstruct sequences from."
        )
elif kind in ("haplotypes", "annotated", "variants"):
    if not isinstance(self._seqs, Haps):
        raise ValueError("Dataset has no genotypes to yield haplotypes/variants from.")
else:
    assert_never(kind)

new_recon = _build_reconstructor(self._seqs, self._tracks, kind)
return evolve(self, _seqs_kind=kind, _recon=new_recon)
```

The validation logic mirrors what the old match cases raised; the construction is delegated to the factory.

Verify the error messages exactly match the originals so any tests asserting on error strings still pass. Look at the originals (the lines you're replacing) to confirm.

- [ ] **Step 2: Verify imports for `Haps`, `Ref` are present**

The validation code uses `isinstance(self._seqs, (Haps, Ref))`. Both classes are already imported at the top of `_impl.py` (per Task 2). No new imports needed.

- [ ] **Step 3: Run tests**

```bash
pixi run -e dev pytest tests/dataset/ -v --tb=short -k "with_seqs or seqs" 2>&1 | tail -30
```

Expected: all pass. If any error message tests fail, align the messages to match the originals.

```bash
pixi run -e dev typecheck
```

- [ ] **Step 4: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): simplify with_seqs via _seqs_kind + factory"
```

---

## Task 5 (revised): Simplify `with_tracks` using view-state

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — the `with_tracks` method (around line 750)

- [ ] **Step 1: Read the current body**

```bash
sed -n '750,815p' python/genvarloader/_dataset/_impl.py
```

The current method has a match on `tracks, self._seqs, self._tracks, self._recon` with ~5 cases. Each case calls `tr.with_tracks(...).to_kind(...)` then constructs a specific reconstructor variant.

- [ ] **Step 2: Replace the match with a smaller flow**

Replace the entire match block (from `match tracks, self._seqs, self._tracks, self._recon:` through the final `assert_never` case) with:

```python
        # Compute the new tracks state (active set + kind).
        if tracks is False:
            # User-deactivate all tracks.
            new_tracks = self._tracks.with_tracks(None)
        elif tracks is None:
            # Keep current active set; only kind may change.
            new_tracks = self._tracks
        elif isinstance(tracks, str):
            new_tracks = self._tracks.with_tracks([tracks])
        else:
            new_tracks = self._tracks.with_tracks(tracks)

        if kind is not None:
            new_tracks = new_tracks.to_kind(_kind)

        # Validate: at least one of (seqs, tracks) must remain active.
        seqs_active = self._seqs_kind is not None and self._seqs is not None
        tracks_active = new_tracks.active_tracks is not None
        if not seqs_active and not tracks_active:
            raise RuntimeError(
                "Dataset is set to only return tracks, so setting tracks to None would"
                " result in a Dataset that cannot return anything."
            )

        new_recon = _build_reconstructor(self._seqs, new_tracks, self._seqs_kind)
        return evolve(self, _tracks=new_tracks, _recon=new_recon)
```

This relies on `_kind` already being set above (the existing code computes it before the match — keep that part). Verify by reading the code preceding the match — it should look like:

```python
        if kind == "tracks":
            _kind = RaggedTracks
        elif kind == "intervals":
            _kind = RaggedIntervals
        elif kind is None:
            _kind = self._tracks.kind
        else:
            assert_never(kind)
```

Keep this block; it precedes the replacement.

Also keep the early-return guard at the top of `with_tracks`:

```python
        if self._tracks is None:
            logger.warning("Dataset has no tracks, so this method has no effect.")
            return self
```

- [ ] **Step 3: Tests**

```bash
pixi run -e dev pytest tests/dataset/ -v --tb=short -k "with_tracks or tracks" 2>&1 | tail -30
pixi run -e dev typecheck
```

- [ ] **Step 4: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): simplify with_tracks via factory + active_tracks check"
```

---

## Task 6 (revised): Collapse `with_settings` propagation

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py` — the `with_settings` method (around line 327)

- [ ] **Step 1: Delete the seqs→recon propagation block (lines ~429–436)**

Find and delete the entire `if "_seqs" in to_evolve:` block that does isinstance dispatch:

```python
        if "_seqs" in to_evolve:
            haps = to_evolve["_seqs"]
            if isinstance(self._recon, Haps):
                recon = haps
                to_evolve["_recon"] = recon
            elif isinstance(self._recon, HapsTracks):
                recon = evolve(self._recon, haps=haps)
                to_evolve["_recon"] = recon
```

- [ ] **Step 2: Delete the var_filter propagation isinstance block**

In the `var_filter` branch (around lines ~471–477), find:

```python
                if isinstance(self._recon, Haps):
                    recon_haps = to_evolve.get("_recon", self._recon)
                    to_evolve["_recon"] = evolve(recon_haps, filter=var_filter)
                elif isinstance(self._recon, HapsTracks):
                    recon = to_evolve.get("_recon", self._recon)
                    new_haps = evolve(recon.haps, filter=var_filter)
                    to_evolve["_recon"] = evolve(recon, haps=new_haps)
```

Delete this block entirely (the factory rebuild at the end handles it).

- [ ] **Step 3: Add a single factory call just before `self = evolve(self, **to_evolve)`**

Insert:

```python
# If any source state changed, rebuild _recon via the factory.
if "_seqs" in to_evolve or "_tracks" in to_evolve:
    new_seqs = to_evolve.get("_seqs", self._seqs)
    new_tracks = to_evolve.get("_tracks", self._tracks)
    to_evolve["_recon"] = _build_reconstructor(new_seqs, new_tracks, self._seqs_kind)
```

- [ ] **Step 4: Tests**

```bash
pixi run -e dev pytest tests/dataset/ -v --tb=short -k "with_settings or settings or filter or splice_info" 2>&1 | tail -30
pixi run -e dev typecheck
```

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): collapse with_settings _recon propagation via factory"
```

---

## Task 7 (revised): Simplify `sequence_type` to a field lookup

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py:1029-1052` — the `sequence_type` property

- [ ] **Step 1: Replace the property body**

Find the property (around line 1029):

```python
    @property
    def sequence_type(self) -> Literal[...] | None:
        """The type of sequences in the dataset."""
        match self._recon:
            case Tracks():
                return
            case (Haps() as haps) | HapsTracks(haps=haps):
                ...
            ...
```

Replace with:

```python
    @property
    def sequence_type(
        self,
    ) -> Literal["haplotypes", "reference", "annotated", "variants"] | None:
        """The type of sequences in the dataset."""
        return self._seqs_kind
```

- [ ] **Step 2: Tests + typecheck**

```bash
pixi run -e dev pytest tests/dataset/ -v --tb=short -k "sequence_type" 2>&1 | tail -15
pixi run -e dev typecheck
```

- [ ] **Step 3: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): sequence_type returns the _seqs_kind field directly"
```

---

## Task 8 (revised): Audit remaining `_recon` construction / inspection

After Tasks 3–7, a few residual `_recon`-dependent sites may remain. Find them.

**Files:**
- Modify (potentially): `python/genvarloader/_dataset/_impl.py`

- [ ] **Step 1: Grep for direct construction**

```bash
grep -n "RefTracks(\|HapsTracks(" python/genvarloader/_dataset/_impl.py
```

Each CONSTRUCTION (not pattern match) is a candidate for replacement. Likely `with_insertion_fill` (around line 840) still does `evolve(self._recon, tracks=new_tracks)`. Replace with:

```python
        new_recon = _build_reconstructor(self._seqs, new_tracks, self._seqs_kind)
        return evolve(self, _tracks=new_tracks, _recon=new_recon)
```

- [ ] **Step 2: Grep for `isinstance(self._recon, ...)` checks**

```bash
grep -n "isinstance(self._recon" python/genvarloader/_dataset/_impl.py
```

For each: ask whether it's checking view state. If yes, replace with a check on `self._seqs_kind` and/or `self._tracks.active_tracks`. Note them in the commit message if you change behavior.

Common case: `with_insertion_fill` has `if not isinstance(self._recon, HapsTracks):` — this checks "is the active view haps+tracks?" Replace with:

```python
if (
    self._seqs_kind not in ("haplotypes", "annotated", "variants")
    or self._tracks is None
    or self._tracks.active_tracks is None
):
    raise ValueError(
        "with_insertion_fill is only meaningful for datasets currently "
        "viewing both haplotypes and tracks."
    )
```

- [ ] **Step 3: Grep for `evolve(self._recon, ...)`**

```bash
grep -n "evolve(self._recon" python/genvarloader/_dataset/_impl.py
```

Each remaining occurrence should either become a factory call or be left only if it's purely updating non-source fields (e.g., a splice plan parameter). Document any that remain with a one-line comment.

- [ ] **Step 4: Tests + typecheck**

```bash
pixi run -e dev pytest tests/dataset/ -v --tb=short 2>&1 | tail -20
pixi run -e dev typecheck
```

- [ ] **Step 5: Commit (if changes)**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(impl): route remaining _recon construction/checks through view-state"
```

If no changes needed, skip.

---

## Task 9 (revised): Full test parity check

- [ ] **Step 1: Full test suite**

```bash
pixi run -e dev test 2>&1 | tail -20
```

Expected: ≥ 487 tests pass (475 original + 12 factory tests), 7 skipped, 2 xfailed.

If anything fails: investigate immediately. Common causes:
- An error message string differs from the original → align.
- A test asserts on `_recon`'s type that no longer matches → was the test depending on an implementation detail? Update the test, OR if the test was checking USER-visible behavior, fix the refactor.

- [ ] **Step 2: `pyrefly` clean**

```bash
pixi run -e dev prek run --all-files
```

Expected: all hooks pass.

---

## Task 10 (revised): PR + monitor + merge

- [ ] **Step 1: Verify branch**

```bash
git log --oneline main..HEAD
```

Expected commits (rough order):
1. `docs(refactor-campaign): re-sequence + plan for PR1 (reconstructor factory)`
2. `feat(reconstruct): add _build_reconstructor factory`
3. `refactor(impl): route Dataset.open construction through factory`
4. `feat(reconstruct): promote view-state to explicit _seqs_kind field`
5. `refactor(impl): simplify with_seqs via _seqs_kind + factory`
6. `refactor(impl): simplify with_tracks via factory + active_tracks check`
7. `refactor(impl): collapse with_settings _recon propagation via factory`
8. `refactor(impl): sequence_type returns the _seqs_kind field directly`
9. (Optional) `refactor(impl): route remaining _recon construction/checks through view-state`

- [ ] **Step 2: Push**

```bash
git push -u origin refactor/pr1-build-reconstructor
```

- [ ] **Step 3: Open PR**

```bash
gh pr create --title "refactor: promote reconstructor view-state to first-class field (PR1)" --base main --head refactor/pr1-build-reconstructor --body "$(cat <<'EOF'
## Summary

PR1 of the refactor campaign. Promotes user view-state (currently encoded in `_recon`'s runtime class) to an explicit `_seqs_kind` field on `Dataset`, and centralizes reconstructor construction in a single `_build_reconstructor(seqs, tracks, seqs_kind)` factory.

### What changes

- **New field**: `Dataset._seqs_kind: Literal["haplotypes", "reference", "annotated", "variants"] | None` — the user's current sequence-view choice, set by `with_seqs(kind)`.
- **Track-active state** continues to live on `Tracks.active_tracks` (no new field needed).
- **Factory** in `_reconstruct.py` is the single source of truth for "given storage + view, which reconstructor class?" Replaces ~150 lines of scattered isinstance dispatch across `Dataset.open`, `with_seqs`, `with_tracks`, `with_settings`, and `with_insertion_fill`.
- **`sequence_type`** is now a one-line field lookup instead of a match on `_recon`.

### Why

`_recon`'s runtime class was doing double duty as (1) the output callable and (2) implicit storage of the user's view choice. This created the propagation ugliness in `with_settings` (isinstance ladder to know "is the user in seqs-only mode?") and the 90-line match in `with_seqs`. Promoting view-state to a typed field cleans this up.

### What's preserved

- The 5 reconstructor classes (`Haps`, `Ref`, `Tracks`, `RefTracks`, `HapsTracks`) — they encode genuinely different combination strategies (esp. `HapsTracks`'s indel-aware joint reconstruction).
- All user-facing behavior. Same error messages, same outputs.

### Campaign re-sequence

Spec at `docs/superpowers/specs/2026-05-23-refactor-campaign-design.md` updated:
- Old PR1 ("DatasetSettings value object") didn't fit the code; reframed as this work.
- Old PR5 ("Pipeline composition") is unnecessary; subsequent PRs renumber.

## Test plan

- [ ] CI: Lint workflow passes
- [ ] CI: Test workflow passes across py310/py311/py312/py313
- [ ] `pixi run -e dev test` exits 0 locally (487+ tests)
- [ ] `pixi run -e dev typecheck` exits 0
- [ ] New unit tests in `tests/dataset/test_build_reconstructor.py` cover 12 paths

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Monitor CI; merge when green**

(Controller handles this.)

---

## Self-Review Checklist (revised)

Before declaring PR1 complete:

- [ ] No new `# type: ignore` / `# pyrefly: ignore` suppressions added in source files
- [ ] PR0's `# pyrefly: ignore[unbound-name]` on the old `_impl.py:306` is REMOVED (verified by Task 2)
- [ ] `_build_reconstructor` is the ONLY place that constructs `RefTracks` or `HapsTracks` in non-test code: `grep -rn "RefTracks(\|HapsTracks(" python/genvarloader/_dataset/` — only `_reconstruct.py` and pattern-match cases should appear
- [ ] `sequence_type` is a one-line field return; no `match self._recon` anywhere in `_impl.py`
- [ ] `Dataset._seqs_kind` is a typed attrs field with the expected `Literal[...] | None` annotation
- [ ] `_impl.py` line count dropped by at least 100 lines net (was 2253 after PR0; aim for ~2120 or less)
- [ ] All commits use conventional-commit format
- [ ] Full test suite exits 0
- [ ] `pixi run -e dev typecheck` exits 0
- [ ] User-facing error messages preserved exactly (any test asserting on string content still passes)
