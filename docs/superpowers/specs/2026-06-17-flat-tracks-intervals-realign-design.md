# Flat tracks/intervals + decoupled track re-alignment

**Repo:** `mcvickerlab/GenVarLoader` · **Date:** 2026-06-17
**Status:** design approved, pre-implementation

## Problem

Two related gaps in the `Dataset` output surface:

1. **Flat track/interval output.** `with_output_format("flat")` already yields
   pure-numpy containers for sequences (`FlatRagged` / `FlatVariants` /
   `FlatAnnotatedHaps` / `FlatVariantWindows`). Tracks are inconsistent:
   - Float tracks (`with_tracks(kind="tracks")`) **already** pass through as
     `FlatRagged` in flat mode (the reconstructor builds a `_Flat` internally),
     but this is untested and undocumented.
   - Interval tracks (`with_tracks(kind="intervals")`) **always** return
     awkward `RaggedIntervals`, even in flat mode — there is no flat interval
     type.

2. **Tracks alongside `variant-windows`.** `_build_reconstructor` explicitly
   raises when `seqs_kind == "variant-windows"` and tracks are active. Users
   modeling per-variant token windows (e.g. genvarformer / gvf-germ-som) cannot
   get an accompanying track signal.

## Decisions (from brainstorming)

- **Semantics: "variant mode = data as-is."** Variant-oriented outputs should be
  able to return tracks/intervals *without* re-aligning them to the would-be
  haplotype. Rather than tie this to the seq mode, expose it as an explicit,
  decoupled setting.
- **API: `with_settings(realign_tracks=True)`.** Runtime setting, default
  `True` (preserves today's haplotypes+tracks re-alignment). `with_settings`
  only — **not** added to `Dataset.open` (scope limit).
- **`variant-windows` + tracks requires `realign_tracks=False`** (raise
  otherwise). Re-aligning a track to the would-be haplotype around a variant is
  conceivable but YAGNI for now; the reference-oriented flanks of the window are
  independent of that question.
- **Interval tracks require `realign_tracks=False` for any variant-aware seq
  mode.** Intervals are not re-aligned in any code path today; this just
  codifies it. This is a (minor) **breaking change**: `haplotypes`+`intervals`
  currently returns un-realigned intervals under the default `realign_tracks=True`
  and will now raise unless `realign_tracks=False`.
- **`with_insertion_fill` raises when `realign_tracks=False`** (insertion fill
  only applies during re-alignment, so it would silently no-op otherwise).
- **Flat scope:** add a `FlatIntervals` type for flat interval output; add test
  + doc coverage for the already-working flat tracks.

## Scope of `realign_tracks`

`realign_tracks` only affects `Haps` + float tracks (`kind="tracks"`). For
`reference`+tracks, tracks-only, or interval tracks it is a no-op (those are
already returned as-is / un-realigned).

## Design

### 1. The `realign_tracks` setting

- New `Dataset` field `realign_tracks: bool = True`.
- Settable via `with_settings(realign_tracks: bool | None = None)`.
- `_build_reconstructor` gains a `realign_tracks: bool` parameter. All four call
  sites pass it:
  - `with_seqs` (`_impl.py:717`)
  - `with_tracks` (`_impl.py:769`)
  - `with_insertion_fill` (`_impl.py:802`)
  - `with_settings` (`_impl.py:435`)
- `with_settings` must rebuild `_recon` when `realign_tracks` changes (today it
  rebuilds only when `_seqs`/`_tracks` change — extend that guard).

### 2. Reconstructor dispatch

Rule table for `Haps + Tracks` in `_build_reconstructor`:

| seqs_kind | track kind | realign_tracks | result |
|---|---|---|---|
| haplotypes / annotated / variants | tracks (float) | `True` (default) | `HapsTracks` (unchanged) |
| haplotypes / annotated / variants | tracks (float) | `False` | as-is: seqs + reference-coord tracks |
| variant-windows | tracks (float) | `True` | **raise** (require `realign_tracks=False`) |
| variant-windows | tracks (float) | `False` | as-is: windows + reference-coord tracks |
| any Haps-backed kind | intervals | `True` | **raise** (intervals can't be re-aligned) |
| any Haps-backed kind | intervals | `False` | as-is: seqs + raw intervals |

`reference`+tracks and tracks-only are unchanged (already as-is).

**Mechanism (chosen):** generalize the existing `RefTracks` reconstructor into a
`SeqsTracks` that pairs *any* seq reconstructor (`Ref` or `Haps` in any kind)
with un-realigned `Tracks`, calling `seqs(...)` and `tracks(...)` independently
(exactly what `RefTracks` does today). `HapsTracks` remains solely for the
`realign_tracks=True` float path with its fused haps+realign compute.

- Rename/replace `RefTracks` → `SeqsTracks` (`seqs: Ref | Haps`, `tracks:
  Tracks`). Update its splice dispatch in `_query.py` (`build_recon_splice_plan`)
  — splicing of `SeqsTracks` stays `NotImplementedError`, consistent with
  today's `RefTracks`/`HapsTracks`.
- `_build_reconstructor`:
  - `Ref + Tracks` → `SeqsTracks`
  - `Haps + Tracks`, float, `realign_tracks=True`, kind ≠ variant-windows →
    `HapsTracks`
  - `Haps + Tracks`, float, `realign_tracks=False` → `SeqsTracks`
  - `Haps + Tracks`, intervals → require `realign_tracks=False` → `SeqsTracks`
  - variant-windows + tracks → require `realign_tracks=False` → `SeqsTracks`

*Alternative considered:* a separate `HapsRawTracks` class instead of
generalizing `RefTracks` (smaller blast radius on splice dispatch, more
duplicated code). Rejected in favor of the single `SeqsTracks`.

**`with_insertion_fill`** raises a clear error when `realign_tracks=False`
(no effect without re-alignment).

### 3. `FlatIntervals` type

Mirror `RaggedIntervals` (`_ragged.py:33`) in flat form:

- Fields: `starts`, `ends`, `values` (flat numpy arrays) + shared `offsets`
  (int64) + `shape` (outer fixed dims with one trailing `None`).
- Methods: `.to_ragged()` → `RaggedIntervals`; `.reshape(...)`; `.squeeze(...)`.
- Exported as `gvl.FlatIntervals` in `python/genvarloader/__init__.py` `__all__`.
- `Tracks._call_intervals(idx, flat=False)` gains a flat branch that builds
  `FlatIntervals` via a pure-numpy offset gather over the stored
  (memmap-backed) interval arrays, instead of `ak.concatenate` /
  `ak`-indexing. Shape `(b, t, ~itvs)` matching `RaggedIntervals`.
- `Tracks.__call__` passes `flat` through to `_call_intervals`.
- `_query.py getitem`:
  - flat mode: pass `FlatIntervals` through; include it in the reshape/squeeze
    handling for flat types; reverse-complement is a no-op for intervals (same
    as `RaggedIntervals`).
  - ragged mode: convert via `FlatIntervals.to_ragged()`.

### 4. Testing & docs

Tests:
- Flat tracks-only and flat haps+tracks return `FlatRagged` (currently untested).
- Flat intervals round-trip: `FlatIntervals.to_ragged()` element-identical to
  the `RaggedIntervals` from ragged mode.
- `realign_tracks=False` float-track values equal the reference-coord
  (un-realigned) track values.
- variant-windows + tracks (float → `FlatRagged`; intervals → `FlatIntervals`),
  in the required flat output format.
- Raise paths: variant-windows + tracks with `realign_tracks=True`;
  intervals + variant-aware seq with `realign_tracks=True`;
  `with_insertion_fill` with `realign_tracks=False`.

Docs:
- `skills/genvarloader/SKILL.md` (required by CLAUDE.md): document the
  `realign_tracks` setting, `FlatIntervals` (flat table + one-liner),
  variant-windows + tracks, the intervals/realign rules, and the
  `haplotypes`+`intervals` breaking change.
- `docs/source/dataset.md`: track output modes / re-alignment.
- `docs/source/changelog.md`: feat entries + breaking-change note.

## Out of scope (YAGNI)

- Re-aligning track *float* values to the would-be haplotype for
  `variant-windows` (conceptually possible; deferred).
- Re-aligning interval tracks to haplotype coordinates (any mode).
- `realign_tracks` on `Dataset.open` (only `with_settings`).
- Splicing of `SeqsTracks`.
