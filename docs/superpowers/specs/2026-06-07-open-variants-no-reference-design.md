# `Dataset.open` without a reference should default to variants

**Date:** 2026-06-07
**Status:** Approved (design)

## Problem

`gvl.Dataset.open(path)` on a dataset that has genotypes but no reference genome
crashes during open:

```python
import genvarloader as gvl
gvl.Dataset.open("tests/data/phased_dataset.vcf.gvl")
# ValueError: Cannot return RaggedSeqs: no reference genome was provided.
```

A genotypes-only, no-reference dataset cannot reconstruct haplotypes or
reference-overlaid sequences. The only sequence view it can serve is
`RaggedVariants`. So `open` should resolve such a dataset to a working
`RaggedDataset[RaggedVariants, ...]` rather than erroring.

The bug is visible at open time and is internally contradictory: `_build_seqs`
already logs a warning that the dataset "can only support `.with_seqs('variants')`",
then `resolve()` immediately tries to build the haplotypes reconstructor anyway.

## Root cause

Two cooperating pieces in `python/genvarloader/_dataset/`:

1. **`OpenRequest._initial_seqs_kind`** (`_open.py:180-186`) returns
   `"haplotypes"` for *any* `Haps` storage, regardless of whether a reference
   is present.
2. **`OpenRequest.resolve`** (`_open.py:75-76`) then calls
   `_build_reconstructor(seqs, tracks, "haplotypes")`, which calls
   `seqs.to_kind(RaggedSeqs)`. `Haps.to_kind` (`_haps.py:495-500`) raises
   `ValueError` for any non-`RaggedVariants` kind when `self.reference is None`.

The reconstructor factory and `to_kind` are behaving correctly — the defect is
that `open` picks an impossible default view kind.

## Fix

Make the default view kind reference-aware. This is a single decision point.

In `python/genvarloader/_dataset/_open.py`, change `_initial_seqs_kind`:

```python
@staticmethod
def _initial_seqs_kind(seqs: Haps | Ref | None) -> SeqsKind:
    # Default view kind for each storage shape.
    if isinstance(seqs, Haps):
        # Without a reference we can't reconstruct haplotypes; the only
        # sequence view available is RaggedVariants.
        return "haplotypes" if seqs.reference is not None else "variants"
    if isinstance(seqs, Ref):
        return "reference"
    return None
```

No other code changes. `_build_reconstructor` stays a pure mapper of
explicit state → reconstructor class (correct layering). The `_build_seqs`
warning text remains accurate.

### Why this is sufficient

- `seqs_kind == "variants"` routes through the `("haplotypes", "annotated",
  "variants")` branch of `_build_reconstructor`, which calls
  `seqs.to_kind(RaggedVariants)` — explicitly allowed without a reference.
- With tracks present, the factory builds `HapsTracks(haps=Haps[RaggedVariants],
  tracks)`. This is a state already reachable today via
  `with_seqs("variants")`, so no new combined-state handling is needed.
- `_check_valid_state` only restricts `variants` for `output_length ==
  "variable"`; `open` uses `output_length == "ragged"`, so the default is valid.
- Users can still call `.with_seqs("haplotypes" | "annotated" | "reference")`
  later; those paths already validate reference presence and raise a clear
  error if it's missing.

## Behavior decisions

- **Genotypes + tracks, no reference:** default to `variants` view with tracks
  active (tracks keep their current auto-activation). Yields `RaggedVariants`
  alongside the active tracks, i.e. `RaggedDataset[RaggedVariants,
  RaggedTracks]`. Tracks default to per-nucleotide `RaggedTracks`
  (`Tracks.from_path` defaults `kind=RaggedTracks`), not `RaggedIntervals`;
  `RaggedTracks` re-aligns to haplotype coordinates using variant indel lengths
  from the genotypes, not the reference sequence, so it works without a
  reference.

## Testing (TDD)

Write the failing test first, confirm it reproduces the `ValueError`, then apply
the fix.

Use the existing reference-less fixtures: `tests/data/phased_dataset.vcf.gvl`,
`phased_dataset.pgen.gvl`, `phased_dataset.svar.gvl`.

1. **Open without reference succeeds** — for each of the three fixtures,
   `gvl.Dataset.open(fixture)` returns a dataset with `sequence_type ==
   "variants"` and does not raise.
2. **Indexing yields RaggedVariants** — `dataset[region_idx, sample_idx]`
   returns `RaggedVariants` end-to-end (at least on the vcf fixture; the three
   share a reconstruction path, so parametrizing open across all three plus one
   indexing assertion is adequate).

Place tests under `tests/unit/dataset/` following existing fixture conventions
in that directory.

## Out of scope

- Changing `Haps.to_kind` validation or `_build_reconstructor` layering.
- Any change to `with_seqs` / `with_settings` behavior.
- Spliced/annotated output for variants (already correctly rejected).
