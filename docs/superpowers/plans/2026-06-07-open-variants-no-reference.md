# `Dataset.open` Without Reference Defaults to Variants — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gvl.Dataset.open(path)` on a genotypes-only dataset with no reference resolve to a working `RaggedDataset[RaggedVariants, ...]` instead of raising `ValueError: Cannot return RaggedSeqs: no reference genome was provided.`

**Architecture:** The single decision point for the default sequence view is `OpenRequest._initial_seqs_kind` in `python/genvarloader/_dataset/_open.py`. It currently returns `"haplotypes"` for any `Haps` storage. We make it reference-aware: a `Haps` with no reference defaults to `"variants"` (the only sequence view `Haps.to_kind` permits without a reference). `_build_reconstructor` and `Haps.to_kind` stay unchanged — they are already correct; only the chosen default was impossible.

**Tech Stack:** Python, pytest, pixi (`pixi run -e dev`), maturin (Rust extension auto-built on test run).

---

## Background (read before starting)

The crash, confirmed by repro:

```python
import genvarloader as gvl
gvl.Dataset.open("tests/data/phased_dataset.vcf.gvl")
# ValueError: Cannot return RaggedSeqs: no reference genome was provided.
```

Chain of events:
- `OpenRequest._initial_seqs_kind` (`_open.py:180-186`) returns `"haplotypes"` for any `Haps`.
- `OpenRequest.resolve` (`_open.py:76`) calls `_build_reconstructor(seqs, tracks, "haplotypes")`.
- `_build_reconstructor` (`_reconstruct.py:269-279`) calls `seqs.to_kind(RaggedSeqs)`.
- `Haps.to_kind` (`_haps.py:495-500`) raises because `kind != RaggedVariants and self.reference is None`.

The fix flips the default to `"variants"` when the `Haps` has no reference. `RaggedVariants` is the only kind `to_kind` allows without a reference, so this is the unique correct default. `with_seqs("haplotypes" | "annotated" | "reference")` already validates reference presence later and raises a clear error if missing, so nothing downstream regresses.

**Test fixtures** live in `tests/conftest.py` (session-scoped, available to all subdirectories):
- `phased_vcf_gvl`, `phased_pgen_gvl`, `phased_svar_gvl` — each yields a `Path` to a genotypes-only, reference-less gvl dataset (one per variant source). These are the exact datasets that trigger the bug.

**Return-shape contract:** For a seqs-only dataset with no active tracks, `RaggedDataset.__getitem__` returns the sequence object directly (the `RaggedDataset[RSEQ, None]` overload returns `RSEQ`), so `ds[region_idx, sample_idx]` yields a `RaggedVariants` — not a tuple. The phased fixtures have no tracks.

---

## File Structure

- **Modify:** `python/genvarloader/_dataset/_open.py` — `_initial_seqs_kind` (lines 179-186). One logic change.
- **Create:** `tests/unit/dataset/test_open_no_reference.py` — regression tests covering all three variant sources plus an end-to-end indexing assertion.

---

## Task 1: Genotypes-only dataset opens as variants and indexes to RaggedVariants

**Files:**
- Create: `tests/unit/dataset/test_open_no_reference.py`
- Modify: `python/genvarloader/_dataset/_open.py:179-186`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/test_open_no_reference.py` with exactly this content:

```python
"""Regression: opening a genotypes-only dataset without a reference must
default to the 'variants' view (RaggedVariants), not crash trying to build
haplotypes. See docs/superpowers/specs/2026-06-07-open-variants-no-reference-design.md.
"""

from __future__ import annotations

import pytest

import genvarloader as gvl


@pytest.mark.parametrize(
    "fixture_name", ["phased_vcf_gvl", "phased_pgen_gvl", "phased_svar_gvl"]
)
def test_open_without_reference_defaults_to_variants(fixture_name, request):
    path = request.getfixturevalue(fixture_name)
    ds = gvl.Dataset.open(path)
    assert ds.sequence_type == "variants"


def test_open_without_reference_indexing_yields_variants(phased_vcf_gvl):
    ds = gvl.Dataset.open(phased_vcf_gvl)
    out = ds[0, 0]
    assert isinstance(out, gvl.RaggedVariants)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_open_no_reference.py -v`

Expected: all cases FAIL with `ValueError: Cannot return RaggedSeqs: no reference genome was provided.` (raised from `Haps.to_kind` during `Dataset.open`).

- [ ] **Step 3: Apply the fix**

In `python/genvarloader/_dataset/_open.py`, replace the `_initial_seqs_kind` method (lines 179-186):

```python
    @staticmethod
    def _initial_seqs_kind(seqs: Haps | Ref | None) -> SeqsKind:
        # Default view kind for each storage shape.
        if isinstance(seqs, Haps):
            return "haplotypes"
        if isinstance(seqs, Ref):
            return "reference"
        return None
```

with:

```python
    @staticmethod
    def _initial_seqs_kind(seqs: Haps | Ref | None) -> SeqsKind:
        # Default view kind for each storage shape.
        if isinstance(seqs, Haps):
            # Without a reference we can't reconstruct haplotypes; the only
            # sequence view Haps.to_kind allows is RaggedVariants.
            return "haplotypes" if seqs.reference is not None else "variants"
        if isinstance(seqs, Ref):
            return "reference"
        return None
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_open_no_reference.py -v`

Expected: all 4 cases PASS.

- [ ] **Step 5: Run the broader dataset suite to confirm no regression**

Run: `pixi run -e dev pytest tests/unit/dataset tests/dataset -q`

Expected: PASS (no new failures). This confirms datasets that *do* have a reference still default to `"haplotypes"`/`"reference"` as before.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_open.py tests/unit/dataset/test_open_no_reference.py
rtk git commit -m "fix(open): default to variants when genotypes have no reference

Dataset.open on a genotypes-only dataset with no reference crashed in
_initial_seqs_kind, which always picked 'haplotypes' and then failed in
Haps.to_kind(RaggedSeqs). Default to 'variants' when reference is None so
the dataset resolves to RaggedDataset[RaggedVariants, ...].

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Fix in `_initial_seqs_kind` → Task 1 Step 3. ✓
- `variants` is the unique correct default without a reference → covered; `Haps.to_kind` only permits `RaggedVariants` without a reference. ✓
- Genotypes + tracks → `RaggedDataset[RaggedVariants, RaggedTracks]` with tracks active → no code change needed (the factory already builds `HapsTracks` for `seqs_kind="variants"`, a state reachable today via `with_seqs("variants")`); the phased fixtures have no tracks so the indexing test asserts the no-tracks shape (direct `RaggedVariants`). The combined state is exercised by existing `with_seqs("variants")` tests, so no new task is required.
- All three fixtures (vcf/pgen/svar) + indexing → Task 1 Steps 1-2 (parametrized open across all three; indexing on vcf). ✓
- TDD (failing test first) → Task 1 Steps 1-2 before Step 3. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/vague steps. Every code step shows full code. ✓

**Type consistency:** `_initial_seqs_kind` signature and `SeqsKind` return type unchanged; `"variants"` is a valid member of `SeqsKind` (`_open.py:36`). `gvl.RaggedVariants` is exported (`__init__.py:45`). `ds.sequence_type` is a public property (`_impl.py:835`). ✓
