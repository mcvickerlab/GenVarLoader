# Rust Migration Phase 5 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Rust migration's Phase 5 — fix the remaining numba/rust correctness divergences, fuse the last deferred read path, freeze the numba oracle as golden fixtures, delete numba, add rayon, and merge `rust-migration → main` once a final `__getitem__` benchmark shows rust at parity-or-better.

**Architecture:** Phase 5 is a strict sequential pipeline of distinct PRs into the `rust-migration` integration branch. Correctness fixes (W1, W2) and the fusion (W3) must land **while numba still exists** as the differential oracle; the final numba-vs-rust verdict (W4) must be captured **before** deletion; only then is it safe to golden-snapshot (W5) and delete numba (W6), add rayon (W7), measure RSS (W8), and merge (W9). **This document fully specifies PR1 (W1).** PR2–PR6 (W2–W9) are scoped at the end and each gets its own detailed plan written at its turn — W2 in particular requires a coordinate-math investigation whose root cause is not yet known and therefore cannot be bite-sized in advance.

**Tech Stack:** Rust (ndarray, PyO3, rayon), Python (numpy, numba — being deleted), pixi (`-e dev`), maturin, pytest + hypothesis, cargo test, memray, pytest-benchmark.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-06-26-rust-migration-phase-5-design.md`. Roadmap (source of truth, must be updated): `docs/roadmaps/rust-migration.md` (Phase 5).
- Byte-identical parity is the landing gate for every kernel change; numba is the oracle until W6 deletes it (W5 freezes it to golden fixtures first).
- Benchmark parity verdict is **single-thread**: `NUMBA_NUM_THREADS=1`, rayon threads=1, `maturin develop --release`, corpus `chr22_geuv.gvl` (format 2.0), Carter HPC (AMD EPYC 7543, linux-64). Node is shared/noisy — use within-session ratios + pedantic min; the durable signal is parity + the recorded instruction-count reductions.
- Dataset/parity tests on the HPC need `--basetemp=$(pwd)/.pytest_tmp` (numba write path's `os.link` fails cross-device, Errno 18).
- Numba-oracle-bug policy: a numba-vs-rust divergence where numba is buggy gets an issue + an isolated fix PR + un-exclusion from parity. W1 and W2 follow this.
- Per-kernel rust core lives in `src/`; PyO3 only in `src/ffi/`. No `unsafe` unless justified by a profile.
- Commits: conventional-commit style; no squash on the final merge (preserve history). Co-author trailer on commits:
  `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.

---

## PR1 (W1): Fix the haplotype/track trailing-fill divergence in BOTH kernels

**Why this is "fix both," not "fix numba to match rust":** reading the actual code, *neither* kernel is correct in the overshoot sub-domain (a deletion drives `ref_idx` past the contig end with output still unfilled). The roadmap's "rust is correct" was an assertion about an untested, parity-excluded sub-domain. Concretely, with `ref=[1,2,3,4]`, a deletion at pos 2 with `ilen=-5` (so `v_ref_end = 2+5+1 = 8`), `out_len=8`, `pad_char=0`:

- Correct output: ref consumed `[1,2]`, allele `[50]`, then **ref is exhausted** → pad the entire tail → `[1,2,50,0,0,0,0,0]`.
- Current **numba** (`_genotypes.py:508`): `writable_ref = min(5, 4-8) = -4`, `out_end_idx = 3 + (-4) = -1`; `out[3:-1] = ref[8:4]` is a numpy shape mismatch inside njit → SystemError / unwritten tail (the bug).
- Current **rust** (`src/reconstruct/mod.rs:245`): `out_end_idx = (3 + (-4)).max(0) = 0`; then `out[0..8] = pad` → `[0,0,0,0,0,0,0,0]` — **overwrites the valid prefix** `[1,2,50]`.

**The fix (both kernels):** when `ref` is exhausted (`writable_ref <= 0`), clamp `out_end_idx` to `out_idx` (not 0) so the right-pad fills exactly the unfilled tail `out[out_idx:length]`. In numba this is `writable_ref = max(0, min(unfilled_length, len(ref) - ref_idx))`. The same latent pattern exists in the track-realign kernels (`_tracks.py:396` numba) — apply the identical clamp.

**Files:**
- Modify: `src/reconstruct/mod.rs:208-260` (rust haplotype trailing-fill; the `else` branch at 240-246) + its in-module test block.
- Modify: `python/genvarloader/_dataset/_genotypes.py:508` (numba haplotype singular kernel).
- Modify: `python/genvarloader/_dataset/_tracks.py:396` (numba track singular kernel).
- Verify/Modify: rust track-realign trailing-fill in `src/tracks*` (check for the same `.max(0)` pattern).
- Test (new): `tests/unit/dataset/test_reconstruct_trailing_fill.py` (numba + rust correctness, deterministic).
- Test (new): `src/reconstruct/mod.rs` cargo unit test `overshoot_ref_past_contig`.
- Modify: `tests/parity/test_reconstruct_haplotypes_parity.py` (remove the 3 exclusion guards once the divergence is gone).
- Check: `tests/parity/test_shift_and_realign_tracks_parity.py`, `tests/parity/test_dataset_parity.py`, `tests/parity/strategies.py`, `tests/parity/_fixtures.py` for analogous overshoot/`max_jitter` exclusions tied to this divergence.

**Interfaces:**
- Consumes: `reconstruct_haplotype_from_sparse(v_idxs, v_starts, ilens, shift, alt_alleles, alt_offsets, ref, ref_start, out, pad_char, keep=None, annot_v_idxs=None, annot_ref_pos=None)` — numba singular kernel, `@nb.njit(nogil=True, cache=True)`, directly importable from `genvarloader._dataset._genotypes`.
- Produces: no signature changes. Behavior change only: overshoot inputs now produce full-tail-pad output, byte-identical across numba and rust.

### Task 1: Characterize the rust overshoot bug (cargo, failing test)

**Files:**
- Test: `src/reconstruct/mod.rs` (add to the `#[cfg(test)] mod tests` block, alongside `deletion`/`del_spanning_ref_start`).

- [ ] **Step 1: Write the failing cargo test**

Add next to the existing `run(...)`-helper tests (the helper signature is
`run(v_idxs, v_starts, ilens, shift, alt_alleles, alt_offsets, ref, ref_start, out_len, pad_char, keep, annotate)`):

```rust
// -------------------------------------------------------------------------
// Case: deletion drives ref_idx past the contig end (overshoot).
// ref = [1,2,3,4] (len 4), ref_start=0, out_len=8.
// variant at pos=2, ilen=-5, allele=[50] (anchor).
//   v_ref_end = 2 - min(0,-5) + 1 = 8  → ref_idx advances to 8 (> len 4).
// Processing: ref[0..2]=[1,2], allele=[50] → out_idx=3.
// Final clause: unfilled=5, ref exhausted (writable_ref = min(5, 4-8) = -4 <= 0).
// CORRECT: no ref left → pad the whole tail → [1,2,50,0,0,0,0,0].
// (Pre-fix rust over-pads from index 0 → all zeros.)
// -------------------------------------------------------------------------
#[test]
fn overshoot_ref_past_contig() {
    let (out, _av, _ap) = run(
        &[0],
        &[2],          // v_pos=2
        &[-5],         // ilen=-5 (deletion past contig end)
        0,             // shift
        &[50u8],       // anchor allele
        &[0i64, 1],
        &[1, 2, 3, 4], // ref, len 4
        0,             // ref_start
        8,             // out_len
        0,             // pad_char
        None,
        false,
    );
    assert_eq!(out, vec![1, 2, 50, 0, 0, 0, 0, 0]);
}
```

- [ ] **Step 2: Run the test to verify it FAILS**

Run: `pixi run -e dev cargo test --lib reconstruct::tests::overshoot_ref_past_contig`
Expected: FAIL — actual `[0, 0, 0, 0, 0, 0, 0, 0]` (rust over-pads from index 0).

- [ ] **Step 3: Commit the failing test**

```bash
rtk git add src/reconstruct/mod.rs
rtk git commit -m "test(reconstruct): pin correct full-tail-pad on ref overshoot (failing)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 2: Fix the rust trailing-fill clamp

**Files:**
- Modify: `src/reconstruct/mod.rs:240-246` (the `else` branch) + the stale comments at 211-218.

- [ ] **Step 1: Apply the clamp-to-`out_idx` fix**

Replace the `else` branch (currently `(out_idx + writable_ref).max(0)`) so an exhausted ref pads exactly the unfilled tail:

```rust
        } else {
            // writable_ref <= 0: ref exhausted (ref_idx at/after contig end).
            // No reference bytes remain to copy, so the entire unfilled tail
            // out[out_idx..length] must be padded. Clamp out_end_idx to out_idx
            // (NOT 0) so the right-pad below fills exactly out[out_idx..length]
            // and never overwrites already-written positions.
            out_idx
        };
```

Also fix the now-inaccurate comment block at lines 211-218 (it describes mirroring numpy's negative-index behavior, which was the bug). Replace with a one-line note that the tail is padded when ref is exhausted.

- [ ] **Step 2: Run the cargo test to verify it PASSES**

Run: `pixi run -e dev cargo test --lib reconstruct::tests::overshoot_ref_past_contig`
Expected: PASS — `[1, 2, 50, 0, 0, 0, 0, 0]`.

- [ ] **Step 3: Run the full rust suite (no regressions)**

Run: `pixi run -e dev cargo-test`
Expected: all pass (the existing `deletion`, `del_spanning_ref_start`, etc. are unaffected — they never overshoot).

- [ ] **Step 4: Commit**

```bash
rtk git add src/reconstruct/mod.rs
rtk git commit -m "fix(reconstruct): pad full tail when ref exhausted, not from index 0

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 3: Characterize + fix the numba haplotype/track kernels

**Files:**
- Test: `tests/unit/dataset/test_reconstruct_trailing_fill.py` (new).
- Modify: `python/genvarloader/_dataset/_genotypes.py:508`.
- Modify: `python/genvarloader/_dataset/_tracks.py:396`.

- [ ] **Step 1: Write the failing numba correctness test**

```python
"""Correctness of the trailing-fill clause when a deletion exhausts the contig.

The overshoot sub-domain (ref_idx past contig end with output unfilled) was
historically excluded from parity because numba and rust diverged AND both were
wrong. Correct behavior: pad the entire unfilled tail (no reference left).
"""

import numpy as np

from genvarloader._dataset._genotypes import reconstruct_haplotype_from_sparse


def test_overshoot_pads_full_tail():
    # ref=[1,2,3,4], deletion at pos 2 (ilen=-5) -> ref_idx advances to 8 (>4).
    # out_len=8: [1,2] ref + [50] allele, then ref exhausted -> pad rest with 0.
    out = np.full(8, 255, dtype=np.uint8)  # 0xFF sentinel: catches unwritten positions
    reconstruct_haplotype_from_sparse(
        np.array([0], dtype=np.int32),        # v_idxs
        np.array([2], dtype=np.int32),        # v_starts
        np.array([-5], dtype=np.int32),       # ilens
        0,                                    # shift
        np.array([50], dtype=np.uint8),       # alt_alleles
        np.array([0, 1], dtype=np.int64),     # alt_offsets
        np.array([1, 2, 3, 4], dtype=np.uint8),  # ref
        0,                                    # ref_start
        out,                                  # out
        0,                                    # pad_char
    )
    np.testing.assert_array_equal(out, np.array([1, 2, 50, 0, 0, 0, 0, 0], dtype=np.uint8))
```

- [ ] **Step 2: Run to verify it FAILS**

Run: `pixi run -e dev pytest tests/unit/dataset/test_reconstruct_trailing_fill.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — numba leaves the tail unwritten (0xFF sentinel leaks through) or raises a numpy shape error inside the njit kernel.

- [ ] **Step 3: Apply the numba clamp (haplotype kernel)**

In `python/genvarloader/_dataset/_genotypes.py:508`, clamp the available ref to be non-negative so an exhausted ref yields `out_end_idx == out_idx` and the right-pad fills the whole tail:

```python
        writable_ref = max(0, min(unfilled_length, len(ref) - ref_idx))
```

- [ ] **Step 4: Apply the same clamp to the numba track kernel**

In `python/genvarloader/_dataset/_tracks.py:396`:

```python
        writable_ref = max(0, min(unfilled_length, len(track) - track_idx))
```

- [ ] **Step 5: Run the numba test to verify it PASSES**

Run: `pixi run -e dev pytest tests/unit/dataset/test_reconstruct_trailing_fill.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS — `[1, 2, 50, 0, 0, 0, 0, 0]`.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_genotypes.py python/genvarloader/_dataset/_tracks.py tests/unit/dataset/test_reconstruct_trailing_fill.py
rtk git commit -m "fix(reconstruct,tracks): pad full tail in numba trailing-fill on ref overshoot

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 4: Verify the rust track-realign kernel + un-exclude parity

**Files:**
- Verify/Modify: rust track trailing-fill (search `src/` for the analog).
- Modify: `tests/parity/test_reconstruct_haplotypes_parity.py`.
- Check: `tests/parity/test_shift_and_realign_tracks_parity.py`, `tests/parity/test_dataset_parity.py`, `tests/parity/strategies.py`, `tests/parity/_fixtures.py`.

- [ ] **Step 1: Verify the rust track kernel has no `.max(0)` over-pad**

Run: `pixi run -e dev grep -n "max(0)\|writable_ref\|out_end" src/tracks.rs src/intervals.rs`
If the track-realign trailing-fill uses the same `(out_idx + writable_ref).max(0)` pattern, apply the identical `out_idx` clamp + add a cargo test mirroring Task 1. If it already clamps to `out_idx` (or has no negative-`writable_ref` path), record that in the commit message and skip.

- [ ] **Step 2: Remove the now-obsolete exclusion guards from the haplotype parity test**

In `tests/parity/test_reconstruct_haplotypes_parity.py`, delete:
- the `_ref_idx_overshoots_contig(...)` helper and both `assume(not _ref_idx_overshoots_contig(inputs))` calls (Guard 1),
- the `_numba_fully_defined(...)` double-init helper and `assume(defined)` calls (Guard 3),
- the `try/except SystemError: assume(False)` wrapper (Guard 2).

The body simplifies to: run numba into `out_n`, run rust into `out_r`, `np.testing.assert_array_equal`. (Both kernels now fully write every position byte-identically across the full generated domain, including overshoot.)

- [ ] **Step 3: Run the haplotype parity suite (both backends, full domain)**

Run: `pixi run -e dev pytest tests/parity/test_reconstruct_haplotypes_parity.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS — hypothesis explores overshoot inputs (no longer assumed away) and finds byte-identity. (The parity helper calls both `numba_fn` and `rust_fn` directly, so one run covers both backends.)

- [ ] **Step 4: Lift analogous exclusions in the track + dataset parity suites**

Inspect `test_shift_and_realign_tracks_parity.py`, `test_dataset_parity.py`, `strategies.py`, `_fixtures.py` for overshoot/`max_jitter`-pinned guards tied to THIS divergence (not the separate #242 `intervals_to_tracks` clip bug — leave those for W2). Remove only the trailing-fill-overshoot exclusions; re-run each touched suite:

Run: `pixi run -e dev pytest tests/parity/test_shift_and_realign_tracks_parity.py tests/parity/test_dataset_parity.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add src/ tests/parity/
rtk git commit -m "test(parity): un-exclude ref-overshoot sub-domain now both kernels pad correctly

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

### Task 5: Full-tree verification, roadmap update, and PR

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (Phase 5 notes/log).

- [ ] **Step 1: Run the full Python tree on the rust backend**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: green (the pre-existing xfails remain xfailed; no new failures).

- [ ] **Step 2: Run the full tree on the numba backend**

Run: `GVL_BACKEND=numba pixi run -e dev pytest tests/dataset tests/unit tests/parity -q --basetemp=$(pwd)/.pytest_tmp`
Expected: green — same pass/xfail profile, confirming byte-identical parity.

- [ ] **Step 3: Lint, format, typecheck, cargo**

Run:
```bash
pixi run -e dev ruff check python/ tests/ && \
pixi run -e dev ruff format --check python/ tests/ && \
pixi run -e dev typecheck && \
pixi run -e dev cargo-test
```
Expected: all clean/green.

- [ ] **Step 4: Record the fix in the roadmap**

Add a dated entry to the Notes & decisions log in `docs/roadmaps/rust-migration.md` noting: the overshoot trailing-fill divergence was fixed in BOTH kernels (clamp `out_end_idx` to `out_idx`; numba `writable_ref = max(0, ...)`), the previously-excluded sub-domain is now parity-covered (Guards 1–3 removed), and reference the filed issue. Do NOT yet mark Phase 5 ✅ (W2–W9 remain).

- [ ] **Step 5: Commit and open the PR**

```bash
rtk git add docs/roadmaps/rust-migration.md
rtk git commit -m "docs(roadmap): record trailing-fill overshoot fix (Phase 5 W1)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
rtk git push -u origin rust-migration   # (or a w1 topic branch, per your PR convention)
```
Then open the PR into `rust-migration` (file the GVL issue first and reference it). Title: `fix: pad full tail on reference overshoot in haplotype/track reconstruction (Phase 5 W1)`.

---

## Subsequent PRs (planned separately, in order)

Each gets its own detailed bite-sized plan written when its predecessor lands. They are **not** bite-sized here because they depend on results that don't exist yet.

- **PR2 (W2) — Fix the #242 `intervals_to_tracks` store-vs-query divergence.** Requires a systematic-debugging investigation: gvl stores intervals at `chromStart - max_jitter` but queries at `chromStart + jitter`, so a stored interval can start before the query window (`max_jitter>0`). The correct reconciliation (kernel clip vs store/query coordinate math) is unknown until investigated and may touch the write path. Fix both backends to agree-and-be-correct; un-exclude the #242 sub-domain across the parity + dataset suites; close issue #242. *Plan written after the investigation; W1 should land first so the oracle is otherwise trustworthy.*

- **PR3 (W3) — Fuse the deferred annotated+spliced intersection path.** Add a fused rust kernel collapsing its remaining FFI crossings (pattern: `reconstruct_annotated_haplotypes_fused` / `reconstruct_haplotypes_spliced_fused`). Parity-gate against the composed numba oracle **while numba still exists**. Extend the parity suite to cover it.

- **PR4 (W4) — Final single-thread numba-vs-rust `__getitem__` A/B.** Benchmark only (no code): `tests/benchmarks/test_e2e.py` pedantic min + `profile.py` wall-clock across all modes, both backends present, one back-to-back session. **Gate:** rust at parity-or-better single-thread → proceed to consolidation.

- **PR5 (W5–W7) — The consolidation PR.** (a) Golden-snapshot the ~17 numba-oracle parity suites to frozen fixtures (storage scheme decided in that plan — compressed `.npz` keyed by generated input, or a bounded seeded sample); (b) delete all numba: ~21 `register()` refs, njit bodies, `_dispatch` registry + `GVL_BACKEND`, every `import numba`; replace `get(name)(...)` with direct rust calls; assert `import genvarloader` pulls neither numba nor llvmlite; (c) add rayon batch parallelism over per-(query,hap) work items, gated byte-identical to the serial golden result.

- **PR6 (W8–W9) — Measure & merge.** Rust-only peak RSS (memray) vs the 3.53 GB numba baseline (expect the ~3.2 GB JIT drop); rayon multi-thread speedup (rayon N vs 1). If RSS and wall-clock are parity-or-better, open `rust-migration → main` (no squash); mark Phase 5 ✅ in the roadmap with the final tables + PR link; update `skills/genvarloader/SKILL.md` for any public-API change (e.g. `GVL_BACKEND` removal).

---

## Self-Review

- **Spec coverage:** W1 (haps trailing-fill) is fully planned as PR1 — and corrected to "fix both kernels," a deviation from the spec's "verify rust already correct" found during planning (documented in the PR1 preamble). W2–W9 map to PR2–PR6. Decisions D1–D7 are all reflected (D4 = PR1; D5 = PR2; D3 = PR3; D6 = PR4; D2 = PR5; D1 = PR5; D7 = separate PRs throughout).
- **Placeholder scan:** PR1 steps contain concrete code, exact commands, and expected output. PR2–PR6 are intentionally high-level (planned separately) and labeled as such — not placeholders within an executable task.
- **Type consistency:** `reconstruct_haplotype_from_sparse` signature and the `run(...)` cargo helper argument order match the source read during planning; `writable_ref`/`out_end_idx`/`out_idx` names match both kernels.
