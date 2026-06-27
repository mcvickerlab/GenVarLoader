# Design: Rust migration Phase 4 close-out (write/update gate + reconcile)

**Date:** 2026-06-26
**Branch:** `phase-4-close-out` (worktree `.claude/worktrees/phase-4-close-out`, off `rust-variant-rc-fold`)
**Roadmap:** `docs/roadmaps/rust-migration.md` — Phase 4 (🚧 → ✅)

## Problem & context

Phase 4 of the Rust migration ("Write / update pipeline") is marked 🚧 with bullets:

- Migrate `_dataset/_write.py`: variant normalization (left-align, bi-allelic, atomize),
  genotype storage, interval extraction + realign.
  - [x] bigWig interval extraction — single-pass streaming Rust writer
  - [x] Table + annot overlap — COITrees Rust engine
- Migrate remaining `_dataset/_utils.py` / `_flat_flanks.py` / `_variants/_sitesonly.py`
  kernels touched by the write path.

**Investigation finding (2026-06-26): the porting is essentially already done.** Tracing the
real `gvl.write()` / `gvl.update()` paths shows the roadmap bullets mischaracterize the work:

- **Variant normalization (left-align, bi-allelic, atomize) is NOT something GVL does.** It is a
  documented *precondition* the user satisfies with `bcftools norm` / `plink2 --normalize`
  (`_write.py:124-129`). The write path only *validates and rejects* non-bi-allelic / symbolic /
  breakend records (`_write.py:599-615`). There is no numba normalization kernel to port.
- **Genotype storage is done by genoray**, via `dense2sparse` / `_dense2sparse_with_length`
  (`genoray._svar`, imported at `_write.py:21-22`). That belongs to **Phase 6 (absorb genoray)**,
  not Phase 4.
- **Interval extraction + realign** on the write path is the bigWig streaming writer (✅) and the
  Table COITrees engine (✅), both already shipped. There is no write-time *realign* — realign is a
  read-path concern.
- Of the remaining-file candidates, the only GVL numba kernel reachable on the write path is
  `splits_sum_le_value` (`_utils.py:165-196`), used solely by `_write_track_legacy`
  (`_write.py:1254-1386`), the dispatch fall-through for custom `IntervalTrack` sources
  (`_write.py:1467`). The Phase 0 notes (roadmap lines 767-780) already document this exact path as
  **dead** for the only concrete public track types (`BigWigs`→Rust, `Table`→Rust). Verified
  2026-06-26: there are **no** concrete `IntervalTrack` subclasses anywhere in the codebase besides
  `BigWigs` and `Table`, and `IntervalTrack` itself is **not exported** in `__init__.py`.
  `_flat_flanks.py::_assemble_alt_windows`, `_sitesonly.py::apply_site_only_variants`, `padded_slice`,
  and the `_tracks.py` kernels are all **read-path**, outside Phase 4.

So "finishing Phase 4" is a **close-out + reconcile**, not a new port. Decisions taken with the
maintainer (2026-06-26):

1. Deliver: close out the gate **and** reconcile the roadmap. Mark Phase 4 ✅.
2. The dead legacy track path is **deleted as dead** (Phase 0 precedent).
3. The gate is measured as a **Carter absolute re-baseline** (the write path is already Rust-only;
   the Python/numba orchestration was deleted at landing, so there is no live numba A/B).

## Scope

### In scope

**A. Delete the dead legacy track path**
- Remove `_write_track_legacy` (`_write.py:1254-1386`).
- Replace the `else` fall-through at `_write.py:1467` with a clear `TypeError` naming the unsupported
  track type and pointing at `BigWigs` / `Table`.
- Remove `splits_sum_le_value` (`_utils.py:165-196`) and its unit test.
- Leave `padded_slice` (`_utils.py:37-72`, read-path numba reference) untouched.
- Confirm no other importers of `splits_sum_le_value` (it is not registered in `_dispatch.py`).
- Net effect: the `gvl.write()` / `gvl.update()` path is **numba-free**.

**B. Measurement gate — Carter absolute re-baseline**
- **`write()` workload:** build the `chr22_geuv` corpus from its sources (PGEN variants + a bigWig
  track; 165 regions × 5 samples, chr22) via `tests/benchmarks/profiling/profile_write.py --op write`.
  Record wall-clock + peak RSS (memray), `NUMBA_NUM_THREADS=1`, release build, Carter HPC
  (AMD EPYC 7543, linux-64).
- **`update()` workload:** open `chr22_geuv.gvl`, `gvl.update()` adding a new per-sample `BigWigs`
  read-depth track — exercises the Rust streaming bigWig writer through the update entry point.
  Record wall-clock + peak RSS. This replaces the 60-row synthetic smoke row.
- Record both as the canonical Phase 4 numbers in the roadmap baseline table; annotate the old
  1.143 s / 3.593 GB write figure as macOS / non-comparable.

**C. Parity confirmation**
- Write-path parity = the already-landed differential tests: the bigWig writer's byte-identical
  test (roadmap 2026-06-19 note, Task 6) and the Table COITrees numpy-oracle + property tests. No new
  A/B (legacy is deleted). Re-run these plus the full tree on both backends to confirm green.

**D. Roadmap + reconciliation**
- Rewrite the Phase 4 section to reflect reality:
  - variant normalization → user precondition (bcftools / plink2), struck from Phase 4;
  - genotype storage / variant IO → explicitly Phase 6 (genoray);
  - bigWig + Table slices ✅;
  - dead legacy path deleted.
- Record the Carter write/update baseline numbers.
- Set Phase 4 ✅ + PR link; add a notes/decisions-log entry.

### Out of scope (explicitly)

- Genotype storage / variant IO (`dense2sparse`) → **Phase 6 (genoray)**.
- All read-path numba kernels (`padded_slice`, `_assemble_alt_windows`, `apply_site_only_variants`,
  `_tracks.py` realign kernels) → retained as Phase-5-deletion references.
- Rayon batch parallelism → Phase 5.
- Any new Rust kernel (nothing on the write path needs one once the dead path is deleted).

## Verification

- Full test tree on **both backends** (`GVL_BACKEND` rust + numba): `pixi run -e dev pytest tests -q`
  (dataset + unit). Read-path parity must be unaffected by the deletion.
- `cargo test` green; lint (`ruff check python/ tests/`), format, `typecheck` clean; abi3 wheel builds.
- `tests/integration/test_scale_guard.py` still green (write path).
- Confirm deleting `_write_track_legacy` breaks no existing test (search for tests that write a custom
  `IntervalTrack`; expect none).
- Public API is unchanged (`IntervalTrack` unexported; `BigWigs` / `Table` untouched) → no SKILL.md
  update expected; verify against the CLAUDE.md skill-maintenance checklist before closing.

## Risks & notes

- **Cross-machine baseline:** the original 1.143 s / 3.593 GB write figure was macOS; the new numbers
  are Carter. They are not directly comparable — the roadmap entry must say so explicitly. Carter
  becomes the canonical write/update baseline going forward.
- **Corpus availability:** `write()` measurement needs the `chr22_geuv` source inputs (PGEN + bigWig)
  reachable via `/carter` or `GVL_BENCH_SOURCE` (per the Phase 0 build_realistic.py note). If sources
  are unavailable, fall back to the synthetic chr21/chr22 slice used for the bigWig write slice.
- **Worktree env:** fresh pixi env per worktree (no symlinked `.pixi`), per the parallel-worktree
  memory; `pixi run -e dev gen` before the first test run.
