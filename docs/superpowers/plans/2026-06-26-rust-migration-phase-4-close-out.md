# Rust Migration Phase 4 Close-out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close out Rust-migration Phase 4 — delete the last dead write-path numba kernel, capture canonical Carter write/update perf + RSS numbers, confirm write-path parity, and reconcile the roadmap to reality (Phase 4 ✅).

**Architecture:** No new Rust kernel. The default `gvl.write()` / `gvl.update()` path is already Rust-backed (bigWig streaming writer + COITrees table engine; variant IO via genoray). The only remaining write-path numba (`splits_sum_le_value`) is reachable solely through `_write_track_legacy`, the dispatch fall-through for custom `IntervalTrack` types — of which there are zero concrete public implementations. We delete it as dead, replace the fall-through with a hard `TypeError`, then measure and document.

**Tech Stack:** Python (pytest, polars, numpy), Rust (PyO3, abi3), pixi (`-e dev`), memray, numba (read-path references only).

## Global Constraints

- Run all dev tasks through `pixi run -e dev <task>` (this worktree has its own fresh pixi env; no symlinked `.pixi`).
- Dataset tests need pytest's tmp on the same filesystem as `tests/data`: pass `--basetemp=$(pwd)/.pytest_tmp` (HPC `os.link` cross-device Errno 18).
- Parity must hold byte-identical across **both** backends (`GVL_BACKEND=rust` default and `GVL_BACKEND=numba`).
- Measurements: `NUMBA_NUM_THREADS=1`, release build (`maturin develop --release` / `pixi run -e dev` release task), Carter HPC (AMD EPYC 7543, linux-64). Report wall-clock + peak RSS (memray).
- Conventional-commit messages; end commit messages with `Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>`.
- Do not touch read-path numba kernels (`padded_slice`, `_assemble_alt_windows`, `apply_site_only_variants`, `_tracks.py` realign) — they are retained Phase-5-deletion references.

---

### Task 1: Delete the dead legacy track path + `splits_sum_le_value`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (delete `_write_track_legacy` lines 1254-1386; change fall-through at line 1467; drop `splits_sum_le_value` from the import at line 41)
- Modify: `python/genvarloader/_dataset/_utils.py` (delete `splits_sum_le_value`, lines 165-196)
- Modify: `tests/unit/test_utils.py` (drop `splits_sum_le_value` from import line 4; delete `test_splits_sum_le_value`, line 63)
- Modify: `tests/unit/dataset/test_dataset_utils.py` (drop `splits_sum_le_value` from import line 13; delete `test_splits_sum_le_value_docstring_example`, lines 81-82)
- Modify: `src/lib.rs:54` (stale docstring — bigWig writer emits SoA `starts/ends/values.npy`, not `intervals.npy`)
- Test: `tests/unit/dataset/test_write.py` (add the new TypeError test; create the file if absent)

**Interfaces:**
- Consumes: `genvarloader._dataset._write._write_track(out_dir, bed, track, samples, max_mem)` — dispatches `BigWigs`→Rust, `Table`→Rust, else now raises.
- Produces: `_write_track` raises `TypeError` for any track that is not `BigWigs`/`Table`. No public symbol changes.

- [ ] **Step 1: Write the failing test**

In `tests/unit/dataset/test_write.py` (create if needed):

```python
from pathlib import Path

import polars as pl
import pytest

from genvarloader._dataset._write import _write_track


def test_write_track_rejects_unsupported_type():
    """Custom IntervalTrack types are unsupported now that the legacy path is gone."""
    with pytest.raises(TypeError, match="BigWigs.*Table"):
        _write_track(Path("/tmp/unused"), pl.DataFrame(), object(), None, 1)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write.py::test_write_track_rejects_unsupported_type -v --basetemp=$(pwd)/.pytest_tmp`
Expected: FAIL — currently the fall-through calls `_write_track_legacy`, which tries to treat `object()` as a track (AttributeError / different error), not `TypeError`.

- [ ] **Step 3: Replace the fall-through and delete `_write_track_legacy`**

In `python/genvarloader/_dataset/_write.py`, change the last line of `_write_track` (line 1467) from:

```python
    return _write_track_legacy(out_dir, bed, track, samples, max_mem)
```

to:

```python
    raise TypeError(
        f"Unsupported track type {type(track).__name__!r}; "
        "tracks must be a genvarloader.BigWigs or genvarloader.Table."
    )
```

Then delete the entire `_write_track_legacy` function (lines 1254-1386, from `def _write_track_legacy(` up to but not including `def _write_track_rust(`).

- [ ] **Step 4: Delete `splits_sum_le_value` and its import**

In `python/genvarloader/_dataset/_write.py` line 41, change:

```python
from ._utils import bed_to_regions, regions_to_bed, splits_sum_le_value
```

to:

```python
from ._utils import bed_to_regions, regions_to_bed
```

In `python/genvarloader/_dataset/_utils.py`, delete the `splits_sum_le_value` function (the `@nb.njit(...)` decorator at line 165 through the end of the function body at line 196). Leave `padded_slice` (lines 37-72) untouched.

- [ ] **Step 5: Delete the two `splits_sum_le_value` unit tests**

In `tests/unit/test_utils.py` line 4, change:

```python
from genvarloader._dataset._utils import bed_to_regions, splits_sum_le_value
```

to:

```python
from genvarloader._dataset._utils import bed_to_regions
```

and delete the `test_splits_sum_le_value` function (starting line 63).

In `tests/unit/dataset/test_dataset_utils.py`, remove `splits_sum_le_value` from the import block (line 13) and delete `test_splits_sum_le_value_docstring_example` (lines 81-82 and its body).

- [ ] **Step 6: Fix the stale Rust docstring**

In `src/lib.rs:54`, change the comment:

```rust
/// Write intervals.npy + offsets.npy for a bigWig track directly to `out_dir`.
```

to:

```rust
/// Write SoA starts/ends/values.npy + offsets.npy for a bigWig track directly to `out_dir`.
```

- [ ] **Step 7: Run the new test + the utils tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write.py::test_write_track_rejects_unsupported_type tests/unit/test_utils.py tests/unit/dataset/test_dataset_utils.py -v --basetemp=$(pwd)/.pytest_tmp`
Expected: PASS (new TypeError test green; no remaining references to `splits_sum_le_value`).

- [ ] **Step 8: Grep to confirm no dangling references**

Run: `grep -rn "splits_sum_le_value\|_write_track_legacy" python/genvarloader/ tests/ --include="*.py"`
Expected: no matches.

- [ ] **Step 9: Rebuild Rust + run the write-path test slice on both backends**

Run: `pixi run -e dev pytest tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp`
Then: `GVL_BACKEND=numba pixi run -e dev pytest tests/dataset tests/unit -q --basetemp=$(pwd)/.pytest_tmp`
Expected: both green (pre-existing xfails unchanged: `test_e2e_variants`, `test_haps_property` ×2, `test_parse_idx[missing]`, `test_getitem[no_regions]`).

- [ ] **Step 10: Commit**

```bash
git add python/genvarloader/_dataset/_write.py python/genvarloader/_dataset/_utils.py \
        tests/unit/test_utils.py tests/unit/dataset/test_dataset_utils.py \
        tests/unit/dataset/test_write.py src/lib.rs
git commit -m "refactor(write): delete dead legacy track path + splits_sum_le_value

_write_track_legacy was reachable only via custom IntervalTrack types (none
exist; IntervalTrack is unexported). Replace the dispatch fall-through with a
TypeError and drop the last write-path numba kernel (splits_sum_le_value) and
its tests. Write path is now numba-free. Fix stale SoA docstring in lib.rs.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Realistic write/update measurement driver

**Files:**
- Create: `tests/benchmarks/profiling/profile_write_realistic.py`

**Interfaces:**
- Consumes: helpers + constants from `tests/benchmarks/data/build_realistic.py` — `choose_samples()`, `copy_regions()`, `slice_pgen(samples, bed_path)`, `drop_unsupported_variants(pgen)`, and module constants `SAMPLE_MAP`, `BW_CHR22_DIR`. Also `genvarloader.write`/`genvarloader.update`, `genvarloader.BigWigs`, `genoray.PGEN`.
- Produces: a CLI `python tests/benchmarks/profiling/profile_write_realistic.py --op {write,update}` printing `op=... corpus=chr22_geuv wall=<s>s (...)`. Times only the `gvl.write` / `gvl.update` call (prep runs untimed). Runnable under `memray run` for peak RSS.

This driver exercises the **full Rust write path** (genoray sparse genotypes + the Rust bigWig streaming writer) on the realistic chr22 corpus, and a real per-sample `BigWigs` track add for `update` (replacing the 60-row synthetic annot smoke).

- [ ] **Step 1: Write the driver**

Create `tests/benchmarks/profiling/profile_write_realistic.py`:

```python
"""Time gvl.write() and a real per-sample BigWigs gvl.update() on the chr22_geuv corpus.

Exercises the full Rust write path (genoray sparse genotypes + Rust bigWig
streaming writer). Prep (sample choice, plink2 slice) runs untimed; only the
gvl.write / gvl.update call is measured.

Usage (needs /carter sources or GVL_BENCH_SOURCE bundle):
    pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op write
    pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op update

Peak RSS:
    NUMBA_NUM_THREADS=1 .pixi/envs/dev/bin/memray run -o w.bin \\
        tests/benchmarks/profiling/profile_write_realistic.py --op write
    .pixi/envs/dev/bin/memray stats w.bin
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

import polars as pl

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.benchmarks.data import build_realistic as br  # noqa: E402

CORPUS_TAG = "chr22_geuv"


def _resolve_bigwig_paths(samples: list[str]) -> dict[str, str]:
    """Resolve per-sample chr22 bigWig paths exactly as build_realistic.build_dataset."""
    smap = pl.read_csv(br.SAMPLE_MAP)
    paths: dict[str, str] = {}
    for sample, full_path in smap.select("sample", "path").iter_rows():
        if sample not in samples:
            continue
        bw = br.BW_CHR22_DIR / Path(full_path).name
        if not bw.exists():
            raise SystemExit(f"Missing chr22 bigwig for {sample}: {bw}")
        paths[sample] = str(bw)
    assert set(paths) == set(samples), set(samples) - set(paths)
    return paths


def _prep() -> tuple[list[str], Path, Path, dict[str, str]]:
    """Untimed prep: choose samples, build regions BED, slice + filter PGEN, resolve bigwigs."""
    samples = br.choose_samples()
    bed_path = br.copy_regions()
    pgen = br.slice_pgen(samples, bed_path)
    pgen = br.drop_unsupported_variants(pgen)
    paths = _resolve_bigwig_paths(samples)
    return samples, pgen, bed_path, paths


def run_write(out: Path) -> float:
    import genvarloader as gvl
    from genoray import PGEN

    samples, pgen, bed_path, paths = _prep()
    tracks = gvl.BigWigs("read-depth", paths)
    t0 = time.perf_counter()
    gvl.write(
        path=out,
        bed=bed_path,
        variants=PGEN(pgen),
        tracks=tracks,
        samples=samples,
        overwrite=True,
        extend_to_length=False,
    )
    return time.perf_counter() - t0


def run_update(out: Path) -> tuple[float, str]:
    import genvarloader as gvl
    from genoray import PGEN

    samples, pgen, bed_path, paths = _prep()
    # Build a base dataset (untimed) to update.
    gvl.write(
        path=out,
        bed=bed_path,
        variants=PGEN(pgen),
        tracks=gvl.BigWigs("read-depth", paths),
        samples=samples,
        overwrite=True,
        extend_to_length=False,
    )
    # Timed: add a SECOND per-sample BigWigs track via update (Rust bigWig writer).
    add = gvl.BigWigs("read-depth-2", paths)
    t0 = time.perf_counter()
    gvl.update(out, tracks=add, max_mem="4g")
    wall = time.perf_counter() - t0
    return wall, f"track=read-depth-2 samples={len(samples)}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--op", choices=["write", "update"], required=True)
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "chr22_geuv_bench.gvl"
        if args.op == "write":
            wall = run_write(out)
            print(f"op=write corpus={CORPUS_TAG} wall={wall:.3f}s")
        else:
            wall, info = run_update(out)
            print(f"op=update corpus={CORPUS_TAG} wall={wall:.3f}s ({info})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run the driver (write) to verify it executes**

Run: `NUMBA_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op write`
Expected: prints `op=write corpus=chr22_geuv wall=<s>s`. If it raises `SystemExit` about missing `/carter` sources, set `GVL_BENCH_SOURCE` to the extracted source bundle and retry; if no source bundle is reachable at all, record that and fall back to the 1kg driver in Task 3 (note the fallback in the roadmap).

- [ ] **Step 3: Smoke-run the driver (update)**

Run: `NUMBA_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op update`
Expected: prints `op=update corpus=chr22_geuv wall=<s>s (track=read-depth-2 samples=5)`.

- [ ] **Step 4: Commit**

```bash
git add tests/benchmarks/profiling/profile_write_realistic.py
git commit -m "test(bench): realistic chr22_geuv write/update perf driver

Times gvl.write (PGEN variants + per-sample BigWigs track) and a real
per-sample BigWigs gvl.update on the chr22_geuv corpus, exercising the full
Rust write path. Replaces the 60-row synthetic annot smoke for the update gate.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Capture the gate — perf + RSS + full-tree parity

**Files:** none (measurement + verification only; outputs feed Task 4).

**Interfaces:**
- Consumes: `profile_write_realistic.py` (Task 2), `memray`, the dual-backend test tree.
- Produces: recorded numbers — `write()` wall + peak RSS, `update()` wall + peak RSS (corpus `chr22_geuv`, Carter) — and confirmation that the full tree is green on both backends. These numbers are pasted into the roadmap in Task 4.

- [ ] **Step 1: Ensure a release build**

Run: `pixi run -e dev maturin develop --release`
Expected: builds clean (abi3).

- [ ] **Step 2: Measure `write()` wall-clock (median of 3)**

Run 3×: `NUMBA_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op write`
Record the median `wall=` value.

- [ ] **Step 3: Measure `write()` peak RSS under memray**

Run: `NUMBA_NUM_THREADS=1 .pixi/envs/dev/bin/memray run -f -o /tmp/w.bin tests/benchmarks/profiling/profile_write_realistic.py --op write && .pixi/envs/dev/bin/memray stats /tmp/w.bin | grep -i "peak memory"`
Record peak RSS.

- [ ] **Step 4: Measure `update()` wall-clock (median of 3) + peak RSS**

Run 3×: `NUMBA_NUM_THREADS=1 pixi run -e dev python tests/benchmarks/profiling/profile_write_realistic.py --op update` (record median wall).
Then: `NUMBA_NUM_THREADS=1 .pixi/envs/dev/bin/memray run -f -o /tmp/u.bin tests/benchmarks/profiling/profile_write_realistic.py --op update && .pixi/envs/dev/bin/memray stats /tmp/u.bin | grep -i "peak memory"`
Record peak RSS.

- [ ] **Step 5: Confirm write-path parity (already-landed differential tests)**

Run: `pixi run -e dev pytest tests/parity -q --basetemp=$(pwd)/.pytest_tmp` and the table/bigwig write tests: `pixi run -e dev pytest -q -k "table or bigwig or write" tests --basetemp=$(pwd)/.pytest_tmp`
Expected: green (bigWig byte-identical writer test; Table COITrees numpy-oracle + property tests).

- [ ] **Step 6: Full tree, both backends**

Run: `pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Then: `GVL_BACKEND=numba pixi run -e dev pytest tests -q --basetemp=$(pwd)/.pytest_tmp`
Expected: both green except the known pre-existing xfails.

- [ ] **Step 7: cargo + lint/format/typecheck + abi3**

Run:
```bash
pixi run -e dev cargo-test
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format --check python/ tests/
pixi run -e dev typecheck
```
Expected: all clean/green.

- [ ] **Step 8: Record the captured numbers in a scratch note**

Write the four numbers + machine/corpus/HEAD into `docs/superpowers/plans/2026-06-26-phase-4-measurements.md` (a short scratch file) so Task 4 can transcribe them into the roadmap. Commit:

```bash
git add docs/superpowers/plans/2026-06-26-phase-4-measurements.md
git commit -m "docs(bench): record Phase 4 Carter write/update perf + RSS

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Reconcile the roadmap + mark Phase 4 ✅

**Files:**
- Modify: `docs/roadmaps/rust-migration.md` (Phase 4 section ~lines 600-610; baseline table ~lines 103-108; notes/decisions log)
- Verify only: `skills/genvarloader/SKILL.md` (expect no change)

**Interfaces:**
- Consumes: the four measured numbers from Task 3.
- Produces: Phase 4 marked ✅ with PR link; baseline table updated; a dated decisions-log entry. No code.

- [ ] **Step 1: Rewrite the Phase 4 section**

In `docs/roadmaps/rust-migration.md`, replace the Phase 4 block (`### Phase 4 — Write / update pipeline 🚧` … through its `**Gate:**` line) with a ✅ version that:
  - marks the phase ✅ and sets `_PR: <link>_` (fill the PR URL when opened);
  - states that variant normalization is a **user precondition** (`bcftools norm` / `plink2 --normalize`), not GVL work, and strikes it from scope;
  - states genotype storage / variant IO (genoray `dense2sparse`) is **deferred to Phase 6 (absorb genoray)**;
  - keeps the two ✅ slices (bigWig streaming writer; Table COITrees);
  - records that the dead `_write_track_legacy` + `splits_sum_le_value` path was deleted (write path now numba-free; custom `IntervalTrack` types raise `TypeError`);
  - records the gate result with the Task-3 numbers.

Example replacement text (fill in the measured numbers):

```markdown
### Phase 4 — Write / update pipeline ✅
_PR: <PR-URL>_

The default `gvl.write()` / `gvl.update()` path is fully Rust-backed; the write path is numba-free.

- [x] bigWig interval extraction — single-pass streaming Rust writer (SoA `starts/ends/values.npy`).
- [x] Table + annot overlap — COITrees Rust engine.
- [x] Deleted the dead `_write_track_legacy` + `splits_sum_le_value` (the last write-path numba),
      reachable only via custom `IntervalTrack` types (none exist; `IntervalTrack` is unexported).
      Unsupported track types now raise `TypeError`.
- **Variant normalization (left-align, bi-allelic, atomize) is NOT GVL work** — it is a user
  precondition (`bcftools norm` / `plink2 --normalize`); the write path only validates/rejects
  non-conforming records. Struck from Phase 4 scope.
- **Genotype storage / variant IO (genoray `dense2sparse`) deferred to Phase 6 (absorb genoray).**

**Gate (parity — MET):** write-path parity = the landed differential tests (bigWig byte-identical;
Table COITrees numpy-oracle + property). Full tree green on both backends.

**Gate (throughput/RSS — Carter re-baseline, chr22_geuv):**

| Op | corpus | wall-clock | peak RSS |
|---|---|---|---|
| `gvl.write()` (PGEN variants + BigWigs track) | chr22_geuv (5 samples × <N> regions, chr22) | <W> s | <R> GB |
| `gvl.update()` (add per-sample BigWigs track) | chr22_geuv | <W> s | <R> GB |

> Carter HPC (AMD EPYC 7543, linux-64), `NUMBA_NUM_THREADS=1`, release build, HEAD `<hash>`. The
> write path is already Rust-only (Python/numba orchestration deleted at landing), so there is no
> live numba A/B; these are the canonical Phase 4 numbers. The old 1.143 s / 3.593 GB write figure
> was macOS / 1kg-VCF and is **not comparable**.
```

- [ ] **Step 2: Annotate the old baseline table row**

In the Baseline metrics table (~line 107), update the `gvl.update()` row: replace the "smoke only" TBD note with a pointer to the Phase 4 chr22_geuv update number, and mark the macOS `gvl.write()` row (line 105) as superseded-for-comparison by the Carter chr22_geuv re-baseline.

- [ ] **Step 3: Add a decisions-log entry**

Prepend to the "Notes & decisions log" section:

```markdown
- 2026-06-26 (Phase 4 close-out; branch `phase-4-close-out`, PR <URL>): Investigation found the
  default write/update path already fully Rust-backed (bigWig streaming writer + COITrees table;
  variant IO via genoray). The roadmap's "variant normalization" bullet was a mischaracterization —
  GVL never normalizes (it is a bcftools/plink2 user precondition); genotype storage is genoray
  (→ Phase 6). Deleted the only remaining write-path numba (`splits_sum_le_value` + the dead
  `_write_track_legacy`; unsupported `IntervalTrack` types now `TypeError`). Captured canonical
  Carter chr22_geuv write/update wall-clock + peak RSS (no live numba A/B — orchestration was
  deleted at landing). Full tree green both backends; cargo + lint/format/typecheck clean; abi3
  builds. Phase 4 ✅.
```

- [ ] **Step 4: Verify the skill needs no update**

Run: `grep -n "write\|update\|IntervalTrack\|BigWigs\|Table" skills/genvarloader/SKILL.md | head`
Confirm: no public-API claim changed (no exported symbol, signature, or default changed; `IntervalTrack` is unexported). If the skill documents a "custom IntervalTrack" capability, add a one-line note that only `BigWigs`/`Table` are supported. Otherwise no change.

- [ ] **Step 5: Commit**

```bash
git add docs/roadmaps/rust-migration.md skills/genvarloader/SKILL.md
git commit -m "docs(roadmap): Phase 4 close-out — write path numba-free, gate captured, scope reconciled

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Spec A (delete dead legacy path) → Task 1. ✅
- Spec B (Carter re-baseline write + real update) → Tasks 2–3. ✅
- Spec C (parity via landed differential tests) → Task 3 steps 5–6. ✅
- Spec D (roadmap reconciliation, Phase 4 ✅, genoray→Phase 6, SKILL check) → Task 4. ✅
- Out-of-scope items (genoray, read-path numba, rayon) are not given tasks. ✅

**Placeholder scan:** Measured numbers (`<W>`, `<R>`, `<hash>`, `<PR-URL>`) are intentional fill-at-runtime values produced by Task 3 / at PR time, not vague instructions — every code step has concrete code. No "TBD/add error handling" placeholders.

**Type consistency:** `_write_track(out_dir, bed, track, samples, max_mem)` signature is used consistently (Task 1 test + dispatch). `profile_write_realistic.py` reuses `build_realistic` helper names verified against the source (`choose_samples`, `copy_regions`, `slice_pgen`, `drop_unsupported_variants`, `SAMPLE_MAP`, `BW_CHR22_DIR`). `gvl.BigWigs(name, paths)` and `gvl.update(path, tracks=...)` match the codebase.
