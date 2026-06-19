# Fix #233 — annot bigWig off-by-one Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the off-by-one in the Rust bigWig interval reader so annot/track readback expands intervals across their full width instead of collapsing wide intervals to `value/span_length`.

**Architecture:** A single-line correction in `src/bigwig.rs` (store `start`, not `start - 1`). bigtools already clips intervals to the query range, so the `- 1` was a pure off-by-one. Add a regression test on the non-experimental bigWig annot path; regenerate the 4 byte-identical characterization snapshots that baked in the bug.

**Tech Stack:** Rust (PyO3 + bigtools), Python, numba (`intervals_to_tracks` kernel), pytest, pyBigWig, pixi, maturin.

## Global Constraints

- All dev commands run under pixi: `pixi run -e dev <task>`.
- Rust changes require `pixi run -e dev maturin develop --release` before Python sees them.
- Conventional commits (commitizen). End commit messages with the `Co-Authored-By` trailer.
- A pre-push hook enforces `ruff format` (separate from `ruff check`) — run both before pushing.
- Before pushing a change touching shared code, run the **full tree**: `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`).
- No public API signature/mode/default changes. No `skills/genvarloader/SKILL.md` update (it documents dataset-level reading, not the internal per-base coordinate).
- No kernel change (`_dataset/_intervals.py` stays as-is); fix is minimal, parity-preserving.

---

### Task 1: Regression test for wide-interval annot readback

**Files:**
- Modify: `tests/integration/tracks/test_annot_tracks.py` (append a new test)
- Source under test: `src/bigwig.rs:124` (the fix), `python/genvarloader/_dataset/_intervals.py` (the kernel consuming the coords)

**Interfaces:**
- Consumes: `gvl.write(out, bed, tracks=[BigWigs(...)], annot_tracks={...})`, `gvl.Dataset.open`, `Dataset.with_seqs(None).with_output_format("flat").with_settings(realign_tracks=False).with_tracks([...])`. The flat result exposes `.data` and `.offsets`.
- Produces: a test `test_annot_bigwig_wide_intervals_full_width` that fails on the unfixed reader and passes on the fixed one.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/tracks/test_annot_tracks.py`:

```python
def test_annot_bigwig_wide_intervals_full_width(tmp_path):
    """#233: annot bigWig intervals wider than the query region must expand to
    their full width on readback, not collapse to value/span_length."""
    import numpy as np
    import pyBigWig
    from genvarloader._bigwig import BigWigs

    bw_path = tmp_path / "binned.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr22", 50_000_000)])
    # 1 kb bins: each region below falls inside one bin (interval wider than region)
    starts = list(range(16_770_000, 16_832_000, 1000))
    ends = [s + 1000 for s in starts]
    vals = [float((s // 1000) % 7) * 0.1 + 0.2 for s in starts]
    bw.addEntries(["chr22"] * len(starts), starts, ends=ends, values=vals)
    bw.close()

    spans = [(16_774_654, 16_775_004), (16_829_376, 16_829_599)]
    bed = pl.DataFrame(
        {
            "chrom": ["chr22"] * len(spans),
            "chromStart": [s for s, _ in spans],
            "chromEnd": [e for _, e in spans],
            "name": [f"r{i}" for i in range(len(spans))],
        }
    )

    out = tmp_path / "repro.gvl"
    gvl.write(
        out,
        bed=bed,
        tracks=[BigWigs("dummy", {"s0": str(bw_path)})],
        annot_tracks={"t": str(bw_path)},
        overwrite=True,
    )
    ds = gvl.Dataset.open(out)
    fr = (
        ds.with_seqs(None)
        .with_output_format("flat")
        .with_settings(realign_tracks=False)
        .with_tracks(["t"])
    )[0 : ds.n_regions, 0]
    data, offs = np.asarray(fr.data), np.asarray(fr.offsets)

    src = pyBigWig.open(str(bw_path))
    for i, (s, e) in enumerate(spans):
        sv = src.values("chr22", s, e, numpy=True)
        sv = sv[~np.isnan(sv)]
        seg = data[offs[i] : offs[i + 1]]
        # full width filled (catches the collapse), value correct
        assert np.count_nonzero(seg) == (e - s)
        assert np.isclose(np.nanmean(seg), sv.mean(), atol=1e-4)
```

- [ ] **Step 2: Confirm the test fails on the unfixed reader**

The fix may already be in the working tree from root-causing. Temporarily revert it to prove fail-first, then restore:

```bash
git stash push -- src/bigwig.rs        # no-op if src/bigwig.rs is unmodified
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/integration/tracks/test_annot_tracks.py::test_annot_bigwig_wide_intervals_full_width -v
```
Expected: FAIL — `count_nonzero(seg)` is far less than `e - s` (e.g. 5 and 1).

If `git stash` reported "No local changes to save" (the fix was not staged in the tree), instead edit `src/bigwig.rs:124` to `start - 1`, rebuild, run, confirm FAIL.

- [ ] **Step 3: Run ruff on the test file**

```bash
pixi run -e dev ruff check tests/integration/tracks/test_annot_tracks.py
pixi run -e dev ruff format tests/integration/tracks/test_annot_tracks.py
```
Expected: clean.

- [ ] **Step 4: Commit the failing test (red)**

```bash
git add tests/integration/tracks/test_annot_tracks.py
git commit -m "test: regression for #233 wide annot bigWig interval readback

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: Apply the off-by-one fix in the Rust reader

**Files:**
- Modify: `src/bigwig.rs:124`

**Interfaces:**
- Consumes: the failing test from Task 1.
- Produces: corrected interval start coordinates for every bigWig read (per-sample `tracks` and `annot_tracks` alike).

- [ ] **Step 1: Restore/apply the fix**

If Task 1 Step 2 used `git stash`:
```bash
git stash pop
```
Otherwise edit `src/bigwig.rs` so line 124 reads:
```rust
                                .write(MaybeUninit::new(start));
```
(was `MaybeUninit::new(start - 1)`). Confirm the surrounding block is intact:
```rust
let Value { start, end, value } = itv.expect("Read interval");
unsafe {
    coords_ptr.add((offset + i) * 2).write(MaybeUninit::new(start));
    coords_ptr.add((offset + i) * 2 + 1).write(MaybeUninit::new(end));
    values_ptr.add(i + offset).write(MaybeUninit::new(value));
}
```

- [ ] **Step 2: Rebuild and run the regression test (green)**

```bash
pixi run -e dev maturin develop --release
pixi run -e dev pytest tests/integration/tracks/test_annot_tracks.py::test_annot_bigwig_wide_intervals_full_width -v
```
Expected: PASS.

- [ ] **Step 3: Run the existing annot test still passes**

```bash
pixi run -e dev pytest tests/integration/tracks/test_annot_tracks.py -v
```
Expected: `test_write_with_annot_tracks` and the new test PASS (the experimental polars-bio test stays skipped without `GVL_TEST_EXPERIMENTAL`).

- [ ] **Step 4: Run cargo tests**

```bash
pixi run -e dev cargo test 2>&1 | tail -20
```
Expected: PASS (no Rust test encoded the off-by-one; if one does, it asserted buggy behavior — update it to the corrected start and note it in the commit).

- [ ] **Step 5: Commit the fix**

```bash
git add src/bigwig.rs
git commit -m "fix: off-by-one in bigWig interval start collapsed wide annot tracks (#233)

bigtools get_interval already clips to the query range; storing start-1
shifted every interval one base 5', which made intervals_to_tracks compute
a negative relative start and write an empty slice for wide intervals
(value/span_length collapse) and a one-base dilution for fine tracks. Also
fixes a latent u32 underflow for contig-start intervals.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Regenerate the affected characterization snapshots

**Files:**
- Modify (regenerate): `tests/dataset/_snapshots/` entries for `tracks_ragged`, `tracks_fixed`, `haps_tracks_fixed`, `haps_tracks_ragged`
- Reference: `tests/dataset/test_flat_getitem_snapshot.py`

**Interfaces:**
- Consumes: the corrected reader from Task 2.
- Produces: updated golden snapshots reflecting the one-base edge correction; seq-only snapshots untouched.

- [ ] **Step 1: Confirm exactly the 4 track snapshots fail (and why)**

```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -v
```
Expected: FAIL only on `tracks_ragged`, `tracks_fixed`, `haps_tracks_fixed`, `haps_tracks_ragged`; all seq-only cases (`haps_*`, `ref_fixed`, `annot_fixed`, `variants_ragged`) PASS. If any non-track case fails, STOP — the fix changed more than coordinates; investigate before regenerating.

- [ ] **Step 2: Hand-verify a diff is a one-base edge correction**

Inspect one failing case to confirm the new output extends a filled run by exactly one base at interval edges (not a wholesale change):

```bash
pixi run -e dev pytest "tests/dataset/test_flat_getitem_snapshot.py::test_getitem_snapshot[tracks_ragged-view4-ragged]" -v 2>&1 | sed -n '1,40p'
```
Expected: the mismatch is a run of values shifted/extended by one position at each interval boundary, consistent with removing the `-1`.

- [ ] **Step 3: Locate the regeneration mechanism**

```bash
sed -n '1,5p' tests/dataset/test_flat_getitem_snapshot.py
grep -n "SNAP\|np.savez\|exists\|os.environ\|getenv" tests/dataset/test_flat_getitem_snapshot.py
```
The header says "First run (no snapshot present) writes reference .npz files from the CURRENT code." So regenerate by deleting the 4 stale snapshot files for the track cases and re-running (the test re-writes any missing snapshot). Confirm the on-disk filenames from the `SNAP` path and the `_save`/`_load` naming used in the test before deleting.

- [ ] **Step 4: Delete the 4 stale snapshots and regenerate**

Using the filenames confirmed in Step 3 (one `.npz` per failing case key — `tracks_ragged`, `tracks_fixed`, `haps_tracks_fixed`, `haps_tracks_ragged`):

```bash
# Example — adjust to the exact paths/extension confirmed in Step 3:
ls tests/dataset/_snapshots/
# remove only the four track-case snapshot files, then:
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -v
```
Expected: the 4 cases regenerate and the whole file PASSES on a second run:
```bash
pixi run -e dev pytest tests/dataset/test_flat_getitem_snapshot.py -v
```
Expected: ALL PASS.

- [ ] **Step 5: Sanity-check the git diff touches only the 4 track snapshots**

```bash
git status --porcelain tests/dataset/_snapshots/
```
Expected: only the 4 track-case files changed. If any seq-only snapshot changed, STOP and investigate.

- [ ] **Step 6: Commit the regenerated snapshots**

```bash
git add tests/dataset/_snapshots/
git commit -m "test: regenerate track snapshots after #233 bigWig coordinate fix

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: Full-tree verification

**Files:** none (verification only).

**Interfaces:**
- Consumes: Tasks 1–3.
- Produces: green full test tree + lint/type checks, ready to push/PR.

- [ ] **Step 1: Run the full Python tree**

```bash
pixi run -e dev pytest tests -q 2>&1 | tail -30
```
Expected: PASS (or only pre-existing unrelated skips). Investigate any failure that references bigWig/track/interval coordinates.

- [ ] **Step 2: Run cargo tests**

```bash
pixi run -e dev cargo test 2>&1 | tail -20
```
Expected: PASS.

- [ ] **Step 3: Lint and type-check (python + tests)**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
```
Expected: clean. Commit any formatting-only changes:
```bash
git add -A && git commit -m "style: ruff format after #233 fix

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

- [ ] **Step 4: Confirm the issue's own reproduction is resolved**

Re-run the issue's minimal repro shape (synthetic binned bigWig) and confirm `readback ≈ source` for each span (already asserted by the Task 1 test; this is a final eyeball). Done.

---

## Self-Review

**Spec coverage:**
- Root-cause fix (`src/bigwig.rs` `start - 1` → `start`) → Task 2. ✓
- Minimal scope, no kernel clamp → respected (no `_intervals.py` change). ✓
- Regression test on non-experimental bigWig annot path → Task 1. ✓
- Snapshot regeneration with diff verification → Task 3. ✓
- No SKILL.md / API change → Global Constraints. ✓
- Full-tree + cargo + lint + type verification → Task 4. ✓

**Placeholder scan:** Task 3 Steps 3–4 intentionally instruct the engineer to confirm exact snapshot filenames before deleting (the on-disk naming is not assumed); all other steps carry concrete code/commands. No TBD/TODO.

**Type consistency:** Test uses `fr.data`/`fr.offsets` (flat result), `ds.n_regions`, `BigWigs(name, {sample: path})`, `gvl.write(..., tracks=[...], annot_tracks={...}, overwrite=True)` — consistent across tasks.
