# Track-Write Window Input Floor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the variant writers from truncating each region's stored `chromEnd` below the input window, so annotation/sample tracks are written over (at least) the user's input regions and read back without a zeroed tail.

**Architecture:** The variant writers (`_write_from_vcf`/`_write_from_pgen` via shared helpers, and `_write_from_svar` inline) set each region's stored `chromEnd` to the furthest retained variant's end. `regions.npy` — read only by the track writers — therefore truncates below the input window. Floor the stored `chromEnd` at the input window end in all three sites; the read path already clips to the input window, so no read change is needed. Add a soft open-time warning so existing (corrupt) variant+track datasets are flagged.

**Tech Stack:** Python, polars, numpy, pyBigWig (tests), genoray (`VCF`/`PGEN`/`SparseVar`), seqpro (`sp.bed.sort`), loguru, pytest, pixi.

## Global Constraints

- Run everything via pixi: `pixi run -e dev <cmd>` (e.g. `pixi run -e dev pytest ...`). Platform linux-64.
- No Rust change in this plan; the dev env already has the extension built. If imports fail, run `pixi run -e dev maturin develop` once.
- On-disk **format unchanged** — do NOT bump `DATASET_FORMAT_VERSION`. The fix is behavior-only; existing variant+track datasets must be rewritten by the user.
- `gvl.write` defaults `max_jitter=None` (no jitter); tests rely on this so `regions.npy` chromEnd == input chromEnd.
- Read path (`_dataset/_reconstruct.py`, `_tracks.py`, `_intervals.py`) is out of scope — do not modify.
- Lint before any push: `pixi run -e dev ruff check python/ tests/` and `pixi run -e dev ruff format python/ tests/` (a pre-push hook enforces `ruff format`).
- Before pushing: full tree `pixi run -e dev pytest tests -q` (scoped runs skip `tests/unit/`).
- All new tests live in `tests/integration/tracks/test_track_window_floor.py`.

---

### Task 1: Floor stored `chromEnd` at the input window (the fix)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`_region_end` ~`:649-658`, `_region_ends_from_list` ~`:661-671`, `_write_from_svar` return ~`:1033`)
- Test: `tests/integration/tracks/test_track_window_floor.py`

**Interfaces:**
- Consumes: `gvl.write(path, bed, variants=...)`; fixtures `vcf_dir`, `pgen_dir`, `filtered_svar`, `ref_fasta`; `regions.npy` is `(n_regions, 4)` int32 in writer-sorted order, columns `[contig_idx, chromStart, chromEnd, strand]`.
- Produces: shared test helpers `_open_variants(source, vcf_dir, pgen_dir, filtered_svar)` and `_chr1_len(ref_fasta)` reused by later tasks.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/tracks/test_track_window_floor.py`:

```python
import numpy as np
import polars as pl
import pyBigWig
import pytest
import seqpro as sp
from genoray import PGEN, VCF, SparseVar
from loguru import logger

import genvarloader as gvl

VARIANT_SOURCES = ["vcf", "pgen", "svar"]


def _open_variants(source, vcf_dir, pgen_dir, filtered_svar):
    if source == "vcf":
        return VCF(vcf_dir / "filtered_source.vcf.gz")
    if source == "pgen":
        return PGEN(pgen_dir / "filtered_source.pgen")
    return SparseVar(filtered_svar)


def _chr1_len(ref_fasta):
    ref = gvl.Reference.from_path(ref_fasta, in_memory=False)
    return int(dict(zip(ref.contigs, np.diff(ref.offsets)))["chr1"])


@pytest.mark.parametrize("source", VARIANT_SOURCES)
def test_stored_window_floored_to_input(
    source, vcf_dir, pgen_dir, filtered_svar, ref_fasta, tmp_path
):
    chr1_len = _chr1_len(ref_fasta)
    # One wide region spanning the chr1 variant cluster out to the contig end.
    # Its tail is variant-free, so a pre-fix writer truncates chromEnd to the
    # rightmost variant (< chr1_len); the fix floors it at the input end.
    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [100], "chromEnd": [chr1_len]}
    )
    variants = _open_variants(source, vcf_dir, pgen_dir, filtered_svar)
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=variants, overwrite=True)

    regions = np.load(out / "regions.npy")  # (n, 4) int32, writer-sorted order
    input_end = sp.bed.sort(bed)["chromEnd"].to_numpy()  # same sorted order
    assert (regions[:, 2] >= input_end).all(), (
        f"{source}: stored chromEnd {regions[:, 2].tolist()} truncated below "
        f"input {input_end.tolist()}"
    )
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py::test_stored_window_floored_to_input -v`
Expected: FAIL for all three params (`vcf`, `pgen`, `svar`) — stored chromEnd is the rightmost-variant end, below `chr1_len`.

- [ ] **Step 3: Floor the two shared helpers**

In `python/genvarloader/_dataset/_write.py`, `_region_end` — change the final return:

```python
    if rag.data.size == 0:
        return int(fallback_end)
    return max(int(fallback_end), int(v_ends[int(rag.data.max())]))
```

`_region_ends_from_list` — change the final return:

```python
    if max_idx < 0:
        return int(fallback_end)
    return max(int(fallback_end), int(v_ends[max_idx]))
```

- [ ] **Step 4: Floor the SVAR writer**

In `_write_from_svar`, replace the return at ~`:1033`:

```python
    return bed.with_columns(
        chromEnd=pl.max_horizontal(pl.Series(max_ends), pl.col("chromEnd"))
    ), svar_link
```

(`bed` here is `gvl_bed`, whose `chromEnd` is the input window end + `max_jitter`; flooring the per-region `max_ends` against it is correct and a no-op for the already-input-valued no-variant rows.)

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py::test_stored_window_floored_to_input -v`
Expected: PASS for `vcf`, `pgen`, `svar`.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_write.py tests/integration/tracks/test_track_window_floor.py
git commit -m "fix: floor track-write window at the input region (#233 follow-up)

Variant writers set stored chromEnd to the furthest retained variant, which
can fall short of the input region end, so tracks were written over a
truncated window. Floor chromEnd at the input window in _region_end,
_region_ends_from_list, and _write_from_svar.

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 2: End-to-end annot-track tail regression

**Files:**
- Test: `tests/integration/tracks/test_track_window_floor.py` (append)

**Interfaces:**
- Consumes: the fix from Task 1; `_chr1_len`; `gvl.Dataset.open(...).with_seqs(None).with_output_format("flat").with_settings(realign_tracks=False).with_tracks([...])`.
- Produces: nothing downstream.

- [ ] **Step 1: Write the regression test**

Append to `tests/integration/tracks/test_track_window_floor.py`:

```python
def test_annot_track_tail_not_truncated_by_variants(vcf_dir, ref_fasta, tmp_path):
    chr1_len = _chr1_len(ref_fasta)
    # Constant-signal bigWig over all of chr1: every position reads 0.5.
    bw_path = tmp_path / "sig.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", chr1_len)])
    bw.addEntries(["chr1"], [0], ends=[chr1_len], values=[0.5])
    bw.close()

    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [100], "chromEnd": [chr1_len]}
    )
    out = tmp_path / "ds.gvl"
    gvl.write(
        out,
        bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        annot_tracks={"sig": str(bw_path)},
        overwrite=True,
    )

    fr = (
        gvl.Dataset.open(out, ref_fasta)
        .with_seqs(None)
        .with_output_format("flat")
        .with_settings(realign_tracks=False)
        .with_tracks(["sig"])
    )[0:1, 0]
    data, offs = np.asarray(fr.data), np.asarray(fr.offsets)
    seg = data[offs[0] : offs[1]]

    width = chr1_len - 100
    assert seg.shape[0] == width
    # Pre-fix the tail past the rightmost variant reads back 0; post-fix it is
    # fully covered at the source value.
    assert np.count_nonzero(seg) == width
    assert np.allclose(seg, 0.5)
```

- [ ] **Step 2: Run the test to verify it passes (with the fix)**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py::test_annot_track_tail_not_truncated_by_variants -v`
Expected: PASS. (Sanity that it is a real regression: temporarily revert Task 1's three edits → this test FAILS with `count_nonzero < width`; then restore.)

- [ ] **Step 3: Commit**

```bash
git add tests/integration/tracks/test_track_window_floor.py
git commit -m "test: annot track tail no longer zeroed by variant-truncated window (#233)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 3: Open-time warning for legacy truncated datasets

**Files:**
- Modify: `python/genvarloader/_dataset/_open.py` (`resolve` ~`:61-94`; add `_warn_truncated_tracks` method)
- Test: `tests/integration/tracks/test_track_window_floor.py` (append)

**Interfaces:**
- Consumes: `self._has_genotypes()`, `self._has_intervals()`, `self.path`, `metadata.max_jitter`, `metadata.version`, and `regions` (sorted input, no jitter) from `_build_indexer`.
- Produces: a `logger.warning` (loguru) containing the word "truncated" when stored `regions.npy` chromEnd is below the input floor.

- [ ] **Step 1: Write the failing test**

Append to `tests/integration/tracks/test_track_window_floor.py`:

```python
def _capture_warnings():
    msgs: list[str] = []
    sink_id = logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    return msgs, sink_id


def test_warns_on_truncated_track_window(vcf_dir, ref_fasta, tmp_path):
    chr1_len = _chr1_len(ref_fasta)
    bw_path = tmp_path / "sig.bw"
    bw = pyBigWig.open(str(bw_path), "w")
    bw.addHeader([("chr1", chr1_len)])
    bw.addEntries(["chr1"], [0], ends=[chr1_len], values=[0.5])
    bw.close()
    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [100], "chromEnd": [chr1_len]}
    )
    out = tmp_path / "ds.gvl"
    gvl.write(
        out,
        bed,
        variants=VCF(vcf_dir / "filtered_source.vcf.gz"),
        annot_tracks={"sig": str(bw_path)},
        overwrite=True,
    )

    # A clean (post-fix) dataset must NOT warn.
    clean_msgs, sid = _capture_warnings()
    try:
        gvl.Dataset.open(out, ref_fasta)
    finally:
        logger.remove(sid)
    assert not any("truncat" in m.lower() for m in clean_msgs)

    # Simulate a legacy writer: pull each stored chromEnd below the input window.
    regions = np.load(out / "regions.npy")
    regions[:, 2] = regions[:, 2] - 50
    np.save(out / "regions.npy", regions)

    dirty_msgs, sid = _capture_warnings()
    try:
        gvl.Dataset.open(out, ref_fasta)
    finally:
        logger.remove(sid)
    assert any("truncat" in m.lower() for m in dirty_msgs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py::test_warns_on_truncated_track_window -v`
Expected: FAIL on the second assertion — the truncated dataset opens without any warning (no detector exists yet).

- [ ] **Step 3: Add the detector method**

In `python/genvarloader/_dataset/_open.py`, add a method to the open-request class (same class that defines `resolve`/`_build_indexer`), e.g. right after `_build_tracks`:

```python
    def _warn_truncated_tracks(
        self, metadata: Metadata, regions: NDArray[np.int32]
    ) -> None:
        """Warn when a dataset's stored track windows were truncated below the
        input regions (variant+track datasets written before the chromEnd-floor
        fix). Such datasets silently drop track signal past each region's
        rightmost variant and must be rewritten with ``gvl.write``/``gvl.update``.
        """
        if not (self._has_genotypes() and self._has_intervals()):
            return
        stored = np.load(self.path / "regions.npy", mmap_mode="r")
        floor = regions[:, 2] + (metadata.max_jitter or 0)
        if bool((stored[:, 2] < floor).any()):
            logger.warning(
                f"Dataset at {self.path} (written by genvarloader "
                f"{metadata.version}) has track windows truncated below its input "
                f"regions: track signal past each region's rightmost variant is "
                f"missing. Rewrite with `gvl.write` / `gvl.update` to fix."
            )
```

If `Metadata` / `NDArray` are not already imported in `_open.py`, add them (`from ._write import Metadata`; `from numpy.typing import NDArray`) — check existing imports first and reuse.

- [ ] **Step 4: Call the detector in `resolve`**

In `resolve`, immediately after `idxer, bed, regions = self._build_indexer(metadata)`:

```python
        idxer, bed, regions = self._build_indexer(metadata)
        self._warn_truncated_tracks(metadata, regions)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py::test_warns_on_truncated_track_window -v`
Expected: PASS (clean open silent, truncated open warns).

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_open.py tests/integration/tracks/test_track_window_floor.py
git commit -m "feat: warn when opening datasets with variant-truncated track windows (#233)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

### Task 4: `Dataset.regions == input` invariant guard + skill re-check

**Files:**
- Test: `tests/integration/tracks/test_track_window_floor.py` (append)
- Check (maybe modify): `skills/genvarloader/SKILL.md`

**Interfaces:**
- Consumes: `gvl.Dataset.open(...).regions`; fixtures `vcf_dir`, `ref_fasta`.
- Produces: nothing downstream.

- [ ] **Step 1: Write the invariant guard test**

Append to `tests/integration/tracks/test_track_window_floor.py`:

```python
def test_dataset_regions_match_input_with_variants(vcf_dir, ref_fasta, tmp_path):
    chr1_len = _chr1_len(ref_fasta)
    bed = pl.DataFrame(
        {
            "chrom": ["chr1", "chr1"],
            "chromStart": [100, 300],
            "chromEnd": [250, chr1_len],
        }
    )
    out = tmp_path / "ds.gvl"
    gvl.write(
        out, bed, variants=VCF(vcf_dir / "filtered_source.vcf.gz"), overwrite=True
    )
    ds = gvl.Dataset.open(out, ref_fasta)
    got = ds.regions.select("chrom", "chromStart", "chromEnd")
    assert got.to_dicts() == bed.to_dicts()
```

- [ ] **Step 2: Run the test (already-true invariant — expect PASS)**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py::test_dataset_regions_match_input_with_variants -v`
Expected: PASS. This pins the existing guarantee that `Dataset.regions` reflects `input_regions.arrow` (the true input), independent of the variant-storage window. If it FAILS, stop — something feeds the variant-extended bed into `input_regions.arrow` and the spec's root-cause assumption is wrong.

- [ ] **Step 3: Re-check the skill for tracks+variants gotchas**

Read `skills/genvarloader/SKILL.md`. No public-API signature/default changed, so no surface edit is expected. If there is a "Common gotchas" entry about tracks with variants (or annot tracks), add one sentence: tracks are stored over at least the input window, and `realign_tracks=False` returns track values over the input region. If no such section fits cleanly, leave the skill unchanged and note that in the commit.

- [ ] **Step 4: Run the whole new test file**

Run: `pixi run -e dev pytest tests/integration/tracks/test_track_window_floor.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Lint + full tree**

```bash
pixi run -e dev ruff check python/ tests/
pixi run -e dev ruff format python/ tests/
pixi run -e dev pytest tests -q
```
Expected: clean lint, format no-op (or fold changes into the commit), full suite green.

- [ ] **Step 6: Commit**

```bash
git add tests/integration/tracks/test_track_window_floor.py skills/genvarloader/SKILL.md
git commit -m "test: pin Dataset.regions == input invariant for variant datasets (#233)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- Floor at three writer sites (`_region_end`, `_region_ends_from_list`, `_write_from_svar`) → Task 1.
- `chromStart` not modified → no task needed (left untouched by design).
- Read path untouched → enforced by Global Constraints; verified by Task 2 passing with no read-side edit.
- Regression across VCF/PGEN/SVAR → Task 1 (parametrized `regions.npy` floor) + Task 2 (VCF end-to-end annot tail). VCF/PGEN share the helper; SVAR has its own edit; all three are written and read in Task 1.
- Sample-track parity (latent same bug) → covered by Task 1 at the storage level (sample + annot writers both consume `regions.npy`, now floored); no separate read assertion (sample-track realign is out of scope, and `realign=False` sample reads use the same kernel proven in Task 2).
- `Dataset.regions == input` invariant → Task 4.
- Open-time silent-corruption warning, no `format_version` bump → Task 3.
- Breaking-change/rewrite remedy → encoded in the warning message (Task 3) and commit messages.

**Placeholder scan:** No TBD/TODO; every code step shows full code; commands have expected output. Task 4 Step 3 is conditional by nature (skill may need no edit) but states the exact decision rule and fallback.

**Type consistency:** `_open_variants`/`_chr1_len` defined in Task 1 and reused verbatim in Tasks 2–4. `_warn_truncated_tracks(self, metadata, regions)` signature matches its call site in `resolve`. `regions.npy` is treated as `(n, 4)` int32 with chromEnd at column 2 throughout.

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-19-track-write-window-input-floor.md`.
