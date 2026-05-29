# Fix #176: splice_info + var_filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `Dataset.open(splice_info=..., var_filter="exonic")` and `.with_seqs("haplotypes").with_settings(splice_info=..., var_filter="exonic")` produce identical, correct results on SVAR-backed data.

**Architecture:** Three fixes: (1) correct 2-D indexing in `choose_exonic_variants`; (2) make `with_settings(var_filter=...)` propagate to `_recon` while preserving `_recon.kind`; (3) unify `Dataset.open`'s splice/filter configuration through `with_settings` so there's one source of truth.

**Tech Stack:** Python 3.10+, numba JIT kernels, attrs `evolve`, polars + awkward arrays, pytest. All commands assume the `pixi` package manager (`pixi run -e dev <task>`).

**Spec:** `docs/superpowers/specs/2026-05-21-splice-exonic-var-filter-design.md`

**Branch:** `worktree-fix-176-splice-exonic-filter` (worktree at `.claude/worktrees/fix-176-splice-exonic-filter`)

---

## Pre-flight

- [ ] **Step 0.1: Confirm worktree state**

Run: `git status && git log --oneline -3`
Expected: clean worktree on `worktree-fix-176-splice-exonic-filter`, top commit is `docs(specs): design for #176 splice + var_filter fix`.

- [ ] **Step 0.2: Confirm SVAR test fixture exists**

Run: `ls tests/data/filtered.svar tests/data/source.bed tests/data/fasta/hg38.fa.bgz`
Expected: all three exist. If `filtered.svar` is missing, run `pixi run -e dev gen` and wait for it to finish.

- [ ] **Step 0.3: Baseline test run**

Run: `pixi run -e dev pytest tests/dataset/genotypes/test_choose_exonic_variants.py -v`
Expected: 2 tests pass (these are the existing tests; they pass against the buggy code because of the wrong 2-D layout — see spec Bug 3).

---

## Task 1: Reproduce Bug 2 with a layout-realistic test

Goal: write a failing test that exercises `choose_exonic_variants` with the real `(2, n_slices)` SVAR offsets layout, with `n_slices > 2` so wrong indexing cannot accidentally yield 2 elements. This test will fail on current `main` and pass after Task 2.

**Files:**
- Modify: `tests/dataset/genotypes/test_choose_exonic_variants.py`

- [ ] **Step 1.1: Replace the 2-D test with a real-layout regression**

Open `tests/dataset/genotypes/test_choose_exonic_variants.py`. Replace the body of `test_choose_exonic_variants_2d_geno_offsets` (lines 45–55) so it builds offsets in the SVAR `(2, n_slices)` shape with `n_slices = 3`. The fixture must logically describe: 3 sparse-genotype slices total. The 1 region × ploidy 2 fixture only consumes the first two slices (`o_idx = 0, 1`); the third slice is padding that makes the row length 3, so wrong indexing (`geno_offsets[o_idx]` returning a 3-vector) fails loud.

Replace the test (and the docstring) with:

```python
def test_choose_exonic_variants_2d_geno_offsets() -> None:
    """SVAR offsets are shape (2, n_slices). Wrong indexing returns a
    length-n_slices row and produces garbage o_s/o_e. Use n_slices > 2
    so wrong indexing cannot accidentally yield 2 elements.
    """
    inputs = _common_inputs()
    # Real SVAR layout: shape (2, n_slices). Row 0 is starts, row 1 is ends.
    # 3 slices total; region 0, haplotype 0 -> slice 0 = [0, 1).
    # region 0, haplotype 1 -> slice 1 = [1, 2). Slice 2 is unused padding.
    inputs["geno_offsets"] = np.asarray(
        [
            [0, 1, 2],  # starts
            [1, 2, 2],  # ends
        ],
        dtype=np.int64,
    )
    keep, keep_offsets = choose_exonic_variants(**inputs)
    # Logical content identical to the 1-D case: 2 variants kept.
    assert keep_offsets.shape == (3,)  # n_regions * ploidy + 1
    assert keep_offsets.tolist() == [0, 1, 2]
    assert keep.tolist() == [True, True]
```

- [ ] **Step 1.2: Run the new test to confirm it fails on buggy code**

Run: `pixi run -e dev pytest tests/dataset/genotypes/test_choose_exonic_variants.py::test_choose_exonic_variants_2d_geno_offsets -v`

Expected: FAIL. Likely failure modes are:
- `ValueError: negative dimensions not allowed` from `keep = np.empty(n_variants, ...)`, OR
- numba-internal error from unpacking a 3-vector into 2 names, OR
- assertion failure on `keep_offsets.tolist()` because the wrong indexing produced bogus lengths.

If it PASSES on current code, stop — the fixture is not actually triggering Bug 2. Verify `n_slices > 2`, that `geno_offsets` is shape `(2, 3)` not `(3, 2)`, and that `geno_offset_idxs` points to the first two columns.

- [ ] **Step 1.3: Confirm the 1-D test still passes**

Run: `pixi run -e dev pytest tests/dataset/genotypes/test_choose_exonic_variants.py::test_choose_exonic_variants_1d_geno_offsets -v`
Expected: PASS.

- [ ] **Step 1.4: Commit the failing test**

```bash
git add tests/dataset/genotypes/test_choose_exonic_variants.py
git commit -m "test(choose_exonic_variants): rebuild 2-D fixture on real (2, n_slices) SVAR layout

The previous fixture used (total_variants, 2) which accidentally made
geno_offsets[o_idx] return a 2-element row. Real SVAR offsets are
(2, n_slices) per Haps.from_path; with n_slices > 2 the existing
indexing returns a row that can't unpack into (o_s, o_e), exposing #176."
```

---

## Task 2: Fix `choose_exonic_variants` 2-D indexing

Goal: change `geno_offsets[o_idx]` to `geno_offsets[:, o_idx]` in both branches; verify Task 1's test now passes.

**Files:**
- Modify: `python/genvarloader/_dataset/_genotypes.py:455-458, 479-482`

- [ ] **Step 2.1: Patch the first loop**

In `python/genvarloader/_dataset/_genotypes.py`, find the first occurrence (around line 455):

```python
            if geno_offsets.ndim == 1:
                o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            else:
                o_s, o_e = geno_offsets[o_idx]
```

Change the `else` branch to:

```python
            if geno_offsets.ndim == 1:
                o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            else:
                o_s, o_e = geno_offsets[:, o_idx]
```

- [ ] **Step 2.2: Patch the second loop**

Find the second occurrence (around line 479), inside the second `prange` loop. The block currently reads:

```python
            if geno_offsets.ndim == 1:
                o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            else:
                o_s, o_e = geno_offsets[o_idx]
```

Change to:

```python
            if geno_offsets.ndim == 1:
                o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            else:
                o_s, o_e = geno_offsets[:, o_idx]
```

Also remove the now-misleading comment above the second block (lines ~472–478) that talks about the slice/array typing — keep just a short one-liner referencing the sibling kernel:

```python
            # Mirror filter_af's (2, n_slices) indexing (sibling kernel below).
            if geno_offsets.ndim == 1:
                o_s, o_e = geno_offsets[o_idx], geno_offsets[o_idx + 1]
            else:
                o_s, o_e = geno_offsets[:, o_idx]
```

- [ ] **Step 2.3: Clear numba cache and re-run the failing test**

Numba caches compiled kernels by file hash so just re-running should pick up the change, but clear to be safe:

Run: `find python/genvarloader -name __pycache__ -prune -exec rm -rf {} + ; rm -rf ~/.numba_cache 2>/dev/null; pixi run -e dev pytest tests/dataset/genotypes/test_choose_exonic_variants.py -v`

Expected: both tests PASS.

- [ ] **Step 2.4: Commit the kernel fix**

```bash
git add python/genvarloader/_dataset/_genotypes.py
git commit -m "fix(choose_exonic_variants): use (2, n_slices) indexing for 2-D offsets

SVAR offsets are constructed as offsets.reshape(2, -1) in
Haps.from_path. geno_offsets[o_idx] returns a length-n_slices row,
not a 2-tuple, producing garbage o_s/o_e -> MemoryError or
negative-dim ValueError. Mirror filter_af's geno_offsets[:, o_idx].

Fixes #176 (kernel half)."
```

---

## Task 3: Make `with_settings(var_filter=...)` propagate to `_recon`

Goal: when `var_filter` evolves `_seqs`, also evolve `_recon`'s underlying `Haps` (preserving `_recon.kind`). Unit-test this directly before changing `Dataset.open`.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py:453-463`
- Create: `tests/dataset/test_with_settings_var_filter.py`

- [ ] **Step 3.1: Write a failing unit test for `_recon.filter` propagation**

Create `tests/dataset/test_with_settings_var_filter.py`:

```python
"""Direct probe for #176 Bug 1: with_settings(var_filter=...) must update _recon, not just _seqs.

After .with_seqs("haplotypes"), _recon is a separate Haps instance
(haps.to_kind(RaggedSeqs) returns a fresh instance via evolve). The
old with_settings only evolved self._seqs, silently dropping the
filter from the code path that __getitem__ actually exercises.
"""

from pathlib import Path

import genvarloader as gvl
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"


@pytest.fixture
def svar_gvl_path(tmp_path):
    svar_path = _DATA_DIR / "filtered.svar"
    bed_path = _DATA_DIR / "source.bed"
    assert svar_path.is_dir(), f"missing fixture {svar_path}; run pixi run -e dev gen"
    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed_path, variants=svar_path, overwrite=True)
    return out


def test_with_settings_var_filter_propagates_to_recon(svar_gvl_path):
    ref_path = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset
        .open(svar_gvl_path, reference=ref_path)
        .with_seqs("haplotypes")
        .with_settings(var_filter="exonic")
    )
    # The bug: previously _recon.filter stayed None.
    assert ds._seqs.filter == "exonic"
    assert ds._recon.filter == "exonic", (
        "with_settings(var_filter=...) failed to propagate to _recon; "
        "__getitem__ uses _recon, so the filter would be silently dropped."
    )


def test_with_settings_var_filter_false_clears_recon(svar_gvl_path):
    ref_path = _DATA_DIR / "fasta" / "hg38.fa.bgz"
    ds = (
        gvl.Dataset
        .open(svar_gvl_path, reference=ref_path)
        .with_seqs("haplotypes")
        .with_settings(var_filter="exonic")
        .with_settings(var_filter=False)
    )
    assert ds._seqs.filter is None
    assert ds._recon.filter is None
```

- [ ] **Step 3.2: Run to confirm it fails on current code**

Run: `pixi run -e dev pytest tests/dataset/test_with_settings_var_filter.py -v`

Expected: `test_with_settings_var_filter_propagates_to_recon` FAILS with `assert None == 'exonic'`. The second test may also fail or pass trivially — both must pass after the fix.

- [ ] **Step 3.3: Patch `with_settings` to propagate `var_filter` to `_recon`**

In `python/genvarloader/_dataset/_impl.py`, locate the var_filter block (around lines 453–463):

```python
        if var_filter is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Filtering variants can only be done when the dataset has variants."
                )

            if var_filter is False:
                var_filter = None

            if var_filter != self._seqs.filter:
                to_evolve["_seqs"] = evolve(self._seqs, filter=var_filter)
```

Replace with:

```python
        if var_filter is not None:
            if not isinstance(self._seqs, Haps):
                raise ValueError(
                    "Filtering variants can only be done when the dataset has variants."
                )

            if var_filter is False:
                var_filter = None

            if var_filter != self._seqs.filter:
                haps = to_evolve.get("_seqs", self._seqs)
                to_evolve["_seqs"] = evolve(haps, filter=var_filter)

                # Propagate to _recon, preserving its kind (set by with_seqs).
                # We must not replace _recon with _seqs wholesale — _recon has
                # a different kind (e.g. RaggedSeqs) than _seqs (RaggedVariants).
                if isinstance(self._recon, Haps):
                    recon_haps = to_evolve.get("_recon", self._recon)
                    to_evolve["_recon"] = evolve(recon_haps, filter=var_filter)
                elif isinstance(self._recon, HapsTracks):
                    recon = to_evolve.get("_recon", self._recon)
                    new_haps = evolve(recon.haps, filter=var_filter)
                    to_evolve["_recon"] = evolve(recon, haps=new_haps)
```

- [ ] **Step 3.4: Verify the import of `HapsTracks` is already present**

Run: `grep -n "HapsTracks" python/genvarloader/_dataset/_impl.py | head`
Expected: `HapsTracks` is already imported and used in the file. If not, add it to the existing `from ._reconstruct import (...)` block.

- [ ] **Step 3.5: Run the unit tests to confirm they pass**

Run: `pixi run -e dev pytest tests/dataset/test_with_settings_var_filter.py -v`
Expected: both tests PASS.

- [ ] **Step 3.6: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py tests/dataset/test_with_settings_var_filter.py
git commit -m "fix(with_settings): propagate var_filter to _recon, preserving kind

After with_seqs(...), _recon is a separate Haps instance with the
user-chosen kind. __getitem__ uses _recon, not _seqs, so updating
only _seqs silently dropped the filter (issue #176 path B).

Mirror the propagation for the HapsTracks recon shape.

Fixes #176 (with_settings half)."
```

---

## Task 4: Unify `Dataset.open` with `with_settings` for splice/var_filter

Goal: `Dataset.open` should configure `splice_info` and `var_filter` by delegating to `self.with_settings(...)` after initial construction, eliminating the two-code-path divergence.

**Files:**
- Modify: `python/genvarloader/_dataset/_impl.py:124-326` (`open` method body, specifically the splice block at :264-278 and the `filter=var_filter` argument at :229)

- [ ] **Step 4.1: Read the current `open` body to confirm structure**

Run: `sed -n '215,326p' python/genvarloader/_dataset/_impl.py`

Confirm:
- Line ~229: `filter=var_filter` is passed to `Haps.from_path`.
- Lines ~264–278: splice_info block constructs `splice_idxer` / `spliced_bed`.
- Line ~314: `_sp_idxer=splice_idxer` and `_spliced_bed=spliced_bed` are passed to `RaggedDataset(...)`.

- [ ] **Step 4.2: Remove `filter=var_filter` from `Haps.from_path` call**

In `python/genvarloader/_dataset/_impl.py`, find the `Haps.from_path(...)` call around line 218–230. Change:

```python
            seqs = Haps.from_path(
                path=path,
                reference=reference,
                regions=regions,
                samples=samples,
                ploidy=ploidy,
                version=metadata.version,
                svar_link=metadata.svar_link,
                svar_override=svar,
                min_af=min_af,
                max_af=max_af,
                filter=var_filter,
            )
```

to (drop the `filter=var_filter` line):

```python
            seqs = Haps.from_path(
                path=path,
                reference=reference,
                regions=regions,
                samples=samples,
                ploidy=ploidy,
                version=metadata.version,
                svar_link=metadata.svar_link,
                svar_override=svar,
                min_af=min_af,
                max_af=max_af,
            )
```

- [ ] **Step 4.3: Remove the inline splice block**

Still in `Dataset.open`, find the splice block at ~lines 264–278:

```python
        if splice_info is not None:
            sm, spliced_bed = SpliceMap.from_bed(splice_info, bed)
            # SpliceIndexer._init performed a bounds check that `from_bed` does not.
            # Preserve it here.
            if (
                ak.max(sm.splice_map, None) >= idxer.n_regions
                or ak.min(sm.splice_map, None) < -idxer.n_regions
            ):
                raise ValueError(
                    "Found indices in the splice map that are out of bounds for the dataset."
                )
            splice_idxer = SpliceIndexer(map=sm, dsi=idxer)
        else:
            splice_idxer = None
            spliced_bed = None
```

Replace with just:

```python
        splice_idxer = None
        spliced_bed = None
```

(The actual splice configuration will be applied via `with_settings` after construction — Step 4.4.)

- [ ] **Step 4.4: Delegate splice/filter to `with_settings` after construction**

Find the end of `Dataset.open` (after the `RaggedDataset(...)` constructor at ~lines 304–322 and before `logger.info(...)` at ~line 324):

```python
        dataset = RaggedDataset(
            path=path,
            ...
            _rng=np.random.default_rng(rng),
        )

        logger.info(f"Opened dataset:\n{dataset}")

        return dataset
```

Insert a `with_settings` call between the constructor and the logger:

```python
        dataset = RaggedDataset(
            path=path,
            ...
            _rng=np.random.default_rng(rng),
        )

        if splice_info is not None or var_filter is not None:
            dataset = dataset.with_settings(
                splice_info=splice_info,
                var_filter=var_filter,
            )

        logger.info(f"Opened dataset:\n{dataset}")

        return dataset
```

- [ ] **Step 4.5: Run the existing splice tests as a smoke check**

Run: `pixi run -e dev pytest tests/dataset/test_rc_packing.py -v`
Expected: all existing tests still pass.

- [ ] **Step 4.6: Run the var_filter unit tests**

Run: `pixi run -e dev pytest tests/dataset/test_with_settings_var_filter.py -v`
Expected: both pass.

- [ ] **Step 4.7: Commit**

```bash
git add python/genvarloader/_dataset/_impl.py
git commit -m "refactor(Dataset.open): delegate splice_info/var_filter to with_settings

One configuration path. Eliminates the open()/with_settings divergence
that masked #176 — open() previously reached into Haps(filter=...) and
constructed SpliceIndexer inline; with_settings had its own copy
(broken for var_filter). Both now flow through with_settings."
```

---

## Task 5: End-to-end path-parity test

Goal: integration test that `Dataset.open(..., splice_info=..., var_filter="exonic")` and `Dataset.open(...).with_seqs("haplotypes").with_settings(splice_info=..., var_filter="exonic")` produce identical output.

**Files:**
- Create: `tests/dataset/test_open_vs_settings_parity.py`

- [ ] **Step 5.1: Write the path-parity test**

Create `tests/dataset/test_open_vs_settings_parity.py`:

```python
"""Regression test for #176: Dataset.open(splice_info, var_filter) must produce
the same result as .with_seqs("haplotypes").with_settings(splice_info, var_filter).

Pre-fix:
- Path A (open(splice_info, var_filter)): hit choose_exonic_variants with a
  2-D SVAR offsets layout, which was mis-indexed -> MemoryError /
  negative-dim ValueError.
- Path B (with_settings): silently dropped var_filter from _recon and produced
  un-filtered haplotypes.

Post-fix: both paths produce the same, correctly-filtered output.
"""

import shutil
from pathlib import Path

import awkward as ak
import genvarloader as gvl
import polars as pl
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "data"


@pytest.fixture(scope="module")
def spliced_svar_ds_path(tmp_path_factory):
    """Build an SVAR-backed GVL store with per-region single-exon transcripts.

    Each BED row becomes its own single-exon transcript so SpliceMap has
    something to do, and the SVAR backend ensures 2-D geno_offsets are
    exercised end-to-end.
    """
    svar_path = _DATA_DIR / "filtered.svar"
    bed_path = _DATA_DIR / "source.bed"
    assert svar_path.is_dir(), f"missing fixture {svar_path}; run pixi run -e dev gen"

    tmp = tmp_path_factory.mktemp("issue_176_parity")
    out = tmp / "ds.gvl"
    gvl.write(path=out, bed=bed_path, variants=svar_path, overwrite=True)

    # Inject transcript_id / exon_number so SpliceMap.from_bed can resolve.
    regions_path = out / "input_regions.arrow"
    bed = pl.read_ipc(regions_path)
    bed = bed.with_columns(
        transcript_id=pl.arange(0, pl.len()).cast(pl.Utf8),
        exon_number=pl.lit(1, pl.Int32),
    )
    tmp_arrow = regions_path.with_suffix(".arrow.tmp")
    bed.write_ipc(tmp_arrow)
    shutil.move(tmp_arrow, regions_path)
    return out


def test_open_vs_with_settings_parity_state(spliced_svar_ds_path):
    """Internal state probe: both paths produce the same filter / spliced state."""
    ref_path = _DATA_DIR / "fasta" / "hg38.fa.bgz"

    ds_a = gvl.Dataset.open(
        spliced_svar_ds_path,
        reference=ref_path,
        splice_info=("transcript_id", "exon_number"),
        var_filter="exonic",
    ).with_seqs("haplotypes")

    ds_b = (
        gvl.Dataset
        .open(spliced_svar_ds_path, reference=ref_path)
        .with_seqs("haplotypes")
        .with_settings(
            splice_info=("transcript_id", "exon_number"),
            var_filter="exonic",
        )
    )

    assert ds_a._seqs.filter == ds_b._seqs.filter == "exonic"
    assert ds_a._recon.filter == ds_b._recon.filter == "exonic"
    assert (ds_a._sp_idxer is None) == (ds_b._sp_idxer is None)
    assert ds_a._sp_idxer is not None and ds_b._sp_idxer is not None


def test_open_vs_with_settings_parity_output(spliced_svar_ds_path):
    """Materialized output: __getitem__ must agree byte-for-byte."""
    ref_path = _DATA_DIR / "fasta" / "hg38.fa.bgz"

    ds_a = gvl.Dataset.open(
        spliced_svar_ds_path,
        reference=ref_path,
        splice_info=("transcript_id", "exon_number"),
        var_filter="exonic",
    ).with_seqs("haplotypes")

    ds_b = (
        gvl.Dataset
        .open(spliced_svar_ds_path, reference=ref_path)
        .with_seqs("haplotypes")
        .with_settings(
            splice_info=("transcript_id", "exon_number"),
            var_filter="exonic",
        )
    )

    haps_a = ds_a[0, :].to_ak()
    haps_b = ds_b[0, :].to_ak()

    assert ak.all(haps_a == haps_b), (
        "Path A (open(splice_info, var_filter)) and Path B "
        "(with_seqs.with_settings) produced different output."
    )
```

- [ ] **Step 5.2: Run the parity tests**

Run: `pixi run -e dev pytest tests/dataset/test_open_vs_settings_parity.py -v`
Expected: both tests PASS.

- [ ] **Step 5.3: Run the full dataset test suite as a regression check**

Run: `pixi run -e dev pytest tests/dataset/ -v`
Expected: all pass.

- [ ] **Step 5.4: Commit**

```bash
git add tests/dataset/test_open_vs_settings_parity.py
git commit -m "test(#176): assert Dataset.open and with_settings produce identical output

End-to-end regression: state probe (filter on _recon, splice indexer
present) and output equality on a small SVAR-backed dataset with
per-region single-exon transcripts. Catches both halves of #176 —
the kernel mis-indexing and the with_settings propagation gap."
```

---

## Task 6: Final verification

- [ ] **Step 6.1: Run the full test suite**

Run: `pixi run -e dev test`
Expected: all pass (pytest + cargo).

- [ ] **Step 6.2: Lint**

Run: `pixi run -e dev ruff check python/ tests/`
Expected: no errors (formatting/style).

- [ ] **Step 6.3: Type check**

Run: `pixi run -e dev basedpyright python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_genotypes.py`
Expected: no new errors (some pre-existing warnings are acceptable; verify the diff doesn't add new ones by comparing to `git stash && basedpyright ... ; git stash pop` if unsure).

- [ ] **Step 6.4: Review the diff**

Run: `git log --oneline main..HEAD && git diff main...HEAD --stat`
Expected commits in order:
1. `docs(specs): design for #176 ...` (pre-existing)
2. `test(choose_exonic_variants): rebuild 2-D fixture ...`
3. `fix(choose_exonic_variants): use (2, n_slices) indexing ...`
4. `fix(with_settings): propagate var_filter to _recon ...`
5. `refactor(Dataset.open): delegate splice_info/var_filter ...`
6. `test(#176): assert Dataset.open and with_settings produce identical output`

- [ ] **Step 6.5: Push and open PR (skip if user prefers to do this manually)**

Confirm with user before pushing. If approved:

```bash
git push -u origin worktree-fix-176-splice-exonic-filter
gh pr create --title "fix(#176): splice + var_filter parity between open() and with_settings()" --body "$(cat <<'EOF'
## Summary
- Fix `choose_exonic_variants` 2-D indexing (was `geno_offsets[o_idx]`, must be `geno_offsets[:, o_idx]` for real SVAR `(2, n_slices)` layout).
- Fix `with_settings(var_filter=...)` to propagate the filter to `_recon`, preserving `_recon.kind`.
- Unify `Dataset.open`'s `splice_info` / `var_filter` configuration through `with_settings` so there's one source of truth.

Closes #176.

## Test plan
- [x] New 2-D layout regression in `tests/dataset/genotypes/test_choose_exonic_variants.py` (n_slices > 2).
- [x] New unit tests in `tests/dataset/test_with_settings_var_filter.py` directly probe `_recon.filter` propagation.
- [x] New end-to-end parity tests in `tests/dataset/test_open_vs_settings_parity.py` assert `Dataset.open(splice_info, var_filter)` and `.with_seqs.with_settings(splice_info, var_filter)` produce identical output on an SVAR-backed dataset.
- [x] `pixi run -e dev test` passes.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes / gotchas

- **Numba caching.** `choose_exonic_variants` is `@nb.njit(..., cache=True)`. After editing it, stale cache can mask the fix; clear with `find python/genvarloader -name __pycache__ -prune -exec rm -rf {} +` and (optionally) `rm -rf ~/.numba_cache`.
- **`pre-existing _recon.kind clobber`.** `_impl.py:427-434` (min_af/max_af/var_fields recon rebuild) replaces `_recon` with `_seqs` wholesale, clobbering `_recon.kind`. The var_filter fix in Task 3 deliberately does *not* rely on that block. **Out of scope for this PR.**
- **`Haps.from_path` `filter=` argument.** After Task 4, `Dataset.open` no longer passes `filter=` to `Haps.from_path`. The argument still exists on `from_path` (Haps stores `filter` as a field) and is exercised indirectly via `with_settings`. We do not remove the `filter` parameter from `from_path` — it remains valid construction sugar, and removing it would be an unnecessary public-facing change.
- **Splice bed validation.** `Dataset.open`'s old splice block had an explicit bounds check (`splice_map indices vs idxer.n_regions`) at lines ~268-274 that `SpliceMap.from_bed` itself does not perform. `SpliceIndexer.__init__` (called from `with_settings:449`) is supposed to perform the equivalent check. Confirm during implementation that the bounds check is still triggered after the refactor — if it isn't, add it inside `with_settings`'s splice block (mirror the existing `if (ak.max(sm.splice_map, ...) >= self._idxer.n_regions or ...)` check at `_impl.py:442-448`, which is already present, so this should be fine — just sanity-check.
