# Index-aware `max_mem` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `gvl.write(..., max_mem=...)` account for the in-memory footprint of genoray's variant index so `max_mem` is a true total cap.

**Architecture:** Two-repo change. First, add a public `nbytes: int` property on genoray's `VCF`, `PGEN`, and `SparseVar` classes (returns resident-memory size; mmap'd data excluded). Release genoray. Then GVL subtracts `variants.nbytes` from `max_mem` before passing the remainder to chunk-sizing in `_write_from_vcf` / `_write_from_pgen`, and warns when the index dominates the budget.

**Tech Stack:** Python, polars (`DataFrame.estimated_size()`), numpy (`ndarray.nbytes`), pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-05-19-max-mem-index-aware-design.md`

---

## Repo layout

- **genoray repo:** `/Users/david/projects/genoray`
- **GVL repo:** `/Users/david/projects/GenVarLoader`

## Files affected

### genoray (`feat/nbytes` branch)
- Modify: `genoray/_vcf.py` (add `nbytes` property on `VCF`)
- Modify: `genoray/_pgen.py` (add `nbytes` property on `PGEN`)
- Modify: `genoray/_svar.py` (add `nbytes` property on `SparseVar`)
- Test: `tests/test_vcf.py` (assert `nbytes`)
- Test: `tests/test_pgen.py` (assert `nbytes`)
- Test: `tests/test_svar.py` (assert `nbytes`)

### GVL (separate branch, after genoray release)
- Modify: `python/genvarloader/_dataset/_write.py` (compute effective_max_mem, warn, pass through)
- Modify: `pyproject.toml` (bump `genoray>=<new version>`)
- Modify: `pixi.toml` (bump `genoray = "==<new version>"`)
- Test: `tests/dataset/test_write.py` (warning is emitted when index dominates max_mem)

---

## Phase 1 — genoray

### Task 1: Create feat/nbytes branch in genoray

**Files:** none

- [ ] **Step 1: Switch to genoray repo and create branch**

```bash
cd /Users/david/projects/genoray
rtk git checkout -b feat/nbytes
rtk git status
```

Expected: `On branch feat/nbytes` with clean working tree.

---

### Task 2: Add `VCF.nbytes` property (TDD)

**Files:**
- Modify: `genoray/_vcf.py` — add property on `VCF` class
- Test: `tests/test_vcf.py` — append new test

- [ ] **Step 1: Write the failing test**

Append to `tests/test_vcf.py`:

```python
def test_nbytes_zero_before_index_loaded():
    # _load_index is not called when with_gvi_index=False and no auto-load occurs
    vcf = VCF(ddir / "biallelic.vcf.gz", with_gvi_index=False)
    assert vcf._index is None
    assert vcf.nbytes == 0


def test_nbytes_positive_after_index_loaded():
    vcf = VCF(ddir / "biallelic.vcf.gz")
    if not vcf._valid_index():
        vcf._write_gvi_index()
    vcf._load_index()
    assert vcf._index is not None
    assert vcf.nbytes > 0
    # sanity: at least one byte per row across CHROM/POS/REF/ALT
    assert vcf.nbytes >= vcf._index.height
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/david/projects/genoray
pixi run -e default pytest tests/test_vcf.py::test_nbytes_zero_before_index_loaded tests/test_vcf.py::test_nbytes_positive_after_index_loaded -v
```

Expected: FAIL with `AttributeError: 'VCF' object has no attribute 'nbytes'`.

- [ ] **Step 3: Add the property**

In `genoray/_vcf.py`, add a property to the `VCF` class. Place it next to the existing `n_samples` / `current_samples` properties (around line 329):

```python
    @property
    def nbytes(self) -> int:
        """Total in-memory footprint, in bytes, of resident (non-mmap'd) data
        structures held by this reader. Currently this is the gvi variant
        index (CHROM/POS/REF/ALT/ILEN). Returns 0 before the index is loaded.
        """
        if self._index is None:
            return 0
        return self._index.estimated_size()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pixi run -e default pytest tests/test_vcf.py::test_nbytes_zero_before_index_loaded tests/test_vcf.py::test_nbytes_positive_after_index_loaded -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
rtk git add genoray/_vcf.py tests/test_vcf.py
rtk git commit -m "feat(vcf): add nbytes property for resident memory size"
```

---

### Task 3: Add `PGEN.nbytes` property (TDD)

**Files:**
- Modify: `genoray/_pgen.py` — add property on `PGEN` class
- Test: `tests/test_pgen.py` — append new test

- [ ] **Step 1: Write the failing test**

Append to `tests/test_pgen.py`:

```python
def test_pgen_nbytes_positive_after_init():
    # PGEN auto-loads the index in __init__ via _init_index
    pgen = PGEN(ddir / "biallelic.pgen")
    assert pgen._index is not None
    assert pgen.nbytes > 0
    # both the index dataframe and the StartsEndsIlens cache should contribute
    assert pgen.nbytes >= pgen._index.estimated_size()


def test_pgen_nbytes_zero_after_free():
    pgen = PGEN(ddir / "biallelic.pgen")
    pgen._free_index()
    assert pgen._index is None
    assert pgen._sei is None
    assert pgen.nbytes == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/david/projects/genoray
pixi run -e default pytest tests/test_pgen.py::test_pgen_nbytes_positive_after_init tests/test_pgen.py::test_pgen_nbytes_zero_after_free -v
```

Expected: FAIL with `AttributeError: 'PGEN' object has no attribute 'nbytes'`.

- [ ] **Step 3: Add the property**

In `genoray/_pgen.py`, add a property to the `PGEN` class. Place it next to `n_samples` (around line 270):

```python
    @property
    def nbytes(self) -> int:
        """Total in-memory footprint, in bytes, of resident (non-mmap'd) data
        structures held by this reader. Sums the gvi variant index dataframe
        and the StartsEndsIlens cache. Returns 0 after `_free_index()`.
        """
        n = 0
        if self._index is not None:
            n += self._index.estimated_size()
        if self._sei is not None:
            n += (
                self._sei.v_starts.nbytes
                + self._sei.v_ends.nbytes
                + self._sei.ilens.nbytes
                + self._sei.alt.estimated_size()
            )
        return n
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pixi run -e default pytest tests/test_pgen.py::test_pgen_nbytes_positive_after_init tests/test_pgen.py::test_pgen_nbytes_zero_after_free -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
rtk git add genoray/_pgen.py tests/test_pgen.py
rtk git commit -m "feat(pgen): add nbytes property summing index + StartsEndsIlens"
```

---

### Task 4: Add `SparseVar.nbytes` property (TDD)

**Files:**
- Modify: `genoray/_svar.py` — add property on `SparseVar` class
- Test: `tests/test_svar.py` — append new test

- [ ] **Step 1: Write the failing test**

Append to `tests/test_svar.py`:

```python
def test_svar_nbytes_index_only():
    svar = SparseVar(ddir / "biallelic.vcf.svar")
    # nbytes counts only the resident polars index, not the mmap'd genos/fields
    assert svar.nbytes == svar.index.estimated_size()
    assert svar.nbytes > 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/david/projects/genoray
pixi run -e default pytest tests/test_svar.py::test_svar_nbytes_index_only -v
```

Expected: FAIL with `AttributeError: 'SparseVar' object has no attribute 'nbytes'`.

- [ ] **Step 3: Add the property**

In `genoray/_svar.py`, add a property to the `SparseVar` class. Place it next to `n_samples` / `n_variants` (around line 146):

```python
    @property
    def nbytes(self) -> int:
        """Total in-memory footprint, in bytes, of resident (non-mmap'd) data
        held by this reader. Only the polars variant index counts; `genos`
        and `fields` are memory-mapped and excluded.
        """
        return self.index.estimated_size()
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pixi run -e default pytest tests/test_svar.py::test_svar_nbytes_index_only -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
rtk git add genoray/_svar.py tests/test_svar.py
rtk git commit -m "feat(svar): add nbytes property covering resident index only"
```

---

### Task 5: Run full genoray test suite

**Files:** none

- [ ] **Step 1: Run all tests**

```bash
cd /Users/david/projects/genoray
pixi run -e default pytest tests/ -v
```

Expected: all tests pass (new + existing).

- [ ] **Step 2: If any failure, stop and investigate**

Do not proceed to release. Fix issues; if they are unrelated to this change, surface them to the user before continuing.

---

### Task 6: Push branch and open PR for genoray

**Files:** none

> **Important:** Do NOT bump the genoray version manually. The genoray project's CI handles release/versioning on merge. Just push the feature branch and open a PR.

- [ ] **Step 1: Confirm with user before pushing**

This is a remote-affecting action — pause for explicit user confirmation before running the push command.

- [ ] **Step 2: Push the branch**

```bash
cd /Users/david/projects/genoray
rtk git push -u origin feat/nbytes
```

- [ ] **Step 3: Open the PR**

```bash
rtk gh pr create --title "feat: add nbytes property to VCF/PGEN/SparseVar" --body "$(cat <<'EOF'
## Summary
- Adds a public `nbytes: int` property to `VCF`, `PGEN`, and `SparseVar` reporting the in-memory footprint of resident (non-mmap'd) data structures.
- `VCF.nbytes`: size of the loaded gvi index (0 if not loaded).
- `PGEN.nbytes`: index dataframe + `StartsEndsIlens` cache.
- `SparseVar.nbytes`: polars index only; `genos` and `fields` are memory-mapped and excluded.

Enables downstream consumers (e.g. GenVarLoader) to treat their `max_mem` budgets as true total caps by subtracting `reader.nbytes` before passing the remainder to chunked readers.

## Test plan
- [x] `pytest tests/test_vcf.py -v` — new `nbytes` tests pass
- [x] `pytest tests/test_pgen.py -v` — new `nbytes` tests pass
- [x] `pytest tests/test_svar.py -v` — new `nbytes` test passes
- [x] Full `pytest tests/` green

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 4: Wait for CI / merge / release**

CI handles version bump and publication on merge. **Phase 2 proceeds against a local editable install** (Task 7), so it does not have to wait for the release to be on the index — but the GVL dep-version bump (deferred to the end of Phase 2) does.

---

## Phase 2 — GVL consumer

> **Prerequisite for dev:** Phase 1 changes are committed locally in `/Users/david/projects/genoray`. We will dev against an **editable local install** of that working tree, so we do not have to wait for the genoray release.
>
> **Prerequisite for merging the GVL PR:** the new genoray version (containing `.nbytes`) is released and available on the package index. The dep-version bump in `pyproject.toml` / `pixi.toml` happens **after** that release lands (Task 11.5, below) — not now.

### Task 7: Create branch and install genoray editably

**Files:** none (no version-string edits at this stage)

- [ ] **Step 1: Switch to GVL repo and branch**

```bash
cd /Users/david/projects/GenVarLoader
rtk git checkout -b feat/max-mem-index-aware
```

- [ ] **Step 2: Install local genoray editably into the dev env**

```bash
pixi run -e dev uv pip install -e /Users/david/projects/genoray
```

If `uv` is not on the dev env, fall back to `pip`:
```bash
pixi run -e dev pip install -e /Users/david/projects/genoray
```

Expected: `Successfully installed genoray-<x.y.z>` (the working-tree version of genoray, which has `nbytes`).

- [ ] **Step 3: Sanity check the new property is importable**

```bash
pixi run -e dev python -c "from genoray import VCF, PGEN, SparseVar; print(hasattr(VCF, 'nbytes'), hasattr(PGEN, 'nbytes'), hasattr(SparseVar, 'nbytes'))"
```

Expected: `True True True`.

- [ ] **Step 4: No commit yet**

The editable install touches the pixi env, not tracked files. Nothing to commit in this task.

---

### Task 8: Write the failing test for warning behavior

**Files:**
- Test: `tests/dataset/test_write.py` — append new test

- [ ] **Step 1: Append the test**

Append to `/Users/david/projects/GenVarLoader/tests/dataset/test_write.py`:

```python
def test_write_warns_when_index_dominates_max_mem(
    bed: pl.DataFrame, tmp_path, monkeypatch
):
    """If variants.nbytes exceeds 50% of max_mem, gvl.write emits a UserWarning."""
    import pytest
    import warnings as _warnings

    vcf = VCF(ddir / "vcf" / "filtered_sample.vcf.gz")
    vcf._write_gvi_index()
    vcf._load_index()

    # Force nbytes to a large value relative to max_mem.
    # max_mem = 1 MiB; nbytes = 800 KiB → 80% of budget, should warn.
    monkeypatch.setattr(type(vcf), "nbytes", property(lambda self: 800 * 1024))

    out = tmp_path / "test.gvl"
    with pytest.warns(UserWarning, match="exceeds 50% of max_mem"):
        gvl.write(out, bed, vcf, max_mem=1024 * 1024)

    # Sanity: dataset directory was actually created
    assert (out / "metadata.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /Users/david/projects/GenVarLoader
pixi run -e dev pytest tests/dataset/test_write.py::test_write_warns_when_index_dominates_max_mem -v
```

Expected: FAIL — either no warning emitted, or `gvl.write` errors because the tiny effective budget hits genoray's "insufficient memory for a single variant" check.

If the failure is the latter (genoray error), that confirms we still need the GVL-side warning to fire *before* genoray's check. The next task addresses both: emit the warning **before** calling into genoray, and use a larger `max_mem` in the test if needed so the write still succeeds.

Adjust the test if needed: pick `max_mem` so that `effective_max_mem` (= max_mem − 800 KiB) is still enough for at least one variant. For the existing tiny fixture this is fine at 1 MiB. If genoray still errors, increase `max_mem` to e.g. `4 * 1024 * 1024` and scale `nbytes` to `3 * 1024 * 1024` so the 50% rule still trips.

---

### Task 9: Implement effective_max_mem in `_write.py`

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` lines 195–215 (inside `write()`, the variants-handling block)

- [ ] **Step 1: Add `format_memory` import**

At the top of `python/genvarloader/_dataset/_write.py`, alongside the existing imports, add:

```python
from genoray._utils import format_memory
```

If `parse_memory` is already imported from `genoray._utils`, extend that import:

```python
from genoray._utils import format_memory, parse_memory
```

(Verify by grepping the file first — `parse_memory` is used on line 123.)

- [ ] **Step 2: Compute effective_max_mem, warn, and pass through**

Replace the existing variants block. The current code (around line 197–216) reads:

```python
if variants is not None:
    logger.info("Writing genotypes.")
    if isinstance(variants, VCF):
        variants.set_samples(samples)
        gvl_bed = _write_from_vcf(path, gvl_bed, variants, max_mem, extend_to_length)
    elif isinstance(variants, PGEN):
        variants.set_samples(samples)
        gvl_bed = _write_from_pgen(path, gvl_bed, variants, max_mem, extend_to_length)
    elif isinstance(variants, SparseVar):
        gvl_bed = _write_from_svar(path, gvl_bed, variants, samples, extend_to_length)
    metadata["ploidy"] = variants.ploidy
    # free memory
    del variants
    gc.collect()
```

Change it to:

```python
if variants is not None:
    logger.info("Writing genotypes.")

    idx_bytes = variants.nbytes
    effective_max_mem = max_mem - idx_bytes
    logger.info(
        f"Variant reader resident size: {format_memory(idx_bytes)}; "
        f"max_mem budget: {format_memory(max_mem)}; "
        f"available for chunking: {format_memory(max(effective_max_mem, 0))}"
    )
    if idx_bytes > max_mem // 2:
        warnings.warn(
            f"Variant index resident size ({format_memory(idx_bytes)}) "
            f"exceeds 50% of max_mem ({format_memory(max_mem)}). "
            f"Consider increasing max_mem.",
            stacklevel=2,
        )

    if isinstance(variants, VCF):
        variants.set_samples(samples)
        gvl_bed = _write_from_vcf(
            path, gvl_bed, variants, effective_max_mem, extend_to_length
        )
    elif isinstance(variants, PGEN):
        variants.set_samples(samples)
        gvl_bed = _write_from_pgen(
            path, gvl_bed, variants, effective_max_mem, extend_to_length
        )
    elif isinstance(variants, SparseVar):
        gvl_bed = _write_from_svar(path, gvl_bed, variants, samples, extend_to_length)
    metadata["ploidy"] = variants.ploidy
    # free memory
    del variants
    gc.collect()
```

Note: `_write_from_svar` does **not** accept `max_mem` (per current signature). The warning still fires for SparseVar; no other change to that call.

- [ ] **Step 3: Run the warning test**

```bash
cd /Users/david/projects/GenVarLoader
pixi run -e dev pytest tests/dataset/test_write.py::test_write_warns_when_index_dominates_max_mem -v
```

Expected: PASS.

- [ ] **Step 4: Run the full write test module**

```bash
pixi run -e dev pytest tests/dataset/test_write.py -v
```

Expected: all tests pass (the existing `test_write` is `@mark.skip`, but the new one and any others should pass).

- [ ] **Step 5: Run the full GVL test suite**

```bash
pixi run -e dev test
```

Expected: all tests pass. If anything that previously passed now fails, investigate before committing. For very small fixtures, `effective_max_mem` ≈ `max_mem` (default 4 GiB) so chunk sizing should be unchanged.

- [ ] **Step 6: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/dataset/test_write.py
rtk git commit -m "feat(write): subtract genoray nbytes from max_mem; warn when index dominates"
```

---

### Task 10: Update `gvl.write` docstring

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` lines 95–96 (the `max_mem` docstring entry)

- [ ] **Step 1: Edit the docstring**

Change:

```
    max_mem
        Approximate maximum memory to use. This is a soft limit and may be exceeded by a small amount.
```

To:

```
    max_mem
        Approximate maximum total memory to use, including the genoray variant
        index already resident before genotype reading begins. The reader's
        :attr:`~genoray.VCF.nbytes` (or equivalent) is subtracted from
        ``max_mem`` to determine the budget available for genotype chunking.
        A warning is emitted if the resident index exceeds 50% of ``max_mem``.
        This is a soft limit and may be exceeded by a small amount.
```

- [ ] **Step 2: Verify docs still build**

```bash
pixi run -e docs doc
```

Expected: docs build succeeds.

- [ ] **Step 3: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py
rtk git commit -m "docs(write): explain index-aware max_mem accounting"
```

---

### Task 11: Final verification

**Files:** none

- [ ] **Step 1: Full test + lint**

```bash
cd /Users/david/projects/GenVarLoader
pixi run -e dev test
pixi run -e dev ruff check python/
pixi run -e dev basedpyright python/
```

Expected: all green. If basedpyright complains about `variants.nbytes` access (because the union includes `Reader` and not all `Reader` implementations have `nbytes`), the access is guarded by the `isinstance(variants, (VCF, PGEN, SparseVar))` branches downstream; if needed, narrow the type up-front:

```python
assert isinstance(variants, (VCF, PGEN, SparseVar))
idx_bytes = variants.nbytes
```

- [ ] **Step 2: Do NOT push yet**

The GVL PR depends on the new genoray release being on the index. Pause here until the user confirms genoray has been released. Phase 2 implementation work is complete on the local branch; the remaining task (11.5) bumps the dep version once the release is available.

---

### Task 11.5: Bump genoray dependency (after genoray release lands)

**Files:**
- Modify: `pyproject.toml`
- Modify: `pixi.toml`

> **Trigger:** Run this task only after the user confirms the new genoray version (containing `.nbytes`) has been published. Replace `<NEW_VERSION>` below with the actual released version (e.g. `2.4.0`).

- [ ] **Step 1: Bump pyproject.toml**

Edit the `genoray` line (currently line 34):
```diff
-"genoray>=2.3.3,<3",
+"genoray>=<NEW_VERSION>,<3",
```

- [ ] **Step 2: Bump pixi.toml**

Edit the `genoray` line (currently line 87):
```diff
-genoray = "==2.3.3"
+genoray = "==<NEW_VERSION>"
```

- [ ] **Step 3: Uninstall the editable install and refresh lockfile**

```bash
pixi run -e dev pip uninstall -y genoray
pixi install -e dev
```

Expected: pixi.lock updated; the released genoray version is resolved.

- [ ] **Step 4: Re-run the full test suite against the released version**

```bash
pixi run -e dev test
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
rtk git add pyproject.toml pixi.toml pixi.lock
rtk git commit -m "chore(deps): bump genoray to <NEW_VERSION> for nbytes property"
```

- [ ] **Step 6: Push and open PR (confirm with user first)**

```bash
rtk git push -u origin feat/max-mem-index-aware
rtk gh pr create --title "feat(write): index-aware max_mem accounting" --body "$(cat <<'EOF'
## Summary
- Subtracts `variants.nbytes` (new in genoray <NEW_VERSION>) from `max_mem` before passing the remainder to genoray chunking in `_write_from_vcf` / `_write_from_pgen`.
- Emits a `UserWarning` when the resident genoray index exceeds 50% of `max_mem`.
- Bumps genoray dependency to `>=<NEW_VERSION>`.

See `docs/superpowers/specs/2026-05-19-max-mem-index-aware-design.md` for the design.

## Test plan
- [x] `pytest tests/dataset/test_write.py::test_write_warns_when_index_dominates_max_mem`
- [x] Full `pixi run -e dev test`
- [x] `ruff check` and `basedpyright` clean

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-review checklist (already applied)

1. **Spec coverage:** every spec section is implemented — genoray `nbytes` on all three readers (Tasks 2–4), GVL subtraction + warning (Task 9), docstring update (Task 10), dependency bump (Task 7), testing (Tasks 2/3/4 for genoray, Task 8/11 for GVL).
2. **Placeholder scan:** no TBDs; every code block is complete.
3. **Type consistency:** property name is `nbytes` everywhere; `effective_max_mem` is the only computed name and it's used consistently in Task 9.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-19-max-mem-index-aware.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
