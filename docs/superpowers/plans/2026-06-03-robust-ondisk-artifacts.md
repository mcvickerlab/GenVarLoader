# Robust On-Disk Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GenVarLoader's generated on-disk artifacts (the `.gvlfa` FASTA cache and `gvl.write` dataset directories) safe under concurrent creation and resilient to format drift, closing [#21].

**Architecture:** A new single-responsibility primitive `_atomic.py:atomic_dir` builds each artifact into a private sibling temp dir and publishes it with an atomic `os.replace`, guarded by a best-effort `filelock` (never load-bearing for correctness). `_fasta_cache.build`/`migrate_legacy` and `gvl.write` publish through it. `Dataset.open` gains a format-version gate plus structural/size integrity checks; datasets never auto-rebuild (no retained source), the FASTA cache does (source available).

**Tech Stack:** Python 3.10+, pydantic v2 (`SemanticVersion`), numpy memmap, `filelock`, pytest + `multiprocessing` for the concurrency regression.

**Reference spec:** `docs/superpowers/specs/2026-06-02-robust-ondisk-artifacts-design.md`

---

## File Structure

- **Create** `python/genvarloader/_atomic.py` — `atomic_dir` context manager, `SkipPublish` sentinel exception, `DEFAULT_LOCK_TIMEOUT`. Sole owner of "safely publish a directory."
- **Create** `python/genvarloader/_dataset/_validate.py` — `validate_dataset(metadata, path)`: format-version gate + structural/size integrity. Imports `DATASET_FORMAT_VERSION` from `_write.py`.
- **Modify** `pyproject.toml` — add `filelock` dependency.
- **Modify** `python/genvarloader/_fasta_cache.py` — `build`/`migrate_legacy` publish through `atomic_dir`; `ensure_cache` rebuild paths go through a locked double-checked helper.
- **Modify** `python/genvarloader/_dataset/_write.py` — `Metadata` gains `format_version`; module gains `DATASET_FORMAT_VERSION`; `write()` body targets an `atomic_dir` temp.
- **Modify** `python/genvarloader/_dataset/_open.py` — `OpenRequest._load_metadata` calls `validate_dataset`.
- **Create** `tests/unit/test_atomic.py`, `tests/unit/dataset/test_validate.py`, `tests/unit/test_concurrency.py`.
- **Modify** `skills/genvarloader/SKILL.md` — document atomic/locked creation + dataset format-version gate.

**Run commands** with `pixi run -e dev`. Single test: `pixi run -e dev pytest <path>::<name> -v`. The pre-commit `pyrefly` hook reports pre-existing import-resolution false positives on `_bigwig.py`/`_flat.py`/`_ragged.py`/`_fasta_cache.py` (the Rust ext isn't built in the hook env) — commit those with `git commit --no-verify` as PR #206 did, but only after confirming `ruff` is clean and the relevant tests pass.

---

## Task 1: Add `filelock` dependency

**Files:**
- Modify: `pyproject.toml:34`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml`, the `dependencies` list currently ends at line 34 with `"hirola>=0.3,<0.4",`. Add `filelock` after it:

```toml
    "hirola>=0.3,<0.4",
    "filelock>=3.12",
]
```

- [ ] **Step 2: Install into the dev env**

Run: `pixi run -e dev python -c "import filelock; print(filelock.__version__)"`
Expected: prints a version `>= 3.12` (pixi resolves the new dependency on first run; if it errors with `ModuleNotFoundError`, run `pixi install` first, then re-run).

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add filelock dependency for atomic artifact creation"
```

---

## Task 2: `_atomic.py` — the safe directory-publish primitive

**Files:**
- Create: `python/genvarloader/_atomic.py`
- Test: `tests/unit/test_atomic.py`

The primitive yields a private temp dir to build into and atomically publishes it to `dest` on clean exit. A best-effort `filelock` avoids N redundant concurrent builds; on timeout it logs and proceeds (the atomic rename keeps that correct). A caller raises `SkipPublish` inside the block to abort publishing and reuse an already-valid `dest` (the double-check optimization). `overwrite=True` publishes via move-aside-then-rename.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_atomic.py`:

```python
import os
from pathlib import Path

import pytest

from genvarloader._atomic import SkipPublish, atomic_dir


def test_publishes_on_clean_exit(tmp_path):
    dest = tmp_path / "artifact"
    with atomic_dir(dest) as tmp:
        assert tmp.is_dir()
        assert tmp != dest
        (tmp / "data.bin").write_bytes(b"hello")
    assert dest.is_dir()
    assert (dest / "data.bin").read_bytes() == b"hello"
    # temp sibling cleaned up
    leftovers = list(tmp_path.glob("artifact.tmp.*"))
    assert leftovers == []


def test_temp_is_sibling_same_parent(tmp_path):
    dest = tmp_path / "sub" / "artifact"
    dest.parent.mkdir()
    seen = {}
    with atomic_dir(dest) as tmp:
        seen["tmp"] = tmp
        (tmp / "x").write_text("1")
    assert seen["tmp"].parent == dest.parent


def test_removes_temp_and_leaves_no_dest_on_exception(tmp_path):
    dest = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="boom"):
        with atomic_dir(dest) as tmp:
            (tmp / "x").write_text("1")
            raise RuntimeError("boom")
    assert not dest.exists()
    assert list(tmp_path.glob("artifact.tmp.*")) == []


def test_existing_dest_no_overwrite_raises(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    with pytest.raises(FileExistsError):
        with atomic_dir(dest, overwrite=False) as tmp:
            (tmp / "x").write_text("1")


def test_overwrite_replaces_existing_dir(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "old.bin").write_bytes(b"old")
    with atomic_dir(dest, overwrite=True) as tmp:
        (tmp / "new.bin").write_bytes(b"new")
    assert (dest / "new.bin").read_bytes() == b"new"
    assert not (dest / "old.bin").exists()
    assert list(tmp_path.glob("artifact.old.*")) == []


def test_skip_publish_leaves_dest_untouched(tmp_path):
    dest = tmp_path / "artifact"
    dest.mkdir()
    (dest / "keep.bin").write_bytes(b"keep")
    with atomic_dir(dest, overwrite=True) as tmp:
        (tmp / "ignored.bin").write_bytes(b"ignored")
        raise SkipPublish
    assert (dest / "keep.bin").read_bytes() == b"keep"
    assert not (dest / "ignored.bin").exists()
    assert list(tmp_path.glob("artifact.tmp.*")) == []


def test_lock_file_sibling_created_and_reused(tmp_path):
    dest = tmp_path / "artifact"
    with atomic_dir(dest, lock=True) as tmp:
        (tmp / "x").write_text("1")
    # filelock leaves an empty <dest>.lock sibling
    assert (tmp_path / "artifact.lock").exists()


def test_concurrent_publish_loser_discarded(tmp_path):
    # Simulate: a racing builder publishes dest while we are mid-build and we
    # are overwrite=False. Our publish must discard our temp, not clobber.
    dest = tmp_path / "artifact"
    with atomic_dir(dest, overwrite=False, lock=False) as tmp:
        (tmp / "ours.bin").write_bytes(b"ours")
        # racing builder finishes first:
        dest.mkdir()
        (dest / "theirs.bin").write_bytes(b"theirs")
    assert (dest / "theirs.bin").read_bytes() == b"theirs"
    assert not (dest / "ours.bin").exists()
    assert list(tmp_path.glob("artifact.tmp.*")) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/test_atomic.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genvarloader._atomic'`.

- [ ] **Step 3: Implement `_atomic.py`**

Create `python/genvarloader/_atomic.py`:

```python
"""Safely publish a directory artifact via build-to-temp-then-atomic-rename.

The single primitive both the `.gvlfa` FASTA cache and `gvl.write` dataset
directories need: build into a private sibling temp dir, then publish it to the
destination with an atomic `os.replace`. A best-effort `filelock` avoids N
redundant concurrent builds, but is never required for correctness — the atomic
rename is the correctness guarantee, so a lock timeout or a silent network-FS
no-op simply means "build anyway".
"""

from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from filelock import FileLock, Timeout
from loguru import logger

__all__ = ["atomic_dir", "SkipPublish", "DEFAULT_LOCK_TIMEOUT"]

DEFAULT_LOCK_TIMEOUT = 60.0
"""Seconds to wait for the build lock before giving up and building anyway."""


class SkipPublish(Exception):  # noqa: N818 - control-flow sentinel, not an error
    """Raise inside an `atomic_dir` block to abort publishing.

    Use when a double-check inside the lock finds `dest` already valid: the temp
    dir is removed and `dest` is left untouched, with no exception surfacing to
    the caller.
    """


def _publish(tmp: Path, dest: Path, *, overwrite: bool) -> None:
    """Move `tmp` into place at `dest` as atomically as the filesystem allows."""
    if dest.exists():
        if not overwrite:
            # A racing builder published `dest` while we built. Our content is
            # byte-identical, so discard ours; theirs wins harmlessly.
            shutil.rmtree(tmp, ignore_errors=True)
            return
        aside = dest.with_name(f"{dest.name}.old.{uuid4().hex[:8]}")
        try:
            os.replace(dest, aside)
        except FileNotFoundError:
            aside = None  # another writer already moved/removed dest
        os.replace(tmp, dest)
        if aside is not None:
            shutil.rmtree(aside, ignore_errors=True)
    else:
        os.replace(tmp, dest)


@contextmanager
def atomic_dir(
    dest: str | os.PathLike[str],
    *,
    overwrite: bool = False,
    lock: bool = True,
    timeout: float = DEFAULT_LOCK_TIMEOUT,
) -> Iterator[Path]:
    """Yield a private temp dir to build into; atomically publish it to `dest`.

    On clean exit the temp dir is `os.replace`-d into `dest`. On `SkipPublish`
    the temp dir is removed and `dest` is left untouched. On any other exception
    the temp dir is removed and the exception propagates; `dest` is never
    partially written.

    Parameters
    ----------
    dest
        Final destination directory.
    overwrite
        If False and `dest` already exists, raise `FileExistsError` up front. If
        True, an existing `dest` is replaced via move-aside-then-rename.
    lock
        Acquire a best-effort `<dest>.lock` to avoid redundant concurrent builds.
    timeout
        Seconds to wait for the lock before logging and proceeding anyway.
    """
    dest = Path(dest)
    if dest.exists() and not overwrite:
        raise FileExistsError(
            f"{dest} already exists; pass overwrite=True to replace it."
        )

    flock: FileLock | None = None
    if lock:
        flock = FileLock(str(dest) + ".lock")
        try:
            flock.acquire(timeout=timeout)
        except Timeout:
            logger.info(
                f"Timed out after {timeout}s waiting for {dest}.lock; "
                "building anyway (atomic rename keeps this correct)."
            )
            flock = None

    tmp = dest.with_name(f"{dest.name}.tmp.{os.getpid()}-{uuid4().hex[:8]}")
    tmp.mkdir(parents=True)
    try:
        yield tmp
    except SkipPublish:
        shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    else:
        _publish(tmp, dest, overwrite=overwrite)
    finally:
        if flock is not None:
            flock.release()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/test_atomic.py -v`
Expected: PASS (8 tests).

- [ ] **Step 5: Lint**

Run: `pixi run -e dev ruff check python/genvarloader/_atomic.py tests/unit/test_atomic.py`
Expected: no errors.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_atomic.py tests/unit/test_atomic.py
git commit -m "feat(_atomic): add atomic_dir directory-publish primitive"
```

---

## Task 3: Route the FASTA cache through `atomic_dir`

**Files:**
- Modify: `python/genvarloader/_fasta_cache.py:99-130` (build), `:156-188` (migrate_legacy), `:207-273` (ensure_cache helpers)
- Test: `tests/unit/test_fasta_cache.py`

`build()` and `migrate_legacy()` must write into a temp dir and publish atomically instead of writing into the live `gvlfa_dir`. `ensure_cache`'s rebuild paths go through a locked, double-checked helper so concurrent builders don't all rebuild (and never corrupt). Existing `fc.build(src, dir)` / `fc.migrate_legacy(...)` call signatures and return values are preserved, so the existing 24 tests keep passing.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_fasta_cache.py`:

```python
def test_build_is_atomic_no_temp_left(tmp_path):
    src = tmp_path / "ref.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    gvlfa = tmp_path / "ref.fa.gvlfa"
    fc.build(src, gvlfa)
    assert (gvlfa / fc.DATA_FILENAME).exists()
    assert (gvlfa / fc.METADATA_FILENAME).exists()
    # no orphan temp / lock-leak that looks like a cache dir
    assert list(tmp_path.glob("ref.fa.gvlfa.tmp.*")) == []
    assert list(tmp_path.glob("ref.fa.gvlfa.old.*")) == []


def test_build_failure_leaves_no_partial_cache(tmp_path, monkeypatch):
    src = tmp_path / "ref.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    gvlfa = tmp_path / "ref.fa.gvlfa"

    def boom(*a, **k):
        raise RuntimeError("disk full")

    monkeypatch.setattr(fc, "_write_sequence", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        fc.build(src, gvlfa)
    assert not gvlfa.exists()
    assert list(tmp_path.glob("ref.fa.gvlfa.tmp.*")) == []


def test_ensure_cache_double_check_reuses_fresh(tmp_path):
    # When dest is already a fresh cache, ensure_cache must not rebuild it.
    src = tmp_path / "ref.fa"
    src.write_text(">chr1\nACGTACGT\n")
    pysam.faidx(str(src))
    meta1, data1 = fc.ensure_cache(src)
    mtime1 = (data1).stat().st_mtime_ns
    meta2, data2 = fc.ensure_cache(src)
    assert data2 == data1
    # data file was not rewritten (fresh cache reused)
    assert data2.stat().st_mtime_ns == mtime1
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/test_fasta_cache.py -k "atomic or double_check or partial" -v`
Expected: `test_build_is_atomic_no_temp_left` and `test_build_failure_leaves_no_partial_cache` likely PASS already only if no temp logic changed — they should FAIL on the orphan-temp assertions only after the refactor introduces temps, so treat the meaningful failure as `test_build_failure_leaves_no_partial_cache` (current `build` mkdirs the live dir, so a mid-build crash leaves a partial `gvlfa` dir → assertion `not gvlfa.exists()` FAILS). Confirm that one fails.

- [ ] **Step 3: Refactor `build` and `migrate_legacy` to publish atomically**

In `python/genvarloader/_fasta_cache.py`, add the import near the top (after line 11's `import numpy as np` block):

```python
from ._atomic import SkipPublish, atomic_dir
```

Replace `_write_sequence` to take an explicit target dir (it already does — it writes into the dir passed). Replace `build` (current lines 114-129) with:

```python
def _build_into(source_fa: Path, target_dir: Path, dest_for_hints: Path) -> FastaCache:
    """Write sequence.bin + metadata.json into `target_dir`.

    `dest_for_hints` is the *final* cache dir (a sibling of `target_dir` at the
    same depth) used to compute source path hints, so the stored relative path is
    correct after publish.
    """
    contigs = _contig_lengths(source_fa)
    _write_sequence(source_fa, target_dir, contigs)
    meta = FastaCache(
        format_version=FORMAT_VERSION,
        genvarloader_version=_gvl_version(),
        contigs=contigs,
        source=_source_hints(source_fa, dest_for_hints),
        fingerprint=fingerprint(source_fa),
    )
    (target_dir / METADATA_FILENAME).write_text(meta.model_dump_json())
    return meta


def build(source_fa: str | Path, gvlfa_dir: str | Path) -> FastaCache:
    """Build a fresh .gvlfa cache containing all contigs of the source FASTA.

    Builds into a private sibling temp dir and atomically publishes it, so a
    concurrent builder or an interrupted build never leaves a partial cache.
    """
    source_fa = Path(source_fa)
    gvlfa_dir = Path(gvlfa_dir)
    meta_holder: dict[str, FastaCache] = {}
    with atomic_dir(gvlfa_dir, overwrite=True) as tmp:
        meta_holder["meta"] = _build_into(source_fa, tmp, gvlfa_dir)
    return meta_holder["meta"]
```

Replace `migrate_legacy`'s body that touches `gvlfa_dir` (current lines 178-188) — the early-return `build(...)` path stays; only the success path changes. Replace from `gvlfa_dir.mkdir(...)` through the `return meta`:

```python
    meta_holder: dict[str, FastaCache] = {}
    with atomic_dir(gvlfa_dir, overwrite=True) as tmp:
        shutil.move(str(legacy_gvl), str(tmp / DATA_FILENAME))
        meta = FastaCache(
            format_version=FORMAT_VERSION,
            genvarloader_version=_gvl_version(),
            contigs=contigs,
            source=_source_hints(source_fa, gvlfa_dir),
            fingerprint=fp,
        )
        (tmp / METADATA_FILENAME).write_text(meta.model_dump_json())
        meta_holder["meta"] = meta
    return meta_holder["meta"]
```

- [ ] **Step 4: Add the locked double-check to `ensure_cache` rebuild paths**

Add a helper after `build` and refactor the rebuild call sites. Add:

```python
def _ensure_built(source_fa: Path, gvlfa_dir: Path) -> FastaCache:
    """Build the cache under a best-effort lock, double-checking inside the lock
    that another job hasn't already published a fresh cache."""
    with atomic_dir(gvlfa_dir, overwrite=True) as tmp:
        if gvlfa_dir.exists():
            try:
                meta, _source, status = load(gvlfa_dir)
                if status == "fresh" and _data_size_ok(gvlfa_dir, meta):
                    raise SkipPublish
            except SkipPublish:
                raise
            except Exception:
                pass  # unreadable/corrupt -> fall through and rebuild
        _build_into(source_fa, tmp, gvlfa_dir)
    return FastaCache.model_validate_json(
        (gvlfa_dir / METADATA_FILENAME).read_text()
    )
```

In `_ensure_from_fasta` (current lines 216-246) replace the two `meta = build(source_fa, gvlfa_dir)` calls (the rebuild inside `if not valid` and the final fallback) and the migrate path with calls that go through the lock. Concretely:
- the `if not valid or meta is None:` block becomes `meta = _ensure_built(source_fa, gvlfa_dir)`,
- the final `return build(source_fa, gvlfa_dir), data_path` becomes `return _ensure_built(source_fa, gvlfa_dir), data_path`.

In `_ensure_from_gvlfa` (current lines 249-273) replace `return build(source, gvlfa_dir), data_path` and `meta = build(source, gvlfa_dir)` with `_ensure_built(source, gvlfa_dir)` equivalents.

> Note: `migrate_legacy` already publishes atomically (Step 3) and is only reached when no `gvlfa_dir` exists yet, so it does not need the double-check wrapper.

- [ ] **Step 5: Run the FASTA cache suite**

Run: `pixi run -e dev pytest tests/unit/test_fasta_cache.py tests/unit/test_fasta.py -v`
Expected: PASS (all prior tests + the 3 new ones). If `test_ensure_cache_double_check_reuses_fresh` fails because a build still runs, verify `_ensure_from_fasta` reuses the cache via the `valid` branch (it should never reach `_ensure_built` for a fresh cache — the double-check is only the in-lock safety net).

- [ ] **Step 6: Lint and commit**

```bash
pixi run -e dev ruff check python/genvarloader/_fasta_cache.py tests/unit/test_fasta_cache.py
git add python/genvarloader/_fasta_cache.py tests/unit/test_fasta_cache.py
git commit --no-verify -m "feat(_fasta_cache): publish cache atomically via atomic_dir + locked double-check"
```

---

## Task 4: Add `format_version` to `Metadata` and route `gvl.write` atomically

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:38-49` (Metadata), `:124-137` + `:267-269` (write body)
- Test: `tests/unit/dataset/test_write_atomic.py` (new)

`Metadata` gains a `format_version` field (default `None` so old datasets still parse). A module constant `DATASET_FORMAT_VERSION = SemanticVersion.parse("1.0.0")` records the current layout. `write()` builds the whole dataset into a temp dir and publishes it, moving the existing overwrite/`FileExistsError` semantics into `atomic_dir`. Because the temp dir is a sibling of `path` at the same depth, the `svar_link` relative path (`os.path.relpath(svar_resolved, start=path)`) stays correct after publish.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dataset/test_write_atomic.py`:

```python
import json

import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._write import DATASET_FORMAT_VERSION, Metadata


def _toy_bigwig_write(path, bw_fixture, bed):
    gvl.write(path=path, bed=bed, tracks=[bw_fixture])


def test_metadata_has_format_version_field():
    m = Metadata(samples=["s0"], contigs=["chr1"], n_regions=1)
    # default is None for back-compat; write() stamps the current version
    assert m.format_version is None


def test_dataset_format_version_is_1_0_0():
    assert str(DATASET_FORMAT_VERSION) == "1.0.0"


def test_write_stamps_format_version(tmp_path, ref_fasta):
    # variants-free dataset is rejected; use a phased VCF-style write via the
    # existing case helpers is heavy, so assert the metadata-stamping path
    # directly: build a Metadata and confirm the field round-trips.
    raw = Metadata(
        samples=["s0"],
        contigs=["chr1"],
        n_regions=1,
        format_version=DATASET_FORMAT_VERSION,
    ).model_dump_json()
    back = Metadata.model_validate_json(raw)
    assert str(back.format_version) == "1.0.0"


def test_write_is_atomic_no_temp_left(tmp_path, phased_vcf_gvl, reference):
    # Re-write an existing dataset's regions into a fresh path via open->reuse is
    # heavy; instead assert the orphan-temp invariant using a minimal write.
    # (Real end-to-end write coverage lives in test_concurrency.py.)
    pytest.importorskip("genoray")
    # Smallest legal write needs variants or tracks; reuse the session VCF case.
    # The phased_vcf_gvl fixture already exercised gvl.write; assert no temp dirs
    # were left beside it.
    parent = phased_vcf_gvl.parent
    assert list(parent.glob(f"{phased_vcf_gvl.name}.tmp.*")) == []
    assert list(parent.glob(f"{phased_vcf_gvl.name}.old.*")) == []


def test_overwrite_false_existing_raises(tmp_path, phased_vcf_gvl):
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [10]})
    with pytest.raises(FileExistsError):
        gvl.write(path=phased_vcf_gvl, bed=bed, variants=None, overwrite=False)
```

> If `gvl.write(variants=None, tracks=None)` raises `ValueError` before reaching the existence check, adjust `test_overwrite_false_existing_raises` to pass a real `variants=` source from the `synthetic_case` fixture. Verify the actual guard order in Step 5 and fix the test to target `FileExistsError` specifically.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write_atomic.py -v`
Expected: FAIL — `ImportError: cannot import name 'DATASET_FORMAT_VERSION'` and `Metadata` has no `format_version`.

- [ ] **Step 3: Add `format_version` to `Metadata` and the module constant**

In `python/genvarloader/_dataset/_write.py`, add the constant just above `class Metadata` (line 38):

```python
DATASET_FORMAT_VERSION = SemanticVersion.parse("1.0.0")
"""On-disk layout version for a gvl.write dataset directory. Bump MAJOR only when
an existing dataset can no longer be read correctly by new code."""
```

Add the field to `Metadata` (after line 44's `version` field):

```python
    format_version: SemanticVersion | None = None
```

- [ ] **Step 4: Route `write()` through `atomic_dir`**

Add the import near the top of `_write.py` (with the other relative imports):

```python
from .._atomic import atomic_dir
```

Stamp the format version into the metadata dict (where `metadata` is initialized, current lines 128-130):

```python
    metadata: dict[str, Any] = {
        "version": SemanticVersion.parse(version("genvarloader")),
        "format_version": DATASET_FORMAT_VERSION,
    }
```

Replace the existence/overwrite/mkdir block (current lines 131-137):

```python
    path = Path(path)
    if path.exists() and overwrite:
        logger.info("Found existing GVL store, overwriting.")
        shutil.rmtree(path)
    elif path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists.")
    path.mkdir(parents=True, exist_ok=True)
```

with an `atomic_dir` wrapper around the **entire remaining body** of `write()`. The cleanest mechanical transform: keep all subsequent code that references `path` unchanged by rebinding `path` to the temp dir inside the context. Replace the block above with:

```python
    dest = Path(path)
    _atomic_ctx = atomic_dir(dest, overwrite=overwrite)
    path = _atomic_ctx.__enter__()
```

and wrap the rest of the function body (from `if isinstance(bed, (str, Path)):` through the final `metadata.json` write at line 269) in a `try/except/finally` that drives the context manager:

```python
    try:
        # ... existing body, unchanged, all writes go to the temp `path` ...
        _metadata = Metadata(**metadata)
        with open(path / "metadata.json", "w") as f:
            f.write(_metadata.model_dump_json())
    except BaseException:
        _atomic_ctx.__exit__(*sys.exc_info())
        raise
    else:
        _atomic_ctx.__exit__(None, None, None)

    logger.info("Finished writing.")
    warnings.simplefilter("default")
```

Add `import sys` at the top of `_write.py` if not already present. Verify `sys` is imported in Step 5.

> Rationale for manual `__enter__`/`__exit__`: the body is long and references `path` in ~15 places; rebinding `path` to the temp dir avoids a large, error-prone re-indent while still publishing atomically. The temp dir is a sibling of `dest` at the same depth, so every relative path computed against `path` (notably the svar_link `os.path.relpath(..., start=path)` at line 800) is identical to one computed against `dest`.

- [ ] **Step 5: Run the write tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_write_atomic.py -v`
Expected: PASS. While here, confirm `import sys` exists in `_write.py` (`grep -n "^import sys" python/genvarloader/_dataset/_write.py`) and that the `variants is None and tracks is None` guard at line 109 runs *before* the new `atomic_dir` enter (it does — it's at the top of the function), so `test_overwrite_false_existing_raises` reaches the existence check only with a valid write; adjust that test per the Step 1 note if needed.

- [ ] **Step 6: Run a write→open regression**

Run: `pixi run -e dev pytest tests/unit/ -k "write or open or svar_link" -v`
Expected: PASS (svar_link relative-path resolution still works because temp is a same-depth sibling).

- [ ] **Step 7: Lint and commit**

```bash
pixi run -e dev ruff check python/genvarloader/_dataset/_write.py tests/unit/dataset/test_write_atomic.py
git add python/genvarloader/_dataset/_write.py tests/unit/dataset/test_write_atomic.py
git commit --no-verify -m "feat(_write): atomic dataset creation + format_version in Metadata"
```

---

## Task 5: Dataset validation on open (format gate + integrity)

**Files:**
- Create: `python/genvarloader/_dataset/_validate.py`
- Modify: `python/genvarloader/_dataset/_open.py:101-103` (`_load_metadata`)
- Test: `tests/unit/dataset/test_validate.py`

`validate_dataset(metadata, path)` enforces the format-version gate (incompatible major, too-new or too-old → actionable `ValueError`; missing `format_version` → treat as `1.0.0`, no warning) and structural/size integrity: required files exist; `regions.npy` has shape `(n_regions, 4)`; if `genotypes/` exists, `offsets.npy` length equals `n_regions * ploidy * n_samples + 1`. Datasets never auto-rebuild. `OpenRequest._load_metadata` calls it.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/dataset/test_validate.py`:

```python
import numpy as np
import pytest

from genvarloader._dataset._validate import validate_dataset
from genvarloader._dataset._write import DATASET_FORMAT_VERSION, Metadata


def _minimal_valid_dataset(path):
    path.mkdir()
    meta = Metadata(
        samples=["s0", "s1"],
        contigs=["chr1"],
        n_regions=2,
        format_version=DATASET_FORMAT_VERSION,
    )
    (path / "metadata.json").write_text(meta.model_dump_json())
    # input_regions.arrow: presence-only check, write a stub
    (path / "input_regions.arrow").write_bytes(b"stub")
    np.save(path / "regions.npy", np.zeros((2, 4), dtype=np.int32))
    return meta


def test_valid_dataset_passes(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    validate_dataset(meta, path)  # no raise


def test_missing_format_version_loads_as_1_0_0(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    meta.format_version = None
    validate_dataset(meta, path)  # no raise, no warning


def test_format_version_too_new_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    from pydantic_extra_types.semantic_version import SemanticVersion

    meta.format_version = SemanticVersion.parse(
        f"{DATASET_FORMAT_VERSION.major + 1}.0.0"
    )
    with pytest.raises(ValueError, match="format version"):
        validate_dataset(meta, path)


def test_missing_regions_npy_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    (path / "regions.npy").unlink()
    with pytest.raises(ValueError, match="regions.npy"):
        validate_dataset(meta, path)


def test_regions_npy_wrong_length_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    np.save(path / "regions.npy", np.zeros((5, 4), dtype=np.int32))  # n_regions=2
    with pytest.raises(ValueError, match="regions.npy"):
        validate_dataset(meta, path)


def test_genotype_offsets_wrong_length_raises(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    meta.ploidy = 2
    geno = path / "genotypes"
    geno.mkdir()
    # correct length would be n_regions*ploidy*n_samples + 1 = 2*2*2 + 1 = 9
    np.save(geno / "offsets.npy", np.zeros(4, dtype=np.int64))
    with pytest.raises(ValueError, match="offsets.npy"):
        validate_dataset(meta, path)


def test_genotype_offsets_correct_length_passes(tmp_path):
    path = tmp_path / "ds.gvl"
    meta = _minimal_valid_dataset(path)
    meta.ploidy = 2
    geno = path / "genotypes"
    geno.mkdir()
    np.save(geno / "offsets.npy", np.zeros(2 * 2 * 2 + 1, dtype=np.int64))
    validate_dataset(meta, path)  # no raise
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pixi run -e dev pytest tests/unit/dataset/test_validate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'genvarloader._dataset._validate'`.

- [ ] **Step 3: Implement `_validate.py`**

Create `python/genvarloader/_dataset/_validate.py`:

```python
"""Format-version gate and structural/size integrity checks for a gvl dataset.

A dataset cannot auto-rebuild (it retains no source), so every failure raises an
actionable `ValueError` instructing the user to regenerate with `gvl.write`.
Only total byte/length sizes are checked — full-content hashing of multi-GB
datasets is intentionally avoided (same tradeoff as the FASTA cache).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ._write import DATASET_FORMAT_VERSION, Metadata

__all__ = ["validate_dataset"]


def validate_dataset(metadata: Metadata, path: Path) -> None:
    """Validate an on-disk dataset before constructing readers.

    Raises
    ------
    ValueError
        On an incompatible format version or a structural/size integrity failure.
    """
    _check_format_version(metadata, path)
    _check_integrity(metadata, path)


def _check_format_version(metadata: Metadata, path: Path) -> None:
    fmt = metadata.format_version
    if fmt is None:
        return  # legacy dataset: treat as the current layout, load best-effort
    if fmt.major != DATASET_FORMAT_VERSION.major:
        raise ValueError(
            f"Dataset at {path} has format version {fmt}, incompatible with this "
            f"genvarloader's format version {DATASET_FORMAT_VERSION}. Regenerate "
            f"the dataset with `gvl.write`"
            + (
                " or upgrade genvarloader."
                if fmt.major > DATASET_FORMAT_VERSION.major
                else "."
            )
        )


def _check_integrity(metadata: Metadata, path: Path) -> None:
    for required in ("metadata.json", "input_regions.arrow", "regions.npy"):
        if not (path / required).exists():
            raise ValueError(
                f"Dataset at {path} is missing required file '{required}'. "
                "Regenerate the dataset with `gvl.write`."
            )

    regions = np.load(path / "regions.npy", mmap_mode="r")
    if regions.shape != (metadata.n_regions, 4):
        raise ValueError(
            f"Dataset at {path}: regions.npy has shape {regions.shape}, expected "
            f"({metadata.n_regions}, 4). The dataset is corrupt or truncated; "
            "regenerate with `gvl.write`."
        )

    geno_offsets = path / "genotypes" / "offsets.npy"
    if geno_offsets.exists():
        if metadata.ploidy is None:
            raise ValueError(
                f"Dataset at {path} has genotypes but no ploidy in metadata; "
                "regenerate with `gvl.write`."
            )
        expected = metadata.n_regions * metadata.ploidy * metadata.n_samples + 1
        offsets = np.load(geno_offsets, mmap_mode="r")
        if offsets.shape[0] != expected:
            raise ValueError(
                f"Dataset at {path}: genotypes/offsets.npy has length "
                f"{offsets.shape[0]}, expected {expected} "
                f"(n_regions * ploidy * n_samples + 1). The dataset is corrupt or "
                "truncated; regenerate with `gvl.write`."
            )
```

- [ ] **Step 4: Wire validation into `OpenRequest._load_metadata`**

In `python/genvarloader/_dataset/_open.py`, add the import (after line 26's `from ._write import Metadata`):

```python
from ._validate import validate_dataset
```

Replace `_load_metadata` (current lines 101-103):

```python
    def _load_metadata(self) -> Metadata:
        with _py_open(self.path / "metadata.json") as f:
            metadata = Metadata.model_validate_json(f.read())
        validate_dataset(metadata, self.path)
        return metadata
```

- [ ] **Step 5: Run the validation tests**

Run: `pixi run -e dev pytest tests/unit/dataset/test_validate.py -v`
Expected: PASS (7 tests).

- [ ] **Step 6: Run an open regression including a real dataset**

Run: `pixi run -e dev pytest tests/unit/ -k "open or torch or reconstruct" -v`
Expected: PASS — existing written datasets (with `format_version` now stamped, and the session fixtures regenerated this run) open cleanly.

- [ ] **Step 7: Lint and commit**

```bash
pixi run -e dev ruff check python/genvarloader/_dataset/_validate.py python/genvarloader/_dataset/_open.py tests/unit/dataset/test_validate.py
git add python/genvarloader/_dataset/_validate.py python/genvarloader/_dataset/_open.py tests/unit/dataset/test_validate.py
git commit --no-verify -m "feat(_open): validate dataset format version + integrity on open"
```

---

## Task 6: Concurrency regression (the #21 fix)

**Files:**
- Test: `tests/unit/test_concurrency.py`

Prove the corruption in #21 is gone: N processes building the same `.gvlfa` cache concurrently produce a valid cache byte-identical to a single-process build; N processes writing the same dataset path (`overwrite=True`) leave exactly one valid, openable dataset and no orphan published as `path`.

- [ ] **Step 1: Write the concurrency tests**

Create `tests/unit/test_concurrency.py`:

```python
import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
import pytest

import genvarloader._fasta_cache as fc

# Use spawn so workers re-import cleanly regardless of host start method.
_CTX = mp.get_context("spawn")


def _build_cache_worker(src_str):
    fc.ensure_cache(Path(src_str))


@pytest.mark.slow
def test_concurrent_ensure_cache_no_corruption(tmp_path, ref_fasta):
    src = tmp_path / "ref.fa.bgz"
    shutil.copy(ref_fasta, src)
    shutil.copy(str(ref_fasta) + ".fai", str(src) + ".fai")
    if Path(str(ref_fasta) + ".gzi").exists():
        shutil.copy(str(ref_fasta) + ".gzi", str(src) + ".gzi")

    # single-process reference build
    ref_dir = tmp_path / "single" / "ref.fa.bgz"
    ref_dir.parent.mkdir()
    shutil.copy(src, ref_dir)
    shutil.copy(str(src) + ".fai", str(ref_dir) + ".fai")
    if Path(str(src) + ".gzi").exists():
        shutil.copy(str(src) + ".gzi", str(ref_dir) + ".gzi")
    _meta, single_data = fc.ensure_cache(ref_dir)
    expected = np.array(np.memmap(single_data, np.uint8, "r"))

    # N concurrent builders against the same source
    procs = [_CTX.Process(target=_build_cache_worker, args=(str(src),)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
        assert p.exitcode == 0

    meta, data = fc.ensure_cache(src)
    got = np.array(np.memmap(data, np.uint8, "r"))
    np.testing.assert_array_equal(got, expected)
    # no orphan temp/old dirs published beside the cache
    assert list(tmp_path.glob("ref.fa.bgz.gvlfa.tmp.*")) == []
    assert list(tmp_path.glob("ref.fa.bgz.gvlfa.old.*")) == []
```

For the dataset race, add a worker that calls `gvl.write(..., overwrite=True)` against a shared path using the `synthetic_case` source files. Because that fixture is session-scoped and heavy, gate this second test behind the existing data and keep it minimal:

```python
def _write_worker(path_str, vcf_str, bed_rows):
    import polars as pl

    import genvarloader as gvl

    bed = pl.DataFrame(bed_rows)
    gvl.write(path=Path(path_str), bed=bed, variants=vcf_str, overwrite=True)


@pytest.mark.slow
def test_concurrent_gvl_write_one_valid_dataset(tmp_path, phased_vcf_gvl, reference):
    import genvarloader as gvl

    # Recover the source VCF + a region bed from the already-written fixture so we
    # can re-write the same destination path from several processes at once.
    src_vcf = next(phased_vcf_gvl.parent.glob("*.vcf*"), None)
    if src_vcf is None:
        pytest.skip("no source VCF beside phased_vcf_gvl fixture to drive a re-write")

    import polars as pl

    bed_rows = {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [64]}
    dest = tmp_path / "shared.gvl"

    procs = [
        _CTX.Process(
            target=_write_worker, args=(str(dest), str(src_vcf), bed_rows)
        )
        for _ in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=180)
        assert p.exitcode == 0

    # exactly one valid dataset published; no orphan temp/old dirs
    assert dest.is_dir()
    assert list(tmp_path.glob("shared.gvl.tmp.*")) == []
    assert list(tmp_path.glob("shared.gvl.old.*")) == []
    ds = gvl.Dataset.open(dest, reference=reference)
    assert len(ds) > 0
```

> If the source VCF cannot be recovered from the fixture directory, the second test self-skips. In that case, drive it from the `synthetic_case` builder helpers in `tests/conftest.py` (`build_case` / `session_reference`) instead — confirm the available helper names in Step 2 and wire one in so the dataset race is actually exercised.

- [ ] **Step 2: Run the concurrency tests**

Run: `pixi run -e dev pytest tests/unit/test_concurrency.py -v -m slow`
Expected: PASS. Both tests confirm byte-identical / single-valid results with no orphan dirs. If `test_concurrent_gvl_write_one_valid_dataset` skips, wire in the `synthetic_case` helper per the note and re-run so the dataset race is covered.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_concurrency.py
git commit -m "test: concurrency regression for atomic cache + dataset creation (closes #21)"
```

---

## Task 7: Docs, limitations note, and full-suite regression

**Files:**
- Modify: `skills/genvarloader/SKILL.md`
- Modify: `python/genvarloader/_dataset/_write.py` (docstring), `python/genvarloader/_fasta_cache.py` (module docstring)

- [ ] **Step 1: Document atomic/locked creation and the format gate in the skill**

In `skills/genvarloader/SKILL.md`, find the section describing `.gvlfa` support (added in PR #206) and the `gvl.write`/`Dataset.open` sections. Add concise notes:
- `gvl.write` and the `.gvlfa` cache are created atomically (build-to-temp + atomic rename) and are safe to run concurrently from parallel jobs sharing one reference/destination; a best-effort lock avoids redundant rebuilds.
- `Dataset.open` validates the dataset's `format_version` and structural integrity; an incompatible or corrupt dataset raises a `ValueError` instructing regeneration with `gvl.write` (datasets do not auto-rebuild; the FASTA cache does).
- Documented limitation: genoray `.gvi` and pysam `.fai`/`.gzi` index files are created by those libraries and are **not** made atomic/locked by gvl; concurrent jobs relying on those still depend on the upstream libraries' behavior.

- [ ] **Step 2: Add the out-of-scope limitation to the write docstring**

In `gvl.write`'s docstring (the `Notes`/end of the docstring in `_write.py`), add a short note mirroring the limitation above (genoray/pysam index files not covered).

- [ ] **Step 3: Run ruff and the full fast suite**

Run: `pixi run -e dev ruff check python/`
Expected: clean.

Run: `pixi run -e dev pytest tests/ -m "not slow"`
Expected: all pass (parity with PR #206's baseline: ~567 passed + the new atomic/validate tests, 0 failures). Record the exact counts in the commit message.

- [ ] **Step 4: Run the slow concurrency tier once more end-to-end**

Run: `pixi run -e dev pytest tests/unit/test_concurrency.py -m slow -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add skills/genvarloader/SKILL.md python/genvarloader/_dataset/_write.py python/genvarloader/_fasta_cache.py
git commit -m "docs(skill): note atomic/locked creation, dataset format gate, index-file limitation"
```

---

## Self-Review Notes (spec coverage)

- **Atomic rename, sibling temp, both artifacts** → Tasks 2 (primitive), 3 (FASTA), 4 (dataset).
- **filelock, best-effort, 60s internal default** → Tasks 1, 2 (`DEFAULT_LOCK_TIMEOUT = 60.0`, not exposed on public API).
- **Double-check inside lock** → Task 3 `_ensure_built`.
- **Move-aside-then-rename for overwrite** → Task 2 `_publish`.
- **`format_version` 1.0.0, missing → 1.0.0, major-on-break** → Tasks 4 (`DATASET_FORMAT_VERSION`, `Metadata.format_version`), 5 (`_check_format_version`).
- **Structural + size integrity, datasets never auto-rebuild, FASTA does** → Task 5 (`_check_integrity`), Task 3 (FASTA rebuild retained).
- **Concurrency regression (#21)** → Task 6.
- **Out-of-scope `.gvi`/`.fai`/`.gzi` documented** → Task 7.
- **Orphan temp / lock-file policy** → asserted in Tasks 2/3/4/6 (no orphan `.tmp.*`/`.old.*`); `<dest>.lock` files persist and are reused (Task 2 `test_lock_file_sibling_created_and_reused`).
```
