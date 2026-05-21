# `link.svar` symlink replacement — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the brittle `link.svar` symlink with a typed `Metadata.svar_link` field (path reference + fingerprint), migrate `Metadata.version` to `SemanticVersion`, and document the on-disk format.

**Architecture:** Extend the existing `Metadata` pydantic model with a nested `SvarLink` field; dispatch read-side resolution on `Metadata.version`; keep a legacy fallback that reads the old symlink for pre-bump datasets; provide `migrate_svar_link()` to upgrade in place. No new sidecar files.

**Tech Stack:** Python, pydantic v2, `pydantic-extra-types` (SemanticVersion), polars, numpy memmap, pytest, pixi.

**Spec:** `docs/superpowers/specs/2026-05-21-svar-link-replacement-design.md`

**NEXT_VERSION placeholder:** Throughout this plan, `NEXT_VERSION` is the next published genvarloader release. Read it once at the start of Task 1 (commitizen / `pyproject.toml`'s configured version-bumping convention) and substitute the literal string (e.g. `"0.3.0"`) everywhere this plan references it. Do not leave the placeholder in code.

---

## File Structure

**New files:**
- `python/genvarloader/_dataset/_svar_link.py` — `SvarFingerprint`, `SvarLink`, `_resolve_svar`, `_verify_fingerprint`, `migrate_svar_link`. One file for everything svar-link-related so the boundary is obvious.
- `tests/dataset/test_svar_link.py` — all new tests for this feature (write/read roundtrip, override, mismatch, missing, sibling, legacy, migration, schema, version parsing).
- `docs/source/format.md` — on-disk format documentation + changelog.

**Modified files:**
- `python/genvarloader/_dataset/_write.py` — switch `version` field to `SemanticVersion`; add `svar_link` field to `Metadata`; populate it from `_write_from_svar`; remove the symlink write.
- `python/genvarloader/_dataset/_reconstruct.py` — replace `Version` import; accept `SvarLink | None` and use `_resolve_svar`; update the `>= Version("0.18.0")` comparison.
- `python/genvarloader/_dataset/_impl.py` — add `svar=` kwarg to all three `Dataset.open` overloads; thread it to `Haps.from_path`; emit deprecation warning when legacy path is taken.
- `python/genvarloader/__init__.py` — re-export `migrate_svar_link`.
- `pyproject.toml` — add `pydantic-extra-types` with the `semver` extra.
- `docs/source/index.md` — add `format` to the toctree.

---

## Task 1: Add `pydantic-extra-types` dependency and confirm `NEXT_VERSION`

**Files:**
- Modify: `pyproject.toml`
- Read: `pyproject.toml` (for the current version), `CHANGELOG.md` if present

- [ ] **Step 1: Inspect current version & decide `NEXT_VERSION`**

Run: `grep -E '^version' pyproject.toml`
Expected: a line like `version = "0.2.1"`.

Pick `NEXT_VERSION` as the next minor bump under commitizen's conventional-commits scheme (e.g. `0.2.1` → `0.3.0`). Record the chosen value at the top of every subsequent file you touch — substitute the literal value where this plan says `NEXT_VERSION`.

- [ ] **Step 2: Add `pydantic-extra-types` to dependencies**

Edit `pyproject.toml`, in the `[project] dependencies` array (the same one that contains `"pydantic>=2,<3"`), add:

```toml
"pydantic-extra-types[semver]>=2.10",
```

- [ ] **Step 3: Install and verify**

Run: `pixi install -e dev`
Then: `pixi run -e dev python -c "from pydantic_extra_types.semantic_version import SemanticVersion; print(SemanticVersion.parse('0.18.0'))"`
Expected: `0.18.0` printed; no ImportError.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml pixi.lock
git commit -m "build: add pydantic-extra-types dep for SemanticVersion"
```

---

## Task 2: Create `_svar_link.py` with `SvarFingerprint` and `SvarLink` models

**Files:**
- Create: `python/genvarloader/_dataset/_svar_link.py`
- Test: `tests/dataset/test_svar_link.py`

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_svar_link.py`:

```python
from pathlib import Path

import pytest
from pydantic import ValidationError

from genvarloader._dataset._svar_link import SvarFingerprint, SvarLink


def test_svar_link_roundtrip():
    link = SvarLink(
        relative_path="../foo.svar",
        absolute_path="/abs/path/foo.svar",
        fingerprint=SvarFingerprint(n_variants=10, variant_idxs_bytes=42),
    )
    payload = link.model_dump_json()
    parsed = SvarLink.model_validate_json(payload)
    assert parsed == link


def test_svar_link_rejects_malformed_fingerprint():
    bad = '{"relative_path":"a","absolute_path":"b","fingerprint":{"n_variants":"not_an_int","variant_idxs_bytes":1}}'
    with pytest.raises(ValidationError):
        SvarLink.model_validate_json(bad)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_svar_link_roundtrip -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'genvarloader._dataset._svar_link'`.

- [ ] **Step 3: Implement the models**

Create `python/genvarloader/_dataset/_svar_link.py`:

```python
"""Resolution and integrity for the GVL dataset → SVAR back-reference."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class SvarFingerprint(BaseModel):
    n_variants: int
    variant_idxs_bytes: int


class SvarLink(BaseModel):
    relative_path: str
    absolute_path: str
    fingerprint: SvarFingerprint
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_svar_link.py tests/dataset/test_svar_link.py
git commit -m "feat(dataset): add SvarLink / SvarFingerprint pydantic models"
```

---

## Task 3: Migrate `Metadata.version` to `SemanticVersion` and add `svar_link` field

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py:25-50`, `:133`
- Test: `tests/dataset/test_svar_link.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_svar_link.py`:

```python
import json

from genvarloader._dataset._write import Metadata
from pydantic_extra_types.semantic_version import SemanticVersion


def test_metadata_version_parses_existing_strings():
    payload = json.dumps(
        {
            "samples": ["s1"],
            "contigs": ["1"],
            "n_regions": 1,
            "version": "0.18.0",
        }
    )
    m = Metadata.model_validate_json(payload)
    assert isinstance(m.version, SemanticVersion)
    assert m.version == SemanticVersion.parse("0.18.0")


def test_metadata_version_serializes_back_to_string():
    m = Metadata(
        samples=["s1"],
        contigs=["1"],
        n_regions=1,
        version=SemanticVersion.parse("0.18.0"),
    )
    dumped = json.loads(m.model_dump_json())
    assert dumped["version"] == "0.18.0"


def test_metadata_svar_link_defaults_to_none():
    m = Metadata(samples=["s1"], contigs=["1"], n_regions=1)
    assert m.svar_link is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: the three new tests FAIL (the current `Metadata` uses `packaging.version.Version` and has no `svar_link`).

- [ ] **Step 3: Update `_write.py` imports and `Metadata`**

In `python/genvarloader/_dataset/_write.py`:

Replace lines 25-26:

```python
from packaging.version import Version
from pydantic import BaseModel, BeforeValidator, PlainSerializer, WithJsonSchema
```

with:

```python
from pydantic import BaseModel
from pydantic_extra_types.semantic_version import SemanticVersion

from ._svar_link import SvarLink
```

Replace the `Metadata` class (currently `_write.py:36-54`) with:

```python
class Metadata(BaseModel, arbitrary_types_allowed=True):
    samples: list[str]
    contigs: list[str]
    n_regions: int
    ploidy: int | None = None
    max_jitter: int = 0
    version: SemanticVersion | None = None
    svar_link: SvarLink | None = None

    @property
    def n_samples(self) -> int:
        return len(self.samples)
```

Also update line 133:

```python
metadata: dict[str, Any] = {"version": Version(version("genvarloader"))}
```

to:

```python
metadata: dict[str, Any] = {"version": SemanticVersion(version("genvarloader"))}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: all five tests in this file PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_write.py tests/dataset/test_svar_link.py
git commit -m "refactor(dataset): use SemanticVersion in Metadata, add svar_link field"
```

---

## Task 4: Update `_reconstruct.py` Version comparison

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py:21`, `:227`, `:265`

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_svar_link.py`:

```python
def test_semantic_version_ordering_for_one_based_dispatch():
    """The legacy comparison '>= 0.18.0' must still work under SemanticVersion."""
    assert SemanticVersion.parse("0.18.0") >= SemanticVersion.parse("0.18.0")
    assert SemanticVersion.parse("0.20.0") >= SemanticVersion.parse("0.18.0")
    assert not (SemanticVersion.parse("0.17.5") >= SemanticVersion.parse("0.18.0"))
```

- [ ] **Step 2: Run it to verify the lib comparison works**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_semantic_version_ordering_for_one_based_dispatch -v`
Expected: PASS (this is a sanity check on the library; you still need to update `_reconstruct.py` so the production code uses the new type).

- [ ] **Step 3: Update imports and type hints in `_reconstruct.py`**

Replace line 21:

```python
from packaging.version import Version
```

with:

```python
from pydantic_extra_types.semantic_version import SemanticVersion
```

Replace line 227:

```python
        version: Version | None,
```

with:

```python
        version: SemanticVersion | None,
```

Replace line 265:

```python
                one_based=version is not None and version >= Version("0.18.0"),
```

with:

```python
                one_based=version is not None and version >= SemanticVersion.parse("0.18.0"),
```

- [ ] **Step 4: Run the dataset tests to ensure nothing regressed**

Run: `pixi run -e dev pytest tests/dataset/ -v -x --timeout=120`
Expected: All existing tests still PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_reconstruct.py tests/dataset/test_svar_link.py
git commit -m "refactor(dataset): switch Haps.from_path version compare to SemanticVersion"
```

---

## Task 5: Populate `svar_link` in `_write_from_svar` (replace symlink)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (function `_write_from_svar`, currently ending at `:657`)

- [ ] **Step 1: Write the failing test**

Append to `tests/dataset/test_svar_link.py`:

```python
import shutil

import genvarloader as gvl


@pytest.fixture
def svar_dataset_paths(tmp_path):
    """Locate the canonical test svar and produce a fresh GVL dataset from it.

    Mirrors the existing svar fixtures used in `tests/dataset/`.
    """
    # The repo's pixi-gen task produces a test svar under tests/data/.
    # Look up the exact path by searching tests/data for *.svar.
    repo_root = Path(__file__).resolve().parents[2]
    svar_candidates = list((repo_root / "tests" / "data").rglob("*.svar"))
    assert svar_candidates, "Run `pixi run -e dev gen` to materialize test svar"
    svar_path = svar_candidates[0]
    bed_path = next((repo_root / "tests" / "data").rglob("*.bed"))

    gvl_path = tmp_path / "ds.gvl"
    gvl.write(path=gvl_path, bed=bed_path, variants=svar_path, overwrite=True)
    return gvl_path, svar_path


def test_write_from_svar_records_svar_link_and_no_symlink(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths

    # No symlink in the new layout.
    assert not (gvl_path / "genotypes" / "link.svar").exists()
    assert not (gvl_path / "genotypes" / "link.svar").is_symlink()

    # Metadata records the svar.
    metadata = Metadata.model_validate_json(
        (gvl_path / "metadata.json").read_text()
    )
    assert metadata.svar_link is not None
    assert Path(metadata.svar_link.absolute_path) == svar_path.resolve()
    # relative_path should resolve back to the svar from the dataset dir.
    assert (gvl_path / metadata.svar_link.relative_path).resolve() == svar_path.resolve()
    # Fingerprint matches the source.
    expected_bytes = (svar_path / "variant_idxs.npy").stat().st_size
    assert metadata.svar_link.fingerprint.variant_idxs_bytes == expected_bytes
    assert metadata.svar_link.fingerprint.n_variants > 0
```

- [ ] **Step 2: Run it to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_write_from_svar_records_svar_link_and_no_symlink -v`
Expected: FAIL — `link.svar` symlink still exists and `metadata.svar_link is None`.

- [ ] **Step 3: Update `_write_from_svar`**

In `python/genvarloader/_dataset/_write.py`, locate the line at the end of `_write_from_svar`:

```python
    (out_dir / "link.svar").symlink_to(svar.path.resolve(), target_is_directory=True)
```

Replace it with:

```python
    import os as _os

    from ._svar_link import SvarFingerprint, SvarLink as _SvarLink

    svar_resolved = svar.path.resolve()
    variant_idxs_path = svar_resolved / "variant_idxs.npy"
    svar_link_obj = _SvarLink(
        relative_path=_os.path.relpath(svar_resolved, start=out_dir.parent).replace(
            _os.sep, "/"
        ),
        absolute_path=str(svar_resolved),
        fingerprint=SvarFingerprint(
            n_variants=svar.index.height,
            variant_idxs_bytes=variant_idxs_path.stat().st_size,
        ),
    )
    # Store it on the in-memory metadata dict so `write()` serializes it.
    # _write_from_svar can't see `metadata` directly — return it.
    # See updated return below.
```

The return signature must change so `write()` can pick up the `SvarLink`. Change the function signature and final return:

```python
def _write_from_svar(
    path: Path,
    bed: pl.DataFrame,
    svar: SparseVar,
    samples: list[str],
    extend_to_length: bool,
) -> tuple[pl.DataFrame, "SvarLink"]:
    ...
    return bed.with_columns(chromEnd=pl.Series(max_ends)), svar_link_obj
```

And in the top-level `write()` function (currently `_write.py:253-256`):

```python
        elif isinstance(variants, SparseVar):
            gvl_bed = _write_from_svar(
                path, gvl_bed, variants, samples, extend_to_length
            )
```

change to:

```python
        elif isinstance(variants, SparseVar):
            gvl_bed, _svar_link = _write_from_svar(
                path, gvl_bed, variants, samples, extend_to_length
            )
            metadata["svar_link"] = _svar_link
```

(The local `import` inside `_write_from_svar` is for clarity; if the linter prefers top-of-file, move `from ._svar_link import SvarFingerprint, SvarLink` to the existing imports block at the top of the file and drop the inline imports.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_write_from_svar_records_svar_link_and_no_symlink -v`
Expected: PASS.

- [ ] **Step 5: Run all dataset tests**

Run: `pixi run -e dev pytest tests/dataset/ -v -x --timeout=120`
Expected: All PASS. (Existing tests should still work because read-side resolution is changed in Task 6, but they ran from the existing symlink layout — until you regenerate, you'll have a mix. Re-run `pixi run -e dev gen` first if pre-existing fixture data still expects the old layout. If gen-produced data is read-only test input, it's fine to leave; tests in this file produce their own dataset via `gvl.write`.)

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_write.py tests/dataset/test_svar_link.py
git commit -m "feat(write): record SvarLink in metadata, drop link.svar symlink"
```

---

## Task 6: Implement `_resolve_svar` and `_verify_fingerprint`

**Files:**
- Modify: `python/genvarloader/_dataset/_svar_link.py`
- Test: `tests/dataset/test_svar_link.py`

- [ ] **Step 1: Write failing tests for each resolution branch**

Append to `tests/dataset/test_svar_link.py`:

```python
from genvarloader._dataset._svar_link import _resolve_svar, _verify_fingerprint


def _make_link(svar_path: Path, gvl_path: Path, override_bytes: int | None = None):
    return SvarLink(
        relative_path=str(Path("../") / svar_path.name),  # not necessarily correct, test resolver handles it
        absolute_path=str(svar_path.resolve()),
        fingerprint=SvarFingerprint(
            n_variants=10,
            variant_idxs_bytes=override_bytes
            if override_bytes is not None
            else (svar_path / "variant_idxs.npy").stat().st_size,
        ),
    )


def test_resolve_svar_prefers_override(svar_dataset_paths, tmp_path):
    gvl_path, svar_path = svar_dataset_paths
    # Pretend the recorded paths are nonsense — override still wins.
    link = SvarLink(
        relative_path="does/not/exist",
        absolute_path="/does/not/exist",
        fingerprint=SvarFingerprint(
            n_variants=1,
            variant_idxs_bytes=(svar_path / "variant_idxs.npy").stat().st_size,
        ),
    )
    resolved = _resolve_svar(gvl_path, link, override=svar_path)
    assert resolved == svar_path


def test_resolve_svar_uses_relative_path(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    metadata = Metadata.model_validate_json((gvl_path / "metadata.json").read_text())
    resolved = _resolve_svar(gvl_path, metadata.svar_link, override=None)
    assert resolved.resolve() == svar_path.resolve()


def test_resolve_svar_falls_back_to_sibling(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    sibling_target = gvl_path.parent / "sibling.svar"
    shutil.copytree(svar_path, sibling_target)

    # Build a link with broken relative/absolute paths.
    link = SvarLink(
        relative_path="nowhere",
        absolute_path="/nowhere",
        fingerprint=SvarFingerprint(
            n_variants=1,
            variant_idxs_bytes=(sibling_target / "variant_idxs.npy").stat().st_size,
        ),
    )
    resolved = _resolve_svar(gvl_path, link, override=None)
    assert resolved.resolve() == sibling_target.resolve()


def test_resolve_svar_raises_when_not_found(tmp_path):
    gvl_path = tmp_path / "ds.gvl"
    gvl_path.mkdir()
    link = SvarLink(
        relative_path="nowhere",
        absolute_path="/nowhere",
        fingerprint=SvarFingerprint(n_variants=1, variant_idxs_bytes=1),
    )
    with pytest.raises(FileNotFoundError, match="svar="):
        _resolve_svar(gvl_path, link, override=None)


def test_verify_fingerprint_mismatch_raises(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    bogus_link = SvarLink(
        relative_path=str(svar_path),
        absolute_path=str(svar_path),
        fingerprint=SvarFingerprint(n_variants=999999, variant_idxs_bytes=1),
    )
    with pytest.raises(ValueError, match="fingerprint"):
        _verify_fingerprint(svar_path, bogus_link)


def test_verify_fingerprint_ok(svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    metadata = Metadata.model_validate_json((gvl_path / "metadata.json").read_text())
    _verify_fingerprint(svar_path, metadata.svar_link)  # does not raise
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: the new resolver/verifier tests FAIL — `_resolve_svar`, `_verify_fingerprint` not defined.

- [ ] **Step 3: Implement `_resolve_svar` and `_verify_fingerprint`**

Edit `python/genvarloader/_dataset/_svar_link.py`, appending:

```python
import json


def _resolve_svar(
    gvl_path: Path,
    link: SvarLink | None,
    override: Path | str | None,
) -> Path:
    """Resolve the SVAR directory referenced by a GVL dataset.

    Order: override → link.relative_path → link.absolute_path → sibling *.svar.
    Raises FileNotFoundError if nothing resolves.
    """
    if override is not None:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(
                f"svar override path does not exist or is not a directory: {p}"
            )
        return p

    if link is not None:
        rel = (gvl_path / link.relative_path).resolve()
        if rel.is_dir():
            return rel
        absp = Path(link.absolute_path)
        if absp.is_dir():
            return absp

    siblings = sorted(gvl_path.parent.glob("*.svar"))
    if len(siblings) == 1:
        return siblings[0]

    expected = (
        Path(link.absolute_path).name if link is not None else "<unknown>.svar"
    )
    raise FileNotFoundError(
        f"Could not locate svar '{expected}' for GVL dataset at {gvl_path}. "
        f"Tried: stored relative path, stored absolute path, sibling *.svar. "
        f"Pass `svar=` to `Dataset.open(...)` to override."
    )


def _verify_fingerprint(svar_path: Path, link: SvarLink | None) -> None:
    """Compare the recorded fingerprint against the resolved svar. Raises ValueError on mismatch.

    Skips silently when `link` is None (legacy dataset path).
    """
    if link is None:
        return

    variant_idxs = svar_path / "variant_idxs.npy"
    if not variant_idxs.exists():
        raise FileNotFoundError(
            f"Expected variant_idxs.npy at {variant_idxs}; the resolved svar is malformed."
        )

    observed_bytes = variant_idxs.stat().st_size

    # SparseVar's own metadata records n_variants implicitly via its index;
    # to avoid importing genoray here, read variant_idxs.npy's shape via numpy header.
    # But simpler: trust the byte size as the primary check, and read index.arrow row count.
    try:
        import polars as pl

        n_variants_observed = pl.scan_ipc(svar_path / "index.arrow").select(
            pl.len()
        ).collect().item()
    except Exception as exc:
        raise ValueError(
            f"Could not read variant index at {svar_path / 'index.arrow'}: {exc}"
        ) from exc

    exp = link.fingerprint
    mismatches = []
    if n_variants_observed != exp.n_variants:
        mismatches.append(
            f"n_variants: expected {exp.n_variants}, observed {n_variants_observed}"
        )
    if observed_bytes != exp.variant_idxs_bytes:
        mismatches.append(
            f"variant_idxs_bytes: expected {exp.variant_idxs_bytes}, "
            f"observed {observed_bytes}"
        )
    if mismatches:
        raise ValueError(
            f"svar fingerprint mismatch at {svar_path}: " + "; ".join(mismatches)
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add python/genvarloader/_dataset/_svar_link.py tests/dataset/test_svar_link.py
git commit -m "feat(dataset): add _resolve_svar and _verify_fingerprint"
```

---

## Task 7: Wire the resolver into `Haps.from_path` with version-dispatched legacy fallback

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py:219-275`
- Modify: `python/genvarloader/_dataset/_impl.py:121-225`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_svar_link.py`:

```python
import warnings


def test_open_dataset_after_relocation_via_override(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    moved = tmp_path / "moved.svar"
    shutil.move(str(svar_path), str(moved))

    ds = gvl.Dataset.open(gvl_path, svar=moved)
    _ = ds[0, 0]  # force eager load


def test_open_dataset_mismatched_svar_raises(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    fake = tmp_path / "fake.svar"
    shutil.copytree(svar_path, fake)
    # Truncate variant_idxs.npy to change its size (but keep file usable as path).
    target = fake / "variant_idxs.npy"
    target.write_bytes(target.read_bytes()[:-8])
    with pytest.raises(ValueError, match="fingerprint"):
        gvl.Dataset.open(gvl_path, svar=fake)


def test_open_dataset_legacy_symlink_layout(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    # Synthesize a legacy dataset: bump version down, strip svar_link, restore symlink.
    meta_path = gvl_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.18.0"
    meta.pop("svar_link", None)
    meta_path.write_text(json.dumps(meta))
    (gvl_path / "genotypes" / "link.svar").symlink_to(
        svar_path.resolve(), target_is_directory=True
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = gvl.Dataset.open(gvl_path)
        _ = ds[0, 0]
        assert any(
            issubclass(w.category, DeprecationWarning)
            and "link.svar" in str(w.message)
            for w in caught
        )
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_open_dataset_after_relocation_via_override -v`
Expected: FAIL — `Dataset.open` does not accept `svar=`.

- [ ] **Step 3: Update `Haps.from_path`**

In `python/genvarloader/_dataset/_reconstruct.py`, change the `from_path` signature (around line 219-231) by adding two new keyword params after `version`:

```python
        version: SemanticVersion | None,
        svar_link: "SvarLink | None" = None,
        svar_override: Path | str | None = None,
```

Add imports near the top of the file:

```python
import warnings
from pathlib import Path

from ._svar_link import SvarLink, _resolve_svar, _verify_fingerprint
```

Replace the body of the `if svar_meta_path.exists():` branch (currently `_reconstruct.py:235-260`) — which is currently keyed on the existence of `svar_meta.json` — with version-dispatched logic:

```python
        if svar_meta_path.exists():
            with open(svar_meta_path) as f:
                metadata = json.load(f)
            shape = cast(tuple[int, ...], tuple(metadata["shape"]))
            dtype = np.dtype(metadata["dtype"])

            offset_path = path / "genotypes" / "offsets.npy"

            # Decide which layout to read from.
            NEXT_VERSION = SemanticVersion.parse("0.25.0")  # substitute literal
            is_new_layout = (
                svar_link is not None
                and version is not None
                and version >= NEXT_VERSION
            )

            if is_new_layout:
                svar_path = _resolve_svar(path, svar_link, svar_override)
                _verify_fingerprint(svar_path, svar_link)
            else:
                legacy_link = path / "genotypes" / "link.svar"
                if not legacy_link.exists():
                    raise FileNotFoundError(
                        f"Legacy GVL dataset at {path} is missing its link.svar "
                        f"symlink and has no svar_link metadata. "
                        f"Pass `svar=` to Dataset.open(...) to recover, or "
                        f"re-run `gvl.write`."
                    )
                warnings.warn(
                    f"GVL dataset at {path} uses the legacy link.svar symlink. "
                    f"Run `genvarloader.migrate_svar_link(path)` to upgrade.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                svar_path = legacy_link.resolve()

            geno_path = svar_path / "variant_idxs.npy"
            dosage_path = svar_path / "dosages.npy"

            offsets = np.memmap(offset_path, shape=shape, dtype=dtype, mode="r")
            v_idxs = np.memmap(geno_path, dtype=V_IDX_TYPE, mode="r")
            rag_shape = (*shape[1:], None)
            genotypes = Ragged.from_offsets(v_idxs, rag_shape, offsets.reshape(2, -1))

            if dosage_path.exists():
                dosages = np.memmap(dosage_path, dtype=DOSAGE_TYPE, mode="r")
                dosages = Ragged.from_offsets(
                    dosages, rag_shape, offsets.reshape(2, -1)
                )

            logger.info("Loading variant data.")
            variants = _Variants.from_table(svar_path / "index.arrow")
```

(Note: the `dosages = None` on line 233 of the original is preserved before the `if`.)

- [ ] **Step 4: Update `Dataset.open` in `_impl.py`**

In `python/genvarloader/_dataset/_impl.py`, add `svar: str | Path | None = None` as a keyword-only argument to all three `Dataset.open` definitions (two `@overload` + the impl, currently lines 86-133). Example for the implementation:

```python
    @staticmethod
    def open(
        path: str | Path,
        reference: str | Path | Reference | None = None,
        jitter: int = 0,
        rng: int | np.random.Generator | None = False,
        deterministic: bool = True,
        rc_neg: bool = True,
        min_af: float | None = None,
        max_af: float | None = None,
        region_names: str | None = None,
        splice_info: str | tuple[str, str] | None = None,
        var_filter: Literal["exonic"] | None = None,
        *,
        svar: str | Path | None = None,
    ) -> RaggedDataset[MaybeRSEQ, MaybeRTRK]:
```

(Apply the same trailing `*, svar: str | Path | None = None` to the two `@overload` stubs.)

In the docstring, under Parameters, add:

```
        svar
            Override the recorded SVAR location. Use when the original SVAR has
            moved and the dataset cannot find it via the stored relative/absolute
            path or by sibling discovery.
```

Then, at the `Haps.from_path(...)` call site (currently `_impl.py:209-219`), thread the new args:

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

- [ ] **Step 5: Run all the new tests**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: all PASS.

- [ ] **Step 6: Run the full dataset test suite to catch regressions**

Run: `pixi run -e dev pytest tests/dataset/ -v -x --timeout=120`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_impl.py tests/dataset/test_svar_link.py
git commit -m "feat(dataset): version-dispatched svar resolution with legacy fallback"
```

---

## Task 8: Implement `migrate_svar_link` and expose it on the package

**Files:**
- Modify: `python/genvarloader/_dataset/_svar_link.py`
- Modify: `python/genvarloader/__init__.py`
- Test: `tests/dataset/test_svar_link.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/dataset/test_svar_link.py`:

```python
def test_migrate_svar_link_upgrades_legacy_dataset(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    # Synthesize a legacy dataset (same as the legacy-open test).
    meta_path = gvl_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.18.0"
    meta.pop("svar_link", None)
    meta_path.write_text(json.dumps(meta))
    (gvl_path / "genotypes" / "link.svar").symlink_to(
        svar_path.resolve(), target_is_directory=True
    )

    gvl.migrate_svar_link(gvl_path)

    upgraded = json.loads(meta_path.read_text())
    assert "svar_link" in upgraded and upgraded["svar_link"] is not None
    assert upgraded["version"] == "NEXT_VERSION"  # substitute literal
    assert not (gvl_path / "genotypes" / "link.svar").exists()

    # Dataset still opens normally — no DeprecationWarning this time.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ds = gvl.Dataset.open(gvl_path)
        _ = ds[0, 0]
        assert not any(issubclass(w.category, DeprecationWarning) for w in caught)


def test_migrate_svar_link_is_idempotent(svar_dataset_paths):
    gvl_path, _ = svar_dataset_paths
    # Already on the new layout from the fixture; migrate is a no-op.
    before = (gvl_path / "metadata.json").read_text()
    gvl.migrate_svar_link(gvl_path)
    after = (gvl_path / "metadata.json").read_text()
    assert before == after


def test_migrate_svar_link_refuses_dangling_symlink(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths
    meta_path = gvl_path / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["version"] = "0.18.0"
    meta.pop("svar_link", None)
    meta_path.write_text(json.dumps(meta))
    (gvl_path / "genotypes" / "link.svar").symlink_to(
        tmp_path / "does_not_exist.svar", target_is_directory=True
    )
    with pytest.raises(FileNotFoundError):
        gvl.migrate_svar_link(gvl_path)
```

- [ ] **Step 2: Run to verify failure**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_migrate_svar_link_upgrades_legacy_dataset -v`
Expected: FAIL — `gvl.migrate_svar_link` is undefined.

- [ ] **Step 3: Implement `migrate_svar_link`**

Append to `python/genvarloader/_dataset/_svar_link.py`:

```python
import os
from typing import Union

from pydantic_extra_types.semantic_version import SemanticVersion

NEXT_VERSION = SemanticVersion.parse("0.25.0")  # substitute literal at top of file


def migrate_svar_link(gvl_path: Union[str, Path]) -> None:
    """Upgrade a pre-NEXT_VERSION GVL dataset's `link.svar` symlink into
    a `Metadata.svar_link` entry, then remove the symlink.

    Idempotent. No-op if the dataset already carries `svar_link`.
    Raises FileNotFoundError if the legacy symlink is dangling.
    """
    gvl_path = Path(gvl_path)
    meta_path = gvl_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json at {meta_path}")

    raw = json.loads(meta_path.read_text())
    if raw.get("svar_link") is not None:
        return  # already migrated

    symlink = gvl_path / "genotypes" / "link.svar"
    if not symlink.exists() and not symlink.is_symlink():
        return  # dataset has no svar dependency; nothing to do

    target = symlink.resolve(strict=False)
    if not target.is_dir():
        raise FileNotFoundError(
            f"link.svar at {symlink} points to {target}, which does not exist. "
            f"Cannot migrate."
        )

    variant_idxs = target / "variant_idxs.npy"
    import polars as pl  # delayed to keep cold-import cheap

    n_variants = pl.scan_ipc(target / "index.arrow").select(pl.len()).collect().item()

    link = SvarLink(
        relative_path=os.path.relpath(target, start=gvl_path).replace(os.sep, "/"),
        absolute_path=str(target),
        fingerprint=SvarFingerprint(
            n_variants=n_variants,
            variant_idxs_bytes=variant_idxs.stat().st_size,
        ),
    )

    raw["svar_link"] = link.model_dump()
    raw["version"] = str(NEXT_VERSION)

    # Atomic-ish rewrite.
    tmp = meta_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(raw))
    tmp.replace(meta_path)

    symlink.unlink()
```

- [ ] **Step 4: Re-export from the package**

Edit `python/genvarloader/__init__.py` — append (or place with other imports):

```python
from ._dataset._svar_link import migrate_svar_link  # noqa: F401
```

If there is an `__all__`, add `"migrate_svar_link"` to it.

- [ ] **Step 5: Run tests**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add python/genvarloader/_dataset/_svar_link.py python/genvarloader/__init__.py tests/dataset/test_svar_link.py
git commit -m "feat: add migrate_svar_link for upgrading legacy datasets"
```

---

## Task 9: Roundtrip-relocation test (relative path preserved)

**Files:**
- Test: `tests/dataset/test_svar_link.py`

- [ ] **Step 1: Add the test**

Append to `tests/dataset/test_svar_link.py`:

```python
def test_open_after_joint_relocation_preserves_relative(tmp_path, svar_dataset_paths):
    gvl_path, svar_path = svar_dataset_paths

    # Move both into a new parent, preserving sibling layout.
    new_parent = tmp_path / "relocated"
    new_parent.mkdir()
    new_gvl = new_parent / gvl_path.name
    new_svar = new_parent / svar_path.name
    shutil.move(str(gvl_path), str(new_gvl))
    shutil.move(str(svar_path), str(new_svar))

    ds = gvl.Dataset.open(new_gvl)
    _ = ds[0, 0]
```

- [ ] **Step 2: Run it**

Run: `pixi run -e dev pytest tests/dataset/test_svar_link.py::test_open_after_joint_relocation_preserves_relative -v`
Expected: PASS — relative_path resolves correctly.

If it fails, the relative path computed in Task 5 likely used the wrong base; double-check it computes relative to `out_dir.parent` (i.e., the dataset root), not `out_dir` itself.

- [ ] **Step 3: Commit**

```bash
git add tests/dataset/test_svar_link.py
git commit -m "test: cover GVL+svar joint relocation via stored relative path"
```

---

## Task 10: Write `docs/source/format.md` and update toctree

**Files:**
- Create: `docs/source/format.md`
- Modify: `docs/source/index.md`

- [ ] **Step 1: Create the format doc**

Create `docs/source/format.md`:

````markdown
# Dataset format

A GVL dataset is a directory written by [`gvl.write`](api.md#genvarloader.write) and read
by [`gvl.Dataset.open`](api.md#genvarloader.Dataset.open). This page is the authoritative
description of its on-disk layout.

## Directory layout

```
dataset_dir/
├── metadata.json          # the Metadata schema (below)
├── input_regions.arrow    # original BED regions + region-index map
├── genotypes/             # present iff variants were provided to gvl.write
│   ├── offsets.npy        # per (region, sample, ploidy) offsets into variant_idxs.npy
│   ├── svar_meta.json     # shape + dtype of offsets.npy — present iff source was .svar
│   ├── variant_idxs.npy   # variant indices; absent when sourced from .svar
│   ├── dosages.npy        # optional, absent when sourced from .svar
│   └── variants.arrow     # variant table; absent when sourced from .svar
└── intervals/             # or annot_intervals/ when annotated; present iff tracks given
```

When the dataset was built from an `.svar`, the heavy per-variant arrays (`variant_idxs.npy`,
`dosages.npy`, `index.arrow`) are **not duplicated** into the dataset. Instead the dataset
records a back-reference to the source `.svar` in `metadata.json` (see `svar_link` below).

## `metadata.json` schema

`metadata.json` is the serialization of `genvarloader._dataset._write.Metadata`:

| Field | Type | Notes |
|-------|------|-------|
| `samples` | `list[str]` | Sample identifiers, sorted. |
| `contigs` | `list[str]` | Contig names used to interpret BED coords. |
| `n_regions` | `int` | Number of regions (after jitter padding). |
| `ploidy` | `int \| None` | Ploidy when the dataset has genotypes. |
| `max_jitter` | `int` | Maximum coordinate jitter (defaults to 0). |
| `version` | `SemanticVersion \| None` | Package version that wrote this dataset. Drives format dispatch. |
| `svar_link` | `SvarLink \| None` | Back-reference to a source `.svar`, when present. |

`SvarLink`:

| Field | Type | Notes |
|-------|------|-------|
| `relative_path` | `str` | POSIX path from `dataset_dir` to the `.svar`. |
| `absolute_path` | `str` | Original absolute path; used as a fallback. |
| `fingerprint` | `SvarFingerprint` | Integrity check (see below). |

`SvarFingerprint`:

| Field | Type | Notes |
|-------|------|-------|
| `n_variants` | `int` | Row count of the svar's `index.arrow`. |
| `variant_idxs_bytes` | `int` | Byte size of the svar's `variant_idxs.npy`. |

## SVAR resolution at open time

When opening a dataset whose `metadata.svar_link` is non-null,
[`Dataset.open`](api.md#genvarloader.Dataset.open) resolves the svar in this order:

1. Caller-provided `svar=...` argument.
2. `svar_link.relative_path` resolved against the dataset directory.
3. `svar_link.absolute_path`.
4. A unique `*.svar` directory next to the dataset.

If none match, a `FileNotFoundError` is raised naming the expected `.svar` basename. After
resolution, the fingerprint is verified; a mismatch raises `ValueError` and lists both
expected and observed values.

## Format changelog

| Version | Change |
|---------|--------|
| `< 0.18.0` | Variant coordinates stored 0-based. |
| `0.18.0` | Variant coordinates switched to 1-based. |
| `NEXT_VERSION` | `metadata.json` gains `svar_link`; old `genotypes/link.svar` symlink layout deprecated. `Metadata.version` typed as `SemanticVersion` (on-disk JSON unchanged). |

> **Upgrading legacy datasets.** A dataset written before `NEXT_VERSION` that was built from
> an `.svar` will still open (with a `DeprecationWarning`). Run
> `genvarloader.migrate_svar_link(path)` to convert the symlink layout to the new metadata
> layout in place.

````

Substitute the literal `NEXT_VERSION` value in the changelog table.

- [ ] **Step 2: Add to toctree**

In `docs/source/index.md`, edit the toctree at the top:

```
dataset
write
format
geuvadis
...
```

(Insert `format` after `write` and before `geuvadis`.)

- [ ] **Step 3: Build the docs to confirm rendering**

Run: `pixi run -e docs doc`
Expected: build succeeds with no warnings about missing references.

- [ ] **Step 4: Commit**

```bash
git add docs/source/format.md docs/source/index.md
git commit -m "docs: add dataset on-disk format reference + changelog"
```

---

## Task 11: Final verification

- [ ] **Step 1: Full test suite**

Run: `pixi run -e dev pytest tests/ -v --timeout=300`
Expected: all tests pass.

- [ ] **Step 2: Lint**

Run: `pixi run -e dev ruff check python/`
Then: `pixi run -e dev basedpyright python/`
Expected: clean (or only pre-existing issues).

- [ ] **Step 3: Confirm no `Version` from `packaging` left in the project**

Run: `grep -rn 'from packaging.version' python/`
Expected: no matches (or only matches outside the dataset module if any).

- [ ] **Step 4: Confirm no `link.svar` writes remain**

Run: `grep -rn 'link.svar' python/`
Expected: only matches inside the legacy-fallback branch of `_reconstruct.py` and the migration code in `_svar_link.py`.

- [ ] **Step 5: Commit any cleanup**

If any small fixups are needed, commit them now. Otherwise this is a no-op.
