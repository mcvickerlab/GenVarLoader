# Test Suite Overhaul — Phase 5 svar_link Component Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract 7 pure-pydantic-model unit tests from `tests/integration/dataset/test_svar_link.py` into `tests/unit/dataset/test_svar_link_models.py`. The 14 dataset-dependent tests stay behind.

**Architecture:** The first 6 tests in the source file (`test_svar_link_roundtrip` through `test_semantic_version_ordering_for_one_based_dispatch`) are pure Pydantic model exercises — JSON roundtrip, validation errors, default values, `SemanticVersion` ordering. A 7th test (`test_resolve_svar_raises_when_not_found`) uses only `tmp_path` to verify an error path in `_resolve_svar`. None of these need `gvl.write` or a real SVAR fixture. Everything else in the file uses the `svar_dataset_paths` fixture (which calls `gvl.write`) — stays in integration.

**Tech Stack:** pytest, pydantic, `pydantic_extra_types.semantic_version`, `genvarloader._dataset._svar_link`, `genvarloader._dataset._write.Metadata`. No new builders, no production code changes.

**Reference docs:**
- Spec: `docs/superpowers/specs/2026-05-24-test-suite-overhaul-design.md`
- Audit: `docs/superpowers/specs/2026-05-24-test-audit.md` (per-file row for `test_svar_link.py`, lines 282-302)

**Diverging from audit:** The audit counts 8 Port tests, including `test_resolve_svar_prefers_override` (line 289) and `test_resolve_svar_falls_back_to_sibling` (line 291). Both are technically portable, but their CURRENT bodies use the `svar_dataset_paths` fixture (which writes a GVL via `gvl.write`). Rewriting them to use only `tmp_path` is real work — it requires synthesizing a fake SVAR directory with a matching `variant_idxs.npy` fingerprint. Per YAGNI, this plan leaves them in integration and revisits when there's reason to invest in the rewrite. Net moves: 7 (audit count: 8).

---

## Pre-flight baseline (after Phase 5 splice)

- Non-slow tier: **351 passed, 3 skipped, 3 deselected, 2 xfailed**
- Coverage: **63%**
- Unit tier: **122 passed, 1 xfailed**

Counts unchanged after this plan (relocations only).

---

## Test classification

### `tests/integration/dataset/test_svar_link.py` — ~21 tests total

**Port (7) → `tests/unit/dataset/test_svar_link_models.py`:**

| Test | Why portable |
|---|---|
| `test_svar_link_roundtrip` | JSON roundtrip on `SvarLink` — pure Pydantic |
| `test_svar_link_rejects_malformed_fingerprint` | Validation error on malformed JSON — pure Pydantic |
| `test_metadata_version_parses_existing_strings` | `Metadata` validates version strings — pure Pydantic |
| `test_metadata_version_serializes_back_to_string` | `Metadata.model_dump_json` round-trip — pure Pydantic |
| `test_metadata_svar_link_defaults_to_none` | Default value on `Metadata` — pure Pydantic |
| `test_semantic_version_ordering_for_one_based_dispatch` | `SemanticVersion` comparison ordering — no library code |
| `test_resolve_svar_raises_when_not_found` | `_resolve_svar` error path; uses only `tmp_path`, no `svar_dataset_paths` |

**Keep-as-integration (~14) → stay in `tests/integration/dataset/test_svar_link.py`:**

All tests that take `svar_dataset_paths` (the `gvl.write`-based fixture) or otherwise require a real written dataset. The Keep set includes write-roundtrip tests, `_resolve_svar` tests that use the fixture, fingerprint-verification tests, and the `Dataset.open` / `migrate_svar_link` integration coverage.

---

## Task 1: Extract 7 unit tests + trim integration file (atomic)

**Files:**
- Create: `tests/unit/dataset/test_svar_link_models.py` (7 extracted tests + imports they need)
- Modify: `tests/integration/dataset/test_svar_link.py` (delete the 7 extracted tests; everything else stays untouched)

### Step 1: Create `tests/unit/dataset/test_svar_link_models.py`

Write the new unit file with this exact content. Bodies are verbatim from source lines 1-72 plus the `test_resolve_svar_raises_when_not_found` body at lines 143-152.

- [ ] **Write the file**

```python
"""Unit tests for ``SvarLink`` / ``Metadata`` Pydantic models and pure
``_resolve_svar`` error-path logic.

Originally lived in ``tests/integration/dataset/test_svar_link.py``;
extracted to the unit tier because each test here either exercises only
Pydantic model construction/validation/serialization or uses ``tmp_path``
to drive an error path in ``_resolve_svar``. None requires a written GVL
dataset.

The 14 integration-tier tests in the original file all take the
``svar_dataset_paths`` fixture (which calls ``gvl.write``) or otherwise
require a real written dataset — they remain in the integration tier.
"""

import json

import pytest
from pydantic import ValidationError
from pydantic_extra_types.semantic_version import SemanticVersion

from genvarloader._dataset._svar_link import (
    SvarFingerprint,
    SvarLink,
    _resolve_svar,
)
from genvarloader._dataset._write import Metadata


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
    bad = (
        '{"relative_path":"a","absolute_path":"b",'
        '"fingerprint":{"n_variants":"not_an_int","variant_idxs_bytes":1}}'
    )
    with pytest.raises(ValidationError):
        SvarLink.model_validate_json(bad)


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


def test_semantic_version_ordering_for_one_based_dispatch():
    """The legacy comparison '>= 0.18.0' must still work under SemanticVersion."""
    assert SemanticVersion.parse("0.18.0") >= SemanticVersion.parse("0.18.0")
    assert SemanticVersion.parse("0.20.0") >= SemanticVersion.parse("0.18.0")
    assert not (SemanticVersion.parse("0.17.5") >= SemanticVersion.parse("0.18.0"))


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
```

Note: the source imports `_verify_fingerprint` alongside the other names. The unit file does NOT need `_verify_fingerprint` — none of the 7 extracted tests use it. Omit it from the import to keep the unit file's imports tight.

### Step 2: Run the new unit file alone

```
pixi run -e dev pytest tests/unit/dataset/test_svar_link_models.py -q 2>&1 | tail -3
```

Expected: **7 passed**.

### Step 3: Trim the integration file

Remove the 7 extracted test functions from `tests/integration/dataset/test_svar_link.py`:

1. Lines 18-26: `test_svar_link_roundtrip`
2. Lines 29-35: `test_svar_link_rejects_malformed_fingerprint`
3. Lines 38-49: `test_metadata_version_parses_existing_strings`
4. Lines 52-60: `test_metadata_version_serializes_back_to_string`
5. Lines 63-65: `test_metadata_svar_link_defaults_to_none`
6. Lines 68-72: `test_semantic_version_ordering_for_one_based_dispatch`
7. Lines 143-152: `test_resolve_svar_raises_when_not_found`

For each block: delete the function definition, its decorator (none in this case — all 7 are plain `def`), and the blank lines that frame it.

After the 6 contiguous deletions at the top (lines 18-72), the imports remain valid because the 14 Keep tests still need `pytest`, `shutil`, `Path`, `pydantic.ValidationError`, `SemanticVersion`, `SvarFingerprint`, `SvarLink`, `_resolve_svar`, `_verify_fingerprint`, and `Metadata`. **Do not modify the imports** — they all stay.

After deleting `test_resolve_svar_raises_when_not_found` (lines 143-152), the surrounding context is unchanged: line 142 ends the previous test, line 153 onwards (`test_verify_fingerprint_mismatch_raises` at line 155 in original numbering) is the next Keep test.

The `svar_dataset_paths` fixture (lines 75-87) and all 14 Keep tests stay.

- [ ] **Edit the integration file**

The cleanest way to do this surgically: read the file's content, locate each test-function block by name (`def test_svar_link_roundtrip(...)`, etc.), and remove just that function plus its trailing two blank lines.

After the edits, the file's top section should look like:

```python
import json
import shutil
from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_extra_types.semantic_version import SemanticVersion

from genvarloader._dataset._svar_link import (
    SvarFingerprint,
    SvarLink,
    _resolve_svar,
    _verify_fingerprint,
)
from genvarloader._dataset._write import Metadata


@pytest.fixture
def svar_dataset_paths(tmp_path, filtered_svar, source_bed):
    ...
```

i.e., the imports stay, the 6 pre-fixture tests are gone, and the `svar_dataset_paths` fixture comes directly after the import block. Similarly, `test_resolve_svar_raises_when_not_found` is gone from its middle position.

**Imports audit:** Even though all imports are still referenced by the remaining 14 tests, run a quick sanity grep after editing:

```
grep -nE "json|shutil|Path|ValidationError|SemanticVersion|SvarFingerprint|SvarLink|_resolve_svar|_verify_fingerprint|Metadata" tests/integration/dataset/test_svar_link.py
```

Every name from the imports must appear at least once outside the import block.

### Step 4: Run the trimmed integration file

```
pixi run -e dev pytest tests/integration/dataset/test_svar_link.py -q 2>&1 | tail -3
```

Expected: **14 passed** (was 21 passed; lost 7 to the unit tier). Adjust ±1 if the original had a parametrize that I missed.

### Step 5: Full non-slow suite

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: `351 passed, 3 skipped, 3 deselected, 2 xfailed` (totals unchanged — 7 tests moved tier).

### Step 6: Commit atomically

```bash
git add tests/integration/dataset/test_svar_link.py tests/unit/dataset/test_svar_link_models.py
git commit -m "test: extract svar_link/Metadata pydantic unit tests to unit/dataset/"
```

Verify `git status` clean.

---

## Task 2: End-of-plan verification

- [ ] **Step 1: Full non-slow suite**

```
pixi run -e dev pytest tests -m "not slow" -q 2>&1 | tail -2
```

Expected: **`351 passed, 3 skipped, 3 deselected, 2 xfailed`**.

- [ ] **Step 2: Unit tier collection**

```
pixi run -e dev pytest tests/unit -q 2>&1 | tail -2
```

Expected: roughly **129 passed, 1 xfailed** (was 122/1; +7 from svar_link extraction).

- [ ] **Step 3: Coverage parity**

```
pixi run -e dev pytest tests -m "not slow" --cov --cov-report=term -q 2>&1 | grep "^TOTAL"
```

Expected: `TOTAL ... 63%` (±1pp).

- [ ] **Step 4: File structure**

```
/bin/ls tests/unit/dataset/
```

Expected: `genotypes/`, `test_build_reconstructor.py`, `test_indexing.py`, `test_svar_link_models.py` (new). Plus possibly a `.gitkeep`.

```
/bin/ls tests/integration/dataset/test_svar_link.py
```

Expected: file exists, contains 14 Keep tests.

- [ ] **Step 5: Commit graph**

```
git log --oneline -3
```

Expected:
```
<sha> test: extract svar_link/Metadata pydantic unit tests to unit/dataset/
<previous Phase 5 splice head>
```

---

## Out of scope (deferred to subsequent component plans)

- **tracks (broader)** — `test_random_nonoverlapping.py` (1 port), `test_write_tracks.py:test_write_duplicate_track_names_rejected` (1 port), `test_table.py` (12 ports).
- **ref/fasta** — `test_fasta.py` (3 ports), `test_ref_ds.py` (2 remaining ports).
- **utility** — `test_utils.py` (5 ports).
- **dataset polymorphism** — `test_with_insertion_fill_rejects_when_no_tracks_active` (once `make_dataset` exists).
- **_resolve_svar rewrites** — `test_resolve_svar_prefers_override` and `test_resolve_svar_falls_back_to_sibling` could move to unit if rewritten to synthesize the SVAR fingerprint directly. Skip until there's a reason to invest in the rewrite.
- **haps** — Builder-only; deferred until a real consumer exists.
