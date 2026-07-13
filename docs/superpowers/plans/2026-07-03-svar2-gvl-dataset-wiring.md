# SVAR2 gvl Dataset Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the SVAR2 format into the gvl `Dataset` the way SVAR1 is — cache genoray's interval-search result at `gvl.write` time and replay it at read time via `gather_ranges` + the existing SVAR2 kernels — so SVAR2 haplotype/track reads stop paying a per-query `SearchTree::build`.

**Architecture:** Mirror the SVAR1 write/read wiring. `_write_from_svar2` streams `SparseVar2.find_ranges(..., out=...)` into a compact per-region/per-hap ranges cache under `genotypes/`, plus a `Svar2Link` back-reference in `metadata.json`. On read, a new `HapsSvar2` reconstructor loads the cached ranges for the requested `(region, sample)` block, calls `SparseVar2.gather_ranges` (tree-free), and feeds the payload to `reconstruct_haplotypes_from_svar2` / `shift_and_realign_tracks_from_svar2`. A source discriminant (`svar2_link` present) selects `HapsSvar2` over the SVAR1 `Haps`.

**Tech Stack:** Python 3.10+, numpy, polars, pydantic, seqpro `Ragged`, genoray `SparseVar2` (from the genoray split plan), pixi, maturin, pytest.

**Repo:** `/carter/users/dlaub/projects/GenVarLoader` worktree `svar2-m6b-kernel`. **Depends on** the shipped genoray wheel from `2026-07-03-svar2-genoray-search-gather-split.md` (`find_ranges`/`gather_ranges`/`read_ranges` with `samples=`/`out=`).

## Global Constraints

- **Depends on genoray >= `<VERSION released by the genoray split plan>`.** Before starting, confirm `pixi run -e dev python -c "from genoray import SparseVar2; SparseVar2.find_ranges; SparseVar2.gather_ranges"` succeeds; if not, bump the genoray pin in `pixi.toml` / `pyproject.toml` and `pixi run -e dev install`.
- **Byte-identical parity contract** (verbatim from spec): cached-path reconstruct ≡ live `read_ranges`/`overlap_batch` reconstruct ≡ `decode` oracle, on the M6b matrix (SNP/INS/DEL × samples × ploids) + real chr21 germline & somatic stores. Track re-alignment matched the same way.
- **Additive:** the SVAR1 path is **byte-unchanged**. The full SVAR1 regression suite stays green (`pixi run -e dev pytest tests -q`; `pixi run -e dev cargo-test` for kernels). Follows the rust-migration byte-identical parity contract and the numba-oracle-bug policy (if the cached path and a numba oracle disagree, check whether numba is the buggy one before "fixing" the new path).
- **Scope:** haplotypes + tracks **only**. SVAR2 `variants` and `annotated` output modes are out of scope (raise a clear `NotImplementedError`); the same cache extends to them later.
- **REBUILD RUST BEFORE PYTHON TESTS:** these changes are pure-Python (no `src/` edits), so `maturin develop` is *not* required for this plan — but if any parity test imports the SVAR2 kernels and you touched `src/`, run `pixi run -e dev maturin develop --release` first.
- **Docs gates:** any public-API change updates `skills/genvarloader/SKILL.md`; `api.md` stays in sync with `__all__`; user-facing docs audited (`README.md`, `docs/source/{api,write,format,faq}.md`). See CLAUDE.md's skill-maintenance + docs-audit rules.
- **Conventional commits** (commitizen). Ensure prek hooks installed before committing.

---

## File Structure

- `python/genvarloader/_dataset/_svar2_link.py` — **new**. `Svar2Fingerprint`, `Svar2Link` pydantic models, `_resolve_svar2`, `_verify_fingerprint2`. Mirrors `_svar_link.py`.
- `python/genvarloader/_dataset/_write.py` — **modify**. Add `.svar2` detection in the variant-source dispatch (~`:225`), a `SparseVar2` branch in the genotype-writing dispatch (~`:325`), and a new `_write_from_svar2(...)` (mirrors `_write_from_svar` at `:961`). Add `svar2_link` to the `Metadata` model (`:86`).
- `python/genvarloader/_dataset/_svar2_source.py` — **modify**. Add a cache-load + `gather_ranges` path; keep the live `overlap_batch` path as the parity oracle.
- `python/genvarloader/_dataset/_haps.py` — **modify**. Add `HapsSvar2` reconstructor (haplotypes-only, `Reconstructor[RaggedSeqs]`) that loads cached ranges + gathers + calls the SVAR2 kernels.
- `python/genvarloader/_dataset/_reconstruct.py` — **modify**. Add `HapsSvar2Tracks` and route `HapsSvar2` through `_build_reconstructor`.
- `python/genvarloader/_dataset/_open.py` — **modify**. `_build_seqs` constructs `HapsSvar2` when `metadata.svar2_link` is present.
- `python/genvarloader/_dataset/_migrate.py` — **verify** `svar2_link` is tolerated (additive metadata key).
- Tests: `tests/dataset/test_write_svar2.py`, `tests/dataset/test_svar2_dataset.py`, `tests/unit/dataset/test_svar2_link.py` — **new**.

---

### Task 1: `Svar2Link` model + resolution/fingerprint

**Files:**
- Create: `python/genvarloader/_dataset/_svar2_link.py`
- Test: `tests/unit/dataset/test_svar2_link.py`

**Interfaces:**
- Produces:
  ```python
  class Svar2Fingerprint(BaseModel):
      n_variants: int
      store_bytes: int
  class Svar2Link(BaseModel):
      relative_path: str
      absolute_path: str
      fingerprint: Svar2Fingerprint
  def _resolve_svar2(gvl_path: Path, link: Svar2Link | None, override: Path | str | None) -> Path
  def _verify_fingerprint2(svar2_path: Path, link: Svar2Link | None) -> None
  ```
- Consumes: nothing gvl-internal beyond the same pattern as `_svar_link.py`. Fingerprint identity for `.svar2` = `n_variants` (from the SparseVar2 index) + a byte count of a canonical store file (see Step 3).

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/test_svar2_link.py`:

```python
from pathlib import Path

import pytest

from genvarloader._dataset._svar2_link import (
    Svar2Fingerprint,
    Svar2Link,
    _resolve_svar2,
)


def _mk_svar2_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cohort.svar2"
    d.mkdir()
    (d / "index.arrow").write_bytes(b"stub")
    return d


def test_resolve_prefers_override(tmp_path):
    d = _mk_svar2_dir(tmp_path)
    other = tmp_path / "other.svar2"
    other.mkdir()
    link = Svar2Link(
        relative_path="cohort.svar2",
        absolute_path=str(d),
        fingerprint=Svar2Fingerprint(n_variants=3, store_bytes=4),
    )
    gvl = tmp_path  # pretend the gvl dataset lives here
    assert _resolve_svar2(gvl, link, other) == other


def test_resolve_falls_back_to_relative_then_absolute(tmp_path):
    d = _mk_svar2_dir(tmp_path)
    gvl = tmp_path / "ds.gvl"
    gvl.mkdir()
    import os
    rel = os.path.relpath(d, start=gvl).replace(os.sep, "/")
    link = Svar2Link(
        relative_path=rel, absolute_path=str(d),
        fingerprint=Svar2Fingerprint(n_variants=3, store_bytes=4),
    )
    assert _resolve_svar2(gvl, link, None) == d


def test_resolve_raises_when_unfindable(tmp_path):
    gvl = tmp_path / "ds.gvl"
    gvl.mkdir()
    link = Svar2Link(
        relative_path="missing.svar2", absolute_path=str(tmp_path / "missing.svar2"),
        fingerprint=Svar2Fingerprint(n_variants=3, store_bytes=4),
    )
    with pytest.raises(FileNotFoundError):
        _resolve_svar2(gvl, link, None)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_link.py -q`
Expected: FAIL — `No module named ..._svar2_link`.

- [ ] **Step 3: Implement `_svar2_link.py`**

Copy `_svar_link.py`'s structure. Fingerprint on a **stable** `.svar2` identity: `n_variants` from the SparseVar2 index (`pl.scan_ipc(<svar2>/index.arrow).select(pl.len())`) and the byte size of a canonical store file. **Verify the actual on-disk `.svar2` layout first** — `rtk ls <a real .svar2 dir>` (e.g. under `tmp/svar2_mvp/`) to confirm the index filename (`index.arrow`?) and pick one always-present store file for `store_bytes` (e.g. the largest `*.npy`/packed buffer). Do not assume `variant_idxs.npy` — that is SVAR1.

```python
"""Resolution and integrity for the GVL dataset → SVAR2 back-reference.

Mirrors _svar_link.py. SVAR2 fingerprint identity = n_variants (from the
SparseVar2 index) + byte count of a canonical store file.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Svar2Fingerprint(BaseModel):
    n_variants: int
    store_bytes: int


class Svar2Link(BaseModel):
    relative_path: str
    absolute_path: str
    fingerprint: Svar2Fingerprint


# The canonical store file used for the byte-count fingerprint. VERIFY against a
# real .svar2 directory and update if the layout differs.
_STORE_FILE = "index.arrow"


def _resolve_svar2(
    gvl_path: Path, link: Svar2Link | None, override: Path | str | None
) -> Path:
    if override is not None:
        p = Path(override)
        if not p.is_dir():
            raise FileNotFoundError(
                f"svar2 override path does not exist or is not a directory: {p}"
            )
        return p
    if link is not None:
        rel = (gvl_path / link.relative_path).resolve()
        if rel.is_dir():
            return rel
        absp = Path(link.absolute_path)
        if absp.is_dir():
            return absp
    siblings = sorted(gvl_path.parent.glob("*.svar2"))
    if len(siblings) == 1:
        return siblings[0]
    expected = Path(link.absolute_path).name if link is not None else "<unknown>.svar2"
    raise FileNotFoundError(
        f"Could not locate svar2 '{expected}' for GVL dataset at {gvl_path}. "
        f"Tried: stored relative path, stored absolute path, sibling *.svar2. "
        f"Pass `svar2=` to `Dataset.open(...)` to override."
    )


def _verify_fingerprint2(svar2_path: Path, link: Svar2Link | None) -> None:
    if link is None:
        return
    store = svar2_path / _STORE_FILE
    if not store.exists():
        raise FileNotFoundError(
            f"Expected {store}; resolved svar2 is malformed."
        )
    import polars as pl

    n_variants_observed = (
        pl.scan_ipc(svar2_path / "index.arrow").select(pl.len()).collect().item()
    )
    observed_bytes = store.stat().st_size
    exp = link.fingerprint
    mismatches: list[str] = []
    if n_variants_observed != exp.n_variants:
        mismatches.append(
            f"n_variants: expected {exp.n_variants}, observed {n_variants_observed}"
        )
    if observed_bytes != exp.store_bytes:
        mismatches.append(
            f"store_bytes: expected {exp.store_bytes}, observed {observed_bytes}"
        )
    if mismatches:
        raise ValueError(
            f"svar2 fingerprint mismatch at {svar2_path}: " + "; ".join(mismatches)
        )
```

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_link.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
cd /carter/users/dlaub/projects/GenVarLoader/.claude/worktrees/svar2-m6b-kernel
rtk git add python/genvarloader/_dataset/_svar2_link.py tests/unit/dataset/test_svar2_link.py
rtk git commit -m "feat(svar2): Svar2Link resolution + fingerprint"
```

---

### Task 2: `Metadata.svar2_link` field + migration tolerance

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` (`Metadata` model, `:86-95`)
- Test: `tests/unit/dataset/test_svar2_link.py` (extend)

**Interfaces:**
- Consumes: `Svar2Link` (Task 1).
- Produces: `Metadata.svar2_link: Svar2Link | None = None`; unchanged datasets (no `svar2_link` key) still validate; `_check_dataset_format_version` and `_migrate.migrate` tolerate the additive key.

- [ ] **Step 1: Write the failing test**

Add to `tests/unit/dataset/test_svar2_link.py`:

```python
def test_metadata_roundtrips_svar2_link():
    from genvarloader._dataset._write import Metadata
    from genvarloader._dataset._svar2_link import Svar2Fingerprint, Svar2Link

    link = Svar2Link(
        relative_path="c.svar2", absolute_path="/abs/c.svar2",
        fingerprint=Svar2Fingerprint(n_variants=5, store_bytes=99),
    )
    m = Metadata(
        contigs=["chr1"], samples=["s0"], ploidy=2, n_regions=1,
        svar2_link=link,
    )
    m2 = Metadata.model_validate_json(m.model_dump_json())
    assert m2.svar2_link is not None
    assert m2.svar2_link.fingerprint.n_variants == 5


def test_metadata_without_svar2_link_still_valid():
    from genvarloader._dataset._write import Metadata

    m = Metadata(contigs=["chr1"], samples=["s0"], ploidy=2, n_regions=1)
    assert m.svar2_link is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_link.py -k metadata -q`
Expected: FAIL — `Metadata` has no field `svar2_link` (unexpected-keyword / validation error).

- [ ] **Step 3: Add the field**

In `python/genvarloader/_dataset/_write.py`, import `Svar2Link` next to `from ._svar_link import SvarLink` and add the field to `Metadata` (after `svar_link`, `:94`):

```python
from ._svar2_link import Svar2Link  # noqa: E402  (near the SvarLink import)
...
    svar_link: SvarLink | None = None
    svar2_link: Svar2Link | None = None
```

- [ ] **Step 4: Run to verify it passes, then confirm migration tolerance**

Run:
```bash
pixi run -e dev pytest tests/unit/dataset/test_svar2_link.py -q
pixi run -e dev pytest tests/unit/dataset -k migrate -q   # existing migration tests still green
```
Expected: PASS. `_migrate.migrate` reads `metadata.json` as raw JSON and only touches `format_version`, so the additive `svar2_link` key passes through untouched — the migration tests confirm no regression. No `_migrate.py` code change needed; note this in the commit.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/unit/dataset/test_svar2_link.py
rtk git commit -m "feat(svar2): additive svar2_link metadata field"
```

---

### Task 3: `_write_from_svar2` + write dispatch

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py`
- Test: `tests/dataset/test_write_svar2.py` (create)

**Interfaces:**
- Consumes: `SparseVar2` (genoray), `SparseVar2.find_ranges(contig, starts, ends, samples=, out=)` (genoray split plan), `Svar2Link`/`Svar2Fingerprint` (Task 1), the existing `_reject_unsupported_variants` (`_write.py`), `atomic_dir`/`_prep_bed`/`_write_regions` (existing write pipeline).
- Produces:
  ```python
  def _write_from_svar2(
      path: Path, bed: pl.DataFrame, svar2: "SparseVar2",
      samples: list[str], extend_to_length: bool,
  ) -> tuple[pl.DataFrame, Svar2Link]
  ```
  Writes cache memmaps + `svar2_meta.json` under `<path>/genotypes/`. The cache layout (all int arrays, region-ordered to match `regions.npy`):
  - `svar2_dense_range.npy` int32 `(R, 2)`
  - `svar2_region_starts.npy` int32 `(R,)`
  - `svar2_vk_snp_range.npy` int64 `(R, S, P, 2)`
  - `svar2_vk_indel_range.npy` int64 `(R, S, P, 2)`
  - `svar2_meta.json` — shapes/dtypes + `sample_cols` + `n_samples`/`ploidy`

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_write_svar2.py`. Build a small `.svar2` (reuse whatever fixture/helper the existing `tests/test_svar2_reconstruct.py` uses to synthesize a `SparseVar2` store; grep it) and assert `gvl.write` produces the cache + link.

```python
import json
from pathlib import Path

import numpy as np
import polars as pl

import genvarloader as gvl


def test_write_svar2_produces_ranges_cache(svar2_store, tmp_path):
    # svar2_store: path to a small .svar2 directory (shared fixture; see conftest).
    from genoray import SparseVar2

    sv = SparseVar2(svar2_store)
    bed = pl.DataFrame(
        {"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]}
    )
    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, variants=Path(svar2_store))

    geno = out / "genotypes"
    assert (geno / "svar2_meta.json").exists()
    meta = json.loads((geno / "svar2_meta.json").read_text())
    R, S, P = 1, sv.n_samples, sv.ploidy
    dr = np.load(geno / "svar2_dense_range.npy")
    assert dr.shape == (R, 2)
    vks = np.load(geno / "svar2_vk_snp_range.npy")
    assert vks.shape == (R, S, P, 2)

    ds_meta = json.loads((out / "metadata.json").read_text())
    assert ds_meta["svar2_link"] is not None
    assert ds_meta["ploidy"] == P
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: FAIL — `write` does not recognize `.svar2` (raises the "unrecognized file extension" ValueError, or SparseVar2 branch missing).

- [ ] **Step 3: Implement detection + dispatch + `_write_from_svar2`**

3a. Import `SparseVar2` at the top of `_write.py` (next to `from genoray import ..., SparseVar`).

3b. In the variant-source dispatch (`_write.py:225`, the `elif variants.is_dir() and variants.suffix == ".svar"` branch), add **before** it:

```python
                    elif variants.is_dir() and variants.suffix == ".svar2":
                        variants = SparseVar2(variants)
```

3c. In the genotype-writing dispatch (`_write.py:325`, the `elif isinstance(variants, SparseVar):` branch), add a sibling branch:

```python
                elif isinstance(variants, SparseVar2):
                    from ._svar2_link import Svar2Link  # local import ok
                    gvl_bed, _svar2_link = _write_from_svar2(
                        path, gvl_bed, variants, samples, extend_to_length
                    )
                    metadata["svar2_link"] = _svar2_link.model_dump()
```

`metadata["ploidy"] = variants.ploidy` at `:330` already runs for any `variants` (SparseVar2 exposes `.ploidy`). Confirm `SparseVar2.available_samples` exists (used at `:233`); if the attribute is `.samples`, add `available_samples` as an alias or special-case the `available_samples` assignment for `SparseVar2`.

3d. Add `_write_from_svar2` (mirror `_write_from_svar` at `:961`; partition bed by contig, stream `find_ranges` into the cache memmaps):

```python
def _write_from_svar2(
    path: Path,
    bed: pl.DataFrame,
    svar2: "SparseVar2",
    samples: list[str],
    extend_to_length: bool,
) -> tuple[pl.DataFrame, "Svar2Link"]:
    import json
    import os

    from ._svar2_link import Svar2Fingerprint, Svar2Link

    _reject_unsupported_variants(svar2.index, "SVAR2")  # verify svar2.index exists

    out_dir = path / "genotypes"
    out_dir.mkdir(parents=True, exist_ok=True)

    R = bed.height
    S = len(samples)
    P = svar2.ploidy

    dense_range = np.memmap(out_dir / "svar2_dense_range.npy", np.int32, "w+", shape=(R, 2))
    region_starts = np.memmap(out_dir / "svar2_region_starts.npy", np.int32, "w+", shape=(R,))
    vk_snp = np.memmap(out_dir / "svar2_vk_snp_range.npy", np.int64, "w+", shape=(R, S, P, 2))
    vk_indel = np.memmap(out_dir / "svar2_vk_indel_range.npy", np.int64, "w+", shape=(R, S, P, 2))

    sample_cols: list[int] | None = None
    contig_offset = 0
    for (c,), df in bed.partition_by("chrom", as_dict=True, maintain_order=True).items():
        c = cast(str, c)
        rc = df.height
        rows = slice(contig_offset, contig_offset + rc)
        # find_ranges returns a dict bundle; stream into the cache slices via out=.
        out = {
            "dense_range": dense_range[rows],
            "region_starts": region_starts[rows],
            "sample_cols": np.empty(S, np.int64),
            "vk_snp_range": vk_snp[rows].reshape(rc * S * P, 2),
            "vk_indel_range": vk_indel[rows].reshape(rc * S * P, 2),
        }
        bundle = svar2.find_ranges(
            c, df["chromStart"], df["chromEnd"], samples=samples, out=out
        )
        if sample_cols is None:
            sample_cols = np.asarray(bundle["sample_cols"], np.int64).tolist()
        contig_offset += rc

    for m in (dense_range, region_starts, vk_snp, vk_indel):
        m.flush()

    with open(out_dir / "svar2_meta.json", "w") as f:
        json.dump(
            {
                "n_regions": R, "n_samples": S, "ploidy": P,
                "sample_cols": sample_cols,
                "dense_range": {"shape": [R, 2], "dtype": "<i4"},
                "region_starts": {"shape": [R], "dtype": "<i4"},
                "vk_snp_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "vk_indel_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
            },
            f,
        )

    svar2_resolved = Path(svar2.path).resolve()
    store_bytes = (svar2_resolved / "index.arrow").stat().st_size  # match _STORE_FILE
    svar2_link = Svar2Link(
        relative_path=os.path.relpath(svar2_resolved, start=path).replace(os.sep, "/"),
        absolute_path=str(svar2_resolved),
        fingerprint=Svar2Fingerprint(
            n_variants=svar2.index.height, store_bytes=store_bytes
        ),
    )
    # SVAR2 max_ends: the region chromEnd already bounds reads (dense/indel spans
    # are handled at read via the ranges). Keep chromEnd as-is unless a chr21
    # parity test shows track windows truncate below the rightmost indel; if so,
    # extend using the cached ranges (mirror _write_from_svar's v_ends logic).
    return bed, svar2_link
```

> Implementation notes: (1) `find_ranges`'s `out=` writes into the provided arrays; passing memmap **slices** streams straight to disk. Confirm the genoray `out=` copy handles the `(rc*S*P, 2)` reshape of a memmap slice (it is C-contiguous per contig-block since the outer axis is region). If a slice is non-contiguous, allocate a per-contig scratch array, `find_ranges(..., out=scratch)`, then assign into the memmap slice. (2) `_reject_unsupported_variants(svar2.index, ...)` — verify `SparseVar2` exposes `.index` (a polars frame with the same reject-able columns); if the schema differs, adapt or skip with a comment (upstream `-V other,bnd` already filters symbolic/breakend ALTs per the spec). (3) `svar2.path` — confirm the attribute name (`SparseVar2.__init__` stores `self.path`).

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py
rtk git commit -m "feat(svar2): write dispatch + _write_from_svar2 ranges cache"
```

---

### Task 4: `HapsSvar2` reconstructor — cached ranges + gather + kernel

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_source.py` (add cache-load + `gather_ranges` path)
- Modify: `python/genvarloader/_dataset/_haps.py` (add `HapsSvar2`)
- Test: `tests/dataset/test_svar2_dataset.py` (create)

**Interfaces:**
- Consumes: the cache memmaps + `svar2_meta.json` (Task 3), `SparseVar2.gather_ranges(contig, bundle)` (genoray split plan), `SparseVar2Source.reconstruct`/`realign_tracks` marshalling (existing, `_svar2_source.py`), `Reference` (for `ref_`/`ref_offsets`/`pad_char`), the `Reconstructor[_H]` protocol (`_protocol.py`).
- Produces:
  ```python
  @dataclass(slots=True)
  class HapsSvar2(Reconstructor[RaggedSeqs]):
      path: Path
      reference: Reference
      svar2: "SparseVar2"
      contigs: list[str]
      samples: list[str]
      ploidy: int
      # cache memmaps + sample_cols loaded in from_path
      def __call__(self, idx, r_idx, regions, output_length, jitter, rng,
                   deterministic, splice_plan=None, flat=False, to_rc=None) -> RaggedSeqs
      @classmethod
      def from_path(cls, path, reference, contigs, samples, ploidy, svar2_link, svar2_override) -> "HapsSvar2"
      def gather_block(self, contig, region_rows, sample_slot_idxs) -> dict  # bundle -> gather_ranges payload
  ```

- [ ] **Step 1: Write the failing test (cached ≡ live oracle)**

Create `tests/dataset/test_svar2_dataset.py`. Oracle = the **live** `SparseVar2Source` path (already decode-validated); assert the cached `HapsSvar2` reconstructs byte-identically.

```python
from pathlib import Path

import numpy as np
import polars as pl

import genvarloader as gvl


def test_svar2_dataset_haps_match_live_source(svar2_store, reference_fasta, tmp_path):
    from genoray import SparseVar2
    from genvarloader._dataset._svar2_source import SparseVar2Source

    sv = SparseVar2(svar2_store)
    regions = [(0, 40)]
    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})

    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, variants=Path(svar2_store))

    ds = gvl.Dataset.open(out, reference=reference_fasta).with_seqs("haplotypes")
    cached = ds[0, :]  # (S, P, ~L) ragged bytes for region 0, all samples

    # Live oracle via the adapter (contig ref bytes + offsets from the fasta).
    ref_bytes, ref_off, pad = _load_contig_ref(reference_fasta, "chr1")
    live = SparseVar2Source(sv).reconstruct(
        "chr1", regions, ref_bytes, ref_off, pad, output_length=-1
    )
    # Compare byte-for-byte over every (sample, ploid).
    np.testing.assert_array_equal(cached.to_packed().data, live.to_packed().data)
    np.testing.assert_array_equal(
        np.asarray(cached.offsets), np.asarray(live.offsets)
    )
```

Add the `_load_contig_ref` helper + `reference_fasta`/`svar2_store` fixtures to `tests/dataset/conftest.py` if absent (mirror the fixtures in `tests/test_svar2_reconstruct.py`).

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q`
Expected: FAIL — `Dataset.open` builds a plain `Haps` (no svar2 dispatch yet) or errors on the missing SVAR1 `svar_meta.json`/`offsets.npy`.

- [ ] **Step 3a: Add the cache-load + gather path to `SparseVar2Source`**

In `_svar2_source.py`, add a classmethod-free helper that builds a `gather_ranges` bundle dict from cached memmap slices and calls `self.svar2.gather_ranges`. Refactor `_query` so its `d` (payload dict) can come from **either** live `overlap_batch` (kept as oracle) or `gather_ranges(bundle)`:

```python
    def _query_cached(self, contig, regions, dense_range, region_starts,
                      vk_snp_range, vk_indel_range, sample_cols):
        """Build a find_ranges bundle from cached slices and gather it (tree-free).
        Shapes: dense_range (R,2), region_starts (R,), vk_*_range (R,S,P,2)."""
        R = dense_range.shape[0]
        S = vk_snp_range.shape[1]
        P = vk_snp_range.shape[2]
        bundle = {
            "dense_range": np.ascontiguousarray(dense_range, np.int32),
            "region_starts": np.ascontiguousarray(region_starts, np.int32),
            "sample_cols": np.ascontiguousarray(sample_cols, np.int64),
            "vk_snp_range": np.ascontiguousarray(vk_snp_range.reshape(R * S * P, 2), np.int64),
            "vk_indel_range": np.ascontiguousarray(vk_indel_range.reshape(R * S * P, 2), np.int64),
            "n_regions": R, "n_samples": S, "ploidy": P,
        }
        d = self.svar2.gather_ranges(contig, bundle)
        reg = np.asarray(regions, np.int32).reshape(R, 2)
        reg_rs = np.repeat(reg, S, axis=0)
        regions_gvl = np.zeros((R * S, 3), np.int32)
        regions_gvl[:, 1:] = reg_rs
        dense_range_gvl = np.ascontiguousarray(
            np.repeat(np.asarray(d["dense_range"], np.int32), S, axis=0), np.int32
        )
        return d, R, S, P, regions_gvl, dense_range_gvl
```

Extract the kernel-call bodies of `reconstruct`/`realign_tracks` so they accept the `(d, R, S, P, regions_gvl, dense_range_gvl)` tuple from **either** `_query` (live) or `_query_cached`. Keep the live `_query` intact — it is the parity oracle.

- [ ] **Step 3b: Add `HapsSvar2` to `_haps.py`**

`HapsSvar2` loads the cache once in `from_path`, groups a read batch by contig, and per contig gathers + reconstructs. Reference is **required** (haplotypes need ref bytes). Only `RaggedSeqs` is supported.

```python
@dataclass(slots=True)
class HapsSvar2(Reconstructor["RaggedSeqs"]):
    path: Path
    reference: Reference
    svar2: "SparseVar2"
    contigs: list[str]
    samples: list[str]
    ploidy: int
    dense_range: NDArray[np.int32]        # (R, 2) memmap
    region_starts: NDArray[np.int32]      # (R,) memmap
    vk_snp_range: NDArray[np.int64]       # (R, S, P, 2) memmap
    vk_indel_range: NDArray[np.int64]     # (R, S, P, 2) memmap
    sample_cols: NDArray[np.int64]        # (S,)

    @classmethod
    def from_path(cls, path, reference, contigs, samples, ploidy,
                  svar2_link, svar2_override):
        import json
        from genoray import SparseVar2
        from ._svar2_link import _resolve_svar2, _verify_fingerprint2

        svar2_path = _resolve_svar2(path, svar2_link, svar2_override)
        _verify_fingerprint2(svar2_path, svar2_link)
        svar2 = SparseVar2(svar2_path)
        geno = path / "genotypes"
        meta = json.loads((geno / "svar2_meta.json").read_text())
        def mm(name):
            spec = meta[name]
            return np.memmap(geno / f"svar2_{name}.npy",
                             dtype=np.dtype(spec["dtype"]),
                             mode="r", shape=tuple(spec["shape"]))
        if reference is None:
            raise ValueError("SVAR2 haplotype output requires a reference genome.")
        return cls(
            path=path, reference=reference, svar2=svar2, contigs=contigs,
            samples=samples, ploidy=ploidy,
            dense_range=mm("dense_range"), region_starts=mm("region_starts"),
            vk_snp_range=mm("vk_snp_range"), vk_indel_range=mm("vk_indel_range"),
            sample_cols=np.asarray(meta["sample_cols"], np.int64),
        )

    def to_kind(self, kind):
        from .._ragged import RaggedSeqs
        if kind is not RaggedSeqs:
            raise NotImplementedError(
                f"SVAR2 datasets support only 'haplotypes' output, not {kind.__name__}."
            )
        return self

    def __call__(self, idx, r_idx, regions, output_length, jitter, rng,
                 deterministic, splice_plan=None, flat=False, to_rc=None):
        if splice_plan is not None:
            raise NotImplementedError("Spliced SVAR2 haplotypes are not supported.")
        # idx -> (region, sample); group by contig; gather+reconstruct per contig;
        # stitch back into batch order. Shifts/jitter mirror Haps._prepare_request.
        ...  # see Step 3c
```

- [ ] **Step 3c: Implement `HapsSvar2.__call__` (per-contig grouping + stitch)**

The batch `regions` is `(b, 3)` = `(contig_idx, start, end)`; `idx` ravels `(region, sample)`. For SVAR2, ploidy is fixed. Group the batch rows by `contig_idx`, and for each contig build the cached bundle for those region rows × the requested samples, gather, and run `SparseVar2Source(...)`'s extracted reconstruct-from-payload. Compute `shifts` exactly as `Haps._prepare_request` does (zeros when `deterministic` or `output_length` is a string; else the same `rng.integers(0, max_shift+1)` where `max_shift = diffs.clip(min=0) + (lengths-output_length).clip(min=0)`). Derive `diffs` from the gathered payload — reuse the SVAR2 kernel's own length computation by first reconstructing with `output_length=-1` (ragged) to learn hap lengths, or expose an ilen helper. **Simplest correct first cut:** support `deterministic`/ragged output (shifts=0) end-to-end, and raise `NotImplementedError` for random-jitter fixed-length until a follow-up — the parity test (Step 1) uses `output_length=-1`, deterministic. Gate non-deterministic with a clear error and a `TODO`.

Stitching: because each `(region, sample)` maps to one output row, collect per-contig `Ragged` outputs and reassemble a single `Ragged[(b, P, ~L)]` (or `(S, P, ~L)` for a single-region all-sample read) in `idx` order. Reuse `_Flat.from_offsets` as `SparseVar2Source.reconstruct` does.

> This is the crux integration. Keep the marshalling in `SparseVar2Source` (already validated) and let `HapsSvar2` own only: cache slicing, per-contig grouping, shift computation, and output stitching. Pin every step with the live-oracle parity test.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q`
Expected: PASS (cached ≡ live, byte-for-byte). If offsets match but bytes differ, the divergence is in shift handling or sample-column mapping — reconcile against the live `_query` path.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_svar2_source.py python/genvarloader/_dataset/_haps.py tests/dataset/test_svar2_dataset.py tests/dataset/conftest.py
rtk git commit -m "feat(svar2): HapsSvar2 cached-ranges + gather reconstructor"
```

---

### Task 5: `Dataset.open` dispatch to `HapsSvar2`

**Files:**
- Modify: `python/genvarloader/_dataset/_open.py` (`_build_seqs`, `:143`)
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (`_build_reconstructor` accepts `HapsSvar2`)
- Modify: `python/genvarloader/_dataset/_impl.py` (`Dataset.open` reads `svar2=` override; the `_recon` union type includes `HapsSvar2`)
- Test: `tests/dataset/test_svar2_dataset.py` (extend)

**Interfaces:**
- Consumes: `metadata.svar2_link` (Task 2), `HapsSvar2.from_path` (Task 4), `self.svar2` override on the open builder (parallel to `self.svar`, `_open.py:160`).
- Produces: `Dataset.open(path, reference=..., svar2=<override>)` returns a dataset whose `_recon` routes haplotype reads through `HapsSvar2`.

- [ ] **Step 1: Write the failing test**

```python
def test_open_routes_svar2_to_hapssvar2(svar2_store, reference_fasta, tmp_path):
    import polars as pl
    from genvarloader._dataset._haps import HapsSvar2

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, variants=Path(svar2_store))
    ds = gvl.Dataset.open(out, reference=reference_fasta).with_seqs("haplotypes")
    assert isinstance(ds._recon, HapsSvar2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -k routes -q`
Expected: FAIL — `_recon` is a plain `Haps`, or `_build_seqs` errors on missing SVAR1 metadata.

- [ ] **Step 3: Wire the dispatch**

3a. In `_open.py::_build_seqs` (`:149`), branch on `metadata.svar2_link`:

```python
        if self._has_genotypes():
            if metadata.ploidy is None:
                raise ValueError("Malformed dataset: found genotypes but not ploidy.")
            if metadata.svar2_link is not None:
                from ._haps import HapsSvar2
                if reference is None:
                    raise ValueError(
                        "SVAR2 datasets require a reference genome for haplotype output."
                    )
                return HapsSvar2.from_path(
                    path=self.path,
                    reference=reference,
                    contigs=metadata.contigs,
                    samples=metadata.samples,
                    ploidy=metadata.ploidy,
                    svar2_link=metadata.svar2_link,
                    svar2_override=getattr(self, "svar2", None),
                )
            seqs = Haps.from_path(...)  # unchanged SVAR1 path
```

`self._has_genotypes()` checks for `genotypes/` — confirm it does not require SVAR1-specific files (`svar_meta.json`); if it does, relax it to also accept `svar2_meta.json`.

3b. Add a `svar2: Path | str | None = None` field to the open-builder dataclass (parallel to `svar`, wherever `self.svar` is defined) and thread it from `Dataset.open`'s signature (`_impl.py`).

3c. In `_reconstruct.py::_build_reconstructor`, accept `HapsSvar2` for the `haplotypes` kind. The simplest wiring: treat `HapsSvar2` like `Haps` in the `seqs_kind in ("haplotypes", ...)` branch but restrict to `"haplotypes"`:

```python
    from ._haps import HapsSvar2
    if isinstance(seqs, HapsSvar2):
        if seqs_kind not in (None, "haplotypes"):
            raise NotImplementedError(
                f"SVAR2 datasets support only 'haplotypes', not {seqs_kind!r}."
            )
        active_seqs = seqs
        # dispatch: HapsSvar2 alone -> itself; with tracks -> HapsSvar2Tracks (Task 6)
```

Add `HapsSvar2` (and `HapsSvar2Tracks` from Task 6) to the `_recon` type union in `_impl.py` (`:899`) and the `match self._recon` in `__getitem__` (`:1028`).

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_open.py python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_impl.py tests/dataset/test_svar2_dataset.py
rtk git commit -m "feat(svar2): Dataset.open routes svar2 datasets to HapsSvar2"
```

---

### Task 6: Track re-alignment via the same cache (`HapsSvar2Tracks`)

**Files:**
- Modify: `python/genvarloader/_dataset/_reconstruct.py` (add `HapsSvar2Tracks`)
- Modify: `python/genvarloader/_dataset/_svar2_source.py` (reuse `realign_tracks` from the cached payload)
- Test: `tests/dataset/test_svar2_dataset.py` (extend)

**Interfaces:**
- Consumes: `HapsSvar2` (Task 4), `Tracks` (existing), `SparseVar2Source.realign_tracks` marshalling (existing), the same cached bundle + `gather_ranges` payload.
- Produces: `HapsSvar2Tracks(haps: HapsSvar2, tracks: Tracks)` implementing `Reconstructor[tuple[RaggedSeqs, _T]]`; `_build_reconstructor` returns it for `(HapsSvar2, Tracks)`.

- [ ] **Step 1: Write the failing test (cached tracks ≡ live tracks)**

```python
def test_svar2_tracks_match_live(svar2_store, reference_fasta, bigwig_track, tmp_path):
    import polars as pl
    from genoray import SparseVar2
    from genvarloader._dataset._svar2_source import SparseVar2Source

    bed = pl.DataFrame({"chrom": ["chr1"], "chromStart": [0], "chromEnd": [40]})
    out = tmp_path / "ds.gvl"
    gvl.write(path=out, bed=bed, variants=Path(svar2_store), tracks=[bigwig_track])

    ds = gvl.Dataset.open(out, reference=reference_fasta).with_seqs("haplotypes").with_tracks(...)
    _, cached_tracks = ds[0, :]

    # Live oracle: SparseVar2Source.realign_tracks with the same track buffer.
    sv = SparseVar2(svar2_store)
    live = SparseVar2Source(sv).realign_tracks("chr1", [(0, 40)], *_track_args(...))
    np.testing.assert_array_equal(
        cached_tracks.to_packed().data, live.to_packed().data
    )
```

Adapt fixtures/`_track_args` to whatever the existing `tests/test_svar2_realign_tracks.py` uses.

- [ ] **Step 2: Run to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -k tracks -q`
Expected: FAIL — `_build_reconstructor` has no `(HapsSvar2, Tracks)` case.

- [ ] **Step 3: Implement `HapsSvar2Tracks` + dispatch**

Model on `HapsTracks.__call__` (`_reconstruct.py:130`) but source haplotypes + realigned tracks from the cached `gather_ranges` payload via the extracted `SparseVar2Source.realign_tracks` body. The track buffer is read from the dataset's `intervals/` exactly as `HapsTracks` does (`self.tracks.intervals[name]`); only the haplotype-coordinate re-alignment uses the SVAR2 kernel + cached ranges instead of the SVAR1 sparse-genotype path. Route `(HapsSvar2, Tracks)` in `_build_reconstructor`:

```python
    if isinstance(active_seqs, HapsSvar2) and active_tracks is not None:
        return HapsSvar2Tracks(haps=active_seqs, tracks=active_tracks)
```

Interval (non-realigned) tracks with SVAR2: keep the existing `realign_tracks=False` guard message from `_build_reconstructor` (`:360`) — reuse verbatim.

- [ ] **Step 4: Run to verify it passes**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q`
Expected: PASS (haps + tracks parity).

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_reconstruct.py python/genvarloader/_dataset/_svar2_source.py tests/dataset/test_svar2_dataset.py
rtk git commit -m "feat(svar2): HapsSvar2Tracks realign via cached ranges"
```

---

### Task 7: End-to-end byte-identical parity + full regression

**Files:**
- Test: `tests/dataset/test_svar2_dataset.py` (extend with the M6b matrix + real chr21)
- Modify: `python/genvarloader/_dataset/_svar2_source.py` — retire `TODO(svar2-dataset-dispatch)` comment (`:7`)

**Interfaces:**
- Consumes: everything above. Produces no new production surface — hardens the contract.

- [ ] **Step 1: Add the M6b matrix parity test**

Parametrize over `{SNP, INS, DEL} × {1, 2, 4} samples × {1, 2} ploidy` (reuse the synthesis helpers from `tests/test_svar2_reconstruct.py`). For each: build a `.svar2`, `gvl.write`, open, and assert `ds[region, samples]` ≡ the live `SparseVar2Source.reconstruct` ≡ genoray `decode` (the cross-check oracle the M6b kernels already validate against). Assert offsets **and** packed bytes equal.

```python
import pytest

@pytest.mark.parametrize("variant_kind", ["snp", "ins", "del"])
@pytest.mark.parametrize("n_samples", [1, 2, 4])
@pytest.mark.parametrize("ploidy", [1, 2])
def test_svar2_cached_matches_decode_matrix(variant_kind, n_samples, ploidy, tmp_path):
    ...  # synth store -> gvl.write -> open -> compare cached vs live vs decode
```

- [ ] **Step 2: Add the real chr21 germline + somatic parity test (slow)**

Mark `@pytest.mark.slow`. Point at the real chr21 SVAR2 stores used by the E1 profiling driver (grep `docs/superpowers/specs/2026-07-03-svar2-profiling-followup.md` and the profiling driver for their paths). Compare cached-`Dataset` reconstruct vs live `SparseVar2Source` over the 3-region × all-samples workload, haps + tracks.

- [ ] **Step 3: Retire the deferred-dispatch TODO**

Remove the `TODO(svar2-dataset-dispatch)` block in `_svar2_source.py:7` (now delivered) and update the module docstring to describe the cache+gather read path.

- [ ] **Step 4: Full tree regression (SVAR1 unchanged)**

Run:
```bash
pixi run -e dev pytest tests -q
pixi run -e dev cargo-test
pixi run -e dev ruff check python/ tests/
pixi run -e dev typecheck
```
Expected: all green. If any SVAR1 test changed behavior, the additive claim is violated — investigate before proceeding.

- [ ] **Step 5: Commit**

```bash
rtk git add python/genvarloader/_dataset/_svar2_source.py tests/dataset/test_svar2_dataset.py
rtk git commit -m "test(svar2): byte-identical cached parity across M6b matrix + chr21"
```

---

### Task 8: Docs, skill, api.md, and roadmaps

**Files:**
- Modify: `skills/genvarloader/SKILL.md`, `docs/source/{api,write,format,faq}.md`, `README.md`, `docs/roadmaps/rust-migration.md`
- Modify: `docs/superpowers/specs/2026-07-03-svar2-dataset-wiring-design.md` (mark delivered) — optional

**Interfaces:**
- Consumes: the shipped feature. Produces: docs true against `main` per the repo's docs-audit + skill-maintenance gates.

- [ ] **Step 1: Document `.svar2` as a `write` variant source**

Update `docs/source/write.md` (and `SKILL.md`'s write section) to list `.svar2` directories as an accepted `variants=` source alongside VCF/PGEN/`.svar`, noting the write-time ranges cache and that `Dataset.open` accepts a `svar2=` override (parallel to `svar=`). Update `docs/source/format.md` with the `genotypes/svar2_*.npy` + `svar2_meta.json` cache layout and the `svar2_link` metadata key.

- [ ] **Step 2: Sync `api.md` with `__all__`**

If any new symbol was exported (e.g. a `svar2=` kwarg is not a new `__all__` symbol, but confirm no new public class leaked), run the sync check:

```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`. If a symbol was added to `__all__`, add its autodoc entry.

- [ ] **Step 3: Update FAQ + README + rust-migration roadmap**

`faq.md`: add "Can I use SVAR2 files with gvl?" → yes, same as `.svar`, with the search/gather cache. `README.md`: add `.svar2` to the supported variant sources if `.svar` is listed. `docs/roadmaps/rust-migration.md`: tick the SVAR2 dataset-wiring task, record the perf-verification result (Step 4), set the phase marker + PR link (per CLAUDE.md's rust-migration rule).

- [ ] **Step 4: Perf verification (same-session before/after)**

Per the spec + the `gvl-rust-perf-gate-shared-node-noise` and `gvl-profiling-perf-not-pyspy-native` memories: profile a warm SVAR2 `Dataset` read with `perf` on the Python process (paranoid=2, no `py-spy --native`) and confirm the DSO split flips from ~80% genoray `SearchTree::build` to gvl-kernel-bound, like SVAR1. Report as a **relative before/after within one allocation** (absolute wall-clock not comparable across allocations on shared Carter nodes). Record the numbers in the rust-migration roadmap.

- [ ] **Step 5: Commit + finish the branch**

```bash
rtk git add skills/ docs/ README.md
rtk git commit -m "docs(svar2): document .svar2 write source + dataset wiring"
```

Then use **superpowers:finishing-a-development-branch** to decide merge/PR. Before the PR: re-run `pixi run -e dev pytest tests -q` (full tree, not scoped) per CLAUDE.md's rename/shared-code gate.

---

## Self-Review

- **Spec coverage (Components B & C + format/parity/docs):**
  - Component B: `.svar2` detection + dispatch + `_write_from_svar2` (Task 3); `_svar2_link.py` (Task 1); `metadata["ploidy"]` set (Task 3); reject unsupported variants (Task 3, Step 3d note). ✅
  - Component C: `Dataset.open` resolve+fingerprint + `svar2=` override (Task 5); `HapsSvar2` haplotype routing retiring `TODO(svar2-dataset-dispatch)` (Tasks 4/7); tracks via same cache (Task 6); `_svar2_source.py` refactor to cache+gather (Tasks 4/6/7). ✅
  - Cache format `.gvl/genotypes/` O(offsets) memmaps + `svar2_meta.json` + `Svar2Link` (Task 3). ✅
  - Parity & testing: byte-identical cached ≡ live ≡ decode on M6b matrix + chr21; SVAR1 additive-green; perf verification (Tasks 4–8). ✅
  - Format version: additive `svar2_link` tolerated by `_check_dataset_format_version`/`_migrate` (Task 2). ✅
  - Docs/roadmaps: skill, api.md, write/format/faq/README, rust-migration (Task 8). ✅
- **Out of scope honored:** `variants`/`annotated` SVAR2 output modes raise `NotImplementedError` (Tasks 4/5); no `.svar2` on-disk format change (write only reads it).
- **Type consistency:** the cache bundle field names (`dense_range`, `region_starts`, `sample_cols`, `vk_snp_range`, `vk_indel_range`, `n_regions`/`n_samples`/`ploidy`) match exactly between `_write_from_svar2` (Task 3), `SparseVar2Source._query_cached` (Task 4), and the genoray `gather_ranges` bundle contract (genoray plan Task 4). `HapsSvar2.from_path`'s memmap names (`svar2_<field>.npy`) match the files `_write_from_svar2` writes.
- **Known risk flagged in-plan:** Task 4 Step 3c bounds the first cut to deterministic/ragged reads and defers random-jitter fixed-length SVAR2 output to a follow-up (explicit `NotImplementedError`), keeping the crux integration testable against the live oracle rather than over-reaching.
