# SVAR2 gvl Read-Bound Dataset Wiring — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the `.svar2` variant source into `gvl.write()` (a write-time 6-array offsets cache, dataset samples only) and `gvl.Dataset.__getitem__` (one all-Rust FFI call per read that gathers off the cache with **no interval-search-tree rebuild and no dense-union rebuild**), so all four output modes (haplotypes, tracks, variants, variant-windows) reconstruct in Rust, matching the SVAR1 read path.

**Architecture:** Mirror the SVAR1 write path (`_write_from_svar` → `offsets.npy` + `svar_meta.json` + `SvarLink`). At write, per contig call genoray's Python `find_ranges(samples=dataset)` (Plan 1) and stream the 6 arrays into memmaps under `genotypes/svar2_ranges/`. At read, gvl's Rust links `genoray_core` (query-only path-dep), opens a `ContigReader` per contig **once** at `Dataset.open` (an `Svar2Store` pyclass), and per read builds a `RangesBundle`-equivalent from the cached memmap slice → calls genoray's flat `gather_haps_readbound` → gets a split-dense `BatchResultSplit` → merges `var_key ⋈ dense_snp ⋈ dense_indel` and reconstructs in Rust, decoding keys inline via `svar2-codec`. The SVAR1 path is byte-unchanged.

**Tech Stack:** Python 3.10+ (pydantic, polars, numpy), Rust 2024 (PyO3 0.28 abi3-py310, `numpy` 0.28, `svar2-codec` path-dep, **new** `genoray_core` path-dep), `genoray` (local wheel), `pixi -e dev`, `maturin develop --release`.

**Depends on:** the genoray PR in `docs/superpowers/plans/2026-07-04-svar2-genoray-readbound-gather.md` — MUST be merged and the local wheel + crate built from the **same commit** first. Record that commit; the gvl path-dep and the genoray Python wheel must match it (the `RangesBundle`/`BatchResultSplit` field layout is the contract).

## Global Constraints

- **Byte-identical parity contract.** For any `contig, regions, samples` and every output mode: read-bound reconstruct ≡ the existing union-based `SparseVar2Source.reconstruct`/`realign_tracks` (which use genoray `overlap_batch`) ≡ genoray `decode` oracle ≡ SVAR1 output for an equivalent dataset — byte-for-byte / field-for-field.
- **Additive.** The SVAR1 gvl path (`_write_from_svar`, `Haps` SVAR1 branch, `reconstruct_haplotypes_fused`, etc.) stays byte-unchanged; full SVAR1 regression green (`pixi run -e dev pytest tests -q` + `cargo test`). The existing standalone `SparseVar2Source` (union path) stays as the parity oracle until Task 8 retires only its live dispatch.
- **Write caches only the dataset's samples `S'`.** `gvl.write` already selects the sample set; the cache is sized to `S'`, not the full `.svar2` cohort — exactly like `_write_from_svar`.
- **Rebuild Rust before Python tests.** After any `src/` edit: `pixi run -e dev maturin develop --release` BEFORE `pixi run -e dev pytest` — otherwise pytest imports the stale `.so`. `cargo test` compiles from source and is unaffected.
- **Full-tree before pushing shared-code changes.** `_write.py`, `_haps.py`, `_open.py`, `_reconstruct.py`, `_impl.py` are shared; run `pixi run -e dev pytest tests -q` (dataset **and** unit), not a scoped subset, before pushing.
- **Lint gate.** `pixi run -e dev ruff check python/ tests/` + `pixi run -e dev ruff format python/ tests/` + `pixi run -e dev typecheck`; `cargo fmt` + `cargo clippy --all-targets` for Rust.
- **Docs/skill gates.** `.svar2` becomes a public `write` variant source ⇒ update `skills/genvarloader/SKILL.md`, `docs/source/{api.md,write.md,format.md,faq.md}`, `README.md`; keep `api.md` in sync with any new `__all__` symbol (Task 9).
- **Reject unsupported variants** (symbolic/breakend) exactly as SVAR1 does (`_reject_unsupported_variants`).
- **genoray repo path is absolute:** `/carter/users/dlaub/projects/genoray` (there is no `../genoray`). gvl already path-deps `svar2-codec = { path = "/carter/users/dlaub/projects/genoray/svar2-codec" }`.

---

## File Structure

- `python/genvarloader/_dataset/_svar2_link.py` — **new**; `Svar2Link`/`Svar2Fingerprint`/`_resolve_svar2`/`_verify_svar2_fingerprint`. Models `_svar_link.py`. *(Task 1)*
- `python/genvarloader/_dataset/_write.py` — add `.svar2` coercion arm (`:~225`), `SparseVar2` dispatch arm (`:~325`), `_write_from_svar2`, `Metadata.svar2_link` field (`:86-98`). *(Tasks 1, 2)*
- `Cargo.toml` — add `genoray_core = { path = "/carter/users/dlaub/projects/genoray", default-features = false }`. *(Task 3)*
- `src/svar2/store.rs` — **new**; `Svar2Store` pyclass wrapping `HashMap<String, genoray_core::query::ContigReader>`. *(Task 3)*
- `src/svar2/mod.rs` — add `merge_hap3` (3-source merge). *(Task 4)*
- `src/ffi/mod.rs` — `reconstruct_haplotypes_from_svar2_readbound`, `shift_and_realign_tracks_from_svar2_readbound`, `decode_variants_from_svar2_readbound`. *(Tasks 4, 5, 6)*
- `src/reconstruct/mod.rs` — internal `reconstruct_haplotypes_from_split` (consumes `BatchResultSplit`). *(Task 4)*
- `src/tracks/mod.rs` — internal `shift_and_realign_tracks_from_split`. *(Task 5)*
- `src/lib.rs` — register the new pyclass + pyfunctions. *(Tasks 3–6)*
- `python/genvarloader/_dataset/_haps.py` — `Haps` source discriminant (`svar` vs `svar2`); route `_reconstruct_haplotypes` / variants; open `Svar2Store`. *(Tasks 4, 6, 7)*
- `python/genvarloader/_dataset/_open.py` — thread `svar2_link`/`svar2` override through `_build_seqs`. *(Task 7)*
- `python/genvarloader/_dataset/_impl.py` — `Dataset.open(svar2=...)` override param. *(Task 7)*
- `python/genvarloader/_dataset/_reconstruct.py` — `HapsTracks` routes to the svar2 track kernel when source is svar2. *(Task 5)*
- `python/genvarloader/_dataset/_svar2_source.py` — retire live `overlap_batch` dispatch; keep as parity oracle only. *(Task 8)*

---

## Task 1: `_svar2_link.py` + `Metadata.svar2_link` field

**Files:**
- Create: `python/genvarloader/_dataset/_svar2_link.py`
- Modify: `python/genvarloader/_dataset/_write.py:86-98` (`Metadata`)
- Test: `tests/unit/dataset/test_svar2_link.py` (new)

**Interfaces:**
- Produces:
  - `class Svar2Fingerprint(BaseModel)`: `n_variants: int`, `store_bytes: int`.
  - `class Svar2Link(BaseModel)`: `relative_path: str`, `absolute_path: str`, `fingerprint: Svar2Fingerprint`.
  - `def _resolve_svar2(gvl_path: Path, link: Svar2Link | None, override: Path | str | None) -> Path` — override → link.relative → link.absolute → sibling `*.svar2`.
  - `def _verify_svar2_fingerprint(svar2_path: Path, link: Svar2Link | None) -> None` — no-op if `link is None`; else compare `n_variants` (from the `.svar2` index) + a canonical store-file byte count; raise `ValueError` on mismatch.
  - `Metadata` gains `svar2_link: Svar2Link | None = None`.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/dataset/test_svar2_link.py`:

```python
from pathlib import Path
import pytest
from genvarloader._dataset._svar2_link import (
    Svar2Link, Svar2Fingerprint, _resolve_svar2, _verify_svar2_fingerprint,
)


def test_resolve_prefers_override(tmp_path: Path):
    real = tmp_path / "cohort.svar2"
    real.mkdir()
    link = Svar2Link(relative_path="nope.svar2", absolute_path="/nope.svar2",
                     fingerprint=Svar2Fingerprint(n_variants=1, store_bytes=1))
    assert _resolve_svar2(tmp_path, link, real) == real


def test_resolve_missing_override_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        _resolve_svar2(tmp_path, None, tmp_path / "absent.svar2")


def test_verify_none_link_is_noop(tmp_path: Path):
    _verify_svar2_fingerprint(tmp_path, None)  # must not raise
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_link.py -q`
Expected: FAIL — `ModuleNotFoundError: _svar2_link`.

- [ ] **Step 3: Write `_svar2_link.py`**

Create the file. Model it on `_svar_link.py` (`_resolve_svar`/`_verify_fingerprint`), changing the fingerprint to the `.svar2` store's stable identity. The `.svar2` index is at `<store>/index.arrow` (confirm the exact filename by inspecting a real `.svar2`; genoray's `SparseVar2` exposes `.index` — the n_variants source); the canonical byte count is the summed size of the on-disk dense/var_key store files:

```python
"""Resolution and integrity for the GVL dataset → .svar2 back-reference.

Mirrors _svar_link.py; the fingerprint keys on the .svar2 store's stable identity
(variant count + a canonical store-file byte count) rather than SVAR1's
variant_idxs.npy, which .svar2 does not have.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel


class Svar2Fingerprint(BaseModel):
    n_variants: int
    store_bytes: int


class Svar2Link(BaseModel):
    relative_path: str
    absolute_path: str
    fingerprint: Svar2Fingerprint


def _svar2_store_bytes(svar2_path: Path) -> int:
    """Canonical, stable byte count of the .svar2 on-disk stores. Sum the sizes of
    the packed dense + var_key key files across contigs, sorted for determinism."""
    total = 0
    for p in sorted(svar2_path.rglob("*")):
        if p.is_file() and p.suffix in {".bin", ".npy"} and "keys" in p.name:
            total += p.stat().st_size
    return total


def _svar2_n_variants(svar2_path: Path) -> int:
    import polars as pl
    # .svar2 index; confirm filename against a real store (SparseVar2().index).
    return pl.scan_ipc(svar2_path / "index.arrow").select(pl.len()).collect().item()


def _resolve_svar2(
    gvl_path: Path, link: "Svar2Link | None", override: "Path | str | None"
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


def _verify_svar2_fingerprint(svar2_path: Path, link: "Svar2Link | None") -> None:
    if link is None:
        return
    n_obs = _svar2_n_variants(svar2_path)
    bytes_obs = _svar2_store_bytes(svar2_path)
    exp = link.fingerprint
    mismatches: list[str] = []
    if n_obs != exp.n_variants:
        mismatches.append(f"n_variants: expected {exp.n_variants}, observed {n_obs}")
    if bytes_obs != exp.store_bytes:
        mismatches.append(f"store_bytes: expected {exp.store_bytes}, observed {bytes_obs}")
    if mismatches:
        raise ValueError(
            f"svar2 fingerprint mismatch at {svar2_path}: " + "; ".join(mismatches)
        )


def make_svar2_link(gvl_path: Path, svar2_path: Path) -> Svar2Link:
    svar2_resolved = svar2_path.resolve()
    return Svar2Link(
        relative_path=os.path.relpath(svar2_resolved, start=gvl_path).replace(os.sep, "/"),
        absolute_path=str(svar2_resolved),
        fingerprint=Svar2Fingerprint(
            n_variants=_svar2_n_variants(svar2_resolved),
            store_bytes=_svar2_store_bytes(svar2_resolved),
        ),
    )
```

> **Confirm before finalizing:** open a real `.svar2` (the MVP fixture) and verify (a) the index filename used by `_svar2_n_variants` and (b) that `_svar2_store_bytes`'s glob captures ≥1 stable file. Adjust the patterns to the actual layout; the contract is only that the count is deterministic and changes iff the store changes.

- [ ] **Step 4: Add the `Metadata` field**

In `python/genvarloader/_dataset/_write.py`, add the import near the `SvarLink` import (`:40`):

```python
from ._svar2_link import Svar2Link
```

and add to `Metadata` (`:86-98`), after `svar_link`:

```python
    svar2_link: Svar2Link | None = None
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pixi run -e dev pytest tests/unit/dataset/test_svar2_link.py -q`
Expected: PASS.
Run: `python -c "from genvarloader._dataset._write import Metadata; print(Metadata.model_fields['svar2_link'])"`
Expected: prints the optional field (default None) — confirms backward/forward compat (no format bump needed; `_check_dataset_format_version` gates only on major, and old datasets fill the default).

- [ ] **Step 6: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_svar2_link.py python/genvarloader/_dataset/_write.py tests/unit/dataset/test_svar2_link.py
git commit -m "feat(dataset): Svar2Link resolution/fingerprint + Metadata.svar2_link

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 2: `_write_from_svar2` + write dispatch (the 6-array cache)

**Files:**
- Modify: `python/genvarloader/_dataset/_write.py` — coercion arm (`:216-234`), dispatch arm (`:285-333`), add `_write_from_svar2`.
- Test: `tests/dataset/test_write_svar2.py` (new).

**Interfaces:**
- Consumes: genoray `SparseVar2` with `.ploidy`, `.index`, and a per-contig batched `find_ranges(contig, starts, ends, samples=...) -> dict` returning the 6 arrays: `vk_snp_range (R,S',P,2)`, `vk_indel_range (R,S',P,2)`, `dense_snp_range (R,2)`, `dense_indel_range (R,2)`, `region_starts (R,)`, `sample_cols (S',)` (Plan 1 Task 5 added `dense_snp_range`/`dense_indel_range` to the dict).
- Produces: `def _write_from_svar2(path: Path, bed: pl.DataFrame, svar2: "SparseVar2", samples: list[str], extend_to_length: bool) -> tuple[pl.DataFrame, Svar2Link]`. Writes memmaps under `path/genotypes/svar2_ranges/` + `svar2_meta.json`; returns the extended bed (max_ends folded in) and the `Svar2Link`.

> **genoray dependency check (do first):** confirm `SparseVar2.find_ranges(contig, starts, ends, samples=...)` exists in the local genoray wheel and returns the 6-key dict. If genoray only exposes `find_ranges` on `PyContigReader` (per-contig, no batched Python entry on `SparseVar2`), add a thin `SparseVar2.find_ranges` wrapper in genoray that dispatches to the contig's `PyContigReader.find_ranges` — a small genoray-side addition; fold it into the genoray PR (Plan 1 Task 5) rather than marshalling in gvl.

- [ ] **Step 1: Write the failing test**

Create `tests/dataset/test_write_svar2.py`. Reuse the `.svar2` fixture the current `SparseVar2Source` tests use (search `tests/` for an existing `.svar2` path / `SparseVar2(` construction; if none, build one from the MVP `svar2_mvp` store or a synthetic genoray fixture). Skeleton:

```python
import json
from pathlib import Path
import numpy as np
import polars as pl
import pytest

import genvarloader as gvl
from genvarloader._dataset._svar2_link import Svar2Link

SVAR2_FIXTURE = ...  # Path to a small .svar2 store (reuse existing test fixture)


def test_write_svar2_emits_cache(tmp_path: Path):
    from genoray import SparseVar2
    svar2 = SparseVar2(SVAR2_FIXTURE)
    bed = pl.DataFrame({
        "chrom": ["chr1", "chr1"],
        "chromStart": [0, 250],
        "chromEnd": [1000, 400],
    })
    out = tmp_path / "ds.gvl"
    gvl.write(out, bed, variants=svar2, samples=None, overwrite=True)

    rd = out / "genotypes" / "svar2_ranges"
    meta = json.loads((rd / "svar2_meta.json").read_text())
    assert set(meta) >= {
        "vk_snp_range", "vk_indel_range", "dense_snp_range",
        "dense_indel_range", "region_starts", "sample_cols",
    }
    # metadata.json carries the link + ploidy.
    md = json.loads((out / "metadata.json").read_text())
    assert md["svar2_link"] is not None
    assert md["ploidy"] == svar2.ploidy
    Svar2Link.model_validate(md["svar2_link"])  # shape check
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: FAIL — `gvl.write` raises "unrecognized file extension" / doesn't dispatch `SparseVar2`.

- [ ] **Step 3: Add the `.svar2` coercion arm**

In `_write.py`, add the import near the `SparseVar` import (`:19`): change `from genoray import PGEN, VCF, Reader, SparseVar` to also import `SparseVar2`:

```python
from genoray import PGEN, VCF, Reader, SparseVar, SparseVar2
```

In the coercion block (`:216-234`), add an arm before the `else` that raises:

```python
                    elif variants.is_dir() and variants.suffix == ".svar2":
                        variants = SparseVar2(variants)
```

- [ ] **Step 4: Add the dispatch arm**

In the genotype-writing dispatch (`:325-330`), after the `SparseVar` branch, add:

```python
                elif isinstance(variants, SparseVar2):
                    gvl_bed, _svar2_link = _write_from_svar2(
                        path, gvl_bed, variants, samples, extend_to_length
                    )
                    metadata["svar2_link"] = _svar2_link
```

(`metadata["ploidy"] = variants.ploidy` at `:330` already runs for all sources, including `SparseVar2`.)

- [ ] **Step 5: Write `_write_from_svar2`**

Add the function near `_write_from_svar` (`:961`). It streams the 6 arrays per contig into memmaps and computes `max_ends` for the bed (mirroring SVAR1's end-extension). Because the read-bound gather needs per-`(region, selected-sample, ploid)` var_key ranges and per-region dense ranges, the memmaps are shaped `(R, S', P, 2)` / `(R, 2)` / `(R,)` / `(S',)`:

```python
def _write_from_svar2(
    path: Path,
    bed: pl.DataFrame,
    svar2: "SparseVar2",
    samples: list[str],
    extend_to_length: bool,
) -> "tuple[pl.DataFrame, Svar2Link]":
    _reject_unsupported_variants(svar2.index, "SVAR2")

    out_dir = path / "genotypes" / "svar2_ranges"
    out_dir.mkdir(parents=True, exist_ok=True)

    R, S, P = bed.height, len(samples), svar2.ploidy
    vk_snp = np.memmap(out_dir / "vk_snp_range.npy", np.int64, "w+", shape=(R, S, P, 2))
    vk_indel = np.memmap(out_dir / "vk_indel_range.npy", np.int64, "w+", shape=(R, S, P, 2))
    dense_snp = np.memmap(out_dir / "dense_snp_range.npy", np.int64, "w+", shape=(R, 2))
    dense_indel = np.memmap(out_dir / "dense_indel_range.npy", np.int64, "w+", shape=(R, 2))
    region_starts = np.memmap(out_dir / "region_starts.npy", np.int64, "w+", shape=(R,))
    # sample_cols: selected slot -> original sample index (same for every contig).
    sample_cols_full = np.asarray(
        [svar2.available_samples.index(s) for s in samples], np.int64
    )
    np.save(out_dir / "sample_cols.npy", sample_cols_full)

    with open(out_dir / "svar2_meta.json", "w") as f:
        json.dump(
            {
                "vk_snp_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "vk_indel_range": {"shape": [R, S, P, 2], "dtype": "<i8"},
                "dense_snp_range": {"shape": [R, 2], "dtype": "<i8"},
                "dense_indel_range": {"shape": [R, 2], "dtype": "<i8"},
                "region_starts": {"shape": [R], "dtype": "<i8"},
                "sample_cols": {"shape": [S], "dtype": "<i8"},
                "ploidy": P,
            },
            f,
        )

    v_ends = svar2.index.select(
        end=pl.col("POS") - pl.col("ILEN").list.first().clip(upper_bound=0)
    )["end"].to_numpy()
    max_ends = np.empty(R, np.int32)
    contig_offset = 0
    pbar = tqdm(total=R, unit=" region")
    for (c,), df in bed.partition_by("chrom", as_dict=True, maintain_order=True).items():
        c = cast(str, c)
        pbar.set_description(f"Processing svar2 ranges for {df.height} regions on {c}")
        lo, hi = contig_offset, contig_offset + df.height
        d = svar2.find_ranges(
            c,
            df["chromStart"].to_numpy(),
            df["chromEnd"].to_numpy() if not extend_to_length else df["chromEnd"].to_numpy(),
            samples=samples,
        )
        rc = df.height
        # find_ranges returns row-major (R, S', P) for vk ranges; reshape into (R,S,P,2).
        vk_snp[lo:hi] = np.asarray(d["vk_snp_range"], np.int64).reshape(rc, S, P, 2)
        vk_indel[lo:hi] = np.asarray(d["vk_indel_range"], np.int64).reshape(rc, S, P, 2)
        dense_snp[lo:hi] = np.asarray(d["dense_snp_range"], np.int64).reshape(rc, 2)
        dense_indel[lo:hi] = np.asarray(d["dense_indel_range"], np.int64).reshape(rc, 2)
        region_starts[lo:hi] = np.asarray(d["region_starts"], np.int64).reshape(rc)

        # max_ends: extend each region to cover the farthest variant end it overlaps.
        # Reuse the svar2 max-end helper if genoray exposes one; else derive from the
        # widest dense/indel + vk window like _write_from_svar does from v_idxs.
        max_ends[lo:hi] = svar2._region_max_ends(
            c, df["chromStart"].to_numpy(), df["chromEnd"].to_numpy(), samples=samples
        )  # SEE NOTE
        contig_offset += df.height
        pbar.update(df.height)
    pbar.close()
    for mm in (vk_snp, vk_indel, dense_snp, dense_indel, region_starts):
        mm.flush()

    from ._svar2_link import make_svar2_link
    svar2_link = make_svar2_link(path, svar2.path)
    return bed.with_columns(
        chromEnd=pl.max_horizontal(pl.Series(max_ends), pl.col("chromEnd"))
    ), svar2_link
```

> **`max_ends` NOTE.** SVAR1 derives `max_ends` from the per-region max `v_idx` and `v_ends[v_idx]`. For SVAR2 the equivalent is the farthest `v_end` among the region's overlapping variants (var_key + dense) for the dataset's samples. If genoray exposes a helper (`_region_max_ends` / a field on the `find_ranges` dict), use it. If not, add `max_end_range` per region to genoray's `find_ranges` dict (small genoray addition, fold into Plan 1 Task 5) rather than recomputing in Python. Confirm the exact source before implementing; do **not** guess a formula — measure it against `_write_from_svar`'s `max_ends` on a shared fixture (Step 7).

- [ ] **Step 6: Rebuild not needed (Python-only) — run the write test**

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: PASS (cache + meta + link written).

- [ ] **Step 7: Add a max_ends parity assertion**

Add to the test file: write the *same* regions from an equivalent `.svar` (SVAR1) and `.svar2` of the same cohort (the MVP fixtures are matched); assert the two extended `chromEnd` columns are equal. This pins the end-extension semantics.

Run: `pixi run -e dev pytest tests/dataset/test_write_svar2.py -q`
Expected: PASS.

- [ ] **Step 8: Lint + commit**

```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_write.py tests/dataset/test_write_svar2.py
git commit -m "feat(write): _write_from_svar2 6-array ranges cache + dispatch

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 3: gvl links `genoray_core`; `Svar2Store` pyclass

**Files:**
- Modify: `Cargo.toml` (`[dependencies]`)
- Create: `src/svar2/store.rs`
- Modify: `src/svar2/mod.rs` (`pub mod store;`), `src/lib.rs` (register pyclass)
- Test: `tests/unit/dataset/test_svar2_store.py` (new)

**Interfaces:**
- Produces: `Svar2Store` pyclass. Python: `Svar2Store(store_path: str, contigs: list[str], n_samples: int, ploidy: int)` opens one `genoray_core::query::ContigReader` per contig, held for the store's lifetime (the SVAR2 analog of SVAR1's once-built `_HapsFfiStatic`). Rust-internal: `fn reader(&self, contig: &str) -> &ContigReader`.

- [ ] **Step 1: Add the `genoray_core` path-dep**

In `Cargo.toml` `[dependencies]`, after the `svar2-codec` line, add:

```toml
genoray_core = { path = "/carter/users/dlaub/projects/genoray", default-features = false }
```

`default-features = false` drops `conversion` (htslib) and `extension-module`, yielding the query-only core (Plan 1 Task 1). Confirm the genoray checkout is at the commit recorded in Plan 1 Task 6.

- [ ] **Step 2: Write the failing test**

Create `tests/unit/dataset/test_svar2_store.py`:

```python
import pytest
from genvarloader.genvarloader import Svar2Store  # compiled ext

SVAR2_FIXTURE = ...  # same fixture as Task 2


def test_store_opens_contigs():
    store = Svar2Store(str(SVAR2_FIXTURE), ["chr1"], n_samples=2, ploidy=2)
    assert store.contigs() == ["chr1"]
```

- [ ] **Step 3: Write `src/svar2/store.rs`**

```rust
use std::collections::HashMap;

use genoray_core::query::ContigReader;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;

/// Opened once at Dataset.open; holds one query-only ContigReader per contig for
/// the store's lifetime (SVAR2 analog of SVAR1's cached _HapsFfiStatic).
#[pyclass]
pub struct Svar2Store {
    readers: HashMap<String, ContigReader>,
}

impl Svar2Store {
    pub fn reader(&self, contig: &str) -> Option<&ContigReader> {
        self.readers.get(contig)
    }
}

#[pymethods]
impl Svar2Store {
    #[new]
    fn new(store_path: &str, contigs: Vec<String>, n_samples: usize, ploidy: usize) -> PyResult<Self> {
        let mut readers = HashMap::with_capacity(contigs.len());
        for c in contigs {
            let r = ContigReader::open(store_path, &c, n_samples, ploidy)
                .map_err(|e| PyIOError::new_err(format!("open contig {c}: {e}")))?;
            readers.insert(c, r);
        }
        Ok(Self { readers })
    }

    fn contigs(&self) -> Vec<String> {
        let mut v: Vec<String> = self.readers.keys().cloned().collect();
        v.sort();
        v
    }
}
```

Add `pub mod store;` to `src/svar2/mod.rs`, and in `src/lib.rs`'s `#[pymodule]` add `m.add_class::<svar2::store::Svar2Store>()?;`.

- [ ] **Step 4: Build + run**

Run: `cargo build 2>&1 | tail -20` (first build compiles genoray_core query-only; may take a bit).
Then: `pixi run -e dev maturin develop --release 2>&1 | tail -20`
Then: `pixi run -e dev pytest tests/unit/dataset/test_svar2_store.py -q`
Expected: PASS.

- [ ] **Step 5: fmt + clippy + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
git add Cargo.toml Cargo.lock src/svar2/store.rs src/svar2/mod.rs src/lib.rs tests/unit/dataset/test_svar2_store.py
git commit -m "feat(rust): link genoray_core (query-only) + Svar2Store pyclass

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 4: Read-bound haplotype kernel (all-Rust, one FFI call)

**Files:**
- Modify: `src/svar2/mod.rs` — add `merge_hap3`.
- Modify: `src/reconstruct/mod.rs` — add `reconstruct_haplotypes_from_split`.
- Modify: `src/ffi/mod.rs` — add `reconstruct_haplotypes_from_svar2_readbound`.
- Modify: `src/lib.rs` — register it.
- Test: `tests/dataset/test_svar2_readbound_haps.py` (parity vs union oracle), `src/svar2/mod.rs` `#[cfg(test)]` (merge_hap3 unit).

**Interfaces:**
- Consumes: `genoray_core::query::{gather_haps_readbound, BatchResultSplit, KeyRef}`; `ContigReader::lut_arrays()` (in-Rust LUT, no Python marshalling); existing `svar2::decode_alt`, `reconstruct::reconstruct_haplotype_core`.
- Produces the FFI pyfunction:
  ```
  reconstruct_haplotypes_from_svar2_readbound(
      store: &Svar2Store, contig: &str,
      region_starts: (n_q,) u32, orig_samples: (n_q,) i64,
      vk_snp_range: (n_q*P, 2) i64, vk_indel_range: (n_q*P, 2) i64,
      dense_snp_range: (n_q, 2) i64, dense_indel_range: (n_q, 2) i64,
      region_bounds: (n_q, 2) i32,          # [start, end) per query, post-jitter
      shifts: (n_q, P) i32,
      ref_: (n_ref,) u8, ref_offsets: (n_contig+1,) i64,
      pad_char: u8, output_length: i64, parallel: bool,
  ) -> (out_data u8, out_offsets i64)
  ```
  `n_q` = number of `(region, sample)` query rows; ploidy `P` inferred from `shifts.shape[1]`.

- [ ] **Step 1: Write the `merge_hap3` unit test**

In `src/svar2/mod.rs` `#[cfg(test)]`, add:

```rust
#[test]
fn merge_hap3_is_position_sorted_stable() {
    // vk at pos 5,20; dense_snp at 10; dense_indel at 10,30.
    let vk = [(5u32, 100u32), (20, 200)];
    let dsnp = [(10u32, 300u32)];
    let dindel = [(10u32, 400u32), (30, 500)];
    let out = merge_hap3(&vk, &dsnp, &dindel);
    let positions: Vec<u32> = out.iter().map(|&(p, _)| p).collect();
    assert_eq!(positions, vec![5, 10, 10, 20, 30]);
    // On the pos-10 tie: vk-source first (none here), then dense_snp before dense_indel.
    assert_eq!(out[1], (10, 300));
    assert_eq!(out[2], (10, 400));
}
```

- [ ] **Step 2: Run it (fails to compile)**

Run: `cargo test svar2::mod 2>&1 | tail -20` (or `cargo test merge_hap3`)
Expected: FAIL — `merge_hap3` undefined.

- [ ] **Step 3: Write `merge_hap3`**

In `src/svar2/mod.rs`, add. It mirrors the existing 2-source `merge_hap` (`:30-51`) but takes three already-position-ordered inputs and stable-sorts by position (vk pushed first, then dense_snp, then dense_indel — matching genoray's `merge_keys(vec![vk, dense_snp, dense_indel])` tie order):

```rust
/// Merge one hap's var_key ⋈ dense_snp ⋈ dense_indel into one position-sorted
/// (pos, key) list. Stable: on a shared position, var_key precedes dense_snp
/// precedes dense_indel — the order genoray's decode oracle uses.
pub fn merge_hap3(
    vk: &[(u32, u32)],
    dense_snp: &[(u32, u32)],
    dense_indel: &[(u32, u32)],
) -> Vec<(u32, u32)> {
    let mut a: Vec<(u32, u32)> = Vec::with_capacity(vk.len() + dense_snp.len() + dense_indel.len());
    a.extend_from_slice(vk);
    a.extend_from_slice(dense_snp);
    a.extend_from_slice(dense_indel);
    a.sort_by_key(|&(p, _)| p); // stable
    a
}
```

- [ ] **Step 4: Run the unit test**

Run: `cargo test merge_hap3 2>&1 | tail -20`
Expected: PASS.

- [ ] **Step 5: Write the internal `reconstruct_haplotypes_from_split`**

In `src/reconstruct/mod.rs`, add a function consuming a `BatchResultSplit`. It mirrors the existing `reconstruct_haplotypes_from_svar2` (`:611`) but sources per-hap merged keys from the split result via `merge_hap3` (extracting present dense entries per hap from the presence bitmasks), and decodes via `svar2::decode_alt` with LUT bytes from the reader. Signature:

```rust
use genoray_core::query::BatchResultSplit;

#[allow(clippy::too_many_arguments)]
pub fn reconstruct_haplotypes_from_split(
    mut out: ArrayViewMut1<u8>,
    out_offsets: ArrayView1<i64>,
    region_bounds: ArrayView2<i32>,   // (n_q, 2)
    shifts: ArrayView2<i32>,          // (n_q, P)
    br: &BatchResultSplit,
    lut_bytes: &[u8],
    lut_off: &[i64],
    ref_: ArrayView1<u8>,
    ref_offsets: ArrayView1<i64>,
    pad_char: u8,
    parallel: bool,
) {
    // For each query q and ploid p (hap h = q*P + p):
    //   vk_h  = br.vk[br.vk_off[h]..br.vk_off[h+1]]        -> (pos, key)
    //   dsnp  = present entries of br.dense_snp[br.dense_snp_range[q]]   via br.dense_snp_present
    //   dind  = present entries of br.dense_indel[br.dense_indel_range[q]] via br.dense_indel_present
    //   merged = svar2::merge_hap3(&vk_h, &dsnp, &dind)
    //   provide(i) decodes merged[i].key via svar2::decode_alt(key, lut_bytes, lut_off),
    //     patching an empty pure-DEL alt with the anchor ref[pos] byte (as the existing
    //     reconstruct_haplotypes_from_svar2 does at :690-710).
    //   reconstruct_haplotype_core(merged.len(), provide, shift, contig_ref, ref_start,
    //     out_view, pad_char, None, None, None)
    // Parallel path: split_at_mut chain over out, one disjoint chunk per hap (mirror
    // reconstruct_haplotypes_from_svar2's parallel arm at :740-879).
    // ...
}
```

Extract "present dense entries for hap h in query q" with a small helper:

```rust
fn present_dense(
    dense: &[genoray_core::query::KeyRef],
    range: (usize, usize),
    present: &[u8],
    bit0: usize,
) -> Vec<(u32, u32)> {
    let (s, e) = range;
    let mut out = Vec::new();
    for (k, j) in (s..e).enumerate() {
        if genoray_core::bits_get_bit(present, bit0 + k) {
            out.push((dense[j].position, dense[j].key));
        }
    }
    out
}
```

(`genoray_core::bits_get_bit` is the shim added in Plan 1 Task 4; `KeyRef` fields `position`/`key` are `pub`.)

> **Sizing pass.** Before reconstruct, size outputs exactly like `reconstruct_haplotypes_from_svar2`: compute per-hap applied-ilen diffs (a `hap_diffs_split` analog of `svar2::hap_diffs_svar2` operating over the merged keys), prefix-sum to `out_offsets`, `uninit_output`. Reuse `svar2::hap_diffs_svar2`'s clipping logic; it only needs `(pos, ilen)` per merged key, which `decode_alt` yields as `v_diff`.

- [ ] **Step 6: Write the FFI `reconstruct_haplotypes_from_svar2_readbound`**

In `src/ffi/mod.rs`, mirror the fused SVAR1 entry (`:618`) / the existing `reconstruct_haplotypes_from_svar2` (`:768`). Body: look up `reader = store.reader(contig)`; convert the `(n,2)` range arrays to `Vec<(usize,usize)>` and `orig_samples` to `Vec<usize>`; call `genoray_core::query::gather_haps_readbound(reader, &region_starts, &orig_samples, &vk_snp_range, &vk_indel_range, &dense_snp_range, &dense_indel_range, ploidy)`; get `(lut_bytes, lut_off)` from `reader.lut_arrays()`; run the sizing pass; allocate; call `reconstruct_haplotypes_from_split`; return `(out_data, out_offsets)`. Register in `src/lib.rs`.

- [ ] **Step 7: Write the parity test (vs the union oracle)**

Create `tests/dataset/test_svar2_readbound_haps.py`. The oracle is the existing `SparseVar2Source.reconstruct` (genoray `overlap_batch`, union path) — the spec's byte-identical contract:

```python
import numpy as np
from genvarloader._dataset._svar2_source import SparseVar2Source
from genvarloader._dataset._svar2_store_py import build_readbound_haps  # thin py wrapper (Task 7)

SVAR2_FIXTURE = ...  # same fixture


def test_readbound_haps_match_union_oracle():
    from genoray import SparseVar2
    svar2 = SparseVar2(SVAR2_FIXTURE)
    contig = "chr1"
    regions = [(0, 1000), (250, 400), (150, 250)]
    ref, ref_offsets, pad = _load_contig_ref(contig)  # helper: contig bytes

    union = SparseVar2Source(svar2).reconstruct(
        contig, regions, ref, ref_offsets, pad, shifts=None, output_length=-1
    )
    readbound = build_readbound_haps(  # opens Svar2Store, slices no cache (direct find_ranges),
        svar2, contig, regions, ref, ref_offsets, pad, shifts=None, output_length=-1
    )
    # Ragged equality: same offsets + same bytes.
    assert np.array_equal(np.asarray(union.offsets), np.asarray(readbound.offsets))
    assert np.array_equal(union.data.view("u1"), readbound.data.view("u1"))
```

> `build_readbound_haps` is a thin Python test helper that, for the given regions, computes the flat per-query `(region_starts, orig_samples, vk/dense ranges)` via `svar2.find_ranges` (full cohort, R×S' flattened to n_q = R*S rows in region-major sample order), opens an `Svar2Store`, and calls `reconstruct_haplotypes_from_svar2_readbound`. It exercises the exact FFI the dataset read path uses (Task 7), decoupled from cache I/O. Put it in a small `python/genvarloader/_dataset/_svar2_store_py.py` alongside the pyclass usage.

- [ ] **Step 8: Rebuild + run parity**

Run: `pixi run -e dev maturin develop --release 2>&1 | tail -20`
Run: `pixi run -e dev pytest tests/dataset/test_svar2_readbound_haps.py -q`
Expected: PASS (byte-identical to the union oracle).

- [ ] **Step 9: cargo test + fmt + clippy + commit**

Run: `cargo test 2>&1 | tail -30` (merge_hap3 + any Rust svar2 tests green).

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
git add src/svar2/mod.rs src/reconstruct/mod.rs src/ffi/mod.rs src/lib.rs python/genvarloader/_dataset/_svar2_store_py.py tests/dataset/test_svar2_readbound_haps.py
git commit -m "feat(rust): read-bound svar2 haplotype kernel (gather_haps_readbound + merge_hap3)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 5: Read-bound tracks kernel

**Files:**
- Modify: `src/tracks/mod.rs` (add `shift_and_realign_tracks_from_split`), `src/ffi/mod.rs` (add `shift_and_realign_tracks_from_svar2_readbound`), `src/lib.rs`.
- Modify: `python/genvarloader/_dataset/_svar2_store_py.py` (a `build_readbound_tracks` helper).
- Test: `tests/dataset/test_svar2_readbound_tracks.py`.

**Interfaces:**
- Produces `shift_and_realign_tracks_from_svar2_readbound(store, contig, <same 8 range/query args as Task 6 FFI>, tracks, track_offsets, params, strategy_id, base_seed, parallel) -> (out_data f32, out_offsets i64)`. Tracks need only `ilen`/`deletion_len` per merged key (no allele bytes), so decoding is cheaper — but reuse the **same** `gather_haps_readbound` + `merge_hap3` merge so the two modes read identical variant sets.

- [ ] **Step 1: Write the failing parity test**

Create `tests/dataset/test_svar2_readbound_tracks.py`, oracle = `SparseVar2Source(svar2).realign_tracks(...)`:

```python
def test_readbound_tracks_match_union_oracle():
    from genoray import SparseVar2
    svar2 = SparseVar2(SVAR2_FIXTURE)
    contig, regions = "chr1", [(0, 1000), (250, 400)]
    tracks, toff, params, strat, seed = _synthetic_track_inputs(regions)
    union = SparseVar2Source(svar2).realign_tracks(
        contig, regions, tracks, toff, params, strat, seed, shifts=None)
    rb = build_readbound_tracks(
        svar2, contig, regions, tracks, toff, params, strat, seed, shifts=None)
    import numpy as np
    assert np.array_equal(np.asarray(union.offsets), np.asarray(rb.offsets))
    assert np.allclose(union.data, rb.data, equal_nan=True)
```

- [ ] **Step 2: Run to confirm it fails** — `pixi run -e dev pytest tests/dataset/test_svar2_readbound_tracks.py -q` → FAIL (`build_readbound_tracks`/FFI missing).

- [ ] **Step 3: Implement the Rust track-from-split kernel + FFI**, mirroring `shift_and_realign_tracks_from_svar2` (`src/tracks/mod.rs:698`, `src/ffi/mod.rs:897`) but sourcing merged keys from `BatchResultSplit` via `merge_hap3` (ilen only). Register in `src/lib.rs`.

- [ ] **Step 4: Rebuild + run** — `pixi run -e dev maturin develop --release` then the test → PASS.

- [ ] **Step 5: fmt + clippy + lint + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
git add src/tracks/mod.rs src/ffi/mod.rs src/lib.rs python/genvarloader/_dataset/_svar2_store_py.py tests/dataset/test_svar2_readbound_tracks.py
git commit -m "feat(rust): read-bound svar2 track re-alignment kernel

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 6: Read-bound variants / variant-windows kernel

**Files:**
- Modify: `src/ffi/mod.rs` (add `decode_variants_from_svar2_readbound`), `src/lib.rs`.
- Modify: `python/genvarloader/_dataset/_svar2_store_py.py` (`build_readbound_variants`).
- Test: `tests/dataset/test_svar2_readbound_variants.py`.

**Interfaces:**
- Produces `decode_variants_from_svar2_readbound(store, contig, <range/query args>) -> RaggedVariants-backing arrays` (per-hap positions, ilens, ALT bytes + offsets), decoding each merged key via `svar2::decode_alt` (`Inline`/`PureDel`/`Lookup`, LUT from `reader.lut_arrays()`), mirroring genoray `decode_hap`. `variant-windows` reuses the same decode + the existing window-materialization gvl already applies for SVAR1 variants.

- [ ] **Step 1: Write the failing parity test** — oracle: decode the same regions via genoray's `read_ranges(...).decode_hap` per hap (or the existing `SparseVar2Source` variants path if present). Assert per-hap `(positions, ilens, alts)` equal.

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `decode_variants_from_svar2_readbound`** — build `BatchResultSplit` via `gather_haps_readbound`, `merge_hap3` per hap, `decode_alt` each key into the `RaggedVariants` SoA. Register in `src/lib.rs`.

- [ ] **Step 4: Rebuild + run → PASS.**

- [ ] **Step 5: fmt + clippy + lint + commit** (message: `feat(rust): read-bound svar2 variants/variant-windows decode`).

---

## Task 7: Dataset read dispatch wiring (Haps discriminant, open, __getitem__)

**Files:**
- Modify: `python/genvarloader/_dataset/_haps.py` — `Haps.from_path` opens an `Svar2Store` when the dataset is svar2-backed; add a `source` discriminant; `_reconstruct_haplotypes` and the variants path branch on it; slice the cache per query.
- Modify: `python/genvarloader/_dataset/_open.py:143-173` (`_build_seqs`) — thread `svar2_link` + `svar2` override.
- Modify: `python/genvarloader/_dataset/_impl.py:124-199` (`Dataset.open`) — add keyword-only `svar2: str | Path | None = None`.
- Modify: `python/genvarloader/_dataset/_reconstruct.py:130-287` (`HapsTracks.__call__`) — route to the svar2 track kernel when source is svar2.
- Test: `tests/dataset/test_svar2_dataset.py` (end-to-end, all four modes).

**Interfaces:**
- Consumes: `svar2_ranges/` cache (Task 2), `Svar2Store` (Task 3), the three read-bound kernels (Tasks 4–6), `_resolve_svar2`/`_verify_svar2_fingerprint` (Task 1).
- Produces: a dataset whose `Haps` carries `source: Literal["svar", "svar2"]`; `Dataset.open(path, svar2=<override>)` resolves + fingerprints the `Svar2Link`; `dataset[region, sample]` issues **one** FFI call to the appropriate read-bound kernel — no interval search, no dense-union at read.

**The cache-slice → FFI mapping (the hot loop).** For a query block of `n_q` rows, each row `q` is a `(region_idx r_q, sample_idx si_q)` pair with post-jitter bounds `[start_q, end_q)`:
- `region_starts[q] = start_q` (post-jitter).
- `orig_samples[q] = sample_cols[si_q]` (from the cache's `sample_cols.npy`).
- `vk_snp_range[q*P + p] = cache.vk_snp_range[r_q, si_q, p]`, likewise `vk_indel_range`.
- `dense_snp_range[q] = cache.dense_snp_range[r_q]`, `dense_indel_range[q] = cache.dense_indel_range[r_q]` (dense is per-region, sample-independent).
- Gather these with numpy fancy-indexing on the memmapped cache (sub-linear; no per-read search), pass to the FFI.

- [ ] **Step 1: Write the end-to-end test (all four modes) — failing**

Create `tests/dataset/test_svar2_dataset.py`. Build two datasets from matched MVP fixtures — one from `.svar` (SVAR1), one from `.svar2` — over the same bed/samples/reference, and assert equality per mode:

```python
import numpy as np
import genvarloader as gvl


def _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref):
    from genoray import SparseVar, SparseVar2
    d1 = tmp_path / "d1.gvl"; d2 = tmp_path / "d2.gvl"
    gvl.write(d1, bed, variants=SparseVar(svar_fixture), overwrite=True)
    gvl.write(d2, bed, variants=SparseVar2(svar2_fixture), overwrite=True)
    return gvl.Dataset.open(d1, reference=ref), gvl.Dataset.open(d2, reference=ref)


def test_svar2_haplotypes_match_svar1(tmp_path, bed, svar_fixture, svar2_fixture, ref):
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_seqs("haplotypes")[:, :]
    b = ds2.with_seqs("haplotypes")[:, :]
    assert np.array_equal(np.asarray(a.offsets), np.asarray(b.offsets))
    assert np.array_equal(a.data.view("u1"), b.data.view("u1"))


def test_svar2_tracks_match_svar1(tmp_path, bed, svar_fixture, svar2_fixture, ref, bigwig):
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_tracks(bigwig)[:, :]
    b = ds2.with_tracks(bigwig)[:, :]
    assert np.allclose(np.asarray(a), np.asarray(b), equal_nan=True)


def test_svar2_variants_match_svar1(tmp_path, bed, svar_fixture, svar2_fixture, ref):
    ds1, ds2 = _open_pair(tmp_path, bed, svar_fixture, svar2_fixture, ref)
    a = ds1.with_seqs("variants")[:, :]
    b = ds2.with_seqs("variants")[:, :]
    assert a == b  # RaggedVariants equality (positions/ilens/alts)
```

> These fixtures (matched `.svar`/`.svar2` of the same cohort, a shared bed, a reference, a bigwig) should be added to `tests/dataset/conftest.py`; reuse the MVP `svar2_mvp` chr21 stores or a small synthetic pair. If a matched SVAR1 store isn't available, use the union-path `SparseVar2Source` as the oracle instead of SVAR1 (weaker cross-format check but still the byte-identical contract).

- [ ] **Step 2: Run → FAIL** (svar2 dataset opens but reconstructs via the SVAR1 path or errors — no dispatch yet).

- [ ] **Step 3: Add the `Svar2Store` open + discriminant in `Haps.from_path`**

In `_haps.py:363`, add a branch: when `path/genotypes/svar2_ranges/svar2_meta.json` exists (the svar2 discriminant, analogous to SVAR1's `svar_meta.json` at `:388`), resolve the `.svar2` via `_resolve_svar2(path, svar2_link, svar2_override)`, `_verify_svar2_fingerprint(...)`, memmap the six cache arrays, open `Svar2Store(str(svar2_path), contigs, n_samples=len(samples), ploidy=ploidy)`, and construct the `Haps` with `source="svar2"` (add the field to `Haps`), the store, and the cache arrays. Leave the existing SVAR1 branch as `source="svar"`. `_build_seqs` already forwards `svar_link`/`svar_override`; add `svar2_link`/`svar2_override` params to `Haps.from_path` and forward them from `_build_seqs`.

- [ ] **Step 4: Branch `_reconstruct_haplotypes` on `source`**

In `_haps.py:809`, at the top of `_reconstruct_haplotypes`, if `self.source == "svar2"`, build the flat per-query FFI inputs from the cache (the mapping above) and call `reconstruct_haplotypes_from_svar2_readbound(self.store, contig, ...)`, returning the `Ragged`. Else fall through to the unchanged SVAR1 `reconstruct_haplotypes_fused` path. Do the same source-branch for the variants path (Task 6 kernel) and, in `_reconstruct.py`'s `HapsTracks.__call__`, for the track kernel (Task 5).

> **Splice / annotated / RC / AF-keep.** This plan wires the four Phase-1 modes for the **unspliced, no-keep, no-in-kernel-RC** path (matching the current svar2 kernels' "first cut minimal"). If the test bed exercises jitter only (no splice, no `min_af`/`max_af`, no annotated), that's covered. Guard the svar2 branch to `raise NotImplementedError` for splice plans / `keep` masks / annotated kind / in-kernel `to_rc` until those `_from_svar2` kernels exist (out of scope here — see the spec's "Annotated out of scope" and the SVAR2 kernel gaps). Add explicit `raise` guards so a user hitting an unsupported combo gets a clear error, not silently-wrong output.

- [ ] **Step 5: Add `Dataset.open(svar2=...)`**

In `_impl.py:124`, add `svar2: str | Path | None = None` (keyword-only, next to `svar`), thread it into `OpenRequest(..., svar2=svar2)`, and in `_open.py` forward `self.svar2` to `_build_seqs` → `Haps.from_path(svar2_override=...)`. Mirror the two `@overload`s if they enumerate kwargs.

- [ ] **Step 6: Rebuild + run the end-to-end tests**

Run: `pixi run -e dev maturin develop --release`
Run: `pixi run -e dev pytest tests/dataset/test_svar2_dataset.py -q`
Expected: all four modes PASS (byte-identical to SVAR1 / union oracle).

- [ ] **Step 7: Full SVAR1 regression (additive guarantee)**

Run: `pixi run -e dev pytest tests -q`
Expected: entire tree green — SVAR1 path byte-unchanged, unit + dataset both covered.

- [ ] **Step 8: Lint + fmt + clippy + commit**

```bash
cargo fmt && cargo clippy --all-targets 2>&1 | tail -20
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/ && pixi run -e dev typecheck
git add python/genvarloader/_dataset/_haps.py python/genvarloader/_dataset/_open.py python/genvarloader/_dataset/_impl.py python/genvarloader/_dataset/_reconstruct.py tests/dataset/test_svar2_dataset.py tests/dataset/conftest.py
git commit -m "feat(dataset): wire svar2 read dispatch (Svar2Store, source discriminant, svar2= override)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 8: Retire the live `overlap_batch` dispatch in `_svar2_source.py`

**Files:**
- Modify: `python/genvarloader/_dataset/_svar2_source.py`.

**Interfaces:** `SparseVar2Source` stays as a **parity oracle** (used by Tasks 4–6 tests) but is no longer on any live read path. Remove the `TODO(svar2-dataset-dispatch)` marker since dispatch now lives in `Haps` (Task 7).

- [ ] **Step 1: Update the module docstring** — replace the `TODO(svar2-dataset-dispatch)` paragraph with a note that live dispatch is wired in `Haps` (read-bound, `_haps.py`) and this adapter is retained only as the union-path parity oracle for tests.

- [ ] **Step 2: Confirm nothing imports it on a live path**

Run: `pixi run -e dev python -c "import genvarloader"` and search: `rtk grep "SparseVar2Source" python/` — expect references only in `_svar2_source.py` and tests, not in `_haps.py`/`_impl.py`/`_open.py`.

- [ ] **Step 3: Run the oracle tests + commit**

Run: `pixi run -e dev pytest tests/dataset/test_svar2_readbound_haps.py tests/dataset/test_svar2_readbound_tracks.py -q`
Expected: PASS (oracle still callable).

```bash
git add python/genvarloader/_dataset/_svar2_source.py
git commit -m "refactor(dataset): retire svar2 live overlap_batch dispatch (oracle-only)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 9: Docs / roadmap / skill / api.md audit

**Files:**
- Modify: `skills/genvarloader/SKILL.md`, `docs/source/{write.md,format.md,faq.md,api.md}`, `README.md`, `docs/roadmaps/rust-migration.md`.

- [ ] **Step 1: Document `.svar2` as a `write` variant source** — in `SKILL.md` and `docs/source/write.md`: `.svar2` accepted alongside `.svar`/VCF/PGEN; note it produces a read-bound ranges cache; `Dataset.open(..., svar2=<override>)` mirrors `svar=`. In `format.md`: document the `genotypes/svar2_ranges/` layout (the six arrays + `svar2_meta.json`) and the `metadata.json` `svar2_link` field. In `faq.md`/`README.md`: `.svar2`'s on-disk size advantage; the read path builds no interval tree / no dense union.

- [ ] **Step 2: api.md ↔ `__all__` sync** — if any new public symbol was added to `python/genvarloader/__init__.py` `__all__` (e.g. a `migrate_svar2_link` analog), add its autodoc entry. Run the gate:

```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`.

- [ ] **Step 3: Roadmap** — in `docs/roadmaps/rust-migration.md`, tick the read-bound SVAR2 wiring; record the parity results (all four modes byte-identical to SVAR1/union oracle) and set the phase marker + link this plan and the genoray plan. Note the byte-identical parity contract is satisfied.

- [ ] **Step 4: Commit**

```bash
git add skills/ docs/ README.md
git commit -m "docs: .svar2 as a write variant source + read-bound wiring (skill, format, faq, roadmap)

Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>"
```

---

## Task 10: Relocate + re-run the MVP benchmark

**Files:** none in-repo (benchmark scripts live in the relocated `svar2_mvp`).

- [ ] **Step 1: Relocate the MVP tree**

```bash
mv /carter/users/dlaub/repos/for_loukik/svar2_mvp /carter/users/dlaub/projects/svar2_mvp
```
Then update any absolute paths in `svar2_mvp/build_source.sh` and the benchmark driver (`rtk grep "for_loukik" /carter/users/dlaub/projects/svar2_mvp`).

- [ ] **Step 2: Re-run the SVAR1-vs-SVAR2 `Dataset.__getitem__` benchmark** on chr21 germline (3202) + somatic (16007) after wiring — latency (same-session before/after within one allocation) + store size. Follow the profiling memory: profile the Python process with `perf` (paranoid=2, no sudo), not `py-spy --native`. Report the perf DSO split.

- [ ] **Step 3: Verify success criteria** — the warm SVAR2 read shows **neither** `SearchTree::build` **nor** a dense-union rebuild (the DSO split flips from ~80% genoray to gvl-kernel-bound, like SVAR1), and SVAR2's store-size advantage holds. Record numbers as a **relative** same-session before/after (absolute wall-clock is not comparable across allocations on shared Carter nodes — per the perf-gate memory).

- [ ] **Step 4: Record results** in the roadmap checkpoint (Task 9's roadmap file) and commit any driver-path edits that live inside the repo (if the benchmark driver is tracked).

---

## Self-Review Notes (traceability to the spec)

- **Spec Component B (write)** → Tasks 1 (`_svar2_link.py` + `Metadata.svar2_link`), 2 (`_write_from_svar2` + coercion + dispatch + 6-array cache). Reject unsupported variants via `_reject_unsupported_variants` (Task 2 Step 5).
- **Spec Component C (read, all-Rust)** → Task 3 (`genoray_core` path-dep + `Svar2Store` opened once = the SVAR1 `ffi_static` analog), Task 4 (one FFI call, `gather_haps_readbound` + `merge_hap3`, LUT via `reader.lut_arrays()` — no numpy round-trip, no Python `gather_ranges`), Task 7 (dispatch discriminant retiring `TODO(svar2-dataset-dispatch)`).
- **Spec "all four output modes, all Rust"** → Task 4 (haplotypes), 5 (tracks), 6 (variants + variant-windows). Annotated explicitly out of scope (guarded `NotImplementedError`, Task 7 Step 4).
- **Spec cache format** → Task 2 Step 5 (six arrays under `svar2_ranges/` + `svar2_meta.json`; O(offsets), bulk stays in the `.svar2`).
- **Spec parity & testing** → union-oracle parity per mode (Tasks 4–6), SVAR1 cross-format + additive regression (Task 7 Steps 6–7), perf verification (Task 10).
- **Spec "no interval search / no contig-wide union at read"** → guaranteed structurally: the read path calls only `gather_haps_readbound` (Plan 1: zero `SearchTree`, no `dense_union`); verified in Plan 1 Task 4's zero-tree test and Task 10's DSO split.
- **Spec open questions** → (channel factoring) resolved in Plan 1 (`BatchResultSplit`: vk merged + dense split). (wheel↔path-dep sync) Task 3 pins the genoray commit; the `Svar2Fingerprint` guards store identity at open. (format version) `svar2_link` is additive/defaulted — no bump (Task 1 Step 5). (arbitrary-(region,sample) mapping) resolved via Plan 1's flat `gather_haps_readbound` + Task 7's cache-slice mapping.
- **Resolved spec inaccuracies** (carried from Plan 1): genoray path is absolute, not `../genoray`; `DenseView` in `query.rs`; `decode_key` = `svar2_codec::decode_key`; htslib reach includes `lib.rs`.
