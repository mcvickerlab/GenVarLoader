# Streaming SVAR2 backend (Phase 1 — parity) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A write-free `StreamingDataset` backend for `.svar2` stores that reconstructs contiguous haplotypes on the fly, byte-identical to `gvl.write()` + `Dataset.open()[r, s]`.

**Architecture:** Mirror the landed SVAR1 backend one format down. The SVAR2 read-bound reconstruction kernel already exists (`reconstruct_haplotypes_from_svar2_readbound`, from the rust migration). Phase 1 computes the per-window range arrays **live** in Python via `SparseVar2._find_ranges` (the same Rust-backed query the writer uses at write time) instead of slicing the written path's on-disk `_Svar2Cache`, then feeds the existing FFI. **No new Rust, no genoray rev bump.** The producer-thread engine (GIL-free Rust range computation) is Phase 2 — a separate, measurement-gated follow-up (see the final section), exactly as SVAR1 split its walking skeleton (PR #274) from its engine (#275/#283).

**Tech Stack:** Python 3.10+ (abi3), genoray `SparseVar2` (Rust-backed via PyO3), existing gvl Rust FFI, polars, numpy, `seqpro.rag.Ragged`, pytest, vcfixture-rs `bulk` CLI (benchmark aid only), bcftools/samtools.

## Global Constraints

- **Target branch:** `streaming` (not `main`). Streaming PRs merge into `streaming` per `CLAUDE.md`.
- **Correctness oracle:** byte-identical parity vs `gvl.write()` + `Dataset.open()[r, s]` under `.with_seqs("haplotypes")`. A faster/streamed variant that fails parity is a bug, not a feature.
- **No genoray rev bump** in Phase 1. `Cargo.toml` stays pinned to `rev = e07477e687c913f9605fc79ea251f1bb3b177aa9`.
- **No new Rust** in Phase 1 (reuse the existing SVAR2 read-bound FFI). Therefore **no `maturin develop` rebuild is required** for Phase-1 tasks — they are pure-Python + tests.
- **Scope: haplotypes only.** No splicing, annotated/variants, `with_len`, `min_af`/`max_af`, `var_fields`, or jitter (all → #277). No tracks/intervals (→ #279).
- **Sample-index convention:** public `sample_idx` indexes the **lexicographically sorted** sample-name order (matching `gvl.write()`'s unconditional sort and `_Svar1Backend`); translate to the store's physical column via a `_phys_sample_idx` map before crossing into any query.
- **vcfixture-rs is a benchmark aid only** — never a hard dependency of a must-pass CI test. Correctness (unit parity) and the #284 cohort-scale gate use portable, vcfixture-free generation; the perf harness (Task 4) uses `vcfixture bulk` and **skips** when the binary is absent.
- **Testing commands:** `pixi run -e dev pytest <path> -v`. Before pushing, run `pixi run -e dev pytest tests/dataset tests/unit -q` (shared code touches both trees). Lint: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/`. Types: `pixi run -e dev typecheck`.
- **Commit hooks:** ensure `prek install` has run in this worktree before the first commit (`.pre-commit-config.yaml` present).

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `python/genvarloader/_dataset/_streaming.py` | Add `_Svar2Backend`; add a `"sync"` `_iter_batches` branch; make `__init__`'s `_prefetch_strategy`/`cell_bytes` backend-derived; wire `.svar2` dispatch (replace the `NotImplementedError` at ~187-191); widen `_backend` to a union. | 2 |
| `tests/dataset/conftest.py` | Add a small, vcfixture-free `svar2_multicontig_fixture` (mirror `svar1_multicontig_fixture`; reuse the `SparseVar2.from_vcf` pattern already referenced from `test_svar2_dataset.py`). | 2 |
| `tests/dataset/test_streaming_parity_svar2.py` | Byte-parity tests over the small fixture (multi-contig, unsorted bed; sample-identity; mixed contig-naming). | 2 |
| `tests/dataset/test_streaming_scale.py` | Add SVAR2 cohort-flatness (#284) gate alongside the SVAR1 ones. | 3 |
| `tests/benchmarks/data/build_svar2_stream_bulk.py` | vcfixture-rs `bulk` → BCF → `.svar2` + contiguous BED + `gvl.write` oracle, at arbitrary cohort scale. Self-contained (does not import the unmerged splice builder). | 1 |
| `benchmarking/streaming/svar2_cold_cache.py` | Cold-cache synchronous-path baseline + IO-vs-CPU bound profiling over the bulk fixture. Measures only. | 4 |
| Docs: `README.md`, `docs/source/{faq,dataset,write}.md`, `skills/genvarloader/SKILL.md`, `docs/roadmaps/streaming-dataset.md` | Add `.svar2` to the supported streaming-input list; fill the roadmap Specs/Plans rows. | 5 |

**Parallelism (per project convention — dispatch with `superpowers:dispatching-parallel-agents` + `subagent-driven-development`):**
- **Wave A (parallel):** Task 1 (bulk builder) ∥ Task 2 (backend + parity). Different files, no shared state.
- **Wave B (after Task 2, parallel):** Task 3 (#284 gate) ∥ Task 4 (perf baseline, also needs Task 1) ∥ Task 5 (docs).
- Use **Sonnet or weaker** for implementation subagents (per user preference); reserve stronger models for a second pass only if an implementer critically fails.

---

## Task 1: vcfixture-rs bulk SVAR2 benchmark fixture builder

**Files:**
- Create: `tests/benchmarks/data/build_svar2_stream_bulk.py`
- Test: `tests/benchmarks/test_build_svar2_stream_bulk.py`

**Interfaces:**
- Produces: `build(out_dir, *, n_samples, records=20_000, seed=42, contig="chr1", region_len=1000, n_regions=64) -> BulkStreamFixture` where
  `BulkStreamFixture = dataclass(gvl_path: Path, svar2_path: Path, reference: Path, bed: pl.DataFrame, n_samples: int, records: int)`.
  Later tasks (4) import `build` and `BulkStreamFixture`.
- Consumes: nothing from other tasks. `vcfixture` binary via `VCFIXTURE_BIN` env or PATH; `bcftools`, `samtools`; `genoray.SparseVar2`; `genvarloader`.

This reuses the proven `gen_cohort` recipe from `tests/benchmarks/data/build_svar2_splice_bulk.py` (currently on `main`, arriving in `streaming` at the next merge) but is **self-contained** and builds a **contiguous-region** bed (streaming reconstructs contiguous haplotypes, not spliced).

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/benchmarks/test_build_svar2_stream_bulk.py
"""Smoke test for the bulk SVAR2 streaming fixture builder. Skips without vcfixture."""
from __future__ import annotations

import shutil
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _has_vcfixture() -> bool:
    return bool(os.environ.get("VCFIXTURE_BIN") or shutil.which("vcfixture"))


@pytest.mark.skipif(not _has_vcfixture(), reason="vcfixture-rs bulk CLI not available")
def test_build_bulk_svar2_stream_fixture(tmp_path: Path) -> None:
    from tests.benchmarks.data.build_svar2_stream_bulk import build, BulkStreamFixture

    fx = build(tmp_path, n_samples=8, records=200, n_regions=4, region_len=100)
    assert isinstance(fx, BulkStreamFixture)
    assert fx.svar2_path.exists() and (fx.svar2_path / "meta.json").exists()
    assert (fx.gvl_path).exists()  # the gvl.write oracle dataset
    assert fx.reference.exists()
    assert fx.bed.height == 4
    assert set(fx.bed.columns) >= {"chrom", "chromStart", "chromEnd"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/benchmarks/test_build_svar2_stream_bulk.py -v -m slow`
Expected: FAIL (collection/import error — `build_svar2_stream_bulk` does not exist), or SKIP if vcfixture is genuinely absent on the box. If it skips, still proceed — the builder must exist for Task 4.

- [ ] **Step 3: Write the builder**

```python
# tests/benchmarks/data/build_svar2_stream_bulk.py
"""Build a *bulk* SVAR2 dataset (contiguous regions) for streaming benchmarks.

vcfixture-rs ``bulk`` synthesizes arbitrarily large cohorts (``n_samples`` toward
AoU scale, the dominating axis) drawn i.i.d. from a fitted 1kGP site-frequency
spectrum. Genotypes have no LD -- irrelevant to a decode/gather-throughput
benchmark, per vcfixture's own ablation. ``SparseVar2.from_vcf(no_reference=True,
skip_out_of_scope=True)`` avoids REF-vs-FASTA validation (vcfixture draws REF
i.i.d.) and drops symbolic/breakend records the short-read codec can't represent.
A synthetic reference (correct contig name + length) is written for gvl.write /
Dataset.open. ``build`` raises FileNotFoundError when a tool is missing; the
benchmark turns that into a skip.

Run standalone:
    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev python \
        tests/benchmarks/data/build_svar2_stream_bulk.py /tmp/svar2_stream_bulk 2000 200
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

DEFAULT_RECORDS = 20_000
DEFAULT_SEED = 42


@dataclass(frozen=True)
class BulkStreamFixture:
    gvl_path: Path
    svar2_path: Path
    reference: Path
    bed: pl.DataFrame
    n_samples: int
    records: int


def _which_vcfixture() -> str:
    cand = os.environ.get("VCFIXTURE_BIN") or shutil.which("vcfixture")
    if not cand or not Path(cand).exists():
        raise FileNotFoundError(
            "vcfixture-rs bulk CLI not found: set VCFIXTURE_BIN or put `vcfixture` "
            "on PATH (build with `cargo build --release --features cli` in vcfixture-rs)."
        )
    return str(cand)


def _require(tool: str) -> None:
    if shutil.which(tool) is None:
        raise FileNotFoundError(f"required tool not on PATH: {tool}")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _contig_span(bcf: Path, contig: str) -> int:
    hdr = subprocess.run(
        ["bcftools", "view", "-h", str(bcf)], check=True, capture_output=True, text=True
    ).stdout
    pat = re.compile(
        r"##contig=<[^>]*ID=" + re.escape(contig) + r"[^>]*length=(\d+)", re.IGNORECASE
    )
    for line in hdr.splitlines():
        m = pat.search(line)
        if m:
            return int(m.group(1))
    raise ValueError(f"no ##contig length for {contig!r} in {bcf} header")


def gen_cohort(
    out_dir: Path,
    n_samples: int,
    records: int,
    *,
    contig: str = "chr1",
    seed: int = DEFAULT_SEED,
    profile: str = "germline-1kgp",
    payload: str = "gt-only",
) -> tuple[Path, int]:
    """Generate + bi-allelic-normalize a cohort BCF. Returns (bcf, span). Cached by shape."""
    vcfixture = _which_vcfixture()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"cohort_s{n_samples}_r{records}_seed{seed}"
    raw = out_dir / f"{stem}.raw.bcf"
    norm = out_dir / f"{stem}.bcf"
    if not norm.exists():
        _run([
            vcfixture, "bulk",
            "--profile", profile,
            "--samples", str(n_samples),
            "--contigs", contig,
            "--records", str(records),
            "--payload", payload,
            "--seed", str(seed),
            "-o", str(raw),
        ])
        # GVL requires bi-allelic, atomized variants (from_vcf skips SV/symbolic ALTs).
        _run(["bcftools", "norm", "-m-", "-Ob", "-o", str(norm), str(raw)])
        _run(["bcftools", "index", "-f", str(norm)])
    return norm, _contig_span(norm, contig)


def make_contiguous_bed(
    contig: str, span: int, n_regions: int, region_len: int
) -> pl.DataFrame:
    """Tile n_regions non-overlapping windows of region_len inside [0, span)."""
    stride = max(span // max(n_regions, 1), region_len)
    rows = []
    for i in range(n_regions):
        start = (i * stride) % max(span - region_len, 1)
        rows.append({"chrom": contig, "chromStart": start, "chromEnd": start + region_len})
    return pl.DataFrame(rows)


def _write_reference(path: Path, contig: str, length: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    seq = rng.choice(np.frombuffer(b"ACGT", "S1"), size=length).tobytes().decode()
    path.write_text(f">{contig}\n{seq}\n")
    _run(["samtools", "faidx", str(path)])


def build(
    out_dir: Path,
    *,
    n_samples: int,
    records: int = DEFAULT_RECORDS,
    seed: int = DEFAULT_SEED,
    contig: str = "chr1",
    region_len: int = 1000,
    n_regions: int = 64,
) -> BulkStreamFixture:
    """Build one bulk SVAR2 streaming dataset. Idempotent (cached by shape)."""
    import genvarloader as gvl
    from genoray import SparseVar2

    _require("bcftools")
    _require("samtools")

    root = Path(out_dir) / f"s{n_samples}_r{records}_{n_regions}x{region_len}"
    gvl_path = root / "ds.gvl"
    ref = root / "ref.fa"
    svar2 = root / "store.svar2"

    bcf, span = gen_cohort(
        Path(out_dir) / "cohorts", n_samples, records, contig=contig, seed=seed
    )
    bed = make_contiguous_bed(contig, span, n_regions, region_len)

    if gvl_path.exists() and ref.exists() and (svar2 / "meta.json").exists():
        return BulkStreamFixture(gvl_path, svar2, ref, bed, n_samples, records)

    root.mkdir(parents=True, exist_ok=True)
    _write_reference(ref, contig, span + region_len)
    if not (svar2 / "meta.json").exists():
        SparseVar2.from_vcf(
            svar2, bcf, no_reference=True, skip_out_of_scope=True, overwrite=True
        )
    gvl.write(gvl_path, bed, variants=SparseVar2(svar2), overwrite=True)
    return BulkStreamFixture(gvl_path, svar2, ref, bed, n_samples, records)


if __name__ == "__main__":
    import sys

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("svar2_stream_bulk.cache")
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    records = int(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_RECORDS
    fx = build(out, n_samples=n_samples, records=records)
    print(f"gvl oracle: {fx.gvl_path}")
    print(f"svar2 store: {fx.svar2_path}")
    print(f"bed: {fx.bed.height} regions")
```

- [ ] **Step 4: Run the smoke test**

Run: `pixi run -e dev pytest tests/benchmarks/test_build_svar2_stream_bulk.py -v -m slow`
Expected: PASS if `vcfixture` is available; SKIP otherwise. Either outcome is acceptable to proceed (the file exists and imports clean). Confirm import cleanliness even on skip: `pixi run -e dev python -c "from tests.benchmarks.data.build_svar2_stream_bulk import build, BulkStreamFixture; print('ok')"` → prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add tests/benchmarks/data/build_svar2_stream_bulk.py tests/benchmarks/test_build_svar2_stream_bulk.py
git commit -m "test(streaming): vcfixture-rs bulk SVAR2 streaming fixture builder (#278)"
```

---

## Task 2: `_Svar2Backend`, `.svar2` dispatch, and the `"sync"` iteration path

**Files:**
- Modify: `python/genvarloader/_dataset/_streaming.py` (add `_Svar2Backend` after `_Svar1Backend` which ends ~line 895; add a `"sync"` `_iter_batches` branch; make `__init__`'s `_prefetch_strategy`/`cell_bytes` backend-derived; replace `.svar2` `NotImplementedError` at ~187-191; widen `_backend` field to a union ~line 122)
- Modify: `tests/dataset/conftest.py` (add `svar2_multicontig_fixture`)
- Create: `tests/dataset/test_streaming_parity_svar2.py`

**Interfaces:**
- Consumes:
  - `SparseVar2._find_ranges(contig, starts, ends, samples) -> RangesBundle` (genoray; Rust-backed). Returns a dict-like bundle with keys `region_starts`, `sample_cols`, `vk_snp_range`, `vk_indel_range`, and dense-range key(s). **Read `genoray/_svar2_batch.py:104` for the exact bundle keys/shapes.**
  - The existing SVAR2 read-bound FFI that `Svar2Haps._reconstruct_haps` calls at `python/genvarloader/_dataset/_svar2_haps.py:615` (name imported at `_svar2_haps.py:44-49`; Rust def `src/ffi/mod.rs:1321-1482`). Its inputs and the exact per-window tuple are assembled in `_svar2_haps.py:_gather_inputs` (`1123-1167`), sourced there from the on-disk `_Svar2Cache` (`_svar2_haps.py:82-96`).
  - `Svar2Store(store_path: str, contigs: list[str], n_samples: int, ploidy: int)` pyclass (`src/svar2/store.rs`).
  - `Reference.from_path` + `Reference._contig_slice(contig_idx)` (used by `_Svar1Backend`).
- Produces:
  - `_Svar2Backend(svar2_path, reference_path, contigs, bed)`: `read_window(r_idx, s_idx) -> window` (a `_find_ranges` bundle), `generate_batch(r_idx, s_idx, window, lo, hi) -> Ragged`, attrs `n_samples`, `ploidy`, `_sample_names`, `_cell_bytes = ploidy*32`, `_default_strategy = "sync"`. Selected by `StreamingDataset.__init__` for `.svar2`.
  - A `"sync"` `_iter_batches` strategy (engine-less, prefetch-less) driving a backend via `read_window`/`generate_batch`.
  - `_backend` field widened to a union `_Svar1Backend | _Svar2Backend | None`; `_Svar1Backend` gains additive `_cell_bytes`/`_default_strategy` attrs but is otherwise untouched. (No formal Protocol in Phase 1 — see Step 4.)

**The core idea:** `_Svar2Backend` is `_Svar1Backend` with the read+generate steps swapped. `read_window` runs `SparseVar2._find_ranges` live (vs SVAR1's `svar1_read_window`); `generate_batch` runs the existing SVAR2 read-bound FFI over those ranges (vs SVAR1's `svar1_generate_batch`). The bundle→FFI-tuple mapping is authoritatively defined by the writer (`_write.py:1214-1220`) and the read-bound tuple by `_svar2_haps.py:_gather_inputs` (`1123-1167`); the parity oracle (TDD) confirms it.

- [ ] **Step 1: Add the small, vcfixture-free SVAR2 parity fixture to conftest**

Mirror `svar1_multicontig_fixture` (`tests/dataset/conftest.py:~120`), but build a `.svar2` store. Reuse the existing SVAR2 VCF referenced by the conftest header comment ("Mirrors the SVAR2-parity fixture in `tests/dataset/test_svar2_dataset.py` (`_src`/`svar_fixture`)"). Read that fixture first to copy its `SparseVar2.from_vcf` call, then add:

```python
# tests/dataset/conftest.py  (add near the other Svar1* fixtures)

@dataclass(slots=True)
class Svar2MultiContigFixture:
    svar2_path: Path
    reference_path: Path
    contigs: list[str]
    bed: pl.DataFrame
    dataset_path: Path


@pytest.fixture(scope="module")
def svar2_multicontig_fixture(tmp_path_factory) -> Svar2MultiContigFixture:
    from genoray import SparseVar2, VCF

    d = tmp_path_factory.mktemp("svar2_mc_src")
    ref = d / "ref.fa"
    ref.write_text(f">chr1\n{_SVAR1_STREAM_REF}\n>chr2\n{_SVAR1_MC_REF2}\n")
    subprocess.run(["samtools", "faidx", str(ref)], check=True)

    vcf = d / "in.vcf"
    vcf.write_text(_SVAR1_MC_VCF)  # same multi-contig VCF the SVAR1 fixture uses
    bcf = d / "in.bcf"
    subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
    subprocess.run(["bcftools", "index", str(bcf)], check=True)

    svar2_path = tmp_path_factory.mktemp("svar2_mc_store") / "store.svar2"
    SparseVar2.from_vcf(svar2_path, VCF(bcf), samples=["S0", "S1", "S2"], overwrite=True)

    starts = [0, 4, 8, 12, 16, 20]
    rows = []
    for i, s in enumerate(starts):
        c1, c2 = ("chr2", "chr1") if i % 2 == 0 else ("chr1", "chr2")
        rows.append({"chrom": c1, "chromStart": s, "chromEnd": s + 20})
        rows.append({"chrom": c2, "chromStart": s, "chromEnd": s + 20})
    bed = pl.DataFrame(rows)

    out = tmp_path_factory.mktemp("svar2_mc_ds") / "d2.gvl"
    gvl.write(out, bed, variants=SparseVar2(svar2_path), samples=None, overwrite=True)

    return Svar2MultiContigFixture(
        svar2_path=svar2_path, reference_path=ref,
        contigs=["chr1", "chr2"], bed=bed, dataset_path=out,
    )
```

Note: if `SparseVar2.from_vcf` in this genoray rev does not accept `samples=`, drop that kwarg and let it take all samples — check its signature via `pixi run -e dev python -c "from genoray import SparseVar2; help(SparseVar2.from_vcf)"`.

- [ ] **Step 2: Write the failing parity test**

```python
# tests/dataset/test_streaming_parity_svar2.py
"""Byte-identical parity: StreamingDataset over a .svar2 store vs a written gvl.Dataset."""
from __future__ import annotations

import numpy as np
import pytest

import genvarloader as gvl


def test_streaming_svar2_matches_written_all_cells(svar2_multicontig_fixture) -> None:
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    written = gvl.Dataset.open(fx.dataset_path, reference=fx.reference_path).with_seqs(
        "haplotypes"
    )

    seen = 0
    for data, r_idx, s_idx in sds.to_iter(batch_size=4, return_indices=True):
        for i in range(len(r_idx)):
            r, s = int(r_idx[i]), int(s_idx[i])
            exp = written[r, s]
            for p in range(sds.ploidy):
                np.testing.assert_array_equal(
                    np.asarray(data[i][p]), np.asarray(exp[p]),
                    err_msg=f"mismatch at region={r} sample={s} ploid={p}",
                )
            seen += 1
    assert seen == fx.bed.height * sds.n_samples


def test_streaming_svar2_covers_every_cell_once(svar2_multicontig_fixture) -> None:
    fx = svar2_multicontig_fixture
    sds = gvl.StreamingDataset(
        fx.bed, reference=fx.reference_path, variants=fx.svar2_path
    ).with_seqs("haplotypes")
    cells = set()
    for _, r_idx, s_idx in sds.to_iter(batch_size=3, return_indices=True):
        for i in range(len(r_idx)):
            cells.add((int(r_idx[i]), int(s_idx[i])))
    assert cells == {(r, s) for r in range(fx.bed.height) for s in range(sds.n_samples)}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py -v`
Expected: FAIL with `NotImplementedError: StreamingDataset does not support SVAR2 stores yet` (the current dispatch at `_streaming.py:187-191`).

- [ ] **Step 4: Add a backend union, a `"sync"` iteration branch, and backend-derived sizing (minimal, SVAR1-untouched)**

The two existing `_iter_batches` strategies (`"engine"`, `"readahead"`, `_streaming.py:308-419`) are both SVAR1-shaped: the engine uses `build_engine`; readahead calls `svar1_prefetch_runs` plus `generate_batch(r_idx, s_idx, cur[0], cur[1], lo, hi)`. `generate_batch` is only ever called from its own backend's strategy branch, so **do not change `_Svar1Backend`'s signature** (an existing test, `test_streaming_scale.py:243`, calls the 6-arg form, and the readahead branch unpacks the 2-tuple). SVAR2 gets a new `"sync"` branch and its own `generate_batch(r_idx, s_idx, window, lo, hi)`. A formal `StreamBackend` Protocol that unifies `generate_batch` is deferred to Phase 2, when both backends are engine-driven and their window shapes converge — YAGNI until then.

**Deviation from the spec's boy-scout item, on purpose:** the spec proposed a `StreamBackend` Protocol now. In Phase 1 the two backends have genuinely different window shapes and iteration strategies (engine vs sync), so unifying `generate_batch`'s arity would force churn on working SVAR1 code and its tests for no runtime benefit. Phase 1 uses a plain union type; the Protocol lands in Phase 2 where it pays off.

**4a.** Widen the dataclass field (`_streaming.py:~122`) and add additive attrs to `_Svar1Backend.__init__` (safe — nothing else touched):

```python
    _backend: "_Svar1Backend | _Svar2Backend | None"   # was: _Svar1Backend | None
```
In `_Svar1Backend.__init__`, add: `self._cell_bytes = int(self.ploidy) * 16` and `self._default_strategy = "engine"`.

**4b.** _(intentionally omitted — no SVAR1 `generate_batch` refactor.)_

**4c.** Add a backend-agnostic `"sync"` branch to `_iter_batches` (after the `"readahead"` elif, before the `else: raise ValueError`, `_streaming.py:~415`) — a plain synchronous, no-prefetch, no-engine drive for backends that (in Phase 1) have neither:

```python
            elif self._prefetch_strategy == "sync":
                # No engine, no prefetch: read each window's ranges, then generate
                # per batch_size row slice. Output is (hi-lo)-bounded (issue #284).
                for r_idx, s_idx in self._plan():
                    window = self._backend.read_window(r_idx, s_idx)
                    n_s = len(s_idx)
                    flat_r = np.repeat(self._sort_order[r_idx], n_s)
                    flat_s = np.tile(s_idx, len(r_idx))
                    n_rows = len(flat_r)
                    for lo in range(0, n_rows, batch_size):
                        hi = min(lo + batch_size, n_rows)
                        data = self._backend.generate_batch(r_idx, s_idx, window, lo, hi)
                        yield data, flat_r[lo:hi], flat_s[lo:hi]
```

**4d.** Make `__init__`'s `_prefetch_strategy` and `cell_bytes` backend-derived instead of hardcoded. Replace the hardcoded `object.__setattr__(self, "_prefetch_strategy", "engine")` (`_streaming.py:231`) and `cell_bytes = int(ploidy) * 16` (`:245-247`) with:

```python
        # at _streaming.py:231 (was: hardcoded "engine"):
        _strat = getattr(_backend_obj, "_default_strategy", "engine")
        object.__setattr__(self, "_prefetch_strategy", _strat)

        # at _streaming.py:245-247 (was: cell_bytes = int(ploidy) * 16):
        cell_bytes = getattr(_backend_obj, "_cell_bytes", int(ploidy) * 16)
```

(The `_backend` field widening is done in Step 4a — the union `_Svar1Backend | _Svar2Backend | None`.)

- [ ] **Step 5: Implement `_Svar2Backend`**

Add after `_Svar1Backend`. This is `_Svar1Backend` with the read+generate steps swapped to source ranges from `SparseVar2._find_ranges` and call the SVAR2 read-bound FFI. The window is the raw `_find_ranges` bundle reshaped to per-`(region, sample)` arrays; `generate_batch` slices rows `[lo, hi)` and builds the EXACT 7-arg tuple `_svar2_haps.py:_gather_inputs` (`1123-1167`) produces — sourced live instead of from the cache. The bundle keys and reshapes are authoritatively defined by the writer at `_write.py:1214-1220` (`vk_snp_range`/`vk_indel_range` are row-major → `(n_reg, n_s, P, 2)`; `dense_snp_range`/`dense_indel_range` → `(n_reg, 2)`).

```python
class _Svar2Backend:
    """StreamingDataset backend for .svar2 stores. Mirrors _Svar1Backend, but per-
    window variant ranges are computed LIVE via SparseVar2._find_ranges (the same
    Rust-backed query gvl.write uses at write time, _write.py:1214) instead of
    slicing an on-disk _Svar2Cache; the ranges feed the existing SVAR2 read-bound
    FFI (reconstruct_haplotypes_from_svar2_readbound). Phase 1: synchronous only."""

    _default_strategy = "sync"

    def __init__(self, svar2_path, reference_path, contigs, bed):
        from genoray import SparseVar2
        from ..genvarloader import Svar2Store  # pyclass, src/svar2/store.rs

        self._sv = SparseVar2(str(svar2_path))
        native = list(self._sv.samples)                       # store column order
        self._sample_names = sorted(native)                   # public sorted order
        col_of = {name: i for i, name in enumerate(native)}
        self._phys_sample_idx = np.array(
            [col_of[n] for n in self._sample_names], dtype=np.int64
        )
        self.n_samples = len(self._sample_names)
        self.ploidy = int(self._sv.ploidy)
        # Resident window buffer holds vk_snp_range + vk_indel_range (i64 pairs) per
        # (region, sample, ploid): 2 arrays x 2 i64 = 32 B/cell. STARTING ESTIMATE
        # (validate in Task 4); gathered flat channels add bounded window-scale extra.
        self._cell_bytes = self.ploidy * 32
        self._contigs = list(contigs)
        self._store = Svar2Store(
            str(svar2_path), self._contigs, self.n_samples, self.ploidy
        )
        self._ref = Reference.from_path(reference_path, self._contigs)
        # Identical region-bounds derivation to _Svar1Backend (and StreamingDataset):
        # r_idx from the plan indexes THIS sorted regions table. Reuse the same
        # imports _Svar1Backend uses (bed_to_regions, ContigNormalizer, sp.bed).
        bed_df = bed if isinstance(bed, pl.DataFrame) else sp.bed.read(bed)
        self._regions = bed_to_regions(
            sp.bed.sort(bed_df), ContigNormalizer(self._contigs)
        )

    def _contig_of(self, r_idx: "NDArray[np.intp]") -> tuple[int, str]:
        contig_idxs = self._regions[r_idx, 0]
        contig_idx = int(contig_idxs[0])
        if not np.all(contig_idxs == contig_idx):
            raise ValueError("_Svar2Backend: window spans multiple contigs")
        return contig_idx, self._contigs[contig_idx]

    def read_window(self, r_idx, s_idx):
        """Compute the window's live ranges. Returns per-(region,sample) arrays plus
        the physical sample columns, reshaped exactly as _write.py:1214-1220 does."""
        r_idx = np.asarray(r_idx, np.intp)
        s_idx = np.asarray(s_idx, np.intp)
        contig_idx, contig = self._contig_of(r_idx)
        rb = self._regions[r_idx, 1:3]                # (n_reg, 2) int
        starts = np.ascontiguousarray(rb[:, 0])
        ends = np.ascontiguousarray(rb[:, 1])
        phys = self._phys_sample_idx[s_idx]           # physical store columns
        d = self._sv._find_ranges(contig, starts, ends, phys)
        n_reg, n_s, P = len(r_idx), len(s_idx), self.ploidy
        return {
            "contig_idx": contig_idx,
            "region_bounds": np.ascontiguousarray(rb, np.int32),  # (n_reg, 2)
            "orig_samples": np.ascontiguousarray(phys, np.int64),  # (n_s,)
            "vk_snp": np.asarray(d["vk_snp_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "vk_indel": np.asarray(d["vk_indel_range"], np.int64).reshape(n_reg, n_s, P, 2),
            "dense_snp": np.asarray(d["dense_snp_range"], np.int64).reshape(n_reg, 2),
            "dense_indel": np.asarray(d["dense_indel_range"], np.int64).reshape(n_reg, 2),
        }

    def generate_batch(self, r_idx, s_idx, window, lo, hi) -> Ragged:
        """Reconstruct C-order (region, sample) rows [lo, hi) of the window. Output is
        (hi-lo)-bounded (issue #284). Builds the same 7-arg tuple as
        _svar2_haps.py:_gather_inputs, then calls the SVAR2 read-bound FFI."""
        from ..genvarloader import reconstruct_haplotypes_from_svar2_readbound

        r_idx = np.asarray(r_idx, np.intp)
        n_s = len(np.asarray(s_idx))
        P = self.ploidy
        contig_idx = window["contig_idx"]
        contig = self._contigs[contig_idx]
        ref_, ref_offsets = self._ref._contig_slice(contig_idx)

        rows = np.arange(lo, hi)
        ri = rows // n_s          # region-in-window per row
        si = rows % n_s           # sample-in-window per row
        # Per-row FFI inputs (C-order (region, sample)), mirroring _gather_inputs:
        region_bounds = np.ascontiguousarray(window["region_bounds"][ri], np.int32)  # (m,2)
        region_starts = np.ascontiguousarray(region_bounds[:, 0], np.uint32)         # (m,)
        orig_samples = np.ascontiguousarray(window["orig_samples"][si], np.int64)    # (m,)
        vk_snp = np.ascontiguousarray(
            window["vk_snp"][ri, si].reshape(-1, 2), np.int64                        # (m*P,2)
        )
        vk_indel = np.ascontiguousarray(
            window["vk_indel"][ri, si].reshape(-1, 2), np.int64
        )
        dense_snp = np.ascontiguousarray(window["dense_snp"][ri], np.int64)          # (m,2)
        dense_indel = np.ascontiguousarray(window["dense_indel"][ri], np.int64)
        m = hi - lo
        shifts = np.zeros((m, P), np.int32)     # jitter out of scope (jitter=0)

        data, offsets = reconstruct_haplotypes_from_svar2_readbound(
            self._store,
            contig,
            region_starts,
            orig_samples,
            vk_snp,
            vk_indel,
            dense_snp,
            dense_indel,
            region_bounds,
            shifts,
            ref_,
            ref_offsets,
            np.uint8(self._ref.pad_char),
            np.int64(-1),      # ragged output (no fixed output_length)
            True,              # parallel
            False,             # filter_exonic (splicing out of scope)
        )
        return Ragged.from_offsets(
            np.asarray(data).view("S1"), (m, P, None), np.asarray(offsets, np.int64)
        )
```

**Imports for `_Svar2Backend`:** reuse exactly what `_Svar1Backend` imports for its region derivation — `bed_to_regions`, `ContigNormalizer`, and `seqpro as sp` (grep `bed_to_regions` / `ContigNormalizer` at the top of `_streaming.py`; they are already imported at module scope, so no new imports are needed). `_regions` is built the same way `_Svar1Backend.__init__` builds it (`_streaming.py:~624`), so a batch's `r_idx` indexes the identical sorted table.

**If parity fails**, it is an indexing/reshape bug here (row order, `ri/si`, or vk reshape), NOT a kernel bug — the kernel is the same one the written path passes and `test_svar2_dataset.py` already covers. Diff your 7-tuple against `_gather_inputs`'s for the same `(contig, r_idx, s_idx)`.

- [ ] **Step 6: Wire the dispatch**

Replace the `.svar2` `NotImplementedError` (`_streaming.py:187-191`) with (matching the `.svar` branch's exact local names — `backend`, `_backend_obj`, `_reconstruct_window`, `samples`):

```python
            elif p.is_dir() and p.suffix == ".svar2":
                from genoray import SparseVar2

                contigs = SparseVar2(str(p)).contigs
                backend = _Svar2Backend(p, reference, contigs, regions)
                n_samples = backend.n_samples
                ploidy = backend.ploidy
                samples = backend._sample_names
                _reconstruct_window = None
                _backend_obj = backend
```

- [ ] **Step 7: Run parity tests to green**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_parity_svar2.py -v`
Expected: PASS (both tests). Debug per Step 5's note if a mismatch appears.

- [ ] **Step 8: Lint, type-check, and confirm no SVAR1 regression**

Run:
```bash
pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format python/ tests/
pixi run -e dev typecheck
pixi run -e dev pytest tests/dataset/test_streaming_parity.py tests/dataset/test_streaming_parity_svar2.py tests/dataset/test_streaming_scale.py -v
```
Expected: clean lint/format, types pass, ALL SVAR1 + SVAR2 parity **and** the existing SVAR1 scale tests pass — the SVAR1 sets confirm the additive `__init__`/union changes did not regress SVAR1 (which still runs `"engine"`).

- [ ] **Step 9: Commit**

```bash
git add python/genvarloader/_dataset/_streaming.py tests/dataset/conftest.py tests/dataset/test_streaming_parity_svar2.py
git commit -m "feat(streaming): SVAR2 backend (live _find_ranges + read-bound kernel), byte-parity (#278)"
```

---

## Task 3: SVAR2 cohort-scale (#284) gate

**Files:**
- Modify: `tests/dataset/test_streaming_scale.py` (add SVAR2 case beside the SVAR1 ones)

**Interfaces:**
- Consumes: `_Svar2Backend`, `StreamingDataset` (Task 2). A **portable, vcfixture-free** cohort generator — reuse the module-scope programmatic VCF→BCF→`SparseVar2.from_vcf` approach the SVAR1 `scale_fixture` uses (`test_streaming_scale.py:55`), swapping `SparseVar` for `SparseVar2`.
- Produces: a deterministic gate proving per-batch (not whole-window) generation at cohort scale.

The #284 property: a fixed-`batch_size` call's output byte count is identical between a small and a large cohort (per-batch generation), while the read window still covers the whole cohort.

- [ ] **Step 1: Write the failing scale test**

This is a direct clone of `test_generate_batch_output_is_flat_in_cohort_size` (`test_streaming_scale.py:145`), with three substitutions: `SparseVar`→`SparseVar2`, `store.svar`→`store.svar2`, and the SVAR2 `read_window`/`generate_batch` call shape (`read_window` returns a `window` object; `generate_batch` takes it opaquely). SNP-only variants keep haplotype length == region length so batch bytes are cohort-independent.

```python
# add to tests/dataset/test_streaming_scale.py

def test_svar2_generate_batch_output_is_flat_in_cohort_size(tmp_path):
    """SVAR2 #284 gate: a fixed-batch_size call's output bytes are identical between a
    50- and a 400-sample cohort (per-batch generation), while the window covers the
    whole cohort. Clone of the SVAR1 test above, SparseVar2 + .svar2."""
    import subprocess

    import numpy as np
    import polars as pl
    from genoray import VCF, SparseVar2

    def build(n_samples: int):
        d = tmp_path / f"n{n_samples}"
        d.mkdir()
        ref = d / "ref.fa"
        rng = np.random.default_rng(1)
        seq = "".join(rng.choice(list("ACGT"), 4000))
        ref.write_text(f">chr1\n{seq}\n")
        subprocess.run(["samtools", "faidx", str(ref)], check=True)
        vcf = d / "in.vcf"
        lines = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr1,length=4000>",
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="GT">',
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t"
            + "\t".join(f"S{i}" for i in range(n_samples)),
        ]
        pos = np.sort(rng.choice(np.arange(2, 3998), 200, replace=False))
        for p in pos:
            gts = "\t".join(
                f"{rng.integers(0, 2)}|{rng.integers(0, 2)}" for _ in range(n_samples)
            )
            lines.append(f"chr1\t{p}\t.\tA\tG\t.\t.\t.\tGT\t{gts}")
        vcf.write_text("\n".join(lines) + "\n")
        bcf = d / "in.bcf"
        subprocess.run(["bcftools", "view", "-Ob", "-o", str(bcf), str(vcf)], check=True)
        subprocess.run(["bcftools", "index", str(bcf)], check=True)
        svar2 = d / "store.svar2"
        SparseVar2.from_vcf(
            svar2, VCF(bcf), samples=[f"S{i}" for i in range(n_samples)], overwrite=True
        )
        return svar2, ref

    def batch_output_bytes(n_samples: int) -> int:
        svar2, ref = build(n_samples)
        bed = pl.DataFrame(
            {
                "chrom": ["chr1"] * 4,
                "chromStart": [0, 100, 200, 300],
                "chromEnd": [100, 200, 300, 400],
            }
        )
        sds = gvl.StreamingDataset(bed, reference=ref, variants=svar2).with_seqs(
            "haplotypes"
        )
        backend = sds._backend
        assert backend is not None, "test requires the real SVAR2 backend"
        r_idx, s_idx = next(iter(sds._plan()))
        assert len(s_idx) == n_samples, (
            f"expected the window to cover the whole cohort ({n_samples}), "
            f"got {len(s_idx)}"
        )
        window = backend.read_window(r_idx, s_idx)          # SVAR2: opaque bundle
        batch_size = 4
        assert batch_size <= len(r_idx) * len(s_idx)
        data = backend.generate_batch(r_idx, s_idx, window, 0, batch_size)
        return int(data.data.nbytes)

    bytes_50 = batch_output_bytes(50)
    bytes_400 = batch_output_bytes(400)
    print(f"[svar2 batch-output-gate] n=50 bytes={bytes_50} n=400 bytes={bytes_400}")
    assert bytes_50 > 0, "counter is not wired (batch produced no output)"
    assert bytes_400 == bytes_50, (
        f"batch output bytes scaled with cohort size (50->{bytes_50}B, "
        f"400->{bytes_400}B) -- whole-window output materialization has returned"
    )
```

If `SparseVar2.from_vcf` in this genoray rev rejects `samples=`, drop that kwarg (it defaults to all samples); the assertion `len(s_idx) == n_samples` still holds.

- [ ] **Step 2: Run to verify it fails, then passes**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scale.py -k svar2 -v`
Expected: FAIL before Task 2's backend exists / if pasted first; PASS once the SVAR2 backend is in. (If Task 2 is already merged, it should PASS immediately — that is the intended end state.)

- [ ] **Step 3 removed — the test above is complete; there is no separate helper to implement.**

- [ ] **Step 4: Run to green**

Run: `pixi run -e dev pytest tests/dataset/test_streaming_scale.py -k svar2 -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/dataset/test_streaming_scale.py
git commit -m "test(streaming): SVAR2 cohort-scale (#284) output-flatness gate (#278)"
```

---

## Task 4: Perf baseline + IO-vs-CPU bound profiling (vcfixture-rs bulk)

**Files:**
- Create: `benchmarking/streaming/svar2_cold_cache.py`

**Interfaces:**
- Consumes: `tests/benchmarks/data/build_svar2_stream_bulk.py::build` (Task 1); `StreamingDataset` over `.svar2` (Task 2). Reference the discipline in `benchmarking/streaming/cold_cache_overlap.py` (SVAR1) — best-of-N on a shared/noisy node, fresh-store-per-run cold cache, ratio is secondary color not a gate.
- Produces: a **measurement report only** (no pass/fail). Two numbers the Phase-2 decision needs: (1) the synchronous-path cold-cache baseline at a few cohort sizes; (2) an IO-vs-CPU split of the window fill (`_find_ranges` + gather+kernel) so we know whether a producer thread (overlap) or rayon-within-kernel is the right Phase-2 lever.

This task **measures, it does not optimize**. It exists so Phase 2 is gated on evidence, per the spec's Performance section and performant-py-rust.

- [ ] **Step 1: Write the harness**

```python
# benchmarking/streaming/svar2_cold_cache.py
"""SVAR2 streaming: synchronous-path cold-cache baseline + IO-vs-CPU bound split.

MEASURES ONLY. Uses the vcfixture-rs bulk builder for cohort-scale stores; skips
cleanly if vcfixture/bcftools/samtools are absent. Reports best-of-N on a shared,
noisy node -- treat ratios as color, not a gate (project perf-gate convention).

Run:
    VCFIXTURE_BIN=/path/to/vcfixture pixi run -e dev \
        python benchmarking/streaming/svar2_cold_cache.py --samples 500 2000 --records 20000
"""
from __future__ import annotations

import argparse, time
from pathlib import Path
import tempfile

import genvarloader as gvl


def _sweep(samples_list, records, repeats):
    from tests.benchmarks.data.build_svar2_stream_bulk import build
    import pyinstrument

    for n in samples_list:
        best = float("inf")
        for rep in range(repeats):
            tmp = Path(tempfile.mkdtemp())          # fresh inode -> never-faulted
            fx = build(tmp, n_samples=n, records=records, seed=1000 + rep)
            sds = gvl.StreamingDataset(
                fx.bed, reference=fx.reference, variants=fx.svar2_path
            ).with_seqs("haplotypes")
            t0 = time.perf_counter()
            for _ in sds.to_iter(batch_size=32):
                pass
            best = min(best, time.perf_counter() - t0)
        print(f"n_samples={n:>6}: synchronous best-of-{repeats} = {best:.3f}s")

        # IO-vs-CPU split on the LAST rep's store, one sweep under pyinstrument.
        prof = pyinstrument.Profiler()
        prof.start()
        for _ in sds.to_iter(batch_size=32):
            pass
        prof.stop()
        print(prof.output_text(unicode=True, color=False))
        print("  -> read `_find_ranges` (search) vs gather+kernel (fill) share above.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, nargs="+", default=[500, 2000])
    ap.add_argument("--records", type=int, default=20_000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()
    try:
        _sweep(args.samples, args.records, args.repeats)
    except FileNotFoundError as e:
        print(f"SKIP: {e}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run (skips without vcfixture)**

Run: `pixi run -e dev python benchmarking/streaming/svar2_cold_cache.py --samples 8 --records 200 --repeats 1`
Expected: either a printed baseline + pyinstrument tree, or `SKIP: ...` if tooling is absent. Both acceptable.

- [ ] **Step 3: Record the bound finding**

If the harness runs, note in the commit message / a scratch note whether the window fill is IO-dominated (page faults in gather) or CPU-dominated (`split_to_flat` decode). This is the input to the Phase-2 lever decision.

- [ ] **Step 4: Commit**

```bash
git add benchmarking/streaming/svar2_cold_cache.py
git commit -m "bench(streaming): SVAR2 synchronous cold-cache baseline + IO/CPU bound split (#278)"
```

---

## Task 5: Docs + roadmap/board

**Files:**
- Modify: `README.md`, `docs/source/faq.md`, `docs/source/dataset.md`, `docs/source/write.md` (only where the streaming-input format list appears), `skills/genvarloader/SKILL.md`, `docs/roadmaps/streaming-dataset.md`.

**Interfaces:**
- Consumes: the shipped `.svar2` streaming support (Task 2).
- Produces: user-facing docs that no longer say SVAR2 streaming is unsupported; roadmap Specs/Plans rows filled.

- [ ] **Step 1: Update the supported-format list**

Grep for the current SVAR2-not-supported claims and the SVAR1 streaming mention, then add `.svar2`:

```bash
grep -rniE "streaming.*svar|svar.*streaming|\.svar2|NotImplementedError.*SVAR2" \
  README.md docs/source/*.md skills/genvarloader/SKILL.md
```
Edit each hit so `.svar` **and** `.svar2` are listed as supported `StreamingDataset` variant sources (VCF/PGEN remain "later"). Keep the sorted-`sample_idx` convention note that already exists for SVAR1 — it applies unchanged.

- [ ] **Step 2: Fill the roadmap rows**

In `docs/roadmaps/streaming-dataset.md`: set the SVAR2 Specs-table row (currently `_TBD_ — issue #278`) to point at `docs/superpowers/specs/2026-07-17-streaming-svar2-backend-design.md` with status `✅ approved`, and add a Plans-table row pointing at this plan with the Phase-1 status; note Phase 2 (engine) is deferred/gated. Set the appropriate ⬜/🚧/✅ marker.

- [ ] **Step 3: API-doc sync check (public surface unchanged, but verify)**

`StreamingDataset` is already public; no new `__all__` symbol. Confirm:
```bash
pixi run -e dev python -c "import re,genvarloader as g; api=open('docs/source/api.md').read(); print('MISSING:', [n for n in g.__all__ if n not in api] or 'none')"
```
Expected: `MISSING: none`.

- [ ] **Step 4: Commit**

```bash
git add README.md docs/source/*.md skills/genvarloader/SKILL.md docs/roadmaps/streaming-dataset.md
git commit -m "docs(streaming): SVAR2 is a supported StreamingDataset source (#278)"
```

---

## Final verification (before opening the PR)

- [ ] **Full parity + scale + unit sweep** (shared code — cover both trees):

Run: `pixi run -e dev pytest tests/dataset tests/unit -q`
Expected: green, including SVAR1 parity + scale (no regression from the additive `__init__`/union changes) and the new SVAR2 parity + scale tests.

- [ ] **Lint/format/type gate:**

Run: `pixi run -e dev ruff check python/ tests/ && pixi run -e dev ruff format --check python/ tests/ && pixi run -e dev typecheck`
Expected: clean.

- [ ] **Open a draft PR into `streaming`** (not `main`), titled `streaming: SVAR2 backend (parity)`, "Closes #278" (Phase 1; note Phase 2 engine follow-up). Add the PR to the StreamingDataset project board.

---

## Phase 2 — producer-thread engine (DEFERRED, gated — not decomposed here)

Out of scope for this plan; a separate spec-linked plan, mirroring how SVAR1's engine (#275/#283) followed its skeleton (#274). Ship it **only if** it clears the measurement gate. Its first steps, in order:

1. **Read Task 4's bound finding.** If the synchronous window fill is IO-dominated, a producer thread (overlap) is the right lever and the SVAR1 result carries over. If CPU-dominated, evaluate rayon-within-kernel (the reconstruct kernel's existing `parallel` flag) first — a single producer caps overlap at ~2×.
2. **Discovery: is GIL-free Rust range computation reachable?** Determine whether `genoray_core::query::ContigReader` exposes a Rust `find_ranges` (the Rust function underneath `SparseVar2._find_ranges`'s `self._readers[contig].find_ranges` pymethod). If yes → implement `Svar2Store.read_window` in Rust (compute ranges + `gather_haps_readbound` into an owned flat-channel buffer) with **no rev bump**. If no → a genoray change + rev bump is required (like SVAR1's `svar1_query`), which is itself gated on the perf win justifying it.
3. **Wire `Svar2StreamEngine`** copying `Svar1StreamEngine` (`src/ffi/stream_engine.rs`) discipline verbatim: 2-slot ping-pong `crossbeam_channel`, shutdown-by-`Sender`-drop, join-then-classify-panics, `py.detach` for the blocking body, `_phys_sample_idx` held once. Buffer = the flat `vk_*/dense_*/lut_*` channels (spec §"window buffer").
4. **Cold-cache A-vs-synchronous measurement** via `benchmarking/streaming/svar2_cold_cache.py` (extended with an engine strategy). Ship the engine as default only if it beats synchronous outside node noise; else keep it behind `_prefetch_strategy` and record the measured reason.
5. **Re-verify** byte-parity + the #284 gate under the engine.

**Remember: no new Rust ships in Phase 2 without a profile confirming the mechanism** (cargo-show-asm / samply), and no engine ships as default without the cold-cache win — performant-py-rust discipline, inherited from the spec.
