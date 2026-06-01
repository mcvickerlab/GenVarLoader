# vcfixture Phase 1 — Synthetic-Reference Test Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the toy test suite's 3 GB hg38 download and hand-authored `tests/data/source.vcf` with a deterministically generated, fully synthetic reference plus a programmatic vcfixture re-encoding of the source VCF — so `pixi run -e dev gen` needs no network and no hg38, while every existing (non-1kg) test passes unchanged.

**Architecture:** A new helper module `tests/data/_synthetic.py` owns two pure functions: `write_synthetic_reference(path, seed)` (writes a bgzipped, faidx-indexed random-ACGT FASTA whose bases at each variant locus are overwritten to match the source VCF's REF alleles, with flank guards that prevent `bcftools norm` from left-shifting indels) and `build_source_vcf(reference_path)` (re-encodes `source.vcf` exactly via `vcfixture.VcfBuilder` — same `chr19`/`chr20` contigs, `NA00001–3` samples, positions, genotypes, INFO/FORMAT). `tests/data/generate_ground_truth.py` keeps its entire downstream pipeline (bcftools norm → consensus oracle → plink2 PGEN → genoray SparseVar → `gvl.write`, plus the polars BED-region heuristic) and only swaps its reference-acquisition and source-VCF-reading front end to call these two helpers. The synthetic reference is written into the already-gitignored `tests/data/fasta/`, so nothing new is committed; `source.vcf` is deleted.

**Tech Stack:** Python, `vcfixture>=0.2.1` (`VcfBuilder`, `Number`, `Type`), `pysam` (bgzip/faidx via `samtools`), `numpy`, `bcftools`, `plink2`, `genoray`, `polars`, `pytest`, `pixi` (`-e dev`).

**Scope:** Phase 1 only. Phase 2 (Hypothesis property tests, deleting the committed `.gvl`/consensus fixtures, standardizing on synthetic contig names + `s0..s2` samples) is a separate follow-up plan. The 1kg slow-tier (`generate_1kg_ground_truth.py`) and bigwig/track fixtures are untouched and keep their own hg38 download.

---

## Reference facts (read before starting)

**The source VCF being re-encoded** (`tests/data/source.vcf`, the canonical VCFv4 example). Records (1-based POS), all FORMAT keys `GT:VAF:HQ` unless noted:

| # | CHROM | POS | REF | ALT | NA00001 | NA00002 | NA00003 | notes |
|---|-------|-----|-----|-----|---------|---------|---------|-------|
| 1 | chr19 | 111 | N | C | `0\|0` | `0\|0` | `0/1` | N-REF locus |
| 2 | chr19 | 1010696 | GAGA | G | `1\|0` | `0\|0` | `0/0` | 3-bp del |
| 3 | chr19 | 1010696 | GAGACGG | G | `0\|0` | `0\|0` | `0/1` | 6-bp del |
| 4 | chr19 | 1010696 | GAGACGGGGCC | G | `0\|1` | `1\|1` | `0/0` | **10-bp del** (test_write_edge_cases) |
| 5 | chr19 | 1110696 | A | TTT | `0\|1` | `1\|1` | `0/0` | insertion |
| 6 | chr19 | 1110696 | A | G | `0\|0` | `0\|0` | `0/1` | |
| 7 | chr19 | 1210696 | C | G | `1\|.` | `0/1` | `1\|1` | missing allele |
| 8 | chr19 | 1210696 | C | G | `.\|1` | `0\|0` | `0/0` | missing allele |
| 9 | chr19 | 1210697 | T | G | `0/0` | `1\|0` | `0/1` | |
| 10 | chr19 | 1210697 | T | A | `0/0` | `1\|0` | `0/1` | |
| 11 | chr20 | 14370 | N | A | `0\|0` | `1\|0` | `1/1` | FORMAT `GT:VAF:GQ:DP:HQ`; INFO `NS=3;DP=14;AF=0.5;DB;H2`; ID `rs6054257`; FILTER PASS |
| 12 | chr20 | 17330 | N | A | `0\|0` | `0\|1` | `0/0` | FORMAT `GT:VAF:GQ:DP:HQ`; INFO `NS=3;DP=11;AF=0.017`; FILTER q10 |
| 13 | chr20 | 1110696 | G | A,T | `1\|2` | `2\|1` | `2/2` | FORMAT `GT:VAF:GQ:DP:HQ`; INFO `NS=2;DP=10;AF=0.333,0.667;AA=T;DB`; ID `rs6040355`; FILTER PASS |
| 14 | chr20 | 1234567 | A | GA,AC | `0/1` | `0/2` | `./.` | FORMAT `GT:VAF:GQ:DP`; INFO `NS=3;DP=9;AA=G;AN=6;AC=3,1`; ID `microsat1`; FILTER PASS |

**INFO header defs** (must all be declared so the header matches; `test_sitesonly.py` requires `NS/DP/AF` defined): `NS` (1,Int), `AN` (1,Int), `AC` (.,Int), `DP` (1,Int), `AF` (.,Float), `AA` (1,String), `DB` (0,Flag), `H2` (0,Flag).
**FORMAT header defs:** `GT` (1,String), `VAF` (A,Float), `GQ` (1,Int), `DP` (1,Int), `HQ` (2,Int).
**FILTER defs:** `q10` ("Quality below 10"), `s50` ("Less than 50% of samples have data"). (`PASS` is built in via `filter=()`.)

**Why FORMAT subfield values can be simplified:** a grep of `tests/` shows no test reads `VAF/HQ/GQ/DP` values; only `GT` genotypes and the INFO header matter. We still re-encode `GT` exactly (it drives haplotype reconstruction) and declare full INFO/FORMAT headers, and we reproduce INFO values on the chr20 records (faithful + cheap). FORMAT non-GT *values* are reproduced where trivial but are not assertion-critical.

**Variant loci needing engineered reference bases** (1-based POS → required REF):
- chr19:111 → `N`
- chr19:1010696 → `GAGACGGGGCC` (the longest REF; shorter REFs `GAGA`, `GAGACGG` are its prefixes — consistent)
- chr19:1110696 → `A`
- chr19:1210696 → `C`
- chr19:1210697 → `T`
- chr20:14370 → `N`
- chr20:17330 → `N`
- chr20:1110696 → `G`
- chr20:1234567 → `A`

Contig minimum lengths: `chr19` ≥ 1,210,697 and `chr20` ≥ 1,234,568. Use `chr19=1_300_000`, `chr20=1_300_000` (≈2.6 MB random FASTA, generated in well under a second; gitignored, regenerated by `gen`).

**Flank-guard rule (prevents `bcftools norm` left-shift):** for each indel anchor position `p` (1-based) — records 2/3/4 at chr19:1010696, record 5 at chr19:1110696, record 14 at chr20:1234567 — overwrite the single reference base immediately 5′ of the anchor (0-based index `p-2`) with a base that is **not equal to the anchor base and not equal to the last base of that record's REF**. A deletion/insertion only left-shifts into a repeat; breaking the immediate 5′ base guarantees no shift. Use `'T'` as the guard base where the anchor/REF tail is not `T`, else `'A'`. (Concrete picks given in Task 3.)

**Fixture coupling already audited** — these files must keep passing unchanged (the re-encode preserves everything they depend on): `tests/integration/dataset/test_write_edge_cases.py`, `tests/dataset/test_with_methods.py`, `tests/integration/dataset/test_write_tracks_e2e.py`, `tests/unit/variants/test_sitesonly.py`, `tests/integration/dataset/test_rc_packing.py`, `tests/integration/dataset/test_write.py`, plus the data-driven `tests/integration/dataset/test_ds_haps*.py`.

**Conftest fixtures touched:** `ref_fasta` (repoint to synthetic FASTA); `source_vcf` (delete — no test consumes the *fixture*; tests read `vcf_dir/"filtered_source.vcf.gz"`, which `gen` still produces).

**Acceptance gate:** `pixi run -e dev gen` completes with **no network/hg38 fetch**, then `pixi run -e dev pytest tests -m "not slow"` is green. The `-m "not slow"` is required: there is no `addopts` auto-excluding slow tests, and the 1kg tests are `@pytest.mark.slow` and need `gen-1kg` (hg38) fixtures. Do **not** run the `test` pixi task — it triggers `gen-1kg` + hg38.

---

## File Structure

- **Create** `tests/data/_synthetic.py` — `write_synthetic_reference(path, seed=0)` and `build_source_vcf(reference_path)`. Single responsibility: produce the synthetic reference + the re-encoded source VCF document. No downstream pipeline logic.
- **Create** `tests/data/test_synthetic_inputs.py` — fast unit tests for the two helpers (locus bases, norm survival, position preservation). Lives under `tests/` so it runs in the normal suite.
- **Modify** `tests/data/generate_ground_truth.py` — swap the reference-acquisition + source-VCF-reading front end to call the helpers; update the manual BED rows + sample list to synthetic loci; keep the rest.
- **Modify** `tests/conftest.py` — repoint `ref_fasta`, delete `source_vcf` fixture.
- **Modify** `pixi.toml` — add `vcfixture>=0.2.1` to `[feature.py310.pypi-dependencies]`.
- **Delete** `tests/data/source.vcf`.

---

## Task 1: Build the worktree environment and confirm the Rust extension imports

This fresh worktree has no compiled Rust extension yet, which is why the pyrefly pre-commit hook fails on `genvarloader.count_intervals`. Build it first so later `gen`/`pytest` runs work and commits don't trip the hook.

**Files:** none (environment only).

- [ ] **Step 1: Trigger a dev-env solve + editable build and confirm import**

Run:
```bash
pixi run -e dev python -c "import genvarloader as gvl; from genvarloader.genvarloader import intervals, count_intervals; print('ok', gvl.__version__)"
```
Expected: prints `ok <version>` with no ImportError (maturin compiles the extension on first install; this may take a few minutes).

- [ ] **Step 2: Confirm the baseline 1kg tests are slow-marked (so plain pytest excludes them)**

Run:
```bash
grep -rn "pytest.mark.slow" tests/integration/dataset/test_ds_haps_1kg.py
```
Expected: at least one match. If there is **no** match, note it — Task 8 must then explicitly deselect the 1kg test by path instead of relying on the slow marker.

- [ ] **Step 3: No commit** (environment-only task).

---

## Task 2: Add vcfixture as a dev dependency

**Files:**
- Modify: `pixi.toml` (under `[feature.py310.pypi-dependencies]`, near `hypothesis = "*"`)

- [ ] **Step 1: Add the dependency**

In `pixi.toml`, inside the `[feature.py310.pypi-dependencies]` table, add:
```toml
vcfixture = ">=0.2.1"
```

- [ ] **Step 2: Install and verify the public API imports**

Run:
```bash
pixi run -e dev python -c "from vcfixture import VcfBuilder, Number, Type; print(VcfBuilder, Number.ONE, Type.FLOAT)"
```
Expected: prints the class and enum members with no ImportError.

- [ ] **Step 3: Commit**

```bash
git add pixi.toml pixi.lock
git commit -m "test: add vcfixture as a dev dependency"
```

---

## Task 3: Write `write_synthetic_reference` (TDD)

**Files:**
- Create: `tests/data/_synthetic.py`
- Create (test): `tests/data/test_synthetic_inputs.py`

- [ ] **Step 1: Write the failing test**

Create `tests/data/test_synthetic_inputs.py`:
```python
"""Unit tests for the synthetic reference + source-VCF re-encoding helpers."""

from __future__ import annotations

from pathlib import Path

import pysam
import pytest

from _synthetic import write_synthetic_reference

# (contig, 1-based pos, expected REF) for every variant locus in the source VCF.
LOCI = [
    ("chr19", 111, "N"),
    ("chr19", 1010696, "GAGACGGGGCC"),
    ("chr19", 1110696, "A"),
    ("chr19", 1210696, "C"),
    ("chr19", 1210697, "T"),
    ("chr20", 14370, "N"),
    ("chr20", 17330, "N"),
    ("chr20", 1110696, "G"),
    ("chr20", 1234567, "A"),
]


def test_reference_has_expected_bases_at_loci(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    assert ref.exists()
    assert ref.with_suffix(ref.suffix + ".fai").exists() or (
        ref.parent / (ref.name + ".fai")
    ).exists()
    with pysam.FastaFile(str(ref)) as fa:
        for contig, pos, expected in LOCI:
            got = fa.fetch(contig, pos - 1, pos - 1 + len(expected)).upper()
            assert got == expected, f"{contig}:{pos} got {got!r}, want {expected!r}"


def test_reference_is_deterministic(tmp_path: Path):
    a = write_synthetic_reference(tmp_path / "a.fa.bgz", seed=0)
    b = write_synthetic_reference(tmp_path / "b.fa.bgz", seed=0)
    with pysam.FastaFile(str(a)) as fa, pysam.FastaFile(str(b)) as fb:
        assert fa.fetch("chr19", 0, 5000) == fb.fetch("chr19", 0, 5000)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pixi run -e dev pytest tests/data/test_synthetic_inputs.py -q
```
Expected: FAIL — `ModuleNotFoundError: No module named '_synthetic'` (or `ImportError`).

- [ ] **Step 3: Implement `write_synthetic_reference`**

Create `tests/data/_synthetic.py`:
```python
"""Synthetic reference + source-VCF re-encoding for the toy test fixtures.

Replaces the hg38 download and hand-authored ``source.vcf``. The reference is
random ACGT with bases at each variant locus overwritten to match the source
VCF's REF alleles, plus single-base flank guards 5' of every indel anchor so
``bcftools norm`` cannot left-shift indels off their hardcoded positions.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# (contig, length) — sized to span every variant locus with headroom.
CONTIGS: list[tuple[str, int]] = [("chr19", 1_300_000), ("chr20", 1_300_000)]

# (contig, 1-based pos, REF) — the longest REF at each shared position.
REF_OVERWRITES: list[tuple[str, int, str]] = [
    ("chr19", 111, "N"),
    ("chr19", 1010696, "GAGACGGGGCC"),
    ("chr19", 1110696, "A"),
    ("chr19", 1210696, "C"),
    ("chr19", 1210697, "T"),
    ("chr20", 14370, "N"),
    ("chr20", 17330, "N"),
    ("chr20", 1110696, "G"),
    ("chr20", 1234567, "A"),
]

# (contig, 1-based anchor pos, guard base) — overwrite the base at index pos-2
# (immediately 5' of the anchor) to break any leftward repeat. Guard base is
# chosen != anchor base and != REF tail base for that record.
FLANK_GUARDS: list[tuple[str, int, str]] = [
    ("chr19", 1010696, "T"),  # anchor G, REF tail C -> T is safe
    ("chr19", 1110696, "G"),  # anchor A, ALT TTT/REF A -> G is safe
    ("chr20", 1234567, "T"),  # anchor A, REF A / ALTs GA,AC -> T is safe
]

_BASES = np.frombuffer(b"ACGT", dtype="S1")


def write_synthetic_reference(path: str | Path, seed: int = 0) -> Path:
    """Write a bgzipped, faidx-indexed synthetic reference to *path*.

    Returns the path to the bgzipped FASTA (``.fa.bgz``).
    """
    path = Path(path)
    rng = np.random.default_rng(seed)

    seqs: dict[str, np.ndarray] = {}
    for contig, length in CONTIGS:
        seqs[contig] = rng.choice(_BASES, size=length)

    for contig, pos, ref in REF_OVERWRITES:
        arr = seqs[contig]
        start = pos - 1  # 0-based
        ref_bytes = np.frombuffer(ref.encode(), dtype="S1")
        arr[start : start + len(ref)] = ref_bytes

    for contig, pos, guard in FLANK_GUARDS:
        # Base immediately 5' of the anchor (0-based index pos-2).
        seqs[contig][pos - 2] = guard.encode()

    # Write plain FASTA (60-col wrapped), then bgzip + faidx via samtools.
    plain = path.with_suffix("")  # strip .bgz
    if plain.suffix != ".fa":
        plain = plain.with_suffix(".fa")
    with open(plain, "w") as f:
        for contig, _ in CONTIGS:
            f.write(f">{contig}\n")
            seq = seqs[contig].tobytes().decode()
            for i in range(0, len(seq), 60):
                f.write(seq[i : i + 60] + "\n")

    subprocess.run(f"bgzip -f -c {plain} > {path}", shell=True, check=True)
    plain.unlink()
    subprocess.run(["samtools", "faidx", str(path)], check=True)
    return path
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pixi run -e dev pytest tests/data/test_synthetic_inputs.py -q
```
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add tests/data/_synthetic.py tests/data/test_synthetic_inputs.py
git commit -m "test: synthetic reference generator for toy fixtures"
```

---

## Task 4: Write `build_source_vcf` re-encoding the source VCF (TDD)

**Files:**
- Modify: `tests/data/_synthetic.py` (add `build_source_vcf`)
- Modify: `tests/data/test_synthetic_inputs.py` (add re-encode tests)

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_synthetic_inputs.py`:
```python
import subprocess

from _synthetic import build_source_vcf


def _norm(vcf_path: Path, ref_path: Path) -> str:
    """Left-align with bcftools norm; return normalized VCF text."""
    out = subprocess.run(
        ["bcftools", "norm", "-f", str(ref_path), str(vcf_path)],
        check=True,
        capture_output=True,
    )
    return out.stdout.decode()


def test_source_vcf_re_encode_matches_header_and_samples(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    doc = build_source_vcf(ref)
    text = doc.render()
    # Samples preserved exactly.
    header_line = next(l for l in text.splitlines() if l.startswith("#CHROM"))
    assert header_line.endswith("NA00001\tNA00002\tNA00003")
    # INFO defs required by test_sitesonly are declared.
    for fid in ("NS", "DP", "AF"):
        assert f"##INFO=<ID={fid}," in text


def test_source_vcf_passes_norm_and_preserves_coupled_positions(tmp_path: Path):
    ref = write_synthetic_reference(tmp_path / "synthetic.fa.bgz", seed=0)
    doc = build_source_vcf(ref)
    vcf = doc.write(tmp_path / "source.vcf", bgzip=False)
    normalized = _norm(vcf, ref)  # raises if any REF mismatches the reference
    # The 10-bp deletion hardcoded by test_write_edge_cases must survive at pos.
    assert any(
        line.startswith("chr19\t1010696\t") and "GAGACGGGGCC" in line
        for line in normalized.splitlines()
    ), "chr19:1010696 10-bp deletion shifted or lost during norm"
    # chr20 multiallelic and microsat records present.
    assert any(l.startswith("chr20\t1110696\t") for l in normalized.splitlines())
    assert any(l.startswith("chr20\t1234567\t") for l in normalized.splitlines())
```

- [ ] **Step 2: Run the test to verify it fails**

Run:
```bash
pixi run -e dev pytest tests/data/test_synthetic_inputs.py -q
```
Expected: FAIL — `ImportError: cannot import name 'build_source_vcf'`.

- [ ] **Step 3: Implement `build_source_vcf`**

Add to `tests/data/_synthetic.py` (imports at top of file):
```python
from vcfixture import Number, Type, VcfBuilder
```
Then append the function:
```python
def build_source_vcf(reference_path: str | Path) -> "object":
    """Re-encode the canonical source VCF against the synthetic reference.

    Returns a built ``vcfixture`` document (has ``.render()`` / ``.write()``).
    REFs are taken verbatim from the table below; they match the engineered
    reference bases written by ``write_synthetic_reference``.
    """
    b = VcfBuilder(
        samples=["NA00001", "NA00002", "NA00003"],
        contigs=[("chr19", 1_300_000), ("chr20", 1_300_000)],
    )
    # INFO defs (must match the original header).
    b.info("NS", Number.ONE, Type.INTEGER)
    b.info("AN", Number.ONE, Type.INTEGER)
    b.info("AC", Number.DOT, Type.INTEGER)
    b.info("DP", Number.ONE, Type.INTEGER)
    b.info("AF", Number.DOT, Type.FLOAT)
    b.info("AA", Number.ONE, Type.STRING)
    b.info("DB", Number.FLAG, Type.FLAG)
    b.info("H2", Number.FLAG, Type.FLAG)
    # FORMAT defs.
    b.fmt("GT")
    b.fmt("VAF", Number.A, Type.FLOAT)
    b.fmt("GQ", Number.ONE, Type.INTEGER)
    b.fmt("DP", Number.ONE, Type.INTEGER)
    b.fmt("HQ", Number.fixed(2), Type.INTEGER)
    # FILTER defs.
    b.filter("q10", "Quality below 10")
    b.filter("s50", "Less than 50% of samples have data")

    # chr19 block — FORMAT GT:VAF:HQ. Non-GT FORMAT values are not assertion-
    # critical (no test reads them); GT is reproduced exactly.
    b.record("chr19", 111, ref="N", alt=["C"], gt=["0|0", "0|0", "0/1"])
    b.record("chr19", 1010696, ref="GAGA", alt=["G"], gt=["1|0", "0|0", "0/0"])
    b.record("chr19", 1010696, ref="GAGACGG", alt=["G"], gt=["0|0", "0|0", "0/1"])
    b.record("chr19", 1010696, ref="GAGACGGGGCC", alt=["G"], gt=["0|1", "1|1", "0/0"])
    b.record("chr19", 1110696, ref="A", alt=["TTT"], gt=["0|1", "1|1", "0/0"])
    b.record("chr19", 1110696, ref="A", alt=["G"], gt=["0|0", "0|0", "0/1"])
    b.record("chr19", 1210696, ref="C", alt=["G"], gt=["1|.", "0/1", "1|1"])
    b.record("chr19", 1210696, ref="C", alt=["G"], gt=[".|1", "0|0", "0/0"])
    b.record("chr19", 1210697, ref="T", alt=["G"], gt=["0/0", "1|0", "0/1"])
    b.record("chr19", 1210697, ref="T", alt=["A"], gt=["0/0", "1|0", "0/1"])

    # chr20 block — carries INFO (test_sitesonly) and IDs/FILTERs.
    b.record(
        "chr20", 14370, ref="N", alt=["A"], ids=["rs6054257"], qual=29.0, filter=(),
        gt=["0|0", "1|0", "1/1"], info={"NS": 3, "DP": 14, "AF": [0.5], "DB": True, "H2": True},
    )
    b.record(
        "chr20", 17330, ref="N", alt=["A"], qual=3.0, filter=["q10"],
        gt=["0|0", "0|1", "0/0"], info={"NS": 3, "DP": 11, "AF": [0.017]},
    )
    b.record(
        "chr20", 1110696, ref="G", alt=["A", "T"], ids=["rs6040355"], qual=67.0, filter=(),
        gt=["1|2", "2|1", "2/2"],
        info={"NS": 2, "DP": 10, "AF": [0.333, 0.667], "AA": "T", "DB": True},
    )
    b.record(
        "chr20", 1234567, ref="A", alt=["GA", "AC"], ids=["microsat1"], qual=50.0, filter=(),
        gt=["0/1", "0/2", "./."],
        info={"NS": 3, "DP": 9, "AA": "G", "AN": 6, "AC": [3, 1]},
    )
    return b.build()
```

- [ ] **Step 4: Run the test to verify it passes**

Run:
```bash
pixi run -e dev pytest tests/data/test_synthetic_inputs.py -q
```
Expected: PASS (4 passed). If `test_source_vcf_passes_norm_and_preserves_coupled_positions` fails with a REF-mismatch from `bcftools norm`, the reference locus bases are wrong — re-check `REF_OVERWRITES`. If the `chr19:1010696` assertion fails because the position shifted, adjust the `FLANK_GUARDS` guard base for that anchor.

- [ ] **Step 5: Commit**

```bash
git add tests/data/_synthetic.py tests/data/test_synthetic_inputs.py
git commit -m "test: re-encode source VCF via vcfixture builder"
```

---

## Task 5: Rewrite the generator front end to use the helpers

Swap only the reference-acquisition and source-VCF-reading sections of `generate_ground_truth.py`. Keep the normalization, filtering, consensus, PGEN, SVAR, BED-heuristic, and `gvl.write` tail intact.

**Files:**
- Modify: `tests/data/generate_ground_truth.py:88-121` (reference block + source read) and `:239-250` (manual BED rows + sample list area)

- [ ] **Step 1: Replace the reference-acquisition block**

In `generate_ground_truth.py`, replace the `pooch.retrieve(...)` hg38 block (the lines from `reference = Path(` through the `samtools faidx` of the reference, roughly lines 88–105) with:
```python
    from _synthetic import build_source_vcf, write_synthetic_reference

    reference = write_synthetic_reference(fasta_dir / "synthetic.fa.bgz", seed=0)
```
Remove the now-unused `pooch` import.

- [ ] **Step 2: Replace the source-VCF read with the builder output**

Replace the block that reads `vcf_path` (`with open(vcf_path, "r") as f: vcf = f.read().encode()`, roughly lines 107–111) with:
```python
    # Re-encode the (formerly hand-authored) source VCF programmatically.
    vcf = build_source_vcf(reference).render().encode()
```
Delete the now-unused `vcf_path = WDIR / f"{name}.vcf"` line.

- [ ] **Step 3: Update the manual BED rows to synthetic loci**

Replace the manual-additions `rows` DataFrame (the `chr19`/`chr1` block, roughly lines 242–248) with rows that exercise the same scenarios on the synthetic contigs — a spanning-deletion region over the chr19:1010696 deletion and a no-variant region on chr20:
```python
    rows = pl.DataFrame(
        {
            # spanning deletion region (over chr19:1010696 10-bp deletion)
            # and a no-variant region on chr20.
            "chrom": ["chr19", "chr20"],
            "start": [1010696, 500_000],
            "end": [1010696 + SEQ_LEN, 500_000 + SEQ_LEN],
        }
    )
```

- [ ] **Step 4: Run the generator**

Run:
```bash
pixi run -e dev gen
```
Expected: completes with no network access; creates `tests/data/fasta/synthetic.fa.bgz`, `tests/data/vcf/filtered_source.vcf.gz`, `tests/data/pgen/`, `tests/data/filtered.svar/`, `tests/data/consensus/`, and `tests/data/phased_dataset.{vcf,pgen,svar}.gvl`. Check the tail of `tests/data/generate_ground_truth.log` for `Finished in`.

- [ ] **Step 5: Commit**

```bash
git add tests/data/generate_ground_truth.py
git commit -m "test: generate toy fixtures from synthetic reference, drop hg38"
```

---

## Task 6: Repoint conftest fixtures and delete source.vcf

**Files:**
- Modify: `tests/conftest.py:34-37` (`ref_fasta`) and `:104-106` (`source_vcf`)
- Delete: `tests/data/source.vcf`

- [ ] **Step 1: Repoint `ref_fasta`**

In `tests/conftest.py`, change the `ref_fasta` fixture body and docstring:
```python
@pytest.fixture(scope="session")
def ref_fasta(data_dir: Path) -> Path:
    """bgzipped synthetic reference used by the toy datasets (generated by `gen`)."""
    return data_dir / "fasta" / "synthetic.fa.bgz"
```

- [ ] **Step 2: Delete the now-unused `source_vcf` fixture**

Remove the `source_vcf` fixture (the `@pytest.fixture` + function at lines ~104–106). No test consumes this fixture; tests read `vcf_dir/"filtered_source.vcf.gz"`.

- [ ] **Step 3: Delete the hand-authored source VCF**

Run:
```bash
git rm tests/data/source.vcf
```

- [ ] **Step 4: Verify no remaining references to the deleted symbols**

Run:
```bash
grep -rn "source_vcf\b" tests/ ; grep -rn "hg38" tests/conftest.py tests/data/generate_ground_truth.py
```
Expected: no matches for `source_vcf` as a fixture name in test files; no `hg38` in the toy generator or conftest (the 1kg generator may still mention hg38 — that's expected and out of scope).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py
git commit -m "test: point ref_fasta at synthetic reference, remove source.vcf"
```

---

## Task 7: Run the full non-1kg suite and confirm parity

**Files:** none (verification).

- [ ] **Step 1: Regenerate fixtures from a clean state**

Run:
```bash
rm -rf tests/data/fasta tests/data/vcf tests/data/pgen tests/data/filtered.svar tests/data/consensus tests/data/phased_dataset.*.gvl
pixi run -e dev gen
```
Expected: regenerates everything from scratch with no network access.

- [ ] **Step 2: Run the coupled tests explicitly first**

Run:
```bash
pixi run -e dev pytest \
  tests/unit/variants/test_sitesonly.py \
  tests/dataset/test_with_methods.py \
  tests/integration/dataset/test_write_edge_cases.py \
  tests/integration/dataset/test_write_tracks_e2e.py \
  tests/integration/dataset/test_rc_packing.py \
  tests/integration/dataset/test_write.py \
  tests/integration/dataset/test_ds_haps.py \
  tests/integration/dataset/test_ds_haps_modes.py -q
```
Expected: all pass. If `test_write_edge_cases.py` fails on the chr19:1010696 spanning deletion, revisit the `FLANK_GUARDS` in `_synthetic.py` (Task 3) and regenerate.

- [ ] **Step 3: Run the entire default (non-slow) suite**

Run:
```bash
pixi run -e dev pytest tests -m "not slow" -q
```
Expected: green (1kg `@pytest.mark.slow` tests are deselected by default). Record the passed/deselected counts.

- [ ] **Step 4: No commit** (verification only; fixtures are gitignored).

---

## Task 8: Confirm the genvarloader skill is unaffected and finalize

**Files:**
- Possibly modify: `skills/genvarloader/SKILL.md` (only if it references toy fixtures — it should not; it tracks public API)

- [ ] **Step 1: Check the skill for fixture coupling**

Run:
```bash
grep -rn "source.vcf\|hg38\|NA00001\|phased_dataset" skills/genvarloader/SKILL.md
```
Expected: no matches. If any appear, update them to reflect the synthetic-fixture reality; otherwise no change.

- [ ] **Step 2: Confirm git status is clean of unexpected artifacts**

Run:
```bash
git status --short
```
Expected: only intended changes; generated fixtures under `tests/data/{fasta,vcf,pgen,filtered.svar,consensus}` and `phased_dataset.*.gvl` are gitignored and must NOT appear as untracked-to-commit.

- [ ] **Step 3: Final verification of the acceptance gate**

Run:
```bash
pixi run -e dev gen && pixi run -e dev pytest tests -m "not slow" -q
```
Expected: `gen` runs offline; suite green. This is the Phase 1 acceptance gate.

- [ ] **Step 4: Commit any skill change (only if Step 1 required one)**

```bash
git add skills/genvarloader/SKILL.md
git commit -m "docs(skill): align with synthetic test fixtures"
```

---

## Notes & follow-ups (not part of Phase 1 execution)

- **CI** (`.github/workflows/test.yaml`) caches hg38 for the `test` task, which still runs `gen-1kg`. The toy `gen` no longer needs hg38, but the 1kg path does — leave the CI hg38 cache as-is. The stale comment in `generate_ground_truth.py` about bumping the `hg38-ref-<...>` cache key is removed along with the pooch block.
- **Phase 2** (separate plan): add an upstream reference-consistent Hypothesis strategy to vcfixture, add a property-test module asserting gvl haplotypes == bcftools consensus and gvl genotypes/AF == vcfixture `GroundTruth`, then delete the committed `phased_dataset.*.gvl`/`consensus/` reliance and standardize on synthetic contig names + `s0..s2` samples.
