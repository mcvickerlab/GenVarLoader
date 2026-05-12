# 1KG bcftools-consensus Parity Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a slow-marked pytest that validates `Dataset[region, sample]` haplotypes against `bcftools consensus` on the real 1000 Genomes chr21/chr22 subset (5 samples) hosted at Zenodo record 20132907, exercising the BCF, PGEN, and SVAR backends.

**Architecture:** A new data-prep script (`tests/data/generate_1kg_ground_truth.py`) pooches the BCF/CSI from Zenodo, derives PGEN (via plink2) and SVAR (via genoray), writes three `.gvl` datasets, picks 100 variant-centered 10 kb regions across chr21/chr22, and emits one bcftools-consensus FASTA per (region, sample, haplotype). A new test (`tests/dataset/test_ds_haps_1kg.py`) mirrors the existing `test_ds_haps.py` and asserts byte-equality across all backends.

**Tech Stack:** Python, pixi, pooch, bcftools, samtools, plink2, genoray (VCF/PGEN/SparseVar), genvarloader, pytest, pytest_cases, pysam, polars, seqpro.

**Reference materials:**
- Spec: `docs/superpowers/specs/2026-05-12-1kg-bcftools-parity-design.md`
- Existing analog data-prep: `tests/data/generate_ground_truth.py`
- Existing analog test: `tests/dataset/test_ds_haps.py`

**TDD note:** This feature is a data-pipeline + integration test. Pure unit TDD does not fit the data-prep script (each step calls an external tool whose output we want to keep). The plan uses incremental smoke checks (run the script with `--stage` flags or partial commands) and the final parity test as the integration-level "failing test → passing test" cycle. The test file is written before the data exists, and we verify it fails for the right reason (missing files) before generating data and watching it pass.

---

## File Structure

| File | Status | Responsibility |
| --- | --- | --- |
| `tests/data/generate_1kg_ground_truth.py` | Create | One-shot data-prep CLI: pooch fetch → normalize → PGEN → SVAR → region BED → 3× gvl.write → bcftools consensus loop |
| `tests/dataset/test_ds_haps_1kg.py` | Create | Parity test (3 backends × 100 regions × 5 samples × 2 haps) |
| `pixi.toml` | Modify | Add `gen-1kg` task |
| `docs/superpowers/specs/2026-05-12-1kg-bcftools-parity-design.md` | Already exists | Spec |

---

### Task 1: Scaffold the data-prep script and pixi task

**Files:**
- Create: `tests/data/generate_1kg_ground_truth.py`
- Modify: `pixi.toml`

- [ ] **Step 1: Create the script skeleton**

Create `tests/data/generate_1kg_ground_truth.py`:

```python
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from time import perf_counter

import typer
from loguru import logger

WDIR = Path(__file__).resolve().parent
ONE_KG_DIR = WDIR / "1kg"
CONS_DIR = WDIR / "1kg_consensus"
REF = WDIR / "fasta" / "hg38.fa.bgz"

N_REGIONS = 100
REGION_LEN = 10_000
SEED = 0

ZENODO_BCF_URL = "https://zenodo.org/records/20132907/files/1kg.chr21_chr22.5samples.bcf"
ZENODO_CSI_URL = "https://zenodo.org/records/20132907/files/1kg.chr21_chr22.5samples.bcf.csi"
# Fill these in on first successful run; the script prints observed hashes
# and exits when they are None.
ZENODO_BCF_HASH: str | None = None
ZENODO_CSI_HASH: str | None = None


def run_shell(cmd: list[str], input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, input=input)
    except subprocess.CalledProcessError as e:
        print("Command:", " ".join(e.cmd))
        print("Stdout:", e.stdout.decode(errors="replace"))
        print("Stderr:", e.stderr.decode(errors="replace"))
        raise


def main() -> None:
    """Generate 1000 Genomes ground-truth haplotypes via bcftools consensus."""
    log_file = WDIR / "generate_1kg_ground_truth.log"
    if log_file.exists():
        log_file.unlink()
    _ = logger.add(log_file, level="DEBUG")

    t0 = perf_counter()

    if not REF.exists():
        raise SystemExit(
            f"Reference {REF} not found. Run `pixi run -e dev gen` first to "
            "fetch hg38."
        )

    ONE_KG_DIR.mkdir(0o777, parents=True, exist_ok=True)
    if CONS_DIR.exists():
        shutil.rmtree(CONS_DIR)
    CONS_DIR.mkdir(0o777, parents=True, exist_ok=True)

    logger.info("Pipeline scaffold OK")
    logger.info(f"Finished in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    typer.run(main)
```

- [ ] **Step 2: Add the pixi task**

Edit `pixi.toml`. Find the `[tasks]` section (the same one that defines `gen` and `test`) and add a single new line after the `test` task line:

```toml
gen-1kg = { cmd = "python tests/data/generate_1kg_ground_truth.py", depends-on = ["gen"] }
```

Do not modify `gen` or `test`.

- [ ] **Step 3: Smoke-run the scaffold**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; logs "Pipeline scaffold OK"; creates `tests/data/1kg/` and `tests/data/1kg_consensus/`. (Assumes `gen` has already been run at least once so `hg38.fa.bgz` exists; if not the script exits with the explicit message above — that's also acceptable for this step.)

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py pixi.toml
rtk git commit -m "test(1kg): scaffold ground-truth generation script"
```

---

### Task 2: Pooch the BCF + CSI from Zenodo

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `fetch_zenodo` helper that prints the observed hash on first run**

Insert below `run_shell`:

```python
def fetch_zenodo(url: str, known_hash: str | None, fname: str) -> Path:
    import pooch

    if known_hash is None:
        # First run: download once without verification, log the observed hash,
        # then fail loudly so the developer can paste it into the script.
        path = Path(
            pooch.retrieve(url, known_hash=None, fname=fname, path=ONE_KG_DIR)
        )
        import hashlib

        h = hashlib.sha256(path.read_bytes()).hexdigest()
        logger.error(
            f"known_hash for {fname} is None. Observed sha256: {h}. "
            "Paste this into ZENODO_BCF_HASH / ZENODO_CSI_HASH and re-run."
        )
        raise SystemExit(2)

    return Path(
        pooch.retrieve(url, known_hash=known_hash, fname=fname, path=ONE_KG_DIR)
    )
```

- [ ] **Step 2: Call it from `main`**

Inside `main`, after the `CONS_DIR.mkdir(...)` line, before the final `logger.info("Pipeline scaffold OK")`, add:

```python
    bcf = fetch_zenodo(ZENODO_BCF_URL, ZENODO_BCF_HASH, "source.bcf")
    csi = fetch_zenodo(ZENODO_CSI_URL, ZENODO_CSI_HASH, "source.bcf.csi")
    logger.info(f"Fetched: {bcf} ({bcf.stat().st_size} bytes)")
    logger.info(f"Fetched: {csi} ({csi.stat().st_size} bytes)")
```

Remove the `logger.info("Pipeline scaffold OK")` line.

- [ ] **Step 3: First run to capture the hash**

Run: `pixi run -e dev gen-1kg`

Expected: exits with code 2 and an error log line containing the observed sha256 for `source.bcf`. Copy that hash and paste it as the value of `ZENODO_BCF_HASH` in the script (replacing `None`).

- [ ] **Step 4: Second run to capture the CSI hash**

Run: `pixi run -e dev gen-1kg`

Expected: same as above but for `source.bcf.csi`. Paste into `ZENODO_CSI_HASH`.

- [ ] **Step 5: Verify both hashes are accepted**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0, logs both file sizes. No re-download (pooch sees the file and verifies).

- [ ] **Step 6: Verify the URL exists if the script reports a 404**

If pooch raises an HTTP error in Step 3, the URL pattern is wrong. The Zenodo record page is `https://zenodo.org/records/20132907`; open it in a browser (or `curl -sI`) and copy the exact filenames into `ZENODO_BCF_URL` and `ZENODO_CSI_URL`. Re-run Step 3.

- [ ] **Step 7: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): pooch BCF + CSI from Zenodo"
```

---

### Task 3: Normalize the BCF (left-align, atomize, split multiallelics)

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `normalize_bcf` helper**

Insert below `fetch_zenodo`:

```python
def normalize_bcf(source_bcf: Path) -> Path:
    """Left-align, atomize, and split multiallelics. Returns indexed filtered.bcf."""
    filtered = ONE_KG_DIR / "filtered.bcf"

    # Step A: left-align
    result = run_shell(
        [
            "bcftools",
            "norm",
            "-f",
            str(REF),
            "-O",
            "u",
            str(source_bcf),
        ]
    )
    logger.info("bcftools norm (left-align) done")

    # Step B: atomize + split multiallelics; emit as bgzipped BCF
    result = run_shell(
        [
            "bcftools",
            "norm",
            "-a",
            "--atom-overlaps",
            ".",
            "-f",
            str(REF),
            "-m",
            "-",
            "-O",
            "b",
            "-o",
            str(filtered),
        ],
        input=result.stdout,
    )
    logger.info("bcftools norm (atomize + split) done")

    # Index
    _ = run_shell(["bcftools", "index", "-f", str(filtered)])
    return filtered
```

- [ ] **Step 2: Wire it into `main`**

Inside `main`, after the `logger.info(f"Fetched: {csi} ...")` line, add:

```python
    filtered = normalize_bcf(bcf)
    logger.info(f"Normalized BCF at {filtered}")
```

- [ ] **Step 3: Run and verify**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; `tests/data/1kg/filtered.bcf` and `filtered.bcf.csi` exist; log shows variant counts in the bcftools stderr.

Sanity check the output:
```bash
bcftools view -h tests/data/1kg/filtered.bcf | tail -3
bcftools view tests/data/1kg/filtered.bcf | head -5
```

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): normalize BCF with bcftools norm"
```

---

### Task 4: Generate PGEN via plink2

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `make_pgen` helper**

Insert below `normalize_bcf`:

```python
def make_pgen(filtered_bcf: Path) -> Path:
    out_prefix = ONE_KG_DIR / "filtered"
    _ = run_shell(
        [
            "plink2",
            "--bcf",
            str(filtered_bcf),
            "--make-pgen",
            "--vcf-half-call",
            "r",
            "--out",
            str(out_prefix),
        ]
    )
    return out_prefix.with_suffix(".pgen")
```

- [ ] **Step 2: Wire it into `main`**

After the `logger.info(f"Normalized BCF ...")` line, add:

```python
    pgen = make_pgen(filtered)
    logger.info(f"PGEN at {pgen}")
```

- [ ] **Step 3: Run and verify**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; `tests/data/1kg/filtered.pgen`, `.pvar`, `.psam` exist.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): generate PGEN via plink2"
```

---

### Task 5: Generate SparseVar via genoray

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `make_svar` helper**

Insert below `make_pgen`:

```python
def make_svar(filtered_bcf: Path) -> Path:
    from genoray import VCF, SparseVar

    out = ONE_KG_DIR / "filtered.svar"
    if out.exists():
        shutil.rmtree(out)
    SparseVar.from_vcf(out, VCF(filtered_bcf), "50mb")
    SparseVar(out).cache_afs()
    return out
```

- [ ] **Step 2: Wire it into `main`**

After the `logger.info(f"PGEN at ...")` line, add:

```python
    svar = make_svar(filtered)
    logger.info(f"SVAR at {svar}")
```

- [ ] **Step 3: Run and verify**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; `tests/data/1kg/filtered.svar/` exists and contains genoray's standard sparse-variant directory layout.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): generate SparseVar via genoray"
```

---

### Task 6: Pick 100 variant-centered 10 kb regions across chr21/chr22

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `pick_regions` helper**

Insert below `make_svar`:

```python
def pick_regions(filtered_bcf: Path) -> Path:
    import numpy as np
    import polars as pl

    bed_path = ONE_KG_DIR / "regions.bed"

    # Pull (chrom, pos) for chr21/chr22 only via bcftools query.
    proc = run_shell(
        [
            "bcftools",
            "query",
            "-f",
            "%CHROM\t%POS\n",
            "-r",
            "chr21,chr22",
            str(filtered_bcf),
        ]
    )
    raw = proc.stdout.decode().strip()
    if not raw:
        # The Zenodo dataset uses GRCh38 contig names without the "chr" prefix
        # on some 1KG releases. Fall back to bare contig names.
        proc = run_shell(
            [
                "bcftools",
                "query",
                "-f",
                "%CHROM\t%POS\n",
                "-r",
                "21,22",
                str(filtered_bcf),
            ]
        )
        raw = proc.stdout.decode().strip()

    if not raw:
        raise SystemExit(
            "No variants found on chr21/chr22 (tried 'chr21,chr22' and '21,22'). "
            "Inspect filtered.bcf with `bcftools view -h` to find the contig "
            "naming convention used."
        )

    df = pl.read_csv(
        raw.encode(),
        separator="\t",
        has_header=False,
        new_columns=["chrom", "pos"],
        schema_overrides={"chrom": pl.Utf8, "pos": pl.Int64},
    )

    # Keep variants with at least REGION_LEN/2 of flanking space on each side.
    half = REGION_LEN // 2
    df = df.filter(pl.col("pos") > half)

    rng = np.random.default_rng(SEED)
    idx = rng.choice(df.height, size=N_REGIONS, replace=False)
    chosen = df[idx.tolist()]

    starts = chosen["pos"].to_numpy() - half
    ends = starts + REGION_LEN
    strand = rng.choice(["+", "-"], size=N_REGIONS, replace=True)

    out = pl.DataFrame(
        {
            "chrom": chosen["chrom"].to_numpy(),
            "start": starts,
            "end": ends,
            "name": ["."] * N_REGIONS,
            "score": ["."] * N_REGIONS,
            "strand": strand,
        }
    )
    out.write_csv(bed_path, include_header=False, separator="\t")
    logger.info(f"Wrote {N_REGIONS} regions to {bed_path}")
    return bed_path
```

- [ ] **Step 2: Wire it into `main`**

After `logger.info(f"SVAR at ...")`, add:

```python
    bed_path = pick_regions(filtered)
```

- [ ] **Step 3: Run and verify**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; `tests/data/1kg/regions.bed` has 100 lines; each line has 6 tab-separated fields; `end - start == 10000` for every row. Sanity check:

```bash
wc -l tests/data/1kg/regions.bed
awk '$3-$2 != 10000' tests/data/1kg/regions.bed | head
```

The second command should produce no output.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): pick 100 variant-centered 10kb regions"
```

---

### Task 7: Write the three .gvl datasets (BCF, PGEN, SVAR)

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `write_datasets` helper**

Insert below `pick_regions`:

```python
def write_datasets(filtered_bcf: Path, pgen: Path, svar: Path, bed_path: Path) -> None:
    import genvarloader as gvl
    from genoray import PGEN, VCF, SparseVar

    bcf_ds = ONE_KG_DIR / "phased_1kg.bcf.gvl"
    pgen_ds = ONE_KG_DIR / "phased_1kg.pgen.gvl"
    svar_ds = ONE_KG_DIR / "phased_1kg.svar.gvl"

    for d in (bcf_ds, pgen_ds, svar_ds):
        if d.exists():
            shutil.rmtree(d)

    vcf_reader = VCF(filtered_bcf)
    if not vcf_reader._valid_index():
        vcf_reader._write_gvi_index()
    _ = vcf_reader._load_index()
    gvl.write(path=bcf_ds, bed=bed_path, variants=vcf_reader)
    logger.info(f"Wrote {bcf_ds}")

    gvl.write(path=pgen_ds, bed=bed_path, variants=PGEN(pgen))
    logger.info(f"Wrote {pgen_ds}")

    gvl.write(path=svar_ds, bed=bed_path, variants=SparseVar(svar))
    logger.info(f"Wrote {svar_ds}")
```

- [ ] **Step 2: Wire it into `main`**

After the `bed_path = pick_regions(filtered)` line, add:

```python
    write_datasets(filtered, pgen, svar, bed_path)
```

- [ ] **Step 3: Run and verify**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; three directories exist under `tests/data/1kg/`:
- `phased_1kg.bcf.gvl/`
- `phased_1kg.pgen.gvl/`
- `phased_1kg.svar.gvl/`

Each should have `metadata.json`, `input_regions.arrow`, and a `genotypes/` subdirectory.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): write BCF/PGEN/SVAR-backed gvl datasets"
```

---

### Task 8: Generate per-(region, sample, hap) bcftools-consensus FASTAs

**Files:**
- Modify: `tests/data/generate_1kg_ground_truth.py`

- [ ] **Step 1: Add a `generate_consensus_fastas` helper**

Insert below `write_datasets`:

```python
def generate_consensus_fastas(filtered_bcf: Path, bed_path: Path) -> None:
    import polars as pl
    from tqdm.auto import tqdm

    # Read samples from the BCF header.
    proc = run_shell(["bcftools", "query", "-l", str(filtered_bcf)])
    samples = proc.stdout.decode().strip().splitlines()
    logger.info(f"Samples: {samples}")

    bed = pl.read_csv(
        bed_path,
        separator="\t",
        has_header=False,
        new_columns=["chrom", "start", "end", "name", "score", "strand"],
        schema_overrides={"chrom": pl.Utf8},
    ).with_row_index()

    total = bed.height * len(samples) * 2
    pbar = tqdm(total=total, desc="bcftools consensus")
    for row_nr, chrom, start, end in bed.select(
        "index", "chrom", "start", "end"
    ).iter_rows():
        subseq = run_shell(
            ["samtools", "faidx", str(REF), f"{chrom}:{start + 1}-{end}"]
        )
        for sample in samples:
            for hap in (0, 1):
                out_fa = CONS_DIR / f"1kg_{sample}_nr{row_nr}_h{hap}.fa"
                _ = run_shell(
                    [
                        "bcftools",
                        "consensus",
                        "-H",
                        str(hap + 1),
                        "-s",
                        sample,
                        "-o",
                        str(out_fa),
                        str(filtered_bcf),
                    ],
                    input=subseq.stdout,
                )
                _ = run_shell(["samtools", "faidx", str(out_fa)])
                _ = pbar.update()
    pbar.close()
```

- [ ] **Step 2: Wire it into `main`**

After `write_datasets(filtered, pgen, svar, bed_path)`, add:

```python
    generate_consensus_fastas(filtered, bed_path)
```

- [ ] **Step 3: Run end-to-end and verify**

Run: `pixi run -e dev gen-1kg`

Expected: exits 0; `tests/data/1kg_consensus/` contains exactly `100 × 5 × 2 = 1000` `.fa` files (plus `.fai` indexes, so 2000 entries):

```bash
ls tests/data/1kg_consensus/*.fa | wc -l
```

Expected: `1000`.

- [ ] **Step 4: Commit**

```bash
rtk git add tests/data/generate_1kg_ground_truth.py
rtk git commit -m "test(1kg): emit bcftools consensus FASTAs for all (region, sample, hap)"
```

---

### Task 9: Write the parametrized parity test

**Files:**
- Create: `tests/dataset/test_ds_haps_1kg.py`

- [ ] **Step 1: Write the test file**

Create `tests/dataset/test_ds_haps_1kg.py`:

```python
from itertools import product
from pathlib import Path

import genvarloader as gvl
import numpy as np
import pysam
import pytest
import seqpro as sp
from genvarloader._ragged import RaggedSeqs
from pytest_cases import parametrize_with_cases

data_dir = Path(__file__).resolve().parents[1] / "data"
ref = data_dir / "fasta" / "hg38.fa.bgz"
cons_dir = data_dir / "1kg_consensus"

pytestmark = pytest.mark.slow


def dataset_bcf():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.bcf.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


def dataset_pgen():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.pgen.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


def dataset_svar():
    return (
        gvl.Dataset.open(data_dir / "1kg" / "phased_1kg.svar.gvl", ref, rc_neg=False)
        .with_len("ragged")
        .with_seqs("haplotypes")
        .with_tracks(False)
    )


@parametrize_with_cases("dataset", cases=".", prefix="dataset_")
def test_ds_haps_1kg(dataset: gvl.RaggedDataset[RaggedSeqs, None]):
    for region, sample in product(range(dataset.n_regions), dataset.samples):
        c, s, e, _ = dataset.regions.select(
            "chrom", "chromStart", "chromEnd", "strand"
        ).row(region)
        haps = dataset[region, sample]
        for h in range(2):
            actual = sp.cast_seqs(haps[h])
            fpath = f"1kg_{sample}_nr{region}_h{h}.fa"
            with pysam.FastaFile(str(cons_dir / fpath)) as f:
                desired = sp.cast_seqs(f.fetch(f.references[0]).upper())
            np.testing.assert_equal(
                actual,
                desired,
                f"region: {region}, sample: {sample}, hap: {h}, "
                f"coords: {c}:{s + 1}-{e}",
            )
```

- [ ] **Step 2: Verify the test is discovered and fails on a missing-data sentinel before generation**

Skip this step if Tasks 1–8 already produced the data (the test will pass instead). Otherwise:

Run: `pixi run -e dev pytest tests/dataset/test_ds_haps_1kg.py -m slow -v --collect-only`

Expected: pytest lists three test parametrizations (`test_ds_haps_1kg[dataset_bcf]`, `[dataset_pgen]`, `[dataset_svar]`).

- [ ] **Step 3: Run the test**

Run: `pixi run -e dev pytest tests/dataset/test_ds_haps_1kg.py -m slow -v`

Expected: all three parametrizations PASS.

If any fail with mismatches, that's the bug the test is designed to surface — diagnose using the per-assertion failure message (`region: N, sample: S, hap: H, coords: ...`).

- [ ] **Step 4: Verify the default test run is unaffected**

Run: `pixi run -e dev pytest tests/dataset/test_ds_haps_1kg.py -v`

Expected: tests are skipped/deselected (without `-m slow`, the slow marker excludes them per project convention).

If they are *not* skipped, check `pyproject.toml` / `pytest.ini` for the marker config — the project's existing convention is that `slow` tests are excluded by default but this is worth confirming.

- [ ] **Step 5: Commit**

```bash
rtk git add tests/dataset/test_ds_haps_1kg.py
rtk git commit -m "test(1kg): add bcftools-consensus parity test across BCF/PGEN/SVAR backends"
```

---

### Task 10: Final verification

**Files:** None.

- [ ] **Step 1: Full default test suite still passes**

Run: `pixi run -e dev pytest tests -q`

Expected: same pass count as before this branch; the new test does not run because it is slow-marked.

- [ ] **Step 2: Slow suite passes**

Run: `pixi run -e dev pytest tests -m slow -v`

Expected: the three new parametrizations PASS along with any other slow tests in the project. (If other slow tests fail and are unrelated to this work, note them and continue — this task is not responsible for pre-existing slow-test failures.)

- [ ] **Step 3: Lint**

Run: `pixi run -e dev ruff check python/ tests/`

Expected: clean (or only pre-existing warnings).

- [ ] **Step 4: Confirm spec coverage**

Open the spec at `docs/superpowers/specs/2026-05-12-1kg-bcftools-parity-design.md` and skim each section. Confirm:
- Data layout matches what was generated (Task 1–8).
- Test structure matches Task 9.
- Pixi task added per spec (Task 1).

No commit unless changes were needed; if so:

```bash
rtk git add -A
rtk git commit -m "test(1kg): final tidy after verification"
```
