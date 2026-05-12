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

ZENODO_BCF_URL = "https://zenodo.org/records/20132907/files/1kgp.thin.bcf"
ZENODO_CSI_URL = "https://zenodo.org/records/20132907/files/1kgp.thin.bcf.csi"
ZENODO_BCF_MD5 = "md5:3bdfed585e4a6b2a51c49d1d7dc7124f"
ZENODO_CSI_MD5 = "md5:8f190a43294404ca320b45a05851d56a"


def run_shell(cmd: list[str], input: bytes | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(cmd, check=True, capture_output=True, input=input)
    except subprocess.CalledProcessError as e:
        print("Command:", " ".join(e.cmd))
        print("Stdout:", e.stdout.decode(errors="replace"))
        print("Stderr:", e.stderr.decode(errors="replace"))
        raise


def fetch_zenodo(url: str, known_hash: str, fname: str) -> Path:
    import pooch

    return Path(
        pooch.retrieve(url, known_hash=known_hash, fname=fname, path=ONE_KG_DIR)
    )


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
    _ = run_shell(
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

    _ = run_shell(["bcftools", "index", "-f", str(filtered)])
    return filtered

def make_pgen(filtered_bcf: Path) -> Path:
    """Generate PGEN from normalized BCF via plink2."""
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


def make_svar(filtered_bcf: Path) -> Path:
    """Generate SparseVar from normalized BCF via genoray."""
    from genoray import VCF, SparseVar

    out = ONE_KG_DIR / "filtered.svar"
    if out.exists():
        shutil.rmtree(out)
    SparseVar.from_vcf(out, VCF(filtered_bcf), "50mb")
    SparseVar(out).cache_afs()
    return out


def pick_regions(filtered_bcf: Path) -> Path:
    import io
    import numpy as np
    import polars as pl

    bed_path = ONE_KG_DIR / "regions.bed"

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
        raise SystemExit(
            "No variants found on chr21/chr22. Inspect filtered.bcf with "
            "`bcftools view -h` to verify contig naming."
        )

    df = pl.read_csv(
        io.BytesIO(raw.encode()),
        separator="\t",
        has_header=False,
        new_columns=["chrom", "pos"],
        schema_overrides={"chrom": pl.Utf8, "pos": pl.Int64},
    )

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

    bcf = fetch_zenodo(ZENODO_BCF_URL, ZENODO_BCF_MD5, "source.bcf")
    csi = fetch_zenodo(ZENODO_CSI_URL, ZENODO_CSI_MD5, "source.bcf.csi")
    logger.info(f"Fetched: {bcf} ({bcf.stat().st_size} bytes)")
    logger.info(f"Fetched: {csi} ({csi.stat().st_size} bytes)")

    filtered = normalize_bcf(bcf)
    logger.info(f"Normalized BCF at {filtered}")
    pgen = make_pgen(filtered)
    logger.info(f"PGEN at {pgen}")
    svar = make_svar(filtered)
    logger.info(f"SVAR at {svar}")
    bed_path = pick_regions(filtered)
    logger.info(f"Finished in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    typer.run(main)
