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

    logger.info(f"Finished in {perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    typer.run(main)
